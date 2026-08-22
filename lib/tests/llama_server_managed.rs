use anyhow::Result;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;
use tempfile::TempDir;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};
use yatima_lib::{
    Agent, AgentStop, ChatMlTemplate, ChatSession, Completer, Dir, GenOpts, JsonToolCall,
    LlamaServer, LlamaServerCompleter, LlamaServerConfig, LlamaServerSpawn, ModelSource,
    PlainTemplate, ReadFile, StopReason, Tools,
};

const WITHIN: Duration = Duration::from_secs(8);

async fn artifact(behavior: &str) -> (TempDir, yatima_lib::GgufArtifact) {
    let directory = tempfile::tempdir().unwrap();
    std::fs::write(directory.path().join(format!("{behavior}.gguf")), "stub").unwrap();
    let resolved =
        ModelSource::from_args(Some(directory.path().to_path_buf()), None, None, true, None)
            .unwrap()
            .resolve_async()
            .await
            .unwrap();
    (directory, resolved.into_gguf().unwrap())
}

async fn spawn_stub(behavior: &str, readiness: Duration) -> (TempDir, LlamaServer) {
    let (directory, artifact) = artifact(behavior).await;
    let mut spec = LlamaServerSpawn::new(artifact);
    spec.binary = PathBuf::from(env!("CARGO_BIN_EXE_llama-server-stub"));
    spec.readiness_timeout = readiness;
    let server = tokio::time::timeout(WITHIN, LlamaServer::spawn(spec))
        .await
        .expect("managed spawn stayed bounded")
        .unwrap();
    (directory, server)
}

#[tokio::test]
async fn spawn_pins_arguments_waits_for_readiness_and_reaps() {
    // upholds: LSRV-1 / LSRV-2 — the stub rejects an unpinned host, slot
    // count, or missing --jinja; delayed readiness exercises repeated health
    // probes, and shutdown returns only after reap and drain joins.
    let (_directory, server) = spawn_stub("ready-after-2", Duration::from_secs(2)).await;
    assert_eq!(server.props().build, "stub-b10520");
    assert_eq!(server.props().total_slots, 1);
    assert!(server.launched_artifact().ends_with("ready-after-2.gguf"));
    let status = tokio::time::timeout(WITHIN, server.shutdown())
        .await
        .expect("shutdown stayed bounded")
        .unwrap();
    assert!(
        !status.success(),
        "SIGKILL is successful cleanup, not child success"
    );
}

#[tokio::test]
async fn stdout_and_stderr_are_drained_concurrently_into_bounded_tails() {
    // upholds: LSRV-1 — each individual pipe and both together exceed normal
    // pipe capacity before readiness; none can wedge child or supervisor.
    for behavior in ["flood-stdout", "flood-stderr", "flood-both"] {
        let (_directory, server) = spawn_stub(behavior, Duration::from_secs(3)).await;
        let diagnostics = server.diagnostics();
        if behavior != "flood-stderr" {
            assert!(diagnostics.contains("stdout-flood-complete"), "{behavior}");
        }
        if behavior != "flood-stdout" {
            assert!(diagnostics.contains("stderr-flood-complete"), "{behavior}");
        }
        assert!(
            diagnostics.len() <= 2 * 64 * 1024 + 64,
            "diagnostic tails must stay bounded: {}",
            diagnostics.len()
        );
        server.shutdown().await.unwrap();
    }
}

#[tokio::test]
async fn readiness_timeout_reaps_and_joins_before_error() {
    // upholds: LSRV-1 — the timeout failure owns a child and therefore must
    // report completed reap and drain joins, not merely a closed port.
    let (_directory, artifact) = artifact("never-ready").await;
    let mut spec = LlamaServerSpawn::new(artifact);
    spec.binary = PathBuf::from(env!("CARGO_BIN_EXE_llama-server-stub"));
    spec.readiness_timeout = Duration::from_millis(300);
    let error = tokio::time::timeout(WITHIN, LlamaServer::spawn(spec))
        .await
        .expect("readiness failure stayed bounded")
        .err()
        .expect("never-ready must fail");
    let message = format!("{error:#}");
    assert!(message.contains("readiness timed out"), "{message}");
    assert!(
        message.contains("reaped child") && message.contains("joined"),
        "{message}"
    );
}

#[tokio::test]
async fn early_exit_is_retried_then_reported_with_cleanup() {
    // upholds: LSRV-1 — early startup is conservatively retried three times;
    // each owned child is reaped and its drains joined.
    let (_directory, artifact) = artifact("exit-immediately").await;
    let mut spec = LlamaServerSpawn::new(artifact);
    spec.binary = PathBuf::from(env!("CARGO_BIN_EXE_llama-server-stub"));
    spec.readiness_timeout = Duration::from_secs(1);
    let error = tokio::time::timeout(WITHIN, LlamaServer::spawn(spec))
        .await
        .expect("retry exhaustion stayed bounded")
        .err()
        .expect("exit-immediately must fail");
    let message = format!("{error:#}");
    for attempt in 1..=3 {
        assert!(message.contains(&format!("attempt {attempt}")), "{message}");
    }
    assert_eq!(message.matches("reaped child").count(), 3, "{message}");
    assert!(message.contains("exit-immediately marker"), "{message}");
}

#[tokio::test]
async fn introspection_failure_reaps_and_joins_before_error() {
    // upholds: LSRV-1 — compatibility failure is also a child-owning path.
    let (_directory, artifact) = artifact("introspection-fail").await;
    let mut spec = LlamaServerSpawn::new(artifact);
    spec.binary = PathBuf::from(env!("CARGO_BIN_EXE_llama-server-stub"));
    spec.readiness_timeout = Duration::from_secs(2);
    let error = tokio::time::timeout(WITHIN, LlamaServer::spawn(spec))
        .await
        .expect("introspection failure stayed bounded")
        .err()
        .expect("bad introspection must fail");
    let message = format!("{error:#}");
    assert!(message.contains("introspection failed"), "{message}");
    assert!(
        message.contains("reaped child") && message.contains("joined"),
        "{message}"
    );
}

#[tokio::test]
async fn child_death_wins_the_in_flight_completion_race() {
    // upholds: LSRV-1 — death after readiness is surfaced as a named process
    // error with status and stderr, not an incidental connection failure.
    let (_directory, mut server) = spawn_stub("die-after-ready", Duration::from_secs(2)).await;
    let error = tokio::time::timeout(WITHIN, server.complete("prompt", &GenOpts::default(), &[]))
        .await
        .expect("child death stayed bounded")
        .unwrap_err();
    let message = format!("{error:#}");
    assert!(message.contains("died during completion"), "{message}");
    assert!(message.contains("die-after-ready marker"), "{message}");
    server.shutdown().await.unwrap();
}

#[tokio::test]
async fn completion_can_win_then_the_child_shuts_down_orderly() {
    // upholds: LSRV-1 / CMP-1 — the successful branch leaves ownership intact.
    let (_directory, mut server) = spawn_stub("serve-one-completion", Duration::from_secs(2)).await;
    fn requires_send<F: Send>(_: &F) {}
    let opts = GenOpts::default();
    let future = server.complete("type witness", &opts, &[]);
    requires_send(&future);
    drop(future);
    let completion =
        tokio::time::timeout(WITHIN, server.complete("prompt", &GenOpts::default(), &[]))
            .await
            .expect("completion stayed bounded")
            .unwrap();
    assert_eq!(completion.text, "stub answer");
    assert_eq!(completion.stop, StopReason::Eos);
    server.shutdown().await.unwrap();
}

#[tokio::test]
async fn live_child_protocol_failure_carries_bounded_diagnostics() {
    let (_directory, mut server) = spawn_stub("incomplete-stream", Duration::from_secs(2)).await;
    let error = tokio::time::timeout(WITHIN, server.complete("prompt", &GenOpts::default(), &[]))
        .await
        .expect("protocol failure stayed bounded")
        .unwrap_err();
    let message = format!("{error:#}");
    assert!(message.contains("without a final stop event"), "{message}");
    assert!(
        message.contains("remained alive") && message.contains("incomplete-stream"),
        "{message}"
    );
    server.shutdown().await.unwrap();
}

#[tokio::test]
async fn missing_binary_is_a_bounded_spawn_error() {
    // upholds: LSRV-1 — no child exists on this path, but the error is named
    // and immediate.
    let (_directory, artifact) = artifact("serve-one-completion").await;
    let mut spec = LlamaServerSpawn::new(artifact);
    spec.binary = PathBuf::from("/definitely/not/a/llama-server");
    let error = tokio::time::timeout(WITHIN, LlamaServer::spawn(spec))
        .await
        .expect("spawn failure stayed bounded")
        .err()
        .expect("missing binary must fail");
    assert!(format!("{error:#}").contains("execute"));
}

#[tokio::test]
async fn agent_tool_round_composes_over_the_http_completer() -> Result<()> {
    // Preserves LSRV-3 / REASON-1 through the generic Agent path without
    // adding user-facing agent backend flags in stage 2.
    let server = MockServer::start().await;
    let response_number = Arc::new(AtomicUsize::new(0));
    let sequence = response_number.clone();
    Mock::given(method("POST"))
        .and(path("/completion"))
        .respond_with(move |_request: &wiremock::Request| {
            let content = if sequence.fetch_add(1, Ordering::SeqCst) == 0 {
                r#"<tool_call>{"name":"read_file","args":{"path":"note.txt"}}</tool_call>"#
            } else {
                "The note says blue."
            };
            let body = format!(
                "data: {}\n\ndata: {{\"content\":\"\",\"stop\":true,\"stop_type\":\"eos\"}}\n\n",
                serde_json::json!({"content": content, "stop": false})
            );
            ResponseTemplate::new(200).set_body_raw(body, "text/event-stream")
        })
        .mount(&server)
        .await;

    let directory = tempfile::tempdir()?;
    std::fs::write(directory.path().join("note.txt"), "blue")?;
    let tools = Tools::new().with(ReadFile::new(Dir::new(directory.path())));
    let config = LlamaServerConfig::new(server.uri())?;
    let mut completer = LlamaServerCompleter::new(config)?;
    let mut agent = Agent::new(
        &mut completer,
        &tools,
        JsonToolCall,
        PlainTemplate,
        "Use the tool.",
        3,
    );
    let run = tokio::time::timeout(WITHIN, agent.run_async("Read note.txt"))
        .await
        .expect("agent-over-HTTP stayed bounded")?;
    assert_eq!(run.stop, AgentStop::Final);
    assert_eq!(run.answer, "The note says blue.");
    assert_eq!(response_number.load(Ordering::SeqCst), 2);
    Ok(())
}

#[tokio::test]
#[ignore = "requires llama-server on PATH and the local Qwen2.5-32B Q4_K_M GGUF"]
async fn live_managed_qwen_chat() -> Result<()> {
    // Live LSRV-1 composition proof; deliberately not part of the offline gate.
    let artifact = ModelSource::from_args(
        None,
        Some("bartowski/Qwen2.5-32B-Instruct-GGUF".into()),
        None,
        true,
        Some("Qwen2.5-32B-Instruct-Q4_K_M.gguf".into()),
    )?
    .resolve_async()
    .await?
    .into_gguf()?;
    let mut server = LlamaServer::spawn(LlamaServerSpawn::new(artifact)).await?;
    let mut chat = ChatSession::new(&mut server, ChatMlTemplate).with_opts(GenOpts {
        max_tokens: 48,
        ..GenOpts::default()
    });
    let answer = chat.turn_async("In one sentence, what is a CRDT?").await?;
    assert!(!answer.trim().is_empty());
    drop(chat);
    server.shutdown().await?;
    Ok(())
}
