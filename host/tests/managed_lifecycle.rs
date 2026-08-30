//! The hermetic host lifecycle battery: the backend thread's ownership of a
//! managed llama-server child, end to end over the protocol stub — startup
//! phases, verified identity, Muse turns on the existing event meanings,
//! and every exit converging on the one reaping epilogue (HOST-3 / LSRV-1 /
//! LSRV-5 at the host boundary). No network, no models, no llama-server.
//!
//! The stub binary is the lib's single protocol stub behind this crate's
//! thin wrapper bin (`llama-server-stub-host`); its `-m` file stem selects
//! behavior, and each test's unique tempdir path doubles as the process
//! marker a leak check can grep for.

use std::path::PathBuf;
use std::time::Duration;

use sha2::{Digest, Sha256};
use tempfile::TempDir;
use yatima_host::{
    spawn_nonblocking, Channel, HostClient, HostConfig, HostEvent, HostOwner, HostRequest,
    ModelIdentity, StartupPhase, StopKind, ToolNoteKind,
};
use yatima_lib::{
    ChatFormat, ChildCleanupFailed, GenOpts, LlamaServerProfile, ModelProfile, ModelSource,
    ProfileBackend, Sha256Digest,
};

const WITHIN: Duration = Duration::from_secs(20);
const STUB_BYTES: &[u8] = b"stub";

/// Serializes the whole battery. Every test here spawns and reaps real OS
/// children inside one process, and tokio's child reaping rides coalesced
/// SIGCHLD delivery: under concurrent churn — plus the two leak-pipe
/// witnesses that each deliberately consume the lib's full 10 s cleanup
/// bound — bystander tests were observed failing with "cleanup timed out"
/// through no fault of their own. The honest verdict propagation rightly
/// refuses to bless an unproven cleanup, so the fix is determinism, not
/// tolerance: one test's bounds measure its own work, never its
/// neighbours'. tokio's Mutex: no std lock is ever held across an await.
static SESSION: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

fn digest(bytes: &[u8]) -> Sha256Digest {
    format!("{:x}", Sha256::digest(bytes)).parse().unwrap()
}

fn stub_profile(bytes: &[u8]) -> LlamaServerProfile {
    LlamaServerProfile {
        expected_sha256: digest(bytes),
        build_floor: 10520,
        template_sha256: digest(b"stub-template"),
        context: 4096,
        top_k: 64,
    }
}

/// The stub-shaped profile: a Muse-format managed profile whose source is
/// the test's tempdir — the validated `HostConfig::managed` constructor is
/// the only door (PROFILE-1/2 by construction), with the launcher pointed
/// at the wrapper stub binary.
fn stub_model_profile(directory: &TempDir, bytes: &[u8]) -> ModelProfile {
    ModelProfile {
        name: "stub-muse".into(),
        backend: ProfileBackend::LlamaServer(stub_profile(bytes)),
        dir: Some(directory.path().to_path_buf()),
        format: Some(ChatFormat::MuseGlimmer),
        ..ModelProfile::default()
    }
}

/// A managed config over the stub: `{behavior}.gguf` holding `bytes` in a
/// fresh tempdir, resolved offline, launched via the wrapper stub binary.
fn managed_config(directory: &TempDir, behavior: &str, bytes: &[u8]) -> HostConfig {
    managed_config_with_readiness(directory, behavior, bytes, Duration::from_secs(15))
}

fn managed_config_with_readiness(
    directory: &TempDir,
    behavior: &str,
    bytes: &[u8],
    readiness: Duration,
) -> HostConfig {
    std::fs::write(directory.path().join(format!("{behavior}.gguf")), bytes).unwrap();
    HostConfig::managed(
        &stub_model_profile(directory, bytes),
        true,
        GenOpts::default(),
        None,
    )
    .expect("a managed profile with a pinned format")
    .with_managed_launcher(
        PathBuf::from(env!("CARGO_BIN_EXE_llama-server-stub-host")),
        readiness,
    )
}

async fn recv(client: &mut HostClient) -> HostEvent {
    tokio::time::timeout(WITHIN, client.event_rx.recv())
        .await
        .expect("an event must arrive within the bound")
        .expect("the actor must not vanish silently")
}

/// Drive a spawned host to `Ready`, asserting the exact managed phase order
/// on the way (PROTO-2's vocabulary at its actual boundaries) and the
/// verified identity carried by `Ready` (LSRV-5 at the host boundary).
async fn ready_after_phases(client: &mut HostClient, expected_digest: &Sha256Digest) -> HostEvent {
    for expected in [
        StartupPhase::ResolvingModel,
        StartupPhase::VerifyingModel,
        StartupPhase::StartingBackend,
    ] {
        match recv(client).await {
            HostEvent::Startup { phase } => assert_eq!(phase, expected, "phase order"),
            other => panic!("expected Startup {expected:?}, got {other:?}"),
        }
    }
    let ready = recv(client).await;
    let HostEvent::Ready(info) = &ready else {
        panic!("expected Ready after the phases, got {ready:?}");
    };
    assert_eq!(
        info.identity,
        ModelIdentity::VerifiedSha256(expected_digest.to_string()),
        "Ready must carry the verified digest, byte for byte"
    );
    assert_eq!(
        info.backend, "b10520-stub",
        "backend names the server build"
    );
    ready
}

/// The reap oracle. pgrep's exit semantics: 0 = matched, 1 = no match;
/// anything else (2 = usage, 3 = cannot get the process list) means the
/// oracle could not answer and MUST fail the test rather than pass it — an
/// oracle that fails open proves nothing. Returns the surviving matches, or
/// `None` for a clean answer of "no such process".
async fn surviving_children(marker: &str) -> Option<String> {
    let output = tokio::process::Command::new("pgrep")
        .args(["-f", marker])
        .output()
        .await
        .expect("run pgrep");
    match output.status.code() {
        Some(0) => Some(String::from_utf8_lossy(&output.stdout).into_owned()),
        Some(1) => None,
        other => panic!(
            "pgrep could not answer (exit {other:?}): {}",
            String::from_utf8_lossy(&output.stderr)
        ),
    }
}

/// No process whose command line mentions `marker` (each test's unique
/// tempdir path) survives: the epilogue reaped the child.
async fn assert_no_child(marker: &str) {
    if let Some(children) = surviving_children(marker).await {
        panic!("stub child leaked: {children}");
    }
}

/// Poll until no `marker` process remains (the drop-fallback path has no
/// join to await), bounded.
async fn wait_for_no_child(marker: &str) {
    for _ in 0..100 {
        if surviving_children(marker).await.is_none() {
            return;
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    panic!("stub child still alive after the bound: {marker}");
}

fn submit(client: &HostClient, turn_id: u64, text: &str) {
    client
        .req_tx
        .send(HostRequest::Submit {
            turn_id,
            text: text.into(),
        })
        .expect("the actor is alive");
}

async fn shutdown_joined(owner: HostOwner) {
    tokio::time::timeout(WITHIN, owner.shutdown())
        .await
        .expect("shutdown stayed bounded")
        .expect("shutdown joined cleanly");
}

#[tokio::test]
async fn managed_muse_chat_turn_rides_the_existing_events() {
    let _session = SESSION.lock().await;
    // upholds: HOST-3 / LSRV-5 / REASON-1 — the full happy path: exact phase
    // order, Ready carrying the verified digest, a Muse turn whose reasoning
    // and answer arrive as ordinary classified Fragments with no ATEM bytes,
    // and a joined shutdown that leaves no child.
    let directory = TempDir::new().unwrap();
    let marker = directory.path().display().to_string();
    let (mut client, owner) =
        spawn_nonblocking(managed_config(&directory, "muse-chat", STUB_BYTES)).unwrap();
    ready_after_phases(&mut client, &digest(STUB_BYTES)).await;

    submit(&client, 1, "what is yatima?");
    let mut reasoning = String::new();
    let mut answer = String::new();
    let mut started = false;
    loop {
        match recv(&mut client).await {
            HostEvent::Started { turn_id: 1 } => started = true,
            HostEvent::Fragment { channel, text, .. } => {
                for framing in ["<|", "atem:"] {
                    assert!(!text.contains(framing), "framing leaked: {text}");
                }
                match channel {
                    Channel::Reasoning => reasoning.push_str(&text),
                    Channel::Answer => answer.push_str(&text),
                }
            }
            HostEvent::Done { stop, .. } => {
                assert_eq!(stop, StopKind::Eos);
                break;
            }
            other => panic!("unexpected event: {other:?}"),
        }
    }
    assert!(started);
    assert_eq!(reasoning, "consider the question");
    assert_eq!(answer, "a clean answer");

    shutdown_joined(owner).await;
    assert_no_child(&marker).await;
}

#[tokio::test]
async fn managed_muse_tool_round_is_typed_activity() {
    let _session = SESSION.lock().await;
    // upholds: HOST-4 / PROTO-1 / REASON-1 — a Muse tool round surfaces as
    // the existing event meanings: a Call note, a typed Failure result (the
    // CAP-2 cross-origin refusal — hermetic, no fetch), then a clean final
    // answer; ATEM bytes appear nowhere, and the internal tool-call channel
    // is consumed, never a panic and never prose (the retired unreachable
    // arm's replacement, witnessed).
    let directory = TempDir::new().unwrap();
    let marker = directory.path().display().to_string();
    let (mut client, owner) =
        spawn_nonblocking(managed_config(&directory, "muse-tool-round", STUB_BYTES)).unwrap();
    ready_after_phases(&mut client, &digest(STUB_BYTES)).await;

    // Grant an origin so the web tools surface (CAP-3); the scripted call
    // targets a *different* origin, so dispatch refuses before any I/O.
    client
        .req_tx
        .send(HostRequest::Grant {
            origin: "https://granted.example".into(),
        })
        .unwrap();
    let HostEvent::Grants { origins, .. } = recv(&mut client).await else {
        panic!("expected the grant report");
    };
    assert_eq!(origins, ["https://granted.example"]);

    submit(&client, 1, "read the page and summarize");
    let mut reasoning = String::new();
    let mut answer = String::new();
    let mut notes = Vec::new();
    loop {
        match recv(&mut client).await {
            HostEvent::Started { .. } => {}
            HostEvent::Fragment { channel, text, .. } => {
                for framing in ["<|", "atem:"] {
                    assert!(!text.contains(framing), "framing leaked: {text}");
                }
                match channel {
                    Channel::Reasoning => reasoning.push_str(&text),
                    Channel::Answer => answer.push_str(&text),
                }
            }
            HostEvent::ToolNote { kind, text, .. } => notes.push((kind, text)),
            HostEvent::RetractAnswer { .. } => {}
            HostEvent::Done { stop, .. } => {
                assert_eq!(stop, StopKind::Eos);
                break;
            }
            other => panic!("unexpected event: {other:?}"),
        }
    }
    assert!(
        notes
            .iter()
            .any(|(kind, text)| *kind == ToolNoteKind::Call && text.contains("read_url")),
        "the invocation must surface as a typed Call note: {notes:?}"
    );
    assert!(
        notes.iter().any(|(kind, _)| *kind == ToolNoteKind::Failure),
        "the refusal must surface as a typed Failure note: {notes:?}"
    );
    assert!(reasoning.contains("fetch the page first"), "{reasoning}");
    assert!(
        reasoning.contains("the fetch was refused"),
        "second-round reasoning: {reasoning}"
    );
    assert_eq!(answer, "the final grounded answer");

    shutdown_joined(owner).await;
    assert_no_child(&marker).await;
}

#[tokio::test]
async fn managed_startup_failure_is_fatal_and_reaped() {
    let _session = SESSION.lock().await;
    // upholds: HOST-3 / LSRV-1 — a child that dies during startup yields the
    // resolving/verifying/starting phases, then Fatal with the lib's
    // reap-and-join diagnostics; nothing survives.
    let directory = TempDir::new().unwrap();
    let marker = directory.path().display().to_string();
    let (mut client, owner) =
        spawn_nonblocking(managed_config(&directory, "exit-immediately", STUB_BYTES)).unwrap();
    for _ in 0..3 {
        let HostEvent::Startup { .. } = recv(&mut client).await else {
            panic!("phases precede the failure");
        };
    }
    let HostEvent::Fatal(message) = recv(&mut client).await else {
        panic!("expected Fatal");
    };
    assert!(message.contains("exited during startup"), "{message}");
    shutdown_joined(owner).await;
    assert_no_child(&marker).await;
}

#[tokio::test]
async fn managed_gate_failure_is_fatal_and_reaped() {
    let _session = SESSION.lock().await;
    // upholds: LSRV-5 / LSRV-1 — a server whose introspection violates the
    // pinned gates never serves: Fatal names the gate, the lib's cleanup
    // reaped and joined, and no child survives.
    let directory = TempDir::new().unwrap();
    let marker = directory.path().display().to_string();
    let (mut client, owner) =
        spawn_nonblocking(managed_config(&directory, "gate-wrong-context", STUB_BYTES)).unwrap();
    let fatal = loop {
        match recv(&mut client).await {
            HostEvent::Startup { .. } => continue,
            HostEvent::Fatal(message) => break message,
            other => panic!("expected Startup or Fatal, got {other:?}"),
        }
    };
    assert!(fatal.contains("context"), "{fatal}");
    assert!(fatal.contains("reaped child"), "{fatal}");
    shutdown_joined(owner).await;
    assert_no_child(&marker).await;
}

#[tokio::test]
async fn recoverable_turn_failure_keeps_the_session() {
    let _session = SESSION.lock().await;
    // upholds: HOST-3 — a mid-stream protocol failure is an Error for its
    // turn, not the end of the session: the next submit is served by the
    // same backend, and shutdown still joins cleanly.
    let directory = TempDir::new().unwrap();
    let marker = directory.path().display().to_string();
    let (mut client, owner) =
        spawn_nonblocking(managed_config(&directory, "error-then-answer", STUB_BYTES)).unwrap();
    ready_after_phases(&mut client, &digest(STUB_BYTES)).await;

    submit(&client, 1, "first");
    loop {
        match recv(&mut client).await {
            HostEvent::Error { turn_id: 1, .. } => break,
            HostEvent::Started { .. } | HostEvent::Fragment { .. } => continue,
            other => panic!("expected the turn error, got {other:?}"),
        }
    }
    submit(&client, 2, "second");
    let mut answer = String::new();
    loop {
        match recv(&mut client).await {
            HostEvent::Started { turn_id: 2 } => {}
            HostEvent::Fragment {
                turn_id: 2,
                channel: Channel::Answer,
                text,
            } => answer.push_str(&text),
            HostEvent::Fragment { .. } => {}
            HostEvent::Done { turn_id: 2, .. } => break,
            other => panic!("the retry must be served: {other:?}"),
        }
    }
    assert_eq!(answer, "stub answer");
    shutdown_joined(owner).await;
    assert_no_child(&marker).await;
}

#[tokio::test]
async fn child_death_is_fatal_loss_through_the_epilogue() {
    let _session = SESSION.lock().await;
    // upholds: HOST-3 / LSRV-1 — a child that dies mid-turn is fatal loss,
    // not a recoverable error: the turn's Error is followed by Fatal naming
    // the exit, the serve loop converges on the epilogue (which reaps the
    // corpse), and the owner's shutdown still joins.
    let directory = TempDir::new().unwrap();
    let marker = directory.path().display().to_string();
    let (mut client, owner) =
        spawn_nonblocking(managed_config(&directory, "die-after-ready", STUB_BYTES)).unwrap();
    ready_after_phases(&mut client, &digest(STUB_BYTES)).await;

    submit(&client, 1, "trigger");
    let mut saw_error = false;
    let fatal = loop {
        match recv(&mut client).await {
            HostEvent::Started { .. } | HostEvent::Fragment { .. } => {}
            HostEvent::Error { turn_id: 1, .. } => saw_error = true,
            HostEvent::Fatal(message) => break message,
            other => panic!("expected Error then Fatal, got {other:?}"),
        }
    };
    assert!(saw_error, "the failed turn reports before the loss");
    assert!(fatal.contains("exited"), "{fatal}");
    shutdown_joined(owner).await;
    assert_no_child(&marker).await;
}

#[tokio::test]
async fn child_death_with_unproven_cleanup_fails_shutdown() {
    let _session = SESSION.lock().await;
    // upholds: LSRV-1 / HOST-3 — the final ownership path: the child dies
    // mid-completion while a descendant holds its pipes, so the death
    // handler's own drain-join fails and consumes the handles — the later
    // epilogue cannot re-prove that cleanup. The turn still reports Error
    // then Fatal, and the owner's shutdown must surface the typed cleanup
    // debt instead of blessing the run.
    let directory = TempDir::new().unwrap();
    let marker = directory.path().display().to_string();
    let (mut client, owner) =
        spawn_nonblocking(managed_config(&directory, "leak-pipe-die", STUB_BYTES)).unwrap();
    ready_after_phases(&mut client, &digest(STUB_BYTES)).await;

    submit(&client, 1, "trigger");
    let mut saw_error = false;
    let fatal = loop {
        match recv(&mut client).await {
            HostEvent::Started { .. } | HostEvent::Fragment { .. } => {}
            HostEvent::Error { turn_id: 1, .. } => saw_error = true,
            HostEvent::Fatal(message) => break message,
            other => panic!("expected Error then Fatal, got {other:?}"),
        }
    };
    assert!(saw_error, "the failed turn reports before the loss");
    assert!(fatal.contains("exited"), "{fatal}");

    let error = tokio::time::timeout(WITHIN, owner.shutdown())
        .await
        .expect("shutdown stayed bounded")
        .expect_err("an unproven mid-completion cleanup must fail shutdown");
    let message = format!("{error:#}");
    assert!(
        message.contains("drain join") || message.contains("cleanup"),
        "the debt names the cleanup: {message}"
    );
    assert!(
        error.downcast_ref::<ChildCleanupFailed>().is_some(),
        "the typed marker survives to the owner: {message}"
    );
    // Clean up the deliberate leaker (its argv0 carries the marker path).
    let _ = tokio::process::Command::new("pkill")
        .args(["-f", &marker])
        .status()
        .await;
    wait_for_no_child(&marker).await;
}

#[tokio::test]
async fn early_cancel_stops_the_turn_over_the_managed_backend() {
    let _session = SESSION.lock().await;
    // upholds: CANCEL-1 (host boundary) — a wire cancel that beats its turn
    // is remembered by the gate and fires the instant the turn arms: the
    // managed completion ends Stopped without a token surfacing.
    let directory = TempDir::new().unwrap();
    let marker = directory.path().display().to_string();
    let (mut client, owner) =
        spawn_nonblocking(managed_config(&directory, "muse-chat", STUB_BYTES)).unwrap();
    ready_after_phases(&mut client, &digest(STUB_BYTES)).await;

    client
        .req_tx
        .send(HostRequest::Cancel { turn_id: 1 })
        .unwrap();
    submit(&client, 1, "never runs");
    let stop = loop {
        match recv(&mut client).await {
            HostEvent::Started { .. } | HostEvent::Fragment { .. } => {}
            HostEvent::Done { stop, .. } => break stop,
            other => panic!("expected Done, got {other:?}"),
        }
    };
    assert_eq!(stop, StopKind::Stopped);
    shutdown_joined(owner).await;
    assert_no_child(&marker).await;
}

#[tokio::test]
async fn owner_shutdown_cancels_an_active_turn_and_reaps() {
    let _session = SESSION.lock().await;
    // upholds: HOST-3 / LSRV-1 / CANCEL-1 — the owner's shutdown lands while
    // a turn is genuinely mid-stream (a fragment has arrived and the stub is
    // still dribbling): `cancel_armed` ends the decode promptly (LSRV-4's
    // pending read races the cancel), the serve loop drains the shutdown
    // request, the epilogue reaps, and the thread joins — all inside the
    // shutdown bound. The cancelled turn surfaces as Done{Stopped}.
    let directory = TempDir::new().unwrap();
    let marker = directory.path().display().to_string();
    let (mut client, owner) =
        spawn_nonblocking(managed_config(&directory, "slow-stream", STUB_BYTES)).unwrap();
    ready_after_phases(&mut client, &digest(STUB_BYTES)).await;

    submit(&client, 1, "stream forever");
    // Prove the turn is active before shutting down: Started, then at least
    // one streamed fragment.
    loop {
        match recv(&mut client).await {
            HostEvent::Started { turn_id: 1 } => {}
            HostEvent::Fragment { turn_id: 1, .. } => break,
            other => panic!("expected the live stream, got {other:?}"),
        }
    }
    shutdown_joined(owner).await;

    // The cancelled turn completed as Stopped before the actor exited; its
    // events were buffered on the plane.
    let stop = loop {
        match tokio::time::timeout(WITHIN, client.event_rx.recv())
            .await
            .expect("buffered events must drain within the bound")
        {
            Some(HostEvent::Fragment { .. }) => {}
            Some(HostEvent::Done { turn_id: 1, stop }) => break stop,
            Some(other) => panic!("expected the cancelled turn's Done, got {other:?}"),
            None => panic!("the plane closed before the turn's Done"),
        }
    };
    assert_eq!(stop, StopKind::Stopped);
    assert_no_child(&marker).await;
}

#[tokio::test]
async fn vanished_event_plane_exits_through_the_epilogue() {
    let _session = SESSION.lock().await;
    // upholds: HOST-3 — when no frontend can ever see another event, the
    // actor stops serving the void at its next wakeup and still reaps.
    let directory = TempDir::new().unwrap();
    let marker = directory.path().display().to_string();
    let (mut client, owner) =
        spawn_nonblocking(managed_config(&directory, "muse-chat", STUB_BYTES)).unwrap();
    ready_after_phases(&mut client, &digest(STUB_BYTES)).await;

    let HostClient {
        req_tx, event_rx, ..
    } = client;
    drop(event_rx);
    // The next request is the wakeup that observes the closed plane.
    req_tx
        .send(HostRequest::Submit {
            turn_id: 1,
            text: "into the void".into(),
        })
        .unwrap();
    shutdown_joined(owner).await;
    assert_no_child(&marker).await;
}

#[tokio::test]
async fn owner_shutdown_during_verification_prevents_launch() {
    let _session = SESSION.lock().await;
    // upholds: HOST-3 / LSRV-1 — lifecycle cancellation during startup: the
    // hash observes the cancel between read chunks, so a shutdown issued
    // during VerifyingModel stops the hash mid-file and prevents the child
    // launch — StartingBackend never happens and no process ever exists.
    // The artifact is large enough that the cancel decisively beats a full
    // hash.
    let directory = TempDir::new().unwrap();
    let marker = directory.path().display().to_string();
    let big = vec![0u8; 32 * 1024 * 1024];
    let (mut client, owner) =
        spawn_nonblocking(managed_config(&directory, "cancelled-verify", &big)).unwrap();

    let HostEvent::Startup {
        phase: StartupPhase::ResolvingModel,
    } = recv(&mut client).await
    else {
        panic!("resolution first");
    };
    let HostEvent::Startup {
        phase: StartupPhase::VerifyingModel,
    } = recv(&mut client).await
    else {
        panic!("verification second");
    };
    shutdown_joined(owner).await;
    // The actor exited without ever reaching StartingBackend: the event
    // plane closed with nothing further on it.
    match tokio::time::timeout(WITHIN, client.event_rx.recv()).await {
        Ok(None) => {}
        other => panic!("expected a closed plane after the cancelled startup, got {other:?}"),
    }
    assert_no_child(&marker).await;
}

#[tokio::test]
async fn dropped_owner_still_winds_the_host_down() {
    let _session = SESSION.lock().await;
    // The Drop fallback: an abandoned owner requests shutdown and the
    // actor's own epilogue still reaps — but nothing joins, so this is
    // observed externally and is never the success witness (HOST-3's
    // joined-exit proof is `shutdown`, above).
    let directory = TempDir::new().unwrap();
    let marker = directory.path().display().to_string();
    let (mut client, owner) =
        spawn_nonblocking(managed_config(&directory, "muse-chat", STUB_BYTES)).unwrap();
    ready_after_phases(&mut client, &digest(STUB_BYTES)).await;
    drop(client);
    drop(owner);
    wait_for_no_child(&marker).await;
}

#[tokio::test]
async fn failed_epilogue_fails_shutdown_not_blesses_it() {
    let _session = SESSION.lock().await;
    // upholds: HOST-3 / LSRV-1 — the joined-success witness must not lie: a
    // descendant of the stub holds the output pipes open, so the epilogue's
    // bounded drain-join fails, and `HostOwner::shutdown` must surface that
    // as an error after the join rather than reporting success over an
    // unproven cleanup.
    let directory = TempDir::new().unwrap();
    let marker = directory.path().display().to_string();
    let (mut client, owner) =
        spawn_nonblocking(managed_config(&directory, "leak-pipe", STUB_BYTES)).unwrap();
    ready_after_phases(&mut client, &digest(STUB_BYTES)).await;

    let error = tokio::time::timeout(WITHIN, owner.shutdown())
        .await
        .expect("shutdown stayed bounded (the cleanup bound is inside it)")
        .expect_err("a failed epilogue must fail shutdown");
    let message = format!("{error:#}");
    assert!(message.contains("epilogue"), "{message}");
    // Clean up the deliberate leaker (its $0 carries the marker path).
    let _ = tokio::process::Command::new("pkill")
        .args(["-f", &marker])
        .status()
        .await;
    wait_for_no_child(&marker).await;
}

#[tokio::test]
async fn failed_startup_cleanup_fails_shutdown_too() {
    let _session = SESSION.lock().await;
    // upholds: LSRV-1 / HOST-3 — a startup failure (bad gate) whose own
    // cleanup also fails (the leak-pipe descendant defeats the bounded
    // drain-join) is an ownership debt, not a self-cleaned failure: the
    // Fatal reports the gate, and the owner's shutdown must surface the
    // typed cleanup failure instead of blessing an unproven reap.
    let directory = TempDir::new().unwrap();
    let marker = directory.path().display().to_string();
    let (mut client, owner) =
        spawn_nonblocking(managed_config(&directory, "leak-pipe-bad-gate", STUB_BYTES)).unwrap();
    let fatal = loop {
        match recv(&mut client).await {
            HostEvent::Startup { .. } => continue,
            HostEvent::Fatal(message) => break message,
            other => panic!("expected Startup or Fatal, got {other:?}"),
        }
    };
    assert!(
        fatal.contains("context"),
        "the gate failure reports: {fatal}"
    );
    let error = tokio::time::timeout(WITHIN, owner.shutdown())
        .await
        .expect("shutdown stayed bounded")
        .expect_err("an unproven startup cleanup must fail shutdown");
    let message = format!("{error:#}");
    assert!(message.contains("cleanup failed"), "{message}");
    // Clean up the deliberate leaker (its argv0 carries the marker path).
    let _ = tokio::process::Command::new("pkill")
        .args(["-f", &marker])
        .status()
        .await;
    wait_for_no_child(&marker).await;
}

#[tokio::test]
async fn owner_shutdown_during_stalled_readiness_stays_bounded() {
    let _session = SESSION.lock().await;
    // upholds: HOST-3 / LSRV-1 — a child stalled in readiness (never-ready)
    // when the owner shuts down: the spawn's own bounded wait expires, the
    // lib reaps within it, the cancelled startup exits quietly (no Fatal
    // for a requested exit), and the owner's join completes inside the
    // bound with nothing left running.
    let directory = TempDir::new().unwrap();
    let marker = directory.path().display().to_string();
    let (mut client, owner) = spawn_nonblocking(managed_config_with_readiness(
        &directory,
        "never-ready",
        STUB_BYTES,
        Duration::from_secs(2),
    ))
    .unwrap();
    for _ in 0..3 {
        let HostEvent::Startup { .. } = recv(&mut client).await else {
            panic!("phases precede the stall");
        };
    }
    // The actor is now inside the readiness wait; shut down mid-stall.
    shutdown_joined(owner).await;
    match tokio::time::timeout(WITHIN, client.event_rx.recv()).await {
        Ok(None) => {}
        other => panic!("expected a closed plane after the cancelled startup, got {other:?}"),
    }
    assert_no_child(&marker).await;
}

#[tokio::test]
async fn engine_startup_failure_is_fatal_after_the_resolving_phase() {
    let _session = SESSION.lock().await;
    // The engine path through the same door: an unresolvable source fails
    // during ResolvingModel with a Fatal naming the resolution error —
    // the 5a fail-closed refusal's successor now that the managed variant
    // is hosted.
    let absent = std::env::temp_dir().join("yatima-host-absent-models-root");
    let source =
        ModelSource::from_args(None, Some("absent/model".into()), Some(absent), true, None)
            .expect("a well-formed source");
    let (mut client, owner) = spawn_nonblocking(HostConfig::engine(
        source,
        true,
        GenOpts::default(),
        None,
        None,
        None,
    ))
    .unwrap();
    let HostEvent::Startup {
        phase: StartupPhase::ResolvingModel,
    } = recv(&mut client).await
    else {
        panic!("resolution is reported before it fails");
    };
    let HostEvent::Fatal(message) = recv(&mut client).await else {
        panic!("expected Fatal");
    };
    assert!(message.contains("absent/model"), "{message}");
    shutdown_joined(owner).await;
}
