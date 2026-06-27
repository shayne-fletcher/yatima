//! Gated live smoke test: the engine actor loads a cached model, runs one turn,
//! and the three-plane protocol delivers Started → Fragment* → Done with a
//! coherent answer. YATIMA_E2E-gated; skips if the model isn't cached.

use yatima_lib::{is_model_present, model_dir, models_root, Channel, GenOpts, ModelId, Sampling};
use yatima_tui::engine_actor::{self, EngineConfig, EngineEvent, EngineRequest, TurnControl};

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn actor_runs_a_turn() -> anyhow::Result<()> {
    if std::env::var_os("YATIMA_E2E").is_none() {
        eprintln!("skipping e2e: set YATIMA_E2E=1 to run");
        return Ok(());
    }
    let repo = "Qwen/Qwen2.5-7B-Instruct";
    let dir = model_dir(&models_root(), &ModelId::parse(repo)?);
    if !is_model_present(&dir) {
        eprintln!("skip: {repo} not cached");
        return Ok(());
    }
    let config = EngineConfig {
        dir,
        cpu: false,
        opts: GenOpts {
            max_tokens: 32,
            sampling: Sampling::Greedy,
            ..Default::default()
        },
        format: None,
        system: None,
        model_label: repo.into(),
    };
    let mut handle = engine_actor::spawn(config).await?;
    handle.req_tx.send(EngineRequest::Submit {
        turn_id: 0,
        user: "What is 2 + 2? Reply with only the number.".into(),
        control: TurnControl::new(),
    })?;

    let mut answer = String::new();
    let mut started = false;
    loop {
        match handle.event_rx.recv().await.expect("event") {
            EngineEvent::Started { .. } => started = true,
            EngineEvent::Fragment {
                channel: Channel::Answer,
                text,
                ..
            } => answer.push_str(&text),
            EngineEvent::Fragment { .. } => {}
            EngineEvent::Done { answer: a, .. } => {
                answer = a;
                break;
            }
            EngineEvent::Error { message, .. } => panic!("engine error: {message}"),
        }
    }
    handle.req_tx.send(EngineRequest::Shutdown)?;
    eprintln!("actor answer: {answer:?}");
    assert!(started, "expected a Started event");
    assert!(answer.contains('4'), "expected 4, got {answer:?}");
    Ok(())
}
