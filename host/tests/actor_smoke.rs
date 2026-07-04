//! Gated live smoke test: the host loads a cached model, runs one turn, and the
//! event plane delivers Started → Fragment* → Done with a coherent answer, then
//! a cancel cuts a long turn short. YATIMA_E2E-gated; skips if uncached.

use yatima_host::{spawn, Channel, HostConfig, HostEvent, HostRequest, StopKind};
use yatima_lib::{is_model_present, model_dir, models_root, GenOpts, ModelId, Sampling};

fn cached_dir() -> Option<std::path::PathBuf> {
    if std::env::var_os("YATIMA_E2E").is_none() {
        eprintln!("skipping e2e: set YATIMA_E2E=1 to run");
        return None;
    }
    let repo = "Qwen/Qwen2.5-7B-Instruct";
    let dir = model_dir(&models_root(), &ModelId::parse(repo).ok()?);
    if !is_model_present(&dir) {
        eprintln!("skip: {repo} not cached");
        return None;
    }
    Some(dir)
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn host_runs_a_turn() -> anyhow::Result<()> {
    let Some(dir) = cached_dir() else {
        return Ok(());
    };
    let config = HostConfig {
        dir,
        cpu: false,
        opts: GenOpts {
            max_tokens: 32,
            sampling: Sampling::Greedy,
            ..Default::default()
        },
        format: None,
        system: None,
        model_label: "Qwen/Qwen2.5-7B-Instruct".into(),
    };
    let (mut handle, _info) = spawn(config).await?;
    handle.req_tx.send(HostRequest::Submit {
        turn_id: 0,
        text: "What is 2 + 2? Reply with only the number.".into(),
    })?;

    let mut answer = String::new();
    let mut started = false;
    loop {
        match handle.event_rx.recv().await.expect("event") {
            HostEvent::Started { .. } => started = true,
            HostEvent::Fragment {
                channel: Channel::Answer,
                text,
                ..
            } => answer.push_str(&text),
            HostEvent::Done { .. } => break,
            HostEvent::Error { message, .. } => panic!("engine error: {message}"),
            _ => {}
        }
    }
    handle.req_tx.send(HostRequest::Shutdown)?;
    eprintln!("host answer: {answer:?}");
    assert!(started, "expected a Started event");
    assert!(answer.contains('4'), "expected 4, got {answer:?}");
    Ok(())
}

/// The cancel gate end to end: a turn flipped via [`yatima_host::CancelGate`]
/// stops in flight with [`StopKind::Stopped`], well before its `max_tokens`
/// budget. Drives a long generation, cancels after the first fragment.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn host_cancels_a_turn_in_flight() -> anyhow::Result<()> {
    let Some(dir) = cached_dir() else {
        return Ok(());
    };
    let config = HostConfig {
        dir,
        cpu: false,
        opts: GenOpts {
            max_tokens: 2048, // long enough that a cancel must cut it short
            sampling: Sampling::Greedy,
            ..Default::default()
        },
        format: None,
        system: None,
        model_label: "Qwen/Qwen2.5-7B-Instruct".into(),
    };
    let (mut handle, _info) = spawn(config).await?;
    handle.req_tx.send(HostRequest::Submit {
        turn_id: 0,
        text: "Write a long, detailed essay about the history of computing.".into(),
    })?;

    let stop = loop {
        match handle.event_rx.recv().await.expect("event") {
            // First token → cancel the in-flight turn through the gate.
            HostEvent::Fragment { .. } => handle.cancel.cancel(0),
            HostEvent::Done { stop, .. } => break stop,
            HostEvent::Error { message, .. } => panic!("engine error: {message}"),
            _ => {}
        }
    };
    handle.req_tx.send(HostRequest::Shutdown)?;
    assert_eq!(
        stop,
        StopKind::Stopped,
        "a cancelled turn must end as Stopped, not run to max_tokens"
    );
    Ok(())
}
