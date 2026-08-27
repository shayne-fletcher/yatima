//! Hermetic 5a boundary witness: a managed llama-server configuration
//! reaching today's actor fails closed — one deliberate Fatal naming stage
//! 5b — before any source resolution, verification, or process work.

use std::time::Duration;

use yatima_host::{spawn_nonblocking, HostBackendConfig, HostConfig, HostEvent};
use yatima_lib::{GenOpts, ModelProfile, ModelSource, ProfileBackend};

#[tokio::test]
async fn managed_config_fails_closed_before_resolution() {
    // The source is deliberately unresolvable (an absent repo under an
    // absent models root, offline): a resolution attempt would surface a
    // resolution error, so receiving the stage-5b refusal instead proves the
    // fail-closed branch precedes resolution — and that no verification or
    // process work existed to fail from.
    let absent_root = std::env::temp_dir().join("yatima-host-absent-models-root");
    let source = ModelSource::from_args(
        None,
        Some("absent/model".into()),
        Some(absent_root),
        true,
        Some("missing.gguf".into()),
    )
    .expect("a well-formed source");
    let profile = ModelProfile::builtin("muse-glimmer").expect("Muse profile is built in");
    let ProfileBackend::LlamaServer(server) = profile.backend.clone() else {
        panic!("Muse must pin the llama-server backend");
    };

    let mut handle = spawn_nonblocking(HostConfig {
        backend: HostBackendConfig::ManagedLlamaServer {
            source,
            profile: server,
        },
        opts: GenOpts::default(),
        format: None,
        system: None,
        model_label: Some("muse-glimmer".into()),
    })
    .expect("spawn the host thread");

    let event = tokio::time::timeout(Duration::from_secs(8), handle.event_rx.recv())
        .await
        .expect("the refusal must arrive well within the bound")
        .expect("the actor must send its first event before exiting");
    let HostEvent::Fatal(message) = event else {
        panic!("expected the deliberate Fatal, got {event:?}");
    };
    assert!(message.contains("not hosted yet (stage 5b)"), "{message}");
    assert!(
        message.contains("yatima chat --profile muse-glimmer"),
        "the refusal must name the supported path: {message}"
    );
    assert!(
        !message.contains("absent/model") && !message.contains("offline"),
        "a resolution error here would mean the branch ran too late: {message}"
    );
}
