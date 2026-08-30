//! `yatima-serve` — the event plane over a WebSocket (see lib.rs for the
//! bridge and the SRV-* registry). This binary is only wiring: resolve a
//! model config the same way the native frontends do, spawn the host, bind
//! where SRV-1 allows, serve.

use std::path::PathBuf;

use anyhow::Result;
use clap::Parser;
use std::sync::Arc;
use yatima_host::{
    init_stderr_logging, resolve_host_model, spawn_nonblocking, HostConfig, HostModelChoices,
};

use yatima_lib::{GenOpts, Sampling};
use yatima_serve::{combined_outcome, serve_until};
use yatima_serve::{validate_bind, Bridge};

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Address to bind — explicit and specific, never 0.0.0.0 (SRV-1);
    /// loopback for local use, the tailnet address for a second device.
    #[arg(long)]
    bind: String,
    /// Directory of the browser client bundle to serve at `/` (trunk's
    /// `dist/`); without it, serve is WebSocket-only.
    #[arg(long)]
    static_dir: Option<PathBuf>,
    /// A built-in model profile (e.g. `qwq`, `qwen32b`).
    #[arg(long)]
    profile: Option<String>,
    /// Explicit model directory.
    #[arg(long)]
    model: Option<PathBuf>,
    /// Repository id, resolved under the models root.
    #[arg(long)]
    repo: Option<String>,
    /// Override the models root (else $YATIMA_MODELS_DIR / XDG cache).
    #[arg(long)]
    models_dir: Option<PathBuf>,
    /// With `--repo`, fetch this single GGUF file (quantized).
    #[arg(long)]
    gguf: Option<String>,
    /// Optional system instruction (applies for the whole session).
    #[arg(long)]
    system: Option<String>,
    /// Maximum tokens generated per turn.
    #[arg(long, default_value_t = 1024)]
    max_tokens: usize,
    /// Sampling temperature; 0.0 is greedy (deterministic).
    #[arg(long, default_value_t = 0.0)]
    temperature: f64,
    /// Nucleus (top-p) sampling cutoff; omit for the full distribution.
    #[arg(long)]
    top_p: Option<f64>,
    /// Sampling RNG seed (reproducible when temperature > 0).
    #[arg(long, default_value_t = 0)]
    seed: u64,
    /// Force CPU instead of the GPU.
    #[arg(long)]
    cpu: bool,
    /// Don't auto-fetch a missing model; error instead.
    #[arg(long)]
    offline: bool,
}

/// The shared host resolver (PROFILE-2), then the serve-shaped config. The
/// third local copy of this resolution is gone; contradictions fail here,
/// before the bind and the host thread. Acquisition happens inside the host
/// thread, after the listener is bound.
fn resolve(args: &Args) -> Result<HostConfig> {
    let resolved = resolve_host_model(HostModelChoices {
        profile: args.profile.clone(),
        model: args.model.clone(),
        repo: args.repo.clone(),
        models_dir: args.models_dir.clone(),
        gguf: args.gguf.clone(),
        cpu: args.cpu,
        offline: args.offline,
    })?;
    let base = GenOpts {
        max_tokens: args.max_tokens,
        sampling: Sampling::nucleus(args.temperature, args.top_p, args.seed),
        ..Default::default()
    };
    Ok(resolved.into_host_config(base, args.system.clone()))
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();
    // serve owns no screen: the console is the operator's view, so logs go
    // to stderr, always ($YATIMA_LOG raises the level — `debug` shows tool
    // calls with args, `trace` adds whole prompts).
    init_stderr_logging("serve")?;
    let bind = validate_bind(&args.bind)?; // SRV-1 before any model load
    let config = resolve(&args)?;

    // Bind before loading the model so an EADDRINUSE fails fast, not after a
    // full (possibly weight-fetching) load.
    let listener = tokio::net::TcpListener::bind(bind).await?;

    eprintln!("loading model… (first run may fetch weights)");
    // The bridge gets only the movable client planes; this main retains the
    // one owner and shuts it down (joined) when the server loop ends. The
    // graceful-shutdown signal wiring is stage 5c.
    let (client, owner) = spawn_nonblocking(config)?;
    let bridge = Bridge::new(client);

    eprintln!(
        "serving on http://{bind}/ (ws at /ws{})",
        if args.static_dir.is_some() {
            ", client at /"
        } else {
            "; no client bundle"
        }
    );
    // The closing edge, end to end: the signal (Ctrl-C or SIGTERM) stops
    // acceptance and closes live sessions under the drain bound inside
    // `serve_until`; the bridge close is re-flipped here so a *spontaneous*
    // server error takes the same tail (idempotent — a second flip is a
    // no-op); then, sequentially, the owner's joined shutdown. No `?` may
    // bypass either step, and both failures are reported if both occur
    // (serving primary).
    let served = serve_until(
        listener,
        Arc::clone(&bridge),
        args.static_dir,
        shutdown_signal(),
    )
    .await;
    bridge.close();
    let joined = owner.shutdown().await;
    combined_outcome(served, joined)
}

/// Resolves when the operator asks serve to stop: Ctrl-C, or SIGTERM where
/// the platform has it. Production's injectable-shutdown supplier for
/// [`serve_until`].
async fn shutdown_signal() {
    let ctrl_c = async {
        let _ = tokio::signal::ctrl_c().await;
    };
    #[cfg(unix)]
    let terminate = async {
        match tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()) {
            Ok(mut signal) => {
                signal.recv().await;
            }
            Err(_) => std::future::pending::<()>().await,
        }
    };
    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();
    tokio::select! {
        _ = ctrl_c => {}
        _ = terminate => {}
    }
}
