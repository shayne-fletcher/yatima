//! `yatima-serve` — the event plane over a WebSocket (see lib.rs for the
//! bridge and the SRV-* registry). This binary is only wiring: resolve a
//! model config the same way the native frontends do, spawn the host, bind
//! where SRV-1 allows, serve.

use std::path::PathBuf;

use anyhow::Result;
use clap::Parser;
use yatima_host::{init_stderr_logging, spawn_nonblocking, HostConfig};
use yatima_lib::{GenOpts, ModelProfile, ModelSource, Sampling};
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

/// Mirror of the GUI/TUI config resolution (a shared-host candidate once a
/// third copy exists; today the duplication is the cheaper debt).
fn resolve(args: &Args) -> Result<HostConfig> {
    let profile = match &args.profile {
        Some(name) => Some(ModelProfile::builtin(name).ok_or_else(|| {
            anyhow::anyhow!(
                "unknown profile {name:?}; built-ins: {:?}",
                ModelProfile::BUILTIN_NAMES
            )
        })?),
        None => None,
    };

    let (dir, label) = match &profile {
        Some(p) => (p.to_source(args.offline)?.resolve()?, p.name.clone()),
        None => {
            let dir = ModelSource::from_args(
                args.model.clone(),
                args.repo.clone(),
                args.models_dir.clone(),
                args.offline,
                args.gguf.clone(),
            )?
            .resolve()?;
            let label = dir.display().to_string();
            (dir, label)
        }
    };

    let base = GenOpts {
        max_tokens: args.max_tokens,
        sampling: Sampling::nucleus(args.temperature, args.top_p, args.seed),
        ..Default::default()
    };
    let opts = match &profile {
        Some(p) => p.apply_gen_overrides(base),
        None => base,
    };
    let format = profile.as_ref().and_then(ModelProfile::format);

    Ok(HostConfig {
        dir,
        cpu: args.cpu,
        opts,
        format,
        system: args.system.clone(),
        model_label: label,
    })
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
    let handle = spawn_nonblocking(config)?;
    let bridge = Bridge::new(handle);

    eprintln!(
        "serving on http://{bind}/ (ws at /ws{})",
        if args.static_dir.is_some() {
            ", client at /"
        } else {
            "; no client bundle"
        }
    );
    axum::serve(listener, bridge.router(args.static_dir)).await?;
    Ok(())
}
