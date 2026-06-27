//! The `yatima-tui` binary: parse args, load the model on the engine thread,
//! enter the terminal, run the event loop, and restore the terminal on exit.

use std::io::{self, Stdout};
use std::path::PathBuf;

use anyhow::Result;
use clap::Parser;
use crossterm::event::{Event, EventStream};
use crossterm::execute;
use crossterm::terminal::{
    disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen,
};
use futures::stream::Stream;
use ratatui::backend::CrosstermBackend;
use ratatui::Terminal;
use yatima_lib::{GenOpts, ModelProfile, ModelSource, Sampling};

use yatima_tui::app::{run_loop, App};
use yatima_tui::engine_actor::{self, EngineConfig};

/// Interactive terminal chat over a local model.
#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// A built-in model profile (e.g. `kimi-dev`, `deepseek-r1`): sets the model,
    /// chat format, and generation defaults. Replaces `--model`/`--repo`.
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
    #[arg(long, default_value_t = 1024)]
    max_tokens: usize,
    #[arg(long, default_value_t = 0.0)]
    temperature: f64,
    #[arg(long, default_value_t = 0)]
    seed: u64,
    /// Force CPU instead of the GPU.
    #[arg(long)]
    cpu: bool,
    /// Don't auto-fetch a missing model; error instead.
    #[arg(long)]
    offline: bool,
}

#[tokio::main(flavor = "multi_thread")]
async fn main() -> Result<()> {
    let args = Args::parse();

    // Resolve the profile (if any) → model dir + format + gen defaults. Done
    // before touching the terminal so an error prints normally.
    let profile = match &args.profile {
        Some(name) => Some(ModelProfile::builtin(name).ok_or_else(|| {
            anyhow::anyhow!(
                "unknown profile {name:?}; built-ins: {:?}",
                ModelProfile::BUILTIN_NAMES
            )
        })?),
        None => None,
    };

    let (dir, model_label) = match &profile {
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
            (dir.clone(), dir.display().to_string())
        }
    };

    let base = GenOpts {
        max_tokens: args.max_tokens,
        sampling: Sampling::from_temperature(args.temperature, args.seed),
        ..Default::default()
    };
    let opts = match &profile {
        Some(p) => p.apply_gen_overrides(base),
        None => base,
    };
    let format = profile.as_ref().and_then(ModelProfile::format);

    let config = EngineConfig {
        dir,
        cpu: args.cpu,
        opts,
        format,
        system: args.system.clone(),
        model_label,
    };

    eprintln!("loading model… (first run may fetch weights)");
    let handle = engine_actor::spawn(config).await?;

    let mut terminal = enter_terminal()?;
    let app = App::new(handle.req_tx, handle.ready);
    let key_events = key_event_stream();
    let result = run_loop(&mut terminal, app, handle.event_rx, key_events).await;
    restore_terminal(&mut terminal)?;
    result
}

/// The crossterm key-event stream, dropping non-key/errored events upstream of
/// the loop's matcher (which only acts on key presses anyway).
fn key_event_stream() -> impl Stream<Item = io::Result<Event>> + Unpin {
    EventStream::new()
}

fn enter_terminal() -> Result<Terminal<CrosstermBackend<Stdout>>> {
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen)?;
    Ok(Terminal::new(CrosstermBackend::new(stdout))?)
}

fn restore_terminal(terminal: &mut Terminal<CrosstermBackend<Stdout>>) -> Result<()> {
    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
    terminal.show_cursor()?;
    Ok(())
}
