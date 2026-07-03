//! The `yatima-tui` binary: parse args, load the model on the engine thread,
//! enter the terminal, run the event loop, and restore the terminal on exit.

use std::io::{self, Stdout};
use std::path::PathBuf;

use anyhow::Result;
use clap::Parser;
use crossterm::event::{
    Event, EventStream, KeyboardEnhancementFlags, PopKeyboardEnhancementFlags,
    PushKeyboardEnhancementFlags,
};
use crossterm::execute;
use crossterm::terminal::{
    disable_raw_mode, enable_raw_mode, supports_keyboard_enhancement, EnterAlternateScreen,
    LeaveAlternateScreen,
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
    /// Nucleus (top-p) sampling cutoff; omit for the full distribution. A profile
    /// may set its own (e.g. reasoning profiles use 0.95).
    #[arg(long)]
    top_p: Option<f64>,
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

    init_file_logging()?;

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
        sampling: Sampling::nucleus(args.temperature, args.top_p, args.seed),
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

    let (mut terminal, enhanced) = enter_terminal()?;
    let app = App::new(handle.req_tx, handle.ready);
    let key_events = key_event_stream();
    let result = run_loop(&mut terminal, app, handle.event_rx, key_events).await;
    restore_terminal(&mut terminal, enhanced)?;
    result
}

/// Install a file-writing tracing subscriber when `$YATIMA_LOG` is set (its
/// value is the filter, e.g. `debug` or `yatima_lib=trace`) — OBS-1: the lib
/// emits spans/events, the host decides where they go. The terminal belongs
/// to ratatui, so logs append to `~/.cache/yatima/tui.log`; `debug` shows
/// each tool call with its full args JSON and outcome, `trace` adds whole
/// prompts and completions. No env var, no subscriber, no cost.
fn init_file_logging() -> Result<()> {
    if std::env::var_os("YATIMA_LOG").is_none() {
        return Ok(());
    }
    let dir = std::env::home_dir()
        .map(|home| home.join(".cache/yatima"))
        .unwrap_or_else(std::env::temp_dir);
    std::fs::create_dir_all(&dir)?;
    let path = dir.join("tui.log");
    let file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)?;
    // A bare level ("debug") scopes to yatima: third-party internals
    // (html5ever narrating every HTML token, hyper, wgpu) drown the log at
    // debug, and the question the log answers is "what is yatima doing".
    // A spec with '='/',' is honored verbatim for when those internals are
    // exactly what's wanted. tui-markdown warns per animation frame about
    // anything it can't render, so it stays quiet unless named.
    let value = std::env::var("YATIMA_LOG").unwrap_or_default();
    let mut spec = if value.contains('=') || value.contains(',') {
        value
    } else {
        format!("warn,yatima_lib={value},yatima_tui={value},yatima_text={value}")
    };
    if !spec.contains("tui_markdown") {
        spec.push_str(",tui_markdown=error");
    }
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::new(spec))
        .with_writer(file)
        .with_ansi(false)
        .init();
    eprintln!("logging to {}", path.display());
    Ok(())
}

/// The crossterm key-event stream, dropping non-key/errored events upstream of
/// the loop's matcher (which only acts on key presses anyway).
fn key_event_stream() -> impl Stream<Item = io::Result<Event>> + Unpin {
    EventStream::new()
}

/// Enter raw mode and the alternate screen. Returns whether the kitty keyboard
/// protocol was enabled: where the terminal supports it (kitty, ghostty, wezterm,
/// foot, iTerm2 with the setting), disambiguating escape codes makes modified
/// Enter — Shift+Enter / Alt+Enter for a newline — arrive as distinct keys.
/// Apple Terminal does not support it (returns false); there, enable "Use Option
/// as Meta key" so Option+Return is delivered as Alt+Enter.
fn enter_terminal() -> Result<(Terminal<CrosstermBackend<Stdout>>, bool)> {
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen)?;
    let enhanced = supports_keyboard_enhancement().unwrap_or(false);
    if enhanced {
        execute!(
            stdout,
            PushKeyboardEnhancementFlags(KeyboardEnhancementFlags::DISAMBIGUATE_ESCAPE_CODES)
        )?;
    }
    Ok((Terminal::new(CrosstermBackend::new(stdout))?, enhanced))
}

fn restore_terminal(
    terminal: &mut Terminal<CrosstermBackend<Stdout>>,
    enhanced: bool,
) -> Result<()> {
    if enhanced {
        execute!(terminal.backend_mut(), PopKeyboardEnhancementFlags)?;
    }
    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
    terminal.show_cursor()?;
    Ok(())
}
