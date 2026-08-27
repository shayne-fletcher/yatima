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
use yatima_host::{init_file_logging, resolve_host_model, spawn, HostModelChoices};
use yatima_lib::{GenOpts, Sampling};

use yatima_tui::app::{run_loop, App};

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

    // The terminal belongs to ratatui, so logs go to ~/.cache/yatima/tui.log;
    // tui-markdown warns per animation frame about glyphs it can't render, so
    // it stays quiet unless the filter names it.
    init_file_logging("tui", &["tui_markdown"])?;

    // Validate the profile/source choices through the shared host resolver
    // (PROFILE-2): every contradiction fails here, before the host thread
    // spawns and before the terminal is touched, so the error prints
    // normally. Acquisition itself happens inside the host thread.
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
    let config = resolved.into_host_config(base, args.system.clone());

    eprintln!("loading model… (first run may fetch weights)");
    let (handle, ready) = spawn(config).await?;

    let (mut terminal, enhanced) = enter_terminal()?;
    let app = App::new(handle.req_tx, handle.cancel, ready);
    let key_events = key_event_stream();
    let result = run_loop(&mut terminal, app, handle.event_rx, key_events).await;
    restore_terminal(&mut terminal, enhanced)?;
    result
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
