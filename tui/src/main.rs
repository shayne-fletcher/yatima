//! The `yatima-tui` binary: parse args, load the model on the engine thread,
//! enter the terminal, run the event loop, and restore the terminal on exit.

use std::io;
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
use yatima_host::{init_file_logging, resolve_host_model, spawn_nonblocking, HostModelChoices};
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
    // The rail's label until Ready carries the real facts: the profile name,
    // or the source argument as given (resolution happens in the host).
    let label = config
        .model_label()
        .map(str::to_string)
        .or_else(|| args.model.as_ref().map(|p| p.display().to_string()))
        .or_else(|| args.repo.clone())
        .unwrap_or_else(|| "local model".to_string());

    // Terminal ownership is established before the backend exists, and every
    // path below runs the full epilogue: restore the terminal, then consume
    // the one owner — shutdown cancels startup or the armed turn, awaits the
    // actor epilogue, and joins the backend thread (HOST-3). No `?` may
    // shortcut past the owner: a failed restore or a failed session still
    // joins, and every failure is reported, none silently dropped.
    let mut guard = TerminalGuard::enter(CrosstermTerm)?; // partial entry already unwound
    let (result, owner) = match Terminal::new(CrosstermBackend::new(io::stdout())) {
        Ok(mut terminal) => match spawn_nonblocking(config) {
            Ok((client, owner)) => {
                let app = App::loading(client.req_tx, client.cancel, label);
                let key_events = key_event_stream();
                let result = run_loop(&mut terminal, app, client.event_rx, key_events).await;
                (result, Some(owner))
            }
            // The thread never spawned: nothing to own, but the terminal
            // must still be restored before the error prints.
            Err(error) => (Err(error), None),
        },
        Err(error) => (Err(error.into()), None),
    };
    // Explicit restore captures errors; the guard's Drop remains the
    // panic-unwind safety net (a panic anywhere above still restores).
    let restored = guard.restore();
    let joined = match owner {
        Some(owner) => owner.shutdown().await,
        None => Ok(()),
    };
    combined_outcome(result, restored, joined)
}

/// Fold the session's three exit results into one report: the session
/// outcome is primary; a failed terminal restore or a failed joined shutdown
/// is appended as context rather than lost (and stands alone when the
/// session itself succeeded).
fn combined_outcome(session: Result<()>, restored: Result<()>, joined: Result<()>) -> Result<()> {
    let mut outcome = session;
    for (label, secondary) in [
        ("restore terminal", restored),
        ("shut down the backend owner", joined),
    ] {
        outcome = match (outcome, secondary) {
            (Ok(()), Ok(())) => Ok(()),
            (Ok(()), Err(error)) => Err(error.context(label)),
            (Err(primary), Ok(())) => Err(primary),
            (Err(primary), Err(error)) => {
                Err(primary.context(format!("{label} also failed: {error:#}")))
            }
        };
    }
    outcome
}

/// The crossterm key-event stream, dropping non-key/errored events upstream of
/// the loop's matcher (which only acts on key presses anyway).
fn key_event_stream() -> impl Stream<Item = io::Result<Event>> + Unpin {
    EventStream::new()
}

/// The terminal transitions, abstracted so the guard's ordering logic is
/// unit-testable with injected failures; the crossterm impl is the only
/// integration-bound part.
trait TermOps {
    fn enable_raw(&mut self) -> Result<()>;
    fn enter_alternate(&mut self) -> Result<()>;
    /// Returns whether the keyboard-enhancement flags were pushed (the
    /// terminal may simply not support them — that is not a failure).
    fn push_enhancement(&mut self) -> Result<bool>;
    fn pop_enhancement(&mut self) -> Result<()>;
    fn leave_alternate(&mut self) -> Result<()>;
    fn disable_raw(&mut self) -> Result<()>;
    fn show_cursor(&mut self) -> Result<()>;
}

/// Which transitions have actually succeeded — the guard's ledger. Every
/// inverse of a recorded transition is attempted on the way out, wherever
/// the exit happens (a partial entry unwinds; a failed restore step never
/// prevents the later steps).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct TermState {
    raw: bool,
    alternate: bool,
    enhanced: bool,
}

/// Enter raw mode, the alternate screen, and (where supported) the kitty
/// keyboard enhancement, recording each success. On a mid-entry failure the
/// transitions already made are unwound before the error returns — raw mode
/// is never left enabled by a failed entry.
fn enter_guarded(ops: &mut impl TermOps) -> Result<TermState> {
    let mut state = TermState::default();
    let entered = (|| -> Result<()> {
        ops.enable_raw()?;
        state.raw = true;
        ops.enter_alternate()?;
        state.alternate = true;
        state.enhanced = ops.push_enhancement()?;
        Ok(())
    })();
    match entered {
        Ok(()) => Ok(state),
        Err(error) => match restore_guarded(ops, state) {
            Ok(()) => Err(error),
            Err(unwound) => Err(error.context(format!("unwind also failed: {unwound:#}"))),
        },
    }
}

/// Attempt every inverse the state records — pop enhancement, leave the
/// alternate screen, disable raw mode, show the cursor — regardless of
/// earlier failures, accumulating errors instead of stopping at the first:
/// a failed enhancement pop must never leave the user's shell in raw mode.
fn restore_guarded(ops: &mut impl TermOps, state: TermState) -> Result<()> {
    let mut failures: Vec<String> = Vec::new();
    if state.enhanced {
        if let Err(error) = ops.pop_enhancement() {
            failures.push(format!("pop keyboard enhancement: {error:#}"));
        }
    }
    if state.alternate {
        if let Err(error) = ops.leave_alternate() {
            failures.push(format!("leave alternate screen: {error:#}"));
        }
    }
    if state.raw {
        if let Err(error) = ops.disable_raw() {
            failures.push(format!("disable raw mode: {error:#}"));
        }
    }
    if let Err(error) = ops.show_cursor() {
        failures.push(format!("show cursor: {error:#}"));
    }
    if failures.is_empty() {
        Ok(())
    } else {
        Err(anyhow::anyhow!(
            "terminal restoration incomplete: {}",
            failures.join("; ")
        ))
    }
}

/// RAII over the guarded transitions: `Drop` restores best-effort during
/// panic unwinding (errors can only be logged from a destructor), while the
/// ordinary path calls [`restore`](TerminalGuard::restore) explicitly to
/// capture accumulated restoration errors — the same drop-is-fallback,
/// explicit-is-witness split as `HostOwner`.
struct TerminalGuard<T: TermOps> {
    ops: T,
    state: TermState,
    armed: bool,
}

impl<T: TermOps> TerminalGuard<T> {
    /// Enter the terminal; a partial entry is already unwound by
    /// [`enter_guarded`] before the error returns.
    fn enter(mut ops: T) -> Result<TerminalGuard<T>> {
        let state = enter_guarded(&mut ops)?;
        Ok(TerminalGuard {
            ops,
            state,
            armed: true,
        })
    }

    /// The explicit, error-carrying restore: attempts every inverse and
    /// disarms the drop fallback.
    fn restore(&mut self) -> Result<()> {
        self.armed = false;
        restore_guarded(&mut self.ops, self.state)
    }
}

impl<T: TermOps> Drop for TerminalGuard<T> {
    fn drop(&mut self) {
        if self.armed {
            // Unwinding (or a forgotten restore): put the terminal back so
            // the panic message is readable; failures here have nowhere to
            // go but the log.
            if let Err(error) = restore_guarded(&mut self.ops, self.state) {
                eprintln!("terminal restore during unwind failed: {error:#}");
            }
        }
    }
}

/// The real transitions: crossterm over stdout.
#[derive(Default)]
struct CrosstermTerm;

impl TermOps for CrosstermTerm {
    fn enable_raw(&mut self) -> Result<()> {
        Ok(enable_raw_mode()?)
    }
    fn enter_alternate(&mut self) -> Result<()> {
        Ok(execute!(io::stdout(), EnterAlternateScreen)?)
    }
    fn push_enhancement(&mut self) -> Result<bool> {
        // Where the terminal supports the kitty keyboard protocol
        // (kitty, ghostty, wezterm, foot, iTerm2 with the setting),
        // disambiguating escape codes make modified Enter — Shift+Enter /
        // Alt+Enter for a newline — arrive as distinct keys. Apple Terminal
        // does not support it; there, enable "Use Option as Meta key" so
        // Option+Return is delivered as Alt+Enter.
        if supports_keyboard_enhancement().unwrap_or(false) {
            execute!(
                io::stdout(),
                PushKeyboardEnhancementFlags(KeyboardEnhancementFlags::DISAMBIGUATE_ESCAPE_CODES)
            )?;
            Ok(true)
        } else {
            Ok(false)
        }
    }
    fn pop_enhancement(&mut self) -> Result<()> {
        Ok(execute!(io::stdout(), PopKeyboardEnhancementFlags)?)
    }
    fn leave_alternate(&mut self) -> Result<()> {
        Ok(execute!(io::stdout(), LeaveAlternateScreen)?)
    }
    fn disable_raw(&mut self) -> Result<()> {
        Ok(disable_raw_mode()?)
    }
    fn show_cursor(&mut self) -> Result<()> {
        execute!(io::stdout(), crossterm::cursor::Show)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::sync::{Arc, Mutex};

    /// A fake terminal recording call order (into a shared log, so a
    /// dropped guard's calls remain observable) and injecting failures.
    #[derive(Default)]
    struct FakeTerm {
        calls: Arc<Mutex<Vec<&'static str>>>,
        fail: Vec<&'static str>,
        supports_enhancement: bool,
    }

    impl FakeTerm {
        fn op(&mut self, name: &'static str) -> Result<()> {
            self.calls.lock().unwrap().push(name);
            if self.fail.contains(&name) {
                anyhow::bail!("{name} failed (injected)");
            }
            Ok(())
        }

        fn log(&self) -> Vec<&'static str> {
            self.calls.lock().unwrap().clone()
        }
    }

    impl TermOps for FakeTerm {
        fn enable_raw(&mut self) -> Result<()> {
            self.op("enable_raw")
        }
        fn enter_alternate(&mut self) -> Result<()> {
            self.op("enter_alternate")
        }
        fn push_enhancement(&mut self) -> Result<bool> {
            self.op("push_enhancement")?;
            Ok(self.supports_enhancement)
        }
        fn pop_enhancement(&mut self) -> Result<()> {
            self.op("pop_enhancement")
        }
        fn leave_alternate(&mut self) -> Result<()> {
            self.op("leave_alternate")
        }
        fn disable_raw(&mut self) -> Result<()> {
            self.op("disable_raw")
        }
        fn show_cursor(&mut self) -> Result<()> {
            self.op("show_cursor")
        }
    }

    #[test]
    fn partial_entry_unwinds_what_succeeded() {
        // Raw mode succeeded, the alternate screen failed: the guard must
        // disable raw mode before returning the error — a failed entry
        // never leaves the user's shell raw.
        let mut term = FakeTerm {
            fail: vec!["enter_alternate"],
            supports_enhancement: true,
            ..Default::default()
        };
        let error = enter_guarded(&mut term).unwrap_err();
        assert!(
            format!("{error:#}").contains("enter_alternate"),
            "{error:#}"
        );
        assert!(
            term.log().contains(&"disable_raw"),
            "raw mode must be unwound: {:?}",
            term.log()
        );
        assert!(
            !term.log().contains(&"pop_enhancement"),
            "never invert a transition that did not happen: {:?}",
            term.log()
        );
    }

    #[test]
    fn restore_attempts_every_inverse_despite_failures() {
        // upholds: the guard's whole point — a failed enhancement pop must
        // not prevent raw-mode disablement, alternate-screen exit, or
        // cursor restoration, and the accumulated error names each failure.
        let mut term = FakeTerm {
            fail: vec!["pop_enhancement", "leave_alternate"],
            supports_enhancement: true,
            ..Default::default()
        };
        let state = TermState {
            raw: true,
            alternate: true,
            enhanced: true,
        };
        let error = restore_guarded(&mut term, state).unwrap_err().to_string();
        assert!(
            term.log()
                == vec![
                    "pop_enhancement",
                    "leave_alternate",
                    "disable_raw",
                    "show_cursor"
                ],
            "every inverse attempted in order: {:?}",
            term.log()
        );
        assert!(error.contains("pop keyboard enhancement"), "{error}");
        assert!(error.contains("leave alternate screen"), "{error}");
    }

    #[test]
    fn guard_restores_during_panic_unwinding() {
        // upholds: the RAII net — a panic between entry and the explicit
        // restore still puts the terminal back (Drop runs on unwind), so a
        // panic message is never printed into a raw-mode alternate screen.
        let calls = Arc::new(Mutex::new(Vec::new()));
        let term = FakeTerm {
            calls: Arc::clone(&calls),
            supports_enhancement: true,
            ..Default::default()
        };
        let panicked = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _guard = TerminalGuard::enter(term).unwrap();
            panic!("mid-session panic");
        }));
        assert!(panicked.is_err());
        let log = calls.lock().unwrap().clone();
        assert!(
            log.contains(&"disable_raw") && log.contains(&"leave_alternate"),
            "unwind must restore: {log:?}"
        );
    }

    #[test]
    fn explicit_restore_disarms_the_drop_fallback() {
        // One restore, not two: the explicit path disarms Drop, so the
        // inverses run exactly once.
        let calls = Arc::new(Mutex::new(Vec::new()));
        let term = FakeTerm {
            calls: Arc::clone(&calls),
            supports_enhancement: false,
            ..Default::default()
        };
        let mut guard = TerminalGuard::enter(term).unwrap();
        guard.restore().unwrap();
        drop(guard);
        let log = calls.lock().unwrap().clone();
        let restores = log.iter().filter(|c| **c == "disable_raw").count();
        assert_eq!(restores, 1, "Drop must not restore twice: {log:?}");
    }

    #[test]
    fn clean_entry_records_exactly_what_happened() {
        let mut term = FakeTerm {
            supports_enhancement: false,
            ..Default::default()
        };
        let state = enter_guarded(&mut term).unwrap();
        assert_eq!(
            state,
            TermState {
                raw: true,
                alternate: true,
                enhanced: false,
            }
        );
        // Restoring that state never pops an enhancement it did not push.
        let mut term = FakeTerm::default();
        restore_guarded(&mut term, state).unwrap();
        assert!(!term.log().contains(&"pop_enhancement"), "{:?}", term.log());
    }
}
