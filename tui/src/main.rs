//! The `yatima-tui` binary: parse args, load the model on the engine thread,
//! enter the terminal, run the event loop, and restore the terminal on exit.

use std::future::Future;
use std::io;
use std::path::{Path, PathBuf};
use std::time::Duration;

use anyhow::Result;
use chrono::Utc;
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
use yatima_drive::{start_recorder, RecorderHandle, RecorderOwner, TapeMeta, TapeRecord};
use yatima_host::{init_file_logging, resolve_host_model, spawn_nonblocking, HostModelChoices};
use yatima_lib::{GenOpts, Sampling};

use yatima_tui::app::{run_loop, App};

const RECORDER_CONTROL_WITHIN: Duration = Duration::from_secs(5);

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
    /// Grant read-only repository tools under this directory.
    #[arg(long)]
    root: Option<PathBuf>,
    /// Record this TUI session. With no DIR, writes under runs/.
    #[arg(long, num_args = 0..=1, value_name = "DIR")]
    tape: Option<Option<PathBuf>>,
}

fn tape_dir(choice: &Option<Option<PathBuf>>, utc_stamp: &str, pid: u32) -> Option<PathBuf> {
    match choice {
        None => None,
        Some(Some(dir)) => Some(dir.clone()),
        Some(None) => Some(PathBuf::from(format!("runs/{utc_stamp}-{pid}-tui"))),
    }
}

fn tape_notice(dir: &Path) -> String {
    let absolute = dir.canonicalize().unwrap_or_else(|_| {
        if dir.is_absolute() {
            dir.to_path_buf()
        } else {
            std::env::current_dir()
                .map(|cwd| cwd.join(dir))
                .unwrap_or_else(|_| dir.to_path_buf())
        }
    });
    format!("recording tape to {}", absolute.display())
}

async fn recorder_control_within<T, F>(what: &str, future: F) -> Result<T>
where
    F: Future<Output = Result<T>>,
{
    tokio::time::timeout(RECORDER_CONTROL_WITHIN, future)
        .await
        .map_err(|_| {
            anyhow::anyhow!("flight recorder {what} timed out after {RECORDER_CONTROL_WITHIN:?}")
        })?
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
    let config = resolved
        .into_host_config(base, args.system.clone())
        .with_repo_root(args.root.clone())?;
    // The rail's label until Ready carries the real facts: the profile name,
    // or the source argument as given (resolution happens in the host).
    let label = config
        .model_label()
        .map(str::to_string)
        .or_else(|| args.model.as_ref().map(|p| p.display().to_string()))
        .or_else(|| args.repo.clone())
        .unwrap_or_else(|| "local model".to_string());
    let utc_stamp = Utc::now().format("%Y%m%dT%H%M%SZ").to_string();
    let requested_tape = tape_dir(&args.tape, &utc_stamp, std::process::id());

    // Terminal ownership is established before the backend exists, and every
    // path below runs the full epilogue: restore the terminal, then consume
    // the one owner — shutdown cancels startup or the armed turn, awaits the
    // actor epilogue, and joins the backend thread (HOST-3). No `?` may
    // shortcut past the owner: a failed restore or a failed session still
    // joins, and every failure is reported, none silently dropped.
    let mut guard = TerminalGuard::enter(CrosstermTerm)?; // partial entry already unwound
    let mut host_owner = None;
    let mut tape_handle: Option<RecorderHandle> = None;
    let mut tape_owner: Option<RecorderOwner> = None;
    let result = match Terminal::new(CrosstermBackend::new(io::stdout())) {
        Ok(mut terminal) => {
            let recorder = match requested_tape {
                Some(dir) => {
                    let meta = TapeMeta {
                        origin: format!("yatima-tui {}", env!("CARGO_PKG_VERSION")),
                        model: label.clone(),
                        notes: std::collections::BTreeMap::from([(
                            "agent_max_steps".to_string(),
                            yatima_host::knobs::AGENT_MAX_STEPS.to_string(),
                        )]),
                    };
                    match recorder_control_within("startup", start_recorder(&dir, meta)).await {
                        Ok((handle, owner)) => Ok(Some((handle, owner, dir))),
                        Err(error) => Err(error.context("start the requested flight recorder")),
                    }
                }
                None => Ok(None),
            };
            match recorder {
                Err(error) => Err(error),
                Ok(recorder) => {
                    let tape_notice = recorder.as_ref().map(|(_, _, dir)| tape_notice(dir));
                    if let Some((handle, owner, _)) = recorder {
                        tape_handle = Some(handle);
                        tape_owner = Some(owner);
                    }
                    match spawn_nonblocking(config) {
                        Ok((client, owner)) => {
                            host_owner = Some(owner);
                            let (app_req_tx, app_req_rx) = std::sync::mpsc::channel();
                            let mut app = App::loading(app_req_tx, client.cancel, label);
                            if let Some(notice) = tape_notice {
                                app.push_entry(yatima_tui::app::Entry::Notice(notice));
                            }
                            run_loop(
                                &mut terminal,
                                app,
                                client.event_rx,
                                key_event_stream(),
                                client.req_tx,
                                app_req_rx,
                                tape_handle.clone(),
                            )
                            .await
                        }
                        // The thread never spawned: nothing to own, but the
                        // terminal and recorder still reach their epilogues.
                        Err(error) => Err(error),
                    }
                }
            }
        }
        Err(error) => Err(error.into()),
    };
    // Explicit restore captures errors; the guard's Drop remains the
    // panic-unwind safety net (a panic anywhere above still restores).
    let restored = guard.restore();
    let shutdown_record = match &tape_handle {
        Some(handle) => {
            recorder_control_within(
                "shutdown record",
                handle.enqueue(TapeRecord::Request(yatima_host::HostRequest::Shutdown)),
            )
            .await
        }
        None => Ok(()),
    };
    let joined = match host_owner {
        Some(owner) => owner.shutdown().await,
        None => Ok(()),
    };
    let disposition = if result.is_err() || restored.is_err() {
        "tui-error"
    } else if joined.is_err() {
        "backend-error"
    } else if shutdown_record.is_err() {
        "recorder-error"
    } else {
        "completed"
    };
    drop(tape_handle);
    let recorded = match tape_owner {
        Some(owner) => recorder_control_within("finish", owner.finish(disposition))
            .await
            .map(|_| ()),
        None => Ok(()),
    };
    combined_outcome(result, restored, shutdown_record, joined, recorded)
}

/// Fold every exit result into one report. The session outcome is primary;
/// terminal, tape-control, host-join, and recorder-finish failures are appended
/// as context rather than lost (HOST-3 / TAPE-1).
fn combined_outcome(
    session: Result<()>,
    restored: Result<()>,
    shutdown_record: Result<()>,
    joined: Result<()>,
    recorded: Result<()>,
) -> Result<()> {
    let mut outcome = session;
    for (label, secondary) in [
        ("restore terminal", restored),
        ("record shutdown", shutdown_record),
        ("shut down the backend owner", joined),
        ("finish the flight recorder", recorded),
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

    #[test]
    fn repository_root_is_an_explicit_tui_startup_choice() {
        let absent = Args::try_parse_from(["yatima-tui"]).unwrap();
        assert_eq!(absent.root, None);
        let present = Args::try_parse_from(["yatima-tui", "--root", "/tmp/repo"]).unwrap();
        assert_eq!(present.root, Some(PathBuf::from("/tmp/repo")));
    }

    #[test]
    fn tape_flag_supports_default_and_explicit_directories() {
        let absent = Args::try_parse_from(["yatima-tui"]).unwrap();
        assert_eq!(absent.tape, None);
        let default = Args::try_parse_from(["yatima-tui", "--tape"]).unwrap();
        assert_eq!(default.tape, Some(None));
        assert_eq!(
            tape_dir(&default.tape, "20260913T140000Z", 41),
            Some(PathBuf::from("runs/20260913T140000Z-41-tui"))
        );
        let explicit = Args::try_parse_from(["yatima-tui", "--tape", "/tmp/tape"]).unwrap();
        assert_eq!(explicit.tape, Some(Some(PathBuf::from("/tmp/tape"))));
        assert_eq!(
            tape_dir(&explicit.tape, "ignored", 0),
            Some(PathBuf::from("/tmp/tape"))
        );

        let run = tempfile::tempdir().unwrap();
        assert_eq!(
            tape_notice(run.path()),
            format!(
                "recording tape to {}",
                run.path().canonicalize().unwrap().display()
            )
        );
    }

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
