//! Logging destinations, one per frontend shape.
//!
//! OBS-1: the lib emits spans/events, the host decides where they go. A
//! frontend that owns its screen (ratatui's terminal, egui's window) cannot
//! share it with logs, so those log to a per-host file under
//! `~/.cache/yatima`, behind `$YATIMA_LOG` ([`init_file_logging`]). serve
//! owns no screen — its console *is* the operator's view of the session —
//! so it logs to stderr, always ([`init_stderr_logging`]).

use anyhow::Result;

/// Install a file-writing tracing subscriber when `$YATIMA_LOG` is set (its
/// value is the filter, e.g. `debug` or `yatima_lib=trace`). Logs append to
/// `~/.cache/yatima/{file_stem}.log` — separate files so two frontends never
/// interleave. `debug` shows each tool call with its full args JSON and
/// outcome, `trace` adds whole prompts and completions. No env var, no
/// subscriber, no cost.
///
/// `default_quiets` names crates whose debug/trace chatter drowns the log and
/// is silenced to `error` unless the caller's filter mentions them (the TUI's
/// `tui_markdown`, which warns per animation frame about glyphs it can't
/// render, is the motivating case).
pub fn init_file_logging(file_stem: &str, default_quiets: &[&str]) -> Result<()> {
    if std::env::var_os("YATIMA_LOG").is_none() {
        return Ok(());
    }
    let dir = std::env::home_dir()
        .map(|home| home.join(".cache/yatima"))
        .unwrap_or_else(std::env::temp_dir);
    std::fs::create_dir_all(&dir)?;
    let path = dir.join(format!("{file_stem}.log"));
    let file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)?;
    // A bare level ("debug") scopes to yatima: third-party internals
    // (html5ever narrating every HTML token, hyper, wgpu) drown the log at
    // debug, and the question the log answers is "what is yatima doing". A
    // spec with '='/',' is honored verbatim for when those internals are
    // exactly what's wanted.
    let value = std::env::var("YATIMA_LOG").unwrap_or_default();
    let mut spec = if value.contains('=') || value.contains(',') {
        value
    } else {
        format!(
            "warn,yatima_lib={value},yatima_host={value},\
             yatima_{file_stem}={value},yatima_text={value}"
        )
    };
    for quiet in default_quiets {
        if !spec.contains(quiet) {
            spec.push_str(&format!(",{quiet}=error"));
        }
    }
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::new(spec))
        .with_writer(file)
        .with_ansi(false)
        .init();
    eprintln!("logging to {}", path.display());
    Ok(())
}

/// The screenless frontend's twin: a stderr subscriber, always installed —
/// a server that says nothing hides the session behind it. `$YATIMA_LOG`
/// sets the level exactly as in [`init_file_logging`] (`debug` shows each
/// tool call with args and outcome, `trace` adds whole prompts); unset
/// defaults to `info`. Bare levels scope to the yatima crates for the same
/// reason as the file twin: the question the console answers is "what is
/// yatima doing", not hyper's inner life.
pub fn init_stderr_logging(file_stem: &str) -> Result<()> {
    let value = std::env::var("YATIMA_LOG").unwrap_or_else(|_| "info".into());
    let spec = if value.contains('=') || value.contains(',') {
        value
    } else {
        format!(
            "warn,yatima_lib={value},yatima_host={value},\
             yatima_{file_stem}={value},yatima_text={value}"
        )
    };
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::new(spec))
        .with_writer(std::io::stderr)
        .init();
    Ok(())
}
