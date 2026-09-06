//! The scenario driver: run a file of GUI-box lines against the real host,
//! record both planes to a tape, and exit through the joined shutdown.
//!
//! `yatima-drive --profile muse-glimmer --offline scenarios/mandelbrot.scenario`
//! is the founding invocation (contract: plans/scenario-driver.plan.md). The
//! tape directory is the product; the exit code is the verdict. The driver
//! mints no law of its own — it exercises HOST-3/LSRV-1 (the joined exit),
//! CANCEL-1 (the monotone cancel plane), CAP-3 (scenario grants), and TAPE-1
//! (both planes through one recorder queue).

use std::future::Future;
use std::path::PathBuf;
use std::pin::Pin;
use std::process::ExitCode;
use std::sync::mpsc::Sender;
use std::time::Duration;

use anyhow::{anyhow, Context, Result};
use chrono::Utc;
use clap::Parser;
use tokio::sync::mpsc::UnboundedReceiver;
use tokio::time::Instant;

use yatima_drive::{start_recorder, RecorderHandle, TapeMeta, TapeRecord};
use yatima_host::{
    init_file_logging, knobs, resolve_host_model, spawn_nonblocking, CancelGate, HostClient,
    HostConfig, HostEvent, HostModelChoices, HostRequest,
};
use yatima_lib::{GenOpts, Sampling};

/// Startup wait: generous for a verified offline launch. A non-offline first
/// fetch is legitimately unbounded — prefetch, or pass `--offline`.
const STARTUP_WITHIN: Duration = Duration::from_secs(600);
/// Bound on admitting one record to the capacity-one recorder queue. The
/// driver's value is bounded liveness, so unlike the GUI's deliberately
/// pausing synchronous edge, a wedged recorder is an evidence failure here.
const RECORD_WITHIN: Duration = Duration::from_secs(5);
/// Bound on recorder control waits: creation and the semantic Shutdown record.
const CONTROL_WITHIN: Duration = Duration::from_secs(5);
/// After a cancel, how long the host gets to settle the turn (a cancelled
/// decode ends at the next token; outliving this means a wedged host).
const SETTLE_GRACE: Duration = Duration::from_secs(30);
/// The epilogue's event-tail drain stops after this much quiet — the fast
/// exit when the sender is open but silent.
const DRAIN_QUIET: Duration = Duration::from_secs(1);
/// The tail drain's absolute deadline, quiet or not: a reaper-transferred
/// actor emitting faster than the quiet cutoff cannot hold the epilogue
/// open. Expiry marks the evidence tail incomplete.
const DRAIN_WITHIN: Duration = Duration::from_secs(10);

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Scenario file: one GUI-box line per utterance (`/grant`, `/revoke`,
    /// prompts; `#` comments and blank lines ignored).
    scenario: PathBuf,
    /// A built-in model profile (e.g. `muse-glimmer`).
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
    /// Force CPU instead of the GPU.
    #[arg(long)]
    cpu: bool,
    /// Don't auto-fetch a missing model; error instead.
    #[arg(long)]
    offline: bool,
    /// Optional system instruction (applies for the whole session).
    #[arg(long)]
    system: Option<String>,
    /// Tape destination (default `runs/<utc-stamp>-<pid>-drive`). Recording
    /// is unconditional: the tape is the product.
    #[arg(long, value_name = "DIR")]
    tape: Option<PathBuf>,
    /// Per-turn wall clock in seconds before the turn is cancelled.
    #[arg(long, default_value_t = 300, value_parser = clap::value_parser!(u64).range(1..))]
    turn_within: u64,
    /// Abort after the first turn that settles Error or times out.
    #[arg(long)]
    stop_on_error: bool,
}

// The scenario language: one trimmed line per utterance.

#[derive(Debug, Clone, PartialEq, Eq)]
enum ScenarioLine {
    Grant(String),
    Revoke(String),
    Prompt(String),
}

/// Parse the whole file before anything spawns: a malformed scenario costs
/// nothing to reap. Lines trim before classification (the GUI trims its box
/// the same way); an unknown `/` line is an error naming its line number —
/// GUI-local commands have no host meaning, and silently prompting the model
/// with one would corrupt the scenario.
fn parse_scenario(text: &str) -> Result<Vec<ScenarioLine>> {
    let mut lines = Vec::new();
    for (index, raw) in text.lines().enumerate() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if let Some(origin) = line.strip_prefix("/grant ") {
            let origin = origin.trim();
            if origin.is_empty() {
                anyhow::bail!("line {}: /grant needs an origin", index + 1);
            }
            lines.push(ScenarioLine::Grant(origin.to_string()));
        } else if let Some(origin) = line.strip_prefix("/revoke ") {
            let origin = origin.trim();
            if origin.is_empty() {
                anyhow::bail!("line {}: /revoke needs an origin", index + 1);
            }
            lines.push(ScenarioLine::Revoke(origin.to_string()));
        } else if line.starts_with('/') {
            anyhow::bail!(
                "line {}: {:?} is not a scenario command (only /grant and /revoke reach the host)",
                index + 1,
                line
            );
        } else {
            lines.push(ScenarioLine::Prompt(line.to_string()));
        }
    }
    Ok(lines)
}

// The outcome algebra: one accumulated session verdict (a severity join),
// with evidence and cleanup as independent axes. `PreHost` failures never
// build these facts — they are the early `Err` exit from `run`.

/// Session facts joined by severity: declaration order IS the join order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Session {
    Clean,
    StartupFailed,
    TurnErrors,
    Timeout,
    Fatal,
    Interrupted,
}

/// The exit-code projection over the outcome product (the session verdict
/// outranks evidence and cleanup, which then ride diagnostics).
fn exit_code(
    session: Session,
    recorder_failed: bool,
    host_join_failed: bool,
    finish_failed: bool,
) -> u8 {
    match session {
        Session::Interrupted => 130,
        Session::Fatal => 3,
        Session::Timeout => 4,
        Session::TurnErrors => 2,
        Session::StartupFailed => 1,
        Session::Clean => {
            if recorder_failed || host_join_failed || finish_failed {
                5
            } else {
                0
            }
        }
    }
}

/// The disposition written by the consuming finish: the strongest
/// session-or-evidence fact; `backend-error` when the session is clean but
/// the already-known host join failed; `completed` only when nothing above
/// it is true. (A finish failure cannot be in its own disposition.)
fn disposition(session: Session, recorder_failed: bool, host_join_failed: bool) -> &'static str {
    match session {
        Session::Interrupted => "interrupted",
        Session::Fatal => "fatal",
        Session::Timeout => "timeout",
        Session::TurnErrors => "errored",
        Session::StartupFailed => "startup-error",
        Session::Clean => {
            if recorder_failed {
                "recorder-error"
            } else if host_join_failed {
                "backend-error"
            } else {
                "completed"
            }
        }
    }
}

// The record-sink seam: the driver core is generic over record admission so
// the stalled-recorder witness is constructible without touching the
// recorder library surface (`RecorderHandle`'s internals are private and
// `start_recorder` always builds a real file writer).

trait RecordSink {
    async fn record(&self, record: TapeRecord) -> Result<()>;
}

/// The admission bound, generic over the inner sink so the stalled-recorder
/// witness drives the exact production boundary with a truly pending fake.
struct Bounded<S> {
    inner: S,
    within: Duration,
}

impl<S: RecordSink> RecordSink for Bounded<S> {
    async fn record(&self, record: TapeRecord) -> Result<()> {
        tokio::time::timeout(self.within, self.inner.record(record))
            .await
            .map_err(|_| anyhow!("record admission timed out after {:?}", self.within))?
    }
}

/// The production inner sink: the D0.5 handle, unbounded here — the bound is
/// `Bounded`'s alone.
struct HandleSink {
    handle: RecorderHandle,
}

impl RecordSink for HandleSink {
    async fn record(&self, record: TapeRecord) -> Result<()> {
        self.handle.enqueue(record).await
    }
}

/// The one pinned interrupt: it owns a branch of every dequeue and of every
/// pre-dispatch request admission, and once fired it stays fired (later
/// waits see a pending future, never a re-poll). Recording of a consumed
/// observation deliberately does not race it — see `record_step`'s doc.
struct Interrupt {
    fired: bool,
    signal: Pin<Box<dyn Future<Output = ()>>>,
}

impl Interrupt {
    fn new(signal: Pin<Box<dyn Future<Output = ()>>>) -> Interrupt {
        Interrupt {
            fired: false,
            signal,
        }
    }

    fn ctrl_c() -> Interrupt {
        Interrupt::new(Box::pin(async {
            let _ = tokio::signal::ctrl_c().await;
        }))
    }

    async fn wait(&mut self) {
        if self.fired {
            std::future::pending::<()>().await;
        }
        self.signal.as_mut().await;
        self.fired = true;
    }
}

/// What one bounded, interruptible step of the driver produced.
enum Step<T> {
    Value(T),
    Interrupted,
    RecorderFailed(anyhow::Error),
    /// The request plane is closed: the host actor is gone. The recorded
    /// request stands on the tape as an attempted dispatch.
    HostGone,
}

/// The movable host planes the core drives (exactly `HostClient`'s fields,
/// taken apart so tests can fabricate them).
struct Planes {
    req_tx: Sender<HostRequest>,
    event_rx: UnboundedReceiver<HostEvent>,
    cancel: CancelGate,
}

impl Planes {
    fn from_client(client: HostClient) -> Planes {
        Planes {
            req_tx: client.req_tx,
            event_rx: client.event_rx,
            cancel: client.cancel,
        }
    }
}

/// What the scenario run accumulated; the epilogue turns it into the exit.
#[derive(Debug, PartialEq, Eq)]
struct DriveFacts {
    session: Session,
    recorder_failed: bool,
}

/// Record a request that has not yet been dispatched, with the interrupt
/// owning a branch of the wait: nothing has been observed or issued yet, so
/// abandoning the record on Ctrl-C loses nothing from the tape.
///
/// This is deliberately NOT used for anything already observed. An event
/// pulled off the plane, or the semantic Cancel of an already-issued turn,
/// is recorded through the sink's own admission bound without racing the
/// interrupt — otherwise the interrupt could punch an unreported interior
/// hole into a tape whose later records still land. The interrupt is still
/// honored within one admission bound, at the next wait.
async fn record_step<S: RecordSink>(
    sink: &S,
    interrupt: &mut Interrupt,
    record: TapeRecord,
) -> Step<()> {
    tokio::select! {
        _ = interrupt.wait() => Step::Interrupted,
        result = sink.record(record) => match result {
            Ok(()) => Step::Value(()),
            Err(error) => Step::RecorderFailed(error),
        },
    }
}

/// Record-before-side-effect at the driver's one dispatch seam: an
/// unrecordable request is not dispatched (the scenario ends as an evidence
/// failure), while requests already issued are never suppressed. A send onto
/// a closed request plane is host loss — the recorded request stays on the
/// tape as an attempted dispatch, and the run ends with the fatal fact.
async fn dispatch<S: RecordSink>(
    sink: &S,
    interrupt: &mut Interrupt,
    req_tx: &Sender<HostRequest>,
    request: HostRequest,
) -> Step<()> {
    match record_step(sink, interrupt, TapeRecord::Request(request.clone())).await {
        Step::Value(()) => {
            if req_tx.send(request).is_err() {
                return Step::HostGone;
            }
            Step::Value(())
        }
        other => other,
    }
}

/// Drive the whole scenario: wait for `Ready`, then submit each line
/// sequentially, recording every event before interpreting it.
async fn drive_scenario<S: RecordSink>(
    lines: &[ScenarioLine],
    sink: &S,
    planes: &mut Planes,
    interrupt: &mut Interrupt,
    turn_within: Duration,
    stop_on_error: bool,
) -> DriveFacts {
    let mut facts = DriveFacts {
        session: Session::Clean,
        recorder_failed: false,
    };

    macro_rules! step {
        ($facts:ident, $step:expr) => {
            match $step {
                Step::Value(value) => value,
                Step::Interrupted => {
                    $facts.session = $facts.session.max(Session::Interrupted);
                    return $facts;
                }
                Step::RecorderFailed(error) => {
                    eprintln!("yatima-drive: evidence failure: {error:#}");
                    $facts.recorder_failed = true;
                    return $facts;
                }
                Step::HostGone => {
                    eprintln!("yatima-drive: host request plane closed mid-run");
                    $facts.session = $facts.session.max(Session::Fatal);
                    return $facts;
                }
            }
        };
    }

    // Startup: consume (and record) events until Ready. Fatal, a closed
    // plane, or the bound elapsing are startup failure; the epilogue still
    // owns every consumer.
    let startup_deadline = Instant::now() + STARTUP_WITHIN;
    loop {
        let event = tokio::select! {
            _ = interrupt.wait() => {
                facts.session = facts.session.max(Session::Interrupted);
                return facts;
            }
            _ = tokio::time::sleep_until(startup_deadline) => {
                eprintln!("yatima-drive: no Ready within {STARTUP_WITHIN:?}");
                facts.session = facts.session.max(Session::StartupFailed);
                return facts;
            }
            event = planes.event_rx.recv() => event,
        };
        let Some(event) = event else {
            eprintln!("yatima-drive: host event plane closed before Ready");
            facts.session = facts.session.max(Session::StartupFailed);
            return facts;
        };
        // Observed events record under the admission bound, never raced
        // against the interrupt: an interior hole would falsify the tape.
        if let Err(error) = sink.record(TapeRecord::Event(event.clone())).await {
            eprintln!("yatima-drive: evidence failure: {error:#}");
            facts.recorder_failed = true;
            return facts;
        }
        match event {
            HostEvent::Ready(_) => break,
            HostEvent::Fatal(message) => {
                eprintln!("yatima-drive: fatal during startup: {message}");
                facts.session = facts.session.max(Session::StartupFailed);
                return facts;
            }
            _ => {}
        }
    }

    let mut next_turn = 0u64;
    for line in lines {
        match line {
            ScenarioLine::Grant(origin) => {
                step!(
                    facts,
                    dispatch(
                        sink,
                        interrupt,
                        &planes.req_tx,
                        HostRequest::Grant {
                            origin: origin.clone(),
                        },
                    )
                    .await
                );
            }
            ScenarioLine::Revoke(origin) => {
                step!(
                    facts,
                    dispatch(
                        sink,
                        interrupt,
                        &planes.req_tx,
                        HostRequest::Revoke {
                            origin: origin.clone(),
                        },
                    )
                    .await
                );
            }
            ScenarioLine::Prompt(text) => {
                // A URL in the prompt is authorization for its origin
                // (CAP-3): granted before the turn runs, the GUI's own rule.
                for origin in yatima_lib::origins_in(text) {
                    step!(
                        facts,
                        dispatch(
                            sink,
                            interrupt,
                            &planes.req_tx,
                            HostRequest::Grant { origin }
                        )
                        .await
                    );
                }
                let turn_id = next_turn;
                next_turn += 1;
                eprintln!("yatima-drive: turn {turn_id}: {text}");
                step!(
                    facts,
                    dispatch(
                        sink,
                        interrupt,
                        &planes.req_tx,
                        HostRequest::Submit {
                            turn_id,
                            text: text.clone(),
                        },
                    )
                    .await
                );

                match settle_turn(sink, planes, interrupt, turn_id, turn_within, &mut facts).await {
                    TurnEnd::Settled => {}
                    TurnEnd::SettledBadly if !stop_on_error => {}
                    TurnEnd::SettledBadly | TurnEnd::RunOver => return facts,
                }
            }
        }
    }
    facts
}

/// How one turn left the run.
enum TurnEnd {
    /// Settled `Done`: the run continues unconditionally.
    Settled,
    /// Settled `Error` or cancelled on timeout: `--stop-on-error` decides.
    SettledBadly,
    /// Fatal, closed plane, interrupt, evidence failure, or a wedged host:
    /// the run is over regardless.
    RunOver,
}

/// Consume events until `turn_id` settles, cancelling at `turn_within`
/// through the monotone plane: record the semantic Cancel, then trip the
/// gate, then give the host `SETTLE_GRACE` to settle.
async fn settle_turn<S: RecordSink>(
    sink: &S,
    planes: &mut Planes,
    interrupt: &mut Interrupt,
    turn_id: u64,
    turn_within: Duration,
    facts: &mut DriveFacts,
) -> TurnEnd {
    let deadline = Instant::now() + turn_within;
    loop {
        let event = tokio::select! {
            _ = interrupt.wait() => {
                facts.session = facts.session.max(Session::Interrupted);
                cancel_and_drain(sink, planes, interrupt, turn_id, facts).await;
                return TurnEnd::RunOver;
            }
            _ = tokio::time::sleep_until(deadline) => {
                eprintln!("yatima-drive: turn {turn_id} timed out; cancelling");
                facts.session = facts.session.max(Session::Timeout);
                return if cancel_and_drain(sink, planes, interrupt, turn_id, facts).await {
                    TurnEnd::SettledBadly
                } else {
                    // Outliving even the grace is a wedged host.
                    TurnEnd::RunOver
                };
            }
            event = planes.event_rx.recv() => event,
        };
        let Some(event) = event else {
            eprintln!("yatima-drive: host event plane closed mid-run");
            facts.session = facts.session.max(Session::Fatal);
            return TurnEnd::RunOver;
        };
        // Observed: record before anything else may preempt (bounded by the
        // sink's admission bound, so the interrupt waits at most that long).
        if let Err(error) = sink.record(TapeRecord::Event(event.clone())).await {
            eprintln!("yatima-drive: evidence failure: {error:#}");
            facts.recorder_failed = true;
            return TurnEnd::RunOver;
        }
        match event {
            HostEvent::Done { turn_id: id, .. } if id == turn_id => return TurnEnd::Settled,
            HostEvent::Error {
                turn_id: id,
                message,
            } if id == turn_id => {
                eprintln!("yatima-drive: turn {turn_id} errored: {message}");
                facts.session = facts.session.max(Session::TurnErrors);
                return TurnEnd::SettledBadly;
            }
            HostEvent::Fatal(message) => {
                eprintln!("yatima-drive: fatal: {message}");
                facts.session = facts.session.max(Session::Fatal);
                return TurnEnd::RunOver;
            }
            _ => {}
        }
    }
}

/// The recorded cancel: the semantic `Cancel` is attempted on the tape, and
/// the out-of-band gate then trips *unconditionally* — cancellation of an
/// already-issued turn is never suppressed by recorder trouble. The host then
/// gets `SETTLE_GRACE` to settle (events still recorded), and every fact
/// observed on the way — interruption, `Fatal`, a closed plane — joins the
/// session severity rather than being flattened into "did not settle".
/// Returns whether the turn settled.
async fn cancel_and_drain<S: RecordSink>(
    sink: &S,
    planes: &mut Planes,
    interrupt: &mut Interrupt,
    turn_id: u64,
    facts: &mut DriveFacts,
) -> bool {
    // The turn is already issued, so its semantic Cancel is recorded like an
    // observed fact — bounded, never raced against the interrupt (a hole
    // here would misdate the cancellation on an otherwise complete tape).
    let mut evidence_failed = false;
    match sink
        .record(TapeRecord::Request(HostRequest::Cancel { turn_id }))
        .await
    {
        Ok(()) => {
            let _ = planes.req_tx.send(HostRequest::Cancel { turn_id });
        }
        Err(error) => {
            eprintln!("yatima-drive: evidence failure: {error:#}");
            facts.recorder_failed = true;
            evidence_failed = true;
        }
    }
    planes.cancel.cancel(turn_id);
    if evidence_failed {
        return false;
    }

    let grace = Instant::now() + SETTLE_GRACE;
    loop {
        let event = tokio::select! {
            _ = interrupt.wait() => {
                // The epilogue's tail drain will still record a late
                // settlement; the user asked out now.
                facts.session = facts.session.max(Session::Interrupted);
                return false;
            }
            _ = tokio::time::sleep_until(grace) => return false,
            event = planes.event_rx.recv() => event,
        };
        let Some(event) = event else {
            eprintln!("yatima-drive: host event plane closed during settle grace");
            facts.session = facts.session.max(Session::Fatal);
            return false;
        };
        if let Err(error) = sink.record(TapeRecord::Event(event.clone())).await {
            eprintln!("yatima-drive: evidence failure: {error:#}");
            facts.recorder_failed = true;
            return false;
        }
        match event {
            HostEvent::Done { turn_id: id, .. } | HostEvent::Error { turn_id: id, .. }
                if id == turn_id =>
            {
                return true;
            }
            HostEvent::Fatal(message) => {
                eprintln!("yatima-drive: fatal during settle grace: {message}");
                facts.session = facts.session.max(Session::Fatal);
                return false;
            }
            _ => {}
        }
    }
}

/// Drain the event tail after the host has joined (its sender is closed
/// then, so the drain sees everything the actor emitted): each event is
/// recorded under the admission bound, and a late `Fatal` becomes a session
/// fact instead of vanishing. Two exits guard the one case where the sender
/// never closes — a timed-out shutdown whose join transferred to the reaper:
/// the quiet cutoff for a silent survivor, and an absolute deadline a chatty
/// survivor cannot reset, whose expiry marks the evidence tail incomplete.
async fn drain_tail<S: RecordSink>(
    sink: &S,
    event_rx: &mut UnboundedReceiver<HostEvent>,
    quiet: Duration,
    within: Duration,
) -> (Session, bool) {
    let mut session = Session::Clean;
    let mut recorder_failed = false;
    let deadline = Instant::now() + within;
    loop {
        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            eprintln!("yatima-drive: event tail incomplete after {within:?} (host outlived its shutdown bound)");
            recorder_failed = true;
            break;
        }
        match tokio::time::timeout(quiet.min(remaining), event_rx.recv()).await {
            Ok(None) => break,
            Err(_) => {
                if Instant::now() >= deadline {
                    eprintln!("yatima-drive: event tail incomplete after {within:?} (host outlived its shutdown bound)");
                    recorder_failed = true;
                }
                break;
            }
            Ok(Some(event)) => {
                // The record admission races the same absolute deadline: the
                // advertised bound is the whole drain, not just dequeuing.
                let remaining = deadline.saturating_duration_since(Instant::now());
                match tokio::time::timeout(remaining, sink.record(TapeRecord::Event(event.clone())))
                    .await
                {
                    Err(_) => {
                        eprintln!(
                            "yatima-drive: event tail incomplete after {within:?} (record admission crossed the deadline)"
                        );
                        recorder_failed = true;
                        break;
                    }
                    Ok(Err(error)) => {
                        eprintln!("yatima-drive: evidence failure in the tail: {error:#}");
                        recorder_failed = true;
                        break;
                    }
                    Ok(Ok(())) => {}
                }
                if let HostEvent::Fatal(message) = &event {
                    eprintln!("yatima-drive: fatal in the event tail: {message}");
                    session = Session::Fatal;
                }
            }
        }
    }
    (session, recorder_failed)
}

/// The epilogue, factored for the stalled-recorder witness: attempt the
/// semantic Shutdown record wherever the host plane ever existed, always
/// await the joined host shutdown, then drain and record the event tail
/// (folding any late `Fatal` into the session severity), then the consuming
/// recorder finish with the honest disposition. Returns the exit code.
async fn close_run<R, H, D, DF, F, FF>(
    facts: DriveFacts,
    shutdown_record: R,
    host_shutdown: H,
    drain: D,
    finish: F,
) -> u8
where
    R: Future<Output = Result<()>>,
    H: Future<Output = Result<()>>,
    D: FnOnce() -> DF,
    DF: Future<Output = (Session, bool)>,
    F: FnOnce(&'static str) -> FF,
    FF: Future<Output = Result<()>>,
{
    let mut recorder_failed = facts.recorder_failed;
    if let Err(error) = shutdown_record.await {
        eprintln!("yatima-drive: shutdown record failed: {error:#}");
        recorder_failed = true;
    }
    let host_join_failed = match host_shutdown.await {
        Ok(()) => false,
        Err(error) => {
            eprintln!("yatima-drive: backend owner shutdown failed: {error:#}");
            true
        }
    };
    let (tail_session, tail_recorder_failed) = drain().await;
    let session = facts.session.max(tail_session);
    recorder_failed |= tail_recorder_failed;
    let disposition = disposition(session, recorder_failed, host_join_failed);
    let finish_failed = match finish(disposition).await {
        Ok(()) => false,
        Err(error) => {
            eprintln!("yatima-drive: flight recorder finish failed: {error:#}");
            true
        }
    };
    exit_code(session, recorder_failed, host_join_failed, finish_failed)
}

/// Requested git provenance, supplied by the invoker: the driver spawns no
/// subprocess for it. A probe child proved impossible to own soundly here —
/// tokio's runtime shutdown waits indefinitely for started blocking tasks,
/// so any wedged `wait` (a failed kill, a descendant holding the stdout
/// pipe) could hang process exit — and the header field is optional
/// provenance, so the economical truth wins: scripts and CI pass
/// `YATIMA_GIT_DESCRIBE="$(git describe --dirty --always)"`; absent that,
/// the field is omitted rather than guessed.
fn git_describe_from_env() -> Option<String> {
    git_describe_note(std::env::var("YATIMA_GIT_DESCRIBE").ok())
}

/// The pure half: trim, and treat empty as absent — omitted, never guessed.
fn git_describe_note(raw: Option<String>) -> Option<String> {
    raw.map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
}

/// The requested display label, mirroring the GUI's title derivation.
fn requested_label(args: &Args) -> String {
    args.profile
        .clone()
        .or_else(|| args.model.as_ref().map(|p| p.display().to_string()))
        .or_else(|| args.repo.clone())
        .unwrap_or_else(|| "local model".to_string())
}

/// The shared resolver, exactly as the GUI: profile-layered generation over
/// the same agent-shaped base; no field-by-field reconstruction.
fn resolve(args: &Args) -> Result<HostConfig> {
    if let Some(config) = test_stub_config(args)? {
        return Ok(config);
    }
    let resolved = resolve_host_model(HostModelChoices {
        profile: args.profile.clone(),
        model: args.model.clone(),
        repo: args.repo.clone(),
        models_dir: args.models_dir.clone(),
        gguf: args.gguf.clone(),
        cpu: args.cpu,
        offline: args.offline,
    })?;
    Ok(resolved.into_host_config(base_gen_opts(), args.system.clone()))
}

fn base_gen_opts() -> GenOpts {
    GenOpts {
        max_tokens: 1024,
        sampling: Sampling::nucleus(0.0, None, 0),
        ..Default::default()
    }
}

/// Test wiring only, on `with_managed_launcher`'s doc-hidden precedent: the
/// hermetic battery must reach the protocol stub through the real binary,
/// but the shared resolver only knows built-in profiles. With
/// `YATIMA_DRIVE_TEST_STUB_DIR` set, a Muse-format stub profile is built
/// over that directory's sole `.gguf` and launched via
/// `YATIMA_DRIVE_TEST_STUB_BIN` (readiness `YATIMA_DRIVE_TEST_STUB_READY_MS`,
/// default 15000). Never set outside the battery.
fn test_stub_config(args: &Args) -> Result<Option<HostConfig>> {
    let Ok(dir) = std::env::var("YATIMA_DRIVE_TEST_STUB_DIR") else {
        return Ok(None);
    };
    let bin = std::env::var("YATIMA_DRIVE_TEST_STUB_BIN")
        .context("YATIMA_DRIVE_TEST_STUB_BIN must accompany YATIMA_DRIVE_TEST_STUB_DIR")?;
    let ready_ms: u64 = std::env::var("YATIMA_DRIVE_TEST_STUB_READY_MS")
        .ok()
        .map(|value| value.parse())
        .transpose()
        .context("parse YATIMA_DRIVE_TEST_STUB_READY_MS")?
        .unwrap_or(15_000);
    let dir = PathBuf::from(dir);
    let mut ggufs = Vec::new();
    for entry in std::fs::read_dir(&dir).with_context(|| format!("read {}", dir.display()))? {
        let path = entry?.path();
        if path.extension().is_some_and(|ext| ext == "gguf") {
            ggufs.push(path);
        }
    }
    let [gguf] = ggufs.as_slice() else {
        anyhow::bail!("stub dir must hold exactly one .gguf: {}", dir.display());
    };
    let bytes = std::fs::read(gguf)?;
    let digest = |bytes: &[u8]| {
        use sha2::{Digest, Sha256};
        format!("{:x}", Sha256::digest(bytes))
            .parse::<yatima_lib::Sha256Digest>()
            .expect("a hex digest parses")
    };
    let profile = yatima_lib::ModelProfile {
        name: "stub-muse".into(),
        backend: yatima_lib::ProfileBackend::LlamaServer(yatima_lib::LlamaServerProfile {
            expected_sha256: digest(&bytes),
            build_floor: 10520,
            template_sha256: digest(b"stub-template"),
            context: 4096,
            top_k: 64,
        }),
        dir: Some(dir),
        format: Some(yatima_lib::ChatFormat::MuseGlimmer),
        ..yatima_lib::ModelProfile::default()
    };
    let config = HostConfig::managed(&profile, true, base_gen_opts(), args.system.clone())?
        .with_managed_launcher(PathBuf::from(bin), Duration::from_millis(ready_ms));
    Ok(Some(config))
}

fn tape_dir(args: &Args, utc_stamp: &str, pid: u32) -> PathBuf {
    args.tape
        .clone()
        .unwrap_or_else(|| PathBuf::from(format!("runs/{utc_stamp}-{pid}-drive")))
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> ExitCode {
    let args = match Args::try_parse() {
        Ok(args) => args,
        Err(error) => {
            let code = if error.exit_code() == 0 { 0 } else { 1 };
            let _ = error.print();
            return ExitCode::from(code as u8);
        }
    };
    match run(args).await {
        Ok(code) => ExitCode::from(code),
        Err(error) => {
            // PreHost: nothing spawned, nothing to consume.
            eprintln!("yatima-drive: {error:#}");
            ExitCode::from(1)
        }
    }
}

async fn run(args: Args) -> Result<u8> {
    init_file_logging("drive", &[])?;
    let text = std::fs::read_to_string(&args.scenario)
        .with_context(|| format!("read scenario {}", args.scenario.display()))?;
    let lines = parse_scenario(&text)?;
    let config = resolve(&args)?;
    // The requested label, the GUI's own precedence: the resolution's label
    // (a profile name) first, then the raw argument shapes.
    let label = config
        .model_label()
        .map(str::to_string)
        .unwrap_or_else(|| requested_label(&args));

    let utc_stamp = Utc::now().format("%Y%m%dT%H%M%SZ").to_string();
    let dir = tape_dir(&args, &utc_stamp, std::process::id());
    let mut notes = std::collections::BTreeMap::from([(
        "agent_max_steps".to_string(),
        knobs::AGENT_MAX_STEPS.to_string(),
    )]);
    if let Some(describe) = git_describe_from_env() {
        notes.insert("git_describe".to_string(), describe);
    }
    let meta = TapeMeta {
        origin: format!("yatima-drive {}", env!("CARGO_PKG_VERSION")),
        model: label,
        notes,
    };
    let (handle, tape_owner) = tokio::time::timeout(CONTROL_WITHIN, start_recorder(&dir, meta))
        .await
        .map_err(|_| anyhow!("flight recorder creation timed out after {CONTROL_WITHIN:?}"))?
        .context("start the flight recorder")?;
    // The evidence path, on stdout, so scripts find it on any exit.
    println!("{}", dir.display());

    let spawned = spawn_nonblocking(config);
    let (client, host_owner) = match spawned {
        Ok(pair) => pair,
        Err(error) => {
            // Startup failure after the recorder exists: no host plane was
            // ever created, so no Shutdown record; the recorder is still
            // consumed with the honest disposition.
            eprintln!("yatima-drive: spawn failed: {error:#}");
            let facts = DriveFacts {
                session: Session::StartupFailed,
                recorder_failed: false,
            };
            let code = close_run(
                facts,
                async { Ok(()) },
                async { Ok(()) },
                || async { (Session::Clean, false) },
                |disposition| async move { tape_owner.finish(disposition).await.map(|_| ()) },
            )
            .await;
            return Ok(code);
        }
    };

    let mut interrupt = Interrupt::ctrl_c();
    let sink = Bounded {
        inner: HandleSink {
            handle: handle.clone(),
        },
        within: RECORD_WITHIN,
    };
    let mut planes = Planes::from_client(client);
    let facts = drive_scenario(
        &lines,
        &sink,
        &mut planes,
        &mut interrupt,
        Duration::from_secs(args.turn_within),
        args.stop_on_error,
    )
    .await;
    eprintln!("yatima-drive: run over: {facts:?}");

    let mut event_rx = planes.event_rx;
    let code = close_run(
        facts,
        async {
            tokio::time::timeout(
                CONTROL_WITHIN,
                handle.enqueue(TapeRecord::Request(HostRequest::Shutdown)),
            )
            .await
            .map_err(|_| anyhow!("shutdown record timed out after {CONTROL_WITHIN:?}"))?
        },
        host_owner.shutdown(),
        || async { drain_tail(&sink, &mut event_rx, DRAIN_QUIET, DRAIN_WITHIN).await },
        |disposition| async move { tape_owner.finish(disposition).await.map(|_| ()) },
    )
    .await;
    Ok(code)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};
    use yatima_host::{ModelExecution, ModelIdentity, ModelInfo};

    const TEST_WITHIN: Duration = Duration::from_secs(15);

    async fn within<T>(what: &str, future: impl Future<Output = T>) -> T {
        tokio::time::timeout(TEST_WITHIN, future)
            .await
            .unwrap_or_else(|_| panic!("{what} did not finish within {TEST_WITHIN:?}"))
    }

    // The parser.

    #[test]
    fn parser_classifies_trimmed_lines() {
        let text = "\n  # indented comment\n/grant https://a.example/\n  /revoke https://b.example/  \n\nask me anything\n";
        let lines = parse_scenario(text).unwrap();
        assert_eq!(
            lines,
            vec![
                ScenarioLine::Grant("https://a.example/".into()),
                ScenarioLine::Revoke("https://b.example/".into()),
                ScenarioLine::Prompt("ask me anything".into()),
            ]
        );
    }

    #[test]
    fn parser_rejects_unknown_slash_lines_by_line_number() {
        let error = parse_scenario("hello\n/stats\n").unwrap_err();
        assert!(format!("{error}").contains("line 2"), "{error}");
        let error = parse_scenario("/grant   \n").unwrap_err();
        assert!(format!("{error}").contains("line 1"), "{error}");
    }

    #[test]
    fn turn_within_must_be_positive() {
        assert!(Args::try_parse_from(["yatima-drive", "--turn-within", "0", "s.txt"]).is_err());
        let args = Args::try_parse_from(["yatima-drive", "s.txt"]).unwrap();
        assert_eq!(args.turn_within, 300);
    }

    // The outcome algebra: the projection and disposition over the product,
    // cross-products included, not just the diagonal.

    #[test]
    fn exit_code_projection_covers_the_product() {
        // Singletons.
        assert_eq!(exit_code(Session::Clean, false, false, false), 0);
        assert_eq!(exit_code(Session::StartupFailed, false, false, false), 1);
        assert_eq!(exit_code(Session::TurnErrors, false, false, false), 2);
        assert_eq!(exit_code(Session::Fatal, false, false, false), 3);
        assert_eq!(exit_code(Session::Timeout, false, false, false), 4);
        assert_eq!(exit_code(Session::Interrupted, false, false, false), 130);
        assert_eq!(exit_code(Session::Clean, true, false, false), 5);
        assert_eq!(exit_code(Session::Clean, false, true, false), 5);
        assert_eq!(exit_code(Session::Clean, false, false, true), 5);
        // Turn-error-then-timeout joins to Timeout.
        assert_eq!(Session::TurnErrors.max(Session::Timeout), Session::Timeout);
        // Every session verdict outranks evidence and cleanup failures.
        for session in [
            Session::StartupFailed,
            Session::TurnErrors,
            Session::Timeout,
            Session::Fatal,
            Session::Interrupted,
        ] {
            assert_eq!(
                exit_code(session, true, true, true),
                exit_code(session, false, false, false),
                "{session:?}"
            );
        }
    }

    #[test]
    fn disposition_never_writes_a_false_completed() {
        assert_eq!(disposition(Session::Clean, false, false), "completed");
        // A clean session whose host join failed is backend-error, never
        // completed (the epilogue knows the join verdict before finishing).
        assert_eq!(disposition(Session::Clean, false, true), "backend-error");
        // Evidence outranks cleanup below the session facts.
        assert_eq!(disposition(Session::Clean, true, true), "recorder-error");
        assert_eq!(
            disposition(Session::StartupFailed, false, false),
            "startup-error"
        );
        assert_eq!(disposition(Session::TurnErrors, true, true), "errored");
        assert_eq!(disposition(Session::Timeout, false, false), "timeout");
        assert_eq!(disposition(Session::Fatal, false, false), "fatal");
        assert_eq!(
            disposition(Session::Interrupted, false, false),
            "interrupted"
        );
    }

    // The driver core over fake planes and sinks.

    #[derive(Clone, Default)]
    struct VecSink {
        records: Arc<Mutex<Vec<TapeRecord>>>,
    }

    impl RecordSink for VecSink {
        async fn record(&self, record: TapeRecord) -> Result<()> {
            self.records.lock().unwrap().push(record);
            Ok(())
        }
    }

    /// Records `admit` records, then pends forever — a truly wedged
    /// capacity-one queue, with no timeout of its own: the bound under test
    /// is the production `Bounded` wrapper's alone.
    struct PendingAfterSink {
        inner: VecSink,
        admit: usize,
    }

    impl RecordSink for PendingAfterSink {
        async fn record(&self, record: TapeRecord) -> Result<()> {
            let admitted = { self.inner.records.lock().unwrap().len() };
            if admitted >= self.admit {
                std::future::pending::<()>().await;
            }
            self.inner.record(record).await
        }
    }

    /// Fails to record exactly the semantic `Cancel`; everything else lands.
    struct CancelRefusingSink {
        inner: VecSink,
    }

    impl RecordSink for CancelRefusingSink {
        async fn record(&self, record: TapeRecord) -> Result<()> {
            if matches!(record, TapeRecord::Request(HostRequest::Cancel { .. })) {
                anyhow::bail!("injected cancel-record failure");
            }
            self.inner.record(record).await
        }
    }

    fn ready_info() -> ModelInfo {
        ModelInfo {
            label: "stub-muse".into(),
            arch: "stall-then-answer".into(),
            backend: "b10520-stub".into(),
            execution: ModelExecution::InProcess {
                device: "test".into(),
            },
            format: "MuseGlimmer".into(),
            sampling: "greedy".into(),
            max_tokens: 1024,
            context_length: Some(4096),
            identity: ModelIdentity::VerifiedSha256("00".into()),
        }
    }

    fn fake_planes() -> (
        Planes,
        std::sync::mpsc::Receiver<HostRequest>,
        tokio::sync::mpsc::UnboundedSender<HostEvent>,
    ) {
        let (req_tx, req_rx) = std::sync::mpsc::channel();
        let (ev_tx, event_rx) = tokio::sync::mpsc::unbounded_channel();
        (
            Planes {
                req_tx,
                event_rx,
                cancel: CancelGate::new(),
            },
            req_rx,
            ev_tx,
        )
    }

    fn never_interrupt() -> Interrupt {
        Interrupt::new(Box::pin(std::future::pending()))
    }

    #[tokio::test]
    async fn auto_grant_precedes_submit_and_both_planes_share_the_tape() {
        within("auto-grant ordering", async {
            let sink = VecSink::default();
            let (mut planes, req_rx, ev_tx) = fake_planes();
            ev_tx.send(HostEvent::Ready(ready_info())).unwrap();
            ev_tx.send(HostEvent::Started { turn_id: 0 }).unwrap();
            ev_tx
                .send(HostEvent::Done {
                    turn_id: 0,
                    stop: yatima_host::StopKind::Eos,
                })
                .unwrap();
            let mut interrupt = never_interrupt();
            let lines = vec![ScenarioLine::Prompt(
                "read https://en.example/page please".into(),
            )];
            let facts = drive_scenario(
                &lines,
                &sink,
                &mut planes,
                &mut interrupt,
                Duration::from_secs(5),
                false,
            )
            .await;
            assert_eq!(facts.session, Session::Clean);
            assert!(!facts.recorder_failed);

            let records = sink.records.lock().unwrap();
            let kinds: Vec<String> = records
                .iter()
                .map(|record| match record {
                    TapeRecord::Request(HostRequest::Grant { origin }) => {
                        format!("grant {origin}")
                    }
                    TapeRecord::Request(HostRequest::Submit { turn_id, .. }) => {
                        format!("submit {turn_id}")
                    }
                    TapeRecord::Request(other) => format!("request {other:?}"),
                    TapeRecord::Event(event) => format!("event {event:?}"),
                })
                .collect();
            assert!(
                kinds[1].starts_with("grant https://en.example"),
                "{kinds:?}"
            );
            assert_eq!(kinds[2], "submit 0", "{kinds:?}");
            // The host received the same order.
            assert!(matches!(req_rx.try_recv(), Ok(HostRequest::Grant { .. })));
            assert!(matches!(req_rx.try_recv(), Ok(HostRequest::Submit { .. })));
        })
        .await;
    }

    #[tokio::test]
    async fn stalled_recorder_is_bounded_by_the_production_wrapper_and_still_shuts_down() {
        within("stalled recorder composed", async {
            // upholds: TAPE-1 / HOST-3 — the exact production admission
            // boundary (`Bounded` over a truly pending sink) turns the wedge
            // into a bounded evidence failure that never unsends what was
            // already issued and never skips the joined backend shutdown.
            let sink = Bounded {
                inner: PendingAfterSink {
                    inner: VecSink::default(),
                    admit: 2, // Ready event + the Grant request are admitted.
                },
                within: Duration::from_millis(50),
            };
            let (mut planes, req_rx, ev_tx) = fake_planes();
            ev_tx.send(HostEvent::Ready(ready_info())).unwrap();
            let mut interrupt = never_interrupt();
            let lines = vec![
                ScenarioLine::Grant("https://a.example/".into()),
                ScenarioLine::Prompt("hello".into()),
            ];
            let started = std::time::Instant::now();
            let facts = drive_scenario(
                &lines,
                &sink,
                &mut planes,
                &mut interrupt,
                Duration::from_secs(60),
                false,
            )
            .await;
            assert!(started.elapsed() < Duration::from_secs(5), "bounded return");
            assert!(facts.recorder_failed);
            assert_eq!(facts.session, Session::Clean);
            // The already-issued Grant stands; the unrecordable Submit was
            // never dispatched (record-before-side-effect).
            assert!(matches!(req_rx.try_recv(), Ok(HostRequest::Grant { .. })));
            assert!(req_rx.try_recv().is_err(), "no suppressed-or-phantom send");

            // The same run's epilogue: host shutdown observed, disposition
            // honest, exit 5's evidence-failure arm.
            let reached = Arc::new(Mutex::new(false));
            let reached_by = reached.clone();
            let seen = Arc::new(Mutex::new(None::<&'static str>));
            let seen_by = seen.clone();
            let code = close_run(
                facts,
                async { Err(anyhow!("shutdown record also stalled")) },
                async {
                    *reached_by.lock().unwrap() = true;
                    Ok(())
                },
                || async { (Session::Clean, false) },
                |disposition| async move {
                    *seen_by.lock().unwrap() = Some(disposition);
                    Err(anyhow!("finish fails after evidence failure"))
                },
            )
            .await;
            assert!(*reached.lock().unwrap(), "host shutdown was reached");
            assert_eq!(*seen.lock().unwrap(), Some("recorder-error"));
            assert_eq!(code, 5);
        })
        .await;
    }

    #[tokio::test]
    async fn a_dead_request_plane_is_host_loss_not_silent_success() {
        within("dead request plane", async {
            // The recorded request stays as an attempted dispatch; the run
            // ends with the fatal fact instead of taping a phantom success.
            let sink = VecSink::default();
            let (mut planes, req_rx, ev_tx) = fake_planes();
            drop(req_rx);
            ev_tx.send(HostEvent::Ready(ready_info())).unwrap();
            let mut interrupt = never_interrupt();
            let facts = drive_scenario(
                &[ScenarioLine::Grant("https://a.example/".into())],
                &sink,
                &mut planes,
                &mut interrupt,
                Duration::from_secs(5),
                false,
            )
            .await;
            assert_eq!(facts.session, Session::Fatal);
            let records = sink.records.lock().unwrap();
            assert!(
                records
                    .iter()
                    .any(|r| matches!(r, TapeRecord::Request(HostRequest::Grant { .. }))),
                "the attempted dispatch is on the tape"
            );
        })
        .await;
    }

    #[tokio::test]
    async fn cancel_record_failure_still_trips_the_gate() {
        within("cancel-record failure", async {
            // Cancellation of an issued turn is never suppressed by
            // recorder trouble: the armed Cancel flips even though the
            // semantic record was refused.
            let sink = CancelRefusingSink {
                inner: VecSink::default(),
            };
            let (mut planes, _req_rx, _ev_tx) = fake_planes();
            let armed = yatima_lib::Cancel::new();
            planes.cancel.arm(0, armed.clone());
            let mut interrupt = never_interrupt();
            let mut facts = DriveFacts {
                session: Session::Timeout,
                recorder_failed: false,
            };
            let settled = cancel_and_drain(&sink, &mut planes, &mut interrupt, 0, &mut facts).await;
            assert!(!settled);
            assert!(facts.recorder_failed, "the refusal is an evidence fact");
            assert!(armed.is_cancelled(), "the gate tripped regardless");
        })
        .await;
    }

    #[tokio::test]
    async fn interrupt_during_settle_grace_joins_interrupted() {
        within("grace interrupt", async {
            let sink = VecSink::default();
            let records = sink.records.clone();
            let (mut planes, _req_rx, _ev_tx) = fake_planes();
            let (fire, fired) = tokio::sync::oneshot::channel::<()>();
            let mut interrupt = Interrupt::new(Box::pin(async {
                let _ = fired.await;
            }));
            // Fire the interrupt once the Cancel is recorded — i.e. during
            // the settle grace.
            let firer = tokio::spawn(async move {
                loop {
                    let cancelled = records.lock().unwrap().iter().any(|record| {
                        matches!(
                            record,
                            TapeRecord::Request(HostRequest::Cancel { turn_id: 0 })
                        )
                    });
                    if cancelled {
                        break;
                    }
                    tokio::time::sleep(Duration::from_millis(5)).await;
                }
                let _ = fire.send(());
            });
            let mut facts = DriveFacts {
                session: Session::Timeout,
                recorder_failed: false,
            };
            let started = std::time::Instant::now();
            let settled = cancel_and_drain(&sink, &mut planes, &mut interrupt, 0, &mut facts).await;
            firer.await.unwrap();
            assert!(!settled);
            assert!(started.elapsed() < Duration::from_secs(5), "prompt exit");
            assert_eq!(facts.session, Session::Interrupted, "joins above timeout");
        })
        .await;
    }

    #[tokio::test]
    async fn fatal_during_settle_grace_joins_fatal() {
        within("grace fatal", async {
            let sink = VecSink::default();
            let (mut planes, _req_rx, ev_tx) = fake_planes();
            ev_tx.send(HostEvent::Fatal("child died".into())).unwrap();
            let mut interrupt = never_interrupt();
            let mut facts = DriveFacts {
                session: Session::Timeout,
                recorder_failed: false,
            };
            let settled = cancel_and_drain(&sink, &mut planes, &mut interrupt, 0, &mut facts).await;
            assert!(!settled);
            assert_eq!(facts.session, Session::Fatal, "fatal outranks timeout");

            // A closed plane during grace is the same fatal fact.
            let (mut planes, _req_rx, ev_tx) = fake_planes();
            drop(ev_tx);
            let mut facts = DriveFacts {
                session: Session::Timeout,
                recorder_failed: false,
            };
            let settled = cancel_and_drain(&sink, &mut planes, &mut interrupt, 0, &mut facts).await;
            assert!(!settled);
            assert_eq!(facts.session, Session::Fatal);
        })
        .await;
    }

    #[tokio::test]
    async fn drain_tail_records_late_events_and_folds_a_late_fatal() {
        within("tail drain", async {
            let sink = VecSink::default();
            let (ev_tx, mut event_rx) = tokio::sync::mpsc::unbounded_channel();
            ev_tx
                .send(HostEvent::Grants {
                    origins: vec!["https://a.example/".into()],
                    message: "granted".into(),
                })
                .unwrap();
            ev_tx.send(HostEvent::Fatal("late death".into())).unwrap();
            drop(ev_tx); // the joined host's sender is closed
            let (session, recorder_failed) =
                drain_tail(&sink, &mut event_rx, DRAIN_QUIET, DRAIN_WITHIN).await;
            assert_eq!(session, Session::Fatal, "the late Fatal is a fact");
            assert!(!recorder_failed);
            let records = sink.records.lock().unwrap();
            assert_eq!(records.len(), 2, "every tail event is on the tape");
        })
        .await;
    }

    #[tokio::test]
    async fn a_chatty_surviving_actor_cannot_hold_the_drain_open() {
        within("chatty tail", async {
            // A reaper-transferred actor emitting faster than the quiet
            // cutoff meets the absolute deadline; the tail is marked
            // incomplete evidence rather than draining forever.
            let sink = VecSink::default();
            let (ev_tx, mut event_rx) = tokio::sync::mpsc::unbounded_channel();
            let chatter = tokio::spawn(async move {
                loop {
                    if ev_tx.send(HostEvent::Note("chatter".into())).is_err() {
                        break;
                    }
                    tokio::time::sleep(Duration::from_millis(10)).await;
                }
            });
            let started = std::time::Instant::now();
            let (session, recorder_failed) = drain_tail(
                &sink,
                &mut event_rx,
                Duration::from_secs(1),
                Duration::from_millis(200),
            )
            .await;
            chatter.abort();
            let _ = chatter.await;
            assert!(started.elapsed() < Duration::from_secs(5), "absolute bound");
            assert_eq!(session, Session::Clean);
            assert!(recorder_failed, "the incomplete tail is an evidence fact");
            assert!(
                !sink.records.lock().unwrap().is_empty(),
                "what arrived before the bound is on the tape"
            );
        })
        .await;
    }

    /// Signals when `record` is entered, then pends forever: the handshake
    /// that makes the no-interior-hole witness deterministic.
    struct SignalThenPendSink {
        entered: Mutex<Option<tokio::sync::oneshot::Sender<()>>>,
    }

    impl RecordSink for SignalThenPendSink {
        async fn record(&self, _record: TapeRecord) -> Result<()> {
            if let Some(entered) = self.entered.lock().unwrap().take() {
                let _ = entered.send(());
            }
            std::future::pending::<()>().await;
            unreachable!()
        }
    }

    #[tokio::test]
    async fn an_observed_event_is_recorded_or_reported_never_a_silent_hole() {
        within("no interior hole", async {
            // The interrupt fires only once the observed event's record has
            // provably been entered — so the event is consumed, and the only
            // honest outcomes are a recorded event or an evidence failure,
            // never a silent hole. The bounded record failure must win.
            let (entered_tx, entered_rx) = tokio::sync::oneshot::channel::<()>();
            let sink = Bounded {
                inner: SignalThenPendSink {
                    entered: Mutex::new(Some(entered_tx)),
                },
                within: Duration::from_millis(50),
            };
            let (mut planes, _req_rx, ev_tx) = fake_planes();
            ev_tx.send(HostEvent::Ready(ready_info())).unwrap();
            let mut interrupt = Interrupt::new(Box::pin(async move {
                let _ = entered_rx.await;
            }));
            let facts = drive_scenario(
                &[ScenarioLine::Prompt("hello".into())],
                &sink,
                &mut planes,
                &mut interrupt,
                Duration::from_secs(60),
                false,
            )
            .await;
            assert!(
                facts.recorder_failed,
                "the unrecorded observed event is reported as evidence failure"
            );
            assert_ne!(
                facts.session,
                Session::Interrupted,
                "the pending interrupt did not preempt the consumed observation"
            );
        })
        .await;
    }

    #[tokio::test]
    async fn interrupt_preempts_a_stalled_record_wait() {
        within("interrupt preempts stall", async {
            // Ready is admitted; the wedge hits the Submit's pre-dispatch
            // request admission — the one wait the interrupt may preempt.
            let sink = Bounded {
                inner: PendingAfterSink {
                    inner: VecSink::default(),
                    admit: 1,
                },
                within: Duration::from_secs(60),
            };
            let (mut planes, _req_rx, ev_tx) = fake_planes();
            ev_tx.send(HostEvent::Ready(ready_info())).unwrap();
            let (fire, fired) = tokio::sync::oneshot::channel::<()>();
            let mut interrupt = Interrupt::new(Box::pin(async {
                let _ = fired.await;
            }));
            fire.send(()).unwrap();
            let started = std::time::Instant::now();
            let facts = drive_scenario(
                &[ScenarioLine::Prompt("hello".into())],
                &sink,
                &mut planes,
                &mut interrupt,
                Duration::from_secs(60),
                false,
            )
            .await;
            assert!(started.elapsed() < Duration::from_secs(5), "bounded return");
            assert_eq!(facts.session, Session::Interrupted);
        })
        .await;
    }

    #[tokio::test]
    async fn timeout_cancels_through_the_gate_and_settles_in_grace() {
        within("timeout cancel", async {
            let sink = VecSink::default();
            let records = sink.records.clone();
            let (mut planes, req_rx, ev_tx) = fake_planes();
            ev_tx.send(HostEvent::Ready(ready_info())).unwrap();
            ev_tx.send(HostEvent::Started { turn_id: 0 }).unwrap();
            // The host settles only after the driver's cancel: a task
            // watches for the recorded Cancel, then sends Done.
            let settle = tokio::spawn(async move {
                loop {
                    let cancelled = records.lock().unwrap().iter().any(|record| {
                        matches!(
                            record,
                            TapeRecord::Request(HostRequest::Cancel { turn_id: 0 })
                        )
                    });
                    if cancelled {
                        break;
                    }
                    tokio::time::sleep(Duration::from_millis(5)).await;
                }
                ev_tx
                    .send(HostEvent::Done {
                        turn_id: 0,
                        stop: yatima_host::StopKind::Eos,
                    })
                    .unwrap();
            });
            let mut interrupt = never_interrupt();
            let facts = drive_scenario(
                &[ScenarioLine::Prompt("stall".into())],
                &sink,
                &mut planes,
                &mut interrupt,
                Duration::from_millis(100),
                false,
            )
            .await;
            settle.await.unwrap();
            assert_eq!(facts.session, Session::Timeout);
            // The semantic Cancel rode the tape and the wire.
            let sent: Vec<HostRequest> = req_rx.try_iter().collect();
            assert!(sent
                .iter()
                .any(|request| matches!(request, HostRequest::Cancel { turn_id: 0 })));
        })
        .await;
    }

    #[tokio::test]
    async fn close_run_reaches_host_shutdown_and_reports_honestly() {
        within("close_run", async {
            // upholds: HOST-3 — a recorder in evidence failure cannot skip
            // the joined backend shutdown; the disposition tells the truth.
            let reached = Arc::new(Mutex::new(false));
            let reached_by = reached.clone();
            let seen = Arc::new(Mutex::new(None::<&'static str>));
            let seen_by = seen.clone();
            let code = close_run(
                DriveFacts {
                    session: Session::Clean,
                    recorder_failed: true,
                },
                async { Err(anyhow!("shutdown record timed out")) },
                async {
                    *reached_by.lock().unwrap() = true;
                    Ok(())
                },
                || async { (Session::Clean, false) },
                |disposition| async move {
                    *seen_by.lock().unwrap() = Some(disposition);
                    Err(anyhow!("finish fails after evidence failure"))
                },
            )
            .await;
            assert!(*reached.lock().unwrap(), "host shutdown was reached");
            assert_eq!(*seen.lock().unwrap(), Some("recorder-error"));
            assert_eq!(code, 5);

            // The clean session whose host join failed: backend-error, 5.
            let seen = Arc::new(Mutex::new(None::<&'static str>));
            let seen_by = seen.clone();
            let code = close_run(
                DriveFacts {
                    session: Session::Clean,
                    recorder_failed: false,
                },
                async { Ok(()) },
                async { Err(anyhow!("reap failed")) },
                || async { (Session::Clean, false) },
                |disposition| async move {
                    *seen_by.lock().unwrap() = Some(disposition);
                    Ok(())
                },
            )
            .await;
            assert_eq!(*seen.lock().unwrap(), Some("backend-error"));
            assert_eq!(code, 5);

            // A fatal fact surfacing only in the drained tail governs both
            // the disposition and the exit code.
            let seen = Arc::new(Mutex::new(None::<&'static str>));
            let seen_by = seen.clone();
            let code = close_run(
                DriveFacts {
                    session: Session::TurnErrors,
                    recorder_failed: false,
                },
                async { Ok(()) },
                async { Ok(()) },
                || async { (Session::Fatal, false) },
                |disposition| async move {
                    *seen_by.lock().unwrap() = Some(disposition);
                    Ok(())
                },
            )
            .await;
            assert_eq!(*seen.lock().unwrap(), Some("fatal"));
            assert_eq!(code, 3);
        })
        .await;
    }

    #[test]
    fn git_provenance_is_env_supplied_or_absent() {
        // The driver spawns no provenance subprocess; whitespace-only or
        // unset means the field is omitted, never guessed.
        assert_eq!(
            git_describe_note(Some("  abc-dirty ".into())),
            Some("abc-dirty".to_string())
        );
        assert_eq!(git_describe_note(Some("   ".into())), None);
        assert_eq!(git_describe_note(None), None);
    }

    #[tokio::test]
    async fn a_pending_record_cannot_outlive_the_drain_deadline() {
        within("drain record admission", async {
            // The absolute deadline bounds the whole drain, record admission
            // included: a sink whose own bound is longer (or absent) still
            // returns at the drain deadline, marked incomplete.
            let sink = PendingAfterSink {
                inner: VecSink::default(),
                admit: 0,
            };
            let (ev_tx, mut event_rx) = tokio::sync::mpsc::unbounded_channel();
            ev_tx.send(HostEvent::Note("queued".into())).unwrap();
            let started = std::time::Instant::now();
            let (session, recorder_failed) = drain_tail(
                &sink,
                &mut event_rx,
                Duration::from_secs(1),
                Duration::from_millis(100),
            )
            .await;
            drop(ev_tx);
            assert!(started.elapsed() < Duration::from_secs(5), "absolute bound");
            assert_eq!(session, Session::Clean);
            assert!(recorder_failed, "the crossed deadline is an evidence fact");
        })
        .await;
    }
}
