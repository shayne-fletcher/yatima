//! The engine-facing host: one implementation every yatima frontend shares.
//!
//! Local decode is `!Send` and runs on the runtime's blocking island (CMP-1 /
//! RT-2), so it cannot live in a `tokio::spawn`. A dedicated **OS thread** owns
//! the backend — the `!Send` Candle [`Engine`] or a managed llama-server
//! child — *and* the [`ChatSession`]/[`Agent`] — the one authoritative
//! prompt history — for its whole life (HOST-3), and, being a plain thread (not
//! a runtime worker), calls the public **sync** decode shims directly; RT-1 is
//! not violated (the managed lifecycle uses the lib's three narrow sync
//! shims — verify, spawn, shutdown — never a general executor). The TUI,
//! GUI, and yatima-serve are thin views over this host; they differ only in
//! how they draw a [`HostEvent`] and where a [`HostRequest`] comes from.
//!
//! Two planes connect the host to a frontend, plus one out-of-band control:
//!
//! - **request** ([`std::sync::mpsc`], frontend→host): [`HostRequest`]. The
//!   actor blocks on receive between turns and never `.await`s.
//! - **event** ([`tokio::sync::mpsc`], host→frontend): [`HostEvent`] — the
//!   frontend's only source of transcript truth.
//! - **cancel** ([`CancelGate`], out-of-band): the actor owns each turn's
//!   [`Cancel`] and arms the gate with it before decoding, so a frontend can
//!   flip it *mid-decode* (the request queue is unserviced while the actor
//!   decodes). A native frontend calls [`CancelGate::cancel`] on Esc; a serve
//!   session maps a wire [`HostRequest::Cancel`] to the same gate. Both reach
//!   the same handle — this is the one genuinely subtle piece of the split.
//!
//! # Invariant & law registry
//!
//! - **HOST-1** frontends drive turns only through the protocol: none
//!   constructs an [`Agent`]/[`ChatSession`] or calls a yatima-lib decode path
//!   directly — the engine lives here, behind [`HostEvent`]/[`HostRequest`]
//!   (grep-enforced by review).
//! - **HOST-2** the grant/refusal/report wording lives only in this crate —
//!   CAP-3's user-facing contract is single-sourced ([`report_grant`],
//!   [`report_revoke`], [`report_grants`], [`refuse_grant`]; cited by
//!   `grant_wording_is_single_sourced` / `chat_only_reports_name_no_authority`).
//! - **HOST-3** one backend thread owns the backend — the `!Send` Candle
//!   engine (created inside the thread, never crossing a thread boundary) or
//!   the managed llama-server child — and the session, for the whole run.
//!   Every post-construction exit converges on the thread's single
//!   backend-consuming epilogue: a managed child is explicitly killed,
//!   reaped, and its drains joined (LSRV-1 at the host boundary) before the
//!   thread ends; [`HostOwner::shutdown`] is the joined-success witness and
//!   `Drop` only a request fallback. Cited by the hermetic lifecycle battery
//!   in `tests/managed_lifecycle.rs`.
//! - **HOST-4** tool activity crosses the wire as `(kind, payload)` —
//!   [`ToolNoteKind`] carries the semantics, and this crate emits no marker
//!   glyphs or note indentation; the vocabulary a note renders under is view
//!   policy (cited by `notes_carry_kind_not_typography`).
//! - **HOST-5** the host keeps every rendered prompt under the depth budget
//!   — scoped to backends that expose a tokenizer: between turns it trims
//!   the committed history (COMPACT-1) back under a low-water mark
//!   ([`compaction_low_water`] = the depth ceiling less the reply and one
//!   run's within-run tool growth), and compaction is always visible —
//!   history is never edited silently. The ceiling tightens to the Metal KV
//!   validated depth on a Metal run (CTX-2; a Candle-engine envelope, never
//!   applied to a managed server). A backend that cannot count tokens
//!   reports no prompt depth, emits no `Context` event, and never trims —
//!   no estimate is presented as evidence (`/tokenize` is the recorded debt
//!   that restores exact metering for llama-server). Wording single-sourced
//!   in [`compaction_note`]; cited by the arithmetic/wording/trigger tests.

use std::collections::BTreeSet;
use std::ops::ControlFlow;
use std::sync::mpsc::{Receiver, Sender};
use std::sync::{Arc, Mutex};

use anyhow::{Context as _, Result};
use chrono::Local;
use tokio::sync::mpsc::{unbounded_channel, UnboundedReceiver, UnboundedSender};
use yatima_lib::{
    device, looks_degenerate, metal_kv_depth_risk, resolve_format, verify_cancellable_sync, Agent,
    AgentEvent, AgentStop, Cancel, Channel as LibChannel, ChatFormat, ChatSession,
    ChildCleanupFailed, Completer, Engine, GenOpts, ImageListing, JsonToolCall, KvDepthRisk,
    LlamaServer, LlamaServerSpawn, ModelSource, MuseAtemCodec, Plot, PlotSandbox, PromptTemplate,
    QwenToolCall, ReadImage, ReadPage, ReadUrl, Sampling, ServerIdentity, StopReason,
    ToolCallCodec, ToolOutcome, Tools, VerifyCancelled, WebOrigins, METAL_KV_VALIDATED,
};

pub mod knobs;
mod logging;
mod resolve;

pub use logging::{init_file_logging, init_stderr_logging};
pub use resolve::{resolve_host_model, HostBackendConfig, HostModelChoices, ResolvedHostModel};
pub use yatima_protocol::{
    Channel, HostEvent, HostRequest, ModelIdentity, ModelInfo, StartupPhase, StopKind, ToolNoteKind,
};

/// A turn identifier, monotonic per session. Lets a frontend ignore stale events.
pub type TurnId = u64;

/// The gate's interior: the turn currently in flight (armed before it
/// decodes), and the turns whose cancel arrived before they armed.
#[derive(Default)]
struct GateState {
    armed: Option<(TurnId, Cancel)>,
    early: BTreeSet<TurnId>,
}

/// The out-of-band cancel handle: the actor arms it with the in-flight turn's
/// [`Cancel`] before decoding; a frontend flips it mid-decode. A cancel that
/// arrives before its turn is armed — the wire ordering `Submit{n}` then
/// `Cancel{n}` for a turn still queued behind a running one — is remembered
/// and applied the instant that turn arms, so a queued turn a user asked to
/// stop never runs anyway. Cloneable and cheap (an `Arc`); [`spawn`] hands one
/// to the frontend and keeps one for the actor.
#[derive(Clone, Default)]
pub struct CancelGate(Arc<Mutex<GateState>>);

/// The most early cancels the gate remembers at once. Turn ids are monotonic
/// and spent ids are pruned as turns arm, so this is only reached by a client
/// spraying cancels for turns it never submits — then the oldest is evicted.
const EARLY_CANCEL_CAP: usize = 1024;

impl CancelGate {
    /// A fresh, disarmed gate.
    pub fn new() -> CancelGate {
        CancelGate::default()
    }

    /// Arm the gate with the turn about to decode (the host's job, per turn).
    /// A cancel that arrived early for this turn fires now; ids at or below it
    /// are spent (monotonic turns) and pruned.
    pub fn arm(&self, turn_id: TurnId, cancel: Cancel) {
        if let Ok(mut state) = self.0.lock() {
            let fire = state.early.remove(&turn_id);
            state.early = state.early.split_off(&turn_id);
            state.armed = Some((turn_id, cancel.clone()));
            if fire {
                cancel.cancel();
            }
        }
    }

    /// Disarm after a turn finishes (a stale `cancel(turn_id)` then no-ops).
    /// Early cancels for turns not yet armed survive — they are the point.
    pub fn disarm(&self) {
        if let Ok(mut state) = self.0.lock() {
            state.armed = None;
        }
    }

    /// Cancel whatever turn is armed right now, whichever id it carries —
    /// the owner's shutdown path: a mid-decode turn must end promptly without
    /// the owner knowing its id. A disarmed gate is a no-op.
    pub fn cancel_armed(&self) {
        if let Ok(state) = self.0.lock() {
            if let Some((_, cancel)) = state.armed.as_ref() {
                cancel.cancel();
            }
        }
    }

    /// Cancel `turn_id`. If it is the one in flight, flip it now. Otherwise it
    /// is either a queued turn not yet armed (remember it — [`arm`] applies it
    /// when the turn starts) or a stale id for a finished turn (harmless: a
    /// monotonic turn id never arms again, and the next arm prunes it).
    pub fn cancel(&self, turn_id: TurnId) {
        if let Ok(mut state) = self.0.lock() {
            match state.armed.as_ref() {
                Some((id, cancel)) if *id == turn_id => cancel.cancel(),
                _ => {
                    if state.early.len() >= EARLY_CANCEL_CAP {
                        if let Some(&oldest) = state.early.iter().next() {
                            state.early.remove(&oldest);
                        }
                    }
                    state.early.insert(turn_id);
                }
            }
        }
    }
}

/// What the host needs to run a model (all `Send`, so it crosses into the
/// thread; the `!Send` [`Engine`] is then *created* inside the thread —
/// HOST-3). The backend arrives **unresolved** ([`HostBackendConfig`]):
/// acquisition happens inside the actor as part of its owned lifecycle, so
/// no frontend holds a resolved path it could substitute before launch.
pub struct HostConfig {
    pub(crate) backend: HostBackendConfig,
    pub(crate) opts: GenOpts,
    pub(crate) format: Option<ChatFormat>,
    pub(crate) system: Option<String>,
    /// Display label; `None` labels with the resolved model directory.
    pub(crate) model_label: Option<String>,
    /// Test/diagnostic wiring only (see [`HostConfig::with_managed_launcher`]).
    pub(crate) managed_launcher: Option<ManagedLauncher>,
}

/// Test/diagnostic override for the managed child: which binary to launch
/// and how long to wait for readiness. Never set by the shared resolver.
#[derive(Debug, Clone)]
pub(crate) struct ManagedLauncher {
    pub(crate) binary: std::path::PathBuf,
    pub(crate) readiness_timeout: std::time::Duration,
}

impl HostConfig {
    /// A profile-less Candle engine session — the explicit-source path
    /// (CLI-1 chose exactly one source). Profile-driven configs go through
    /// [`resolve_host_model`] + [`ResolvedHostModel::into_host_config`],
    /// which layer the profile's recipe (PROFILE-1) and cannot be
    /// recombined field-by-field: this type is opaque so the invalid states
    /// the resolver removed cannot be reconstructed (PROFILE-2).
    pub fn engine(
        source: ModelSource,
        cpu: bool,
        opts: GenOpts,
        format: Option<ChatFormat>,
        system: Option<String>,
        model_label: Option<String>,
    ) -> HostConfig {
        HostConfig {
            backend: HostBackendConfig::Engine { source, cpu },
            opts,
            format,
            system,
            model_label,
            managed_launcher: None,
        }
    }

    /// A managed llama-server session, valid by construction: only a
    /// profile that pins the llama-server backend and a chat format can
    /// build one, its source is the profile's own (PROFILE-2), and its
    /// generation options are the profile's recipe layered over `base`
    /// (PROFILE-1) — a mismatched format or an undercut recipe cannot be
    /// assembled.
    pub fn managed(
        profile: &yatima_lib::ModelProfile,
        offline: bool,
        base: GenOpts,
        system: Option<String>,
    ) -> Result<HostConfig> {
        let yatima_lib::ProfileBackend::LlamaServer(server) = &profile.backend else {
            anyhow::bail!(
                "profile {:?} does not pin the llama-server backend",
                profile.name
            );
        };
        let Some(format) = profile.format() else {
            anyhow::bail!(
                "managed profile {:?} does not pin a chat format",
                profile.name
            );
        };
        Ok(HostConfig {
            backend: HostBackendConfig::ManagedLlamaServer {
                source: profile.to_source(offline)?,
                profile: server.clone(),
            },
            opts: profile.apply_gen_overrides(base),
            format: Some(format),
            system,
            model_label: Some(profile.name.clone()),
            managed_launcher: None,
        })
    }

    /// The display label, when the resolution carried one (a profile name).
    /// Read-only: observation cannot reconstruct the invalid states this
    /// type's opacity removed.
    pub fn model_label(&self) -> Option<&str> {
        self.model_label.as_deref()
    }

    /// Test/diagnostic wiring only: launch `binary` as the managed server
    /// (instead of `llama-server` from `PATH`) and wait `readiness_timeout`
    /// for it. The hermetic battery points this at the protocol stub; the
    /// shared resolver never sets it.
    #[doc(hidden)]
    pub fn with_managed_launcher(
        mut self,
        binary: std::path::PathBuf,
        readiness_timeout: std::time::Duration,
    ) -> HostConfig {
        self.managed_launcher = Some(ManagedLauncher {
            binary,
            readiness_timeout,
        });
        self
    }
}

/// The movable frontend planes of a running host: the request sender, the
/// event receiver, and the turn-cancel gate. Views may move this freely (the
/// GUI's event pump, serve's bridge); thread ownership stays behind
/// [`HostOwner`].
pub struct HostClient {
    pub req_tx: Sender<HostRequest>,
    pub event_rx: UnboundedReceiver<HostEvent>,
    pub cancel: CancelGate,
}

/// The one owner of the backend thread: lifecycle cancellation, a shutdown
/// path of its own, the actor-epilogue completion signal, and the OS
/// `JoinHandle`. Exactly one exists per host; [`shutdown`](HostOwner::shutdown)
/// consumes it. Dropping it without shutdown only *requests* shutdown — the
/// fallback is never the joined-success witness (HOST-3 / LSRV-1).
pub struct HostOwner {
    req_tx: Sender<HostRequest>,
    lifecycle: Cancel,
    cancel: CancelGate,
    done_rx: Option<tokio::sync::oneshot::Receiver<Result<()>>>,
    thread: Option<std::thread::JoinHandle<()>>,
}

impl HostOwner {
    /// Shut the host down and prove it finished: flip the lifecycle cancel
    /// (startup observes it between phases and inside the hash), cancel any
    /// armed turn ([`CancelGate::cancel_armed`] — token-level, so a
    /// mid-decode turn ends promptly), request shutdown on the wire, await
    /// the actor's epilogue under [`knobs::SHUTDOWN_WITHIN`], join the OS
    /// thread, and only then report the epilogue's own result — a failed
    /// kill/reap/drain-join is an error here, never blessed (HOST-3 /
    /// LSRV-1: this is the joined-success witness, so it must not lie).
    ///
    /// If the bound elapses (the actor is inside phase-unbounded work — a
    /// model fetch has no finite bound), ownership is not silently dropped:
    /// the join obligation transfers to a background reaper task that
    /// awaits the epilogue and joins the thread whenever it finishes,
    /// logging the outcome — and the returned error says so.
    pub async fn shutdown(mut self) -> Result<()> {
        self.lifecycle.cancel();
        self.cancel.cancel_armed();
        let _ = self.req_tx.send(HostRequest::Shutdown);
        let done_rx = self.done_rx.take().expect("shutdown consumes the owner");
        let thread = self.thread.take().expect("shutdown consumes the owner");
        await_epilogue(done_rx, thread, knobs::SHUTDOWN_WITHIN).await
    }
}

/// The owner's wait: epilogue result under `within`, then the thread join,
/// then the epilogue's own verdict. On timeout the join obligation is
/// handed to a background reaper (never dropped), and the error names the
/// transfer. Factored from [`HostOwner::shutdown`] so the timeout path has
/// a hermetic witness.
async fn await_epilogue(
    mut done_rx: tokio::sync::oneshot::Receiver<Result<()>>,
    thread: std::thread::JoinHandle<()>,
    within: std::time::Duration,
) -> Result<()> {
    match tokio::time::timeout(within, &mut done_rx).await {
        Ok(signal) => {
            tokio::task::spawn_blocking(move || thread.join())
                .await
                .context("join the backend thread")?
                .map_err(|panic| anyhow::anyhow!("backend thread panicked: {panic:?}"))?;
            match signal {
                // The epilogue's own result: a failed managed kill/reap/
                // drain-join surfaces here, after the join.
                Ok(epilogue) => epilogue,
                // Sender dropped without a value: the actor was torn down
                // some other way; the join above already surfaced a panic.
                Err(_) => Ok(()),
            }
        }
        Err(_) => {
            spawn_reaper(done_rx, thread);
            anyhow::bail!(
                "host actor did not finish within {within:?} (it is inside \
                 phase-unbounded work, e.g. a model fetch); the join obligation \
                 was handed to a background reaper thread — the actor's own \
                 epilogue still reaps its child when the work completes"
            )
        }
    }
}

/// Hand the actor's join obligation to a dedicated **OS thread** — never a
/// tokio task, which the frontend's runtime would abort at teardown, and
/// never dropped: the reaper blocks on the completion signal (a tokio
/// oneshot supports a sync `blocking_recv` off-runtime), joins the actor
/// thread, and logs both verdicts. Process exit remains the outer bound —
/// no user-space reaper survives it — but within the process the actor is
/// always joined by someone. Returns the reaper's own handle for the
/// hermetic witness; production detaches it (the reaper holds everything it
/// needs).
fn spawn_reaper(
    done_rx: tokio::sync::oneshot::Receiver<Result<()>>,
    thread: std::thread::JoinHandle<()>,
) -> std::thread::JoinHandle<()> {
    std::thread::Builder::new()
        .name("yatima-backend-reaper".into())
        .spawn(move || {
            let epilogue = done_rx.blocking_recv();
            let joined = thread.join();
            tracing::warn!(
                "backend thread finished after the shutdown bound: epilogue {:?}, join {:?}",
                epilogue.map(|r| r.map_err(|e| format!("{e:#}"))),
                joined.as_ref().map_err(|p| format!("{p:?}"))
            );
        })
        .expect("spawn the backend reaper thread")
}

impl Drop for HostOwner {
    /// Fallback only: request shutdown and cancel work so an abandoned host
    /// winds down (the actor's own epilogue still reaps its child), but
    /// nothing here waits or joins — that proof requires
    /// [`shutdown`](HostOwner::shutdown).
    fn drop(&mut self) {
        if self.done_rx.is_some() {
            self.lifecycle.cancel();
            self.cancel.cancel_armed();
            let _ = self.req_tx.send(HostRequest::Shutdown);
        }
    }
}

/// Launch the backend thread and return the movable client planes and the one
/// owner at once. The thread resolves, verifies (when the profile pins a
/// digest), and starts its backend, reporting each transition as
/// [`HostEvent::Startup`] and then [`HostEvent::Ready`] (or
/// [`HostEvent::Fatal`]); nothing here waits for any of it.
pub fn spawn_nonblocking(config: HostConfig) -> Result<(HostClient, HostOwner)> {
    let (req_tx, req_rx) = std::sync::mpsc::channel::<HostRequest>();
    let (event_tx, event_rx) = unbounded_channel::<HostEvent>();
    let (done_tx, done_rx) = tokio::sync::oneshot::channel::<Result<()>>();
    let gate = CancelGate::new();
    let lifecycle = Cancel::new();
    let actor_gate = gate.clone();
    let actor_lifecycle = lifecycle.clone();
    let thread = std::thread::Builder::new()
        .name("yatima-backend".into())
        .spawn(move || {
            let epilogue = actor_main(config, req_rx, event_tx, actor_gate, actor_lifecycle);
            // Signalled after the epilogue, carrying its result: the owner's
            // await, then its thread join, observe a finished actor — and a
            // failed reap is reported, never blessed (HOST-3 / LSRV-1).
            let _ = done_tx.send(epilogue);
        })?;
    Ok((
        HostClient {
            req_tx: req_tx.clone(),
            event_rx,
            cancel: gate.clone(),
        },
        HostOwner {
            req_tx,
            lifecycle,
            cancel: gate,
            done_rx: Some(done_rx),
            thread: Some(thread),
        },
    ))
}

/// Launch the host and wait for readiness, returning the planes, the owner,
/// and what's running — or an error if startup failed (surfaced *before* the
/// caller touches its screen). Startup phase events are consumed here; a
/// frontend that displays them uses [`spawn_nonblocking`] instead.
pub async fn spawn(config: HostConfig) -> Result<(HostClient, HostOwner, ModelInfo)> {
    let (mut client, owner) = spawn_nonblocking(config)?;
    loop {
        match client.event_rx.recv().await {
            Some(HostEvent::Startup { .. }) => continue,
            Some(HostEvent::Ready(info)) => return Ok((client, owner, info)),
            Some(HostEvent::Fatal(message)) => {
                let joined = owner.shutdown().await;
                return Err(match joined {
                    Ok(()) => anyhow::anyhow!(message),
                    Err(join_error) => anyhow::anyhow!(message).context(join_error),
                });
            }
            _ => {
                let _ = owner.shutdown().await;
                anyhow::bail!("backend thread exited before reporting readiness");
            }
        }
    }
}

/// The actor's body: build the backend (reporting startup phases), report
/// readiness, serve requests until an exit condition, then run the **one
/// backend-consuming epilogue** — every post-construction exit converges
/// there, so a managed child is always explicitly killed, reaped, and its
/// drains joined before the thread ends (HOST-3 / LSRV-1).
fn actor_main(
    config: HostConfig,
    req_rx: Receiver<HostRequest>,
    event_tx: UnboundedSender<HostEvent>,
    gate: CancelGate,
    lifecycle: Cancel,
) -> Result<()> {
    let HostConfig {
        backend,
        opts,
        format: format_choice,
        system,
        model_label,
        managed_launcher,
    } = config;
    let built = match build_backend(
        backend,
        format_choice,
        managed_launcher,
        &lifecycle,
        &event_tx,
    ) {
        Ok(Some(built)) => built,
        // Lifecycle cancelled during startup: a requested exit, not a
        // failure — every child-owning cancellation branch reaped before
        // returning, or propagated its failure as a CleanupFailed error.
        Ok(None) => return Ok(()),
        Err(e) => {
            // The actor's return value is its *ownership* verdict, not its
            // startup verdict: a startup failure that self-cleaned (the lib
            // reaps before erroring) is reported on the event plane; only an
            // unproven cleanup — the lib's typed ChildCleanupFailed marker,
            // stamped on any spawn failure whose own kill/reap/drain-join
            // also failed, or on this actor's cancelled-after-spawn reap —
            // may fail the owner's joined shutdown.
            let cleanup_debt = e.downcast_ref::<ChildCleanupFailed>().is_some();
            if !lifecycle.is_cancelled() {
                let _ = event_tx.send(HostEvent::Fatal(format!("{e:#}")));
            }
            return if cleanup_debt { Err(e) } else { Ok(()) };
        }
    };
    let (mut backend, facts) = built;

    let label = model_label.unwrap_or(facts.default_label);
    let format = facts.format;
    let info = ModelInfo {
        label,
        arch: facts.arch,
        backend: facts.backend,
        device: facts.device,
        format: format!("{format:?}"),
        sampling: sampling_summary(opts.sampling),
        max_tokens: opts.max_tokens,
        context_length: facts.context_length,
        identity: facts.identity,
    };
    // The CTX-2 / HOST-5 surface is scoped to the tokenizing engine: a
    // managed server counts no tokens, so its watch never sees a depth and
    // never warns or trims (no estimate stands in — HOST-5).
    let watch = DepthWatch {
        metal: facts.engine_on_metal,
        max_tokens: opts.max_tokens,
        context_length: facts.watch_context,
    };
    // Muse's template carries the runtime date; chosen once at host startup
    // so every turn of the session renders the same prompt bytes.
    let current_date = Local::now().format("%Y-%m-%d").to_string();

    let served = if event_tx.send(HostEvent::Ready(info)).is_ok() {
        serve_session(
            &mut backend,
            format,
            current_date,
            system,
            opts,
            watch,
            &req_rx,
            &event_tx,
            &gate,
        )
    } else {
        Ok(())
    };
    // The one epilogue. Its verdict joins any debt the serve loop carried
    // out (a mid-completion child death whose drain-join failed consumed
    // the handles — the epilogue cannot re-prove that cleanup): both reach
    // the owner through the completion signal and fail
    // `HostOwner::shutdown` — the joined-success witness must never report
    // success over an unproven reap (HOST-3 / LSRV-1).
    let disposed = backend.dispose().context("backend epilogue");
    match (served, disposed) {
        (Ok(()), disposed) => disposed,
        (Err(debt), Ok(())) => Err(debt),
        (Err(debt), Err(disposed)) => {
            Err(debt.context(format!("backend epilogue also failed: {disposed:#}")))
        }
    }
}

/// The backend the actor thread owns for the whole run (HOST-3): the `!Send`
/// Candle [`Engine`] (created in-thread) or a managed [`LlamaServer`] child.
/// [`Completer`] by delegation, so every serve loop is backend-independent;
/// consumed only by [`dispose`](HostBackend::dispose) — the epilogue.
enum HostBackend {
    // Both boxed for variant-size parity (clippy::large_enum_variant): each
    // is a large value, and the enum is created once per session.
    Engine(Box<Engine>),
    LlamaServer(Box<LlamaServer>),
}

/// The marker context a turn error carries when the managed child is found
/// dead ([`LlamaServer::exited`]): the serve loop downcasts to distinguish
/// fatal backend loss (exit through the epilogue) from a recoverable turn
/// error (session continues).
#[derive(Debug)]
struct BackendLost(String);

impl std::fmt::Display for BackendLost {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

/// Classify a child probe after a failed completion: `Some(message)` is
/// fatal backend loss (the serve loop exits through the epilogue), `None`
/// leaves the error recoverable. An exited child is loss; a probe *failure*
/// is also loss — inability to observe the child is not evidence of a
/// healthy backend (pure, witnessed by unit test).
fn probe_verdict(probe: std::io::Result<Option<std::process::ExitStatus>>) -> Option<String> {
    match probe {
        Ok(Some(status)) => Some(format!("managed llama-server exited ({status})")),
        Ok(None) => None,
        Err(error) => Some(format!(
            "cannot observe the managed llama-server child (try_wait failed: {error}); \
             treating the backend as lost"
        )),
    }
}

impl HostBackend {
    /// The one backend-consuming epilogue: a managed child is explicitly
    /// killed, reaped, and its output drains joined (LSRV-1, through the
    /// narrow RT-1 shutdown shim); the engine drops in-thread. `Drop` never
    /// substitutes for this.
    fn dispose(self) -> Result<()> {
        match self {
            Self::Engine(_) => Ok(()),
            Self::LlamaServer(server) => (*server).shutdown_sync().map(|_| ()),
        }
    }
}

impl Completer for HostBackend {
    async fn complete(
        &mut self,
        prompt: &str,
        opts: &GenOpts,
        stops: &[String],
    ) -> Result<yatima_lib::Completion> {
        match self {
            Self::Engine(engine) => engine.complete(prompt, opts, stops).await,
            Self::LlamaServer(server) => server.complete(prompt, opts, stops).await,
        }
    }

    fn count_tokens(&self, text: &str) -> Option<usize> {
        match self {
            Self::Engine(engine) => engine.count_tokens(text),
            // No tokenizer: no prompt depth is ever reported, so no Context
            // event, no depth warning, and no compaction claim (HOST-5 is
            // scoped to tokenizing backends; /tokenize is the recorded debt).
            Self::LlamaServer(_) => None,
        }
    }

    async fn complete_streaming(
        &mut self,
        prompt: &str,
        opts: &GenOpts,
        stops: &[String],
        cancel: &Cancel,
        on_token: &mut dyn FnMut(&str),
    ) -> Result<yatima_lib::Completion> {
        match self {
            Self::Engine(engine) => {
                engine
                    .complete_streaming(prompt, opts, stops, cancel, on_token)
                    .await
            }
            Self::LlamaServer(server) => {
                let outcome = server
                    .complete_streaming(prompt, opts, stops, cancel, on_token)
                    .await;
                match outcome {
                    Err(error) => {
                        // A dying child's socket error can precede its
                        // reapability by a few milliseconds; probe briefly
                        // (bounded) so loss is not misread as recoverable.
                        let mut probe = server.exited();
                        for _ in 0..3 {
                            if !matches!(probe, Ok(None)) {
                                break;
                            }
                            tokio::time::sleep(std::time::Duration::from_millis(20)).await;
                            probe = server.exited();
                        }
                        match probe_verdict(probe) {
                            // The child is gone — or cannot even be observed
                            // (an owner without observation holds no evidence
                            // of a healthy backend): mark the error so the
                            // serve loop exits through the epilogue instead
                            // of retrying a backend it cannot account for.
                            Some(loss) => Err(error.context(BackendLost(loss))),
                            None => Err(error),
                        }
                    }
                    ok => ok,
                }
            }
        }
    }
}

/// Backend construction facts the actor needs after the backend exists —
/// every display field preformatted here, where the evidence lives.
struct BackendFacts {
    default_label: String,
    arch: String,
    /// The resolved chat format: inferred from the engine's architecture
    /// (FMT-1/FMT-2), or the managed profile's pin (a server exposes no
    /// architecture to infer from).
    format: ChatFormat,
    backend: String,
    device: String,
    context_length: Option<usize>,
    identity: ModelIdentity,
    /// Whether the *engine* decodes on Metal (drives the CTX-2 KV-depth
    /// warning — a Candle-specific envelope, never applied to a server).
    engine_on_metal: bool,
    /// The depth ceiling the HOST-5 watch enforces; `None` for a backend
    /// whose prompt depth is never known (no tokenizer — nothing to enforce,
    /// nothing to estimate).
    watch_context: Option<usize>,
}

/// Build the configured backend inside the actor thread, reporting each
/// startup phase transition (PROTO-2's vocabulary) and observing the
/// lifecycle cancel between phases and *inside* the hash (checked between
/// read chunks — a shutdown mid-hash is answered within one chunk, never
/// after the whole file). `Ok(None)` is a cancelled startup; a cancel
/// observed after the child launched reaps it before returning, and a
/// failed reap is the typed ownership debt the owner reports (LSRV-1).
fn build_backend(
    config: HostBackendConfig,
    format_choice: Option<ChatFormat>,
    managed_launcher: Option<ManagedLauncher>,
    lifecycle: &Cancel,
    event_tx: &UnboundedSender<HostEvent>,
) -> Result<Option<(HostBackend, BackendFacts)>> {
    let phase = |phase: StartupPhase| {
        let _ = event_tx.send(HostEvent::Startup { phase });
    };
    match config {
        HostBackendConfig::Engine { source, cpu } => {
            phase(StartupPhase::ResolvingModel);
            let dir = source.resolve()?.into_directory();
            if lifecycle.is_cancelled() {
                return Ok(None);
            }
            phase(StartupPhase::StartingBackend);
            let engine = Engine::load(&dir, device(cpu)?)?;
            if lifecycle.is_cancelled() {
                return Ok(None);
            }
            let (format, _mismatch) = resolve_format(engine.arch(), format_choice);
            let facts = BackendFacts {
                default_label: dir.display().to_string(),
                arch: format!("{:?}", engine.arch()),
                format,
                backend: engine.backend(),
                device: if cpu { "cpu" } else { "gpu" }.to_string(),
                context_length: engine.context_length(),
                // A directory load verifies no digest: no authenticated
                // identity evidence exists (LSRV-5's verified form is the
                // managed path's).
                identity: ModelIdentity::Unverified,
                engine_on_metal: !cpu,
                watch_context: engine.context_length(),
            };
            Ok(Some((HostBackend::Engine(Box::new(engine)), facts)))
        }
        HostBackendConfig::ManagedLlamaServer { source, profile } => {
            // No engine architecture exists to infer a format from, so the
            // profile's pin is the only source (the resolver guarantees it).
            // Reject a hand-built config without one before any resolution,
            // hashing, or child-owning work.
            let Some(format) = format_choice else {
                anyhow::bail!("a managed llama-server backend requires a pinned chat format");
            };
            phase(StartupPhase::ResolvingModel);
            let artifact = source.resolve()?.into_gguf()?;
            if lifecycle.is_cancelled() {
                return Ok(None);
            }
            phase(StartupPhase::VerifyingModel);
            // The hash observes the lifecycle cancel between chunks: an
            // owner shutting down mid-hash is answered within one chunk,
            // never after the whole 17 GB (the real Muse hash exceeds any
            // polite shutdown bound).
            let verified =
                match verify_cancellable_sync(artifact, &profile.expected_sha256, lifecycle) {
                    Ok(verified) => verified,
                    Err(error) if error.downcast_ref::<VerifyCancelled>().is_some() => {
                        return Ok(None);
                    }
                    Err(error) => return Err(error),
                };
            if lifecycle.is_cancelled() {
                // Cancelled in the gap after hashing: still before launch.
                return Ok(None);
            }
            phase(StartupPhase::StartingBackend);
            let mut spec = LlamaServerSpawn::verified(verified, profile.server_gates());
            spec.context = Some(profile.context);
            spec.top_k = profile.top_k;
            if let Some(launcher) = managed_launcher {
                spec.binary = launcher.binary;
                spec.readiness_timeout = launcher.readiness_timeout;
            }
            let server = LlamaServer::spawn_sync(spec)?;
            if lifecycle.is_cancelled() {
                // Cancelled after launch: the child must still be reaped
                // before this thread gives up on it (LSRV-1) — and a failed
                // reap is an ownership debt the owner's shutdown reports.
                server.shutdown_sync().context(ChildCleanupFailed)?;
                return Ok(None);
            }
            let identity = match server.identity() {
                ServerIdentity::Verified { digest } => {
                    ModelIdentity::VerifiedSha256(digest.to_string())
                }
                // Unreachable for a verified spawn, but stated rather than
                // assumed: identity never exceeds its evidence (LSRV-5).
                _ => ModelIdentity::Unverified,
            };
            let arch = server
                .launched_artifact()
                .file_stem()
                .map(|stem| stem.to_string_lossy().into_owned())
                .unwrap_or_else(|| "gguf".to_string());
            let facts = BackendFacts {
                default_label: server.launched_artifact().display().to_string(),
                arch,
                format,
                backend: server.props().build.clone(),
                device: "external".to_string(),
                context_length: Some(server.props().n_ctx as usize),
                identity,
                engine_on_metal: false,
                watch_context: None,
            };
            Ok(Some((HostBackend::LlamaServer(Box::new(server)), facts)))
        }
    }
}

/// The one-line sampling summary for the status rail.
fn sampling_summary(sampling: Sampling) -> String {
    match sampling {
        Sampling::Greedy => "greedy".to_string(),
        Sampling::Sample {
            temperature,
            top_p,
            seed,
        } => match top_p {
            Some(p) => format!("temp {temperature:.2} · top-p {p:.2} · seed {seed}"),
            None => format!("temp {temperature:.2} · seed {seed}"),
        },
    }
}

/// Per-turn facts the CTX-2 / HOST-5 surface needs: whether decode runs on
/// Metal, the token budget the risk bound adds to the prompt depth, and the
/// model's declared context window (the compaction ceiling when it is tighter
/// than the Metal KV depth, or the only ceiling off Metal).
#[derive(Clone, Copy)]
struct DepthWatch {
    metal: bool,
    max_tokens: usize,
    context_length: Option<usize>,
}

/// CTX-2, surfaced: the engine logs the depth risk where no user looks
/// (`tracing::warn!` behind `$YATIMA_LOG`); the frontends must *show* it, or
/// a degenerate answer reads as a broken model rather than a known Metal
/// cliff (`notes/metal-kv-cliff.md`). Quiet on CPU and in the mitigated
/// band — only the unreliable depth speaks, as an always-visible app-plane
/// [`HostEvent::Note`], never inside a foldable reasoning pane.
fn warn_kv_depth(event_tx: &UnboundedSender<HostEvent>, watch: DepthWatch, prompt_tokens: usize) {
    if !watch.metal {
        return;
    }
    if metal_kv_depth_risk(prompt_tokens, watch.max_tokens) == Some(KvDepthRisk::Unreliable) {
        let _ = event_tx.send(HostEvent::Note(format!(
            "warning: this turn's context (~{prompt_tokens} tokens, up to \
             ~{} with the reply) is past the ~{METAL_KV_VALIDATED} the Metal \
             corruption workaround is validated to — output may degenerate; \
             /reset starts clean (grants survive) [CTX-2]",
            prompt_tokens.saturating_add(watch.max_tokens),
        )));
    }
}

/// The depth ceiling every rendered prompt must stay under (HOST-5): the
/// model's declared context window, tightened to the Metal KV validated depth
/// on a Metal run. `None` off Metal with no declared window — nothing bounds
/// depth, so compaction never fires.
fn depth_ceiling(watch: DepthWatch) -> Option<usize> {
    match (watch.metal, watch.context_length) {
        (true, Some(c)) => Some(c.min(METAL_KV_VALIDATED)),
        (true, None) => Some(METAL_KV_VALIDATED),
        (false, c) => c,
    }
}

/// The token budget compaction trims the committed history down to (HOST-5):
/// the depth ceiling less the reply budget (`max_tokens`) and one run's
/// within-run tool growth ([`knobs::TOOL_HEADROOM`]), so the deepest step of
/// the next turn stays under the ceiling. `None` when no ceiling applies.
fn compaction_low_water(watch: DepthWatch) -> Option<usize> {
    let ceiling = depth_ceiling(watch)?;
    Some(
        ceiling
            .saturating_sub(watch.max_tokens)
            .saturating_sub(knobs::TOOL_HEADROOM),
    )
}

/// The always-visible compaction notice (HOST-5). Wording single-sourced here
/// like the grant wording (HOST-2); unit-tested. Names the depth budget so the
/// drop reads as a known limit, not lost memory by accident.
fn compaction_note(exchanges: usize, watch: DepthWatch) -> String {
    let ceiling = depth_ceiling(watch).unwrap_or(METAL_KV_VALIDATED);
    let plural = if exchanges == 1 {
        "exchange"
    } else {
        "exchanges"
    };
    format!(
        "compacted: dropped the {exchanges} oldest {plural} to stay under the \
         reliable context depth (~{ceiling} tokens on this backend) — older \
         turns are gone from memory; /reset clears everything"
    )
}

/// COMPACT-1's *policy* (HOST-5): between turns, if the run just served
/// reached deeper than the low-water mark, trim the committed history back
/// under it via `trim` (which returns how many turns it dropped) and tell the
/// user, always visibly. A no-op when no depth ceiling applies, when the run
/// stayed under the mark, or when nothing needed dropping (a deep run whose
/// depth was all within-run tool growth leaves history untouched and stays
/// silent). Never mid-run: the serve loop calls this only after a turn ends.
fn compact_after_turn(
    event_tx: &UnboundedSender<HostEvent>,
    watch: DepthWatch,
    last_prompt_tokens: Option<usize>,
    trim: impl FnOnce(usize) -> usize,
) {
    let Some(low_water) = compaction_low_water(watch) else {
        return;
    };
    let Some(depth) = last_prompt_tokens else {
        return;
    };
    if depth <= low_water {
        return;
    }
    let dropped_turns = trim(low_water);
    if dropped_turns >= 2 {
        let _ = event_tx.send(HostEvent::Note(compaction_note(dropped_turns / 2, watch)));
    }
}

/// Tell the user when a final answer looked degenerate and so was not kept
/// (CHAT-2 / AGENT-3's degenerate case — the lib already withheld the
/// commit; without this note the silent non-commit would be indistinguishable
/// from normal memory).
fn note_degenerate_answer(event_tx: &UnboundedSender<HostEvent>, answer: &str) {
    if looks_degenerate(answer) {
        let _ = event_tx.send(HostEvent::Note(
            "the answer above looks degenerate (decode corruption), so it was \
             not kept in session history — re-ask, or /reset if it recurs"
                .to_string(),
        ));
    }
}

/// Serve the whole session over any backend: [`ChatFormat::supports_tools`]
/// is the one tool-eligibility decision (CAPS-1) — a tool-trained format
/// pairs its native codec with its dated template and serves the sessionful
/// agent; every other format serves plain chat. The web tools start with an
/// empty origin set — hidden from the model (CAP-3a) and inert until a user
/// grant arrives (CAP-3).
#[allow(clippy::too_many_arguments)]
fn serve_session<C: Completer>(
    completer: &mut C,
    format: ChatFormat,
    current_date: String,
    system: Option<String>,
    opts: GenOpts,
    watch: DepthWatch,
    req_rx: &Receiver<HostRequest>,
    event_tx: &UnboundedSender<HostEvent>,
    gate: &CancelGate,
) -> Result<()> {
    if !format.supports_tools() {
        return serve_chat(
            completer,
            format,
            current_date,
            system,
            opts,
            watch,
            req_rx,
            event_tx,
            gate,
        );
    }
    let origins = WebOrigins::new();
    // Client construction cannot practically fail; degrade to empty tools
    // (the model simply never sees web tools) rather than dying.
    let tools = web_tools(&origins).unwrap_or_default();
    let system = system.unwrap_or_else(|| knobs::DEFAULT_AGENT_SYSTEM.to_string());
    let template = format.template_with_date(Some(current_date));
    match format {
        ChatFormat::Qwen => serve_agent(
            completer,
            &tools,
            QwenToolCall,
            template,
            system,
            opts,
            watch,
            &origins,
            req_rx,
            event_tx,
            gate,
        ),
        ChatFormat::Plain => serve_agent(
            completer,
            &tools,
            JsonToolCall,
            template,
            system,
            opts,
            watch,
            &origins,
            req_rx,
            event_tx,
            gate,
        ),
        ChatFormat::MuseGlimmer => serve_agent(
            completer,
            &tools,
            MuseAtemCodec,
            template,
            system,
            opts,
            watch,
            &origins,
            req_rx,
            event_tx,
            gate,
        ),
        // supports_tools() names exactly the codec-backed formats; a format
        // it admits without an arm here is a compile-time review failure,
        // caught by the caps_for/supports_tools witnesses (CAPS-1).
        other => unreachable!("supports_tools admitted {other:?} without a codec"),
    }
}

/// The chat serve loop for chat-only formats: the plain streaming
/// [`ChatSession`]. Grants are always refused here (CAPS-1 — a chat-only format
/// cannot enter the tool path); tool-trained formats never enter this loop.
#[allow(clippy::too_many_arguments)]
fn serve_chat<C: Completer>(
    completer: &mut C,
    format: ChatFormat,
    current_date: String,
    system: Option<String>,
    opts: GenOpts,
    watch: DepthWatch,
    req_rx: &Receiver<HostRequest>,
    event_tx: &UnboundedSender<HostEvent>,
    gate: &CancelGate,
) -> Result<()> {
    let template = format.template_with_date(Some(current_date));
    let mut session = ChatSession::new(completer, template).with_opts(opts);
    if let Some(system) = system {
        session = session.with_system(system);
    }

    while let Ok(req) = req_rx.recv() {
        // A vanished event plane means no frontend can ever see another
        // event: exit through the epilogue rather than serving the void.
        if event_tx.is_closed() {
            return Ok(());
        }
        match req {
            HostRequest::Submit { turn_id, text } => {
                let cancel = Cancel::new();
                gate.arm(turn_id, cancel.clone());
                let outcome = run_turn(&mut session, event_tx, turn_id, &text, &cancel, watch);
                gate.disarm();
                if let ControlFlow::Break(debt) = outcome {
                    // Fatal backend loss: converge on the epilogue, carrying
                    // any unproven-cleanup debt to the actor's verdict.
                    return debt.map_or(Ok(()), Err);
                }
                // Between turns, keep the next prompt under the depth budget
                // (HOST-5) — never mid-run.
                compact_after_turn(event_tx, watch, session.last_prompt_tokens(), |budget| {
                    session
                        .trim_history_to(budget, knobs::COMPACTION_KEEP_LAST)
                        .len()
                });
            }
            HostRequest::Cancel { turn_id } => gate.cancel(turn_id),
            HostRequest::Reset => session.reset(),
            HostRequest::Grant { origin } => refuse_grant(event_tx, format, &origin),
            HostRequest::Revoke { origin } => report_revoke(event_tx, None, &origin),
            HostRequest::ListGrants => report_grants(event_tx, None),
            HostRequest::Shutdown => return Ok(()),
            _ => {} // a future request variant this host predates: ignore it.
        }
    }
    // Request disconnect: every sender is gone — a requested end, no debt.
    Ok(())
}

/// The agent serve loop: one sessionful [`Agent`] (AGENT-3) serves every turn,
/// seeded with the chat phase's history. Later grants/revokes mutate the shared
/// origin set in place — the specs re-render each run (CAP-3a).
#[allow(clippy::too_many_arguments)]
fn serve_agent<C: Completer, K: ToolCallCodec, T: PromptTemplate>(
    completer: &mut C,
    tools: &Tools,
    codec: K,
    template: T,
    system: String,
    opts: GenOpts,
    watch: DepthWatch,
    origins: &WebOrigins,
    req_rx: &Receiver<HostRequest>,
    event_tx: &UnboundedSender<HostEvent>,
    gate: &CancelGate,
) -> Result<()> {
    let mut agent = Agent::new(
        completer,
        tools,
        codec,
        template,
        system,
        knobs::AGENT_MAX_STEPS,
    )
    .with_opts(opts);

    while let Ok(req) = req_rx.recv() {
        // A vanished event plane means no frontend can ever see another
        // event: exit through the epilogue rather than serving the void.
        if event_tx.is_closed() {
            return Ok(());
        }
        match req {
            HostRequest::Submit { turn_id, text } => {
                let cancel = Cancel::new();
                gate.arm(turn_id, cancel.clone());
                let outcome = run_agent_turn(&mut agent, event_tx, turn_id, &text, &cancel, watch);
                gate.disarm();
                if let ControlFlow::Break(debt) = outcome {
                    // Fatal backend loss: converge on the epilogue, carrying
                    // any unproven-cleanup debt to the actor's verdict.
                    return debt.map_or(Ok(()), Err);
                }
                // Between turns, keep the next prompt under the depth budget
                // (HOST-5) — never mid-run.
                compact_after_turn(event_tx, watch, agent.last_prompt_tokens(), |budget| {
                    agent
                        .trim_history_to(budget, knobs::COMPACTION_KEEP_LAST)
                        .len()
                });
            }
            HostRequest::Cancel { turn_id } => gate.cancel(turn_id),
            HostRequest::Reset => agent.reset(),
            HostRequest::Grant { origin } => report_grant(event_tx, origins, &origin),
            HostRequest::Revoke { origin } => report_revoke(event_tx, Some(origins), &origin),
            HostRequest::ListGrants => report_grants(event_tx, Some(origins)),
            HostRequest::Shutdown => return Ok(()),
            _ => {} // a future request variant this host predates: ignore it.
        }
    }
    // Request disconnect: every sender is gone — a requested end, no debt.
    Ok(())
}

/// Grant an origin and report it (both the first-grant "web tools enabled"
/// tail and the plain subsequent form). CAP-3 wording; HOST-2.
fn report_grant(event_tx: &UnboundedSender<HostEvent>, origins: &WebOrigins, origin: &str) {
    let (list, message) = match origins.grant(origin) {
        Ok(true) => {
            let list = origins.list();
            let message = if list.len() == 1 {
                format!("granted read access to {origin} — web tools enabled")
            } else {
                format!("granted read access to {origin}")
            };
            (list, message)
        }
        Ok(false) => (origins.list(), format!("{origin} was already granted")),
        Err(e) => (origins.list(), format!("grant failed: {e}")),
    };
    let _ = event_tx.send(HostEvent::Grants {
        origins: list,
        message,
    });
}

/// Refuse a grant on a chat-only format, naming the format and the way out
/// (CAP-3 / CAPS-1; HOST-2).
fn refuse_grant(event_tx: &UnboundedSender<HostEvent>, format: ChatFormat, origin: &str) {
    let _ = event_tx.send(HostEvent::Grants {
        origins: vec![],
        message: format!(
            "cannot grant {origin}: the {format} format is chat-only \
             (tool calling needs a tool-capable format)"
        ),
    });
}

/// Answer a revoke request (both phases; HOST-2).
fn report_revoke(
    event_tx: &UnboundedSender<HostEvent>,
    origins: Option<&WebOrigins>,
    origin: &str,
) {
    let Some(origins) = origins else {
        let _ = event_tx.send(HostEvent::Grants {
            origins: vec![],
            message: "nothing granted (chat-only format)".to_string(),
        });
        return;
    };
    let message = match origins.revoke(origin) {
        Ok(true) => format!("revoked {origin}"),
        Ok(false) => format!("{origin} was not granted"),
        Err(e) => format!("revoke failed: {e}"),
    };
    let _ = event_tx.send(HostEvent::Grants {
        origins: origins.list(),
        message,
    });
}

/// Answer a list request (both phases; HOST-2).
fn report_grants(event_tx: &UnboundedSender<HostEvent>, origins: Option<&WebOrigins>) {
    let (list, message) = match origins {
        None => (vec![], "no web tools (chat-only format)".to_string()),
        Some(origins) => {
            let list = origins.list();
            let message = if list.is_empty() {
                "no origins granted — type a URL or /grant <origin>".to_string()
            } else {
                format!("granted: {}", list.join(", "))
            };
            (list, message)
        }
    };
    let _ = event_tx.send(HostEvent::Grants {
        origins: list,
        message,
    });
}

/// The web tools over a shared (growable) origin set. Present from the start on
/// tool-trained formats; hidden from the model while the set is empty (CAP-3a).
/// The plot tool rides along when a python-with-matplotlib is present (PLOT-1..3:
/// declarative specs only, output confined to `~/.cache/yatima/plots` — stable
/// and content-hash named so re-renders are idempotent) — and quietly doesn't
/// when it isn't; the model never sees a tool it cannot use.
fn web_tools(origins: &WebOrigins) -> Result<Tools> {
    // One listing cell per session (IMG-3): read_page publishes its numbered
    // [images] list into it, read_image selects from it by number.
    let listing = ImageListing::default();
    let mut tools = Tools::new().with(ReadUrl::new(origins.clone())?).with(
        ReadPage::with_limits(
            origins.clone(),
            knobs::READ_PAGE_MAX_INPUT_BYTES,
            knobs::READ_PAGE_MAX_CHARS,
        )?
        .with_listing(listing.clone()),
    );
    let cache = std::env::home_dir()
        .map(|home| home.join(".cache/yatima"))
        .unwrap_or_else(std::env::temp_dir);
    tools =
        tools.with(ReadImage::new(origins.clone(), cache.join("images"))?.with_listing(listing));
    match PlotSandbox::system(cache.join("plots")) {
        Ok(sandbox) => tools = tools.with(Plot::new(sandbox)),
        Err(e) => eprintln!("plot tool unavailable: {e}"),
    }
    Ok(tools)
}

/// Read back the image an artifact tool just announced (a plot render, a
/// fetched image), returning its bytes and filename. The path arrives on the
/// typed artifact event (IMG-2) — the tool emitted it, so it always points
/// inside the tool's own sandbox (PLOT-2 / IMG-1) and only ever names an
/// artifact the user has not seen this session. Format-agnostic — an SVG's
/// raw bytes pass through; a view that cannot show SVG rasterizes on receipt
/// (that stays a view concern).
fn read_artifact(path: &std::path::Path) -> Result<(Vec<u8>, String)> {
    let bytes = std::fs::read(path)?;
    let name = path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("artifact")
        .to_string();
    Ok((bytes, name))
}

/// Run one chat turn: stream `turn_streaming`'s raw fragments through the
/// session template's own [`ResponseClassifier`] (REASON-1: each emitted
/// [`HostEvent::Fragment`] is already classified — marker-split for marker
/// formats, the ATEM machine for Muse), report the prompt-token count for
/// the meter when the backend can count, then `Done`/`Error`.
/// `Break` = fatal backend loss: the caller exits through the epilogue,
/// carrying any unproven-cleanup debt to the actor's verdict.
fn run_turn<C: Completer>(
    session: &mut ChatSession<'_, C, Box<dyn PromptTemplate>>,
    event_tx: &UnboundedSender<HostEvent>,
    turn_id: TurnId,
    user: &str,
    cancel: &Cancel,
    watch: DepthWatch,
) -> ControlFlow<Option<anyhow::Error>> {
    let _ = event_tx.send(HostEvent::Started { turn_id });

    let mut classifier = session.classifier();

    let outcome = {
        let mut on_token = |frag: &str| {
            classifier.push(frag, |channel, text| {
                if let Some(channel) = to_proto_channel(channel) {
                    let _ = event_tx.send(HostEvent::Fragment {
                        turn_id,
                        channel,
                        text: text.to_string(),
                    });
                }
            });
        };
        session
            .turn_streaming_cancellable(user, cancel, &mut on_token)
            .map(|answer| answer.to_string())
    };
    // `on_token` is dropped at the block end, releasing the classifier so the
    // tail can be flushed.
    classifier.finish(|channel, text| {
        if let Some(channel) = to_proto_channel(channel) {
            let _ = event_tx.send(HostEvent::Fragment {
                turn_id,
                channel,
                text: text.to_string(),
            });
        }
    });

    match outcome {
        Ok(answer) => {
            // The streamed Fragment channels are the answer's authoritative
            // form; Done carries only why it stopped.
            if let Some(used) = session.last_prompt_tokens() {
                let _ = event_tx.send(HostEvent::Context {
                    prompt_tokens: used,
                });
                warn_kv_depth(event_tx, watch, used);
            }
            note_degenerate_answer(event_tx, &answer);
            let stop = session.last_stop().unwrap_or(StopReason::Eos);
            let _ = event_tx.send(HostEvent::Done {
                turn_id,
                stop: to_proto_stop(stop),
            });
            ControlFlow::Continue(())
        }
        Err(e) => report_turn_error(event_tx, turn_id, e),
    }
}

/// Report a failed turn: always an [`HostEvent::Error`] for the turn; when
/// the error carries the [`BackendLost`] marker (the managed child is dead),
/// also a [`HostEvent::Fatal`] and `Break`, so the serve loop exits through
/// the backend-consuming epilogue instead of retrying a dead backend. Any
/// other failure is recoverable: the session stands and the next submit is
/// served.
///
/// A lost backend whose error chain also carries the lib's typed
/// [`ChildCleanupFailed`] — the mid-completion death path consumed the
/// drain handles, so the later epilogue cannot re-prove the cleanup —
/// breaks with the error itself as **ownership debt**: the serve loop
/// returns it to the actor, whose final verdict fails
/// [`HostOwner::shutdown`] rather than blessing an unproven reap (LSRV-1 /
/// HOST-3).
fn report_turn_error(
    event_tx: &UnboundedSender<HostEvent>,
    turn_id: TurnId,
    error: anyhow::Error,
) -> ControlFlow<Option<anyhow::Error>> {
    let lost = error.downcast_ref::<BackendLost>().map(|l| l.0.clone());
    let _ = event_tx.send(HostEvent::Error {
        turn_id,
        message: format!("{error:#}"),
    });
    match lost {
        Some(message) => {
            let _ = event_tx.send(HostEvent::Fatal(message));
            let debt = error.downcast_ref::<ChildCleanupFailed>().is_some();
            ControlFlow::Break(debt.then_some(error))
        }
        None => ControlFlow::Continue(()),
    }
}

/// Run one agent turn, folding [`AgentEvent`]s onto the event plane. Each step's
/// decode **streams** (AGENT-4): classified fragments arrive live — reasoning on
/// [`Channel::Reasoning`], answer prose on [`Channel::Answer`], tool activity as
/// [`HostEvent::ToolNote`] — and the turn's `cancel` is token-level. A step that
/// turns out to be a tool call retracts its streamed narration from the answer
/// pane ([`HostEvent::RetractAnswer`]) and replays it as reasoning, ahead of
/// the [`ToolNoteKind::Call`] activity line. A successful plot/read_image
/// ships its bytes as [`HostEvent::Image`].
fn run_agent_turn<C: Completer, K: ToolCallCodec, T: PromptTemplate>(
    agent: &mut Agent<'_, C, K, T>,
    event_tx: &UnboundedSender<HostEvent>,
    turn_id: TurnId,
    user: &str,
    cancel: &Cancel,
    watch: DepthWatch,
) -> ControlFlow<Option<anyhow::Error>> {
    let _ = event_tx.send(HostEvent::Started { turn_id });

    let fragment = |channel: LibChannel, text: String| {
        if let Some(channel) = to_proto_channel(channel) {
            let _ = event_tx.send(HostEvent::Fragment {
                turn_id,
                channel,
                text,
            });
        }
    };
    let note = |kind: ToolNoteKind, text: String| {
        let _ = event_tx.send(HostEvent::ToolNote {
            turn_id,
            kind,
            text,
        });
    };

    // Answer prose streamed during the *current* step; a ToolCall event means it
    // was narration, not answer — retract and reclassify.
    let mut step_answer = String::new();

    let result = agent.run_with_cancellable(user, cancel, (), |(), event| {
        match event {
            AgentEvent::Fragment { channel, text } => {
                if channel == LibChannel::Answer {
                    step_answer.push_str(&text);
                }
                fragment(channel, text);
            }
            // The per-step aggregate; already streamed via Fragment (AGENT-4).
            AgentEvent::Reasoning(_) => {}
            AgentEvent::ToolCall(call) => {
                if !step_answer.is_empty() {
                    let narration = std::mem::take(&mut step_answer);
                    let _ = event_tx.send(HostEvent::RetractAnswer {
                        turn_id,
                        chars: narration.chars().count(),
                    });
                    fragment(LibChannel::Reasoning, format!("{}\n", narration.trim_end()));
                }
                note(
                    ToolNoteKind::Call,
                    format!("{} {}", call.name, clip(&call.args.to_string(), 160)),
                );
            }
            AgentEvent::ToolStarted(_) => {}
            AgentEvent::ToolProgress(message) => {
                note(ToolNoteKind::Progress, message);
            }
            AgentEvent::ToolOutcome(outcome) => {
                // Bare payloads under a semantic kind (HOST-4): clipping is
                // host policy; the marker each frontend draws is its own.
                let (kind, text) = match &outcome {
                    ToolOutcome::Success { content } => {
                        let flat = content.trim();
                        let text = if flat.chars().count() <= knobs::TOOL_NOTE_MAX_CHARS
                            && !flat.contains('\n')
                        {
                            flat.to_string()
                        } else {
                            format!("{} chars", content.chars().count())
                        };
                        (ToolNoteKind::Success, text)
                    }
                    other => (
                        ToolNoteKind::Failure,
                        clip(&other.render_for_model("").content, 160),
                    ),
                };
                note(kind, text);
                step_answer.clear();
            }
            AgentEvent::ToolArtifact(path) => {
                // IMG-2: the typed artifact event is the display license —
                // result prose is model-facing only. Every successful
                // read_image emits one (repeats included): the memo governs
                // narration, not pixels — a view may have reloaded since
                // the first showing.
                match read_artifact(&path) {
                    Ok((bytes, name)) => {
                        let _ = event_tx.send(HostEvent::Image {
                            turn_id,
                            bytes,
                            name,
                        });
                    }
                    Err(e) => note(ToolNoteKind::Failure, format!("artifact: {e}")),
                }
            }
            // Already streamed fragment-by-fragment; Done carries the stop.
            AgentEvent::Final(_) => {}
        }
        Ok(ControlFlow::Continue(()))
    });

    match result {
        Ok(((), run)) => {
            // The run's deepest step prompt feeds the context meter (the
            // agent path reported nothing before — precisely the mode that
            // ran off the Metal cliff unmetered) and the CTX-2 warning.
            if let Some(used) = agent.last_prompt_tokens() {
                let _ = event_tx.send(HostEvent::Context {
                    prompt_tokens: used,
                });
                warn_kv_depth(event_tx, watch, used);
            }
            if run.stop == AgentStop::Final {
                note_degenerate_answer(event_tx, &run.answer);
            }
            let stop = match run.stop {
                AgentStop::Final => StopReason::Eos,
                AgentStop::Stopped => StopReason::Stopped,
                AgentStop::MaxSteps => {
                    note(
                        ToolNoteKind::Warning,
                        format!("tool-step budget exhausted ({})", knobs::AGENT_MAX_STEPS),
                    );
                    StopReason::MaxTokens
                }
                AgentStop::NoAnswer => {
                    note(
                        ToolNoteKind::Warning,
                        "the model produced no answer text (reasoning-only reply; \
                         nothing committed)"
                            .to_string(),
                    );
                    StopReason::Stopped
                }
            };
            let _ = event_tx.send(HostEvent::Done {
                turn_id,
                stop: to_proto_stop(stop),
            });
            ControlFlow::Continue(())
        }
        Err(e) => report_turn_error(event_tx, turn_id, e),
    }
}

/// Convert a yatima-lib channel to its wire mirror. A free function, not a
/// `From` impl: both types are foreign to this crate, so the orphan rule forbids
/// the trait impl here (and yatima-protocol may not depend on the lib).
///
/// `ToolCall` has no wire mirror and never panics: in the agent, tool
/// material becomes typed [`AgentEvent`] activity (surfaced as
/// [`HostEvent::ToolNote`]); in a *chat* stream — where a Muse reply may
/// still address a tool nobody advertised — the fragment is protocol
/// material, not prose, and is deliberately consumed before the wire
/// (REASON-1: framing never surfaces as reasoning or answer text).
fn to_proto_channel(channel: LibChannel) -> Option<Channel> {
    match channel {
        LibChannel::Reasoning => Some(Channel::Reasoning),
        LibChannel::Answer => Some(Channel::Answer),
        LibChannel::ToolCall => None,
    }
}

/// Convert a yatima-lib stop reason to its wire mirror (see [`to_proto_channel`]
/// on why this is a free function).
fn to_proto_stop(stop: StopReason) -> StopKind {
    match stop {
        StopReason::Eos => StopKind::Eos,
        StopReason::MaxTokens => StopKind::MaxTokens,
        StopReason::Stopped => StopKind::Stopped,
        StopReason::Repetition => StopKind::Repetition,
    }
}

/// Truncate a note payload to `max` characters (with an ellipsis) — activity
/// lines summarize; the model, not the pane, consumes full payloads.
fn clip(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        s.to_string()
    } else {
        let cut: String = s.chars().take(max).collect();
        format!("{cut}…")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lib_types_map_to_wire_mirrors() {
        // The two free conversions cover every variant (a new one that isn't
        // handled fails to compile — the matches are exhaustive). ToolCall
        // deliberately has no wire mirror and no panic: tool material is
        // consumed or projected as typed activity before conversion
        // (REASON-1; the retired `unreachable!` arm was a stage-5 landmine).
        assert_eq!(to_proto_channel(LibChannel::Answer), Some(Channel::Answer));
        assert_eq!(
            to_proto_channel(LibChannel::Reasoning),
            Some(Channel::Reasoning)
        );
        assert_eq!(to_proto_channel(LibChannel::ToolCall), None);
        assert_eq!(to_proto_stop(StopReason::Eos), StopKind::Eos);
        assert_eq!(to_proto_stop(StopReason::MaxTokens), StopKind::MaxTokens);
        assert_eq!(to_proto_stop(StopReason::Stopped), StopKind::Stopped);
        assert_eq!(to_proto_stop(StopReason::Repetition), StopKind::Repetition);
    }

    #[test]
    fn probe_verdict_treats_unobservable_children_as_lost() {
        // upholds: LSRV-1 (host boundary) — an exited child is loss, a
        // running child is not, and a probe FAILURE is also loss: an owner
        // that cannot observe its child holds no evidence of a healthy
        // backend and must not keep retrying it.
        use std::os::unix::process::ExitStatusExt;
        let exited = std::process::ExitStatus::from_raw(9);
        assert!(probe_verdict(Ok(Some(exited)))
            .expect("exited is loss")
            .contains("exited"));
        assert_eq!(probe_verdict(Ok(None)), None, "running is recoverable");
        let failure = std::io::Error::other("no such process table");
        assert!(probe_verdict(Err(failure))
            .expect("unobservable is loss")
            .contains("cannot observe"));
    }

    #[tokio::test]
    async fn await_epilogue_propagates_the_epilogue_verdict_after_join() {
        // upholds: HOST-3 / LSRV-1 — the joined-success witness must not
        // lie: a failed epilogue fails the wait even though the join
        // succeeded, and a clean epilogue passes.
        let (tx, rx) = tokio::sync::oneshot::channel();
        let thread = std::thread::spawn(move || {
            let _ = tx.send(Err(anyhow::anyhow!("drain join failed")));
        });
        let error = await_epilogue(rx, thread, std::time::Duration::from_secs(5))
            .await
            .expect_err("a failed epilogue must fail the wait");
        assert!(format!("{error:#}").contains("drain join failed"));

        let (tx, rx) = tokio::sync::oneshot::channel();
        let thread = std::thread::spawn(move || {
            let _ = tx.send(Ok(()));
        });
        await_epilogue(rx, thread, std::time::Duration::from_secs(5))
            .await
            .expect("a clean epilogue passes");
    }

    #[test]
    fn reaper_thread_joins_the_actor_without_any_runtime() {
        // upholds: HOST-3 — the reaper is an OS thread, not a tokio task: a
        // frontend runtime's teardown cannot abort it. Proven by running the
        // whole transfer with NO runtime anywhere — the reaper's own join
        // completing is the witness that the late actor was joined, not
        // detached.
        let (tx, rx) = tokio::sync::oneshot::channel();
        let actor = std::thread::spawn(move || {
            std::thread::sleep(std::time::Duration::from_millis(200));
            let _ = tx.send(Ok(()));
        });
        let reaper = spawn_reaper(rx, actor);
        reaper
            .join()
            .expect("the reaper joined the late actor and exited cleanly");
    }

    #[tokio::test]
    async fn await_epilogue_timeout_transfers_the_join_to_the_reaper() {
        // upholds: HOST-3 — when the bound elapses the error names the
        // transfer; the reaper mechanism itself is witnessed runtime-free
        // above. The runtime this test creates then dies immediately — with
        // an OS-thread reaper that is safe, which is the blocker this
        // replaces (a tokio-spawned reaper died with it).
        let (tx, rx) = tokio::sync::oneshot::channel();
        let thread = std::thread::spawn(move || {
            std::thread::sleep(std::time::Duration::from_millis(200));
            let _ = tx.send(Ok(()));
        });
        let error = await_epilogue(rx, thread, std::time::Duration::from_millis(50))
            .await
            .expect_err("the bound elapsed");
        assert!(format!("{error:#}").contains("reaper"), "{error:#}");
    }

    #[test]
    fn managed_config_is_valid_by_construction() {
        // upholds: PROFILE-1 / PROFILE-2 (host boundary) — only a profile
        // pinning the llama-server backend and a chat format can build a
        // managed config; its options are the profile's recipe layered over
        // the base and its format is the pin — a mismatched format or an
        // undercut recipe cannot be assembled field-by-field.
        let profile = yatima_lib::ModelProfile::builtin("muse-glimmer").unwrap();
        let config = HostConfig::managed(&profile, true, GenOpts::default(), None).unwrap();
        assert_eq!(config.format, Some(ChatFormat::MuseGlimmer));
        assert_eq!(config.model_label.as_deref(), Some("muse-glimmer"));
        let layered = profile.apply_gen_overrides(GenOpts::default());
        assert_eq!(config.opts.max_tokens, layered.max_tokens);
        assert_eq!(config.opts.sampling, layered.sampling);
        assert!(config.managed_launcher.is_none(), "never set by default");

        let engine_profile = yatima_lib::ModelProfile::builtin("kimi-dev").unwrap();
        let Err(error) = HostConfig::managed(&engine_profile, true, GenOpts::default(), None)
        else {
            panic!("an engine profile must not build a managed config");
        };
        assert!(
            error
                .to_string()
                .contains("does not pin the llama-server backend"),
            "{error:#}"
        );

        let mut formatless = profile.clone();
        formatless.format = None;
        let Err(error) = HostConfig::managed(&formatless, true, GenOpts::default(), None) else {
            panic!("a format-less profile must not build a managed config");
        };
        assert!(
            error.to_string().contains("does not pin a chat format"),
            "{error:#}"
        );
    }

    #[test]
    fn cancel_armed_flips_only_the_armed_turn() {
        // The owner's shutdown path: whatever turn is armed cancels without
        // the owner knowing its id; a disarmed gate is a no-op, and early
        // cancels for queued turns are untouched.
        let gate = CancelGate::new();
        gate.cancel_armed(); // disarmed: nothing to flip, nothing to panic.
        let cancel = Cancel::new();
        gate.arm(3, cancel.clone());
        gate.cancel_armed();
        assert!(cancel.is_cancelled());
        gate.disarm();
        let next = Cancel::new();
        gate.arm(4, next.clone());
        assert!(!next.is_cancelled(), "a later turn must not inherit it");
    }

    #[test]
    fn grant_wording_is_single_sourced() {
        // upholds: HOST-2 — the CAP-3 grant wording lives only here; the first
        // grant carries the "web tools enabled" tail, later grants do not.
        let (tx, mut rx) = unbounded_channel();
        let origins = WebOrigins::new();
        report_grant(&tx, &origins, "https://example.com");
        let HostEvent::Grants {
            message,
            origins: list,
        } = rx.try_recv().unwrap()
        else {
            panic!("expected a Grants event");
        };
        assert_eq!(
            message,
            "granted read access to https://example.com — web tools enabled"
        );
        assert_eq!(list, ["https://example.com"]);

        report_grant(&tx, &origins, "https://other.example");
        let HostEvent::Grants { message, .. } = rx.try_recv().unwrap() else {
            panic!("expected a Grants event");
        };
        assert_eq!(message, "granted read access to https://other.example");

        report_grant(&tx, &origins, "https://example.com");
        let HostEvent::Grants { message, .. } = rx.try_recv().unwrap() else {
            panic!("expected a Grants event");
        };
        assert_eq!(message, "https://example.com was already granted");
    }

    #[test]
    fn chat_only_reports_name_no_authority() {
        // upholds: HOST-2 — the chat-only grant/revoke/list reports are single
        // sourced here, and none claims web authority a chat format cannot hold.
        let (tx, mut rx) = unbounded_channel();
        report_grants(&tx, None);
        let HostEvent::Grants { message, .. } = rx.try_recv().unwrap() else {
            panic!("expected a Grants event");
        };
        assert_eq!(message, "no web tools (chat-only format)");

        report_revoke(&tx, None, "https://x.example");
        let HostEvent::Grants { message, .. } = rx.try_recv().unwrap() else {
            panic!("expected a Grants event");
        };
        assert_eq!(message, "nothing granted (chat-only format)");
    }

    #[test]
    fn artifact_read_takes_the_event_path() {
        // upholds: IMG-2 — the display path starts from the typed artifact
        // event's path, never from parsing result prose: a missing file
        // errors; a real file yields its bytes and bare filename (the wire's
        // Image.name).
        assert!(read_artifact(std::path::Path::new("/nonexistent/x.png")).is_err());

        let path = std::env::temp_dir().join("yatima-host-artifact-test.png");
        std::fs::write(&path, b"PNGDATA").unwrap();
        let (bytes, name) = read_artifact(&path).unwrap();
        assert_eq!(bytes, b"PNGDATA");
        assert_eq!(name, "yatima-host-artifact-test.png");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn cancel_gate_flips_only_the_armed_turn() {
        // The armed turn cancels; a cancel for a different turn never touches
        // the in-flight one (it is remembered for that turn, not applied here).
        let gate = CancelGate::new();
        let cancel = Cancel::new();
        gate.arm(5, cancel.clone());
        gate.cancel(6); // a different (queued) turn: must not touch turn 5
        assert!(!cancel.is_cancelled());
        gate.cancel(5);
        assert!(cancel.is_cancelled());
    }

    #[test]
    fn cancel_gate_remembers_a_cancel_that_beats_its_turn() {
        // A Submit{n}/Cancel{n} for a turn still queued behind a running one:
        // the cancel arrives before the turn arms and must apply the instant it
        // does, so the queued turn a user stopped never runs.
        let gate = CancelGate::new();
        let running = Cancel::new();
        gate.arm(7, running.clone());
        gate.cancel(8); // turn 8 not armed yet: remembered
        assert!(!running.is_cancelled(), "cancel for 8 must not touch 7");
        gate.disarm();
        let queued = Cancel::new();
        gate.arm(8, queued.clone());
        assert!(queued.is_cancelled(), "early cancel must fire when 8 arms");
    }

    #[test]
    fn cancel_gate_prunes_spent_early_cancels() {
        // A cancel for a turn that never arms is pruned by a later arm
        // (monotonic ids), so it can never leak onto a newer turn.
        let gate = CancelGate::new();
        gate.cancel(1); // never submitted; remembered
        let later = Cancel::new();
        gate.arm(2, later.clone()); // arming 2 prunes ids <= 2, incl. stale 1
        assert!(
            !later.is_cancelled(),
            "turn 2 must not inherit a stale cancel"
        );
    }

    #[test]
    fn clip_is_char_safe() {
        assert_eq!(clip("hello", 10), "hello");
        assert_eq!(clip("hello", 3), "hel…");
    }

    #[test]
    fn kv_depth_warning_reaches_the_user_only_when_unreliable() {
        // upholds: CTX-2 (surfaced) — the unreliable depth warns on the
        // always-visible Note plane; the mitigated band and CPU runs stay
        // quiet (the engine's debug log covers them).
        let watch = |metal| DepthWatch {
            metal,
            max_tokens: 1024,
            context_length: None,
        };
        let (tx, mut rx) = unbounded_channel();
        warn_kv_depth(&tx, watch(true), 16_000);
        let Ok(HostEvent::Note(message)) = rx.try_recv() else {
            panic!("expected a Note past the validated depth");
        };
        assert!(message.contains("~16000 tokens"), "{message}");
        assert!(message.contains("may degenerate"), "{message}");
        assert!(message.contains("/reset"), "{message}");

        warn_kv_depth(&tx, watch(true), 9_000); // mitigated band: quiet
        warn_kv_depth(&tx, watch(true), 2_000); // shallow: quiet
        warn_kv_depth(&tx, watch(false), 16_000); // cpu: quiet
        assert!(rx.try_recv().is_err(), "no other depth may warn");
    }

    #[test]
    fn compaction_budget_reserves_reply_and_tool_headroom() {
        // upholds: HOST-5 — the low-water mark is the depth ceiling less the
        // reply budget and one run's within-run tool growth; on Metal the
        // ceiling is tightened to the validated KV depth, off Metal it is the
        // model's declared window (or none, so nothing is trimmed).
        let metal = |ctx| DepthWatch {
            metal: true,
            max_tokens: 1024,
            context_length: ctx,
        };
        let cpu = |ctx| DepthWatch {
            metal: false,
            max_tokens: 1024,
            context_length: ctx,
        };
        let headroom = knobs::TOOL_HEADROOM;
        // Metal, no declared window: the validated depth is the ceiling.
        assert_eq!(
            compaction_low_water(metal(None)),
            Some(METAL_KV_VALIDATED - 1024 - headroom)
        );
        // A larger declared window is still capped at the validated depth…
        assert_eq!(
            compaction_low_water(metal(Some(128_000))),
            Some(METAL_KV_VALIDATED - 1024 - headroom)
        );
        // …a smaller one binds instead.
        assert_eq!(
            compaction_low_water(metal(Some(8_000))),
            Some(8_000 - 1024 - headroom)
        );
        // Off Metal the declared window is the ceiling; none means no trimming.
        assert_eq!(
            compaction_low_water(cpu(Some(32_000))),
            Some(32_000 - 1024 - headroom)
        );
        assert_eq!(compaction_low_water(cpu(None)), None);
    }

    #[test]
    fn compaction_note_is_single_sourced_and_names_the_depth() {
        // upholds: HOST-5 — the compaction wording lives only here, names the
        // depth budget, pluralizes, and points at /reset (like the grant
        // wording, HOST-2).
        let watch = DepthWatch {
            metal: true,
            max_tokens: 1024,
            context_length: None,
        };
        let one = compaction_note(1, watch);
        assert!(one.contains("dropped the 1 oldest exchange "), "{one}");
        assert!(
            one.contains(&format!("~{METAL_KV_VALIDATED} tokens")),
            "{one}"
        );
        assert!(one.contains("/reset"), "{one}");
        let many = compaction_note(3, watch);
        assert!(many.contains("dropped the 3 oldest exchanges"), "{many}");
    }

    #[test]
    fn compaction_only_notes_when_history_is_actually_dropped() {
        // upholds: HOST-5 — compaction is retrospective and always visible: it
        // fires only when the run went past the low-water mark AND trimming
        // dropped committed exchanges. A run at/under the mark never even
        // attempts a trim; a deep run whose depth was all within-run tool
        // growth (history already fits) drops nothing and stays silent.
        let watch = DepthWatch {
            metal: true,
            max_tokens: 1024,
            context_length: None,
        };
        let low_water = compaction_low_water(watch).unwrap();

        // At/under the mark: trimming is not even attempted, no note.
        let (tx, mut rx) = unbounded_channel();
        let mut attempted = false;
        compact_after_turn(&tx, watch, Some(low_water), |_| {
            attempted = true;
            0
        });
        assert!(!attempted, "at/under the mark, no trim is attempted");
        assert!(rx.try_recv().is_err());

        // Past the mark but nothing droppable (history already fits): silent,
        // and the trim was asked for exactly the low-water budget.
        let (tx, mut rx) = unbounded_channel();
        compact_after_turn(&tx, watch, Some(low_water + 1), |budget| {
            assert_eq!(budget, low_water);
            0
        });
        assert!(rx.try_recv().is_err(), "no exchanges dropped → no note");

        // Past the mark and two turns dropped: one visible note, one exchange.
        let (tx, mut rx) = unbounded_channel();
        compact_after_turn(&tx, watch, Some(low_water + 1), |_| 2);
        let Ok(HostEvent::Note(msg)) = rx.try_recv() else {
            panic!("expected a compaction note");
        };
        assert!(msg.contains("dropped the 1 oldest exchange"), "{msg}");
    }

    #[test]
    fn notes_carry_kind_not_typography() {
        // upholds: HOST-4 — the wire carries meaning, never typography: no
        // marker glyph appears anywhere in this crate's source (the escapes
        // below keep the scan from tripping on this test itself). The
        // frontends own the vocabulary a ToolNoteKind renders under.
        let src = concat!(
            include_str!("lib.rs"),
            include_str!("knobs.rs"),
            include_str!("logging.rs")
        );
        for glyph in ['\u{2713}', '\u{2717}', '\u{2699}', '\u{26a0}'] {
            assert!(
                !src.contains(glyph),
                "the host emits marker glyph {glyph}; markers are view policy"
            );
        }
    }
}
