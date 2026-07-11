//! The engine-facing host: one implementation every yatima frontend shares.
//!
//! Local decode is `!Send` and runs on the runtime's blocking island (CMP-1 /
//! RT-2), so it cannot live in a `tokio::spawn`. A dedicated **OS thread** owns
//! the [`Engine`] *and* the [`ChatSession`]/[`Agent`] — the one authoritative
//! prompt history — for its whole life (HOST-3), and, being a plain thread (not
//! a runtime worker), calls the public **sync** decode shims directly; RT-1 is
//! not violated. The TUI, GUI, and coming yatima-serve are thin views over this
//! host; they differ only in how they draw a [`HostEvent`] and where a
//! [`HostRequest`] comes from.
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
//! - **HOST-3** one engine thread owns the `!Send` engine and session for the
//!   whole run; the `!Send` types are created inside the thread and never
//!   cross a thread boundary.
//! - **HOST-4** tool activity crosses the wire as `(kind, payload)` —
//!   [`ToolNoteKind`] carries the semantics, and this crate emits no marker
//!   glyphs or note indentation; the vocabulary a note renders under is view
//!   policy (cited by `notes_carry_kind_not_typography`).
//! - **HOST-5** the host keeps every rendered prompt under the depth budget:
//!   between turns it trims the committed history (COMPACT-1) back under a
//!   low-water mark ([`compaction_low_water`] = the depth ceiling less the
//!   reply and one run's within-run tool growth), and compaction is always
//!   visible — history is never edited silently. The ceiling tightens to the
//!   Metal KV validated depth on a Metal run (CTX-2). Wording single-sourced
//!   in [`compaction_note`]; cited by the arithmetic/wording/trigger tests.

use std::collections::BTreeSet;
use std::ops::ControlFlow;
use std::path::PathBuf;
use std::sync::mpsc::{Receiver, Sender};
use std::sync::{Arc, Mutex};

use anyhow::Result;
use tokio::sync::mpsc::{unbounded_channel, UnboundedReceiver, UnboundedSender};
use yatima_lib::{
    device, looks_degenerate, metal_kv_depth_risk, resolve_format, Agent, AgentEvent, AgentStop,
    Cancel, Channel as LibChannel, ChatFormat, ChatMlTemplate, ChatSession, Engine, GenOpts,
    ImageListing, JsonToolCall, KvDepthRisk, PlainTemplate, Plot, PlotSandbox, PromptTemplate,
    QwenToolCall, ReadImage, ReadPage, ReadUrl, ReasoningSplitter, Sampling, StopReason,
    ToolCallCodec, ToolOutcome, Tools, WebOrigins, METAL_KV_VALIDATED,
};

pub mod knobs;
mod logging;

pub use logging::init_file_logging;
pub use yatima_protocol::{Channel, HostEvent, HostRequest, ModelInfo, StopKind, ToolNoteKind};

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

/// What the host needs to load a model (all `Send`, so it crosses into the
/// thread; the `!Send` [`Engine`] is then *created* inside the thread — HOST-3).
pub struct HostConfig {
    pub dir: PathBuf,
    pub cpu: bool,
    pub opts: GenOpts,
    pub format: Option<ChatFormat>,
    pub system: Option<String>,
    pub model_label: String,
}

/// The frontend-side handle to a running host.
pub struct HostHandle {
    pub req_tx: Sender<HostRequest>,
    pub event_rx: UnboundedReceiver<HostEvent>,
    pub cancel: CancelGate,
}

/// Launch the host thread and return its handle at once. The thread loads the
/// model and sends [`HostEvent::Ready`] (or [`HostEvent::Fatal`]) as its first
/// event; nothing here waits for it. This is the shape a GUI wants (it renders
/// a loading state and drains events on its own clock).
pub fn spawn_nonblocking(config: HostConfig) -> Result<HostHandle> {
    let (req_tx, req_rx) = std::sync::mpsc::channel::<HostRequest>();
    let (event_tx, event_rx) = unbounded_channel::<HostEvent>();
    let gate = CancelGate::new();
    let actor_gate = gate.clone();
    std::thread::Builder::new()
        .name("yatima-engine".into())
        .spawn(move || actor_main(config, req_rx, event_tx, actor_gate))?;
    Ok(HostHandle {
        req_tx,
        event_rx,
        cancel: gate,
    })
}

/// Launch the host and wait for the model to load, returning the handle and
/// what's running — or an error if the load failed (surfaced *before* the
/// caller touches its screen; the TUI prints it as a plain stderr line). The
/// blocking shape a terminal frontend wants; the first event is consumed here.
pub async fn spawn(config: HostConfig) -> Result<(HostHandle, ModelInfo)> {
    let mut handle = spawn_nonblocking(config)?;
    match handle.event_rx.recv().await {
        Some(HostEvent::Ready(info)) => Ok((handle, info)),
        Some(HostEvent::Fatal(message)) => Err(anyhow::anyhow!(message)),
        _ => Err(anyhow::anyhow!(
            "engine thread exited before reporting readiness"
        )),
    }
}

/// The actor's body: load, report readiness, then serve requests until
/// shutdown. Owns the engine and the session/agent for the whole run (HOST-3).
fn actor_main(
    config: HostConfig,
    req_rx: Receiver<HostRequest>,
    event_tx: UnboundedSender<HostEvent>,
    gate: CancelGate,
) {
    let mut engine = match load_engine(&config) {
        Ok(engine) => engine,
        Err(e) => {
            let _ = event_tx.send(HostEvent::Fatal(e.to_string()));
            return;
        }
    };
    let (format, _mismatch) = resolve_format(engine.arch(), config.format);
    let info = build_model_info(&engine, &config, format);

    // Tool-trained formats always carry the web tools, initially with an empty
    // origin set — hidden from the model (CAP-3a) and inert until a grant
    // arrives (sandbox by omission; CAP-3: grants come only from the user, via
    // the frontend). Chat-only formats get none.
    let tool_trained = matches!(format, ChatFormat::Qwen | ChatFormat::Plain);
    let origins = WebOrigins::new();
    // Client construction cannot practically fail; degrade to empty tools (the
    // model simply never sees web tools) rather than dying.
    let tools = tool_trained.then(|| web_tools(&origins).unwrap_or_default());

    if event_tx.send(HostEvent::Ready(info)).is_err() {
        return; // the frontend gave up during load.
    }

    // What the CTX-2 surface needs per turn: whether decode runs on Metal
    // (mirrors ModelInfo's device judgment) and the per-turn budget the risk
    // bound adds to the prompt depth.
    let watch = DepthWatch {
        metal: !config.cpu,
        max_tokens: config.opts.max_tokens,
        context_length: engine.context_length(),
    };

    // Tool-trained formats serve the sessionful agent from turn one: the web
    // tools hide themselves while the origin set is empty (CAP-3a), so pre-grant
    // the model sees exactly the no-authority tools (plot), and a grant simply
    // surfaces the web tools mid-session — /grant mints authority, it is not a
    // mode switch. Chat-only formats stay on the plain chat path forever.
    let Some(tools) = tools else {
        serve_chat(
            &mut engine,
            format,
            config.system.clone(),
            config.opts.clone(),
            watch,
            &req_rx,
            &event_tx,
            &gate,
        );
        return;
    };
    let system = config
        .system
        .unwrap_or_else(|| knobs::DEFAULT_AGENT_SYSTEM.to_string());
    match format {
        ChatFormat::Qwen => serve_agent(
            &mut engine,
            &tools,
            QwenToolCall,
            ChatMlTemplate,
            system,
            config.opts,
            watch,
            &origins,
            &req_rx,
            &event_tx,
            &gate,
        ),
        ChatFormat::Plain => serve_agent(
            &mut engine,
            &tools,
            JsonToolCall,
            PlainTemplate,
            system,
            config.opts,
            watch,
            &origins,
            &req_rx,
            &event_tx,
            &gate,
        ),
        _ => unreachable!("the agent path is only taken on tool-trained formats"),
    }
}

/// Snapshot what's running for the status rail — every field a pre-formatted
/// string so the frontend is a pure view (built here, where the engine and
/// config live).
fn build_model_info(engine: &Engine, config: &HostConfig, format: ChatFormat) -> ModelInfo {
    ModelInfo {
        label: config.model_label.clone(),
        arch: format!("{:?}", engine.arch()),
        backend: engine.backend(),
        device: if config.cpu { "cpu" } else { "gpu" }.to_string(),
        format: format!("{format:?}"),
        sampling: sampling_summary(config.opts.sampling),
        max_tokens: config.opts.max_tokens,
        context_length: engine.context_length(),
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

/// The chat serve loop for chat-only formats: the plain streaming
/// [`ChatSession`]. Grants are always refused here (CAPS-1 — a chat-only format
/// cannot enter the tool path); tool-trained formats never enter this loop.
#[allow(clippy::too_many_arguments)]
fn serve_chat(
    engine: &mut Engine,
    format: ChatFormat,
    system: Option<String>,
    opts: GenOpts,
    watch: DepthWatch,
    req_rx: &Receiver<HostRequest>,
    event_tx: &UnboundedSender<HostEvent>,
    gate: &CancelGate,
) {
    let template = format.template();
    let mut session = ChatSession::new(engine, template).with_opts(opts);
    if let Some(system) = system {
        session = session.with_system(system);
    }

    while let Ok(req) = req_rx.recv() {
        match req {
            HostRequest::Submit { turn_id, text } => {
                let cancel = Cancel::new();
                gate.arm(turn_id, cancel.clone());
                run_turn(
                    &mut session,
                    event_tx,
                    format,
                    turn_id,
                    &text,
                    &cancel,
                    watch,
                );
                gate.disarm();
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
            HostRequest::Shutdown => return,
            _ => {} // a future request variant this host predates: ignore it.
        }
    }
}

/// The agent serve loop: one sessionful [`Agent`] (AGENT-3) serves every turn,
/// seeded with the chat phase's history. Later grants/revokes mutate the shared
/// origin set in place — the specs re-render each run (CAP-3a).
#[allow(clippy::too_many_arguments)]
fn serve_agent<K: ToolCallCodec, T: PromptTemplate>(
    engine: &mut Engine,
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
) {
    let mut agent = Agent::new(
        engine,
        tools,
        codec,
        template,
        system,
        knobs::AGENT_MAX_STEPS,
    )
    .with_opts(opts);

    while let Ok(req) = req_rx.recv() {
        match req {
            HostRequest::Submit { turn_id, text } => {
                let cancel = Cancel::new();
                gate.arm(turn_id, cancel.clone());
                run_agent_turn(&mut agent, event_tx, turn_id, &text, &cancel, watch);
                gate.disarm();
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
            HostRequest::Shutdown => return,
            _ => {} // a future request variant this host predates: ignore it.
        }
    }
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
             (tool calling needs qwen or plain)"
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

fn load_engine(config: &HostConfig) -> Result<Engine> {
    let dev = device(config.cpu)?;
    Engine::load(&config.dir, dev)
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

/// Run one chat turn: stream `turn_streaming`'s raw fragments through a
/// [`ReasoningSplitter`] (so each emitted [`HostEvent::Fragment`] is already
/// classified), report the prompt-token count for the meter, then `Done`/`Error`.
#[allow(clippy::too_many_arguments)]
fn run_turn(
    session: &mut ChatSession<'_, Engine, Box<dyn PromptTemplate>>,
    event_tx: &UnboundedSender<HostEvent>,
    format: ChatFormat,
    turn_id: TurnId,
    user: &str,
    cancel: &Cancel,
    watch: DepthWatch,
) {
    let _ = event_tx.send(HostEvent::Started { turn_id });

    let mut splitter = if format.pre_seeds_reasoning() {
        ReasoningSplitter::seeded()
    } else {
        ReasoningSplitter::new()
    };

    let outcome = {
        let mut on_token = |frag: &str| {
            splitter.push(frag, |channel, text| {
                let _ = event_tx.send(HostEvent::Fragment {
                    turn_id,
                    channel: to_proto_channel(channel),
                    text: text.to_string(),
                });
            });
        };
        session
            .turn_streaming_cancellable(user, cancel, &mut on_token)
            .map(|answer| answer.to_string())
    };
    // `on_token` is dropped at the block end, releasing `splitter` so the tail
    // can be flushed.
    splitter.finish(|channel, text| {
        let _ = event_tx.send(HostEvent::Fragment {
            turn_id,
            channel: to_proto_channel(channel),
            text: text.to_string(),
        });
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
        }
        Err(e) => {
            let _ = event_tx.send(HostEvent::Error {
                turn_id,
                message: e.to_string(),
            });
        }
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
fn run_agent_turn<K: ToolCallCodec, T: PromptTemplate>(
    agent: &mut Agent<'_, Engine, K, T>,
    event_tx: &UnboundedSender<HostEvent>,
    turn_id: TurnId,
    user: &str,
    cancel: &Cancel,
    watch: DepthWatch,
) {
    let _ = event_tx.send(HostEvent::Started { turn_id });

    let fragment = |channel: LibChannel, text: String| {
        let _ = event_tx.send(HostEvent::Fragment {
            turn_id,
            channel: to_proto_channel(channel),
            text,
        });
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
                // result prose is model-facing only, so a memo-served repeat
                // (which mentions the file but emits no event) never
                // re-shows the image.
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
            };
            let _ = event_tx.send(HostEvent::Done {
                turn_id,
                stop: to_proto_stop(stop),
            });
        }
        Err(e) => {
            let _ = event_tx.send(HostEvent::Error {
                turn_id,
                message: e.to_string(),
            });
        }
    }
}

/// Convert a yatima-lib channel to its wire mirror. A free function, not a
/// `From` impl: both types are foreign to this crate, so the orphan rule forbids
/// the trait impl here (and yatima-protocol may not depend on the lib).
fn to_proto_channel(channel: LibChannel) -> Channel {
    match channel {
        LibChannel::Reasoning => Channel::Reasoning,
        LibChannel::Answer => Channel::Answer,
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
        // handled fails to compile — the matches are exhaustive).
        assert_eq!(to_proto_channel(LibChannel::Answer), Channel::Answer);
        assert_eq!(to_proto_channel(LibChannel::Reasoning), Channel::Reasoning);
        assert_eq!(to_proto_stop(StopReason::Eos), StopKind::Eos);
        assert_eq!(to_proto_stop(StopReason::MaxTokens), StopKind::MaxTokens);
        assert_eq!(to_proto_stop(StopReason::Stopped), StopKind::Stopped);
        assert_eq!(to_proto_stop(StopReason::Repetition), StopKind::Repetition);
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
