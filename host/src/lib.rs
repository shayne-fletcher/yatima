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

use std::ops::ControlFlow;
use std::path::PathBuf;
use std::sync::mpsc::{Receiver, Sender};
use std::sync::{Arc, Mutex};

use anyhow::Result;
use tokio::sync::mpsc::{unbounded_channel, UnboundedReceiver, UnboundedSender};
use yatima_lib::{
    device, resolve_format, Agent, AgentEvent, AgentStop, Cancel, Channel as LibChannel,
    ChatFormat, ChatMlTemplate, ChatSession, Engine, GenOpts, JsonToolCall, PlainTemplate, Plot,
    PlotSandbox, PromptTemplate, QwenToolCall, ReadImage, ReadPage, ReadUrl, ReasoningSplitter,
    Sampling, StopReason, ToolCallCodec, ToolOutcome, Tools, WebOrigins,
};

pub mod knobs;
mod logging;

pub use logging::init_file_logging;
pub use yatima_protocol::{Channel, HostEvent, HostRequest, ModelInfo, StopKind};

/// A turn identifier, monotonic per session. Lets a frontend ignore stale events.
pub type TurnId = u64;

/// The out-of-band cancel handle: the actor arms it with the in-flight turn's
/// [`Cancel`] before decoding; a frontend flips it mid-decode. Cloneable and
/// cheap (an `Arc`); [`spawn`] hands one to the frontend and keeps one for the
/// actor.
#[derive(Clone, Default)]
pub struct CancelGate(Arc<Mutex<Option<(TurnId, Cancel)>>>);

impl CancelGate {
    /// A fresh, disarmed gate.
    pub fn new() -> CancelGate {
        CancelGate::default()
    }

    /// Arm the gate with the turn about to decode (the host's job, per turn).
    pub fn arm(&self, turn_id: TurnId, cancel: Cancel) {
        if let Ok(mut slot) = self.0.lock() {
            *slot = Some((turn_id, cancel));
        }
    }

    /// Disarm after a turn finishes (a stale `cancel(turn_id)` then no-ops).
    pub fn disarm(&self) {
        if let Ok(mut slot) = self.0.lock() {
            *slot = None;
        }
    }

    /// Cancel `turn_id` if it is the one in flight; otherwise a no-op (nothing
    /// armed, or a different turn — a late Esc never touches the wrong turn).
    pub fn cancel(&self, turn_id: TurnId) {
        if let Ok(slot) = self.0.lock() {
            if let Some((id, cancel)) = slot.as_ref() {
                if *id == turn_id {
                    cancel.cancel();
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

/// The chat serve loop for chat-only formats: the plain streaming
/// [`ChatSession`]. Grants are always refused here (CAPS-1 — a chat-only format
/// cannot enter the tool path); tool-trained formats never enter this loop.
#[allow(clippy::too_many_arguments)]
fn serve_chat(
    engine: &mut Engine,
    format: ChatFormat,
    system: Option<String>,
    opts: GenOpts,
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
                run_turn(&mut session, event_tx, format, turn_id, &text, &cancel);
                gate.disarm();
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
                run_agent_turn(&mut agent, event_tx, turn_id, &text, &cancel);
                gate.disarm();
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
    let mut tools = Tools::new()
        .with(ReadUrl::new(origins.clone())?)
        .with(ReadPage::with_limits(
            origins.clone(),
            knobs::READ_PAGE_MAX_INPUT_BYTES,
            knobs::READ_PAGE_MAX_CHARS,
        )?);
    let cache = std::env::home_dir()
        .map(|home| home.join(".cache/yatima"))
        .unwrap_or_else(std::env::temp_dir);
    tools = tools.with(ReadImage::new(origins.clone(), cache.join("images"))?);
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

/// Read back the image an artifact tool just wrote (a plot render, a fetched
/// image), returning its bytes and filename. The path is parsed from the tool's
/// `wrote <path> (…)` summary and so always points inside the tool's own
/// sandbox (PLOT-2 / IMG-1); this only ever fires for an artifact the user just
/// asked for. Format-agnostic — an SVG's raw bytes pass through; a view that
/// cannot show SVG rasterizes on receipt (that stays a view concern).
fn read_artifact(content: &str) -> Result<(Vec<u8>, String)> {
    let path = content
        .strip_prefix("wrote ")
        .and_then(|rest| rest.rsplit_once(" ("))
        .map(|(path, _)| path)
        .ok_or_else(|| anyhow::anyhow!("unrecognized artifact summary: {content:?}"))?;
    let bytes = std::fs::read(path)?;
    let name = std::path::Path::new(path)
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("artifact")
        .to_string();
    Ok((bytes, name))
}

/// Run one chat turn: stream `turn_streaming`'s raw fragments through a
/// [`ReasoningSplitter`] (so each emitted [`HostEvent::Fragment`] is already
/// classified), report the prompt-token count for the meter, then `Done`/`Error`.
fn run_turn(
    session: &mut ChatSession<'_, Engine, Box<dyn PromptTemplate>>,
    event_tx: &UnboundedSender<HostEvent>,
    format: ChatFormat,
    turn_id: TurnId,
    user: &str,
    cancel: &Cancel,
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
        Ok(_answer) => {
            // The streamed Fragment channels are the answer's authoritative
            // form; Done carries only why it stopped.
            if let Some(used) = session.last_prompt_tokens() {
                let _ = event_tx.send(HostEvent::Context {
                    prompt_tokens: used,
                });
            }
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
/// pane ([`HostEvent::RetractAnswer`]) and replays it as reasoning, ahead of the
/// `⚙` activity line. A successful plot/read_image ships its bytes as
/// [`HostEvent::Image`].
fn run_agent_turn<K: ToolCallCodec, T: PromptTemplate>(
    agent: &mut Agent<'_, Engine, K, T>,
    event_tx: &UnboundedSender<HostEvent>,
    turn_id: TurnId,
    user: &str,
    cancel: &Cancel,
) {
    let _ = event_tx.send(HostEvent::Started { turn_id });

    let fragment = |channel: LibChannel, text: String| {
        let _ = event_tx.send(HostEvent::Fragment {
            turn_id,
            channel: to_proto_channel(channel),
            text,
        });
    };
    let note = |text: String| {
        let _ = event_tx.send(HostEvent::ToolNote { turn_id, text });
    };

    // Answer prose streamed during the *current* step; a ToolCall event means it
    // was narration, not answer — retract and reclassify.
    let mut step_answer = String::new();
    // Name of the tool whose outcome we're awaiting, so the outcome handler
    // knows who produced it (a plot/read_image success ships its bytes).
    let mut pending_tool = String::new();

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
                pending_tool = call.name.clone();
                note(format!(
                    "\n⚙ {} {}\n",
                    call.name,
                    clip(&call.args.to_string(), 160)
                ));
            }
            AgentEvent::ToolStarted(_) => {}
            AgentEvent::ToolProgress(message) => {
                note(format!("  {message}\n"));
            }
            AgentEvent::ToolOutcome(outcome) => {
                // Plain `ok`/`failed:` vocabulary: shared host text renders in
                // egui too, whose built-in fonts lack ✓/✗ (they show as tofu).
                let text = match &outcome {
                    ToolOutcome::Success { content } => {
                        let flat = content.trim();
                        if flat.chars().count() <= knobs::TOOL_NOTE_MAX_CHARS
                            && !flat.contains('\n')
                        {
                            format!("  ok {flat}\n")
                        } else {
                            format!("  ok {} chars\n", content.chars().count())
                        }
                    }
                    other => format!(
                        "  failed: {}\n",
                        clip(&other.render_for_model("").content, 160)
                    ),
                };
                note(text);
                if matches!(pending_tool.as_str(), "plot" | "read_image") {
                    if let ToolOutcome::Success { content } = &outcome {
                        match read_artifact(content) {
                            Ok((bytes, name)) => {
                                let _ = event_tx.send(HostEvent::Image {
                                    turn_id,
                                    bytes,
                                    name,
                                });
                            }
                            Err(e) => note(format!("  failed: artifact: {e}\n")),
                        }
                    }
                }
                step_answer.clear();
            }
            // Already streamed fragment-by-fragment; Done carries the stop.
            AgentEvent::Final(_) => {}
        }
        Ok(ControlFlow::Continue(()))
    });

    match result {
        Ok(((), run)) => {
            let stop = match run.stop {
                AgentStop::Final => StopReason::Eos,
                AgentStop::Stopped => StopReason::Stopped,
                AgentStop::MaxSteps => {
                    note(format!(
                        "\n⚠ tool-step budget exhausted ({})\n",
                        knobs::AGENT_MAX_STEPS
                    ));
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
    fn artifact_summary_parses_the_wrote_contract() {
        // The `wrote <path> (…)` contract shared by plot and read_image: an
        // unrecognized summary and a missing file both error; a real file
        // yields its bytes and bare filename (the wire's Image.name).
        assert!(read_artifact("no such summary").is_err());
        assert!(read_artifact("wrote /nonexistent/x.png (png, 5 bytes)").is_err());

        let path = std::env::temp_dir().join("yatima-host-artifact-test.png");
        std::fs::write(&path, b"PNGDATA").unwrap();
        let content = format!("wrote {} (png, 7 bytes)", path.display());
        let (bytes, name) = read_artifact(&content).unwrap();
        assert_eq!(bytes, b"PNGDATA");
        assert_eq!(name, "yatima-host-artifact-test.png");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn cancel_gate_flips_only_the_armed_turn() {
        // The gate cancels the in-flight turn and ignores a stale id — a late
        // Esc for a finished turn never touches a newer one.
        let gate = CancelGate::new();
        let cancel = Cancel::new();
        gate.arm(0, cancel.clone());
        gate.cancel(1); // wrong turn: no-op
        assert!(!cancel.is_cancelled());
        gate.cancel(0);
        assert!(cancel.is_cancelled());
        gate.disarm();
        let stale = Cancel::new();
        gate.cancel(0); // disarmed: no-op
        assert!(!stale.is_cancelled());
    }

    #[test]
    fn clip_is_char_safe() {
        assert_eq!(clip("hello", 10), "hello");
        assert_eq!(clip("hello", 3), "hel…");
    }
}
