//! The engine-thread actor and the three-plane protocol.
//!
//! Local decode is `!Send` and runs on the runtime's blocking island (CMP-1 /
//! RT-2), so it cannot live in a `tokio::spawn`. A dedicated **OS thread** owns
//! the [`Engine`] *and* the [`ChatSession`] — the one authoritative prompt
//! history — and, because it is a plain thread (not a runtime worker), it calls
//! the public **sync** shim [`ChatSession::turn_streaming`] directly; the lib's
//! `block_on` / blocking-island machinery does its job and RT-1 is not violated.
//!
//! Three planes connect it to the async UI (TUI design keystone):
//!
//! - **request** (`std::sync::mpsc`, UI→actor): [`EngineRequest`]. The actor
//!   *blocks* on receive between turns and never `.await`s.
//! - **event** (`tokio::sync::mpsc`, actor→UI): [`EngineEvent`] — the UI's only
//!   source of transcript truth; the async loop `select!`s on it.
//! - **control** (shared [`TurnControl`], *not* queued): carried in `Submit` and
//!   held by both the UI and the decode callback, so a cancel is reachable while
//!   the actor is busy decoding (Slice 3 acts on it; Slice 1 plumbs it inert).

use std::path::PathBuf;
use std::sync::mpsc::{Receiver, Sender};

use anyhow::Result;
use tokio::sync::mpsc::{unbounded_channel, UnboundedReceiver, UnboundedSender};
use tokio::sync::oneshot;
use yatima_lib::{
    device, resolve_format, Agent, AgentEvent, AgentStop, Arch, Cancel, Channel, ChatFormat,
    ChatMlTemplate, ChatSession, Engine, GenOpts, JsonToolCall, PlainTemplate, Plot, PlotSandbox,
    PromptTemplate, QwenToolCall, ReadImage, ReadPage, ReadUrl, ReasoningSplitter, StopReason,
    ToolCallCodec, ToolOutcome, Tools, WebOrigins,
};

/// A turn identifier, monotonic per session. Lets the UI ignore stale events.
pub type TurnId = u64;

/// Request plane: UI → actor (queued; the actor blocks on it between turns).
pub enum EngineRequest {
    /// Run one turn. `control` is the per-turn **control-plane** handle (shared
    /// memory, out-of-band): the UI flips it and the decode loop polls it per
    /// token, so a cancel is reachable while the actor is busy decoding (TUI-6).
    Submit {
        turn_id: TurnId,
        user: String,
        control: Cancel,
    },
    /// Clear the conversation back to the system prompt. Granted origins are
    /// capability state, not conversation state — a reset keeps them (CAP-3).
    Reset,
    /// Grant a web origin for the session (CAP-3: the UI sends this only for
    /// user utterances — a typed URL or an explicit /grant). The first grant
    /// on a tool-trained format switches the session to the agent path.
    Grant { origin: String },
    /// Revoke a previously granted origin.
    Revoke { origin: String },
    /// Report the granted origins (a `Grants` event answers).
    ListGrants,
    /// Stop the actor and drop the engine.
    Shutdown,
}

/// Event plane: actor → UI (queued; the async loop `select!`s on it). The UI's
/// only source of transcript truth.
pub enum EngineEvent {
    /// A turn began. (`prompt_tokens` for the exact context meter arrives in
    /// Slice 2, once the lib exposes it.)
    Started { turn_id: TurnId },
    /// A classified slice of the live completion (REASON-1: reasoning vs answer).
    Fragment {
        turn_id: TurnId,
        channel: Channel,
        text: String,
    },
    /// The turn finished: the answer (reasoning stripped), why it stopped, and
    /// the prompt's token count (for the context meter), if known.
    Done {
        turn_id: TurnId,
        answer: String,
        stop: StopReason,
        prompt_tokens: Option<usize>,
    },
    /// The turn failed.
    Error { turn_id: TurnId, message: String },
    /// Retract the last `chars` characters streamed on the answer channel
    /// (AGENT-4): the step they belonged to turned out to be a tool call, so
    /// they were narration — the actor replays them on the reasoning channel.
    RetractAnswer { turn_id: TurnId, chars: usize },
    /// The granted-origin set after a grant/revoke/list, with a line for the
    /// transcript ("granted read access to …", an error, or the listing).
    Grants {
        origins: Vec<String>,
        message: String,
    },
}

/// What the actor needs to load a model (all `Send`, so it crosses into the
/// thread; the `!Send` `Engine` is then *created* inside the thread).
pub struct EngineConfig {
    pub dir: PathBuf,
    pub cpu: bool,
    pub opts: GenOpts,
    pub format: Option<ChatFormat>,
    pub system: Option<String>,
    pub model_label: String,
}

/// Tool rounds per turn before the agent gives up (AGENT-1); mirrors the CLI's
/// `--max-steps` default.
const AGENT_MAX_STEPS: usize = 6;

/// The base system prompt for tool-enabled sessions when `--system` is absent.
const DEFAULT_AGENT_SYSTEM: &str =
    "You are a helpful assistant. Call a tool when it helps, then answer. \
     Markdown image links do not render here: to show the user an image or \
     chart, call read_image (or plot) — its result is displayed \
     automatically.";

/// `read_page`'s readable-text budget for interactive use. The tool's own
/// default (40k chars ≈ 10–12k tokens) makes the next step's prefill take
/// minutes on a 32B local model; ~12k chars is plenty for summarize-and-answer
/// and keeps a tool turn interactive.
const READ_PAGE_MAX_CHARS: usize = 12_000;

/// `read_page`'s raw-input cap (unchanged from the tool's default).
const READ_PAGE_MAX_INPUT_BYTES: usize = 4_000_000;

/// A successful tool result at most this long (and single-line) is shown
/// verbatim in the reasoning fold; anything bigger is summarized as a char
/// count. Short results — a file path, a count, an ID — *are* the
/// deliverable, and counting their characters would hide them (the plot
/// tool's "wrote <path> …" being the motivating case).
const TOOL_NOTE_MAX_CHARS: usize = 200;

/// Model metadata reported once after a successful load — for the status bar.
#[derive(Clone)]
pub struct Ready {
    pub backend: String,
    pub arch: Arch,
    pub format: ChatFormat,
    /// The model's context window in tokens (for the meter denominator), if the
    /// model declares it.
    pub context_length: Option<usize>,
    pub model_label: String,
}

/// The UI-side handle to a running engine actor.
pub struct EngineHandle {
    pub req_tx: Sender<EngineRequest>,
    pub event_rx: UnboundedReceiver<EngineEvent>,
    pub ready: Ready,
}

/// Spawn the engine actor: load the model on its own OS thread and return a
/// handle once the model is ready (or an error if the load failed — before the
/// UI ever enters the alternate screen).
pub async fn spawn(config: EngineConfig) -> Result<EngineHandle> {
    let (req_tx, req_rx) = std::sync::mpsc::channel::<EngineRequest>();
    let (event_tx, event_rx) = unbounded_channel::<EngineEvent>();
    let (ready_tx, ready_rx) = oneshot::channel::<Result<Ready, String>>();

    std::thread::Builder::new()
        .name("yatima-engine".into())
        .spawn(move || actor_main(config, req_rx, event_tx, ready_tx))?;

    match ready_rx.await {
        Ok(Ok(ready)) => Ok(EngineHandle {
            req_tx,
            event_rx,
            ready,
        }),
        Ok(Err(message)) => Err(anyhow::anyhow!(message)),
        Err(_) => Err(anyhow::anyhow!(
            "engine thread exited before reporting readiness"
        )),
    }
}

/// The actor's body: load, then serve requests until shutdown. Owns the engine
/// and the session for the whole run; never crosses `!Send` over a thread.
fn actor_main(
    config: EngineConfig,
    req_rx: Receiver<EngineRequest>,
    event_tx: UnboundedSender<EngineEvent>,
    ready_tx: oneshot::Sender<Result<Ready, String>>,
) {
    let mut engine = match load_engine(&config) {
        Ok(engine) => engine,
        Err(e) => {
            let _ = ready_tx.send(Err(e.to_string()));
            return;
        }
    };
    let backend = engine.backend();
    let arch = engine.arch();
    // Capture before the session borrows the engine (for the context meter).
    let context_length = engine.context_length();
    let (format, _mismatch) = resolve_format(arch, config.format);

    // Tool-trained formats always carry the web tools, initially with an
    // empty origin set — hidden from the model (CAP-3a) and inert until a
    // grant arrives (sandbox by omission; CAP-3: grants come only from the
    // user, via the UI). Chat-only formats get none.
    let tool_trained = matches!(format, ChatFormat::Qwen | ChatFormat::Plain);
    let origins = WebOrigins::new();
    // Client construction cannot practically fail; degrade to empty tools
    // (the model simply never sees web tools) rather than dying.
    let tools = tool_trained.then(|| web_tools(&origins).unwrap_or_default());

    if ready_tx
        .send(Ok(Ready {
            backend,
            arch,
            format,
            context_length,
            model_label: config.model_label,
        }))
        .is_err()
    {
        return; // the UI gave up during load.
    }

    // Tool-trained formats serve the sessionful agent from turn one: the
    // web tools hide themselves while the origin set is empty (CAP-3a), so
    // pre-grant the model sees exactly the no-authority tools (plot), and a
    // grant simply surfaces the web tools mid-session — /grant mints
    // authority, it is not a mode switch. Chat-only formats stay on the
    // plain chat path forever.
    let Some(tools) = tools else {
        serve_chat(
            &mut engine,
            format,
            config.system.clone(),
            config.opts.clone(),
            &req_rx,
            &event_tx,
        );
        return;
    };
    let system = config
        .system
        .unwrap_or_else(|| DEFAULT_AGENT_SYSTEM.to_string());
    match format {
        ChatFormat::Qwen => serve_agent(
            &mut engine,
            &tools,
            QwenToolCall,
            ChatMlTemplate,
            system,
            config.opts,
            &origins,
            req_rx,
            event_tx,
        ),
        ChatFormat::Plain => serve_agent(
            &mut engine,
            &tools,
            JsonToolCall,
            PlainTemplate,
            system,
            config.opts,
            &origins,
            req_rx,
            event_tx,
        ),
        _ => unreachable!("the switch is only offered on tool-trained formats"),
    }
}

/// The chat serve loop for chat-only formats: the plain streaming
/// [`ChatSession`]. Grants are always refused here (CAPS-1 — a chat-only
/// format cannot enter the tool path); tool-trained formats never enter
/// this loop (they serve the agent from turn one).
fn serve_chat(
    engine: &mut Engine,
    format: ChatFormat,
    system: Option<String>,
    opts: GenOpts,
    req_rx: &Receiver<EngineRequest>,
    event_tx: &UnboundedSender<EngineEvent>,
) {
    let template = format.template();
    let mut session = ChatSession::new(engine, template).with_opts(opts);
    if let Some(system) = system {
        session = session.with_system(system);
    }

    while let Ok(req) = req_rx.recv() {
        match req {
            EngineRequest::Submit {
                turn_id,
                user,
                control,
            } => run_turn(&mut session, event_tx, format, turn_id, &user, &control),
            EngineRequest::Reset => session.reset(),
            EngineRequest::Grant { origin } => {
                let _ = event_tx.send(EngineEvent::Grants {
                    origins: vec![],
                    message: format!(
                        "cannot grant {origin}: the {format} format is chat-only \
                         (tool calling needs qwen or plain)"
                    ),
                });
            }
            EngineRequest::Revoke { origin } => {
                report_revoke(event_tx, None, &origin);
            }
            EngineRequest::ListGrants => {
                report_grants(event_tx, None);
            }
            EngineRequest::Shutdown => return,
        }
    }
}

/// The agent serve loop: one sessionful [`Agent`] (AGENT-3) serves every turn,
/// seeded with the chat phase's history. Later grants/revokes mutate the
/// shared origin set in place — the specs re-render each run (CAP-3a).
#[allow(clippy::too_many_arguments)]
fn serve_agent<K: ToolCallCodec, T: PromptTemplate>(
    engine: &mut Engine,
    tools: &Tools,
    codec: K,
    template: T,
    system: String,
    opts: GenOpts,
    origins: &WebOrigins,
    req_rx: Receiver<EngineRequest>,
    event_tx: UnboundedSender<EngineEvent>,
) {
    let mut agent =
        Agent::new(engine, tools, codec, template, system, AGENT_MAX_STEPS).with_opts(opts);

    while let Ok(req) = req_rx.recv() {
        match req {
            EngineRequest::Submit {
                turn_id,
                user,
                control,
            } => run_agent_turn(&mut agent, &event_tx, turn_id, &user, &control),
            EngineRequest::Reset => agent.reset(),
            EngineRequest::Grant { origin } => match origins.grant(&origin) {
                Ok(true) => {
                    let list = origins.list();
                    let message = if list.len() == 1 {
                        format!("granted read access to {origin} — web tools enabled")
                    } else {
                        format!("granted read access to {origin}")
                    };
                    let _ = event_tx.send(EngineEvent::Grants {
                        origins: list,
                        message,
                    });
                }
                Ok(false) => {
                    let _ = event_tx.send(EngineEvent::Grants {
                        origins: origins.list(),
                        message: format!("{origin} was already granted"),
                    });
                }
                Err(e) => {
                    let _ = event_tx.send(EngineEvent::Grants {
                        origins: origins.list(),
                        message: format!("grant failed: {e}"),
                    });
                }
            },
            EngineRequest::Revoke { origin } => {
                report_revoke(&event_tx, Some(origins), &origin);
            }
            EngineRequest::ListGrants => {
                report_grants(&event_tx, Some(origins));
            }
            EngineRequest::Shutdown => break,
        }
    }
}

/// Answer a revoke request (both phases).
fn report_revoke(
    event_tx: &UnboundedSender<EngineEvent>,
    origins: Option<&WebOrigins>,
    origin: &str,
) {
    let Some(origins) = origins else {
        let _ = event_tx.send(EngineEvent::Grants {
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
    let _ = event_tx.send(EngineEvent::Grants {
        origins: origins.list(),
        message,
    });
}

/// Answer a list request (both phases).
fn report_grants(event_tx: &UnboundedSender<EngineEvent>, origins: Option<&WebOrigins>) {
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
    let _ = event_tx.send(EngineEvent::Grants {
        origins: list,
        message,
    });
}

/// The web tools over a shared (growable) origin set. Present from the start
/// on tool-trained formats; hidden from the model while the set is empty
/// (CAP-3a). The plot tool rides along when a python-with-matplotlib is
/// present (PLOT-1..3: declarative specs only, output confined to
/// `~/.cache/yatima/plots` — stable and discoverable, and content-hash
/// names make re-renders idempotent across sessions) — and quietly doesn't
/// when it isn't; the model never sees a tool it cannot use.
fn web_tools(origins: &WebOrigins) -> Result<Tools> {
    let mut tools = Tools::new()
        .with(ReadUrl::new(origins.clone())?)
        .with(ReadPage::with_limits(
            origins.clone(),
            READ_PAGE_MAX_INPUT_BYTES,
            READ_PAGE_MAX_CHARS,
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

fn load_engine(config: &EngineConfig) -> Result<Engine> {
    let dev = device(config.cpu)?;
    Engine::load(&config.dir, dev)
}

/// Run one turn: stream `turn_streaming`'s raw fragments through a
/// [`ReasoningSplitter`] (so each emitted [`EngineEvent::Fragment`] is already
/// classified — channel *classification* lives with the actor that owns the
/// format), then report `Done`/`Error`.
fn run_turn(
    session: &mut ChatSession<'_, Engine, Box<dyn yatima_lib::PromptTemplate>>,
    event_tx: &UnboundedSender<EngineEvent>,
    format: ChatFormat,
    turn_id: TurnId,
    user: &str,
    cancel: &Cancel,
) {
    let _ = event_tx.send(EngineEvent::Started { turn_id });

    let mut splitter = if format.pre_seeds_reasoning() {
        ReasoningSplitter::seeded()
    } else {
        ReasoningSplitter::new()
    };

    let outcome = {
        let mut on_token = |frag: &str| {
            splitter.push(frag, |channel, text| {
                let _ = event_tx.send(EngineEvent::Fragment {
                    turn_id,
                    channel,
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
        let _ = event_tx.send(EngineEvent::Fragment {
            turn_id,
            channel,
            text: text.to_string(),
        });
    });

    match outcome {
        Ok(answer) => {
            let stop = session.last_stop().unwrap_or(StopReason::Eos);
            let _ = event_tx.send(EngineEvent::Done {
                turn_id,
                answer,
                stop,
                prompt_tokens: session.last_prompt_tokens(),
            });
        }
        Err(e) => {
            let _ = event_tx.send(EngineEvent::Error {
                turn_id,
                message: e.to_string(),
            });
        }
    }
}

/// Pop a just-produced image artifact (a plot render, a fetched image) in
/// the platform viewer (macOS `open`). Fire-and-forget: viewing is a
/// courtesy, never an error — failures are ignored and a reaper thread
/// waits the child so no zombies accrue. The path is parsed from the
/// tool's `wrote <path> (…)` summary and so always points inside the
/// tool's own sandbox (PLOT-2 / IMG-1); this only ever fires for an
/// artifact the user just asked for.
fn open_artifact(content: &str) {
    #[cfg(target_os = "macos")]
    if let Some((path, _)) = content
        .strip_prefix("wrote ")
        .and_then(|rest| rest.rsplit_once(" ("))
    {
        if let Ok(mut child) = std::process::Command::new("open").arg(path).spawn() {
            std::thread::spawn(move || {
                let _ = child.wait();
            });
        }
    }
    #[cfg(not(target_os = "macos"))]
    let _ = content;
}

/// Run one agent turn, folding [`AgentEvent`]s onto the event plane. Each
/// step's decode **streams** (AGENT-4): classified fragments arrive live —
/// reasoning and tool activity on [`Channel::Reasoning`], answer prose on
/// [`Channel::Answer`] — and the turn's `cancel` is token-level. A step that
/// turns out to be a tool call retracts its streamed narration from the
/// answer pane (RetractAnswer) and replays it as working matter in the
/// reasoning pane, ahead of the ⚙ activity line.
fn run_agent_turn<K: ToolCallCodec, T: PromptTemplate>(
    agent: &mut Agent<'_, Engine, K, T>,
    event_tx: &UnboundedSender<EngineEvent>,
    turn_id: TurnId,
    user: &str,
    cancel: &Cancel,
) {
    let _ = event_tx.send(EngineEvent::Started { turn_id });

    let fragment = |channel: Channel, text: String| {
        let _ = event_tx.send(EngineEvent::Fragment {
            turn_id,
            channel,
            text,
        });
    };

    // Answer prose streamed during the *current* step; a ToolCall event means
    // it was narration, not answer — retract and reclassify.
    let mut step_answer = String::new();

    // Name of the tool whose outcome we're awaiting, so the outcome handler
    // knows who produced it (the plot tool's success pops a viewer).
    let mut pending_tool = String::new();

    let result = agent.run_with_cancellable(user, cancel, (), |(), event| {
        match event {
            AgentEvent::Fragment { channel, text } => {
                if channel == Channel::Answer {
                    step_answer.push_str(&text);
                }
                fragment(channel, text);
            }
            // The per-step aggregate; already streamed via Fragment (AGENT-4).
            AgentEvent::Reasoning(_) => {}
            AgentEvent::ToolCall(call) => {
                if !step_answer.is_empty() {
                    let narration = std::mem::take(&mut step_answer);
                    let _ = event_tx.send(EngineEvent::RetractAnswer {
                        turn_id,
                        chars: narration.chars().count(),
                    });
                    fragment(Channel::Reasoning, format!("{}\n", narration.trim_end()));
                }
                pending_tool = call.name.clone();
                fragment(
                    Channel::Reasoning,
                    format!("\n⚙ {} {}\n", call.name, clip(&call.args.to_string(), 160)),
                );
            }
            AgentEvent::ToolStarted(_) => {}
            AgentEvent::ToolProgress(message) => {
                fragment(Channel::Reasoning, format!("  {message}\n"));
            }
            AgentEvent::ToolOutcome(outcome) => {
                let note = match &outcome {
                    ToolOutcome::Success { content } => {
                        let flat = content.trim();
                        if flat.chars().count() <= TOOL_NOTE_MAX_CHARS && !flat.contains('\n') {
                            format!("  ✓ {flat}\n")
                        } else {
                            format!("  ✓ {} chars\n", content.chars().count())
                        }
                    }
                    other => format!("  ✗ {}\n", clip(&other.render_for_model("").content, 160)),
                };
                fragment(Channel::Reasoning, note);
                if matches!(pending_tool.as_str(), "plot" | "read_image") {
                    if let ToolOutcome::Success { content } = &outcome {
                        open_artifact(content);
                    }
                }
                step_answer.clear();
            }
            // Already streamed fragment-by-fragment; Done carries the answer.
            AgentEvent::Final(_) => {}
        }
        Ok(std::ops::ControlFlow::Continue(()))
    });

    match result {
        Ok(((), run)) => {
            let stop = match run.stop {
                AgentStop::Final => StopReason::Eos,
                AgentStop::Stopped => StopReason::Stopped,
                AgentStop::MaxSteps => {
                    fragment(
                        Channel::Reasoning,
                        format!("\n⚠ tool-step budget exhausted ({AGENT_MAX_STEPS})\n"),
                    );
                    StopReason::MaxTokens
                }
            };
            let _ = event_tx.send(EngineEvent::Done {
                turn_id,
                answer: run.answer,
                stop,
                prompt_tokens: None,
            });
        }
        Err(e) => {
            let _ = event_tx.send(EngineEvent::Error {
                turn_id,
                message: e.to_string(),
            });
        }
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
