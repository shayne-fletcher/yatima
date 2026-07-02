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
    ChatMlTemplate, ChatSession, Engine, GenOpts, JsonToolCall, PlainTemplate, PromptTemplate,
    QwenToolCall, ReadPage, ReadUrl, ReasoningSplitter, StopReason, ToolCallCodec, ToolOutcome,
    Tools, WebOrigin,
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
    /// Clear the conversation back to the system prompt.
    Reset,
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
}

/// What the actor needs to load a model (all `Send`, so it crosses into the
/// thread; the `!Send` `Engine` is then *created* inside the thread).
pub struct EngineConfig {
    pub dir: PathBuf,
    pub cpu: bool,
    pub opts: GenOpts,
    pub format: Option<ChatFormat>,
    pub system: Option<String>,
    /// Grant the model HTTP tools (`read_url`, `read_page`) scoped to this
    /// origin (CAP-2). When set, turns run through the sessionful tool-calling
    /// [`Agent`] instead of the plain [`ChatSession`].
    pub web_origin: Option<String>,
    pub model_label: String,
}

/// Tool rounds per turn before the agent gives up (AGENT-1); mirrors the CLI's
/// `--max-steps` default.
const AGENT_MAX_STEPS: usize = 6;

/// The base system prompt for tool-enabled sessions when `--system` is absent.
const DEFAULT_AGENT_SYSTEM: &str =
    "You are a helpful assistant. You can fetch web pages with the provided \
     tools. Call a tool when it helps, then answer.";

/// `read_page`'s readable-text budget for interactive use. The tool's own
/// default (40k chars ≈ 10–12k tokens) makes the next step's prefill take
/// minutes on a 32B local model; ~12k chars is plenty for summarize-and-answer
/// and keeps a tool turn interactive.
const READ_PAGE_MAX_CHARS: usize = 12_000;

/// `read_page`'s raw-input cap (unchanged from the tool's default).
const READ_PAGE_MAX_INPUT_BYTES: usize = 4_000_000;

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

    // Build the granted tools before reporting ready, so a bad origin or a
    // chat-only format fails the load cleanly (before the alternate screen).
    let tools = match agent_tools(config.web_origin.as_deref(), format) {
        Ok(tools) => tools,
        Err(e) => {
            let _ = ready_tx.send(Err(e.to_string()));
            return;
        }
    };

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

    match tools {
        None => serve_chat(
            &mut engine,
            format,
            config.system,
            config.opts,
            req_rx,
            event_tx,
        ),
        // A `--web-origin` grant routes turns through the sessionful agent.
        // The codec/template pair is monomorphic per format (as in the CLI);
        // `agent_tools` has already rejected chat-only formats (CAPS-1).
        Some(tools) => {
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
                    req_rx,
                    event_tx,
                ),
                _ => unreachable!("agent_tools rejects chat-only formats"),
            }
        }
    }
}

/// The chat serve loop: the plain [`ChatSession`], no tools (the pre-agent
/// behavior, byte for byte).
fn serve_chat(
    engine: &mut Engine,
    format: ChatFormat,
    system: Option<String>,
    opts: GenOpts,
    req_rx: Receiver<EngineRequest>,
    event_tx: UnboundedSender<EngineEvent>,
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
            } => run_turn(&mut session, &event_tx, format, turn_id, &user, &control),
            EngineRequest::Reset => session.reset(),
            EngineRequest::Shutdown => break,
        }
    }
}

/// The agent serve loop: one sessionful [`Agent`] (AGENT-3) serves every turn,
/// so exchanges remember each other while tool rounds stay ephemeral.
#[allow(clippy::too_many_arguments)]
fn serve_agent<K: ToolCallCodec, T: PromptTemplate>(
    engine: &mut Engine,
    tools: &Tools,
    codec: K,
    template: T,
    system: String,
    opts: GenOpts,
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
            EngineRequest::Shutdown => break,
        }
    }
}

/// The HTTP tools granted by `--web-origin`, or `None` without one. Rejects
/// chat-only formats: the tool loop needs a tool-trained codec (CAPS-1).
fn agent_tools(origin: Option<&str>, format: ChatFormat) -> Result<Option<Tools>> {
    let Some(origin) = origin else {
        return Ok(None);
    };
    if !matches!(format, ChatFormat::Qwen | ChatFormat::Plain) {
        anyhow::bail!(
            "--web-origin needs a tool-trained chat format (qwen or plain); \
             {format} is chat-only"
        );
    }
    Ok(Some(
        Tools::new()
            .with(ReadUrl::new(WebOrigin::new(origin)?)?)
            .with(ReadPage::with_limits(
                WebOrigin::new(origin)?,
                READ_PAGE_MAX_INPUT_BYTES,
                READ_PAGE_MAX_CHARS,
            )?),
    ))
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

/// Run one agent turn, folding [`AgentEvent`]s onto the event plane. Reasoning
/// and tool activity ride the [`Channel::Reasoning`] fragments — tool rounds
/// are working matter, so the reasoning pane is their honest home — and the
/// final answer rides [`Channel::Answer`]. The fold polls `cancel`, so a
/// cancel takes effect at the next event boundary (step granularity: the
/// agent's decode itself is not yet token-cancellable, unlike the chat path).
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

    let result = agent.run_with(user, (), |(), event| {
        match event {
            AgentEvent::Reasoning(text) => fragment(Channel::Reasoning, format!("{text}\n")),
            AgentEvent::ToolCall(call) => fragment(
                Channel::Reasoning,
                format!("\n⚙ {} {}\n", call.name, clip(&call.args.to_string(), 160)),
            ),
            AgentEvent::ToolStarted(_) => {}
            AgentEvent::ToolProgress(message) => {
                fragment(Channel::Reasoning, format!("  {message}\n"));
            }
            AgentEvent::ToolOutcome(outcome) => {
                let note = match &outcome {
                    ToolOutcome::Success { content } => {
                        format!("  ✓ {} chars\n", content.chars().count())
                    }
                    other => format!("  ✗ {}\n", clip(&other.render_for_model("").content, 160)),
                };
                fragment(Channel::Reasoning, note);
            }
            AgentEvent::Final(text) => fragment(Channel::Answer, text),
        }
        Ok(if cancel.is_cancelled() {
            std::ops::ControlFlow::Break(())
        } else {
            std::ops::ControlFlow::Continue(())
        })
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
