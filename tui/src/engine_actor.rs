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
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, Sender};
use std::sync::Arc;

use anyhow::Result;
use tokio::sync::mpsc::{unbounded_channel, UnboundedReceiver, UnboundedSender};
use tokio::sync::oneshot;
use yatima_lib::{
    device, resolve_format, Arch, Channel, ChatFormat, ChatSession, Engine, GenOpts,
    ReasoningSplitter, StopReason,
};

/// A turn identifier, monotonic per session. Lets the UI ignore stale events.
pub type TurnId = u64;

/// The per-turn **control plane** (shared memory, out-of-band). The UI flips
/// `cancel`; the decode callback polls it mid-decode. `Arc<AtomicBool>` is the
/// simplest correct first version for a flag crossing a plain OS thread and a
/// sync callback; a `CancellationToken` is the drop-in upgrade later.
#[derive(Clone)]
pub struct TurnControl {
    pub cancel: Arc<AtomicBool>,
}

impl TurnControl {
    pub fn new() -> TurnControl {
        TurnControl {
            cancel: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Signal cancellation (Slice 3 — the decode callback cannot yet act on it).
    #[allow(dead_code)]
    pub fn cancel(&self) {
        self.cancel.store(true, Ordering::SeqCst);
    }

    /// Whether cancellation has been requested.
    #[allow(dead_code)]
    pub fn is_cancelled(&self) -> bool {
        self.cancel.load(Ordering::SeqCst)
    }
}

impl Default for TurnControl {
    fn default() -> TurnControl {
        TurnControl::new()
    }
}

/// Request plane: UI → actor (queued; the actor blocks on it between turns).
pub enum EngineRequest {
    /// Run one turn. `control` is the per-turn control-plane handle (inert in
    /// Slice 1).
    Submit {
        turn_id: TurnId,
        user: String,
        control: TurnControl,
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
    pub model_label: String,
}

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

    let template = format.template();
    let mut session = ChatSession::new(&mut engine, template).with_opts(config.opts);
    if let Some(system) = config.system {
        session = session.with_system(system);
    }

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

    while let Ok(req) = req_rx.recv() {
        match req {
            EngineRequest::Submit {
                turn_id,
                user,
                control: _control, // Slice 1: plumbed, inert (turn_streaming can't break yet).
            } => run_turn(&mut session, &event_tx, format, turn_id, &user),
            EngineRequest::Reset => session.reset(),
            EngineRequest::Shutdown => break,
        }
    }
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
            .turn_streaming(user, &mut on_token)
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
