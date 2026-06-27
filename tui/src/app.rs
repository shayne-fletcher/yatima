//! UI state (a render *mirror*), input handling, and the async event loop.
//!
//! The engine thread owns the authoritative `ChatSession`; [`App`] holds only a
//! mirror rebuilt from [`EngineEvent`]s, plus input/scroll/status. State changes
//! flow through one of two places: a key [`Intent`] (`apply`) or an engine event
//! (`on_engine_event`); the transcript grows only through [`App::push_entry`]
//! (TUI-3). Rendering is a pure projection (`render::ui(&App)`, TUI-2).

use std::io;
use std::sync::mpsc::Sender;
use std::time::Instant;

use anyhow::Result;
use crossterm::event::{Event, KeyCode, KeyEvent, KeyEventKind, KeyModifiers};
use futures::{Stream, StreamExt};
use ratatui::backend::Backend;
use ratatui::Terminal;
use tokio::sync::mpsc::UnboundedReceiver;
use yatima_lib::{Channel, StopReason};

use crate::engine_actor::{EngineEvent, EngineRequest, Ready, TurnControl, TurnId};
use crate::render;

/// One rendered transcript entry (the mirror; the actor's session is truth).
pub enum Entry {
    User(String),
    Assistant {
        reasoning: String,
        answer: String,
        stop: Option<StopReason>,
    },
    Error(String),
}

/// The single in-flight turn (TUI-7: at most one at a time).
pub struct InFlight {
    pub turn_id: TurnId,
    pub started: Instant,
    pub frags: usize,
    /// Whether the answer channel has begun (vs still reasoning) — drives the
    /// "thinking" → "answering" phase in the activity indicator.
    pub answering: bool,
    /// Control-plane handle (Slice 1: plumbed, inert).
    pub control: TurnControl,
}

/// Status-bar facts.
pub struct Status {
    pub model_label: String,
    pub backend: String,
    pub format: String,
}

/// What a key press means (classified pure of effects).
#[derive(Debug, PartialEq, Eq)]
pub enum Intent {
    None,
    Quit,
    Submit,
    Backspace,
    Insert(char),
    ScrollUp,
    ScrollDown,
}

/// The UI render model.
pub struct App {
    pub req_tx: Sender<EngineRequest>,
    pub transcript: Vec<Entry>,
    pub input: String,
    /// Lines scrolled up from the bottom (0 = follow latest). Display is always
    /// clamped by [`scroll_y`] (TUI-1).
    pub scroll_back: usize,
    pub in_flight: Option<InFlight>,
    pub status: Status,
    pub should_quit: bool,
    next_turn_id: TurnId,
}

impl App {
    pub fn new(req_tx: Sender<EngineRequest>, ready: Ready) -> App {
        App {
            req_tx,
            transcript: Vec::new(),
            input: String::new(),
            scroll_back: 0,
            in_flight: None,
            status: Status {
                model_label: ready.model_label,
                backend: ready.backend,
                format: ready.format.to_string(),
            },
            should_quit: false,
            next_turn_id: 0,
        }
    }

    /// Classify a key press into an [`Intent`] (no effects — testable).
    pub fn classify(key: KeyEvent) -> Intent {
        let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
        match key.code {
            KeyCode::Char('c') if ctrl => Intent::Quit,
            KeyCode::Char('d') if ctrl => Intent::Quit,
            KeyCode::PageUp => Intent::ScrollUp,
            KeyCode::PageDown => Intent::ScrollDown,
            KeyCode::Enter => Intent::Submit,
            KeyCode::Backspace => Intent::Backspace,
            KeyCode::Char(c) => Intent::Insert(c),
            _ => Intent::None,
        }
    }

    pub fn on_key(&mut self, key: KeyEvent) {
        self.apply(App::classify(key));
    }

    /// Apply an intent's effect.
    pub fn apply(&mut self, intent: Intent) {
        match intent {
            Intent::None => {}
            Intent::Quit => self.should_quit = true,
            Intent::Submit => self.start_turn(),
            Intent::Backspace => {
                self.input.pop();
            }
            Intent::Insert(c) => self.input.push(c),
            Intent::ScrollUp => self.scroll_back = self.scroll_back.saturating_add(3),
            Intent::ScrollDown => self.scroll_back = self.scroll_back.saturating_sub(3),
        }
    }

    /// Begin a turn — unless input is empty or a turn is already in flight
    /// (TUI-7 single-in-flight).
    fn start_turn(&mut self) {
        let user = self.input.trim().to_string();
        if user.is_empty() || self.in_flight.is_some() {
            return;
        }
        let turn_id = self.next_turn_id;
        self.next_turn_id += 1;
        let control = TurnControl::new();
        self.push_entry(Entry::User(user.clone()));
        self.in_flight = Some(InFlight {
            turn_id,
            started: Instant::now(),
            frags: 0,
            answering: false,
            control: control.clone(),
        });
        let _ = self.req_tx.send(EngineRequest::Submit {
            turn_id,
            user,
            control,
        });
        self.input.clear();
        self.scroll_back = 0; // jump to the latest
    }

    /// The single transcript-append path (TUI-3).
    pub fn push_entry(&mut self, entry: Entry) {
        self.transcript.push(entry);
    }

    fn is_current(&self, turn_id: TurnId) -> bool {
        self.in_flight
            .as_ref()
            .is_some_and(|f| f.turn_id == turn_id)
    }

    /// Fold an engine event into the render mirror (the only event entry point).
    pub fn on_engine_event(&mut self, event: EngineEvent) {
        match event {
            EngineEvent::Started { turn_id } if self.is_current(turn_id) => {
                self.push_entry(Entry::Assistant {
                    reasoning: String::new(),
                    answer: String::new(),
                    stop: None,
                });
            }
            EngineEvent::Fragment {
                turn_id,
                channel,
                text,
            } if self.is_current(turn_id) => {
                if let Some(f) = self.in_flight.as_mut() {
                    f.frags += 1;
                    if channel == Channel::Answer {
                        f.answering = true;
                    }
                }
                self.append_fragment(channel, &text);
            }
            EngineEvent::Done {
                turn_id,
                answer,
                stop,
            } if self.is_current(turn_id) => {
                self.finish_assistant(answer, stop);
                self.in_flight = None;
            }
            EngineEvent::Error { turn_id, message } if self.is_current(turn_id) => {
                self.push_entry(Entry::Error(message));
                self.in_flight = None;
            }
            _ => {} // stale event for a turn that is no longer current.
        }
    }

    /// Append a classified fragment to the assistant entry in progress.
    fn append_fragment(&mut self, channel: Channel, text: &str) {
        if let Some(Entry::Assistant {
            reasoning, answer, ..
        }) = self.transcript.last_mut()
        {
            match channel {
                Channel::Reasoning => reasoning.push_str(text),
                Channel::Answer => answer.push_str(text),
            }
        }
    }

    /// Finalize the assistant entry with the authoritative answer + stop reason.
    fn finish_assistant(&mut self, answer: String, stop: StopReason) {
        if let Some(Entry::Assistant {
            answer: a, stop: s, ..
        }) = self.transcript.last_mut()
        {
            *a = answer;
            *s = Some(stop);
        }
    }
}

/// The top row index to render so the displayed window is always within bounds
/// (TUI-1): the result is in `[0, total.saturating_sub(viewport)]`.
pub fn scroll_y(total: usize, viewport: usize, scroll_back: usize) -> usize {
    total.saturating_sub(viewport).saturating_sub(scroll_back)
}

/// The async event loop: draw, then `select!` over key events and engine events.
/// Generic over the key-event stream so it is testable with an injected stream
/// and a `TestBackend` (TUI-4). Generation runs on the engine thread, so this
/// loop never blocks on decode.
pub async fn run_loop<B, S>(
    terminal: &mut Terminal<B>,
    mut app: App,
    mut event_rx: UnboundedReceiver<EngineEvent>,
    mut key_events: S,
) -> Result<()>
where
    B: Backend,
    S: Stream<Item = io::Result<Event>> + Unpin,
{
    // A periodic tick redraws on a timer so the live activity indicator (spinner
    // + elapsed) animates even when the model stalls between tokens — without it,
    // a slow reasoning stretch looks hung (TUI: thinking ≠ frozen).
    let mut tick = tokio::time::interval(std::time::Duration::from_millis(120));
    tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

    loop {
        terminal.draw(|frame| render::ui(frame, &app))?;
        tokio::select! {
            _ = tick.tick() => {} // wake to re-draw the animated indicator
            maybe_key = key_events.next() => {
                match maybe_key {
                    Some(Ok(Event::Key(key))) if key.kind == KeyEventKind::Press => app.on_key(key),
                    Some(Ok(_)) => {} // resize / other
                    Some(Err(_)) | None => app.should_quit = true,
                }
            }
            maybe_event = event_rx.recv() => {
                match maybe_event {
                    Some(event) => app.on_engine_event(event),
                    None => app.should_quit = true, // actor gone
                }
            }
        }
        if app.should_quit {
            break;
        }
    }
    let _ = app.req_tx.send(EngineRequest::Shutdown);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine_actor::Ready;
    use ratatui::backend::TestBackend;
    use std::time::Duration;
    use tokio::sync::mpsc::unbounded_channel;
    use yatima_lib::{Arch, ChatFormat};

    fn test_app() -> (App, std::sync::mpsc::Receiver<EngineRequest>) {
        let (tx, rx) = std::sync::mpsc::channel();
        let ready = Ready {
            backend: "test".into(),
            arch: Arch::Qwen2,
            format: ChatFormat::Qwen,
            model_label: "test-model".into(),
        };
        (App::new(tx, ready), rx)
    }

    fn key(code: KeyCode) -> KeyEvent {
        KeyEvent::new(code, KeyModifiers::NONE)
    }

    #[test]
    fn scroll_y_is_always_in_bounds() {
        // upholds: TUI-1 — the displayed top row never exceeds the max scroll.
        for &(total, viewport, back) in &[
            (0, 24, 0),
            (10, 24, 0),
            (100, 24, 0),
            (100, 24, 5),
            (100, 24, 1000),
            (5, 5, 3),
        ] {
            let y = scroll_y(total, viewport, back);
            assert!(y <= total.saturating_sub(viewport), "y={y} total={total}");
        }
    }

    #[test]
    fn transcript_grows_only_via_push_entry() {
        // upholds: TUI-3 — fragments mutate the last entry, never append. A turn
        // adds exactly two entries (User on submit, Assistant on Started).
        let (mut app, _rx) = test_app();
        app.input = "hi".into();
        app.apply(Intent::Submit);
        assert_eq!(app.transcript.len(), 1); // User
        app.on_engine_event(EngineEvent::Started { turn_id: 0 });
        assert_eq!(app.transcript.len(), 2); // + Assistant
        for _ in 0..5 {
            app.on_engine_event(EngineEvent::Fragment {
                turn_id: 0,
                channel: Channel::Answer,
                text: "x".into(),
            });
        }
        app.on_engine_event(EngineEvent::Done {
            turn_id: 0,
            answer: "xxxxx".into(),
            stop: StopReason::Eos,
        });
        assert_eq!(app.transcript.len(), 2, "fragments/done must not append");
    }

    #[test]
    fn submit_is_blocked_while_in_flight() {
        // upholds: TUI-7 — a new prompt cannot start while a turn is active.
        let (mut app, rx) = test_app();
        app.input = "first".into();
        app.apply(Intent::Submit);
        assert!(app.in_flight.is_some());
        assert!(matches!(rx.try_recv(), Ok(EngineRequest::Submit { .. })));

        // A second submit while in flight is a no-op: no new request, input kept.
        app.input = "second".into();
        app.apply(Intent::Submit);
        assert_eq!(app.input, "second");
        assert!(rx.try_recv().is_err(), "no second Submit while in flight");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn ui_stays_live_during_generation() {
        // upholds: TUI-4 — generation runs off the UI loop, so a key is serviced
        // promptly even while the "engine" is mid-decode. A background thread
        // feeds fragments with 100ms gaps (≈500ms total); a Quit key is ready at
        // once. If decode blocked the loop, quitting would wait ~500ms.
        let (mut app, _rx) = test_app();
        app.in_flight = Some(InFlight {
            turn_id: 0,
            started: Instant::now(),
            frags: 0,
            answering: false,
            control: TurnControl::new(),
        });

        let (event_tx, event_rx) = unbounded_channel();
        std::thread::spawn(move || {
            for i in 0..5 {
                std::thread::sleep(Duration::from_millis(100));
                if event_tx
                    .send(EngineEvent::Fragment {
                        turn_id: 0,
                        channel: Channel::Answer,
                        text: format!("t{i}"),
                    })
                    .is_err()
                {
                    return;
                }
            }
        });

        let keys = futures::stream::iter(vec![Ok(Event::Key(KeyEvent::new(
            KeyCode::Char('c'),
            KeyModifiers::CONTROL,
        )))]);
        let mut terminal = Terminal::new(TestBackend::new(80, 24)).unwrap();

        let start = Instant::now();
        run_loop(&mut terminal, app, event_rx, keys).await.unwrap();
        assert!(
            start.elapsed() < Duration::from_millis(250),
            "UI blocked during generation: took {:?}",
            start.elapsed()
        );
    }

    #[test]
    fn classify_maps_keys_to_intents() {
        assert_eq!(
            App::classify(KeyEvent::new(KeyCode::Char('c'), KeyModifiers::CONTROL)),
            Intent::Quit
        );
        assert_eq!(App::classify(key(KeyCode::Enter)), Intent::Submit);
        assert_eq!(App::classify(key(KeyCode::Char('a'))), Intent::Insert('a'));
        assert_eq!(App::classify(key(KeyCode::PageUp)), Intent::ScrollUp);
    }
}
