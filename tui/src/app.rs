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
use ratatui::style::Style;
use ratatui::Terminal;
use tokio::sync::mpsc::UnboundedReceiver;
use tui_textarea::{CursorMove, Input, TextArea};
use yatima_lib::{Cancel, Channel, StopReason};

use crate::engine_actor::{EngineEvent, EngineRequest, Ready, TurnId};
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
    /// Whether a cancel has been requested for this turn (the decode stops at the
    /// next token boundary; the indicator shows "cancelling…" until `Done`).
    pub cancelling: bool,
    /// Control-plane handle: flip it to cancel this turn in flight (TUI-6).
    pub control: Cancel,
}

/// Status-bar facts.
pub struct Status {
    pub model_label: String,
    pub backend: String,
    pub format: String,
    /// The model's context window (meter denominator), if declared.
    pub context_length: Option<usize>,
    /// Tokens in the most recent prompt (meter numerator), once a turn completes.
    pub prompt_tokens: Option<usize>,
}

/// What a key press means (classified pure of effects). The keys the app owns
/// (submit, cancel, scroll, quit, reasoning-fold) are named variants; every
/// other key is an [`Intent::Edit`] routed to the input editor, which carries
/// the full readline/emacs keymap (Ctrl+A/E/K/U/W, Alt+B/F/D, word/char motion,
/// insert-at-point, undo).
#[derive(Debug, PartialEq, Eq)]
pub enum Intent {
    None,
    Quit,
    Submit,
    /// Request cancellation of the in-flight turn (TUI-6).
    Cancel,
    /// A key for the input editor (anything the app does not own itself).
    Edit(Input),
    /// Move the editor cursor back/forward one word (Alt+←/→).
    WordBack,
    WordForward,
    /// Insert a literal newline in the prompt (Alt+Enter) — Enter submits, so a
    /// multi-line prompt is composed with this.
    Newline,
    /// ↑: recall the previous prompt, or move up a line in a multi-line draft.
    Up,
    /// ↓: recall the next prompt (or the live draft), or move down a line.
    Down,
    ScrollUp,
    ScrollDown,
    /// Expand/collapse the reasoning regions of completed turns (TUI-5).
    ToggleReasoning,
}

/// The UI render model.
pub struct App {
    pub req_tx: Sender<EngineRequest>,
    pub transcript: Vec<Entry>,
    /// The input editor — a `tui-textarea` widget owning the prompt buffer,
    /// cursor, and edit history. The app feeds it [`Intent::Edit`] keys and
    /// reads its text on submit; rendering draws it at the cursor (TUI-2 stays
    /// pure — render only borrows it).
    pub input: TextArea<'static>,
    /// Submitted prompts, oldest first — recalled with ↑/↓ (shell-style).
    pub history: Vec<String>,
    /// Position while browsing [`history`] with ↑/↓: `None` when editing the
    /// live draft, `Some(i)` when showing `history[i]`.
    history_browse: Option<usize>,
    /// The live input stashed when history browsing began, restored on ↓ past
    /// the newest entry.
    draft: Option<String>,
    /// Lines scrolled up from the bottom (0 = follow latest). Display is always
    /// clamped by [`scroll_y`] (TUI-1).
    pub scroll_back: usize,
    pub in_flight: Option<InFlight>,
    pub status: Status,
    /// Whether completed turns' reasoning is expanded (TUI-5). The in-flight
    /// turn always streams its reasoning live regardless.
    pub reasoning_expanded: bool,
    pub should_quit: bool,
    next_turn_id: TurnId,
}

impl App {
    pub fn new(req_tx: Sender<EngineRequest>, ready: Ready) -> App {
        App {
            req_tx,
            transcript: Vec::new(),
            input: fresh_input(),
            history: Vec::new(),
            history_browse: None,
            draft: None,
            scroll_back: 0,
            in_flight: None,
            status: Status {
                model_label: ready.model_label,
                backend: ready.backend,
                format: ready.format.to_string(),
                context_length: ready.context_length,
                prompt_tokens: None,
            },
            reasoning_expanded: false,
            should_quit: false,
            next_turn_id: 0,
        }
    }

    /// Classify a key press into an [`Intent`] (no effects — testable). The app
    /// owns a small set of keys; everything else becomes an [`Intent::Edit`] for
    /// the input editor, whose own keymap supplies emacs/readline editing.
    pub fn classify(key: KeyEvent) -> Intent {
        let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
        let alt = key.modifiers.contains(KeyModifiers::ALT);
        let shift = key.modifiers.contains(KeyModifiers::SHIFT);
        match key.code {
            KeyCode::Char('c') if ctrl => Intent::Quit,
            KeyCode::Char('d') if ctrl => Intent::Quit,
            KeyCode::Char('r') if ctrl => Intent::ToggleReasoning,
            KeyCode::Esc => Intent::Cancel,
            KeyCode::PageUp => Intent::ScrollUp,
            KeyCode::PageDown => Intent::ScrollDown,
            // Alt+Enter / Shift+Enter insert a newline (the latter needs a
            // terminal that reports modified Enter — see `enter_terminal`).
            KeyCode::Enter if alt || shift => Intent::Newline,
            KeyCode::Enter => Intent::Submit,
            // Alt+←/→ jump by word (the editor's own Ctrl+←/→ and Alt+B/F still
            // work; these add the arrow combo regardless of Option-as-Meta).
            KeyCode::Left if alt => Intent::WordBack,
            KeyCode::Right if alt => Intent::WordForward,
            // ↑/↓ recall prior prompts (shell-style); inside a multi-line draft
            // they move between lines first (handled in `apply`).
            KeyCode::Up => Intent::Up,
            KeyCode::Down => Intent::Down,
            _ => Intent::Edit(key.into()),
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
            Intent::Cancel => self.cancel_in_flight(),
            Intent::Edit(input) => {
                self.input.input(input);
            }
            Intent::WordBack => self.input.move_cursor(CursorMove::WordBack),
            Intent::WordForward => self.input.move_cursor(CursorMove::WordForward),
            Intent::Newline => self.input.insert_newline(),
            Intent::Up => {
                if self.input.cursor().0 == 0 {
                    self.history_prev();
                } else {
                    self.input.move_cursor(CursorMove::Up);
                }
            }
            Intent::Down => {
                let last_row = self.input.lines().len().saturating_sub(1);
                if self.input.cursor().0 >= last_row {
                    self.history_next();
                } else {
                    self.input.move_cursor(CursorMove::Down);
                }
            }
            Intent::ScrollUp => self.scroll_back = self.scroll_back.saturating_add(3),
            Intent::ScrollDown => self.scroll_back = self.scroll_back.saturating_sub(3),
            Intent::ToggleReasoning => self.reasoning_expanded = !self.reasoning_expanded,
        }
    }

    /// Begin a turn — unless input is empty or a turn is already in flight
    /// (TUI-7 single-in-flight). A leading-slash command (`/reset`) is handled
    /// here instead of submitting.
    fn start_turn(&mut self) {
        let user = self.input_text().trim().to_string();
        if user.is_empty() || self.in_flight.is_some() {
            return;
        }
        self.remember(&user);
        // `/reset` clears the conversation — the in-session recovery/escape hatch
        // (clears the engine's authoritative history and the UI mirror).
        if user == "/reset" {
            let _ = self.req_tx.send(EngineRequest::Reset);
            self.transcript.clear();
            self.input = fresh_input();
            self.scroll_back = 0;
            return;
        }
        let turn_id = self.next_turn_id;
        self.next_turn_id += 1;
        let control = Cancel::new();
        self.push_entry(Entry::User(user.clone()));
        self.in_flight = Some(InFlight {
            turn_id,
            started: Instant::now(),
            frags: 0,
            answering: false,
            cancelling: false,
            control: control.clone(),
        });
        let _ = self.req_tx.send(EngineRequest::Submit {
            turn_id,
            user,
            control,
        });
        self.input = fresh_input();
        self.scroll_back = 0; // jump to the latest
    }

    /// The current prompt text (the editor's lines joined). Enter submits before
    /// reaching the editor, so this is normally a single line.
    pub fn input_text(&self) -> String {
        self.input.lines().join("\n")
    }

    /// Replace the prompt text (cursor lands at the end).
    pub fn set_input(&mut self, text: &str) {
        let mut ta = fresh_input();
        ta.insert_str(text);
        self.input = ta;
    }

    /// Record a submitted prompt for ↑/↓ recall and reset the browse cursor.
    /// Consecutive duplicates are coalesced (a re-sent prompt isn't stored twice).
    fn remember(&mut self, prompt: &str) {
        if self.history.last().map(String::as_str) != Some(prompt) {
            self.history.push(prompt.to_string());
        }
        self.history_browse = None;
        self.draft = None;
    }

    /// Recall the previous (older) prompt into the editor. On first step the live
    /// draft is stashed so ↓ can restore it.
    fn history_prev(&mut self) {
        if self.history.is_empty() {
            return;
        }
        let target = match self.history_browse {
            None => {
                self.draft = Some(self.input_text());
                self.history.len() - 1
            }
            Some(0) => 0, // already at the oldest
            Some(i) => i - 1,
        };
        self.history_browse = Some(target);
        let text = self.history[target].clone();
        self.set_input(&text);
    }

    /// Recall the next (newer) prompt; stepping past the newest restores the
    /// stashed draft and leaves browsing.
    fn history_next(&mut self) {
        match self.history_browse {
            None => {}
            Some(i) if i + 1 < self.history.len() => {
                self.history_browse = Some(i + 1);
                let text = self.history[i + 1].clone();
                self.set_input(&text);
            }
            Some(_) => {
                self.history_browse = None;
                let draft = self.draft.take().unwrap_or_default();
                self.set_input(&draft);
            }
        }
    }

    /// Request cancellation of the in-flight turn (TUI-6): flip the shared
    /// control flag the decode loop polls. The turn stops at the next token
    /// boundary and arrives as a normal `Done` with `StopReason::Stopped`; until
    /// then the indicator shows "cancelling…". A no-op when nothing is in flight.
    fn cancel_in_flight(&mut self) {
        if let Some(f) = self.in_flight.as_mut() {
            f.control.cancel();
            f.cancelling = true;
        }
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
                answer: _, // the streamed Fragment channels are authoritative
                stop,
                prompt_tokens,
            } if self.is_current(turn_id) => {
                self.finish_assistant(stop);
                self.status.prompt_tokens = prompt_tokens;
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

    /// Record the stop reason. The answer is NOT overwritten here: the streamed
    /// Fragment channels (classified by the actor's splitter, seeded per format)
    /// are the single source of truth. `Done.answer` comes from the lib's
    /// *non-seeded* batch split, which disagrees for a pre-seeded model that
    /// emitted no `</think>` (it would mislabel the whole reasoning as the
    /// answer, duplicating it — the "no boundary" bug).
    fn finish_assistant(&mut self, stop: StopReason) {
        if let Some(Entry::Assistant { stop: s, .. }) = self.transcript.last_mut() {
            *s = Some(stop);
        }
    }
}

/// A blank input editor configured for the prompt box: a "message" placeholder
/// and no whole-line cursor highlight (the cursor itself marks the point).
fn fresh_input() -> TextArea<'static> {
    let mut ta = TextArea::default();
    ta.set_placeholder_text("message");
    ta.set_cursor_line_style(Style::default());
    ta
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
    // ratatui 0.30 made `Backend::Error` an associated type; `?` into `anyhow`
    // needs it to be a `Send + Sync + 'static` std error.
    B::Error: std::error::Error + Send + Sync + 'static,
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
            context_length: Some(32768),
            model_label: "test-model".into(),
        };
        (App::new(tx, ready), rx)
    }

    fn key(code: KeyCode) -> KeyEvent {
        KeyEvent::new(code, KeyModifiers::NONE)
    }

    #[test]
    fn ctrl_r_toggles_reasoning_and_done_records_prompt_tokens() {
        // upholds: TUI-5 — Ctrl+R flips the reasoning fold; Done carries the
        // prompt-token count for the context meter.
        let (mut app, _rx) = test_app();
        assert_eq!(
            App::classify(KeyEvent::new(KeyCode::Char('r'), KeyModifiers::CONTROL)),
            Intent::ToggleReasoning
        );
        assert!(!app.reasoning_expanded);
        app.apply(Intent::ToggleReasoning);
        assert!(app.reasoning_expanded);
        app.apply(Intent::ToggleReasoning);
        assert!(!app.reasoning_expanded);

        app.set_input("hi");
        app.apply(Intent::Submit);
        app.on_engine_event(EngineEvent::Started { turn_id: 0 });
        app.on_engine_event(EngineEvent::Done {
            turn_id: 0,
            answer: "ok".into(),
            stop: StopReason::Eos,
            prompt_tokens: Some(2048),
        });
        assert_eq!(app.status.prompt_tokens, Some(2048));
    }

    #[test]
    fn esc_cancels_the_in_flight_turn() {
        // upholds: TUI-6 — Esc flips the shared control flag the decode loop polls
        // and marks the turn "cancelling"; a no-op when nothing is in flight.
        assert_eq!(App::classify(key(KeyCode::Esc)), Intent::Cancel);
        let (mut app, _rx) = test_app();
        app.apply(Intent::Cancel); // nothing in flight: harmless
        assert!(app.in_flight.is_none());

        app.set_input("hi");
        app.apply(Intent::Submit);
        let control = app.in_flight.as_ref().unwrap().control.clone();
        assert!(!control.is_cancelled());
        app.apply(Intent::Cancel);
        assert!(control.is_cancelled(), "Esc must flip the control flag");
        assert!(
            app.in_flight.as_ref().unwrap().cancelling,
            "the turn is marked cancelling for the indicator"
        );
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
        app.set_input("hi");
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
            prompt_tokens: Some(123),
        });
        assert_eq!(app.transcript.len(), 2, "fragments/done must not append");
    }

    #[test]
    fn done_does_not_overwrite_the_streamed_answer() {
        // Regression for the "no boundary" bug: a pre-seeded reasoning model that
        // emits no </think> streams all Reasoning (answer stays empty); Done's
        // non-seeded batch answer must NOT clobber the empty streamed answer
        // (else reasoning and answer show the same text).
        let (mut app, _rx) = test_app();
        app.set_input("hi");
        app.apply(Intent::Submit);
        app.on_engine_event(EngineEvent::Started { turn_id: 0 });
        app.on_engine_event(EngineEvent::Fragment {
            turn_id: 0,
            channel: Channel::Reasoning,
            text: "deep thoughts".into(),
        });
        app.on_engine_event(EngineEvent::Done {
            turn_id: 0,
            answer: "deep thoughts".into(), // the batch-split (mis)answer
            stop: StopReason::MaxTokens,
            prompt_tokens: Some(10),
        });
        let Entry::Assistant {
            reasoning, answer, ..
        } = &app.transcript[1]
        else {
            panic!("expected an assistant entry");
        };
        assert_eq!(reasoning, "deep thoughts");
        assert_eq!(
            answer, "",
            "Done.answer must not overwrite the streamed answer"
        );
    }

    #[test]
    fn slash_reset_clears_mirror_and_signals_engine() {
        // /reset is the in-session recovery: it clears the UI mirror and tells
        // the engine to reset its authoritative history — no turn is submitted.
        let (mut app, rx) = test_app();
        app.set_input("hi");
        app.apply(Intent::Submit);
        app.on_engine_event(EngineEvent::Started { turn_id: 0 });
        app.on_engine_event(EngineEvent::Done {
            turn_id: 0,
            answer: "ok".into(),
            stop: StopReason::Eos,
            prompt_tokens: Some(123),
        });
        assert!(!app.transcript.is_empty());
        let _ = rx.try_recv(); // drain the Submit

        app.set_input("/reset");
        app.apply(Intent::Submit);
        assert!(app.transcript.is_empty(), "reset clears the mirror");
        assert!(app.in_flight.is_none(), "reset does not start a turn");
        assert!(matches!(rx.try_recv(), Ok(EngineRequest::Reset)));
    }

    #[test]
    fn submit_is_blocked_while_in_flight() {
        // upholds: TUI-7 — a new prompt cannot start while a turn is active.
        let (mut app, rx) = test_app();
        app.set_input("first");
        app.apply(Intent::Submit);
        assert!(app.in_flight.is_some());
        assert!(matches!(rx.try_recv(), Ok(EngineRequest::Submit { .. })));

        // A second submit while in flight is a no-op: no new request, input kept.
        app.set_input("second");
        app.apply(Intent::Submit);
        assert_eq!(app.input_text(), "second");
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
            cancelling: false,
            control: Cancel::new(),
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
        assert_eq!(App::classify(key(KeyCode::PageUp)), Intent::ScrollUp);
        // An unowned key routes to the editor as an Edit intent.
        assert_eq!(
            App::classify(key(KeyCode::Char('a'))),
            Intent::Edit(KeyEvent::from(KeyCode::Char('a')).into())
        );
    }

    #[test]
    fn editor_keys_edit_the_prompt_buffer() {
        // The readline keymap lives in the editor: type, then Ctrl+A (start) and
        // Ctrl+K (kill to end) clear the line — proof the keys reach it.
        let (mut app, _rx) = test_app();
        for c in "hello".chars() {
            app.on_key(key(KeyCode::Char(c)));
        }
        assert_eq!(app.input_text(), "hello");
        app.on_key(KeyEvent::new(KeyCode::Char('a'), KeyModifiers::CONTROL));
        app.on_key(KeyEvent::new(KeyCode::Char('k'), KeyModifiers::CONTROL));
        assert_eq!(app.input_text(), "", "Ctrl+A then Ctrl+K kills the line");
    }

    #[test]
    fn alt_enter_inserts_a_newline_without_submitting() {
        // Enter submits; Alt+Enter composes a multi-line prompt instead.
        assert_eq!(
            App::classify(KeyEvent::new(KeyCode::Enter, KeyModifiers::ALT)),
            Intent::Newline
        );
        assert_eq!(
            App::classify(KeyEvent::new(KeyCode::Enter, KeyModifiers::SHIFT)),
            Intent::Newline
        );
        let (mut app, rx) = test_app();
        for c in "line one".chars() {
            app.on_key(key(KeyCode::Char(c)));
        }
        app.on_key(KeyEvent::new(KeyCode::Enter, KeyModifiers::ALT));
        for c in "line two".chars() {
            app.on_key(key(KeyCode::Char(c)));
        }
        assert_eq!(app.input_text(), "line one\nline two");
        assert!(app.in_flight.is_none(), "Alt+Enter must not submit");
        assert!(rx.try_recv().is_err(), "no turn submitted");
    }

    #[test]
    fn up_down_recall_prior_prompts() {
        // ↑/↓ browse submitted prompts; ↓ past the newest restores the draft.
        assert_eq!(App::classify(key(KeyCode::Up)), Intent::Up);
        assert_eq!(App::classify(key(KeyCode::Down)), Intent::Down);
        let (mut app, _rx) = test_app();
        let done = |id| EngineEvent::Done {
            turn_id: id,
            answer: "ok".into(),
            stop: StopReason::Eos,
            prompt_tokens: None,
        };
        app.set_input("first");
        app.apply(Intent::Submit);
        app.on_engine_event(done(0)); // clear in-flight (TUI-7)
        app.set_input("second");
        app.apply(Intent::Submit);
        app.on_engine_event(done(1));
        // A half-typed draft, then browse back through history.
        app.set_input("draf");
        app.apply(Intent::Up);
        assert_eq!(app.input_text(), "second");
        app.apply(Intent::Up);
        assert_eq!(app.input_text(), "first");
        app.apply(Intent::Up); // already oldest — stays
        assert_eq!(app.input_text(), "first");
        app.apply(Intent::Down);
        assert_eq!(app.input_text(), "second");
        app.apply(Intent::Down); // past the newest — restore the draft
        assert_eq!(app.input_text(), "draf");
    }

    #[test]
    fn alt_arrows_move_by_word() {
        // Alt+← / Alt+→ jump word boundaries (the keys the user named), landing
        // the cursor at the start of the previous word.
        assert_eq!(
            App::classify(KeyEvent::new(KeyCode::Left, KeyModifiers::ALT)),
            Intent::WordBack
        );
        let (mut app, _rx) = test_app();
        app.set_input("alpha beta gamma"); // cursor is at end after set
        app.input.move_cursor(CursorMove::End);
        app.apply(Intent::WordBack);
        // Now at the start of "gamma" (col 11); typing inserts there.
        app.on_key(key(KeyCode::Char('!')));
        assert_eq!(app.input_text(), "alpha beta !gamma");
    }
}
