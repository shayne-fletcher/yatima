//! UI state (a render *mirror*), input handling, and the async event loop.
//!
//! The host thread owns the authoritative session; [`App`] holds only a mirror
//! rebuilt from [`HostEvent`]s, plus input/scroll/status. State changes flow
//! through one of two places: a key [`Intent`] (`apply`) or a host event
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
use yatima_host::{
    CancelGate, Channel, HostEvent, HostRequest, ModelInfo, StopKind, ToolNoteKind, TurnId,
};

use crate::render;

/// One rendered transcript entry (the mirror; the actor's session is truth).
pub enum Entry {
    User(String),
    Assistant {
        reasoning: String,
        answer: String,
        stop: Option<StopKind>,
    },
    Error(String),
    /// A host notice (grant/revoke confirmations and the like) — visible
    /// authority changes belong in the transcript (CAP-3).
    Notice(String),
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
    /// The granted web origins (CAP-3) — the session's live web authority.
    pub grants: Vec<String>,
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
    /// Ctrl+G: compose the prompt in `$VISUAL`/`$EDITOR` (the TUI suspends,
    /// the editor gets the draft, the result lands back in the input box —
    /// reviewed, never auto-submitted).
    Compose,
    ScrollUp,
    ScrollDown,
    /// Expand/collapse the reasoning regions of completed turns (TUI-5).
    ToggleReasoning,
}

/// The UI render model.
pub struct App {
    pub req_tx: Sender<HostRequest>,
    /// The host's cancel gate: Esc trips it for the in-flight turn (TUI-6).
    pub cancel: CancelGate,
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
    /// Set by Ctrl+G; consumed by the run loop, which owns the terminal
    /// choreography the compose needs.
    compose_requested: bool,
    pub should_quit: bool,
    /// A first Ctrl+C/Ctrl+D armed the quit; the next confirms, any other key
    /// stands down. Render shows the confirm hint while armed.
    pub quit_armed: bool,
    next_turn_id: TurnId,
}

impl App {
    pub fn new(req_tx: Sender<HostRequest>, cancel: CancelGate, ready: ModelInfo) -> App {
        App {
            req_tx,
            cancel,
            transcript: Vec::new(),
            input: fresh_input(),
            history: Vec::new(),
            history_browse: None,
            draft: None,
            scroll_back: 0,
            in_flight: None,
            status: Status {
                model_label: ready.label,
                backend: ready.backend,
                format: ready.format,
                context_length: ready.context_length,
                prompt_tokens: None,
                grants: Vec::new(),
            },
            reasoning_expanded: false,
            compose_requested: false,
            should_quit: false,
            quit_armed: false,
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
            KeyCode::Char('g') if ctrl => Intent::Compose,
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
        // Quitting takes a confirmation: the first Ctrl+C/Ctrl+D arms it (the
        // input box shows the hint), the next confirms, and any other key
        // stands down — a stray reflex never tears the session down.
        if !matches!(intent, Intent::Quit) {
            self.quit_armed = false;
        }
        match intent {
            Intent::None => {}
            Intent::Quit => {
                if self.quit_armed {
                    self.should_quit = true;
                } else {
                    self.quit_armed = true;
                }
            }
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
            Intent::Compose => self.compose_requested = true,
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
        // (clears the engine's authoritative history and the UI mirror). Granted
        // origins survive: capability state is not conversation state (CAP-3).
        if user == "/reset" {
            let _ = self.req_tx.send(HostRequest::Reset);
            self.transcript.clear();
            self.input = fresh_input();
            self.scroll_back = 0;
            return;
        }
        // Grant management (CAP-3: these, plus URLs typed in a message, are
        // the *only* sources of web authority).
        if user == "/grants" {
            let _ = self.req_tx.send(HostRequest::ListGrants);
            self.input = fresh_input();
            return;
        }
        if let Some(origin) = user.strip_prefix("/grant ") {
            let _ = self.req_tx.send(HostRequest::Grant {
                origin: origin.trim().to_string(),
            });
            self.input = fresh_input();
            return;
        }
        if let Some(origin) = user.strip_prefix("/revoke ") {
            let _ = self.req_tx.send(HostRequest::Revoke {
                origin: origin.trim().to_string(),
            });
            self.input = fresh_input();
            return;
        }
        // Auto-grant: a URL in the *user's own message* is authorization for
        // its origin (CAP-3) — granted before the turn runs, so the model can
        // act on it immediately. URLs from any other source never pass
        // through here.
        for origin in yatima_lib::origins_in(&user) {
            let _ = self.req_tx.send(HostRequest::Grant { origin });
        }
        let turn_id = self.next_turn_id;
        self.next_turn_id += 1;
        self.push_entry(Entry::User(user.clone()));
        self.in_flight = Some(InFlight {
            turn_id,
            started: Instant::now(),
            frags: 0,
            answering: false,
            cancelling: false,
        });
        let _ = self.req_tx.send(HostRequest::Submit {
            turn_id,
            text: user,
        });
        self.input = fresh_input();
        self.scroll_back = 0; // jump to the latest
    }

    /// The current prompt text (the editor's lines joined). Enter submits before
    /// reaching the editor, so this is normally a single line.
    pub fn input_text(&self) -> String {
        self.input.lines().join("\n")
    }

    /// Consume a pending Ctrl+G compose request (run-loop side).
    pub fn take_compose_request(&mut self) -> bool {
        std::mem::take(&mut self.compose_requested)
    }

    /// Surface an app-plane notice in the transcript (compose failures land
    /// here — audible, never fatal).
    pub fn notice(&mut self, message: String) {
        self.push_entry(Entry::Notice(message));
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

    /// Request cancellation of the in-flight turn (TUI-6): trip the host's
    /// cancel gate for this turn, which flips the shared flag the decode loop
    /// polls. The turn stops at the next token boundary and arrives as a normal
    /// `Done` with `StopKind::Stopped`; until then the indicator shows
    /// "cancelling…". A no-op when nothing is in flight.
    fn cancel_in_flight(&mut self) {
        if let Some(f) = self.in_flight.as_mut() {
            self.cancel.cancel(f.turn_id);
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

    /// Fold a host event into the render mirror (the only event entry point).
    pub fn on_engine_event(&mut self, event: HostEvent) {
        match event {
            HostEvent::Started { turn_id } if self.is_current(turn_id) => {
                self.push_entry(Entry::Assistant {
                    reasoning: String::new(),
                    answer: String::new(),
                    stop: None,
                });
            }
            HostEvent::Fragment {
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
            // Tool activity folds into the reasoning pane under the terminal's
            // marker vocabulary (HOST-4: the wire carries kind + payload; the
            // glyphs are this view's). A successful artifact write carries the
            // `wrote <path>` contract, which opens in the platform viewer.
            HostEvent::ToolNote {
                turn_id,
                kind,
                text,
            } if self.is_current(turn_id) => {
                self.append_fragment(Channel::Reasoning, &tool_note_line(kind, &text));
                if kind == ToolNoteKind::Success {
                    open_artifact(&text);
                }
            }
            // The most recent prompt's token count (the meter numerator); a
            // status fact, set outside the turn guard (single-in-flight, TUI-7).
            HostEvent::Context { prompt_tokens } => {
                self.status.prompt_tokens = Some(prompt_tokens);
            }
            HostEvent::Done { turn_id, stop } if self.is_current(turn_id) => {
                self.finish_assistant(stop);
                self.in_flight = None;
            }
            HostEvent::Error { turn_id, message } if self.is_current(turn_id) => {
                self.push_entry(Entry::Error(message));
                self.in_flight = None;
            }
            // A step that became a tool call retracts its streamed narration
            // from the answer (the host replays it as reasoning) — AGENT-4.
            HostEvent::RetractAnswer { turn_id, chars } if self.is_current(turn_id) => {
                if let Some(Entry::Assistant { answer, .. }) = self.transcript.last_mut() {
                    let keep = answer.chars().count().saturating_sub(chars);
                    *answer = answer.chars().take(keep).collect();
                }
            }
            // Authority changes are always current (not tied to a turn): show
            // the notice and refresh the status rail's grant list (CAP-3).
            HostEvent::Grants { origins, message } => {
                self.status.grants = origins;
                self.push_entry(Entry::Notice(message));
            }
            // The host reads artifact bytes and ships an Image for a texturing
            // frontend; the terminal opens the file via the ToolNote path
            // instead, so these bytes go unused here.
            HostEvent::Image { .. } => {}
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

    /// Record the stop reason on the completed assistant entry. The answer is
    /// never overwritten from the host: the streamed Fragment channels
    /// (classified by the host's splitter, seeded per format) are the single
    /// source of truth — `Done` carries only why the turn stopped.
    fn finish_assistant(&mut self, stop: StopKind) {
        if let Some(Entry::Assistant { stop: s, .. }) = self.transcript.last_mut() {
            *s = Some(stop);
        }
    }
}

/// Render a tool-note payload as a line in the terminal's marker vocabulary
/// (HOST-4: the wire carries `(kind, payload)`; glyphs and indentation are the
/// view's). A kind this build doesn't know renders unmarked — the protocol
/// enum is `#[non_exhaustive]`, and the payload alone is still legible.
fn tool_note_line(kind: ToolNoteKind, text: &str) -> String {
    match kind {
        ToolNoteKind::Call => format!("\n⚙ {text}\n"),
        ToolNoteKind::Success => format!("  ✓ {text}\n"),
        ToolNoteKind::Failure => format!("  ✗ {text}\n"),
        ToolNoteKind::Warning => format!("\n⚠ {text}\n"),
        // Progress, and any kind newer than this build, renders unmarked.
        _ => format!("  {text}\n"),
    }
}

/// Open a just-written image artifact (a plot render, a fetched image) in the
/// platform viewer (macOS `open`), parsed from a successful tool outcome's
/// payload carrying the `wrote <path> (…)` contract (PLOT-2 / IMG-1).
/// Fire-and-forget: viewing is a courtesy, never an error — failures are
/// ignored and a reaper thread waits the child so no zombies accrue. This only
/// ever fires for an artifact the user just asked for.
fn open_artifact(note: &str) {
    #[cfg(target_os = "macos")]
    if let Some((path, _)) = note
        .split_once("wrote ")
        .and_then(|(_, rest)| rest.rsplit_once(" ("))
    {
        if let Ok(mut child) = std::process::Command::new("open").arg(path).spawn() {
            std::thread::spawn(move || {
                let _ = child.wait();
            });
        }
    }
    #[cfg(not(target_os = "macos"))]
    let _ = note;
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
    mut event_rx: UnboundedReceiver<HostEvent>,
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
        if app.take_compose_request() {
            // Ctrl+G: hand the draft to $VISUAL/$EDITOR. The engine keeps
            // running (events buffer in the channel and land on return);
            // the terminal is suspended for exactly the editor's lifetime.
            match tokio::task::block_in_place(|| compose_in_editor(&app.input_text())) {
                Ok(Some(text)) => app.set_input(&text),
                Ok(None) => {} // unchanged or emptied: the draft stands
                Err(e) => app.notice(format!("compose: {e}")),
            }
            // External writes trashed the alternate screen; repaint from
            // scratch on the next draw.
            terminal.clear()?;
        }
        if app.should_quit {
            break;
        }
    }
    let _ = app.req_tx.send(HostRequest::Shutdown);
    Ok(())
}

/// Suspend the TUI, run `$VISUAL`/`$EDITOR` on a temp file seeded with the
/// draft, and return the edited text — `None` when the user left it
/// unchanged or emptied it (their draft stands either way). The terminal is
/// restored before any error propagates, whatever the editor did.
fn compose_in_editor(draft: &str) -> Result<Option<String>> {
    use crossterm::event::{
        KeyboardEnhancementFlags, PopKeyboardEnhancementFlags, PushKeyboardEnhancementFlags,
    };
    use crossterm::terminal::{
        disable_raw_mode, enable_raw_mode, supports_keyboard_enhancement, EnterAlternateScreen,
        LeaveAlternateScreen,
    };

    let editor = std::env::var("VISUAL")
        .or_else(|_| std::env::var("EDITOR"))
        .map_err(|_| anyhow::anyhow!("set $VISUAL or $EDITOR"))?;
    let mut words = editor.split_whitespace();
    let program = words
        .next()
        .ok_or_else(|| anyhow::anyhow!("$VISUAL/$EDITOR is empty"))?
        .to_string();
    let args: Vec<String> = words.map(str::to_string).collect();

    let path = std::env::temp_dir().join(format!("yatima-compose-{}.md", std::process::id()));
    std::fs::write(&path, draft)?;

    // Suspend: mirror enter_terminal/restore_terminal in main.rs (the
    // enhancement query answers the same for the same terminal).
    let enhanced = supports_keyboard_enhancement().unwrap_or(false);
    if enhanced {
        crossterm::execute!(io::stdout(), PopKeyboardEnhancementFlags)?;
    }
    crossterm::execute!(io::stdout(), LeaveAlternateScreen)?;
    disable_raw_mode()?;

    let status = std::process::Command::new(&program)
        .args(&args)
        .arg(&path)
        .status();

    // Resume unconditionally — the terminal must come back even if the
    // editor never started.
    enable_raw_mode()?;
    crossterm::execute!(io::stdout(), EnterAlternateScreen)?;
    if enhanced {
        crossterm::execute!(
            io::stdout(),
            PushKeyboardEnhancementFlags(KeyboardEnhancementFlags::DISAMBIGUATE_ESCAPE_CODES)
        )?;
    }

    let status = status.map_err(|e| anyhow::anyhow!("could not run {program}: {e}"))?;
    if !status.success() {
        anyhow::bail!("{program} exited with {status}; draft kept");
    }
    let text = std::fs::read_to_string(&path)?;
    let _ = std::fs::remove_file(&path);
    let text = text.trim_end();
    Ok((!text.is_empty() && text != draft).then(|| text.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ratatui::backend::TestBackend;
    use std::time::Duration;
    use tokio::sync::mpsc::unbounded_channel;
    use yatima_lib::Cancel;

    fn test_app() -> (App, std::sync::mpsc::Receiver<HostRequest>) {
        let (tx, rx) = std::sync::mpsc::channel();
        let ready = ModelInfo {
            label: "test-model".into(),
            arch: "Qwen2".into(),
            backend: "test".into(),
            device: "cpu".into(),
            format: "Qwen".into(),
            sampling: "greedy".into(),
            max_tokens: 1024,
            context_length: Some(32768),
        };
        (App::new(tx, CancelGate::new(), ready), rx)
    }

    fn key(code: KeyCode) -> KeyEvent {
        KeyEvent::new(code, KeyModifiers::NONE)
    }

    #[test]
    fn ctrl_g_requests_compose_once() {
        // Ctrl+G classifies to Compose and raises the flag; the run loop
        // consumes it exactly once (classify stays pure, apply is the only
        // writer).
        let key = KeyEvent::new(KeyCode::Char('g'), KeyModifiers::CONTROL);
        assert!(matches!(App::classify(key), Intent::Compose));
        let (mut app, _rx) = test_app();
        app.apply(Intent::Compose);
        assert!(app.take_compose_request());
        assert!(!app.take_compose_request(), "consumed, not sticky");
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
        app.on_engine_event(HostEvent::Started { turn_id: 0 });
        app.on_engine_event(HostEvent::Context {
            prompt_tokens: 2048,
        });
        app.on_engine_event(HostEvent::Done {
            turn_id: 0,
            stop: StopKind::Eos,
        });
        assert_eq!(app.status.prompt_tokens, Some(2048));
    }

    #[test]
    fn quit_takes_two_presses() {
        // A stray Ctrl+C/Ctrl+D must not tear the session down: the first
        // press arms (render shows the confirm hint), the second confirms,
        // and any other key stands down.
        let (mut app, _rx) = test_app();
        app.apply(Intent::Quit);
        assert!(!app.should_quit, "one press must not quit");
        assert!(app.quit_armed, "the first press arms");
        app.apply(Intent::Quit);
        assert!(app.should_quit, "the second press confirms");

        let (mut app, _rx) = test_app();
        app.apply(Intent::Quit);
        app.apply(Intent::ToggleReasoning);
        assert!(!app.quit_armed, "any other key stands down");
        app.apply(Intent::Quit);
        assert!(
            !app.should_quit,
            "after standing down, a press only re-arms"
        );
    }

    #[test]
    fn esc_cancels_the_in_flight_turn() {
        // upholds: TUI-6 — Esc trips the host's cancel gate for the in-flight
        // turn and marks it "cancelling"; a no-op when nothing is in flight.
        assert_eq!(App::classify(key(KeyCode::Esc)), Intent::Cancel);
        let (mut app, _rx) = test_app();
        app.apply(Intent::Cancel); // nothing in flight: harmless
        assert!(app.in_flight.is_none());

        app.set_input("hi");
        app.apply(Intent::Submit);
        // The host arms the gate with the turn's cancel before it decodes;
        // simulate that here, then check Esc flips it.
        let turn_id = app.in_flight.as_ref().unwrap().turn_id;
        let cancel = Cancel::new();
        app.cancel.arm(turn_id, cancel.clone());
        assert!(!cancel.is_cancelled());
        app.apply(Intent::Cancel);
        assert!(
            cancel.is_cancelled(),
            "Esc must trip the gate for the in-flight turn"
        );
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
    fn tool_notes_render_in_the_terminal_vocabulary() {
        // upholds: HOST-4 — the wire carries (kind, payload); the ✓/✗/⚙/⚠
        // markers and the fold's indentation are this view's own, and the
        // rendered line folds into the reasoning pane.
        assert_eq!(
            tool_note_line(ToolNoteKind::Call, "plot {…}"),
            "\n⚙ plot {…}\n"
        );
        assert_eq!(
            tool_note_line(ToolNoteKind::Progress, "fetching"),
            "  fetching\n"
        );
        assert_eq!(
            tool_note_line(ToolNoteKind::Success, "142 chars"),
            "  ✓ 142 chars\n"
        );
        assert_eq!(tool_note_line(ToolNoteKind::Failure, "boom"), "  ✗ boom\n");
        assert_eq!(
            tool_note_line(ToolNoteKind::Warning, "tool-step budget exhausted (6)"),
            "\n⚠ tool-step budget exhausted (6)\n"
        );

        let (mut app, _rx) = test_app();
        app.set_input("hi");
        app.apply(Intent::Submit);
        app.on_engine_event(HostEvent::Started { turn_id: 0 });
        app.on_engine_event(HostEvent::ToolNote {
            turn_id: 0,
            kind: ToolNoteKind::Failure,
            text: "boom".into(),
        });
        let Some(Entry::Assistant { reasoning, .. }) = app.transcript.last() else {
            panic!("expected an assistant entry");
        };
        assert_eq!(reasoning, "  ✗ boom\n");
    }

    #[test]
    fn transcript_grows_only_via_push_entry() {
        // upholds: TUI-3 — fragments mutate the last entry, never append. A turn
        // adds exactly two entries (User on submit, Assistant on Started).
        let (mut app, _rx) = test_app();
        app.set_input("hi");
        app.apply(Intent::Submit);
        assert_eq!(app.transcript.len(), 1); // User
        app.on_engine_event(HostEvent::Started { turn_id: 0 });
        assert_eq!(app.transcript.len(), 2); // + Assistant
        for _ in 0..5 {
            app.on_engine_event(HostEvent::Fragment {
                turn_id: 0,
                channel: Channel::Answer,
                text: "x".into(),
            });
        }
        app.on_engine_event(HostEvent::Done {
            turn_id: 0,
            stop: StopKind::Eos,
        });
        assert_eq!(app.transcript.len(), 2, "fragments/done must not append");
    }

    #[test]
    fn streamed_channels_are_the_answer_of_record() {
        // A pre-seeded reasoning model that emits no </think> streams all
        // Reasoning (the answer stays empty); the streamed Fragment channels are
        // authoritative and Done — which carries no answer — leaves them intact
        // (the old "no boundary" bug was Done clobbering the empty answer).
        let (mut app, _rx) = test_app();
        app.set_input("hi");
        app.apply(Intent::Submit);
        app.on_engine_event(HostEvent::Started { turn_id: 0 });
        app.on_engine_event(HostEvent::Fragment {
            turn_id: 0,
            channel: Channel::Reasoning,
            text: "deep thoughts".into(),
        });
        app.on_engine_event(HostEvent::Done {
            turn_id: 0,
            stop: StopKind::MaxTokens,
        });
        let Entry::Assistant {
            reasoning, answer, ..
        } = &app.transcript[1]
        else {
            panic!("expected an assistant entry");
        };
        assert_eq!(reasoning, "deep thoughts");
        assert_eq!(answer, "", "the streamed (empty) answer stands");
    }

    #[test]
    fn slash_reset_clears_mirror_and_signals_engine() {
        // /reset is the in-session recovery: it clears the UI mirror and tells
        // the engine to reset its authoritative history — no turn is submitted.
        let (mut app, rx) = test_app();
        app.set_input("hi");
        app.apply(Intent::Submit);
        app.on_engine_event(HostEvent::Started { turn_id: 0 });
        app.on_engine_event(HostEvent::Done {
            turn_id: 0,
            stop: StopKind::Eos,
        });
        assert!(!app.transcript.is_empty());
        let _ = rx.try_recv(); // drain the Submit

        app.set_input("/reset");
        app.apply(Intent::Submit);
        assert!(app.transcript.is_empty(), "reset clears the mirror");
        assert!(app.in_flight.is_none(), "reset does not start a turn");
        assert!(matches!(rx.try_recv(), Ok(HostRequest::Reset)));
    }

    #[test]
    fn submit_is_blocked_while_in_flight() {
        // upholds: TUI-7 — a new prompt cannot start while a turn is active.
        let (mut app, rx) = test_app();
        app.set_input("first");
        app.apply(Intent::Submit);
        assert!(app.in_flight.is_some());
        assert!(matches!(rx.try_recv(), Ok(HostRequest::Submit { .. })));

        // A second submit while in flight is a no-op: no new request, input kept.
        app.set_input("second");
        app.apply(Intent::Submit);
        assert_eq!(app.input_text(), "second");
        assert!(rx.try_recv().is_err(), "no second Submit while in flight");
    }

    #[test]
    fn user_typed_urls_auto_grant_before_the_turn() {
        // upholds: CAP-3 — a URL in the user's own message grants its origin,
        // and the Grant request precedes the Submit so the model can act on
        // it immediately. Duplicate origins collapse.
        let (mut app, rx) = test_app();
        app.set_input(
            "compare https://en.wikipedia.org/wiki/A and https://en.wikipedia.org/wiki/B",
        );
        app.apply(Intent::Submit);
        match rx.try_recv() {
            Ok(HostRequest::Grant { origin }) => {
                assert_eq!(origin, "https://en.wikipedia.org")
            }
            other => panic!("expected Grant first, got {:?}", other.is_ok()),
        }
        assert!(
            matches!(rx.try_recv(), Ok(HostRequest::Submit { .. })),
            "Submit follows the grant"
        );
        assert!(rx.try_recv().is_err(), "one grant per distinct origin");
    }

    #[test]
    fn grant_commands_manage_authority_without_starting_turns() {
        // upholds: CAP-3 — /grant, /grants, and /revoke are user utterances
        // that manage authority; none of them submits a turn.
        let (mut app, rx) = test_app();
        for input in [
            "/grant https://a.example",
            "/grants",
            "/revoke https://a.example",
        ] {
            app.set_input(input);
            app.apply(Intent::Submit);
            assert!(app.in_flight.is_none(), "{input}: no turn starts");
        }
        assert!(matches!(
            rx.try_recv(),
            Ok(HostRequest::Grant { origin }) if origin == "https://a.example"
        ));
        assert!(matches!(rx.try_recv(), Ok(HostRequest::ListGrants)));
        assert!(matches!(
            rx.try_recv(),
            Ok(HostRequest::Revoke { origin }) if origin == "https://a.example"
        ));
    }

    #[test]
    fn grants_events_update_status_and_leave_a_notice() {
        // upholds: CAP-3 — authority changes are visible: the status rail
        // mirrors the granted set and the transcript records the change. (The
        // message is host-generated — HOST-2 — so this fixture uses a synthetic
        // one; the wording itself is tested in yatima-host.)
        let (mut app, _rx) = test_app();
        app.on_engine_event(HostEvent::Grants {
            origins: vec!["https://a.example".into()],
            message: "<a grant notice from the host>".into(),
        });
        assert_eq!(app.status.grants, ["https://a.example"]);
        assert!(matches!(
            app.transcript.last(),
            Some(Entry::Notice(text)) if text.contains("grant notice")
        ));
    }

    #[test]
    fn retract_answer_pulls_narration_out_of_the_live_entry() {
        // upholds: AGENT-4 — when a streamed step turns out to be a tool
        // call, RetractAnswer removes exactly its narration chars from the
        // live answer (the actor replays them as reasoning).
        let (mut app, _rx) = test_app();
        app.set_input("go");
        app.apply(Intent::Submit);
        app.on_engine_event(HostEvent::Started { turn_id: 0 });
        for text in ["Let me ", "fetch that."] {
            app.on_engine_event(HostEvent::Fragment {
                turn_id: 0,
                channel: Channel::Answer,
                text: text.into(),
            });
        }
        app.on_engine_event(HostEvent::RetractAnswer {
            turn_id: 0,
            chars: "Let me fetch that.".chars().count(),
        });
        let Some(Entry::Assistant { answer, .. }) = app.transcript.last() else {
            panic!("expected a live assistant entry");
        };
        assert!(answer.is_empty(), "narration retracted, got {answer:?}");
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
        });

        let (event_tx, event_rx) = unbounded_channel();
        std::thread::spawn(move || {
            for i in 0..5 {
                std::thread::sleep(Duration::from_millis(100));
                if event_tx
                    .send(HostEvent::Fragment {
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
        let done = |id| HostEvent::Done {
            turn_id: id,
            stop: StopKind::Eos,
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
