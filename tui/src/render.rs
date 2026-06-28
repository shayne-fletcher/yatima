//! Pure rendering: `ui(frame, &App)` mutates nothing (TUI-2). The transcript,
//! input box, and status bar are drawn as a projection of [`App`] state.

use std::time::Duration;

use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Paragraph, Wrap};
use ratatui::Frame;

use yatima_lib::StopReason;

use crate::app::{scroll_y, App, Entry};

/// The animated glyph for an activity phase. Reasoning gets a HAL-ish red
/// breathing **eye**; answering gets a green **pulse**. The frame advances on the
/// redraw tick (from elapsed time), so it animates even when the model stalls.
#[derive(Clone, Copy, PartialEq, Eq)]
enum ActivitySkin {
    Eye,
    Pulse,
}

const EYE: [&str; 6] = ["◌", "◎", "◉", "●", "◉", "◎"];
const PULSE: [&str; 4] = ["·", "•", "●", "•"];

impl ActivitySkin {
    fn frames(self) -> &'static [&'static str] {
        match self {
            ActivitySkin::Eye => &EYE,
            ActivitySkin::Pulse => &PULSE,
        }
    }

    fn color(self) -> Color {
        match self {
            ActivitySkin::Eye => Color::Red,
            ActivitySkin::Pulse => Color::Green,
        }
    }

    /// The glyph for this moment, plus a style that *breathes*: bright + bold at
    /// the open frames (`●`/`◉`/`•`), dim at the closed ones — so the eye pulses
    /// in brightness, not just shape.
    fn glyph(self, elapsed: Duration) -> (&'static str, Style) {
        let frames = self.frames();
        let glyph = frames[(elapsed.as_millis() / 180) as usize % frames.len()];
        let open = matches!(glyph, "●" | "◉" | "•");
        let style = Style::default().fg(self.color()).add_modifier(if open {
            Modifier::BOLD
        } else {
            Modifier::DIM
        });
        (glyph, style)
    }
}

/// The text trailing the activity glyph (pure; testable): phase, elapsed `m:ss`,
/// token count, and the rate. The rate distinguishes "slow" (low but non-zero,
/// e.g. a large model under memory pressure) from "stalled" (decaying toward 0).
fn activity_text(answering: bool, elapsed: Duration, frags: usize) -> String {
    let secs = elapsed.as_secs();
    let phase = if answering { "answering" } else { "thinking" };
    let rate = frags as f64 / elapsed.as_secs_f64().max(0.1);
    format!(
        " {phase}… · {}:{:02} · {} tok · {:.1} tok/s",
        secs / 60,
        secs % 60,
        frags,
        rate
    )
}

/// The live "the model is working, not hung" indicator as styled spans: a
/// breathing colored glyph (red eye = reasoning, green pulse = answering) and the
/// phase/elapsed/tokens/rate, the whole line tinted by phase.
fn activity_spans(answering: bool, elapsed: Duration, frags: usize) -> Vec<Span<'static>> {
    let skin = if answering {
        ActivitySkin::Pulse
    } else {
        ActivitySkin::Eye
    };
    let (glyph, glyph_style) = skin.glyph(elapsed);
    vec![
        Span::styled(glyph, glyph_style),
        Span::styled(
            activity_text(answering, elapsed, frags),
            Style::default().fg(skin.color()),
        ),
    ]
}

/// Draw the whole UI for one frame.
pub fn ui(frame: &mut Frame, app: &App) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Min(3),    // transcript
            Constraint::Length(3), // input
            Constraint::Length(1), // status
        ])
        .split(frame.area());

    render_transcript(frame, chunks[0], app);
    render_input(frame, chunks[1], app);
    render_status(frame, chunks[2], app);
}

fn render_transcript(frame: &mut Frame, area: Rect, app: &App) {
    let mut lines: Vec<Line> = Vec::new();
    let last = app.transcript.len().saturating_sub(1);
    for (idx, entry) in app.transcript.iter().enumerate() {
        match entry {
            Entry::User(text) => {
                lines.push(Line::from(Span::styled(
                    "you",
                    Style::default()
                        .fg(Color::Cyan)
                        .add_modifier(Modifier::BOLD),
                )));
                push_wrapped(&mut lines, text, Style::default());
                lines.push(Line::from(""));
            }
            Entry::Assistant {
                reasoning,
                answer,
                stop,
            } => {
                lines.push(Line::from(Span::styled(
                    "assistant",
                    Style::default()
                        .fg(Color::Green)
                        .add_modifier(Modifier::BOLD),
                )));
                if !reasoning.is_empty() {
                    // The in-flight turn streams its reasoning live; completed
                    // turns collapse to a one-line summary unless expanded (TUI-5,
                    // Ctrl+R). Collapsing keeps the answer from being buried — but
                    // only collapse when there's an answer to show instead; if the
                    // turn produced no answer (e.g. ran out of budget mid-think),
                    // keep the reasoning visible since it's all there is.
                    let streaming = app.in_flight.is_some() && idx == last;
                    if app.reasoning_expanded || streaming || answer.trim().is_empty() {
                        lines.push(Line::from(Span::styled(
                            if streaming {
                                "▾ reasoning (live)".to_string()
                            } else {
                                "▾ reasoning".to_string()
                            },
                            Style::default().fg(Color::DarkGray),
                        )));
                        push_wrapped(
                            &mut lines,
                            reasoning,
                            Style::default().add_modifier(Modifier::DIM),
                        );
                    } else {
                        lines.push(Line::from(Span::styled(
                            format!("▸ reasoning ({} lines · Ctrl+R)", reasoning.lines().count()),
                            Style::default().fg(Color::DarkGray),
                        )));
                    }
                }
                push_wrapped(&mut lines, answer, Style::default());
                // Surface a non-EOS stop so a truncated / collapsed turn is not
                // mistaken for a complete answer.
                if let Some(note) = stop_note(*stop) {
                    lines.push(Line::from(Span::styled(
                        note,
                        Style::default()
                            .fg(Color::Yellow)
                            .add_modifier(Modifier::DIM),
                    )));
                }
                lines.push(Line::from(""));
            }
            Entry::Error(text) => {
                push_wrapped(
                    &mut lines,
                    &format!("error: {text}"),
                    Style::default().fg(Color::Red),
                );
                lines.push(Line::from(""));
            }
        }
    }

    let inner_width = area.width.saturating_sub(2); // borders
    let viewport = area.height.saturating_sub(2) as usize; // borders

    let block = Block::default().borders(Borders::ALL).title("yatima");
    let paragraph = Paragraph::new(lines)
        .block(block)
        .wrap(Wrap { trim: false });
    // Auto-follow uses the *wrapped* row count, not the logical line count: each
    // logical line can wrap to several rows, so scrolling by logical lines would
    // leave streaming output below the fold (looks frozen while it's still
    // streaming). `line_count` reports the rows Paragraph will actually render.
    let total_rows = paragraph.line_count(inner_width);
    let top = scroll_y(total_rows, viewport, app.scroll_back);
    let paragraph = paragraph.scroll((top as u16, 0));
    frame.render_widget(paragraph, area);
}

/// A short note for a non-`Eos` stop reason, or `None` for a clean finish.
fn stop_note(stop: Option<StopReason>) -> Option<&'static str> {
    match stop {
        Some(StopReason::MaxTokens) => Some("[stopped: hit max tokens]"),
        Some(StopReason::Repetition) => Some("[stopped: repetition detected]"),
        Some(StopReason::Stopped) => Some("[stopped: cancelled]"),
        Some(StopReason::Eos) | None => None,
    }
}

/// Wrap a (possibly multi-line) string into styled `Line`s.
fn push_wrapped(lines: &mut Vec<Line<'static>>, text: &str, style: Style) {
    for line in text.split('\n') {
        lines.push(Line::from(Span::styled(line.to_string(), style)));
    }
}

fn render_input(frame: &mut Frame, area: Rect, app: &App) {
    // While a turn is in flight, the input box title carries the live activity
    // indicator (a breathing colored glyph + stats) — unmistakably working,
    // never apparently hung.
    let title: Line = match &app.in_flight {
        Some(f) => Line::from(activity_spans(f.answering, f.started.elapsed(), f.frags)),
        None => Line::from("message"),
    };
    let busy = app.in_flight.is_some();
    let prompt = if busy {
        Style::default().add_modifier(Modifier::DIM)
    } else {
        Style::default()
    };
    let body = format!("{}{}", app.input, if busy { "" } else { "▏" });
    let paragraph = Paragraph::new(Span::styled(body, prompt))
        .block(Block::default().borders(Borders::ALL).title(title))
        .wrap(Wrap { trim: false });
    frame.render_widget(paragraph, area);
}

fn render_status(frame: &mut Frame, area: Rect, app: &App) {
    let hint = if app.in_flight.is_some() {
        "^C quit (cancel: soon)"
    } else {
        "^C quit · /reset · ^R reasoning · PgUp/PgDn"
    };
    let mut parts = vec![
        app.status.model_label.clone(),
        format!("[{}]", app.status.backend),
        format!("fmt:{}", app.status.format),
    ];
    if let Some(ctx) = context_label(app.status.prompt_tokens, app.status.context_length) {
        parts.push(ctx);
    }
    parts.push(hint.to_string());
    let status = Line::from(Span::styled(
        parts.join("  ·  "),
        Style::default().fg(Color::DarkGray),
    ));
    frame.render_widget(Paragraph::new(status), area);
}

/// Compact token count, e.g. `2.1k`, `512`.
fn kfmt(n: usize) -> String {
    if n >= 1000 {
        format!("{:.1}k", n as f64 / 1000.0)
    } else {
        n.to_string()
    }
}

/// The context-meter label, or `None` before the first turn / tokenizer-less
/// completer. `ctx 2.1k/32k` when the window is known, else `ctx 2.1k`.
fn context_label(prompt_tokens: Option<usize>, context_length: Option<usize>) -> Option<String> {
    let used = prompt_tokens?;
    Some(match context_length {
        Some(total) => format!("ctx {}/{}", kfmt(used), kfmt(total)),
        None => format!("ctx {}", kfmt(used)),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn activity_text_shows_phase_elapsed_tokens_and_rate() {
        let t = activity_text(false, Duration::from_secs(75), 812);
        assert!(t.contains("thinking…"));
        assert!(t.contains("1:15")); // 75s = 1:15
        assert!(t.contains("812 tok"));
        assert!(t.contains("tok/s")); // the slow-vs-stalled signal
        let t = activity_text(true, Duration::from_secs(3), 40);
        assert!(t.contains("answering…"));
        assert!(t.contains("0:03"));
    }

    #[test]
    fn skin_is_phase_colored_and_breathes() {
        // Reasoning = red eye, answering = green pulse.
        assert_eq!(ActivitySkin::Eye.color(), Color::Red);
        assert_eq!(ActivitySkin::Pulse.color(), Color::Green);
        // The glyph advances with time (animates on the redraw tick), and the
        // open frames are bold while the closed ones are dim (it breathes).
        let (g0, _) = ActivitySkin::Eye.glyph(Duration::from_millis(0)); // ◌ (closed)
        let (g3, s3) = ActivitySkin::Eye.glyph(Duration::from_millis(3 * 180)); // ● (open)
        assert_ne!(g0, g3, "the eye glyph should change over time");
        assert!(
            s3.add_modifier.contains(Modifier::BOLD),
            "open frame is bold"
        );
    }

    #[test]
    fn context_label_formats_used_and_total() {
        assert_eq!(context_label(None, Some(32768)), None); // before any turn
        assert_eq!(
            context_label(Some(2100), Some(32768)).unwrap(),
            "ctx 2.1k/32.8k"
        );
        assert_eq!(context_label(Some(512), None).unwrap(), "ctx 512");
    }
}
