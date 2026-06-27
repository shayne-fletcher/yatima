//! Pure rendering: `ui(frame, &App)` mutates nothing (TUI-2). The transcript,
//! input box, and status bar are drawn as a projection of [`App`] state.

use std::time::Duration;

use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Paragraph, Wrap};
use ratatui::Frame;

use crate::app::{scroll_y, App, Entry};

/// Braille spinner frames for the live activity indicator.
const SPINNER: [char; 10] = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'];

/// The live "the model is working, not hung" indicator (pure, so it is testable):
/// an animated spinner, the phase (thinking vs answering), elapsed `m:ss`, and a
/// token count. The spinner frame is derived from elapsed time so it animates on
/// the periodic redraw tick even when the model stalls between tokens.
pub fn activity_line(answering: bool, elapsed: Duration, frags: usize) -> String {
    let frame = SPINNER[(elapsed.as_millis() / 100) as usize % SPINNER.len()];
    let secs = elapsed.as_secs();
    let phase = if answering { "answering" } else { "thinking" };
    format!(
        "{frame} {phase}… · {}:{:02} · {} tok",
        secs / 60,
        secs % 60,
        frags
    )
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
    for entry in &app.transcript {
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
                stop: _,
            } => {
                lines.push(Line::from(Span::styled(
                    "assistant",
                    Style::default()
                        .fg(Color::Green)
                        .add_modifier(Modifier::BOLD),
                )));
                if !reasoning.is_empty() {
                    // Slice 2 will make this foldable; for now it is dimmed.
                    push_wrapped(
                        &mut lines,
                        reasoning,
                        Style::default().add_modifier(Modifier::DIM),
                    );
                }
                push_wrapped(&mut lines, answer, Style::default());
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

    let viewport = area.height.saturating_sub(2) as usize; // borders
    let top = scroll_y(lines.len(), viewport, app.scroll_back);

    let block = Block::default().borders(Borders::ALL).title("yatima");
    let paragraph = Paragraph::new(lines)
        .block(block)
        .wrap(Wrap { trim: false })
        .scroll((top as u16, 0));
    frame.render_widget(paragraph, area);
}

/// Wrap a (possibly multi-line) string into styled `Line`s.
fn push_wrapped(lines: &mut Vec<Line<'static>>, text: &str, style: Style) {
    for line in text.split('\n') {
        lines.push(Line::from(Span::styled(line.to_string(), style)));
    }
}

fn render_input(frame: &mut Frame, area: Rect, app: &App) {
    // While a turn is in flight, the input box title carries the live activity
    // indicator (so it is unmistakably working, never apparently hung).
    let title: String = match &app.in_flight {
        Some(f) => activity_line(f.answering, f.started.elapsed(), f.frags),
        None => "message".to_string(),
    };
    let busy = app.in_flight.is_some();
    let prompt = if busy {
        Style::default().add_modifier(Modifier::DIM)
    } else {
        Style::default()
    };
    let body = format!("{}{}", app.input, if busy { "" } else { "▏" });
    let title_style = if busy {
        Style::default()
            .fg(Color::Yellow)
            .add_modifier(Modifier::BOLD)
    } else {
        Style::default()
    };
    let paragraph = Paragraph::new(Span::styled(body, prompt))
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title(Span::styled(title, title_style)),
        )
        .wrap(Wrap { trim: false });
    frame.render_widget(paragraph, area);
}

fn render_status(frame: &mut Frame, area: Rect, app: &App) {
    let hint = if app.in_flight.is_some() {
        "^C quit (cancel: soon)"
    } else {
        "^C quit · PgUp/PgDn scroll"
    };
    let parts = [
        app.status.model_label.clone(),
        format!("[{}]", app.status.backend),
        format!("fmt:{}", app.status.format),
        hint.to_string(),
    ];
    let status = Line::from(Span::styled(
        parts.join("  ·  "),
        Style::default().fg(Color::DarkGray),
    ));
    frame.render_widget(Paragraph::new(status), area);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn activity_line_shows_phase_elapsed_and_tokens() {
        let line = activity_line(false, Duration::from_secs(75), 812);
        assert!(line.contains("thinking…"));
        assert!(line.contains("1:15")); // 75s = 1:15
        assert!(line.contains("812 tok"));
        let line = activity_line(true, Duration::from_secs(3), 40);
        assert!(line.contains("answering…"));
        assert!(line.contains("0:03"));
    }

    #[test]
    fn spinner_animates_with_elapsed_time() {
        // Different elapsed → (generally) different spinner frame, so the
        // indicator visibly moves on the redraw tick even between tokens.
        let a = activity_line(false, Duration::from_millis(0), 0);
        let b = activity_line(false, Duration::from_millis(500), 0);
        assert_ne!(
            a.chars().next(),
            b.chars().next(),
            "spinner frame should advance with time"
        );
    }
}
