//! Pure rendering: `ui(frame, &App)` mutates nothing (TUI-2). The transcript,
//! input box, and status bar are drawn as a projection of [`App`] state.

use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Paragraph, Wrap};
use ratatui::Frame;

use crate::app::{scroll_y, App, Entry};

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
    let busy = app.in_flight.is_some();
    let title = if busy { "generating…" } else { "message" };
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
    let mut parts = vec![
        app.status.model_label.clone(),
        format!("[{}]", app.status.backend),
        format!("fmt:{}", app.status.format),
    ];
    if let Some(f) = &app.in_flight {
        let secs = f.started.elapsed().as_secs_f64().max(0.001);
        let toks_per_s = f.frags as f64 / secs;
        parts.push(format!("⠿ {:.1} tok/s", toks_per_s));
    }
    parts.push("^C quit · PgUp/PgDn scroll".to_string());
    let status = Line::from(Span::styled(
        parts.join("  ·  "),
        Style::default().fg(Color::DarkGray),
    ));
    frame.render_widget(Paragraph::new(status), area);
}
