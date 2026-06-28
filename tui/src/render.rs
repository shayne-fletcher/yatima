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

// Aurora — northern-lights greens, teals, blues and violets shimmering to pink,
// over the 256-color cube (not 24-bit RGB, so it renders in Apple Terminal,
// which has no truecolor). The one palette the live UI animates with. The ramp
// is *ping-ponged*, so the open ends shimmer back and forth instead of jumping.
const AURORA: [u8; 12] = [48, 43, 50, 51, 45, 39, 33, 63, 99, 141, 177, 213];
const COLOR_STEP_MS: u128 = 140;

// A single quadrant orbiting the cell corners — the activity glyph, a smooth
// spin in the logo's block idiom.
const ORBIT: [&str; 4] = ["▘", "▝", "▗", "▖"];

/// The orbit glyph for this moment.
fn orbit_glyph(elapsed: Duration) -> &'static str {
    ORBIT[(elapsed.as_millis() / 180) as usize % ORBIT.len()]
}

/// An aurora color sampled at position `pos` along the (ping-ponged) ramp.
fn aurora_at(pos: usize) -> Color {
    let n = AURORA.len();
    let period = 2 * (n - 1);
    let t = pos % period;
    let i = if t < n { t } else { period - t };
    Color::Indexed(AURORA[i])
}

/// The activity glyph's shimmering color this moment.
fn aurora_now(elapsed: Duration) -> Color {
    aurora_at((elapsed.as_millis() / COLOR_STEP_MS) as usize)
}

/// The transcript pane's "yatima" label, colored by UI state. Idle: dim and
/// still. In flight: a single lit letter skips left-to-right along the word
/// (the rest dim), shimmering through the aurora ramp as it travels — a cute
/// little runner. It quickens while answering (excited, busting to share).
fn yatima_title(state: Option<(Duration, bool)>) -> Line<'static> {
    const WORD: &str = "yatima";
    let Some((elapsed, answering)) = state else {
        return Line::from(Span::styled(WORD, Style::default().fg(Color::DarkGray)));
    };
    let step_ms = if answering { 90 } else { 200 }; // answering = a quicker, excited skip
    let tick = (elapsed.as_millis() / step_ms) as usize;
    let len = WORD.chars().count();
    let lit = tick % len; // the one lit letter, advancing left → right
    let spans: Vec<Span<'static>> = WORD
        .chars()
        .enumerate()
        .map(|(i, ch)| {
            let style = if i == lit {
                Style::default()
                    .fg(aurora_at(tick)) // the runner shimmers as it moves
                    .add_modifier(Modifier::BOLD)
            } else {
                Style::default().fg(Color::DarkGray)
            };
            Span::styled(ch.to_string(), style)
        })
        .collect();
    Line::from(spans)
}

/// The text trailing the activity glyph (pure; testable): phase, elapsed `m:ss`,
/// token count, and the rate. The rate distinguishes "slow" (low but non-zero,
/// e.g. a large model under memory pressure) from "stalled" (decaying toward 0).
/// A requested-but-not-yet-effected cancel shows "cancelling…" (the decode stops
/// at the next token boundary), so the key press is never apparently ignored.
fn activity_text(answering: bool, cancelling: bool, elapsed: Duration, frags: usize) -> String {
    let secs = elapsed.as_secs();
    let phase = if cancelling {
        "cancelling"
    } else if answering {
        "answering"
    } else {
        "thinking"
    };
    let rate = frags as f64 / elapsed.as_secs_f64().max(0.1);
    format!(
        " {phase}… · {}:{:02} · {} tok · {:.1} tok/s",
        secs / 60,
        secs % 60,
        frags,
        rate
    )
}

/// The live "the model is working, not hung" indicator as styled spans: an
/// orbiting glyph shimmering through the aurora palette, trailed by the
/// phase/elapsed/tokens/rate in a steady, legible tint.
fn activity_spans(
    answering: bool,
    cancelling: bool,
    elapsed: Duration,
    frags: usize,
) -> Vec<Span<'static>> {
    // The glyph orbits and shimmers through aurora; the stats keep a steady,
    // legible tint so the numbers never strobe.
    vec![
        Span::styled(
            orbit_glyph(elapsed),
            Style::default()
                .fg(aurora_now(elapsed))
                .add_modifier(Modifier::BOLD),
        ),
        Span::styled(
            activity_text(answering, cancelling, elapsed, frags),
            Style::default().fg(Color::Indexed(51)), // steady aurora cyan
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
    let inner_width = area.width.saturating_sub(2); // borders
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
                // Render the answer as Markdown (headings, bold, lists, rules)
                // rather than raw text. Partial Markdown mid-stream renders fine,
                // so this works while the answer is still arriving. The reasoning
                // scratchpad stays plain.
                lines.extend(render_answer(answer, inner_width));
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

    let viewport = area.height.saturating_sub(2) as usize; // borders

    // The "yatima" label carries the live UI state (a cute aurora runner).
    let title = yatima_title(
        app.in_flight
            .as_ref()
            .map(|f| (f.started.elapsed(), f.answering)),
    );
    let block = Block::default().borders(Borders::ALL).title(title);
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

/// Render the assistant's answer as Markdown lines. `tui-markdown` handles
/// emphasis, lists and inline styling; we post-process the two things it leaves
/// raw: strip the leading `#` from ATX headings (keeping the color it applied)
/// and turn a thematic break (`---`/`***`/`___`) into a rule drawn across `width`.
fn render_answer(answer: &str, width: u16) -> Vec<Line<'static>> {
    let mut out: Vec<Line<'static>> = Vec::new();
    for line in tui_markdown::from_str(answer).lines {
        let plain: String = line.spans.iter().map(|s| s.content.as_ref()).collect();
        let trimmed = plain.trim();
        // Thematic break: a line of only -, * or _ (three or more).
        if trimmed.len() >= 3 && trimmed.chars().all(|c| matches!(c, '-' | '*' | '_')) {
            out.push(Line::from(Span::styled(
                "─".repeat(width as usize),
                Style::default().fg(Color::DarkGray),
            )));
            continue;
        }
        // Heading: strip the `#` markers, keep tui-markdown's heading style.
        if let Some(text) = heading_text(&plain) {
            let style = line.spans.first().map(|s| s.style).unwrap_or_default();
            out.push(Line::from(Span::styled(text, style)));
            continue;
        }
        // Otherwise keep tui-markdown's rendering (owned so the line is 'static).
        out.push(Line::from(
            line.spans
                .into_iter()
                .map(|s| Span::styled(s.content.into_owned(), s.style))
                .collect::<Vec<_>>(),
        ));
    }
    out
}

/// The text of a Markdown ATX heading (`## Title` → `Title`), or `None` if the
/// line is not a heading.
fn heading_text(line: &str) -> Option<String> {
    let s = line.trim_start();
    let hashes = s.chars().take_while(|&c| c == '#').count();
    if (1..=6).contains(&hashes) && s[hashes..].starts_with(' ') {
        Some(s[hashes..].trim().to_string())
    } else {
        None
    }
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

/// Wrap a (possibly multi-line) string into styled `Line`s. Generic over the
/// buffer lifetime so it composes with borrowed lines (the Markdown answer
/// borrows the transcript); the owned lines it pushes coerce in.
fn push_wrapped<'a>(lines: &mut Vec<Line<'a>>, text: &str, style: Style) {
    for line in text.split('\n') {
        lines.push(Line::from(Span::styled(line.to_string(), style)));
    }
}

fn render_input(frame: &mut Frame, area: Rect, app: &App) {
    // While a turn is in flight, the input box title carries the live activity
    // indicator (a breathing colored glyph + stats) — unmistakably working,
    // never apparently hung.
    let title: Line = match &app.in_flight {
        Some(f) => Line::from(activity_spans(
            f.answering,
            f.cancelling,
            f.started.elapsed(),
            f.frags,
        )),
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
        "Esc cancel · ^C quit"
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
        let t = activity_text(false, false, Duration::from_secs(75), 812);
        assert!(t.contains("thinking…"));
        assert!(t.contains("1:15")); // 75s = 1:15
        assert!(t.contains("812 tok"));
        assert!(t.contains("tok/s")); // the slow-vs-stalled signal
        let t = activity_text(true, false, Duration::from_secs(3), 40);
        assert!(t.contains("answering…"));
        assert!(t.contains("0:03"));
        // A requested cancel overrides the phase word so the key isn't apparently
        // ignored while the decode winds down to the next token boundary.
        let t = activity_text(true, true, Duration::from_secs(3), 40);
        assert!(t.contains("cancelling…"), "cancel shows in the indicator");
    }

    #[test]
    fn glyph_orbits_and_shimmers_through_aurora() {
        // The aurora shimmer moves over time and is 256-indexed (Apple Terminal
        // has no truecolor); the orbit visits all four quadrants over its cycle.
        let c0 = aurora_now(Duration::from_millis(0));
        let c5 = aurora_now(Duration::from_millis(5 * COLOR_STEP_MS as u64));
        assert_ne!(c0, c5, "the aurora shimmer flows over time");
        assert!(matches!(c0, Color::Indexed(_)), "256-color, not truecolor");
        let quadrants: std::collections::HashSet<_> = (0..4)
            .map(|k| orbit_glyph(Duration::from_millis(k * 180)))
            .collect();
        assert_eq!(quadrants.len(), 4, "the orbit visits every corner");
    }

    #[test]
    fn yatima_title_runs_one_lit_letter_left_to_right() {
        // Idle: a single dim, unanimated span.
        let idle = yatima_title(None);
        assert_eq!(idle.spans.len(), 1);

        // In flight: exactly one letter is lit (not DarkGray); it advances over
        // time and is bold/quicker while answering.
        let lit_index = |line: &Line| {
            line.spans
                .iter()
                .position(|s| s.style.fg != Some(Color::DarkGray))
        };
        let a = yatima_title(Some((Duration::from_millis(0), false)));
        let b = yatima_title(Some((Duration::from_millis(200), false)));
        assert_eq!(a.spans.len(), 6, "each of yatima's letters is its own span");
        assert_eq!(
            a.spans
                .iter()
                .filter(|s| s.style.fg != Some(Color::DarkGray))
                .count(),
            1
        );
        assert_ne!(
            lit_index(&a),
            lit_index(&b),
            "the lit letter moves over time"
        );

        // Answering is the excited mode: the lit letter is bold.
        let ans = yatima_title(Some((Duration::from_millis(0), true)));
        assert!(ans
            .spans
            .iter()
            .any(|s| s.style.add_modifier.contains(Modifier::BOLD)));
    }

    #[test]
    fn answer_markdown_is_rendered_not_raw() {
        // The answer pane parses Markdown: emphasis is rendered (the `**` markup
        // is consumed, the text styled) and the content survives. (tui-markdown
        // keeps the `#` heading marker, styled, as a visual cue — that's fine.)
        let text = tui_markdown::from_str("# Heading\n\nsome **bold** words");
        let rendered: String = text
            .lines
            .iter()
            .flat_map(|l| l.spans.iter().map(|s| s.content.as_ref()))
            .collect();
        assert!(rendered.contains("Heading"), "heading text is kept");
        assert!(rendered.contains("bold"), "emphasis text is kept");
        assert!(
            !rendered.contains("**"),
            "the '**' emphasis markup is consumed, not shown raw"
        );
        // Some span carries a non-default style (the parse produced styling).
        assert!(
            text.lines
                .iter()
                .flat_map(|l| &l.spans)
                .any(|s| s.style != Style::default()),
            "markdown applied styling"
        );
    }

    #[test]
    fn heading_text_parses_atx() {
        assert_eq!(heading_text("### 1. Carbon").as_deref(), Some("1. Carbon"));
        assert_eq!(heading_text("# Title").as_deref(), Some("Title"));
        assert_eq!(heading_text("no heading"), None);
        assert_eq!(heading_text("####### too many"), None); // 7 hashes isn't a heading
        assert_eq!(heading_text("#nospace"), None);
    }

    #[test]
    fn answer_strips_heading_hashes_and_draws_rules() {
        let lines = render_answer("### Heading\n\n---\n\nbody **x**", 12);
        let texts: Vec<String> = lines
            .iter()
            .map(|l| l.spans.iter().map(|s| s.content.as_ref()).collect())
            .collect();
        assert!(
            texts.iter().all(|t| !t.contains('#')),
            "no '#' markup leaks through: {texts:?}"
        );
        assert!(
            texts.iter().any(|t| t.contains("Heading")),
            "heading text survives"
        );
        assert!(
            texts
                .iter()
                .any(|t| t.chars().filter(|&c| c == '─').count() >= 3),
            "a thematic break becomes a drawn rule: {texts:?}"
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
