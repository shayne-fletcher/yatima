//! Pure rendering: `ui(frame, &App)` mutates nothing (TUI-2). The transcript,
//! input box, and status bar are drawn as a projection of [`App`] state.

use std::sync::OnceLock;
use std::time::Duration;

use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Paragraph, Wrap};
use ratatui::Frame;
use syntect::easy::HighlightLines;
use syntect::highlighting::{Theme, ThemeSet};
use syntect::parsing::SyntaxSet;

use yatima_host::StopKind;
use yatima_text::prettify_math;

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
/// still. In flight: a single lit letter skips left-to-right along "yatima"
/// (the rest dim), shimmering through the aurora ramp as it travels — a cute
/// little runner. It quickens while answering (excited, busting to share).
fn yatima_title(state: Option<(Duration, bool)>) -> Line<'static> {
    const WORD: &str = "yatima";
    let Some((elapsed, answering)) = state else {
        return Line::from(vec![Span::styled(
            WORD,
            Style::default().fg(Color::DarkGray),
        )]);
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
    // The input box grows with the prompt (Alt+Enter adds lines), capped so it
    // never crowds out the transcript; +2 for the borders.
    let input_rows = (app.input.lines().len().clamp(1, 8) + 2) as u16;
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Min(3),             // transcript
            Constraint::Length(input_rows), // input
            Constraint::Length(1),          // status
        ])
        .split(frame.area());

    render_transcript(frame, chunks[0], app);
    render_input(frame, chunks[1], app);
    render_status(frame, chunks[2], app);
}

/// The user's speaker label: their login name (`$USER`), falling back to
/// "you". Resolved once — it cannot change mid-session.
fn user_label() -> &'static str {
    static LABEL: OnceLock<String> = OnceLock::new();
    LABEL.get_or_init(|| {
        std::env::var("USER")
            .ok()
            .filter(|u| !u.trim().is_empty())
            .unwrap_or_else(|| "you".to_string())
    })
}

fn render_transcript(frame: &mut Frame, area: Rect, app: &App) {
    let inner_width = area.width.saturating_sub(2); // borders
    let mut lines: Vec<Line> = Vec::new();
    let last = app.transcript.len().saturating_sub(1);
    for (idx, entry) in app.transcript.iter().enumerate() {
        match entry {
            Entry::User(text) => {
                lines.push(Line::from(Span::styled(
                    user_label(),
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
                    "yatima",
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
                        lines.extend(render_reasoning(reasoning, inner_width));
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
                // mistaken for a complete answer. A user cancel is called out
                // boldly (it's a deliberate act and should read as one); the
                // automatic stops stay quiet and dim.
                if let Some(note) = stop_note(*stop) {
                    let style = if matches!(stop, Some(StopKind::Stopped)) {
                        Style::default().fg(Color::Red).add_modifier(Modifier::BOLD)
                    } else {
                        Style::default()
                            .fg(Color::Yellow)
                            .add_modifier(Modifier::DIM)
                    };
                    lines.push(Line::from(Span::styled(note, style)));
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
            // Host notices (grants and the like): dim, quiet, but on the
            // record — authority changes are visible history (CAP-3).
            Entry::Notice(text) => {
                push_wrapped(
                    &mut lines,
                    &format!("◆ {text}"),
                    Style::default().fg(Color::Indexed(140)),
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

/// Render the assistant's answer as Markdown lines. The raw text is first run
/// through [`prettify_math`] (LaTeX → Unicode). GFM pipe tables — which
/// `tui-markdown` does not render — are pulled out and drawn as aligned columns
/// by [`render_table`]; everything else goes through [`render_markdown_block`].
fn render_answer(answer: &str, width: u16) -> Vec<Line<'static>> {
    // Images first (file:// artifact echoes drop, remote ones become plain
    // links) — tui-markdown would only warn-and-vanish them.
    let answer = prettify_math(&yatima_text::tame_markdown_images(answer));
    let lines: Vec<&str> = answer.lines().collect();
    let mut out: Vec<Line<'static>> = Vec::new();
    let mut buf: Vec<&str> = Vec::new();
    let mut i = 0;
    while i < lines.len() {
        // A fenced code block: ``` (or ~~~) opens it, the same fence closes it.
        // We render it ourselves (verbatim lines + syntax highlight) — leaving it
        // to the Markdown pass collapses the newlines and leaks the fences.
        if is_code_fence(lines[i]) {
            if !buf.is_empty() {
                out.extend(render_markdown_block(&buf.join("\n"), width));
                buf.clear();
            }
            let lang = fence_lang(lines[i]);
            let mut j = i + 1;
            while j < lines.len() && !is_code_fence(lines[j]) {
                j += 1;
            }
            out.extend(render_code_block(&lang, &lines[i + 1..j], width, false));
            i = if j < lines.len() { j + 1 } else { j }; // skip the closing fence
            continue;
        }
        // A GFM table starts with a `|`-bearing header row immediately followed
        // by a separator row (dashes/colons), and runs while rows carry `|`.
        if i + 1 < lines.len() && lines[i].contains('|') && is_table_separator(lines[i + 1]) {
            if !buf.is_empty() {
                out.extend(render_markdown_block(&buf.join("\n"), width));
                buf.clear();
            }
            let mut j = i + 2;
            while j < lines.len() && lines[j].contains('|') && !lines[j].trim().is_empty() {
                j += 1;
            }
            out.extend(render_table(lines[i], &lines[i + 2..j], width));
            i = j;
        } else {
            buf.push(lines[i]);
            i += 1;
        }
    }
    if !buf.is_empty() {
        out.extend(render_markdown_block(&buf.join("\n"), width));
    }
    out
}

/// Render the reasoning scratchpad: dim plain text (LaTeX prettified), but with
/// fenced code blocks syntax-highlighted in muted form — code stays legible
/// without competing with the answer.
fn render_reasoning(reasoning: &str, width: u16) -> Vec<Line<'static>> {
    let text = prettify_math(reasoning);
    let lines: Vec<&str> = text.lines().collect();
    let dim = Style::default().add_modifier(Modifier::DIM);
    let mut out: Vec<Line<'static>> = Vec::new();
    let mut i = 0;
    while i < lines.len() {
        if is_code_fence(lines[i]) {
            let lang = fence_lang(lines[i]);
            let mut j = i + 1;
            while j < lines.len() && !is_code_fence(lines[j]) {
                j += 1;
            }
            out.extend(render_code_block(&lang, &lines[i + 1..j], width, true));
            i = if j < lines.len() { j + 1 } else { j };
        } else {
            out.push(Line::from(Span::styled(lines[i].to_string(), dim)));
            i += 1;
        }
    }
    out
}

/// Render a Markdown fragment via `tui-markdown`, post-processing the two things
/// it leaves raw: strip the leading `#` from ATX headings (keeping the color it
/// applied) and turn a thematic break (`---`/`***`/`___`) into a drawn rule.
fn render_markdown_block(text: &str, width: u16) -> Vec<Line<'static>> {
    let mut out: Vec<Line<'static>> = Vec::new();
    for line in tui_markdown::from_str(text).lines {
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

/// Whether `line` opens or closes a fenced code block (``` or ~~~).
fn is_code_fence(line: &str) -> bool {
    let t = line.trim_start();
    t.starts_with("```") || t.starts_with("~~~")
}

/// The language token on a fence line (` ```rust ` → `rust`), lowercased.
fn fence_lang(line: &str) -> String {
    line.trim()
        .trim_start_matches(['`', '~'])
        .trim()
        .to_lowercase()
}

/// The default syntax/theme sets, loaded once. `nonewlines` matches our
/// line-at-a-time highlighting (we split the block and strip the `\n`).
fn syntax_set() -> &'static SyntaxSet {
    static SET: OnceLock<SyntaxSet> = OnceLock::new();
    SET.get_or_init(SyntaxSet::load_defaults_nonewlines)
}

fn code_theme() -> &'static Theme {
    static THEME: OnceLock<Theme> = OnceLock::new();
    THEME.get_or_init(|| {
        let mut themes = ThemeSet::load_defaults().themes;
        themes
            .remove("base16-ocean.dark")
            .or_else(|| themes.into_values().next())
            .expect("syntect ships default themes")
    })
}

/// Map a 24-bit color to the nearest xterm-256 index (the 6×6×6 cube or the
/// grayscale ramp, whichever is closer). Terminals without truecolor — Apple
/// Terminal among them — render indexed colors faithfully where they mangle RGB.
fn rgb_to_256(r: u8, g: u8, b: u8) -> u8 {
    const STEPS: [u8; 6] = [0, 95, 135, 175, 215, 255];
    let nearest = |v: u8| -> usize {
        STEPS
            .iter()
            .enumerate()
            .min_by_key(|(_, &s)| (s as i32 - v as i32).abs())
            .map(|(i, _)| i)
            .unwrap_or(0)
    };
    let (ri, gi, bi) = (nearest(r), nearest(g), nearest(b));
    let cube_rgb = (STEPS[ri], STEPS[gi], STEPS[bi]);
    let cube = 16 + 36 * ri + 6 * gi + bi;

    let avg = (r as u32 + g as u32 + b as u32) / 3;
    let gray_level = (avg.saturating_sub(8) / 10).min(23) as u8;
    let gray_v = 8 + gray_level * 10;
    let gray_idx = 232 + gray_level as usize;

    let dist = |(ar, ag, ab): (u8, u8, u8)| {
        let d = |x: u8, y: u8| (x as i32 - y as i32).pow(2);
        d(ar, r) + d(ag, g) + d(ab, b)
    };
    if dist(cube_rgb) <= dist((gray_v, gray_v, gray_v)) {
        cube as u8
    } else {
        gray_idx as u8
    }
}

/// Render a fenced code block: each line verbatim (indentation preserved), syntax
/// highlighted via `syntect` with colors mapped to the 256-cube, and framed by a
/// dim left gutter. In the answer pane (`muted == false`) the theme background
/// tints the whole block so it reads as a panel; in the reasoning scratchpad
/// (`muted == true`) the highlight is kept but dimmed and the panel tint dropped,
/// so code stays subordinate to the answer.
fn render_code_block(lang: &str, code: &[&str], width: u16, muted: bool) -> Vec<Line<'static>> {
    let ss = syntax_set();
    let syntax = (!lang.is_empty())
        .then(|| ss.find_syntax_by_token(lang))
        .flatten()
        .unwrap_or_else(|| ss.find_syntax_plain_text());
    let theme = code_theme();
    let mut hl = HighlightLines::new(syntax, theme);

    // Answer: tint the whole block with the theme background (a panel). Reasoning:
    // no tint, everything dimmed.
    let base = if muted {
        Style::default().add_modifier(Modifier::DIM)
    } else {
        match theme.settings.background {
            Some(c) => Style::default().bg(Color::Indexed(rgb_to_256(c.r, c.g, c.b))),
            None => Style::default(),
        }
    };

    const GUTTER: &str = "▏ ";
    let body = width as usize;

    code.iter()
        .map(|line| {
            let ranges = hl.highlight_line(line, ss).unwrap_or_default();
            let mut spans = vec![Span::styled(GUTTER, base.fg(Color::Indexed(238)))];
            for (style, text) in ranges {
                let fg = style.foreground;
                spans.push(Span::styled(
                    text.to_string(),
                    base.fg(Color::Indexed(rgb_to_256(fg.r, fg.g, fg.b))),
                ));
            }
            // Pad the row so the panel tint fills the width (only when tinted).
            let used = GUTTER.chars().count() + line.chars().count();
            if !muted && used < body {
                spans.push(Span::styled(" ".repeat(body - used), base));
            }
            Line::from(spans)
        })
        .collect()
}

/// Whether `line` is a GFM table separator row: dashes/colons and pipes only,
/// with at least one dash (e.g. `| --- | :--: |` or `---|---`).
fn is_table_separator(line: &str) -> bool {
    let t = line.trim();
    t.contains('-') && t.contains('|') && t.chars().all(|c| matches!(c, '|' | '-' | ':' | ' '))
}

/// Split a table row into trimmed cell strings (dropping the outer pipes).
fn table_cells(row: &str) -> Vec<String> {
    let t = row.trim();
    let t = t.strip_prefix('|').unwrap_or(t);
    let t = t.strip_suffix('|').unwrap_or(t);
    t.split('|').map(|c| c.trim().to_string()).collect()
}

/// Pad or truncate `s` to exactly `w` display columns (char-count approximation;
/// truncation appends `…`).
fn fit_cell(s: &str, w: usize) -> String {
    let n = s.chars().count();
    if n > w {
        let keep = w.saturating_sub(1);
        let mut t: String = s.chars().take(keep).collect();
        t.push('…');
        t
    } else {
        format!("{s:<w$}")
    }
}

/// Draw a GFM table as aligned columns: a bold header, a rule, then the body.
/// Column widths fit content, scaled down evenly if the natural table exceeds
/// `width` (cells then truncate with `…`).
fn render_table(header: &str, body: &[&str], width: u16) -> Vec<Line<'static>> {
    let head = table_cells(header);
    let rows: Vec<Vec<String>> = body.iter().map(|r| table_cells(r)).collect();
    let ncols = head.len().max(rows.iter().map(Vec::len).max().unwrap_or(0));
    if ncols == 0 {
        return Vec::new();
    }
    let cell = |row: &[String], c: usize| row.get(c).cloned().unwrap_or_default();

    // Natural width per column, then scale to fit (3 cols of " │ " overhead).
    let mut widths: Vec<usize> = (0..ncols)
        .map(|c| {
            let h = head.get(c).map(|s| s.chars().count()).unwrap_or(0);
            let b = rows
                .iter()
                .map(|r| cell(r, c).chars().count())
                .max()
                .unwrap_or(0);
            h.max(b).max(1)
        })
        .collect();
    let sep = " │ ";
    let overhead = ncols.saturating_sub(1) * sep.chars().count();
    let budget = (width as usize).saturating_sub(overhead).max(ncols);
    if widths.iter().sum::<usize>() > budget {
        let each = (budget / ncols).max(1);
        for w in &mut widths {
            *w = (*w).min(each);
        }
    }

    let row_line = |cells: &[String], style: Style| -> Line<'static> {
        let mut spans: Vec<Span<'static>> = Vec::new();
        for (c, w) in widths.iter().enumerate() {
            if c > 0 {
                spans.push(Span::styled(sep, Style::default().fg(Color::DarkGray)));
            }
            spans.push(Span::styled(fit_cell(&cell(cells, c), *w), style));
        }
        Line::from(spans)
    };

    let mut out: Vec<Line<'static>> = Vec::new();
    let header_style = Style::default()
        .fg(Color::Cyan)
        .add_modifier(Modifier::BOLD);
    out.push(row_line(&head, header_style));
    // Rule under the header: per-column dashes joined by `─┼─`.
    let rule = widths
        .iter()
        .map(|w| "─".repeat(*w))
        .collect::<Vec<_>>()
        .join("─┼─");
    out.push(Line::from(Span::styled(
        rule,
        Style::default().fg(Color::DarkGray),
    )));
    for r in &rows {
        out.push(row_line(r, Style::default()));
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

/// A short note for a non-`Eos` stop, or `None` for a clean finish.
fn stop_note(stop: Option<StopKind>) -> Option<&'static str> {
    match stop {
        Some(StopKind::MaxTokens) => Some("[stopped: hit max tokens]"),
        Some(StopKind::Repetition) => Some("[stopped: repetition detected]"),
        Some(StopKind::Stopped) => Some("⊘ interrupted"),
        Some(StopKind::Eos) | None => None,
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
    // never apparently hung. An armed quit outranks it: the confirm hint must
    // be seen for the second Ctrl+C/Ctrl+D to mean anything.
    let title: Line = if app.quit_armed {
        Line::from(Span::styled(
            "press ^C or ^D again to quit",
            Style::default().fg(Color::Yellow),
        ))
    } else {
        match &app.in_flight {
            Some(f) => Line::from(activity_spans(
                f.answering,
                f.cancelling,
                f.started.elapsed(),
                f.frags,
            )),
            None => Line::from("message"),
        }
    };
    // The input editor (`tui-textarea`) renders itself — cursor at the point,
    // horizontal scroll, placeholder. We draw the bordered block (its title is
    // the live activity indicator) and hand the editor the inner rect.
    let block = Block::default().borders(Borders::ALL).title(title);
    let inner = block.inner(area);
    frame.render_widget(block, area);
    frame.render_widget(&app.input, inner);
}

fn render_status(frame: &mut Frame, area: Rect, app: &App) {
    let hint = if app.in_flight.is_some() {
        "Esc cancel · ^C^C quit"
    } else {
        "^C^C quit · /reset · ^R reasoning · ^G editor · PgUp/PgDn"
    };
    let mut parts = vec![
        app.status.model_label.clone(),
        format!("[{}]", app.status.backend),
        format!("fmt:{}", app.status.format),
    ];
    if let Some(ctx) = context_label(app.status.prompt_tokens, app.status.context_length) {
        parts.push(ctx);
    }
    if let Some(web) = grants_label(&app.status.grants) {
        parts.push(web);
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

/// The status rail's web-authority label (CAP-3), or `None` before any grant:
/// the lone origin's host, or a count when several are granted.
fn grants_label(grants: &[String]) -> Option<String> {
    match grants {
        [] => None,
        [one] => Some(format!(
            "web:{}",
            one.trim_start_matches("https://")
                .trim_start_matches("http://")
        )),
        many => Some(format!("web:{} origins", many.len())),
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
        // Idle: the dim "yatima" label, nothing else.
        let idle = yatima_title(None);
        let idle_text: String = idle.spans.iter().map(|s| s.content.as_ref()).collect();
        assert_eq!(idle_text, "yatima");

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
    fn answer_renders_gfm_tables_as_aligned_columns() {
        let md = "| Type | Formula |\n|------|---------|\n| Recurrence | F_n = F_{n-1} + F_{n-2} |\n| Closed | Binet |";
        let lines = render_answer(md, 80);
        let texts: Vec<String> = lines
            .iter()
            .map(|l| l.spans.iter().map(|s| s.content.as_ref()).collect())
            .collect();
        // No raw pipe-table markup leaks through.
        assert!(
            !texts.iter().any(|t| t.contains("|---")),
            "separator row is drawn, not raw: {texts:?}"
        );
        // Header and a body cell survive, drawn with the column separator.
        assert!(
            texts.iter().any(|t| t.contains("Type") && t.contains('│')),
            "header row with column separator: {texts:?}"
        );
        assert!(
            texts.iter().any(|t| t.contains("Recurrence")),
            "body cell survives: {texts:?}"
        );
        // The header rule uses the ─┼─ junction.
        assert!(
            texts.iter().any(|t| t.contains("┼")),
            "a header rule is drawn: {texts:?}"
        );
    }

    #[test]
    fn answer_renders_fenced_code_verbatim_and_highlighted() {
        let md = "intro\n\n```rust\nfn main() {\n    let x = 1;\n}\n```\n\nafter";
        let lines = render_answer(md, 40);
        let texts: Vec<String> = lines
            .iter()
            .map(|l| l.spans.iter().map(|s| s.content.as_ref()).collect())
            .collect();
        assert!(
            !texts.iter().any(|t| t.contains("```")),
            "fences are consumed: {texts:?}"
        );
        // Each code line survives on its own row (newlines preserved, not mashed).
        assert!(texts.iter().any(|t| t.contains("fn main()")));
        assert!(
            texts
                .iter()
                .any(|t| t.contains("let x = 1;") && !t.contains("fn main")),
            "code lines are not collapsed together: {texts:?}"
        );
        // Indentation is preserved.
        assert!(
            texts.iter().any(|t| t.contains("    let x = 1;")),
            "leading indentation kept: {texts:?}"
        );
        // Highlighting produced styled (non-default) spans.
        assert!(
            lines
                .iter()
                .flat_map(|l| &l.spans)
                .any(|s| s.style.fg.is_some()),
            "syntax highlighting applied a color"
        );
    }

    #[test]
    fn reasoning_highlights_code_muted() {
        let r = "let me try:\n\n```rust\nfn f() {}\n```\n\nthat works";
        let lines = render_reasoning(r, 40);
        let texts: Vec<String> = lines
            .iter()
            .map(|l| l.spans.iter().map(|s| s.content.as_ref()).collect())
            .collect();
        assert!(
            !texts.iter().any(|t| t.contains("```")),
            "fences consumed in reasoning: {texts:?}"
        );
        assert!(texts.iter().any(|t| t.contains("fn f()")), "code survives");
        // The code line is highlighted (a colored span) AND muted (DIM).
        let code_line = lines
            .iter()
            .find(|l| {
                l.spans
                    .iter()
                    .map(|s| s.content.as_ref())
                    .collect::<String>()
                    .contains("fn f()")
            })
            .expect("code line present");
        assert!(
            code_line.spans.iter().any(|s| s.style.fg.is_some()),
            "reasoning code is highlighted"
        );
        assert!(
            code_line
                .spans
                .iter()
                .all(|s| s.style.add_modifier.contains(Modifier::DIM)),
            "reasoning code is muted (DIM)"
        );
    }

    #[test]
    fn code_fence_helpers_parse() {
        assert!(is_code_fence("```rust"));
        assert!(is_code_fence("  ~~~"));
        assert!(!is_code_fence("not code"));
        assert_eq!(fence_lang("```rust"), "rust");
        assert_eq!(fence_lang("```  Python "), "python");
        assert_eq!(fence_lang("```"), "");
    }

    #[test]
    fn rgb_to_256_maps_into_indexed_range() {
        assert_eq!(rgb_to_256(0, 0, 0), 16); // cube origin
        assert_eq!(rgb_to_256(255, 255, 255), 231); // cube apex
        let mid = rgb_to_256(128, 128, 128);
        assert!((16..=255).contains(&mid)); // some grayscale/cube index
    }

    #[test]
    fn is_table_separator_recognizes_gfm_rules() {
        assert!(is_table_separator("|------|---------|"));
        assert!(is_table_separator("| :--- | ---: |"));
        assert!(is_table_separator("---|---"));
        assert!(!is_table_separator("| a | b |")); // a data row, not a rule
        assert!(!is_table_separator("just prose"));
    }

    #[test]
    fn fit_cell_pads_and_truncates() {
        assert_eq!(fit_cell("hi", 5), "hi   ");
        assert_eq!(fit_cell("hello", 5), "hello");
        assert_eq!(fit_cell("toolong", 5), "tool…");
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
