//! A readable projection of a recorded tape (headless-drive D3).
//!
//! The raw tape carries one record per streamed token. Reading it means
//! reconstructing the model's text by hand and timing turns with a
//! calculator, so every reviewer wrote their own script. This module is that
//! script, once: it coalesces each maximal uninterrupted run of `Fragment`
//! records with the same turn and channel into one span, and measures each
//! turn from its `Submit` to its first tool call, first image, and `Done`.
//!
//! It is only a projection. The raw per-fragment tape remains the canonical
//! evidence for replay, ordering, and timing; nothing here is written back.
//!
//! - **REPORT-1** coalescing is lossless and never crosses a record: the
//!   concatenated text of the spans projected for one `(turn, channel)`
//!   equals the concatenated text of that channel's fragments in tape order,
//!   and every non-fragment record — a request, a tool note, an image, a
//!   `Done` — sits between spans exactly where it sat between fragments.
//!   Cited by `coalescing_preserves_channel_text_and_crosses_no_record`.

use std::fmt::Write as _;
use std::path::Path;

use anyhow::{Context, Result};
use serde_json::Value;

use crate::{TapeHeader, TapeLine, TapePayload};

/// One maximal run of same-turn, same-channel fragments.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Span {
    pub turn: Option<u64>,
    pub channel: String,
    pub text: String,
    pub fragments: usize,
    pub first_seq: u64,
    pub last_seq: u64,
    pub first_ms: u64,
    pub last_ms: u64,
}

/// A tape record as the report shows it: fragments folded into spans, every
/// other record kept whole and in place.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Row {
    Span(Span),
    Record(TapeLine),
}

/// The `Fragment` payload of an event line, if it is one.
fn fragment(line: &TapeLine) -> Option<(u64, &str, &str)> {
    let TapePayload::Event(event) = &line.payload else {
        return None;
    };
    let fragment = event.get("Fragment")?;
    Some((
        fragment.get("turn_id")?.as_u64()?,
        fragment.get("channel")?.as_str()?,
        fragment.get("text")?.as_str()?,
    ))
}

/// Fold fragment runs into spans, in tape order (REPORT-1).
pub fn coalesce(lines: &[TapeLine]) -> Vec<Row> {
    let mut rows: Vec<Row> = Vec::new();
    for line in lines {
        match fragment(line) {
            Some((turn, channel, text)) => {
                let extends = matches!(
                    rows.last(),
                    Some(Row::Span(span)) if span.turn == Some(turn) && span.channel == channel
                );
                if extends {
                    let Some(Row::Span(span)) = rows.last_mut() else {
                        unreachable!("matched above");
                    };
                    span.text.push_str(text);
                    span.fragments += 1;
                    span.last_seq = line.seq;
                    span.last_ms = line.t_ms;
                } else {
                    rows.push(Row::Span(Span {
                        turn: Some(turn),
                        channel: channel.to_string(),
                        text: text.to_string(),
                        fragments: 1,
                        first_seq: line.seq,
                        last_seq: line.seq,
                        first_ms: line.t_ms,
                        last_ms: line.t_ms,
                    }));
                }
            }
            None => rows.push(Row::Record(line.clone())),
        }
    }
    rows
}

/// One user turn's timeline, measured from its `Submit`.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct TurnReport {
    pub turn_id: u64,
    pub prompt: String,
    pub submitted_ms: u64,
    /// Elapsed from `Submit` to the first tool call, first image, and `Done`.
    pub first_tool_call_ms: Option<u64>,
    pub first_image_ms: Option<u64>,
    pub done_ms: Option<u64>,
    pub stop: Option<String>,
    pub tool_calls: usize,
    pub tool_failures: usize,
    pub images: usize,
    /// `RetractAnswer` events: answers the agent withheld and corrected.
    pub retractions: usize,
    pub cancelled: bool,
    pub errored: bool,
}

/// Per-turn measurements over the whole tape, in submission order.
pub fn turns(lines: &[TapeLine]) -> Vec<TurnReport> {
    let mut turns: Vec<TurnReport> = Vec::new();
    let find = |turns: &mut Vec<TurnReport>, id: u64| -> Option<usize> {
        turns.iter().position(|t| t.turn_id == id)
    };
    for line in lines {
        match &line.payload {
            TapePayload::Request(request) => {
                let value = serde_json::to_value(request).unwrap_or(Value::Null);
                if let Some(submit) = value.get("Submit") {
                    if let Some(id) = submit.get("turn_id").and_then(Value::as_u64) {
                        turns.push(TurnReport {
                            turn_id: id,
                            prompt: submit
                                .get("text")
                                .and_then(Value::as_str)
                                .unwrap_or_default()
                                .to_string(),
                            submitted_ms: line.t_ms,
                            ..TurnReport::default()
                        });
                    }
                } else if let Some(cancel) = value.get("Cancel") {
                    if let Some(i) = cancel
                        .get("turn_id")
                        .and_then(Value::as_u64)
                        .and_then(|id| find(&mut turns, id))
                    {
                        turns[i].cancelled = true;
                    }
                }
            }
            TapePayload::Event(event) => {
                let Some((kind, body)) = event.as_object().and_then(|o| o.iter().next()) else {
                    continue;
                };
                let Some(i) = body
                    .get("turn_id")
                    .and_then(Value::as_u64)
                    .and_then(|id| find(&mut turns, id))
                else {
                    continue;
                };
                let turn = &mut turns[i];
                let since = line.t_ms.saturating_sub(turn.submitted_ms);
                match kind.as_str() {
                    "ToolNote" => match body.get("kind").and_then(Value::as_str) {
                        Some("Call") => {
                            turn.tool_calls += 1;
                            turn.first_tool_call_ms.get_or_insert(since);
                        }
                        Some("Failure") => turn.tool_failures += 1,
                        _ => {}
                    },
                    "Image" => {
                        turn.images += 1;
                        turn.first_image_ms.get_or_insert(since);
                    }
                    "RetractAnswer" => turn.retractions += 1,
                    "Done" => {
                        turn.done_ms = Some(since);
                        turn.stop = body.get("stop").and_then(Value::as_str).map(String::from);
                    }
                    "Error" => {
                        turn.done_ms = Some(since);
                        turn.errored = true;
                    }
                    _ => {}
                }
            }
        }
    }
    turns
}

/// Read a run directory (or a `tape.jsonl` path) into its header and lines.
/// A trailing incomplete line is discarded, as TAPE-1 permits.
pub fn read_tape(path: &Path) -> Result<(TapeHeader, Vec<TapeLine>)> {
    let path = if path.is_dir() {
        path.join("tape.jsonl")
    } else {
        path.to_path_buf()
    };
    let text =
        std::fs::read_to_string(&path).with_context(|| format!("read tape {}", path.display()))?;
    let mut lines = text.lines();
    let header: TapeHeader =
        serde_json::from_str(lines.next().context("empty tape")?).context("parse tape header")?;
    let complete = text.ends_with('\n');
    let mut records = Vec::new();
    let body: Vec<&str> = lines.collect();
    let usable = if complete {
        body.len()
    } else {
        body.len().saturating_sub(1)
    };
    for (n, line) in body.iter().take(usable).enumerate() {
        let record: TapeLine = serde_json::from_str(line)
            .with_context(|| format!("parse tape line {} of {}", n + 2, path.display()))?;
        records.push(record);
    }
    Ok((header, records))
}

fn secs(ms: u64) -> String {
    format!("{:.1}s", ms as f64 / 1000.0)
}

fn opt_secs(ms: Option<u64>) -> String {
    ms.map_or_else(|| "-".to_string(), secs)
}

/// Clip `text` to `limit` characters for display (0 = no clipping).
fn clip(text: &str, limit: usize) -> String {
    if limit == 0 || text.chars().count() <= limit {
        return text.to_string();
    }
    let head: String = text.chars().take(limit).collect();
    format!("{head}… [{} chars]", text.chars().count())
}

/// Render the whole report: header, per-turn table, then the coalesced
/// timeline. `clip_chars` bounds each span's shown text (0 = full).
pub fn render(header: &TapeHeader, lines: &[TapeLine], clip_chars: usize) -> String {
    let mut out = String::new();
    let _ = writeln!(
        out,
        "tape: {} · model {} · {} records",
        header.meta.origin,
        header.meta.model,
        lines.len()
    );
    for (key, value) in &header.meta.notes {
        let _ = writeln!(out, "  {key}: {value}");
    }

    let turns = turns(lines);
    if !turns.is_empty() {
        let _ = writeln!(out, "\nturns (elapsed from Submit):");
        let _ = writeln!(
            out,
            "  turn  1st-call   1st-img      done  calls  imgs  fail  retract  prompt"
        );
        for t in &turns {
            let stop = match (&t.stop, t.cancelled, t.errored) {
                (_, true, _) => " (cancelled)".to_string(),
                (_, _, true) => " (error)".to_string(),
                (Some(s), _, _) if s != "Eos" => format!(" ({s})"),
                _ => String::new(),
            };
            let _ = writeln!(
                out,
                "  {:>4}  {:>8}  {:>8}  {:>8}  {:>5}  {:>4}  {:>4}  {:>7}  {}{}",
                t.turn_id,
                opt_secs(t.first_tool_call_ms),
                opt_secs(t.first_image_ms),
                opt_secs(t.done_ms),
                t.tool_calls,
                t.images,
                t.tool_failures,
                t.retractions,
                clip(&t.prompt, 60),
                stop
            );
        }
    }

    let _ = writeln!(out, "\ntimeline:");
    for row in coalesce(lines) {
        match row {
            Row::Span(span) => {
                let _ = writeln!(
                    out,
                    "\n--- {} turn {} · {} fragments · {}–{} (seq {}–{}):\n{}",
                    span.channel,
                    span.turn.map_or_else(|| "-".to_string(), |t| t.to_string()),
                    span.fragments,
                    secs(span.first_ms),
                    secs(span.last_ms),
                    span.first_seq,
                    span.last_seq,
                    clip(&span.text, clip_chars)
                );
            }
            Row::Record(line) => {
                let (kind, body) = match &line.payload {
                    TapePayload::Request(request) => {
                        let value = serde_json::to_value(request).unwrap_or(Value::Null);
                        describe(&value)
                    }
                    TapePayload::Event(event) => describe(event),
                };
                if kind == "Startup" {
                    continue;
                }
                let _ = writeln!(
                    out,
                    "[{} {}] {kind}: {}",
                    line.seq,
                    secs(line.t_ms),
                    clip(&body, 600)
                );
            }
        }
    }
    out
}

/// An externally tagged wire value as `(variant, compact body)`.
fn describe(value: &Value) -> (String, String) {
    match value {
        Value::Object(map) if map.len() == 1 => {
            let (kind, body) = map.iter().next().expect("one entry");
            (kind.clone(), body.to_string())
        }
        Value::String(s) => (s.clone(), String::new()),
        other => ("?".to_string(), other.to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TapeMeta;
    use yatima_protocol::HostRequest;

    fn event(seq: u64, t_ms: u64, turn: Option<u64>, event: Value) -> TapeLine {
        TapeLine {
            seq,
            t_ms,
            turn,
            payload: TapePayload::Event(event),
        }
    }

    fn fragment(seq: u64, t_ms: u64, turn: u64, channel: &str, text: &str) -> TapeLine {
        event(
            seq,
            t_ms,
            Some(turn),
            serde_json::json!({"Fragment": {"turn_id": turn, "channel": channel, "text": text}}),
        )
    }

    fn submit(seq: u64, t_ms: u64, turn: u64, text: &str) -> TapeLine {
        TapeLine {
            seq,
            t_ms,
            turn: Some(turn),
            payload: TapePayload::Request(HostRequest::Submit {
                turn_id: turn,
                text: text.to_string(),
            }),
        }
    }

    fn note(seq: u64, t_ms: u64, turn: u64, kind: &str, text: &str) -> TapeLine {
        event(
            seq,
            t_ms,
            Some(turn),
            serde_json::json!({"ToolNote": {"turn_id": turn, "kind": kind, "text": text}}),
        )
    }

    /// A tape shaped like a real tool turn: reasoning, a call, a result,
    /// more reasoning, an image, an answer, done; then a second turn whose
    /// answer is retracted and cancelled.
    fn fixture() -> Vec<TapeLine> {
        vec![
            submit(1, 1000, 0, "find images"),
            event(
                2,
                1001,
                Some(0),
                serde_json::json!({"Started": {"turn_id": 0}}),
            ),
            fragment(3, 1100, 0, "Reasoning", "We "),
            fragment(4, 1150, 0, "Reasoning", "search"),
            note(5, 3000, 0, "Call", "web_search {}"),
            note(6, 3200, 0, "Success", "10 results"),
            fragment(7, 3300, 0, "Reasoning", "Now "),
            fragment(8, 3350, 0, "Reasoning", "pick"),
            note(9, 5000, 0, "Call", "read_image {}"),
            event(
                10,
                5100,
                Some(0),
                serde_json::json!({"Image": {"turn_id": 0, "name": "img.png", "artifact": "artifacts/0000-img.png"}}),
            ),
            note(11, 5101, 0, "Success", "1 shown"),
            fragment(12, 5200, 0, "Answer", "Here "),
            fragment(13, 5250, 0, "Answer", "it is."),
            event(
                14,
                6000,
                Some(0),
                serde_json::json!({"Done": {"turn_id": 0, "stop": "Eos"}}),
            ),
            submit(15, 9000, 1, "more"),
            event(
                16,
                9001,
                Some(1),
                serde_json::json!({"Started": {"turn_id": 1}}),
            ),
            fragment(17, 9100, 1, "Answer", "Displayed!"),
            event(
                18,
                9200,
                Some(1),
                serde_json::json!({"RetractAnswer": {"turn_id": 1, "chars": 10}}),
            ),
            note(19, 9201, 1, "Failure", "final answer withheld"),
            fragment(20, 9300, 1, "Reasoning", "Hmm"),
            TapeLine {
                seq: 21,
                t_ms: 9500,
                turn: Some(1),
                payload: TapePayload::Request(HostRequest::Cancel { turn_id: 1 }),
            },
            event(
                22,
                9501,
                Some(1),
                serde_json::json!({"Done": {"turn_id": 1, "stop": "Stopped"}}),
            ),
        ]
    }

    #[test]
    fn coalescing_preserves_channel_text_and_crosses_no_record() {
        // upholds: REPORT-1 — spans concatenate to the channel's fragment
        // text in order, and every non-fragment record keeps its place
        // between them (the run breaks at the record, never across it).
        let lines = fixture();
        let rows = coalesce(&lines);
        // Reference: the channel text straight from the fragments.
        let mut expected: std::collections::BTreeMap<(u64, String), String> = Default::default();
        for line in &lines {
            if let Some((turn, channel, text)) = super::fragment(line) {
                expected
                    .entry((turn, channel.to_string()))
                    .or_default()
                    .push_str(text);
            }
        }
        let mut projected: std::collections::BTreeMap<(u64, String), String> = Default::default();
        for row in &rows {
            if let Row::Span(span) = row {
                projected
                    .entry((span.turn.unwrap(), span.channel.clone()))
                    .or_default()
                    .push_str(&span.text);
            }
        }
        assert_eq!(projected, expected, "lossless per (turn, channel)");

        // No record crossed: the rows' seq order is the tape's seq order,
        // and a span's seq range contains only fragments.
        let mut seqs: Vec<u64> = Vec::new();
        for row in &rows {
            match row {
                Row::Span(span) => {
                    for line in &lines {
                        if (span.first_seq..=span.last_seq).contains(&line.seq) {
                            assert!(
                                super::fragment(line).is_some(),
                                "seq {} inside a span",
                                line.seq
                            );
                        }
                    }
                    seqs.push(span.first_seq);
                }
                Row::Record(line) => seqs.push(line.seq),
            }
        }
        assert!(seqs.windows(2).all(|w| w[0] < w[1]), "tape order kept");
        // The tool call at seq 5 split turn 0's reasoning into two spans.
        let reasoning: Vec<&Span> = rows
            .iter()
            .filter_map(|r| match r {
                Row::Span(s) if s.channel == "Reasoning" && s.turn == Some(0) => Some(s),
                _ => None,
            })
            .collect();
        assert_eq!(reasoning.len(), 2);
        assert_eq!(reasoning[0].text, "We search");
        assert_eq!(
            (
                reasoning[0].fragments,
                reasoning[0].first_ms,
                reasoning[0].last_ms
            ),
            (2, 1100, 1150)
        );
        assert_eq!(reasoning[1].text, "Now pick");
    }

    #[test]
    fn turn_metrics_measure_from_submit() {
        let t = turns(&fixture());
        assert_eq!(t.len(), 2);
        assert_eq!(t[0].prompt, "find images");
        assert_eq!(t[0].first_tool_call_ms, Some(2000));
        assert_eq!(t[0].first_image_ms, Some(4100));
        assert_eq!(t[0].done_ms, Some(5000));
        assert_eq!((t[0].tool_calls, t[0].images, t[0].retractions), (2, 1, 0));
        assert_eq!(t[0].stop.as_deref(), Some("Eos"));
        assert_eq!(t[1].first_tool_call_ms, None);
        assert_eq!(t[1].first_image_ms, None);
        assert_eq!((t[1].retractions, t[1].tool_failures), (1, 1));
        assert!(t[1].cancelled);
        assert_eq!(t[1].stop.as_deref(), Some("Stopped"));
    }

    #[test]
    fn render_shows_the_table_and_the_folded_timeline() {
        let header = TapeHeader {
            schema: 1,
            meta: TapeMeta {
                origin: "test".into(),
                model: "stub".into(),
                notes: Default::default(),
            },
        };
        let text = render(&header, &fixture(), 0);
        assert!(text.contains("1st-call"), "{text}");
        assert!(text.contains("2.0s"), "first call at 2.0s: {text}");
        assert!(text.contains("4.1s"), "first image at 4.1s: {text}");
        assert!(text.contains("(cancelled)"), "{text}");
        assert!(
            text.contains("--- Reasoning turn 0 · 2 fragments"),
            "{text}"
        );
        assert!(text.contains("We search"), "{text}");
        assert!(text.contains("ToolNote:"), "{text}");
        assert!(!text.contains("\"Fragment\""), "no raw fragments: {text}");
    }

    #[test]
    fn read_tape_drops_only_an_incomplete_final_line() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("tape.jsonl");
        let header =
            serde_json::json!({"schema": 1, "meta": {"origin": "t", "model": "m", "notes": {}}});
        let good = r#"{"seq": 0, "t_ms": 1, "turn": 0, "request": {"Submit": {"turn_id": 0, "text": "hi"}}}"#;
        std::fs::write(
            &path,
            format!("{header}\n{good}\n{{\"seq\": 1, \"t_ms\": 2, \"event\": {{\"Done\": {{\"tur"),
        )
        .unwrap();
        let (_, lines) = read_tape(dir.path()).unwrap();
        assert_eq!(
            lines.len(),
            1,
            "the torn tail is discarded, the prefix kept"
        );
        assert_eq!(lines[0].seq, 0);
    }
}
