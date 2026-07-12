//! The browser client's transcript model: a deliberately small miniature of
//! the GUI's `Msg` list — its *semantics* copied, not its code. The model is
//! plain Rust over `yatima-protocol` types (plus the `image` crate for
//! artifact decode), so everything subtle here — char-boundary retraction,
//! the image path, the commit-on-Done rules — is unit-tested natively,
//! without a browser in the loop. The egui app in `main.rs` is a thin view
//! over [`Transcript`] and compiles only for wasm32.
//!
//! Spike cuts (chosen, not discovered — see plans/wasm-spike.plan.md): plain
//! text only (no markdown), PNG/JPEG only (an SVG or other format renders as
//! a named placeholder line, never an error), no grants UI (grant reports
//! render as notes), no context meter beyond a token count.
//!
//! # Invariant & law registry
//!
//! The client copies the GUI/TUI transcript *semantics*; these are the laws
//! that copy must uphold, each cited by a test (`// upholds: <id>`).
//!
//! - **WEB-1** the transcript is a pure mirror: it holds no truth of its own
//!   but folds the `HostEvent` stream (`fold`) into what the host has
//!   emitted — the host's session is authoritative, this is a view of it.
//! - **WEB-2** single in flight: at most one turn runs at a time; a submit is
//!   refused while one is in flight (enforced at the view's send gate).
//! - **WEB-3** a live turn always shows live: any turn activity — `Started`,
//!   an answer or reasoning fragment, or a tool-note — arms the in-flight
//!   marker on demand, so a client that resumes or preempts (SRV-3) into a
//!   turn it never saw begin still renders it running (spinner, stop,
//!   submit-gate).
//! - **WEB-4** a settle names its turn: only a `Done`/`Error` for the turn
//!   believed live settles it, so a stale settle redelivered at the reconnect
//!   seam cannot clear a newer turn.
//! - **WEB-5** a turn is always locally endable: stop settles the mirror at
//!   once (commit what streamed, disarm) without waiting on a host round trip
//!   the one-deep carry slot can drop — the client-side dual of SRV-3's "a
//!   session can always end".
//! - **WEB-6** commit only real output: on settle an answer commits iff it
//!   carried non-whitespace text (a fully-retracted turn commits nothing,
//!   never a blank bubble); retraction counts characters, never bytes; an
//!   artifact renders as an image or a named placeholder, never an error.
//!
//! The marker vocabulary a tool-note renders in is this view's own, not the
//! wire's (**HOST-4**): egui's fonts lack `✓`/`✗`, so `tool_note_line` spells
//! `ok`/`failed:` and keeps `⚙`/`⚠`.

use yatima_protocol::{Channel, HostEvent, ModelInfo, ToolNoteKind};

/// One committed transcript entry (the streaming turn lives in the buffers
/// until `Done`/`Error` commits or discards it).
pub enum Entry {
    User(String),
    Assistant {
        answer: String,
        /// The chain-of-thought, kept so the reasoning toggle can reveal it
        /// after the fact (collapsed by default, as in the GUI).
        reasoning: Option<String>,
    },
    /// A decoded image artifact: raw RGBA the view layer textures.
    Image(DecodedImage),
    /// An app-plane or grant-report line, rendered muted.
    Note(String),
    Error(String),
}

/// An artifact decoded to raw RGBA — everything the view needs to make a
/// texture, with no view types in the model (that keeps this half testable
/// off-browser).
pub struct DecodedImage {
    pub name: String,
    pub size: [usize; 2],
    pub rgba: Vec<u8>,
}

/// Decode PNG/JPEG bytes to RGBA. `None` is not an error: the spike renders
/// unknown formats (SVG, WebP, …) as a named placeholder line.
pub fn decode_rgba(bytes: &[u8]) -> Option<DecodedImage> {
    let rgba = image::load_from_memory(bytes).ok()?.to_rgba8();
    let (w, h) = rgba.dimensions();
    Some(DecodedImage {
        name: String::new(),
        size: [w as usize, h as usize],
        rgba: rgba.into_raw(),
    })
}

/// Render a tool-note payload in this view's marker vocabulary — the same
/// one the GUI uses, for the same reason: egui's built-in fonts lack `✓`/`✗`
/// (tofu), while `⚙`/`⚠` survive via the emoji fallback. A kind this build
/// doesn't know renders unmarked (the enum is `#[non_exhaustive]`).
pub fn tool_note_line(kind: ToolNoteKind, text: &str) -> String {
    match kind {
        ToolNoteKind::Call => format!("\n⚙ {text}\n"),
        ToolNoteKind::Success => format!("  ok {text}\n"),
        ToolNoteKind::Failure => format!("  failed: {text}\n"),
        ToolNoteKind::Warning => format!("\n⚠ {text}\n"),
        _ => format!("  {text}\n"),
    }
}

/// The UI mirror the app renders: committed entries plus the turn in flight.
/// Fold every [`HostEvent`] through [`Transcript::fold`]; the host's session
/// is truth, this is a view of it.
#[derive(Default)]
pub struct Transcript {
    pub entries: Vec<Entry>,
    /// The answer streaming in, if a turn is in flight (armed on submit, or
    /// on demand — see `fold`'s Fragment arm).
    streaming: Option<String>,
    /// The chain-of-thought (and tool notes) streaming in alongside it.
    streaming_reasoning: String,
    /// The turn in flight, if any — drives the spinner and the stop button.
    pub in_flight: Option<u64>,
    /// What's running (set by `Ready`; the status line shows the label).
    pub model: Option<ModelInfo>,
    /// The most recent prompt token count (`Context`), for the status line.
    pub prompt_tokens: Option<usize>,
    /// A fatal load error: the session never starts.
    pub fatal: Option<String>,
}

impl Transcript {
    /// Record the user's submit and arm the streaming buffers (the mirror of
    /// the GUI's `submit`).
    pub fn push_user(&mut self, turn_id: u64, text: &str) {
        self.entries.push(Entry::User(text.to_string()));
        self.streaming = Some(String::new());
        self.streaming_reasoning.clear();
        self.in_flight = Some(turn_id);
    }

    /// The live answer, if a turn is streaming (for the view).
    pub fn streaming_answer(&self) -> Option<&str> {
        self.streaming.as_deref()
    }

    /// The live reasoning fold (empty when nothing streamed).
    pub fn streaming_reasoning(&self) -> &str {
        &self.streaming_reasoning
    }

    /// Settle the streaming turn locally: commit its answer (with the
    /// reasoning fold) if it carried text, then disarm. A fully-retracted
    /// turn commits nothing — an empty answer would render a blank bubble.
    /// Shared by a clean `Done` and by `abort` (the stop button).
    fn settle(&mut self) {
        self.in_flight = None;
        let reasoning = std::mem::take(&mut self.streaming_reasoning);
        let reasoning = reasoning.trim();
        if let Some(buf) = self.streaming.take() {
            if !buf.trim().is_empty() {
                let reasoning = (!reasoning.is_empty()).then(|| reasoning.to_string());
                self.entries.push(Entry::Assistant {
                    answer: buf,
                    reasoning,
                });
            }
        }
    }

    /// End the in-flight turn locally — the stop button's escape hatch. The
    /// user pressed stop; commit whatever streamed and disarm now, without
    /// waiting for the host's `Done` to make the round trip. That round trip
    /// is exactly what the reconnect seam can drop (serve's carry slot is one
    /// deep), which would otherwise leave the spinner wedged and submit
    /// disabled forever. A later `Done` for this turn lands on an
    /// already-settled mirror and is a no-op.
    pub fn abort(&mut self) {
        self.settle();
    }

    /// Fold one host event into the mirror — the *step* of the left fold the
    /// client runs over the event stream (the fold itself is `drain_socket`'s
    /// loop). Its dual is the host's *unfold*: from its session state the host
    /// produces the stream this consumes. Producer unfolds, consumer folds;
    /// the socket is the tape between them.
    pub fn fold(&mut self, ev: HostEvent) {
        match ev {
            HostEvent::Ready(info) => self.model = Some(info),
            HostEvent::Started { turn_id } => {
                self.in_flight = Some(turn_id);
                // Usually armed by push_user; arm here too so a client that
                // attaches mid-turn (serve's reconnect resumes the stream)
                // still renders the rest of the turn.
                if self.streaming.is_none() {
                    self.streaming = Some(String::new());
                }
            }
            HostEvent::Fragment {
                turn_id,
                channel: Channel::Answer,
                text,
            } => {
                // Arm on demand rather than drop: after a reconnect the carry
                // slot can redeliver a mid-turn fragment before any Started,
                // and a client that takes over a live turn (SRV-3 preemption)
                // must show it in flight — spinner, stop, submit-gate — though
                // it never saw Started. `get_or_insert` so a stale fragment
                // can't hijack a different live turn's id.
                self.in_flight.get_or_insert(turn_id);
                self.streaming
                    .get_or_insert_with(String::new)
                    .push_str(&text);
            }
            HostEvent::Fragment {
                turn_id,
                channel: Channel::Reasoning,
                text,
            } => {
                self.in_flight.get_or_insert(turn_id);
                self.streaming_reasoning.push_str(&text);
            }
            HostEvent::ToolNote {
                turn_id,
                kind,
                text,
            } => {
                self.in_flight.get_or_insert(turn_id);
                self.streaming_reasoning
                    .push_str(&tool_note_line(kind, &text));
            }
            HostEvent::RetractAnswer { chars, .. } => {
                // The streamed tail was narration ahead of a tool call — pull
                // it back out of the answer; it replays as reasoning. The
                // GUI's exact arithmetic: chars, never bytes (a multibyte
                // fragment truncated by bytes would panic or shear a char).
                if let Some(buf) = self.streaming.as_mut() {
                    let keep = buf.chars().count().saturating_sub(chars);
                    let cut = buf.char_indices().nth(keep).map_or(buf.len(), |(i, _)| i);
                    buf.truncate(cut);
                }
            }
            HostEvent::Image { bytes, name, .. } => match decode_rgba(&bytes) {
                Some(mut img) => {
                    img.name = name;
                    self.entries.push(Entry::Image(img));
                }
                // Unknown format: a named placeholder line, never an error
                // (the artifact exists; this build just doesn't render it).
                None => self
                    .entries
                    .push(Entry::Note(format!("[image {name} — not rendered here]"))),
            },
            HostEvent::Note(message) | HostEvent::Grants { message, .. } => {
                self.entries.push(Entry::Note(message))
            }
            HostEvent::Context { prompt_tokens } => self.prompt_tokens = Some(prompt_tokens),
            HostEvent::Done { turn_id, .. } => {
                // Settle unless this Done is for a turn we've already moved
                // past: at the reconnect seam a stale Done can be redelivered,
                // and it must not clear a newer turn's in-flight state.
                if self.in_flight.is_none_or(|live| live == turn_id) {
                    self.settle();
                }
            }
            HostEvent::Error { turn_id, message } => {
                // Same guard as Done: a stale Error can't disarm a newer turn.
                if self.in_flight.is_none_or(|live| live == turn_id) {
                    self.in_flight = None;
                    self.streaming = None;
                    self.streaming_reasoning.clear();
                }
                self.entries.push(Entry::Error(message));
            }
            HostEvent::Fatal(message) => {
                self.streaming = None;
                self.streaming_reasoning.clear();
                self.fatal = Some(message);
            }
            _ => {} // a future event variant this view predates
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use yatima_protocol::StopKind;

    fn fragment(text: &str) -> HostEvent {
        HostEvent::Fragment {
            turn_id: 1,
            channel: Channel::Answer,
            text: text.into(),
        }
    }

    fn done() -> HostEvent {
        HostEvent::Done {
            turn_id: 1,
            stop: StopKind::Eos,
        }
    }

    #[test]
    fn retraction_counts_chars_not_bytes() {
        // upholds: WEB-6 — retraction counts characters, never bytes.
        // The citing multibyte case: "héllo — ≥" is 9 chars but 14 bytes;
        // retracting by bytes would shear the em-dash or panic mid-char.
        let mut t = Transcript::default();
        t.push_user(1, "go");
        t.fold(fragment("héllo — ≥"));
        t.fold(HostEvent::RetractAnswer {
            turn_id: 1,
            chars: 3,
        });
        assert_eq!(t.streaming_answer(), Some("héllo "));
        // Retracting more than remains empties the buffer, never panics.
        t.fold(HostEvent::RetractAnswer {
            turn_id: 1,
            chars: 99,
        });
        assert_eq!(t.streaming_answer(), Some(""));
    }

    #[test]
    fn retraction_spans_fragment_boundaries() {
        // upholds: WEB-6 — retraction spans the whole accumulated answer.
        // Retraction applies to the accumulated buffer, not the last frame.
        let mut t = Transcript::default();
        t.push_user(1, "go");
        t.fold(fragment("aé"));
        t.fold(fragment("îo"));
        t.fold(HostEvent::RetractAnswer {
            turn_id: 1,
            chars: 3,
        });
        assert_eq!(t.streaming_answer(), Some("a"));
    }

    #[test]
    fn fully_retracted_turn_commits_nothing() {
        // upholds: WEB-6 — an empty answer commits nothing, never a blank bubble.
        // A turn whose narration was all pulled back (it replays as
        // reasoning) must not leave an empty Assistant bubble.
        let mut t = Transcript::default();
        t.push_user(1, "go");
        t.fold(fragment("calling a tool…"));
        t.fold(HostEvent::RetractAnswer {
            turn_id: 1,
            chars: 15,
        });
        t.fold(done());
        assert!(t.in_flight.is_none());
        assert!(
            !t.entries
                .iter()
                .any(|e| matches!(e, Entry::Assistant { .. })),
            "no Assistant entry for an empty answer"
        );
    }

    #[test]
    fn done_commits_answer_with_reasoning_fold() {
        // upholds: WEB-6 — a Done commits the answer that carried text, with
        // the reasoning fold alongside it.
        let mut t = Transcript::default();
        t.push_user(1, "go");
        t.fold(HostEvent::Fragment {
            turn_id: 1,
            channel: Channel::Reasoning,
            text: "thinking…".into(),
        });
        t.fold(HostEvent::ToolNote {
            turn_id: 1,
            kind: ToolNoteKind::Success,
            text: "read_page ok".into(),
        });
        t.fold(fragment("the answer"));
        t.fold(done());
        match t.entries.last() {
            Some(Entry::Assistant { answer, reasoning }) => {
                assert_eq!(answer, "the answer");
                let r = reasoning.as_deref().unwrap();
                assert!(r.contains("thinking…"), "{r}");
                assert!(r.contains("ok read_page ok"), "{r}");
            }
            _ => panic!("expected a committed Assistant entry"),
        }
    }

    #[test]
    fn fragment_before_started_is_kept_not_dropped() {
        // upholds: WEB-3 — an answer fragment before Started arms the turn.
        // The reconnect seam: serve's carry slot can redeliver a mid-turn
        // fragment before this client ever sees Started — arm on demand. And
        // arming means in flight: a client that takes over a live turn (SRV-3
        // preemption) must show the spinner, offer stop, and refuse a second
        // submit, though it never saw Started.
        let mut t = Transcript::default();
        t.fold(fragment("resumed mid-turn"));
        assert_eq!(t.streaming_answer(), Some("resumed mid-turn"));
        assert_eq!(t.in_flight, Some(1), "a bare fragment marks the turn live");
    }

    #[test]
    fn reasoning_or_tool_note_before_started_marks_the_turn_live() {
        // upholds: WEB-3 — reasoning or a tool-note arms the turn too.
        // The same seam, but the turn is mid-reasoning or mid-tool-call when
        // the client attaches: still in flight, even with no answer text yet.
        let mut t = Transcript::default();
        t.fold(HostEvent::Fragment {
            turn_id: 7,
            channel: Channel::Reasoning,
            text: "thinking…".into(),
        });
        assert_eq!(t.in_flight, Some(7));

        let mut t = Transcript::default();
        t.fold(HostEvent::ToolNote {
            turn_id: 7,
            kind: ToolNoteKind::Call,
            text: "plot(...)".into(),
        });
        assert_eq!(t.in_flight, Some(7));
    }

    #[test]
    fn abort_commits_partial_and_disarms() {
        // upholds: WEB-5 — stop settles locally; WEB-6 keeps what streamed.
        // The stop button: settle the turn locally without waiting for a Done
        // the seam may have dropped. Whatever streamed is kept (as on Done),
        // the spinner clears, and a late Done for the turn is a no-op.
        let mut t = Transcript::default();
        t.push_user(1, "go");
        t.fold(fragment("partial answer"));
        t.abort();
        assert!(t.in_flight.is_none(), "stop clears the spinner");
        assert!(
            matches!(t.entries.last(), Some(Entry::Assistant { answer, .. }) if answer == "partial answer"),
            "stop keeps what streamed"
        );
        let before = t.entries.len();
        t.fold(done()); // the host's Done finally arrives — already settled
        assert_eq!(
            t.entries.len(),
            before,
            "a late Done double-commits nothing"
        );
    }

    #[test]
    fn abort_from_a_wedged_spinner_commits_nothing_but_disarms() {
        // upholds: WEB-5 — stop is the escape hatch out of a wedged spinner.
        // The wedge the phone hit: in flight, nothing streamed yet, Done lost
        // at the seam. Stop must disarm without leaving a blank bubble.
        let mut t = Transcript::default();
        t.push_user(1, "go");
        t.abort();
        assert!(t.in_flight.is_none());
        assert!(
            !t.entries
                .iter()
                .any(|e| matches!(e, Entry::Assistant { .. })),
            "no empty Assistant bubble"
        );
    }

    #[test]
    fn stale_done_does_not_disarm_a_newer_turn() {
        // upholds: WEB-4 — a settle clears only the turn it names.
        // At the reconnect seam a Done for a finished turn can be redelivered
        // after the next turn is already live; it must not clear the new
        // turn's in-flight state (which would wrongly re-enable submit).
        let mut t = Transcript::default();
        t.push_user(5, "second turn");
        t.fold(fragment("streaming"));
        t.fold(HostEvent::Done {
            turn_id: 1, // a stale Done from the previous turn
            stop: StopKind::Eos,
        });
        assert_eq!(t.in_flight, Some(5), "the live turn stays armed");
        assert_eq!(
            t.streaming_answer(),
            Some("streaming"),
            "its buffer survives"
        );
    }

    #[test]
    fn ready_sets_the_model() {
        // upholds: WEB-1 — the mirror folds a Ready event into the model.
        let mut t = Transcript::default();
        assert!(t.model.is_none());
        t.fold(HostEvent::Ready(ModelInfo {
            label: "qwen32b".into(),
            arch: "Qwen2".into(),
            backend: "metal/BF16".into(),
            device: "gpu".into(),
            format: "Qwen".into(),
            sampling: "greedy".into(),
            max_tokens: 4096,
            context_length: Some(32768),
        }));
        assert_eq!(t.model.as_ref().map(|m| m.label.as_str()), Some("qwen32b"));
    }

    #[test]
    fn tool_note_line_marks_each_kind() {
        // upholds: HOST-4 — the marker vocabulary is this view's own.
        // The view's marker vocabulary (egui fonts lack ✓/✗, so ⚙/⚠ + words):
        // a call opens with the gear, warnings with the sign, success/failure
        // read as words, and an unmarked kind (Progress) is plain indented.
        assert!(tool_note_line(ToolNoteKind::Call, "plot(...)").contains("⚙ plot(...)"));
        assert!(tool_note_line(ToolNoteKind::Warning, "budget").contains("⚠ budget"));
        assert!(tool_note_line(ToolNoteKind::Success, "ok").contains("ok ok"));
        assert!(tool_note_line(ToolNoteKind::Failure, "boom").contains("failed: boom"));
        let progress = tool_note_line(ToolNoteKind::Progress, "step 2");
        assert!(!progress.contains('⚙') && progress.contains("step 2"));
    }

    #[test]
    fn png_decodes_jpeg_decodes_unknown_is_a_named_placeholder() {
        // upholds: WEB-6 — an artifact is an image or a named placeholder,
        // never an error.
        // A real 2×1 PNG and JPEG round-trip through the decode path; SVG
        // (not in the spike's formats) becomes a named placeholder line,
        // never an Error entry.
        let mut png = Vec::new();
        let mut jpg = Vec::new();
        let img = image::RgbaImage::from_raw(2, 1, vec![255, 0, 0, 255, 0, 255, 0, 255]).unwrap();
        let dynimg = image::DynamicImage::ImageRgba8(img);
        dynimg
            .write_to(&mut std::io::Cursor::new(&mut png), image::ImageFormat::Png)
            .unwrap();
        dynimg
            .to_rgb8() // JPEG has no alpha
            .write_to(
                &mut std::io::Cursor::new(&mut jpg),
                image::ImageFormat::Jpeg,
            )
            .unwrap();

        let mut t = Transcript::default();
        t.fold(HostEvent::Image {
            turn_id: 1,
            bytes: png,
            name: "plot.png".into(),
        });
        t.fold(HostEvent::Image {
            turn_id: 1,
            bytes: jpg,
            name: "photo.jpg".into(),
        });
        t.fold(HostEvent::Image {
            turn_id: 1,
            bytes: b"<svg xmlns='http://www.w3.org/2000/svg'/>".to_vec(),
            name: "figure.svg".into(),
        });

        match &t.entries[0] {
            Entry::Image(img) => {
                assert_eq!(img.name, "plot.png");
                assert_eq!(img.size, [2, 1]);
                assert_eq!(img.rgba.len(), img.size[0] * img.size[1] * 4);
            }
            _ => panic!("expected a decoded PNG"),
        }
        assert!(matches!(&t.entries[1], Entry::Image(i) if i.name == "photo.jpg"));
        match &t.entries[2] {
            Entry::Note(line) => assert!(line.contains("figure.svg"), "{line}"),
            _ => panic!("unknown format must be a named placeholder note"),
        }
        assert!(
            !t.entries.iter().any(|e| matches!(e, Entry::Error(_))),
            "an unrenderable image is never an error"
        );
    }

    #[test]
    fn status_events_drive_the_status_fields() {
        // upholds: WEB-1 — the mirror folds status events; WEB-4 — an Error
        // for the live turn disarms it.
        let mut t = Transcript::default();
        t.fold(HostEvent::Started { turn_id: 4 });
        assert_eq!(t.in_flight, Some(4));
        t.fold(HostEvent::Context { prompt_tokens: 777 });
        assert_eq!(t.prompt_tokens, Some(777));
        t.fold(HostEvent::Error {
            turn_id: 4,
            message: "boom".into(),
        });
        assert!(t.in_flight.is_none());
        assert!(matches!(t.entries.last(), Some(Entry::Error(m)) if m == "boom"));
        t.fold(HostEvent::Fatal("no model".into()));
        assert_eq!(t.fatal.as_deref(), Some("no model"));
    }

    #[test]
    fn notes_and_grants_render_as_muted_lines() {
        // upholds: WEB-1 — app-plane notes and grant reports fold to muted lines.
        let mut t = Transcript::default();
        t.fold(HostEvent::Note("about yatima".into()));
        t.fold(HostEvent::Grants {
            origins: vec!["https://example.com".into()],
            message: "granted https://example.com".into(),
        });
        assert!(matches!(&t.entries[0], Entry::Note(m) if m == "about yatima"));
        assert!(matches!(&t.entries[1], Entry::Note(m) if m.contains("granted")));
    }
}
