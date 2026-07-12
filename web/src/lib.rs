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
/// Fold every [`HostEvent`] through [`Transcript::apply`]; the host's session
/// is truth, this is a view of it.
#[derive(Default)]
pub struct Transcript {
    pub entries: Vec<Entry>,
    /// The answer streaming in, if a turn is in flight (armed on submit, or
    /// on demand — see `apply`'s Fragment arm).
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

    /// Fold one host event into the mirror.
    pub fn apply(&mut self, ev: HostEvent) {
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
                channel: Channel::Answer,
                text,
                ..
            } => {
                // Arm on demand rather than drop: after a reconnect the carry
                // slot can redeliver a mid-turn fragment before any Started.
                self.streaming
                    .get_or_insert_with(String::new)
                    .push_str(&text);
            }
            HostEvent::Fragment {
                channel: Channel::Reasoning,
                text,
                ..
            } => self.streaming_reasoning.push_str(&text),
            HostEvent::ToolNote { kind, text, .. } => self
                .streaming_reasoning
                .push_str(&tool_note_line(kind, &text)),
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
            HostEvent::Done { .. } => {
                self.in_flight = None;
                // Commit a reply only if the answer carried text: a fully
                // retracted turn streams none, and committing an empty
                // Assistant entry would render a blank bubble.
                let reasoning = std::mem::take(&mut self.streaming_reasoning);
                if let Some(buf) = self.streaming.take() {
                    if !buf.trim().is_empty() {
                        let reasoning =
                            (!reasoning.trim().is_empty()).then(|| reasoning.trim().to_string());
                        self.entries.push(Entry::Assistant {
                            answer: buf,
                            reasoning,
                        });
                    }
                }
            }
            HostEvent::Error { message, .. } => {
                self.in_flight = None;
                self.streaming = None;
                self.streaming_reasoning.clear();
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
        // The citing multibyte case: "héllo — ≥" is 9 chars but 14 bytes;
        // retracting by bytes would shear the em-dash or panic mid-char.
        let mut t = Transcript::default();
        t.push_user(1, "go");
        t.apply(fragment("héllo — ≥"));
        t.apply(HostEvent::RetractAnswer {
            turn_id: 1,
            chars: 3,
        });
        assert_eq!(t.streaming_answer(), Some("héllo "));
        // Retracting more than remains empties the buffer, never panics.
        t.apply(HostEvent::RetractAnswer {
            turn_id: 1,
            chars: 99,
        });
        assert_eq!(t.streaming_answer(), Some(""));
    }

    #[test]
    fn retraction_spans_fragment_boundaries() {
        // Retraction applies to the accumulated buffer, not the last frame.
        let mut t = Transcript::default();
        t.push_user(1, "go");
        t.apply(fragment("aé"));
        t.apply(fragment("îo"));
        t.apply(HostEvent::RetractAnswer {
            turn_id: 1,
            chars: 3,
        });
        assert_eq!(t.streaming_answer(), Some("a"));
    }

    #[test]
    fn fully_retracted_turn_commits_nothing() {
        // A turn whose narration was all pulled back (it replays as
        // reasoning) must not leave an empty Assistant bubble.
        let mut t = Transcript::default();
        t.push_user(1, "go");
        t.apply(fragment("calling a tool…"));
        t.apply(HostEvent::RetractAnswer {
            turn_id: 1,
            chars: 15,
        });
        t.apply(done());
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
        let mut t = Transcript::default();
        t.push_user(1, "go");
        t.apply(HostEvent::Fragment {
            turn_id: 1,
            channel: Channel::Reasoning,
            text: "thinking…".into(),
        });
        t.apply(HostEvent::ToolNote {
            turn_id: 1,
            kind: ToolNoteKind::Success,
            text: "read_page ok".into(),
        });
        t.apply(fragment("the answer"));
        t.apply(done());
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
        // The reconnect seam: serve's carry slot can redeliver a mid-turn
        // fragment before this client ever sees Started — arm on demand.
        let mut t = Transcript::default();
        t.apply(fragment("resumed mid-turn"));
        assert_eq!(t.streaming_answer(), Some("resumed mid-turn"));
    }

    #[test]
    fn png_decodes_jpeg_decodes_unknown_is_a_named_placeholder() {
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
        t.apply(HostEvent::Image {
            turn_id: 1,
            bytes: png,
            name: "plot.png".into(),
        });
        t.apply(HostEvent::Image {
            turn_id: 1,
            bytes: jpg,
            name: "photo.jpg".into(),
        });
        t.apply(HostEvent::Image {
            turn_id: 1,
            bytes: b"<svg xmlns='http://www.w3.org/2000/svg'/>".to_vec(),
            name: "figure.svg".into(),
        });

        match &t.entries[0] {
            Entry::Image(img) => {
                assert_eq!(img.name, "plot.png");
                assert_eq!(img.size, [2, 1]);
                assert_eq!(img.rgba.len(), 2 * 1 * 4);
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
        let mut t = Transcript::default();
        t.apply(HostEvent::Started { turn_id: 4 });
        assert_eq!(t.in_flight, Some(4));
        t.apply(HostEvent::Context { prompt_tokens: 777 });
        assert_eq!(t.prompt_tokens, Some(777));
        t.apply(HostEvent::Error {
            turn_id: 4,
            message: "boom".into(),
        });
        assert!(t.in_flight.is_none());
        assert!(matches!(t.entries.last(), Some(Entry::Error(m)) if m == "boom"));
        t.apply(HostEvent::Fatal("no model".into()));
        assert_eq!(t.fatal.as_deref(), Some("no model"));
    }

    #[test]
    fn notes_and_grants_render_as_muted_lines() {
        let mut t = Transcript::default();
        t.apply(HostEvent::Note("about yatima".into()));
        t.apply(HostEvent::Grants {
            origins: vec!["https://example.com".into()],
            message: "granted https://example.com".into(),
        });
        assert!(matches!(&t.entries[0], Entry::Note(m) if m == "about yatima"));
        assert!(matches!(&t.entries[1], Entry::Note(m) if m.contains("granted")));
    }
}
