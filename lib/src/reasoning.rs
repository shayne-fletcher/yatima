//! The reasoning channel: separating a model's chain-of-thought from its answer.
//!
//! Reasoning models (Kimi-Dev, Qwen3, the DeepSeek-R1 family) emit an inline
//! *thinking* span before their answer, wrapped in model-specific markers. That
//! span is **ephemeral**: it must not be surfaced to the user, and — crucially —
//! must not enter the transcript that is re-rendered into the next prompt, or the
//! model re-reads its own stale reasoning off-distribution (trained chat
//! templates drop prior think spans). [`split_reasoning`] performs that split at
//! the completion→turn boundary (REASON-1).
//!
//! The split is the identity when no marker is present, so it is safe for any
//! model or format — a non-reasoning model's output passes through unchanged.

/// One reasoning-marker dialect: the open/close pair a model wraps its
/// chain-of-thought in.
struct Dialect {
    open: &'static str,
    close: &'static str,
}

/// Every dialect we recognize. A model emits at most one; the spellings are
/// unambiguous and non-overlapping, so scanning all of them is safe.
const DIALECTS: &[Dialect] = &[
    // Qwen3, DeepSeek-R1 distills, and the de-facto generic spelling.
    Dialect {
        open: "<think>",
        close: "</think>",
    },
    // Kimi (Moonshot) — special tokens, not ASCII angle brackets.
    Dialect {
        open: "◁think▷",
        close: "◁/think▷",
    },
];

/// A completion split into its (optional) reasoning span and the answer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Reasoned {
    /// The chain-of-thought between the markers, trimmed — `None` when the
    /// completion carried no recognized reasoning span.
    pub reasoning: Option<String>,
    /// The surfaced answer: everything after the reasoning close marker,
    /// trimmed. Equal to the whole (trimmed) input when there is no marker.
    pub answer: String,
}

/// Split a raw completion into reasoning + answer (REASON-1).
///
/// The answer is everything after the **last** recognized close marker; the span
/// before it (minus the open marker, if present) is the reasoning. With no
/// recognized marker this is the identity: the whole trimmed text is the answer
/// and `reasoning` is `None`. An unterminated open marker (no close) is also
/// treated as no split — content is never lost to a half-emitted marker.
///
/// A trailing tool call (`<tool_call>…`) sits after the close marker, so it
/// stays in `answer` and the agent codec still parses it.
pub fn split_reasoning(text: &str) -> Reasoned {
    // Pick the dialect whose close marker appears latest — a model uses one
    // dialect, and "latest close" matches the old strip-to-last-`</think>`
    // behavior when a span itself contains the marker text.
    let split = DIALECTS
        .iter()
        .filter_map(|d| text.rfind(d.close).map(|close_at| (d, close_at)))
        .max_by_key(|(_, close_at)| *close_at);

    match split {
        None => Reasoned {
            reasoning: None,
            answer: text.trim().to_string(),
        },
        Some((dialect, close_at)) => {
            let answer = text[close_at + dialect.close.len()..].trim().to_string();
            let before = &text[..close_at];
            // Drop the open marker if present; whatever precedes the close is the
            // reasoning, even if the open was never emitted.
            let reasoning = match before.find(dialect.open) {
                Some(open_at) => &before[open_at + dialect.open.len()..],
                None => before,
            }
            .trim();
            let reasoned = Reasoned {
                reasoning: (!reasoning.is_empty()).then(|| reasoning.to_string()),
                answer,
            };
            tracing::trace!(
                dialect = dialect.close,
                reasoning_chars = reasoned.reasoning.as_deref().map_or(0, str::len),
                answer_chars = reasoned.answer.len(),
                "reasoning split"
            );
            reasoned
        }
    }
}

/// The answer only — `split_reasoning(text).answer`. The drop-in for callers
/// that don't need the reasoning trace.
pub fn strip_reasoning(text: &str) -> String {
    split_reasoning(text).answer
}

/// Which channel a streamed span belongs to.
///
/// Intentionally binary: it is the *complete* partition for what streams today
/// (the chat path, which has no tools — reasoning vs. answer is everything). A
/// tool call is not a stream channel here because the only tool consumer, the
/// [`Agent`](crate::Agent), runs non-streaming and extracts calls downstream via
/// its [`ToolCallCodec`](crate::ToolCallCodec) (PROTO-1). If the agent ever
/// streams, or a harmony/gpt-oss-style multi-channel model is enabled, this enum
/// grows a `ToolCall` (and perhaps `Commentary`) arm — a non-breaking addition.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Channel {
    /// The model's chain-of-thought (between reasoning markers).
    Reasoning,
    /// The surfaced answer.
    Answer,
}

/// The streaming dual of [`split_reasoning`] (REASON-1): an incremental
/// classifier that routes each fragment of a *streamed* completion to
/// [`Channel::Reasoning`] or [`Channel::Answer`] as it arrives, so a live UI can
/// fold or dim the chain-of-thought. It recognizes the same dialects and handles
/// a marker that straddles fragment boundaries (split across two `push` calls).
/// Marker text itself is control, never emitted.
pub struct ReasoningSplitter {
    in_reasoning: bool,
    buf: String,
}

impl Default for ReasoningSplitter {
    fn default() -> ReasoningSplitter {
        ReasoningSplitter::new()
    }
}

impl ReasoningSplitter {
    /// A splitter for output that *begins in the answer* and enters reasoning on
    /// an open marker — the usual case (Kimi/Qwen3 emit `◁think▷`/`<think>`
    /// first).
    pub fn new() -> ReasoningSplitter {
        ReasoningSplitter {
            in_reasoning: false,
            buf: String::new(),
        }
    }

    /// A splitter for output that *begins inside the reasoning block* — used when
    /// the prompt pre-seeds the opener (DeepSeek's `<｜Assistant｜><think>` cue),
    /// so the stream's first marker is the close. See
    /// [`ChatFormat::pre_seeds_reasoning`](crate::ChatFormat::pre_seeds_reasoning).
    pub fn seeded() -> ReasoningSplitter {
        ReasoningSplitter {
            in_reasoning: true,
            buf: String::new(),
        }
    }

    /// Feed the next raw fragment; `emit(channel, text)` is called for each
    /// classified piece (zero or more times).
    pub fn push(&mut self, fragment: &str, mut emit: impl FnMut(Channel, &str)) {
        self.buf.push_str(fragment);
        self.drain(&mut emit);
    }

    /// Flush any buffered tail at end of stream. A partial marker that never
    /// completed is treated as content on the current channel.
    pub fn finish(mut self, mut emit: impl FnMut(Channel, &str)) {
        self.drain(&mut emit);
        if !self.buf.is_empty() {
            emit(self.channel(), &self.buf);
            self.buf.clear();
        }
    }

    fn channel(&self) -> Channel {
        if self.in_reasoning {
            Channel::Reasoning
        } else {
            Channel::Answer
        }
    }

    fn drain(&mut self, emit: &mut impl FnMut(Channel, &str)) {
        loop {
            // The earliest complete marker — open *or* close — controls the
            // channel. A marker *sets* state (open→reasoning, close→answer)
            // rather than toggling, so a stray or duplicated marker (e.g. a model
            // that emits `</think>` twice while degenerating) is always consumed,
            // never leaked into a channel.
            let hit = all_markers()
                .filter_map(|(text, opens)| self.buf.find(text).map(|i| (i, text, opens)))
                .min_by_key(|(i, ..)| *i);
            match hit {
                Some((i, text, opens)) => {
                    if i > 0 {
                        let ch = self.channel();
                        emit(ch, &self.buf[..i]);
                    }
                    self.buf = self.buf.split_off(i + text.len());
                    let was = self.in_reasoning;
                    self.in_reasoning = opens;
                    tracing::trace!(
                        marker = text,
                        opens,
                        was_reasoning = was,
                        now_reasoning = self.in_reasoning,
                        "reasoning channel marker"
                    );
                }
                None => {
                    // No complete marker: emit all but a tail that could be the
                    // start of one, so a boundary-straddling marker is caught on
                    // the next push.
                    let keep = held_back_len(&self.buf);
                    let upto = self.buf.len() - keep;
                    if upto > 0 {
                        let ch = self.channel();
                        emit(ch, &self.buf[..upto]);
                        self.buf.drain(..upto);
                    }
                    break;
                }
            }
        }
    }
}

/// Every marker the stream watches — both ends of every dialect, paired with
/// whether it *opens* a reasoning span — derived from the single [`DIALECTS`]
/// source so the batch and streaming splitters never drift.
fn all_markers() -> impl Iterator<Item = (&'static str, bool)> {
    DIALECTS
        .iter()
        .flat_map(|d| [(d.open, true), (d.close, false)])
}

/// Bytes to hold back at the tail of `buf`: the longest suffix that is a proper
/// prefix of any marker (at a marker char boundary, so the kept split is always
/// a valid `str` boundary), in case the marker completes in the next fragment.
/// No complete marker is present here (the caller already searched), so the
/// overlap is always shorter than the marker.
fn held_back_len(buf: &str) -> usize {
    let mut best = 0;
    for (m, _opens) in all_markers() {
        let mut k = m.len().min(buf.len());
        while k > best {
            if m.is_char_boundary(k) && buf.as_bytes().ends_with(&m.as_bytes()[..k]) {
                best = k;
                break;
            }
            k -= 1;
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_marker_is_the_identity() {
        let r = split_reasoning("no markers here");
        assert_eq!(r.reasoning, None);
        assert_eq!(r.answer, "no markers here");
        // trims like the old strip_think.
        assert_eq!(strip_reasoning("  padded  "), "padded");
    }

    #[test]
    fn splits_the_think_dialect() {
        let r = split_reasoning("<think>weighing it</think>\nthe answer");
        assert_eq!(r.reasoning.as_deref(), Some("weighing it"));
        assert_eq!(r.answer, "the answer");
    }

    #[test]
    fn splits_the_kimi_dialect() {
        let r = split_reasoning("◁think▷let me see◁/think▷ 4");
        assert_eq!(r.reasoning.as_deref(), Some("let me see"));
        assert_eq!(r.answer, "4");
    }

    #[test]
    fn keeps_text_after_the_last_close() {
        // Matches the old strip_think: split on the last close marker.
        let r = split_reasoning("a</think>b</think>final");
        assert_eq!(r.answer, "final");
        assert_eq!(r.reasoning.as_deref(), Some("a</think>b"));
    }

    #[test]
    fn unterminated_open_is_left_intact() {
        let r = split_reasoning("<think>still thinking");
        assert_eq!(r.reasoning, None);
        assert_eq!(r.answer, "<think>still thinking");
    }

    #[test]
    fn close_without_open_still_yields_an_answer() {
        // Some setups suppress the open marker; treat everything before the
        // close as reasoning rather than leaking it into the answer.
        let r = split_reasoning("hidden reasoning</think>visible answer");
        assert_eq!(r.reasoning.as_deref(), Some("hidden reasoning"));
        assert_eq!(r.answer, "visible answer");
    }

    #[test]
    fn answer_retains_a_trailing_tool_call() {
        // The agent must still parse a tool call that follows the reasoning.
        let r = split_reasoning("<think>which file?</think>\n<tool_call>\n{}\n</tool_call>");
        assert_eq!(r.answer, "<tool_call>\n{}\n</tool_call>");
        assert!(r.answer.contains("<tool_call>"));
    }

    #[test]
    fn empty_reasoning_span_is_none() {
        let r = split_reasoning("<think></think>answer");
        assert_eq!(r.reasoning, None);
        assert_eq!(r.answer, "answer");
    }

    /// Run a splitter over `fragments`, collecting per-channel output.
    fn stream(mut s: ReasoningSplitter, fragments: &[&str]) -> (String, String) {
        let mut reasoning = String::new();
        let mut answer = String::new();
        let mut sink = |ch: Channel, t: &str| match ch {
            Channel::Reasoning => reasoning.push_str(t),
            Channel::Answer => answer.push_str(t),
        };
        for f in fragments {
            s.push(f, &mut sink);
        }
        s.finish(&mut sink);
        (reasoning, answer)
    }

    #[test]
    fn splitter_classifies_a_single_fragment() {
        let (r, a) = stream(ReasoningSplitter::new(), &["<think>reason</think>answer"]);
        assert_eq!(r, "reason");
        assert_eq!(a, "answer");
    }

    #[test]
    fn splitter_handles_markers_across_boundaries() {
        // The open and close markers are each split across pushes.
        let (r, a) = stream(
            ReasoningSplitter::new(),
            &["<th", "ink>hi the", "re</thi", "nk>by", "e"],
        );
        assert_eq!(r, "hi there");
        assert_eq!(a, "bye");
    }

    #[test]
    fn splitter_seeded_starts_in_reasoning() {
        // DeepSeek pre-seeds `<think>`, so the stream opens mid-thought and the
        // first marker is the close.
        let (r, a) = stream(
            ReasoningSplitter::seeded(),
            &["thinking…", "</think>", "the answer"],
        );
        assert_eq!(r, "thinking…");
        assert_eq!(a, "the answer");
    }

    #[test]
    fn splitter_handles_the_kimi_dialect() {
        let (r, a) = stream(ReasoningSplitter::new(), &["◁think▷w◁/think▷4"]);
        assert_eq!(r, "w");
        assert_eq!(a, "4");
    }

    #[test]
    fn splitter_with_no_markers_is_all_answer() {
        let (r, a) = stream(ReasoningSplitter::new(), &["just ", "an ", "answer"]);
        assert_eq!(r, "");
        assert_eq!(a, "just an answer");
    }

    #[test]
    fn splitter_flushes_an_unterminated_partial_marker() {
        // A dangling `<thi` at end of stream is content, not a swallowed marker.
        let (r, a) = stream(ReasoningSplitter::new(), &["answer <thi"]);
        assert_eq!(r, "");
        assert_eq!(a, "answer <thi");
    }

    /// Drive the splitter one *character* at a time — the most adversarial
    /// fragmentation (every marker is split maximally) — and never leak a marker.
    fn stream_char_by_char(s: ReasoningSplitter, text: &str) -> (String, String) {
        let frags: Vec<String> = text.chars().map(|c| c.to_string()).collect();
        let refs: Vec<&str> = frags.iter().map(String::as_str).collect();
        stream(s, &refs)
    }

    #[test]
    fn splitter_seeded_consumes_close_amid_real_text() {
        // Regression: a DeepSeek-style stream (seeded, close marker after real
        // punctuation `]\n`) fed char-by-char must consume `</think>`, not leak
        // it into the answer.
        let raw = "reasoning\n\\boxed{3}\n]\n</think>\n\nThe answer is 3.";
        let (r, a) = stream_char_by_char(ReasoningSplitter::seeded(), raw);
        assert!(!a.contains("think"), "marker leaked into answer: {a:?}");
        assert!(!r.contains("think"), "marker leaked into reasoning: {r:?}");
        // The stream preserves whitespace (live display); trim for the compare.
        assert_eq!(a.trim(), "The answer is 3.");
        assert!(r.contains("\\boxed{3}"));
    }

    #[test]
    fn splitter_consumes_a_stray_or_duplicate_close() {
        // Regression for the live bug: a degenerating model emitted `</think>`
        // twice. With a toggle, the second close (seen while already in the
        // answer) leaked; set-semantics consume every marker. Reproduced
        // synthetically, no model needed.
        let raw = "think one</think>answer one</think>answer two";
        let (r, a) = stream_char_by_char(ReasoningSplitter::seeded(), raw);
        assert!(!a.contains("think"), "stray close leaked: {a:?}");
        assert_eq!(r, "think one");
        assert_eq!(a, "answer oneanswer two");
    }

    #[test]
    fn splitter_ignores_a_stray_open_while_reasoning() {
        // The dual: a second open while already reasoning is consumed, not leaked.
        let raw = "<think>a<think>b</think>done";
        let (r, a) = stream_char_by_char(ReasoningSplitter::new(), raw);
        assert!(!r.contains("think") && !a.contains("think"));
        assert_eq!(r, "ab");
        assert_eq!(a, "done");
    }

    #[test]
    fn splitter_open_then_close_char_by_char() {
        // The new() path under the same adversarial fragmentation.
        let raw = "<think>weigh it</think>final";
        let (r, a) = stream_char_by_char(ReasoningSplitter::new(), raw);
        assert_eq!(r, "weigh it");
        assert_eq!(a, "final");
    }
}
