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
            Reasoned {
                reasoning: (!reasoning.is_empty()).then(|| reasoning.to_string()),
                answer,
            }
        }
    }
}

/// The answer only — `split_reasoning(text).answer`. The drop-in for callers
/// that don't need the reasoning trace.
pub fn strip_reasoning(text: &str) -> String {
    split_reasoning(text).answer
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
}
