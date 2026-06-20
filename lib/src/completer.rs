//! The model seam.
//!
//! [`Completer`] is the one thing the agent loop needs from a model: turn a
//! prompt into text, stopping at EOS / `max_tokens` / a caller-supplied stop
//! string. The real [`crate::Engine`] implements it over `generate_with`; tests
//! implement it with canned outputs (a *scripted* completer), so the agent's
//! laws are provable with no GPU. It is also the engine-swap seam — mistral.rs
//! or llama.cpp would be another `Completer`.

use crate::{Engine, GenOpts, StopReason};
use anyhow::Result;
use std::ops::ControlFlow;

/// A model completion: the generated `text` and why generation stopped.
///
/// When `complete` is given stop strings, `text` **includes** the matched stop
/// string, so a [`crate::ToolCallCodec`] sees the whole block (e.g. the closing
/// `</tool_call>`).
#[derive(Debug, Clone)]
pub struct Completion {
    pub text: String,
    pub stop: StopReason,
}

/// The model as a function: prompt → completion.
pub trait Completer {
    /// Generate from `prompt` under `opts`, stopping at EOS, `max_tokens`, or as
    /// soon as any string in `stops` appears in the output. The returned
    /// [`Completion::text`] includes the matched stop string.
    fn complete(&mut self, prompt: &str, opts: &GenOpts, stops: &[String]) -> Result<Completion>;
}

/// The real engine as a [`Completer`]: a thin adapter over `generate_with` that
/// accumulates decoded text and breaks at the first stop string, **including**
/// that marker in the result (so a `ToolCallCodec` sees the whole block). This
/// is the seam an alternate backend (mistral.rs / llama.cpp) would also fill.
impl Completer for Engine {
    fn complete(&mut self, prompt: &str, opts: &GenOpts, stops: &[String]) -> Result<Completion> {
        let (text, generation) =
            self.generate_with(prompt, opts, String::new(), |mut acc, fragment| {
                acc.push_str(fragment);
                match first_stop_end(&acc, stops) {
                    Some(end) => {
                        acc.truncate(end);
                        Ok(ControlFlow::Break(acc))
                    }
                    None => Ok(ControlFlow::Continue(acc)),
                }
            })?;
        Ok(Completion {
            text,
            stop: generation.stop,
        })
    }
}

/// The earliest byte offset *past* any stop string in `text` (so truncating to
/// it keeps the marker). `None` if no stop string is present. Searching the
/// whole accumulator each step lets a marker span fragment boundaries.
fn first_stop_end(text: &str, stops: &[String]) -> Option<usize> {
    stops
        .iter()
        .filter(|s| !s.is_empty())
        .filter_map(|s| text.find(s.as_str()).map(|i| i + s.len()))
        .min()
}

#[cfg(test)]
mod tests {
    use super::first_stop_end;

    #[test]
    fn stop_end_includes_the_marker_and_picks_earliest() {
        let stops = vec!["</tool_call>".to_string(), "STOP".to_string()];
        let text = "call <tool_call>{}</tool_call> trailing";
        let end = first_stop_end(text, &stops).unwrap();
        assert_eq!(&text[..end], "call <tool_call>{}</tool_call>");
        assert!(first_stop_end("no markers here", &stops).is_none());
        assert!(first_stop_end("anything", &[]).is_none());
    }
}
