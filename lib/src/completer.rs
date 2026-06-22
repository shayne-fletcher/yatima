//! The model seam.
//!
//! [`Completer`] is the one thing the agent loop needs from a model: turn a
//! prompt into text, stopping at EOS / `max_tokens` / a caller-supplied stop
//! string. The real [`crate::Engine`] implements it over `generate_with`; tests
//! implement it with canned outputs (a *scripted* completer), so the agent's
//! laws are provable with no GPU. It is also the engine-swap seam — mistral.rs,
//! llama.cpp, or a **remote/HTTP model** would be another `Completer`.
//!
//! # The seam is async, and `Send` is inferred per impl (CMP-1)
//!
//! `Completer` is an **async** trait so the seam generalizes beyond local
//! blocking compute to a remote model that is fundamentally async I/O. Crucially
//! it uses **native `async fn` in trait** (stable since Rust 1.75), *not* the
//! `async_trait` crate. That choice is the whole point, and it is non-obvious —
//! read this before changing it:
//!
//! - **`Send` is inferred per implementation, not fixed at the trait.** With
//!   native `async fn`, each impl's returned future is exactly as `Send` as the
//!   state it captures. The local [`Engine`] owns GPU handles
//!   (`Box<dyn CausalLm>`, no `Send` bound) so its completion future is naturally
//!   `!Send`; a remote/HTTP completer captures only `Send` state so its future is
//!   naturally `Send`. We never write `?Send` (which would strip `Send` from
//!   *every* completer, penalising the remote case) and never force `Engine:
//!   Send` (a lie about the rented engine that buys nothing — it is one-
//!   generation-at-a-time and `block_in_place`-pinned regardless). The decision
//!   lives where the truth is: each impl.
//!
//! - **Each impl owns the *operational shape* of the effect.** `complete` is the
//!   effect boundary; the impl chooses how it is discharged. The local engine
//!   runs its synchronous decode under [`crate::run_blocking`] — native `async`
//!   alone does **not** make candle inference non-blocking; without
//!   `run_blocking` the sync decode would stall the executor (RT-1). A remote
//!   completer instead `.await`s network I/O and blocks no thread. Callers
//!   (`Agent`, `ChatSession`) just `.await` and assume nothing about whether
//!   completion is CPU- or I/O-bound.
//!
//! - **`Completer` is intentionally not `dyn`-compatible.** Native `async fn` in
//!   trait cannot be made into a trait object, and that is fine: every consumer
//!   is generic and monomorphic (`Agent<C: Completer>`, `ChatSession<C>`), never
//!   `dyn Completer`. (Contrast [`crate::Tool`], which *is* stored as
//!   `Arc<dyn Tool>` and whose futures are `tokio::spawn`ed across tasks — so it
//!   correctly uses `#[async_trait]` with the default `Send` bound. Same project,
//!   two async traits, two mechanisms, for principled reasons.)
//!
//! - **Sync callers still bridge through [`crate::run_blocking`]'s sibling,
//!   `runtime::block_on`.** The synchronous shims (`ChatSession::turn`,
//!   `Agent::run`) wrap the async primitive via the one runtime bridge, so a
//!   sync call from inside a current-thread runtime hits the directed panic
//!   (RT-1), not a deadlock.
//!
//! The cost is one lint: a public trait with native `async fn` trips
//! `clippy::async_fn_in_trait` (callers cannot name a `Send` bound on the
//! method). We `#[allow]` it deliberately — *because* completion futures are
//! never spawned, we intentionally impose no global `Send` bound. If a future
//! engine-actor ever needs to move a completion across threads, that is when to
//! reach for return-type-notation or `trait_variant`, not now.

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

/// The model as a function: prompt → completion. See the module docs (CMP-1)
/// for why this is a native `async fn` trait with per-impl `Send`.
///
// `async_fn_in_trait`: deliberate. Completion futures are awaited inline and
// never spawned across threads, so we intentionally impose no global `Send`
// bound — `Send` is inferred per impl (the local engine is `!Send`, a remote
// completer is `Send`). This is the design, not an oversight (CMP-1).
#[allow(async_fn_in_trait)]
pub trait Completer {
    /// Generate from `prompt` under `opts`, stopping at EOS, `max_tokens`, or as
    /// soon as any string in `stops` appears in the output. The returned
    /// [`Completion::text`] includes the matched stop string.
    ///
    /// The impl owns the operational shape: the local [`Engine`] runs sync
    /// decode under [`crate::run_blocking`]; a remote completer `.await`s I/O.
    async fn complete(
        &mut self,
        prompt: &str,
        opts: &GenOpts,
        stops: &[String],
    ) -> Result<Completion>;

    /// Like [`complete`](Completer::complete), but delivers text to `on_token` as
    /// it is produced (for live UIs / streaming chat). The default emits the whole
    /// completion once — so every `Completer` works unchanged; backends that can
    /// stream (e.g. [`Engine`]) override this to forward fragments incrementally.
    async fn complete_streaming(
        &mut self,
        prompt: &str,
        opts: &GenOpts,
        stops: &[String],
        on_token: &mut dyn FnMut(&str),
    ) -> Result<Completion> {
        let completion = self.complete(prompt, opts, stops).await?;
        on_token(&completion.text);
        Ok(completion)
    }
}

/// The real engine as a [`Completer`]: a thin adapter over `generate_with` that
/// accumulates decoded text and breaks at the first stop string, **including**
/// that marker in the result (so a `ToolCallCodec` sees the whole block). This
/// is the seam an alternate backend (mistral.rs / llama.cpp) would also fill.
impl Completer for Engine {
    async fn complete(
        &mut self,
        prompt: &str,
        opts: &GenOpts,
        stops: &[String],
    ) -> Result<Completion> {
        // candle decode is synchronous compute; `run_blocking` keeps it off the
        // async executor's critical path (RT-1). Native `async fn` does not make
        // it non-blocking — the impl owns that (CMP-1).
        crate::run_blocking(|| {
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
        })
    }

    async fn complete_streaming(
        &mut self,
        prompt: &str,
        opts: &GenOpts,
        stops: &[String],
        on_token: &mut dyn FnMut(&str),
    ) -> Result<Completion> {
        // Stream each newly-committed slice as it is decoded; never emit past a
        // stop marker (truncate to it, like `complete`). Runs under run_blocking
        // (RT-1); `on_token` is called from within the blocking section.
        crate::run_blocking(|| {
            let mut emitted = 0usize;
            let (text, generation) =
                self.generate_with(prompt, opts, String::new(), |mut acc, fragment| {
                    acc.push_str(fragment);
                    let flow = match first_stop_end(&acc, stops) {
                        Some(end) => {
                            acc.truncate(end);
                            ControlFlow::Break(())
                        }
                        None => ControlFlow::Continue(()),
                    };
                    if acc.len() > emitted {
                        on_token(&acc[emitted..]);
                        emitted = acc.len();
                    }
                    Ok(match flow {
                        ControlFlow::Break(()) => ControlFlow::Break(acc),
                        ControlFlow::Continue(()) => ControlFlow::Continue(acc),
                    })
                })?;
            Ok(Completion {
                text,
                stop: generation.stop,
            })
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
