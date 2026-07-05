//! Multi-turn chat as a reusable, embeddable session.
//!
//! [`ChatSession`] is the conversation fold one level simpler than [`Agent`]: no
//! tools, just instruction-following with memory. It holds the transcript and,
//! each [`turn`](ChatSession::turn), re-renders the *whole* history through a
//! [`PromptTemplate`] and asks the [`Completer`] for the next answer — so memory
//! lives in the prompt, not the model (the engine stays stateless per call).
//!
//! It mirrors [`Agent`]'s shape (generic over [`Completer`]), so it is testable
//! with a scripted completer and embeddable over a real [`crate::Engine`]:
//!
//! ```no_run
//! use yatima_lib::{ChatSession, ChatMlTemplate, Engine, device};
//! # fn main() -> anyhow::Result<()> {
//! let mut engine = Engine::load(std::path::Path::new("/models/qwen"), device(false)?)?;
//! let mut chat = ChatSession::new(&mut engine, ChatMlTemplate).with_system("Be brief.");
//! println!("{}", chat.turn("My name is Ada.")?);
//! println!("{}", chat.turn("What is my name?")?); // recalls "Ada"
//! # Ok(()) }
//! ```
//! It borrows the completer (`&mut`), like [`Agent`], so one loaded `Engine` can
//! back several sessions in turn.
//!
//! [`Agent`]: crate::Agent

use crate::completer::Completer;
use crate::reasoning::{split_reasoning, Reasoned};
use crate::template::PromptTemplate;
use crate::transcript::{Role, Turn};
use crate::{Cancel, GenOpts, StopReason};
use anyhow::Result;

/// A stateful, tool-free conversation over a [`Completer`] and a
/// [`PromptTemplate`]. Borrows the completer (like [`crate::Agent`]); owns the
/// transcript that `turn` advances.
pub struct ChatSession<'a, C: Completer, T: PromptTemplate> {
    completer: &'a mut C,
    template: T,
    opts: GenOpts,
    turns: Vec<Turn>,
    /// How many leading turns are the seeded system prompt (kept on `reset`).
    system_len: usize,
    /// Why the most recent turn stopped (for run metadata); `None` before any.
    last_stop: Option<StopReason>,
    /// The most recent reply's reasoning span, if it was a reasoning model;
    /// `None` otherwise. Kept out of the transcript (REASON-1) but surfaced here.
    last_reasoning: Option<String>,
    /// Tokens in the most recent rendered prompt (for a host's context meter),
    /// if the completer exposes a tokenizer; `None` otherwise.
    last_prompt_tokens: Option<usize>,
    /// The most recent *degenerate* answer, held outside the transcript
    /// (CHAT-2) so the turn can still hand its text back as `&str`.
    uncommitted: String,
}

impl<'a, C: Completer, T: PromptTemplate> ChatSession<'a, C, T> {
    /// Start a session. Default generation options (greedy); add a system prompt
    /// with [`with_system`](ChatSession::with_system).
    pub fn new(completer: &'a mut C, template: T) -> ChatSession<'a, C, T> {
        ChatSession {
            completer,
            template,
            opts: GenOpts::default(),
            turns: Vec::new(),
            system_len: 0,
            last_stop: None,
            last_reasoning: None,
            last_prompt_tokens: None,
            uncommitted: String::new(),
        }
    }

    /// Seed a system instruction that persists across turns (and `reset`).
    pub fn with_system(mut self, system: impl Into<String>) -> ChatSession<'a, C, T> {
        self.turns.insert(
            0,
            Turn {
                role: Role::System,
                content: system.into(),
            },
        );
        self.system_len = self.turns.len();
        self
    }

    /// Override the per-turn generation options (default greedy).
    pub fn with_opts(mut self, opts: GenOpts) -> ChatSession<'a, C, T> {
        self.opts = opts;
        self
    }

    /// Send a user message and get the assistant's reply (async). Appends both
    /// to the transcript so later turns remember them; the reply is rendered from
    /// the *whole* history (no stop strings — chat has no tools). This is the
    /// primitive; [`turn`](ChatSession::turn) is its sync shim.
    pub async fn turn_async(&mut self, user: &str) -> Result<&str> {
        self.turns.push(Turn {
            role: Role::User,
            content: user.to_string(),
        });
        let prompt = self.template.render(&self.turns);
        self.last_prompt_tokens = self.completer.count_tokens(&prompt);
        // Just await: the Completer impl owns whether this is sync compute (the
        // local Engine, under run_blocking) or I/O (a remote completer). CMP-1.
        // A turn is atomic (CHAT-1): on error, roll back the user turn so a failed
        // turn leaves the transcript exactly as before — never a dangling
        // unanswered user turn that poisons every later prompt.
        let completion = match self.completer.complete(&prompt, &self.opts, &[]).await {
            Ok(completion) => completion,
            Err(e) => {
                self.turns.pop();
                return Err(e);
            }
        };
        self.last_stop = Some(completion.stop);
        // Keep only the answer in history; the reasoning span is surfaced via
        // `last_reasoning`, never re-fed into the next prompt (REASON-1).
        let Reasoned { reasoning, answer } = split_reasoning(&completion.text);
        self.last_reasoning = reasoning;
        // A degenerate answer never enters history (CHAT-2): committed garbage
        // re-renders into every later prompt and poisons the session. The
        // exchange rolls back whole, as CHAT-1 does on error; the text is
        // still handed back (stashed outside the transcript) so the caller
        // can show what happened.
        if looks_degenerate(&answer) {
            tracing::warn!(
                chars = answer.chars().count(),
                "final answer looks degenerate; exchange not committed (CHAT-2)"
            );
            self.turns.pop();
            self.uncommitted = answer;
            return Ok(&self.uncommitted);
        }
        self.turns.push(Turn {
            role: Role::Assistant,
            content: answer,
        });
        Ok(&self.turns.last().expect("just pushed").content)
    }

    /// Sync shim over [`turn_async`](ChatSession::turn_async) for non-async
    /// callers, bridged through the one runtime (RT-1: panics, with direction, if
    /// called from within a current-thread runtime).
    pub fn turn(&mut self, user: &str) -> Result<&str> {
        crate::runtime::block_on(self.turn_async(user))
    }

    /// Like [`turn_async`](ChatSession::turn_async), but streams the reply to
    /// `on_token` as it is produced (live chat UIs). Same transcript bookkeeping.
    pub async fn turn_streaming_async(
        &mut self,
        user: &str,
        on_token: &mut dyn FnMut(&str),
    ) -> Result<&str> {
        self.turn_streaming_cancellable_async(user, &Cancel::new(), on_token)
            .await
    }

    /// [`turn_streaming_async`](ChatSession::turn_streaming_async) with a
    /// [`Cancel`] handle the caller can flip to stop the turn in flight (TUI-6).
    /// On cancel the turn ends with [`StopReason::Stopped`] and the partial reply
    /// is stored like any other completed turn (history stays clean per REASON-1).
    pub async fn turn_streaming_cancellable_async(
        &mut self,
        user: &str,
        cancel: &Cancel,
        on_token: &mut dyn FnMut(&str),
    ) -> Result<&str> {
        self.turns.push(Turn {
            role: Role::User,
            content: user.to_string(),
        });
        let prompt = self.template.render(&self.turns);
        self.last_prompt_tokens = self.completer.count_tokens(&prompt);
        // Atomic on error (CHAT-1): roll back the user turn. Any fragments already
        // streamed to `on_token` cannot be un-emitted, but the *stored* history
        // stays clean, so the next turn re-renders consistent prompt history.
        let completion = match self
            .completer
            .complete_streaming(&prompt, &self.opts, &[], cancel, on_token)
            .await
        {
            Ok(completion) => completion,
            Err(e) => {
                self.turns.pop();
                return Err(e);
            }
        };
        self.last_stop = Some(completion.stop);
        // The live `on_token` stream is raw (reasoning tokens included; a
        // channel-tagged stream is a follow-up), but the stored turn is
        // answer-only so history stays clean (REASON-1).
        let Reasoned { reasoning, answer } = split_reasoning(&completion.text);
        self.last_reasoning = reasoning;
        // CHAT-2: a degenerate answer rolls the exchange back (see
        // `turn_async`); the streamed tokens cannot be un-emitted, but the
        // stored history stays clean.
        if looks_degenerate(&answer) {
            tracing::warn!(
                chars = answer.chars().count(),
                "final answer looks degenerate; exchange not committed (CHAT-2)"
            );
            self.turns.pop();
            self.uncommitted = answer;
            return Ok(&self.uncommitted);
        }
        self.turns.push(Turn {
            role: Role::Assistant,
            content: answer,
        });
        Ok(&self.turns.last().expect("just pushed").content)
    }

    /// Sync shim over [`turn_streaming_async`](ChatSession::turn_streaming_async).
    pub fn turn_streaming(&mut self, user: &str, on_token: &mut dyn FnMut(&str)) -> Result<&str> {
        crate::runtime::block_on(self.turn_streaming_async(user, on_token))
    }

    /// Sync shim over
    /// [`turn_streaming_cancellable_async`](ChatSession::turn_streaming_cancellable_async).
    pub fn turn_streaming_cancellable(
        &mut self,
        user: &str,
        cancel: &Cancel,
        on_token: &mut dyn FnMut(&str),
    ) -> Result<&str> {
        crate::runtime::block_on(self.turn_streaming_cancellable_async(user, cancel, on_token))
    }

    /// Why the most recent turn stopped (EOS / max tokens / cancelled), or
    /// `None` before the first turn — for run metadata (META-1).
    pub fn last_stop(&self) -> Option<StopReason> {
        self.last_stop
    }

    /// The reasoning span of the most recent reply (a reasoning model's
    /// chain-of-thought), or `None` for a non-reasoning model or before any
    /// turn. The span is never part of [`history`](ChatSession::history)
    /// (REASON-1); this is the only place it is surfaced.
    pub fn last_reasoning(&self) -> Option<&str> {
        self.last_reasoning.as_deref()
    }

    /// Tokens in the most recent rendered prompt, if the completer exposes a
    /// tokenizer — for a host's context-usage meter (with
    /// [`crate::Engine::context_length`] as the denominator). `None` before any
    /// turn or for a tokenizer-less completer.
    pub fn last_prompt_tokens(&self) -> Option<usize> {
        self.last_prompt_tokens
    }

    /// Clear the conversation back to the seeded system prompt.
    pub fn reset(&mut self) {
        self.turns.truncate(self.system_len);
    }

    /// Drop the oldest committed exchanges until the rendered prompt of the
    /// remaining transcript fits `budget` tokens, keeping the newest
    /// `keep_last` exchanges (COMPACT-1). Returns the dropped turns, oldest
    /// first (a host may summarize them; rung 2). The seeded system prompt
    /// (`system_len` leading turns) is never dropped, and exchanges are
    /// dropped as indivisible user+assistant pairs, so template alternation is
    /// never broken. If the system prompt alone exceeds `budget` only the
    /// protected turns remain.
    pub fn trim_history_to(&mut self, budget: usize, keep_last: usize) -> Vec<Turn> {
        let protected_tail = keep_last.saturating_mul(2);
        let mut dropped = Vec::new();
        loop {
            let droppable = self
                .turns
                .len()
                .saturating_sub(self.system_len)
                .saturating_sub(protected_tail);
            if droppable < 2 {
                break;
            }
            let rendered = self.template.render(&self.turns);
            if self.count_prompt_tokens(&rendered) <= budget {
                break;
            }
            // Oldest exchange = the two turns just after the system prefix.
            dropped.push(self.turns.remove(self.system_len));
            dropped.push(self.turns.remove(self.system_len));
        }
        dropped
    }

    /// Token count of `rendered` under the completer's tokenizer, falling back
    /// to `chars/4` when it exposes none (COMPACT-1's deterministic fallback).
    fn count_prompt_tokens(&self, rendered: &str) -> usize {
        self.completer
            .count_tokens(rendered)
            .unwrap_or_else(|| rendered.chars().count() / 4)
    }

    /// The transcript so far (system / user / assistant turns, in order).
    pub fn history(&self) -> &[Turn] {
        &self.turns
    }
}

/// True when a final answer's tail looks like decode degeneration — the
/// Metal KV-cliff garbage modes (ASCII punctuation soup `,,,,!0…`, digit
/// runs; `notes/metal-kv-cliff.md`) rather than prose or code. This is the
/// judgment CHAT-2 and AGENT-3's degenerate case gate history commits on,
/// and a host uses to tell the user why an answer was not kept.
///
/// Deliberately conservative: only a *sustained* non-alphabetic tail
/// convicts — prose and code both keep letters flowing, and an answer
/// shorter than the window never convicts. Known tradeoffs: a long purely
/// numeric table tail can false-positive (cost: the exchange is re-asked,
/// not lost); the non-ASCII "wave" mode (`퓮퓮…`) is deliberately out of
/// scope — those codepoints are alphabetic, and convicting on them would
/// convict every CJK answer. That mode trips the repetition guard instead.
pub fn looks_degenerate(answer: &str) -> bool {
    const TAIL_CHARS: usize = 120;
    const ALPHA_MIN_PERCENT: usize = 20;
    let tail: Vec<char> = answer.chars().rev().take(TAIL_CHARS).collect();
    if tail.len() < TAIL_CHARS {
        return false;
    }
    let alpha = tail.iter().filter(|c| c.is_alphabetic()).count();
    alpha * 100 < TAIL_CHARS * ALPHA_MIN_PERCENT
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::completer::Completion;
    use crate::template::PlainTemplate;
    use crate::StopReason;

    /// A [`Completer`] that echoes back the prompt it was given (so tests can
    /// assert what history reached the model) after a canned reply, replayed
    /// from a script.
    struct Scripted {
        replies: Vec<String>,
        i: usize,
        last_prompt: String,
    }

    impl Scripted {
        fn new(replies: &[&str]) -> Scripted {
            Scripted {
                replies: replies.iter().map(|s| s.to_string()).collect(),
                i: 0,
                last_prompt: String::new(),
            }
        }
    }

    impl Completer for Scripted {
        async fn complete(
            &mut self,
            prompt: &str,
            _: &GenOpts,
            _: &[String],
        ) -> Result<Completion> {
            self.last_prompt = prompt.to_string();
            let text = self.replies.get(self.i).cloned().unwrap_or_default();
            self.i += 1;
            Ok(Completion {
                text,
                stop: StopReason::Eos,
            })
        }
    }

    /// A [`Completer`] that errors on its first call, then succeeds — so we can
    /// assert a failed turn does not poison the session (CHAT-1).
    struct FailsThenWorks {
        calls: usize,
        last_prompt: String,
    }

    impl Completer for FailsThenWorks {
        async fn complete(
            &mut self,
            prompt: &str,
            _: &GenOpts,
            _: &[String],
        ) -> Result<Completion> {
            self.last_prompt = prompt.to_string();
            self.calls += 1;
            if self.calls == 1 {
                anyhow::bail!("simulated engine error");
            }
            Ok(Completion {
                text: "recovered".to_string(),
                stop: StopReason::Eos,
            })
        }
    }

    #[test]
    fn a_failed_turn_is_atomic_and_recovers() {
        // upholds: CHAT-1 — a turn whose completion errors rolls back its user
        // turn, leaving the transcript unchanged, so the session is not poisoned:
        // a later turn re-renders clean history and succeeds, and the failed
        // message never enters the prompt.
        let mut model = FailsThenWorks {
            calls: 0,
            last_prompt: String::new(),
        };
        let mut chat = ChatSession::new(&mut model, PlainTemplate).with_system("sys");
        assert_eq!(chat.history().len(), 1); // system only

        assert!(chat.turn("poison me").is_err(), "first turn errors");
        assert_eq!(
            chat.history().len(),
            1,
            "a failed turn must leave no dangling user turn"
        );

        let reply = chat.turn("hello again").unwrap().to_string();
        assert_eq!(reply, "recovered");
        // history: system, user(hello again), assistant(recovered).
        assert_eq!(chat.history().len(), 3);
        assert!(
            !chat.completer.last_prompt.contains("poison me"),
            "the failed message must not reach the model's prompt"
        );
    }

    #[test]
    fn degeneration_judgment_convicts_soup_spares_prose_and_code() {
        // The KV-cliff garbage modes convict; real answer shapes do not.
        let soup = format!(
            "The Holy Sepulchre Bicycle] ({}!0 ... -lnd",
            ",".repeat(120)
        );
        assert!(looks_degenerate(&soup), "punctuation soup convicts");
        let digits = format!("answer: {}", "8, 0, 0, ".repeat(20));
        assert!(looks_degenerate(&digits), "digit runs convict");

        assert!(!looks_degenerate("short"), "short answers never convict");
        let prose = "The road has historical significance as part of the \
                     route used by soldiers traveling south during the \
                     Invasion of Waikato in the 1860s, and later became a \
                     center for brewers.";
        assert!(!looks_degenerate(prose), "prose is spared");
        let code = format!(
            "fn scroll_y(total: usize, viewport: usize) -> usize {{\n    \
             total.saturating_sub(viewport)\n}}\n{}",
            "// the result is in [0, total - viewport]\n".repeat(3)
        );
        assert!(!looks_degenerate(&code), "code keeps letters flowing");
    }

    #[test]
    fn a_degenerate_turn_is_not_committed() {
        // upholds: CHAT-2 — a final answer that looks like decode
        // degeneration rolls the exchange back (like CHAT-1 on error): the
        // caller still sees the text, but the next prompt re-renders clean
        // history, so one poisoned answer cannot poison the session.
        let soup = format!("garbage{}", ",!0.".repeat(40));
        let scripted = [soup.as_str(), "recovered"];
        let mut model = Scripted::new(&scripted);
        let mut chat = ChatSession::new(&mut model, PlainTemplate).with_system("sys");

        let reply = chat.turn("first question").unwrap().to_string();
        assert_eq!(reply, soup, "the degenerate text is still handed back");
        assert_eq!(
            chat.history().len(),
            1,
            "a degenerate exchange must leave history unchanged (CHAT-2)"
        );

        let reply = chat.turn("second question").unwrap().to_string();
        assert_eq!(reply, "recovered");
        assert!(
            !chat.completer.last_prompt.contains("garbage"),
            "the degenerate answer must never re-enter a prompt"
        );
        assert_eq!(chat.history().len(), 3); // sys, user, assistant
    }

    #[test]
    fn turn_accumulates_and_remembers_history() {
        let mut model = Scripted::new(&["Hi Ada!", "Your name is Ada."]);
        let mut chat = ChatSession::new(&mut model, PlainTemplate);
        assert_eq!(chat.turn("My name is Ada.").unwrap(), "Hi Ada!");
        let second = chat.turn("What is my name?").unwrap().to_string();
        assert_eq!(second, "Your name is Ada.");

        // The second call's prompt must contain the whole prior exchange — that's
        // where memory comes from (history re-rendered, engine stateless).
        let p = &chat.completer.last_prompt;
        assert!(p.contains("My name is Ada."), "user turn 1 in prompt");
        assert!(p.contains("Hi Ada!"), "assistant turn 1 in prompt");
        assert!(p.contains("What is my name?"), "user turn 2 in prompt");

        // transcript = user, assistant, user, assistant
        assert_eq!(chat.history().len(), 4);
        assert_eq!(chat.history()[0].role, Role::User);
        assert_eq!(chat.history()[1].role, Role::Assistant);
    }

    #[test]
    fn turn_streaming_delivers_and_accumulates() {
        let mut model = Scripted::new(&["streamed reply"]);
        let mut chat = ChatSession::new(&mut model, PlainTemplate);
        let mut got = String::new();
        let answer = chat
            .turn_streaming("hi", &mut |piece| got.push_str(piece))
            .unwrap()
            .to_string();
        assert_eq!(answer, "streamed reply");
        assert_eq!(got, "streamed reply"); // delivered via the callback
        assert_eq!(chat.history().len(), 2); // user + assistant
    }

    #[test]
    fn reasoning_is_split_from_the_reply_and_history() {
        // upholds: REASON-1 — a reasoning model's think span is surfaced via
        // last_reasoning, but the stored/returned reply is answer-only, so it is
        // not re-fed into the next prompt.
        let mut model = Scripted::new(&["<think>recall the name</think>Your name is Ada."]);
        let mut chat = ChatSession::new(&mut model, PlainTemplate);
        let reply = chat.turn("What is my name?").unwrap().to_string();
        assert_eq!(reply, "Your name is Ada.");
        assert_eq!(chat.last_reasoning(), Some("recall the name"));
        // History (re-rendered into the next prompt) holds the answer only.
        let assistant = &chat.history()[1];
        assert_eq!(assistant.role, Role::Assistant);
        assert_eq!(assistant.content, "Your name is Ada.");
        assert!(!assistant.content.contains("<think>"));
    }

    #[test]
    fn trim_protects_system_and_newest_and_drops_oldest_pairs() {
        // upholds: COMPACT-1 — on a session with a seeded system prompt,
        // trimming drops the oldest user/assistant pairs *after* the system
        // turn, oldest first, and never the system prompt nor the newest
        // keep_last exchanges. The `chars/4` fallback (the scripted completer
        // counts no tokens) makes the drop deterministic.
        let replies: Vec<String> = (0..6)
            .map(|i| format!("answer {i} {}", "y".repeat(100)))
            .collect();
        let reply_refs: Vec<&str> = replies.iter().map(String::as_str).collect();
        let mut model = Scripted::new(&reply_refs);
        let mut chat = ChatSession::new(&mut model, PlainTemplate).with_system("SYSTEM PROMPT");
        for i in 0..6 {
            chat.turn(&format!("question {i} {}", "x".repeat(100)))
                .unwrap();
        }
        assert_eq!(chat.history().len(), 13); // system + 6 exchanges

        let dropped = chat.trim_history_to(200, 2);
        assert!(!dropped.is_empty(), "a too-deep session must be trimmed");
        assert_eq!(dropped.len() % 2, 0, "exchanges drop as whole pairs");
        assert_eq!(dropped[0].role, Role::User, "oldest turn first");
        assert!(dropped[0].content.contains("question 0"));
        // The system prompt is never dropped and stays at the front.
        assert_eq!(chat.history()[0].role, Role::System);
        assert_eq!(chat.history()[0].content, "SYSTEM PROMPT");
        // Alternation intact: the turn right after the system prefix is a user.
        assert_eq!(chat.history()[1].role, Role::User);
        // The newest two exchanges (4 turns) survive alongside the system turn.
        assert!(chat.history().len() >= 5 && chat.history().len() < 13);
        assert!(chat.history().last().unwrap().content.contains("answer 5"));
    }

    #[test]
    fn a_long_scripted_session_stays_bounded_under_repeated_trims() {
        // upholds: COMPACT-1 (with the host's HOST-5 policy simulated here) —
        // over a 30-turn scripted run, trimming to a fixed small budget after
        // every turn keeps history bounded, drops whole pairs oldest-first
        // (each dropped exchange exactly once, in order), and never touches the
        // system prompt.
        let replies: Vec<String> = (0..30)
            .map(|i| format!("reply {i} {}", "z".repeat(60)))
            .collect();
        let reply_refs: Vec<&str> = replies.iter().map(String::as_str).collect();
        let mut model = Scripted::new(&reply_refs);
        let mut chat = ChatSession::new(&mut model, PlainTemplate).with_system("SYS");

        let budget = 200; // chars/4 fallback: only a handful of exchanges fit
        let mut next_oldest = 0usize; // the exchange index we expect to drop next
        let mut total_dropped = 0usize;
        for i in 0..30 {
            chat.turn(&format!("ask {i} {}", "q".repeat(60))).unwrap();
            let dropped = chat.trim_history_to(budget, 2);
            assert_eq!(dropped.len() % 2, 0, "whole pairs only");
            for pair in dropped.chunks(2) {
                assert_eq!(pair[0].role, Role::User);
                assert!(
                    pair[0].content.starts_with(&format!("ask {next_oldest} ")),
                    "dropped out of order: {:?}",
                    pair[0].content
                );
                assert_eq!(pair[1].role, Role::Assistant);
                next_oldest += 1;
            }
            total_dropped += dropped.len();
            assert_eq!(chat.history()[0].content, "SYS", "system prompt survives");
        }
        assert!(total_dropped > 0, "a 30-turn run must trigger drops");
        assert!(
            (5..=15).contains(&chat.history().len()),
            "history stayed bounded: {}",
            chat.history().len()
        );
    }

    #[test]
    fn trim_leaves_a_fitting_session_untouched() {
        // upholds: COMPACT-1 — a session already under budget is returned
        // whole; nothing is dropped.
        let mut model = Scripted::new(&["ok"]);
        let mut chat = ChatSession::new(&mut model, PlainTemplate).with_system("sys");
        chat.turn("hi").unwrap();
        let before = chat.history().len();
        let dropped = chat.trim_history_to(100_000, 0);
        assert!(dropped.is_empty());
        assert_eq!(chat.history().len(), before);
    }

    #[test]
    fn system_persists_and_reset_keeps_it() {
        let mut model = Scripted::new(&["a", "b"]);
        let mut chat = ChatSession::new(&mut model, PlainTemplate).with_system("Be terse.");
        chat.turn("one").unwrap();
        assert_eq!(chat.history().len(), 3); // system, user, assistant
        assert_eq!(chat.history()[0].role, Role::System);

        chat.reset();
        assert_eq!(chat.history().len(), 1); // only the system turn remains
        assert_eq!(chat.history()[0].content, "Be terse.");

        // a turn after reset still carries the system prompt into the prompt.
        chat.turn("two").unwrap();
        assert!(chat.completer.last_prompt.contains("Be terse."));
    }
}
