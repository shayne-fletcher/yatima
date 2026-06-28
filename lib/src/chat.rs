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
use crate::{GenOpts, StopReason};
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
            .complete_streaming(&prompt, &self.opts, &[], on_token)
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

    /// The transcript so far (system / user / assistant turns, in order).
    pub fn history(&self) -> &[Turn] {
        &self.turns
    }
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
