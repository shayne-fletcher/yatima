//! The agent loop — a fold over turns.
//!
//! One level up from `generate_with` (which folds *tokens* into a value), the
//! agent folds *turns*: the model emits a tool call, a capability-scoped tool
//! runs, its result is fed back, and the loop repeats until the model answers or
//! `max_steps` is reached. [`Agent::run`] collects the final answer;
//! [`Agent::run_with_async`] is the fold a future actor/TUI streams
//! [`AgentEvent`]s into. The model turns are sequential, but tool calls are
//! async tasks: callers can observe starts/progress/results and cancellation
//! boundaries while the agent still waits for the result before the next model
//! turn.

use crate::completer::Completer;
use crate::reasoning::{split_reasoning, Channel, Reasoned, ReasoningSplitter};
use crate::template::PromptTemplate;
use crate::tool::{
    ToolCall, ToolCallCodec, ToolEvent, ToolOutcome, ToolRejection, ToolResult, Tools,
};
use crate::transcript::{Role, Turn};
use crate::{Cancel, GenOpts};
use anyhow::Result;
use std::cell::RefCell;
use std::ops::ControlFlow;

/// An observable step of a run, delivered to [`Agent::run_with`]'s fold.
#[derive(Debug, Clone)]
pub enum AgentEvent {
    ToolCall(ToolCall),
    ToolStarted(ToolCall),
    ToolProgress(String),
    ToolOutcome(ToolOutcome),
    /// A live slice of the current step's decode (AGENT-4), classified as it
    /// streams: chain-of-thought on [`Channel::Reasoning`], prose on
    /// [`Channel::Answer`]. Codec markup never reaches the answer channel —
    /// text that turns out to open a tool call is withheld; the call itself
    /// arrives as [`AgentEvent::ToolCall`]. Answer fragments of a step that
    /// ends in a tool call are *narration* (prose the model wrote before
    /// calling): the following `ToolCall` event licenses a consumer to fold
    /// them into working matter.
    Fragment {
        channel: Channel,
        text: String,
    },
    /// A reasoning model's chain-of-thought for the step just completed. Emitted
    /// before the resulting `ToolCall`/`Final`; observational (for UIs/logging).
    /// The trace is *not* re-fed into the transcript (REASON-1). Streaming
    /// consumers get the same text incrementally via [`AgentEvent::Fragment`];
    /// this remains the complete per-step record.
    Reasoning(String),
    Final(String),
}

/// Why a run ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AgentStop {
    /// The model produced a final answer.
    Final,
    /// The `max_steps` tool-round budget was exhausted (AGENT-1).
    MaxSteps,
    /// The caller's fold returned `ControlFlow::Break`.
    Stopped,
}

/// The outcome of a run.
#[derive(Debug, Clone)]
pub struct Run {
    pub answer: String,
    pub transcript: Vec<Turn>,
    pub steps: usize,
    pub stop: AgentStop,
}

/// An agent: a [`Completer`] driven against a set of [`Tools`], using a
/// [`PromptTemplate`] to speak the model's native format and a [`ToolCallCodec`]
/// to encode/parse tool calls, bounded by `max_steps`.
///
/// The agent is **sessionful**: successive [`run`](Agent::run)s form one
/// conversation. Each completed exchange persists its user turn and final
/// answer into [`history`](Agent::history) and is rendered into later prompts;
/// tool rounds and reasoning stay ephemeral to their run (AGENT-3). One-shot
/// use is the fresh-`Agent`-per-run special case, unchanged.
pub struct Agent<'a, C: Completer, K: ToolCallCodec, T: PromptTemplate> {
    completer: &'a mut C,
    tools: &'a Tools,
    codec: K,
    template: T,
    system: String,
    max_steps: usize,
    opts: GenOpts,
    /// Session memory across runs: the user turn and final answer of each
    /// completed exchange, in order (AGENT-3). Tool rounds and reasoning are
    /// ephemeral to their run and never re-enter a later prompt; an
    /// interrupted or step-exhausted run leaves this untouched.
    history: Vec<Turn>,
}

impl<'a, C: Completer, K: ToolCallCodec, T: PromptTemplate> Agent<'a, C, K, T> {
    /// Build an agent. `system` is the base system prompt; the codec's tool
    /// instructions are appended to it, and the whole transcript is rendered by
    /// `template` into the model's native prompt format.
    pub fn new(
        completer: &'a mut C,
        tools: &'a Tools,
        codec: K,
        template: T,
        system: impl Into<String>,
        max_steps: usize,
    ) -> Agent<'a, C, K, T> {
        Agent {
            completer,
            tools,
            codec,
            template,
            system: system.into(),
            max_steps,
            opts: GenOpts::default(),
            history: Vec::new(),
        }
    }

    /// Override the generation options used for each turn (default greedy).
    pub fn with_opts(mut self, opts: GenOpts) -> Agent<'a, C, K, T> {
        self.opts = opts;
        self
    }

    /// The session history: each completed exchange's user turn and final
    /// answer, in order (AGENT-3). Empty until a run reaches a final answer.
    pub fn history(&self) -> &[Turn] {
        &self.history
    }

    /// Seed the session history (builder style) — the transplant a host uses
    /// when a plain chat session becomes tool-bearing mid-conversation (the
    /// first origin grant): both histories are user/answer `Turn`s, so the
    /// seam is invisible (AGENT-3).
    pub fn with_history(mut self, history: Vec<Turn>) -> Self {
        self.history = history;
        self
    }

    /// Clear the session history (the system prompt and tools are unchanged).
    pub fn reset(&mut self) {
        self.history.clear();
    }

    /// Run to a final answer (or `max_steps`), discarding per-step events.
    pub fn run(&mut self, user: &str) -> Result<Run> {
        let ((), run) = self.run_with(user, (), |(), _event| Ok(ControlFlow::Continue(())))?;
        Ok(run)
    }

    /// Async variant of [`Agent::run`]. This is the primitive path for agents
    /// with network/process tools; [`Agent::run`] is a compatibility wrapper.
    pub async fn run_async(&mut self, user: &str) -> Result<Run> {
        let ((), run) = self
            .run_with_async(user, (), |(), _event| Ok(ControlFlow::Continue(())))
            .await?;
        Ok(run)
    }

    /// Run while folding each [`AgentEvent`] into an accumulator. Returning
    /// `ControlFlow::Break` stops the run early ([`AgentStop::Stopped`]). This is
    /// the primitive; [`Agent::run`] is the `acc = ()` specialization.
    pub fn run_with<A>(
        &mut self,
        user: &str,
        init: A,
        mut step: impl FnMut(A, AgentEvent) -> Result<ControlFlow<A, A>>,
    ) -> Result<(A, Run)> {
        crate::runtime::block_on(self.run_with_async(user, init, &mut step))
    }

    /// [`Agent::run_with`] with a token-level cancel (sync shim).
    pub fn run_with_cancellable<A>(
        &mut self,
        user: &str,
        cancel: &Cancel,
        init: A,
        mut step: impl FnMut(A, AgentEvent) -> Result<ControlFlow<A, A>>,
    ) -> Result<(A, Run)> {
        crate::runtime::block_on(self.run_with_cancellable_async(user, cancel, init, &mut step))
    }

    /// Run while folding each [`AgentEvent`] into an accumulator. Returning
    /// `ControlFlow::Break` stops the run early ([`AgentStop::Stopped`]). The
    /// decode streams (AGENT-4) but without an external cancel handle a stop
    /// still lands at the next fragment; see
    /// [`run_with_cancellable_async`](Agent::run_with_cancellable_async).
    pub async fn run_with_async<A>(
        &mut self,
        user: &str,
        init: A,
        mut step: impl FnMut(A, AgentEvent) -> Result<ControlFlow<A, A>>,
    ) -> Result<(A, Run)> {
        let cancel = Cancel::new();
        self.run_with_cancellable_async(user, &cancel, init, &mut step)
            .await
    }

    /// Run while folding each [`AgentEvent`] into an accumulator, with a
    /// token-level cancel. Each step's decode **streams** (AGENT-4): fragments
    /// arrive live as [`AgentEvent::Fragment`], classified reasoning/answer,
    /// with codec markup withheld from the answer channel. Flipping `cancel`
    /// (or returning `ControlFlow::Break` from the fold, which flips it too)
    /// stops the decode at the next token — [`AgentStop::Stopped`], history
    /// untouched (AGENT-3).
    pub async fn run_with_cancellable_async<A>(
        &mut self,
        user: &str,
        cancel: &Cancel,
        init: A,
        mut step: impl FnMut(A, AgentEvent) -> Result<ControlFlow<A, A>>,
    ) -> Result<(A, Run)> {
        tracing::info!(
            max_steps = self.max_steps,
            tool_count = self.tools.specs().len(),
            "agent run started"
        );
        let rendered_tools = self.codec.render_system(&self.tools.specs());
        let system = if rendered_tools.is_empty() {
            self.system.clone()
        } else {
            format!("{}\n\n{rendered_tools}", self.system)
        };
        // Seed the working transcript with the session history (AGENT-3): prior
        // exchanges' user/answer turns only — their tool rounds and reasoning
        // were ephemeral to their runs.
        let mut transcript = Vec::with_capacity(self.history.len() + 2);
        transcript.push(Turn {
            role: Role::System,
            content: system,
        });
        transcript.extend(self.history.iter().cloned());
        transcript.push(Turn {
            role: Role::User,
            content: user.to_string(),
        });

        let stops = self.codec.stop_strings();
        let mut acc = init;
        let mut steps = 0usize;
        let mut answer = String::new();
        let stop;

        loop {
            let prompt = self.template.render(&transcript);
            tracing::trace!(step = steps, prompt_chars = prompt.len(), prompt = %prompt,
                "agent step prompt");
            // Stream the step's decode (AGENT-4): fragments are classified
            // live — reasoning via the splitter, answer text through the
            // opener gate (codec markup withheld) — and folded as they
            // arrive. A fold `Break` (or error) flips `cancel`, so the
            // decode stops at the next token; the Completer impl owns its
            // operational shape (CMP-1 / RT-1).
            let fold = RefCell::new(StepFold {
                acc: Some(acc),
                broke: false,
                error: None,
            });
            let completion = {
                let mut deliver = |channel: Channel, text: String| {
                    let mut f = fold.borrow_mut();
                    if f.broke || f.error.is_some() {
                        return;
                    }
                    let Some(a) = f.acc.take() else { return };
                    match step(a, AgentEvent::Fragment { channel, text }) {
                        Ok(ControlFlow::Continue(a)) => f.acc = Some(a),
                        Ok(ControlFlow::Break(a)) => {
                            f.acc = Some(a);
                            f.broke = true;
                            cancel.cancel();
                        }
                        Err(e) => {
                            f.error = Some(e);
                            cancel.cancel();
                        }
                    }
                };
                let mut splitter = ReasoningSplitter::new();
                let mut gate = AnswerGate::new(self.codec.open_marker());
                let mut on_token = |frag: &str| {
                    splitter.push(frag, |channel, text| match channel {
                        Channel::Reasoning => deliver(Channel::Reasoning, text.to_string()),
                        Channel::Answer => {
                            if let Some(safe) = gate.push(text) {
                                deliver(Channel::Answer, safe);
                            }
                        }
                    });
                };
                let completion = self
                    .completer
                    .complete_streaming(&prompt, &self.opts, &stops, cancel, &mut on_token)
                    .await?;
                // Flush the splitter's tail through the same pipeline, then
                // the gate's held answer text (a final step may end on a
                // partial-opener lookalike that turned out to be prose).
                splitter.finish(|channel, text| match channel {
                    Channel::Reasoning => deliver(Channel::Reasoning, text.to_string()),
                    Channel::Answer => {
                        if let Some(safe) = gate.push(text) {
                            deliver(Channel::Answer, safe);
                        }
                    }
                });
                if let Some(rest) = gate.finish() {
                    deliver(Channel::Answer, rest);
                }
                completion
            };
            let StepFold {
                acc: a,
                broke,
                error,
            } = fold.into_inner();
            if let Some(e) = error {
                return Err(e);
            }
            acc = a.expect("fold accumulator survives the stream");
            // A cancel is detected via the handle, never via `completion.stop`:
            // the engine reports `Stopped` for *any* early fold break, which
            // includes ordinary stop-string termination — i.e. every
            // successful tool-call step (the `</tool_call>` stop). Only the
            // handle distinguishes "the user stopped us" from "the reply is
            // complete".
            if broke || cancel.is_cancelled() {
                stop = AgentStop::Stopped;
                break;
            }
            tracing::trace!(step = steps, completion_stop = ?completion.stop,
                completion = %completion.text, "agent step completion");
            // Split off the reasoning span at the completion→turn boundary: the
            // transcript (re-rendered into the next prompt) carries only the
            // answer, never the chain-of-thought (REASON-1). The reply still
            // holds any trailing tool call, so the codec parses it below.
            let Reasoned {
                reasoning,
                answer: reply,
            } = split_reasoning(&completion.text);
            transcript.push(Turn {
                role: Role::Assistant,
                content: reply.clone(),
            });
            if let Some(reasoning) = reasoning {
                match step(acc, AgentEvent::Reasoning(reasoning))? {
                    ControlFlow::Continue(a) => acc = a,
                    ControlFlow::Break(a) => {
                        acc = a;
                        stop = AgentStop::Stopped;
                        break;
                    }
                }
            }

            match self.codec.parse(&reply) {
                // A plain answer: the run is done (the reasoning span, if any, has
                // already been stripped from `reply`).
                None => {
                    match step(acc, AgentEvent::Final(reply.clone()))? {
                        ControlFlow::Continue(a) | ControlFlow::Break(a) => acc = a,
                    }
                    answer = reply;
                    stop = AgentStop::Final;
                    break;
                }
                // A tool call (well-formed or not): dispatch / make an error
                // result, feed it back, and continue under the step budget.
                Some(parsed) => {
                    let (tool_name, outcome) = match parsed {
                        Ok(call) => {
                            match step(acc, AgentEvent::ToolCall(call.clone()))? {
                                ControlFlow::Continue(a) => acc = a,
                                ControlFlow::Break(a) => {
                                    acc = a;
                                    stop = AgentStop::Stopped;
                                    break;
                                }
                            }
                            let tool_name = call.name.clone();
                            let mut task = self.tools.spawn(call);
                            let outcome = loop {
                                match task.recv().await {
                                    Some(ToolEvent::Started { call, .. }) => {
                                        match step(acc, AgentEvent::ToolStarted(call))? {
                                            ControlFlow::Continue(a) => acc = a,
                                            ControlFlow::Break(a) => {
                                                task.cancel();
                                                let _ = task.join().await;
                                                tracing::info!(
                                                    steps,
                                                    stop = ?AgentStop::Stopped,
                                                    "agent run finished"
                                                );
                                                return Ok((
                                                    a,
                                                    Run {
                                                        answer,
                                                        transcript,
                                                        steps,
                                                        stop: AgentStop::Stopped,
                                                    },
                                                ));
                                            }
                                        }
                                    }
                                    Some(ToolEvent::Progress { message, .. }) => {
                                        match step(acc, AgentEvent::ToolProgress(message))? {
                                            ControlFlow::Continue(a) => acc = a,
                                            ControlFlow::Break(a) => {
                                                task.cancel();
                                                let _ = task.join().await;
                                                tracing::info!(
                                                    steps,
                                                    stop = ?AgentStop::Stopped,
                                                    "agent run finished"
                                                );
                                                return Ok((
                                                    a,
                                                    Run {
                                                        answer,
                                                        transcript,
                                                        steps,
                                                        stop: AgentStop::Stopped,
                                                    },
                                                ));
                                            }
                                        }
                                    }
                                    Some(ToolEvent::Finished { outcome, .. }) => break outcome,
                                    Some(ToolEvent::Cancelled { .. }) => {}
                                    None => break task.join().await,
                                }
                            };
                            (tool_name, outcome)
                        }
                        Err(e) => (
                            String::new(),
                            ToolOutcome::Rejected(ToolRejection::InvalidArgs {
                                message: format!("malformed tool call: {e}"),
                            }),
                        ),
                    };

                    let result = outcome.render_for_model(&tool_name);
                    transcript.push(Turn {
                        role: Role::Tool,
                        content: render_result(&result),
                    });
                    match step(acc, AgentEvent::ToolOutcome(outcome))? {
                        ControlFlow::Continue(a) => acc = a,
                        ControlFlow::Break(a) => {
                            acc = a;
                            stop = AgentStop::Stopped;
                            break;
                        }
                    }

                    steps += 1;
                    if steps >= self.max_steps {
                        stop = AgentStop::MaxSteps;
                        break;
                    }
                }
            }
        }

        tracing::info!(steps, stop = ?stop, "agent run finished");
        // Persist the exchange into session history only when it completed
        // (AGENT-3): the user turn and the final answer. Interrupted or
        // step-exhausted runs leave history untouched, so the caller can
        // simply re-ask.
        if stop == AgentStop::Final {
            self.history.push(Turn {
                role: Role::User,
                content: user.to_string(),
            });
            self.history.push(Turn {
                role: Role::Assistant,
                content: answer.clone(),
            });
        }
        Ok((
            acc,
            Run {
                answer,
                transcript,
                steps,
                stop,
            },
        ))
    }
}

/// Render a tool result as the `tool`-turn content the model reads back.
fn render_result(result: &ToolResult) -> String {
    let tag = if result.is_error { "error" } else { "ok" };
    format!("[{} {}] {}", result.name, tag, result.content)
}

/// The fold's state while a step streams: the accumulator threads through the
/// `on_token` pipeline (which cannot return `ControlFlow` itself), and a
/// `Break`/error is recorded and converted into a token-level cancel.
struct StepFold<A> {
    acc: Option<A>,
    broke: bool,
    error: Option<anyhow::Error>,
}

/// Withholds tool-call markup from a live answer stream (AGENT-4): text is
/// buffered while its tail could still become the codec's open marker; once
/// the marker completes, everything from it on is suppressed (the parsed call
/// arrives as [`AgentEvent::ToolCall`] instead); a lookalike that diverges is
/// released as ordinary prose.
struct AnswerGate {
    opener: String,
    held: String,
    suppressed: bool,
}

impl AnswerGate {
    fn new(opener: String) -> AnswerGate {
        AnswerGate {
            opener,
            held: String::new(),
            suppressed: false,
        }
    }

    /// Feed `text`; returns answer-safe output to emit now (never empty).
    fn push(&mut self, text: &str) -> Option<String> {
        if self.suppressed {
            return None;
        }
        self.held.push_str(text);
        if let Some(at) = self.held.find(&self.opener) {
            let safe = self.held[..at].to_string();
            self.suppressed = true;
            self.held.clear();
            return (!safe.is_empty()).then_some(safe);
        }
        let hold = longest_opener_prefix_suffix(&self.held, &self.opener);
        let emit = self.held.len() - hold;
        if emit == 0 {
            return None;
        }
        let safe: String = self.held.drain(..emit).collect();
        Some(safe)
    }

    /// The stream ended without a complete opener: release what was held.
    fn finish(self) -> Option<String> {
        (!self.suppressed && !self.held.is_empty()).then_some(self.held)
    }
}

/// The length of the longest *proper* suffix of `held` that is a prefix of
/// `opener` (a complete opener is found by `find` before this runs). Both
/// strings are ASCII markers in practice, but boundaries are checked.
fn longest_opener_prefix_suffix(held: &str, opener: &str) -> usize {
    let max = held.len().min(opener.len().saturating_sub(1));
    for len in (1..=max).rev() {
        if !held.is_char_boundary(held.len() - len) {
            continue;
        }
        if opener
            .as_bytes()
            .starts_with(&held.as_bytes()[held.len() - len..])
        {
            return len;
        }
    }
    0
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::completer::Completion;
    use crate::template::PlainTemplate;
    use crate::tool::{JsonToolCall, ReadFile};
    use crate::{capability::Dir, StopReason};
    use std::io::Write;

    /// A [`Completer`] that replays canned completions — the agent's laws are
    /// provable with no model. Panics if the loop asks for more than scripted.
    /// Records every rendered prompt so tests can assert what a later step (or
    /// a later run, AGENT-3) actually re-renders.
    struct Scripted {
        script: Vec<Completion>,
        i: usize,
        prompts: Vec<String>,
    }

    impl Scripted {
        fn new(texts: &[&str]) -> Scripted {
            let script = texts
                .iter()
                .map(|t| Completion {
                    text: (*t).to_string(),
                    stop: StopReason::Eos,
                })
                .collect();
            Scripted {
                script,
                i: 0,
                prompts: Vec::new(),
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
            self.prompts.push(prompt.to_string());
            let c = self
                .script
                .get(self.i)
                .cloned()
                .unwrap_or_else(|| panic!("scripted completer exhausted at step {}", self.i));
            self.i += 1;
            Ok(c)
        }

        // Stream in 3-char chunks, honoring `cancel` between chunks — small
        // enough that markers split across fragments, exercising the gate and
        // splitter reassembly (AGENT-4).
        async fn complete_streaming(
            &mut self,
            prompt: &str,
            opts: &GenOpts,
            stops: &[String],
            cancel: &Cancel,
            on_token: &mut dyn FnMut(&str),
        ) -> Result<Completion> {
            let completion = self.complete(prompt, opts, stops).await?;
            let chars: Vec<char> = completion.text.chars().collect();
            for chunk in chars.chunks(3) {
                if cancel.is_cancelled() {
                    return Ok(Completion {
                        text: completion.text.clone(),
                        stop: StopReason::Stopped,
                    });
                }
                on_token(&chunk.iter().collect::<String>());
            }
            Ok(completion)
        }
    }

    fn tmp_with_file(name: &str, body: &str) -> tempfile::TempDir {
        let dir = tempfile::tempdir().unwrap();
        let mut f = std::fs::File::create(dir.path().join(name)).unwrap();
        write!(f, "{body}").unwrap();
        dir
    }

    fn call(tool: &str, path: &str) -> String {
        format!(
            "<tool_call>{{\"name\": \"{tool}\", \"args\": {{\"path\": \"{path}\"}}}}</tool_call>"
        )
    }

    #[test]
    fn happy_path_tool_call_then_answer() {
        // upholds: AGENT-1 — valid call → tool result → final, in one round.
        let tmp = tmp_with_file("note.txt", "the sky is blue");
        let tools = Tools::new().with(ReadFile::new(Dir::new(tmp.path())));
        let mut model = Scripted::new(&[
            &call("read_file", "note.txt"),
            "Based on the file, the sky is blue.",
        ]);

        let mut agent = Agent::new(
            &mut model,
            &tools,
            JsonToolCall,
            PlainTemplate,
            "You are a helper.",
            5,
        );
        let run = agent.run("What does note.txt say?").unwrap();

        assert_eq!(run.stop, AgentStop::Final);
        assert_eq!(run.steps, 1);
        assert_eq!(run.answer, "Based on the file, the sky is blue.");
        // the tool result was fed back into the transcript
        assert!(run
            .transcript
            .iter()
            .any(|t| t.role == Role::Tool && t.content.contains("the sky is blue")));
    }

    #[test]
    fn unknown_tool_recovers_then_answers() {
        // upholds: AGENT-2, PROTO-1 — an unknown tool yields an error result the
        // model sees, then recovers to a final answer.
        let tools = Tools::new();
        let mut model =
            Scripted::new(&[&call("nonexistent", "x"), "Sorry, I will just answer: 4."]);

        let mut agent = Agent::new(&mut model, &tools, JsonToolCall, PlainTemplate, "helper", 5);
        let run = agent.run("compute 2+2").unwrap();

        assert_eq!(run.stop, AgentStop::Final);
        assert_eq!(run.answer, "Sorry, I will just answer: 4.");
        assert!(run.transcript.iter().any(|t| t.role == Role::Tool
            && t.content.contains("error")
            && t.content.contains("unknown tool")));
    }

    #[test]
    fn malformed_call_recovers_then_answers() {
        // upholds: PROTO-1 — a malformed call becomes an error turn, not a panic
        // or silent mis-execution; the model recovers.
        let tools = Tools::new();
        let mut model =
            Scripted::new(&["<tool_call>{not valid json}</tool_call>", "Answer: done."]);

        let mut agent = Agent::new(&mut model, &tools, JsonToolCall, PlainTemplate, "helper", 5);
        let run = agent.run("go").unwrap();

        assert_eq!(run.stop, AgentStop::Final);
        assert_eq!(run.answer, "Answer: done.");
        assert!(run
            .transcript
            .iter()
            .any(|t| t.role == Role::Tool && t.content.contains("malformed tool call")));
    }

    #[test]
    fn max_steps_terminates_a_looping_model() {
        // upholds: AGENT-1 — a model that only ever calls tools still terminates,
        // bounded by max_steps.
        let tmp = tmp_with_file("a.txt", "x");
        let tools = Tools::new().with(ReadFile::new(Dir::new(tmp.path())));
        // more tool calls than the budget; none is ever a final answer.
        let c = call("read_file", "a.txt");
        let mut model = Scripted::new(&[&c, &c, &c, &c, &c]);

        let mut agent = Agent::new(&mut model, &tools, JsonToolCall, PlainTemplate, "helper", 3);
        let run = agent.run("loop").unwrap();

        assert_eq!(run.stop, AgentStop::MaxSteps);
        assert_eq!(run.steps, 3);
        assert!(run.answer.is_empty());
    }

    #[test]
    fn run_and_run_with_agree() {
        // upholds: AGENT-1 — run is the acc=() specialization of run_with; both
        // produce the same Run, and run_with observes the events in order.
        let tmp = tmp_with_file("note.txt", "contents here");
        let script = [call("read_file", "note.txt"), "Final answer.".to_string()];
        let texts: Vec<&str> = script.iter().map(String::as_str).collect();

        let tools = Tools::new().with(ReadFile::new(Dir::new(tmp.path())));

        let mut m1 = Scripted::new(&texts);
        let run_plain = Agent::new(&mut m1, &tools, JsonToolCall, PlainTemplate, "helper", 5)
            .run("q")
            .unwrap();

        let mut m2 = Scripted::new(&texts);
        let (events, run_folded) =
            Agent::new(&mut m2, &tools, JsonToolCall, PlainTemplate, "helper", 5)
                .run_with("q", Vec::new(), |mut acc, event| {
                    acc.push(event);
                    Ok(ControlFlow::Continue(acc))
                })
                .unwrap();

        // the two runs agree on the observable outcome
        assert_eq!(run_plain.answer, run_folded.answer);
        assert_eq!(run_plain.steps, run_folded.steps);
        assert_eq!(run_plain.stop, run_folded.stop);
        assert_eq!(run_plain.transcript.len(), run_folded.transcript.len());

        // run_with saw ToolCall, ToolStarted, ToolOutcome, Final in order
        // (Fragment events interleave — AGENT-4 — and are checked below).
        let marks: Vec<&AgentEvent> = events
            .iter()
            .filter(|e| !matches!(e, AgentEvent::Fragment { .. }))
            .collect();
        assert!(matches!(marks[0], AgentEvent::ToolCall(_)));
        assert!(matches!(marks[1], AgentEvent::ToolStarted(_)));
        assert!(matches!(marks[2], AgentEvent::ToolOutcome(_)));
        assert!(matches!(marks[3], AgentEvent::Final(_)));

        // upholds: AGENT-4 — the streamed answer fragments reconstruct the
        // final answer (the tool-call step contributed none: markup never
        // reaches the answer channel).
        let streamed: String = events
            .iter()
            .filter_map(|e| match e {
                AgentEvent::Fragment {
                    channel: Channel::Answer,
                    text,
                } => Some(text.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(streamed.trim(), run_folded.answer);
    }

    #[test]
    fn caller_break_stops_with_stopped() {
        // upholds: AGENT-1 — run_with's ControlFlow::Break halts the run, and
        // that outcome is reported precisely as AgentStop::Stopped (distinct from
        // Final / MaxSteps), before the step budget advances.
        let tmp = tmp_with_file("a.txt", "x");
        let tools = Tools::new().with(ReadFile::new(Dir::new(tmp.path())));
        let mut model = Scripted::new(&[&call("read_file", "a.txt"), "unreached"]);

        let mut agent = Agent::new(&mut model, &tools, JsonToolCall, PlainTemplate, "helper", 5);
        let (observed, run) = agent
            .run_with("q", 0usize, |n, event| {
                Ok(match event {
                    AgentEvent::ToolOutcome(_) => ControlFlow::Break(n + 1),
                    _ => ControlFlow::Continue(n + 1),
                })
            })
            .unwrap();

        assert_eq!(run.stop, AgentStop::Stopped);
        assert_eq!(run.steps, 0, "break happens before the round is counted");
        assert!(run.answer.is_empty());
        assert_eq!(
            observed, 3,
            "saw ToolCall, ToolStarted, then ToolResult, then stopped"
        );
    }

    #[test]
    fn multi_step_two_tool_calls_then_answer() {
        // upholds: AGENT-1 — the loop chains multiple tool rounds then answers
        // (the no-GPU analogue of the servers.txt + runbook.md demo).
        let tmp = tmp_with_file("servers.txt", "web02: DOWN");
        {
            let mut f = std::fs::File::create(tmp.path().join("runbook.md")).unwrap();
            write!(f, "restart it and page on-call").unwrap();
        }
        let tools = Tools::new().with(ReadFile::new(Dir::new(tmp.path())));
        let mut model = Scripted::new(&[
            &call("read_file", "servers.txt"),
            &call("read_file", "runbook.md"),
            "Restart web02 and page the on-call.",
        ]);

        let mut agent = Agent::new(&mut model, &tools, JsonToolCall, PlainTemplate, "helper", 5);
        let run = agent.run("what should I do?").unwrap();

        assert_eq!(run.stop, AgentStop::Final);
        assert_eq!(run.steps, 2);
        assert_eq!(run.answer, "Restart web02 and page the on-call.");
        let tool_turns: Vec<&Turn> = run
            .transcript
            .iter()
            .filter(|t| t.role == Role::Tool)
            .collect();
        assert_eq!(tool_turns.len(), 2);
        assert!(tool_turns[0].content.contains("DOWN"));
        assert!(tool_turns[1].content.contains("restart it"));
    }

    #[test]
    fn multi_step_recovers_from_failed_tool() {
        // upholds: PROTO-1 — a failed tool call (missing file) yields an error
        // result the model recovers from, then a correct answer.
        let tmp = tmp_with_file("real.txt", "the data");
        let tools = Tools::new().with(ReadFile::new(Dir::new(tmp.path())));
        let mut model = Scripted::new(&[
            &call("read_file", "missing.txt"),
            &call("read_file", "real.txt"),
            "Got it: the data.",
        ]);

        let mut agent = Agent::new(&mut model, &tools, JsonToolCall, PlainTemplate, "helper", 5);
        let run = agent.run("q").unwrap();

        assert_eq!(run.stop, AgentStop::Final);
        assert_eq!(run.steps, 2);
        let tool_turns: Vec<&Turn> = run
            .transcript
            .iter()
            .filter(|t| t.role == Role::Tool)
            .collect();
        assert!(tool_turns[0].content.contains("error"));
        assert!(!tool_turns[1].content.contains("error"));
        assert_eq!(run.answer, "Got it: the data.");
    }

    #[test]
    fn immediate_answer_when_no_tool_call() {
        // upholds: AGENT-1 — a model that answers directly yields 0 tool rounds.
        let tools = Tools::new();
        let mut model = Scripted::new(&["The answer is 42."]);
        let mut agent = Agent::new(&mut model, &tools, JsonToolCall, PlainTemplate, "helper", 5);
        let run = agent.run("q").unwrap();
        assert_eq!(run.stop, AgentStop::Final);
        assert_eq!(run.steps, 0);
        assert_eq!(run.answer, "The answer is 42.");
    }

    #[test]
    fn session_history_carries_across_runs() {
        // upholds: AGENT-3 — a second run's prompt re-renders the first
        // exchange's user turn and final answer.
        let tools = Tools::new();
        let mut model = Scripted::new(&["Paris.", "About two million."]);
        let mut agent = Agent::new(&mut model, &tools, JsonToolCall, PlainTemplate, "helper", 5);

        let run1 = agent.run("What is the capital of France?").unwrap();
        assert_eq!(run1.answer, "Paris.");
        let run2 = agent.run("And its population?").unwrap();
        assert_eq!(run2.answer, "About two million.");

        // History holds both completed exchanges, in order.
        let roles: Vec<Role> = agent.history().iter().map(|t| t.role).collect();
        assert_eq!(
            roles,
            [Role::User, Role::Assistant, Role::User, Role::Assistant]
        );

        drop(agent);
        let second_prompt = &model.prompts[1];
        assert!(second_prompt.contains("What is the capital of France?"));
        assert!(second_prompt.contains("Paris."));
        assert!(second_prompt.contains("And its population?"));
    }

    /// The Answer-channel fragments of a fold's events, concatenated.
    fn answer_stream(events: &[AgentEvent]) -> String {
        events
            .iter()
            .filter_map(|e| match e {
                AgentEvent::Fragment {
                    channel: Channel::Answer,
                    text,
                } => Some(text.as_str()),
                _ => None,
            })
            .collect()
    }

    #[test]
    fn tool_call_markup_never_streams_on_the_answer_channel() {
        // upholds: AGENT-4 — a step that is a tool call emits no answer
        // fragments (the opener gate withholds from the first marker byte
        // on); the final prose step streams normally, and its fragments
        // reconstruct the run's answer.
        let tmp = tmp_with_file("note.txt", "hello");
        let tools = Tools::new().with(ReadFile::new(Dir::new(tmp.path())));
        let mut model = Scripted::new(&[&call("read_file", "note.txt"), "All done here."]);
        let mut agent = Agent::new(&mut model, &tools, JsonToolCall, PlainTemplate, "helper", 5);
        let (events, run) = agent
            .run_with("q", Vec::new(), |mut acc, e| {
                acc.push(e);
                Ok(ControlFlow::Continue(acc))
            })
            .unwrap();
        let streamed = answer_stream(&events);
        assert!(
            !streamed.contains("<tool_call>") && !streamed.contains("</tool_call>"),
            "markup leaked: {streamed:?}"
        );
        assert_eq!(streamed.trim(), run.answer);
    }

    #[test]
    fn prose_with_angle_brackets_survives_the_gate() {
        // upholds: AGENT-4 — text that merely *looks* like it might open a
        // tool call (a `<` mid-prose, even `<tool…` lookalikes) is released
        // once it diverges from the marker; nothing is swallowed.
        let tools = Tools::new();
        let text = "For x < y, use <toolboxes> as <tool_kits do.";
        let mut model = Scripted::new(&[text]);
        let mut agent = Agent::new(&mut model, &tools, JsonToolCall, PlainTemplate, "helper", 5);
        let (events, run) = agent
            .run_with("q", Vec::new(), |mut acc, e| {
                acc.push(e);
                Ok(ControlFlow::Continue(acc))
            })
            .unwrap();
        assert_eq!(answer_stream(&events).trim(), text);
        assert_eq!(run.answer, text);
    }

    #[test]
    fn reasoning_streams_on_the_reasoning_channel() {
        // upholds: AGENT-4 + REASON-1 — chain-of-thought streams live on the
        // reasoning channel and never contaminates the answer stream; the
        // answer fragments still reconstruct the surfaced answer.
        let tools = Tools::new();
        let mut model = Scripted::new(&["<think>working it out</think>The answer is 42."]);
        let mut agent = Agent::new(&mut model, &tools, JsonToolCall, PlainTemplate, "helper", 5);
        let (events, run) = agent
            .run_with("q", Vec::new(), |mut acc, e| {
                acc.push(e);
                Ok(ControlFlow::Continue(acc))
            })
            .unwrap();
        let reasoned: String = events
            .iter()
            .filter_map(|e| match e {
                AgentEvent::Fragment {
                    channel: Channel::Reasoning,
                    text,
                } => Some(text.as_str()),
                _ => None,
            })
            .collect();
        assert!(reasoned.contains("working it out"), "{reasoned:?}");
        assert!(
            !reasoned.contains("</think>"),
            "markers stay out: {reasoned:?}"
        );
        let streamed = answer_stream(&events);
        assert!(!streamed.contains("think"), "{streamed:?}");
        assert_eq!(streamed.trim(), run.answer);
        assert_eq!(run.answer, "The answer is 42.");
    }

    #[test]
    fn a_break_on_a_fragment_stops_token_level() {
        // upholds: AGENT-4 + AGENT-3 — breaking the fold on the first
        // streamed fragment cancels the decode in flight: the run reports
        // Stopped, the decode never ran to completion, and nothing persists
        // into session history.
        let tools = Tools::new();
        let mut model = Scripted::new(&["a long final answer that will be interrupted early"]);
        let mut agent = Agent::new(&mut model, &tools, JsonToolCall, PlainTemplate, "helper", 5);
        let cancel = Cancel::new();
        let (fragments, run) = agent
            .run_with_cancellable("q", &cancel, 0usize, |n, e| {
                Ok(match e {
                    AgentEvent::Fragment { .. } => ControlFlow::Break(n + 1),
                    _ => ControlFlow::Continue(n),
                })
            })
            .unwrap();
        assert_eq!(fragments, 1, "stopped on the first fragment");
        assert_eq!(run.stop, AgentStop::Stopped);
        assert!(cancel.is_cancelled(), "the break flipped the cancel");
        drop(agent);
        let mut model2 = Scripted::new(&[]);
        let agent2 = Agent::new(
            &mut model2,
            &tools,
            JsonToolCall,
            PlainTemplate,
            "helper",
            5,
        );
        assert!(agent2.history().is_empty());
    }

    #[test]
    fn stop_string_termination_is_not_a_cancel() {
        // upholds: AGENT-4 — the engine reports `StopReason::Stopped` for
        // any early fold break, which includes ordinary stop-string
        // termination (every successful tool-call step ends at
        // `</tool_call>`). The agent must read the cancel *handle*, not the
        // ambiguous stop reason: an engine-like script whose tool-call
        // completion says `Stopped` still runs to a Final answer.
        let tmp = tmp_with_file("note.txt", "hi");
        let tools = Tools::new().with(ReadFile::new(Dir::new(tmp.path())));
        let mut model = Scripted {
            script: vec![
                Completion {
                    text: call("read_file", "note.txt"),
                    stop: StopReason::Stopped, // engine: cut at the stop string
                },
                Completion {
                    text: "Read it.".to_string(),
                    stop: StopReason::Eos,
                },
            ],
            i: 0,
            prompts: Vec::new(),
        };
        let mut agent = Agent::new(&mut model, &tools, JsonToolCall, PlainTemplate, "helper", 5);
        let run = agent.run("q").unwrap();
        assert_eq!(run.stop, AgentStop::Final, "not misread as a cancel");
        assert_eq!(run.answer, "Read it.");
    }

    #[test]
    fn answer_gate_holds_markers_and_releases_lookalikes() {
        // upholds: AGENT-4 — the gate's unit algebra: complete openers
        // suppress from the marker on; partial-opener tails are held, then
        // released when they diverge or the stream ends.
        let mut gate = AnswerGate::new("<tool_call>".to_string());
        assert_eq!(gate.push("Hello "), Some("Hello ".to_string()));
        assert_eq!(gate.push("<tool"), None); // could still become the opener
        assert_eq!(gate.push("boxes> ok"), Some("<toolboxes> ok".to_string()));
        assert_eq!(gate.push("<tool_call>{}"), None); // suppressed from here on
        assert_eq!(gate.push("more"), None);
        assert_eq!(gate.finish(), None);

        let mut gate = AnswerGate::new("<tool_call>".to_string());
        assert_eq!(
            gate.push("ends on a cliff <tool_ca"),
            Some("ends on a cliff ".to_string())
        );
        assert_eq!(gate.finish(), Some("<tool_ca".to_string()));
    }

    #[test]
    fn transplanted_history_seeds_the_next_run() {
        // upholds: AGENT-3 — a host switching a plain chat to a tool-bearing
        // agent mid-session (the first origin grant) seeds the agent with the
        // chat's user/answer turns, and the next run's prompt re-renders them.
        let tools = Tools::new();
        let mut model = Scripted::new(&["Blue, as you said."]);
        let prior = vec![
            Turn {
                role: Role::User,
                content: "My favourite colour is blue.".to_string(),
            },
            Turn {
                role: Role::Assistant,
                content: "Noted: blue.".to_string(),
            },
        ];
        let mut agent = Agent::new(&mut model, &tools, JsonToolCall, PlainTemplate, "helper", 5)
            .with_history(prior);

        let run = agent.run("What colour did I say?").unwrap();
        assert_eq!(run.answer, "Blue, as you said.");
        drop(agent);
        let prompt = &model.prompts[0];
        assert!(prompt.contains("My favourite colour is blue."));
        assert!(prompt.contains("Noted: blue."));
    }

    #[test]
    fn tool_rounds_are_ephemeral_across_runs() {
        // upholds: AGENT-3 — run 1's tool result is fed back within run 1 but
        // never re-rendered into run 2's prompt; only the final answer is.
        let tmp = tmp_with_file("note.txt", "SECRET-77");
        let tools = Tools::new().with(ReadFile::new(Dir::new(tmp.path())));
        let mut model = Scripted::new(&[
            &call("read_file", "note.txt"),
            "The note mentions a number.",
            "Yes, just one.",
        ]);
        let mut agent = Agent::new(&mut model, &tools, JsonToolCall, PlainTemplate, "helper", 5);

        let run1 = agent.run("What does note.txt mention?").unwrap();
        assert_eq!(run1.answer, "The note mentions a number.");
        let run2 = agent.run("Just one?").unwrap();
        assert_eq!(run2.answer, "Yes, just one.");

        drop(agent);
        // Within run 1 the tool result was visible (step 2's prompt)…
        assert!(model.prompts[1].contains("SECRET-77"));
        // …but run 2's prompt carries only the exchange's conclusion.
        let run2_prompt = model.prompts.last().unwrap();
        assert!(run2_prompt.contains("The note mentions a number."));
        assert!(!run2_prompt.contains("SECRET-77"));
    }

    #[test]
    fn interrupted_runs_leave_history_untouched() {
        // upholds: AGENT-3 — MaxSteps and Stopped runs persist nothing; a
        // subsequent completed run starts the history.
        let tmp = tmp_with_file("note.txt", "x");
        let tools = Tools::new().with(ReadFile::new(Dir::new(tmp.path())));
        let looping = call("read_file", "note.txt");
        let mut model = Scripted::new(&[&looping, &looping, "Done: x."]);
        let mut agent = Agent::new(&mut model, &tools, JsonToolCall, PlainTemplate, "helper", 2);

        let run1 = agent.run("read forever").unwrap();
        assert_eq!(run1.stop, AgentStop::MaxSteps);
        assert!(agent.history().is_empty());

        let run2 = agent.run("and now?").unwrap();
        assert_eq!(run2.stop, AgentStop::Final);
        assert_eq!(agent.history().len(), 2);

        agent.reset();
        assert!(agent.history().is_empty());
    }

    #[test]
    fn reasoning_is_stripped_from_the_surfaced_answer() {
        // The agent surfaces (and re-feeds) only the answer; the reasoning span
        // is dropped from the transcript (REASON-1). Marker coverage and edge
        // cases are tested in `crate::reasoning`.
        let tools = Tools::new();
        let mut model = Scripted::new(&["<think>2+2 is 4</think>The answer is 4."]);
        let mut agent = Agent::new(
            &mut model,
            &tools,
            JsonToolCall,
            PlainTemplate,
            "be brief",
            4,
        );
        let run = agent.run("what is 2+2?").unwrap();
        assert_eq!(run.answer, "The answer is 4.");
        assert_eq!(run.stop, AgentStop::Final);
        // The reasoning span is absent from the re-rendered transcript.
        let assistant = run
            .transcript
            .iter()
            .find(|t| t.role == Role::Assistant)
            .unwrap();
        assert_eq!(assistant.content, "The answer is 4.");
        assert!(!assistant.content.contains("<think>"));
    }

    // End-to-end agent runs over a real, tool-trained model (Qwen2.5-Instruct,
    // Qwen2 arch). Gated: need the weights and `YATIMA_E2E=1`, skip fast
    // otherwise. Assertions are *tool-side* (the tool returns the real file
    // content regardless of how the model phrases its answer), so they are
    // deterministic given that Qwen2.5 reliably calls the tool. These lock in the
    // manual CLI demos. Run with `--features metal --nocapture` to read the
    // transcripts.

    /// Skip-guarded Qwen model dir, or `None` if the e2e gate/weights are absent.
    fn e2e_qwen_dir() -> Option<PathBuf> {
        if std::env::var_os("YATIMA_E2E").is_none() {
            eprintln!("skipping e2e: set YATIMA_E2E=1 to run");
            return None;
        }
        let dir = crate::model_dir(
            &crate::models_root(),
            &crate::ModelId::parse("Qwen/Qwen2.5-7B-Instruct").unwrap(),
        );
        if !dir.join("config.json").exists() {
            eprintln!("skipping e2e: weights absent at {}", dir.display());
            return None;
        }
        Some(dir)
    }

    fn dump(run: &Run) {
        for turn in &run.transcript {
            eprintln!("── {:?} ──\n{}\n", turn.role, turn.content);
        }
        eprintln!("[{} steps, {:?}]", run.steps, run.stop);
    }

    fn tool_turns(run: &Run) -> Vec<&Turn> {
        run.transcript
            .iter()
            .filter(|t| t.role == Role::Tool)
            .collect()
    }

    use std::path::PathBuf;

    fn qwen_agent<'a>(
        engine: &'a mut crate::Engine,
        tools: &'a Tools,
        max_steps: usize,
    ) -> Agent<'a, crate::Engine, crate::tool::QwenToolCall, crate::template::ChatMlTemplate> {
        Agent::new(
            engine,
            tools,
            crate::tool::QwenToolCall,
            crate::template::ChatMlTemplate,
            "You are a helpful assistant. Read files with the provided tools.",
            max_steps,
        )
    }

    // Both demos in one test so the ~15 GB model is loaded once and reused (an
    // Engine is one-generation-at-a-time; scenarios run sequentially on the same
    // engine). Assertions are tool-side, hence deterministic given Qwen calls the
    // tool. Run with `--features metal --nocapture` to watch the transcripts.
    #[test]
    fn e2e_agent_demos() -> Result<()> {
        let Some(dir) = e2e_qwen_dir() else {
            return Ok(());
        };
        let mut engine = crate::Engine::load(&dir, crate::device(false)?)?;

        // Single-read demo (the "launch code" CLI run).
        {
            let tmp = tmp_with_file("secret.txt", "The launch code is ZEBRA-42.");
            let tools = Tools::new().with(ReadFile::new(Dir::new(tmp.path())));
            let run = qwen_agent(&mut engine, &tools, 4)
                .run("Read secret.txt and tell me the launch code.")?;
            dump(&run);
            assert!(run.steps >= 1, "the model should have called a tool");
            assert!(run.steps <= 4, "AGENT-1: steps stay within max_steps");
            assert!(
                tool_turns(&run)
                    .iter()
                    .any(|t| t.content.contains("ZEBRA-42") && !t.content.contains("error")),
                "the tool must have fed back the real file content"
            );
        }

        // Multi-step investigation demo (servers.txt + runbook.md).
        {
            let tmp = tmp_with_file("servers.txt", "web01: healthy\nweb02: DOWN\ndb01: healthy");
            {
                let mut f = std::fs::File::create(tmp.path().join("runbook.md")).unwrap();
                write!(
                    f,
                    "If a web server is DOWN, restart it and page the on-call."
                )
                .unwrap();
            }
            let tools = Tools::new()
                .with(ReadFile::new(Dir::new(tmp.path())))
                .with(crate::ListDir::new(Dir::new(tmp.path())));
            let run = qwen_agent(&mut engine, &tools, 6).run(
                "Check servers.txt for any DOWN servers, then consult runbook.md for the \
                 procedure, and tell me exactly what action to take.",
            )?;
            dump(&run);
            assert!(run.steps >= 1, "the model should have called a tool");
            assert!(run.steps <= 6, "AGENT-1: steps stay within max_steps");
            let reads: String = tool_turns(&run).iter().map(|t| t.content.clone()).collect();
            assert!(reads.contains("DOWN"), "servers.txt content should be read");
        }

        Ok(())
    }

    /// The GGUF quantized path through the *agent* (a 32B-Q4 Qwen2.5). Gated on
    /// `YATIMA_E2E=1` and the cached GGUF; proves the quantized model drives the
    /// full read-file-then-answer loop. Self-contained: it summarizes a known tmp
    /// file, so assertions (tool fired, Final, the answer mentions the subject)
    /// are stable. Run with `--features metal --nocapture`.
    #[test]
    fn e2e_gguf_agent_reads_and_answers() -> Result<()> {
        if std::env::var_os("YATIMA_E2E").is_none() {
            eprintln!("skipping e2e: set YATIMA_E2E=1 to run");
            return Ok(());
        }
        let dir = crate::model_dir(
            &crate::models_root(),
            &crate::ModelId::parse("bartowski/Qwen2.5-32B-Instruct-GGUF").unwrap(),
        );
        if !crate::is_model_present(&dir) {
            eprintln!("skipping e2e: GGUF model not cached at {}", dir.display());
            return Ok(());
        }

        let tmp = tmp_with_file(
            "about.txt",
            "Yatima is a Rust runtime for language-integrated LLMs: it calls a local \
             model as an in-process function and lets it act through capability-scoped tools.",
        );
        let tools = Tools::new().with(ReadFile::new(Dir::new(tmp.path())));

        let mut engine = crate::Engine::load(&dir, crate::device(false)?)?;
        let run = qwen_agent(&mut engine, &tools, 4)
            .run("Read about.txt and tell me in one sentence what Yatima is.")?;
        dump(&run);

        assert!(run.steps >= 1, "the model should have called read_file");
        assert_eq!(run.stop, AgentStop::Final);
        assert!(
            tool_turns(&run)
                .iter()
                .any(|t| t.content.contains("Rust runtime") && !t.content.contains("error")),
            "the tool must have fed back about.txt"
        );
        let answer = run.answer.to_lowercase();
        assert!(
            answer.contains("rust") || answer.contains("runtime") || answer.contains("llm"),
            "the quantized model's answer should be grounded in the file: {:?}",
            run.answer
        );
        Ok(())
    }
}
