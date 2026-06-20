//! The agent loop — a fold over turns.
//!
//! One level up from `generate_with` (which folds *tokens* into a value), the
//! agent folds *turns*: the model emits a tool call, a capability-scoped tool
//! runs, its result is fed back, and the loop repeats until the model answers or
//! `max_steps` is reached. [`Agent::run`] collects the final answer;
//! [`Agent::run_with`] is the fold a future actor/TUI streams [`AgentEvent`]s
//! into. The loop is sync (turns are sequential and compute-bound) and provable
//! against a scripted [`Completer`] with no GPU.

use crate::completer::Completer;
use crate::template::PromptTemplate;
use crate::tool::{ToolCall, ToolCallCodec, ToolResult, Tools};
use crate::GenOpts;
use anyhow::Result;
use std::ops::ControlFlow;

/// A role in the transcript — mirrors the de-facto standard (system / user /
/// assistant / tool).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    System,
    User,
    Assistant,
    Tool,
}

/// One transcript entry.
#[derive(Debug, Clone)]
pub struct Turn {
    pub role: Role,
    pub content: String,
}

/// An observable step of a run, delivered to [`Agent::run_with`]'s fold.
#[derive(Debug, Clone)]
pub enum AgentEvent {
    ToolCall(ToolCall),
    ToolResult(ToolResult),
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
pub struct Agent<'a, C: Completer, K: ToolCallCodec, T: PromptTemplate> {
    completer: &'a mut C,
    tools: &'a Tools,
    codec: K,
    template: T,
    system: String,
    max_steps: usize,
    opts: GenOpts,
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
        }
    }

    /// Override the generation options used for each turn (default greedy).
    pub fn with_opts(mut self, opts: GenOpts) -> Agent<'a, C, K, T> {
        self.opts = opts;
        self
    }

    /// Run to a final answer (or `max_steps`), discarding per-step events.
    pub fn run(&mut self, user: &str) -> Result<Run> {
        let ((), run) = self.run_with(user, (), |(), _event| Ok(ControlFlow::Continue(())))?;
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
        let system = format!(
            "{}\n\n{}",
            self.system,
            self.codec.render_system(&self.tools.specs())
        );
        let mut transcript = vec![
            Turn {
                role: Role::System,
                content: system,
            },
            Turn {
                role: Role::User,
                content: user.to_string(),
            },
        ];

        let stops = self.codec.stop_strings();
        let mut acc = init;
        let mut steps = 0usize;
        let mut answer = String::new();
        let stop;

        loop {
            let prompt = self.template.render(&transcript);
            let completion = self.completer.complete(&prompt, &self.opts, &stops)?;
            let text = completion.text;
            transcript.push(Turn {
                role: Role::Assistant,
                content: text.clone(),
            });

            match self.codec.parse(&text) {
                // A plain answer: the run is done (a model's reasoning block, if
                // any, is stripped from the surfaced answer).
                None => {
                    let final_answer = strip_think(&text);
                    match step(acc, AgentEvent::Final(final_answer.clone()))? {
                        ControlFlow::Continue(a) | ControlFlow::Break(a) => acc = a,
                    }
                    answer = final_answer;
                    stop = AgentStop::Final;
                    break;
                }
                // A tool call (well-formed or not): dispatch / make an error
                // result, feed it back, and continue under the step budget.
                Some(parsed) => {
                    let result = match parsed {
                        Ok(call) => {
                            match step(acc, AgentEvent::ToolCall(call.clone()))? {
                                ControlFlow::Continue(a) => acc = a,
                                ControlFlow::Break(a) => {
                                    acc = a;
                                    stop = AgentStop::Stopped;
                                    break;
                                }
                            }
                            self.tools.dispatch(&call)
                        }
                        Err(e) => ToolResult {
                            name: String::new(),
                            content: format!("malformed tool call: {e}"),
                            is_error: true,
                        },
                    };

                    transcript.push(Turn {
                        role: Role::Tool,
                        content: render_result(&result),
                    });
                    match step(acc, AgentEvent::ToolResult(result))? {
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

/// Strip a leading reasoning block from a model's answer: keep only what follows
/// the last `</think>`. A no-op for output with no think block (so it is safe
/// for any template/codec).
fn strip_think(text: &str) -> String {
    match text.rfind("</think>") {
        Some(i) => text[i + "</think>".len()..].trim().to_string(),
        None => text.trim().to_string(),
    }
}

/// Render a tool result as the `tool`-turn content the model reads back.
fn render_result(result: &ToolResult) -> String {
    let tag = if result.is_error { "error" } else { "ok" };
    format!("[{} {}] {}", result.name, tag, result.content)
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
    struct Scripted {
        script: Vec<Completion>,
        i: usize,
    }

    impl Scripted {
        fn new(texts: &[&str]) -> Scripted {
            let script = texts
                .iter()
                .map(|t| Completion {
                    text: (*t).to_string(),
                    stop: StopReason::Stopped,
                })
                .collect();
            Scripted { script, i: 0 }
        }
    }

    impl Completer for Scripted {
        fn complete(&mut self, _: &str, _: &GenOpts, _: &[String]) -> Result<Completion> {
            let c = self
                .script
                .get(self.i)
                .cloned()
                .unwrap_or_else(|| panic!("scripted completer exhausted at step {}", self.i));
            self.i += 1;
            Ok(c)
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

        // and run_with saw ToolCall, ToolResult, Final in order
        assert!(matches!(events[0], AgentEvent::ToolCall(_)));
        assert!(matches!(events[1], AgentEvent::ToolResult(_)));
        assert!(matches!(events[2], AgentEvent::Final(_)));
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
                    AgentEvent::ToolResult(_) => ControlFlow::Break(n + 1),
                    _ => ControlFlow::Continue(n + 1),
                })
            })
            .unwrap();

        assert_eq!(run.stop, AgentStop::Stopped);
        assert_eq!(run.steps, 0, "break happens before the round is counted");
        assert!(run.answer.is_empty());
        assert_eq!(observed, 2, "saw ToolCall then ToolResult, then stopped");
    }

    // End-to-end agent loop over the real model + a real ReadFile capability.
    // Gated like the engine e2e: needs the weights and `YATIMA_E2E=1`, skips
    // fast otherwise. Best-effort — base distills aren't reliably tool-callers,
    // so we assert the loop *terminates cleanly within budget* (the contract),
    // not that the model chose to call a tool. Run with `--features metal` and
    // `--nocapture` to read the transcript.
    #[test]
    fn e2e_agent_loop_terminates() -> Result<()> {
        if std::env::var_os("YATIMA_E2E").is_none() {
            eprintln!("skipping e2e: set YATIMA_E2E=1 to run");
            return Ok(());
        }
        let repo = crate::ModelId::parse("deepseek-ai/DeepSeek-R1-Distill-Qwen-7B").unwrap();
        let dir = crate::model_dir(&crate::models_root(), &repo);
        if !dir.join("config.json").exists() {
            eprintln!("skipping e2e: weights absent at {}", dir.display());
            return Ok(());
        }

        let tmp = tmp_with_file("fact.txt", "The capital of France is Paris.");
        let tools = Tools::new()
            .with(ReadFile::new(Dir::new(tmp.path())))
            .with(crate::ListDir::new(Dir::new(tmp.path())));

        let mut engine = crate::Engine::load(&dir, crate::device(false)?)?;
        let mut agent = Agent::new(
            &mut engine,
            &tools,
            crate::tool::DeepSeekToolCall,
            crate::template::DeepSeekR1Template,
            "You are a helpful assistant. Use the read_file tool to read files.",
            4,
        );
        let run = agent.run("Read fact.txt and tell me the capital of France.")?;

        for turn in &run.transcript {
            eprintln!("── {:?} ──\n{}\n", turn.role, turn.content);
        }
        eprintln!("[{} steps, {:?}]", run.steps, run.stop);

        assert!(run.steps <= 4, "AGENT-1: steps stay within max_steps");
        assert!(
            matches!(run.stop, AgentStop::Final | AgentStop::MaxSteps),
            "the loop terminates by answering or exhausting the budget"
        );
        Ok(())
    }
}
