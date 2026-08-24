//! Prompt templates — rendering a transcript into a model's *native* prompt
//! string.
//!
//! A model is acutely sensitive to its trained chat format: feed it a generic
//! `<|role|>` layout and it can destabilise (degenerate repetition, no
//! instruction-following). [`PromptTemplate`] is the boundary that makes the
//! format per-model; [`ChatMlTemplate`] matches Qwen2.5's trained format, and
//! [`PlainTemplate`] keeps the minimal layout for models with no known template
//! and for tests.

use crate::reasoning::{
    split_reasoning, split_seeded_reasoning, AtemInterpreter, Reasoned, ResponseClassifier,
    ATEM_EOT as MUSE_EOT, ATEM_MESSAGE as MUSE_MESSAGE, ATEM_START as MUSE_START,
};
use crate::transcript::{render_json_inline, Role, ToolArguments, Turn};

/// Render a transcript into the prompt string fed to the model, ending with the
/// cue that makes the model speak next.
pub trait PromptTemplate {
    fn render(&self, turns: &[Turn]) -> String;

    /// Compose an agent's base system instruction with model-facing tool
    /// instructions. Most formats simply append the latter. Muse overrides
    /// this so its reasoning directive remains ahead of tool definitions, as
    /// required by the native template.
    fn compose_system(&self, system: &str, tool_instructions: &str) -> String {
        if tool_instructions.is_empty() {
            system.to_string()
        } else {
            format!("{system}\n\n{tool_instructions}")
        }
    }

    /// Construct the streaming classifier for replies in this template's
    /// protocol. Marker-based formats use the default; pre-seeded marker
    /// formats and Muse override it. Keeping this choice on the template avoids
    /// a second format switch drifting from final interpretation (REASON-1).
    fn classifier(&self) -> ResponseClassifier {
        ResponseClassifier::markers()
    }

    /// Interpret the model's **completed** reply into reasoning and answer —
    /// the inverse direction of [`render`](PromptTemplate::render), owned by
    /// the same template that selects [`classifier`](Self::classifier). The
    /// default is the marker-based [`split_reasoning`] (`<think>` dialects);
    /// [`MuseGlimmerTemplate`] feeds the completed reply through the same ATEM
    /// machine used for its live stream. `ChatSession` commits only a non-empty
    /// returned `answer` to history (REASON-1).
    fn interpret_response(&self, raw: &str) -> Reasoned {
        split_reasoning(raw)
    }
}

/// A boxed template is a template — lets a runtime-chosen `Box<dyn
/// PromptTemplate>` (e.g. the CLI's `--format`) satisfy generic bounds like
/// `ChatSession<_, T: PromptTemplate>`. Forwards **every** method: a default
/// here would silently strip an override (a boxed Muse template must keep its
/// ATEM interpretation).
impl<T: PromptTemplate + ?Sized> PromptTemplate for Box<T> {
    fn render(&self, turns: &[Turn]) -> String {
        (**self).render(turns)
    }

    fn classifier(&self) -> ResponseClassifier {
        (**self).classifier()
    }

    fn compose_system(&self, system: &str, tool_instructions: &str) -> String {
        (**self).compose_system(system, tool_instructions)
    }

    fn interpret_response(&self, raw: &str) -> Reasoned {
        (**self).interpret_response(raw)
    }
}

/// A minimal, backend-agnostic role layout. Not any model's trained format —
/// fine for scripted tests and as a fallback, but off-distribution for a real
/// instruction/reasoning model.
pub struct PlainTemplate;

impl PromptTemplate for PlainTemplate {
    fn render(&self, turns: &[Turn]) -> String {
        let mut s = String::new();
        for turn in turns {
            let (tag, content) = match turn {
                Turn::System(content) => ("system", content.clone()),
                Turn::User(content) => ("user", content.clone()),
                Turn::Assistant(content) => ("assistant", content.clone()),
                Turn::AssistantToolCall { name, arguments } => (
                    "assistant",
                    format!(
                        "<tool_call>{}</tool_call>",
                        render_call_json(name, "args", arguments)
                    ),
                ),
                Turn::ToolResult {
                    name,
                    content,
                    is_error,
                } => ("tool", render_tool_result(name, content, *is_error)),
            };
            s.push_str(&format!("<|{tag}|>\n{content}\n"));
        }
        s.push_str("<|assistant|>\n");
        s
    }
}

const IM_START: &str = "<|im_start|>";
const IM_END: &str = "<|im_end|>";

/// ChatML, as used by Qwen2.5-Instruct: `<|im_start|>{role}\n{content}<|im_end|>`
/// turns, no BOS, and a trailing `<|im_start|>assistant\n` cue. Tool results are
/// fed back the way Qwen expects — as a `user` turn wrapping a `<tool_response>`
/// (the tool-definition block lives in the system turn, produced by the codec).
pub struct ChatMlTemplate;

impl PromptTemplate for ChatMlTemplate {
    fn render(&self, turns: &[Turn]) -> String {
        render_chatml(turns, false)
    }
}

/// ChatML that **pre-seeds `<think>\n`** in the assistant cue — the format of
/// reasoning Qwen models (QwQ-32B, Qwen3 in thinking mode), whose chat template
/// ends `<|im_start|>assistant\n<think>\n`. Because the opener is in the prompt,
/// the model emits only the *closing* `</think>`. Its `PromptTemplate`
/// implementation therefore selects [`crate::ReasoningSplitter::seeded`] for
/// streaming and the matching seeded final interpretation (REASON-1).
pub struct ChatMlThinkTemplate;

impl PromptTemplate for ChatMlThinkTemplate {
    fn render(&self, turns: &[Turn]) -> String {
        render_chatml(turns, true)
    }

    fn classifier(&self) -> ResponseClassifier {
        ResponseClassifier::seeded_markers()
    }

    /// The cue opened the think block, so a close-less reply is a truncated
    /// chain-of-thought, not an answer (REASON-1).
    fn interpret_response(&self, raw: &str) -> Reasoned {
        split_seeded_reasoning(raw)
    }
}

/// Render ChatML turns; `seed_think` appends `<think>\n` to the assistant cue.
fn render_chatml(turns: &[Turn], seed_think: bool) -> String {
    let mut s = String::new();
    for turn in turns {
        match turn {
            Turn::System(content) => block(&mut s, "system", content),
            Turn::User(content) => block(&mut s, "user", content),
            Turn::Assistant(content) => block(&mut s, "assistant", content),
            Turn::AssistantToolCall { name, arguments } => block(
                &mut s,
                "assistant",
                &format!(
                    "<tool_call>\n{}\n</tool_call>",
                    render_call_json(name, "arguments", arguments)
                ),
            ),
            Turn::ToolResult {
                name,
                content,
                is_error,
            } => block(
                &mut s,
                "user",
                &format!(
                    "<tool_response>\n{}\n</tool_response>",
                    render_tool_result(name, content, *is_error)
                ),
            ),
        }
    }
    s.push_str(IM_START);
    s.push_str("assistant\n");
    if seed_think {
        s.push_str("<think>\n");
    }
    s
}

/// Append one `<|im_start|>{role}\n{content}<|im_end|>\n` block.
fn block(s: &mut String, role: &str, content: &str) {
    s.push_str(IM_START);
    s.push_str(role);
    s.push('\n');
    s.push_str(content);
    s.push_str(IM_END);
    s.push('\n');
}

fn render_call_json(name: &str, arguments_key: &str, arguments: &ToolArguments) -> String {
    let arguments = arguments.to_json_object();
    format!(
        "{{\"name\":{},\"{arguments_key}\":{arguments}}}",
        serde_json::to_string(name).expect("a string always serializes")
    )
}

fn render_tool_result(name: &str, content: &str, is_error: bool) -> String {
    let outcome = if is_error { "error" } else { "ok" };
    format!("[{name} {outcome}] {content}")
}

fn generic_tool_call(name: &str, arguments: &ToolArguments) -> String {
    format!(
        "<tool_call>{}</tool_call>",
        render_call_json(name, "arguments", arguments)
    )
}

/// Gemma-2's trained chat format: `<start_of_turn>{role}\n{content}<end_of_turn>`
/// turns with `assistant`→`model`. Gemma has **no system role**, so any system
/// text is folded into the next user turn. Emits **no `<bos>`**: Gemma's
/// tokenizer adds it automatically (its `TemplateProcessing` post-processor on
/// `encode(_, true)`), so a literal one would double-BOS. Chat-only (no tools).
pub struct GemmaTemplate;

impl PromptTemplate for GemmaTemplate {
    fn render(&self, turns: &[Turn]) -> String {
        let mut s = String::new();
        let mut pending_system: Option<String> = None;
        for turn in turns {
            match turn {
                Turn::System(content) => pending_system = Some(content.clone()),
                Turn::Assistant(content) => gemma_turn(&mut s, "model", content),
                Turn::AssistantToolCall { name, arguments } => {
                    gemma_turn(&mut s, "model", &generic_tool_call(name, arguments));
                }
                Turn::User(content) => {
                    let content = match pending_system.take() {
                        Some(sys) => format!("{sys}\n\n{content}"),
                        None => content.clone(),
                    };
                    gemma_turn(&mut s, "user", &content);
                }
                Turn::ToolResult {
                    name,
                    content,
                    is_error,
                } => gemma_turn(
                    &mut s,
                    "user",
                    &render_tool_result(name, content, *is_error),
                ),
            }
        }
        s.push_str("<start_of_turn>model\n");
        s
    }
}

fn gemma_turn(s: &mut String, role: &str, content: &str) {
    s.push_str("<start_of_turn>");
    s.push_str(role);
    s.push('\n');
    s.push_str(content);
    s.push_str("<end_of_turn>\n");
}

/// Mistral-v0.3's plain `[INST] … [/INST]` chat format (chat-only — **no**
/// `[TOOL_CALLS]` tool markers). System text folds into the first `[INST]`;
/// `[/INST]` is itself the generation cue. Emits **no `<s>`**: Mistral's
/// tokenizer adds it via `TemplateProcessing` on `encode(_, true)`, like Gemma.
pub struct MistralTemplate;

impl PromptTemplate for MistralTemplate {
    fn render(&self, turns: &[Turn]) -> String {
        let mut s = String::new();
        let mut pending_system: Option<String> = None;
        for turn in turns {
            match turn {
                Turn::System(content) => pending_system = Some(content.clone()),
                Turn::Assistant(content) => {
                    s.push(' ');
                    s.push_str(content);
                    s.push_str("</s>");
                }
                Turn::AssistantToolCall { name, arguments } => {
                    s.push(' ');
                    s.push_str(&generic_tool_call(name, arguments));
                    s.push_str("</s>");
                }
                Turn::User(content) => {
                    let content = match pending_system.take() {
                        Some(sys) => format!("{sys}\n\n{content}"),
                        None => content.clone(),
                    };
                    s.push_str("[INST] ");
                    s.push_str(&content);
                    s.push_str("[/INST]");
                }
                Turn::ToolResult {
                    name,
                    content,
                    is_error,
                } => {
                    s.push_str("[INST] ");
                    s.push_str(&render_tool_result(name, content, *is_error));
                    s.push_str("[/INST]");
                }
            }
        }
        s
    }
}

/// GLM-4's chat format: a literal `[gMASK]<sop>` prefix, then
/// `<|{role}|>\n{content}` turns (`system`/`user`/`assistant`, and
/// `observation` for tool output), with an `<|assistant|>\n` generation cue.
/// GLM **has a system role** (no folding). The `[gMASK]<sop>` prefix is emitted
/// literally — GLM's tokenizer has `add_bos_token` unset, so nothing adds it
/// otherwise (the emit-side of the no-double-BOS rule). Chat-only.
pub struct GlmTemplate;

impl PromptTemplate for GlmTemplate {
    fn render(&self, turns: &[Turn]) -> String {
        let mut s = String::from("[gMASK]<sop>");
        for turn in turns {
            let (role, content) = match turn {
                Turn::System(content) => ("system", content.clone()),
                Turn::User(content) => ("user", content.clone()),
                Turn::Assistant(content) => ("assistant", content.clone()),
                Turn::AssistantToolCall { name, arguments } => {
                    ("assistant", generic_tool_call(name, arguments))
                }
                Turn::ToolResult {
                    name,
                    content,
                    is_error,
                } => ("observation", render_tool_result(name, content, *is_error)),
            };
            s.push_str("<|");
            s.push_str(role);
            s.push_str("|>\n");
            s.push_str(&content);
        }
        s.push_str("<|assistant|>\n");
        s
    }
}

/// DeepSeek's chat format (DeepSeek-V2/V3 and the R1 distills): a literal
/// `<｜begin▁of▁sentence｜>` prefix, the system text raw (no wrapper, hoisted to
/// the front — DeepSeek has a system role), then `<｜User｜>{content}` and
/// `<｜Assistant｜>{content}<｜end▁of▁sentence｜>` turns, ending in a
/// `<｜Assistant｜><think>\n` cue. Two deliberate choices:
///
/// - **The BOS is emitted literally.** DeepSeek tokenizers carry no
///   `TemplateProcessing` post-processor (verified: plain ByteLevel), so nothing
///   adds it otherwise — the emit-side of the no-double-BOS rule (TMPL-1), like
///   [`GlmTemplate`].
/// - **The cue pre-seeds `<think>\n`.** R1 reasons reliably only when the
///   assistant turn opens inside the think block, so the model's output carries
///   the *closing* `</think>` but not the opening one — the close-without-open
///   case [`crate::split_reasoning`] handles (REASON-1). Prior assistant turns
///   are answer-only already (REASON-1), matching DeepSeek's own template, which
///   drops history reasoning via `content.split('</think>')[-1]`.
///
/// Chat-only (no DeepSeek tool-call markers); a `Tool` turn is rendered as a
/// single tool-output block for completeness.
pub struct DeepSeekTemplate;

const DS_BOS: &str = "<\u{ff5c}begin\u{2581}of\u{2581}sentence\u{ff5c}>";
const DS_USER: &str = "<\u{ff5c}User\u{ff5c}>";
const DS_ASSISTANT: &str = "<\u{ff5c}Assistant\u{ff5c}>";
const DS_EOS: &str = "<\u{ff5c}end\u{2581}of\u{2581}sentence\u{ff5c}>";
const DS_TOOL_OUT_BEGIN: &str = "<\u{ff5c}tool\u{2581}output\u{2581}begin\u{ff5c}>";
const DS_TOOL_OUT_END: &str = "<\u{ff5c}tool\u{2581}output\u{2581}end\u{ff5c}>";

impl PromptTemplate for DeepSeekTemplate {
    /// The cue opened the think block, so a close-less reply is a truncated
    /// chain-of-thought, not an answer (REASON-1).
    fn interpret_response(&self, raw: &str) -> Reasoned {
        split_seeded_reasoning(raw)
    }

    fn render(&self, turns: &[Turn]) -> String {
        let mut s = String::from(DS_BOS);
        // System text is hoisted to the front, raw (no wrapper).
        for turn in turns {
            if let Turn::System(content) = turn {
                s.push_str(content);
            }
        }
        for turn in turns {
            match turn {
                Turn::System(_) => {}
                Turn::User(content) => {
                    s.push_str(DS_USER);
                    s.push_str(content);
                }
                Turn::Assistant(content) => {
                    s.push_str(DS_ASSISTANT);
                    s.push_str(content);
                    s.push_str(DS_EOS);
                }
                Turn::AssistantToolCall { name, arguments } => {
                    s.push_str(DS_ASSISTANT);
                    s.push_str(&generic_tool_call(name, arguments));
                    s.push_str(DS_EOS);
                }
                Turn::ToolResult {
                    name,
                    content,
                    is_error,
                } => {
                    s.push_str(DS_TOOL_OUT_BEGIN);
                    s.push_str(&render_tool_result(name, content, *is_error));
                    s.push_str(DS_TOOL_OUT_END);
                }
            }
        }
        s.push_str(DS_ASSISTANT);
        s.push_str("<think>\n");
        s
    }

    fn classifier(&self) -> ResponseClassifier {
        ResponseClassifier::seeded_markers()
    }
}

/// Muse Glimmer's ATEM format: `<|start|>role<|message|>content<|eot|>` turns
/// and a bare `<|start|>assistant` cue. The addressed-output protocol uses
/// `to=self` for reasoning, `to=user` for answers, `to=<tool>` for structured
/// invocations, and `<|eom|>` to join messages. [`AtemInterpreter`] classifies
/// replies for live display, final transcript commit, and tool extraction; the
/// renderer speaks the other direction from structured [`Turn`] variants.
///
/// Byte-checked against the `/apply-template` oracle renders captured from
/// the official GGUF's embedded template (`tests/fixtures/llama_server/`,
/// template sha256 in provenance.json). Two behaviors mirror that template
/// exactly:
///
/// - A system turn is normalized ("Reasoning effort" → "Reasoning strength",
///   four casings) and, unless it already states one, a
///   `Reasoning strength: {level}.` line is appended; every system block ends
///   with the advertised recipient list (chat subset: `"self", "user"`).
/// - With no system turn, the template's default system block is emitted
///   (assistant greeting + knowledge cutoff). `current_date` is optional and
///   omitted when `None` so renders stay deterministic for tests — the
///   upstream template injects the wall-clock date here, which oracle
///   comparisons must inject explicitly.
///
/// Emits no BOS (the oracle renders carry none). A prior assistant answer is
/// answer-only (REASON-1) and renders as `to=user` with terminal `<|eot|>`.
/// Tool calls and results retain their names and arguments as transcript data;
/// rendering never recovers protocol meaning by parsing display text.
#[derive(Default)]
pub struct MuseGlimmerTemplate {
    /// Rendered as `Reasoning strength: {level}.`; the upstream template's
    /// default is `high` (the [`ReasoningStrength`] default).
    pub reasoning_strength: ReasoningStrength,
    /// `Current date: {date}.` in the default system block; `None` omits the
    /// line.
    pub current_date: Option<String>,
}

/// Muse Glimmer's reasoning-strength directive — the four levels the model is
/// trained on, as a sum type so an invalid strength is unrepresentable rather
/// than a string caught (or not) at render time.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ReasoningStrength {
    Low,
    Medium,
    #[default]
    High,
    XHigh,
}

impl ReasoningStrength {
    /// The directive spelling the upstream template uses.
    pub fn as_str(self) -> &'static str {
        match self {
            ReasoningStrength::Low => "low",
            ReasoningStrength::Medium => "medium",
            ReasoningStrength::High => "high",
            ReasoningStrength::XHigh => "xhigh",
        }
    }
}

const MUSE_KNOWLEDGE_CUTOFF: &str = "2026-01-04";
const MUSE_RECIPIENTS: &str = "# Valid recipients: \"self\", \"user\".";
const MUSE_RECIPIENTS_PREFIX: &str = "# Valid recipients:";

impl MuseGlimmerTemplate {
    fn reasoning_line(&self) -> String {
        format!("Reasoning strength: {}.", self.reasoning_strength.as_str())
    }

    /// The block the upstream template emits when the caller supplies no
    /// system message.
    fn default_system(&self, s: &mut String) {
        s.push_str(MUSE_START);
        s.push_str("system");
        s.push_str(MUSE_MESSAGE);
        s.push_str("You are a helpful AI assistant.");
        s.push_str(&format!("\nKnowledge cutoff: {MUSE_KNOWLEDGE_CUTOFF}."));
        if let Some(date) = &self.current_date {
            s.push_str(&format!("\nCurrent date: {date}."));
        }
        s.push_str("\n\n");
        s.push_str(&self.reasoning_line());
        s.push_str("\n\n");
        s.push_str(MUSE_RECIPIENTS);
        s.push_str(MUSE_EOT);
    }
}

/// The upstream template's normalization of caller-written directives (jinja
/// has no case-insensitive replace, hence the four realistic casings).
fn muse_normalize_effort(text: &str) -> String {
    text.replace("Reasoning effort", "Reasoning strength")
        .replace("Reasoning Effort", "Reasoning Strength")
        .replace("reasoning effort", "reasoning strength")
        .replace("REASONING EFFORT", "REASONING STRENGTH")
}

impl PromptTemplate for MuseGlimmerTemplate {
    fn render(&self, turns: &[Turn]) -> String {
        let mut s = String::new();
        if !turns.iter().any(|turn| turn.role() == Role::System) {
            self.default_system(&mut s);
        }
        for turn in turns {
            match turn {
                Turn::System(content) => {
                    let sys = muse_normalize_effort(content);
                    s.push_str(MUSE_START);
                    s.push_str("system");
                    s.push_str(MUSE_MESSAGE);
                    s.push_str(&sys);
                    if !sys.to_lowercase().contains("reasoning strength") {
                        s.push_str("\n\n");
                        s.push_str(&self.reasoning_line());
                    }
                    if !sys
                        .lines()
                        .any(|line| line.trim_start().starts_with(MUSE_RECIPIENTS_PREFIX))
                    {
                        s.push_str("\n\n");
                        s.push_str(MUSE_RECIPIENTS);
                    }
                    s.push_str(MUSE_EOT);
                }
                Turn::User(content) => {
                    s.push_str(MUSE_START);
                    s.push_str("user");
                    s.push_str(MUSE_MESSAGE);
                    s.push_str(content);
                    s.push_str(MUSE_EOT);
                }
                Turn::Assistant(content) => {
                    s.push_str(MUSE_START);
                    s.push_str("assistant to=user");
                    s.push_str(MUSE_MESSAGE);
                    s.push_str(content);
                    s.push_str(MUSE_EOT);
                }
                Turn::AssistantToolCall { name, arguments } => {
                    s.push_str(MUSE_START);
                    s.push_str("assistant to=");
                    s.push_str(name);
                    s.push_str(MUSE_MESSAGE);
                    render_muse_call(&mut s, name, arguments);
                    s.push_str(MUSE_EOT);
                }
                Turn::ToolResult {
                    name,
                    content,
                    is_error: _,
                } => {
                    s.push_str(MUSE_START);
                    s.push_str("tool ");
                    s.push_str(name);
                    s.push_str(MUSE_MESSAGE);
                    s.push_str("<tool_output name=\"");
                    s.push_str(name);
                    s.push_str("\">\n");
                    s.push_str(content);
                    s.push_str("\n</tool_output>");
                    s.push_str(MUSE_EOT);
                }
            }
        }
        s.push_str(MUSE_START);
        s.push_str("assistant");
        s
    }

    fn compose_system(&self, system: &str, tool_instructions: &str) -> String {
        if tool_instructions.is_empty() {
            return system.to_string();
        }
        let mut system = muse_normalize_effort(system);
        if !system.to_lowercase().contains("reasoning strength") {
            system.push_str("\n\n");
            system.push_str(&self.reasoning_line());
        }
        system.push_str("\n\n");
        system.push_str(tool_instructions);
        system
    }

    fn classifier(&self) -> ResponseClassifier {
        ResponseClassifier::atem()
    }

    fn interpret_response(&self, raw: &str) -> Reasoned {
        AtemInterpreter::interpret(raw)
    }
}

fn render_muse_call(s: &mut String, name: &str, arguments: &ToolArguments) {
    s.push_str("<atem:function_calls>\n<atem:invoke name=\"");
    s.push_str(name);
    s.push_str("\">\n");
    for (parameter, value) in arguments.iter() {
        s.push_str("<atem:parameter name=\"");
        s.push_str(parameter);
        s.push_str("\">");
        match value {
            serde_json::Value::String(value) => s.push_str(value),
            other => s.push_str(&render_json_inline(other)),
        }
        s.push_str("</atem:parameter>\n");
    }
    s.push_str("</atem:invoke>\n</atem:function_calls>");
}

#[cfg(test)]
mod tests {
    use super::*;

    fn turn(role: Role, content: &str) -> Turn {
        match role {
            Role::System => Turn::system(content),
            Role::User => Turn::user(content),
            Role::Assistant => Turn::assistant(content),
            Role::Tool => panic!("tool turns require a name and outcome"),
        }
    }

    #[test]
    fn chatml_wraps_roles_and_tool_responses() {
        let s = ChatMlTemplate.render(&[
            turn(Role::System, "SYS"),
            turn(Role::User, "hi"),
            Turn::assistant_tool_call(
                "read_file",
                ToolArguments::try_from_pairs([(
                    "path".to_string(),
                    serde_json::Value::String("note.txt".to_string()),
                )])
                .unwrap(),
            ),
            Turn::tool_result("read_file", "X", false),
        ]);
        assert_eq!(
            s,
            "<|im_start|>system\nSYS<|im_end|>\n\
             <|im_start|>user\nhi<|im_end|>\n\
             <|im_start|>assistant\n<tool_call>\n{\"name\":\"read_file\",\"arguments\":{\"path\":\"note.txt\"}}\n</tool_call><|im_end|>\n\
             <|im_start|>user\n<tool_response>\n[read_file ok] X\n</tool_response><|im_end|>\n\
             <|im_start|>assistant\n"
        );
    }

    #[test]
    fn gemma_folds_system_and_cues_model_without_bos() {
        // upholds: TMPL-1, TMPL-2
        let s = GemmaTemplate.render(&[turn(Role::System, "Be brief."), turn(Role::User, "hi")]);
        assert_eq!(
            s,
            "<start_of_turn>user\nBe brief.\n\nhi<end_of_turn>\n<start_of_turn>model\n"
        );
        assert!(
            !s.contains("<bos>"),
            "Gemma template must not emit a literal <bos>"
        );
    }

    #[test]
    fn gemma_plain_user_turn() {
        let s = GemmaTemplate.render(&[turn(Role::User, "explain rust")]);
        assert_eq!(
            s,
            "<start_of_turn>user\nexplain rust<end_of_turn>\n<start_of_turn>model\n"
        );
    }

    #[test]
    fn mistral_folds_system_into_inst_without_bos() {
        // upholds: TMPL-1, TMPL-2
        let s = MistralTemplate.render(&[turn(Role::System, "Be brief."), turn(Role::User, "hi")]);
        assert_eq!(s, "[INST] Be brief.\n\nhi[/INST]");
        assert!(
            !s.contains("<s>"),
            "Mistral template must not emit a literal <s>"
        );
    }

    #[test]
    fn mistral_plain_user_turn() {
        let s = MistralTemplate.render(&[turn(Role::User, "explain rust")]);
        assert_eq!(s, "[INST] explain rust[/INST]");
    }

    #[test]
    fn glm_prefixes_gmask_and_keeps_system_role() {
        // upholds: TMPL-1 — the [gMASK]<sop> prefix is emitted exactly once (GLM's
        // tokenizer doesn't add it). GLM has a real system role (no folding).
        let s = GlmTemplate.render(&[turn(Role::System, "Be brief."), turn(Role::User, "hi")]);
        assert_eq!(
            s,
            "[gMASK]<sop><|system|>\nBe brief.<|user|>\nhi<|assistant|>\n"
        );
        assert_eq!(s.matches("[gMASK]").count(), 1, "exactly one gMASK prefix");
    }

    #[test]
    fn deepseek_emits_bos_keeps_system_and_seeds_think() {
        // upholds: TMPL-1 — the BOS is emitted exactly once (DeepSeek tokenizers
        // don't add it). System has a real role (hoisted, not folded). The cue
        // pre-seeds <think> so the model's output carries only the close marker.
        let s = DeepSeekTemplate.render(&[turn(Role::System, "Be brief."), turn(Role::User, "hi")]);
        assert_eq!(
            s,
            "<\u{ff5c}begin\u{2581}of\u{2581}sentence\u{ff5c}>Be brief.\
             <\u{ff5c}User\u{ff5c}>hi<\u{ff5c}Assistant\u{ff5c}><think>\n"
        );
        assert_eq!(
            s.matches("<\u{ff5c}begin\u{2581}of\u{2581}sentence\u{ff5c}>")
                .count(),
            1,
            "exactly one BOS"
        );
        assert!(s.ends_with("<think>\n"), "cue pre-seeds the think block");
    }

    #[test]
    fn deepseek_history_uses_answer_only_assistant_turns() {
        // upholds: REASON-1 — a prior assistant turn (already answer-only) renders
        // wrapped in the EOS; combined with the seeded <think> cue, a reasoning
        // model's output is a clean reasoning…</think>answer the split recovers.
        let s = DeepSeekTemplate.render(&[
            turn(Role::User, "2+2?"),
            turn(Role::Assistant, "4"),
            turn(Role::User, "x3?"),
        ]);
        assert!(s.contains(
            "<\u{ff5c}Assistant\u{ff5c}>4<\u{ff5c}end\u{2581}of\u{2581}sentence\u{ff5c}>"
        ));
        assert!(s.ends_with("<\u{ff5c}Assistant\u{ff5c}><think>\n"));
    }

    /// The `/apply-template` oracle renders captured in stage 0 (see
    /// tests/fixtures/llama_server/provenance.json for the template digest
    /// and capture provenance).
    const ORACLE1: &str = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/llama_server/oracle1-prompt.txt"
    ));
    const ORACLE2: &str = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/llama_server/oracle2-prompt.txt"
    ));
    const ORACLE3: &str = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/llama_server/oracle3-prompt.txt"
    ));

    #[test]
    fn muse_matches_the_oracle_render_byte_for_byte() {
        // The renderer is checked against the official embedded template via
        // the stage-0 /apply-template capture, not against our own reading of
        // the jinja. oracle1: explicit system + user, default strength.
        let s = MuseGlimmerTemplate::default().render(&[
            turn(Role::System, "You are a helpful assistant."),
            turn(Role::User, "What is a river?"),
        ]);
        assert_eq!(s, ORACLE1);
    }

    #[test]
    fn muse_matches_oracle2_minus_the_reasoning_replay() {
        // oracle2 exercises effort→strength normalization and multi-turn
        // history. The upstream render replays prior reasoning as a
        // `to=self …<|eom|>` message; yatima's transcript is answer-only by
        // law (REASON-1), so the expectation is the oracle with that one
        // segment excised — a deliberate subset, not an approximation.
        let replay =
            "<|start|>assistant to=self<|message|>User wants any river. Nile is canonical.<|eom|>";
        let expected = ORACLE2.replacen(replay, "", 1);
        assert_ne!(expected, ORACLE2, "fixture contains the replay segment");
        let s = MuseGlimmerTemplate::default().render(&[
            turn(Role::System, "You are terse.\n\nReasoning effort: low."),
            turn(Role::User, "Name a river."),
            turn(Role::Assistant, "The Nile."),
            turn(Role::User, "Another?"),
        ]);
        assert_eq!(s, expected);
    }

    #[test]
    fn muse_renders_the_complete_oracle3_tool_round() {
        // upholds: AGENT-3, REASON-1 — the working transcript preserves the
        // structured invocation and result required by the next prompt. The
        // official template's complete tool round is the byte-level oracle.
        use crate::{MuseAtemCodec, ToolCallCodec, ToolSpec};

        let template = MuseGlimmerTemplate {
            reasoning_strength: ReasoningStrength::High,
            current_date: Some("2026-08-21".to_string()),
        };
        let codec = MuseAtemCodec;
        let spec = ToolSpec {
            name: "fs.stat_file".to_string(),
            description: "Report file metadata.".to_string(),
            params: serde_json::json!({
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "follow_symlinks": {"type": "boolean"}
                },
                "required": ["path"]
            }),
        };
        let system = template.compose_system(
            "You are a helpful AI assistant.\nKnowledge cutoff: 2026-01-04.\nCurrent date: 2026-08-21.",
            &codec.render_system(&[spec]),
        );
        let arguments = ToolArguments::try_from_pairs([
            ("path".to_string(), serde_json::json!("README.md")),
            ("follow_symlinks".to_string(), serde_json::json!(true)),
        ])
        .unwrap();
        let rendered = template.render(&[
            Turn::system(system),
            Turn::user("How large is README.md?"),
            Turn::assistant_tool_call("fs.stat_file", arguments),
            Turn::tool_result("fs.stat_file", "{\"bytes\": 4096}", false),
        ]);

        assert_eq!(rendered, ORACLE3);
    }

    #[test]
    fn muse_default_system_is_deterministic_without_a_date() {
        // With no system turn the default block is emitted; current_date None
        // omits the date line entirely (the upstream template would inject
        // wall-clock time — the determinism knob oracle comparisons must set).
        let s = MuseGlimmerTemplate::default().render(&[turn(Role::User, "hi")]);
        assert!(s.starts_with(
            "<|start|>system<|message|>You are a helpful AI assistant.\n\
             Knowledge cutoff: 2026-01-04.\n\nReasoning strength: high."
        ));
        assert!(!s.contains("Current date:"));
        assert!(s.ends_with("<|start|>assistant"));
    }

    #[test]
    fn muse_interprets_addressed_output() {
        // One definition of the protocol: the same interpretation feeds the
        // caller-visible answer and the committed turn. gen1's observed shape:
        // reasoning continues the bare cue, the answer opens a fresh header.
        let raw = " to=self<|message|>User wants a river. Nile is canonical.\
                   <|start|>assistant to=user<|message|>The Nile.";
        let r = AtemInterpreter::interpret(raw);
        assert_eq!(
            r.reasoning.as_deref(),
            Some("User wants a river. Nile is canonical.")
        );
        assert_eq!(r.answer, "The Nile.");
    }

    #[test]
    fn muse_interpretation_consumes_framing_markers() {
        // <|eom|> joins in-turn messages; a trailing <|eot|> may survive when
        // it arrives as a caller stop rather than EOS. Neither reaches text.
        let raw = " to=self<|message|>thinking<|eom|>\
                   <|start|>assistant to=user<|message|>Answer.<|eot|>";
        let r = AtemInterpreter::interpret(raw);
        assert_eq!(r.reasoning.as_deref(), Some("thinking"));
        assert_eq!(r.answer, "Answer.");
    }

    #[test]
    fn muse_unaddressed_output_is_the_answer() {
        // A model that ignored the protocol loses nothing: no <|message|>
        // marker means the whole text is the answer.
        let r = AtemInterpreter::interpret("Just a plain reply.");
        assert_eq!(r.reasoning, None);
        assert_eq!(r.answer, "Just a plain reply.");
    }

    #[test]
    fn muse_tool_recipients_reach_neither_reasoning_nor_answer() {
        // upholds: REASON-1 — an addressed tool invocation is retained on the
        // machine's tool channel and never leaks into either prose bucket.
        let raw = " to=self<|message|>plan<|start|>assistant \
                   to=fs.stat_file<|message|><atem:function_calls>…</atem:function_calls>\
                   <|start|>assistant to=user<|message|>Done.";
        let r = AtemInterpreter::interpret(raw);
        assert_eq!(r.answer, "Done.");
        assert_eq!(r.reasoning.as_deref(), Some("plan"));
    }

    #[test]
    fn seeded_templates_interpret_truncation_as_reasoning() {
        // upholds: REASON-1 — a pre-seeded cue means a close-less reply never
        // left the think block; the final interpreter must agree with
        // ReasoningSplitter::seeded rather than surface the span as answer.
        for t in [
            &ChatMlThinkTemplate as &dyn PromptTemplate,
            &DeepSeekTemplate,
        ] {
            let r = t.interpret_response("truncated reasoning only");
            assert_eq!(r.answer, "");
            assert_eq!(r.reasoning.as_deref(), Some("truncated reasoning only"));
        }
    }

    #[test]
    fn boxed_template_keeps_its_interpretation() {
        // upholds: REASON-1 — the CLI carries templates as Box<dyn
        // PromptTemplate>; the Box impl must forward interpret_response, or a
        // default would silently strip the Muse override and recommit raw
        // ATEM.
        let b: Box<dyn PromptTemplate> = Box::new(MuseGlimmerTemplate::default());
        let r = b.interpret_response(" to=self<|message|>t<|start|>assistant to=user<|message|>a");
        assert_eq!(r.answer, "a");
        assert_eq!(r.reasoning.as_deref(), Some("t"));
    }

    #[test]
    fn muse_strength_levels_render_their_directive() {
        // The sum type makes an invalid strength unrepresentable; each level
        // renders the upstream template's spelling.
        for (level, spelled) in [
            (ReasoningStrength::Low, "low"),
            (ReasoningStrength::Medium, "medium"),
            (ReasoningStrength::High, "high"),
            (ReasoningStrength::XHigh, "xhigh"),
        ] {
            let t = MuseGlimmerTemplate {
                reasoning_strength: level,
                current_date: None,
            };
            let s = t.render(&[turn(Role::User, "hi")]);
            assert!(s.contains(&format!("Reasoning strength: {spelled}.")));
        }
    }

    #[test]
    fn muse_respects_a_caller_stated_strength() {
        // A system prompt that already states a strength (after
        // normalization) is not given a second directive line.
        let s = MuseGlimmerTemplate::default().render(&[
            turn(Role::System, "Reasoning effort: low."),
            turn(Role::User, "hi"),
        ]);
        assert!(s.contains("Reasoning strength: low."));
        assert!(!s.contains("Reasoning strength: high."));
    }

    #[test]
    fn templates_render_multi_turn_history_with_cue() {
        // upholds: TMPL-2 — a mid-conversation transcript carries prior turns so
        // the model has memory, and ends with the generation cue. This is what
        // makes the chat REPL remember (history lives in the prompt).
        let convo = [
            turn(Role::User, "My name is Ada."),
            turn(Role::Assistant, "Nice to meet you, Ada."),
            turn(Role::User, "What is my name?"),
        ];

        let qwen = ChatMlTemplate.render(&convo);
        assert!(qwen.contains("My name is Ada."), "history present (qwen)");
        assert!(
            qwen.contains("Nice to meet you, Ada."),
            "prior answer present"
        );
        assert!(
            qwen.ends_with("<|im_start|>assistant\n"),
            "ends with the cue"
        );

        let gemma = GemmaTemplate.render(&convo);
        assert!(gemma.contains("My name is Ada."), "history present (gemma)");
        assert!(
            gemma.contains("<start_of_turn>model\nNice to meet you, Ada.<end_of_turn>"),
            "prior assistant turn rendered as model"
        );
        assert!(
            gemma.ends_with("<start_of_turn>model\n"),
            "ends with the cue"
        );
    }
}
