//! Prompt templates — rendering a transcript into a model's *native* prompt
//! string.
//!
//! A base model is acutely sensitive to its trained chat format: feed an R1
//! distill a generic `<|role|>` layout and it destabilises (degenerate
//! repetition, no instruction-following). [`PromptTemplate`] is the seam that
//! makes the format per-model; [`DeepSeekR1Template`] matches the tokenizer's
//! `chat_template` (BOS, `<｜User｜>`/`<｜Assistant｜>`, an opening `<think>`, and
//! native tool-output framing) so the model behaves and can emit native tool
//! calls. [`PlainTemplate`] keeps the minimal layout for models with no known
//! template and for tests.

use crate::agent::{Role, Turn};

/// Render a transcript into the prompt string fed to the model, ending with the
/// cue that makes the model speak next.
pub trait PromptTemplate {
    fn render(&self, turns: &[Turn]) -> String;
}

/// A minimal, backend-agnostic role layout. Not any model's trained format —
/// fine for scripted tests and as a fallback, but off-distribution for a real
/// instruction/reasoning model.
pub struct PlainTemplate;

impl PromptTemplate for PlainTemplate {
    fn render(&self, turns: &[Turn]) -> String {
        let mut s = String::new();
        for turn in turns {
            let tag = match turn.role {
                Role::System => "system",
                Role::User => "user",
                Role::Assistant => "assistant",
                Role::Tool => "tool",
            };
            s.push_str(&format!("<|{tag}|>\n{}\n", turn.content));
        }
        s.push_str("<|assistant|>\n");
        s
    }
}

const BOS: &str = "<｜begin▁of▁sentence｜>";
const USER: &str = "<｜User｜>";
const ASSISTANT: &str = "<｜Assistant｜>";
const THINK_OPEN: &str = "<think>\n";
const OUTPUTS_BEGIN: &str = "<｜tool▁outputs▁begin｜>";
const OUTPUTS_END: &str = "<｜tool▁outputs▁end｜>";
const OUTPUT_BEGIN: &str = "<｜tool▁output▁begin｜>";
const OUTPUT_END: &str = "<｜tool▁output▁end｜>";

/// The DeepSeek-R1(-Distill) native chat format, mirroring the model's trained
/// `chat_template`: a leading BOS, the system prompt prepended raw, then
/// `<｜User｜>` / `<｜Assistant｜>` turns and native `<｜tool▁output…｜>` framing for
/// tool results. The generation cue opens a forced `<think>` block on a fresh
/// assistant turn; after tool outputs the assistant turn continues, so no new
/// tag is added (matching the template).
pub struct DeepSeekR1Template;

impl PromptTemplate for DeepSeekR1Template {
    fn render(&self, turns: &[Turn]) -> String {
        let mut s = String::from(BOS);
        // The system prompt sits right after BOS (the template concatenates it
        // raw, with no role wrapper).
        for turn in turns.iter().filter(|t| t.role == Role::System) {
            s.push_str(&turn.content);
        }
        for turn in turns {
            match turn.role {
                Role::System => {}
                Role::User => {
                    s.push_str(USER);
                    s.push_str(&turn.content);
                }
                Role::Assistant => {
                    s.push_str(ASSISTANT);
                    s.push_str(&turn.content);
                }
                Role::Tool => {
                    s.push_str(OUTPUTS_BEGIN);
                    s.push_str(OUTPUT_BEGIN);
                    s.push_str(&turn.content);
                    s.push_str(OUTPUT_END);
                    s.push_str(OUTPUTS_END);
                }
            }
        }
        // Cue the next turn. After a tool output the assistant continues the turn
        // that made the call, so no `<｜Assistant｜>` is added.
        if turns.last().map(|t| t.role) != Some(Role::Tool) {
            s.push_str(ASSISTANT);
            s.push_str(THINK_OPEN);
        }
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
        let mut s = String::new();
        for turn in turns {
            match turn.role {
                Role::System => block(&mut s, "system", &turn.content),
                Role::User => block(&mut s, "user", &turn.content),
                Role::Assistant => block(&mut s, "assistant", &turn.content),
                Role::Tool => block(
                    &mut s,
                    "user",
                    &format!("<tool_response>\n{}\n</tool_response>", turn.content),
                ),
            }
        }
        s.push_str(IM_START);
        s.push_str("assistant\n");
        s
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    fn turn(role: Role, content: &str) -> Turn {
        Turn {
            role,
            content: content.to_string(),
        }
    }

    #[test]
    fn deepseek_renders_native_tokens_with_think_cue() {
        let s = DeepSeekR1Template.render(&[turn(Role::System, "SYS"), turn(Role::User, "hi")]);
        assert_eq!(
            s,
            "<｜begin▁of▁sentence｜>SYS<｜User｜>hi<｜Assistant｜><think>\n"
        );
    }

    #[test]
    fn deepseek_continues_after_tool_output_without_new_tag() {
        let s = DeepSeekR1Template.render(&[
            turn(Role::System, "S"),
            turn(Role::User, "u"),
            turn(Role::Assistant, "A<｜tool▁call▁end｜>"),
            turn(Role::Tool, "[read_file ok] X"),
        ]);
        assert_eq!(
            s,
            "<｜begin▁of▁sentence｜>S<｜User｜>u<｜Assistant｜>A<｜tool▁call▁end｜>\
             <｜tool▁outputs▁begin｜><｜tool▁output▁begin｜>[read_file ok] X\
             <｜tool▁output▁end｜><｜tool▁outputs▁end｜>"
        );
        assert!(
            !s.ends_with("<think>\n"),
            "no new assistant cue after a tool output"
        );
    }

    #[test]
    fn chatml_wraps_roles_and_tool_responses() {
        let s = ChatMlTemplate.render(&[
            turn(Role::System, "SYS"),
            turn(Role::User, "hi"),
            turn(Role::Assistant, "<tool_call>\n{}\n</tool_call>"),
            turn(Role::Tool, "[read_file ok] X"),
        ]);
        assert_eq!(
            s,
            "<|im_start|>system\nSYS<|im_end|>\n\
             <|im_start|>user\nhi<|im_end|>\n\
             <|im_start|>assistant\n<tool_call>\n{}\n</tool_call><|im_end|>\n\
             <|im_start|>user\n<tool_response>\n[read_file ok] X\n</tool_response><|im_end|>\n\
             <|im_start|>assistant\n"
        );
    }
}
