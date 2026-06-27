//! Prompt templates — rendering a transcript into a model's *native* prompt
//! string.
//!
//! A model is acutely sensitive to its trained chat format: feed it a generic
//! `<|role|>` layout and it can destabilise (degenerate repetition, no
//! instruction-following). [`PromptTemplate`] is the boundary that makes the
//! format per-model; [`ChatMlTemplate`] matches Qwen2.5's trained format, and
//! [`PlainTemplate`] keeps the minimal layout for models with no known template
//! and for tests.

use crate::transcript::{Role, Turn};

/// Render a transcript into the prompt string fed to the model, ending with the
/// cue that makes the model speak next.
pub trait PromptTemplate {
    fn render(&self, turns: &[Turn]) -> String;
}

/// A boxed template is a template — lets a runtime-chosen `Box<dyn
/// PromptTemplate>` (e.g. the CLI's `--format`) satisfy generic bounds like
/// `ChatSession<_, T: PromptTemplate>`.
impl<T: PromptTemplate + ?Sized> PromptTemplate for Box<T> {
    fn render(&self, turns: &[Turn]) -> String {
        (**self).render(turns)
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
            match turn.role {
                Role::System => pending_system = Some(turn.content.clone()),
                Role::Assistant => gemma_turn(&mut s, "model", &turn.content),
                Role::User | Role::Tool => {
                    let content = match pending_system.take() {
                        Some(sys) => format!("{sys}\n\n{}", turn.content),
                        None => turn.content.clone(),
                    };
                    gemma_turn(&mut s, "user", &content);
                }
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
            match turn.role {
                Role::System => pending_system = Some(turn.content.clone()),
                Role::Assistant => {
                    s.push(' ');
                    s.push_str(&turn.content);
                    s.push_str("</s>");
                }
                Role::User | Role::Tool => {
                    let content = match pending_system.take() {
                        Some(sys) => format!("{sys}\n\n{}", turn.content),
                        None => turn.content.clone(),
                    };
                    s.push_str("[INST] ");
                    s.push_str(&content);
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
            let role = match turn.role {
                Role::System => "system",
                Role::User => "user",
                Role::Assistant => "assistant",
                Role::Tool => "observation",
            };
            s.push_str("<|");
            s.push_str(role);
            s.push_str("|>\n");
            s.push_str(&turn.content);
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
    fn render(&self, turns: &[Turn]) -> String {
        let mut s = String::from(DS_BOS);
        // System text is hoisted to the front, raw (no wrapper).
        for turn in turns {
            if turn.role == Role::System {
                s.push_str(&turn.content);
            }
        }
        for turn in turns {
            match turn.role {
                Role::System => {}
                Role::User => {
                    s.push_str(DS_USER);
                    s.push_str(&turn.content);
                }
                Role::Assistant => {
                    s.push_str(DS_ASSISTANT);
                    s.push_str(&turn.content);
                    s.push_str(DS_EOS);
                }
                Role::Tool => {
                    s.push_str(DS_TOOL_OUT_BEGIN);
                    s.push_str(&turn.content);
                    s.push_str(DS_TOOL_OUT_END);
                }
            }
        }
        s.push_str(DS_ASSISTANT);
        s.push_str("<think>\n");
        s
    }
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
