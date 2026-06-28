//! Chat-format selection: the `&str`/`Arch` ⇄ prompt-template mapping a host
//! program needs. Dependency-light (no `clap`) — a `ChatFormat` parses from a
//! name via [`FromStr`] and lists its spellings in [`ChatFormat::NAMES`], so a
//! clap front end can wrap it with a `PossibleValuesParser` without the library
//! taking a CLI dependency.

use crate::{
    Arch, ChatMlTemplate, ChatMlThinkTemplate, DeepSeekTemplate, GemmaTemplate, GlmTemplate,
    MistralTemplate, PlainTemplate, PromptTemplate,
};
use serde::{Deserialize, Serialize};
use std::str::FromStr;

/// Which model-native chat format to speak. `Qwen`/`Plain` also carry a
/// tool-call codec (so they work with the agent loop); `Gemma`/`Mistral`/`Glm`
/// are chat-only.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ChatFormat {
    /// Qwen2.5-Instruct: ChatML (+ `<tool_call>` tools).
    Qwen,
    /// Gemma-2-it: `<start_of_turn>` (chat only).
    Gemma,
    /// Mistral-v0.3: `[INST] … [/INST]` (chat only).
    Mistral,
    /// GLM-4: `[gMASK]<sop><|role|>` (chat only).
    Glm,
    /// DeepSeek-V2/V3 and R1 distills: `<｜begin▁of▁sentence｜>…<｜User｜>…
    /// <｜Assistant｜><think>` (chat only; reasoning model).
    DeepSeek,
    /// ChatML with a pre-seeded `<think>` cue — reasoning Qwen models (QwQ-32B,
    /// Qwen3 thinking). Same turns as `Qwen`, but the assistant cue is
    /// `<|im_start|>assistant\n<think>\n`, so the model emits only the closing
    /// `</think>` (chat only; reasoning model).
    QwenThink,
    /// Minimal `<|role|>` layout + `<tool_call>{json}</tool_call>` (fallback).
    Plain,
}

impl ChatFormat {
    /// Every format's name, in declaration order — the accepted spellings for
    /// [`FromStr`] and a ready-made list for a CLI `PossibleValuesParser`.
    pub const NAMES: [&'static str; 7] = [
        "qwen",
        "gemma",
        "mistral",
        "glm",
        "deepseek",
        "qwen-think",
        "plain",
    ];

    /// The lowercase name (the inverse of [`FromStr`]).
    pub fn name(self) -> &'static str {
        match self {
            ChatFormat::Qwen => "qwen",
            ChatFormat::Gemma => "gemma",
            ChatFormat::Mistral => "mistral",
            ChatFormat::Glm => "glm",
            ChatFormat::DeepSeek => "deepseek",
            ChatFormat::QwenThink => "qwen-think",
            ChatFormat::Plain => "plain",
        }
    }

    /// The prompt template for this format (used by `chat` and the examples).
    pub fn template(self) -> Box<dyn PromptTemplate> {
        match self {
            ChatFormat::Qwen => Box::new(ChatMlTemplate),
            ChatFormat::Gemma => Box::new(GemmaTemplate),
            ChatFormat::Mistral => Box::new(MistralTemplate),
            ChatFormat::Glm => Box::new(GlmTemplate),
            ChatFormat::DeepSeek => Box::new(DeepSeekTemplate),
            ChatFormat::QwenThink => Box::new(ChatMlThinkTemplate),
            ChatFormat::Plain => Box::new(PlainTemplate),
        }
    }

    /// Whether this format carries a tool-call codec, i.e. is usable by the
    /// agent loop. Only `Qwen` and `Plain` do; the rest are chat-only. The agent
    /// gates on this (CAPS-1).
    pub fn supports_tools(self) -> bool {
        matches!(self, ChatFormat::Qwen | ChatFormat::Plain)
    }

    /// Whether this format's generation cue pre-seeds the reasoning opener (so a
    /// model's output begins *inside* the think block and carries only the close
    /// marker). `DeepSeek` and `QwenThink` do. A streaming host uses this to pick
    /// [`ReasoningSplitter::seeded`](crate::ReasoningSplitter::seeded) over
    /// [`new`](crate::ReasoningSplitter::new). It MUST agree with whether the
    /// format's template seeds `<think>` — see the `pre_seeds_matches_template`
    /// test (the consistency that prevents the QwQ-style mis-classification).
    pub fn pre_seeds_reasoning(self) -> bool {
        matches!(self, ChatFormat::DeepSeek | ChatFormat::QwenThink)
    }

    /// The format a model of this architecture speaks natively — the default a
    /// host uses when the user doesn't pass one (HOST-1). Delegates to the
    /// capability table [`caps_for`].
    pub fn default_for(arch: Arch) -> ChatFormat {
        caps_for(arch).default_format
    }
}

impl FromStr for ChatFormat {
    type Err = anyhow::Error;
    fn from_str(s: &str) -> anyhow::Result<ChatFormat> {
        match s {
            "qwen" => Ok(ChatFormat::Qwen),
            "gemma" => Ok(ChatFormat::Gemma),
            "mistral" => Ok(ChatFormat::Mistral),
            "glm" => Ok(ChatFormat::Glm),
            "deepseek" => Ok(ChatFormat::DeepSeek),
            "qwen-think" => Ok(ChatFormat::QwenThink),
            "plain" => Ok(ChatFormat::Plain),
            other => anyhow::bail!(
                "unknown chat format {other:?}; expected one of {:?}",
                Self::NAMES
            ),
        }
    }
}

impl std::fmt::Display for ChatFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}

/// The host-side capability table: what a host program needs to know about a
/// model from its architecture alone — distinct from engine-native runtime
/// policy ([`Arch::metal_prefill_chunk`]), which names no format type. This is
/// the single source of truth for format/role capabilities; the README matrix
/// and CLI/examples read it rather than re-encoding it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Caps {
    /// The chat format this architecture speaks natively.
    pub default_format: ChatFormat,
    /// Whether the architecture has a distinct system role (vs folding system
    /// text into the first user turn — Gemma/Mistral; see TMPL-2).
    pub supports_system_role: bool,
}

/// The [`Caps`] for an architecture.
pub fn caps_for(arch: Arch) -> Caps {
    match arch {
        // Qwen2.5 and Qwen3 share the ChatML format.
        Arch::Qwen2 | Arch::Qwen3 | Arch::Qwen3Moe => Caps {
            default_format: ChatFormat::Qwen,
            supports_system_role: true,
        },
        Arch::Glm4 => Caps {
            default_format: ChatFormat::Glm,
            supports_system_role: true,
        },
        // Gemma-2 and Gemma-3 share the `<start_of_turn>` format (no system role).
        Arch::Gemma2 | Arch::Gemma3 => Caps {
            default_format: ChatFormat::Gemma,
            supports_system_role: false,
        },
        Arch::Mistral => Caps {
            default_format: ChatFormat::Mistral,
            supports_system_role: false,
        },
        // DeepSeek-V2/V3 (and R1) speak DeepSeek's native format. Note the R1
        // *distills* are Qwen2/Llama arch, so a host selects `deepseek` for them
        // explicitly (via profile or --format); the arch default below covers
        // the genuine DeepSeek-arch models.
        Arch::DeepSeek2 => Caps {
            default_format: ChatFormat::DeepSeek,
            supports_system_role: true,
        },
        // No native instruct chat template wired up — fall back to Plain.
        Arch::Llama | Arch::Phi3 | Arch::Starcoder2 => Caps {
            default_format: ChatFormat::Plain,
            supports_system_role: true,
        },
    }
}

/// An explicit `--format` that contradicts the model's architecture default —
/// the basis for a host warning (HOST-2). Not an error: the user's choice is
/// honored, but a mismatch is usually a mistake (a mis-rendered prompt → quiet
/// garbage), so it is surfaced.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FormatMismatch {
    pub supplied: ChatFormat,
    pub arch: Arch,
    pub expected: ChatFormat,
}

impl std::fmt::Display for FormatMismatch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "--format {} but the model is {:?} (expected --format {}); \
             the prompt may be mis-rendered",
            self.supplied, self.arch, self.expected
        )
    }
}

/// Resolve the chat format for a loaded model: an explicit choice wins,
/// otherwise the architecture default (HOST-1). When an explicit choice differs
/// from the architecture default, also return a [`FormatMismatch`] so the caller
/// can warn (HOST-2). Pure, so both invariants are unit-testable without a GPU.
pub fn resolve_format(
    arch: Arch,
    explicit: Option<ChatFormat>,
) -> (ChatFormat, Option<FormatMismatch>) {
    let expected = ChatFormat::default_for(arch);
    match explicit {
        None => (expected, None),
        Some(f) if f == expected => (f, None),
        Some(supplied) => (
            supplied,
            Some(FormatMismatch {
                supplied,
                arch,
                expected,
            }),
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Role, Turn};

    #[test]
    fn chat_format_round_trips_through_name() {
        for name in ChatFormat::NAMES {
            let fmt: ChatFormat = name.parse().unwrap();
            assert_eq!(fmt.name(), name);
        }
    }

    #[test]
    fn chat_format_rejects_unknown() {
        assert!("llama".parse::<ChatFormat>().is_err());
    }

    #[test]
    fn chat_format_serde_is_lowercase_name() {
        // serde-ready (config files later) round-trips through the same names.
        let json = serde_json::to_string(&ChatFormat::Glm).unwrap();
        assert_eq!(json, "\"glm\"");
        assert_eq!(
            serde_json::from_str::<ChatFormat>("\"glm\"").unwrap(),
            ChatFormat::Glm
        );
    }

    #[test]
    fn chat_format_template_matches_format() {
        let rendered = ChatFormat::Mistral.template().render(&[Turn {
            role: Role::User,
            content: "x".into(),
        }]);
        assert!(rendered.contains("[INST]"));
    }

    #[test]
    fn only_qwen_and_plain_support_tools() {
        // upholds: CAPS-1 — chat-only formats carry no tool codec.
        assert!(ChatFormat::Qwen.supports_tools());
        assert!(ChatFormat::Plain.supports_tools());
        for fmt in [
            ChatFormat::Gemma,
            ChatFormat::Mistral,
            ChatFormat::Glm,
            ChatFormat::DeepSeek,
            ChatFormat::QwenThink,
        ] {
            assert!(!fmt.supports_tools());
        }
    }

    #[test]
    fn pre_seeds_matches_template() {
        // The guard that prevents the QwQ-style mis-classification: a format's
        // `pre_seeds_reasoning()` must agree with whether its template actually
        // seeds `<think>` in the cue. If they disagree, the streaming splitter is
        // selected wrong (seeded vs new) and reasoning is mis-classified. Pure —
        // no model needed; runs every commit, for every format.
        for name in ChatFormat::NAMES {
            let fmt: ChatFormat = name.parse().unwrap();
            let rendered = fmt.template().render(&[Turn {
                role: Role::User,
                content: "hi".into(),
            }]);
            let template_seeds_think = rendered.trim_end().ends_with("<think>");
            assert_eq!(
                template_seeds_think,
                fmt.pre_seeds_reasoning(),
                "format {name}: template-seeds-<think> ({template_seeds_think}) must \
                 equal pre_seeds_reasoning() ({})",
                fmt.pre_seeds_reasoning()
            );
        }
    }

    #[test]
    fn default_format_is_inferred_per_arch() {
        // upholds: HOST-1 — every arch maps to a native default format.
        assert_eq!(ChatFormat::default_for(Arch::Qwen2), ChatFormat::Qwen);
        assert_eq!(ChatFormat::default_for(Arch::Glm4), ChatFormat::Glm);
        assert_eq!(ChatFormat::default_for(Arch::Gemma2), ChatFormat::Gemma);
        assert_eq!(ChatFormat::default_for(Arch::Mistral), ChatFormat::Mistral);
        assert_eq!(ChatFormat::default_for(Arch::Llama), ChatFormat::Plain);
    }

    #[test]
    fn resolve_format_infers_and_flags_mismatch() {
        // upholds: HOST-1 — omitted format resolves to the arch default.
        assert_eq!(resolve_format(Arch::Glm4, None), (ChatFormat::Glm, None));
        // a matching explicit format is no mismatch.
        assert_eq!(
            resolve_format(Arch::Glm4, Some(ChatFormat::Glm)),
            (ChatFormat::Glm, None)
        );
        // upholds: HOST-2 — a contradicting explicit format is honored but flagged.
        let (chosen, mismatch) = resolve_format(Arch::Qwen2, Some(ChatFormat::Glm));
        assert_eq!(chosen, ChatFormat::Glm);
        let m = mismatch.expect("mismatch should be reported");
        assert_eq!(m.expected, ChatFormat::Qwen);
        assert_eq!(m.supplied, ChatFormat::Glm);
    }
}
