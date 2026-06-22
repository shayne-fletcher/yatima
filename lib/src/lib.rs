//! a Rust runtime for language-integrated LLMs — inference as an in-process
//! library function.
//!
//! # Invariant & law registry
//!
//! The canonical list of the contracts this crate upholds. They are stated, not
//! compiler-enforced; each is protected by a test that cites its id (grep the
//! `invariant`/`law` comments in the test modules). `notes/design.md` explains
//! them in prose. (CLI-level invariants `CLI-1`/`CLI-2` live in `yatima-cli`.)
//!
//! Library layering:
//! - **LAYER-1** dependencies point *down* the module layer DAG
//!   (primitives → model seam → {config, action} → edges); a type lives at the
//!   lowest layer that needs it, and a lower layer never depends on a higher one
//!   (`engine` names no `host`/`agent` type; `Role`/`Turn` live in `transcript`,
//!   not `agent`). Stated, not compiler-enforced within one crate — see the
//!   layer diagram in `notes/design.md`; a future crate split would enforce it.
//!
//! Model store & discovery:
//! - **MS-1** `models_root` precedence: `$YATIMA_MODELS_DIR`, else
//!   `${XDG_CACHE_HOME}/yatima/models`, else `$HOME/.cache/yatima/models`.
//! - **MS-2** [`model_dir`] mirrors possum's `<root>/<org>/<name>` layout.
//! - **MS-3** a [`ModelId`] and index shard names never escape the root / model
//!   directory (untrusted input is contained).
//! - **MD-1** unsharded discovery is every `*.safetensors`, sorted.
//! - **MD-2** indexed discovery is the unique `weight_map` values, deduped and
//!   sorted (also covers the dedup/order half of **DISC**).
//! - **MD-3** `presence` = `config.json` ∧ `tokenizer.json` ∧ all shards; a
//!   partial shard set is never a false cache hit.
//! - **EOS-1** EOS ids are read from `config.json` / `generation_config.json`
//!   as a *set* — never hard-coded token strings.
//! - **FETCH-1** `ensure_model` re-checks `presence` after download; a
//!   partial directory never reaches [`Engine::load`] (gated e2e / fetch path).
//!
//! Generation:
//! - **SAM-1** every [`Sampling`] maps to exactly one candle `LogitsProcessor`;
//!   **SAM-2** `Greedy` ignores any seed.
//! - **STOP-1** every successful generation returns exactly one [`StopReason`].
//! - **GEN-3** a generation emits at most `max_tokens` tokens.
//! - **GE-1** stateless: repeated `Greedy` runs on the same engine + prompt are
//!   byte-identical (gated e2e).
//!
//! Architecture & model configuration (single source of truth, no drift):
//! - **ARCH-1** every loaded [`Engine`] has exactly one detected [`Arch`]; the
//!   safetensors and GGUF load paths both normalize through that public enum.
//! - **ARCH-2** GGUF `general.architecture` strings are normalized to [`Arch`]
//!   at the load boundary (`glm4`/`chatglm` → `Glm4`, …); raw metadata strings
//!   never leak into dispatch logic.
//! - **PREFILL-1** device-sensitive prefill defaults are owned by the loaded
//!   engine ([`Engine::default_prefill_chunk`], from [`Arch::metal_prefill_chunk`]
//!   gated on the device); profiles and CLI flags only override deliberately.
//! - **HOST-1** an omitted chat format resolves to the architecture default
//!   ([`ChatFormat::default_for`] / [`resolve_format`], from [`caps_for`]).
//! - **HOST-2** a supplied format differing from the architecture default is
//!   honored but surfaced as a [`FormatMismatch`] warning, never silently
//!   mis-rendered.
//! - **CAPS-1** the agent/tool path is gated by host capability
//!   ([`ChatFormat::supports_tools`]); a chat-only format cannot enter it.
//! - **PROFILE-1** generation-option precedence is explicit and pure
//!   ([`ModelProfile::apply_gen_overrides`]): profile fields override a caller
//!   base `GenOpts`, and an unset `prefill_chunk` defers to the engine default.
//! - **PROFILE-2** a [`ModelProfile`] resolves to exactly one source (`repo`
//!   xor `dir`) before load.
//! - **CTX-1** the context window is discovered from model config at load
//!   ([`Engine::context_length`], from `max_position_embeddings` /
//!   `<arch>.context_length`) and enforced: a prompt plus `max_tokens` that
//!   would exceed it is refused before decode, rather than silently overflowing.
//!   An undeclared window imposes no constraint. (Hosts use the same budget to
//!   trim/compact a transcript — the higher rungs of the ladder in
//!   `notes/design.md`.)
//!
//! Runtime & concurrency (async-first, one owned runtime):
//! - **RT-1** the library owns exactly one (multi-thread) runtime and never
//!   builds one per call; every sync API is a thin shim over its async primitive
//!   via the single `runtime::block_on` bridge (which panics, with direction, if
//!   misused from a current-thread runtime); blocking compute (inference) runs
//!   only under [`run_blocking`], never directly on an async worker.
//! - **RT-2** local inference is synchronous, but an async caller may reach it
//!   **only through the runtime's blocking island**: the `Engine` decode
//!   primitives require a `BlockingIsland` witness that only `run_blocking_island`
//!   mints, so the `Completer` impl cannot perform model decode on an async
//!   worker — the executor-stalling path does not type-check. (Executor
//!   *liveness* then follows from the multi-thread commitment, RT-1, and is
//!   guarded by a liveness test. The low-level `generate_with` stays island-free
//!   as the deliberate sync escape hatch.)
//! - **CMP-1** [`Completer`] is a **native `async fn`** trait (not `async_trait`)
//!   so `Send` is inferred per impl — the local [`Engine`] is `!Send` and runs
//!   sync decode under the blocking island (RT-2), a remote completer is `Send`
//!   and awaits I/O. It is generic-only (never `dyn`), awaited inline (never
//!   spawned), so no global `Send` bound is imposed (`async_fn_in_trait` is
//!   `#[allow]`ed with that rationale). Contrast [`Tool`], which is `dyn` +
//!   spawned, hence `#[async_trait]` + `Send`.
//!
//! Agent & tools (capability-scoped action):
//! - **AGENT-1** the agent loop terminates in ≤ `max_steps` tool rounds.
//! - **AGENT-2** only tools in the agent's set are dispatchable — an unknown
//!   name is an `is_error` result, never ambient execution (sandbox by omission).
//! - **TOOL-1** tool calls are async task executions: they can be awaited,
//!   joined, watched through [`ToolEvent`], and cooperatively cancelled without
//!   changing their argument schema.
//! - **TOOL-2** [`ToolOutcome`] is the runtime truth of tool execution; the
//!   model-facing [`ToolResult`] is a projection at the protocol boundary.
//! - **CAP-1** a [`Dir`]-scoped tool cannot reach paths outside its root
//!   (containment, reusing `is_safe_relative` / MS-3).
//! - **CAP-2** the agent's effects ⊆ the union of its tools' capabilities —
//!   enforced for omission (AGENT-2) and containment (CAP-1); by construction
//!   otherwise (tools hold their caps, no ambient `std::fs` or arbitrary
//!   network destination). Stated, not compiler-absolute — see `notes/design.md`.
//! - **PROTO-1** a malformed/unknown tool call becomes a typed non-success
//!   [`ToolOutcome`] and then an `is_error` [`ToolResult`] the model can recover
//!   from, never a silent mis-execution.
//!
//! Observability:
//! - **OBS-1** `yatima-lib` emits `tracing` spans/events but never installs a
//!   global subscriber; hosts own collection and formatting.
//! - **OBS-2** info-level tracing never records prompts, generated text, tool
//!   arguments, fetched payloads, auth tokens, or whole user structs.
//! - **OBS-3** async spans are attached to futures; span guards are not held
//!   across `.await`.
//! - **OBS-4** telemetry data is structured and bounded: event messages name
//!   facts, while fields carry typed dimensions such as model, backend, tool,
//!   call id, token counts, stop reason, and outcome.
//!
//! Chat templates (instruction-following prompt rendering):
//! - **TMPL-1** a [`PromptTemplate`] emits no literal BOS when the model's
//!   tokenizer adds one (Gemma `<bos>`, Mistral `<s>`) — never double-BOS.
//! - **TMPL-2** for a model with no system role (Gemma, Mistral), system text is
//!   folded into the first user turn rather than emitted as a system turn.

mod agent;
mod capability;
mod chat;
mod completer;
mod engine;
mod host;
mod runtime;
mod template;
mod token_output_stream;
mod tool;
mod transcript;

pub use agent::{Agent, AgentEvent, AgentStop, Run};
pub use capability::{Dir, NtfyTopic, WebOrigin, WriteDir};
pub use chat::ChatSession;
pub use completer::{Completer, Completion};
#[cfg(feature = "fetch")]
pub use engine::ensure_model_blocking;
pub use engine::{
    device, is_model_present, Arch, Engine, GenOpts, Generation, PrefillLogits, PrefillProgress,
    Sampling, StopReason, TokenLogit,
};
pub use host::{
    caps_for, resolve_format, Caps, ChatFormat, FormatMismatch, ModelProfile, ModelSource,
};
pub use runtime::run_blocking;
pub use template::{
    ChatMlTemplate, GemmaTemplate, GlmTemplate, MistralTemplate, PlainTemplate, PromptTemplate,
};
pub use tool::{
    JsonToolCall, ListDir, QwenToolCall, ReadFile, ReadUrl, SendNotification, Tool, ToolCall,
    ToolCallCodec, ToolCallId, ToolCtx, ToolEvent, ToolFailure, ToolOutcome, ToolRejection,
    ToolResult, ToolSpec, ToolTask, Tools, WriteFile,
};
pub use transcript::{Role, Turn};

use anyhow::{bail, Result};
use std::ffi::OsString;
use std::path::{Component, Path, PathBuf};

/// The directory under which models are stored.
///
/// Resolution order: `$YATIMA_MODELS_DIR`, else
/// `${XDG_CACHE_HOME:-$HOME/.cache}/yatima/models`. Weights are
/// re-downloadable, so the default lives under the XDG cache.
pub fn models_root() -> PathBuf {
    resolve_models_root(
        std::env::var_os("YATIMA_MODELS_DIR"),
        std::env::var_os("XDG_CACHE_HOME"),
        std::env::var_os("HOME"),
    )
}

/// A validated Hugging Face repository id (e.g.
/// `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B`).
///
/// Parsing rejects anything that could escape the models root when joined —
/// empty ids, absolute paths, `..`, and empty path components — so that
/// [`model_dir`] is containment-safe by construction (invariant MS-3). The id
/// is untrusted input (a CLI flag), so this is the security boundary.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelId(String);

impl ModelId {
    /// Parse and validate a repository id.
    pub fn parse(s: &str) -> Result<ModelId> {
        if s.is_empty() {
            bail!("empty repository id");
        }
        if s.split('/').any(|seg| seg.is_empty()) {
            bail!("repository id '{s}' has an empty path component");
        }
        if !is_safe_relative(s) {
            bail!("repository id '{s}' must be relative with no '.' / '..' / root components");
        }
        Ok(ModelId(s.to_string()))
    }

    /// The underlying id string.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for ModelId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::str::FromStr for ModelId {
    type Err = anyhow::Error;
    fn from_str(s: &str) -> Result<Self> {
        ModelId::parse(s)
    }
}

/// The leaf directory holding a repository's files under `models_root`,
/// mirroring possum's on-disk layout (`<root>/<org>/<name>`). Safe by
/// construction: [`ModelId`] cannot escape the root.
pub fn model_dir(models_root: &Path, repo: &ModelId) -> PathBuf {
    models_root.join(repo.as_str())
}

/// Whether a path string is a relative path made only of normal components
/// (no root/prefix, no `..`) — i.e. it cannot escape a directory it is joined
/// onto. Used to validate both [`ModelId`]s and shard names from an index
/// manifest (untrusted data).
pub(crate) fn is_safe_relative(s: &str) -> bool {
    let p = Path::new(s);
    p.is_relative() && p.components().all(|c| matches!(c, Component::Normal(_)))
}

/// Pure core of [`models_root`], taking the relevant environment values as
/// arguments so it can be tested without mutating process state.
fn resolve_models_root(
    yatima_models_dir: Option<OsString>,
    xdg_cache_home: Option<OsString>,
    home: Option<OsString>,
) -> PathBuf {
    if let Some(dir) = yatima_models_dir {
        return PathBuf::from(dir);
    }
    let cache = xdg_cache_home
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(home.unwrap_or_default()).join(".cache"));
    cache.join("yatima").join("models")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn models_root_prefers_yatima_models_dir() {
        // upholds: MS-1
        let r = resolve_models_root(Some("/m".into()), Some("/c".into()), Some("/h".into()));
        assert_eq!(r, PathBuf::from("/m"));
    }

    #[test]
    fn models_root_falls_back_to_xdg_cache_home() {
        // upholds: MS-1
        let r = resolve_models_root(None, Some("/c".into()), Some("/h".into()));
        assert_eq!(r, PathBuf::from("/c/yatima/models"));
    }

    #[test]
    fn models_root_falls_back_to_home_cache() {
        // upholds: MS-1
        let r = resolve_models_root(None, None, Some("/h".into()));
        assert_eq!(r, PathBuf::from("/h/.cache/yatima/models"));
    }

    #[test]
    fn model_dir_mirrors_possum_layout() {
        // upholds: MS-2
        let root = PathBuf::from("/models");
        let id = ModelId::parse("deepseek-ai/DeepSeek-R1-Distill-Qwen-7B").unwrap();
        assert_eq!(
            model_dir(&root, &id),
            PathBuf::from("/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"),
        );
    }

    #[test]
    fn model_id_accepts_valid_ids() {
        // upholds: MS-3
        for id in [
            "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
            "Qwen/Qwen2.5-Coder-7B",
            "gpt2",
        ] {
            assert!(ModelId::parse(id).is_ok(), "{id} should parse");
        }
    }

    #[test]
    fn model_id_rejects_escaping_ids() {
        // upholds: MS-3
        for id in ["", "../etc", "a/../../b", "/abs/path", "a//b", "./x"] {
            assert!(ModelId::parse(id).is_err(), "{id:?} should be rejected");
        }
    }

    #[test]
    fn model_id_cannot_escape_model_dir() {
        // upholds: MS-3 — even constructed by hand, a parsed id stays under root.
        let root = PathBuf::from("/models");
        let id = ModelId::parse("org/name").unwrap();
        assert!(model_dir(&root, &id).starts_with(&root));
    }

    use proptest::prelude::*;

    proptest! {
        // upholds: MS-3 — for ANY input string, a parsed ModelId joins to a path
        // under the root with no `..`; parse rejects everything else.
        #[test]
        fn model_id_never_escapes(s in ".*") {
            let root = PathBuf::from("/models");
            if let Ok(id) = ModelId::parse(&s) {
                let dir = model_dir(&root, &id);
                prop_assert!(dir.starts_with(&root));
                prop_assert!(dir
                    .components()
                    .all(|c| !matches!(c, std::path::Component::ParentDir)));
            }
        }

        // upholds: MS-1 — models_root always follows the declared precedence.
        #[test]
        fn models_root_follows_precedence(
            ym in proptest::option::of("[^\u{0}/][^\u{0}]{0,16}"),
            xc in proptest::option::of("[^\u{0}/][^\u{0}]{0,16}"),
            home in "[^\u{0}/][^\u{0}]{0,16}",
        ) {
            let r = resolve_models_root(
                ym.clone().map(Into::into),
                xc.clone().map(Into::into),
                Some(home.clone().into()),
            );
            let expected = match (&ym, &xc) {
                (Some(m), _) => PathBuf::from(m),
                (None, Some(c)) => PathBuf::from(c).join("yatima").join("models"),
                (None, None) => PathBuf::from(home).join(".cache").join("yatima").join("models"),
            };
            prop_assert_eq!(r, expected);
        }
    }
}
