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
//!   (primitives → model boundary → {config, action} → edges); a type lives
//!   at the lowest layer that needs it, and a lower layer never depends on a
//!   higher one (`engine` names no `host`/`agent` type; `Role`/`Turn` live in
//!   `transcript`, not `agent`). Stated, not compiler-enforced within one
//!   crate — see the layer diagram in `notes/design.md`; a future crate split
//!   would enforce it.
//!
//! Model store & discovery:
//! - **MS-1** `models_root` precedence: `$YATIMA_MODELS_DIR`, else
//!   `${XDG_CACHE_HOME}/yatima/models`, else `$HOME/.cache/yatima/models`.
//! - **MS-2** [`model_dir`] mirrors possum's `<root>/<org>/<name>` layout.
//! - **MS-3** a [`ModelId`] and index shard names never escape the root / model
//!   directory (untrusted input is contained).
//! - **MS-4** exact GGUF resolution preserves a requested safe literal basename; one
//!   cached GGUF can never satisfy a request for another. An unnamed local
//!   directory resolves to a managed-server artifact only when it contains
//!   exactly one GGUF.
//! - **MD-1** unsharded discovery is every `*.safetensors`, sorted.
//! - **MD-2** indexed discovery is the unique `weight_map` values, deduped and
//!   sorted (also covers the dedup/order half of **DISC**).
//! - **MD-3** `presence` = `config.json` ∧ `tokenizer.json` ∧ all shards; a
//!   partial shard set is never a false cache hit.
//! - **EOS-1** EOS ids are read from `config.json` / `generation_config.json`
//!   as a *set* — never hard-coded token strings.
//! - **FETCH-1** `ensure_model` re-checks `presence` after download; a
//!   partial directory never reaches [`Engine::load`] (gated e2e / fetch path).
//! - **MEM-1** [`Engine::load`] refuses weights that exceed a safe fraction of
//!   physical RAM *before* allocating — an oversized model can exhaust memory and
//!   hang the machine (a raised Metal `iogpu.wired_limit_mb` makes it worse).
//!   Overridable with `YATIMA_ALLOW_OVERSIZED_MODEL`; skipped when RAM is unknown.
//! - **MEM-2** [`Engine::generate_with`] refuses to *start a turn* when this
//!   process's live resident footprint already exceeds the same safe fraction of
//!   RAM — the working set grows during decode (KV cache, activations, and Metal
//!   allocations not reclaimed between turns), so a later turn can tip the machine
//!   into swap and hang it even when the static weights fit. Same predicate as
//!   MEM-1 applied to live RSS; same override; skipped when RAM/RSS is unknown.
//!
//! Generation:
//! - **SAM-1** every [`Sampling`] maps to exactly one candle `LogitsProcessor`
//!   (temperature, with optional `top_p` nucleus truncation); **SAM-2** `Greedy`
//!   ignores any seed, while a seeded `Sample` is reproducible — the seed is
//!   threaded through to the sampler (gated e2e `seeded_sampling_is_reproducible`).
//! - **SAM-3** a backend adapter preserves Yatima's requested sampling policy:
//!   every backend sampler knob that can change the distribution is either set
//!   from Yatima's policy or explicitly neutralized; backend defaults never
//!   participate silently. The llama-server request-shape test is the dynamic
//!   witness.
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
//! - **PREFILL-1** device- and dtype-sensitive prefill defaults are owned by the
//!   loaded engine ([`Engine::default_prefill_chunk`], from
//!   [`Arch::metal_prefill_chunk`] gated on Metal **and** an F32 runtime dtype);
//!   a BF16/F16 model never chunks its prefill (chunked prefill would hit a
//!   Candle KV-cache `cat` dtype mismatch). Profiles and CLI flags only override
//!   deliberately.
//! - **FMT-1** an omitted chat format resolves to the architecture default
//!   ([`ChatFormat::default_for`] / [`resolve_format`], from [`caps_for`]).
//! - **FMT-2** a supplied format differing from the architecture default is
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
//!   trim/compact a transcript — the trim rung is built: COMPACT-1 below, with
//!   the host policy at HOST-5; the higher rungs of the ladder in
//!   `notes/design.md`.)
//! - **COMPACT-1** [`ChatSession::trim_history_to`] / [`Agent::trim_history_to`]
//!   drop whole committed exchanges (user+assistant pairs), oldest first, until
//!   the remaining history's rendered prompt fits the caller's token budget —
//!   never touching the seeded system prompt or the newest `keep_last`
//!   exchanges, so template alternation is never broken and only the protected
//!   turns can remain. Token counting uses the completer's tokenizer, falling
//!   back to a deterministic `chars/4` when it exposes none. This is the trim
//!   rung of the context ladder (`notes/design.md`); the *policy* (when to
//!   trim, to what budget) lives in a host (HOST-5). Cited by unit tests on
//!   both session types.
//! - **CTX-2** stock candle's Metal backend deterministically corrupts the
//!   KV state from depth 8,192 (`notes/metal-kv-cliff.md`). The pinned candle
//!   fork's sync workaround restores correctness at moderate depths
//!   (canary-validated ~8.4k, field-observed to ~15.5k) but not in the
//!   deepest water (garbage at ≥18k despite the syncs), so the engine flags
//!   Metal runs entering the band: debug-level through the validated depth,
//!   a warning past it. The `metal_kv_cliff_canary` test is the drill for
//!   every candle bump: green on pure upstream means the fork can be dropped.
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
//! - **CANCEL-1** [`Cancel`] is one monotone state with synchronous and
//!   asynchronous observations: cancellation by any clone is permanent, and
//!   every present or future poll or wait observes it without a lost wakeup.
//! - **LSRV-1** a managed llama-server child has one owner; both output pipes
//!   are drained concurrently into bounded tails, and every explicit shutdown
//!   or child-owning startup failure kills, reaps, and joins those drains under
//!   a bound. In-flight completion races child death; `Drop` is only a
//!   kill-request fallback.
//! - **LSRV-2** every llama-server endpoint is a validated HTTP loopback origin.
//!   Managed mode constructs its own `127.0.0.1` endpoint; attached mode
//!   accepts loopback literals or `localhost`, normalizing the latter before a
//!   request is built. Paths, userinfo, query, fragment, and non-loopback hosts
//!   are unrepresentable after configuration, and HTTP redirects are never
//!   followed.
//! - **LSRV-3** llama-server stop fidelity: returned and streamed text includes
//!   a matched caller-supplied stop marker even though the server excludes it
//!   from content (re-appended from the final event's `stopping_word`), and a
//!   word stop that names no marker — or names one the caller never supplied —
//!   is a protocol error, never a quiet `Stopped`. EOS is not required to
//!   appear in text (an EOS-ended turn carries no marker).
//! - **LSRV-4** llama-server cancellation races the pending initial response
//!   and every response-stream read through CANCEL-1; continuously buffered
//!   events also check the same state before delivery. The result carries only
//!   pre-cancellation text, with `Stopped`, and no polling interval participates.
//!
//! Agent & tools (capability-scoped action):
//! - **AGENT-1** the agent loop terminates in ≤ `max_steps` tool rounds.
//! - **AGENT-2** only tools in the agent's set are dispatchable — an unknown
//!   name is an `is_error` result, never ambient execution (sandbox by omission).
//! - **AGENT-3** across runs, an [`Agent`]'s session history carries only each
//!   completed exchange's user turn and final answer — tool rounds and
//!   reasoning are ephemeral to their run (the working-matter analogue of
//!   REASON-1), and an interrupted run leaves history untouched. A Final
//!   answer that looks like decode degeneration ([`looks_degenerate`], the
//!   CHAT-2 judgment) also commits nothing (cited by
//!   `a_degenerate_final_answer_commits_nothing`).
//! - **AGENT-4** agent steps stream: each decode's text arrives live as
//!   classified [`AgentEvent::Fragment`]s (reasoning vs answer — REASON-1
//!   holds mid-stream), codec markup never reaches the answer channel (the
//!   opener gate withholds it; the parsed call arrives as `ToolCall`), the
//!   final step's answer fragments concatenate to the run's answer (up to
//!   surrounding whitespace), and a fold `Break` or external [`Cancel`]
//!   stops the decode at token granularity — `Stopped`, history untouched.
//!   Answer fragments of a step that ends in a tool call are narration; the
//!   following `ToolCall` event licenses folding them into working matter.
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
//!   network destination). A web tool's authority is exactly its held origin
//!   *set* ([`WebOrigins`]): membership checked at call time, escapes refused
//!   before any network I/O, relative targets resolving only when exactly one
//!   origin is granted — and every **redirect hop is re-checked** like a
//!   fresh request (the network must not carry a granted request to an
//!   ungranted origin; the ntfy publisher follows no redirects at all).
//!   Coverage is **https-upgrade-tolerant**: a granted `http://X` covers
//!   `https://X` (same host/port) — the same authority over strictly
//!   stronger transport — never the reverse (no silent downgrade).
//!   Stated, not compiler-absolute — see `notes/design.md`.
//! - **CAP-3** web authority derives only from **user utterances**: an origin
//!   enters a session's [`WebOrigins`] iff the user typed a URL (auto-grant,
//!   scanned by [`origins_in`]) or issued an explicit grant command. Grants
//!   accumulate (session authority is the union), never persist across
//!   sessions, and shrink only by explicit revoke. Nothing a tool returns or
//!   the model generates reaches [`WebOrigins::grant`] — no such code path
//!   exists, so a fetched page cannot mint authority.
//! - **CAP-3a** the rendered tool specs state the model's live authority: a
//!   tool whose capability is empty is absent from [`Tools::specs`] (the
//!   model never sees a tool it cannot use), and a web tool's description
//!   enumerates its granted origins.
//! - **PAGE-1** within a session, [`ReadPage`] fetches each resolved URL at
//!   most once: repeat and continuation (`offset`) reads are served from a
//!   per-tool, FIFO-bounded cache and never touch the network — re-fetching
//!   is the expensive act for throttled hosts (EDGAR), re-reading is free.
//! - **WIN-1** `read_page` windows tile: successive windows are adjacent and
//!   non-overlapping, concatenating them in offset order reconstructs the
//!   article prefix exactly, and every truncation marker names the next
//!   window's `offset` (the marker is the pagination API). An offset at or
//!   past the end is a helpful error naming the length, never silent-empty.
//! - **PLOT-1** the [`Plot`] tool executes no model-authored code, ever: the
//!   model submits a declarative spec against a **closed schema** (unknown
//!   fields, unknown kinds, and anything code-shaped are typed rejections),
//!   and the sandbox's pinned interpreter runs only the library's generator.
//!   Function series (`expr`) are parsed and evaluated **host-side in Rust**
//!   against a closed grammar — numbers, `x`, `pi`, `e`, arithmetic, and a
//!   whitelisted function alphabet; the interpreter still receives only
//!   literal arrays. Data arrives inline, by expression, or by naming a
//!   host-registered dataset — the program supplies numbers, the model
//!   supplies labels and choices.
//! - **PLOT-2** plot output is confined to the [`PlotSandbox`]'s directory
//!   (CAP-1 containment reused); one PNG per call; the model never chooses
//!   the path.
//! - **PLOT-3** rendering is deterministic per machine (Agg backend, fixed
//!   size/dpi, stable metadata, spec-hash filenames): the same spec yields
//!   the same artifact — a plot can be journaled like any other evidence.
//! - **IMG-1** the [`ReadImage`] tool fetches only from granted origins
//!   (CAP-2/CAP-3 reused), gates on image honesty (SVG/PNG/JPEG by
//!   content-type, magic-byte sniff when the server is silent — anything
//!   else is a teaching rejection), caps input size while streaming, and
//!   confines output to its [`WriteDir`] at a content-hash name: the model
//!   never chooses the path, identical bytes share an artifact, and each
//!   host displays it in its medium's idiom.
//! - **IMG-2** display authority is the typed artifact event
//!   ([`ToolCtx::emit_artifact`] → `ToolEvent::Artifact` →
//!   `AgentEvent::ToolArtifact`), never a parse of result prose — and
//!   **every successful call emits it**. `read_image`'s fetch-once memo
//!   (by URL *and* by content hash) governs the *network* and the
//!   *narration* — a repeat never refetches, and its teaching text tells
//!   the model not to present the picture as new — but never the pixels:
//!   the host cannot assume a view still holds them (serve's browser
//!   client reloads; withholding the event on "already shown" left a
//!   reloaded view silently empty while the model narrated success —
//!   observed live). Display dedup, if any view wants it, is that view's
//!   policy. `{"again": true}` survives as the explicit re-show wording.
//!   When every listed image has been shown, the teach states that
//!   exhaustion as a computed fact. Cited by the repeat/duplicate,
//!   re-show, and exhaustion tests on `read_image` and the artifact-event
//!   tests.
//! - **IMG-3** picking a picture is an index copy, never a URL
//!   transcription: `read_page`'s first window publishes its numbered
//!   `[images]` list into a session-shared [`ImageListing`], and
//!   `read_image {"image": N}` selects from it (the exact-url form remains
//!   for user-supplied targets). The listing covers the whole fetched page
//!   (article-region images first, the rest deduped after — the extraction
//!   alone misses galleries and navboxes), every entry is selectable even
//!   past the printed head, and truncation is always spoken, never silent.
//!   No listing yet and out-of-range numbers teach rather than fail
//!   opaquely. Cited by the numbered-listing, page-wide-coverage,
//!   spoken-truncation, and select-by-number tests.
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
//! - **REASON-1** every active response-framing protocol separates reasoning
//!   from answer text at the completion→turn boundary: reasoning and protocol
//!   framing never enter the committed transcript re-rendered into a later
//!   prompt, and a reply with **no** answer text (all reasoning/framing — a
//!   truncated think block, a Muse turn that never reaches `to=user`) commits
//!   nothing at all. The boundary is [`PromptTemplate::interpret_response`] —
//!   owned by the same template that renders the prompt, so each protocol has
//!   one definition of its *final* interpretation (live-stream display is
//!   classified separately until the stage-3 streaming interpreter).
//!   Implementations: the marker-dialect splitter ([`split_reasoning`],
//!   identity when no marker is present; an unterminated opener classifies as
//!   reasoning), its seeded variant (`split_seeded_reasoning`, crate-private,
//!   used by the pre-seeded templates — a close-less reply never left the
//!   think block), and the Muse ATEM interpreter ([`MuseGlimmerTemplate`]'s
//!   override).
//! - **CHAT-1** a [`ChatSession`] turn is atomic: if its completion errors, the
//!   user turn is rolled back so the transcript is unchanged. A failed turn never
//!   poisons the session — a later turn re-renders clean history and succeeds.
//! - **CHAT-2** a final answer that looks like decode degeneration
//!   ([`looks_degenerate`] — the Metal KV-cliff garbage modes) rolls the
//!   exchange back the same way: the caller still sees the text, but it never
//!   re-enters a prompt (cited by `a_degenerate_turn_is_not_committed`).

mod agent;
mod backend;
mod cancel;
mod capability;
mod chat;
mod completer;
mod engine;
mod expr;
mod host;
mod reasoning;
mod runtime;
mod template;
mod token_output_stream;
mod tool;
mod transcript;

pub use agent::{Agent, AgentEvent, AgentStop, Run};
pub use backend::{
    LlamaServer, LlamaServerCompleter, LlamaServerConfig, LlamaServerSpawn, ServerProps,
};
pub use cancel::Cancel;
pub use capability::{origins_in, Dir, NtfyTopic, PlotSandbox, WebOrigin, WebOrigins, WriteDir};
pub use chat::{looks_degenerate, ChatSession};
pub use completer::{Completer, Completion};
#[cfg(feature = "fetch")]
pub use engine::ensure_model_blocking;
pub use engine::{
    device, is_model_present, metal_kv_depth_risk, Arch, Engine, GenOpts, Generation, KvDepthRisk,
    PrefillLogits, PrefillProgress, Sampling, StopReason, TokenLogit, METAL_KV_VALIDATED,
};
pub use host::{
    caps_for, resolve_format, Caps, ChatFormat, FormatMismatch, GgufArtifact, ModelProfile,
    ModelSource, ResolvedModel, REASONING_MIN_TOKENS,
};
pub use reasoning::{split_reasoning, strip_reasoning, Channel, Reasoned, ReasoningSplitter};
pub use runtime::run_blocking;
pub use template::{
    ChatMlTemplate, ChatMlThinkTemplate, DeepSeekTemplate, GemmaTemplate, GlmTemplate,
    MistralTemplate, MuseGlimmerTemplate, PlainTemplate, PromptTemplate, ReasoningStrength,
};
pub use tool::{
    ImageListing, JsonToolCall, ListDir, Plot, PlotBound, PlotSeries, QwenToolCall, ReadFile,
    ReadImage, ReadPage, ReadUrl, SendNotification, Tool, ToolCall, ToolCallCodec, ToolCallId,
    ToolCtx, ToolEvent, ToolFailure, ToolOutcome, ToolRejection, ToolResult, ToolSpec, ToolTask,
    Tools, WriteFile,
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

/// Whether a literal filename would be interpreted as a pattern by possum's
/// `glob`-based include filter.
pub(crate) fn has_glob_metachar(s: &str) -> bool {
    s.chars()
        .any(|c| matches!(c, '*' | '?' | '[' | ']' | '{' | '}' | '\\'))
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
    fn registry_ids_are_unique_across_crate_docs() {
        // Registry well-formedness: one id names at most one declared law.
        // A declaration is a crate-doc bullet beginning `- **ID**`; additional
        // bold ids in the same bullet are declarations too (SAM-2 is written
        // this way). Bold references outside a declaration bullet are ignored.
        let root = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .expect("yatima-lib lives one directory below the workspace root");
        let mut declarations = std::collections::BTreeMap::<String, Vec<String>>::new();

        for entry in std::fs::read_dir(root).expect("read workspace root") {
            let dir = entry.expect("read workspace entry").path();
            if !dir.join("Cargo.toml").is_file() {
                continue;
            }
            for source in [dir.join("src/lib.rs"), dir.join("src/main.rs")] {
                let Ok(text) = std::fs::read_to_string(&source) else {
                    continue;
                };
                for (line_no, id) in declared_law_ids(&text) {
                    declarations.entry(id).or_default().push(format!(
                        "{}:{}",
                        source.strip_prefix(root).unwrap_or(&source).display(),
                        line_no
                    ));
                }
            }
        }

        let duplicates: Vec<_> = declarations
            .into_iter()
            .filter(|(_, sites)| sites.len() > 1)
            .collect();
        assert!(
            duplicates.is_empty(),
            "duplicate invariant/law ids: {duplicates:#?}"
        );
    }

    fn declared_law_ids(text: &str) -> Vec<(usize, String)> {
        let mut ids = Vec::new();
        let mut declaration_bullet = false;
        for (line_no, line) in text.lines().enumerate() {
            let Some(doc) = line.strip_prefix("//!") else {
                declaration_bullet = false;
                continue;
            };
            let body = doc.trim_start();
            if body.starts_with("- **") {
                declaration_bullet = true;
            } else if body.starts_with("- ") || body.is_empty() {
                declaration_bullet = false;
            }
            if declaration_bullet {
                ids.extend(bold_law_ids(body).into_iter().map(|id| (line_no + 1, id)));
            }
        }
        ids
    }

    fn bold_law_ids(line: &str) -> Vec<String> {
        let mut ids = Vec::new();
        let mut rest = line;
        while let Some(open) = rest.find("**") {
            rest = &rest[open + 2..];
            let Some(close) = rest.find("**") else {
                break;
            };
            let candidate = &rest[..close];
            let Some((prefix, ordinal)) = candidate.split_once('-') else {
                rest = &rest[close + 2..];
                continue;
            };
            let digit_count = ordinal.chars().take_while(char::is_ascii_digit).count();
            let (number, suffix) = ordinal.split_at(digit_count);
            if !prefix.is_empty()
                && prefix.chars().all(|c| c.is_ascii_uppercase())
                && !number.is_empty()
                && number.chars().all(|c| c.is_ascii_digit())
                && suffix.chars().all(|c| c.is_ascii_lowercase())
            {
                ids.push(candidate.to_owned());
            }
            rest = &rest[close + 2..];
        }
        ids
    }

    #[test]
    fn registry_parser_includes_letter_suffixed_ids_after_attributes() {
        let source = concat!(
            "#![allow(dead_code)]\n",
            "//! Registry\n",
            "//! - **CAP-3a** first declaration\n",
            "//! - **SAM-1** and **SAM-2** share a bullet\n",
            "fn item() {}\n",
            "//! - **CAP-3a** planted duplicate\n",
        );
        let ids: Vec<_> = declared_law_ids(source)
            .into_iter()
            .map(|(_, id)| id)
            .collect();
        assert_eq!(ids, ["CAP-3a", "SAM-1", "SAM-2", "CAP-3a"]);
        assert_eq!(ids.iter().filter(|id| id.as_str() == "CAP-3a").count(), 2);
    }

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
