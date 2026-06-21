# yatima — design notes

`yatima` is a small Rust runtime for **language-integrated LLMs**: calling a
local model as an ordinary in-process function. That's a *building block*, not a
fixed product shape — embed it in an app, wrap it in a service, drive it from a
TUI, compose it however the work demands; the in-process function is the
foundation those are built on, not an alternative to them. We **own the runtime,
rent the engine** — `yatima-lib` owns loading, the generation loop, and
capability-scoped tools; the inference engine
([candle](https://github.com/huggingface/candle)) is a swappable dependency.

## Crates

- **`yatima-lib`** — the capability as a function: `Engine::{load, generate,
  generate_with}`, `Sampling`/`GenOpts`/`Generation`/`StopReason`, the
  `ModelId`/`models_root`/`model_dir` resolver, the `presence`/`model_shards`
  discovery, and (behind the `fetch` feature) `ensure_model`. Plus the acting
  layer: the `Completer` model seam, `Dir` capabilities, `Tool`/`Tools` with the
  `ToolCallCodec` protocol, and the `Agent` loop.
- **`yatima-cli`** — a thin wrapper: `yatima generate`, `yatima chat`, `yatima
  agent`, and `yatima models-dir`, with model selection parsed into a
  `ModelSource` ADT at the edge.

## Three layers

The runtime exposes three increasing-capability modes over the same `Engine`:

- **`generate`** — raw completion, no chat template. The primitive.
- **`chat`** — instruction-following: apply the model's native chat template (no
  tools). Renders a transcript via a `PromptTemplate` (`--format
  qwen|gemma|mistral|plain`) then streams `Engine::generate`. One-shot with
  `--prompt`; **omit it for an interactive multi-turn session** (reads stdin;
  `/exit` quits, `/reset` clears). Conversation memory comes from re-rendering the
  whole growing transcript each turn — the `Engine` stays stateless per call, so
  history lives in the prompt, not the KV cache. This is the layer that makes an
  *instruct* model behave as trained — without it, raw text underperforms.
- **`agent`** — the tool loop, for **tool-trained** models only.

The split matters because **chat needs only a chat template, but agent needs the
model to be trained to emit tool calls** — two different bars. Gemma-2 clears the
first, not the second. Capability by model family:

| Model family      | generate | chat  | agent/tools |
|-------------------|----------|-------|-------------|
| Qwen2.5-Instruct  | yes      | yes   | yes         |
| GLM-4 (9B / 32B)  | yes      | yes   | no          |
| Gemma-2-it        | yes      | yes   | no          |
| Mistral-v0.3      | yes      | yes   | later/complex |
| TinyLlama-chat    | yes      | yes   | no          |
| StarCoder2        | yes      | maybe | no          |

Chat templates omit a literal BOS when the model's tokenizer adds one
(Gemma `<bos>`, Mistral `<s>` via `TemplateProcessing`) — never double-BOS
(TMPL-1); models without a system role fold system text into the first user turn
(TMPL-2).

## Generation: an effectful fold (the contract)

`generate_with` is the primitive; `generate` is the `acc = ()` specialization
that just streams fragments to a side-effecting callback.

```rust
fn generate_with<A>(&mut self, prompt: &str, opts: &GenOpts, init: A,
    step: impl FnMut(A, &str) -> Result<ControlFlow<A, A>>) -> Result<(A, Generation)>;
```

The portable contract (what the later Haskell study reasons about):

- **Stateless per call** — the KV cache is cleared on entry; no conversation or
  cache retained across calls (GE-1).
- **Raw completion** — the prompt is fed as-is; no chat template.
- `step` receives **decoded text fragments** (not token ids), **in generation
  order**, via an incremental detokenizer (`TokenOutputStream`, a Mealy machine
  `state → token → (state, Option fragment)`).
- It returns `ControlFlow`: `Continue(acc)` keeps folding, `Break(acc)` stops
  voluntarily (`StopReason::Stopped`), `Err` is propagated. Generation also stops
  on EOS or `max_tokens` — **exactly one `StopReason` per run** (STOP-1), and
  `tokens ≤ max_tokens` (GEN-3).
- **Sampling** is an explicit choice (no `temperature ≤ 0` sentinel):
  `Sampling::Greedy` is deterministic and seed-free (SAM-2); `Sample
  { temperature, seed }` is seeded. Every `Sampling` maps to exactly one candle
  `LogitsProcessor` (SAM-1).

EOS ids are read from `config.json` / `generation_config.json` (a *set*, e.g.
DeepSeek's `<｜end▁of▁sentence｜>` = 151643) — never hard-coded strings.

**North star — a hylomorphism.** Generation *unfolds* a fragment stream (a
coalgebra deciding termination on EOS/max/break) and *folds* it with the caller's
`step` algebra: `generate = ana ; cata = hylo`. The hot loop stays imperative;
this is the denotation, not the implementation — the recursion-scheme reading is
what the (planned) Haskell study formalizes.

## Acting: capability-scoped tools & the agent loop

If `generate_with` folds *tokens* into a value, the agent loop folds *turns*: the
model emits a tool call, a capability-scoped tool runs, its result is fed back,
and the loop repeats until the model answers or `max_steps` is reached (the
hylomorphism nests one level up). It is synchronous (turns are sequential and
compute-bound, per the concurrency note) and provable against a *scripted*
`Completer` with no GPU.

The design is **small composable seams**, simplest concrete impl behind each:

- **`Completer`** (the model seam) — `complete(prompt, opts, stops) ->
  Completion { text, stop }`, stopping at EOS / `max_tokens` / a stop string,
  with the stop marker **included** so the codec sees the whole block. `Engine`
  implements it over `generate_with`; tests use a canned `Completer`. This is
  also the engine-swap seam (mistral.rs / llama.cpp would be another impl).
- **`Dir`** (authority as a value) — a rooted filesystem capability;
  `resolve(rel)` rejects escapes via the same `is_safe_relative` check as
  `ModelId` (MS-3 / CAP-1). A tool *holds* its capabilities; we never hand it
  ambient `std::fs`.
- **`Tool` / `Tools`** — a tool advertises a `ToolSpec` (JSON-Schema params, the
  de-facto standard) and runs `call(args)`. `Tools::dispatch` **never
  hard-errors**: an unknown name (AGENT-2) or a tool failure becomes an
  `is_error` `ToolResult` the model can recover from (PROTO-1). First tools:
  `ReadFile`, `ListDir` (read-only).
- **`PromptTemplate`** (the chat format) — renders the transcript into the
  model's *native* prompt string. `ChatMlTemplate` (Qwen2.5), `PlainTemplate`
  (fallback/tests). A model is acutely sensitive to its trained format; the wrong
  one degenerates it.
- **`ToolCallCodec`** (the protocol) — `QwenToolCall` (ChatML/Hermes
  `<tool_call>{json}</tool_call>` with `arguments`) and `JsonToolCall` (a plain
  convention, for tests / the `plain` fallback). `parse` returns
  `None` (a plain answer), `Some(Ok)` (a call), or `Some(Err)` (malformed ⇒ an
  error turn), and is **panic-proof on any input** (proptest). Each does strict
  JSON first, then a **tolerant** pass that recovers common real-model defects
  (an unquoted name, braces inside string values) — this is the actual line of
  defense for tool-call validity (constrained decoding was tried and shelved; see
  Deferred).
- **`Agent`** — `run` collects the final answer; `run_with` is the fold a future
  actor/TUI streams `AgentEvent`s into (`run` is the `acc = ()` specialization).
  Bounded by `max_steps`; `AgentStop` is `Final` / `MaxSteps` / `Stopped` (the
  last when the caller's fold returns `ControlFlow::Break`).

**Speaking the model's native format (hard-won).** Getting a base model to
*reliably* act took three corrections, each now a guarded invariant:
- **Detokenization must be faithful.** Streaming decode emits a fragment unless
  the text ends in U+FFFD (an unfinished byte sequence) — the canonical TGI/vLLM
  condition. candle's example heuristic (emit on a trailing alphanumeric) buffers
  punctuation and, on a stop, drops it — silently truncating tool-call JSON
  (`"`, `}`, `</tool_call>`). A weights-free round-trip test pins this.
- **Repetition penalty is per-call.** ~1.1 keeps prose from degenerating
  (repeated words) but mangles structured JSON punctuation; it is a `GenOpts`
  knob, kept on (prose) and absorbed for tool calls by the tolerant parser.
- **Model choice matters.** R1 distills were trained on reasoning, not
  tool-calling, and won't emit tool calls even when shown the format; Qwen2.5
  (Qwen2 arch, loads unchanged) is trained for it and is the agent default.

Tool-call *validity* is handled by native format + tolerant parsing (constrained
decoding was tried and did not earn its keep — see Deferred). Free-text answer
*quality* (e.g. a 7B greedily misreading a value back) is a separate, model-bound
concern that neither addresses.

**Honesty (where we partially, not wholly, match Anil's ocap story).** Rust gives
*capabilities by construction + enforced containment*, not Eio's
language-enforced object-capabilities. The enforced parts: a tool not in the
agent's set is uncallable (sandbox by omission, AGENT-2), and a `Dir`-scoped path
is containment-checked (CAP-1). The convention part: tools are written against
capability handles, not ambient effects. The capability model — little
agent-market precedent (MCP ≈ "trust the server"; function-calling has no
capability notion) — is the part worth owning; its lineage is ocap *systems*
(Eio, WASI Preview 2, Pony/E).

**Interop.** Schemas and roles follow the de-facto standard (JSON-Schema params;
system/user/assistant/**tool** turns) rather than reinvent. MCP is a different
problem — a transport for *out-of-process* tool servers; later it can ride the
same seams at the edge (consume an MCP server *as* a `Tool`, or expose our tools
*as* an MCP server), without changing the in-process core.

## Model storage & loading

Weights are re-downloadable ⇒ they live under the XDG **cache**:

```
$YATIMA_MODELS_DIR  (else  ${XDG_CACHE_HOME:-~/.cache}/yatima/models)
  └── <org>/<name>/        # = model_dir(models_root(), &ModelId)
        config.json  tokenizer.json  *.safetensors  [model.safetensors.index.json]
```

This mirrors the layout written by
[`possum`](https://github.com/shayne-fletcher/possum), our standalone Hugging
Face downloader: **possum acquires, yatima loads** — agreement by *convention,
not coupling* (MS-2). `Engine::load` is HF-agnostic (takes a directory).

- **`ModelId`** is a validated newtype: a `--repo` id (untrusted input) is parsed
  rejecting empty / absolute / `..` / empty-component ids, so `model_dir` cannot
  escape the root (MS-3). The same `is_safe_relative` check guards shard names
  read from the (untrusted) index `weight_map`.
- **`model_shards`** is the single discovery rule used by both `Engine::load`
  (what to mmap) and `presence` (what must exist): index `weight_map` values when
  present (deduped, sorted, contained), else all `*.safetensors` (MD-1/MD-2).
- **`presence(dir) -> { complete, missing }`**: `complete` is the conjunction
  over `config.json`, `tokenizer.json`, and every shard, so a partial shard set
  is never a false cache hit; `missing` names what's absent.

### Supported architectures

`Engine::load` dispatches on the model's architecture rather than assuming one.
`detect_arch` reads `config.json`'s `architectures` class name (and falls back to
`model_type`), then a private `CausalLm` trait gives the generation loop one
uniform `forward`/`reset` regardless of family. candle's models come in two
shapes: most self-manage a KV cache (`forward(ids, offset)` + `clear_kv_cache`);
**Llama** threads an external cache and has no clear (reset = rebuild it), and
returns last-token logits as `[1, vocab]` vs the others' `[1, 1, vocab]` — both
normalized by the single `last_token_logits` helper. Unsupported models fail with
a clear "unsupported architecture" error, not a serde mismatch.

Loadable today: **Qwen2, Llama, Mistral, Phi-3, Gemma-2, StarCoder2, GLM-4**
(safetensors) plus **GGUF/quantized** Qwen2, Llama, and GLM-4. Note this covers
*loading + `generate`* (raw completion) for all of them, and `chat` for those with
a chat template (Qwen/Gemma/Mistral/GLM); the **agent** path still assumes the
Qwen/ChatML tool format.

**GGUF / quantized.** A model dir with a single `*.gguf` takes the quantized
path: `Engine::load_gguf` reads the file's metadata for the architecture
(`general.architecture` → `quantized_qwen2` / `quantized_llama`) and EOS
(`tokenizer.ggml.eos_token_id`, unioned with any `<|im_end|>`/`<|endoftext|>` in
the vocab). The quantized model types fit the same self-cache `CausalLm` shape,
and quantized matmul runs on Metal — so a **32B-Q4** (~20 GB) runs on a 48 GB Mac,
the real lever for answer quality on long prose. **GGUF is self-contained:** the
tokenizer is built from the file's embedded `tokenizer.ggml.*` metadata
(candle-core's `TokenizerFromGguf`) — no sibling `tokenizer.json` needed (one is
used only if present). So `--repo <id> --gguf <file>` fetches a single file and
just works; `--model <dir>` also works for a local `.gguf`. Deferred: sharded
multi-file GGUF, more quantized arches.

## Auto-fetch (the `fetch` feature)

`yatima generate --repo <id>` fetches on a cache miss by calling
[`possum-lib`](https://github.com/shayne-fletcher/possum) in-process (no shelling
out): cache hit → quiet load; miss → `ensure_model` downloads (include
`*.safetensors`/`*.json`, exclude `figures/*`, `ProgressMode::Auto`), **re-checks
`presence`** (FETCH-2: never hand a partial dir to `load`), then loads;
`--offline` never touches the network; `--model <dir>` bypasses resolution.
Gated repos (e.g. Gemma) authenticate when `HF_TOKEN` is set in the environment
(passed to possum as a bearer token; unset → public-only, as before).

## Concurrency

Decode is sequential and compute-bound — token *n+1* depends on *n*, and the hot
loop does GPU/CPU work with nothing to await. So the **core stays synchronous**:
an `async fn generate` would be a lie that blocks the executor and starves other
tasks. Concurrency belongs to *delivery*, not decode.

The async form is therefore an **adapter over `generate_with`, not a second
engine path** — run the blocking fold on `spawn_blocking` and let `step` send
fragments down a bounded channel:

```rust
pub fn generate_events(engine, prompt, opts) -> mpsc::Receiver<Event>;
pub enum Event { Fragment(String), Done(Generation) }
```

Two properties fall out of choices already made:

- **Cancellation** — if the consumer drops the receiver, `blocking_send` fails,
  `step` returns `ControlFlow::Break`, and generation stops
  (`StopReason::Stopped`). This is why the `generate_with` primitive exists; the
  plain `generate` callback (`-> Result<()>`) can't express it.
- **Backpressure** — a *bounded* channel parks the decode thread when the
  consumer is slow, so generation tracks the client's pace.

Categorically: decode is the coalgebra (unfold), `step` the algebra (fold), and
the channel adapter is the natural transformation carrying a blocking fold into
an async stream — without touching the core.

Scope honesty: per-completion decode is sequential and GPU-bound; *cross-request*
parallelism is a batching/scheduling concern rented from the engine / a future
server, not faked here. One `Engine` (mutable KV cache, ~15 GB weights) is
one-generation-at-a-time.

**Deferred** (build when a real async client exists): the durable shape is an
**engine actor** owning the `Engine` and serving `(prompt, opts, reply)` requests
over a channel — also the home for conversation/KV-cache reuse and the agent
loop, so concurrency and the capability-scoped tool runtime are the *same* future
component.

## Registries

The **canonical** invariant & law registry lives in the crate docs — see the
`yatima-lib` crate doc (`lib.rs`: model store/discovery + generation) and the
`yatima-cli` `main.rs` doc (`CLI-1`/`CLI-2`). Each is protected by a test that
cites its id in an `// upholds: <id>` comment (`grep -r 'upholds:'`).

In brief: model store & discovery (**MS-1/2/3**, **MD-1/2/3**, **EOS-1**,
**FETCH-1**, dedup/order under **DISC**); generation (**SAM-1/2**, **STOP-1**,
**GEN-3**, **GE-1**); agent & tools (**AGENT-1/2**, **CAP-1/2**, **PROTO-1**);
chat templates (**TMPL-1/2**); CLI (**CLI-1/2**).

## State machines

Model acquisition / loading:

```mermaid
stateDiagram-v2
    [*] --> SourceSelected
    SourceSelected --> Present: present
    SourceSelected --> Missing: absent
    Missing --> Error: offline
    Missing --> Fetching: online
    Fetching --> Present: recheck presence
    Fetching --> Error: download failed or incomplete
    Present --> Loaded: load
    Loaded --> [*]
```

Generation:

```mermaid
stateDiagram-v2
    [*] --> Prefill
    Prefill --> Decode
    Decode --> Emit: fragment decoded
    Decode --> Stop: EOS or max_tokens
    Emit --> Stop: step Break or Err
    Emit --> Decode: step Continue
    Stop --> Flush
    Flush --> [*]
```

## References

- Anil Madhavapeddy, *"Language Integrated LLMs as an OCaml Function"* —
  https://anil.recoil.org/notes/language-integrated-llms. Kindred motivation for
  calling a local model as an ordinary in-process function.
- **Algebra (related, not a dependency):**
  [`axiom`](https://github.com/shayne-fletcher/axiom) — a lawful-composition
  library (`All`/lattices, `Fix`/`fold`). yatima doesn't depend on it today; it's
  the intended home for the recursion-scheme formalization (see the Haskell study
  in Deferred).

## Roadmap (future features)

Grouped by area; rough priority within each. Items marked *(lesson)* were tried
and deliberately shelved — the note records why so we don't repeat them.

### Chat
- **Context-window management** — the multi-turn REPL grows the transcript
  unbounded and will degrade/err past the model's window; add trimming (drop or
  summarize oldest turns) keyed off a token budget.
- **Readline niceties** — line editing + history (e.g. `rustyline`); the loop is
  plain `stdin` today.
- **Conversation persistence** — save/load a session transcript.
- **Reasoning-model think-stripping in chat** — instruct defaults don't emit a
  `<think>` block; add stripping if a reasoning model is used via `chat`.

### Tools / agent
- **Constrained decoding** *(lesson)* — a top-K JSON-prefix masking prototype was
  built and reverted: generic JSON validity forces *structure* but not *meaning*,
  so Qwen2.5 shifted to a valid-prefix-but-wrong run-on string the tolerant
  parser misreads — net **worse** than the parser alone. The real fix is
  **schema-aware** masking (the `name` constrained to a prefix of a real tool
  name, forcing the closing quote on match; arguments to the param schema). Only
  worth doing schema-aware. Addresses tool-call *validity* only, never free-text
  quality.
- **Native Mistral tool codec** (`[TOOL_CALLS]` / `[AVAILABLE_TOOLS]`) — a second
  tool-trained family in the agent; complex (special-token detok, 9-char IDs).
- **Richer capabilities** — `write` / `net` / `proc` capabilities and their tools
  (read-only `Dir` today); multi-tool-per-turn / planning.
- **MCP edge adapter** — consume an MCP server *as* a `Tool`, or expose our tools
  *as* an MCP server (out-of-process; rides the same seams at the edge).

### Models / engine
- **Engine swappability** — cross-arch dispatch + GGUF/quantized + self-contained
  GGUF tokenizers are **done**. Remaining: sharded multi-file GGUF + more
  quantized arches; other backends (mistral.rs, llama.cpp) via more `Completer`
  impls.
- **More chat templates** — Llama-3, Zephyr/TinyLlama (same shape as Gemma/Mistral).
- **Sampling quality** — `top_p`/`top_k` nucleus sampling (only temperature
  today) for better free-text on smaller models; download integrity/resume.

### Embedding (library surface)
- **`ChatSession` — done.** The multi-turn fold is a public lib type
  (`ChatSession<C: Completer, T: PromptTemplate>`, borrows the completer like
  `Agent`), exercised by `lib/examples/embed.rs` (a real non-CLI consumer:
  conversation + model→`enum`→native branch) and dogfooded by the CLI's one-shot
  `chat`. This is what makes "language-integrated" a demonstrated fact.
- **Streaming `Completer`** — the next API step: a token-callback variant so
  `ChatSession` can stream too (today streaming is `Engine::generate`-only, so the
  interactive REPL keeps its own loop). Would let the CLI fully dogfood
  `ChatSession`.
- A worked **service/TUI** embedder — the next consumer after the example.

### Architecture / research
- **Async engine-actor** owning the `Engine` and wrapping `run_with` — the home
  for cross-request concurrency and KV-cache reuse (see Concurrency).
- **`lexicon` crate** — a shared, dependency-light home for `ModelId` + the
  `<root>/<org>/<name>` layout, extracted once there's a real trigger (possum
  validating its own ids, or a second consumer).
- **The Haskell study** — formalize the `generate` and agent contracts
  (GE/STOP/AGENT/CAP/PROTO/TMPL as propositions); the planned home for the
  recursion-scheme reading and `axiom`.
