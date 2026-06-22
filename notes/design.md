# yatima — design notes

This is the deep architecture record. The README is the public front door; this
note keeps the contracts, invariants, state machines, and hard-won design
details close to the code.

`yatima` is a small Rust runtime for **language-integrated LLMs**: calling a
local model as an ordinary in-process function and letting it act only through
explicit capabilities. That's a *building block*, not a fixed product shape —
embed it in an app, wrap it in a service, drive it from a TUI, compose it however
the work demands. We **own the runtime, rent the engine** — `yatima-lib` owns
loading, the generation loop, chat/session state, and capability-scoped tools;
the inference engine ([candle](https://github.com/huggingface/candle)) is a
swappable dependency.

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

## Module layering (LAYER-1)

One law underlies several boundary fixes: **dependencies point *down* this DAG,
and a type lives at the lowest layer that needs it.** A lower layer never depends
on a higher one. This is the rule that, stated, catches a whole class of
organic-growth bug at review time — `template`/`chat` once imported `Turn` from
`agent`; `engine` once carried a `ChatFormat`-shaped `Caps`. Both were *upward*
dependencies; both are now fixed by moving the type to its altitude.

```mermaid
flowchart TD
    edges["edges — cli / examples (#[tokio::main])"]
    host["config — host (ChatFormat, ModelSource, ModelProfile, Caps)"]
    action["action — capability · tool · agent · chat"]
    seam["model seam — engine · completer · template"]
    prim["primitives — transcript · runtime · token_output_stream · root ids/paths"]
    edges --> host
    edges --> action
    host --> seam
    action --> seam
    seam --> prim
```

An arrow is "may depend on". Reading it: **primitives** (`transcript` = `Role`/
`Turn`, `runtime` = the one bridge + island, `token_output_stream`, the
`ModelId`/`model_dir` resolver) depend on nothing in-crate. The **model seam**
(`engine`, `completer`, `template`) sits on primitives (`completer` over
`engine`). **config** (`host`) and **action** (`capability` → `tool` →
`agent`/`chat`) are *siblings* — both consume the seam, neither depends on the
other (the agent never imports `host`; `host` never imports the agent). The
**edges** (CLI, examples) sit on top and may use everything.

Enforcement: within one crate the compiler permits any module to `use` any
other, so LAYER-1 is a *stated* law — held by review and the cheap
relocate-on-catch discipline (the `Turn` move was a single commit). A future
multi-crate split (e.g. extracting `transcript`/`lexicon` — see Roadmap) would
make it compiler-enforced; deferred until there is a real trigger.

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

## Observability

`yatima-lib` emits structured `tracing` spans and events; applications decide how
to collect them. The CLI installs a small `tracing-subscriber` layer driven by
`RUST_LOG`, but the library never installs a global subscriber. This keeps Yatima
embeddable: a service, TUI, notebook bridge, or Python wrapper can attach its
own subscriber without fighting the runtime.

The discipline is borrowed from Hyperactor:

- **Fields, not prose.** Use typed fields such as `arch = ?engine.arch()`,
  `backend = %engine.backend()`, `tool = %call.name`, `outcome = ?outcome`,
  `prompt_tokens`, and `prefill_chunk`. The message string names the event; the
  fields carry the data.
- **No accidental payload capture.** Instrumented functions should use
  `skip_all` or explicit spans/fields. Do not log prompts, generated text,
  tool arguments, SEC payloads, auth tokens, or whole structs at info level.
- **Spans for duration, events for facts.** `model.load`, `engine.generate`,
  prefill chunks, and `tool.call` are duration-bearing operations. `loaded
  model`, `generation finished`, `tool finished`, and `agent run finished` are
  facts about those operations.
- **Async spans must be attached to futures.** Do not hold a `span.enter()` guard
  across `.await`; use `#[tracing::instrument(skip_all, fields(...))]` or
  `future.instrument(span).await`.
- **Info stays sparse.** Per-chunk and per-tool-detail records are debug-level;
  per-token tracing is deferred until there is a concrete profiling need.
- **Perfetto later, same events.** Perfetto should arrive as a subscriber/layer
  over this field vocabulary, not as a second ad-hoc instrumentation system.

### Prefill scheduling

The first generation step can either feed the whole prompt to the model at once
or split it into bounded prefill chunks. The public knob is
`GenOpts::prefill_chunk`:

- `None` — use the model/backend default.
- `Some(0)` — force one full-prompt prefill.
- `Some(n)` — feed at most `n` prompt tokens per prefill forward pass.

Chunking does not change the logical prompt or the generated-token loop: each
chunk advances the same KV cache with the correct `start_pos`, and the final
prefill logits are taken from the last prompt token. It is therefore a scheduling
choice, not a prompt transformation.

Why it exists: GLM-4-32B GGUF on Metal was observed to produce incoherent output
on a long structured SEC research prompt when evaluated as one full prefill.
The same model, prompt, tokenizer, and sampler produced coherent output when the
prefill was bounded; forcing `prefill_chunk = 0` reproduced the bad output.
Source comparison with llama.cpp suggests this is the same broad concern that
llama.cpp avoids through bounded batch/ubatch evaluation rather than requiring a
single giant prompt prefill. Yatima therefore defaults GLM-4 GGUF on Metal to a
64-token prefill chunk, while preserving the explicit override for benchmarking,
diagnosis, and future Candle-side fixes.

**North star — a hylomorphism.** Generation *unfolds* a fragment stream (a
coalgebra deciding termination on EOS/max/break) and *folds* it with the caller's
`step` algebra: `generate = ana ; cata = hylo`. The hot loop stays imperative;
this is the denotation, not the implementation — the recursion-scheme reading is
what the (planned) Haskell study formalizes.

## Acting: capability-scoped tools & the agent loop

If `generate_with` folds *tokens* into a value, the agent loop folds *turns*: the
model emits a tool call, a capability-scoped tool runs, its result is fed back,
and the loop repeats until the model answers or `max_steps` is reached (the
hylomorphism nests one level up). Model turns are sequential, but tool execution
is async: a caller can await the result, watch lifecycle/progress events, join
the task, or request cooperative cancellation. The loop is still provable against
a *scripted* `Completer` with no GPU.

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
  de-facto standard) and runs `async call(args, ctx)`. `Tools::dispatch_async`
  **never hard-errors**: an unknown name (AGENT-2), invalid arguments,
  cancellation, or a tool failure becomes a typed `ToolOutcome`; `ToolResult` is
  only the model-facing projection the transcript receives (PROTO-1).
  `Tools::spawn` returns a `ToolTask` with `ToolEvent`s, `join`, and cooperative
  cancellation. Current tools: `ReadFile`, `ListDir`, `WriteFile`, `ReadUrl`,
  and `SendNotification`, each holding its own capability.
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
- **`Agent`** — `run_async` collects the final answer; `run_with_async` is the
  fold a future actor/TUI streams `AgentEvent`s into (`run_async` is the `acc =
  ()` specialization). Synchronous `run` / `run_with` wrappers remain for simple
  callers. Bounded by `max_steps`; `AgentStop` is `Final` / `MaxSteps` /
  `Stopped` (the last when the caller's fold returns `ControlFlow::Break`).

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
loop does GPU/CPU work with nothing to await. So the **generation core stays
synchronous**: an `async fn generate` would be a lie that blocks the executor and
starves other tasks. Concurrency belongs to *delivery*, not decode.

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

Tool execution is different: tools are effectful boundary calls (filesystem,
HTTP, notifications, future process/MCP adapters), so the tool core is async.
`ToolCtx` carries lifecycle emission and cooperative cancellation; `Tools::spawn`
is the join/watch/cancel surface, while the agent normally awaits each typed
`ToolOutcome`, projects it to a model-facing `ToolResult`, and then proceeds to
the next model turn.

**Deferred**: the durable engine-side shape is an **engine actor** owning the
`Engine` and serving `(prompt, opts, reply)` requests over a channel — also the
home for conversation/KV-cache reuse and cross-request scheduling. That actor can
compose with the async tool runtime, but it is not required to make tools
awaitable/watchable today.

### Runtime ownership & the sync↔async bridge (RT-1)

The library is **async-first** (the agent loop and tool dispatch are async) but
still offers thin **synchronous shims** for non-async embedders, and runs a
**synchronous inference core**. Rather than scatter `Handle::try_current` dances
and per-call runtimes, everything funnels through `runtime.rs`:

- **One owned multi-thread runtime** (a lazy `OnceLock<Runtime>`); the library
  never builds a runtime per call.
- **`block_on` — the single sync→async bridge.** Its ambient policy is explicit:
  no runtime → the owned runtime; a multi-thread runtime → `block_in_place` +
  the current handle (don't nest runtimes; keep other tasks progressing); a
  **current-thread runtime → panic with direction** ("call the async API"),
  because that case is genuinely unsupportable (no worker to hand off to; a
  nested `block_on` would deadlock). Every sync API (`Agent::run`,
  `Tools::dispatch`, `ensure_model_blocking`) is a one-line shim over its async
  primitive through this bridge.
- **`run_blocking` — the compute island.** Blocking work *whose result the async
  fn needs inline* (model inference inside the agent/chat loop) runs here:
  `block_in_place` on a multi-thread runtime so the executor stays live, a plain
  call otherwise. This is what un-paints the earlier half-paint, where the
  synchronous `complete()` sat directly in the async loop and starved tool
  watchers during generation.

`block_in_place` (not `spawn_blocking`) is correct for the inline case: the work
borrows `&mut Engine`/`&mut Completer`, which is neither `'static` nor `Send`, so
it can't be *moved* to a blocking pool — but it can relocate *other* tasks off
the worker. `spawn_blocking` + a bounded channel remains the right tool for the
*detached* streaming adapter sketched above (`generate_events`), where the work
is owned and the consumer wants a stream rather than an awaited result.

The binary edges (CLI + examples) are `#[tokio::main(flavor = "multi_thread")]`
and call the async APIs directly, so they dogfood the runtime rather than hide a
`block_on`; the sync shims exist for embedders that aren't async.

**The blocking island is type-enforced (RT-2).** "Inference must not block the
executor" is, stated literally, false — inference *does* block a thread. The
enforceable invariant is sharper: *local inference may be reached by an async
caller only through the runtime's blocking island.* We make that a compile-time
obligation with a borrow-scoped **capability witness**. `run_blocking_island`
mints a `BlockingIsland<'_>` (private field → unforgeable; HRTB lifetime → can't
escape the closure), and the `Engine` decode primitives (`complete_on`,
`complete_streaming_on`) *require* one. So `impl Completer for Engine` cannot be
written to decode on an async worker — the executor-stalling version does not
type-check. The low-level `generate_with` stays island-free on purpose: it is the
honest synchronous escape hatch for advanced callers. Types fix the *path*; the
operational *property* (liveness) rides on the multi-thread commitment and is
pinned by a liveness test (a ticker task progresses while a completion is in
flight). This is `Send`-gates-`spawn` for blocking: a witness that you are inside
the island gates the call, exactly as `Send` gates a cross-thread move.

### The model seam is async, `Send` inferred per impl (CMP-1)

`Completer` is the effect boundary: prompt → completion. It is an **async** trait
so it generalizes past local blocking compute to a **remote/HTTP model** that is
fundamentally async I/O. The non-obvious, deliberate choice is *how* it is async:
**native `async fn` in trait** (Rust ≥ 1.75), **not** the `async_trait` crate.

Why that matters: with native `async fn`, the returned future's `Send`-ness is
**inferred per implementation** rather than fixed at the trait. The local
`Engine` owns GPU handles (`Box<dyn CausalLm>`, no `Send` bound) so its
completion future is naturally `!Send`; a remote completer captures only `Send`
state so its future is naturally `Send`. We thereby avoid *both* bad global
choices: `?Send` (strips `Send` from every completer, penalising the remote
case) and forcing `Engine: Send` (a lie about the rented engine that buys
nothing — it is one-generation-at-a-time and `block_in_place`-pinned anyway). The
`Send` decision lives where the truth is: each impl.

Each impl also owns the **operational shape** of the effect. Native `async`
alone does not make candle inference non-blocking — so `Engine::complete` runs
its sync decode under `run_blocking`; a remote completer instead `.await`s
network I/O. Callers (`Agent`, `ChatSession`) just `.await` and assume nothing
about whether completion is CPU- or I/O-bound; the synchronous `turn`/`run`
shims bridge through `runtime::block_on`.

This is the principled counterpart to `Tool`, and the contrast is the point:

| trait | shape | stored as | concurrency | mechanism |
|-------|-------|-----------|-------------|-----------|
| `Tool` | `dyn`-compatible | `Arc<dyn Tool>` | `tokio::spawn`ed, watched, cancelled across tasks | `#[async_trait]` + `Send` |
| `Completer` | generic, monomorphic | `Agent<C>` / `ChatSession<C>` | awaited inline, one at a time | native `async fn`, `Send` per impl |

Tradeoff, accepted: a public native-`async fn` trait trips
`clippy::async_fn_in_trait` (you cannot name a `Send` bound on the method). We
`#[allow]` it *because* completions are never spawned, so imposing a global
`Send` bound would be wrong, not merely unnecessary. The day an engine-actor
needs to move a completion across threads, that is when to reach for
return-type-notation or `trait_variant` — not before.

## Registries

The **canonical** invariant & law registry lives in the crate docs — see the
`yatima-lib` crate doc and the `yatima-cli` `main.rs` doc. Each is protected by
a test that cites its id in an `// upholds: <id>` comment (`grep -r
'upholds:'`).

In brief: model store & discovery (**MS-1/2/3**, **MD-1/2/3**, **EOS-1**,
**FETCH-1**, dedup/order under **DISC**); generation (**SAM-1/2**, **STOP-1**,
**GEN-3**, **GE-1**); agent & tools (**AGENT-1/2**, **TOOL-1/2**, **CAP-1/2**,
**PROTO-1**); observability (**OBS-1/2/3/4**); chat templates
(**TMPL-1/2**); CLI (**CLI-1/2/3**).

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
- **Context-length handling — a staged, robust progression.** The multi-turn
  REPL (and the agent loop sooner, via tool results) grows the transcript until
  it exceeds the model's window; today **nothing checks**, so the failure is
  *silent* — RoPE positions degrade, then KV/position errors. Give the local path
  simple, robust behaviour built **down a ladder**, each rung shippable on its
  own and the earlier rungs sufficient for most use:
  1. **Know the budget** — `Engine::context_length()` (engine-native fact, same
     shape as `arch()` / `default_prefill_chunk()`; sourced from GGUF
     `<arch>.context_length` / safetensors `max_position_embeddings`). Not
     surfaced today; this is the prerequisite for everything below.
  2. **Detect + fail clean** — before a turn, `token_count(prompt) + max_tokens`
     vs `context_length`; a typed error / `StopReason`, never silent garbage.
     This rung alone fixes the real bug — it is a *correctness signal*, not a
     feature.
  3. **Trim (evict oldest)** — keep the system turn + the most-recent turns that
     fit. Deterministic and lossy; covers most chat pain.
  4. **Summarize (compaction)** — fold the evicted prefix into one synthetic turn
     *via the `Completer` itself* (the local analog of server-side compaction);
     heaviest and lossiest, gated on real long-session use.
  - Agent-specific rung: **elide old tool-result payloads** (keep "called
    `read_file` → ok", drop the body) — the analog of context-editing, cheaper
    than summarizing prose, since tool output is the actual bloat.

  Layering (LAYER-1): the *budget* is an engine fact (`context_length`); the
  *policy* lives in `ChatSession`/`Agent` — the engine never decides what to
  drop. Introduces **CTX-1** when built: a turn never sends more than
  `context_length` tokens — over-budget is trimmed/compacted/rejected, never
  silently truncated by the backend. Build rungs 1–2 soon (correctness); 3–4 on
  trigger. (`generate` one-shot and short chats don't need this; the `embed`
  classify loop `reset()`s per item, so it never grows.)
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
- **Richer capabilities** — process capabilities and tools, plus multi-tool-per
  turn / planning. `WriteDir`, `WebOrigin`, and `NtfyTopic` are now present as
  first slices of the broader capability model.
- **MCP edge adapter** — consume an MCP server *as* a `Tool`, or expose our tools
  *as* an MCP server (out-of-process; rides the same seams at the edge).

### Models / engine
- **Engine swappability** — cross-arch dispatch + GGUF/quantized + self-contained
  GGUF tokenizers are **done**. Remaining: sharded multi-file GGUF + more
  quantized arches; other backends (mistral.rs, llama.cpp) via more `Completer`
  impls.
- **Remote `Completer` (Anthropic / OpenAI)** — the payoff of the async-`Completer`
  generalization (CMP-1): a `RemoteCompleter` holds only `Send` state, so its
  future is naturally `Send` (per-impl inference, no `?Send`), and it **awaits
  HTTP directly** — no `run_blocking`, no `BlockingIsland` (RT-2 gates only the
  local sync decode). Rust has no official Anthropic SDK, so use raw `reqwest`
  (already a dep) against `POST /v1/messages`: headers `x-api-key` +
  `anthropic-version: 2023-06-01`; body `{model, max_tokens, system?, messages,
  stop_sequences?}`; model ids bare (`claude-opus-4-8`, …, no date suffix);
  `complete_streaming` reads SSE `content_block_delta` → `text_delta` onto
  `on_token`; budget via `POST /v1/messages/count_tokens` + response `usage`.
  Three real impedance points to design around, not gloss:
  1. **Stop sequences aren't echoed.** Our `Completer` contract *includes* the
     matched stop marker (so a `ToolCallCodec` sees `</tool_call>`); the Messages
     API omits it on `stop_reason: "stop_sequence"`. The impl must re-append it.
  2. **`temperature`/`top_p` are 400s on current Claude models.** `Sampling`
     doesn't forward 1:1 — drop it, or map "more/less thinking" onto `effort`.
  3. **The seam passes a flat prompt string; the chat API wants structured
     `messages`.** A first cut sends the rendered prompt as one `user` message
     (works, loses role structure). Doing it well is the seam refinement below.
  Also: handle `stop_reason: "refusal"` before reading content. A flat-prompt,
  greedy-only remote completer is buildable today with no seam surgery.
  **Scope decision (deliberate): a remote `Completer` is chat/generate, text
  only — no tools.** Our tools live in the `Agent` loop via a *text* codec gated
  to local tool-trained formats (Qwen/Plain); a hosted model's **native** tool
  use (`tool_use`/`tool_result` blocks, `stop_reason: "tool_use"`) cannot ride
  the flat-string `Completer` seam and would need a separate "agent backend"
  abstraction (most likely letting the provider run its own tool loop). That is
  explicitly **kicked down the road** — do not conflate it with the `Completer`.
- **More chat templates** — Llama-3, Zephyr/TinyLlama (same shape as Gemma/Mistral).
- **Sampling quality** — `top_p`/`top_k` nucleus sampling (only temperature
  today) for better free-text on smaller models; download integrity/resume.

### Embedding (library surface)
- **`ChatSession` — done.** The multi-turn fold is a public lib type
  (`ChatSession<C: Completer, T: PromptTemplate>`, borrows the completer like
  `Agent`), exercised by `lib/examples/embed.rs` (a real non-CLI consumer:
  conversation + model→`enum`→native branch) and dogfooded by the CLI's one-shot
  `chat`. This is what makes "language-integrated" a demonstrated fact.
- **Streaming `Completer` — done.** `Completer::complete_streaming` adds a
  token-callback variant with a default impl (emit the whole completion once, so
  every existing `Completer` keeps working); `Engine` overrides it to forward each
  decoded fragment as it arrives. `ChatSession::turn_streaming` streams a turn
  through it, and the CLI's interactive `chat` REPL now drives a `ChatSession`
  end-to-end (the hand-rolled loop is gone) — so the CLI fully dogfoods the
  library on both the one-shot and streaming paths.
- **A TUI chat app (sometime soon).** A "ChatGPT-style" terminal app to prove
  out and play with capabilities interactively — the next real consumer after
  the examples, and the natural **forcing function**: a long interactive session
  is exactly what exercises streaming chat end-to-end *and* surfaces the
  context-length ladder (rungs 1–2 become load-bearing the moment you chat past
  the window). It would also be where agent/tool capabilities get hands-on play.
  Drives demand for: the context-length rungs above, readline niceties, and
  eventually the engine-actor (if it grows beyond one session). Stays a pure
  embedder over `ChatSession`/`Agent` — no library changes it needs that aren't
  already wanted.

### Architecture / research
- **Invariant reviewer example — done.** `lib/examples/invariant_reviewer.rs`
  gathers bounded git/file/test-citation context in Rust and asks an in-process
  chat model for a cited report. This is the first "Yatima helps improve Yatima"
  slice: advisory, not patch-writing. Git is host-side evidence gathering here;
  a future interactive agent should get explicit `GitStatus`/`GitDiff`
  capabilities rather than ambient shell access.
- **Async engine-actor** owning the `Engine` and wrapping `run_with_async` — the
  home for cross-request concurrency and KV-cache reuse (see Concurrency).
- **Structured-message `Completer` seam** — `complete` takes a *rendered prompt
  string* today, which suits a local model fed one prompt but loses role
  structure for a hosted chat API (Anthropic/OpenAI take `messages[]`, not a
  flat string). A richer seam would hand the completer the structured turns
  (`&[Turn]`) and let each impl render: local impls run their `PromptTemplate`,
  a remote impl maps turns → API `messages`. Trigger: the remote `Completer`
  above — until then the flat string is fine. **Prerequisite done:** `Role`/`Turn`
  now live in a neutral `transcript` module (were in `agent`), so the seam would
  not make the model layer depend upward into the agent layer.
- **Serving & scale (deferred — mostly a different *process/tier*, not a
  module).** `yatima-lib` is the in-process **leaf**; serving concerns must not
  leak into it (no auth/tenancy/routing in `Engine`/`ChatSession`/`Agent`). The
  map, by where each concern lives:
  - *Node concurrency* (many clients, one model) → the **engine-actor** above
    (owns the `Engine`, serves requests over a channel; home of KV reuse). A
    request queue gives modest concurrency cheaply; real throughput needs
    continuous batching — a large engine build, probably **rented** (see fork).
  - *Tenancy / auth / quotas / multi-client API* → a separate **service tier**
    (e.g. `yatima-serve`) mapping tenant → capabilities → budget → model. Builds
    on the capability primitive (`Dir`/`WriteDir`/`WebOrigin`/`NtfyTopic` —
    bounded effects are exactly what isolation needs); never enters the library.
  - *Clusters / routing / model placement* → a **control-plane** tier above the
    service; the library is the leaf and knows nothing of it.
  - *Model sharding* → engine-internal, **rented from candle/backend**, surfaced
    as a load fact (like device/dtype), not a yatima subsystem.
  LAYER-1 extends one tier up: each serving tier is a new *edge* depending
  **down** into the library, never up. **Identity fork:** yatima as *embedded
  library / orchestrator* (scale by composing rented backends behind remote
  `Completer`s — vLLM / TGI / llama.cpp-server / cloud) **vs** yatima as *serving
  engine* (continuous batching — far larger, lower-level). The remote `Completer`
  is the escape hatch: front a real serving engine and let yatima orchestrate.
  The async runtime (RT-1/RT-2), `Send`-per-impl `Completer` (CMP-1), the
  engine-actor, and capability scoping are the substrate the service tier
  consumes — the serving tier is the first consumer that would pull the
  engine-actor off this list.
- **`lexicon` crate** — a shared, dependency-light home for `ModelId` + the
  `<root>/<org>/<name>` layout, extracted once there's a real trigger (possum
  validating its own ids, or a second consumer).
- **The Haskell study** — formalize the `generate` and agent contracts
  (GE/STOP/AGENT/CAP/PROTO/TMPL as propositions); the planned home for the
  recursion-scheme reading and `axiom`.
