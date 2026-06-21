<p align="center">
  <img src="./images/logo.png" width="340" alt="yatima logo">
</p>
<h1 align="center">yatima</h1>
<p align="center">
  language-integrated llms
</p>
<p align="center">
  <a href="https://github.com/shayne-fletcher/yatima/actions/workflows/build-and-test.yml">
    <img src="https://github.com/shayne-fletcher/yatima/actions/workflows/build-and-test.yml/badge.svg" alt="rust ci">
  </a>
  <a href="https://shayne-fletcher.github.io/yatima/">
    <img src="https://img.shields.io/badge/docs-github.io-blue" alt="docs">
  </a>
</p>

`yatima` is a Rust runtime for language-integrated LLMs — calling a local model
as an ordinary in-process function, and letting it *act* through tools whose
authority is bounded by construction. It's a building block to compose freely,
not a fixed product shape: weights are acquired by
[`possum`](https://github.com/shayne-fletcher/possum) and loaded by `yatima`,
which owns the runtime (loading, generation, and capability-scoped tools) while
renting the inference engine ([candle](https://github.com/huggingface/candle)).

## What it does

Three layers, increasing in capability:

- **`generate`** — raw completion. `Engine::{load, generate, generate_with}` runs
  a stateless loop streaming decoded fragments to a fold; the contract is stated
  as laws and protected by tests. The loader dispatches on architecture — Qwen2,
  Llama, Mistral, Phi-3, Gemma-2, StarCoder2, GLM-4 — and loads **GGUF/quantized**
  weights too (self-contained: the tokenizer is built from the GGUF), so a 32B
  runs on a Mac.
- **`chat`** — instruction-following, no tools. Applies a model's native chat
  template (`--format qwen|gemma|mistral|glm|plain`) so an instruct model behaves
  as trained. Works for any instruct model.
- **`agent`** — a tool loop for **tool-trained** models (currently Qwen/ChatML).

**Capability matrix** (what each model can do):

| Model family      | generate | chat  | agent/tools |
|-------------------|----------|-------|-------------|
| Qwen2.5-Instruct  | yes      | yes   | yes         |
| GLM-4 (9B / 32B)  | yes      | yes   | no          |
| Gemma-2-it        | yes      | yes   | no          |
| Mistral-v0.3      | yes      | yes   | later/complex |
| TinyLlama-chat    | yes      | yes   | no          |
| StarCoder2        | yes      | maybe | no          |
- **Capability-scoped tools** — a tool *holds* its authority (e.g. a `Dir`
  rooted at one directory); the file tools cannot reach outside it, a tool not in
  the agent's set is uncallable, and a malformed or unknown call becomes an error
  the model can recover from — never a silent mis-execution.
- **An agent loop** — `Agent::run_async` / `run_with_async` fold model turns:
  the model emits a tool call, a capability-scoped async tool runs, the result is
  fed back, bounded by `max_steps`. Tool calls are awaitable, joinable,
  watchable, and cooperatively cancellable; the synchronous `run` wrappers remain
  for simple callers.

Kindred in spirit to Anil Madhavapeddy's
[*Language Integrated LLMs as an OCaml Function*](https://anil.recoil.org/notes/language-integrated-llms).

## Building

```bash
cargo build                            # build
cargo test                             # the whole suite (no GPU needed)
cargo run --bin yatima -- --help       # explore the CLI
```

## Try it

The most compact demo is an agent that can read only the directory you grant it,
then asks a local Qwen-format model to summarize this README:

```bash
cargo run -p yatima-cli --release --bin yatima --features metal -- agent \
  --model ~/.cache/yatima/models/bartowski/Qwen2.5-32B-Instruct-GGUF \
  --format qwen \
  --root . \
  --prompt "Read README.md and summarize what yatima is in three sentences." \
  --max-tokens 256
```

Expected shape:

```text
loaded .../Qwen2.5-32B-Instruct-GGUF [metal/F32]; tools rooted at .
Yatima is a Rust runtime designed for language-integrated LLMs, allowing local
models to be called as in-process functions. ...
[1 steps, Final]
```

That one command exercises the core path: local model load, an agent turn, a
capability-scoped `read_file` tool call under `--root`, and a grounded final
answer.

The library tool set now includes read/list directory tools, a separate
`WriteDir`-scoped `write_file`, `WebOrigin`-scoped `read_url`, and
`NtfyTopic`-scoped `send_notification`. Each tool holds its own authority; the
model supplies only arguments within that authority.

Tools execute as async tasks. A caller can use `Tools::dispatch_async` when it
only needs the result, or `Tools::spawn` to receive `ToolEvent`s, join the task,
and request cooperative cancellation. This is the intended foundation for TUIs,
supervising agents, long-running downloads, process tools, and network effects.

There is also an opt-in live test for the notification tool. Subscribe your
phone to an ntfy topic first, then run:

```bash
YATIMA_NTFY_TOPIC=we-could-be-coding-haskell \
  cargo test -p yatima-lib e2e_send_notification_to_phone -- --ignored
```

The normal test suite never publishes to ntfy.sh; this ignored test exists so
you can prove the `NtfyTopic` capability and `send_notification` tool end to end.

```bash
# generate a completion (raw)
cargo run -p yatima-cli --release --bin yatima --features metal -- generate \
  --repo deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
  --prompt "Rust is"

# chat — one-shot (applies the model's chat template)
cargo run -p yatima-cli --release --bin yatima --features metal -- chat \
  --repo google/gemma-2-2b-it --format gemma \
  --prompt "Explain Rust in two sentences."

# chat — interactive multi-turn (omit --prompt); /exit quits, /reset clears
cargo run -p yatima-cli --release --bin yatima --features metal -- chat \
  --repo Qwen/Qwen2.5-7B-Instruct --format qwen

# run an agent with read-only file tools scoped to a directory
cargo run -p yatima-cli --release --bin yatima --features metal -- agent \
  --model ~/.cache/yatima/models/bartowski/Qwen2.5-32B-Instruct-GGUF \
  --format qwen \
  --root . \
  --prompt "What's in README.md?" \
  --verbose
```

A missing model is fetched on demand (the `fetch` feature) via `possum`;
`--offline` never touches the network.

### Prefill chunking

`yatima generate`, `yatima chat`, and `yatima agent` accept
`--prefill-chunk <n>` to bound how many prompt tokens are evaluated in one
prefill step. Omit it for the model/backend default; use `--prefill-chunk 0` to
force one full-prompt prefill.

This is mostly invisible, but it matters for some large quantized Metal models.
GLM-4 GGUF on Metal defaults to a 64-token prefill chunk because full-prompt
prefill on the 32B GGUF path can destabilize generation on long structured
prompts, while bounded prefill preserves the same KV-cache semantics and keeps
output coherent. The override is there for benchmarking and diagnosis.

For a cheap reproducer that compares next-token logits without running a long
generation, see [`notes/glm4-prefill-reproducer.md`](notes/glm4-prefill-reproducer.md)
and `cargo run -p yatima-lib --release --example prefill_compare --features metal`.

## Embedding

The CLI is just one consumer — `yatima-lib` is meant to be called *as a library*,
the model as an ordinary in-process function woven into your own control flow:

```rust
use yatima_lib::{ChatSession, ChatMlTemplate, Engine, device};

let mut engine = Engine::load(model_dir, device(false)?)?;
let mut chat = ChatSession::new(&mut engine, ChatMlTemplate).with_system("Be brief.");
let answer = chat.turn("My name is Ada.")?;        // remembers across turns
let recall = chat.turn("What is my name?")?;       // -> "Your name is Ada."

// …or stream the reply token-by-token for a live UI:
chat.turn_streaming("Tell me a joke.", &mut |piece| print!("{piece}"))?;
```

Because it's in-process, model output flows straight into native code — e.g. ask
for a label and `match` on a Rust `enum`, no serialization or service boundary.
See [`lib/examples/embed.rs`](lib/examples/embed.rs) (a conversation **and** a
classify-then-branch triage loop): `cargo run --example embed --features metal`.

## Auditable research

The investment-research example is the most complete demonstration of the
library shape: Rust resolves a ticker through SEC EDGAR, fetches public XBRL
company facts, normalizes them into cited evidence records, embeds a local chat
model, asks for a concise research note, then audits the generated thesis against
the evidence it supplied.

```bash
SEC_USER_AGENT="your-name your-email@example.com" \
  cargo run -p yatima-lib --release --example investment_thesis --features metal -- AAPL
```

The example defaults to the local Qwen 32B GGUF path used in the agent demo; pass
a model directory as the second argument to override it. The generated note is
not investment advice. It is a grounded-output demo: every factual claim is
expected to cite the SEC accession, filing period/date, and XBRL tag it came
from. A small example-local validator warns when the model cites unknown tags or
accessions, drifts from the normalized `value_text`, omits citation fields on a
quantity-bearing line, or uses trend language when only one period was supplied.
For GLM-4 GGUF, pass `glm` as the third argument; an optional fourth argument
sets the prefill chunk (`0` means full-prompt prefill).

## Notes

- [Design](notes/design.md) — the `generate` contract, model storage and
  auto-fetch, the capability-scoped tool and agent layer, and the concurrency
  model.
