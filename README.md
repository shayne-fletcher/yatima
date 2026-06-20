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
  Llama, Mistral, Phi-3, Gemma-2, StarCoder2 — and loads **GGUF/quantized**
  weights too, so a 32B-Q4 runs on a Mac.
- **`chat`** — instruction-following, no tools. Applies a model's native chat
  template (`--format qwen|gemma|mistral|plain`) so an instruct model behaves as
  trained. Works for any instruct model.
- **`agent`** — a tool loop for **tool-trained** models (currently Qwen/ChatML).

**Capability matrix** (what each model can do):

| Model family      | generate | chat  | agent/tools |
|-------------------|----------|-------|-------------|
| Qwen2.5-Instruct  | yes      | yes   | yes         |
| Gemma-2-it        | yes      | yes   | no          |
| Mistral-v0.3      | yes      | yes   | later/complex |
| TinyLlama-chat    | yes      | yes   | no          |
| StarCoder2        | yes      | maybe | no          |
- **Capability-scoped tools** — a tool *holds* its authority (e.g. a `Dir`
  rooted at one directory); the file tools cannot reach outside it, a tool not in
  the agent's set is uncallable, and a malformed or unknown call becomes an error
  the model can recover from — never a silent mis-execution.
- **An agent loop** — `Agent::run` / `run_with` fold model turns: the model emits
  a tool call, a capability-scoped tool runs, the result is fed back, bounded by
  `max_steps`. The loop is synchronous and provable against a scripted model with
  no GPU.

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

## Notes

- [Design](notes/design.md) — the `generate` contract, model storage and
  auto-fetch, the capability-scoped tool and agent layer, and the concurrency
  model.
