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

- **Inference as a function** — `Engine::{load, generate, generate_with}` runs a
  stateless, raw-completion loop, streaming decoded fragments to a fold. The
  generation contract is stated as laws and protected by tests.
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

```bash
# generate a completion
yatima generate --repo deepseek-ai/DeepSeek-R1-Distill-Qwen-7B --prompt "Rust is"

# run an agent with read-only file tools scoped to a directory
yatima agent --repo deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
  --root ./docs --prompt "What's in README.md?" --verbose
```

A missing model is fetched on demand (the `fetch` feature) via `possum`;
`--offline` never touches the network.

## Notes

- [Design](notes/design.md) — the `generate` contract, model storage and
  auto-fetch, the capability-scoped tool and agent layer, and the concurrency
  model.
