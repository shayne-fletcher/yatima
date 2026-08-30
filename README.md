<p align="center">
  <img src="./images/yatima-2.png" width="340" alt="yatima logo">
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

`yatima` is a Rust runtime for local LLMs. Models run in-process through [Candle](https://github.com/huggingface/candle) or through a supervised [llama-server](https://github.com/ggml-org/llama.cpp).

Yatima keeps model calls in ordinary Rust control flow. It owns prompts, transcripts, tool protocols, and capability checks regardless of which backend generates the text.

<p align="center">
  <img src="./images/yatima-penrose.png" width="820" alt="yatima-tui mid-turn: a typed URL auto-granted its origin, the read_page tool ran, and the answer is streaming live at 1.4 tok/s">
</p>

That is one unstaged turn. The URL granted its origin for the session, the model fetched the page through a capability-scoped tool, and the answer streamed live. Fetched content cannot grant new authority.

## Quickstart

```bash
# Interactive TUI. Type a URL to grant its origin.
cargo run -p yatima-tui --release --features metal -- --profile qwen32b

# Verified Muse agent. read_file is confined to --root.
cargo run -p yatima-cli --release -- agent \
  --profile muse-glimmer --offline --root . \
  --prompt "Read README.md and explain what Yatima is."
```

Managed mode requires `llama-server` on `PATH`. Yatima verifies the model, starts the server on loopback, and reaps it on success, failure, or Ctrl-C. Missing weights are fetched through [`possum`](https://github.com/shayne-fletcher/possum); `--offline` disables network access.

## What it does

- Run local models through Candle or llama.cpp behind one `Completer` interface.
- Use Yatima through the CLI, TUI, native egui app, browser viewer, or embedded in Rust.
- Chat and run capability-scoped tools. Muse Glimmer's native ATEM protocol works through the CLI, TUI, native GUI, and browser viewer.
- Give tools explicit authority such as a directory or a set of web origins.
- Stream reasoning, answers, and tool status without leaking protocol markup.

The crate registry names the important guarantees, and tests cite the laws they witness.

## Documentation

See the [articles](articles/README.md) for user and developer guides. See the [design notes](notes/design.md) for invariants and implementation rationale.

## License

BSD-3-Clause. See [LICENSE](LICENSE).
