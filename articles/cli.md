# CLI usage

The `yatima` binary (crate `yatima-cli`) exposes `generate`, `chat`, and `agent`.
Build with `--features metal` on Apple Silicon.

```bash
cargo build
cargo test
cargo run --bin yatima -- --help
```

## Agent with capability-scoped tools

Ask a local Qwen-format model to read only the directory you grant it and
summarize this README:

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

That command exercises the core path: local model load, prompt rendering, an
agent turn, a capability-scoped `read_file` tool call under `--root`, and a
grounded final answer.

The CLI agent also takes `--web-origin <url>` to pre-grant one HTTP(S) origin
for a one-shot run — the batch shape. For interactive work prefer the
[TUI](tui.md), where web authority is granted at runtime by simply typing a
URL (CAP-3), grants accumulate across the session, and long pages stream and
paginate live.

## Generate and chat

```bash
# raw completion
cargo run -p yatima-cli --release --bin yatima --features metal -- generate \
  --repo deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
  --prompt "Rust is"

# one-shot chat with a chat-only instruct model
cargo run -p yatima-cli --release --bin yatima --features metal -- chat \
  --repo google/gemma-2-2b-it --format gemma \
  --prompt "Explain Rust in two sentences."

# interactive multi-turn chat; /exit quits, /reset clears history
cargo run -p yatima-cli --release --bin yatima --features metal -- chat \
  --repo Qwen/Qwen2.5-7B-Instruct --format qwen
```

A missing model is fetched on demand when the `fetch` feature is enabled;
`--offline` never touches the network.
