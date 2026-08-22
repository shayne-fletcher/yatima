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

## Muse Glimmer through an attached llama-server

The current llama-server backend is an attached development path: start `llama-server` yourself, then point `yatima chat` at it. Managed startup is planned but not implemented yet.

From the repository root, start a named tmux session:

```bash
tmux new-session -s yatima-glimmer -c "$PWD"
```

In the first pane, launch the server with the exact model used by the stage-0 proof:

```bash
MODEL="$HOME/.cache/yatima/models/meta-models/Muse-Glimmer-30B-GGUF/Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf"

llama-server \
  -m "$MODEL" \
  --jinja \
  -np 1 \
  -c 131072 \
  --host 127.0.0.1 \
  --port 8080
```

Type `Ctrl-b %` to open a second pane. Wait for readiness, then inspect what the server reports:

```bash
until curl -fsS http://127.0.0.1:8080/health >/dev/null; do sleep 1; done

curl -fsS http://127.0.0.1:8080/props | jq '{
  model_path,
  model_ftype,
  context: .default_generation_settings.n_ctx,
  total_slots,
  build_info
}'
```

For the recorded artifact, `build_info` is `b10520-cd644c395`, the context is `131072`, and `total_slots` is `1`. Check the local file separately:

```bash
MODEL="$HOME/.cache/yatima/models/meta-models/Muse-Glimmer-30B-GGUF/Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf"
shasum -a 256 "$MODEL"
```

The expected SHA-256 is:

```text
4cc57c0f51040a226e5a72cc47b7613f7772950e460a665f7083de89f183f60e
```

Start the interactive Yatima chat in the second pane:

```bash
cargo run -p yatima-cli --release -- chat \
  --backend llama-server \
  --server-url http://127.0.0.1:8080 \
  --format muse-glimmer \
  --max-tokens 2048 \
  --temperature 1.0 \
  --top-p 0.95 \
  --seed 7
```

Use `/reset` to clear the conversation and `/exit` to leave. Up and Down recall prompts entered during the current run; `/reset` clears the model conversation but leaves that input history available. Stage 1 still displays Muse's raw ATEM framing while generation is in progress; final transcript commits are interpreted.

In tmux, `Ctrl-b o` changes panes. To scroll, type `Ctrl-b [`, use Page Up, Page Down, or the arrow keys, then type `q` to leave copy mode. After `/exit`, change to the server pane and type `Ctrl-c`; wait for the shell prompt before closing the session. This orderly stop matters because the attached path does not own or reap the server process.

### What the checks establish

The file digest proves the bytes at `MODEL`. `/props` helps catch an accidental wrong process, model path, build, context, or template. Neither a generated statement such as "I am Muse Glimmer" nor `/props` authenticates an arbitrary attached process: both are claims made by that process, and the digest does not prove that it loaded the file you checked. Treat attached mode as operator-attested and unverified. The planned managed path closes this gap by verifying the profile-pinned artifact, passing that exact path to a child Yatima owns, and refusing prompts when its compatibility checks fail.
