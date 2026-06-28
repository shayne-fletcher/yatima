# Embedding

The CLI is just one consumer. `yatima-lib` is meant to be called as a library,
with model output flowing into native Rust values and branches:

```rust
use yatima_lib::{device, ChatMlTemplate, ChatSession, Engine};

let mut engine = Engine::load(model_dir, device(false)?)?;
let mut chat = ChatSession::new(&mut engine, ChatMlTemplate).with_system("Be brief.");

let answer = chat.turn("My name is Ada.")?;
let recall = chat.turn("What is my name?")?;

chat.turn_streaming("Tell me a joke.", &mut |piece| print!("{piece}"))?;
```

yatima is async-first: the agent loop and tool dispatch are async, and inference
runs as a synchronous compute island that never stalls the executor. From an
async program, call the `_async` APIs (`turn_async`, `Agent::run_async`,
`Tools::dispatch_async`) directly; the synchronous methods above are thin shims
over them for non-async embedders.

```rust
// inside #[tokio::main(flavor = "multi_thread")]
let answer = chat.turn_async("My name is Ada.").await?;
```

See [`lib/examples/embed.rs`](../lib/examples/embed.rs) for a conversation and a
classify-then-branch triage loop:

```bash
cargo run -p yatima-lib --release --example embed --features metal
```

## Invariant reviewer

yatima can inspect its own invariant discipline. The invariant reviewer example
gathers the crate registries, `upholds:` test citations, changed files, and a
bounded git diff, then asks a local chat model for a cited review:

```bash
cargo run -p yatima-lib --release --example invariant_reviewer --features metal -- \
  --profile mistral \
  --max-tokens 450 \
  --max-prompt-tokens 2500
```

To include the current working-tree diff, keep the evidence bounded:

```bash
cargo run -p yatima-lib --release --example invariant_reviewer --features metal -- \
  --profile mistral \
  --diff \
  --max-tokens 350 \
  --max-prompt-tokens 4000 \
  --max-file-bytes 1200 \
  --max-diff-bytes 5000 \
  --upholds-limit 8
```

For Qwen 32B on Metal, keep the first pass smaller and chunk prefill:

```bash
cargo run -p yatima-lib --release --example invariant_reviewer --features metal -- \
  --profile qwen32b \
  --max-tokens 350 \
  --max-prompt-tokens 2200 \
  --max-file-bytes 3000 \
  --upholds-limit 16 \
  --prefill-chunk 64
```

This is deliberately advisory: the model does not edit files. It produces a short
report about missing invariants, weak tests, wrong citations, or docs drift for a
human to judge.
