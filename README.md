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

`yatima` is a Rust runtime for using local LLMs inside typed programs. It loads
models in-process, renders the right chat template for each family, and lets
tool-trained models act only through explicit capabilities.

The point is not to wrap an LLM service. The point is to make model calls part
of ordinary Rust control flow: fetch public evidence, normalize it into typed
values, ask a local model for a thesis or decision, then validate what it said
against the data your program actually supplied.

Weights are acquired by [`possum`](https://github.com/shayne-fletcher/possum)
and loaded by `yatima`. `yatima` owns the runtime surface — loading, generation,
chat, tools, capabilities, examples — while renting the inference engine from
[candle](https://github.com/huggingface/candle).

Kindred in spirit to Anil Madhavapeddy's
[*Language Integrated LLMs as an OCaml Function*](https://anil.recoil.org/notes/language-integrated-llms).

## What You Can Do

- **Generate** raw completions with local safetensors or GGUF/quantized weights.
- **Chat** with instruction-tuned models using their native prompt templates.
- **Run agents** where tool-trained models act through capability-scoped tools.
- **Embed** the runtime in Rust programs instead of crossing a service boundary.
- **Audit outputs** by keeping public evidence, model claims, and validation in
  the same typed program.

Supported loading/generation covers Qwen2, Llama, Mistral, Phi-3, Gemma-2,
StarCoder2, and GLM-4, with GGUF support for quantized Qwen2, Llama, and GLM-4.
The agent/tool path is narrower by design: it requires a model trained to emit
tool calls, and today the practical default is Qwen/ChatML.

| Model family      | generate | chat  | agent/tools |
|-------------------|----------|-------|-------------|
| Qwen2.5-Instruct  | yes      | yes   | yes         |
| GLM-4 (9B / 32B)  | yes      | yes   | no          |
| Gemma-2-it        | yes      | yes   | no          |
| Mistral-v0.3      | yes      | yes   | later       |
| TinyLlama-chat    | yes      | yes   | no          |
| StarCoder2        | yes      | maybe | no          |

## Try It

Build and test:

```bash
cargo build
cargo test
cargo run --bin yatima -- --help
```

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

Other useful commands:

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

## Auditable Research

The investment-research example is the clearest demonstration of why this shape
is interesting. Rust resolves a ticker through SEC EDGAR, fetches public XBRL
company facts, normalizes them into cited evidence records, embeds a local chat
model, asks for a concise research note, then audits the generated thesis against
the evidence it supplied.

```bash
SEC_USER_AGENT="your-name your-email@example.com" \
  cargo run -p yatima-lib --release --example investment_thesis --features metal -- \
    --ticker AAPL
```

Compare several models on the same evidence:

```bash
SEC_USER_AGENT="yatima research example shayne@shayne-fletcher.org" \
cargo run -p yatima-lib --release --example investment_thesis --features metal -- \
  --ticker META \
  --compare qwen32b,gemma2,mistral \
  --temperature 0 \
  --max-tokens 900
```

The generated note is not investment advice. It is a grounded-output demo:
quantity-bearing claims are expected to cite the SEC accession, period/filed
date, and XBRL tag they came from. The example-local validator warns when the
model cites unknown tags or accessions, drifts from normalized `value_text`,
omits citation fields, or uses trend language when only one period was supplied.

The insider-watchlist example looks for a different kind of stock-selection
signal: recent Form 4 open-market insider purchases. Rust resolves tickers,
fetches SEC submissions and filing-directory indexes, parses ownership XML,
keeps only non-derivative `P` purchases with acquired/disposed code `A`, scores
the typed transactions mechanically, then optionally asks a local model to rank
the watchlist from that evidence.

For a reliable live demo, fetch one known SEC filing, save the normalized
evidence, and run the model in the same pass:

```bash
SEC_USER_AGENT="your-name your-email@example.com" \
cargo run -p yatima-lib --release --example insider_watchlist --features metal -- \
  --filing ENR:1632790:0001140361-26-026118:form4.xml:2026-06-23 \
  --sec-delay-ms 2000 \
  --sec-max-requests 2 \
  --save-evidence /tmp/insider-watchlist-enr.json \
  --profile mistral \
  --max-tokens 600
```

Replay the captured evidence to compare models without touching SEC:

```bash
cargo run -p yatima-lib --release --example insider_watchlist --features metal -- \
  --evidence /tmp/insider-watchlist-enr.json \
  --profile qwen32b \
  --prefill-chunk 64 \
  --max-tokens 600
```

The example routes SEC traffic through a metered client: `--sec-delay-ms`
defaults to 1000, `--sec-max-requests` defaults to 25, and HTTP 429 stops the
run immediately rather than retrying into a longer block. The model-facing prompt
requires citations back to ticker, owner, accession, transaction date, shares,
price, and normalized `value_text`; the validator warns on unknown cited
accessions.

## Embedding

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

See [`lib/examples/embed.rs`](lib/examples/embed.rs) for a conversation and a
classify-then-branch triage loop:

```bash
cargo run -p yatima-lib --release --example embed --features metal
```

Yatima can also inspect its own invariant discipline. The invariant reviewer
example gathers the crate registries, `upholds:` test citations, changed files,
and a bounded git diff, then asks a local chat model for a cited review:

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

This is deliberately advisory: the model does not edit files. It produces a
short report about missing invariants, weak tests, wrong citations, or docs drift
for a human to judge.

## Tools And Capabilities

Tools hold their authority. A `ReadFile` tool constructed with a `Dir` can only
read under that root; `WriteFile` uses a separate `WriteDir`; `ReadUrl` is scoped
to a `WebOrigin`; `SendNotification` is scoped to a pre-shared `NtfyTopic`.
The model supplies arguments, not authority.

Tool execution is async and observable. Runtime code sees a typed `ToolOutcome`
algebra; the model sees only the projected `ToolResult` turn. A caller can use
`Tools::dispatch_async` for a result, or `Tools::spawn` to watch `ToolEvent`s,
join the task, and request cooperative cancellation.

There is an opt-in live test for the notification tool. Subscribe your phone to
an ntfy topic first, then run:

```bash
YATIMA_NTFY_TOPIC=we-could-be-coding-haskell \
  cargo test -p yatima-lib e2e_send_notification_to_phone -- --ignored
```

The normal test suite never publishes to ntfy.sh.

## Architecture Notes

`generate_with` is the primitive: an effectful fold over decoded text fragments.
`chat` adds native prompt templates and transcript memory. `agent` adds a
tool-call protocol, capability-scoped async tools, and typed outcomes.

Generation itself remains synchronous and compute-bound: token `n + 1` depends
on token `n`, and the current `Engine` owns mutable model state. Async belongs at
the effect and UX boundaries: tool calls, future service/TUI/Python wrappers,
and an eventual engine actor that owns the blocking kernel.

`yatima-lib` emits structured `tracing` fields; `yatima-cli` installs the
subscriber. For diagnostics:

```bash
RUST_LOG=yatima_lib=debug,yatima_cli=info \
  cargo run -p yatima-cli --release --bin yatima -- chat ...
```

The library does not log prompts, generated text, tool arguments, or fetched
payloads at info level. Perfetto support should layer over the same structured
events later.

For the full invariant registry, state machines, model-loading contract,
concurrency discussion, and deferred work, see [notes/design.md](notes/design.md).

For the GLM-4 GGUF Metal prefill investigation and reproducer, see
[notes/glm4-prefill-reproducer.md](notes/glm4-prefill-reproducer.md).
