# Architecture

The `Completer` trait is the model-facing boundary. The in-process Candle `Engine` and the HTTP `LlamaServerCompleter` both implement it. `chat` adds prompt templates and transcript memory; `agent` adds capability-scoped tools and typed outcomes.

Candle generation is synchronous and compute-bound: each token depends on the previous token, and the `Engine` owns mutable model state. The llama-server implementation waits asynchronously on a supervised local process. `yatima-host` gives both the same execution shape: one dedicated backend thread owns the selected backend and the authoritative session, while frontends exchange requests, events, and cancellation signals with it.

The TUI, native GUI, and browser are views over that host. They render the same `HostEvent` stream and never own a second model session. The browser reaches it through one WebSocket provided by `yatima-serve`; [the browser viewer](browser-viewer.md) follows that path from startup through shutdown.

## Streaming agent steps (AGENT-4)

Each agent step drives the completer's streaming path. Fragments are
classified live — chain-of-thought onto a reasoning channel via a per-step
splitter (REASON-1 holds mid-stream), prose onto the answer channel through an
**opener gate** that withholds tool-call markup: text is buffered while its
tail could still become the codec's open marker, suppressed once the marker
completes (the parsed call arrives as a `ToolCall` event instead), and
released as ordinary prose when a lookalike diverges. The final step's answer
fragments concatenate to the run's answer; a step that turns out to be a tool
call marks its streamed prose as narration, which the TUI retracts from the
answer pane and replays as reasoning. Cancellation is token-level on both the
chat and agent paths: a fold `Break` or an external `Cancel` stops the decode
at the next token, and an interrupted run persists nothing (AGENT-3).

The agent is sessionful (AGENT-3): completed exchanges persist their user turn
and final answer; tool rounds and reasoning are ephemeral to their run. In the
TUI, sessions start on the plain streaming chat path and the first origin
grant transplants the chat history into the agent — both histories are
user/answer turns, so the switch is invisible.

## Diagnostics

`yatima-lib` emits structured `tracing` fields; `yatima-cli` installs the
subscriber:

```bash
RUST_LOG=yatima_lib=debug,yatima_cli=info \
  cargo run -p yatima-cli --release --bin yatima -- chat ...
```

The library does not log prompts, generated text, tool arguments, or fetched
payloads at info level (agent step prompts/completions are available at trace
level for forensics). Perfetto support should layer over the same structured
events later.

## Further reading

- The serve/web browser viewer — components, the wire, and the reconnect
  seam: [articles/browser-viewer.md](browser-viewer.md).
- The full invariant registry, state machines, model-loading contract,
  concurrency discussion, and deferred work: [notes/design.md](../notes/design.md).
- The Metal KV-depth corruption investigation, workaround, and upgrade canary:
  [notes/metal-kv-cliff.md](../notes/metal-kv-cliff.md).
- The GLM-4 GGUF Metal prefill investigation and reproducer:
  [notes/glm4-prefill-reproducer.md](../notes/glm4-prefill-reproducer.md).
