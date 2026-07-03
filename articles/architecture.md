# Architecture

`generate_with` is the primitive: an effectful fold over decoded text fragments.
`chat` adds native prompt templates and transcript memory. `agent` adds a
tool-call protocol, capability-scoped async tools, and typed outcomes — and is
itself a fold, one level up: over turns instead of tokens.

Generation itself remains synchronous and compute-bound: token `n + 1` depends on
token `n`, and the current `Engine` owns mutable model state. Async belongs at the
effect and UX boundaries: tool calls, the TUI engine actor, future service/Python
wrappers. The TUI runs decode on a dedicated engine thread behind a three-plane
protocol (request / event / control), so the UI loop never blocks on decode and a
turn is cancellable in flight.

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

- The full invariant registry, state machines, model-loading contract,
  concurrency discussion, and deferred work: [notes/design.md](../notes/design.md).
- The Metal KV-depth corruption investigation, workaround, and upgrade canary:
  [notes/metal-kv-cliff.md](../notes/metal-kv-cliff.md).
- The GLM-4 GGUF Metal prefill investigation and reproducer:
  [notes/glm4-prefill-reproducer.md](../notes/glm4-prefill-reproducer.md).
