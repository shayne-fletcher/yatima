# Architecture

`generate_with` is the primitive: an effectful fold over decoded text fragments.
`chat` adds native prompt templates and transcript memory. `agent` adds a
tool-call protocol, capability-scoped async tools, and typed outcomes.

Generation itself remains synchronous and compute-bound: token `n + 1` depends on
token `n`, and the current `Engine` owns mutable model state. Async belongs at the
effect and UX boundaries: tool calls, the TUI engine actor, future service/Python
wrappers. The TUI runs decode on a dedicated engine thread behind a three-plane
protocol (request / event / control), so the UI loop never blocks on decode and a
turn is cancellable in flight.

## Diagnostics

`yatima-lib` emits structured `tracing` fields; `yatima-cli` installs the
subscriber:

```bash
RUST_LOG=yatima_lib=debug,yatima_cli=info \
  cargo run -p yatima-cli --release --bin yatima -- chat ...
```

The library does not log prompts, generated text, tool arguments, or fetched
payloads at info level. Perfetto support should layer over the same structured
events later.

## Further reading

- The full invariant registry, state machines, model-loading contract,
  concurrency discussion, and deferred work: [notes/design.md](../notes/design.md).
- The GLM-4 GGUF Metal prefill investigation and reproducer:
  [notes/glm4-prefill-reproducer.md](../notes/glm4-prefill-reproducer.md).
