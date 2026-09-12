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

The agent is sessionful (AGENT-3): completed exchanges persist their user turn and final answer; tool rounds and reasoning are ephemeral to their run. Tool-capable formats use this path from the first turn. Tools whose authority is empty stay out of the prompt; configured search is available immediately because it discovers sources without granting them.

When the non-final tool-step budget is exhausted, the agent offers the model one final answer-only completion. No tool is advertised or dispatched in that reserve round. This preserves termination without turning a long but useful research run into silence.

## Web research

`WebSearch` discovers sources through a configured SearXNG endpoint or Brave Search. It publishes stable numbered results into a session registry shared by `read_page` and `read_url`, so the model can select `{"result": N}` without copying a URL. Selection is only addressing: the result's origin must still be granted.

After a turn asks for new origins, the host emits one typed `GrantProposal`. The GUI and browser render buttons; the TUI accepts `/grant N` or `/grant all`. Frontends serialize multiple grants and retry the original prompt once after the whole proposed set lands. The host reports every grant and revoke to the model on its next prompt, so model state does not depend on interpreting the UI.

An approved page's current image listing may derive authority for each exact public image it names. The derivation records the source page, survives a cross-origin image host, and dies when the source grant or listing dies. It never inserts the image host into the origin set.

The `yatima-drive` binary runs a line-oriented scenario against the same host and records requests, events, timings, and image artifacts through `yatima-drive`'s asynchronous flight recorder. This makes live failures reproducible without creating a second host path. See [Web research](web-research.md) for usage.

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
