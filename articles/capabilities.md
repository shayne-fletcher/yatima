# Tools & capabilities

Tools hold their authority. A `ReadFile` tool constructed with a `Dir` can only
read under that root; `WriteFile` uses a separate `WriteDir`; `ReadUrl` (raw body)
and `ReadPage` (readable main article from HTML) are both scoped to a `WebOrigin`;
`SendNotification` is scoped to a pre-shared `NtfyTopic`. The model supplies
arguments, not authority. `Tool` is public and `Tools::with` takes any `impl
Tool`, so a consumer crate can register its own domain tools.

Tool execution is async and observable. Runtime code sees a typed `ToolOutcome`
algebra; the model sees only the projected `ToolResult` turn. A caller can use
`Tools::dispatch_async` for a result, or `Tools::spawn` to watch `ToolEvent`s,
join the task, and request cooperative cancellation.

## Live notification test

There is an opt-in live test for the notification tool. Subscribe your phone to
an ntfy topic first, then run:

```bash
YATIMA_NTFY_TOPIC=we-could-be-coding-haskell \
  cargo test -p yatima-lib e2e_send_notification_to_phone -- --ignored
```

The normal test suite never publishes to ntfy.sh.
