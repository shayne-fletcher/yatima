# Tools & capabilities

Tools hold their authority. A `ReadFile` tool constructed with a `Dir` can only read under that root; `WriteFile` uses a separate `WriteDir`; `ReadUrl` and `ReadPage` share a growable set of HTTP(S) origins; and `SendNotification` is scoped to a pre-shared `NtfyTopic`. `WebSearch` discovers numbered sources but grants no authority. `ReadImage` can use an explicit origin grant or the exact listing derived from an approved page. The model supplies arguments, not authority (CAP-2). `Tool` is public and `Tools::with` takes any `impl Tool`, so a consumer crate can register its own domain tools.

## Runtime grants (CAP-3 / CAP-4)

Only the user can add an origin to a session's `WebOrigins`: type a URL, approve a proposed origin, or issue an explicit grant command. Grants accumulate for the session, never persist across sessions, and shrink only by explicit revoke. Search results and model output cannot grant an origin.

A successfully read page may confer narrower authority: `read_image {"image": N}` may fetch the exact public image URL published in that page's current image listing, even when the image is hosted elsewhere. This does not grant the image host or permit navigation there. The authority disappears when the source page is revoked or its listing is replaced. Private, loopback, and otherwise non-public derived targets are refused; a user can still grant a local origin explicitly.

The prompt always states the model's live origin grants (CAP-3a). A web reader with no applicable authority is omitted from the advertised tool specs. Membership is checked before network I/O, and every redirect is checked again. A relative URL resolves only when exactly one origin is granted.

For the complete interactive workflow, see [Web research](web-research.md).

## Fetch-once pagination (FETCH-1 / WIN-1)

`read_page` reads long articles one window at a time. Windows tile exactly:
each truncation marker names the next window's `offset`, and continuation
calls are served from a per-tool, FIFO-bounded cache — **one network fetch per
URL per session**. Re-fetching is the expensive act for throttled hosts (SEC
EDGAR's request budget); re-reading is free. Downstream tools inherit the
contract and refine the addressing (a filings tool wants "Item 1A", not a
blind offset).

## Observable async execution

Tool execution is async and observable. Runtime code sees a typed
`ToolOutcome` algebra; the model sees only the projected `ToolResult` turn
(PROTO-1: a malformed call becomes a recoverable error turn, never a silent
mis-execution). A caller can use `Tools::dispatch_async` for a result, or
`Tools::spawn` to watch `ToolEvent`s, join the task, and request cooperative
cancellation.

## Live notification test

There is an opt-in live test for the notification tool. Subscribe your phone to
an ntfy topic first, then run:

```bash
YATIMA_NTFY_TOPIC=we-could-be-coding-haskell \
  cargo test -p yatima-lib e2e_send_notification_to_phone -- --ignored
```

The normal test suite never publishes to ntfy.sh.
