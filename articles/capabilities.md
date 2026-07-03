# Tools & capabilities

Tools hold their authority. A `ReadFile` tool constructed with a `Dir` can only
read under that root; `WriteFile` uses a separate `WriteDir`; `ReadUrl` (raw
body) and `ReadPage` (readable main article from HTML) share a `WebOrigins` —
a growable **set** of HTTP(S) origins; `SendNotification` is scoped to a
pre-shared `NtfyTopic`. The model supplies arguments, not authority (CAP-2).
`Tool` is public and `Tools::with` takes any `impl Tool`, so a consumer crate
can register its own domain tools.

## Runtime grants (CAP-3)

Web authority is granted at runtime, and only by **user utterances**: an
origin enters a session's `WebOrigins` when the user types a URL (the host
scans user-typed text only) or issues an explicit grant command. Grants
accumulate — session authority is the union — never persist across sessions,
and shrink only by explicit revoke. Nothing a tool returns or the model
generates can reach `WebOrigins::grant`: no such code path exists, so a
malicious page cannot mint authority.

The prompt always states the model's live authority (CAP-3a): a web tool with
an empty origin set omits itself from the advertised tool specs — the model
never sees a tool it cannot use — and once granted, the tool's description
enumerates its origins. Membership is checked at call time; an out-of-set URL
is refused before any network I/O, and a relative URL resolves only when
exactly one origin is granted.

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
