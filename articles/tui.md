# The TUI

`yatima-tui` is an interactive session over a local model: streaming chat,
foldable reasoning, and — on tool-trained formats — a web-capable agent whose
authority you grant at runtime, mid-conversation.

```bash
cargo run -p yatima-tui --release --features metal -- --profile qwen32b
```

No web flags exist. A session starts with **zero** web authority.

## Granting web access (CAP-3)

Web authority derives only from *your* utterances:

- **Type a URL.** `summarize https://en.wikipedia.org/wiki/Roger_Penrose`
  auto-grants `https://en.wikipedia.org` for the session before the turn runs —
  a visible `◆ granted read access…` notice lands in the transcript and the
  status rail shows `web:en.wikipedia.org`.
- **`/grant <origin>`** is the explicit form; **`/grants`** lists the set;
  **`/revoke <origin>`** shrinks it.

Grants are origin-scoped (`https://en.wikipedia.org`, path stripped — the path
is the model's business at call time, the origin is the authority), accumulate
for the session (the rail shows `web:N origins`), and never persist across
sessions. Crucially, a URL the model *encounters* — in a fetched page, in its
own output — grants nothing: there is no code path from content to authority.
`/reset` clears the conversation but keeps grants; capability is not
conversation.

The first grant switches the session from the plain chat path to the
tool-calling agent, carrying the conversation across invisibly. On chat-only
formats (e.g. the reasoning profiles), grants are refused with a clear
message — tool calling needs a tool-trained format (`qwen` or `plain`).

## What a tool turn looks like (AGENT-4)

Everything streams. During a turn you see, live:

- the model's chain-of-thought and tool activity in the **reasoning fold**
  (`▾ reasoning (live)`) — `⚙ read_page {"url": …}` when a call dispatches,
  `✓ 12141 chars` when it lands;
- the answer, token by token, in the answer area — tool-call markup never
  appears there (an opener gate withholds it);
- an honest activity bar: `answering… · 0:39 · 55 tok · 1.4 tok/s`.

If the model narrates before calling a tool ("Let me fetch that…"), the prose
streams into the answer area and then *retracts* into the reasoning fold when
the call dispatches — narration is working matter, not answer.

**Esc cancels at the next token**, mid-decode, on both the chat and agent
paths. An interrupted exchange leaves no trace in session memory (AGENT-3) —
just re-ask.

## Long pages (FETCH-1 / WIN-1)

`read_page` returns long articles one ~12k-char window at a time; each window's
truncation marker names the offset for the next
(`[chars 0..12000 of 54765; call read_page again with offset=12000 …]`), and
the model follows it unprompted when the question warrants. Continuations are
served from a fetch-once cache: **a URL is fetched over the network at most
once per session** — re-reads and follow-up windows are instant and free, which
is also exactly the discipline a rate-limited host (SEC EDGAR) demands.

Session memory is deliberately lean (AGENT-3): a completed exchange persists
your question and the final answer; the tool windows themselves are ephemeral
to their turn. A follow-up ("what prize did he win?") answers from memory when
the earlier answer contains it, and re-reads from cache — not the network —
when it doesn't.

## Keys and commands

| key / command | effect |
|---|---|
| Enter | submit (Alt+Enter / Shift+Enter: newline) |
| Esc | cancel the in-flight turn (token-level) |
| Ctrl+R | expand/collapse completed turns' reasoning |
| ↑ / ↓ | recall prior prompts (shell-style) |
| PgUp / PgDn | scroll the transcript |
| Ctrl+C / Ctrl+D | quit |
| `/reset` | clear the conversation (grants survive) |
| `/grant <origin>` · `/grants` · `/revoke <origin>` | manage web authority |

The transcript speakers are your login name and `yatima`; the bottom rail
carries the machine facts — profile, backend, chat format, context meter,
granted origins.
