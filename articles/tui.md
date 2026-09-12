# The TUI

`yatima-tui` is an interactive session over a local model: streaming chat, foldable reasoning, and, on tool-trained formats, a web-capable agent whose authority you grant at runtime.

```bash
cargo run -p yatima-tui --release --features metal -- --profile qwen32b
```

Configure `YATIMA_SEARCH_URL` or `YATIMA_BRAVE_KEY` to add web search. A session still starts with **zero page-reading authority**: search discovers sources but does not approve them.

## Granting web access (CAP-3)

Only you can grant an origin:

- **Type a URL.** `summarize https://en.wikipedia.org/wiki/Roger_Penrose`
  auto-grants `https://en.wikipedia.org` for the session before the turn runs —
  a visible `◆ granted read access…` notice lands in the transcript and the
  status rail shows `web:en.wikipedia.org`.
- **Approve a proposal.** After a search, Yatima displays numbered source origins. Use **`/grant N`** to approve them individually or **`/grant all`** to approve the set. Once every proposed origin has landed, Yatima retries the original question once.
- **`/grant <origin>`** is the explicit form; **`/grants`** lists the set; **`/revoke <origin>`** shrinks it.

Grants are origin-scoped (`https://en.wikipedia.org`, with the path stripped), accumulate for the session, and never persist across sessions. A page may authorize only the exact public images in its current image listing, including images hosted on another origin; it cannot grant that origin or authorize arbitrary links. `/reset` clears the conversation but keeps grants because capability state is separate from conversation state.

On a tool-capable format the sessionful agent serves from turn one. Before the first grant, page readers stay hidden; configured search remains available because discovery needs no page authority. A grant changes the tools' authority, not the session mode. On chat-only formats, grants are refused because tool calling needs a tool-capable format.

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
| Ctrl+C twice | quit (Ctrl+D stays an editing key: delete-char) |
| `/reset` | clear the conversation (grants survive) |
| `/grant <origin>` · `/grant N` · `/grant all` · `/grants` · `/revoke <origin>` | manage web authority and proposals |

The transcript speakers are your login name and `yatima`; the bottom rail carries the machine facts: profile, backend, chat format, context meter, and granted origins. See [Web research](web-research.md) for search setup and the full authority flow.
