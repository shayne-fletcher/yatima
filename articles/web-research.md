# Web research

Yatima can search the web, ask you which source origins to trust, read approved pages, and display images those pages list. The model chooses what to inspect; Yatima retains authority and records the provenance of displayed images.

## Configure search

Set one search provider before starting the GUI, TUI, browser server, CLI agent, or scenario driver:

```bash
# A SearXNG-compatible JSON endpoint. This takes precedence when both are set.
export YATIMA_SEARCH_URL=https://search.example.org/search

# Or Brave Search. The key is sent only to Brave's fixed API endpoint.
export YATIMA_BRAVE_KEY=your-subscription-key
```

With neither variable set, `web_search` is absent. Page reading still works for origins you grant directly.

## Interactive flow

Ask a tool-capable model to find something rather than supplying a URL. A typical run is:

1. The model calls `web_search` and receives numbered results.
2. It names the best source origins. The host emits a typed proposal rather than asking each view to parse prose.
3. In the GUI or browser, approve origins individually or click **grant all**. In the TUI, enter `/grant N` or `/grant all`.
4. Grants are sent one at a time. When the whole proposed set has landed, Yatima retries the original question once.
5. The model reads a source by its result number. Long pages are cached and exposed in adjacent windows rather than fetched repeatedly.
6. If the approved page lists images, the model selects them by list number. Each rendered image keeps its number, label, source URL, and derivation page.

Search discovers URLs but grants no reading authority. Only the user can add an origin. An approved page may authorize the exact public images in its current listing, including images on a separate CDN, but it cannot grant that CDN or authorize arbitrary links. Revoking the source page also removes authority derived from it.

The CLI agent is one-shot and has no proposal controls. Pre-grant the source origin with `--web-origin` when using it for page reading. Search results are stable only for that process.

## Record a GUI session

Pass `--tape` to record the GUI's host requests and events:

```bash
cargo run -p yatima-gui --release -- \
  --profile muse-glimmer --offline --tape
```

With no directory after `--tape`, the run is written under `runs/<timestamp>-<pid>-gui/`. `tape.jsonl` holds ordered protocol records, images live under `artifacts/`, and `summary.json` closes a clean run with counts and time to first artifact.

## Reproduce a session headlessly

`yatima-drive` submits one GUI-style input per nonempty line and always records a tape. Blank lines and `#` comments are ignored; `/grant` and `/revoke` are the only scenario commands. The repository includes `scenarios/mandelbrot.scenario`, which preserves the original three-turn image investigation.

```bash
cargo run -p yatima-drive --release -- \
  --profile muse-glimmer --offline \
  --tape runs/mandelbrot \
  scenarios/mandelbrot.scenario
```

That preserved scenario includes the explicit image-host grant used during the original investigation. New scenarios do not need one when `read_image` selects an exact public image from an approved page's listing.

The driver prints the tape directory on stdout and exits nonzero for startup failure, turn error, timeout, fatal host failure, interruption, or incomplete evidence. It uses the same host, tools, grants, model profile, and joined shutdown path as the interactive frontends.

## Related guides

- [The TUI](tui.md) covers keys, streaming, and grant commands.
- [The browser viewer](browser-viewer.md) covers the WebSocket bridge and WASM client.
- [Tools and capabilities](capabilities.md) states the authority boundaries.
- [Architecture](architecture.md) traces the shared host path.
