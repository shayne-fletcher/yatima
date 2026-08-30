# The browser viewer

`yatima-serve` runs the native host and bridges its protocol to a browser. `yatima-web` is the WASM client served to that browser. The browser is a view of the host session, not a second model runtime.

```bash
# Build the WASM client.
cd web && trunk build --release
cd ..

# Verify Muse, start llama-server, and serve the browser client.
cargo run -p yatima-serve --release -- \
  --profile muse-glimmer --offline \
  --bind 127.0.0.1:8787 --static-dir web/dist
```

Open `http://127.0.0.1:8787/`. The page loads the WASM app, which opens one WebSocket back to `/ws` on the same origin. To use another device, bind to that machine's specific tailnet address instead. There is no default bind, and wildcard addresses are refused (`SRV-1`), so network exposure must be explicit.

## One fold, one unfold

The shape underneath is a single duality. The host *unfolds*: from its
session state it produces the `HostEvent` stream, one event at a time.
The client *folds*: it consumes that stream into a `Transcript` mirror
it renders. `Transcript::fold` is the fold's step; the loop in
`drain_socket` is the fold itself; the socket is the tape between them.
Everything else in these two crates exists to keep that tape honest
across disconnects.

## The nouns, host side (`yatima-serve`)

The `yatima-serve` binary draws nothing. Its `main` function retains the one `HostOwner`, while `Bridge` receives only the movable fields of `HostClient`. That split lets the bridge carry browser traffic without giving a socket task ownership of the backend or its managed child.

- **`Bridge`** — the shared state behind the router: the host's request sender, its `CancelGate`, the one event stream, the takeover signal, and the closing signal.
- **`EventStream`** — the host's event receiver plus a one-deep **carry
  slot** (`pending`): the last event a session *attempted* to send. A
  buffered `socket.send` is not proof the peer read the frame, so on
  handoff that one event rides to the next session and is delivered
  first — at-least-once at the seam, never a hole (SRV-3).
- **`StreamLease`** — how a session borrows the stream. Its `Drop`
  restores the stream to the bridge, so a failed upgrade or a panicking
  session cannot strand it: one holder at every instant, enforced by
  ownership rather than discipline.
- **The host connection** — requests flow from browser to host over a `std::sync::mpsc` sender, events flow back over a `tokio::sync::mpsc` receiver, and cancellation uses the out-of-band `CancelGate`. A wire `Cancel` bypasses the request queue, which is not serviced while a model turn runs, so stop can take effect during generation.

On Ctrl-C or SIGTERM, `Bridge::close` refuses new WebSocket upgrades and asks a live session to return its `StreamLease`. Axum drains under a bound; only then does `main` consume `HostOwner::shutdown()` and wait for the backend thread and any managed `llama-server` child. A server error follows the same close-and-join path.

## The nouns, on the wire (SRV-2)

The wire is exactly the `yatima-protocol` enums as externally-tagged
JSON: every `HostEvent` is one text frame out; every inbound text frame
is a `HostRequest`. serve defines no message types of its own, so a
client that speaks the protocol crate speaks serve by construction —
which is why the browser client could be a miniature rather than a
port.

## The nouns, browser side (`yatima-web`)

The crate splits along the browser line so the subtle half stays
testable without a browser:

- **`Transcript`** (`lib.rs` — plain Rust, unit-tested natively) — the
  mirror: committed `entries` plus the turn in flight. The live turn is
  one sum, `Turn { Idle, Live { id, answer, reasoning } }`, so "a
  streaming buffer without a live turn" — the state behind the wedged
  spinner the first phone demo found — cannot be constructed
  (WEB-3/4/5 are structural over it).
- **`BackendState`** — the private startup state: `Loading`, `Ready(ModelInfo)`, or `Failed`. Startup events name model resolution, digest verification, and backend launch. Only `Ready` enables input, and only `VerifiedSha256` earns a `verified:<digest prefix>` label.
- **`Entry`** — committed history: `User`, `Assistant` (answer plus the
  reasoning fold), `Image(DecodedImage)`, `Note`, `Error`. Artifact
  bytes decode to raw RGBA in the model; only the view makes textures.
  A format this build doesn't decode renders as a named placeholder
  line, never an error (WEB-6).
- **`WebApp`** (`main.rs` — compiles only for wasm32) — the thin egui
  view: one socket, `drain_socket` folding frames, the status line, the
  input row, the transcript scroll. It renders the mirror; it holds no
  truth of its own (WEB-1).

## The verbs: one turn, end to end

Before a turn, the host emits typed startup phases and then `Ready(ModelInfo)`. The browser folds those events into `BackendState`; an open socket alone does not enable input. Once connected and ready, pressing send stamps a client-local `turn_id`, records the user line in the mirror, arms `Turn::Live`, and sends `Submit { turn_id, text }` as one JSON frame. Serve forwards it to the host, and every returned `HostEvent` becomes one frame. Fragments append to the answer or reasoning text, tool notes remain separate, images become textures, and `Done` commits a nonempty answer and disarms the turn.

```mermaid
sequenceDiagram
    participant B as browser (yatima-web)
    participant S as serve (Bridge)
    participant H as host (backend thread)
    B->>S: Submit {turn_id, text} — one JSON frame
    S->>H: HostRequest over the request plane
    H-->>S: Started · Fragment* · ToolNote* · Image* · Done
    S-->>B: each HostEvent as one frame (event plane)
    Note over B: Transcript::fold per frame — the client's fold
    B->>S: Cancel {turn_id} — stop
    S->>H: CancelGate.cancel(turn_id) — trips the gate mid-decode
```

## Web authority from a browser (CAP-3)

Grants work exactly as in the TUI and GUI — authority derives only from
*your* utterances — with one relocation: the browser client is
protocol-only and cannot scan for origins itself (`origins_in` lives in
`yatima-lib`, which never compiles to wasm), so **serve, the browser's
native edge, owns the auto-grant**. Type a URL in your message and the
bridge grants its origin before the turn runs; `/grant <origin>`,
`/grants`, and `/revoke <origin>` are the explicit forms, parsed
client-side into the protocol's requests. Grant reports come back as
muted notes in the transcript. A URL the model encounters still grants
nothing — there is no code path from content to authority.

## The seam: what a phone actually tests

Reconnect semantics are the first thing a phone exercises — idle tabs
drop, and a frozen tab's network process keeps answering protocol pings
on its behalf. Three behaviors make the seam honest:

- **Preemption (SRV-3): the newest connection wins.** A second
  connection signals the live session (the watch counter — an edge per
  bump, re-bumped each poll round so a session mid-send cannot miss
  it); the session yields at its next await; the handshake completes
  holding the same stream, carry slot intact. Refusing would protect a
  zombie socket over a live human; 409 survives only as the
  takeover-deadline fallback. A session is also always *able* to yield:
  every await is capped (the send stall cap) or paced (the keepalive
  ping), so a half-open peer is reaped rather than holding the stream.
- **At-least-once at the handoff.** Events emitted while nobody is
  connected wait in the channel; the one event already pulled rides the
  carry slot to the next session and is delivered first. A viewer
  tolerates a repeated final fragment far better than a hole.
- **The client absorbs the seam.** Any turn activity arms the mirror on
  demand, so a client that attaches mid-turn renders it running though
  it never saw `Started` (WEB-3). A stale `Done` cannot disarm a newer
  turn (WEB-4). And stop settles locally — commit what streamed, disarm
  now — so a `Done` lost at the seam can never wedge the spinner
  (WEB-5). The reconnect button swaps the dead socket and keeps the
  mirror and its textures; a browser refresh would wipe them.

A reconnect within the same app keeps the transcript and resumes the event stream. A full page reload creates a new mirror. Until `plans/web-replay.plan.md` adds state replay, a page loaded after another client already consumed `Ready` honestly remains in `loading…` with input disabled; it does not invent backend state.

## Laws

The canonical registries live in the crate docs, each id cited by a
test (`grep -rn 'upholds:'`): **SRV-1/2/3** in `serve/src/lib.rs`,
**WEB-1..7** in `web/src/lib.rs`, the host planes (**HOST-1..5**) in
`host/src/lib.rs`, and the wire (**PROTO-2**, **WASM-1**) in
`protocol/src/lib.rs`. This article narrates them; the crate docs
define them.
