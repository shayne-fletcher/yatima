# Vision log

Dated entries, newest last: the durable *why* of yatima's direction.
design.md holds the *how* — architecture, laws, the roadmap — and each
roadmap idea born from a vision entry points back here for provenance.
People appear by role, never by name (house convention). yatima's
earliest vision entries predate this file and live in sieve's log
(sieve `notes/vision.md`, 2026-07-03/05): serve as the event plane over
a websocket, every client a viewer, and the surfaces ladder whose rung
2 the wasm spike built.

## 2026-07-12 — rung 2 demonstrated: a phone on the tailnet

The spike's demo sentence was performed today: a phone browser tab on
the tailnet streamed a live qwen32b turn — fragments, a tool-rendered
sine chart fitted to the screen, an input box that submits, a stop that
cancels. The client is an eframe/wasm build over `yatima-protocol`
alone; serve carried the event plane; the model ran a formula through
the python plot tool to a PNG that crossed the wire as bytes and became
a texture on a phone. Every layer of the last month's preparation — the
protocol extraction, the host split, the capability doctrine — was load
bearing and none of it had to move.

Shared outside the project, the demo drew two things from a
compilers/PL friend. First an endorsement of the direction itself:
"text UIs are fun but so fundamentally limiting" — the same conviction
that opened the GPU-frontend line in the roadmap, now arriving from
outside unprompted. Second, a protocol idea recorded in the roadmap
(design.md, "Speculative decoding over the wire"): a featherweight
local drafter speculating for a remote verifier that batch-confirms to
token N and corrects token N+1, with grammar filtering stacked on top
for codegen — the dual of the remote-`Completer` fork, landing exactly
on the serve seam. The spike's first external dividend: the surface
invited a collaborator to design *with* it, which is what a surface is
for.

What the day taught, spike-style (the findings are recorded in the
serve roadmap entry, design.md): reconnect semantics are the first
thing a phone tests — idle tabs drop, zombie sockets answer keepalives
on behalf of frozen pages, and SRV-3's single-client refusal read as
"server confused" from the outside. Preemption — newest connection
wins — shipped as the amendment the same day, and the daily case closed
as a pair: a reconnect button that preserves the in-memory mirror, plus
a stop that settles the turn locally (the escape hatch a dropped Done
would otherwise wedge). Model temperament is a real variable: QwQ
tutorializes where qwen32b calls the tool. And a debug wasm bundle is
21 MB — release builds before anyone else is invited (measured since:
4.8 MB with wasm-opt).
