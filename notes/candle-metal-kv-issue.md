# Metal: deterministic corruption in quantized generation once the KV
# cache reaches depth 8,192 (evidence points at a missed synchronization)

*(draft for filing against huggingface/candle — title above, body below)*

## Summary

Quantized GGUF generation on Metal degrades into deterministic garbage
the moment the KV cache reaches 8,192 entries — whether the crossing
happens during prefill or mid-decode. Extensive bisection (details
below) exonerates the math, the allocator, RoPE, and tokenization, and
pins the rescue to forced GPU synchronization at the crossing step:
tiny per-layer readbacks around depth 8,192 make the corruption vanish
entirely. The evidence points at a dependency that escapes candle's own
fence tracking (all Metal buffers run `HazardTrackingModeUntracked`
with a buffer→last-writer fence map); under Metal's fixed encoder
schedule the stale read is misread the *same way every run* —
deterministic, yet a synchronization defect.

## Environment

- candle main at 31f35b1 (0.11.0), which already includes the recent
  Metal sync fixes #3532, #3595, #3394 — the defect is live on top of
  them
- macOS 26, Apple Silicon (M-series, 48 GB)
- `quantized_qwen2` (GGUF, Q4_K_M), observed with
  Qwen2.5-32B-Instruct-Q4_K_M (8 KV heads × head dim 128; the f32 KV
  cache reaches exactly 2^25 bytes at depth 8,192 — see below)

## Reproduction

Greedy (`--temperature 0` — the example's default is sampled), so
byte-for-byte deterministic. Any prompt that takes the templated token
count past 8,192 shows it. **`--split-prompt` is required at this
length**: without it the example runs the whole prompt through a
single forward, and each layer then materializes a full f32
attention-scores tensor (`[1, 40, n, n]` ≈ 10 GB at n ≈ 8,400, with
masked and softmaxed copies alive alongside it) — transiently tens of
GB on top of ~19 GB of weights, enough to force a 48 GB machine into
an unrecoverable swap spiral. Single-token prefill is memory-flat and
still crosses the cliff (chunk size is immaterial — see below), at the
cost of prompt processing taking on the order of ten minutes for this
model.

```bash
MODEL=/path/to/Qwen2.5-32B-Instruct-Q4_K_M.gguf
# 8,414 tokens after the example's chat template — past the 8,192 cliff.
# (Not named PROMPT: in an interactive zsh that variable is the shell
# prompt itself, and themes rewrite it under you.)
TEXT=$(python3 -c 'print("Summarize this: " + "lorem ipsum " * 4200)')
cargo run --release --features metal --example quantized-qwen2-instruct -- \
  --model "$MODEL" --prompt "$TEXT" --split-prompt \
  --sample-len 60 --temperature 0
```

(For the decode-side crossing instead, `* 4000` — 8,014 tokens after
the template — with `--sample-len 400` derails partway through
generation.)

Two onset modes, both landing at the same depth:

- prompt ≥ 8,192 tokens: garbage from the first sampled token (with a
  repeat penalty the garbage argmax lands on low-frequency added tokens
  like `</tool_call>`; without it, naked degeneration such as
  `000퓮퓮퓮…`);
- shorter prompt, long decode: fluent text until the cache crosses
  8,192, then word-salad. Measured onsets: an 8,100-token prompt
  derailed at generated token 93 (kv 8,193); an 8,018-token prompt at
  generated token ~175 (kv ≈ 8,193), reproducibly across runs. Past the
  crossing many sampled tokens are UTF-8 fragments an incremental
  detokenizer never completes.

The same prompts on CPU are clean, and identical scaffolds with the
filler cut to ≤ 8,100 tokens are clean on Metal — the bracket was
tightened to (8,100 … 8,300) with pinned-token prompts, and
decode-crossing runs put the onset at kv 8,192–8,194.

## What it is not (each exonerated by direct experiment)

- **Not content or prompt structure**: identical scaffold, filler cut
  to N tokens, flips clean→garbage between 8,100 and 8,300 regardless
  of content.
- **Not tokenization**: the markup tokens involved are added tokens,
  encoded as single pieces.
- **Not prefill scheduling**: prefill chunk sizes 64/32/16 all fail
  identically.
- **Not the attention ops in isolation**: a standalone probe compares
  q·kᵀ, masked/unmasked softmax, probs·v, and `cat` on Metal vs CPU at
  kv ∈ {8,000 … 11,605} for both decode and prefill-chunk shapes — max
  |Δ| ≈ 1e-7 everywhere.
- **Not RoPE**: table values probed to position 32,767 and the rope op
  probed with offset views and contiguous copies to 16,384 — clean.
  (The GGUF path builds its tables in f32, so this is distinct from
  the bf16 RoPE-table issue in the safetensors `models/qwen2.rs`,
  #3520.)
- **Not the buffer pool/allocator**: at depth 8,192 the KV `cat` is
  served from a long-idle 32 MiB power-of-two bucket, which made the
  pool the leading suspect — but a scoped kill-switch (fresh buffers
  for exact power-of-two requests ≥ 16 MiB, allocation logging
  confirming zero reuse at the crossing) produced output
  **byte-identical** to the baseline: same clean prefix, same
  derailment, same garbage. Fresh buffers do not move the cliff.
- **Not a data-path provenance gap**: a replica probe replays one
  decode step exactly as `quantized_qwen2::forward_attn` does (4D batch
  shapes, 8-head KV cache grown by `cat`, `repeat_kv` cat + reshape,
  RoPE through narrowed table views at the true positions, no mask) at
  positions 8,189–8,194 — every intermediate agrees across devices at
  float epsilon.

## Why depth 8,192

It is the unique power-of-two crossing for this shape: 8 KV heads ×
8,192 positions × 128 dim × 4 bytes = 32 MiB = 2^25 bytes per K/V
cache tensor (4,096 bytes per position). Deeper water shows a second
failure regime past 16,384 (2^26) — see "Limits" below.

## The pinning experiment: forced syncs make it vanish

Instrumenting the real model settled it. Per-layer stat readbacks —
each a forced GPU sync — around the crossing make the corruption
disappear, and bisection narrowed the rescue to almost nothing:

- syncs at positions 8,186–8,200, all 64 layers, 3 taps each: clean;
- syncs at positions 8,191–8,192 only: clean;
- one tap (layer output) only: clean;
- **layers 0 and 1 only — four readbacks total: clean**, the entire
  120-token generation coherent through the exact byte where the
  unsynced baseline derails.

So the poison forms once, early in the crossing step, and an early
flush re-phases the encoder stream so everything downstream computes
correctly. "Byte-identical across allocation regimes ⇒ not a race" is
a red herring: with every buffer `HazardTrackingModeUntracked` and
candle substituting its own fence map (buffer pointer → last-writer
fence, fed by `Output::new` annotations), a dependency that escapes
that map is misread the same way every run under a fixed encoder
schedule — deterministic, yet a synchronization defect. The exact
escaping edge (plausibly tied to a kernel/dispatch change at dimension
8,192) is what needs finding; the fence system is where to look.

## Workaround (available as a branch)

https://github.com/shayne-fletcher/candle/tree/yatima-0.11.0-metal-kv-sync
(upstream 0.11.0 plus one commit): `quantized_qwen2::forward` calls
`device.synchronize()` during any forward whose KV positions extend
past 8,192 — after layers 0 and 1 for decode steps (the
bisection-proven minimum), after every layer for prefill chunks
(prefill-side crossings empirically need the wider net; the two-layer
variant fails there with a distinct, *nondeterministic*
punctuation-soup mode). Shallow contexts are untouched. Debug
instrumentation (env-gated layer stats, a scoped pool kill-switch,
allocation logging) rides the same branch.

## Limits of the workaround (may help localize the defect)

The syncs restore correctness to moderate depth only: coherent through
a ~15.5k-token prefill in the field, but an ~18.2k-token prefill —
first to cross 16,384, where the cache passes 2^26 bytes — is garbage
(`!!!!…`) even with a sync after **every** layer of every forward past
8,192. Whatever breaks in the deepest water is not interruptible by
these syncs, which suggests the same escaping dependency has a second,
harder manifestation at the next power-of-two crossing.

## Offer

The standalone probes used above (attention-op device comparison at
depth, RoPE table/op probes, the forward-replica probe, and the
allocation-logging / no-reuse kill-switch) are available and can be
PR'd or shared if useful.
