# Metal KV corruption at depth 8,192 (CTX-2)

## Symptom

Quantized GGUF generation on Metal degrades into deterministic garbage the
moment the KV cache reaches 8,192 entries — however that depth is reached:

- A prompt of ≥ 8,192 tokens produces garbage from the first sampled token
  (with the default 1.1 repeat penalty the garbage argmax lands on tokens
  like `</tool_call>`, which a tool-call codec then misreads as an answer;
  with the penalty off it is naked degeneration, `000퓮퓮퓮…`, halted by the
  degeneration guard).
- A shorter prompt with a long decode derails mid-generation exactly when
  the cache crosses 8,192: an 8,100-token prompt produced fluent text
  through generated token 93 (kv 8,193) then word-salad; an 8,018-token
  prompt derailed at generated token ~175 (kv ≈ 8,193) across two
  independent runs. Past the crossing many sampled tokens are UTF-8
  fragments the incremental detokenizer never completes, so a host sees the
  stream stall as well as corrupt.

First observed as "the model answered `</tool_call>`" in the TUI acceptance
run for HTTP tools (Qwen2.5-32B-Instruct Q4_K_M, 11.6k-token step-2 prompt
carrying a 40k-char `read_page` result).

## What it is not (each exonerated by direct experiment)

- Not the transcript/codec machinery: the failing step-2 prompt was captured
  via agent trace logging and is byte-perfect ChatML; the stop marker is kept
  (`first_stop_end` truncates *past* it).
- Not tokenization: `<tool_call>`/`</tool_call>` are added tokens in the
  sibling tokenizer.json, encoded as single pieces.
- Not the context window: 11.6k ≪ 32k, and CTX-1 would have refused loudly.
- Not content: identical scaffold with the tool response cut to N tokens
  flips from clean (≤ 8,100) to garbage (≥ 8,300) regardless of content; the
  bracket was tightened to (8,100 … 8,300) with pinned-token prompts, and
  decode-crossing runs put the onset at kv 8,192–8,194.
- Not prefill scheduling: chunk 64 (the GLM-4 mitigation), 32, and 16 all
  fail identically; chunk-64 prefill necessarily passes through kv = 8,192
  for any prompt ≥ 8,192 (64·128), which is why every long-prompt run hit it.
- Not the f32 attention ops in isolation: `lib/examples/metal_depth_probe.rs`
  compares q·kᵀ, softmax (masked and unmasked), probs·v, and cat on Metal vs
  CPU at kv ∈ {8,000 … 11,605} for decode and prefill-chunk shapes — max
  |Δ| ≈ 1e-7 everywhere.
- Not the RoPE tables or kernel: `metal_trig_probe.rs` (table values at
  positions to 32,767) and `metal_rope_probe.rs` (the rope op with offset
  views and contiguous copies at positions to 16,384) are both clean.
  (candle #3520 — bf16 RoPE tables in `models/qwen2.rs` — is the same
  disease in the safetensors path, but the GGUF path builds tables in f32.)

## Where the evidence points

The corruption appears only in the composed model run, precisely at the KV
depth where the K/V cache tensors reach exactly 2^25 bytes (8 kv-heads ×
8,192 positions × 128 dim × 4 bytes = 32 MiB — 4,096 bytes per position, so
depth 8,192 is the unique power-of-two crossing).

**The allocator is exonerated.** The leading suspect was candle's Metal
buffer pool (power-of-two buckets, reuse on `Arc` count 1, all buffers
`HazardTrackingModeUntracked`): at kv = 8,192 the cache `cat` is served
from the long-idle 32 MiB bucket rather than the steadily-recycled 64 MiB
one (confirmed by allocation logging on the instrumented local candle,
branch `metal-pool-debug` in the sibling clone). But a scoped kill-switch
(`CANDLE_METAL_NO_REUSE_POW2=1`: fresh buffers for exact power-of-two
requests ≥ 16 MiB, allocation logging confirming zero reuse at the
crossing) produced output **byte-identical** to the baseline — same clean
prefix, same derailment at kv ≈ 8,196, same garbage. Fresh buffers do not
move the cliff. The byte-identical determinism across two allocation
regimes also argues against any scheduling race: races are flaky, this is
clockwork.

What remains is a deterministic compute defect at kv ≥ 8,192 that only
manifests in the composed model. The provenance gap was closed too:
`metal_attn_replica_probe.rs` replays one decode step exactly as
`quantized_qwen2::forward_attn` does (4D batch shapes, 8-head KV cache
grown by `cat`, `repeat_kv` cat-of-5 + reshape, RoPE through narrowed
table views at the true positions, no mask) at positions 8,189–8,194 —
every intermediate agrees across devices at float-epsilon, with no anomaly
at the crossing. So the attention data path is clean even with exact
provenance; what the replica still lacks is what the real model adds: the
quantized weight matmuls (QMatMul) interleaved in the same stream, the
rest of the layer (RMSNorm, MLP), and 64 layers of encoder/command-buffer
batching per step. Localizing further requires layer-by-layer dumps inside
a vendored copy of the model on the 32B weights — the expensive hunt. The
cheap, model-free methods end here; stopped on cost grounds.

(A *global* no-reuse switch is not viable for experiments: fresh Metal
buffers for every op balloon unboundedly — a run took the machine down at
228 GB+. The scoped variant on the candle branch is bounded by design.)

## Status / mitigation

- The engine warns (CTX-2, `warn_metal_kv_cliff`) when a Metal run will
  reach depth 8,192. A warning, not a refusal: the depth is empirical and
  per-stack (validated on Qwen2.5-32B Q4_K_M, macOS 26 / M-series 48 GB,
  candle 0.11.0 = upstream main at 31f35b1).
- Hosts keep working prompts under the cliff: the TUI's `read_page` budget
  is 12k chars (≈ 3.3k tokens) for latency *and* headroom.
- Upstream: candle issue to be filed with this diagnosis; the pin already
  contains all recent Metal sync fixes (#3532, #3595, #3394), so the defect
  is live on their main.

## Reproducing

Any ≥ 8,192-token prompt through `yatima generate` on a Metal quantized
32B GGUF shows it (greedy, so deterministic):

```bash
python3 - <<'EOF' > /tmp/prompt.txt
print("<|im_start|>user\nSummarize this:\n" + ("lorem ipsum " * 4000) +
      "\n<|im_end|>\n<|im_start|>assistant\n")
EOF
yatima generate --repo bartowski/Qwen2.5-32B-Instruct-GGUF \
  --gguf Qwen2.5-32B-Instruct-Q4_K_M.gguf --max-tokens 60 < /tmp/prompt.txt
```

Sharper: a prompt pinned to 8,100 tokens with `--max-tokens 300` derails
mid-decode at generated token ~92. The probes
(`lib/examples/metal_{depth,trig,rope}_probe.rs`) document the exonerated
ops.
