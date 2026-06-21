# GLM4 GGUF Metal Prefill Reproducer

## Symptom

`bartowski/THUDM_GLM-4-32B-0414-GGUF` can produce coherent chat output through
Yatima when prompt prefill is chunked, but long full-prompt prefill on Metal can
destabilize immediately. The failure mode is incoherent next-token selection
followed by repetitive nonsense during generation.

The clearest reproduction was the investment thesis example:

- `glm` default on Metal, which currently chunks GLM4 GGUF prefill at 64 tokens:
  coherent output.
- explicit `glm 64`: coherent output.
- explicit `glm 0`, forcing one full-prompt prefill: incoherent output.

This points at prefill scheduling or a long-prefill backend path, not at the SEC
example, the GLM chat template, or ordinary token-by-token decoding.

## Current Mitigation

Yatima exposes `GenOpts::prefill_chunk`:

- `None`: use the model/backend default.
- `Some(0)`: force one full-prompt prefill.
- `Some(n)`: prefill in chunks of at most `n` tokens.

For quantized GLM4 GGUF on Metal, the backend default is currently `Some(64)`.
This mirrors llama.cpp's practical bounded prompt evaluation behavior more
closely than one huge prefill and avoids the observed degeneration.

This is a mitigation, not a root-cause fix.

## Reproducer

`lib/examples/prefill_compare.rs` compares the next-token logits after full
prefill and chunked prefill without generating any text.

Default GLM4 run:

```bash
cargo run -p yatima-lib --release --example prefill_compare --features metal -- \
  ~/.cache/yatima/models/bartowski/THUDM_GLM-4-32B-0414-GGUF glm
```

Smaller synthetic prompt while the machine is memory-constrained:

```bash
cargo run -p yatima-lib --release --example prefill_compare --features metal -- \
  ~/.cache/yatima/models/bartowski/THUDM_GLM-4-32B-0414-GGUF glm synthetic:64 64 8
```

With an explicit prompt file:

```bash
cargo run -p yatima-lib --release --example prefill_compare --features metal -- \
  ~/.cache/yatima/models/bartowski/THUDM_GLM-4-32B-0414-GGUF glm /tmp/prompt.txt 64 20
```

Arguments:

1. model directory
2. format: `qwen`, `gemma`, `mistral`, `glm`, `plain`, or `raw`
3. optional prompt file, `-` for stdin, or `synthetic:N` for an N-row synthetic
   SEC-like prompt
4. optional chunk size, default `64`
5. optional top-k count, default `12`

The example prints:

- progress for each prefill chunk
- prompt token count
- vocab size
- non-finite logit counts (`NaN`, `+inf`, `-inf`)
- max absolute logit delta
- RMS logit delta
- full-prefill top-k next tokens among finite logits
- chunked-prefill top-k next tokens among finite logits
- top-k overlap

The intended use is to make the bug cheaper and sharper than a long generation:
if full prefill and chunked prefill disagree sharply on the first next-token
distribution for the same prompt, the failure is already present before
sampling and decode.

Avoid running this concurrently with the investment thesis example on a
48GB-memory machine. The 32B GGUF load alone is large enough that two model
processes can cause severe resource contention and make the reproducer look
hung before it reaches its diagnostic output.

## Source Comparison Notes

The local Candle pin and current upstream Candle `main` were checked for:

- `candle-transformers/src/models/quantized_glm4.rs`
- `candle-core/src/quantized/metal.rs`
- `candle-metal-kernels/src/metal_src/quantized.metal`

No relevant upstream delta was found at the time of investigation.

Candle's quantized GLM4 attention path materializes the full attention score
tensor for prefill:

```text
q.matmul(k.transpose(2, 3)) -> softmax -> probs.matmul(v)
```

llama.cpp evaluates prompts through bounded batches/micro-batches and also marks
some GLM4 matrix multiplications for F32 accumulation because of documented
numerical issues. The Yatima mitigation attacks the scheduling side by avoiding
large single-shot prefill.

## Open Questions

- Is the divergence in the attention path, the quantized projection matmuls, or
  a Metal kernel shape threshold?
- Does the same divergence occur on CPU for the same GGUF and prompt?
- What is the smallest prompt token length where full prefill and chunked
  prefill materially diverge?
- Would a deeper Candle patch be better expressed as bounded prefill scheduling,
  more precise Metal accumulation for specific GLM4 matmuls, or an attention
  implementation change?
