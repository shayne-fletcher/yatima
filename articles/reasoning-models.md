# Reasoning models

Reasoning models (QwQ-32B, Kimi-Dev, the DeepSeek-R1 distills) emit an inline
chain-of-thought before their answer. yatima separates it: the trace is stripped
from the answer and kept out of conversation history, and the chat UIs dim/fold
it — it never pollutes the reply or the next prompt.

Run one with a profile, which also raises the token budget so the think block
isn't truncated:

```bash
# the recommended reasoning model: a strong reasoner that fits comfortably (~20 GB GGUF)
cargo run -p yatima-tui --release --features metal -- --profile qwq

# DeepSeek-R1 distill
cargo run -p yatima-tui --release --features metal -- --profile deepseek-r1
```

## Seeded vs. emitted markers

Some reasoning models *pre-seed* the `<think>` opener in their prompt (QwQ →
`qwen-think` format, DeepSeek → `deepseek`), emitting only the closing marker;
others *emit* the opener themselves. The profiles pin the right format so the
reasoning is classified either way.

The split markers are *special tokens* in some tokenizers (QwQ's `</think>` is
flagged like `<|im_end|>`). The decode keeps special tokens so the close marker
survives to the splitter — otherwise the answer is silently mislabeled as
reasoning. End-of-turn / EOS ids are filtered by id before decode, and the
splitter consumes the reasoning markers so they never leak to the UI (REASON-1).
