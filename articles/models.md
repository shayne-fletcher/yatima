# Models & quantization

yatima loads local **safetensors** and **GGUF/quantized** weights and dispatches
across model families from a single engine. A CI consistency harness checks every
family's wiring — architecture detection, GGUF normalization, and chat-format
mapping.

`‡` marks families that are **wired + harness-tested but not yet runtime-validated
with weights**: loading and generation are unverified on a real model of that
family.

| Model family        | generate | chat  | agent/tools |
|---------------------|----------|-------|-------------|
| Qwen2.5-Instruct    | yes      | yes   | yes         |
| Qwen3 ‡             | yes      | yes   | yes         |
| Qwen3-MoE ‡         | yes      | yes   | yes         |
| GLM-4 (9B / 32B)    | yes      | yes   | no          |
| Gemma-2-it          | yes      | yes   | no          |
| Gemma-3 ‡           | yes      | yes   | no          |
| Mistral-v0.3        | yes      | yes   | later       |
| TinyLlama-chat      | yes      | yes   | no          |
| StarCoder2          | yes      | maybe | no          |
| DeepSeek-V2/V3 ‡    | yes      | yes   | no          |

Supported architectures: **Qwen2, Qwen3, Qwen3-MoE, Llama, Mistral, Phi-3,
Gemma-2, Gemma-3, StarCoder2, GLM-4, DeepSeek-V2/V3** (safetensors), with
**GGUF/quantized** loading for Qwen2, Qwen3, Qwen3-MoE, Llama, Gemma-3, and GLM-4
(DeepSeek is safetensors-only — candle has no quantized DeepSeek loader). The
agent/tool path is narrower by design: it needs a model trained to emit tool
calls (today: Qwen/ChatML).

## GGUF quant note

candle reads standard quant types (`Q4_0/1`, `Q5_0/1`, `Q8_0`, `Q2_K`–`Q6_K`) but
**no i-quants** (`IQ*`). Many modern community GGUFs embed `IQ4_NL` tensors and
will fail to load (`unknown dtype 20`); pick a standard-type or `--pure` quant.

A missing model is fetched on demand when the `fetch` feature is enabled;
`--offline` never touches the network. Weights are acquired by
[`possum`](https://github.com/shayne-fletcher/possum) and loaded by yatima.
