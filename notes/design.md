# yatima — design notes

`yatima` is a small Rust runtime for **language-integrated LLMs**: calling a
local model as an ordinary in-process function rather than a network service.
We **own the runtime, rent the engine** — `yatima-lib` owns loading, the
generation loop, and (next) capability-scoped tools; the inference engine
([candle](https://github.com/huggingface/candle)) is a swappable dependency.

## Crates

- **`yatima-lib`** — the capability as a function: `Engine::load` + `generate`,
  the model path resolver, and (behind the `fetch` feature) auto-fetch.
- **`yatima-cli`** — a thin wrapper: `yatima generate` and `yatima models-dir`.

## The `Engine::generate` contract

This is the portable idea (what a later Haskell study would reason about):

- **Stateless per call** — the KV cache is cleared on entry; no conversation or
  cache is retained across calls.
- **Raw completion** — the prompt is fed as-is; no chat template.
- The callback receives **decoded text fragments** (not token ids), **in
  generation order**, via an incremental detokenizer (`TokenOutputStream`).
- Generation **stops** on EOS / `max_tokens` / the callback returning `Err`
  (which is surfaced to the caller — cancellation, stdout errors).
- **`temperature == 0` is deterministic** (greedy/argmax); otherwise output
  depends on `seed`. Greedy determinism is the hard test gate; seeded-sampling
  reproducibility is a bonus.

EOS ids are read from `config.json` / `generation_config.json` (a *set*, e.g.
DeepSeek's `<｜end▁of▁sentence｜>` = 151643) — never hard-coded strings.

## Model storage & loading

Weights are re-downloadable ⇒ they live under the XDG **cache**:

```
$YATIMA_MODELS_DIR  (else  ${XDG_CACHE_HOME:-~/.cache}/yatima/models)
  └── <org>/<name>/        # = model_dir(models_root(), repo)
        config.json  tokenizer.json  *.safetensors  [model.safetensors.index.json]
```

This is exactly possum's `--to <root>` → `<org>/<name>` layout, so the two
tools agree by **convention, not coupling**. `Engine::load` is HF-agnostic — it
takes a directory.

`model_shards()` is the single discovery rule used by **both** `Engine::load`
(what to mmap) and `is_model_present()` (what must exist): if
`model.safetensors.index.json` is present, the unique files in its `weight_map`;
otherwise all `*.safetensors`. So a *partial* shard set is never a false cache
hit.

## Auto-fetch (the `fetch` feature)

`yatima generate --repo <id>` fetches on a cache miss by calling
**`possum-lib`** in-process (no shelling out):

- **cache hit** → quiet load.
- **cache miss** → `ensure_model` downloads via `possum_lib::model::download`
  (include `*.safetensors`/`*.json`, exclude `figures/*`, `ProgressMode::Auto`
  for real progress bars), re-checks completeness, then loads.
- **`--offline`** → never touches the network; clear error if absent.
- **`--model <dir>`** → bypasses resolution/fetch entirely.

possum's download stays an ergonomic async library API (`download(req).await`,
bounded concurrency, aggregate errors, `indicatif` progress); yatima depends on
it via a pinned git rev.

## Deferred

Capability-scoped tool runtime + agent loop + conversation state (the next
layer); engine swappability (mistral.rs / llama.cpp-GGUF — the `Engine` API
allows it); download integrity/resume; a Haskell study of the `generate`
contract.
