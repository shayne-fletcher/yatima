//! Inference as an in-process library function.
//!
//! `Engine::load` reads a local model directory (HF-agnostic) and
//! `Engine::generate` runs a stateless, raw-completion generation loop,
//! streaming decoded text fragments to a callback. The engine rents candle's
//! transformer implementations, dispatching on the model's architecture (see
//! `CausalLm` / `detect_arch`); we own the load/generate boundary.

use std::collections::HashSet;
use std::ops::ControlFlow;
use std::path::{Path, PathBuf};

use anyhow::{anyhow, bail, Result};
use candle_core::quantized::gguf_file;
use candle_core::quantized::tokenizer::TokenizerFromGguf;
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::generation::LogitsProcessor;
use candle_transformers::models::{
    gemma2, glm4_new, llama, mistral, phi3, quantized_glm4, quantized_llama, quantized_qwen2,
    qwen2, starcoder2,
};
use candle_transformers::utils::apply_repeat_penalty;
use tokenizers::Tokenizer;

use crate::token_output_stream::TokenOutputStream;

/// The upstream qwen example's anti-repetition default; without it a
/// temperature-0 raw *prose* completion of an instruction-tuned distill
/// degenerates. It is the wrong default for *structured* output, though — it
/// penalises the punctuation JSON repeats (`"`, `{`, `<`), corrupting tool
/// calls — so it is a per-call [`GenOpts`] knob the agent turns off.
const DEFAULT_REPEAT_PENALTY: f32 = 1.1;
const REPEAT_LAST_N: usize = 64;
const GLM4_METAL_PREFILL_CHUNK: usize = 64;

/// How the next token is chosen. Replaces the old `temperature <= 0` sentinel
/// with an explicit choice — greedy carries no temperature, and seed is
/// meaningless for greedy.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Sampling {
    /// Greedy / argmax — deterministic and seed-free.
    Greedy,
    /// Seeded sampling at the given temperature (expected `> 0`).
    Sample { temperature: f64, seed: u64 },
}

impl Sampling {
    /// Build candle's `LogitsProcessor` for this policy.
    ///
    /// Total (law SAM-1): every `Sampling` maps to exactly one processor.
    /// `Greedy` passes a fixed seed, so it is seed-free by construction (SAM-2).
    fn logits_processor(self) -> LogitsProcessor {
        match self {
            Sampling::Greedy => LogitsProcessor::new(0, None, None),
            Sampling::Sample { temperature, seed } => {
                LogitsProcessor::new(seed, Some(temperature), None)
            }
        }
    }

    /// Map a `(temperature, seed)` pair to a policy: a non-positive temperature
    /// is deterministic [`Greedy`](Sampling::Greedy) (the seed is ignored, per
    /// SAM-2); a positive temperature is seeded [`Sample`](Sampling::Sample).
    pub fn from_temperature(temperature: f64, seed: u64) -> Sampling {
        if temperature <= 0.0 {
            Sampling::Greedy
        } else {
            Sampling::Sample { temperature, seed }
        }
    }
}

/// Options for a single generation.
#[derive(Debug, Clone)]
pub struct GenOpts {
    pub max_tokens: usize,
    pub sampling: Sampling,
    /// Repetition penalty over the last `REPEAT_LAST_N` tokens. `1.0` disables
    /// it (the right choice for structured/tool-call output); ~`1.1` suits prose.
    pub repeat_penalty: f32,
    /// Prompt prefill chunk size in tokens.
    ///
    /// `None` uses the model/backend default. `Some(0)` explicitly feeds the
    /// whole prompt in one prefill. `Some(n)` feeds chunks of at most `n`
    /// tokens. This is useful for diagnosing or avoiding backend-specific
    /// prefill kernels while preserving the same public generation loop.
    pub prefill_chunk: Option<usize>,
}

impl Default for GenOpts {
    fn default() -> Self {
        Self {
            max_tokens: 256,
            sampling: Sampling::Greedy,
            repeat_penalty: DEFAULT_REPEAT_PENALTY,
            prefill_chunk: None,
        }
    }
}

/// Why a generation stopped. Exactly one per successful generation (law STOP-1).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StopReason {
    /// An end-of-sequence token was sampled.
    Eos,
    /// `max_tokens` were produced.
    MaxTokens,
    /// The fold step returned `ControlFlow::Break` (voluntary cancellation).
    Stopped,
}

/// The outcome of a generation: how many tokens were produced and why it ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Generation {
    pub tokens: usize,
    pub stop: StopReason,
}

/// One decoded candidate from a diagnostic logit ranking.
#[derive(Debug, Clone, PartialEq)]
pub struct TokenLogit {
    pub id: u32,
    pub text: String,
    pub logit: f32,
}

/// Diagnostic logits for the next token after prompt prefill.
#[derive(Debug, Clone, PartialEq)]
pub struct PrefillLogits {
    pub token_count: usize,
    pub logits: Vec<f32>,
}

/// Progress for diagnostic prompt prefill.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PrefillProgress {
    /// Zero-based chunk index.
    pub chunk_index: usize,
    /// Total chunks for this prefill schedule.
    pub chunk_count: usize,
    /// Inclusive token offset fed to the model for this chunk.
    pub start_pos: usize,
    /// Exclusive token offset fed to the model for this chunk.
    pub end_pos: usize,
    /// Total prompt tokens.
    pub token_count: usize,
    /// `false` just before the model call, `true` just after it returns.
    pub finished: bool,
}

/// A causal language model behind one uniform interface, so the generation loop
/// is architecture-agnostic. candle's models come in two shapes: most manage an
/// internal KV cache (`forward(ids, seqlen_offset)` + `clear_kv_cache`), while
/// Llama threads an external [`llama::Cache`]. This trait hides that difference.
trait CausalLm {
    /// Logits for the next token. `pos` is the sequence offset (0 on prefill).
    /// The returned tensor's shape may be `[1, 1, vocab]` or `[1, vocab]`
    /// depending on the model — callers normalize via [`last_token_logits`].
    fn forward(&mut self, input: &Tensor, pos: usize) -> Result<Tensor>;
    /// Clear/rebuild the KV cache so the next call starts fresh (stateless per
    /// generation — upholds GE-1).
    fn reset(&mut self) -> Result<()>;
}

/// Implement [`CausalLm`] for a candle model with a self-managed KV cache
/// (`forward(&mut self, ids, offset)` + `clear_kv_cache`). All the supported
/// architectures except Llama fit this shape.
macro_rules! self_cache_causal_lm {
    ($ty:ty) => {
        impl CausalLm for $ty {
            fn forward(&mut self, input: &Tensor, pos: usize) -> Result<Tensor> {
                Ok(<$ty>::forward(self, input, pos)?)
            }
            fn reset(&mut self) -> Result<()> {
                self.clear_kv_cache();
                Ok(())
            }
        }
    };
}

self_cache_causal_lm!(qwen2::ModelForCausalLM);
self_cache_causal_lm!(mistral::Model);
self_cache_causal_lm!(phi3::Model);
self_cache_causal_lm!(gemma2::Model);
self_cache_causal_lm!(starcoder2::Model);
self_cache_causal_lm!(glm4_new::ModelForCausalLM);

// Quantized (GGUF) models fit the same self-cache shape: `forward(&mut self, x,
// index_pos)` + `clear_kv_cache()`.
self_cache_causal_lm!(quantized_qwen2::ModelWeights);
self_cache_causal_lm!(quantized_llama::ModelWeights);

// Quantized GLM-4 has the same `forward(input, offset)` but no public
// `clear_kv_cache`: it auto-resets its KV cache when `offset == 0` (the prefill
// our loop always hits first), so `reset` is a safe no-op.
impl CausalLm for quantized_glm4::ModelWeights {
    fn forward(&mut self, input: &Tensor, pos: usize) -> Result<Tensor> {
        Ok(quantized_glm4::ModelWeights::forward(self, input, pos)?)
    }
    fn reset(&mut self) -> Result<()> {
        Ok(())
    }
}

/// Llama threads an external cache and has no `clear_kv_cache`; this wrapper
/// holds the cache (and what's needed to rebuild it) so it fits [`CausalLm`].
struct LlamaLm {
    model: llama::Llama,
    cache: llama::Cache,
    cfg: llama::Config,
    device: Device,
    dtype: DType,
}

impl CausalLm for LlamaLm {
    fn forward(&mut self, input: &Tensor, pos: usize) -> Result<Tensor> {
        Ok(self.model.forward(input, pos, &mut self.cache)?)
    }
    fn reset(&mut self) -> Result<()> {
        // Llama has no clear_kv_cache — a fresh Cache is the reset.
        self.cache = llama::Cache::new(true, self.dtype, &self.cfg, &self.device)?;
        Ok(())
    }
}

/// Reduce a model's raw output to a 1-D `[vocab]` F32 logit vector for the last
/// position. The single normalization point: self-cache models return
/// `[1, 1, vocab]` and Llama returns `[1, vocab]`; `flatten_all` handles both,
/// and any future shape only needs fixing here.
fn last_token_logits(logits: &Tensor) -> Result<Tensor> {
    Ok(logits.flatten_all()?.to_dtype(DType::F32)?)
}

fn effective_prefill_chunk(configured: Option<usize>, default: Option<usize>, len: usize) -> usize {
    match configured.or(default) {
        Some(0) | None => len,
        Some(n) => n,
    }
}

/// The model architectures the runtime can load. The single public spine both
/// load paths normalize through (ARCH-1): safetensors via `detect_arch` and
/// GGUF via `arch_from_gguf`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Arch {
    Qwen2,
    Llama,
    Mistral,
    Phi3,
    Gemma2,
    Starcoder2,
    Glm4,
}

impl Arch {
    /// The engine-native prefill-chunk default for this architecture **on
    /// Metal** (`None` = one full-prompt prefill). Runtime policy only — it
    /// names no host/format type, so the inference core never depends upward on
    /// the hosting layer. GLM-4 GGUF degrades on a single long Metal prefill
    /// (matrix-matrix kernel precision), so it is bounded; see
    /// `notes/glm4-prefill-reproducer.md`. The caller gates on `device.is_metal`.
    pub fn metal_prefill_chunk(self) -> Option<usize> {
        match self {
            Arch::Glm4 => Some(GLM4_METAL_PREFILL_CHUNK),
            _ => None,
        }
    }
}

/// Normalize a GGUF `general.architecture` string to an [`Arch`] at the load
/// boundary (ARCH-2), so raw metadata strings never leak into dispatch. Only the
/// architectures with a quantized loader are accepted.
fn arch_from_gguf(s: &str) -> Result<Arch> {
    match s {
        "qwen2" => Ok(Arch::Qwen2),
        "llama" => Ok(Arch::Llama),
        "glm4" | "chatglm" => Ok(Arch::Glm4),
        other => bail!("unsupported GGUF architecture: {other} (supported: qwen2, llama, glm4)"),
    }
}

/// Detect the architecture from `config.json`, accepting both the
/// `architectures` class name (e.g. `Qwen2ForCausalLM`) and the `model_type`
/// short form (e.g. `qwen2`), so the fallback is real.
fn detect_arch(config: &serde_json::Value) -> Result<Arch> {
    let class = config
        .get("architectures")
        .and_then(|a| a.get(0))
        .and_then(|s| s.as_str());
    let model_type = config.get("model_type").and_then(|s| s.as_str());
    match class.or(model_type) {
        Some(name) => match name {
            "Qwen2ForCausalLM" | "qwen2" => Ok(Arch::Qwen2),
            "LlamaForCausalLM" | "llama" => Ok(Arch::Llama),
            "MistralForCausalLM" | "mistral" => Ok(Arch::Mistral),
            "Phi3ForCausalLM" | "phi3" => Ok(Arch::Phi3),
            "Gemma2ForCausalLM" | "gemma2" => Ok(Arch::Gemma2),
            "Starcoder2ForCausalLM" | "starcoder2" => Ok(Arch::Starcoder2),
            "Glm4ForCausalLM" | "glm4" => Ok(Arch::Glm4),
            other => bail!(
                "unsupported architecture: {other} (supported: Qwen2, Llama, \
                 Mistral, Phi3, Gemma2, Starcoder2, Glm4)"
            ),
        },
        None => bail!("config.json has no `architectures` or `model_type`"),
    }
}

/// The single `*.gguf` in `dir`, if present — the signal to take the quantized
/// load path. `None` means a safetensors (or absent) model.
fn gguf_in(dir: &Path) -> Option<PathBuf> {
    std::fs::read_dir(dir).ok()?.find_map(|e| {
        let p = e.ok()?.path();
        (p.extension().is_some_and(|x| x == "gguf")).then_some(p)
    })
}

/// EOS ids for a GGUF model: the file's `tokenizer.ggml.eos_token_id` metadata,
/// plus any chat end-of-turn tokens present in the tokenizer vocab (Qwen's
/// `<|im_end|>`, `<|endoftext|>`) so instruct models stop cleanly. A missing EOS
/// only degrades to stopping on `max_tokens`, never a wrong result.
fn gguf_eos_ids(content: &gguf_file::Content, tokenizer: &Tokenizer) -> HashSet<u32> {
    let mut eos = HashSet::new();
    if let Some(id) = content
        .metadata
        .get("tokenizer.ggml.eos_token_id")
        .and_then(|v| v.to_u32().ok())
    {
        eos.insert(id);
    }
    let vocab = tokenizer.get_vocab(true);
    for tok in ["<|im_end|>", "<|endoftext|>"] {
        if let Some(&id) = vocab.get(tok) {
            eos.insert(id);
        }
    }
    eos
}

/// A loaded model ready to generate. Construct with [`Engine::load`].
pub struct Engine {
    model: Box<dyn CausalLm>,
    tokenizer: Tokenizer,
    device: Device,
    eos: HashSet<u32>,
    dtype: DType,
    arch: Arch,
    prefill_chunk: Option<usize>,
}

impl Engine {
    /// Load weights + tokenizer from a local model directory.
    ///
    /// Two layouts are supported. A **GGUF** dir (a single `*.gguf` plus
    /// `tokenizer.json`, no `config.json`) loads the quantized model — see
    /// `load_gguf`. Otherwise the **safetensors** layout
    /// (`config.json`, `tokenizer.json`, `*.safetensors`) is loaded, dispatched
    /// by `detect_arch`. EOS ids come from the config / GGUF metadata, never
    /// hard-coded.
    pub fn load(model_dir: &Path, device: Device) -> Result<Self> {
        let span = tracing::info_span!("model.load", model = %model_dir.display());
        let _enter = span.enter();
        if let Some(gguf) = gguf_in(model_dir) {
            return Self::load_gguf(model_dir, &gguf, device);
        }

        let config_path = model_dir.join("config.json");
        let tokenizer_path = model_dir.join("tokenizer.json");
        if !config_path.exists() {
            bail!("missing config.json in {}", model_dir.display());
        }
        if !tokenizer_path.exists() {
            bail!("missing tokenizer.json in {}", model_dir.display());
        }

        let config_bytes = std::fs::read(&config_path)?;
        let config_value: serde_json::Value = serde_json::from_slice(&config_bytes)?;
        let arch = detect_arch(&config_value)?;

        let gen_config_path = model_dir.join("generation_config.json");
        let gen_config_value = if gen_config_path.exists() {
            Some(serde_json::from_slice::<serde_json::Value>(
                &std::fs::read(&gen_config_path)?,
            )?)
        } else {
            None
        };
        let eos = extract_eos_ids(&config_value, gen_config_value.as_ref());

        let tokenizer =
            Tokenizer::from_file(&tokenizer_path).map_err(|e| anyhow!("loading tokenizer: {e}"))?;

        let shards = model_shards(model_dir)?;
        // dtype is an implementation detail, not a gate: bf16 on the GPU, f32
        // on CPU. The actual choice is recorded and exposed via `backend`.
        let dtype = if device.is_metal() || device.is_cuda() {
            DType::BF16
        } else {
            DType::F32
        };
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&shards, dtype, &device)? };

        // Build the arch-specific model behind the uniform `CausalLm` interface.
        // Each `Config` is parsed from the same bytes; errors name the arch.
        let parse =
            |what: &str| -> anyhow::Error { anyhow!("parsing config.json as a {what} config") };
        let model: Box<dyn CausalLm> = match arch {
            Arch::Qwen2 => {
                let cfg: qwen2::Config =
                    serde_json::from_slice(&config_bytes).map_err(|_| parse("Qwen2"))?;
                Box::new(qwen2::ModelForCausalLM::new(&cfg, vb)?)
            }
            Arch::Mistral => {
                let cfg: mistral::Config =
                    serde_json::from_slice(&config_bytes).map_err(|_| parse("Mistral"))?;
                Box::new(mistral::Model::new(&cfg, vb)?)
            }
            Arch::Phi3 => {
                let cfg: phi3::Config =
                    serde_json::from_slice(&config_bytes).map_err(|_| parse("Phi3"))?;
                Box::new(phi3::Model::new(&cfg, vb)?)
            }
            Arch::Gemma2 => {
                let cfg: gemma2::Config =
                    serde_json::from_slice(&config_bytes).map_err(|_| parse("Gemma2"))?;
                Box::new(gemma2::Model::new(false, &cfg, vb)?)
            }
            Arch::Starcoder2 => {
                let cfg: starcoder2::Config =
                    serde_json::from_slice(&config_bytes).map_err(|_| parse("Starcoder2"))?;
                Box::new(starcoder2::Model::new(&cfg, vb)?)
            }
            Arch::Glm4 => {
                let cfg: glm4_new::Config =
                    serde_json::from_slice(&config_bytes).map_err(|_| parse("Glm4"))?;
                Box::new(glm4_new::ModelForCausalLM::new(&cfg, vb)?)
            }
            Arch::Llama => {
                let cfg: llama::LlamaConfig =
                    serde_json::from_slice(&config_bytes).map_err(|_| parse("Llama"))?;
                let cfg = cfg.into_config(false);
                let cache = llama::Cache::new(true, dtype, &cfg, &device)?;
                let model = llama::Llama::load(vb, &cfg)?;
                Box::new(LlamaLm {
                    model,
                    cache,
                    cfg,
                    device: device.clone(),
                    dtype,
                })
            }
        };

        let engine = Self {
            model,
            tokenizer,
            device,
            eos,
            dtype,
            arch,
            prefill_chunk: None,
        };
        tracing::info!(
            model = %model_dir.display(),
            arch = ?engine.arch(),
            backend = %engine.backend(),
            prefill_chunk = ?engine.default_prefill_chunk(),
            "loaded model"
        );
        Ok(engine)
    }

    /// Load a quantized model from a GGUF file. A GGUF carries its own metadata
    /// (architecture, EOS) *and* tokenizer, so it is self-contained: the
    /// tokenizer is built from the GGUF metadata unless a sibling
    /// `tokenizer.json` is present (which then wins). Quantized weights run on
    /// the device (Metal-quantized is supported).
    fn load_gguf(model_dir: &Path, gguf_path: &Path, device: Device) -> Result<Self> {
        let span = tracing::info_span!(
            "model.load_gguf",
            model = %model_dir.display(),
            gguf = %gguf_path.display()
        );
        let _enter = span.enter();
        let mut file = std::fs::File::open(gguf_path)?;
        let content = gguf_file::Content::read(&mut file)
            .map_err(|e| anyhow!("reading GGUF {}: {e}", gguf_path.display()))?;

        // A sibling tokenizer.json wins; otherwise build the tokenizer from the
        // GGUF's embedded `tokenizer.ggml.*` metadata (candle-core's builder).
        let tokenizer_path = model_dir.join("tokenizer.json");
        let tokenizer = if tokenizer_path.exists() {
            Tokenizer::from_file(&tokenizer_path).map_err(|e| anyhow!("loading tokenizer: {e}"))?
        } else {
            Tokenizer::from_gguf(&content)
                .map_err(|e| anyhow!("building tokenizer from GGUF metadata: {e}"))?
        };

        let arch_str = content
            .metadata
            .get("general.architecture")
            .and_then(|v| v.to_string().ok())
            .ok_or_else(|| anyhow!("GGUF metadata has no general.architecture"))?;
        let arch = arch_from_gguf(arch_str)?;

        let eos = gguf_eos_ids(&content, &tokenizer);
        // GGUF weights carry their own quantized dtype; report q-on-device.
        let dtype = DType::F32;

        // Engine-native, device-aware prefill default. llama.cpp evaluates
        // prompts through bounded micro-batches (`n_ubatch`); feeding GLM-4 GGUF
        // a long prompt as one Metal prefill destabilizes generation on Candle.
        // The per-arch policy lives on `Arch`; we gate it on the actual device.
        let prefill_chunk = if device.is_metal() {
            arch.metal_prefill_chunk()
        } else {
            None
        };

        let model: Box<dyn CausalLm> = match arch {
            Arch::Qwen2 => Box::new(quantized_qwen2::ModelWeights::from_gguf(
                content, &mut file, &device,
            )?),
            Arch::Llama => Box::new(quantized_llama::ModelWeights::from_gguf(
                content, &mut file, &device,
            )?),
            Arch::Glm4 => Box::new(quantized_glm4::ModelWeights::from_gguf(
                content, &mut file, &device, dtype,
            )?),
            // `arch_from_gguf` only yields the three above.
            other => unreachable!("GGUF arch {other:?} has no quantized loader"),
        };

        let engine = Self {
            model,
            tokenizer,
            device,
            eos,
            dtype,
            arch,
            prefill_chunk,
        };
        tracing::info!(
            model = %model_dir.display(),
            gguf = %gguf_path.display(),
            arch = ?engine.arch(),
            backend = %engine.backend(),
            prefill_chunk = ?engine.default_prefill_chunk(),
            "loaded model"
        );
        Ok(engine)
    }

    /// A short "backend/dtype" label for diagnostics, e.g. `metal/BF16`.
    pub fn backend(&self) -> String {
        let dev = match &self.device {
            Device::Cpu => "cpu",
            Device::Cuda(_) => "cuda",
            Device::Metal(_) => "metal",
        };
        format!("{dev}/{:?}", self.dtype)
    }

    /// The architecture this engine loaded — the single detected [`Arch`]
    /// (ARCH-1), so callers can infer chat format and capabilities from the
    /// model itself rather than re-supplying them.
    pub fn arch(&self) -> Arch {
        self.arch
    }

    /// The **effective**, device-aware prefill-chunk default this engine applies
    /// when [`GenOpts::prefill_chunk`] is `None` (PREFILL-1). Owned by the loaded
    /// engine because the policy is device-specific (e.g. GLM-4 only chunks on
    /// Metal); profiles and CLI flags layer their own override over this.
    pub fn default_prefill_chunk(&self) -> Option<usize> {
        self.prefill_chunk
    }

    /// The number of tokens `text` encodes to under this model's tokenizer — a
    /// small helper for run metadata (reproducible comparisons).
    pub fn token_count(&self, text: &str) -> Result<usize> {
        Ok(self.encode_prompt(text)?.len())
    }

    fn encode_prompt(&self, prompt: &str) -> Result<Vec<u32>> {
        let tokens = self
            .tokenizer
            .encode(prompt, true)
            .map_err(|e| anyhow!("tokenizing prompt: {e}"))?
            .get_ids()
            .to_vec();
        if tokens.is_empty() {
            bail!("tokenized prompt is empty");
        }
        Ok(tokens)
    }

    fn prefill_last_logits_for_tokens(
        &mut self,
        tokens: &[u32],
        configured: Option<usize>,
    ) -> Result<Tensor> {
        self.prefill_last_logits_for_tokens_with_progress(tokens, configured, &mut |_| {})
    }

    fn prefill_last_logits_for_tokens_with_progress(
        &mut self,
        tokens: &[u32],
        configured: Option<usize>,
        on_progress: &mut impl FnMut(PrefillProgress),
    ) -> Result<Tensor> {
        let chunk = effective_prefill_chunk(configured, self.prefill_chunk, tokens.len());
        let chunk_count = tokens.len().div_ceil(chunk);
        let mut logits = None;
        for (chunk_index, start_pos) in (0..tokens.len()).step_by(chunk).enumerate() {
            let end = (start_pos + chunk).min(tokens.len());
            tracing::debug!(
                chunk_index,
                chunk_count,
                start_pos,
                end_pos = end,
                prompt_tokens = tokens.len(),
                "prefill chunk started"
            );
            on_progress(PrefillProgress {
                chunk_index,
                chunk_count,
                start_pos,
                end_pos: end,
                token_count: tokens.len(),
                finished: false,
            });
            let input = Tensor::new(&tokens[start_pos..end], &self.device)?.unsqueeze(0)?;
            logits = Some(self.model.forward(&input, start_pos)?);
            tracing::debug!(
                chunk_index,
                chunk_count,
                start_pos,
                end_pos = end,
                prompt_tokens = tokens.len(),
                "prefill chunk finished"
            );
            on_progress(PrefillProgress {
                chunk_index,
                chunk_count,
                start_pos,
                end_pos: end,
                token_count: tokens.len(),
                finished: true,
            });
        }
        logits.ok_or_else(|| anyhow!("tokenized prompt is empty"))
    }

    /// Return the next-token logits after prompt prefill without sampling or
    /// generating. This is a diagnostic hook for backend/scheduling work: it
    /// exercises the same prefill path as [`generate_with`](Engine::generate_with)
    /// and then stops before decode.
    ///
    /// `prefill_chunk` has the same meaning as [`GenOpts::prefill_chunk`]:
    /// `None` uses the model/backend default, `Some(0)` forces one full-prompt
    /// prefill, and `Some(n)` feeds chunks of at most `n` tokens.
    pub fn prefill_logits(
        &mut self,
        prompt: &str,
        prefill_chunk: Option<usize>,
    ) -> Result<PrefillLogits> {
        self.prefill_logits_with_progress(prompt, prefill_chunk, |_| {})
    }

    /// Like [`prefill_logits`](Engine::prefill_logits), but reports each prefill
    /// chunk before and after the model call.
    pub fn prefill_logits_with_progress(
        &mut self,
        prompt: &str,
        prefill_chunk: Option<usize>,
        mut on_progress: impl FnMut(PrefillProgress),
    ) -> Result<PrefillLogits> {
        self.model.reset()?;
        let tokens = self.encode_prompt(prompt)?;
        let logits = self.prefill_last_logits_for_tokens_with_progress(
            &tokens,
            prefill_chunk,
            &mut on_progress,
        )?;
        let logits = last_token_logits(&logits)?.to_vec1::<f32>()?;
        Ok(PrefillLogits {
            token_count: tokens.len(),
            logits,
        })
    }

    /// Decode the top `k` logits with the model tokenizer for diagnostics.
    pub fn topk_from_logits(&self, logits: &[f32], k: usize) -> Vec<TokenLogit> {
        let mut ranked: Vec<(usize, f32)> = logits
            .iter()
            .copied()
            .enumerate()
            .filter(|(_, logit)| logit.is_finite())
            .collect();
        ranked.sort_unstable_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        ranked
            .into_iter()
            .take(k)
            .map(|(id, logit)| {
                let id = id as u32;
                let text = self
                    .tokenizer
                    .decode(&[id], false)
                    .unwrap_or_else(|_| format!("<decode-error:{id}>"));
                TokenLogit { id, text, logit }
            })
            .collect()
    }

    /// Convenience wrapper around [`prefill_logits`](Engine::prefill_logits)
    /// plus [`topk_from_logits`](Engine::topk_from_logits).
    pub fn prefill_topk(
        &mut self,
        prompt: &str,
        prefill_chunk: Option<usize>,
        k: usize,
    ) -> Result<(usize, Vec<TokenLogit>)> {
        let prefill = self.prefill_logits(prompt, prefill_chunk)?;
        let topk = self.topk_from_logits(&prefill.logits, k);
        Ok((prefill.token_count, topk))
    }

    /// Generate, folding each decoded text fragment into an accumulator — the
    /// primitive of which [`generate`](Engine::generate) is the `acc = ()`
    /// specialization. Stateless request-reply (the KV cache is cleared on
    /// entry); raw completion (prompt fed as-is, no chat template).
    ///
    /// `step` threads the accumulator and returns a [`ControlFlow`]:
    /// `Continue(acc)` keeps going, `Break(acc)` stops voluntarily
    /// (`StopReason::Stopped`), and an `Err` is propagated to the caller.
    /// Generation also stops on EOS or `max_tokens`. (This is an effectful fold
    /// over the generated fragment stream; `notes/design.md` gives the
    /// categorical reading.)
    pub fn generate_with<A>(
        &mut self,
        prompt: &str,
        opts: &GenOpts,
        init: A,
        mut step: impl FnMut(A, &str) -> Result<ControlFlow<A, A>>,
    ) -> Result<(A, Generation)> {
        let span = tracing::info_span!(
            "engine.generate",
            arch = ?self.arch(),
            backend = %self.backend(),
            max_tokens = opts.max_tokens,
            sampling = ?opts.sampling,
            repeat_penalty = opts.repeat_penalty,
            prefill_chunk = ?opts.prefill_chunk,
            default_prefill_chunk = ?self.default_prefill_chunk()
        );
        let _enter = span.enter();
        self.model.reset()?;

        let mut stream = TokenOutputStream::new(self.tokenizer.clone());
        let mut tokens = self.encode_prompt(prompt)?;
        let prompt_tokens = tokens.len();

        let mut logits_processor = opts.sampling.logits_processor();
        let mut acc = init;
        let mut generated = 0usize;
        let mut stop = StopReason::MaxTokens;

        for index in 0..opts.max_tokens {
            // Prefill the prompt on the first step, then feed one token at a
            // time; the model advances its own KV cache via `start_pos`.
            //
            // Some quantized Metal backends are less stable on large prefill
            // matrix-matrix kernels than on the decode-style matrix-vector
            // path. `prefill_chunk` lets those models use smaller prefill
            // chunks without changing the public API.
            let logits = if index == 0 {
                self.prefill_last_logits_for_tokens(&tokens, opts.prefill_chunk)?
            } else {
                let start_pos = tokens.len().saturating_sub(1);
                let input = Tensor::new(&tokens[start_pos..], &self.device)?.unsqueeze(0)?;
                self.model.forward(&input, start_pos)?
            };
            let logits = last_token_logits(&logits)?;
            let logits = if opts.repeat_penalty == 1.0 {
                logits
            } else {
                let start = tokens.len().saturating_sub(REPEAT_LAST_N);
                apply_repeat_penalty(&logits, opts.repeat_penalty, &tokens[start..])?
            };

            let next = logits_processor.sample(&logits)?;
            tokens.push(next);
            generated += 1;
            if self.eos.contains(&next) {
                stop = StopReason::Eos;
                break;
            }
            if let Some(piece) = stream.next_token(next)? {
                match step(acc, &piece)? {
                    ControlFlow::Continue(a) => acc = a,
                    ControlFlow::Break(a) => {
                        acc = a;
                        stop = StopReason::Stopped;
                        break;
                    }
                }
            }
        }

        // Always flush trailing buffered text. The incremental detokenizer holds
        // back punctuation until the next alphanumeric token, so on *any* exit
        // (EOS, max_tokens, or a caller `Break`) the buffered tail — closing
        // quotes, `}`, a stop marker like `</tool_call>` — must be delivered, or
        // the accumulated text is silently truncated (corrupting tool-call JSON).
        if let Some(rest) = stream.decode_rest()? {
            acc = match step(acc, &rest)? {
                ControlFlow::Continue(a) | ControlFlow::Break(a) => a,
            };
        }

        let generation = Generation {
            tokens: generated,
            stop,
        };
        tracing::info!(
            prompt_tokens,
            generated_tokens = generation.tokens,
            stop = ?generation.stop,
            "generation finished"
        );
        Ok((acc, generation))
    }

    /// Run inference, streaming decoded text fragments to `on_token` (the
    /// `acc = ()` specialization of [`generate_with`](Engine::generate_with)).
    /// Returning `Err` from `on_token` stops generation and is surfaced.
    pub fn generate(
        &mut self,
        prompt: &str,
        opts: &GenOpts,
        mut on_token: impl FnMut(&str) -> Result<()>,
    ) -> Result<Generation> {
        let ((), generation) = self.generate_with(prompt, opts, (), |(), piece| {
            on_token(piece)?;
            Ok(ControlFlow::Continue(()))
        })?;
        Ok(generation)
    }
}

/// Select a compute device. `cpu == false` prefers Metal (when built with the
/// `metal` feature), falling back to CPU if it is unavailable.
pub fn device(cpu: bool) -> Result<Device> {
    if cpu {
        return Ok(Device::Cpu);
    }
    #[cfg(feature = "metal")]
    {
        match Device::new_metal(0) {
            Ok(d) => Ok(d),
            Err(e) => {
                eprintln!("metal unavailable ({e}); falling back to CPU");
                Ok(Device::Cpu)
            }
        }
    }
    #[cfg(not(feature = "metal"))]
    {
        Ok(Device::Cpu)
    }
}

/// The safetensors shards of a model directory, sorted. If
/// `model.safetensors.index.json` is present, the unique files referenced by
/// its `weight_map` are returned (the authoritative sharded set); otherwise
/// all `*.safetensors` in the directory. Errors if neither yields any file.
///
/// Used by both [`Engine::load`] (what to mmap) and [`is_model_present`] (what
/// must exist) so the two never disagree.
pub(crate) fn model_shards(dir: &Path) -> Result<Vec<PathBuf>> {
    let index = dir.join("model.safetensors.index.json");
    if index.exists() {
        let value: serde_json::Value = serde_json::from_slice(&std::fs::read(&index)?)?;
        let names: std::collections::BTreeSet<String> = value
            .get("weight_map")
            .and_then(|m| m.as_object())
            .map(|m| {
                m.values()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect()
            })
            .unwrap_or_default();
        if names.is_empty() {
            bail!("empty weight_map in {}", index.display());
        }
        // The index is untrusted data; a shard name must not escape `dir`.
        let mut shards = Vec::with_capacity(names.len());
        for n in names {
            if !crate::is_safe_relative(&n) {
                bail!(
                    "shard '{n}' in {} escapes the model directory",
                    index.display()
                );
            }
            shards.push(dir.join(n));
        }
        Ok(shards)
    } else {
        let mut shards: Vec<PathBuf> = std::fs::read_dir(dir)?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().is_some_and(|x| x == "safetensors"))
            .collect();
        shards.sort();
        if shards.is_empty() {
            bail!("no *.safetensors weights found in {}", dir.display());
        }
        Ok(shards)
    }
}

/// The completeness of a model directory: whether it is loadable, and which
/// required files are missing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Presence {
    pub complete: bool,
    pub missing: Vec<PathBuf>,
}

/// Is `dir` a loadable model, and what's missing?
///
/// Two layouts: a **GGUF** dir is complete on the strength of its `*.gguf` alone
/// — the tokenizer is built from the file's metadata, no `tokenizer.json` (or
/// `config.json`) required; otherwise the **safetensors** layout requires
/// `config.json`, `tokenizer.json`, and every shard from [`model_shards`] — so a
/// partial shard set is never a false cache hit. `missing` names the absent ones.
pub(crate) fn presence(dir: &Path) -> Presence {
    if gguf_in(dir).is_some() {
        // The GGUF is self-contained (weights + tokenizer + metadata).
        return Presence {
            complete: true,
            missing: Vec::new(),
        };
    }

    let mut required = vec![dir.join("config.json"), dir.join("tokenizer.json")];
    match model_shards(dir) {
        Ok(shards) => required.extend(shards),
        Err(_) => {
            // No resolvable weights ⇒ incomplete; flag a weights sentinel.
            return Presence {
                complete: false,
                missing: vec![dir.join("*.safetensors")],
            };
        }
    }

    let missing: Vec<PathBuf> = required.into_iter().filter(|p| !p.exists()).collect();
    Presence {
        complete: missing.is_empty(),
        missing,
    }
}

/// Whether `dir` holds a loadable model (the `complete` flag of `presence`).
pub fn is_model_present(dir: &Path) -> bool {
    presence(dir).complete
}

/// Ensure the weights for `repo` are present under `models_root`, fetching them
/// with possum on a cache miss; returns the model directory. Re-checks
/// completeness after download so a partial fetch is never handed to
/// [`Engine::load`].
///
/// `gguf` selects a single quantized file to fetch (plus `*.json` for the
/// tokenizer) instead of the safetensors shards — for `--repo <id> --gguf
/// <file>`. Note GGUF repos often omit `tokenizer.json`; if it's missing after
/// the fetch, the completeness check fails with a clear error.
#[cfg(feature = "fetch")]
pub(crate) async fn ensure_model(
    repo: &crate::ModelId,
    models_root: &Path,
    gguf: Option<&str>,
) -> Result<PathBuf> {
    let dir = crate::model_dir(models_root, repo);
    if is_model_present(&dir) {
        return Ok(dir);
    }
    let include = match gguf {
        Some(file) => vec![file.to_string(), "*.json".to_string()],
        None => vec!["*.safetensors".to_string(), "*.json".to_string()],
    };
    let request = possum_lib::model::DownloadRequest {
        repository: repo.as_str().to_string(),
        to: dir.clone(),
        include,
        exclude: vec!["figures/*".to_string()],
        concurrency: 4,
        progress: possum_lib::model::ProgressMode::Auto,
        // Authenticate gated repos (e.g. Gemma) when `HF_TOKEN` is set; `None`
        // otherwise, so public repos download exactly as before.
        token: std::env::var("HF_TOKEN").ok(),
        ..Default::default()
    };
    possum_lib::model::download(&request)
        .await
        .map_err(|e| anyhow!("fetching {repo}: {e}"))?;
    let p = presence(&dir);
    if !p.complete {
        bail!(
            "model {repo} still incomplete after fetch at {} (missing: {:?}). \
             For a GGUF repo without tokenizer.json, place the .gguf + a \
             tokenizer.json in a dir and use --model.",
            dir.display(),
            p.missing
        );
    }
    Ok(dir)
}

/// Blocking wrapper around `ensure_model` for synchronous callers; drives the
/// async fetch through the library's single runtime bridge (RT-1).
#[cfg(feature = "fetch")]
pub fn ensure_model_blocking(
    repo: &crate::ModelId,
    models_root: &Path,
    gguf: Option<&str>,
) -> Result<PathBuf> {
    crate::runtime::block_on(ensure_model(repo, models_root, gguf))
}

/// Collect EOS token ids from the model config and (optional) generation
/// config. `eos_token_id` may be a single integer or an array; both are
/// gathered. Reading ids from config avoids hard-coding tokenizer-specific
/// EOS strings (e.g. DeepSeek's `<｜end▁of▁sentence｜>` = 151643).
fn extract_eos_ids(
    config: &serde_json::Value,
    gen_config: Option<&serde_json::Value>,
) -> HashSet<u32> {
    let mut ids = HashSet::new();
    for source in [gen_config, Some(config)].into_iter().flatten() {
        if let Some(eos) = source.get("eos_token_id") {
            collect_ids(eos, &mut ids);
        }
    }
    ids
}

fn collect_ids(value: &serde_json::Value, out: &mut HashSet<u32>) {
    match value {
        serde_json::Value::Number(n) => {
            if let Some(u) = n.as_u64() {
                out.insert(u as u32);
            }
        }
        serde_json::Value::Array(a) => {
            for v in a {
                collect_ids(v, out);
            }
        }
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gen_opts_defaults_are_greedy() {
        // upholds: SAM-2 (the default sampling is the deterministic, seed-free one)
        let o = GenOpts::default();
        assert_eq!(o.max_tokens, 256);
        assert_eq!(o.sampling, Sampling::Greedy);
    }

    #[test]
    fn detect_arch_maps_class_names() {
        // upholds: ARCH-1 — safetensors loading normalizes supported class names
        // onto the single public architecture enum before dispatch.
        for (class, want) in [
            ("Qwen2ForCausalLM", Arch::Qwen2),
            ("LlamaForCausalLM", Arch::Llama),
            ("MistralForCausalLM", Arch::Mistral),
            ("Phi3ForCausalLM", Arch::Phi3),
            ("Gemma2ForCausalLM", Arch::Gemma2),
            ("Starcoder2ForCausalLM", Arch::Starcoder2),
        ] {
            let cfg = serde_json::json!({ "architectures": [class] });
            assert_eq!(detect_arch(&cfg).unwrap(), want, "class {class}");
        }
    }

    #[test]
    fn detect_arch_falls_back_to_model_type() {
        // upholds: ARCH-1 — the fallback path still normalizes to the same
        // public architecture enum.
        // No `architectures`, only the short `model_type` form — the fallback
        // must be real for every family.
        for (mt, want) in [
            ("qwen2", Arch::Qwen2),
            ("llama", Arch::Llama),
            ("mistral", Arch::Mistral),
            ("phi3", Arch::Phi3),
            ("gemma2", Arch::Gemma2),
            ("starcoder2", Arch::Starcoder2),
        ] {
            let cfg = serde_json::json!({ "model_type": mt });
            assert_eq!(detect_arch(&cfg).unwrap(), want, "model_type {mt}");
        }
    }

    #[test]
    fn detect_arch_rejects_unknown_and_missing() {
        assert!(detect_arch(&serde_json::json!({ "architectures": ["FooForCausalLM"] })).is_err());
        assert!(detect_arch(&serde_json::json!({ "model_type": "foo" })).is_err());
        assert!(detect_arch(&serde_json::json!({})).is_err());
    }

    #[test]
    fn arch_from_gguf_normalizes_at_the_boundary() {
        // upholds: ARCH-2 — raw GGUF strings normalize to the public enum,
        // including the `chatglm` alias; unsupported strings are rejected.
        assert_eq!(arch_from_gguf("qwen2").unwrap(), Arch::Qwen2);
        assert_eq!(arch_from_gguf("llama").unwrap(), Arch::Llama);
        assert_eq!(arch_from_gguf("glm4").unwrap(), Arch::Glm4);
        assert_eq!(arch_from_gguf("chatglm").unwrap(), Arch::Glm4);
        assert!(arch_from_gguf("gemma2").is_err()); // no quantized loader
        assert!(arch_from_gguf("nope").is_err());
    }

    #[test]
    fn metal_prefill_chunk_is_glm4_only() {
        // upholds: PREFILL-1 — only GLM-4 carries a bounded Metal prefill default.
        assert_eq!(
            Arch::Glm4.metal_prefill_chunk(),
            Some(GLM4_METAL_PREFILL_CHUNK)
        );
        for arch in [
            Arch::Qwen2,
            Arch::Llama,
            Arch::Mistral,
            Arch::Phi3,
            Arch::Gemma2,
            Arch::Starcoder2,
        ] {
            assert_eq!(arch.metal_prefill_chunk(), None, "{arch:?}");
        }
    }

    #[test]
    fn sampling_from_temperature_maps_greedy_and_sample() {
        // upholds: SAM-1, SAM-2 — every user-facing temperature maps to one
        // explicit sampling algebra; non-positive temperature is greedy (seed
        // ignored).
        assert_eq!(Sampling::from_temperature(0.0, 7), Sampling::Greedy);
        assert_eq!(Sampling::from_temperature(-1.0, 7), Sampling::Greedy);
        assert_eq!(
            Sampling::from_temperature(0.8, 7),
            Sampling::Sample {
                temperature: 0.8,
                seed: 7
            }
        );
    }

    #[test]
    fn eos_from_generation_config_single_id() {
        // upholds: EOS-1
        let cfg = serde_json::json!({ "eos_token_id": 151643 });
        let gen = serde_json::json!({ "eos_token_id": 151643 });
        assert_eq!(extract_eos_ids(&cfg, Some(&gen)), HashSet::from([151643]));
    }

    #[test]
    fn eos_handles_array() {
        // upholds: EOS-1
        let cfg = serde_json::json!({ "eos_token_id": [151643, 151645] });
        assert_eq!(extract_eos_ids(&cfg, None), HashSet::from([151643, 151645]));
    }

    #[test]
    fn eos_empty_when_absent() {
        // upholds: EOS-1
        let cfg = serde_json::json!({ "hidden_size": 3584 });
        assert!(extract_eos_ids(&cfg, None).is_empty());
    }

    #[test]
    fn is_model_present_requires_all_indexed_shards() {
        // upholds: MD-3
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path();
        std::fs::write(p.join("config.json"), "{}").unwrap();
        std::fs::write(p.join("tokenizer.json"), "{}").unwrap();
        std::fs::write(
            p.join("model.safetensors.index.json"),
            r#"{"weight_map": {"a": "model-1.safetensors", "b": "model-2.safetensors"}}"#,
        )
        .unwrap();
        // Only one of the two referenced shards exists ⇒ not a cache hit.
        std::fs::write(p.join("model-1.safetensors"), "x").unwrap();
        assert!(!is_model_present(p));
        // Both present ⇒ present.
        std::fs::write(p.join("model-2.safetensors"), "x").unwrap();
        assert!(is_model_present(p));
    }

    #[test]
    fn is_model_present_false_without_config_or_tokenizer() {
        // upholds: MD-3
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path();
        std::fs::write(p.join("model.safetensors"), "x").unwrap();
        assert!(!is_model_present(p)); // missing config.json + tokenizer.json
    }

    #[test]
    fn is_model_present_unsharded_ok() {
        // upholds: MD-1
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path();
        std::fs::write(p.join("config.json"), "{}").unwrap();
        std::fs::write(p.join("tokenizer.json"), "{}").unwrap();
        std::fs::write(p.join("model.safetensors"), "x").unwrap();
        assert!(is_model_present(p));
    }

    #[test]
    fn gguf_dir_is_self_contained() {
        // A GGUF dir is complete on the strength of its *.gguf alone — the
        // tokenizer is built from the file's metadata, no tokenizer.json or
        // config.json required.
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path();
        std::fs::write(p.join("model.gguf"), "x").unwrap();
        assert_eq!(gguf_in(p), Some(p.join("model.gguf")));
        assert!(is_model_present(p)); // no tokenizer.json / config.json needed
    }

    #[test]
    fn safetensors_dir_is_not_gguf() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path();
        std::fs::write(p.join("config.json"), "{}").unwrap();
        std::fs::write(p.join("model.safetensors"), "x").unwrap();
        assert_eq!(gguf_in(p), None);
    }

    #[test]
    fn model_shards_rejects_escaping_index() {
        // upholds: MS-3
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path();
        std::fs::write(
            p.join("model.safetensors.index.json"),
            r#"{"weight_map": {"a": "../evil.safetensors"}}"#,
        )
        .unwrap();
        assert!(model_shards(p).is_err());
    }

    #[test]
    fn model_shards_dedups_and_sorts_indexed() {
        // upholds: MD-2 / DISC — duplicate shard refs collapse; order is stable.
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path();
        std::fs::write(
            p.join("model.safetensors.index.json"),
            r#"{"weight_map": {"a": "m-2.safetensors", "b": "m-1.safetensors", "c": "m-2.safetensors"}}"#,
        )
        .unwrap();
        let shards = model_shards(p).unwrap();
        assert_eq!(
            shards,
            vec![p.join("m-1.safetensors"), p.join("m-2.safetensors")],
        );
    }

    #[test]
    fn presence_reports_missing_shards() {
        // upholds: MD-3, FETCH-1 — the completeness predicate used after fetch
        // rejects a partial shard set before it can reach Engine::load.
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path();
        std::fs::write(p.join("config.json"), "{}").unwrap();
        std::fs::write(p.join("tokenizer.json"), "{}").unwrap();
        std::fs::write(
            p.join("model.safetensors.index.json"),
            r#"{"weight_map": {"a": "model-1.safetensors", "b": "model-2.safetensors"}}"#,
        )
        .unwrap();
        std::fs::write(p.join("model-1.safetensors"), "x").unwrap();
        let pres = presence(p);
        assert!(!pres.complete);
        assert!(pres
            .missing
            .iter()
            .any(|m| m.ends_with("model-2.safetensors")));
    }

    // End-to-end inference over the real model. Gated: needs the ~15 GB
    // weights and `YATIMA_E2E=1`; skips fast otherwise so CI stays green.
    #[test]
    fn e2e_generate_is_deterministic_at_temp_zero() -> Result<()> {
        // upholds: GE-1, GEN-3, STOP-1
        if std::env::var_os("YATIMA_E2E").is_none() {
            eprintln!("skipping e2e: set YATIMA_E2E=1 to run");
            return Ok(());
        }
        let repo = crate::ModelId::parse("deepseek-ai/DeepSeek-R1-Distill-Qwen-7B").unwrap();
        let dir = crate::model_dir(&crate::models_root(), &repo);
        if !dir.join("config.json").exists() {
            eprintln!("skipping e2e: weights absent at {}", dir.display());
            return Ok(());
        }

        let mut engine = Engine::load(&dir, device(false)?)?;
        let opts = GenOpts {
            max_tokens: 16,
            sampling: Sampling::Greedy,
            ..Default::default()
        };
        let prompt = "Rust is a systems programming language that";

        let run = |engine: &mut Engine| -> Result<(String, Generation)> {
            let mut out = String::new();
            let generation = engine.generate(prompt, &opts, |s| {
                out.push_str(s);
                Ok(())
            })?;
            Ok((out, generation))
        };

        let (first, gen1) = run(&mut engine)?;
        assert!(!first.trim().is_empty(), "expected a non-empty completion");
        assert!(gen1.tokens <= 16, "GEN-3: tokens never exceed max_tokens");
        assert!(
            matches!(gen1.stop, StopReason::Eos | StopReason::MaxTokens),
            "STOP-1: greedy run stops by EOS or max_tokens"
        );
        let (second, _) = run(&mut engine)?;
        assert_eq!(
            first, second,
            "GE-1: greedy generation must be deterministic"
        );
        Ok(())
    }

    // Per-family load+generate over the real models. Gated on `YATIMA_E2E=1`;
    // each family is skipped if its weights aren't cached, so this passes
    // whether one or all six are present. One representative model per arch.
    #[test]
    fn e2e_each_architecture_loads_and_generates() -> Result<()> {
        if std::env::var_os("YATIMA_E2E").is_none() {
            eprintln!("skipping e2e: set YATIMA_E2E=1 to run");
            return Ok(());
        }
        // Safetensors families + one GGUF (quantized) repo; the GGUF entry also
        // exercises the quantized load path. Presence is layout-aware, so any
        // absent model is skipped.
        let models = [
            "Qwen/Qwen2.5-7B-Instruct",
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "mistralai/Mistral-7B-Instruct-v0.3",
            "microsoft/Phi-3-mini-4k-instruct",
            "google/gemma-2-2b-it",
            "bigcode/starcoder2-3b",
            "zai-org/GLM-4-9B-0414",
            "bartowski/Qwen2.5-32B-Instruct-GGUF",
        ];

        let mut ran = 0;
        for repo in models {
            let id = crate::ModelId::parse(repo).unwrap();
            let dir = crate::model_dir(&crate::models_root(), &id);
            if !is_model_present(&dir) {
                eprintln!("skip {repo}: not cached");
                continue;
            }
            eprintln!("e2e {repo} …");
            let mut engine = Engine::load(&dir, device(false)?)?;
            let opts = GenOpts {
                max_tokens: 12,
                sampling: Sampling::Greedy,
                ..Default::default()
            };
            let mut out = String::new();
            let gen = engine.generate("Rust is", &opts, |s| {
                out.push_str(s);
                Ok(())
            })?;
            assert!(!out.trim().is_empty(), "{repo}: empty completion");
            assert!(gen.tokens <= 12, "{repo}: GEN-3 tokens ≤ max_tokens");
            assert!(
                matches!(gen.stop, StopReason::Eos | StopReason::MaxTokens),
                "{repo}: STOP-1"
            );
            eprintln!("  ok: {:?}", out.trim());
            ran += 1;
        }
        eprintln!("e2e ran {ran}/6 families");
        Ok(())
    }

    // Proves the Engine's `complete_streaming` override is *real* streaming: a
    // multi-token answer must reach `on_token` in more than one call (the trait's
    // default impl would deliver the whole text in a single call). Gated, skips
    // fast if the weights aren't cached.
    #[tokio::test(flavor = "multi_thread")]
    async fn e2e_engine_streams_in_multiple_chunks() -> Result<()> {
        use crate::completer::Completer;
        if std::env::var_os("YATIMA_E2E").is_none() {
            eprintln!("skipping e2e: set YATIMA_E2E=1 to run");
            return Ok(());
        }
        let repo = crate::ModelId::parse("deepseek-ai/DeepSeek-R1-Distill-Qwen-7B").unwrap();
        let dir = crate::model_dir(&crate::models_root(), &repo);
        if !is_model_present(&dir) {
            eprintln!("skipping e2e: weights absent at {}", dir.display());
            return Ok(());
        }

        let mut engine = Engine::load(&dir, device(false)?)?;
        let opts = GenOpts {
            max_tokens: 16,
            sampling: Sampling::Greedy,
            ..Default::default()
        };
        let mut calls = 0usize;
        let mut acc = String::new();
        let completion = engine
            .complete_streaming(
                "Rust is a systems programming language that",
                &opts,
                &[],
                &mut |piece| {
                    calls += 1;
                    acc.push_str(piece);
                },
            )
            .await?;
        assert!(calls > 1, "expected >1 streamed chunk, got {calls}");
        assert_eq!(acc, completion.text, "streamed pieces reconstruct the text");
        Ok(())
    }
}
