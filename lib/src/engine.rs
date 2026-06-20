//! Inference as an in-process library function.
//!
//! `Engine::load` reads a local model directory (HF-agnostic) and
//! `Engine::generate` runs a stateless, raw-completion generation loop,
//! streaming decoded text fragments to a callback. The engine rents candle's
//! Qwen2 implementation; we own the load/generate boundary.

use std::collections::HashSet;
use std::ops::ControlFlow;
use std::path::{Path, PathBuf};

use anyhow::{anyhow, bail, Context, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::generation::LogitsProcessor;
use candle_transformers::models::qwen2::{Config, ModelForCausalLM};
use candle_transformers::utils::apply_repeat_penalty;
use tokenizers::Tokenizer;

use crate::token_output_stream::TokenOutputStream;

/// Match the upstream qwen example's anti-repetition defaults; without them a
/// temperature-0 raw completion of an instruction-tuned distill degenerates.
const REPEAT_PENALTY: f32 = 1.1;
const REPEAT_LAST_N: usize = 64;

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
}

/// Options for a single generation.
#[derive(Debug, Clone)]
pub struct GenOpts {
    pub max_tokens: usize,
    pub sampling: Sampling,
}

impl Default for GenOpts {
    fn default() -> Self {
        Self {
            max_tokens: 256,
            sampling: Sampling::Greedy,
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

/// A loaded model ready to generate. Construct with [`Engine::load`].
pub struct Engine {
    model: ModelForCausalLM,
    tokenizer: Tokenizer,
    device: Device,
    eos: HashSet<u32>,
    dtype: DType,
}

impl Engine {
    /// Load weights + tokenizer from a local model directory.
    ///
    /// Strict discovery: `config.json` and `tokenizer.json` are required; all
    /// `*.safetensors` in the directory are loaded (the sharded case). Fails
    /// loudly otherwise. EOS ids are read from the config(s), not hard-coded.
    pub fn load(model_dir: &Path, device: Device) -> Result<Self> {
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
        let config: Config = serde_json::from_slice(&config_bytes)
            .context("parsing config.json as a Qwen2 config")?;

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
        let model = ModelForCausalLM::new(&config, vb)?;

        Ok(Self {
            model,
            tokenizer,
            device,
            eos,
            dtype,
        })
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

    /// Generation as an **effectful fold** over the stream of decoded text
    /// fragments — the primitive of which [`generate`](Engine::generate) is the
    /// `acc = ()` specialization. Stateless request-reply (the KV cache is
    /// cleared on entry); raw completion (prompt fed as-is, no chat template).
    ///
    /// `step` threads an accumulator and returns a [`ControlFlow`]:
    /// `Continue(acc)` keeps folding, `Break(acc)` stops voluntarily
    /// (`StopReason::Stopped`), and an `Err` is propagated to the caller.
    /// Generation also stops on EOS or `max_tokens`. Conceptually a
    /// hylomorphism: an unfold of fragments (terminating on EOS/max/break)
    /// consumed by `step` (see `axiom::fix` for the recursion-scheme vocabulary).
    pub fn generate_fold<A>(
        &mut self,
        prompt: &str,
        opts: &GenOpts,
        init: A,
        mut step: impl FnMut(A, &str) -> Result<ControlFlow<A, A>>,
    ) -> Result<(A, Generation)> {
        self.model.clear_kv_cache();

        let mut stream = TokenOutputStream::new(self.tokenizer.clone());
        let mut tokens = self
            .tokenizer
            .encode(prompt, true)
            .map_err(|e| anyhow!("tokenizing prompt: {e}"))?
            .get_ids()
            .to_vec();

        let mut logits_processor = opts.sampling.logits_processor();
        let mut acc = init;
        let mut generated = 0usize;
        let mut stop = StopReason::MaxTokens;

        for index in 0..opts.max_tokens {
            // Prefill the whole prompt on the first step, then feed one token
            // at a time; the model advances its own KV cache via `start_pos`.
            let context_size = if index > 0 { 1 } else { tokens.len() };
            let start_pos = tokens.len().saturating_sub(context_size);
            let ctxt = &tokens[start_pos..];
            let input = Tensor::new(ctxt, &self.device)?.unsqueeze(0)?;
            let logits = self.model.forward(&input, start_pos)?;
            let logits = logits.squeeze(0)?.squeeze(0)?.to_dtype(DType::F32)?;
            let logits = if REPEAT_PENALTY == 1.0 {
                logits
            } else {
                let start = tokens.len().saturating_sub(REPEAT_LAST_N);
                apply_repeat_penalty(&logits, REPEAT_PENALTY, &tokens[start..])?
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

        // Flush trailing buffered text — but not when the caller asked to stop.
        if stop != StopReason::Stopped {
            if let Some(rest) = stream.decode_rest()? {
                acc = match step(acc, &rest)? {
                    ControlFlow::Continue(a) | ControlFlow::Break(a) => a,
                };
            }
        }

        Ok((
            acc,
            Generation {
                tokens: generated,
                stop,
            },
        ))
    }

    /// Run inference, streaming decoded text fragments to `on_token` (the
    /// `acc = ()` specialization of [`generate_fold`](Engine::generate_fold)).
    /// Returning `Err` from `on_token` stops generation and is surfaced.
    pub fn generate(
        &mut self,
        prompt: &str,
        opts: &GenOpts,
        mut on_token: impl FnMut(&str) -> Result<()>,
    ) -> Result<Generation> {
        let ((), generation) = self.generate_fold(prompt, opts, (), |(), piece| {
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
pub fn model_shards(dir: &Path) -> Result<Vec<PathBuf>> {
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
pub struct Presence {
    pub complete: bool,
    pub missing: Vec<PathBuf>,
}

/// Is `dir` a loadable model, and what's missing?
///
/// `complete` is the conjunction over the required files — `config.json`,
/// `tokenizer.json`, and every shard from [`model_shards`] — computed with
/// axiom's `bool` meet (the `All` lattice; ⊤ = true is the identity), so a
/// partial shard set is never a false cache hit.
pub fn presence(dir: &Path) -> Presence {
    use axiom::{BoundedMeetSemilattice, MeetSemilattice};

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

    // presence = ⋀ over required files (bool meet = AND; ⊤ = identity).
    let mut complete = bool::top();
    let mut missing = Vec::new();
    for path in required {
        let exists = path.exists();
        complete = complete.meet(&exists);
        if !exists {
            missing.push(path);
        }
    }
    Presence { complete, missing }
}

/// Whether `dir` holds a loadable model (the `complete` flag of [`presence`]).
pub fn is_model_present(dir: &Path) -> bool {
    presence(dir).complete
}

/// Ensure the weights for `repo` are present under `models_root`, fetching them
/// with possum on a cache miss; returns the model directory. Re-checks
/// completeness after download so a partial fetch is never handed to
/// [`Engine::load`].
#[cfg(feature = "fetch")]
pub async fn ensure_model(repo: &crate::RepoId, models_root: &Path) -> Result<PathBuf> {
    let dir = crate::model_dir(models_root, repo);
    if is_model_present(&dir) {
        return Ok(dir);
    }
    let request = possum_lib::model::DownloadRequest {
        repository: repo.as_str().to_string(),
        to: dir.clone(),
        include: vec!["*.safetensors".to_string(), "*.json".to_string()],
        exclude: vec!["figures/*".to_string()],
        concurrency: 4,
        progress: possum_lib::model::ProgressMode::Auto,
        ..Default::default()
    };
    possum_lib::model::download(&request)
        .await
        .map_err(|e| anyhow!("fetching {repo}: {e}"))?;
    let p = presence(&dir);
    if !p.complete {
        bail!(
            "model {repo} still incomplete after fetch at {} (missing: {:?})",
            dir.display(),
            p.missing
        );
    }
    Ok(dir)
}

/// Blocking wrapper around [`ensure_model`] for synchronous callers (the CLI);
/// drives the async fetch on a private tokio runtime.
#[cfg(feature = "fetch")]
pub fn ensure_model_blocking(repo: &crate::RepoId, models_root: &Path) -> Result<PathBuf> {
    let runtime = tokio::runtime::Runtime::new()?;
    runtime.block_on(ensure_model(repo, models_root))
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
        let o = GenOpts::default();
        assert_eq!(o.max_tokens, 256);
        assert_eq!(o.sampling, Sampling::Greedy);
    }

    #[test]
    fn eos_from_generation_config_single_id() {
        let cfg = serde_json::json!({ "eos_token_id": 151643 });
        let gen = serde_json::json!({ "eos_token_id": 151643 });
        assert_eq!(extract_eos_ids(&cfg, Some(&gen)), HashSet::from([151643]));
    }

    #[test]
    fn eos_handles_array() {
        let cfg = serde_json::json!({ "eos_token_id": [151643, 151645] });
        assert_eq!(extract_eos_ids(&cfg, None), HashSet::from([151643, 151645]));
    }

    #[test]
    fn eos_empty_when_absent() {
        let cfg = serde_json::json!({ "hidden_size": 3584 });
        assert!(extract_eos_ids(&cfg, None).is_empty());
    }

    #[test]
    fn is_model_present_requires_all_indexed_shards() {
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
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path();
        std::fs::write(p.join("model.safetensors"), "x").unwrap();
        assert!(!is_model_present(p)); // missing config.json + tokenizer.json
    }

    #[test]
    fn is_model_present_unsharded_ok() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path();
        std::fs::write(p.join("config.json"), "{}").unwrap();
        std::fs::write(p.join("tokenizer.json"), "{}").unwrap();
        std::fs::write(p.join("model.safetensors"), "x").unwrap();
        assert!(is_model_present(p));
    }

    #[test]
    fn model_shards_rejects_escaping_index() {
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
    fn presence_reports_missing_shards() {
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
        if std::env::var_os("YATIMA_E2E").is_none() {
            eprintln!("skipping e2e: set YATIMA_E2E=1 to run");
            return Ok(());
        }
        let repo = crate::RepoId::parse("deepseek-ai/DeepSeek-R1-Distill-Qwen-7B").unwrap();
        let dir = crate::model_dir(&crate::models_root(), &repo);
        if !dir.join("config.json").exists() {
            eprintln!("skipping e2e: weights absent at {}", dir.display());
            return Ok(());
        }

        let mut engine = Engine::load(&dir, device(false)?)?;
        let opts = GenOpts {
            max_tokens: 16,
            sampling: Sampling::Greedy,
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
        assert_eq!(first, second, "greedy generation must be deterministic");
        Ok(())
    }
}
