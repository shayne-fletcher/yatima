//! Named model profiles: a data-like description of "which model, and how to
//! run it" so a host can say `--profile glm4-32b` instead of repeating a repo
//! id, quant filename, format, and generation knobs on every command. Profiles
//! are serde-ready (a config-file loader is a trivial later add) and currently
//! sourced from a small compiled-in [`builtin`] registry.

use crate::{ChatFormat, GenOpts, ModelSource, Sampling};
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// A model and its run configuration. Every field beyond `name` is optional:
/// an unset field falls back to the loaded engine's default (e.g.
/// `prefill_chunk` → [`crate::Engine::default_prefill_chunk`]) or a caller base
/// [`GenOpts`], so a profile is a *layer of overrides*, never a full snapshot.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct ModelProfile {
    /// The profile's name (the `--profile` key).
    pub name: String,
    /// A repository id, resolved (and fetched on a cache miss) via [`ModelSource`].
    pub repo: Option<String>,
    /// An explicit local model directory (mutually exclusive with `repo`).
    pub dir: Option<PathBuf>,
    /// The single GGUF quant to fetch for a `repo` (ignored once cached: the
    /// loader finds the `.gguf` in the directory).
    pub gguf: Option<String>,
    /// The chat format; `None` infers it from the model's architecture.
    pub format: Option<ChatFormat>,
    /// Prompt prefill chunk override; `None` keeps the engine's device default.
    pub prefill_chunk: Option<usize>,
    pub max_tokens: Option<usize>,
    pub temperature: Option<f64>,
    pub seed: Option<u64>,
    pub repeat_penalty: Option<f32>,
}

impl ModelProfile {
    /// Look up a built-in profile by name. The registry is data-like (repo id +
    /// recommended quant + format), so it stays portable — no machine-specific
    /// paths — and `repo`-based profiles resolve via the cache (no fetch on a
    /// hit). Returns `None` for an unknown name.
    pub fn builtin(name: &str) -> Option<ModelProfile> {
        let p = |repo: &str, gguf: Option<&str>, format: ChatFormat| ModelProfile {
            name: name.to_string(),
            repo: Some(repo.to_string()),
            gguf: gguf.map(str::to_string),
            format: Some(format),
            ..Default::default()
        };
        let profile = match name {
            "qwen32b" => p(
                "bartowski/Qwen2.5-32B-Instruct-GGUF",
                Some("Qwen2.5-32B-Instruct-Q4_K_M.gguf"),
                ChatFormat::Qwen,
            ),
            // Kimi-Dev-72B is a Qwen2.5-72B finetune (GGUF arch `qwen2`,
            // ChatML/Qwen template). The K-quants (Q4_K_M/Q5_K_M/Q6_K) ship as
            // 2-part split GGUFs that the single-file loader can't take, so this
            // pins the largest single-file 4-bit (Q4_1, ~45.7 GB).
            "kimi-dev" => p(
                "unsloth/Kimi-Dev-72B-GGUF",
                Some("Kimi-Dev-72B-Q4_1.gguf"),
                ChatFormat::Qwen,
            ),
            "glm4-32b" => p(
                "bartowski/THUDM_GLM-4-32B-0414-GGUF",
                Some("THUDM_GLM-4-32B-0414-Q6_K_L.gguf"),
                ChatFormat::Glm,
            ),
            "gemma2" => p("google/gemma-2-2b-it", None, ChatFormat::Gemma),
            "mistral" => p(
                "mistralai/Mistral-7B-Instruct-v0.3",
                None,
                ChatFormat::Mistral,
            ),
            _ => return None,
        };
        Some(profile)
    }

    /// The names of every built-in profile (for `--help` / listing).
    pub const BUILTIN_NAMES: [&'static str; 5] =
        ["qwen32b", "glm4-32b", "gemma2", "mistral", "kimi-dev"];

    /// The model source this profile names — a directory **xor** a repository
    /// (PROFILE-2, via [`ModelSource::from_args`]).
    pub fn to_source(&self, offline: bool) -> Result<ModelSource> {
        ModelSource::from_args(
            self.dir.clone(),
            self.repo.clone(),
            None,
            offline,
            self.gguf.clone(),
        )
    }

    /// Layer this profile's set fields over a caller-built `base` (PROFILE-1:
    /// the caller chooses the use-case base — chat keeps [`GenOpts::default`],
    /// the agent sets `repeat_penalty = 1.0`). `prefill_chunk` is override-only:
    /// an unset profile leaves `base.prefill_chunk` untouched (typically `None`),
    /// so the loaded engine's device-aware default wins (PREFILL-1). Pure.
    pub fn apply_gen_overrides(&self, base: GenOpts) -> GenOpts {
        let mut opts = base;
        if let Some(max_tokens) = self.max_tokens {
            opts.max_tokens = max_tokens;
        }
        if let Some(repeat_penalty) = self.repeat_penalty {
            opts.repeat_penalty = repeat_penalty;
        }
        if let Some(temperature) = self.temperature {
            opts.sampling = Sampling::from_temperature(temperature, self.seed.unwrap_or(0));
        }
        if let Some(prefill_chunk) = self.prefill_chunk {
            opts.prefill_chunk = Some(prefill_chunk);
        }
        opts
    }

    /// The chat format this profile pins, if any (`None` defers to architecture
    /// inference via [`crate::resolve_format`]).
    pub fn format(&self) -> Option<ChatFormat> {
        self.format
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builtin_lookup() {
        let glm = ModelProfile::builtin("glm4-32b").expect("glm4-32b is built in");
        assert_eq!(glm.format, Some(ChatFormat::Glm));
        assert!(glm.repo.as_deref().unwrap().contains("GLM-4-32B"));
        assert!(ModelProfile::builtin("nope").is_none());
        for name in ModelProfile::BUILTIN_NAMES {
            assert!(ModelProfile::builtin(name).is_some(), "{name}");
        }
    }

    #[test]
    fn to_source_is_repo_xor_dir() {
        // upholds: PROFILE-2 — a profile resolves to exactly one source.
        let repo = ModelProfile::builtin("gemma2").unwrap();
        assert!(repo.to_source(false).is_ok());

        let dir = ModelProfile {
            name: "local".into(),
            dir: Some(PathBuf::from("/models/x")),
            ..Default::default()
        };
        assert!(dir.to_source(false).is_ok());

        let both = ModelProfile {
            name: "bad".into(),
            repo: Some("org/name".into()),
            dir: Some(PathBuf::from("/models/x")),
            ..Default::default()
        };
        assert!(both.to_source(false).is_err());

        let neither = ModelProfile {
            name: "bad".into(),
            ..Default::default()
        };
        assert!(neither.to_source(false).is_err());
    }

    #[test]
    fn overrides_layer_over_base_and_default_to_engine() {
        // upholds: PROFILE-1 — set fields override the base, in declared precedence.
        let profile = ModelProfile {
            name: "x".into(),
            max_tokens: Some(512),
            temperature: Some(0.7),
            seed: Some(9),
            repeat_penalty: Some(1.0),
            prefill_chunk: Some(32),
            ..Default::default()
        };
        let opts = profile.apply_gen_overrides(GenOpts::default());
        assert_eq!(opts.max_tokens, 512);
        assert_eq!(opts.repeat_penalty, 1.0);
        assert_eq!(
            opts.sampling,
            Sampling::Sample {
                temperature: 0.7,
                seed: 9
            }
        );
        assert_eq!(opts.prefill_chunk, Some(32));
    }

    #[test]
    fn empty_profile_leaves_base_untouched() {
        // upholds: PREFILL-1 — an unset prefill leaves base None so the engine
        // device default wins; other unset fields keep the base.
        let base = GenOpts::default();
        let opts = ModelProfile {
            name: "x".into(),
            ..Default::default()
        }
        .apply_gen_overrides(base.clone());
        assert_eq!(opts.prefill_chunk, None);
        assert_eq!(opts.max_tokens, base.max_tokens);
        assert_eq!(opts.sampling, base.sampling);
        assert_eq!(opts.repeat_penalty, base.repeat_penalty);
    }

    #[test]
    fn serde_round_trips() {
        // serde-ready: a profile survives a JSON/TOML-shaped round trip.
        let profile = ModelProfile::builtin("qwen32b").unwrap();
        let json = serde_json::to_string(&profile).unwrap();
        let back: ModelProfile = serde_json::from_str(&json).unwrap();
        assert_eq!(profile, back);
    }
}
