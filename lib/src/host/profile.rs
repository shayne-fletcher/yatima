//! Named model profiles: a data-like description of "which model, and how to
//! run it" so a host can say `--profile glm4-32b` instead of repeating a repo
//! id, quant filename, format, and generation knobs on every command. Profiles
//! are serde-ready (a config-file loader is a trivial later add) and currently
//! sourced from a small compiled-in [`builtin`] registry.

use crate::{ChatFormat, GenOpts, ModelSource, Sampling, ServerGates, Sha256Digest};
use anyhow::{bail, Result};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// The minimum token budget a reasoning profile guarantees: enough headroom for
/// a think block *and* an answer. Reasoning models spend the default 256-token
/// budget mid-thought, never reaching the answer; a reasoning profile floors the
/// budget here (raising it only — never reducing a larger caller budget).
pub const REASONING_MIN_TOKENS: usize = 2048;

/// Which inference implementation a profile requires. The default preserves
/// existing profiles: they load through the in-process Candle engine.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum ProfileBackend {
    #[default]
    Engine,
    LlamaServer(LlamaServerProfile),
}

/// The llama-server-specific part of a profile's launch recipe.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LlamaServerProfile {
    pub expected_sha256: Sha256Digest,
    pub build_floor: u32,
    pub template_sha256: Sha256Digest,
    pub context: u32,
    pub top_k: u32,
}

impl LlamaServerProfile {
    /// Compatibility gates for the managed process. Profiles use one slot so
    /// the declared context is not divided among concurrent server slots.
    pub fn server_gates(&self) -> ServerGates {
        ServerGates::new(self.build_floor, self.template_sha256, self.context, 1)
    }
}

/// A model and its run configuration. The backend defaults to the local engine;
/// unset generation fields fall back to the loaded engine's default or a caller
/// base [`GenOpts`], so a profile is a layer of overrides rather than a full
/// runtime snapshot.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct ModelProfile {
    /// The profile's name (the `--profile` key).
    pub name: String,
    /// The inference implementation this profile requires.
    #[serde(default)]
    pub backend: ProfileBackend,
    /// A repository id, resolved (and fetched on a cache miss) via [`ModelSource`].
    pub repo: Option<String>,
    /// An explicit local model directory (mutually exclusive with `repo`).
    pub dir: Option<PathBuf>,
    /// The exact GGUF quant to fetch and resolve for a `repo`.
    pub gguf: Option<String>,
    /// The chat format; `None` infers it from the model's architecture.
    pub format: Option<ChatFormat>,
    /// Whether this is a reasoning model (emits a chain-of-thought before its
    /// answer). When set, [`apply_gen_overrides`](ModelProfile::apply_gen_overrides)
    /// raises `max_tokens` to at least [`REASONING_MIN_TOKENS`] so the think
    /// block is not truncated, and a host knows to surface the reasoning channel.
    #[serde(default)]
    pub reasoning: bool,
    /// Prompt prefill chunk override; `None` keeps the engine's device default.
    pub prefill_chunk: Option<usize>,
    pub max_tokens: Option<usize>,
    pub temperature: Option<f64>,
    /// Nucleus (top-p) sampling cutoff; `None` samples the full distribution.
    /// Reasoning models want this (e.g. 0.95) to curb repetition degeneration.
    #[serde(default)]
    pub top_p: Option<f64>,
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
            // ChatML/Qwen template). candle reads no i-quants and the community
            // K-quants embed IQ4_NL, so this pins the legacy Q4_0 (~41.4 GB;
            // dtypes F32/Q4_0/Q4_1/Q6_K — verified candle-loadable and confirmed
            // running on Metal). On a 48 GB Mac, load it with
            // `sudo sysctl iogpu.wired_limit_mb=46080` (default budget is too low
            // for 41 GB of weights).
            "kimi-dev" => ModelProfile {
                reasoning: true,
                // Reasoning models want temperature + nucleus sampling, not greedy
                // (greedy/full-dist collapses into repetition); ~0.6 + top-p 0.95
                // is the family's recommended setting.
                temperature: Some(0.6),
                top_p: Some(0.95),
                ..p(
                    "unsloth/Kimi-Dev-72B-GGUF",
                    Some("Kimi-Dev-72B-Q4_0.gguf"),
                    ChatFormat::Qwen,
                )
            },
            // DeepSeek-R1-Distill-Qwen-7B: a Qwen2-arch distill trained on
            // DeepSeek's format, so it pins `deepseek` explicitly (the arch alone
            // would say Qwen/ChatML). A reasoning model (`<think>` dialect).
            // QwQ-32B: Qwen's reasoning model (ChatML + `<think>`). Q4_K_M GGUF
            // (~20 GB, F32/Q4_K/Q6_K — candle-loadable) fits a 48 GB machine with
            // headroom, unlike Kimi. A strong reasoning model that actually runs.
            "qwq" => ModelProfile {
                reasoning: true,
                temperature: Some(0.6),
                top_p: Some(0.95),
                // QwQ is exceptionally verbose: it routinely thinks for >2k tokens
                // before answering, so the 2048 reasoning floor would truncate it
                // mid-thought (no answer). Give it room to actually finish.
                max_tokens: Some(4096),
                ..p(
                    "bartowski/Qwen_QwQ-32B-GGUF",
                    Some("Qwen_QwQ-32B-Q4_K_M.gguf"),
                    // QwenThink, not Qwen: QwQ's template pre-seeds `<think>`, so
                    // its output carries only the close marker — the seeded
                    // splitter classifies it (a plain-Qwen format would mis-show
                    // the reasoning as the answer).
                    ChatFormat::QwenThink,
                )
            },
            "deepseek-r1" => ModelProfile {
                reasoning: true,
                // DeepSeek-R1's own recommendation: temperature 0.6 + top-p 0.95.
                temperature: Some(0.6),
                top_p: Some(0.95),
                ..p(
                    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
                    None,
                    ChatFormat::DeepSeek,
                )
            },
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
            "muse-glimmer" => ModelProfile {
                backend: ProfileBackend::LlamaServer(LlamaServerProfile {
                    expected_sha256:
                        "4cc57c0f51040a226e5a72cc47b7613f7772950e460a665f7083de89f183f60e"
                            .parse()
                            .expect("built-in Muse artifact digest is valid"),
                    build_floor: 10353,
                    template_sha256:
                        "6cd0e94d9b489fdb9a000743791f20e7e52b62c1e5cbb66fcc91c6716595223a"
                            .parse()
                            .expect("built-in Muse template digest is valid"),
                    context: 131_072,
                    top_k: 64,
                }),
                reasoning: true,
                max_tokens: Some(4096),
                temperature: Some(1.0),
                top_p: Some(0.95),
                ..p(
                    "meta-models/Muse-Glimmer-30B-GGUF",
                    Some("Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf"),
                    ChatFormat::MuseGlimmer,
                )
            },
            _ => return None,
        };
        Some(profile)
    }

    /// The names of every built-in profile (for `--help` / listing).
    pub const BUILTIN_NAMES: [&'static str; 8] = [
        "qwen32b",
        "glm4-32b",
        "gemma2",
        "mistral",
        "kimi-dev",
        "deepseek-r1",
        "qwq",
        "muse-glimmer",
    ];

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

    /// Resolve this profile only when it is compatible with the in-process
    /// Candle engine. Native frontends use this boundary until they can host
    /// alternate backends themselves.
    pub fn to_engine_source(&self, offline: bool) -> Result<ModelSource> {
        if matches!(self.backend, ProfileBackend::LlamaServer(_)) {
            bail!(
                "profile {:?} is served by a managed llama-server; supported by `yatima chat` only until stage 5",
                self.name
            );
        }
        self.to_source(offline)
    }

    /// Layer this profile's set fields over a caller-built `base` (PROFILE-1:
    /// the caller chooses the use-case base — chat keeps [`GenOpts::default`],
    /// the agent sets `repeat_penalty = 1.0`). `prefill_chunk` is override-only:
    /// an unset profile leaves `base.prefill_chunk` untouched (typically `None`),
    /// so the loaded engine's device-aware default wins (PREFILL-1). Pure.
    pub fn apply_gen_overrides(&self, base: GenOpts) -> GenOpts {
        let mut opts = base;
        if let Some(max_tokens) = self.max_tokens {
            opts.max_tokens = if self.reasoning {
                opts.max_tokens.max(max_tokens)
            } else {
                max_tokens
            };
        }
        if let Some(repeat_penalty) = self.repeat_penalty {
            opts.repeat_penalty = repeat_penalty;
        }
        // Sampling overrides compose *per component* over the base, so a profile
        // that sets only `temperature` keeps the caller's `--seed`/`--top-p`
        // rather than silently resetting them (the seed-drop bug). Rebuild from
        // the resolved (temperature, top_p, seed).
        let (base_temp, base_top_p, base_seed) = match opts.sampling {
            Sampling::Greedy => (0.0, None, 0),
            Sampling::Sample {
                temperature,
                top_p,
                seed,
            } => (temperature, top_p, seed),
        };
        if self.temperature.is_some() || self.top_p.is_some() || self.seed.is_some() {
            opts.sampling = Sampling::nucleus(
                self.temperature.unwrap_or(base_temp),
                self.top_p.or(base_top_p),
                self.seed.unwrap_or(base_seed),
            );
        }
        if let Some(prefill_chunk) = self.prefill_chunk {
            opts.prefill_chunk = Some(prefill_chunk);
        }
        // A reasoning model needs room to think *and* answer: floor the budget so
        // the think block is never truncated, raising it only (a deliberately
        // larger caller/profile budget is kept).
        if self.reasoning {
            opts.max_tokens = opts.max_tokens.max(REASONING_MIN_TOKENS);
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
        assert_eq!(glm.backend, ProfileBackend::Engine);
        assert!(glm.repo.as_deref().unwrap().contains("GLM-4-32B"));
        assert!(ModelProfile::builtin("nope").is_none());
        for name in ModelProfile::BUILTIN_NAMES {
            assert!(ModelProfile::builtin(name).is_some(), "{name}");
        }
    }

    #[test]
    fn muse_glimmer_pins_the_verified_llama_server_recipe() {
        // upholds: LSRV-5 — the built-in recipe supplies the artifact digest
        // and every compatibility gate required by verified construction.
        let profile = ModelProfile::builtin("muse-glimmer").unwrap();
        assert_eq!(profile.format, Some(ChatFormat::MuseGlimmer));
        assert_eq!(
            profile.repo.as_deref(),
            Some("meta-models/Muse-Glimmer-30B-GGUF")
        );
        assert_eq!(
            profile.gguf.as_deref(),
            Some("Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf")
        );
        assert!(profile.reasoning);
        assert_eq!(profile.max_tokens, Some(4096));
        assert_eq!(profile.temperature, Some(1.0));
        assert_eq!(profile.top_p, Some(0.95));
        let opts = profile.apply_gen_overrides(GenOpts::default());
        assert_eq!(opts.max_tokens, 4096);
        assert_eq!(
            opts.sampling,
            Sampling::Sample {
                temperature: 1.0,
                top_p: Some(0.95),
                seed: 0,
            }
        );
        let ProfileBackend::LlamaServer(server) = profile.backend else {
            panic!("Muse must select llama-server structurally");
        };
        assert_eq!(server.build_floor, 10353);
        assert_eq!(server.context, 131_072);
        assert_eq!(server.top_k, 64);
        assert_eq!(
            server.expected_sha256.to_string(),
            "4cc57c0f51040a226e5a72cc47b7613f7772950e460a665f7083de89f183f60e"
        );
        assert_eq!(
            server.template_sha256.to_string(),
            "6cd0e94d9b489fdb9a000743791f20e7e52b62c1e5cbb66fcc91c6716595223a"
        );
        assert_eq!(
            server.server_gates(),
            ServerGates::new(10353, server.template_sha256, 131_072, 1)
        );
    }

    #[test]
    fn kimi_dev_is_a_single_file_qwen_gguf() {
        // Kimi-Dev-72B is a Qwen2.5 finetune → ChatML/Qwen; the loader needs a
        // single-file GGUF (the K-quants ship split, so we pin Q4_1).
        let p = ModelProfile::builtin("kimi-dev").expect("kimi-dev is built in");
        assert_eq!(p.format, Some(ChatFormat::Qwen));
        assert!(p.repo.as_deref().unwrap().contains("Kimi-Dev-72B"));
        let gguf = p.gguf.as_deref().expect("kimi-dev pins a gguf quant");
        assert!(
            gguf.ends_with(".gguf") && !gguf.contains("-of-"),
            "expected a single-file gguf, got {gguf}"
        );
        assert!(p.reasoning, "kimi-dev is a reasoning model");
    }

    #[test]
    fn reasoning_profile_floors_the_token_budget() {
        // upholds: PROFILE-1 — a reasoning profile raises a too-small budget to
        // its floor but never reduces a larger caller budget.
        let p = ModelProfile::builtin("deepseek-r1").expect("deepseek-r1 is built in");
        assert!(p.reasoning);
        assert_eq!(p.format, Some(ChatFormat::DeepSeek));
        // default base (256) → floored up.
        let opts = p.apply_gen_overrides(GenOpts::default());
        assert_eq!(opts.max_tokens, REASONING_MIN_TOKENS);
        // a larger caller budget is kept (floor only raises).
        let big = GenOpts {
            max_tokens: REASONING_MIN_TOKENS * 4,
            ..Default::default()
        };
        assert_eq!(
            p.apply_gen_overrides(big).max_tokens,
            REASONING_MIN_TOKENS * 4
        );
        for name in ["qwq", "muse-glimmer"] {
            let profile = ModelProfile::builtin(name).unwrap();
            let big = GenOpts {
                max_tokens: 8192,
                ..Default::default()
            };
            assert_eq!(
                profile.apply_gen_overrides(big).max_tokens,
                8192,
                "{name} profile budget is a floor"
            );
        }
        // a non-reasoning profile leaves the budget alone.
        let plain = ModelProfile::builtin("gemma2").unwrap();
        assert!(!plain.reasoning);
        assert_eq!(
            plain.apply_gen_overrides(GenOpts::default()).max_tokens,
            GenOpts::default().max_tokens
        );
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
    fn engine_source_rejects_a_llama_server_profile() {
        let muse = ModelProfile::builtin("muse-glimmer").unwrap();
        let error = muse
            .to_engine_source(true)
            .err()
            .expect("Muse must not enter Candle")
            .to_string();
        assert!(error.contains("managed llama-server"), "{error}");
        assert!(error.contains("yatima chat"), "{error}");

        assert!(ModelProfile::builtin("gemma2")
            .unwrap()
            .to_engine_source(true)
            .is_ok());
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
                top_p: None,
                seed: 9
            }
        );
        assert_eq!(opts.prefill_chunk, Some(32));
    }

    #[test]
    fn profile_temperature_keeps_the_caller_seed_and_top_p() {
        // Regression: a profile that overrides only `temperature` must NOT reset
        // the caller's seed/top_p (the seed-drop bug). Components compose.
        let profile = ModelProfile {
            name: "x".into(),
            temperature: Some(0.6),
            ..Default::default()
        };
        let base = GenOpts {
            sampling: Sampling::nucleus(0.9, Some(0.8), 42),
            ..Default::default()
        };
        assert_eq!(
            profile.apply_gen_overrides(base).sampling,
            Sampling::Sample {
                temperature: 0.6, // profile overrides temperature
                top_p: Some(0.8), // base top_p preserved
                seed: 42,         // base seed preserved (the bug)
            }
        );
    }

    #[test]
    fn reasoning_profiles_pin_temperature_and_top_p() {
        // The shipped reasoning profiles request nucleus sampling (the repetition
        // mitigation): temperature 0.6 + top-p 0.95.
        for name in ["deepseek-r1", "kimi-dev", "qwq"] {
            let p = ModelProfile::builtin(name).unwrap();
            assert_eq!(p.temperature, Some(0.6), "{name} temperature");
            assert_eq!(p.top_p, Some(0.95), "{name} top_p");
            assert_eq!(
                p.apply_gen_overrides(GenOpts::default()).sampling,
                Sampling::Sample {
                    temperature: 0.6,
                    top_p: Some(0.95),
                    seed: 0,
                },
                "{name} resolves to nucleus sampling"
            );
        }
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
        for name in ["qwen32b", "muse-glimmer"] {
            let profile = ModelProfile::builtin(name).unwrap();
            let json = serde_json::to_string(&profile).unwrap();
            let back: ModelProfile = serde_json::from_str(&json).unwrap();
            assert_eq!(profile, back, "{name}");
        }
    }
}
