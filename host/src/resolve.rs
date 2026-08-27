//! The shared native-frontend model resolution: the raw profile/source
//! choices a frontend's CLI collects become one validated
//! [`HostBackendConfig`] before the host thread spawns. One resolver serves
//! TUI, GUI, and serve — the same policy, not three copies — with the same
//! PROFILE-2 behavior as the CLI's private resolver: a profile's pinned
//! source is authoritative, and every contradictory override is rejected by
//! name, never silently ignored.
//!
//! Pure — no filesystem, network, or process work — so the PROFILE-2 matrix
//! is unit-testable and every contradiction fails in the frontend's `main`,
//! before `spawn_nonblocking`. Acquisition itself happens inside the host
//! thread as part of its owned lifecycle: no frontend holds a resolved path
//! it could substitute before launch.
//!
//! The native frontends expose no `--format` flag, so the CLI's
//! pinned-format rejection row has no counterpart here. If this resolver
//! ever accepts an explicit format, it must reject one conflicting with the
//! profile's pin and add that matrix row.

use std::path::PathBuf;

use anyhow::{bail, Result};
use yatima_lib::{
    ChatFormat, GenOpts, LlamaServerProfile, ModelProfile, ModelSource, ProfileBackend,
};

use crate::HostConfig;

/// The host's model backend, deliberately unresolved: the actor thread
/// performs acquisition (and, from stage 5b, verification and process
/// startup) behind one door.
pub enum HostBackendConfig {
    /// The in-process Candle engine.
    Engine {
        source: ModelSource,
        /// Force CPU instead of the GPU (a Candle-only choice).
        cpu: bool,
    },
    /// A managed llama-server child, to be launched under the profile's
    /// pinned digest, compatibility gates, context, and sampling. Nothing
    /// here is verified yet — verification is 5b's launch-time work. 5a
    /// carries the configuration only; today's actor fails closed on this
    /// variant with no verification or process work.
    ManagedLlamaServer {
        source: ModelSource,
        profile: LlamaServerProfile,
    },
}

// `ModelSource` is deliberately opaque (no `Debug`), so these impls name the
// structure and elide the source rather than deriving.
impl std::fmt::Debug for HostBackendConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Engine { cpu, .. } => f
                .debug_struct("Engine")
                .field("cpu", cpu)
                .finish_non_exhaustive(),
            Self::ManagedLlamaServer { profile, .. } => f
                .debug_struct("ManagedLlamaServer")
                .field("profile", profile)
                .finish_non_exhaustive(),
        }
    }
}

impl std::fmt::Debug for ResolvedHostModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ResolvedHostModel")
            .field("backend", &self.backend)
            .field("label", &self.label)
            .field("profile", &self.profile.as_ref().map(|p| &p.name))
            .finish()
    }
}

/// A frontend's raw model choices, exactly as its CLI collects them (owned
/// and clap-free, so this crate stays a library concern).
#[derive(Debug, Clone, Default)]
pub struct HostModelChoices {
    pub profile: Option<String>,
    pub model: Option<PathBuf>,
    pub repo: Option<String>,
    pub models_dir: Option<PathBuf>,
    pub gguf: Option<String>,
    pub cpu: bool,
    pub offline: bool,
}

/// The resolver's product. The validated pieces are private: the only way
/// out is [`into_host_config`](ResolvedHostModel::into_host_config), so the
/// backend, profile recipe, format, and label cannot drift apart between
/// frontends.
pub struct ResolvedHostModel {
    backend: HostBackendConfig,
    /// The display label: the profile's name, or `None` for an explicit
    /// source — the actor labels those with the directory it resolves.
    label: Option<String>,
    /// The selected profile, kept whole so generation layering stays the
    /// profile's own (PROFILE-1).
    profile: Option<ModelProfile>,
}

impl ResolvedHostModel {
    /// The profile's pinned chat format, if any (the host still infers from
    /// the model architecture when this is `None`, FMT-1/FMT-2).
    fn format(&self) -> Option<ChatFormat> {
        self.profile.as_ref().and_then(ModelProfile::format)
    }

    /// Layer the profile's generation recipe over the frontend's use-case
    /// base (PROFILE-1); the base passes through untouched without a profile.
    fn gen_opts(&self, base: GenOpts) -> GenOpts {
        match &self.profile {
            Some(profile) => profile.apply_gen_overrides(base),
            None => base,
        }
    }

    /// The one composition every frontend uses: the validated backend, the
    /// profile-layered generation options (PROFILE-1), and the pinned format
    /// become the host configuration in a single consuming step.
    pub fn into_host_config(self, base: GenOpts, system: Option<String>) -> HostConfig {
        let opts = self.gen_opts(base);
        let format = self.format();
        HostConfig {
            backend: self.backend,
            opts,
            format,
            system,
            model_label: self.label,
        }
    }
}

/// Validate one frontend's raw choices into the backend sum. A profile is
/// authoritative for its source and backend (PROFILE-2): model-source
/// overrides are rejected by name, a llama-server profile additionally
/// rejects the Candle-only `--cpu`, and without a profile the explicit
/// source flags must name exactly one source (CLI-1, via
/// [`ModelSource::from_args`]).
pub fn resolve_host_model(choices: HostModelChoices) -> Result<ResolvedHostModel> {
    let profile = match &choices.profile {
        Some(name) => Some(ModelProfile::builtin(name).ok_or_else(|| {
            anyhow::anyhow!(
                "unknown profile {name:?}; built-ins: {:?}",
                ModelProfile::BUILTIN_NAMES
            )
        })?),
        None => None,
    };

    if let Some(profile) = &profile {
        let mut overrides = Vec::new();
        if choices.model.is_some() {
            overrides.push("--model");
        }
        if choices.repo.is_some() {
            overrides.push("--repo");
        }
        if choices.gguf.is_some() {
            overrides.push("--gguf");
        }
        if choices.models_dir.is_some() {
            overrides.push("--models-dir");
        }
        if !overrides.is_empty() {
            bail!(
                "profile {:?} rejects model-source overrides: {}",
                profile.name,
                overrides.join(", ")
            );
        }
    }

    match profile {
        Some(profile) => {
            let source = profile.to_source(choices.offline)?;
            let backend = match &profile.backend {
                ProfileBackend::LlamaServer(server) => {
                    if choices.cpu {
                        bail!(
                            "Candle-only flags are not valid with the llama-server \
                             backend: --cpu"
                        );
                    }
                    HostBackendConfig::ManagedLlamaServer {
                        source,
                        profile: server.clone(),
                    }
                }
                ProfileBackend::Engine => HostBackendConfig::Engine {
                    source,
                    cpu: choices.cpu,
                },
            };
            Ok(ResolvedHostModel {
                backend,
                label: Some(profile.name.clone()),
                profile: Some(profile),
            })
        }
        None => Ok(ResolvedHostModel {
            backend: HostBackendConfig::Engine {
                source: ModelSource::from_args(
                    choices.model,
                    choices.repo,
                    choices.models_dir,
                    choices.offline,
                    choices.gguf,
                )?,
                cpu: choices.cpu,
            },
            label: None,
            profile: None,
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn choices() -> HostModelChoices {
        HostModelChoices::default()
    }

    #[test]
    fn explicit_source_without_a_profile_is_engine() {
        // upholds: PROFILE-2 (matrix) — no profile plus an explicit source
        // gives the Candle engine, with the frontend's cpu choice honored.
        let resolved = resolve_host_model(HostModelChoices {
            repo: Some("org/name".into()),
            cpu: true,
            ..choices()
        })
        .unwrap();
        assert!(matches!(
            resolved.backend,
            HostBackendConfig::Engine { cpu: true, .. }
        ));
        assert!(
            resolved.label.is_none(),
            "explicit sources have no profile label"
        );
        assert!(resolved.profile.is_none());
        assert_eq!(resolved.format(), None);
    }

    #[test]
    fn engine_profile_is_engine_and_accepts_cpu() {
        // upholds: PROFILE-2 (matrix) — an Engine profile gives Engine, and
        // --cpu remains a valid Candle choice beside it.
        let resolved = resolve_host_model(HostModelChoices {
            profile: Some("kimi-dev".into()),
            cpu: true,
            ..choices()
        })
        .unwrap();
        assert!(matches!(
            resolved.backend,
            HostBackendConfig::Engine { cpu: true, .. }
        ));
        assert_eq!(resolved.label.as_deref(), Some("kimi-dev"));
        assert_eq!(resolved.format(), Some(ChatFormat::Qwen));
    }

    #[test]
    fn muse_profile_is_managed_llama_server() {
        // upholds: PROFILE-2 (matrix) — muse-glimmer structurally selects the
        // managed llama-server variant with its pinned format.
        let resolved = resolve_host_model(HostModelChoices {
            profile: Some("muse-glimmer".into()),
            ..choices()
        })
        .unwrap();
        assert!(matches!(
            resolved.backend,
            HostBackendConfig::ManagedLlamaServer { .. }
        ));
        assert_eq!(resolved.label.as_deref(), Some("muse-glimmer"));
        assert_eq!(resolved.format(), Some(ChatFormat::MuseGlimmer));
    }

    #[test]
    fn every_profile_rejects_each_model_source_override() {
        // upholds: PROFILE-2 (matrix) — a profile's pinned source is
        // authoritative on every backend: each override flag is a named
        // contradiction. Pure, so every rejection precedes thread spawn.
        for profile in ["kimi-dev", "muse-glimmer"] {
            for (flag, with_override) in [
                (
                    "--model",
                    HostModelChoices {
                        model: Some("/tmp/model".into()),
                        ..choices()
                    },
                ),
                (
                    "--repo",
                    HostModelChoices {
                        repo: Some("someone/else".into()),
                        ..choices()
                    },
                ),
                (
                    "--gguf",
                    HostModelChoices {
                        gguf: Some("another.gguf".into()),
                        ..choices()
                    },
                ),
                (
                    "--models-dir",
                    HostModelChoices {
                        models_dir: Some("/tmp/models".into()),
                        ..choices()
                    },
                ),
            ] {
                let error = resolve_host_model(HostModelChoices {
                    profile: Some(profile.into()),
                    ..with_override
                })
                .unwrap_err()
                .to_string();
                assert!(
                    error.contains("rejects model-source overrides"),
                    "{profile} {flag}: {error}"
                );
                assert!(error.contains(flag), "{profile} {flag}: {error}");
            }
        }
    }

    #[test]
    fn muse_rejects_the_candle_only_cpu_flag() {
        // upholds: PROFILE-2 (matrix) — --cpu is a Candle choice; beside the
        // managed llama-server profile it is a rejected contradiction.
        let error = resolve_host_model(HostModelChoices {
            profile: Some("muse-glimmer".into()),
            cpu: true,
            ..choices()
        })
        .unwrap_err()
        .to_string();
        assert!(error.contains("--cpu"), "{error}");
        assert!(error.contains("Candle-only"), "{error}");
    }

    #[test]
    fn no_profile_and_no_source_is_an_error() {
        // upholds: PROFILE-2 (matrix) / CLI-1 — nothing to run is a clear
        // error before any thread or I/O work.
        assert!(resolve_host_model(choices()).is_err());
    }

    #[test]
    fn unknown_profile_is_a_named_error() {
        let error = resolve_host_model(HostModelChoices {
            profile: Some("nope".into()),
            ..choices()
        })
        .unwrap_err()
        .to_string();
        assert!(error.contains("unknown profile"), "{error}");
        assert!(error.contains("built-ins"), "{error}");
    }

    #[test]
    fn muse_recipe_survives_resolution_structurally() {
        // upholds: LSRV-5 (structural preservation only) — the built-in Muse
        // profile reaches the managed configuration carrying the exact Stage 3
        // launch recipe: pinned digest, compatibility gates, context, and the
        // generation recipe via PROFILE-1 layering. No claim is made that a
        // model was verified or a process launched — those witnesses are 5b.
        let builtin = ModelProfile::builtin("muse-glimmer").unwrap();
        let ProfileBackend::LlamaServer(expected) = &builtin.backend else {
            panic!("Muse must pin the llama-server backend");
        };
        let resolved = resolve_host_model(HostModelChoices {
            profile: Some("muse-glimmer".into()),
            offline: true,
            ..choices()
        })
        .unwrap();
        let HostBackendConfig::ManagedLlamaServer { profile, .. } = &resolved.backend else {
            panic!("Muse must select managed llama-server");
        };
        assert_eq!(
            profile, expected,
            "the launch recipe must pass through whole"
        );
        // `ModelSource` is opaque, so the promised repository and exact GGUF
        // are asserted on the carried profile the source was constructed
        // from (`to_source` builds from exactly these fields).
        let carried = resolved
            .profile
            .as_ref()
            .expect("the profile passes through");
        assert_eq!(
            carried.repo.as_deref(),
            Some("meta-models/Muse-Glimmer-30B-GGUF")
        );
        assert_eq!(
            carried.gguf.as_deref(),
            Some("Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf")
        );
        assert_eq!(
            profile.expected_sha256,
            "4cc57c0f51040a226e5a72cc47b7613f7772950e460a665f7083de89f183f60e"
                .parse()
                .unwrap()
        );
        assert_eq!(profile.context, 131072);

        let opts = resolved.gen_opts(GenOpts::default());
        let layered = builtin.apply_gen_overrides(GenOpts::default());
        assert_eq!(opts.max_tokens, layered.max_tokens);
        assert_eq!(opts.sampling, layered.sampling);
    }
}
