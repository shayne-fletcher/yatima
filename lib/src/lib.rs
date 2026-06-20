//! a Rust runtime for language-integrated LLMs — inference as an in-process
//! library function.
//!
//! # Invariant & law registry
//!
//! The canonical list of the contracts this crate upholds. They are stated, not
//! compiler-enforced; each is protected by a test that cites its id (grep the
//! `invariant`/`law` comments in the test modules). `notes/design.md` explains
//! them in prose. (CLI-level invariants `CLI-1`/`CLI-2` live in `yatima-cli`.)
//!
//! Model store & discovery:
//! - **MS-1** `models_root` precedence: `$YATIMA_MODELS_DIR`, else
//!   `${XDG_CACHE_HOME}/yatima/models`, else `$HOME/.cache/yatima/models`.
//! - **MS-2** [`model_dir`] mirrors possum's `<root>/<org>/<name>` layout.
//! - **MS-3** a [`RepoId`] and index shard names never escape the root / model
//!   directory (untrusted input is contained).
//! - **MD-1** unsharded discovery is every `*.safetensors`, sorted.
//! - **MD-2** indexed discovery is the unique `weight_map` values, deduped and
//!   sorted (also covers the dedup/order half of **DISC**).
//! - **MD-3** [`presence`] = `config.json` ∧ `tokenizer.json` ∧ all shards; a
//!   partial shard set is never a false cache hit.
//! - **EOS-1** EOS ids are read from `config.json` / `generation_config.json`
//!   as a *set* — never hard-coded token strings.
//! - **FETCH-1** [`ensure_model`] re-checks [`presence`] after download; a
//!   partial directory never reaches [`Engine::load`] (gated e2e / fetch path).
//!
//! Generation:
//! - **SAM-1** every [`Sampling`] maps to exactly one candle `LogitsProcessor`;
//!   **SAM-2** `Greedy` ignores any seed.
//! - **STOP-1** every successful generation returns exactly one [`StopReason`].
//! - **GEN-3** a generation emits at most `max_tokens` tokens.
//! - **GE-1** stateless: repeated `Greedy` runs on the same engine + prompt are
//!   byte-identical (gated e2e).

mod engine;
mod token_output_stream;

pub use engine::{
    device, is_model_present, model_shards, presence, Engine, GenOpts, Generation, Presence,
    Sampling, StopReason,
};
#[cfg(feature = "fetch")]
pub use engine::{ensure_model, ensure_model_blocking};

use anyhow::{bail, Result};
use std::ffi::OsString;
use std::path::{Component, Path, PathBuf};

/// The directory under which models are stored.
///
/// Resolution order: `$YATIMA_MODELS_DIR`, else
/// `${XDG_CACHE_HOME:-$HOME/.cache}/yatima/models`. Weights are
/// re-downloadable, so the default lives under the XDG cache.
pub fn models_root() -> PathBuf {
    resolve_models_root(
        std::env::var_os("YATIMA_MODELS_DIR"),
        std::env::var_os("XDG_CACHE_HOME"),
        std::env::var_os("HOME"),
    )
}

/// A validated Hugging Face repository id (e.g.
/// `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B`).
///
/// Parsing rejects anything that could escape the models root when joined —
/// empty ids, absolute paths, `..`, and empty path components — so that
/// [`model_dir`] is containment-safe by construction (invariant MS-3). The id
/// is untrusted input (a CLI flag), so this is the security boundary.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RepoId(String);

impl RepoId {
    /// Parse and validate a repository id.
    pub fn parse(s: &str) -> Result<RepoId> {
        if s.is_empty() {
            bail!("empty repository id");
        }
        if s.split('/').any(|seg| seg.is_empty()) {
            bail!("repository id '{s}' has an empty path component");
        }
        if !is_safe_relative(s) {
            bail!("repository id '{s}' must be relative with no '.' / '..' / root components");
        }
        Ok(RepoId(s.to_string()))
    }

    /// The underlying id string.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for RepoId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::str::FromStr for RepoId {
    type Err = anyhow::Error;
    fn from_str(s: &str) -> Result<Self> {
        RepoId::parse(s)
    }
}

/// The leaf directory holding a repository's files under `models_root`,
/// mirroring possum's on-disk layout (`<root>/<org>/<name>`). Safe by
/// construction: [`RepoId`] cannot escape the root.
pub fn model_dir(models_root: &Path, repo: &RepoId) -> PathBuf {
    models_root.join(repo.as_str())
}

/// Whether a path string is a relative path made only of normal components
/// (no root/prefix, no `..`) — i.e. it cannot escape a directory it is joined
/// onto. Used to validate both [`RepoId`]s and shard names from an index
/// manifest (untrusted data).
pub(crate) fn is_safe_relative(s: &str) -> bool {
    let p = Path::new(s);
    p.is_relative() && p.components().all(|c| matches!(c, Component::Normal(_)))
}

/// Pure core of [`models_root`], taking the relevant environment values as
/// arguments so it can be tested without mutating process state.
fn resolve_models_root(
    yatima_models_dir: Option<OsString>,
    xdg_cache_home: Option<OsString>,
    home: Option<OsString>,
) -> PathBuf {
    if let Some(dir) = yatima_models_dir {
        return PathBuf::from(dir);
    }
    let cache = xdg_cache_home
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(home.unwrap_or_default()).join(".cache"));
    cache.join("yatima").join("models")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn models_root_prefers_yatima_models_dir() {
        let r = resolve_models_root(Some("/m".into()), Some("/c".into()), Some("/h".into()));
        assert_eq!(r, PathBuf::from("/m"));
    }

    #[test]
    fn models_root_falls_back_to_xdg_cache_home() {
        let r = resolve_models_root(None, Some("/c".into()), Some("/h".into()));
        assert_eq!(r, PathBuf::from("/c/yatima/models"));
    }

    #[test]
    fn models_root_falls_back_to_home_cache() {
        let r = resolve_models_root(None, None, Some("/h".into()));
        assert_eq!(r, PathBuf::from("/h/.cache/yatima/models"));
    }

    #[test]
    fn model_dir_mirrors_possum_layout() {
        let root = PathBuf::from("/models");
        let repo = RepoId::parse("deepseek-ai/DeepSeek-R1-Distill-Qwen-7B").unwrap();
        assert_eq!(
            model_dir(&root, &repo),
            PathBuf::from("/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"),
        );
    }

    #[test]
    fn repo_id_accepts_valid_ids() {
        for id in [
            "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
            "Qwen/Qwen2.5-Coder-7B",
            "gpt2",
        ] {
            assert!(RepoId::parse(id).is_ok(), "{id} should parse");
        }
    }

    #[test]
    fn repo_id_rejects_escaping_ids() {
        for id in ["", "../etc", "a/../../b", "/abs/path", "a//b", "./x"] {
            assert!(RepoId::parse(id).is_err(), "{id:?} should be rejected");
        }
    }

    #[test]
    fn repo_id_cannot_escape_model_dir() {
        // Even constructed by hand, a parsed id stays under the root.
        let root = PathBuf::from("/models");
        let repo = RepoId::parse("org/name").unwrap();
        assert!(model_dir(&root, &repo).starts_with(&root));
    }

    use proptest::prelude::*;

    proptest! {
        // MS-3: for ANY input string, a parsed RepoId joins to a path that
        // stays under the root and has no `..` — parse rejects everything else.
        #[test]
        fn repo_id_never_escapes(s in ".*") {
            let root = PathBuf::from("/models");
            if let Ok(id) = RepoId::parse(&s) {
                let dir = model_dir(&root, &id);
                prop_assert!(dir.starts_with(&root));
                prop_assert!(dir
                    .components()
                    .all(|c| !matches!(c, std::path::Component::ParentDir)));
            }
        }

        // MS-1: models_root always follows the declared precedence.
        #[test]
        fn models_root_follows_precedence(
            ym in proptest::option::of("[^\u{0}/][^\u{0}]{0,16}"),
            xc in proptest::option::of("[^\u{0}/][^\u{0}]{0,16}"),
            home in "[^\u{0}/][^\u{0}]{0,16}",
        ) {
            let r = resolve_models_root(
                ym.clone().map(Into::into),
                xc.clone().map(Into::into),
                Some(home.clone().into()),
            );
            let expected = match (&ym, &xc) {
                (Some(m), _) => PathBuf::from(m),
                (None, Some(c)) => PathBuf::from(c).join("yatima").join("models"),
                (None, None) => PathBuf::from(home).join(".cache").join("yatima").join("models"),
            };
            prop_assert_eq!(r, expected);
        }
    }
}
