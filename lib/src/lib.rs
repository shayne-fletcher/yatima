//! a Rust runtime for language-integrated LLMs — inference as an in-process
//! library function.

mod engine;
mod token_output_stream;

pub use engine::{device, is_model_present, model_shards, Engine, GenOpts};
#[cfg(feature = "fetch")]
pub use engine::{ensure_model, ensure_model_blocking};

use std::ffi::OsString;
use std::path::{Path, PathBuf};

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

/// The leaf directory holding a repository's files under `models_root`,
/// mirroring possum's on-disk layout (`<root>/<org>/<name>`).
pub fn model_dir(models_root: &Path, repo: &str) -> PathBuf {
    models_root.join(repo)
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
        assert_eq!(
            model_dir(&root, "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"),
            PathBuf::from("/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"),
        );
    }
}
