//! Where a model's files come from: a directory **xor** a repository, parsed at
//! the edge so the rest of a program never sees an invalid combination, and
//! resolved to a concrete directory (fetching on a cache miss when online).

use crate::{is_model_present, model_dir, models_root, ModelId};
use anyhow::{bail, Result};
use std::path::{Path, PathBuf};

/// Where a model's files come from — exactly one source (CLI-1). Opaque: build
/// one with [`from_args`](ModelSource::from_args) and turn it into a directory
/// with [`resolve`](ModelSource::resolve).
pub struct ModelSource(Source);

enum Source {
    Directory(PathBuf),
    Repository {
        id: ModelId,
        root: PathBuf,
        fetch: FetchPolicy,
        /// A single GGUF file to fetch instead of safetensors shards.
        gguf: Option<String>,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FetchPolicy {
    Online,
    Offline,
}

impl ModelSource {
    /// Parse a `(--model, --repo, --models-dir, --offline, --gguf)` argument set
    /// into exactly one source: a directory **xor** a repository (CLI-1). An
    /// untrusted repo id is validated through [`ModelId`] (MS-3).
    pub fn from_args(
        model: Option<PathBuf>,
        repo: Option<String>,
        models_dir: Option<PathBuf>,
        offline: bool,
        gguf: Option<String>,
    ) -> Result<ModelSource> {
        let source = match (model, repo) {
            (Some(dir), None) => Source::Directory(dir),
            (None, Some(repo)) => Source::Repository {
                id: ModelId::parse(&repo)?,
                gguf,
                root: models_dir.unwrap_or_else(models_root),
                fetch: if offline {
                    FetchPolicy::Offline
                } else {
                    FetchPolicy::Online
                },
            },
            (Some(_), Some(_)) => bail!("pass only one of --model / --repo"),
            (None, None) => bail!("specify --model <dir> or --repo <id>"),
        };
        Ok(ModelSource(source))
    }

    /// Resolve to a concrete model directory, fetching on a cache miss when the
    /// policy is `Online` (CLI-2: `Offline` never touches the network). Fetching
    /// needs the `fetch` feature; without it, an absent model is an error.
    pub fn resolve(self) -> Result<PathBuf> {
        match self.0 {
            Source::Directory(dir) => Ok(dir),
            Source::Repository {
                id,
                root,
                fetch,
                gguf,
            } => {
                let dir = model_dir(&root, &id);
                if is_model_present(&dir) {
                    return Ok(dir);
                }
                match fetch {
                    FetchPolicy::Offline => bail!(
                        "model '{id}' not present at {} (drop --offline to fetch, or run: \
                         possum model download --repository {id} --to {})",
                        dir.display(),
                        root.display()
                    ),
                    FetchPolicy::Online => fetch_model(&id, &root, gguf.as_deref()),
                }
            }
        }
    }
}

#[cfg(feature = "fetch")]
fn fetch_model(id: &ModelId, root: &Path, gguf: Option<&str>) -> Result<PathBuf> {
    eprintln!("fetching {id} …");
    crate::ensure_model_blocking(id, root, gguf)
}

#[cfg(not(feature = "fetch"))]
fn fetch_model(id: &ModelId, _root: &Path, _gguf: Option<&str>) -> Result<PathBuf> {
    bail!("model '{id}' not present and yatima was built without the `fetch` feature")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn source_directory() {
        // upholds: CLI-1
        let s = ModelSource::from_args(Some(PathBuf::from("/m")), None, None, false, None).unwrap();
        assert!(matches!(s.0, Source::Directory(_)));
    }

    #[test]
    fn source_repository_online_and_offline() {
        // upholds: CLI-1
        let on = ModelSource::from_args(None, Some("org/name".into()), None, false, None).unwrap();
        assert!(matches!(
            on.0,
            Source::Repository {
                fetch: FetchPolicy::Online,
                ..
            }
        ));
        let off = ModelSource::from_args(None, Some("org/name".into()), None, true, None).unwrap();
        assert!(matches!(
            off.0,
            Source::Repository {
                fetch: FetchPolicy::Offline,
                ..
            }
        ));
    }

    #[test]
    fn source_is_exclusive_and_required() {
        // upholds: CLI-1 — exactly one model source.
        assert!(ModelSource::from_args(
            Some(PathBuf::from("/m")),
            Some("org/name".into()),
            None,
            false,
            None
        )
        .is_err());
        assert!(ModelSource::from_args(None, None, None, false, None).is_err());
    }

    #[test]
    fn source_rejects_escaping_model_id() {
        // upholds: MS-3
        assert!(ModelSource::from_args(None, Some("../escape".into()), None, false, None).is_err());
    }

    #[test]
    fn offline_absent_errors_without_network() {
        // upholds: CLI-2 — offline + absent model errors, never fetches.
        let src = ModelSource::from_args(
            None,
            Some("org/name".into()),
            Some(PathBuf::from("/nonexistent-yatima-models-xyzzy")),
            true,
            None,
        )
        .unwrap();
        assert!(src.resolve().is_err());
    }
}
