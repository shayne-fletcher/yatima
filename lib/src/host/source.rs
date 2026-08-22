//! Where a model's files come from: a directory **xor** a repository, parsed at
//! the edge so the rest of a program never sees an invalid combination, and
//! resolved to either a model directory or one exact GGUF artifact (fetching
//! on a cache miss when online).

use crate::{is_model_present, model_dir, models_root, ModelId};
use anyhow::{bail, Context, Result};
use std::path::{Path, PathBuf};

/// Where a model's files come from — exactly one source (CLI-1). Opaque: build
/// one with [`from_args`](ModelSource::from_args), then resolve it with
/// [`resolve`](ModelSource::resolve) or
/// [`resolve_async`](ModelSource::resolve_async).
pub struct ModelSource(Source);

/// The result of model-source resolution. Candle consumes its directory; a
/// managed llama-server requires that resolution established one exact GGUF.
/// The representation is private so callers cannot manufacture a resolution.
#[derive(Debug, Clone)]
pub struct ResolvedModel(Resolution);

#[derive(Debug, Clone)]
enum Resolution {
    Directory {
        directory: PathBuf,
        gguf_candidates: Vec<PathBuf>,
    },
    Gguf {
        directory: PathBuf,
        artifact: GgufArtifact,
    },
}

impl ResolvedModel {
    pub fn directory(&self) -> &Path {
        match &self.0 {
            Resolution::Directory { directory, .. } | Resolution::Gguf { directory, .. } => {
                directory
            }
        }
    }

    pub fn into_directory(self) -> PathBuf {
        match self.0 {
            Resolution::Directory { directory, .. } | Resolution::Gguf { directory, .. } => {
                directory
            }
        }
    }

    /// Require one exact GGUF for managed llama-server execution (MS-4).
    pub fn into_gguf(self) -> Result<GgufArtifact> {
        match self.0 {
            Resolution::Gguf { artifact, .. } => Ok(artifact),
            Resolution::Directory {
                directory,
                gguf_candidates,
            } if gguf_candidates.is_empty() => bail!(
                "managed llama-server requires one GGUF in {}; found none",
                directory.display()
            ),
            Resolution::Directory {
                directory,
                gguf_candidates,
            } => bail!(
                "managed llama-server requires one GGUF in {}; found: {}",
                directory.display(),
                display_candidates(&gguf_candidates)
            ),
        }
    }
}

/// A canonical GGUF file established by model-source resolution. Its field is
/// private so a managed server cannot be launched from an unchecked path.
#[derive(Debug, Clone)]
pub struct GgufArtifact(PathBuf);

impl GgufArtifact {
    pub fn path(&self) -> &Path {
        &self.0
    }

    fn from_resolved(directory: &Path, path: PathBuf) -> Result<GgufArtifact> {
        let canonical_directory = directory
            .canonicalize()
            .with_context(|| format!("resolve model directory {}", directory.display()))?;
        let canonical_path = path
            .canonicalize()
            .with_context(|| format!("resolve GGUF {}", path.display()))?;
        if !canonical_path.starts_with(&canonical_directory) {
            bail!(
                "GGUF {} escapes model directory {}",
                path.display(),
                directory.display()
            );
        }
        if !canonical_path.is_file() || !has_gguf_extension(&canonical_path) {
            bail!("resolved artifact is not a GGUF file: {}", path.display());
        }
        Ok(GgufArtifact(canonical_path))
    }
}

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
        if let Some(file) = gguf.as_deref() {
            validate_gguf_basename(file)?;
        }
        let source = match (model, repo) {
            (Some(_), None) if gguf.is_some() => bail!("--gguf requires --repo"),
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

    /// Synchronous shim over [`resolve_async`](ModelSource::resolve_async).
    pub fn resolve(self) -> Result<ResolvedModel> {
        crate::runtime::block_on(self.resolve_async())
    }

    /// Resolve to a directory or exact GGUF, fetching on a cache miss when the
    /// policy is `Online` (CLI-2: `Offline` never touches the network).
    pub async fn resolve_async(self) -> Result<ResolvedModel> {
        match self.0 {
            Source::Directory(dir) => resolve_directory(dir),
            Source::Repository {
                id,
                root,
                fetch,
                gguf,
            } => {
                let dir = model_dir(&root, &id);
                if let Some(resolved) = resolve_cached(&dir, gguf.as_deref())? {
                    return Ok(resolved);
                }
                match fetch {
                    FetchPolicy::Offline => bail!(
                        "model '{id}'{} not present at {} (drop --offline to fetch, or run: \
                         possum model download --repository {id} --to {})",
                        gguf.as_deref()
                            .map(|name| format!(" GGUF {name:?}"))
                            .unwrap_or_default(),
                        dir.display(),
                        root.display()
                    ),
                    FetchPolicy::Online => {
                        fetch_model(&id, &root, gguf.as_deref()).await?;
                        resolve_cached(&dir, gguf.as_deref())?.ok_or_else(|| {
                            anyhow::anyhow!(
                                "model {id} still incomplete after fetch at {}",
                                dir.display()
                            )
                        })
                    }
                }
            }
        }
    }
}

#[cfg(feature = "fetch")]
async fn fetch_model(id: &ModelId, root: &Path, gguf: Option<&str>) -> Result<PathBuf> {
    eprintln!("fetching {id} …");
    crate::engine::ensure_model(id, root, gguf).await
}

#[cfg(not(feature = "fetch"))]
async fn fetch_model(id: &ModelId, _root: &Path, _gguf: Option<&str>) -> Result<PathBuf> {
    bail!("model '{id}' not present and yatima was built without the `fetch` feature")
}

fn resolve_directory(directory: PathBuf) -> Result<ResolvedModel> {
    let mut paths = gguf_candidates(&directory)?;
    match paths.len() {
        1 => Ok(ResolvedModel(Resolution::Gguf {
            artifact: GgufArtifact::from_resolved(&directory, paths.remove(0))?,
            directory,
        })),
        _ => Ok(ResolvedModel(Resolution::Directory {
            directory,
            gguf_candidates: paths,
        })),
    }
}

/// Pure cache decision used before any fetch. `None` means the requested
/// artifact/layout is absent; another cached GGUF never substitutes (MS-4).
fn resolve_cached(directory: &Path, requested_gguf: Option<&str>) -> Result<Option<ResolvedModel>> {
    if let Some(file) = requested_gguf {
        validate_gguf_basename(file)?;
        let path = directory.join(file);
        if !path.is_file() {
            return Ok(None);
        }
        return Ok(Some(ResolvedModel(Resolution::Gguf {
            artifact: GgufArtifact::from_resolved(directory, path)?,
            directory: directory.to_path_buf(),
        })));
    }
    if !is_model_present(directory) {
        return Ok(None);
    }
    resolve_directory(directory.to_path_buf()).map(Some)
}

fn validate_gguf_basename(file: &str) -> Result<()> {
    let path = Path::new(file);
    if !crate::is_safe_relative(file)
        || crate::has_glob_metachar(file)
        || path.components().count() != 1
        || !has_gguf_extension(path)
    {
        bail!("GGUF name {file:?} must be one safe literal .gguf basename");
    }
    Ok(())
}

fn has_gguf_extension(path: &Path) -> bool {
    path.extension()
        .and_then(|ext| ext.to_str())
        .is_some_and(|ext| ext.eq_ignore_ascii_case("gguf"))
}

fn gguf_candidates(directory: &Path) -> Result<Vec<PathBuf>> {
    let mut paths = match std::fs::read_dir(directory) {
        Ok(entries) => {
            let mut paths = Vec::new();
            for entry in entries {
                let path = entry
                    .with_context(|| format!("read entry in {}", directory.display()))?
                    .path();
                if has_gguf_extension(&path)
                    && std::fs::metadata(&path)
                        .with_context(|| format!("inspect GGUF candidate {}", path.display()))?
                        .is_file()
                {
                    paths.push(path);
                }
            }
            paths
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Vec::new(),
        Err(error) => {
            return Err(error)
                .with_context(|| format!("read model directory {}", directory.display()))
        }
    };
    paths.sort();
    Ok(paths)
}

fn display_candidates(paths: &[PathBuf]) -> String {
    paths
        .iter()
        .map(|path| {
            path.file_name()
                .unwrap_or(path.as_os_str())
                .to_string_lossy()
                .into_owned()
        })
        .collect::<Vec<_>>()
        .join(", ")
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

    #[test]
    fn exact_cached_gguf_is_the_resolved_artifact() {
        // upholds: MS-4 — an exact cache hit returns that canonical file even
        // when another quant is present; no directory-order choice remains.
        let root = tempfile::tempdir().unwrap();
        let directory = root.path().join("org/name");
        std::fs::create_dir_all(&directory).unwrap();
        std::fs::write(directory.join("wanted.gguf"), "wanted").unwrap();
        std::fs::write(directory.join("other.gguf"), "other").unwrap();
        let source = ModelSource::from_args(
            None,
            Some("org/name".into()),
            Some(root.path().to_path_buf()),
            true,
            Some("wanted.gguf".into()),
        )
        .unwrap();
        let resolved = source.resolve().unwrap();
        let artifact = resolved.into_gguf().unwrap();
        assert_eq!(
            artifact.path(),
            directory.join("wanted.gguf").canonicalize().unwrap()
        );
    }

    #[test]
    fn another_cached_quant_never_satisfies_the_request() {
        // upholds: MS-4 — this is the offline side of the fetch decision. In
        // online mode the same `None` decision enters the downloader; it can
        // never substitute `other.gguf` for `wanted.gguf`.
        let root = tempfile::tempdir().unwrap();
        let directory = root.path().join("org/name");
        std::fs::create_dir_all(&directory).unwrap();
        std::fs::write(directory.join("other.gguf"), "other").unwrap();
        assert!(resolve_cached(&directory, Some("wanted.gguf"))
            .unwrap()
            .is_none());

        let source = ModelSource::from_args(
            None,
            Some("org/name".into()),
            Some(root.path().to_path_buf()),
            true,
            Some("wanted.gguf".into()),
        )
        .unwrap();
        let err = source.resolve().unwrap_err();
        assert!(err.to_string().contains("wanted.gguf"), "{err}");
    }

    #[test]
    fn unsafe_gguf_basenames_are_rejected() {
        // upholds: MS-4 / MS-3 — a requested artifact cannot escape or name a
        // nested path.
        for name in [
            "../escape.gguf",
            "nested/model.gguf",
            "/tmp/model.gguf",
            "model.bin",
            "model*.gguf",
            "model?.gguf",
            "model[12].gguf",
            "model\\*.gguf",
        ] {
            assert!(
                ModelSource::from_args(
                    None,
                    Some("org/name".into()),
                    None,
                    true,
                    Some(name.into())
                )
                .is_err(),
                "{name}"
            );
        }
    }

    #[test]
    fn a_named_gguf_requires_a_repository_source() {
        // upholds: CLI-1 / MS-4 — a filename is never silently ignored on a
        // local-directory source.
        let error = ModelSource::from_args(
            Some(PathBuf::from("/models/model")),
            None,
            None,
            true,
            Some("model.gguf".into()),
        )
        .err()
        .expect("--model with --gguf must be rejected");
        assert!(error.to_string().contains("--gguf requires --repo"));
    }

    #[test]
    fn managed_resolution_rejects_zero_ggufs() {
        // upholds: MS-4 — a directory with no exact artifact cannot spawn.
        let directory = tempfile::tempdir().unwrap();
        let err = resolve_directory(directory.path().to_path_buf())
            .unwrap()
            .into_gguf()
            .unwrap_err();
        assert!(err.to_string().contains("found none"), "{err}");
    }

    #[test]
    fn managed_resolution_rejects_several_ggufs_and_lists_them() {
        // upholds: MS-4 — no directory-order selection is allowed.
        let directory = tempfile::tempdir().unwrap();
        std::fs::write(directory.path().join("a.gguf"), "a").unwrap();
        std::fs::write(directory.path().join("b.gguf"), "b").unwrap();
        let err = resolve_directory(directory.path().to_path_buf())
            .unwrap()
            .into_gguf()
            .unwrap_err();
        let message = err.to_string();
        assert!(
            message.contains("a.gguf") && message.contains("b.gguf"),
            "{message}"
        );
    }

    #[test]
    fn one_local_gguf_becomes_the_only_spawnable_artifact() {
        // upholds: MS-4 — local directory resolution establishes the opaque
        // value only in the unambiguous one-file case.
        let directory = tempfile::tempdir().unwrap();
        std::fs::write(directory.path().join("only.gguf"), "one").unwrap();
        let resolved = resolve_directory(directory.path().to_path_buf()).unwrap();
        assert_eq!(
            resolved.into_gguf().unwrap().path(),
            directory.path().join("only.gguf").canonicalize().unwrap()
        );
    }
}
