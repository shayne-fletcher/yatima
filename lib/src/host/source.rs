//! Where a model's files come from: a directory **xor** a repository, parsed at
//! the edge so the rest of a program never sees an invalid combination, and
//! resolved to either a model directory or one exact GGUF artifact (fetching
//! on a cache miss when online).

use crate::{is_model_present, model_dir, models_root, ModelId};
use anyhow::{bail, Context, Result};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use sha2::{Digest, Sha256};
use std::fmt;
use std::fs::File;
use std::io::BufReader;
use std::path::{Path, PathBuf};
use std::str::FromStr;

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

/// A SHA-256 digest with one canonical textual form: 64 lowercase hex digits.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct Sha256Digest([u8; 32]);

impl Sha256Digest {
    pub(crate) fn of_bytes(bytes: &[u8]) -> Sha256Digest {
        Sha256Digest(Sha256::digest(bytes).into())
    }
}

impl fmt::Debug for Sha256Digest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("Sha256Digest")
            .field(&self.to_string())
            .finish()
    }
}

impl fmt::Display for Sha256Digest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for byte in self.0 {
            write!(f, "{byte:02x}")?;
        }
        Ok(())
    }
}

impl FromStr for Sha256Digest {
    type Err = anyhow::Error;

    fn from_str(text: &str) -> Result<Sha256Digest> {
        if text.len() != 64 || !text.bytes().all(|byte| byte.is_ascii_hexdigit()) {
            bail!("SHA-256 digest must be exactly 64 hexadecimal characters");
        }
        let mut bytes = [0u8; 32];
        for (index, byte) in bytes.iter_mut().enumerate() {
            *byte = u8::from_str_radix(&text[index * 2..index * 2 + 2], 16)
                .expect("ASCII hex pair was validated");
        }
        Ok(Sha256Digest(bytes))
    }
}

impl Serialize for Sha256Digest {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(&self.to_string())
    }
}

impl<'de> Deserialize<'de> for Sha256Digest {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Sha256Digest, D::Error>
    where
        D: Deserializer<'de>,
    {
        let text = String::deserialize(deserializer)?;
        text.parse().map_err(serde::de::Error::custom)
    }
}

/// A resolved GGUF proven to match an expected SHA-256 digest by one of
/// two admissible evidence paths: a full hash performed now, or a full
/// hash from an earlier launch carried forward by a verification stamp
/// whose recorded file identity (size + mtime) still matches exactly
/// (trust boundary and accepted gap: LSRV-5). Only the verify family can
/// construct this refinement.
#[derive(Debug, Clone)]
pub struct VerifiedModelArtifact {
    artifact: GgufArtifact,
    digest: Sha256Digest,
}

impl VerifiedModelArtifact {
    pub(crate) fn artifact(&self) -> &GgufArtifact {
        &self.artifact
    }

    pub fn path(&self) -> &Path {
        self.artifact.path()
    }

    pub fn digest(&self) -> &Sha256Digest {
        &self.digest
    }
}

/// The marker context a cancelled verification carries: an owner that flips
/// its monotone [`Cancel`](crate::Cancel) mid-hash gets this within one read
/// chunk, distinguishable by downcast from a genuine hashing failure.
#[derive(Debug)]
pub struct VerifyCancelled;

impl std::fmt::Display for VerifyCancelled {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("SHA-256 verification cancelled")
    }
}

impl std::error::Error for VerifyCancelled {}

/// Prove the exact canonical GGUF selected by resolution matches
/// `expected` and refine it to a verified artifact (LSRV-5): the full
/// hash, unless a matching verification stamp carries a prior launch's
/// proof of the same expectation over the observably-unchanged file.
pub async fn verify(
    artifact: GgufArtifact,
    expected: &Sha256Digest,
) -> Result<VerifiedModelArtifact> {
    verify_hash(artifact, expected, None).await
}

/// [`verify`] with a cancel observed between read chunks: a 17 GB hash is a
/// long blocking island, and an owner shutting down must not wait out the
/// whole of it. Cancellation yields an error carrying [`VerifyCancelled`].
pub async fn verify_cancellable(
    artifact: GgufArtifact,
    expected: &Sha256Digest,
    cancel: &crate::Cancel,
) -> Result<VerifiedModelArtifact> {
    verify_hash(artifact, expected, Some(cancel.clone())).await
}

/// Sync shim over [`verify_cancellable`] (RT-1; the narrow lifecycle shim a
/// plain-thread host uses).
pub fn verify_cancellable_sync(
    artifact: GgufArtifact,
    expected: &Sha256Digest,
    cancel: &crate::Cancel,
) -> Result<VerifiedModelArtifact> {
    crate::runtime::block_on(verify_cancellable(artifact, expected, cancel))
}

async fn verify_hash(
    artifact: GgufArtifact,
    expected: &Sha256Digest,
    cancel: Option<crate::Cancel>,
) -> Result<VerifiedModelArtifact> {
    let path = artifact.path().to_path_buf();
    let expected = *expected;
    let actual = crate::run_blocking(move || -> Result<Sha256Digest> {
        // A matching stamp discharges the hash: the digest was proven over
        // these observably-identical bytes on an earlier launch (LSRV-5,
        // stamp amendment). Anything else — no stamp, wrong size or mtime,
        // a re-pinned expectation, unparseable text — pays the full hash.
        if let Some(digest) = stamp_match(&path, &expected) {
            return Ok(digest);
        }
        // The stamp binds the digest to ONE observed identity: sampled
        // before hashing and confirmed unchanged after. A file replaced
        // mid-hash has no stable identity and writes no stamp — stamping
        // the after-image would bind this digest to bytes never hashed.
        let before = stat_identity(&path);
        let actual = sha256_file(&path, cancel.as_ref())?;
        if actual == expected {
            if let Some(identity) = hashed_identity(before, stat_identity(&path)) {
                write_stamp(&path, identity, &actual);
            }
        }
        Ok(actual)
    })?;
    if actual != expected {
        bail!(
            "SHA-256 mismatch for {}: expected {expected}, found {actual}",
            artifact.path().display()
        );
    }
    Ok(VerifiedModelArtifact {
        artifact,
        digest: actual,
    })
}

/// The stamp's sibling path: `<artifact>.sha256-stamp`.
fn stamp_path(path: &Path) -> PathBuf {
    let mut os = path.as_os_str().to_owned();
    os.push(".sha256-stamp");
    PathBuf::from(os)
}

/// A file's observable identity: size, mtime seconds, mtime subsec nanos.
type FileIdentity = (u64, u64, u32);

/// The identity a completed hash may be bound to: only when the samples
/// taken immediately before and after hashing agree. A file replaced or
/// rewritten mid-hash has no stable identity; `None` writes no stamp and
/// the next launch simply re-hashes.
fn hashed_identity(
    before: Option<FileIdentity>,
    after: Option<FileIdentity>,
) -> Option<FileIdentity> {
    match (before, after) {
        (Some(b), Some(a)) if b == a => Some(b),
        _ => None,
    }
}

/// The artifact's [`FileIdentity`] — what the stamp is keyed to. A file
/// whose mtime predates the epoch gets no stamp service.
fn stat_identity(path: &Path) -> Option<FileIdentity> {
    let meta = std::fs::metadata(path).ok()?;
    let mtime = meta
        .modified()
        .ok()?
        .duration_since(std::time::UNIX_EPOCH)
        .ok()?;
    Some((meta.len(), mtime.as_secs(), mtime.subsec_nanos()))
}

/// The stamped digest, only when the stamp parses, matches the file's
/// current size and mtime exactly, and records exactly the expected digest
/// (a re-pinned profile must re-prove, not inherit). The accepted gap is
/// local tampering that rewrites bytes while preserving both stat fields —
/// an actor who can do that owns the machine; the pin defends against
/// wrong, stale, or corrupt artifacts, not root.
fn stamp_match(path: &Path, expected: &Sha256Digest) -> Option<Sha256Digest> {
    let (size, mtime_s, mtime_ns) = stat_identity(path)?;
    let raw = std::fs::read_to_string(stamp_path(path)).ok()?;
    let mut fields = raw.split_whitespace();
    let stamped_size: u64 = fields.next()?.parse().ok()?;
    let stamped_s: u64 = fields.next()?.parse().ok()?;
    let stamped_ns: u32 = fields.next()?.parse().ok()?;
    let stamped: Sha256Digest = fields.next()?.parse().ok()?;
    (fields.next().is_none()
        && stamped_size == size
        && stamped_s == mtime_s
        && stamped_ns == mtime_ns
        && stamped == *expected)
        .then_some(stamped)
}

/// Record the proven digest against the identity captured around the
/// hash (never re-sampled here — see [`hashed_identity`]). Best effort:
/// verification already succeeded, and a failed write only costs the
/// next launch a re-hash.
fn write_stamp(path: &Path, identity: FileIdentity, digest: &Sha256Digest) {
    let (size, mtime_s, mtime_ns) = identity;
    let _ = std::fs::write(
        stamp_path(path),
        format!("{size} {mtime_s} {mtime_ns} {digest}\n"),
    );
}

/// Sync shim over [`verify`], bridged through the one runtime (RT-1) — one of
/// the three narrow lifecycle shims a plain-thread host uses (with
/// [`crate::LlamaServer::spawn_sync`] and
/// [`crate::LlamaServer::shutdown_sync`]); no general executor handle is
/// exposed. This variant carries no cancel and hashes to completion; an
/// owner that must answer a shutdown mid-hash uses
/// [`verify_cancellable_sync`], whose cancel is observed between chunks.
pub fn verify_sync(
    artifact: GgufArtifact,
    expected: &Sha256Digest,
) -> Result<VerifiedModelArtifact> {
    crate::runtime::block_on(verify(artifact, expected))
}

fn sha256_file(path: &Path, cancel: Option<&crate::Cancel>) -> Result<Sha256Digest> {
    use std::io::Read;
    let file = File::open(path).with_context(|| format!("open {} for SHA-256", path.display()))?;
    let mut reader = BufReader::with_capacity(1024 * 1024, file);
    let mut hasher = Sha256::new();
    let mut chunk = vec![0u8; 1024 * 1024];
    loop {
        if cancel.is_some_and(crate::Cancel::is_cancelled) {
            return Err(anyhow::Error::new(VerifyCancelled)
                .context(format!("hash {} for SHA-256", path.display())));
        }
        let read = reader
            .read(&mut chunk)
            .with_context(|| format!("read {} for SHA-256", path.display()))?;
        if read == 0 {
            break;
        }
        hasher.update(&chunk[..read]);
    }
    Ok(Sha256Digest(hasher.finalize().into()))
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

    #[test]
    fn sha256_digest_parses_displays_and_round_trips_serde() {
        let lower = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
        let upper = lower.to_ascii_uppercase();
        let digest: Sha256Digest = upper.parse().unwrap();
        assert_eq!(digest.to_string(), lower);
        let json = serde_json::to_string(&digest).unwrap();
        assert_eq!(json, format!("\"{lower}\""));
        assert_eq!(serde_json::from_str::<Sha256Digest>(&json).unwrap(), digest);
    }

    #[test]
    fn sha256_digest_rejects_wrong_length_and_non_hex() {
        for invalid in ["", "00", &"g".repeat(64), &"0".repeat(63), &"0".repeat(65)] {
            assert!(invalid.parse::<Sha256Digest>().is_err(), "{invalid:?}");
        }
    }

    #[tokio::test]
    async fn verify_refines_only_the_matching_artifact() {
        // upholds: LSRV-5 — the opaque verified value is returned only after
        // hashing the complete canonical file selected by resolution.
        let directory = tempfile::tempdir().unwrap();
        let bytes = b"small deterministic GGUF fixture";
        std::fs::write(directory.path().join("model.gguf"), bytes).unwrap();
        let artifact = resolve_directory(directory.path().to_path_buf())
            .unwrap()
            .into_gguf()
            .unwrap();
        let expected = Sha256Digest::of_bytes(bytes);
        let verified = verify(artifact, &expected).await.unwrap();
        assert_eq!(verified.digest(), &expected);
        assert_eq!(
            verified.path(),
            directory.path().join("model.gguf").canonicalize().unwrap()
        );
    }

    #[tokio::test]
    async fn stamp_discharges_the_rehash_and_pins_its_trust_boundary() {
        // upholds: LSRV-5 (stamp amendment) — a successful full verify
        // writes the stamp, and a later verify whose file stats match
        // trusts it without re-hashing. The second half deliberately pins
        // the ACCEPTED GAP, not an aspiration: bytes rewritten while the
        // stamp is forged to match the new stats still pass, because
        // size+mtime is the whole of the change detection.
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("model.gguf");
        let bytes = b"small deterministic GGUF fixture";
        std::fs::write(&path, bytes).unwrap();
        let resolve = || {
            resolve_directory(directory.path().to_path_buf())
                .unwrap()
                .into_gguf()
                .unwrap()
        };
        let expected = Sha256Digest::of_bytes(bytes);
        verify(resolve(), &expected).await.unwrap();
        let stamp = stamp_path(&path.canonicalize().unwrap());
        assert!(stamp.exists(), "the full verify wrote its stamp");

        // Corrupt the bytes (same length), then forge the stamp's stat
        // fields to the corrupted file's identity: stamp-trust passes
        // without hashing — the documented gap, witnessed on purpose.
        std::fs::write(&path, b"corrupt deterministic GGUF fixtur").unwrap();
        let canonical = path.canonicalize().unwrap();
        let (size, mtime_s, mtime_ns) = stat_identity(&canonical).unwrap();
        std::fs::write(&stamp, format!("{size} {mtime_s} {mtime_ns} {expected}\n")).unwrap();
        let trusted = verify(resolve(), &expected).await.unwrap();
        assert_eq!(trusted.digest(), &expected);
    }

    #[test]
    fn a_shifting_identity_is_never_stamped() {
        // upholds: LSRV-5 (stamp amendment) — the stamp binds a digest to
        // the one identity observed both before and after hashing. A file
        // replaced mid-hash (differing samples) or unstattable at either
        // edge yields no identity, so no stamp can bind the digest to
        // bytes that were never hashed.
        let a = (10u64, 100u64, 5u32);
        let b = (10u64, 100u64, 6u32);
        assert_eq!(hashed_identity(Some(a), Some(a)), Some(a));
        assert_eq!(hashed_identity(Some(a), Some(b)), None);
        assert_eq!(hashed_identity(None, Some(a)), None);
        assert_eq!(hashed_identity(Some(a), None), None);
    }

    #[tokio::test]
    async fn any_observable_change_or_repin_pays_the_full_hash() {
        // upholds: LSRV-5 (stamp amendment) — a size change, an mtime
        // change (any rewrite), a re-pinned expectation, and a malformed
        // stamp each fall back to the full hash, which then tells the
        // truth about the bytes.
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("model.gguf");
        let bytes = b"small deterministic GGUF fixture";
        std::fs::write(&path, bytes).unwrap();
        let resolve = || {
            resolve_directory(directory.path().to_path_buf())
                .unwrap()
                .into_gguf()
                .unwrap()
        };
        let expected = Sha256Digest::of_bytes(bytes);
        verify(resolve(), &expected).await.unwrap();
        let canonical = path.canonicalize().unwrap();
        let stamp = stamp_path(&canonical);

        // Same-length corruption: the rewrite moves mtime, the stamp
        // mismatches, the full hash runs and catches it.
        std::fs::write(&path, b"corrupt deterministic GGUF fixtur").unwrap();
        let error = verify(resolve(), &expected).await.unwrap_err();
        assert!(error.to_string().contains("SHA-256 mismatch"), "{error}");

        // Restore; a passing full hash re-stamps.
        std::fs::write(&path, bytes).unwrap();
        verify(resolve(), &expected).await.unwrap();

        // Size change: caught.
        std::fs::write(&path, b"grown").unwrap();
        std::fs::write(&path, [bytes.as_slice(), b"x"].concat()).unwrap();
        let error = verify(resolve(), &expected).await.unwrap_err();
        assert!(error.to_string().contains("SHA-256 mismatch"), "{error}");

        // Restore and stamp; a re-pinned expectation ignores the stamp and
        // re-proves against the new pin (here: honestly failing).
        std::fs::write(&path, bytes).unwrap();
        verify(resolve(), &expected).await.unwrap();
        let repinned = Sha256Digest::of_bytes(b"a different pin");
        let error = verify(resolve(), &repinned).await.unwrap_err();
        assert!(error.to_string().contains("SHA-256 mismatch"), "{error}");

        // Malformed stamp: ignored, full verify passes and rewrites it.
        std::fs::write(&stamp, "not a stamp at all").unwrap();
        verify(resolve(), &expected).await.unwrap();
        let rewritten = std::fs::read_to_string(&stamp).unwrap();
        assert!(rewritten.ends_with(&format!("{expected}\n")), "{rewritten}");
    }

    #[tokio::test]
    async fn wrong_digest_cannot_construct_a_verified_artifact() {
        // upholds: LSRV-5 — verification failure occurs before a verified
        // spawn input exists; the error names both expected and actual bytes.
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("model.gguf");
        std::fs::write(&path, b"actual").unwrap();
        let artifact = resolve_directory(directory.path().to_path_buf())
            .unwrap()
            .into_gguf()
            .unwrap();
        let expected = Sha256Digest::of_bytes(b"expected");
        let actual = Sha256Digest::of_bytes(b"actual");
        let error = verify(artifact, &expected).await.unwrap_err();
        let message = error.to_string();
        assert!(message.contains(&expected.to_string()), "{message}");
        assert!(message.contains(&actual.to_string()), "{message}");
    }
}
