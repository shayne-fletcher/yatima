//! Authority as values.
//!
//! A capability is a value a tool must *hold* to act, and whose authority is
//! bounded by construction. [`Dir`] is a rooted filesystem capability: the only
//! way the filesystem tools touch disk, and they can only reach paths under the
//! root (CAP-1). Future capabilities (`Net`, `Proc`, write variants) follow the
//! same shape.
//!
//! Honesty: Rust gives *capabilities by construction + enforced containment*,
//! not language-enforced object-capabilities (cf. Eio). We don't hand tools
//! ambient `std::fs`/`std::process`; the containment below is enforced.

use anyhow::{bail, Result};
use std::path::{Path, PathBuf};

/// A rooted filesystem capability: authority to reach paths under `root`, and
/// nowhere else.
#[derive(Debug, Clone)]
pub struct Dir {
    root: PathBuf,
}

impl Dir {
    /// Create a capability rooted at `root`.
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    /// The capability's root directory.
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Resolve a relative path under the root, rejecting anything that could
    /// escape it — absolute paths, `..`, or other non-normal components (CAP-1,
    /// reusing the [`crate::is_safe_relative`] containment check / MS-3).
    pub fn resolve(&self, rel: &str) -> Result<PathBuf> {
        if !crate::is_safe_relative(rel) {
            bail!("path {rel:?} escapes the capability root {:?}", self.root);
        }
        Ok(self.root.join(rel))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dir_resolve_contains_paths() {
        // upholds: CAP-1
        let d = Dir::new("/data");
        assert_eq!(
            d.resolve("a/b.txt").unwrap(),
            PathBuf::from("/data/a/b.txt")
        );
        for bad in ["../etc/passwd", "/etc/passwd", "a/../../b", "./x"] {
            assert!(d.resolve(bad).is_err(), "{bad:?} must be rejected");
        }
    }
}
