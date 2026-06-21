//! Authority as values.
//!
//! A capability is a value a tool must *hold* to act, and whose authority is
//! bounded by construction. [`Dir`] is a rooted filesystem capability: the only
//! way the filesystem tools touch disk, and they can only reach paths under the
//! root (CAP-1). [`WriteDir`] is the write-side filesystem capability: authority
//! to write files under a root, separate from read/list authority. [`WebOrigin`]
//! grants HTTP(S) read authority for one origin. [`NtfyTopic`] is a notification
//! capability: authority to publish to one pre-shared ntfy topic, not arbitrary
//! URLs or topics.
//!
//! Honesty: Rust gives *capabilities by construction + enforced containment*,
//! not language-enforced object-capabilities (cf. Eio). We don't hand tools
//! ambient `std::fs`/`std::process`; the containment below is enforced.

use anyhow::{bail, Result};
use reqwest::Url;
use std::path::{Path, PathBuf};

const DEFAULT_NTFY_SERVER: &str = "https://ntfy.sh";

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
    /// reusing the `is_safe_relative` containment check / MS-3).
    pub fn resolve(&self, rel: &str) -> Result<PathBuf> {
        if !crate::is_safe_relative(rel) {
            bail!("path {rel:?} escapes the capability root {:?}", self.root);
        }
        Ok(self.root.join(rel))
    }
}

/// A rooted filesystem write capability: authority to write paths under `root`,
/// and nowhere else.
#[derive(Debug, Clone)]
pub struct WriteDir {
    root: PathBuf,
}

impl WriteDir {
    /// Create a write capability rooted at `root`.
    pub fn new(root: impl Into<PathBuf>) -> WriteDir {
        WriteDir { root: root.into() }
    }

    /// The capability's root directory.
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Resolve a relative path under the root, rejecting anything that could
    /// escape it.
    pub fn resolve(&self, rel: &str) -> Result<PathBuf> {
        if !crate::is_safe_relative(rel) {
            bail!(
                "path {rel:?} escapes the write capability root {:?}",
                self.root
            );
        }
        Ok(self.root.join(rel))
    }
}

/// Authority to read HTTP(S) URLs from one origin.
#[derive(Debug, Clone)]
pub struct WebOrigin {
    origin: Url,
}

impl WebOrigin {
    /// Create a capability for one HTTP(S) origin, e.g. `https://example.com`.
    ///
    /// The origin must not include a path, query, or fragment. The resulting
    /// capability may resolve relative paths or absolute URLs on the same
    /// origin, but not arbitrary hosts.
    pub fn new(origin: &str) -> Result<WebOrigin> {
        Ok(WebOrigin {
            origin: parse_origin_url(origin, "web origin")?,
        })
    }

    /// The configured origin.
    pub fn origin(&self) -> &Url {
        &self.origin
    }

    /// Resolve `target` under this origin. Absolute targets must have the same
    /// scheme/host/port. Fragments are rejected because they are client-side only
    /// and should not affect a fetched document.
    pub fn resolve(&self, target: &str) -> Result<Url> {
        let url = Url::parse(target).or_else(|_| self.origin.join(target))?;
        if !same_origin(&self.origin, &url) {
            bail!(
                "url {url} escapes web origin {}",
                self.origin.as_str().trim_end_matches('/')
            );
        }
        if url.fragment().is_some() {
            bail!("url fragments are not fetchable");
        }
        Ok(url)
    }
}

/// Authority to publish notifications to one pre-shared ntfy topic.
///
/// ntfy topics are created outside Yatima by the user subscribing/publishing to
/// them; the topic string is effectively a shared secret on public ntfy.sh. The
/// capability fixes both server and topic at construction time so a tool holding
/// it can send notifications, but cannot choose arbitrary network destinations
/// or publish to a different topic.
#[derive(Debug, Clone)]
pub struct NtfyTopic {
    server: Url,
    topic: String,
}

impl NtfyTopic {
    /// Create a capability for `topic` on the public ntfy.sh server.
    pub fn new(topic: impl Into<String>) -> Result<NtfyTopic> {
        Self::with_server(DEFAULT_NTFY_SERVER, topic)
    }

    /// Create a capability for `topic` on an explicit ntfy server.
    ///
    /// `server` must be an `http` or `https` origin URL with no path, query, or
    /// fragment. `topic` follows ntfy's public topic syntax:
    /// `[-_A-Za-z0-9]`, 1 to 64 characters.
    pub fn with_server(server: &str, topic: impl Into<String>) -> Result<NtfyTopic> {
        let server = parse_ntfy_server(server)?;
        let topic = topic.into();
        validate_ntfy_topic(&topic)?;
        Ok(NtfyTopic { server, topic })
    }

    /// The configured ntfy server origin.
    pub fn server(&self) -> &Url {
        &self.server
    }

    /// The pre-shared topic this capability may publish to.
    pub fn topic(&self) -> &str {
        &self.topic
    }

    /// The publish endpoint for this capability.
    pub fn endpoint(&self) -> Url {
        let mut url = self.server.clone();
        url.set_path(&self.topic);
        url
    }
}

fn parse_ntfy_server(server: &str) -> Result<Url> {
    parse_origin_url(server, "ntfy server")
}

fn parse_origin_url(server: &str, what: &str) -> Result<Url> {
    let url = Url::parse(server)?;
    match url.scheme() {
        "http" | "https" => {}
        other => bail!("{what} must use http or https, got {other:?}"),
    }
    if url.host_str().is_none() {
        bail!("{what} must include a host");
    }
    if !matches!(url.path(), "" | "/") || url.query().is_some() || url.fragment().is_some() {
        bail!("{what} must be an origin URL with no path, query, or fragment");
    }
    Ok(url)
}

fn same_origin(a: &Url, b: &Url) -> bool {
    a.scheme() == b.scheme()
        && a.host_str() == b.host_str()
        && a.port_or_known_default() == b.port_or_known_default()
}

fn validate_ntfy_topic(topic: &str) -> Result<()> {
    if topic.is_empty() || topic.len() > 64 {
        bail!("ntfy topic must be 1 to 64 characters");
    }
    if !topic
        .bytes()
        .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_'))
    {
        bail!("ntfy topic may only contain letters, numbers, '-' and '_'");
    }
    Ok(())
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

    #[test]
    fn ntfy_topic_defaults_server_and_validates_topic() {
        // upholds: CAP-2 — notification authority is a value fixed at
        // construction, bounded to one ntfy origin/topic.
        let topic = NtfyTopic::new("we-could-be-coding-haskell").unwrap();
        assert_eq!(topic.server().as_str(), "https://ntfy.sh/");
        assert_eq!(topic.topic(), "we-could-be-coding-haskell");
        assert_eq!(
            topic.endpoint().as_str(),
            "https://ntfy.sh/we-could-be-coding-haskell"
        );

        for bad in ["", "has space", "slash/topic", "dot.topic"] {
            assert!(NtfyTopic::new(bad).is_err(), "{bad:?} must be rejected");
        }
        assert!(NtfyTopic::new("a".repeat(65)).is_err());
    }

    #[test]
    fn write_dir_resolve_contains_paths() {
        // upholds: CAP-1
        let d = WriteDir::new("/data");
        assert_eq!(
            d.resolve("a/b.txt").unwrap(),
            PathBuf::from("/data/a/b.txt")
        );
        for bad in ["../etc/passwd", "/etc/passwd", "a/../../b", "./x"] {
            assert!(d.resolve(bad).is_err(), "{bad:?} must be rejected");
        }
    }

    #[test]
    fn web_origin_resolves_only_same_origin() {
        // upholds: CAP-2 — web read authority is bounded to one origin.
        let web = WebOrigin::new("https://example.com").unwrap();
        assert_eq!(
            web.resolve("/docs?q=rust").unwrap().as_str(),
            "https://example.com/docs?q=rust"
        );
        assert_eq!(
            web.resolve("docs/page").unwrap().as_str(),
            "https://example.com/docs/page"
        );
        assert!(web.resolve("https://example.com/ok").is_ok());
        assert!(web.resolve("http://example.com/no").is_err());
        assert!(web.resolve("https://evil.example/no").is_err());
        assert!(web.resolve("https://example.com/page#frag").is_err());
    }

    #[test]
    fn ntfy_server_is_origin_only() {
        // upholds: CAP-2 — the capability cannot smuggle arbitrary URL paths or
        // query strings into the publish destination.
        assert!(NtfyTopic::with_server("http://127.0.0.1:8080", "topic").is_ok());
        for bad in [
            "ftp://ntfy.sh",
            "https://ntfy.sh/base",
            "https://ntfy.sh?x=1",
            "https://ntfy.sh/#frag",
        ] {
            assert!(
                NtfyTopic::with_server(bad, "topic").is_err(),
                "{bad:?} must be rejected"
            );
        }
    }
}
