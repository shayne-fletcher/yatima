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

/// A **growable set** of HTTP(S) origins — the runtime-grant generalization of
/// [`WebOrigin`] (CAP-2: a web tool's authority is exactly its held origin
/// set). The set is shared: the host keeps a clone and grants/revokes
/// mid-session (CAP-3 — grants derive only from user utterances; nothing a
/// tool or the model produces reaches [`grant`](WebOrigins::grant)); tools
/// holding the same instance see the change on their next call, and their
/// rendered specs reflect it (CAP-3a). Starts empty: sandbox-by-omission
/// holds at every launch.
#[derive(Debug, Clone, Default)]
pub struct WebOrigins {
    origins: std::sync::Arc<std::sync::RwLock<Vec<WebOrigin>>>,
}

impl WebOrigins {
    /// An empty origin set — no web authority at all.
    pub fn new() -> WebOrigins {
        WebOrigins::default()
    }

    /// A set holding a single pre-granted origin — the one-shot (CLI) shape.
    pub fn one(origin: &str) -> Result<WebOrigins> {
        let set = WebOrigins::new();
        set.grant(origin)?;
        Ok(set)
    }

    /// Grant `origin` (e.g. `https://example.com`) for the rest of the
    /// session. Idempotent; returns whether the set actually grew.
    pub fn grant(&self, origin: &str) -> Result<bool> {
        let o = WebOrigin::new(origin)?;
        let mut set = self.origins.write().expect("origin set poisoned");
        if set.iter().any(|w| same_origin(w.origin(), o.origin())) {
            return Ok(false);
        }
        set.push(o);
        Ok(true)
    }

    /// Revoke `origin`; returns whether it was present.
    pub fn revoke(&self, origin: &str) -> Result<bool> {
        let o = WebOrigin::new(origin)?;
        let mut set = self.origins.write().expect("origin set poisoned");
        let before = set.len();
        set.retain(|w| !same_origin(w.origin(), o.origin()));
        Ok(set.len() < before)
    }

    /// The granted origins, render-ready (scheme://host[:port], no trailing /).
    pub fn list(&self) -> Vec<String> {
        self.origins
            .read()
            .expect("origin set poisoned")
            .iter()
            .map(|w| w.origin().as_str().trim_end_matches('/').to_string())
            .collect()
    }

    /// True when no origin has been granted.
    pub fn is_empty(&self) -> bool {
        self.origins.read().expect("origin set poisoned").is_empty()
    }

    /// Resolve `target` against the set: an absolute URL must match a granted
    /// origin; a relative path resolves only when exactly one origin is
    /// granted (with several, relative targets are ambiguous and refused).
    pub fn resolve(&self, target: &str) -> Result<Url> {
        let set = self.origins.read().expect("origin set poisoned");
        if set.is_empty() {
            bail!("no web origin granted");
        }
        if let Ok(url) = Url::parse(target) {
            let Some(origin) = set.iter().find(|w| same_origin(w.origin(), &url)) else {
                bail!(
                    "url {url} escapes the granted web origins [{}]",
                    set.iter()
                        .map(|w| w.origin().as_str().trim_end_matches('/'))
                        .collect::<Vec<_>>()
                        .join(", ")
                );
            };
            return origin.resolve(target);
        }
        match set.as_slice() {
            [only] => only.resolve(target),
            _ => bail!(
                "relative url {target:?} is ambiguous with {} origins granted; \
                 use an absolute url",
                set.len()
            ),
        }
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

/// The HTTP(S) origins named by URLs appearing in `text`.
///
/// This is the auto-grant scanner and it is only ever fed **user utterances**
/// (CAP-3): a URL the model or a fetched page produces must never pass
/// through here. Mid-sentence URLs are found, trailing punctuation is
/// trimmed, duplicates collapse (first occurrence wins), and non-HTTP
/// schemes are ignored.
pub fn origins_in(text: &str) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    for word in text.split_whitespace() {
        let Some(at) = word.find("http://").or_else(|| word.find("https://")) else {
            continue;
        };
        let candidate =
            word[at..].trim_end_matches(['.', ',', ';', ':', '!', '?', ')', ']', '"', '\'', '>']);
        let Ok(url) = Url::parse(candidate) else {
            continue;
        };
        if url.host_str().is_none() {
            continue;
        }
        let origin = format!(
            "{}://{}{}",
            url.scheme(),
            url.host_str().unwrap_or_default(),
            url.port().map(|p| format!(":{p}")).unwrap_or_default()
        );
        if !out.contains(&origin) {
            out.push(origin);
        }
    }
    out
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
    fn web_origins_grant_accumulates_and_resolves_by_membership() {
        // upholds: CAP-2 (origin-set form) — authority is exactly the granted
        // set: in-set absolute URLs resolve, out-of-set are refused, and the
        // set is the union of grants.
        let set = WebOrigins::new();
        assert!(set.resolve("https://a.example/x").is_err(), "empty = none");
        assert!(set.grant("https://a.example").unwrap());
        assert!(!set.grant("https://a.example").unwrap(), "idempotent");
        assert!(set.grant("https://b.example").unwrap());
        assert!(set.resolve("https://a.example/x").is_ok());
        assert!(set.resolve("https://b.example/y").is_ok());
        let err = set.resolve("https://c.example/z").unwrap_err().to_string();
        assert!(err.contains("escapes the granted web origins"), "{err}");
        assert_eq!(set.list(), ["https://a.example", "https://b.example"]);
    }

    #[test]
    fn web_origins_relative_needs_exactly_one_origin() {
        // upholds: CAP-2 — a relative path is unambiguous only with a single
        // granted origin; with several it is refused, not guessed.
        let set = WebOrigins::one("https://a.example").unwrap();
        assert_eq!(
            set.resolve("/wiki/x").unwrap().as_str(),
            "https://a.example/wiki/x"
        );
        set.grant("https://b.example").unwrap();
        let err = set.resolve("/wiki/x").unwrap_err().to_string();
        assert!(err.contains("ambiguous"), "{err}");
    }

    #[test]
    fn web_origins_revoke_shrinks_authority() {
        // upholds: CAP-3 — /revoke is the only way authority shrinks, and it
        // works immediately.
        let set = WebOrigins::one("https://a.example").unwrap();
        assert!(set.revoke("https://a.example").unwrap());
        assert!(!set.revoke("https://a.example").unwrap(), "already gone");
        assert!(set.is_empty());
        assert!(set.resolve("https://a.example/x").is_err());
    }

    #[test]
    fn origins_in_finds_user_typed_urls_only_by_construction() {
        // upholds: CAP-3 — the auto-grant scanner: mid-sentence URLs found,
        // trailing punctuation trimmed, duplicates collapsed, non-HTTP
        // schemes ignored. (That it is only ever fed user utterances is the
        // host's obligation; the lib offers no other grant path.)
        assert_eq!(
            origins_in("summarize https://en.wikipedia.org/wiki/Roger_Penrose please"),
            ["https://en.wikipedia.org"]
        );
        assert_eq!(
            origins_in("see (https://a.example/x), and http://b.example:8080/y."),
            ["https://a.example", "http://b.example:8080"]
        );
        assert_eq!(
            origins_in("https://a.example/1 then https://a.example/2"),
            ["https://a.example"],
            "duplicates collapse"
        );
        assert!(origins_in("ftp://files.example/x and no urls").is_empty());
        assert!(origins_in("plain text").is_empty());
    }

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
