//! llama-server backends behind [`Completer`].
//!
//! [`LlamaServerCompleter`] is the HTTP transport adapter for an already
//! running loopback server. [`LlamaServer`] owns a child process and delegates
//! the same protocol to an inner adapter. Both send raw rendered prompts and
//! receive streamed text; Yatima remains the owner of transcript and format
//! meaning. Their I/O is asynchronous, with futures whose `Send` is inferred
//! per instantiation (CMP-1): [`complete`](Completer::complete) is `Send`, while
//! [`complete_streaming`](Completer::complete_streaming) is not generally
//! `Send` because its callback has no such bound.
//! Managed child death is observed during startup, introspection, completion,
//! and shutdown. There is deliberately no monitor while the backend is idle;
//! the next completion or explicit shutdown observes an intervening exit.
//!
//! Wire behavior is implemented against the stage-0 observations recorded in
//! `tests/fixtures/llama_server/provenance.json` (llama.cpp build 10520):
//!
//! - The server **excludes** a matched stop word from emitted content and
//!   names it in the final event's `stopping_word`; the `Completion` contract
//!   includes the marker, so it is re-appended and emitted.
//! - `stop_type`: `word` → [`StopReason::Stopped`], `eos` → `Eos`, `limit` →
//!   `MaxTokens`; anything else — or a stream that ends without a final stop
//!   event — is a protocol error, never a completion with an invented reason.
//! - The server's sampling defaults are non-neutral (`top_k 40`, `top_p
//!   0.95`, `min_p 0.05`), so every request pins the full recipe explicitly;
//!   omission would silently change yatima's policy.

use super::sse::SseFramer;
use crate::{Cancel, Completer, Completion, GenOpts, GgufArtifact, Sampling, StopReason};
use anyhow::{bail, Context, Result};
use serde_json::Value;
use std::collections::VecDeque;
use std::net::{Ipv4Addr, Ipv6Addr};
use std::path::{Path, PathBuf};
use std::process::{ExitStatus, Stdio};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::io::{AsyncRead, AsyncReadExt};
use tokio::process::{Child, Command};
use tokio::task::JoinHandle;
use tokio::time::{Instant, MissedTickBehavior};
use url::{Host, Url};

const CONTROL_TIMEOUT: Duration = Duration::from_secs(10);
const DEFAULT_READINESS_TIMEOUT: Duration = Duration::from_secs(180);
const CLEANUP_TIMEOUT: Duration = Duration::from_secs(10);
const CHILD_DEATH_GRACE: Duration = Duration::from_millis(50);
const STARTUP_ATTEMPTS: usize = 3;
const TAIL_CAPACITY: usize = 64 * 1024;

#[derive(Debug, Clone)]
struct ValidatedEndpoint(Url);

impl ValidatedEndpoint {
    fn parse(base_url: &str) -> Result<ValidatedEndpoint> {
        if base_url.contains('@') {
            // `url` normalizes some empty-userinfo spellings away, including
            // slash-deficient forms. No accepted origin can legitimately
            // contain `@`, so reject that syntax before parsing.
            bail!("llama-server URL must not contain userinfo");
        }
        let mut url = Url::parse(base_url).context("parse llama-server URL")?;
        if url.scheme() != "http" {
            bail!("llama-server URL must use http");
        }
        if url.cannot_be_a_base() {
            bail!("llama-server URL must be an origin");
        }
        if !url.username().is_empty() || url.password().is_some() {
            bail!("llama-server URL must not contain userinfo");
        }
        if url.path() != "/" || url.query().is_some() || url.fragment().is_some() {
            bail!("llama-server URL must be an origin with no path, query, or fragment");
        }

        match url.host() {
            Some(Host::Ipv4(ip)) if ip.is_loopback() => {}
            Some(Host::Ipv6(ip)) if ip == Ipv6Addr::LOCALHOST => {}
            Some(Host::Domain(name)) if name.eq_ignore_ascii_case("localhost") => {
                url.set_host(Some(&Ipv4Addr::LOCALHOST.to_string()))
                    .map_err(|_| anyhow::anyhow!("normalize localhost llama-server URL"))?;
            }
            _ => bail!("llama-server URL must name a loopback address"),
        }
        Ok(ValidatedEndpoint(url))
    }

    fn join(&self, path: &str) -> Url {
        self.0
            .join(path)
            .expect("validated origin accepts relative endpoint paths")
    }

    fn as_str(&self) -> &str {
        self.0.as_str()
    }
}

/// Configuration for an attached llama-server backend.
#[derive(Debug, Clone)]
pub struct LlamaServerConfig {
    endpoint: ValidatedEndpoint,
    /// Server-side top-k filter; `0` disables it (the neutral default — the
    /// server's own default is 40). Non-zero only when a model's recipe
    /// deliberately asks for it (Muse Glimmer: 64, stage 3).
    top_k: u32,
    control_timeout: Duration,
}

impl LlamaServerConfig {
    /// Validate and normalize an attached server's loopback origin (LSRV-2).
    pub fn new(base_url: impl AsRef<str>) -> Result<LlamaServerConfig> {
        Ok(LlamaServerConfig {
            endpoint: ValidatedEndpoint::parse(base_url.as_ref())?,
            top_k: 0,
            control_timeout: CONTROL_TIMEOUT,
        })
    }

    /// Select a server-side top-k filter. Zero disables the filter.
    pub fn with_top_k(mut self, top_k: u32) -> LlamaServerConfig {
        self.top_k = top_k;
        self
    }

    /// The normalized loopback origin used for requests.
    pub fn base_url(&self) -> &str {
        self.endpoint.as_str()
    }

    #[cfg(test)]
    fn with_control_timeout(mut self, timeout: Duration) -> LlamaServerConfig {
        self.control_timeout = timeout;
        self
    }
}

/// Untrusted compatibility and diagnostic data reported by `/props`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ServerProps {
    pub build: String,
    pub model_self_report: String,
    pub n_ctx: u32,
    pub total_slots: u32,
}

/// The transport adapter. Holds only `Send` state, so the non-streaming
/// completion future is `Send` (asserted in tests).
pub struct LlamaServerCompleter {
    client: reqwest::Client,
    config: LlamaServerConfig,
}

impl LlamaServerCompleter {
    pub fn new(config: LlamaServerConfig) -> Result<LlamaServerCompleter> {
        let client = reqwest::Client::builder()
            .connect_timeout(Duration::from_secs(5))
            .redirect(reqwest::redirect::Policy::none())
            .build()
            .context("build llama-server HTTP client")?;
        Ok(LlamaServerCompleter { client, config })
    }

    /// Read the server's self-reported properties under the control timeout.
    /// This establishes only that the expected `/props` fields can be parsed;
    /// the model path and build string are diagnostics, not identity evidence.
    pub async fn introspect(&self) -> Result<ServerProps> {
        let url = self.config.endpoint.join("props");
        let value = self.control_json(url.clone()).await?;
        let build = required_str(&value, "build_info")?;
        let model_self_report = required_str(&value, "model_path")?;
        let n_ctx = value
            .pointer("/default_generation_settings/n_ctx")
            .and_then(Value::as_u64)
            .ok_or_else(|| anyhow::anyhow!("llama-server /props has no integer n_ctx"))?
            .try_into()
            .context("llama-server /props n_ctx does not fit u32")?;
        let total_slots = value
            .get("total_slots")
            .and_then(Value::as_u64)
            .ok_or_else(|| anyhow::anyhow!("llama-server /props has no integer total_slots"))?
            .try_into()
            .context("llama-server /props total_slots does not fit u32")?;
        Ok(ServerProps {
            build,
            model_self_report,
            n_ctx,
            total_slots,
        })
    }

    async fn control_json(&self, url: Url) -> Result<Value> {
        let request = async {
            let response = self
                .client
                .get(url.clone())
                .send()
                .await
                .with_context(|| format!("GET {url}"))?;
            let status = response.status();
            let bytes = response
                .bytes()
                .await
                .with_context(|| format!("read {url} body"))?;
            if !status.is_success() {
                bail!(
                    "llama-server at {url} returned {status}: {}",
                    String::from_utf8_lossy(&bytes)
                );
            }
            serde_json::from_slice(&bytes).with_context(|| format!("parse {url} response"))
        };
        tokio::time::timeout(self.config.control_timeout, request)
            .await
            .with_context(|| {
                format!(
                    "llama-server control request to {url} timed out after {:?}",
                    self.config.control_timeout
                )
            })?
    }

    async fn health(&self) -> Result<bool> {
        let url = self.config.endpoint.join("health");
        let request = async {
            let response = self
                .client
                .get(url.clone())
                .send()
                .await
                .with_context(|| format!("GET {url}"))?;
            let status = response.status();
            response
                .bytes()
                .await
                .with_context(|| format!("read {url} body"))?;
            Ok(status.is_success())
        };
        tokio::time::timeout(self.config.control_timeout, request)
            .await
            .with_context(|| {
                format!(
                    "llama-server control request to {url} timed out after {:?}",
                    self.config.control_timeout
                )
            })?
    }

    /// The one generation loop behind both `Completer` methods, generic over
    /// the token sink so `Send` is inferred per instantiation: `complete`
    /// passes a no-capture closure (future `Send`), `complete_streaming`
    /// passes the caller's `&mut dyn FnMut` (future not `Send`).
    async fn run<F: FnMut(&str)>(
        &mut self,
        prompt: &str,
        opts: &GenOpts,
        stops: &[String],
        cancel: &Cancel,
        on_token: &mut F,
    ) -> Result<Completion> {
        let url = self.config.endpoint.join("completion");
        let body = request_body(&self.config, prompt, opts, stops);
        let send = self.client.post(url.clone()).json(&body).send();
        let mut resp = tokio::select! {
            _ = cancel.cancelled() => return Ok(cancelled_completion(String::new())),
            response = send => response.with_context(|| format!("POST {url}"))?,
        };
        if !resp.status().is_success() {
            let status = resp.status();
            let body = tokio::select! {
                _ = cancel.cancelled() => return Ok(cancelled_completion(String::new())),
                body = tokio::time::timeout(self.config.control_timeout, resp.text()) => {
                    body.with_context(|| {
                        format!(
                            "llama-server error body from {url} timed out after {:?}",
                            self.config.control_timeout
                        )
                    })?.unwrap_or_default()
                }
            };
            bail!("llama-server at {url} returned {status}: {body}");
        }

        let mut framer = SseFramer::new();
        let mut text = String::new();
        loop {
            // LSRV-4: cancellation is observed per iteration so a continuously
            // ready stream cannot starve the cancellation branch.
            if cancel.is_cancelled() {
                return Ok(cancelled_completion(text));
            }
            // The awaitable side of CANCEL-1 races the pending read directly;
            // a silent server therefore has no polling interval to hide in.
            let chunk = tokio::select! {
                _ = cancel.cancelled() => return Ok(cancelled_completion(text)),
                read = resp.chunk() => read.with_context(|| format!("read {url} stream"))?,
            };
            let Some(chunk) = chunk else {
                bail!("llama-server stream ended without a final stop event (protocol error)");
            };
            for payload in framer.push(&chunk)? {
                // LSRV-4: one network chunk can carry many buffered events;
                // honor a cancel raised mid-batch (e.g. from `on_token`
                // itself) without draining the rest.
                if cancel.is_cancelled() {
                    return Ok(cancelled_completion(text));
                }
                let event: serde_json::Value = serde_json::from_str(&payload)
                    .context("malformed llama-server stream event")?;
                if let Some(content) = event.get("content").and_then(|v| v.as_str()) {
                    if !content.is_empty() {
                        text.push_str(content);
                        on_token(content);
                    }
                }
                if event.get("stop").and_then(|v| v.as_bool()) == Some(true) {
                    let stop_type = event.get("stop_type").and_then(|v| v.as_str());
                    let stop = match stop_type {
                        Some("eos") => StopReason::Eos,
                        Some("limit") => StopReason::MaxTokens,
                        Some("word") => {
                            // LSRV-3: the server excluded the matched stop
                            // string; the Completion contract includes it, and
                            // a codec needs the complete block — re-append and
                            // emit. A word stop that names no marker, or names
                            // one the caller never supplied, breaks that
                            // contract: protocol error, never a quiet Stopped.
                            let word = event
                                .get("stopping_word")
                                .and_then(|v| v.as_str())
                                .unwrap_or("");
                            if word.is_empty() {
                                bail!(
                                    "llama-server reported stop_type \"word\" without a \
                                     stopping_word (protocol error)"
                                );
                            }
                            if !stops.iter().any(|s| s == word) {
                                bail!(
                                    "llama-server stopping_word {word:?} was not among the \
                                     requested stop strings (protocol error)"
                                );
                            }
                            text.push_str(word);
                            on_token(word);
                            StopReason::Stopped
                        }
                        other => bail!(
                            "llama-server reported unexpected stop_type {other:?} \
                             (protocol error)"
                        ),
                    };
                    return Ok(Completion { text, stop });
                }
            }
        }
    }
}

fn cancelled_completion(text: String) -> Completion {
    Completion {
        text,
        stop: StopReason::Stopped,
    }
}

fn required_str(value: &Value, field: &str) -> Result<String> {
    value
        .get(field)
        .and_then(Value::as_str)
        .map(str::to_owned)
        .ok_or_else(|| anyhow::anyhow!("llama-server /props has no string {field}"))
}

impl Completer for LlamaServerCompleter {
    async fn complete(
        &mut self,
        prompt: &str,
        opts: &GenOpts,
        stops: &[String],
    ) -> Result<Completion> {
        self.run(prompt, opts, stops, &Cancel::new(), &mut |_| {})
            .await
    }

    async fn complete_streaming(
        &mut self,
        prompt: &str,
        opts: &GenOpts,
        stops: &[String],
        cancel: &Cancel,
        on_token: &mut dyn FnMut(&str),
    ) -> Result<Completion> {
        self.run(prompt, opts, stops, cancel, &mut |s: &str| on_token(s))
            .await
    }
}

/// How to launch one managed llama-server process.
pub struct LlamaServerSpawn {
    pub binary: PathBuf,
    pub gguf: GgufArtifact,
    pub context: Option<u32>,
    pub readiness_timeout: Duration,
    pub top_k: u32,
}

impl LlamaServerSpawn {
    pub fn new(gguf: GgufArtifact) -> LlamaServerSpawn {
        LlamaServerSpawn {
            binary: PathBuf::from("llama-server"),
            gguf,
            context: None,
            readiness_timeout: DEFAULT_READINESS_TIMEOUT,
            top_k: 0,
        }
    }
}

/// A managed llama-server child and its HTTP completer. The child handle has
/// one owner; only the bounded diagnostic tails are shared with drain tasks.
pub struct LlamaServer {
    child: Child,
    stdout_drain: Option<JoinHandle<std::io::Result<()>>>,
    stderr_drain: Option<JoinHandle<std::io::Result<()>>>,
    stdout_tail: SharedTail,
    stderr_tail: SharedTail,
    completer: LlamaServerCompleter,
    props: Option<ServerProps>,
    artifact: GgufArtifact,
}

impl LlamaServer {
    /// Spawn, await readiness, and introspect a managed server. Early exits are
    /// retried with a fresh port; all child-owning failures reap and join their
    /// drains before returning (LSRV-1).
    pub async fn spawn(spec: LlamaServerSpawn) -> Result<LlamaServer> {
        let mut early_failures = Vec::new();
        for attempt in 1..=STARTUP_ATTEMPTS {
            let port = pick_loopback_port().await?;
            let mut server = LlamaServer::start_attempt(&spec, port)
                .with_context(|| format!("spawn managed llama-server attempt {attempt}"))?;
            let deadline = Instant::now() + spec.readiness_timeout;
            match server.wait_until_ready(deadline).await {
                Ok(()) => {}
                Err(StartupWait::EarlyExit(status)) => {
                    let cleanup = server.cleanup().await;
                    let diagnostics = server.diagnostics();
                    early_failures.push(format!(
                        "attempt {attempt} exited during startup ({status}); {cleanup}; \
                         diagnostics:\n{diagnostics}"
                    ));
                    continue;
                }
                Err(StartupWait::Timeout) => {
                    let cleanup = server.cleanup().await;
                    let diagnostics = server.diagnostics();
                    bail!(
                        "managed llama-server readiness timed out after {:?}; {cleanup}; \
                         diagnostics:\n{diagnostics}",
                        spec.readiness_timeout
                    );
                }
                Err(StartupWait::Io(error)) => {
                    let cleanup = server.cleanup().await;
                    let diagnostics = server.diagnostics();
                    return Err(error).context(format!(
                        "managed llama-server readiness failed; {cleanup}; diagnostics:\n{diagnostics}"
                    ));
                }
            }

            enum Introspection {
                Props(Result<ServerProps>),
                Child(std::io::Result<ExitStatus>),
                Deadline,
            }
            let outcome = {
                let request = server.completer.introspect();
                tokio::pin!(request);
                tokio::select! {
                    props = &mut request => Introspection::Props(props),
                    status = server.child.wait() => Introspection::Child(status),
                    _ = tokio::time::sleep_until(deadline) => Introspection::Deadline,
                }
            };
            match outcome {
                Introspection::Props(Ok(props)) => {
                    server.props = Some(props);
                    return Ok(server);
                }
                Introspection::Props(Err(error)) => {
                    let cleanup = server.cleanup().await;
                    let diagnostics = server.diagnostics();
                    return Err(error).context(format!(
                        "managed llama-server introspection failed; {cleanup}; \
                         diagnostics:\n{diagnostics}"
                    ));
                }
                Introspection::Child(status) => {
                    let status = match status {
                        Ok(status) => status.to_string(),
                        Err(error) => format!("wait failed: {error}"),
                    };
                    let cleanup = server.cleanup().await;
                    let diagnostics = server.diagnostics();
                    bail!(
                        "managed llama-server exited during introspection ({status}); {cleanup}; \
                         diagnostics:\n{diagnostics}"
                    );
                }
                Introspection::Deadline => {
                    let cleanup = server.cleanup().await;
                    let diagnostics = server.diagnostics();
                    bail!(
                        "managed llama-server introspection exceeded the startup deadline; \
                         {cleanup}; diagnostics:\n{diagnostics}"
                    );
                }
            }
        }

        bail!(
            "managed llama-server failed during startup after {STARTUP_ATTEMPTS} attempts:\n{}",
            early_failures.join("\n")
        )
    }

    fn start_attempt(spec: &LlamaServerSpawn, port: u16) -> Result<LlamaServer> {
        let config =
            LlamaServerConfig::new(format!("http://127.0.0.1:{port}"))?.with_top_k(spec.top_k);
        let completer = LlamaServerCompleter::new(config)?;
        let mut command = Command::new(&spec.binary);
        command
            .arg("-m")
            .arg(spec.gguf.path())
            .arg("--host")
            .arg("127.0.0.1")
            .arg("--port")
            .arg(port.to_string())
            .arg("-np")
            .arg("1")
            .arg("--jinja")
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true);
        if let Some(context) = spec.context {
            command.arg("-c").arg(context.to_string());
        }
        let mut child = command
            .spawn()
            .with_context(|| format!("execute {}", spec.binary.display()))?;
        let stdout = child.stdout.take().expect("stdout was configured as piped");
        let stderr = child.stderr.take().expect("stderr was configured as piped");
        let stdout_tail = SharedTail::new();
        let stderr_tail = SharedTail::new();
        let stdout_drain = tokio::spawn(drain_pipe(stdout, stdout_tail.clone()));
        let stderr_drain = tokio::spawn(drain_pipe(stderr, stderr_tail.clone()));
        Ok(LlamaServer {
            child,
            stdout_drain: Some(stdout_drain),
            stderr_drain: Some(stderr_drain),
            stdout_tail,
            stderr_tail,
            completer,
            props: None,
            artifact: spec.gguf.clone(),
        })
    }

    async fn wait_until_ready(
        &mut self,
        deadline: Instant,
    ) -> std::result::Result<(), StartupWait> {
        let mut retry = tokio::time::interval(Duration::from_millis(100));
        retry.set_missed_tick_behavior(MissedTickBehavior::Delay);
        loop {
            enum Observation {
                Health(Result<bool>),
                Child(std::io::Result<ExitStatus>),
                Deadline,
            }
            let observation = {
                let health = self.completer.health();
                tokio::pin!(health);
                tokio::select! {
                    result = &mut health => Observation::Health(result),
                    status = self.child.wait() => Observation::Child(status),
                    _ = tokio::time::sleep_until(deadline) => Observation::Deadline,
                }
            };
            match observation {
                Observation::Health(Ok(true)) => return Ok(()),
                Observation::Health(Ok(false) | Err(_)) => {}
                Observation::Child(Ok(status)) => return Err(StartupWait::EarlyExit(status)),
                Observation::Child(Err(error)) => return Err(StartupWait::Io(error.into())),
                Observation::Deadline => return Err(StartupWait::Timeout),
            }

            tokio::select! {
                _ = retry.tick() => {}
                status = self.child.wait() => {
                    return match status {
                        Ok(status) => Err(StartupWait::EarlyExit(status)),
                        Err(error) => Err(StartupWait::Io(error.into())),
                    };
                }
                _ = tokio::time::sleep_until(deadline) => return Err(StartupWait::Timeout),
            }
        }
    }

    /// Kill, reap, and join both output drains. Consuming `self` makes explicit
    /// shutdown the normal ownership end; `Drop` remains a weaker fallback.
    pub async fn shutdown(mut self) -> Result<ExitStatus> {
        self.cleanup_result().await
    }

    pub fn diagnostics(&self) -> String {
        format!(
            "stdout:\n{}\nstderr:\n{}",
            self.stdout_tail.read(),
            self.stderr_tail.read()
        )
    }

    pub fn props(&self) -> &ServerProps {
        self.props
            .as_ref()
            .expect("LlamaServer::spawn returns only after introspection")
    }

    pub fn launched_artifact(&self) -> &Path {
        self.artifact.path()
    }

    async fn cleanup(&mut self) -> String {
        match self.cleanup_result().await {
            Ok(status) => {
                format!("cleanup reaped child ({status}) and joined stdout/stderr drains")
            }
            Err(error) => format!("cleanup failed: {error:#}"),
        }
    }

    async fn cleanup_result(&mut self) -> Result<ExitStatus> {
        let cleanup = async {
            let (already_exited, poll_error) = match self.child.try_wait() {
                Ok(status) => (status.is_some(), None),
                Err(error) => (false, Some(error)),
            };
            let kill_error = if !already_exited {
                match self.child.start_kill() {
                    Ok(()) => None,
                    Err(error) if error.kind() == std::io::ErrorKind::InvalidInput => None,
                    Err(error) => Some(error),
                }
            } else {
                None
            };
            let wait = self.child.wait().await;
            let drains = self.join_drains().await;

            let status = match wait {
                Ok(status) => status,
                Err(error) => {
                    let poll = poll_error
                        .map(|poll| format!("; initial poll also failed: {poll}"))
                        .unwrap_or_default();
                    let drains = drains
                        .err()
                        .map(|drain| format!("; drain join also failed: {drain:#}"))
                        .unwrap_or_default();
                    bail!("reap managed llama-server child: {error}{poll}{drains}");
                }
            };
            drains?;
            if let Some(error) = kill_error {
                return Err(error).context(format!(
                    "kill managed llama-server child (child was nevertheless reaped as {status})"
                ));
            }
            Ok(status)
        };
        tokio::time::timeout(CLEANUP_TIMEOUT, cleanup)
            .await
            .context("managed llama-server cleanup timed out")?
    }

    async fn join_drains(&mut self) -> Result<()> {
        let stdout = self.stdout_drain.take();
        let stderr = self.stderr_drain.take();
        let join_stdout = async {
            if let Some(task) = stdout {
                task.await.context("join llama-server stdout drain")??;
            }
            Ok::<_, anyhow::Error>(())
        };
        let join_stderr = async {
            if let Some(task) = stderr {
                task.await.context("join llama-server stderr drain")??;
            }
            Ok::<_, anyhow::Error>(())
        };
        let (stdout, stderr) = tokio::join!(join_stdout, join_stderr);
        stdout?;
        stderr?;
        Ok(())
    }

    async fn child_death_error(&mut self, status: std::io::Result<ExitStatus>) -> anyhow::Error {
        let (status, cleanup_note) = match status {
            Ok(status) => {
                let drains = tokio::time::timeout(CLEANUP_TIMEOUT, self.join_drains()).await;
                let note = match drains {
                    Ok(Ok(())) => "stdout/stderr drains joined".to_string(),
                    Ok(Err(error)) => format!("drain join failed: {error:#}"),
                    Err(_) => "drain join timed out".to_string(),
                };
                (status.to_string(), note)
            }
            Err(error) => {
                let cleanup = self.cleanup().await;
                (format!("wait failed: {error}"), cleanup)
            }
        };
        anyhow::anyhow!(
            "managed llama-server died during completion ({status}); {cleanup_note}; diagnostics:\n{}",
            self.diagnostics()
        )
    }

    async fn reconcile_completion(&mut self, result: Result<Completion>) -> Result<Completion> {
        match result {
            Ok(completion) => Ok(completion),
            Err(transport_error) => {
                // Socket closure and process exit become observable on
                // different kernel paths. Give the owned child a small,
                // bounded reconciliation window before reporting a transport
                // error; a live child leaves this wait pending and preserves
                // the original error.
                match tokio::time::timeout(CHILD_DEATH_GRACE, self.child.wait()).await {
                    Ok(status) => Err(self.child_death_error(status).await),
                    Err(_) => Err(transport_error).context(format!(
                        "managed llama-server remained alive after the completion failure; \
                         diagnostics:\n{}",
                        self.diagnostics()
                    )),
                }
            }
        }
    }
}

impl Completer for LlamaServer {
    async fn complete(
        &mut self,
        prompt: &str,
        opts: &GenOpts,
        stops: &[String],
    ) -> Result<Completion> {
        enum Outcome {
            Completion(Result<Completion>),
            Child(std::io::Result<ExitStatus>),
        }
        let outcome = {
            let completion = self.completer.complete(prompt, opts, stops);
            tokio::pin!(completion);
            tokio::select! {
                biased;
                status = self.child.wait() => Outcome::Child(status),
                result = &mut completion => Outcome::Completion(result),
            }
        };
        match outcome {
            Outcome::Completion(result) => self.reconcile_completion(result).await,
            Outcome::Child(status) => Err(self.child_death_error(status).await),
        }
    }

    async fn complete_streaming(
        &mut self,
        prompt: &str,
        opts: &GenOpts,
        stops: &[String],
        cancel: &Cancel,
        on_token: &mut dyn FnMut(&str),
    ) -> Result<Completion> {
        enum Outcome {
            Completion(Result<Completion>),
            Child(std::io::Result<ExitStatus>),
        }
        let outcome = {
            let completion = self
                .completer
                .complete_streaming(prompt, opts, stops, cancel, on_token);
            tokio::pin!(completion);
            tokio::select! {
                biased;
                status = self.child.wait() => Outcome::Child(status),
                result = &mut completion => Outcome::Completion(result),
            }
        };
        match outcome {
            Outcome::Completion(result) => self.reconcile_completion(result).await,
            Outcome::Child(status) => Err(self.child_death_error(status).await),
        }
    }
}

impl Drop for LlamaServer {
    fn drop(&mut self) {
        let _ = self.child.start_kill();
    }
}

enum StartupWait {
    EarlyExit(ExitStatus),
    Timeout,
    Io(anyhow::Error),
}

#[derive(Clone)]
struct SharedTail(Arc<Mutex<VecDeque<u8>>>);

impl SharedTail {
    fn new() -> SharedTail {
        SharedTail(Arc::new(Mutex::new(VecDeque::with_capacity(TAIL_CAPACITY))))
    }

    fn append(&self, bytes: &[u8]) {
        let mut tail = self
            .0
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if bytes.len() >= TAIL_CAPACITY {
            tail.clear();
            tail.extend(&bytes[bytes.len() - TAIL_CAPACITY..]);
            return;
        }
        let overflow = tail
            .len()
            .saturating_add(bytes.len())
            .saturating_sub(TAIL_CAPACITY);
        tail.drain(..overflow);
        tail.extend(bytes);
    }

    fn read(&self) -> String {
        let tail = self
            .0
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let bytes: Vec<_> = tail.iter().copied().collect();
        String::from_utf8_lossy(&bytes).into_owned()
    }
}

async fn drain_pipe<R: AsyncRead + Unpin>(mut pipe: R, tail: SharedTail) -> std::io::Result<()> {
    let mut buffer = [0u8; 8192];
    loop {
        let read = pipe.read(&mut buffer).await?;
        if read == 0 {
            return Ok(());
        }
        tail.append(&buffer[..read]);
    }
}

async fn pick_loopback_port() -> Result<u16> {
    let listener = tokio::net::TcpListener::bind((Ipv4Addr::LOCALHOST, 0))
        .await
        .context("reserve a loopback port for llama-server")?;
    let port = listener.local_addr()?.port();
    drop(listener);
    Ok(port)
}

/// Build the `/completion` request. Pure, so the sampling-preservation rules
/// are unit-testable: the server's non-neutral defaults mean every knob is
/// pinned explicitly — `top_p: 1.0` when yatima samples the full
/// distribution, `top_k`/`min_p` disabled unless the backend config asks —
/// and `prefill_chunk` is Candle-specific and deliberately not represented.
fn request_body(
    config: &LlamaServerConfig,
    prompt: &str,
    opts: &GenOpts,
    stops: &[String],
) -> serde_json::Value {
    let (temperature, top_p, seed) = match opts.sampling {
        Sampling::Greedy => (0.0, 1.0, 0),
        Sampling::Sample {
            temperature,
            top_p,
            seed,
        } => (temperature, top_p.unwrap_or(1.0), seed),
    };
    serde_json::json!({
        "prompt": prompt,
        "stream": true,
        "cache_prompt": true,
        "n_predict": opts.max_tokens,
        "stop": stops,
        "temperature": temperature,
        "top_p": top_p,
        "seed": seed,
        "top_k": config.top_k,
        "min_p": 0.0,
        "repeat_penalty": opts.repeat_penalty,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    const WITHIN: Duration = Duration::from_secs(5);

    async fn within<F: std::future::Future>(what: &str, fut: F) -> F::Output {
        tokio::time::timeout(WITHIN, fut)
            .await
            .unwrap_or_else(|_| panic!("timed out after {WITHIN:?}: {what}"))
    }

    fn completer(base_url: &str) -> LlamaServerCompleter {
        LlamaServerCompleter::new(LlamaServerConfig::new(base_url).unwrap()).unwrap()
    }

    fn completer_with_timeout(base_url: &str, timeout: Duration) -> LlamaServerCompleter {
        let config = LlamaServerConfig::new(base_url)
            .unwrap()
            .with_control_timeout(timeout);
        LlamaServerCompleter::new(config).unwrap()
    }

    async fn serve(body: &str) -> MockServer {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/completion"))
            .respond_with(ResponseTemplate::new(200).set_body_raw(body, "text/event-stream"))
            .mount(&server)
            .await;
        server
    }

    #[test]
    fn attached_endpoints_are_loopback_origins() {
        // upholds: LSRV-2 — validation is the configuration constructor, and
        // the accepted set is exactly HTTP loopback origins.
        for accepted in [
            "http://127.0.0.1",
            "http://127.42.7.9:8080/",
            "http://[::1]:9090",
            "http://localhost:8080/",
            "HTTP://LOCALHOST:8080",
        ] {
            assert!(LlamaServerConfig::new(accepted).is_ok(), "{accepted}");
        }
        for rejected in [
            "not a URL",
            "https://127.0.0.1",
            "http://user@127.0.0.1",
            "http://@127.0.0.1/",
            "http:@127.0.0.1",
            "http:/@127.0.0.1",
            "http:\t//@127.0.0.1",
            "http://127.0.0.1/path",
            "http://127.0.0.1/?query=yes",
            "http://127.0.0.1/#fragment",
            "http://localhost.evil.com",
            "http://localhost.",
            "http://0.0.0.0",
            "http://192.168.1.2",
            "http://[::ffff:127.0.0.1]",
        ] {
            assert!(LlamaServerConfig::new(rejected).is_err(), "{rejected}");
        }
    }

    #[tokio::test]
    async fn localhost_is_normalized_before_a_request() {
        // upholds: LSRV-2 — this observes the URL used at the transport, not
        // merely constructor acceptance.
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/props"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "build_info": "b10520-test",
                "model_path": "/models/test.gguf",
                "total_slots": 1,
                "default_generation_settings": { "n_ctx": 4096 }
            })))
            .mount(&server)
            .await;
        let localhost = server.uri().replace("127.0.0.1", "localhost");
        let config = LlamaServerConfig::new(&localhost).unwrap();
        assert_eq!(
            config.base_url(),
            server.uri() + "/",
            "localhost is removed at the validated-value boundary"
        );
        let c = LlamaServerCompleter::new(config).unwrap();
        c.introspect().await.unwrap();
        let requests = server.received_requests().await.unwrap();
        assert_eq!(requests.len(), 1);
        let host = requests[0]
            .headers
            .get("host")
            .expect("HTTP request carries Host")
            .to_str()
            .unwrap();
        assert!(host.starts_with("127.0.0.1:"), "observed Host: {host}");
    }

    #[tokio::test]
    async fn redirects_are_not_followed() {
        // upholds: LSRV-2 — a validated loopback request cannot be redirected
        // to a second origin by the attached process.
        let sink = MockServer::start().await;
        let origin = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/props"))
            .respond_with(
                ResponseTemplate::new(302)
                    .insert_header("location", format!("{}/escaped", sink.uri())),
            )
            .mount(&origin)
            .await;

        let error = completer(&origin.uri()).introspect().await.unwrap_err();
        assert!(error.to_string().contains("302"), "{error:#}");
        assert!(
            sink.received_requests().await.unwrap().is_empty(),
            "redirect target must receive no request"
        );
    }

    #[tokio::test]
    async fn introspection_parses_only_the_declared_diagnostics() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/props"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "build_info": "b10520-cd644c395",
                "model_path": "/models/qwen.gguf",
                "total_slots": 1,
                "default_generation_settings": { "n_ctx": 32768 },
                "untrusted_extra": "ignored"
            })))
            .mount(&server)
            .await;
        let props = completer(&server.uri()).introspect().await.unwrap();
        assert_eq!(
            props,
            ServerProps {
                build: "b10520-cd644c395".into(),
                model_self_report: "/models/qwen.gguf".into(),
                n_ctx: 32768,
                total_slots: 1,
            }
        );

        let malformed = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/props"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "model_path": "/models/qwen.gguf"
            })))
            .mount(&malformed)
            .await;
        let err = completer(&malformed.uri()).introspect().await.unwrap_err();
        assert!(err.to_string().contains("build_info"), "{err}");
    }

    async fn stalled_response(status: u16) -> std::net::SocketAddr {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.unwrap();
            let mut request = [0u8; 4096];
            let _ = socket.read(&mut request).await;
            let reason = if status == 200 {
                "OK"
            } else {
                "Service Unavailable"
            };
            let response =
                format!("HTTP/1.1 {status} {reason}\r\ncontent-length: 100\r\n\r\npartial");
            socket.write_all(response.as_bytes()).await.unwrap();
            socket.flush().await.unwrap();
            tokio::time::sleep(Duration::from_secs(60)).await;
        });
        addr
    }

    #[tokio::test]
    async fn stalled_props_body_obeys_the_control_timeout() {
        let addr = stalled_response(200).await;
        let c = completer_with_timeout(&format!("http://{addr}"), Duration::from_millis(100));
        let err = within("stalled /props body", c.introspect())
            .await
            .unwrap_err();
        assert!(err.to_string().contains("timed out"), "{err}");
    }

    #[tokio::test]
    async fn stalled_completion_error_body_obeys_the_control_timeout() {
        let addr = stalled_response(503).await;
        let mut c = completer_with_timeout(&format!("http://{addr}"), Duration::from_millis(100));
        let err = within(
            "stalled completion error body",
            c.complete("p", &GenOpts::default(), &[]),
        )
        .await
        .unwrap_err();
        assert!(err.to_string().contains("timed out"), "{err}");
    }

    #[tokio::test]
    async fn cancellation_is_observed_while_send_is_pending() {
        // upholds: LSRV-4 / CANCEL-1 — cancellation races the initial HTTP
        // response, before any stream exists.
        use tokio::io::AsyncReadExt;
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.unwrap();
            let mut request = [0u8; 4096];
            let _ = socket.read(&mut request).await;
            tokio::time::sleep(Duration::from_secs(60)).await;
        });
        let mut c = completer(&format!("http://{addr}"));
        let cancel = Cancel::new();
        let trigger = cancel.clone();
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(50)).await;
            trigger.cancel();
        });
        let done = within(
            "cancel pending send",
            c.complete_streaming("p", &GenOpts::default(), &[], &cancel, &mut |_| {}),
        )
        .await
        .unwrap();
        assert!(done.text.is_empty());
        assert_eq!(done.stop, StopReason::Stopped);
    }

    #[test]
    fn request_pins_the_full_sampling_recipe() {
        // upholds: SAM-3 — every llama-server sampler knob is selected by
        // Yatima or explicitly neutralized; server defaults never participate.
        // The server's defaults are top_k 40 / top_p 0.95 / min_p 0.05 —
        // omitting a knob silently changes yatima's policy, so every request
        // pins all of them (sampling honesty).
        let config = LlamaServerConfig::new("http://127.0.0.1").unwrap();
        let greedy = request_body(&config, "p", &GenOpts::default(), &[]);
        assert_eq!(greedy["temperature"], 0.0);
        assert_eq!(greedy["top_p"], 1.0);
        assert_eq!(greedy["top_k"], 0);
        assert_eq!(greedy["min_p"], 0.0);
        assert_eq!(greedy["cache_prompt"], true);

        let full = GenOpts {
            sampling: Sampling::nucleus(1.0, None, 7),
            ..GenOpts::default()
        };
        let body = request_body(&config, "p", &full, &[]);
        // top_p None means the full distribution — which must be *pinned* as
        // 1.0, not omitted (the server would default to 0.95).
        assert_eq!(body["top_p"], 1.0);
        assert_eq!(body["seed"], 7);
    }

    #[test]
    fn complete_future_is_send() {
        // upholds: CMP-1 — per-impl Send inference: this backend's
        // non-streaming completion future is Send (the remote case the trait
        // design anticipated). The future is genuinely constructed (never
        // polled; nothing connects), so the assertion tracks the real type.
        // complete_streaming is deliberately not asserted: its &mut dyn FnMut
        // parameter carries no Send bound.
        fn requires_send<F: Send>(_: &F) {}
        let mut c = completer("http://127.0.0.1:9");
        let opts = GenOpts::default();
        let fut = c.complete("p", &opts, &[]);
        requires_send(&fut);
        drop(fut);
    }

    #[tokio::test]
    async fn cancellation_is_observed_during_a_rapid_stream() {
        // upholds: LSRV-4 (active stream) — a server emitting continuous
        // chunks must not starve cancellation: the flag is polled per
        // iteration and per buffered event, so a cancel raised mid-stream
        // (here from on_token itself) exits promptly with only the
        // pre-cancellation text, instead of draining the remaining stream.
        let mut body = String::new();
        for _ in 0..200 {
            body.push_str("data: {\"content\":\"x\",\"stop\":false}\n\n");
        }
        body.push_str("data: {\"content\":\"\",\"stop\":true,\"stop_type\":\"eos\"}\n\n");
        let server = serve(&body).await;
        let mut c = completer(&server.uri());
        let cancel = Cancel::new();
        let mut seen = 0usize;
        let done = within(
            "cancel mid-stream",
            c.complete_streaming("p", &GenOpts::default(), &[], &cancel, &mut |_| {
                seen += 1;
                if seen == 5 {
                    cancel.cancel();
                }
            }),
        )
        .await
        .unwrap();
        assert_eq!(done.stop, StopReason::Stopped);
        assert_eq!(seen, 5, "no fragment is emitted after cancellation");
        assert_eq!(done.text, "x".repeat(5), "only pre-cancellation text");
    }

    #[tokio::test]
    async fn cancellation_is_observed_while_the_stream_is_silent() {
        // upholds: LSRV-4 (silent stream) — after the response is
        // established, a server that goes quiet (prefill, stall) must not
        // block cancellation: the pending read is raced directly against the
        // awaitable cancellation state. The stub accepts one request, sends
        // headers plus one chunked SSE event, then holds the connection open.
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            let (mut sock, _) = listener.accept().await.unwrap();
            let mut buf = [0u8; 4096];
            let _ = sock.read(&mut buf).await;
            let event = "data: {\"content\":\"x\",\"stop\":false}\n\n";
            let head = format!(
                "HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\n\
                 transfer-encoding: chunked\r\n\r\n{:x}\r\n{event}\r\n",
                event.len()
            );
            let _ = sock.write_all(head.as_bytes()).await;
            let _ = sock.flush().await;
            // Hold the stream open, silently, far past the test's bounds.
            tokio::time::sleep(Duration::from_secs(60)).await;
        });

        let mut c = completer(&format!("http://{addr}"));
        let cancel = Cancel::new();
        let flipper = cancel.clone();
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(300)).await;
            flipper.cancel();
        });
        let done = within(
            "cancel during silence",
            c.complete_streaming("p", &GenOpts::default(), &[], &cancel, &mut |_| {}),
        )
        .await
        .unwrap();
        assert_eq!(done.stop, StopReason::Stopped);
        assert_eq!(done.text, "x", "the pre-silence fragment is preserved");
    }

    #[tokio::test]
    async fn word_stop_without_a_word_is_a_protocol_error() {
        // upholds: LSRV-3 — a word stop that names no marker (field missing
        // or empty) would hand a codec a truncated block; both malformed
        // shapes are protocol errors, never a quiet Stopped.
        for final_event in [
            "data: {\"content\":\"\",\"stop\":true,\"stop_type\":\"word\"}\n\n",
            "data: {\"content\":\"\",\"stop\":true,\"stop_type\":\"word\",\
             \"stopping_word\":\"\"}\n\n",
        ] {
            let server = serve(final_event).await;
            let mut c = completer(&server.uri());
            let err = within(
                "wordless word stop",
                c.complete("p", &GenOpts::default(), &["five".to_string()]),
            )
            .await
            .unwrap_err();
            assert!(err.to_string().contains("without a stopping_word"), "{err}");
        }
    }

    #[tokio::test]
    async fn foreign_stopping_word_is_a_protocol_error() {
        // upholds: LSRV-3 — a stopping word the caller never supplied has no
        // place in the contract; re-appending it would corrupt the text.
        let server = serve(
            "data: {\"content\":\"\",\"stop\":true,\"stop_type\":\"word\",\
             \"stopping_word\":\"zebra\"}\n\n",
        )
        .await;
        let mut c = completer(&server.uri());
        let err = within(
            "foreign stopping word",
            c.complete("p", &GenOpts::default(), &["five".to_string()]),
        )
        .await
        .unwrap_err();
        assert!(err.to_string().contains("not among the requested"), "{err}");
    }

    #[tokio::test]
    async fn word_stop_is_reappended_and_streamed() {
        // upholds: LSRV-3 — text includes the matched caller-supplied stop
        // string even though llama-server excludes it (stage-0 probe P3), and
        // the marker also reaches on_token.
        let server = serve(concat!(
            "data: {\"content\":\"one two three four \",\"stop\":false}\n\n",
            "data: {\"content\":\"\",\"stop\":true,\"stop_type\":\"word\",",
            "\"stopping_word\":\"five\"}\n\n",
        ))
        .await;
        let mut c = completer(&server.uri());
        let mut streamed = String::new();
        let done = within(
            "word-stop completion",
            c.complete_streaming(
                "p",
                &GenOpts::default(),
                &["five".to_string()],
                &Cancel::new(),
                &mut |s| streamed.push_str(s),
            ),
        )
        .await
        .unwrap();
        assert_eq!(done.text, "one two three four five");
        assert_eq!(done.stop, StopReason::Stopped);
        assert_eq!(streamed, done.text, "the marker reaches on_token too");
    }

    #[tokio::test]
    async fn split_final_word_stop_is_reassembled_and_reappended() {
        // upholds: LSRV-3 — the final event and its stopping word may cross
        // transport chunks; framing completes before stop fidelity is applied.
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.unwrap();
            let mut request = [0u8; 4096];
            let _ = socket.read(&mut request).await;
            socket
                .write_all(
                    b"HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\n\
                      transfer-encoding: chunked\r\n\r\n",
                )
                .await
                .unwrap();
            for fragment in [
                "data: {\"content\":\"one two three four \",\"stop\":false}\n\n",
                "data: {\"content\":\"\",\"stop\":true,\"stop_type\":\"word\",\"stopping_",
                "word\":\"fi",
                "ve\"}\n\n",
            ] {
                socket
                    .write_all(format!("{:x}\r\n{fragment}\r\n", fragment.len()).as_bytes())
                    .await
                    .unwrap();
                socket.flush().await.unwrap();
                tokio::task::yield_now().await;
            }
            socket.write_all(b"0\r\n\r\n").await.unwrap();
        });

        let mut c = completer(&format!("http://{addr}"));
        let mut streamed = String::new();
        let done = within(
            "split final word stop",
            c.complete_streaming(
                "p",
                &GenOpts::default(),
                &["five".to_string()],
                &Cancel::new(),
                &mut |text| streamed.push_str(text),
            ),
        )
        .await
        .unwrap();
        assert_eq!(done.text, "one two three four five");
        assert_eq!(streamed, done.text);
        assert_eq!(done.stop, StopReason::Stopped);
    }

    #[tokio::test]
    async fn eos_and_limit_map_to_their_reasons() {
        let server = serve(concat!(
            "data: {\"content\":\"hi\",\"stop\":false}\n\n",
            "data: {\"content\":\"\",\"stop\":true,\"stop_type\":\"eos\",",
            "\"stopping_word\":\"\"}\n\n",
        ))
        .await;
        let mut c = completer(&server.uri());
        let done = within("eos completion", c.complete("p", &GenOpts::default(), &[]))
            .await
            .unwrap();
        assert_eq!((done.text.as_str(), done.stop), ("hi", StopReason::Eos));

        let server =
            serve("data: {\"content\":\"x\",\"stop\":true,\"stop_type\":\"limit\"}\n\n").await;
        let mut c = completer(&server.uri());
        let done = within(
            "limit completion",
            c.complete("p", &GenOpts::default(), &[]),
        )
        .await
        .unwrap();
        assert_eq!(done.stop, StopReason::MaxTokens);
    }

    #[tokio::test]
    async fn missing_final_event_is_a_protocol_error() {
        // A stream that just ends is a protocol error, never a successful
        // completion with an invented stop reason.
        let server = serve("data: {\"content\":\"partial\",\"stop\":false}\n\n").await;
        let mut c = completer(&server.uri());
        let err = within(
            "truncated stream",
            c.complete("p", &GenOpts::default(), &[]),
        )
        .await
        .unwrap_err();
        assert!(err.to_string().contains("without a final stop event"));
    }

    #[tokio::test]
    async fn unknown_stop_type_is_a_protocol_error() {
        let server =
            serve("data: {\"content\":\"\",\"stop\":true,\"stop_type\":\"novel\"}\n\n").await;
        let mut c = completer(&server.uri());
        let err = within(
            "unknown stop_type",
            c.complete("p", &GenOpts::default(), &[]),
        )
        .await
        .unwrap_err();
        assert!(err.to_string().contains("unexpected stop_type"));
    }

    #[tokio::test]
    async fn http_error_carries_status_and_body() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/completion"))
            .respond_with(ResponseTemplate::new(503).set_body_string("loading model"))
            .mount(&server)
            .await;
        let mut c = completer(&server.uri());
        let err = within("503 response", c.complete("p", &GenOpts::default(), &[]))
            .await
            .unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("503") && msg.contains("loading model"),
            "{msg}"
        );
    }
}
