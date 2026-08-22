//! An attached llama-server as a [`Completer`] — stage 1 of
//! plans/llama-server.plan.md.
//!
//! This is the pure HTTP transport adapter: raw rendered prompt in, streamed
//! text out, against an **already-running** `llama-server` (started by hand;
//! child supervision is stage 2's). It discharges the `Completer` effect by
//! awaiting I/O — no blocking island, and futures whose `Send` is inferred
//! per instantiation (CMP-1): [`complete`](Completer::complete) is `Send`,
//! while [`complete_streaming`](Completer::complete_streaming) is not (its
//! `&mut dyn FnMut` parameter carries no `Send` bound).
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
use crate::{Cancel, Completer, Completion, GenOpts, Sampling, StopReason};
use anyhow::{bail, Context, Result};
use std::time::Duration;

/// How often the pending stream read is interrupted to poll cancellation —
/// prefill produces no chunks for a long time, so cancel must not wait for
/// one (the read is raced against this tick).
const CANCEL_POLL: Duration = Duration::from_millis(100);

/// Configuration for an attached llama-server backend.
pub struct LlamaServerConfig {
    /// Base URL of the running server, e.g. `http://127.0.0.1:8080`.
    pub base_url: String,
    /// Server-side top-k filter; `0` disables it (the neutral default — the
    /// server's own default is 40). Non-zero only when a model's recipe
    /// deliberately asks for it (Muse Glimmer: 64, stage 3).
    pub top_k: u32,
}

impl LlamaServerConfig {
    pub fn new(base_url: impl Into<String>) -> LlamaServerConfig {
        LlamaServerConfig {
            base_url: base_url.into(),
            top_k: 0,
        }
    }
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
            .build()
            .context("build llama-server HTTP client")?;
        Ok(LlamaServerCompleter { client, config })
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
        let url = format!("{}/completion", self.config.base_url.trim_end_matches('/'));
        let body = request_body(&self.config, prompt, opts, stops);
        let mut resp = self
            .client
            .post(&url)
            .json(&body)
            .send()
            .await
            .with_context(|| format!("POST {url}"))?;
        if !resp.status().is_success() {
            let status = resp.status();
            let detail = resp.text().await.unwrap_or_default();
            bail!("llama-server at {url} returned {status}: {detail}");
        }

        let mut framer = SseFramer::new();
        let mut text = String::new();
        loop {
            // LSRV-4: cancellation is observed per iteration — a server
            // emitting chunks faster than the tick would otherwise starve the
            // timeout branch and never see the flag.
            if cancel.is_cancelled() {
                return Ok(Completion {
                    text,
                    stop: StopReason::Stopped,
                });
            }
            // Race the pending read against a bounded cancellation tick (the
            // silent-stream half of LSRV-4); the server aborts generation
            // when the response is dropped.
            let chunk = loop {
                match tokio::time::timeout(CANCEL_POLL, resp.chunk()).await {
                    Ok(read) => break read.with_context(|| format!("read {url} stream"))?,
                    Err(_elapsed) => {
                        if cancel.is_cancelled() {
                            return Ok(Completion {
                                text,
                                stop: StopReason::Stopped,
                            });
                        }
                    }
                }
            };
            let Some(chunk) = chunk else {
                bail!("llama-server stream ended without a final stop event (protocol error)");
            };
            for payload in framer.push(&chunk)? {
                // LSRV-4: one network chunk can carry many buffered events;
                // honor a cancel raised mid-batch (e.g. from `on_token`
                // itself) without draining the rest.
                if cancel.is_cancelled() {
                    return Ok(Completion {
                        text,
                        stop: StopReason::Stopped,
                    });
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
        LlamaServerCompleter::new(LlamaServerConfig::new(base_url)).unwrap()
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
    fn request_pins_the_full_sampling_recipe() {
        // upholds: SAM-3 — every llama-server sampler knob is selected by
        // Yatima or explicitly neutralized; server defaults never participate.
        // The server's defaults are top_k 40 / top_p 0.95 / min_p 0.05 —
        // omitting a knob silently changes yatima's policy, so every request
        // pins all of them (sampling honesty).
        let config = LlamaServerConfig::new("http://x");
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
        // block cancellation: the pending read is raced against a bounded
        // tick. The stub accepts one request, sends headers plus one chunked
        // SSE event, then holds the connection open silently.
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
