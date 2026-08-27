//! The frontend wire protocol: the event and request planes every yatima
//! frontend speaks to the engine host.
//!
//! The TUI, the GUI, and the coming `yatima-serve` WASM client are three
//! views over one engine-facing host ([`yatima-host`](../yatima_host)). They
//! differ only in how they draw a [`HostEvent`] and where a [`HostRequest`]
//! comes from — never in what the planes carry. That common carrier is these
//! types, and because the WASM client deserializes them in a browser, this
//! crate is serde-only and depends on nothing that cannot reach wasm32
//! (yatima-lib, which drags candle, is deliberately NOT a dependency).
//!
//! [`Channel`] and [`StopKind`] mirror the yatima-lib enums of the same
//! shape; yatima-host owns the conversions (this crate cannot, without
//! depending on the lib). Tool activity crosses the wire as a semantic
//! [`ToolNoteKind`] plus a bare payload ([`HostEvent::ToolNote`]): the
//! clip/verbatim rules are host policy, identical across frontends, while
//! the marker vocabulary is each view's own — a terminal draws `✓`/`✗`,
//! egui (whose built-in fonts lack those glyphs) spells `ok`/`failed:`.
//! The wire carries meaning, never typography (HOST-4, registered in
//! yatima-host).
//!
//! # Invariant & law registry
//!
//! - **PROTO-2** every [`HostEvent`] and [`HostRequest`] variant round-trips
//!   through serde losslessly. The enums are externally tagged (serde's
//!   default); no variant uses `#[serde(untagged)]`, which would silently
//!   make wire evolution ambiguous (and does not compose with
//!   `deny_unknown_fields`). `#[non_exhaustive]` governs Rust matching only:
//!   downstream folds must carry a wildcard arm, so adding a variant keeps
//!   consumer *code* compiling — but an older deserializer still **rejects
//!   an unknown variant as an error**. The real compatibility rule is
//!   deployment order: consumers ship at or ahead of producers (here, one
//!   repository built in lockstep). Cited by `host_events_round_trip` /
//!   `host_requests_round_trip`.
//! - **WASM-1** this crate compiles for `wasm32-unknown-unknown` — the
//!   "serde-only, WASM-clean" paragraph above is enforced, not aspirational.
//!   Checked by `scripts/check-wasm.sh` locally and by the CI "wasm check"
//!   step on every push; a dependency added to this crate that cannot reach
//!   wasm32 fails the build, not the browser.

use serde::{Deserialize, Serialize};

/// Which stream a completion fragment belongs to — the wire mirror of
/// `yatima_lib::Channel` (yatima-host converts between them).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Channel {
    /// The model's chain-of-thought (folded away unless the frontend reveals it).
    Reasoning,
    /// The surfaced answer.
    Answer,
}

/// Why a turn stopped — the wire mirror of `yatima_lib::StopReason` (yatima-host
/// converts between them). The agent's tool-step-budget exhaustion is reported
/// as [`StopKind::MaxTokens`], as the TUI has always surfaced it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum StopKind {
    /// An end-of-sequence token was sampled (a clean finish).
    Eos,
    /// The token budget (or, for an agent, the tool-step budget) was reached.
    MaxTokens,
    /// The turn was cancelled (the user pressed Esc / the stop button).
    Stopped,
    /// The degeneration guard stopped a short repeating cycle.
    Repetition,
}

/// What a [`HostEvent::ToolNote`] line reports — the semantic half of tool
/// activity. The rendered marker (a terminal's `✓`/`✗`, egui's `ok`/
/// `failed:`) and the fold's indentation are view policy; the wire carries
/// meaning, never typography (HOST-4).
///
/// `#[non_exhaustive]`: consumers must carry a wildcard arm (rendering the
/// payload unmarked is a fine fallback), so a new kind keeps their *code*
/// compiling — Rust source compatibility, not wire compatibility: an older
/// deserializer still rejects an unknown kind (PROTO-2's deployment-order
/// rule).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum ToolNoteKind {
    /// The model called a tool (payload: the name and clipped arguments).
    Call,
    /// A mid-run progress message from the tool.
    Progress,
    /// The tool succeeded (payload: the clipped content or a size summary).
    Success,
    /// The tool failed (payload: the clipped error).
    Failure,
    /// A host warning (e.g. the tool-step budget was exhausted).
    Warning,
}

/// Which pre-`Ready` startup phase the host is in — the payload of
/// [`HostEvent::Startup`]. 5a defines the vocabulary; the host emits phase
/// transitions from stage 5b. Startup is loading state, not transcript: a
/// view shows the current phase (animating elapsed time on its own clock —
/// the wire carries no ticker) and drops it at `Ready` or `Fatal`.
///
/// A closed sum like [`Channel`]/[`StopKind`]: adding a phase is a reviewed
/// protocol change that must update every exhaustive consumer and fixture.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum StartupPhase {
    /// Choosing and acquiring the exact model artifact (may fetch weights on
    /// a first run).
    ResolvingModel,
    /// Hashing the artifact against a profile's pinned digest.
    VerifyingModel,
    /// Loading the engine, or launching and gating the managed server.
    StartingBackend,
}

/// How the running model's identity is authenticated — the wire form of the
/// verified/unverified distinction (LSRV-5 is registered in yatima-lib; this
/// mirror carries its outcome, never a claim of its own). No self-report can
/// promote [`Unverified`](ModelIdentity::Unverified) to verified.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ModelIdentity {
    /// The artifact's SHA-256 was verified against a pinned digest before
    /// launch. The lowercase hex digest string is preserved byte-for-byte.
    VerifiedSha256(String),
    /// No authenticated identity evidence exists: nothing proved which
    /// artifact is running. A backend may still self-report a name (a Candle
    /// directory load, an attached server's `/props`), but a self-report is
    /// a claim, not evidence, and cannot promote this variant.
    Unverified,
}

/// What is running, reported once the model is ready. Every field is a
/// pre-formatted string so a frontend is a pure view — it renders these, it
/// does not compute them. A given frontend reads the subset it displays (the
/// TUI shows `backend`; the GUI's status rail shows `arch`/`device`/`sampling`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelInfo {
    /// The model's display label (a profile name or the model directory).
    pub label: String,
    /// The detected architecture, formatted (e.g. `Qwen2`).
    pub arch: String,
    /// The engine's backend/dtype label (e.g. `metal/BF16`).
    pub backend: String,
    /// Where decode runs, coarsely: `cpu` or `gpu`.
    pub device: String,
    /// The resolved chat format, formatted (e.g. `Qwen`).
    pub format: String,
    /// The sampling summary (e.g. `greedy` or `temp 0.70 · top-p 0.95 · seed 0`).
    pub sampling: String,
    /// The per-turn token budget.
    pub max_tokens: usize,
    /// The model's context window in tokens (the meter denominator), if declared.
    pub context_length: Option<usize>,
    /// How the model's identity is authenticated (verified digest or not).
    pub identity: ModelIdentity,
}

/// Event plane: host → frontend. A frontend's only source of transcript truth;
/// it renders each event and never reaches past this plane to the engine.
///
/// `#[non_exhaustive]`: consumers must carry a wildcard arm, so a new event
/// (a structured tool record, a flight-recorder tick) keeps their *code*
/// compiling — Rust source compatibility, not wire compatibility (PROTO-2's
/// deployment-order rule).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum HostEvent {
    /// The host is starting up: emitted once on each phase transition before
    /// `Ready`/`Fatal`. Loading state, never transcript content.
    Startup { phase: StartupPhase },
    /// The model loaded and is ready to serve turns (carries what's running).
    Ready(ModelInfo),
    /// A turn began.
    Started { turn_id: u64 },
    /// A classified slice of the live completion (reasoning vs answer).
    Fragment {
        turn_id: u64,
        channel: Channel,
        text: String,
    },
    /// Retract the last `chars` characters streamed on the answer channel: the
    /// step they belonged to turned out to be a tool call, so they were
    /// narration and replay on the reasoning channel (AGENT-4).
    RetractAnswer { turn_id: u64, chars: usize },
    /// A line of tool activity for the reasoning fold. `kind` carries the
    /// semantics; `text` is the bare payload, already clipped by host policy.
    /// The marker it renders under — glyph, word, indentation — is the view's
    /// (HOST-4).
    ToolNote {
        turn_id: u64,
        kind: ToolNoteKind,
        text: String,
    },
    /// An image artifact the turn produced (a plot render, a fetched image):
    /// the file's bytes, already read by the host, and its filename. A
    /// frontend textures/ships them; the terminal frontend may instead open
    /// the file named in the accompanying `ToolNote`.
    Image {
        turn_id: u64,
        bytes: Vec<u8>,
        name: String,
    },
    /// The granted-origin set after a grant/revoke/list, with a line for the
    /// transcript (CAP-3 authority is visible history).
    Grants {
        origins: Vec<String>,
        message: String,
    },
    /// An app-plane message (help, about, a chat-only refusal) — not model text.
    Note(String),
    /// The most recent rendered prompt's token count (the meter numerator).
    Context { prompt_tokens: usize },
    /// The turn finished.
    Done { turn_id: u64, stop: StopKind },
    /// The turn failed.
    Error { turn_id: u64, message: String },
    /// The model could not be loaded; the session never starts.
    Fatal(String),
}

/// Request plane: frontend → host. Grants ride the same plane as prompts so
/// the host sees them in order (CAP-3: a frontend sends `Grant` only for a
/// user utterance — a typed URL or an explicit `/grant`).
///
/// `#[non_exhaustive]`: the host must carry a wildcard arm, so a new request
/// keeps its *code* compiling — Rust source compatibility, not wire
/// compatibility (PROTO-2's deployment-order rule).
///
/// [`HostRequest::Cancel`] is the wire form of a cancel; a native frontend
/// also has the out-of-band `CancelGate` for the mid-decode path (the engine
/// thread is deaf to this queue while decoding). See yatima-host.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum HostRequest {
    /// Run one turn.
    Submit { turn_id: u64, text: String },
    /// Cancel a turn (honored between turns; the mid-decode path is the gate).
    Cancel { turn_id: u64 },
    /// Clear the conversation back to the system prompt (grants survive — CAP-3).
    Reset,
    /// Grant a web origin for the session.
    Grant { origin: String },
    /// Revoke a previously granted origin.
    Revoke { origin: String },
    /// Report the granted origins (a `Grants` event answers).
    ListGrants,
    /// Stop the host and drop the engine.
    Shutdown,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn model_info() -> ModelInfo {
        ModelInfo {
            label: "qwq".into(),
            arch: "Qwen2".into(),
            backend: "metal/BF16".into(),
            device: "gpu".into(),
            format: "Qwen".into(),
            sampling: "greedy".into(),
            max_tokens: 1024,
            context_length: Some(32768),
            identity: ModelIdentity::Unverified,
        }
    }

    const DIGEST: &str = "4cc57c0f51040a226e5a72cc47b7613f7772950e460a665f7083de89f183f60e";

    fn verified_model_info() -> ModelInfo {
        ModelInfo {
            identity: ModelIdentity::VerifiedSha256(DIGEST.into()),
            ..model_info()
        }
    }

    /// A sample of every [`HostEvent`] variant (kept exhaustive by the match
    /// below, which fails to compile if a variant is added without a sample).
    fn every_event() -> Vec<HostEvent> {
        // Every ToolNoteKind rides the wire (kept exhaustive by this match:
        // a new kind added without a sample here is a compile error).
        let kinds = [
            ToolNoteKind::Call,
            ToolNoteKind::Progress,
            ToolNoteKind::Success,
            ToolNoteKind::Failure,
            ToolNoteKind::Warning,
        ];
        for kind in &kinds {
            match kind {
                ToolNoteKind::Call
                | ToolNoteKind::Progress
                | ToolNoteKind::Success
                | ToolNoteKind::Failure
                | ToolNoteKind::Warning => {}
            }
        }
        // Every StartupPhase rides the wire (exhaustive by this match: a new
        // phase added without a sample here is a compile error).
        let phases = [
            StartupPhase::ResolvingModel,
            StartupPhase::VerifyingModel,
            StartupPhase::StartingBackend,
        ];
        for phase in &phases {
            match phase {
                StartupPhase::ResolvingModel
                | StartupPhase::VerifyingModel
                | StartupPhase::StartingBackend => {}
            }
        }
        let mut all = vec![
            HostEvent::Ready(model_info()),
            // Both ModelIdentity variants ride the wire inside Ready.
            HostEvent::Ready(verified_model_info()),
            HostEvent::Started { turn_id: 1 },
            HostEvent::Fragment {
                turn_id: 1,
                channel: Channel::Answer,
                text: "hi".into(),
            },
            HostEvent::RetractAnswer {
                turn_id: 1,
                chars: 7,
            },
            HostEvent::Image {
                turn_id: 1,
                bytes: vec![0x89, 0x50, 0x4e, 0x47],
                name: "chart.png".into(),
            },
            HostEvent::Grants {
                origins: vec!["https://example.com".into()],
                message: "granted".into(),
            },
            HostEvent::Note("about".into()),
            HostEvent::Context {
                prompt_tokens: 2048,
            },
            HostEvent::Done {
                turn_id: 1,
                stop: StopKind::Eos,
            },
            HostEvent::Error {
                turn_id: 1,
                message: "boom".into(),
            },
            HostEvent::Fatal("no model".into()),
        ];
        all.extend(kinds.iter().map(|&kind| HostEvent::ToolNote {
            turn_id: 1,
            kind,
            text: "plot {…}".into(),
        }));
        all.extend(phases.iter().map(|&phase| HostEvent::Startup { phase }));
        // Exhaustiveness guard: this match must name every variant, so adding
        // one without adding it to `all` above is a compile error.
        for ev in &all {
            match ev {
                HostEvent::Startup { .. }
                | HostEvent::Ready(_)
                | HostEvent::Started { .. }
                | HostEvent::Fragment { .. }
                | HostEvent::RetractAnswer { .. }
                | HostEvent::ToolNote { .. }
                | HostEvent::Image { .. }
                | HostEvent::Grants { .. }
                | HostEvent::Note(_)
                | HostEvent::Context { .. }
                | HostEvent::Done { .. }
                | HostEvent::Error { .. }
                | HostEvent::Fatal(_) => {}
            }
        }
        all
    }

    fn every_request() -> Vec<HostRequest> {
        let all = vec![
            HostRequest::Submit {
                turn_id: 1,
                text: "hello".into(),
            },
            HostRequest::Cancel { turn_id: 1 },
            HostRequest::Reset,
            HostRequest::Grant {
                origin: "https://example.com".into(),
            },
            HostRequest::Revoke {
                origin: "https://example.com".into(),
            },
            HostRequest::ListGrants,
            HostRequest::Shutdown,
        ];
        for req in &all {
            match req {
                HostRequest::Submit { .. }
                | HostRequest::Cancel { .. }
                | HostRequest::Reset
                | HostRequest::Grant { .. }
                | HostRequest::Revoke { .. }
                | HostRequest::ListGrants
                | HostRequest::Shutdown => {}
            }
        }
        all
    }

    #[test]
    fn host_events_round_trip() {
        // upholds: PROTO-2 — every event survives serialize → deserialize
        // byte-for-byte as a value.
        for ev in every_event() {
            let json = serde_json::to_string(&ev).expect("serialize");
            let back: HostEvent = serde_json::from_str(&json).expect("deserialize");
            assert_eq!(ev, back, "round-trip mismatch for {json}");
        }
    }

    #[test]
    fn host_requests_round_trip() {
        // upholds: PROTO-2 — every request survives the round trip.
        for req in every_request() {
            let json = serde_json::to_string(&req).expect("serialize");
            let back: HostRequest = serde_json::from_str(&json).expect("deserialize");
            assert_eq!(req, back, "round-trip mismatch for {json}");
        }
    }

    #[test]
    fn identity_preserves_the_digest_and_startup_is_tagged() {
        // upholds: PROTO-2 — both identity variants and the startup event
        // take the externally tagged shape, and the verified digest string
        // survives the round trip byte-for-byte (LSRV-5's wire form carries
        // the digest; it must not be re-cased, re-encoded, or truncated).
        let verified = ModelIdentity::VerifiedSha256(DIGEST.into());
        let json = serde_json::to_string(&verified).unwrap();
        assert_eq!(json, format!("{{\"VerifiedSha256\":\"{DIGEST}\"}}"));
        let ModelIdentity::VerifiedSha256(digest) = serde_json::from_str(&json).unwrap() else {
            panic!("expected the verified variant back");
        };
        assert_eq!(digest.as_bytes(), DIGEST.as_bytes());

        assert_eq!(
            serde_json::to_string(&ModelIdentity::Unverified).unwrap(),
            "\"Unverified\""
        );
        assert_eq!(
            serde_json::to_string(&HostEvent::Startup {
                phase: StartupPhase::VerifyingModel,
            })
            .unwrap(),
            "{\"Startup\":{\"phase\":\"VerifyingModel\"}}"
        );
    }

    #[test]
    fn enums_are_externally_tagged() {
        // upholds: PROTO-2 — externally tagged (serde's default), NOT untagged:
        // the variant name is the JSON key, which is what keeps the wire
        // unambiguous as variants are added.
        let json = serde_json::to_string(&HostEvent::Started { turn_id: 3 }).unwrap();
        assert_eq!(json, r#"{"Started":{"turn_id":3}}"#);
        let json = serde_json::to_string(&HostRequest::Reset).unwrap();
        assert_eq!(json, r#""Reset""#);
    }
}
