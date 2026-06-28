//! `yatima-tui` — an interactive terminal UI over `yatima-lib`.
//!
//! A separate crate so its UI deps (`ratatui`, `crossterm`) never touch the lean
//! `yatima` CLI. It reuses only `yatima-lib`'s public API; LAYER-1 holds (the TUI
//! is an edge, like the CLI).
//!
//! # Architecture keystone
//!
//! The engine thread owns the [`Engine`](yatima_lib::Engine) **and** the
//! [`ChatSession`](yatima_lib::ChatSession) — the one authoritative prompt
//! history — and the UI owns only a render model derived from engine events.
//! Local decode is `!Send` and runs on the blocking island (CMP-1 / RT-2), so it
//! cannot live in a `tokio::spawn`; a dedicated OS thread runs it and, being a
//! plain thread (not a runtime worker), calls the public sync
//! `ChatSession::turn_streaming` directly. See [`engine_actor`].
//!
//! # Invariant registry (TUI-N), each protected by a test citing its id
//!
//! - **TUI-1 cursor-bounds** — the scrolled viewport top is always within
//!   `[0, total - viewport]` ([`app::scroll_y`]).
//! - **TUI-2 pure-render** — [`render::ui`] mutates nothing; state changes flow
//!   only through a key [`Intent`](app::Intent) or an
//!   [`EngineEvent`](engine_actor::EngineEvent).
//! - **TUI-3 single-append** — the transcript grows only through
//!   [`App::push_entry`](app::App::push_entry); fragments mutate the last entry.
//! - **TUI-4 ui-liveness** — generation runs on the engine thread, so the event
//!   loop services input while a turn is in flight (the keystone).
//! - **TUI-5 reasoning-foldable** — a completed turn's reasoning collapses to a
//!   one-line summary (Ctrl+R toggles); the in-flight turn always streams it
//!   live. The reasoning is never lost (REASON-1 carried into the UI).
//! - **TUI-6 prompt-cancel** — Esc flips a shared [`Cancel`](yatima_lib::Cancel)
//!   the decode loop polls per token; the turn stops at the next token boundary
//!   as a clean `StopReason::Stopped`, partial output preserved, and the
//!   indicator shows "cancelling…" until `Done`.
//! - **TUI-7 single-in-flight** — at most one turn at a time; a submit while one
//!   is active is a no-op.

pub mod app;
pub mod engine_actor;
pub mod render;
