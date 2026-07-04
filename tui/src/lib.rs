//! `yatima-tui` — an interactive terminal UI over `yatima-lib`.
//!
//! A separate crate so its UI deps (`ratatui`, `crossterm`) never touch the lean
//! `yatima` CLI. It is a thin view over [`yatima_host`], which owns the engine
//! and speaks the [`yatima_protocol`](yatima_host) event/request planes; LAYER-1
//! holds (the TUI is an edge, above yatima-host).
//!
//! # Architecture keystone
//!
//! The engine lives in [`yatima_host`], on a dedicated thread that owns the
//! `!Send` engine and session for its whole life (HOST-3). The UI owns only a
//! render model derived from [`HostEvent`](yatima_host::HostEvent)s and never
//! touches a decode path itself (HOST-1). Because generation runs on the host
//! thread, the async event loop services input while a turn is in flight.
//!
//! # Invariant registry (TUI-N), each protected by a test citing its id
//!
//! - **TUI-1 cursor-bounds** — the scrolled viewport top is always within
//!   `[0, total - viewport]` ([`app::scroll_y`]).
//! - **TUI-2 pure-render** — [`render::ui`] mutates nothing; state changes flow
//!   only through a key [`Intent`](app::Intent) or a
//!   [`HostEvent`](yatima_host::HostEvent).
//! - **TUI-3 single-append** — the transcript grows only through
//!   [`App::push_entry`](app::App::push_entry); fragments mutate the last entry.
//! - **TUI-4 ui-liveness** — generation runs on the host thread, so the event
//!   loop services input while a turn is in flight (the keystone).
//! - **TUI-5 reasoning-foldable** — a completed turn's reasoning collapses to a
//!   one-line summary (Ctrl+R toggles); the in-flight turn always streams it
//!   live. The reasoning is never lost (REASON-1 carried into the UI).
//! - **TUI-6 prompt-cancel** — Esc trips the host's
//!   [`CancelGate`](yatima_host::CancelGate) for the in-flight turn; decode
//!   stops at the next token boundary as a clean `StopKind::Stopped`, partial
//!   output preserved, and the indicator shows "cancelling…" until `Done`.
//! - **TUI-7 single-in-flight** — at most one turn at a time; a submit while one
//!   is active is a no-op.

pub mod app;
pub mod render;
