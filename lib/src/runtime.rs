//! The one place the library owns its async runtime and bridges sync ↔ async.
//!
//! yatima is async-first (the agent loop, tool dispatch, and fetch are async),
//! but it still offers thin synchronous APIs for non-async embedders and runs a
//! synchronous inference core. Rather than scatter `Handle::try_current` dances
//! and per-call runtimes across the crate, everything funnels through here:
//!
//! - `block_on` is the *only* sync→async bridge (RT-1).
//! - [`run_blocking`] is the async→sync *compute island*: it runs blocking work
//!   (model inference) without stalling the async executor.

use std::future::Future;
use std::marker::PhantomData;
use std::sync::OnceLock;

use tokio::runtime::{Handle, Runtime, RuntimeFlavor};

/// Private witness type — only this module can mint one, so a `BlockingIsland`
/// cannot be forged elsewhere.
mod island {
    pub struct Token;
}

/// The process-wide multi-thread runtime backing the synchronous shims. Built
/// once, lazily; multi-thread so `block_on`'s `block_in_place` branch and the
/// tools' `spawn`/watch/cancel features work.
fn shared() -> &'static Runtime {
    static RT: OnceLock<Runtime> = OnceLock::new();
    RT.get_or_init(|| {
        tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .expect("build yatima shared tokio runtime")
    })
}

/// Drive an async future to completion from synchronous code — the single
/// sync→async bridge (RT-1). The ambient-runtime policy is explicit:
///
/// - **no runtime**: use the owned [`shared`] runtime.
/// - **multi-thread runtime**: `block_in_place` + the current handle, so we
///   don't start a runtime within a runtime and other tasks keep progressing.
/// - **current-thread runtime**: panic with a directed message — this case is
///   unsupportable (no worker to hand off to; a nested `block_on` would
///   deadlock), and the caller should use the async API instead.
pub(crate) fn block_on<F: Future>(f: F) -> F::Output {
    match Handle::try_current() {
        Err(_) => shared().block_on(f),
        Ok(handle) => match handle.runtime_flavor() {
            RuntimeFlavor::MultiThread => tokio::task::block_in_place(|| handle.block_on(f)),
            _ => panic!(
                "yatima: a sync API was called from within a current-thread Tokio runtime; \
                 call the async API (e.g. run_async / dispatch_async) instead"
            ),
        },
    }
}

/// Run blocking work from async code without stalling the executor. On a
/// multi-thread runtime this is `block_in_place`, which relocates other tasks
/// off the worker for the duration; otherwise (no runtime, or a current-thread
/// runtime that has no worker to free) it just runs the closure. Either way it
/// never panics and never blocks sibling tasks on a live multi-thread executor
/// (RT-1). This is the general-purpose island for ad-hoc blocking (model load,
/// reqwest, …); model decode goes through `run_blocking_island`.
pub fn run_blocking<R>(f: impl FnOnce() -> R) -> R {
    dispatch(f)
}

/// Proof that the holder is executing inside the runtime's blocking island
/// (RT-2). Minted only by `run_blocking_island`; the lifetime is HRTB-scoped
/// to the closure so it cannot escape, and the field is module-private so it
/// cannot be forged. Gating a synchronous decode method on `&BlockingIsland`
/// (see `Engine::complete_on`) makes it a **compile error** to perform model
/// decode on an async worker without first entering the island — the
/// executor-stalling path simply does not type-check.
pub(crate) struct BlockingIsland<'a>(PhantomData<&'a island::Token>);

/// Run blocking model decode inside the island, handing the closure a
/// `BlockingIsland` witness (RT-2). Same executor semantics as
/// [`run_blocking`]; the only difference is the minted proof token, which the
/// decode primitive requires.
pub(crate) fn run_blocking_island<R>(f: impl for<'a> FnOnce(BlockingIsland<'a>) -> R) -> R {
    dispatch(|| f(BlockingIsland(PhantomData)))
}

/// Shared executor-liveness policy for the blocking islands above.
fn dispatch<R>(f: impl FnOnce() -> R) -> R {
    match Handle::try_current() {
        Ok(handle) => match handle.runtime_flavor() {
            RuntimeFlavor::MultiThread => tokio::task::block_in_place(f),
            _ => f(),
        },
        Err(_) => f(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn block_on_from_sync_uses_owned_runtime() {
        // upholds: RT-1 — no ambient runtime → the owned runtime drives it.
        assert_eq!(block_on(async { 1 + 1 }), 2);
    }

    #[test]
    fn run_blocking_from_sync_just_runs() {
        assert_eq!(run_blocking(|| 7), 7);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn block_on_inside_multi_thread_uses_block_in_place() {
        // upholds: RT-1 — ambient multi-thread runtime → block_in_place + handle.
        assert_eq!(block_on(async { 2 + 3 }), 5);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn run_blocking_inside_multi_thread_runs() {
        assert_eq!(run_blocking(|| 6 * 7), 42);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn run_blocking_keeps_the_executor_live() {
        // upholds: RT-1 — blocking compute under run_blocking (block_in_place)
        // must not freeze the executor: a concurrently spawned task still makes
        // progress on another worker while this one blocks. This is the property
        // that un-paints the half-paint (tool watchers stay live during decode).
        use std::sync::atomic::{AtomicBool, Ordering};
        use std::sync::Arc;

        let progressed = Arc::new(AtomicBool::new(false));
        let flag = progressed.clone();
        let handle = tokio::spawn(async move {
            flag.store(true, Ordering::SeqCst);
        });

        // Block this worker the way inference does. On a live multi-thread
        // executor the spawned task runs on the other worker meanwhile.
        run_blocking(|| std::thread::sleep(std::time::Duration::from_millis(150)));

        // Checked *before* awaiting the handle: it must have run *during* the
        // block, not after — i.e. the executor stayed live.
        assert!(
            progressed.load(Ordering::SeqCst),
            "spawned task did not progress while run_blocking blocked the worker"
        );
        handle.await.unwrap();
    }

    #[tokio::test(flavor = "current_thread")]
    #[should_panic(expected = "current-thread")]
    async fn block_on_inside_current_thread_panics_with_direction() {
        // upholds: RT-1 — the one unsupportable case fails loudly with the fix.
        let _ = block_on(async { 1 });
    }
}
