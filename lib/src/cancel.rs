//! Cooperative cancellation for an in-flight generation (the control plane).

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use tokio::sync::Notify;

/// A cooperative cancellation flag for a generation in flight. The decode loop
/// polls it once per token ([`crate::Engine::generate_with`]); flipping it from
/// another thread — a UI key handler, a memory watchdog — stops the turn at the
/// next token boundary with [`crate::StopReason::Stopped`], and the partial
/// output already produced is preserved (it is a clean stop, not an error).
///
/// Cheap to clone (a shared `Arc`): hand one clone to the decode
/// thread and keep another at the requester. A default / fresh handle never
/// fires until [`cancel`](Cancel::cancel) is called, so the non-cancelling
/// callers pass `&Cancel::new()` and pay nothing but an atomic load.
#[derive(Clone, Default)]
pub struct Cancel(Arc<State>);

#[derive(Default)]
struct State {
    cancelled: AtomicBool,
    notify: Notify,
}

impl Cancel {
    /// A fresh, un-cancelled handle.
    pub fn new() -> Cancel {
        Cancel::default()
    }

    /// Request cancellation. Idempotent; safe to call from any thread.
    pub fn cancel(&self) {
        if !self.0.cancelled.swap(true, Ordering::AcqRel) {
            self.0.notify.notify_waiters();
        }
    }

    /// Whether cancellation has been requested.
    pub fn is_cancelled(&self) -> bool {
        self.0.cancelled.load(Ordering::Acquire)
    }

    /// Wait until cancellation is requested.
    ///
    /// This is the asynchronous half of the same monotone state observed by
    /// [`is_cancelled`](Cancel::is_cancelled). Registering the notification
    /// before the second atomic check prevents a cancellation between the
    /// check and the await from being lost (CANCEL-1).
    pub async fn cancelled(&self) {
        loop {
            if self.is_cancelled() {
                return;
            }
            let notified = self.0.notify.notified();
            tokio::pin!(notified);
            notified.as_mut().enable();
            if self.is_cancelled() {
                return;
            }
            notified.await;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cancel_is_shared_and_one_way() {
        // upholds: CANCEL-1 (synchronous half)
        let a = Cancel::new();
        let b = a.clone();
        assert!(!a.is_cancelled() && !b.is_cancelled());
        b.cancel(); // a flip on any clone is visible on every clone
        assert!(a.is_cancelled(), "cancellation is shared across clones");
        b.cancel(); // idempotent
        assert!(a.is_cancelled());
    }

    #[tokio::test]
    async fn cancellation_before_wait_is_not_lost() {
        // upholds: CANCEL-1 — a future waiter observes the stored state.
        let cancel = Cancel::new();
        cancel.cancel();
        tokio::time::timeout(std::time::Duration::from_secs(1), cancel.cancelled())
            .await
            .expect("cancel-before-wait must complete");
    }

    #[tokio::test]
    async fn cancellation_wakes_a_present_waiter() {
        // upholds: CANCEL-1 — the awaitable side observes cancellation while
        // suspended rather than relying on a polling interval.
        let cancel = Cancel::new();
        let waiter = cancel.clone();
        let task = tokio::spawn(async move { waiter.cancelled().await });
        tokio::task::yield_now().await;
        cancel.cancel();
        tokio::time::timeout(std::time::Duration::from_secs(1), task)
            .await
            .expect("cancel-during-wait must wake")
            .expect("waiter task must finish");
    }

    #[tokio::test]
    async fn cancellation_crosses_clones_for_poll_and_wait() {
        // upholds: CANCEL-1 — every clone observes one shared state through
        // both interfaces.
        let poller = Cancel::new();
        let waiter = poller.clone();
        let trigger = poller.clone();
        let task = tokio::spawn(async move { waiter.cancelled().await });
        trigger.cancel();
        tokio::time::timeout(std::time::Duration::from_secs(1), task)
            .await
            .expect("cross-clone wait must wake")
            .expect("waiter task must finish");
        assert!(poller.is_cancelled());
    }
}
