//! Cooperative cancellation for an in-flight generation (the control plane).

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

/// A cooperative cancellation flag for a generation in flight. The decode loop
/// polls it once per token ([`crate::Engine::generate_with`]); flipping it from
/// another thread — a UI key handler, a memory watchdog — stops the turn at the
/// next token boundary with [`crate::StopReason::Stopped`], and the partial
/// output already produced is preserved (it is a clean stop, not an error).
///
/// Cheap to clone (a shared `Arc<AtomicBool>`): hand one clone to the decode
/// thread and keep another at the requester. A default / fresh handle never
/// fires until [`cancel`](Cancel::cancel) is called, so the non-cancelling
/// callers pass `&Cancel::new()` and pay nothing but a relaxed atomic load.
#[derive(Clone, Default)]
pub struct Cancel(Arc<AtomicBool>);

impl Cancel {
    /// A fresh, un-cancelled handle.
    pub fn new() -> Cancel {
        Cancel::default()
    }

    /// Request cancellation. Idempotent; safe to call from any thread.
    pub fn cancel(&self) {
        self.0.store(true, Ordering::Relaxed);
    }

    /// Whether cancellation has been requested.
    pub fn is_cancelled(&self) -> bool {
        self.0.load(Ordering::Relaxed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cancel_is_shared_and_one_way() {
        let a = Cancel::new();
        let b = a.clone();
        assert!(!a.is_cancelled() && !b.is_cancelled());
        b.cancel(); // a flip on any clone is visible on every clone
        assert!(a.is_cancelled(), "cancellation is shared across clones");
        b.cancel(); // idempotent
        assert!(a.is_cancelled());
    }
}
