//! Alternate inference backends behind the [`crate::Completer`] seam.
//!
//! The local Candle [`crate::Engine`] is the reference `Completer`; the
//! modules here discharge the same effect over HTTP instead of a blocking
//! island. llama.cpp owns inference (architectures, quantization, kernels);
//! yatima keeps meaning (templates, reasoning channels, tools, the agent
//! fold) — the `Completer` boundary is where the two meet, exactly as
//! anticipated in completer.rs's module docs.

mod llama_server;
mod sse;

pub use llama_server::{
    ChildCleanupFailed, LlamaServer, LlamaServerCompleter, LlamaServerConfig, LlamaServerSpawn,
    ServerGates, ServerIdentity, ServerProps,
};
