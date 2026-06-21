//! Host-side conveniences for *using* `yatima-lib` — the small glue a program
//! embedding the runtime needs but the core does not: selecting a model's chat
//! format (by name or inferred from its architecture) and resolving where a
//! model's files come from. The CLI and the examples share these rather than
//! each re-deriving them.
//!
//! This layer depends on the engine ([`Arch`](crate::Arch)) but the engine never
//! depends on it — host capabilities (format/role) live here, engine-native
//! runtime policy (prefill) lives on `Arch`.

mod format;
mod profile;
mod source;

pub use format::{caps_for, resolve_format, Caps, ChatFormat, FormatMismatch};
pub use profile::ModelProfile;
pub use source::ModelSource;
