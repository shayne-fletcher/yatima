//! Every host tunable on one documented page. These were duplicated verbatim
//! across the TUI and GUI; they live here now, so a change is made once and
//! every frontend (and yatima-serve) inherits it.

/// Tool rounds per turn before the agent gives up (AGENT-1); mirrors the CLI's
/// `--max-steps` default.
pub const AGENT_MAX_STEPS: usize = 6;

/// The base system prompt for tool-enabled sessions when `--system` is absent.
pub const DEFAULT_AGENT_SYSTEM: &str =
    "You are a helpful assistant. Call a tool when it helps, then answer. \
     Markdown image links do not render here: to show the user an image or \
     chart, call read_image (or plot) — its result is displayed \
     automatically.";

/// `read_page`'s readable-text budget for interactive use. The tool's own
/// default (40k chars ≈ 10–12k tokens) makes the next step's prefill take
/// minutes on a 32B local model; ~12k chars is plenty for summarize-and-answer
/// and keeps a tool turn interactive.
pub const READ_PAGE_MAX_CHARS: usize = 12_000;

/// `read_page`'s raw-input cap (unchanged from the tool's default).
pub const READ_PAGE_MAX_INPUT_BYTES: usize = 4_000_000;

/// Headroom (tokens) reserved above the compactable history for one run's
/// *within-run* growth — the current user turn plus the tool results that
/// accumulate over its steps (COMPACT-1 trims only committed history, never
/// mid-run). `read_page`'s window ([`READ_PAGE_MAX_CHARS`], ~3k tokens) is the
/// dominant contributor and can recur across [`AGENT_MAX_STEPS`] rounds;
/// `read_image` bytes ride out-of-band, leaving only a short summary. Sized
/// from the plan's P0 measurement (a live 6–8 turn image/read_page session);
/// tune here if a run's deepest step still crosses the depth budget. Compaction
/// keeps `system + history` under `depth_ceiling - max_tokens - TOOL_HEADROOM`
/// so the deepest step stays under the reliable depth (HOST-5).
pub const TOOL_HEADROOM: usize = 6_000;

/// The newest committed exchanges compaction always keeps, so a trimmed
/// session never snaps to an empty context mid-conversation (COMPACT-1's
/// `keep_last`).
pub const COMPACTION_KEEP_LAST: usize = 2;

/// A successful tool result at most this long (and single-line) is shown
/// verbatim in the reasoning fold; anything bigger is summarized as a char
/// count. Short results — a file path, a count, an ID — *are* the deliverable,
/// and counting their characters would hide them (the plot tool's
/// "wrote <path> …" being the motivating case).
pub const TOOL_NOTE_MAX_CHARS: usize = 200;
