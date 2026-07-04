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

/// A successful tool result at most this long (and single-line) is shown
/// verbatim in the reasoning fold; anything bigger is summarized as a char
/// count. Short results — a file path, a count, an ID — *are* the deliverable,
/// and counting their characters would hide them (the plot tool's
/// "wrote <path> …" being the motivating case).
pub const TOOL_NOTE_MAX_CHARS: usize = 200;
