//! The transcript vocabulary: [`Role`] and [`Turn`].
//!
//! These are the conversation primitives shared by the chat, template, and agent
//! layers — and by a future structured `Completer` boundary that takes turns
//! rather than a rendered string. They live here, *below* all of those, so
//! nothing depends upward into the agent layer for them (they previously lived
//! in `agent`, which made `template` and `chat` depend on the agent module for a
//! type that has nothing to do with tools).

/// A role in the transcript — mirrors the de-facto standard (system / user /
/// assistant / tool). `Tool` carries a tool result fed back to the model in the
/// agent loop; "tool" is part of the standard chat-message vocabulary (every
/// chat API has it), not an agent-private concept, so it belongs here.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    System,
    User,
    Assistant,
    Tool,
}

/// One transcript entry: a role and its text content.
///
/// `content` is renderable turn text. For an [`Assistant`](Role::Assistant) turn
/// it is the *answer* only: a reasoning model's chain-of-thought is split off at
/// the completion→turn boundary (REASON-1, [`crate::split_reasoning`]) before the
/// turn is built, so it never enters a transcript that is re-rendered into a
/// later prompt.
#[derive(Debug, Clone)]
pub struct Turn {
    pub role: Role,
    pub content: String,
}
