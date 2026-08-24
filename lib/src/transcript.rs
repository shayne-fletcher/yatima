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

/// One transcript entry. The variants keep role-specific data structural: an
/// assistant tool invocation always has a name and JSON arguments, and a tool
/// result always has a name, content, and outcome bit. No prompt template has
/// to recover those meanings by parsing display text.
///
/// [`Assistant`](Turn::Assistant) contains answer text only. Reasoning and
/// protocol framing are split off at the completion-to-turn boundary
/// (REASON-1) before the turn is built. [`AssistantToolCall`](Turn::AssistantToolCall)
/// and [`ToolResult`](Turn::ToolResult) exist only in an agent run's working
/// transcript; persistent history still contains final user/assistant pairs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Turn {
    System(String),
    User(String),
    Assistant(String),
    AssistantToolCall {
        name: String,
        arguments: ToolArguments,
    },
    ToolResult {
        name: String,
        content: String,
        is_error: bool,
    },
}

/// Ordered, uniquely named arguments on an assistant tool invocation. ATEM
/// renders parameters in order, while dispatch consumes the equivalent JSON
/// object; this type carries both meanings without an unstructured string.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct ToolArguments(Vec<(String, serde_json::Value)>);

impl ToolArguments {
    pub fn try_from_pairs(
        pairs: impl IntoIterator<Item = (String, serde_json::Value)>,
    ) -> Result<Self, String> {
        let mut arguments = Vec::new();
        for (name, value) in pairs {
            if arguments.iter().any(|(seen, _)| seen == &name) {
                return Err(format!("duplicate tool parameter {name:?}"));
            }
            arguments.push((name, value));
        }
        Ok(Self(arguments))
    }

    pub fn from_json_object(arguments: &serde_json::Value) -> Result<Self, String> {
        let object = arguments
            .as_object()
            .ok_or_else(|| "tool arguments must be a JSON object".to_string())?;
        Self::try_from_pairs(
            object
                .iter()
                .map(|(name, value)| (name.clone(), value.clone())),
        )
    }

    pub fn iter(&self) -> impl Iterator<Item = (&str, &serde_json::Value)> {
        self.0.iter().map(|(name, value)| (name.as_str(), value))
    }

    pub fn to_json_object(&self) -> serde_json::Value {
        serde_json::Value::Object(self.0.iter().cloned().collect())
    }
}

/// Render JSON the way the Muse template's `tojson` filter does: compact on
/// one line, but with a space after separators. Object insertion order is
/// preserved by the crate's `serde_json/preserve_order` feature.
pub(crate) fn render_json_inline(value: &serde_json::Value) -> String {
    match value {
        serde_json::Value::Null => "null".to_string(),
        serde_json::Value::Bool(value) => value.to_string(),
        serde_json::Value::Number(value) => value.to_string(),
        serde_json::Value::String(value) => {
            serde_json::to_string(value).expect("serializing a JSON string cannot fail")
        }
        serde_json::Value::Array(values) => format!(
            "[{}]",
            values
                .iter()
                .map(render_json_inline)
                .collect::<Vec<_>>()
                .join(", ")
        ),
        serde_json::Value::Object(values) => format!(
            "{{{}}}",
            values
                .iter()
                .map(|(name, value)| format!(
                    "{}: {}",
                    serde_json::to_string(name).expect("serializing a JSON object key cannot fail"),
                    render_json_inline(value)
                ))
                .collect::<Vec<_>>()
                .join(", ")
        ),
    }
}

impl Turn {
    pub fn system(content: impl Into<String>) -> Self {
        Self::System(content.into())
    }

    pub fn user(content: impl Into<String>) -> Self {
        Self::User(content.into())
    }

    pub fn assistant(content: impl Into<String>) -> Self {
        Self::Assistant(content.into())
    }

    pub fn assistant_tool_call(name: impl Into<String>, arguments: ToolArguments) -> Self {
        Self::AssistantToolCall {
            name: name.into(),
            arguments,
        }
    }

    pub fn tool_result(
        name: impl Into<String>,
        content: impl Into<String>,
        is_error: bool,
    ) -> Self {
        Self::ToolResult {
            name: name.into(),
            content: content.into(),
            is_error,
        }
    }

    pub fn role(&self) -> Role {
        match self {
            Self::System(_) => Role::System,
            Self::User(_) => Role::User,
            Self::Assistant(_) | Self::AssistantToolCall { .. } => Role::Assistant,
            Self::ToolResult { .. } => Role::Tool,
        }
    }

    /// Text carried by an ordinary message or tool result. Tool invocations
    /// carry structured arguments and therefore have no text projection.
    pub fn content(&self) -> Option<&str> {
        match self {
            Self::System(content)
            | Self::User(content)
            | Self::Assistant(content)
            | Self::ToolResult { content, .. } => Some(content),
            Self::AssistantToolCall { .. } => None,
        }
    }
}

impl std::fmt::Display for Turn {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::System(content) | Self::User(content) | Self::Assistant(content) => {
                f.write_str(content)
            }
            Self::AssistantToolCall { name, arguments } => {
                write!(f, "{name} {}", arguments.to_json_object())
            }
            Self::ToolResult {
                name,
                content,
                is_error,
            } => write!(
                f,
                "[{name} {}] {content}",
                if *is_error { "error" } else { "ok" }
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn turn_variants_keep_tool_meaning_structural() {
        let arguments = ToolArguments::try_from_pairs([(
            "path".to_string(),
            serde_json::Value::String("README.md".to_string()),
        )])
        .unwrap();
        let call = Turn::assistant_tool_call("read_file", arguments);
        let result = Turn::tool_result("read_file", "yatima", false);

        assert_eq!(call.role(), Role::Assistant);
        assert_eq!(call.content(), None, "a call has no lossy text projection");
        assert!(matches!(
            call,
            Turn::AssistantToolCall { ref name, .. } if name == "read_file"
        ));
        assert_eq!(result.role(), Role::Tool);
        assert!(matches!(
            result,
            Turn::ToolResult {
                ref name,
                ref content,
                is_error: false,
            } if name == "read_file" && content == "yatima"
        ));
    }

    #[test]
    fn tool_arguments_preserve_order_and_reject_duplicates() {
        let arguments = ToolArguments::try_from_pairs([
            ("path".to_string(), serde_json::json!("README.md")),
            ("follow_symlinks".to_string(), serde_json::json!(true)),
        ])
        .unwrap();
        assert_eq!(
            arguments.iter().map(|(name, _)| name).collect::<Vec<_>>(),
            ["path", "follow_symlinks"]
        );
        assert!(ToolArguments::try_from_pairs([
            ("path".to_string(), serde_json::json!("a")),
            ("path".to_string(), serde_json::json!("b")),
        ])
        .is_err());
    }
}
