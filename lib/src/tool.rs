//! Typed, capability-holding tools and the call protocol.
//!
//! A [`Tool`] is a Rust function the model may invoke; it *holds* its
//! capabilities (it is constructed with them), so its authority is bounded by
//! construction — we never hand it ambient `std::fs`. [`Tools`] is the set an
//! agent may use; [`Tools::dispatch`] never hard-errors — an unknown name
//! (AGENT-2) or a tool failure becomes an `is_error` [`ToolResult`] the model
//! can see and recover from (PROTO-1).
//!
//! [`ToolCallCodec`] is the wire format between model text and a [`ToolCall`].
//! [`JsonToolCall`] is the first impl: the `<tool_call>{json}</tool_call>`
//! convention for a base model with no native tool tokens. Schemas follow the
//! de-facto standard (JSON Schema params, name + JSON args); a future codec can
//! emit a native tool-call form or a constrained grammar.

use crate::capability::Dir;
use anyhow::{anyhow, bail, Result};
use serde_json::Value;

/// What a tool advertises to the model: its name, a description, and a JSON
/// Schema for its arguments.
#[derive(Debug, Clone)]
pub struct ToolSpec {
    pub name: String,
    pub description: String,
    pub params: Value,
}

/// A parsed request to call a tool: a name and JSON arguments.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolCall {
    pub name: String,
    pub args: Value,
}

/// The outcome of a tool call, fed back to the model. `is_error` distinguishes
/// a recoverable failure (unknown tool, bad args, IO error) from a result, so
/// the model can react — the failure is never silent (PROTO-1).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolResult {
    pub name: String,
    pub content: String,
    pub is_error: bool,
}

impl ToolResult {
    fn ok(name: &str, content: String) -> ToolResult {
        ToolResult {
            name: name.to_string(),
            content,
            is_error: false,
        }
    }

    fn error(name: &str, content: String) -> ToolResult {
        ToolResult {
            name: name.to_string(),
            content,
            is_error: true,
        }
    }
}

/// A tool the model may call. Implementors hold their capabilities and act only
/// through them.
pub trait Tool {
    /// What the model is told about this tool.
    fn spec(&self) -> ToolSpec;
    /// Run the tool. Returning `Err` is fine — [`Tools::dispatch`] turns it into
    /// an `is_error` [`ToolResult`]; the tool need not format failures itself.
    fn call(&self, args: &Value) -> Result<String>;
}

/// The set of tools an agent may use. The agent can call *only* these — a name
/// not present is uncallable (sandbox by omission, AGENT-2).
#[derive(Default)]
pub struct Tools {
    tools: Vec<Box<dyn Tool>>,
}

impl Tools {
    pub fn new() -> Tools {
        Tools::default()
    }

    /// Add a tool (builder style).
    pub fn with(mut self, tool: impl Tool + 'static) -> Tools {
        self.tools.push(Box::new(tool));
        self
    }

    /// The specs to advertise to the model.
    pub fn specs(&self) -> Vec<ToolSpec> {
        self.tools.iter().map(|t| t.spec()).collect()
    }

    /// Dispatch a call to the named tool. Never hard-errors: an unknown name or
    /// a tool-level failure becomes an `is_error` [`ToolResult`] (AGENT-2 /
    /// PROTO-1).
    pub fn dispatch(&self, call: &ToolCall) -> ToolResult {
        match self.tools.iter().find(|t| t.spec().name == call.name) {
            None => ToolResult::error(&call.name, format!("unknown tool '{}'", call.name)),
            Some(tool) => match tool.call(&call.args) {
                Ok(content) => ToolResult::ok(&call.name, content),
                Err(e) => ToolResult::error(&call.name, e.to_string()),
            },
        }
    }
}

/// The protocol between model text and tool calls.
pub trait ToolCallCodec {
    /// Instructions appended to the system prompt: how to call the tools, and
    /// the tools available.
    fn render_system(&self, specs: &[ToolSpec]) -> String;
    /// Strings at which generation should stop so the codec sees a complete
    /// call (the stop string is *included* in the completion text).
    fn stop_strings(&self) -> Vec<String>;
    /// Parse a completion: `None` if it is a plain answer (no call attempted),
    /// `Some(Ok(call))` for a well-formed call, `Some(Err(_))` for an attempted
    /// but malformed one (which becomes an error turn — PROTO-1).
    fn parse(&self, text: &str) -> Option<Result<ToolCall>>;
}

/// The `<tool_call>{ "name": ..., "args": {...} }</tool_call>` convention.
pub struct JsonToolCall;

const OPEN: &str = "<tool_call>";
const CLOSE: &str = "</tool_call>";

impl ToolCallCodec for JsonToolCall {
    fn render_system(&self, specs: &[ToolSpec]) -> String {
        let mut s = String::from(
            "You may call a tool. To do so, emit exactly one block and then stop:\n\
             <tool_call>{\"name\": \"<tool>\", \"args\": { ... }}</tool_call>\n\
             You will be shown the tool's result and may call again or answer. To \
             answer, reply with prose and no tool_call block.\n\nTools:\n",
        );
        for spec in specs {
            s.push_str(&format!(
                "- {}: {} (args schema: {})\n",
                spec.name, spec.description, spec.params
            ));
        }
        s
    }

    fn stop_strings(&self) -> Vec<String> {
        vec![CLOSE.to_string()]
    }

    fn parse(&self, text: &str) -> Option<Result<ToolCall>> {
        let start = text.find(OPEN)?;
        let rest = &text[start + OPEN.len()..];
        let json = match rest.find(CLOSE) {
            Some(end) => &rest[..end],
            None => return Some(Err(anyhow!("unterminated <tool_call> (no closing tag)"))),
        };
        Some(parse_call_json(json))
    }
}

/// Parse the JSON object inside a `<tool_call>` block into a [`ToolCall`]. Done
/// without `serde` derive (one fewer dep): a value with a string `name` and an
/// optional `args` object.
fn parse_call_json(json: &str) -> Result<ToolCall> {
    let value: Value =
        serde_json::from_str(json.trim()).map_err(|e| anyhow!("malformed tool_call JSON: {e}"))?;
    let name = value
        .get("name")
        .and_then(Value::as_str)
        .ok_or_else(|| anyhow!("tool_call missing string field 'name'"))?
        .to_string();
    let args = value
        .get("args")
        .cloned()
        .unwrap_or_else(|| Value::Object(Default::default()));
    Ok(ToolCall { name, args })
}

/// The DeepSeek-R1 native tool-call format. The model is *trained* to emit it,
/// so we speak its language rather than impose a convention: a call is
/// `<｜tool▁call▁begin｜>function<｜tool▁sep｜>NAME\n```json\nARGS\n```<｜tool▁call▁end｜>`.
/// These markers are plain text (not special tokens), so they survive
/// detokenization and a string-stop on `<｜tool▁call▁end｜>` works.
pub struct DeepSeekToolCall;

const DS_CALL_BEGIN: &str = "<｜tool▁call▁begin｜>";
const DS_CALL_END: &str = "<｜tool▁call▁end｜>";
const DS_SEP: &str = "<｜tool▁sep｜>";

impl ToolCallCodec for DeepSeekToolCall {
    fn render_system(&self, specs: &[ToolSpec]) -> String {
        let mut s = String::from(
            "You have access to the following tools. When a tool helps, call it \
             (you know the tool-call format); otherwise answer directly.\n\n\
             ## Tools\n",
        );
        for spec in specs {
            s.push_str(&format!(
                "- {}: {}\n  arguments (JSON Schema): {}\n",
                spec.name, spec.description, spec.params
            ));
        }
        s
    }

    fn stop_strings(&self) -> Vec<String> {
        vec![DS_CALL_END.to_string()]
    }

    fn parse(&self, text: &str) -> Option<Result<ToolCall>> {
        let begin = text.find(DS_CALL_BEGIN)?;
        Some(parse_deepseek_call(&text[begin + DS_CALL_BEGIN.len()..]))
    }
}

/// Parse the body after `<｜tool▁call▁begin｜>`: `function<｜tool▁sep｜>NAME\n```json
/// ARGS```…`. The name runs to the first newline; the arguments are the fenced
/// JSON block (falling back to the first balanced `{…}`).
fn parse_deepseek_call(after: &str) -> Result<ToolCall> {
    let sep = after
        .find(DS_SEP)
        .ok_or_else(|| anyhow!("tool call missing '{DS_SEP}'"))?;
    let rest = &after[sep + DS_SEP.len()..];
    let nl = rest
        .find('\n')
        .ok_or_else(|| anyhow!("tool call missing a name terminator"))?;
    let name = rest[..nl].trim().to_string();
    if name.is_empty() {
        bail!("tool call has an empty name");
    }
    let args = extract_json_block(&rest[nl..])?;
    Ok(ToolCall { name, args })
}

/// Extract a JSON value from a fenced ```json … ``` block, or failing that the
/// first balanced `{…}` span.
fn extract_json_block(s: &str) -> Result<Value> {
    let body = if let Some(i) = s.find("```json") {
        let rest = &s[i + "```json".len()..];
        let end = rest.find("```").unwrap_or(rest.len());
        &rest[..end]
    } else if let (Some(a), Some(b)) = (s.find('{'), s.rfind('}')) {
        &s[a..=b]
    } else {
        bail!("tool call missing JSON arguments");
    };
    serde_json::from_str(body.trim()).map_err(|e| anyhow!("malformed tool-call JSON: {e}"))
}

/// Read a UTF-8 text file under a [`Dir`] capability.
pub struct ReadFile {
    dir: Dir,
}

impl ReadFile {
    pub fn new(dir: Dir) -> ReadFile {
        ReadFile { dir }
    }
}

impl Tool for ReadFile {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "read_file".to_string(),
            description: "Read a UTF-8 text file, given a path relative to the root.".to_string(),
            params: serde_json::json!({
                "type": "object",
                "properties": {
                    "path": { "type": "string", "description": "file path relative to the root" }
                },
                "required": ["path"]
            }),
        }
    }

    fn call(&self, args: &Value) -> Result<String> {
        let path = args
            .get("path")
            .and_then(Value::as_str)
            .ok_or_else(|| anyhow!("read_file: missing string argument 'path'"))?;
        let full = self.dir.resolve(path)?; // CAP-1
        Ok(std::fs::read_to_string(&full)?)
    }
}

/// List directory entries under a [`Dir`] capability.
pub struct ListDir {
    dir: Dir,
}

impl ListDir {
    pub fn new(dir: Dir) -> ListDir {
        ListDir { dir }
    }
}

impl Tool for ListDir {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "list_dir".to_string(),
            description: "List entries of a directory, given a path relative to the root \
                          (use \"\" for the root)."
                .to_string(),
            params: serde_json::json!({
                "type": "object",
                "properties": {
                    "path": { "type": "string", "description": "directory path relative to the root" }
                },
                "required": ["path"]
            }),
        }
    }

    fn call(&self, args: &Value) -> Result<String> {
        let path = args
            .get("path")
            .and_then(Value::as_str)
            .ok_or_else(|| anyhow!("list_dir: missing string argument 'path'"))?;
        let full = self.dir.resolve(path)?; // CAP-1
        let mut names = Vec::new();
        for entry in std::fs::read_dir(&full)? {
            names.push(entry?.file_name().to_string_lossy().into_owned());
        }
        names.sort();
        Ok(names.join("\n"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn json(s: &str) -> Value {
        serde_json::from_str(s).unwrap()
    }

    #[test]
    fn codec_parses_whole_block_including_stop() {
        // upholds: PROTO-1 — the stop marker is included; the whole block parses.
        let codec = JsonToolCall;
        let text = "sure, let me look\n<tool_call>{\"name\": \"read_file\", \
                    \"args\": {\"path\": \"a.txt\"}}</tool_call>";
        let call = codec.parse(text).unwrap().unwrap();
        assert_eq!(call.name, "read_file");
        assert_eq!(call.args, json(r#"{"path": "a.txt"}"#));
    }

    #[test]
    fn codec_plain_answer_is_none() {
        // upholds: PROTO-1 — no call attempted ⇒ not an error, just an answer.
        assert!(JsonToolCall.parse("the answer is 42").is_none());
    }

    #[test]
    fn codec_malformed_is_some_err() {
        // upholds: PROTO-1 — an attempted-but-broken call is a parse error,
        // distinct from a plain answer.
        let codec = JsonToolCall;
        assert!(codec
            .parse("<tool_call>{not json}</tool_call>")
            .unwrap()
            .is_err());
        assert!(codec
            .parse("<tool_call>{\"args\":{}}</tool_call>")
            .unwrap()
            .is_err());
        assert!(codec.parse("<tool_call>{\"name\":\"x\"").unwrap().is_err());
    }

    #[test]
    fn deepseek_codec_parses_native_call() {
        // upholds: PROTO-1 — the model's native format parses to a ToolCall; the
        // think block before it and the stop marker after are ignored.
        let codec = DeepSeekToolCall;
        let text = "<think>\nlet me read it\n</think>\n\
                    <｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>read_file\n\
                    ```json\n{\"path\": \"a.txt\"}\n```<｜tool▁call▁end｜>";
        let call = codec.parse(text).unwrap().unwrap();
        assert_eq!(call.name, "read_file");
        assert_eq!(call.args, json(r#"{"path": "a.txt"}"#));
    }

    #[test]
    fn deepseek_codec_plain_answer_is_none_and_stops_on_call_end() {
        // upholds: PROTO-1 — pure reasoning/answer has no call; the stop string
        // is the native call terminator.
        let codec = DeepSeekToolCall;
        assert!(codec
            .parse("<think>done</think>\nThe answer is 42.")
            .is_none());
        assert_eq!(
            codec.stop_strings(),
            vec!["<｜tool▁call▁end｜>".to_string()]
        );
    }

    #[test]
    fn deepseek_codec_malformed_is_some_err() {
        // upholds: PROTO-1 — an attempted call with no JSON is a parse error.
        let codec = DeepSeekToolCall;
        let text = "<｜tool▁call▁begin｜>function<｜tool▁sep｜>read_file\nno json here";
        assert!(codec.parse(text).unwrap().is_err());
    }

    #[test]
    fn dispatch_unknown_tool_is_error_not_panic() {
        // upholds: AGENT-2 — a name not in the set is uncallable, surfaced as an
        // error result rather than ambient execution.
        let tools = Tools::new();
        let result = tools.dispatch(&ToolCall {
            name: "rm_rf".to_string(),
            args: Value::Null,
        });
        assert!(result.is_error);
        assert!(result.content.contains("unknown tool"));
    }

    #[test]
    fn read_file_is_confined_to_capability_root() {
        // upholds: CAP-1 — a Dir-scoped tool cannot read outside its root, even
        // when asked to; the failure is a recoverable error result.
        let tmp = tempfile::tempdir().unwrap();
        let mut f = std::fs::File::create(tmp.path().join("in.txt")).unwrap();
        write!(f, "hello").unwrap();

        let tools = Tools::new().with(ReadFile::new(Dir::new(tmp.path())));

        let ok = tools.dispatch(&ToolCall {
            name: "read_file".to_string(),
            args: json(r#"{"path": "in.txt"}"#),
        });
        assert!(!ok.is_error);
        assert_eq!(ok.content, "hello");

        let escape = tools.dispatch(&ToolCall {
            name: "read_file".to_string(),
            args: json(r#"{"path": "../../../etc/passwd"}"#),
        });
        assert!(escape.is_error);
    }

    #[test]
    fn list_dir_lists_sorted_entries_and_is_confined() {
        // upholds: CAP-1 — ListDir is capability-scoped like ReadFile; "" is the
        // root, and an escaping path is a recoverable error result.
        let tmp = tempfile::tempdir().unwrap();
        std::fs::File::create(tmp.path().join("b.txt")).unwrap();
        std::fs::File::create(tmp.path().join("a.txt")).unwrap();

        let tools = Tools::new().with(ListDir::new(Dir::new(tmp.path())));

        let listed = tools.dispatch(&ToolCall {
            name: "list_dir".to_string(),
            args: json(r#"{"path": ""}"#),
        });
        assert!(!listed.is_error);
        assert_eq!(listed.content, "a.txt\nb.txt");

        let escape = tools.dispatch(&ToolCall {
            name: "list_dir".to_string(),
            args: json(r#"{"path": ".."}"#),
        });
        assert!(escape.is_error);
    }
}
