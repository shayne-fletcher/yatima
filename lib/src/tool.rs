//! Typed, capability-holding tools and the call protocol.
//!
//! A [`Tool`] is a Rust function the model may invoke; it *holds* its
//! capabilities (it is constructed with them), so its authority is bounded by
//! construction — we never hand it ambient `std::fs`. [`Tools`] is the set an
//! agent may use; [`Tools::dispatch_async`] never hard-errors — an unknown name
//! (AGENT-2), invalid arguments, cancellation, or a tool failure becomes a typed
//! [`ToolOutcome`]. The model sees only the projected [`ToolResult`] (PROTO-1).
//!
//! [`ToolCallCodec`] is the wire format between model text and a [`ToolCall`].
//! A real model uses its native codec ([`QwenToolCall`]); [`JsonToolCall`] is a
//! neutral `<tool_call>{json}</tool_call>` convention that is *not* any specific
//! model's format — it backs the model-free agent-loop tests and the `plain`
//! fallback for a model with no known native format. Schemas follow the de-facto
//! standard (JSON Schema params, name + JSON args).

use crate::capability::{Dir, NtfyTopic, WebOrigin, WriteDir};
use anyhow::{anyhow, bail, Result};
use async_trait::async_trait;
use reqwest::Client;
use serde_json::Value;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;
use thiserror::Error;
use tokio::sync::broadcast;
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;
use tracing::Instrument;

const DEFAULT_READ_URL_MAX_BYTES: usize = 1_000_000;

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

/// The model-facing projection of a tool call. Runtime code should reason over
/// [`ToolOutcome`]; this is the protocol value fed back to the model.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolResult {
    pub name: String,
    pub content: String,
    pub is_error: bool,
}

impl ToolResult {
    pub fn ok(name: &str, content: String) -> ToolResult {
        ToolResult {
            name: name.to_string(),
            content,
            is_error: false,
        }
    }

    pub fn error(name: &str, content: String) -> ToolResult {
        ToolResult {
            name: name.to_string(),
            content,
            is_error: true,
        }
    }
}

/// Runtime truth of a tool call. This is the algebra agents and supervisors
/// should reason over; [`ToolResult`] is only the model-facing projection.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ToolOutcome {
    Success { content: String },
    Rejected(ToolRejection),
    Failed(ToolFailure),
    Cancelled { reason: Option<String> },
    TimedOut { after: Duration },
}

impl ToolOutcome {
    pub fn success(content: impl Into<String>) -> ToolOutcome {
        ToolOutcome::Success {
            content: content.into(),
        }
    }

    pub fn render_for_model(&self, name: &str) -> ToolResult {
        match self {
            ToolOutcome::Success { content } => ToolResult::ok(name, content.clone()),
            ToolOutcome::Rejected(reason) => ToolResult::error(name, reason.to_string()),
            ToolOutcome::Failed(error) => ToolResult::error(name, error.to_string()),
            ToolOutcome::Cancelled { reason } => {
                let content = reason
                    .clone()
                    .unwrap_or_else(|| "tool call cancelled".to_string());
                ToolResult::error(name, content)
            }
            ToolOutcome::TimedOut { after } => {
                ToolResult::error(name, format!("tool call timed out after {after:?}"))
            }
        }
    }

    pub fn is_success(&self) -> bool {
        matches!(self, ToolOutcome::Success { .. })
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum ToolRejection {
    #[error("unknown tool '{name}'")]
    UnknownTool { name: String },
    #[error("invalid arguments: {message}")]
    InvalidArgs { message: String },
    #[error("capability denied: {message}")]
    CapabilityDenied { message: String },
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[error("tool failed: {message}")]
pub struct ToolFailure {
    pub message: String,
}

pub type ToolCallId = u64;

/// Tool execution context supplied by the dispatcher. It carries observability
/// and cooperative cancellation without making those concerns part of every
/// tool's argument schema.
#[derive(Clone)]
pub struct ToolCtx {
    call_id: ToolCallId,
    cancel: CancellationToken,
    events: broadcast::Sender<ToolEvent>,
}

impl ToolCtx {
    fn new(
        call_id: ToolCallId,
        cancel: CancellationToken,
        events: broadcast::Sender<ToolEvent>,
    ) -> ToolCtx {
        ToolCtx {
            call_id,
            cancel,
            events,
        }
    }

    pub fn call_id(&self) -> ToolCallId {
        self.call_id
    }

    pub fn is_cancelled(&self) -> bool {
        self.cancel.is_cancelled()
    }

    pub async fn cancelled(&self) {
        self.cancel.cancelled().await
    }

    pub fn emit_progress(&self, message: impl Into<String>) {
        let _ = self.events.send(ToolEvent::Progress {
            call_id: self.call_id,
            message: message.into(),
        });
    }
}

/// Observable lifecycle events for a tool call.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ToolEvent {
    Started {
        call_id: ToolCallId,
        call: ToolCall,
    },
    Progress {
        call_id: ToolCallId,
        message: String,
    },
    Finished {
        call_id: ToolCallId,
        outcome: ToolOutcome,
    },
    Cancelled {
        call_id: ToolCallId,
    },
}

/// A spawned tool call. The agent can join it, watch lifecycle events, or ask
/// the tool to stop via cooperative cancellation.
pub struct ToolTask {
    call_id: ToolCallId,
    cancel: CancellationToken,
    events: broadcast::Receiver<ToolEvent>,
    join: JoinHandle<ToolOutcome>,
}

impl ToolTask {
    pub fn call_id(&self) -> ToolCallId {
        self.call_id
    }

    pub fn cancel(&self) {
        self.cancel.cancel();
    }

    pub fn subscribe(&self) -> broadcast::Receiver<ToolEvent> {
        self.events.resubscribe()
    }

    pub async fn recv(&mut self) -> Option<ToolEvent> {
        loop {
            match self.events.recv().await {
                Ok(event) => return Some(event),
                Err(broadcast::error::RecvError::Lagged(_)) => continue,
                Err(broadcast::error::RecvError::Closed) => return None,
            }
        }
    }

    pub async fn join(self) -> ToolOutcome {
        match self.join.await {
            Ok(result) => result,
            Err(e) => ToolOutcome::Failed(ToolFailure {
                message: format!("tool task failed: {e}"),
            }),
        }
    }
}

/// A tool the model may call. Implementors hold their capabilities and act only
/// through them.
#[async_trait]
pub trait Tool: Send + Sync {
    /// What the model is told about this tool.
    fn spec(&self) -> ToolSpec;
    /// Run the tool. Returning `Err` is fine — [`Tools::dispatch_async`] turns it
    /// into a typed [`ToolOutcome`]; the tool need not format failures itself.
    async fn call(&self, args: Value, ctx: ToolCtx) -> Result<String>;
}

/// The set of tools an agent may use. The agent can call *only* these — a name
/// not present is uncallable (sandbox by omission, AGENT-2).
#[derive(Default)]
pub struct Tools {
    tools: Vec<Arc<dyn Tool>>,
    next_call_id: AtomicU64,
}

impl Tools {
    pub fn new() -> Tools {
        Tools::default()
    }

    /// Add a tool (builder style).
    pub fn with(mut self, tool: impl Tool + 'static) -> Tools {
        self.tools.push(Arc::new(tool));
        self
    }

    /// The specs to advertise to the model.
    pub fn specs(&self) -> Vec<ToolSpec> {
        self.tools.iter().map(|t| t.spec()).collect()
    }

    /// Dispatch a call to the named tool from synchronous code. This returns the
    /// model-facing projection for compatibility; async/runtime callers should
    /// prefer [`Tools::dispatch_async`] to get the full [`ToolOutcome`] algebra.
    pub fn dispatch(&self, call: &ToolCall) -> ToolResult {
        let outcome = match tokio::runtime::Handle::try_current() {
            Ok(handle) => {
                tokio::task::block_in_place(|| handle.block_on(self.dispatch_async(call)))
            }
            Err(_) => tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .expect("build tokio runtime for tool dispatch")
                .block_on(self.dispatch_async(call)),
        };
        outcome.render_for_model(&call.name)
    }

    /// Async dispatch for callers already in a Tokio runtime.
    pub async fn dispatch_async(&self, call: &ToolCall) -> ToolOutcome {
        self.spawn(call.clone()).join().await
    }

    /// Spawn a tool call as a Tokio task. Call this from within a Tokio runtime;
    /// use [`Tools::dispatch`] from synchronous code.
    pub fn spawn(&self, call: ToolCall) -> ToolTask {
        let call_id = self.next_call_id.fetch_add(1, Ordering::Relaxed) + 1;
        let tool_name = call.name.clone();
        let (events_tx, events_rx) = broadcast::channel(32);
        let cancel = CancellationToken::new();
        let ctx = ToolCtx::new(call_id, cancel.clone(), events_tx.clone());
        let tool = self
            .tools
            .iter()
            .find(|t| t.spec().name == call.name)
            .cloned();
        let task_call = call.clone();
        let span = tracing::debug_span!("tool.call", call_id, tool = %tool_name);
        // upholds: OBS-3 — attach the span to the future; do not hold an entered
        // span guard across await points inside the task.
        let join = tokio::spawn(
            async move {
                tracing::debug!(call_id, tool = %task_call.name, "tool started");
                let _ = events_tx.send(ToolEvent::Started {
                    call_id,
                    call: task_call.clone(),
                });
                let outcome = match tool {
                    None => ToolOutcome::Rejected(ToolRejection::UnknownTool {
                        name: task_call.name.clone(),
                    }),
                    Some(tool) => {
                        let fut = tool.call(task_call.args.clone(), ctx.clone());
                        tokio::select! {
                            _ = ctx.cancelled() => ToolOutcome::Cancelled { reason: None },
                            result = fut => match result {
                                Ok(content) => ToolOutcome::success(content),
                                Err(e) => classify_tool_error(e),
                            }
                        }
                    }
                };
                let outcome = if ctx.is_cancelled() && outcome.is_success() {
                    ToolOutcome::Cancelled { reason: None }
                } else {
                    outcome
                };
                if ctx.is_cancelled() {
                    let _ = events_tx.send(ToolEvent::Cancelled { call_id });
                }
                tracing::debug!(
                    call_id,
                    tool = %task_call.name,
                    outcome = ?outcome,
                    "tool finished"
                );
                let _ = events_tx.send(ToolEvent::Finished {
                    call_id,
                    outcome: outcome.clone(),
                });
                outcome
            }
            .instrument(span),
        );
        ToolTask {
            call_id,
            cancel,
            events: events_rx,
            join,
        }
    }
}

fn classify_tool_error(error: anyhow::Error) -> ToolOutcome {
    match error.downcast::<ToolRejection>() {
        Ok(rejection) => ToolOutcome::Rejected(rejection),
        Err(error) => ToolOutcome::Failed(ToolFailure {
            message: error.to_string(),
        }),
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

/// A neutral `<tool_call>{ "name": ..., "args": {...} }</tool_call>` convention —
/// not any specific model's native format (note `"args"`, vs Qwen's
/// `"arguments"`). Used by the model-free agent-loop tests and the `plain`
/// fallback; for a real model prefer its native codec ([`QwenToolCall`]).
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

/// The Qwen2.5-Instruct native tool-call format (ChatML + Hermes-style). The
/// model is trained for it: tool signatures are advertised in the system prompt
/// inside `<tools></tools>`, and a call is `<tool_call>\n{"name": ...,
/// "arguments": {...}}\n</tool_call>`.
pub struct QwenToolCall;

const QWEN_OPEN: &str = "<tool_call>";
const QWEN_CLOSE: &str = "</tool_call>";

impl ToolCallCodec for QwenToolCall {
    fn render_system(&self, specs: &[ToolSpec]) -> String {
        let mut s = String::from(
            "# Tools\n\nYou may call one or more functions to assist with the user \
             query.\n\nYou are provided with function signatures within \
             <tools></tools> XML tags:\n<tools>",
        );
        for spec in specs {
            let signature = serde_json::json!({
                "type": "function",
                "function": {
                    "name": spec.name,
                    "description": spec.description,
                    "parameters": spec.params,
                }
            });
            s.push('\n');
            s.push_str(&signature.to_string());
        }
        s.push_str(
            "\n</tools>\n\nFor each function call, return a json object with function \
             name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n\
             {\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call>\n\n\
             The name and all string values must be double-quoted. For example:\n\
             <tool_call>\n{\"name\": \"read_file\", \"arguments\": {\"path\": \"README.md\"}}\n\
             </tool_call>",
        );
        s
    }

    fn stop_strings(&self) -> Vec<String> {
        vec![QWEN_CLOSE.to_string()]
    }

    fn parse(&self, text: &str) -> Option<Result<ToolCall>> {
        let start = text.find(QWEN_OPEN)?;
        let rest = &text[start + QWEN_OPEN.len()..];
        let json = match rest.find(QWEN_CLOSE) {
            Some(end) => &rest[..end],
            None => return Some(Err(anyhow!("unterminated <tool_call> (no closing tag)"))),
        };
        Some(parse_qwen_call(json))
    }
}

/// Parse Qwen's `{"name": ..., "arguments": {...}}` call object. Strict JSON
/// first; on failure, a tolerant pass repairs the common real-model defect of an
/// unquoted name (the model imitating the `<function-name>` placeholder), as long
/// as the `arguments` object itself is valid JSON.
fn parse_qwen_call(json: &str) -> Result<ToolCall> {
    let json = json.trim();
    if let Ok(value) = serde_json::from_str::<Value>(json) {
        let name = value
            .get("name")
            .and_then(Value::as_str)
            .ok_or_else(|| anyhow!("tool_call missing string field 'name'"))?
            .to_string();
        let args = value
            .get("arguments")
            .cloned()
            .unwrap_or_else(|| Value::Object(Default::default()));
        return Ok(ToolCall { name, args });
    }
    lenient_qwen_call(json)
}

/// Tolerant recovery for not-quite-JSON tool calls: extract the function name
/// (quoted or a bare identifier) and the `arguments` object (which must itself
/// parse as JSON).
fn lenient_qwen_call(json: &str) -> Result<ToolCall> {
    let name = field_after(json, "\"name\"")
        .map(parse_name_token)
        .ok_or_else(|| anyhow!("malformed tool_call JSON: could not find a 'name'"))?;
    if name.is_empty() {
        bail!("tool_call has an empty name");
    }
    let args = match field_after(json, "\"arguments\"").and_then(balanced_object) {
        Some(obj) => {
            serde_json::from_str(&obj).map_err(|e| anyhow!("malformed tool_call arguments: {e}"))?
        }
        None => Value::Object(Default::default()),
    };
    Ok(ToolCall { name, args })
}

/// The substring just after `key` and its `:` separator.
fn field_after<'a>(s: &'a str, key: &str) -> Option<&'a str> {
    let i = s.find(key)?;
    let rest = &s[i + key.len()..];
    let colon = rest.find(':')?;
    Some(rest[colon + 1..].trim_start())
}

/// The name token at the start of `s`: a quoted string, or a bare identifier.
fn parse_name_token(s: &str) -> String {
    let s = s.trim_start();
    if let Some(rest) = s.strip_prefix('"') {
        rest.split('"').next().unwrap_or("").to_string()
    } else {
        s.chars()
            .take_while(|c| c.is_alphanumeric() || matches!(c, '_' | '-' | '.'))
            .collect()
    }
}

/// The first balanced `{...}` span at the start of `s`, counting braces only
/// outside JSON strings — so an argument value containing `{` or `}` (a path, a
/// snippet of code) does not close the object early.
fn balanced_object(s: &str) -> Option<String> {
    let start = s.find('{')?;
    let mut depth = 0i32;
    let mut in_string = false;
    let mut escaped = false;
    for (offset, ch) in s[start..].char_indices() {
        if in_string {
            match ch {
                _ if escaped => escaped = false,
                '\\' => escaped = true,
                '"' => in_string = false,
                _ => {}
            }
            continue;
        }
        match ch {
            '"' => in_string = true,
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    return Some(s[start..start + offset + 1].to_string());
                }
            }
            _ => {}
        }
    }
    None
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

#[async_trait]
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

    async fn call(&self, args: Value, _ctx: ToolCtx) -> Result<String> {
        let path = args
            .get("path")
            .and_then(Value::as_str)
            .ok_or_else(|| invalid_args("read_file: missing string argument 'path'"))?;
        let full = self.dir.resolve(path)?; // CAP-1
        Ok(tokio::fs::read_to_string(&full).await?)
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

/// Write a UTF-8 text file under a [`WriteDir`] capability.
pub struct WriteFile {
    dir: WriteDir,
}

impl WriteFile {
    pub fn new(dir: WriteDir) -> WriteFile {
        WriteFile { dir }
    }
}

#[async_trait]
impl Tool for WriteFile {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "write_file".to_string(),
            description: "Write UTF-8 text to a file, given a path relative to the write root."
                .to_string(),
            params: serde_json::json!({
                "type": "object",
                "properties": {
                    "path": { "type": "string", "description": "file path relative to the write root" },
                    "content": { "type": "string", "description": "UTF-8 text content to write" },
                    "create_dirs": {
                        "type": "boolean",
                        "description": "create missing parent directories before writing; defaults to false"
                    }
                },
                "required": ["path", "content"]
            }),
        }
    }

    async fn call(&self, args: Value, _ctx: ToolCtx) -> Result<String> {
        let path = required_string(&args, "write_file", "path")?;
        let content = required_string(&args, "write_file", "content")?;
        let create_dirs = optional_bool(&args, "write_file", "create_dirs")?.unwrap_or(false);
        let full = self.dir.resolve(path)?; // CAP-1
        if create_dirs {
            if let Some(parent) = full.parent() {
                tokio::fs::create_dir_all(parent).await?;
            }
        }
        tokio::fs::write(&full, content).await?;
        Ok(format!("wrote {} bytes", content.len()))
    }
}

/// Read a text response from a URL under a [`WebOrigin`] capability.
pub struct ReadUrl {
    origin: WebOrigin,
    client: Client,
    max_bytes: usize,
}

impl ReadUrl {
    pub fn new(origin: WebOrigin) -> Result<ReadUrl> {
        Self::with_max_bytes(origin, DEFAULT_READ_URL_MAX_BYTES)
    }

    pub fn with_max_bytes(origin: WebOrigin, max_bytes: usize) -> Result<ReadUrl> {
        let client = Client::builder().timeout(Duration::from_secs(10)).build()?;
        Ok(ReadUrl {
            origin,
            client,
            max_bytes,
        })
    }
}

#[async_trait]
impl Tool for ReadUrl {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "read_url".to_string(),
            description: "Read a UTF-8/text web URL under the configured origin.".to_string(),
            params: serde_json::json!({
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "absolute same-origin URL or path relative to the configured origin"
                    }
                },
                "required": ["url"]
            }),
        }
    }

    async fn call(&self, args: Value, _ctx: ToolCtx) -> Result<String> {
        let target = required_string(&args, "read_url", "url")?;
        let url = self.origin.resolve(target)?;
        let response = self.client.get(url).send().await?;
        let status = response.status();
        let body = response.bytes().await?;
        if !status.is_success() {
            bail!(
                "read_url failed with HTTP {status}: {}",
                String::from_utf8_lossy(&body)
            );
        }
        if body.len() > self.max_bytes {
            bail!(
                "read_url response too large: {} bytes exceeds {} byte limit",
                body.len(),
                self.max_bytes
            );
        }
        Ok(String::from_utf8(body.to_vec())?)
    }
}

/// Send a notification to a fixed ntfy topic held as a capability.
pub struct SendNotification {
    publisher: NtfyPublisher,
}

impl SendNotification {
    pub fn new(topic: NtfyTopic) -> Result<SendNotification> {
        Ok(SendNotification {
            publisher: NtfyPublisher::new(topic)?,
        })
    }
}

#[async_trait]
impl Tool for SendNotification {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "send_notification".to_string(),
            description: "Send a notification to the configured ntfy topic.".to_string(),
            params: serde_json::json!({
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string",
                        "description": "notification body"
                    },
                    "title": {
                        "type": "string",
                        "description": "optional notification title"
                    },
                    "priority": {
                        "type": "string",
                        "description": "optional ntfy priority: min, low, default, high, urgent, or 1-5",
                        "enum": ["min", "low", "default", "high", "urgent", "1", "2", "3", "4", "5"]
                    },
                    "tags": {
                        "type": "array",
                        "description": "optional ntfy tags/emojis",
                        "items": { "type": "string" }
                    }
                },
                "required": ["message"]
            }),
        }
    }

    async fn call(&self, args: Value, _ctx: ToolCtx) -> Result<String> {
        self.publisher
            .publish(&Notification::from_args(&args)?)
            .await
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct Notification {
    message: String,
    title: Option<String>,
    priority: Option<String>,
    tags: Vec<String>,
}

impl Notification {
    fn from_args(args: &Value) -> Result<Notification> {
        let message = required_string(args, "send_notification", "message")?.to_string();
        if message.is_empty() {
            return Err(invalid_args("send_notification: message must not be empty"));
        }
        let title = optional_string(args, "title")?.map(str::to_string);
        let priority = optional_string(args, "priority")?.map(str::to_string);
        if let Some(priority) = &priority {
            validate_ntfy_priority(priority)?;
        }
        let tags = optional_tags(args)?.unwrap_or_default();
        Ok(Notification {
            message,
            title,
            priority,
            tags,
        })
    }
}

struct NtfyPublisher {
    topic: NtfyTopic,
    client: Client,
}

impl NtfyPublisher {
    fn new(topic: NtfyTopic) -> Result<NtfyPublisher> {
        let client = Client::builder().timeout(Duration::from_secs(10)).build()?;
        Ok(NtfyPublisher { topic, client })
    }

    async fn publish(&self, notification: &Notification) -> Result<String> {
        let mut req = self
            .client
            .post(self.topic.endpoint())
            .body(notification.message.clone());
        if let Some(title) = &notification.title {
            req = req.header("Title", title);
        }
        if let Some(priority) = &notification.priority {
            req = req.header("Priority", priority);
        }
        if !notification.tags.is_empty() {
            req = req.header("Tags", notification.tags.join(","));
        }
        let response = req.send().await?;
        let status = response.status();
        let body = response.text().await.unwrap_or_default();
        if !status.is_success() {
            bail!("ntfy publish failed with HTTP {status}: {body}");
        }
        Ok(if body.trim().is_empty() {
            "notification sent".to_string()
        } else {
            body
        })
    }
}

fn required_string<'a>(args: &'a Value, tool: &str, field: &str) -> Result<&'a str> {
    args.get(field)
        .and_then(Value::as_str)
        .ok_or_else(|| invalid_args(format!("{tool}: missing string argument '{field}'")))
}

fn optional_string<'a>(args: &'a Value, field: &str) -> Result<Option<&'a str>> {
    match args.get(field) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(s)) => Ok(Some(s)),
        Some(_) => Err(invalid_args(format!(
            "send_notification: optional argument '{field}' must be a string"
        ))),
    }
}

fn optional_bool(args: &Value, tool: &str, field: &str) -> Result<Option<bool>> {
    match args.get(field) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::Bool(b)) => Ok(Some(*b)),
        Some(_) => Err(invalid_args(format!(
            "{tool}: optional argument '{field}' must be a boolean"
        ))),
    }
}

fn validate_ntfy_priority(priority: &str) -> Result<()> {
    match priority {
        "min" | "low" | "default" | "high" | "urgent" | "1" | "2" | "3" | "4" | "5" => Ok(()),
        _ => Err(invalid_args(format!(
            "send_notification: invalid priority {priority:?}"
        ))),
    }
}

fn optional_tags(args: &Value) -> Result<Option<Vec<String>>> {
    match args.get("tags") {
        None | Some(Value::Null) => Ok(None),
        Some(Value::Array(values)) => values
            .iter()
            .map(|v| {
                let tag = v
                    .as_str()
                    .ok_or_else(|| invalid_args("send_notification: tags must be strings"))?;
                if tag.contains(',') || tag.contains('\n') || tag.contains('\r') {
                    return Err(invalid_args(
                        "send_notification: tags must not contain commas or newlines",
                    ));
                }
                Ok(tag.to_string())
            })
            .collect::<Result<Vec<_>>>()
            .map(Some),
        Some(_) => Err(invalid_args(
            "send_notification: optional argument 'tags' must be an array",
        )),
    }
}

fn invalid_args(message: impl Into<String>) -> anyhow::Error {
    ToolRejection::InvalidArgs {
        message: message.into(),
    }
    .into()
}

#[async_trait]
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

    async fn call(&self, args: Value, _ctx: ToolCtx) -> Result<String> {
        let path = args
            .get("path")
            .and_then(Value::as_str)
            .ok_or_else(|| invalid_args("list_dir: missing string argument 'path'"))?;
        let full = self.dir.resolve(path)?; // CAP-1
        let mut names = Vec::new();
        let mut entries = tokio::fs::read_dir(&full).await?;
        while let Some(entry) = entries.next_entry().await? {
            names.push(entry.file_name().to_string_lossy().into_owned());
        }
        names.sort();
        Ok(names.join("\n"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;
    use std::io::Write;
    use std::sync::{Arc, Mutex};
    use tracing::field::{Field, Visit};
    use tracing::Subscriber;
    use tracing_subscriber::layer::Context;
    use tracing_subscriber::prelude::*;
    use tracing_subscriber::Layer;
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    fn json(s: &str) -> Value {
        serde_json::from_str(s).unwrap()
    }

    #[derive(Clone, Debug, Default)]
    struct CapturedTracing {
        events: Arc<Mutex<Vec<CapturedRecord>>>,
    }

    impl CapturedTracing {
        fn records(&self) -> Vec<CapturedRecord> {
            self.events.lock().unwrap().clone()
        }
    }

    #[derive(Clone, Debug, PartialEq, Eq)]
    struct CapturedRecord {
        name: String,
        fields: BTreeSet<String>,
    }

    impl<S> Layer<S> for CapturedTracing
    where
        S: Subscriber,
    {
        fn on_new_span(
            &self,
            attrs: &tracing::span::Attributes<'_>,
            _id: &tracing::span::Id,
            _ctx: Context<'_, S>,
        ) {
            let mut fields = FieldNames::default();
            attrs.record(&mut fields);
            self.events.lock().unwrap().push(CapturedRecord {
                name: attrs.metadata().name().to_string(),
                fields: fields.0,
            });
        }

        fn on_event(&self, event: &tracing::Event<'_>, _ctx: Context<'_, S>) {
            let mut fields = FieldNames::default();
            event.record(&mut fields);
            self.events.lock().unwrap().push(CapturedRecord {
                name: event.metadata().name().to_string(),
                fields: fields.0,
            });
        }
    }

    #[derive(Default)]
    struct FieldNames(BTreeSet<String>);

    impl Visit for FieldNames {
        fn record_debug(&mut self, field: &Field, _value: &dyn std::fmt::Debug) {
            self.0.insert(field.name().to_string());
        }
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

    proptest::proptest! {
        // upholds: PROTO-1 — no model output, however malformed, may panic a
        // codec's parse; it must return None or Some(Ok/Err).
        #[test]
        fn codecs_never_panic_on_arbitrary_text(s in ".*") {
            let _ = JsonToolCall.parse(&s);
            let _ = QwenToolCall.parse(&s);
            let wrapped = format!("<tool_call>{s}</tool_call>");
            let _ = JsonToolCall.parse(&wrapped);
            let _ = QwenToolCall.parse(&wrapped);
        }
    }

    #[test]
    fn qwen_codec_parses_native_call() {
        // upholds: PROTO-1 — Qwen's ChatML/Hermes call object parses (note the
        // 'arguments' key, distinct from our JsonToolCall 'args').
        let codec = QwenToolCall;
        let text = "<tool_call>\n{\"name\": \"read_file\", \"arguments\": {\"path\": \"a.txt\"}}\n</tool_call>";
        let call = codec.parse(text).unwrap().unwrap();
        assert_eq!(call.name, "read_file");
        assert_eq!(call.args, json(r#"{"path": "a.txt"}"#));
    }

    #[test]
    fn qwen_codec_tolerates_unquoted_name() {
        // upholds: PROTO-1 — real Qwen output sometimes leaves the name unquoted
        // (imitating the <function-name> placeholder); recover it when the
        // arguments object is valid JSON (the exact first-call string we saw).
        let codec = QwenToolCall;
        let text =
            "<tool_call>\n{\"name\": read_file, \"arguments\": {\"path\": \"secret.txt\"}}\n</tool_call>";
        let call = codec.parse(text).unwrap().unwrap();
        assert_eq!(call.name, "read_file");
        assert_eq!(call.args, json(r#"{"path": "secret.txt"}"#));
    }

    #[test]
    fn qwen_codec_strict_with_braces_in_values() {
        // upholds: PROTO-1 — valid JSON whose values contain braces (code, paths)
        // parses via the strict path untouched.
        let codec = QwenToolCall;
        let text = "<tool_call>\n{\"name\": \"write_file\", \"arguments\": \
                    {\"path\": \"m.rs\", \"content\": \"fn main() {}\"}}\n</tool_call>";
        let call = codec.parse(text).unwrap().unwrap();
        assert_eq!(call.name, "write_file");
        assert_eq!(call.args["content"], "fn main() {}");
    }

    #[test]
    fn qwen_codec_tolerant_with_braces_in_values() {
        // upholds: PROTO-1 — the tolerant path (unquoted name) must still capture
        // an arguments object whose string values contain `}` (the balanced_object
        // string-awareness fix); a naive brace counter would truncate here.
        let codec = QwenToolCall;
        let text = "<tool_call>\n{\"name\": write_file, \"arguments\": \
                    {\"path\": \"m.rs\", \"content\": \"x } y { z\"}}\n</tool_call>";
        let call = codec.parse(text).unwrap().unwrap();
        assert_eq!(call.name, "write_file");
        assert_eq!(call.args["path"], "m.rs");
        assert_eq!(call.args["content"], "x } y { z");
    }

    #[test]
    fn qwen_codec_tolerant_with_nested_args() {
        // upholds: PROTO-1 — unquoted name with a nested arguments object.
        let codec = QwenToolCall;
        let text = "<tool_call>\n{\"name\": configure, \"arguments\": \
                    {\"opts\": {\"a\": 1, \"b\": [2, 3]}}}\n</tool_call>";
        let call = codec.parse(text).unwrap().unwrap();
        assert_eq!(call.name, "configure");
        assert_eq!(call.args["opts"]["b"][1], 3);
    }

    #[test]
    fn qwen_codec_plain_and_malformed() {
        // upholds: PROTO-1
        let codec = QwenToolCall;
        assert!(codec.parse("Just an answer.").is_none());
        assert!(codec
            .parse("<tool_call>\nnot json\n</tool_call>")
            .unwrap()
            .is_err());
        assert_eq!(codec.stop_strings(), vec!["</tool_call>".to_string()]);
    }

    #[test]
    fn dispatch_unknown_tool_is_error_not_panic() {
        // upholds: AGENT-2 — a name not in the set is uncallable, surfaced as an
        // error result rather than ambient execution.
        let tools = Tools::new();
        let call = ToolCall {
            name: "rm_rf".to_string(),
            args: Value::Null,
        };
        let result = tools.dispatch(&call);
        assert!(result.is_error);
        assert!(result.content.contains("unknown tool"));
        let outcome = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap()
            .block_on(tools.dispatch_async(&call));
        assert_eq!(
            outcome,
            ToolOutcome::Rejected(ToolRejection::UnknownTool {
                name: "rm_rf".to_string()
            })
        );
    }

    #[test]
    fn read_file_missing_path_arg_is_error() {
        // upholds: PROTO-1 — a call missing a required argument is a recoverable
        // error result, not a panic.
        let tmp = tempfile::tempdir().unwrap();
        let tools = Tools::new().with(ReadFile::new(Dir::new(tmp.path())));
        let call = ToolCall {
            name: "read_file".to_string(),
            args: json("{}"),
        };
        let r = tools.dispatch(&call);
        assert!(r.is_error);
        let r2 = tools.dispatch(&ToolCall {
            name: "read_file".to_string(),
            args: Value::Null,
        });
        assert!(r2.is_error);

        let outcome = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap()
            .block_on(tools.dispatch_async(&call));
        assert_eq!(
            outcome,
            ToolOutcome::Rejected(ToolRejection::InvalidArgs {
                message: "read_file: missing string argument 'path'".to_string()
            })
        );
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

    #[test]
    fn write_file_writes_under_write_capability_root() {
        // upholds: CAP-1 — a WriteDir-scoped tool cannot write outside its root.
        let tmp = tempfile::tempdir().unwrap();
        let tools = Tools::new().with(WriteFile::new(WriteDir::new(tmp.path())));

        let ok = tools.dispatch(&ToolCall {
            name: "write_file".to_string(),
            args: json(r#"{"path": "notes/out.txt", "content": "hello", "create_dirs": true}"#),
        });
        assert!(!ok.is_error, "{}", ok.content);
        assert_eq!(
            std::fs::read_to_string(tmp.path().join("notes/out.txt")).unwrap(),
            "hello"
        );

        let escape = tools.dispatch(&ToolCall {
            name: "write_file".to_string(),
            args: json(r#"{"path": "../out.txt", "content": "bad"}"#),
        });
        assert!(escape.is_error);
    }

    #[test]
    fn write_file_validates_args() {
        // upholds: PROTO-1 — bad write args are recoverable tool errors.
        let tmp = tempfile::tempdir().unwrap();
        let tools = Tools::new().with(WriteFile::new(WriteDir::new(tmp.path())));
        for args in [
            json("{}"),
            json(r#"{"path": "x"}"#),
            json(r#"{"path": "x", "content": 1}"#),
            json(r#"{"path": "x", "content": "y", "create_dirs": "yes"}"#),
        ] {
            let result = tools.dispatch(&ToolCall {
                name: "write_file".to_string(),
                args,
            });
            assert!(result.is_error, "{result:?}");
        }
    }

    #[tokio::test]
    async fn send_notification_posts_to_capability_topic() {
        // upholds: CAP-2 — the notification tool's network authority is exactly
        // the held NtfyTopic capability, not a topic/server supplied by args.
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/we-could-be-coding-haskell"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "id": "test",
                "event": "message",
                "topic": "we-could-be-coding-haskell",
                "message": "Build finished"
            })))
            .expect(1)
            .mount(&server)
            .await;

        let server_uri = server.uri();
        let args = json(
            r#"{
                "message": "Build finished",
                "title": "yatima",
                "priority": "high",
                "tags": ["white_check_mark", "rust"],
                "topic": "attacker-topic",
                "server": "https://example.com"
            }"#,
        );

        let cap = NtfyTopic::with_server(&server_uri, "we-could-be-coding-haskell").unwrap();
        let tools = Tools::new().with(SendNotification::new(cap).unwrap());
        let result = tools
            .dispatch_async(&ToolCall {
                name: "send_notification".to_string(),
                args,
            })
            .await;

        let ToolOutcome::Success { content } = result else {
            panic!("unexpected outcome: {result:?}");
        };
        assert!(content.contains(r#""event":"message""#));
        let requests = server.received_requests().await.unwrap();
        assert_eq!(requests.len(), 1);
        let request = &requests[0];
        assert_eq!(request.url.path(), "/we-could-be-coding-haskell");
        assert_eq!(request.body, b"Build finished");
        assert_eq!(header_value(request, "title"), "yatima");
        assert_eq!(header_value(request, "priority"), "high");
        assert_eq!(header_value(request, "tags"), "white_check_mark,rust");
    }

    #[tokio::test]
    async fn read_url_gets_only_capability_origin() {
        // upholds: CAP-2 — web read authority is exactly the held WebOrigin.
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/doc"))
            .respond_with(ResponseTemplate::new(200).set_body_string("hello web"))
            .expect(1)
            .mount(&server)
            .await;

        let server_uri = server.uri();
        let origin = WebOrigin::new(&server_uri).unwrap();
        let tools = Tools::new().with(ReadUrl::new(origin).unwrap());
        let result = tools
            .dispatch_async(&ToolCall {
                name: "read_url".to_string(),
                args: json(r#"{"url": "/doc"}"#),
            })
            .await;
        assert_eq!(
            result,
            ToolOutcome::Success {
                content: "hello web".to_string()
            }
        );

        let requests = server.received_requests().await.unwrap();
        assert_eq!(requests.len(), 1);
        assert_eq!(requests[0].url.path(), "/doc");
    }

    #[test]
    fn read_url_rejects_escaping_origins_before_network() {
        // upholds: CAP-2 — an arbitrary host cannot be smuggled through args.
        let origin = WebOrigin::new("https://example.com").unwrap();
        let tools = Tools::new().with(ReadUrl::new(origin).unwrap());
        let result = tools.dispatch(&ToolCall {
            name: "read_url".to_string(),
            args: json(r#"{"url": "https://evil.example/doc"}"#),
        });
        assert!(result.is_error);
        assert!(result.content.contains("escapes web origin"));
    }

    #[test]
    fn notification_args_parse_to_typed_notification() {
        let notification = Notification::from_args(&json(
            r#"{
                "message": "Build finished",
                "title": "yatima",
                "priority": "high",
                "tags": ["white_check_mark", "rust"],
                "topic": "ignored",
                "server": "ignored"
            }"#,
        ))
        .unwrap();
        assert_eq!(
            notification,
            Notification {
                message: "Build finished".to_string(),
                title: Some("yatima".to_string()),
                priority: Some("high".to_string()),
                tags: vec!["white_check_mark".to_string(), "rust".to_string()]
            }
        );
    }

    #[test]
    fn send_notification_validates_args_before_publish() {
        // upholds: PROTO-1 — bad tool arguments are recoverable tool errors,
        // not malformed HTTP requests. No server is needed: validation is pure.
        let cap = NtfyTopic::with_server("http://127.0.0.1:1", "topic").unwrap();
        let tools = Tools::new().with(SendNotification::new(cap).unwrap());

        for args in [
            json("{}"),
            json(r#"{"message": ""}"#),
            json(r#"{"message": "x", "priority": "panic"}"#),
            json(r#"{"message": "x", "tags": ["bad,tag"]}"#),
        ] {
            let result = tools.dispatch(&ToolCall {
                name: "send_notification".to_string(),
                args,
            });
            assert!(result.is_error, "{result:?}");
        }
    }

    #[test]
    fn tool_outcome_projects_to_model_result() {
        let ok = ToolOutcome::Success {
            content: "42".to_string(),
        }
        .render_for_model("calc");
        assert_eq!(ok, ToolResult::ok("calc", "42".to_string()));

        let rejected = ToolOutcome::Rejected(ToolRejection::CapabilityDenied {
            message: "outside root".to_string(),
        })
        .render_for_model("read_file");
        assert_eq!(rejected.name, "read_file");
        assert!(rejected.is_error);
        assert_eq!(rejected.content, "capability denied: outside root");

        let failed = ToolOutcome::Failed(ToolFailure {
            message: "disk full".to_string(),
        })
        .render_for_model("write_file");
        assert!(failed.is_error);
        assert_eq!(failed.content, "tool failed: disk full");

        let timed_out = ToolOutcome::TimedOut {
            after: Duration::from_secs(3),
        }
        .render_for_model("read_url");
        assert!(timed_out.is_error);
        assert_eq!(timed_out.content, "tool call timed out after 3s");
    }

    struct ProgressTool;

    #[async_trait]
    impl Tool for ProgressTool {
        fn spec(&self) -> ToolSpec {
            ToolSpec {
                name: "progress".to_string(),
                description: "test progress".to_string(),
                params: serde_json::json!({ "type": "object" }),
            }
        }

        async fn call(&self, _args: Value, ctx: ToolCtx) -> Result<String> {
            ctx.emit_progress("halfway");
            Ok("done".to_string())
        }
    }

    #[tokio::test]
    async fn spawned_tool_is_watchable_and_joinable() {
        // upholds: AGENT-2 / CAP-2 — a tool call has a concrete lifecycle the
        // agent can observe without broadening the tool's authority.
        let tools = Tools::new().with(ProgressTool);
        let mut task = tools.spawn(ToolCall {
            name: "progress".to_string(),
            args: json("{}"),
        });

        let started = task.recv().await.unwrap();
        assert!(matches!(started, ToolEvent::Started { .. }));
        let progress = task.recv().await.unwrap();
        assert_eq!(
            progress,
            ToolEvent::Progress {
                call_id: task.call_id(),
                message: "halfway".to_string()
            }
        );
        let finished = task.recv().await.unwrap();
        assert!(matches!(
            finished,
            ToolEvent::Finished {
                outcome: ToolOutcome::Success { .. },
                ..
            }
        ));

        let result = task.join().await;
        assert_eq!(
            result,
            ToolOutcome::Success {
                content: "done".to_string()
            }
        );
    }

    #[test]
    fn tool_tracing_records_bounded_structured_fields() {
        // upholds: OBS-1, OBS-2, OBS-3, OBS-4 — the library emits structured
        // tool telemetry through a caller-supplied subscriber, attaches the
        // span to the async task, and records bounded fields rather than args.
        let captured = CapturedTracing::default();
        let subscriber = tracing_subscriber::registry().with(captured.clone());

        tracing::subscriber::with_default(subscriber, || {
            tracing::callsite::rebuild_interest_cache();
            let runtime = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .unwrap();
            runtime.block_on(async {
                let tools = Tools::new().with(ProgressTool);
                let result = tools
                    .dispatch_async(&ToolCall {
                        name: "progress".to_string(),
                        args: serde_json::json!({ "secret": "not a field" }),
                    })
                    .await;
                assert_eq!(
                    result,
                    ToolOutcome::Success {
                        content: "done".to_string()
                    }
                );
            });
        });

        let records = captured.records();
        let tool_span = records
            .iter()
            .find(|record| record.name == "tool.call")
            .expect("tool span was recorded");
        assert!(tool_span.fields.contains("call_id"));
        assert!(tool_span.fields.contains("tool"));
        assert!(!tool_span.fields.contains("args"));

        assert!(
            records.iter().all(|record| !record.fields.contains("args")),
            "tool telemetry must not expose tool args as fields: {records:#?}"
        );
    }

    struct CancelTool;

    #[async_trait]
    impl Tool for CancelTool {
        fn spec(&self) -> ToolSpec {
            ToolSpec {
                name: "cancel_me".to_string(),
                description: "waits for cancellation".to_string(),
                params: serde_json::json!({ "type": "object" }),
            }
        }

        async fn call(&self, _args: Value, ctx: ToolCtx) -> Result<String> {
            ctx.cancelled().await;
            Ok("unreachable".to_string())
        }
    }

    #[tokio::test]
    async fn spawned_tool_is_cancellable() {
        // upholds: AGENT-1 — cancellation gives a supervising agent a bounded
        // way out of a long-running tool.
        let tools = Tools::new().with(CancelTool);
        let mut task = tools.spawn(ToolCall {
            name: "cancel_me".to_string(),
            args: json("{}"),
        });

        assert!(matches!(
            task.recv().await.unwrap(),
            ToolEvent::Started { .. }
        ));
        task.cancel();
        let result = task.join().await;
        assert_eq!(result, ToolOutcome::Cancelled { reason: None });
    }

    fn header_value<'a>(request: &'a wiremock::Request, name: &str) -> &'a str {
        request.headers[name].to_str().unwrap()
    }

    #[tokio::test]
    #[ignore = "sends a real ntfy notification; set YATIMA_NTFY_TOPIC to run"]
    async fn e2e_send_notification_to_phone() {
        // upholds: CAP-2 — a real publish still goes through the same
        // pre-shared NtfyTopic capability.
        let topic = std::env::var("YATIMA_NTFY_TOPIC")
            .expect("set YATIMA_NTFY_TOPIC to a topic subscribed on your phone");
        let server =
            std::env::var("YATIMA_NTFY_SERVER").unwrap_or_else(|_| "https://ntfy.sh".to_string());
        let message = std::env::var("YATIMA_NTFY_MESSAGE").unwrap_or_else(|_| {
            format!(
                "Yatima live notification test from {}",
                std::env::var("USER").unwrap_or_else(|_| "your workspace".to_string())
            )
        });

        let cap = NtfyTopic::with_server(&server, topic).unwrap();
        let tools = Tools::new().with(SendNotification::new(cap).unwrap());
        let result = tools
            .dispatch_async(&ToolCall {
                name: "send_notification".to_string(),
                args: serde_json::json!({
                    "message": message,
                    "title": "yatima",
                    "priority": "default",
                    "tags": ["bell"]
                }),
            })
            .await;

        assert!(matches!(result, ToolOutcome::Success { .. }), "{result:?}");
    }
}
