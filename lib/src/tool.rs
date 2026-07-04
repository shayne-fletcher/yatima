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

use crate::capability::{Dir, NtfyTopic, PlotSandbox, WebOrigins, WriteDir};
use anyhow::{anyhow, bail, Result};
use async_trait::async_trait;
use reqwest::{Client, Url};
use serde_json::Value;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;
use thiserror::Error;
use tokio::sync::broadcast;
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;
use tracing::Instrument;

const DEFAULT_READ_URL_MAX_BYTES: usize = 1_000_000;

/// `read_page` input cap: the streamed body is rejected once it exceeds this, so
/// a pathological page cannot be buffered into memory.
const DEFAULT_READ_PAGE_MAX_BYTES: usize = 4_000_000;
/// `read_page` output budget, in characters of the *readable article text* (the
/// returned string may exceed this by the title, URL, and truncation marker).
const DEFAULT_READ_PAGE_MAX_CHARS: usize = 40_000;
/// Structural guard handed to the readability extractor, independent of bytes.
const READ_PAGE_MAX_ELEMENTS: usize = 100_000;
/// Below this many non-whitespace chars, an extraction is treated as "no
/// readable content" (extractors return title-only/boilerplate on non-articles).
const READ_PAGE_MIN_TEXT_CHARS: usize = 20;
/// How many extracted pages a `ReadPage` keeps for continuation reads before
/// evicting the oldest (fetch-once: `offset` calls re-read the cache, never
/// the network — re-fetching is the expensive act for throttled hosts).
const READ_PAGE_CACHE_PAGES: usize = 16;

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

    pub fn kind(&self) -> &'static str {
        match self {
            ToolOutcome::Success { .. } => "success",
            ToolOutcome::Rejected(_) => "rejected",
            ToolOutcome::Failed(_) => "failed",
            ToolOutcome::Cancelled { .. } => "cancelled",
            ToolOutcome::TimedOut { .. } => "timed_out",
        }
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
    /// Whether the tool currently has any authority to act (CAP-3a): a tool
    /// whose capability is empty — e.g. a web tool before any origin grant —
    /// returns `false` and is left out of the advertised specs, so the prompt
    /// never names a tool the model cannot use. Calls still dispatch (and fail
    /// with the tool's own clear error) if the model tries anyway.
    fn available(&self) -> bool {
        true
    }
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

    /// The specs to advertise to the model: available tools only (CAP-3a) —
    /// the prompt always states the model's true, current authority.
    pub fn specs(&self) -> Vec<ToolSpec> {
        self.tools
            .iter()
            .filter(|t| t.available())
            .map(|t| t.spec())
            .collect()
    }

    /// Dispatch a call to the named tool from synchronous code. This returns the
    /// model-facing projection for compatibility; async/runtime callers should
    /// prefer [`Tools::dispatch_async`] to get the full [`ToolOutcome`] algebra.
    pub fn dispatch(&self, call: &ToolCall) -> ToolResult {
        let outcome = crate::runtime::block_on(self.dispatch_async(call));
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
                tracing::debug!(
                    call_id,
                    tool = %task_call.name,
                    args = %task_call.args,
                    "tool started"
                );
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
                    outcome = outcome.kind(),
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
    /// The marker that opens a tool call in this codec's wire format. A
    /// streaming consumer withholds answer text from the first (possibly
    /// partial) occurrence on, so codec markup never reaches a live answer
    /// channel (AGENT-4).
    fn open_marker(&self) -> String;
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
        // No advertised tools → no tool-calling instructions: the prompt
        // states the model's true authority (CAP-3a), and an agent session
        // with nothing granted reads as plain chat.
        if specs.is_empty() {
            return String::new();
        }
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

    fn open_marker(&self) -> String {
        OPEN.to_string()
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
        // As for the JSON codec: zero advertised tools, zero instructions
        // (CAP-3a — the prompt never claims authority the model lacks).
        if specs.is_empty() {
            return String::new();
        }
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

    fn open_marker(&self) -> String {
        QWEN_OPEN.to_string()
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

/// Read a text response from a URL under a [`WebOrigins`] capability.
/// The `User-Agent` for every web-touching tool. Many origins (Wikipedia's bot
/// policy, SEC EDGAR) reject or throttle anonymous clients — a descriptive UA
/// with a contact URL is required politeness, and reqwest sends none by
/// default.
const WEB_USER_AGENT: &str = concat!(
    "yatima/",
    env!("CARGO_PKG_VERSION"),
    " (+https://github.com/shayne-fletcher/yatima)"
);

pub struct ReadUrl {
    origins: WebOrigins,
    client: Client,
    max_bytes: usize,
}

impl ReadUrl {
    pub fn new(origins: WebOrigins) -> Result<ReadUrl> {
        Self::with_max_bytes(origins, DEFAULT_READ_URL_MAX_BYTES)
    }

    pub fn with_max_bytes(origins: WebOrigins, max_bytes: usize) -> Result<ReadUrl> {
        let client = Client::builder()
            .user_agent(WEB_USER_AGENT)
            .timeout(Duration::from_secs(10))
            .build()?;
        Ok(ReadUrl {
            origins,
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
            // The spec states the tool's live authority (CAP-3a).
            description: format!(
                "Read a UTF-8/text web URL. May read only these origins: {}.",
                self.origins.list().join(", ")
            ),
            params: serde_json::json!({
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "absolute URL on a granted origin (or a relative path when exactly one origin is granted)"
                    }
                },
                "required": ["url"]
            }),
        }
    }

    fn available(&self) -> bool {
        !self.origins.is_empty()
    }

    async fn call(&self, args: Value, _ctx: ToolCtx) -> Result<String> {
        let target = required_string(&args, "read_url", "url")?;
        let url = self.origins.resolve(target)?;
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

/// Read the **readable main content** of an HTML page under a [`WebOrigins`]
/// capability: fetch, extract the article (title + text) with a readability pass,
/// and return it as plain text truncated to a budget.
///
/// Distinct from [`ReadUrl`], which returns the raw body verbatim — use that for
/// JSON/plaintext/APIs. `read_page` is for **server-rendered HTML only**: it does
/// not execute JavaScript, bypass paywalls, or follow links to other origins
/// (CAP-2: authority is exactly the held origin). v1 decodes the body as UTF-8
/// and emits plain text (link `href`s are dropped; a markdown-with-links variant
/// is a possible v2).
pub struct ReadPage {
    origins: WebOrigins,
    client: Client,
    max_input_bytes: usize,
    max_output_chars: usize,
    /// Fetch-once cache: resolved URL → extracted page, so `offset`
    /// continuation calls are pure cache reads (zero network, zero throttle
    /// spend). FIFO-evicted at [`READ_PAGE_CACHE_PAGES`]; session-lifetime
    /// only. A std `Mutex` — never held across an `.await`.
    cache: std::sync::Mutex<PageCache>,
}

/// The extracted article a continuation call re-reads.
struct CachedPage {
    title: String,
    text: String,
    /// `(url, alt)` of the readable region's images, absolute and deduped —
    /// discovery metadata for `read_image` (listed in window 0's header).
    images: Vec<(String, String)>,
}

#[derive(Default)]
struct PageCache {
    pages: std::collections::HashMap<String, Arc<CachedPage>>,
    order: std::collections::VecDeque<String>,
}

impl PageCache {
    fn get(&self, url: &str) -> Option<Arc<CachedPage>> {
        self.pages.get(url).cloned()
    }

    fn insert(&mut self, url: String, page: Arc<CachedPage>) {
        if self.pages.insert(url.clone(), page).is_none() {
            self.order.push_back(url);
            while self.order.len() > READ_PAGE_CACHE_PAGES {
                if let Some(oldest) = self.order.pop_front() {
                    self.pages.remove(&oldest);
                }
            }
        }
    }
}

impl ReadPage {
    /// A capability-scoped page reader with default budgets.
    pub fn new(origins: WebOrigins) -> Result<ReadPage> {
        Self::with_limits(
            origins,
            DEFAULT_READ_PAGE_MAX_BYTES,
            DEFAULT_READ_PAGE_MAX_CHARS,
        )
    }

    /// A capability-scoped page reader with explicit input/output budgets.
    pub fn with_limits(
        origins: WebOrigins,
        max_input_bytes: usize,
        max_output_chars: usize,
    ) -> Result<ReadPage> {
        let client = Client::builder()
            .user_agent(WEB_USER_AGENT)
            .timeout(Duration::from_secs(15))
            .build()?;
        Ok(ReadPage {
            origins,
            client,
            max_input_bytes,
            max_output_chars,
            cache: std::sync::Mutex::new(PageCache::default()),
        })
    }

    /// Render one `max_output_chars` window of a cached page, starting at
    /// `offset` (in characters of the readable text). The trailing marker
    /// tells the model how to continue, so pagination is model-driven.
    fn render_window(&self, url: &str, page: &CachedPage, offset: usize) -> Result<String> {
        let total = page.text.chars().count();
        if offset >= total && !(offset == 0 && total == 0) {
            bail!(
                "read_page: offset {offset} is past the end of the article \
                 ({total} chars) at {url}"
            );
        }
        let body: String = page
            .text
            .chars()
            .skip(offset)
            .take(self.max_output_chars)
            .collect();
        let end = offset + body.chars().count();

        let mut out = String::new();
        let title = page.title.trim();
        if !title.is_empty() {
            out.push_str("# ");
            out.push_str(title);
            out.push('\n');
        }
        out.push_str(url);
        // Image discovery rides in the header (single-newline lines, so the
        // header/body/marker window structure is untouched — WIN-1), once,
        // in the first window.
        if offset == 0 && !page.images.is_empty() {
            out.push_str(
                "\n[images — call read_image to display one; markdown image \
                 links do not render:",
            );
            for (src, alt) in &page.images {
                out.push_str("\n  ");
                out.push_str(src);
                if !alt.is_empty() {
                    out.push_str(" (");
                    out.push_str(alt);
                    out.push(')');
                }
            }
            // Wikipedia-shaped sites serve images from a sibling origin
            // (upload.wikimedia.org vs en.wikipedia.org). Naming the missing
            // grant turns a doomed read_image into a request to the user —
            // who alone mints authority (CAP-3).
            let ungranted = ungranted_image_origins(&page.images, &self.origins);
            if !ungranted.is_empty() {
                out.push_str("\n  not granted: ");
                out.push_str(&ungranted.join(", "));
                out.push_str(
                    " — read_image will refuse these; ask the user \
                              to /grant first",
                );
            }
            out.push(']');
        }
        out.push_str("\n\n");
        out.push_str(&body);
        if end < total {
            out.push_str(&format!(
                "\n\n[chars {offset}..{end} of {total}; call read_page again \
                 with offset={end} for the rest]"
            ));
        } else if offset > 0 {
            out.push_str(&format!("\n\n[chars {offset}..{end} of {total}; end]"));
        }
        Ok(out)
    }
}

#[async_trait]
impl Tool for ReadPage {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "read_page".to_string(),
            // The spec states the tool's live authority (CAP-3a).
            description: format!(
                "Read the readable main content (title + article text) of an HTML page, \
                 listing the article's images (fetch one with read_image). \
                 May read only these origins: {}. Long articles are returned one window \
                 at a time; a truncation marker gives the offset to pass to read the \
                 next window (continuations are served from cache — no refetch). For \
                 raw or non-HTML responses (JSON, plaintext, APIs) use read_url \
                 instead. Server-rendered HTML only — no JavaScript, paywalls, or \
                 cross-origin links.",
                self.origins.list().join(", ")
            ),
            params: serde_json::json!({
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "absolute URL on a granted origin (or a relative path when exactly one origin is granted)"
                    },
                    "offset": {
                        "type": "integer",
                        "description": "character offset to continue a previously truncated read from (default 0)"
                    }
                },
                "required": ["url"]
            }),
        }
    }

    fn available(&self) -> bool {
        !self.origins.is_empty()
    }

    async fn call(&self, args: Value, _ctx: ToolCtx) -> Result<String> {
        let target = required_string(&args, "read_page", "url")?;
        let offset = match args.get("offset") {
            None | Some(Value::Null) => 0,
            Some(v) => v.as_u64().ok_or_else(|| {
                anyhow!("read_page: `offset` must be a non-negative integer, got {v}")
            })? as usize,
        };
        let url = self.origins.resolve(target)?;

        // Fetch-once: a cached extraction serves every continuation without
        // touching the network (or a throttled host's request budget).
        let cached = self
            .cache
            .lock()
            .expect("read_page cache poisoned")
            .get(url.as_str());
        if let Some(page) = cached {
            return self.render_window(url.as_str(), &page, offset);
        }

        let mut response = self.client.get(url.clone()).send().await?;
        let status = response.status();
        if !status.is_success() {
            bail!("read_page failed with HTTP {status} for {url}");
        }

        // Content-type gate (case-insensitive): reject obvious non-HTML before
        // reading or extracting. A missing/blank type is allowed (servers are
        // sloppy); the meaningful-content check below catches true non-HTML.
        if let Some(ct) = response
            .headers()
            .get(reqwest::header::CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
        {
            if !ct.to_ascii_lowercase().contains("html") {
                bail!(
                    "read_page: non-HTML response ({ct}) for {url}; use read_url for raw content"
                );
            }
        }

        // Content-Length preflight, then a streamed hard cap: never buffer a
        // pathological body via `bytes().await`.
        if let Some(len) = response.content_length() {
            if len as usize > self.max_input_bytes {
                bail!(
                    "read_page: response too large ({len} bytes > {} limit) for {url}; use read_url",
                    self.max_input_bytes
                );
            }
        }
        let mut buf: Vec<u8> = Vec::new();
        while let Some(chunk) = response.chunk().await? {
            if buf.len() + chunk.len() > self.max_input_bytes {
                bail!(
                    "read_page: response exceeded the {} byte limit for {url}; use read_url",
                    self.max_input_bytes
                );
            }
            buf.extend_from_slice(&chunk);
        }
        let html = String::from_utf8(buf).map_err(|_| {
            anyhow!("read_page: response was not valid UTF-8 HTML for {url}; use read_url")
        })?;

        // Extract off the async worker — also required because the extractor's
        // types are `!Send`, so they are created and consumed entirely here and
        // only owned `String`s escape.
        let url_string = url.to_string();
        let base = url.clone();
        let (title, text, images) = tokio::task::spawn_blocking(
            move || -> Result<(String, String, Vec<(String, String)>)> {
            let cfg = dom_smoothie::Config {
                max_elements_to_parse: READ_PAGE_MAX_ELEMENTS,
                ..Default::default()
            };
            let mut readability =
                dom_smoothie::Readability::new(html, Some(url_string.as_str()), Some(cfg)).map_err(
                    |e| {
                        anyhow!("read_page: could not initialize the extractor for {url_string}: {e}; use read_url for raw content")
                    },
                )?;
            let article = readability.parse().map_err(|e| {
                anyhow!("read_page: no readable article at {url_string}: {e}; use read_url for raw content")
            })?;
            let images = article_images(&article.content, &base);
            Ok((
                article.title.to_string(),
                article.text_content.to_string(),
                images,
            ))
        },)
        .await??;

        // Meaningful-content check (extractors return title-only/boilerplate on
        // non-articles); errors carry the resolved URL for debuggability.
        let text = text.trim();
        if text.chars().filter(|c| !c.is_whitespace()).count() < READ_PAGE_MIN_TEXT_CHARS {
            bail!("read_page: no readable article content at {url}; use read_url for raw content");
        }

        let page = Arc::new(CachedPage {
            title,
            text: text.to_string(),
            images,
        });
        self.cache
            .lock()
            .expect("read_page cache poisoned")
            .insert(url.to_string(), page.clone());
        self.render_window(url.as_str(), &page, offset)
    }
}

/// Cap on the images listed per page — discovery metadata, not a sitemap.
const READ_PAGE_MAX_IMAGES: usize = 12;

/// Cap on a fetched image's size — real diagrams and photos fit; a
/// pathological body can't buffer unbounded (the guard streams).
const DEFAULT_READ_IMAGE_MAX_BYTES: usize = 8_000_000;

/// The image types `read_image` will save, with their extensions and magic
/// signatures (the sniff when a server sends no content-type).
const IMAGE_TYPES: &[(&str, &str)] = &[
    ("image/svg+xml", "svg"),
    ("image/png", "png"),
    ("image/jpeg", "jpg"),
];

/// A quoted attribute's value inside an HTML tag body (`src="…"`), with a
/// boundary check so `srcset`/`data-src` never match as `src`. A scanner,
/// not a parser: good enough for discovery metadata, never authoritative.
fn attr_value(tag: &str, name: &str) -> Option<String> {
    let mut rest = tag;
    loop {
        let at = rest.find(name)?;
        let boundary = at == 0 || {
            let b = rest.as_bytes()[at - 1];
            !(b.is_ascii_alphanumeric() || b == b'-' || b == b'_')
        };
        let after = rest[at + name.len()..].trim_start();
        if boundary {
            if let Some(after) = after.strip_prefix('=') {
                let after = after.trim_start();
                let quote = after.chars().next()?;
                if quote == '"' || quote == '\'' {
                    let inner = &after[1..];
                    let end = inner.find(quote)?;
                    return Some(inner[..end].to_string());
                }
            }
        }
        rest = &rest[at + name.len()..];
    }
}

/// The readable region's `<img>` sources with their alt text: resolved
/// absolute against the page URL, deduped in document order, capped at
/// The origins of listed images the held origin set cannot reach — deduped,
/// render-ready (`scheme://host[:port]`), listing order. These are the
/// grants a `read_image` on the listing would need but does not have.
fn ungranted_image_origins(images: &[(String, String)], origins: &WebOrigins) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    for (src, _) in images {
        if origins.resolve(src).is_ok() {
            continue;
        }
        let Ok(url) = Url::parse(src) else {
            continue;
        };
        let origin = url.origin().ascii_serialization();
        if origin == "null" {
            continue; // opaque (data:, blob:) — no origin to name
        }
        if !out.contains(&origin) {
            out.push(origin);
        }
    }
    out
}

/// [`READ_PAGE_MAX_IMAGES`] — what `read_page` lists so the model can
/// discover something worth a `read_image` call.
fn article_images(content_html: &str, base: &Url) -> Vec<(String, String)> {
    let mut out: Vec<(String, String)> = Vec::new();
    let mut rest = content_html;
    while let Some(pos) = rest.find("<img") {
        rest = &rest[pos + 4..];
        let end = rest.find('>').unwrap_or(rest.len());
        let tag = &rest[..end];
        rest = &rest[end..];
        let Some(src) = attr_value(tag, "src") else {
            continue;
        };
        let Ok(resolved) = base.join(&src) else {
            continue;
        };
        let url = resolved.to_string();
        if out.iter().any(|(u, _)| *u == url) {
            continue;
        }
        let alt = attr_value(tag, "alt").unwrap_or_default();
        out.push((url, alt));
        if out.len() >= READ_PAGE_MAX_IMAGES {
            break;
        }
    }
    out
}

/// FNV-1a over `bytes` — content-hash filenames for artifacts (identical
/// bytes share an artifact; the model never chooses a path).
fn fnv1a(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf29ce484222325u64;
    for b in bytes {
        hash ^= u64::from(*b);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

/// Fetch an image from the granted origins and save it as a viewable
/// artifact.
///
/// The artifact-plane sibling of [`ReadUrl`]/[`ReadPage`], on the same
/// origin-set capability (CAP-2/CAP-3): the deliverable is a *file the user
/// can see*, and each host displays it in its medium's idiom (IMG-1 — the
/// GUI as an inline texture, the TUI via the platform viewer). The type
/// gate is honest: SVG/PNG/JPEG by content-type, magic-byte sniff when the
/// server is silent, and anything else teaches `read_url`/`read_page`.
/// Output is confined to the tool's [`WriteDir`] at a content-hash name.
pub struct ReadImage {
    origins: WebOrigins,
    dir: WriteDir,
    client: Client,
    max_bytes: usize,
    /// Fetch-once memo: resolved URL → the summary already returned. A
    /// repeat call is served from here (zero network, like `read_page`'s
    /// page cache) with a teaching tail: the user has already seen this
    /// image, so a model padding "show me another" with a re-fetch learns
    /// so instead of presenting a rerun as new. Session-lifetime only.
    fetched: std::sync::Mutex<std::collections::HashMap<String, String>>,
}

impl ReadImage {
    pub fn new(origins: WebOrigins, dir: impl Into<PathBuf>) -> Result<ReadImage> {
        Self::with_max_bytes(origins, dir, DEFAULT_READ_IMAGE_MAX_BYTES)
    }

    pub fn with_max_bytes(
        origins: WebOrigins,
        dir: impl Into<PathBuf>,
        max_bytes: usize,
    ) -> Result<ReadImage> {
        let dir = dir.into();
        std::fs::create_dir_all(&dir)?;
        let client = Client::builder()
            .user_agent(WEB_USER_AGENT)
            .timeout(Duration::from_secs(10))
            .build()?;
        Ok(ReadImage {
            origins,
            dir: WriteDir::new(dir),
            client,
            max_bytes,
            fetched: std::sync::Mutex::new(std::collections::HashMap::new()),
        })
    }
}

/// The saved-image extension: by content-type when the server sent a real
/// one, else by magic-byte sniff (`application/octet-stream` counts as "the
/// server said nothing"). `None` means "not an image we save".
fn image_ext(content_type: Option<&str>, body: &[u8]) -> Option<&'static str> {
    if let Some(ct) = content_type {
        let ct = ct.to_ascii_lowercase();
        if !ct.contains("application/octet-stream") {
            return IMAGE_TYPES
                .iter()
                .find(|(mime, _)| ct.contains(mime))
                .map(|(_, ext)| *ext);
        }
    }
    if body.starts_with(b"\x89PNG\r\n\x1a\n") {
        Some("png")
    } else if body.starts_with(&[0xFF, 0xD8, 0xFF]) {
        Some("jpg")
    } else {
        let head = &body[..body.len().min(512)];
        let head = String::from_utf8_lossy(head);
        head.contains("<svg").then_some("svg")
    }
}

#[async_trait]
impl Tool for ReadImage {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "read_image".to_string(),
            // The spec states the tool's live authority (CAP-3a).
            description: format!(
                "Fetch an image (SVG/PNG/JPEG) and save it for the user to \
                 view. Use an exact URL from a read_page [images] list — do \
                 not construct URLs. May read only these origins: {}. \
                 Returns the file path.",
                self.origins.list().join(", ")
            ),
            params: serde_json::json!({
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "absolute image URL on a granted origin"
                    }
                },
                "required": ["url"]
            }),
        }
    }

    fn available(&self) -> bool {
        !self.origins.is_empty()
    }

    async fn call(&self, args: Value, _ctx: ToolCtx) -> Result<String> {
        let target = required_string(&args, "read_image", "url")?;
        let url = self.origins.resolve(target)?; // CAP-2 before any network
                                                 // Fetch-once: a repeat of a URL this session re-teaches, never
                                                 // re-fetches — the artifact is already on disk and the user has
                                                 // already seen it.
        let memo = self
            .fetched
            .lock()
            .expect("read_image memo poisoned")
            .get(url.as_str())
            .cloned();
        if let Some(summary) = memo {
            return Ok(format!(
                "{summary} — already fetched and shown this session; pick a \
                 different image from the read_page [images] list if the \
                 user wants another"
            ));
        }
        let mut response = self.client.get(url.clone()).send().await?;
        let status = response.status();
        if !status.is_success() {
            bail!(
                "read_image failed with HTTP {status} for {url} — image URLs \
                 must be copied exactly from a read_page [images] list, never \
                 constructed (thumbnail URLs encode content hashes and a \
                 fixed size whitelist — both unguessable)"
            );
        }
        let content_type = response
            .headers()
            .get(reqwest::header::CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .map(str::to_owned);

        // Content-Length preflight, then a streamed hard cap: never buffer a
        // pathological body via `bytes().await`.
        if let Some(len) = response.content_length() {
            if len as usize > self.max_bytes {
                bail!(
                    "read_image: response too large ({len} bytes > {} byte limit) for {url}",
                    self.max_bytes
                );
            }
        }
        let mut body: Vec<u8> = Vec::new();
        while let Some(chunk) = response.chunk().await? {
            if body.len() + chunk.len() > self.max_bytes {
                bail!(
                    "read_image: response exceeded the {} byte limit for {url}",
                    self.max_bytes
                );
            }
            body.extend_from_slice(&chunk);
        }

        // Type honesty: only image types a host can show are saved; anything
        // else teaches the text tools.
        let Some(ext) = image_ext(content_type.as_deref(), &body) else {
            bail!(
                "read_image: {url} is not an SVG/PNG/JPEG image ({}); use \
                 read_page for HTML or read_url for raw content",
                content_type.as_deref().unwrap_or("no content-type")
            );
        };

        let name = format!("img-{:016x}.{ext}", fnv1a(&body));
        let out = self.dir.resolve(&name)?; // IMG-1 confinement
        tokio::fs::write(&out, &body).await?;
        let summary = format!("wrote {} ({ext}, {} bytes)", out.display(), body.len());
        self.fetched
            .lock()
            .expect("read_image memo poisoned")
            .insert(url.to_string(), summary.clone());
        Ok(summary)
    }
}

/// The generator the plot tool runs — the **only** code the sandbox's
/// interpreter ever executes (PLOT-1: the model supplies a declarative spec,
/// never code). Reads the validated spec as JSON on stdin; renders with the
/// Agg backend at fixed size/dpi and stable metadata, so the same spec
/// re-renders byte-identical on a machine (PLOT-3).
const PLOT_GENERATOR: &str = r#"
import sys, json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

spec = json.load(sys.stdin)
fig, ax = plt.subplots(figsize=(8, 4.5), dpi=120)
kind = spec["kind"]
for s in spec["series"]:
    y = s["y"]
    x = s.get("x") or list(range(len(y)))
    name = s.get("name")
    if kind == "line":
        ax.plot(x, y, label=name)
    elif kind == "scatter":
        ax.scatter(x, y, label=name, s=14)
    elif kind == "bar":
        ax.bar(x, y, label=name)
    elif kind == "hist":
        ax.hist(y, bins=spec.get("bins") or 30, label=name)
if spec.get("aspect") == "equal":
    ax.set_aspect("equal")
if spec.get("title"):
    ax.set_title(spec["title"])
if spec.get("xlabel"):
    ax.set_xlabel(spec["xlabel"])
if spec.get("ylabel"):
    ax.set_ylabel(spec["ylabel"])
if any(s.get("name") for s in spec["series"]):
    ax.legend()
ax.grid(True, alpha=0.25)
fig.tight_layout()
fig.savefig(spec["_out"], metadata={"Software": "yatima plot"})
"#;

/// Cap on series and total points — a plot is a summary, not a data dump.
const PLOT_MAX_SERIES: usize = 16;
const PLOT_MAX_POINTS: usize = 200_000;

/// Sample count for an `expr` series when the spec doesn't say — enough for
/// a smooth curve at the fixed render size.
const PLOT_EXPR_DEFAULT_SAMPLES: usize = 400;

/// The legal move, quoted in every rejection that stems from trying to put
/// code where numbers belong: tool errors are prompts, and one example
/// teaches better than six retries.
const PLOT_EXPR_EXAMPLE: &str = r#"{"expr": "sin(x)", "from": 0, "to": "2 * pi", "samples": 512}"#;

/// The chart vocabulary — closed by construction (PLOT-1): serde rejects
/// anything outside it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Deserialize, serde::Serialize)]
#[serde(rename_all = "snake_case")]
enum PlotKind {
    Line,
    Bar,
    Scatter,
    Hist,
}

/// Axis aspect ratio — `equal` makes circles circles.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Deserialize, serde::Serialize)]
#[serde(rename_all = "snake_case")]
enum PlotAspect {
    Auto,
    Equal,
}

/// One series, inline in the spec or host-registered by name. Two forms:
/// **data** (literal `y`, optional `x`) and **function** (`expr` over
/// `from..to`, sampled host-side — see [`crate::expr`]); exactly one of
/// `y` / `expr`.
#[derive(Debug, Clone, Default, serde::Deserialize, serde::Serialize)]
#[serde(deny_unknown_fields)]
pub struct PlotSeries {
    /// Legend label; a function series defaults to its expression text.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// Optional x values (data form); indices 0..n when omitted.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub x: Option<Vec<f64>>,
    /// Literal y values (data form).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub y: Option<Vec<f64>>,
    /// A function of `x` in the closed plot grammar (function form),
    /// e.g. `sin(x) * exp(-x/10)`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub expr: Option<String>,
    /// Sample range start (function form; required with `expr`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub from: Option<PlotBound>,
    /// Sample range end (function form; required with `expr`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub to: Option<PlotBound>,
    /// Sample count (function form; default [`PLOT_EXPR_DEFAULT_SAMPLES`]).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub samples: Option<u32>,
}

/// A range bound: a number, or a **constant** expression in the same closed
/// grammar — models speak trig ranges symbolically (`"9 * pi"`, `"2*pi"`),
/// and making them hand-compute 28.274 is the enumerate-by-hand failure in
/// miniature.
#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
#[serde(untagged)]
pub enum PlotBound {
    Num(f64),
    Expr(String),
}

impl PlotBound {
    fn resolve(&self, which: &str) -> Result<f64> {
        match self {
            PlotBound::Num(n) => Ok(*n),
            PlotBound::Expr(src) => {
                let e = crate::expr::parse(src)?;
                if e.references_x() {
                    bail!("plot: {which} must be a constant — it bounds the range x samples over");
                }
                let v = e.eval(0.0);
                if !v.is_finite() {
                    bail!("plot: {which} {src:?} is not finite");
                }
                Ok(v)
            }
        }
    }
}

/// Resolve one series to literal data: a data series validates and passes
/// through; a function series is parsed against the closed grammar and
/// sampled **here, in Rust** (PLOT-1: the interpreter still only ever sees
/// literal arrays). Every rejection teaches the legal move.
fn resolve_plot_series(s: &PlotSeries) -> Result<PlotSeries> {
    match (&s.y, &s.expr) {
        (Some(_), Some(_)) => bail!("plot: a series takes y or expr, not both"),
        (None, None) => bail!(
            "plot: a series needs y (literal numbers) or expr (a function \
             of x), e.g. {PLOT_EXPR_EXAMPLE}"
        ),
        (Some(y), None) => {
            if s.from.is_some() || s.to.is_some() || s.samples.is_some() {
                bail!("plot: from/to/samples belong to expr series");
            }
            if y.is_empty() {
                bail!("plot: a series has no y values");
            }
            if let Some(x) = &s.x {
                if x.len() != y.len() {
                    bail!(
                        "plot: series x/y length mismatch ({} vs {})",
                        x.len(),
                        y.len()
                    );
                }
            }
            Ok(s.clone())
        }
        (None, Some(src)) => {
            if s.x.is_some() {
                bail!("plot: an expr series samples its own x; give from/to instead");
            }
            let (from, to) = match (&s.from, &s.to) {
                (Some(f), Some(t)) => (f.resolve("from")?, t.resolve("to")?),
                _ => bail!("plot: expr needs from and to, e.g. {PLOT_EXPR_EXAMPLE}"),
            };
            if !from.is_finite() || !to.is_finite() || from >= to {
                bail!("plot: expr needs finite from < to");
            }
            let n = s.samples.map_or(PLOT_EXPR_DEFAULT_SAMPLES, |v| v as usize);
            if !(2..=PLOT_MAX_POINTS).contains(&n) {
                bail!("plot: samples must be between 2 and {PLOT_MAX_POINTS}");
            }
            let f = crate::expr::parse(src)?;
            let step = (to - from) / (n - 1) as f64;
            let xs: Vec<f64> = (0..n).map(|i| from + step * i as f64).collect();
            let ys: Vec<f64> = xs.iter().map(|&x| f.eval(x)).collect();
            if let Some(i) = ys.iter().position(|y| !y.is_finite()) {
                bail!(
                    "plot: {src:?} is non-finite at x = {} (asymptote or \
                     domain edge) — adjust from/to",
                    xs[i]
                );
            }
            Ok(PlotSeries {
                name: s.name.clone().or_else(|| Some(src.clone())),
                x: Some(xs),
                y: Some(ys),
                ..PlotSeries::default()
            })
        }
    }
}

/// The model-facing spec (PLOT-1): a closed schema — unknown fields, unknown
/// kinds, and anything code-shaped are typed rejections, never executed.
#[derive(Debug, serde::Deserialize, serde::Serialize)]
#[serde(deny_unknown_fields)]
struct PlotSpec {
    kind: PlotKind,
    #[serde(skip_serializing_if = "Option::is_none")]
    title: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    xlabel: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    ylabel: Option<String>,
    /// Histogram bin count (hist only).
    #[serde(skip_serializing_if = "Option::is_none")]
    bins: Option<u32>,
    /// Axis aspect; `equal` for shapes whose geometry matters.
    #[serde(skip_serializing_if = "Option::is_none")]
    aspect: Option<PlotAspect>,
    /// Inline data. Exactly one of `series` / `dataset`.
    #[serde(skip_serializing_if = "Option::is_none")]
    series: Option<Vec<PlotSeries>>,
    /// A host-registered dataset name. Exactly one of `series` / `dataset`.
    #[serde(skip_serializing_if = "Option::is_none")]
    dataset: Option<String>,
}

/// Render charts from a declarative spec inside a [`PlotSandbox`].
///
/// The model never writes code (PLOT-1): it submits a spec against a closed
/// schema, and the sandbox's pinned interpreter runs only the library's
/// generator. Data arrives inline (small series the model already holds),
/// as a **function of x** (`expr` — parsed against the closed grammar in
/// [`crate::expr`] and sampled host-side, so symbolic intent has a legal
/// channel and the interpreter still sees only literal arrays), or by naming
/// a **host-registered dataset** — the embedding program supplies the
/// numbers, the model supplies at most labels and choices. Output is one
/// PNG per call, confined to the sandbox (PLOT-2), named by the resolved
/// spec's hash so identical requests share an artifact (PLOT-3: same spec,
/// same bytes).
pub struct Plot {
    sandbox: PlotSandbox,
    datasets: std::collections::HashMap<String, Vec<PlotSeries>>,
}

impl Plot {
    pub fn new(sandbox: PlotSandbox) -> Plot {
        Plot {
            sandbox,
            datasets: std::collections::HashMap::new(),
        }
    }

    /// Register a named dataset (builder style) — the program-supplies-data
    /// shape: the model may reference it by name but never sees or alters
    /// the numbers.
    pub fn with_dataset(mut self, name: impl Into<String>, series: Vec<PlotSeries>) -> Plot {
        self.datasets.insert(name.into(), series);
        self
    }
}

#[async_trait]
impl Tool for Plot {
    fn spec(&self) -> ToolSpec {
        let datasets = if self.datasets.is_empty() {
            String::new()
        } else {
            let mut names: Vec<&str> = self.datasets.keys().map(String::as_str).collect();
            names.sort_unstable();
            format!(" Registered datasets: {}.", names.join(", "))
        };
        ToolSpec {
            name: "plot".to_string(),
            description: format!(
                "Render a chart to a PNG file from a declarative spec (no \
                 code). A series is either literal data (y required, x \
                 optional) or a function of x: {PLOT_EXPR_EXAMPLE} — grammar: \
                 numbers, x, pi, e, + - * / ^, parentheses, and sin cos tan \
                 sinh cosh tanh asin acos atan exp ln log log10 log2 sqrt \
                 abs floor ceil round sign. Prefer expr for mathematical \
                 functions; \
                 never enumerate function values by hand. Or name a \
                 registered dataset.{datasets} Returns the file path."
            ),
            params: serde_json::json!({
                "type": "object",
                "properties": {
                    "kind": { "type": "string", "enum": ["line", "bar", "scatter", "hist"] },
                    "title": { "type": "string" },
                    "xlabel": { "type": "string" },
                    "ylabel": { "type": "string" },
                    "bins": { "type": "integer", "description": "histogram bins (hist only)" },
                    "aspect": { "type": "string", "enum": ["auto", "equal"], "description": "equal makes circles circles" },
                    "series": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": { "type": "string" },
                                "x": { "type": "array", "items": { "type": "number" } },
                                "y": { "type": "array", "items": { "type": "number" } },
                                "expr": { "type": "string", "description": "function of x, e.g. sin(x)*exp(-x/10) — instead of y" },
                                "from": { "type": ["number", "string"], "description": "expr range start — a number or constant expression like \"2 * pi\"" },
                                "to": { "type": ["number", "string"], "description": "expr range end — a number or constant expression like \"9 * pi\"" },
                                "samples": { "type": "integer", "description": "expr sample count (default 400)" }
                            }
                        }
                    },
                    "dataset": { "type": "string", "description": "a registered dataset name (instead of series)" }
                },
                "required": ["kind"]
            }),
        }
    }

    async fn call(&self, args: Value, _ctx: ToolCtx) -> Result<String> {
        // PLOT-1: the closed schema is the whole authority story — unknown
        // fields (e.g. anything code-shaped) fail deserialization, and the
        // rejection teaches the legal channel for symbolic intent.
        let spec: PlotSpec = serde_json::from_value(args).map_err(|e| {
            anyhow!(
                "plot: invalid spec (closed schema): {e} — series data must \
                 be literal numbers; to plot a function of x, use an expr \
                 series, e.g. {PLOT_EXPR_EXAMPLE}"
            )
        })?;

        let series: Vec<PlotSeries> = match (&spec.series, &spec.dataset) {
            (Some(s), None) => s.clone(),
            (None, Some(name)) => self
                .datasets
                .get(name)
                .cloned()
                .ok_or_else(|| anyhow!("plot: unknown dataset {name:?}"))?,
            (Some(_), Some(_)) => bail!("plot: give either series or dataset, not both"),
            (None, None) => bail!("plot: give series or a dataset name"),
        };
        if series.is_empty() || series.len() > PLOT_MAX_SERIES {
            bail!("plot: between 1 and {PLOT_MAX_SERIES} series");
        }
        // Resolve every series to literal data (expr series sample here, in
        // Rust) before anything is counted, hashed, or rendered.
        let series: Vec<PlotSeries> = series
            .iter()
            .map(resolve_plot_series)
            .collect::<Result<_>>()?;
        let points: usize = series
            .iter()
            .map(|s| s.y.as_ref().map_or(0, Vec::len))
            .sum();
        if points > PLOT_MAX_POINTS {
            bail!("plot: {points} points exceeds the {PLOT_MAX_POINTS} cap");
        }

        // Resolved spec: the data the generator actually renders. Its hash
        // names the artifact (PLOT-3: identical request, identical file).
        let resolved = serde_json::json!({
            "kind": spec.kind,
            "title": spec.title,
            "xlabel": spec.xlabel,
            "ylabel": spec.ylabel,
            "bins": spec.bins,
            "aspect": spec.aspect,
            "series": series,
        });
        let payload = serde_json::to_string(&resolved)?;
        let name = format!("plot-{:016x}.png", fnv1a(payload.as_bytes()));
        let out = self.sandbox.resolve(&name)?; // PLOT-2 confinement

        let mut full = resolved;
        full["_out"] = serde_json::Value::String(out.display().to_string());

        let mut child = tokio::process::Command::new(self.sandbox.python())
            .args(["-c", PLOT_GENERATOR])
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::piped())
            .spawn()
            .map_err(|e| anyhow!("plot: could not run the interpreter: {e}"))?;
        {
            use tokio::io::AsyncWriteExt;
            let mut stdin = child.stdin.take().expect("piped stdin");
            stdin
                .write_all(serde_json::to_string(&full)?.as_bytes())
                .await?;
        }
        let output = child.wait_with_output().await?;
        if !output.status.success() {
            bail!(
                "plot: render failed: {}",
                String::from_utf8_lossy(&output.stderr).trim()
            );
        }

        let bytes = tokio::fs::read(&out).await?;
        let (w, h) = png_dims(&bytes)
            .ok_or_else(|| anyhow!("plot: generator produced an unreadable PNG"))?;
        Ok(format!(
            "wrote {} ({w}x{h}, {} bytes)",
            out.display(),
            bytes.len()
        ))
    }
}

/// Width/height from a PNG's IHDR chunk.
fn png_dims(bytes: &[u8]) -> Option<(u32, u32)> {
    if bytes.len() < 24 || &bytes[..8] != b"\x89PNG\r\n\x1a\n" {
        return None;
    }
    let be = |b: &[u8]| u32::from_be_bytes([b[0], b[1], b[2], b[3]]);
    Some((be(&bytes[16..20]), be(&bytes[20..24])))
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
        let client = Client::builder()
            .user_agent(WEB_USER_AGENT)
            .timeout(Duration::from_secs(10))
            .build()?;
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
    use std::io::Write;
    use wiremock::matchers::{header, method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

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
        let origin = WebOrigins::one(&server_uri).unwrap();
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

    #[tokio::test]
    async fn web_tools_send_a_descriptive_user_agent() {
        // Wikipedia's bot policy and SEC EDGAR both reject anonymous clients:
        // the descriptive UA must actually go out on the wire.
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/doc"))
            .and(header("user-agent", WEB_USER_AGENT))
            .respond_with(ResponseTemplate::new(200).set_body_string("ok"))
            .expect(1)
            .mount(&server)
            .await;

        let origin = WebOrigins::one(&server.uri()).unwrap();
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
                content: "ok".to_string()
            }
        );
    }

    #[test]
    fn read_url_rejects_escaping_origins_before_network() {
        // upholds: CAP-2 — an arbitrary host cannot be smuggled through args.
        let origin = WebOrigins::one("https://example.com").unwrap();
        let tools = Tools::new().with(ReadUrl::new(origin).unwrap());
        let result = tools.dispatch(&ToolCall {
            name: "read_url".to_string(),
            args: json(r#"{"url": "https://evil.example/doc"}"#),
        });
        assert!(result.is_error);
        assert!(result.content.contains("escapes the granted web origins"));
    }

    #[tokio::test]
    async fn read_page_lists_article_images_for_discovery() {
        // upholds: WIN-1 (header metadata never disturbs the window tiling)
        // + the read_page → read_image discovery seam: the readable region's
        // images are listed once, in window 0's header — absolute (relative
        // srcs resolved against the page), alt-labeled, srcset never
        // mistaken for src.
        let html = r#"<!DOCTYPE html><html><head><title>Impossible Objects</title></head>
<body><article>
<h1>Impossible Objects</h1>
<img src="/img/tri.svg" alt="Penrose triangle" srcset="/img/tri-2x.png 2x">
<p>The Penrose triangle is an impossible object first popularized in the
nineteen fifties, appearing widely in art and mathematical illustration as a
canonical example of a figure that cannot be realized in three dimensions.</p>
<img src="https://files.example/stairs.png" alt="Penrose stairs">
<p>The related Penrose stairs construction loops a staircase back onto
itself, ascending forever within a closed circuit, and features in several
well known works of art depicting paradoxical architecture.</p>
</article></body></html>"#;
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/objects"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_raw(html.as_bytes().to_vec(), "text/html; charset=utf-8"),
            )
            .mount(&server)
            .await;

        let origins = WebOrigins::one(&server.uri()).unwrap();
        let tools = Tools::new().with(ReadPage::with_limits(origins, 1_000_000, 80).unwrap());
        let first = read_window(&tools, "/objects", 0).await;
        assert!(!first.is_error, "{}", first.content);
        assert!(
            first
                .content
                .contains("[images — call read_image to display one"),
            "{}",
            first.content
        );
        assert!(
            first
                .content
                .contains(&format!("{}/img/tri.svg (Penrose triangle)", server.uri())),
            "relative src resolves absolute, alt rides along: {}",
            first.content
        );
        assert!(
            first
                .content
                .contains("https://files.example/stairs.png (Penrose stairs)"),
            "{}",
            first.content
        );
        assert!(
            !first.content.contains("tri-2x"),
            "srcset is not src: {}",
            first.content
        );
        // The off-origin image's missing grant is named (CAP-3: the model
        // must ask the user, not fail into a refusal it can't see coming);
        // the on-origin image draws no note.
        assert!(
            first
                .content
                .contains("not granted: https://files.example — read_image will refuse"),
            "{}",
            first.content
        );

        // Discovery rides window 0 only; continuations stay pure text.
        let next = next_offset(&first.content).expect("truncated at 80 chars");
        let cont = read_window(&tools, "/objects", next).await;
        assert!(!cont.is_error);
        assert!(
            !cont.content.contains("[images"),
            "listed once, in the first window: {}",
            cont.content
        );
    }

    #[test]
    fn ungranted_image_origins_name_the_missing_grants() {
        // The origins a read_image on the listing would need but lacks:
        // deduped, opaque schemes skipped, granted ones silent.
        let origins = WebOrigins::one("https://en.wikipedia.org").unwrap();
        let img = |src: &str| (src.to_string(), String::new());
        let images = vec![
            img("https://en.wikipedia.org/logo.png"),
            img("https://upload.wikimedia.org/a.jpg"),
            img("https://upload.wikimedia.org/b.jpg"),
            img("data:image/png;base64,AAAA"),
        ];
        assert_eq!(
            ungranted_image_origins(&images, &origins),
            ["https://upload.wikimedia.org"]
        );
        assert!(
            ungranted_image_origins(&images[..1], &origins).is_empty(),
            "an all-granted listing draws no note"
        );
    }

    #[tokio::test]
    async fn read_image_saves_typed_confined_and_content_hashed() {
        // upholds: IMG-1 — the artifact lands inside the tool's WriteDir at
        // a content-hash name with an honest extension; identical bytes
        // (even via a different URL) share an artifact. A *repeat* of the
        // same URL never touches the network again (expect(1) enforces) and
        // teaches that the user has already seen the image.
        let server = MockServer::start().await;
        let png: &[u8] = b"\x89PNG\r\n\x1a\nrest-of-image-bytes";
        for route in ["/tri.png", "/tri-copy.png"] {
            Mock::given(method("GET"))
                .and(path(route))
                .respond_with(
                    ResponseTemplate::new(200)
                        .insert_header("content-type", "image/png")
                        .set_body_bytes(png),
                )
                .expect(1)
                .mount(&server)
                .await;
        }

        let dir = tempfile::tempdir().unwrap();
        let origins = WebOrigins::one(&server.uri()).unwrap();
        let tools = Tools::new().with(ReadImage::new(origins, dir.path().join("images")).unwrap());
        let call = |route: &str| ToolCall {
            name: "read_image".to_string(),
            args: json(&format!(r#"{{"url": "{route}"}}"#)),
        };
        let first = tools.dispatch_async(&call("/tri.png")).await;
        let ToolOutcome::Success { content } = &first else {
            panic!("{first:?}");
        };
        assert!(content.starts_with("wrote "), "{content}");
        let path = content.split_whitespace().nth(1).unwrap();
        assert!(
            std::path::Path::new(path).starts_with(dir.path().join("images")),
            "IMG-1 confinement: {path}"
        );
        assert!(path.ends_with(".png"), "honest extension: {path}");
        assert_eq!(std::fs::read(path).unwrap(), png, "bytes saved verbatim");

        // Same URL again: a memo hit (the mock's expect(1) proves zero
        // network), same artifact, and the already-shown teaching tail.
        let repeat = tools.dispatch_async(&call("/tri.png")).await;
        let repeat = repeat.render_for_model("").content;
        assert!(
            repeat.contains(path),
            "the memo returns the artifact: {repeat}"
        );
        assert!(
            repeat.contains("already fetched and shown this session"),
            "a rerun teaches, not re-presents: {repeat}"
        );

        // A different URL with identical bytes fetches (its own expect(1))
        // and lands on the same content-hash artifact, without the note.
        let copy = tools.dispatch_async(&call("/tri-copy.png")).await;
        let copy = copy.render_for_model("").content;
        assert!(
            copy.contains(path),
            "identical bytes share an artifact: {copy}"
        );
        assert!(
            !copy.contains("already fetched"),
            "a fresh URL is not a rerun: {copy}"
        );
    }

    #[tokio::test]
    async fn read_image_sniffs_svg_when_the_server_is_silent() {
        // upholds: IMG-1 — an absent/octet-stream content-type falls back
        // to a magic-byte sniff; an SVG body saves as .svg.
        let server = MockServer::start().await;
        let svg = br#"<?xml version="1.0"?><svg xmlns="http://www.w3.org/2000/svg"/>"#;
        Mock::given(method("GET"))
            .and(path("/penrose"))
            .respond_with(ResponseTemplate::new(200).set_body_bytes(svg.as_slice()))
            .mount(&server)
            .await;

        let dir = tempfile::tempdir().unwrap();
        let origins = WebOrigins::one(&server.uri()).unwrap();
        let tools = Tools::new().with(ReadImage::new(origins, dir.path()).unwrap());
        let result = tools
            .dispatch_async(&ToolCall {
                name: "read_image".to_string(),
                args: json(r#"{"url": "/penrose"}"#),
            })
            .await;
        let ToolOutcome::Success { content } = &result else {
            panic!("{result:?}");
        };
        assert!(content.contains(".svg"), "sniffed as svg: {content}");
    }

    #[tokio::test]
    async fn read_image_gates_non_images_and_size() {
        // upholds: IMG-1 — a non-image response is a teaching rejection
        // (never saved), and the input cap trips while streaming.
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/page"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "text/html")
                    .set_body_string("<html>not an image</html>"),
            )
            .mount(&server)
            .await;
        Mock::given(method("GET"))
            .and(path("/huge.png"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "image/png")
                    .set_body_bytes(vec![0u8; 64]),
            )
            .mount(&server)
            .await;

        let dir = tempfile::tempdir().unwrap();
        let origins = WebOrigins::one(&server.uri()).unwrap();
        let tools = Tools::new().with(ReadImage::new(origins.clone(), dir.path()).unwrap());
        let capped = Tools::new().with(ReadImage::with_max_bytes(origins, dir.path(), 16).unwrap());

        let html = tools
            .dispatch_async(&ToolCall {
                name: "read_image".to_string(),
                args: json(r#"{"url": "/page"}"#),
            })
            .await
            .render_for_model("read_image");
        assert!(html.is_error);
        assert!(
            html.content.contains("read_page"),
            "teaches the text tools: {}",
            html.content
        );

        let huge = capped
            .dispatch_async(&ToolCall {
                name: "read_image".to_string(),
                args: json(r#"{"url": "/huge.png"}"#),
            })
            .await
            .render_for_model("read_image");
        assert!(huge.is_error);
        assert!(huge.content.contains("byte limit"), "{}", huge.content);
    }

    #[test]
    fn read_image_rejects_escaping_origins_before_network() {
        // upholds: CAP-2 — an arbitrary host cannot be smuggled through args.
        let dir = tempfile::tempdir().unwrap();
        let origins = WebOrigins::one("https://example.com").unwrap();
        let tools = Tools::new().with(ReadImage::new(origins, dir.path()).unwrap());
        let result = tools.dispatch(&ToolCall {
            name: "read_image".to_string(),
            args: json(r#"{"url": "https://evil.example/tri.svg"}"#),
        });
        assert!(result.is_error);
        assert!(result.content.contains("escapes the granted web origins"));
    }

    // A realistic article with nav/script/footer noise around the body.
    const ARTICLE_HTML: &str = r#"<!DOCTYPE html><html><head><title>The Quarterly Report</title></head>
<body>
<nav>HOME ABOUT CONTACT SUBSCRIBE_NAV_LINK</nav>
<script>var tracking = "BEACON_PIXEL_12345";</script>
<article>
<h1>The Quarterly Report</h1>
<p>Acme Corporation announced today that quarterly revenue rose sharply across all
divisions, driven by strong demand in the industrial segment and disciplined cost
control throughout the period under review by the board.</p>
<p>Management reaffirmed full-year guidance and highlighted continued investment in
research and development as a core priority for sustaining the company competitive
position over the coming years.</p>
</article>
<footer>Copyright 2026 Acme Corporation</footer>
</body></html>"#;

    #[tokio::test]
    async fn read_page_extracts_readable_article() {
        // upholds: the new behaviour vs read_url — main article, not page chrome.
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/post"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_raw(ARTICLE_HTML.as_bytes().to_vec(), "text/html; charset=utf-8"),
            )
            .mount(&server)
            .await;

        let tools =
            Tools::new().with(ReadPage::new(WebOrigins::one(&server.uri()).unwrap()).unwrap());
        let result = tools
            .dispatch_async(&ToolCall {
                name: "read_page".to_string(),
                args: json(r#"{"url": "/post"}"#),
            })
            .await
            .render_for_model("read_page");

        assert!(!result.is_error, "got error: {}", result.content);
        assert!(result.content.contains("Quarterly Report")); // title
        assert!(result.content.contains("quarterly revenue rose sharply")); // article prose
        assert!(!result.content.contains("BEACON_PIXEL_12345")); // script dropped
        assert!(!result.content.contains("SUBSCRIBE_NAV_LINK")); // nav dropped
    }

    #[test]
    fn read_page_rejects_escaping_origins_before_network() {
        // upholds: CAP-2 — same origin discipline as read_url.
        let origin = WebOrigins::one("https://example.com").unwrap();
        let tools = Tools::new().with(ReadPage::new(origin).unwrap());
        let result = tools.dispatch(&ToolCall {
            name: "read_page".to_string(),
            args: json(r#"{"url": "https://evil.example/doc"}"#),
        });
        assert!(result.is_error);
        assert!(result.content.contains("escapes the granted web origins"));
    }

    #[tokio::test]
    async fn read_page_truncates_article_text_not_failing() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_raw(ARTICLE_HTML.as_bytes().to_vec(), "text/html"),
            )
            .mount(&server)
            .await;

        // Tiny output budget; generous input budget.
        let reader =
            ReadPage::with_limits(WebOrigins::one(&server.uri()).unwrap(), 1_000_000, 50).unwrap();
        let result = Tools::new()
            .with(reader)
            .dispatch_async(&ToolCall {
                name: "read_page".to_string(),
                args: json(r#"{"url": "/x"}"#),
            })
            .await
            .render_for_model("read_page");

        assert!(!result.is_error, "got error: {}", result.content);
        // upholds: WIN-1 — the marker names the next window's offset.
        assert!(
            result.content.contains("[chars 0..50 of"),
            "{}",
            result.content
        );
        assert!(result.content.contains("offset=50"), "{}", result.content);
        assert!(result.content.contains("Acme Corporation")); // start of the text kept
    }

    #[tokio::test]
    async fn read_page_rejects_non_html_content_type() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_raw(br#"{"ok":true}"#.to_vec(), "application/json"),
            )
            .mount(&server)
            .await;

        let tools =
            Tools::new().with(ReadPage::new(WebOrigins::one(&server.uri()).unwrap()).unwrap());
        let result = tools
            .dispatch_async(&ToolCall {
                name: "read_page".to_string(),
                args: json(r#"{"url": "/api"}"#),
            })
            .await
            .render_for_model("read_page");

        assert!(result.is_error);
        assert!(result.content.contains("read_url"));
    }

    #[tokio::test]
    async fn read_page_accepts_parameterized_and_mixed_case_html() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_raw(ARTICLE_HTML.as_bytes().to_vec(), "Text/HTML; charset=UTF-8"),
            )
            .mount(&server)
            .await;

        let tools =
            Tools::new().with(ReadPage::new(WebOrigins::one(&server.uri()).unwrap()).unwrap());
        let result = tools
            .dispatch_async(&ToolCall {
                name: "read_page".to_string(),
                args: json(r#"{"url": "/x"}"#),
            })
            .await
            .render_for_model("read_page");

        assert!(!result.is_error, "got error: {}", result.content);
        assert!(result.content.contains("quarterly revenue rose sharply"));
    }

    #[tokio::test]
    async fn read_page_enforces_input_cap() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_raw(ARTICLE_HTML.as_bytes().to_vec(), "text/html"),
            )
            .mount(&server)
            .await;

        // Input cap far below the body size → guarded before extraction.
        let reader =
            ReadPage::with_limits(WebOrigins::one(&server.uri()).unwrap(), 100, 40_000).unwrap();
        let result = Tools::new()
            .with(reader)
            .dispatch_async(&ToolCall {
                name: "read_page".to_string(),
                args: json(r#"{"url": "/x"}"#),
            })
            .await
            .render_for_model("read_page");

        assert!(result.is_error);
        assert!(result.content.contains("read_url"));
    }

    #[tokio::test]
    async fn read_page_reports_non_2xx() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .respond_with(ResponseTemplate::new(404))
            .mount(&server)
            .await;

        let tools =
            Tools::new().with(ReadPage::new(WebOrigins::one(&server.uri()).unwrap()).unwrap());
        let result = tools
            .dispatch_async(&ToolCall {
                name: "read_page".to_string(),
                args: json(r#"{"url": "/missing"}"#),
            })
            .await
            .render_for_model("read_page");

        assert!(result.is_error);
        assert!(result.content.contains("404"));
    }

    #[tokio::test]
    async fn read_page_rejects_non_utf8_body() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .respond_with(
                ResponseTemplate::new(200).set_body_raw(vec![0xff, 0xff, 0xff, 0xfe], "text/html"),
            )
            .mount(&server)
            .await;

        let tools =
            Tools::new().with(ReadPage::new(WebOrigins::one(&server.uri()).unwrap()).unwrap());
        let result = tools
            .dispatch_async(&ToolCall {
                name: "read_page".to_string(),
                args: json(r#"{"url": "/x"}"#),
            })
            .await
            .render_for_model("read_page");

        assert!(result.is_error);
        assert!(result.content.contains("UTF-8"));
        assert!(result.content.contains("read_url"));
    }

    #[tokio::test]
    async fn read_page_empty_extraction_points_at_read_url() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .respond_with(ResponseTemplate::new(200).set_body_raw(
                b"<html><head><title>Empty</title></head><body></body></html>".to_vec(),
                "text/html",
            ))
            .mount(&server)
            .await;

        let tools =
            Tools::new().with(ReadPage::new(WebOrigins::one(&server.uri()).unwrap()).unwrap());
        let result = tools
            .dispatch_async(&ToolCall {
                name: "read_page".to_string(),
                args: json(r#"{"url": "/empty"}"#),
            })
            .await
            .render_for_model("read_page");

        assert!(result.is_error);
        assert!(result.content.contains("read_url"));
    }

    /// Strip a `read_page` window down to its article-text body: drop the
    /// title/URL header and the trailing `[chars …]` marker.
    fn window_body(content: &str) -> &str {
        let body = match content.find("\n\n") {
            Some(i) => &content[i + 2..],
            None => content,
        };
        match body.rfind("\n\n[chars ") {
            Some(i) => &body[..i],
            None => body,
        }
    }

    /// The next-window offset a truncation marker names, if any.
    fn next_offset(content: &str) -> Option<usize> {
        let at = content.rfind("offset=")? + "offset=".len();
        let digits: String = content[at..]
            .chars()
            .take_while(char::is_ascii_digit)
            .collect();
        digits.parse().ok()
    }

    async fn read_window(tools: &Tools, url: &str, offset: usize) -> ToolResult {
        tools
            .dispatch_async(&ToolCall {
                name: "read_page".to_string(),
                args: json(&format!(r#"{{"url": "{url}", "offset": {offset}}}"#)),
            })
            .await
            .render_for_model("read_page")
    }

    #[tokio::test]
    async fn read_page_fetches_once_across_windows() {
        // upholds: FETCH-1 — continuation reads are cache hits; the mock's
        // expect(1) proves the network was touched exactly once.
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/post"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_raw(ARTICLE_HTML.as_bytes().to_vec(), "text/html; charset=utf-8"),
            )
            .expect(1)
            .mount(&server)
            .await;

        let tools = Tools::new().with(
            ReadPage::with_limits(WebOrigins::one(&server.uri()).unwrap(), 1_000_000, 80).unwrap(),
        );
        let mut windows = 0;
        let mut offset = 0;
        loop {
            let result = read_window(&tools, "/post", offset).await;
            assert!(!result.is_error, "window at {offset}: {}", result.content);
            windows += 1;
            match next_offset(&result.content) {
                Some(next) => offset = next,
                None => break,
            }
        }
        assert!(windows >= 3, "expected several windows, got {windows}");
        // MockServer verifies expect(1) on drop.
    }

    #[tokio::test]
    async fn read_page_windows_tile_exactly() {
        // upholds: WIN-1 — windows are adjacent and non-overlapping, and
        // their concatenation reconstructs the whole-article read exactly.
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/post"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_raw(ARTICLE_HTML.as_bytes().to_vec(), "text/html; charset=utf-8"),
            )
            .expect(2) // one fetch for the paged reader, one for the reference
            .mount(&server)
            .await;

        let origin = || WebOrigins::one(&server.uri()).unwrap();
        let paged = Tools::new().with(ReadPage::with_limits(origin(), 1_000_000, 80).unwrap());
        let whole = Tools::new().with(ReadPage::new(origin()).unwrap());

        let mut assembled = String::new();
        let mut offset = 0;
        loop {
            let result = read_window(&paged, "/post", offset).await;
            assert!(!result.is_error, "window at {offset}: {}", result.content);
            assembled.push_str(window_body(&result.content));
            match next_offset(&result.content) {
                Some(next) => {
                    // Adjacent: the marker names exactly where this window ended.
                    assert_eq!(next, offset + window_body(&result.content).chars().count());
                    offset = next;
                }
                None => break,
            }
        }
        let reference = read_window(&whole, "/post", 0).await;
        assert!(!reference.is_error);
        assert_eq!(assembled, window_body(&reference.content));
    }

    #[tokio::test]
    async fn read_page_offset_past_end_is_helpful() {
        // upholds: WIN-1 — an offset past the article is an error naming the
        // length, never a silent empty window.
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/post"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_raw(ARTICLE_HTML.as_bytes().to_vec(), "text/html; charset=utf-8"),
            )
            .mount(&server)
            .await;

        let tools =
            Tools::new().with(ReadPage::new(WebOrigins::one(&server.uri()).unwrap()).unwrap());
        let result = read_window(&tools, "/post", 999_999).await;
        assert!(result.is_error);
        assert!(
            result.content.contains("past the end"),
            "{}",
            result.content
        );
        assert!(result.content.contains("chars"), "{}", result.content);
    }

    #[tokio::test]
    async fn read_page_cache_is_per_url() {
        // upholds: FETCH-1 — the cache keys on the resolved URL: two pages
        // fetch once each, and re-reads of either stay off the network.
        let server = MockServer::start().await;
        for p in ["/a", "/b"] {
            Mock::given(method("GET"))
                .and(path(p))
                .respond_with(
                    ResponseTemplate::new(200)
                        .set_body_raw(ARTICLE_HTML.as_bytes().to_vec(), "text/html; charset=utf-8"),
                )
                .expect(1)
                .mount(&server)
                .await;
        }

        let tools =
            Tools::new().with(ReadPage::new(WebOrigins::one(&server.uri()).unwrap()).unwrap());
        for url in ["/a", "/b", "/a", "/b"] {
            let result = read_window(&tools, url, 0).await;
            assert!(!result.is_error, "{url}: {}", result.content);
        }
        // MockServer verifies each expect(1) on drop.
    }

    /// A sandbox for plot tests, or None (skip) when python3/matplotlib is
    /// unavailable on this machine.
    fn plot_sandbox() -> Option<(tempfile::TempDir, PlotSandbox)> {
        let dir = tempfile::tempdir().unwrap();
        match PlotSandbox::system(dir.path().join("plots")) {
            Ok(sb) => Some((dir, sb)),
            Err(e) => {
                eprintln!("skip: {e}");
                None
            }
        }
    }

    fn plot_call(tools: &Tools, args: &str) -> ToolResult {
        tools.dispatch(&ToolCall {
            name: "plot".to_string(),
            args: json(args),
        })
    }

    #[test]
    fn plot_rejects_code_shaped_and_unknown_specs() {
        // upholds: PLOT-1 — the schema is closed: unknown fields (anything
        // code-shaped), unknown kinds, and dataset/series confusion are
        // typed rejections; nothing is ever executed for them.
        let Some((_tmp, sb)) = plot_sandbox() else {
            return;
        };
        let tools = Tools::new().with(Plot::new(sb));

        let smuggled = plot_call(
            &tools,
            r#"{"kind": "line", "series": [{"y": [1.0]}], "code": "import os"}"#,
        );
        assert!(smuggled.is_error, "unknown field must reject");
        assert!(
            smuggled.content.contains("closed schema"),
            "{}",
            smuggled.content
        );

        let bad_kind = plot_call(&tools, r#"{"kind": "exec", "series": [{"y": [1.0]}]}"#);
        assert!(bad_kind.is_error, "unknown kind must reject");

        let bad_aspect = plot_call(
            &tools,
            r#"{"kind": "line", "series": [{"y": [1.0]}], "aspect": "round"}"#,
        );
        assert!(bad_aspect.is_error, "unknown aspect must reject");

        let neither = plot_call(&tools, r#"{"kind": "line"}"#);
        assert!(neither.is_error, "series or dataset required");

        let unknown_ds = plot_call(&tools, r#"{"kind": "line", "dataset": "nope"}"#);
        assert!(unknown_ds.is_error);
        assert!(
            unknown_ds.content.contains("unknown dataset"),
            "{}",
            unknown_ds.content
        );

        let mismatch = plot_call(
            &tools,
            r#"{"kind": "line", "series": [{"x": [1.0, 2.0], "y": [1.0]}]}"#,
        );
        assert!(mismatch.is_error, "x/y length mismatch must reject");

        // Code smuggled as data — the live incident: a comprehension where
        // numbers belong. The rejection must teach the expr channel.
        let smuggled_y = plot_call(
            &tools,
            r#"{"kind": "line", "series": [{"y": ["Math.sin(x) for x in range(628)"]}]}"#,
        );
        assert!(smuggled_y.is_error);
        assert!(
            smuggled_y.content.contains("expr"),
            "the rejection teaches the legal move: {}",
            smuggled_y.content
        );
    }

    #[test]
    fn plot_expr_is_a_closed_grammar_not_code() {
        // upholds: PLOT-1 — expr series are parsed against the closed
        // grammar and sampled host-side; anything code-shaped, rangeless,
        // over-sampled, or non-finite is a typed rejection that teaches.
        let Some((_tmp, sb)) = plot_sandbox() else {
            return;
        };
        let tools = Tools::new().with(Plot::new(sb));

        let code = plot_call(
            &tools,
            r#"{"kind": "line", "series": [{"expr": "__import__('os')", "from": 0, "to": 1}]}"#,
        );
        assert!(code.is_error);
        assert!(
            code.content.contains("sin cos tan"),
            "the rejection names the alphabet: {}",
            code.content
        );

        let both = plot_call(
            &tools,
            r#"{"kind": "line", "series": [{"expr": "x", "y": [1.0], "from": 0, "to": 1}]}"#,
        );
        assert!(both.is_error, "y and expr are exclusive");

        let rangeless = plot_call(&tools, r#"{"kind": "line", "series": [{"expr": "x"}]}"#);
        assert!(rangeless.is_error);
        assert!(
            rangeless.content.contains("from and to"),
            "{}",
            rangeless.content
        );

        let backwards = plot_call(
            &tools,
            r#"{"kind": "line", "series": [{"expr": "x", "from": 1, "to": 0}]}"#,
        );
        assert!(backwards.is_error, "from must precede to");

        let oversampled = plot_call(
            &tools,
            r#"{"kind": "line", "series": [{"expr": "x", "from": 0, "to": 1, "samples": 4000000}]}"#,
        );
        assert!(oversampled.is_error, "samples over the points cap reject");

        let asymptote = plot_call(
            &tools,
            r#"{"kind": "line", "series": [{"expr": "ln(x)", "from": 0, "to": 1}]}"#,
        );
        assert!(asymptote.is_error);
        assert!(
            asymptote.content.contains("non-finite"),
            "domain edges teach a range fix: {}",
            asymptote.content
        );

        // Range bounds are constants in the same grammar: x is out of scope
        // there, and a bound that fails the grammar teaches the alphabet.
        let x_bound = plot_call(
            &tools,
            r#"{"kind": "line", "series": [{"expr": "sin(x)", "from": 0, "to": "x + 1"}]}"#,
        );
        assert!(x_bound.is_error);
        assert!(x_bound.content.contains("constant"), "{}", x_bound.content);

        let bad_bound = plot_call(
            &tools,
            r#"{"kind": "line", "series": [{"expr": "sin(x)", "from": 0, "to": "range(10)"}]}"#,
        );
        assert!(bad_bound.is_error);
        assert!(
            bad_bound.content.contains("plot expr"),
            "{}",
            bad_bound.content
        );
    }

    #[test]
    fn plot_expr_series_render_smooth_and_deterministic() {
        // upholds: PLOT-1 + PLOT-3 — symbolic intent renders through the
        // legal channel (sampled in Rust, literal arrays to the generator),
        // lands in the sandbox, and re-renders byte-identical.
        let Some((_tmp, sb)) = plot_sandbox() else {
            return;
        };
        let root = _tmp.path().join("plots");
        let tools = Tools::new().with(Plot::new(sb));
        // The live incident, verbatim: a symbolic bound ("9 * pi") — legal.
        // aspect exercises the generator's equal-aspect path end to end.
        let spec = r#"{"kind": "line", "title": "sine", "aspect": "equal",
            "series": [{"expr": "sin(x)", "from": 0, "to": "9 * pi", "samples": 64}]}"#;

        let first = plot_call(&tools, spec);
        assert!(!first.is_error, "{}", first.content);
        let path = first
            .content
            .split_whitespace()
            .nth(1)
            .expect("wrote <path>");
        assert!(
            std::path::Path::new(path).starts_with(&root),
            "PLOT-2: {path} inside {root:?}"
        );
        let bytes1 = std::fs::read(path).unwrap();

        let second = plot_call(&tools, spec);
        assert!(!second.is_error);
        assert!(second.content.contains(path), "same expr, same artifact");
        assert_eq!(bytes1, std::fs::read(path).unwrap(), "PLOT-3 holds");
    }

    #[test]
    fn plot_renders_confined_and_deterministic() {
        // upholds: PLOT-2 + PLOT-3 — the artifact lands inside the sandbox
        // (at a spec-hash name the model never chose), and the same spec
        // re-renders byte-identical.
        let Some((_tmp, sb)) = plot_sandbox() else {
            return;
        };
        let root = _tmp.path().join("plots");
        let tools = Tools::new().with(Plot::new(sb));
        let spec =
            r#"{"kind": "line", "title": "t", "series": [{"name": "s", "y": [1.0, 3.0, 2.0]}]}"#;

        let first = plot_call(&tools, spec);
        assert!(!first.is_error, "{}", first.content);
        let path = first
            .content
            .split_whitespace()
            .nth(1)
            .expect("wrote <path>");
        assert!(
            std::path::Path::new(path).starts_with(&root),
            "PLOT-2: {path} is inside {root:?}"
        );
        let bytes1 = std::fs::read(path).unwrap();
        assert!(bytes1.len() > 1000, "a real PNG");

        let second = plot_call(&tools, spec);
        assert!(!second.is_error);
        assert!(second.content.contains(path), "same spec, same artifact");
        let bytes2 = std::fs::read(path).unwrap();
        assert_eq!(bytes1, bytes2, "PLOT-3: byte-identical re-render");
    }

    #[test]
    fn plot_datasets_are_host_supplied() {
        // upholds: PLOT-1 — a registered dataset renders by name (the
        // program supplied the numbers), and the spec advertises it.
        let Some((_tmp, sb)) = plot_sandbox() else {
            return;
        };
        let tools = Tools::new().with(Plot::new(sb).with_dataset(
            "curve",
            vec![PlotSeries {
                name: Some("equity".into()),
                y: Some(vec![1.0, 1.1, 1.3, 1.2]),
                ..PlotSeries::default()
            }],
        ));
        assert!(
            tools.specs()[0].description.contains("curve"),
            "the spec names registered datasets"
        );
        let result = plot_call(&tools, r#"{"kind": "line", "dataset": "curve"}"#);
        assert!(!result.is_error, "{}", result.content);
        assert!(result.content.contains("wrote "), "{}", result.content);
    }

    #[test]
    fn empty_specs_render_no_tool_instructions() {
        // upholds: CAP-3a — a codec renders zero tool-calling instructions
        // for zero advertised tools: an agent session with nothing granted
        // reads as plain chat; the prompt never claims absent authority.
        assert_eq!(QwenToolCall.render_system(&[]), "");
        assert_eq!(JsonToolCall.render_system(&[]), "");
        let spec = ToolSpec {
            name: "plot".into(),
            description: "d".into(),
            params: serde_json::json!({}),
        };
        assert!(!QwenToolCall.render_system(&[spec]).is_empty());
    }

    #[test]
    fn web_tool_specs_reflect_live_authority() {
        // upholds: CAP-3a — an empty origin set hides the web tools from the
        // advertised specs; a grant surfaces them with the origin enumerated
        // in the description; revoking back to empty hides them again.
        let origins = WebOrigins::new();
        let tools = Tools::new()
            .with(ReadPage::new(origins.clone()).unwrap())
            .with(ReadUrl::new(origins.clone()).unwrap());
        assert!(
            tools.specs().is_empty(),
            "no grants → no web tools advertised"
        );

        origins.grant("https://a.example").unwrap();
        let specs = tools.specs();
        assert_eq!(specs.len(), 2);
        assert!(
            specs
                .iter()
                .all(|s| s.description.contains("https://a.example")),
            "the spec names the granted origin"
        );

        origins.revoke("https://a.example").unwrap();
        assert!(tools.specs().is_empty(), "revoked back to hidden");
    }

    #[test]
    fn page_cache_evicts_oldest() {
        // upholds: FETCH-1's bound — the cache is FIFO-capped, so a session
        // reading many pages cannot grow memory without limit.
        let mut cache = PageCache::default();
        for i in 0..=READ_PAGE_CACHE_PAGES {
            cache.insert(
                format!("https://example.com/{i}"),
                Arc::new(CachedPage {
                    title: String::new(),
                    text: format!("page {i}"),
                    images: Vec::new(),
                }),
            );
        }
        assert!(
            cache.get("https://example.com/0").is_none(),
            "oldest evicted"
        );
        assert!(
            cache
                .get(&format!("https://example.com/{READ_PAGE_CACHE_PAGES}"))
                .is_some(),
            "newest present"
        );
        assert!(cache.order.len() <= READ_PAGE_CACHE_PAGES);
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
        // upholds: OBS-2 / OBS-4 — tool telemetry exposes bounded dimensions,
        // never model-supplied args or tool output payloads. Runtime subscriber
        // capture is intentionally not tested here: tracing callsite interest is
        // global and brittle under parallel tests.
        const TOOL_CALL_TRACE_FIELDS: &[&str] = &["call_id", "tool"];
        const TOOL_FINISHED_TRACE_FIELDS: &[&str] = &["call_id", "tool", "outcome"];

        assert_eq!(TOOL_CALL_TRACE_FIELDS, &["call_id", "tool"]);
        assert_eq!(TOOL_FINISHED_TRACE_FIELDS, &["call_id", "tool", "outcome"]);
        assert!(!TOOL_CALL_TRACE_FIELDS.contains(&"args"));
        assert!(!TOOL_FINISHED_TRACE_FIELDS.contains(&"args"));
        assert!(!TOOL_FINISHED_TRACE_FIELDS.contains(&"content"));
        assert_eq!(ToolOutcome::success("secret payload").kind(), "success");
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
