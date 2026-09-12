# 1. Transcript and templates

This chapter begins with Yatima's common conversation data and follows it into a model-native prompt. It reads [`transcript.rs`](https://github.com/shayne-fletcher/yatima/blob/402d34a26bedd9d254e76a51be8c034961e28db1/lib/src/transcript.rs) and [`template.rs`](https://github.com/shayne-fletcher/yatima/blob/402d34a26bedd9d254e76a51be8c034961e28db1/lib/src/template.rs) at commit `402d34a26bedd9d254e76a51be8c034961e28db1`.

The central fact is simple: conversation memory is structured data. A prompt template renders that data into the syntax expected by one model family and interprets the reply using the matching response protocol. The model backend itself does not remember earlier calls.

## `transcript.rs`: conversation data

The private `transcript` module defines the public `Role` enum, `Turn` enum, and `ToolArguments` struct. `lib.rs` re-exports all three from the `yatima_lib` crate root.

```rust
pub enum Role {
    System,
    User,
    Assistant,
    Tool,
}

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
```

`Role` is the smaller vocabulary needed by code that only asks who supplied a turn. `Turn` retains the data needed to render that turn correctly:

- `System`, `User`, and `Assistant` carry ordinary text.
- `AssistantToolCall` carries a tool name and structured arguments.
- `ToolResult` carries the corresponding name, returned text, and success or error status.

This representation prevents several invalid states. A tool invocation cannot exist without a name and arguments, and a tool result cannot lose the name needed by the next prompt. Templates receive these meanings directly; they do not recover them by parsing display strings such as `[read_file ok]`.

The `Turn::role` method maps both assistant variants to `Role::Assistant` and maps `ToolResult` to `Role::Tool`. The `Turn::content` method returns text for ordinary turns and tool results, but returns `None` for a tool call because its payload is structured rather than a single string.

`ToolArguments` owns an ordered list of unique name/value pairs. `try_from_pairs` rejects duplicate parameter names. `from_json_object` converts a JSON object into the same representation, while `to_json_object` converts it back for tool dispatch. Preserving order matters to Muse's ATEM renderer even though dispatch sees an equivalent JSON object.

There is no `Transcript` type here. `ChatSession` and `Agent` each own a `Vec<Turn>` and decide when an exchange is committed, rolled back, or discarded. The transcript module defines valid entries; it does not impose session policy or require roles to alternate.

An `Assistant` turn contains answer text only. Reasoning and protocol framing are removed before construction. `AssistantToolCall` and `ToolResult` are temporary working turns during an agent run; persistent agent history contains completed user and assistant exchanges.

## `template.rs`: both directions of a model protocol

Instruction-tuned models expect particular control tokens, role names, and response framing. The public `PromptTemplate` trait keeps those choices together:

```rust
pub trait PromptTemplate {
    fn render(&self, turns: &[Turn]) -> String;

    fn compose_system(&self, system: &str, tool_instructions: &str) -> String;
    fn classifier(&self) -> ResponseClassifier;
    fn interpret_response(&self, raw: &str) -> Reasoned;
}
```

Only `render` is required. The other methods have defaults suitable for ordinary marker-based formats.

- `render` turns the complete transcript into a fresh prompt ending at the next assistant cue.
- `compose_system` combines the caller's system instruction with model-facing tool declarations.
- `classifier` constructs the state machine used to classify streamed fragments as reasoning, answer, or tool protocol material.
- `interpret_response` performs the final whole-response interpretation used before transcript commit.

Keeping these methods on one trait prevents prompt selection and response interpretation from drifting apart. A Muse prompt must be read as ATEM; a pre-seeded reasoning prompt must begin its response classifier inside reasoning.

The blanket implementation for `Box<T>` forwards every method:

```rust
impl<T: PromptTemplate + ?Sized> PromptTemplate for Box<T> {
    // render, compose_system, classifier, and interpret_response all delegate
}
```

This lets a runtime-selected `Box<dyn PromptTemplate>` satisfy the generic template parameter of `ChatSession` or `Agent`. Forwarding every method is important: using a default for a boxed Muse template would silently replace its ATEM interpreter with the marker splitter.

The file defines eight built-in template structs:

- `PlainTemplate`
- `ChatMlTemplate`
- `ChatMlThinkTemplate`
- `GemmaTemplate`
- `MistralTemplate`
- `GlmTemplate`
- `DeepSeekTemplate`
- `MuseGlimmerTemplate`

Downstream crates may implement `PromptTemplate`, so this is the built-in inventory rather than a closed set.

## Marker-based formats

`ChatMlTemplate` renders Qwen-style blocks and a final assistant cue. Its reasoning variant, `ChatMlThinkTemplate`, adds `<think>` to that cue. Because the prompt has already opened reasoning, the latter overrides both `classifier` and `interpret_response` with their seeded forms.

`GemmaTemplate` and `MistralTemplate` fold system text into the first user turn because their prompt formats have no separate system role. `GlmTemplate` emits its required `[gMASK]<sop>` prefix. `DeepSeekTemplate` emits its beginning token and also pre-seeds `<think>`.

The renderer sees structured tool turns. For example, a Qwen working transcript can be built without embedding protocol strings in `Turn`:

```rust
let arguments = ToolArguments::try_from_pairs([(
    "path".to_string(),
    serde_json::Value::String("README.md".to_string()),
)])?;

let turns = vec![
    Turn::user("Read README.md and name the project."),
    Turn::assistant_tool_call("read_file", arguments),
    Turn::tool_result("read_file", "# Yatima", false),
];

let prompt = ChatMlTemplate.render(&turns);
```

`ChatMlTemplate` renders the invocation inside `<tool_call>` and the result inside `<tool_response>`. Other templates can render the same typed turns differently.

## Muse Glimmer and ATEM

`MuseGlimmerTemplate` has two fields: a `ReasoningStrength` enum and an optional current date. Unlike the fieldless templates, it carries runtime configuration used during rendering.

Muse uses addressed messages rather than an inline `<think>` span:

```text
<|start|>assistant to=self<|message|>reasoning<|eom|>
<|start|>assistant to=user<|message|>answer<|eot|>
```

The bare assistant cue is already present at the end of the prompt, so the first generated header continues that cue. `to=self` carries reasoning, `to=user` carries the surfaced answer, and another recipient names a tool.

`MuseGlimmerTemplate::render` writes ordinary transcript turns, assistant tool invocations, and tool results in ATEM syntax. It also adds the model's reasoning-strength directive and recipient list to the system block. `compose_system` places tool declarations in that same native system format.

For the reverse direction, `classifier` returns `ResponseClassifier::Atem` and `interpret_response` runs the same `AtemInterpreter` over the completed reply. Chapter 2 opens that interpreter. This symmetry is the practical contract: the type that writes a protocol also selects the code that reads it.

## Selection and failure

The `ChatFormat` enum, introduced in detail later, maps configured formats to these template implementations. A model profile pins a known model to its expected format. Muse's profile additionally supplies its model artifact, generation recipe, and managed `llama-server` requirements.

Rendering returns `String`, not `Result<String>`. Selecting the wrong format therefore does not fail locally; it sends the model unfamiliar syntax and usually produces poor or malformed output. The profile and frontend resolvers reject contradictory format overrides before that can happen for a pinned profile.

Two template rules matter when adding a format. `TMPL-1` requires beginning-of-sequence material to be emitted exactly once between tokenizer and template. `TMPL-2` requires formats without a system role to fold that text into the first user turn.

## Maintainer checkpoint

- `Turn` stores role-specific meanings directly. Do not encode tool identity or success inside display text.
- A session owns its `Vec<Turn>`; the transcript module owns no conversation.
- `PromptTemplate` owns prompt rendering, system/tool composition, streaming classifier selection, and final response interpretation for one protocol.
- Add a marker-based model by implementing its renderer and selecting the correct ordinary or seeded marker classifier.
- Add an addressed-message model by giving it a protocol interpreter rather than pretending its messages are another reasoning-marker dialect.
- Preserve `REASON-1`: reasoning, framing, and tool protocol material must not enter the assistant answer committed for the next prompt.
