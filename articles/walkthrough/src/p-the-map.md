# P. The map

Yatima can be embedded as a Rust library or run through its CLI, terminal UI, desktop GUI, or browser view. This chapter shows where a request goes, which part owns the conversation, where model and tool work happens, and where a maintainer should begin common changes.

The chapter describes commit `d724b9ed2f07b709dff29597ff91f24aff5ac8ad`. Later chapters open the implementation one part at a time.

## A frontend turn

The CLI calls `yatima-lib` directly. Turns from the TUI, GUI, and browser all go through `yatima-host`. The `HostRequest` and `HostEvent` enums carry a turn between those frontends and the host. The host owns an instance of the `Engine` struct and handles each turn with either the `ChatSession` struct or the `Agent` struct; both ask for model output through the `Completer` trait. Each arrow below means that the value or request above passes to the step below.

```text
HostRequest::Submit { turn_id, text }
    -> yatima-host's engine thread
    -> ChatSession or Agent borrows the Engine
    -> Engine's Completer implementation runs complete_streaming
    -> optional tool call and another model step
    -> HostEvent::Fragment / ToolNote / Done
    -> frontend updates its display
```

The relevant parts of their declarations are:

```rust
pub struct ChatSession<'a, C: Completer, T: PromptTemplate> {
    completer: &'a mut C,
    template: T,
    // conversation state
}

pub struct Agent<'a, C: Completer, K: ToolCallCodec, T: PromptTemplate> {
    completer: &'a mut C,
    tools: &'a Tools,
    codec: K,
    template: T,
    // conversation state
}
```

- `C: Completer` means that `C` may be any type implementing the `Completer` trait.
- In the host, `C` is `Engine`; the `Engine` struct implements `Completer`.
- Both structs store `&'a mut C`, so they borrow the engine rather than own it. The lifetime `'a` prevents either borrow from outliving the engine.
- Both own an implementation of the `PromptTemplate` trait as `T`. The `Agent` struct additionally borrows a `Tools` struct and owns an implementation of the `ToolCallCodec` trait as `K`.
- The `Agent` struct does not implement `Completer` and cannot fill `C`; it is one of the enclosing structs parameterized by `C`.

The host thread therefore owns the engine instance and the conversation history. A frontend owns only its input and a display built from `HostEvent` values. It may show partial output while a turn is running, but it does not keep a second model session.

A program embedding `yatima-lib` directly takes on that ownership itself. It creates an `Engine` instance, then uses it directly or passes a mutable reference to one of two session structs: `ChatSession` or `Agent`.

## Workspace crates

The workspace is divided by responsibility:

| Crate | Direct Yatima dependencies | Job |
|---|---|---|
| `yatima-lib` | none | Load models, generate text, keep chat history, and run tools and agents. |
| `yatima-protocol` | none | Define serializable requests and events shared with the browser. |
| `yatima-text` | none | Prettify output without owning model or session state. |
| `yatima-host` | `yatima-lib`, `yatima-protocol` | Own the engine thread and conversation used by frontends. |
| `yatima-cli` | `yatima-lib` | Provide the direct command-line embedding. |
| `yatima-tui`, `yatima-gui` | `yatima-host`, `yatima-lib`, `yatima-text` | Maintain native user-interface state and render host events. |
| `yatima-serve` | `yatima-host`, `yatima-lib`, `yatima-protocol` | Carry host requests and events over WebSocket. |
| `yatima-web` | `yatima-protocol`, `yatima-text` | Render the browser view. |

`yatima-protocol` deliberately knows nothing about `yatima-lib`. The host converts between library values and wire messages, so the browser can use the protocol without compiling Candle or native model code to WASM.

`yatima-web` is excluded from the native Cargo workspace and built separately for the `wasm32` target. The TUI, GUI, and server depend on `yatima-lib` to construct configuration values such as the `GenOpts` and `ModelProfile` structs and to parse origins, but they do not call model decoding directly. Frontend turns still pass through the host protocol.

## Inside `yatima-lib`

Most names below are Rust modules declared in `lib/src/lib.rs`. Read them in this broad order:

| Order | Modules | Purpose |
|---|---|---|
| 1. Foundations and helpers | `cancel`, `expr`, `reasoning`, `runtime`, `token_output_stream`, `transcript` | Shared data, pure parsing, cancellation, and runtime support. |
| 2. Model code | `engine`, `completer`, `template` | Load models, produce completions, and render prompts. |
| 3a. Configuration | `host` (`lib/src/host/`) | Define model formats, profiles, and sources. This is not the separate `yatima-host` crate. |
| 3b. Conversations and actions | `capability`, `tool`, `chat`, `agent` | Define permissions, external actions, and conversation state. |

The `ModelId` struct and the `models_root` and `model_dir` functions are defined in the crate root rather than in separate modules; they belong with the foundations for dependency purposes. The CLI is a separate crate, and examples are Cargo example targets. Both are callers of `yatima-lib`, not modules inside it.

Model code may import foundations. The configuration and conversation/action groups may import model code and foundations, but they should not import one another. Lower groups must not import higher groups. Rust does not enforce this ordering between modules in one crate, so the [`LAYER-1` law](https://github.com/shayne-fletcher/yatima/blob/d724b9ed2f07b709dff29597ff91f24aff5ac8ad/notes/design.md#L112) makes it a review check.

Put a shared type in the lowest-level module that needs to understand it. The `Role` enum and `Turn` struct, for example, live in `transcript` because templates, chat, and agents all use them. Within the conversation-and-tool group, the `capability` module defines permissions and the `tool` module uses them. The `chat` module defines the tool-free `ChatSession` struct; the `agent` module defines the `Agent` struct that combines model completion with tools.

## Three ways to use a model

- The `Engine::generate` method sends a raw prompt to the model, passes generated text fragments to a callback, and returns a `Generation` struct containing the token count and stop reason. It has no conversation history or prompt template.
- The `ChatSession::turn` method renders the stored conversation as a prompt, obtains one completion, and adds the answer to its history.
- The `Agent::run` method also starts from stored conversation history, but it may run a tool and ask the model again before it reaches a final answer.

The same normal paths can be compared compactly:

```text
direct generation: Prompt -> GeneratedText
chat session:      Transcript -> rendered Prompt -> Completion -> updated Transcript
agent run:         Transcript -> (Completion -> ToolCall -> ToolOutcome)* -> Completion -> final answer -> updated Transcript
```

This is a process sketch, not a Rust or Haskell type signature. An arrow means "the step on the left produces the step on the right," and `*` means that the grouped tool round may happen zero or more times. `Completion` and `ToolCall` are structs from the source, and `ToolOutcome` is an enum; `Prompt`, `GeneratedText`, and `Transcript` are descriptive labels, not Rust types. Cancellation, errors, generation metadata, and the agent's step limit are left out of this normal-path summary.

The `ChatSession` and `Agent` structs are siblings built on the same `Completer` and `PromptTemplate` traits and `Turn` struct; neither is built from the other. Both re-render their stored history because the `Engine` struct does not remember previous prompts.

## Model work: the `Completer` trait

The [`Completer`](https://github.com/shayne-fletcher/yatima/blob/d724b9ed2f07b709dff29597ff91f24aff5ac8ad/lib/src/completer.rs#L88) trait is how chat and agent code ask for model output. Its core method takes a prompt, a `GenOpts` struct, and stop strings, then returns a `Completion` struct:

```rust
async fn complete(
    &mut self,
    prompt: &str,
    opts: &GenOpts,
    stops: &[String],
) -> Result<Completion>;
```

The `complete_streaming` trait method adds a token callback and cancellation handle to the same operation. Callers await either method directly, one model step at a time.

The implementation decides how the work runs. The current `Engine` implementation performs Candle's synchronous, mutable decoding on Yatima's blocking path. The returned `Completion` struct contains the generated text and the reason generation stopped.

This is where a new model backend begins. The `ChatSession` and `Agent` structs are already generic over the `Completer` trait; `yatima-host` still constructs an `Engine` instance, so making a new backend available to frontends also requires changing host ownership and configuration.

## External actions: the `Tool` trait

A tool implements the [`Tool`](https://github.com/shayne-fletcher/yatima/blob/d724b9ed2f07b709dff29597ff91f24aff5ac8ad/lib/src/tool.rs#L289) trait. Its `call` method receives model arguments in the `serde_json::Value` type and a dispatcher-created `ToolCtx` struct, then returns text or an error:

```rust
async fn call(&self, args: Value, ctx: ToolCtx) -> Result<String>;
```

The tool context carries the call id, cancellation token, and event sender. The tool itself holds any directory, web-origin, plotting, or notification permission it needs. A model cannot grant itself access merely by asking for it.

The `Tools` struct stores the set an agent is allowed to call. It rejects unknown names, starts registered calls as tasks, and records each result as a variant of the `ToolOutcome` enum: `Success`, `Rejected`, `Failed`, `Cancelled`, or `TimedOut`. The runtime uses that full enum; its model-facing form is the smaller `ToolResult` struct.

This is where a new external action begins: define its permission type in the `capability` module, implement the `Tool` trait, and add an instance to the `Tools` struct supplied to an agent.

## Who owns the state

| Program part | State it keeps |
|---|---|
| Direct library caller | Its `Engine`, `ChatSession` or `Agent`, and supplied tools. |
| `yatima-host` engine thread | The engine, one conversation, active tool permissions, and the current turn. |
| `yatima-serve` | The WebSocket connection and temporary ownership of the host event receiver, not a second conversation. |
| Frontend | Input, scroll and selection state, and a display reconstructed from host events. |

Model calls use the mutable engine one at a time. `ChatSession` or `Agent` owns the conversation kept for later prompts. Tool calls have a shorter task lifecycle. A frontend may disappear and rebuild its display without becoming the owner of either the engine or the conversation.

## Maintainer checkpoint

| Change | Start here |
|---|---|
| Model loading, sampling, or token generation | [`engine.rs`](https://github.com/shayne-fletcher/yatima/blob/d724b9ed2f07b709dff29597ff91f24aff5ac8ad/lib/src/engine.rs) |
| The common model interface or another backend | [`completer.rs`](https://github.com/shayne-fletcher/yatima/blob/d724b9ed2f07b709dff29597ff91f24aff5ac8ad/lib/src/completer.rs); then `yatima-host` if frontends must select it |
| Transcript vocabulary or prompt rendering | [`transcript.rs`](https://github.com/shayne-fletcher/yatima/blob/d724b9ed2f07b709dff29597ff91f24aff5ac8ad/lib/src/transcript.rs) and [`template.rs`](https://github.com/shayne-fletcher/yatima/blob/d724b9ed2f07b709dff29597ff91f24aff5ac8ad/lib/src/template.rs) |
| Tool permissions | [`capability.rs`](https://github.com/shayne-fletcher/yatima/blob/d724b9ed2f07b709dff29597ff91f24aff5ac8ad/lib/src/capability.rs) |
| Tool protocol or a concrete external action | [`tool.rs`](https://github.com/shayne-fletcher/yatima/blob/d724b9ed2f07b709dff29597ff91f24aff5ac8ad/lib/src/tool.rs) |
| Chat history and commit policy | [`chat.rs`](https://github.com/shayne-fletcher/yatima/blob/d724b9ed2f07b709dff29597ff91f24aff5ac8ad/lib/src/chat.rs) |
| The model/tool loop | [`agent.rs`](https://github.com/shayne-fletcher/yatima/blob/d724b9ed2f07b709dff29597ff91f24aff5ac8ad/lib/src/agent.rs) |
| Frontend session ownership, cancellation, or live grants | [`host/src/lib.rs`](https://github.com/shayne-fletcher/yatima/blob/d724b9ed2f07b709dff29597ff91f24aff5ac8ad/host/src/lib.rs) |
| Serializable request or event meaning | [`protocol/src/lib.rs`](https://github.com/shayne-fletcher/yatima/blob/d724b9ed2f07b709dff29597ff91f24aff5ac8ad/protocol/src/lib.rs), with conversion in the host |
| Display behavior | the relevant frontend, or `yatima-text` for shared pure formatting |

The next chapter starts with the lowest shared vocabulary: `Role`, `Turn`, and the templates that turn a transcript into a model prompt.
