# P. The map

Yatima runs local language models behind one conversation and tool interface. A caller can embed `yatima-lib` directly, use the CLI, or send turns through the shared host from the terminal UI, desktop GUI, or browser. This chapter identifies the principal types and shows where a turn goes.

The chapter describes commit `402d34a26bedd9d254e76a51be8c034961e28db1`. Later chapters open the implementation one file at a time.

## A frontend turn

The `yatima-cli` crate can call `yatima-lib` directly. The TUI, GUI, server, and browser use the `yatima-host` crate. A frontend sends a `HostRequest` enum value and rebuilds its display from `HostEvent` enum values; it does not own a second model session.

In plain English, a hosted turn follows this path:

```text
frontend submits text
    -> host backend thread receives it
    -> ChatSession or Agent renders the conversation
    -> the selected Completer produces a streamed reply
    -> an optional tool call starts another model step
    -> host events update the frontend
```

Each arrow means that the step on the left passes work or data to the step on the right. The host backend is either Candle's in-process `Engine` struct or a supervised `LlamaServer` child process. Both implement the `Completer` trait, so chat and agent code do not contain separate conversation loops for the two inference systems.

The relevant declarations are:

```rust
pub struct ChatSession<'a, C: Completer, T: PromptTemplate> {
    completer: &'a mut C,
    template: T,
    // options and conversation state
}

pub struct Agent<'a, C: Completer, K: ToolCallCodec, T: PromptTemplate> {
    completer: &'a mut C,
    tools: &'a Tools,
    codec: K,
    template: T,
    // options and conversation state
}
```

- `C: Completer` means that `C` may be any type implementing the `Completer` trait. The host uses its private `HostBackend` enum for `C`; that enum delegates to either `Engine` or `LlamaServer`.
- Both structs store `&'a mut C`, so they borrow the backend rather than own it. The lifetime `'a` prevents the borrow from outliving the backend.
- Both own a `T: PromptTemplate`, which renders prompts and interprets replies in one model protocol.
- The `Agent` struct additionally borrows the allowed `Tools` and owns a `K: ToolCallCodec` that turns model protocol messages into typed tool calls.

`ChatSession` and `Agent` are siblings; neither contains the other. Both own conversation history and re-render it because a `Completer` call has no implicit memory of earlier prompts.

## Workspace crates

| Crate | Job |
|---|---|
| `yatima-lib` | Model backends, prompt and response protocols, transcript state, capabilities, tools, and agent execution. |
| `yatima-protocol` | Serializable request and event meanings shared with the browser. |
| `yatima-text` | Pure output formatting shared by views. |
| `yatima-host` | The backend thread, authoritative conversation, capability changes, and request/event/control planes. |
| `yatima-cli` | Command parsing and direct library use. |
| `yatima-tui`, `yatima-gui` | Native views over the host. |
| `yatima-serve` | The WebSocket bridge to the host. |
| `yatima-web` | The WASM browser view and event mirror. |

`yatima-protocol` knows nothing about native model types. The host converts library events into protocol events, allowing `yatima-web` to compile for WASM without Candle or `llama-server` dependencies.

## Inside `yatima-lib`

Most names in this table are private Rust modules declared in `lib/src/lib.rs`:

| Read first | Modules | Purpose |
|---|---|---|
| Foundations | `cancel`, `expr`, `reasoning`, `runtime`, `token_output_stream`, `transcript` | Shared data, parsing, cancellation, response classification, and runtime support. |
| Model interface and implementations | `completer`, `engine`, `backend`, `template` | Define the common completion interface, run Candle or `llama-server`, and speak model-native protocols. |
| Configuration | `host` under `lib/src/host/` | Define formats, profiles, model sources, and generation recipes. This is not the `yatima-host` crate. |
| Conversations and actions | `capability`, `tool`, `chat`, `agent` | Define authority, external actions, session state, and the model/tool loop. |

Inside the crate, these modules are intended to depend in that direction. Rust does not enforce the ordering between modules in one crate, so `LAYER-1` is a review rule: shared types belong in the lowest module that needs to understand them. The `Turn` enum, for example, lives in `transcript` because templates, chat, and agents all use it.

## Three ways to ask a model

- `Engine::generate` performs raw generation. It has no transcript or prompt template.
- `ChatSession::turn` renders stored conversation history, asks one `Completer` for a response, interprets it, and commits a clean answer.
- `Agent::run` starts from history but may interpret a tool call, execute it under explicit capabilities, and ask the model again before committing a final answer.

The normal paths can be summarized as follows:

```text
generation: prompt -> generated text
chat:       transcript -> rendered prompt -> reply -> updated transcript
agent:      transcript -> reply -> zero or more tool rounds -> final answer
```

These are process sketches, not Rust type signatures. Each arrow means “is followed by.”

## The common model interface

The `Completer` trait is the boundary used by both `ChatSession` and `Agent`. Its asynchronous methods accept a rendered prompt, generation options, and stop strings, then return a `Completion` struct or stream classified fragments.

`Engine` implements the trait using in-process Candle inference. `LlamaServerCompleter` implements it over loopback HTTP, while `LlamaServer` owns and supervises the corresponding child process and delegates completion to that adapter. A managed Muse profile resolves one exact GGUF, verifies its digest and server properties, starts the child, and reports the verified identity through the host.

The backend decides how inference runs; conversation code still owns prompts, transcripts, reasoning separation, tool semantics, and commit policy.

## External actions

A concrete tool implements the `Tool` trait. Its `call` method receives model-supplied JSON arguments and a dispatcher-created `ToolCtx` struct. The tool instance itself holds authority such as an allowed filesystem root or set of web origins. The model supplies arguments, not permission.

The `Tools` struct stores the actions available to one agent. The `ToolCallCodec` trait interprets model-specific request syntax. Qwen uses ChatML tool-call markup; Muse uses addressed ATEM messages. Both become the same typed tool invocation before dispatch.

## Who owns the state

| Program part | State it keeps |
|---|---|
| Direct library caller | Its backend, session or agent, and supplied tools. |
| `yatima-host` backend thread | One backend, the authoritative conversation, active capabilities, and current turn. |
| `yatima-serve` | The WebSocket connection and temporary access to host events, not another conversation. |
| Frontend | Input and display state reconstructed from host events. |

The host retains a separate owner handle so shutdown can cancel work, join its thread, and prove that a managed `llama-server` child was reaped. Frontends may disappear or reconnect without becoming owners of model or conversation state.

## Maintainer checkpoint

| Change | Start here |
|---|---|
| Candle model loading or token generation | `lib/src/engine.rs` |
| Common completion interface | `lib/src/completer.rs` |
| Managed `llama-server` transport or lifecycle | `lib/src/backend/llama_server.rs` |
| Transcript vocabulary or model protocol | `lib/src/transcript.rs`, `lib/src/template.rs`, and `lib/src/reasoning.rs` |
| Model profiles and sources | `lib/src/host/` |
| Tool permission or implementation | `lib/src/capability.rs` and `lib/src/tool.rs` |
| Chat or agent commit policy | `lib/src/chat.rs` and `lib/src/agent.rs` |
| Hosted ownership and event projection | `host/src/lib.rs` |
| Wire meaning | `protocol/src/lib.rs` |
| View behavior | the relevant frontend, or `yatima-text` for shared formatting |

The next chapter begins with the common transcript vocabulary and the templates that speak each model's prompt and response protocol.
