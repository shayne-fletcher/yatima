# 1. Transcript and templates

This chapter begins with the smallest conversation types in Yatima and follows them through prompt rendering. It reads [`transcript.rs`](https://github.com/shayne-fletcher/yatima/blob/d724b9ed2f07b709dff29597ff91f24aff5ac8ad/lib/src/transcript.rs) and [`template.rs`](https://github.com/shayne-fletcher/yatima/blob/d724b9ed2f07b709dff29597ff91f24aff5ac8ad/lib/src/template.rs) at commit `d724b9ed2f07b709dff29597ff91f24aff5ac8ad`.

The central fact is simple: conversation memory is a list of turns, and a prompt template rebuilds the complete model prompt from that list whenever Yatima asks for another completion. The model engine does not retain the conversation between calls.

## `transcript.rs`: conversation data

The private `transcript` module defines two public types. `lib.rs` re-exports them as `yatima_lib::Role` and `yatima_lib::Turn`, so callers do not import the private module itself.

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    System,
    User,
    Assistant,
    Tool,
}

#[derive(Debug, Clone)]
pub struct Turn {
    pub role: Role,
    pub content: String,
}
```

The `Role` enum says who supplied one piece of transcript text:

- `System` is an instruction that governs the conversation.
- `User` is text supplied by the user.
- `Assistant` is text supplied by the model.
- `Tool` is the result of a tool call fed back to the model. It is not the request to call a tool; the request is model output and currently remains inside an assistant turn while an agent run is in progress.

The `Turn` struct pairs one role with an owned `String`. A turn has no methods and does not borrow any session or model state. Cloning a turn clones its text.

There is deliberately no `Transcript` struct in this file. The chat and agent code own their respective `Vec<Turn>` values and decide when to append, roll back, retain, or discard turns. This module supplies their common vocabulary without taking responsibility for session policy.

Any code can construct any sequence of turns because both fields are public. The types do not enforce alternation, require a system turn to come first, or prohibit two adjacent user turns. Those rules, where needed, belong to the chat or agent code that owns the list.

An assistant turn contains no private reasoning text. The completion is split before the turn is constructed, and only the answer side enters a transcript that may be rendered again. During a tool-using agent run that answer-side text can still contain the model's tool-call syntax. Chapter 2 follows the reasoning split; Chapters 8 and 10 follow tool-call parsing and the agent's working transcript.

## `template.rs`: model-native prompts

Different instruction-tuned models expect different control tokens and role names. The private `template` module hides those spellings behind the public `PromptTemplate` trait:

```rust
pub trait PromptTemplate {
    fn render(&self, turns: &[Turn]) -> String;
}
```

The `render` method borrows a template, borrows a slice of turns, and returns a newly allocated prompt string. It does not change the turns or retain state between calls. It returns `String`, not `Result<String>`, because rendering itself has no recoverable error path.

The file defines seven built-in template structs:

- `PlainTemplate`
- `ChatMlTemplate`
- `ChatMlThinkTemplate`
- `GemmaTemplate`
- `MistralTemplate`
- `GlmTemplate`
- `DeepSeekTemplate`

All seven implement the `PromptTemplate` trait. Other crates may implement the public trait for their own template types, so this is the current built-in set rather than a closed list.

Every implementation performs the same broad operation: visit the supplied turns in order, write each one using the selected model family's syntax, and finish with the cue that tells the model to produce the next assistant message.

The trait also has this forwarding implementation:

```rust
impl<T: PromptTemplate + ?Sized> PromptTemplate for Box<T> {
    fn render(&self, turns: &[Turn]) -> String {
        (**self).render(turns)
    }
}
```

This lets `Box<dyn PromptTemplate>` satisfy a `T: PromptTemplate` bound. A host can therefore choose a concrete format at runtime, put that value in a box, and pass the box to a generic `ChatSession` or `Agent`. The box owns the selected template; it does not own the transcript. Yatima's current templates happen to have no fields, but neither the trait nor this implementation requires that.

### Plain text

The `PlainTemplate` struct is the fallback and the simplest implementation. It writes a literal role tag before every turn and then appends an empty assistant tag:

```text
<|system|>
Be brief.
<|user|>
What is 2 + 2?
<|assistant|>
```

This format is useful in tests and when no trained format is known. It is not claimed to be any model family's native chat format, so it is a poor default for a real instruction-tuned model when a specific template is available.

### ChatML and reasoning ChatML

The `ChatMlTemplate` and `ChatMlThinkTemplate` structs share the private `render_chatml` function. System, user, and assistant turns become ordinary ChatML blocks. A tool result is represented as a ChatML user block containing `<tool_response>` markers because that is the shape expected by the Qwen tool protocol.

Both templates end with an assistant cue. `ChatMlThinkTemplate` additionally places `<think>` in that cue. The model therefore begins generating inside a reasoning block and is expected to emit its closing marker. The streaming code must know about that choice so it classifies the output correctly; Chapter 2 explains that stateful side of the arrangement.

### Gemma and Mistral

The `GemmaTemplate` and `MistralTemplate` structs handle models that have no separate system role. Each renderer saves system text in a local `pending_system` value and folds it into the next user turn:

```rust
let content = match pending_system.take() {
    Some(sys) => format!("{sys}\n\n{}", turn.content),
    None => turn.content.clone(),
};
```

For the turns `System("Be brief.")` and `User("hi")`, Gemma receives one user message containing both pieces:

```text
<start_of_turn>user
Be brief.

hi<end_of_turn>
<start_of_turn>model
```

Mistral performs the same fold but renders it as `[INST] Be brief.\n\nhi[/INST]`; the closing `[/INST]` is also its generation cue. Neither renderer writes its model's beginning-of-sequence token: the corresponding tokenizer adds that token, and writing it here would insert it twice.

These templates are chat-only in Yatima, but for different reasons. Gemma-2-it's native chat format defines only user and model turns; it has no trained syntax for tool declarations, calls, or results. Supporting tools with that model would require an invented prompting convention rather than implementing a missing native protocol. Mistral-7B-Instruct-v0.3 is trained for function calling, but `MistralTemplate` implements only its plain `[INST]` chat format; Yatima has not implemented Mistral's `[AVAILABLE_TOOLS]` and `[TOOL_CALLS]` syntax or a codec for parsing it. Both renderers can place tool-result text in a prompt by treating it like user text, but neither currently supports a complete tool exchange in Yatima. Format selection therefore reports that both are chat-only.

### GLM and DeepSeek

The `GlmTemplate` and `DeepSeekTemplate` structs preserve system text as system text because those formats support it. They also write their required opening token or prefix themselves because their tokenizers do not add it.

GLM maps `Role::Tool` to its `observation` role and ends with an assistant cue. DeepSeek writes system text at the front, renders the other turns with its native markers, and ends with an assistant cue followed by `<think>`. Because that opening marker is already in the prompt, it will not appear in the generated text. The `ReasoningSplitter` struct reads generated text and separates the reasoning from the answer. For DeepSeek it starts in reasoning mode, treats the initial output as reasoning, and switches to the answer when the model emits `</think>`. Chapter 2 explains its implementation.

## One rendering from beginning to end

Suppose a chat session has completed one exchange and the user asks a follow-up question. The session's history can be rendered directly:

```rust
let turns = vec![
    Turn {
        role: Role::User,
        content: "My name is Ada.".into(),
    },
    Turn {
        role: Role::Assistant,
        content: "Nice to meet you, Ada.".into(),
    },
    Turn {
        role: Role::User,
        content: "What is my name?".into(),
    },
];

let prompt = ChatMlTemplate.render(&turns);
```

The private `render_chatml` helper function starts with an empty `String` and visits the slice from first turn to last. It writes each turn as a ChatML block, then appends the cue for a new assistant message:

```text
<|im_start|>user
My name is Ada.<|im_end|>
<|im_start|>assistant
Nice to meet you, Ada.<|im_end|>
<|im_start|>user
What is my name?<|im_end|>
<|im_start|>assistant
```

The returned string contains the whole conversation, not only the latest question. The model can answer "Ada" because the earlier user and assistant turns have been placed in its prompt again.

An agent uses the same operation after a tool finishes. Assuming its system instructions have already advertised `read_file`, its working transcript might be:

```rust
let working_transcript = vec![
    Turn {
        role: Role::User,
        content: "Read README.md and tell me the project name.".into(),
    },
    Turn {
        role: Role::Assistant,
        content: concat!(
            "<tool_call>\n",
            r#"{"name":"read_file","arguments":{"path":"README.md"}}"#,
            "\n</tool_call>",
        )
        .into(),
    },
    Turn {
        role: Role::Tool,
        content: "[read_file ok] # Yatima".into(),
    },
];

let next_prompt = ChatMlTemplate.render(&working_transcript);
```

For the `Role::Tool` turn, the `render_chatml` helper function writes a ChatML user block containing:

```text
<tool_response>
[read_file ok] # Yatima
</tool_response>
```

The agent has already parsed the request, executed the tool, and constructed the result turn. The template's only job is to put the enlarged transcript into the syntax Qwen expects. Tool-call parsing and validation belong to the codec examined in Chapter 8.

Nothing in this path can return a Rust error. The practical failure is selecting a format that does not match the model: rendering still succeeds, but the model receives unfamiliar control tokens and may stop following instructions or produce repetitive text. The `ChatFormat` enum maps a configured format to one of these template implementations, while model profiles select a format for known models. Chapter 8 introduces that enum, its mapping, and its tool-support checks.

## Contract for callers

Callers provide an ordered slice of owned transcript entries and choose a template appropriate for the model. In return, the template preserves the supplied history in that model's native syntax and leaves the prompt ready for the model's next assistant output.

The template does not choose itself, validate the transcript, decide whether tools are supported, split generated reasoning, or retain conversation state. Those responsibilities remain with the format, chat, agent, and reasoning code above it.

## Maintainer checkpoint

- Conversation text is owned by the `Vec<Turn>` inside a chat session or agent run; `transcript.rs` defines the entries but owns no session.
- `PromptTemplate::render` borrows the complete turn list and returns a fresh prompt string. It is pure and has no error or cancellation path.
- To add a model's prompt syntax, implement the `PromptTemplate` trait here, then update the `ChatFormat` enum and model profiles described in Chapter 8.
- Before adding a template, determine whether the tokenizer or the template writes the beginning-of-sequence token (`TMPL-1`) and whether the model supports a system role or needs that text folded into a user turn (`TMPL-2`).
