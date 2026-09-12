# 2. From tokens to answer text

Chapter 1 ended with a prompt ready for a model backend. This chapter follows output in the other direction: Candle token ids become text fragments, then the selected response protocol separates reasoning, answer text, and tool-directed material before anything enters conversation history.

The chapter reads [`token_output_stream.rs`](https://github.com/shayne-fletcher/yatima/blob/402d34a26bedd9d254e76a51be8c034961e28db1/lib/src/token_output_stream.rs) and [`reasoning.rs`](https://github.com/shayne-fletcher/yatima/blob/402d34a26bedd9d254e76a51be8c034961e28db1/lib/src/reasoning.rs) at commit `402d34a26bedd9d254e76a51be8c034961e28db1`.

`TokenOutputStream` is specific to the in-process Candle path. The reasoning and response classifiers operate on text and are shared by every `Completer`, including `llama-server`.

## `token_output_stream.rs`: token ids to text

A model generates integer token ids, not complete Rust strings. Decoding each id independently is incorrect because several ids may contribute bytes to one character, and a tokenizer may need preceding tokens to decide spacing. The private `token_output_stream` module provides the `TokenOutputStream` struct that retains enough context to decode incrementally.

```rust
pub struct TokenOutputStream {
    tokenizer: Tokenizer,
    tokens: Vec<u32>,
    prev_index: usize,
    current_index: usize,
}
```

The `Tokenizer` struct comes from the `tokenizers` crate. `TokenOutputStream` owns it and every generated id passed to it. `prev_index` starts the retained decoding window; `current_index` separates the emitted prefix from ids whose text is still pending.

The struct is public inside a private module and is not re-exported from `lib.rs`, so it remains an `Engine` implementation detail.

### Producing one fragment

The ordinary method is:

```rust
pub fn next_token(&mut self, token: u32) -> Result<Option<String>>
```

It decodes the retained tokens before and after appending the new id. If the decoded text has grown and does not end in Unicode's replacement character `U+FFFD`, the new suffix is complete enough to emit. A trailing replacement character usually means byte-fallback tokens have produced only part of a UTF-8 character, so the method returns `Ok(None)` and waits for another id.

The private `decode` method calls `Tokenizer::decode(tokens, false)`. The `false` argument preserves special tokens. The engine removes end-of-sequence ids before this point, but reasoning and response markers must survive so the next stage can classify them.

When generation stops, `decode_rest` returns any decoded suffix that has not yet been emitted:

```rust
pub fn decode_rest(&self) -> Result<Option<String>>
```

It borrows rather than consumes the stream, so its caller must invoke it once. `Engine` does so for every stop reason, including cancellation and token limits. The test-only `decode_all` method supplies a reference value against which the accumulated fragments are checked.

## `reasoning.rs`: text to semantic channels

The private `reasoning` module is backend-independent. `lib.rs` re-exports its important public types:

- `Reasoned`, `split_reasoning`, and `strip_reasoning` for complete marker-based replies;
- `ReasoningSplitter` for streamed marker-based replies;
- `AtemInterpreter`, `AtemResponse`, and `AtemToolMessage` for Muse's addressed-message protocol;
- `ResponseClassifier`, which holds the classifier chosen by a prompt template;
- `Channel`, which identifies reasoning, answer, or tool-call material.

There are two protocol families in this file. Marker models embed reasoning between delimiters such as `<think>` and `</think>`. Muse emits separate assistant messages addressed to `self`, `user`, or a tool.

## Complete marker replies

The `Reasoned` struct is the common answer projection:

```rust
pub struct Reasoned {
    pub reasoning: Option<String>,
    pub answer: String,
}
```

`split_reasoning` recognizes two marker dialects: the ordinary `<think>` spelling and Kimi's `◁think▷` spelling. It finds the latest recognized close marker, removes the corresponding framing, and returns trimmed reasoning and answer strings.

When no marker occurs, the function is a trimmed identity: the whole reply is the answer. When an opening marker occurs without a close, the text after the opener is classified as reasoning and the prefix before it is the answer. A typical truncated reasoning reply therefore produces an empty answer and is not committed.

Some templates place the opening `<think>` in the prompt rather than waiting for the model to generate it. Their crate-private `split_seeded_reasoning` helper treats a reply without a close marker as entirely reasoning, because the generated text never left the already-open block.

`strip_reasoning` is the convenience function for a caller that needs only the answer projection.

## Streamed marker replies

The `ReasoningSplitter` struct performs the same classification incrementally:

```rust
pub struct ReasoningSplitter {
    in_reasoning: bool,
    buf: String,
}
```

`ReasoningSplitter::new` starts in the answer channel and changes state when it sees an opening marker. `ReasoningSplitter::seeded` starts inside reasoning for templates whose assistant cue already supplied the opener.

`push` accepts an arbitrary text fragment and emits every portion whose channel is known. It retains a suffix such as `<thi` when that suffix might become a marker after the next fragment. `finish` consumes the splitter and releases the final buffered suffix on the current channel.

The complete and streaming policies agree on truncation: text inside an unclosed reasoning block is reasoning, not an answer that may be stored.

## Muse's addressed messages

Muse Glimmer uses ATEM. A normal reply can contain one message for private reasoning followed by one for the user:

```text
 to=self<|message|>check the evidence<|eom|>
<|start|>assistant to=user<|message|>The answer is 42.<|eot|>
```

The first line begins partway through a header because `MuseGlimmerTemplate` already ended the prompt with `<|start|>assistant`. Later messages include the complete start marker and assistant role.

The source states the accepted complete-response grammar as EBNF:

```ebnf
completion      = first-message, { EOM, message }, [ EOT ] ;
first-message   = [ address ], MESSAGE, body ;
message         = START, "assistant", [ address ], MESSAGE, body ;
address         = " to=", recipient ;
recipient       = ? a nonempty name containing no whitespace or "<" ? ;
body            = ? text containing none of START, MESSAGE, EOM, or EOT ? ;
START           = "<|start|>" ;
MESSAGE         = "<|message|>" ;
EOM             = "<|eom|>" ;
EOT             = "<|eot|>" ;
```

Read plainly: the generated reply begins with the remainder of the first assistant header and its body. It may then contain more complete assistant messages separated by `<|eom|>`, and it may finish with `<|eot|>`.

The public `AtemInterpreter` struct is the incremental state machine implementing that protocol. Its principal state is:

- whether it is reading a turn start, header, body, or rejected response;
- the current recipient;
- bounded header and possible-marker buffers;
- accumulated reasoning and answer text;
- completed and current tool-directed messages;
- whether the initial partial header has passed and whether the turn ended.

`push` may receive fragments at any byte-valid string boundary. An absent recipient or `to=user` emits `Channel::Answer`; `to=self` emits `Channel::Reasoning`; any other recipient emits `Channel::ToolCall` and is retained as an `AtemToolMessage` for the Muse codec.

The grammar describes valid complete replies. The state machine also defines operational behavior for streaming: it allows a plain-text response that never begins an ATEM header, holds partial control markers across fragments, limits headers to 256 bytes, and treats incomplete framing as reasoning rather than answer. Invalid or overlong framing moves the machine to `Rejected`, clears any candidate answer into reasoning, and leaves the final answer empty.

`finish` consumes the interpreter and returns an `AtemResponse`:

```rust
pub struct AtemResponse {
    pub reasoned: Reasoned,
    pub tool_messages: Vec<AtemToolMessage>,
    pub rejected: bool,
}
```

Chat uses the `reasoned` projection. `MuseAtemCodec` also inspects `tool_messages`, validates the ATEM invocation body, and turns one accepted invocation into Yatima's typed tool call. Rejected messages remain diagnostic data and are never executable.

## One selector for streaming

The `ResponseClassifier` enum lets a template select the right streaming machine without making chat or agent code switch on model names:

```rust
pub enum ResponseClassifier {
    Markers(ReasoningSplitter),
    Atem(AtemInterpreter),
}
```

Its `push` and `finish` methods delegate to the contained classifier. Ordinary templates return marker classification by default. Pre-seeded templates return a seeded marker splitter. `MuseGlimmerTemplate` returns an ATEM interpreter.

The same template also owns final interpretation through `PromptTemplate::interpret_response`. `ChatSession` and `Agent` therefore follow this shape:

```rust
let classifier = template.classifier();       // live fragments
let interpreted = template.interpret_response(&completion.text); // commit
```

Live classification controls presentation. Final interpretation controls storage. A UI cannot make protocol framing persistent merely by displaying a fragment incorrectly.

## The commit boundary

After completion, `ChatSession` asks its template to interpret the whole reply. It stores the returned reasoning separately and commits `Turn::assistant(answer)` only when the answer is nonempty and nondegenerate. An all-reasoning, truncated, or rejected response rolls back the user turn instead of leaving an empty assistant envelope in history.

`Agent` uses the same template boundary. Its codec may additionally extract a typed tool call from the raw and interpreted response. Tool-call and tool-result turns exist only in the working transcript for that run; only a final user/assistant exchange enters persistent history.

This is `REASON-1` in operational terms: every active response protocol separates reasoning from answer at the completion-to-turn boundary, and reasoning or framing never enters the history rendered into a later prompt.

## Maintainer checkpoint

- Candle creates one `TokenOutputStream` per generation. Other backends already supply text and enter after this stage.
- A prompt template selects both the streaming `ResponseClassifier` and the final response interpreter.
- Add a marker spelling to `DIALECTS` only for a genuinely marker-delimited protocol.
- Change ATEM behavior in `AtemInterpreter`, keeping complete interpretation invariant under fragment boundaries.
- `Channel::ToolCall` is internal protocol material; agent code turns it into typed tool activity before frontend events are emitted.
- Preserve `REASON-1`: only a valid answer may become an `Assistant` turn used by the next model call.
