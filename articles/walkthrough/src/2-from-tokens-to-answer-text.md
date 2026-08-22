# 2. From tokens to answer text

Chapter 1 ended with a prompt ready for the model. This chapter follows output in the other direction: generated token ids become text fragments, reasoning is separated from the answer, and the complete response is reduced to answer text before it enters conversation history.

The chapter reads [`token_output_stream.rs`](https://github.com/shayne-fletcher/yatima/blob/d724b9ed2f07b709dff29597ff91f24aff5ac8ad/lib/src/token_output_stream.rs) and [`reasoning.rs`](https://github.com/shayne-fletcher/yatima/blob/d724b9ed2f07b709dff29597ff91f24aff5ac8ad/lib/src/reasoning.rs) at commit `d724b9ed2f07b709dff29597ff91f24aff5ac8ad`.

## `token_output_stream.rs`: token ids to text

A model generates integer token ids, not complete Rust strings. Decoding each id independently is incorrect because several ids may contribute bytes to one character, and a tokenizer may need preceding tokens to decide spacing. The private `token_output_stream` module provides the `TokenOutputStream` struct that retains enough context to decode incrementally.

The struct is declared `pub`, but its module is private and `lib.rs` does not re-export it. It is therefore an internal helper used by the `Engine` struct rather than part of the public `yatima-lib` API.

```rust
pub struct TokenOutputStream {
    tokenizer: Tokenizer,
    tokens: Vec<u32>,
    prev_index: usize,
    current_index: usize,
}
```

The `Tokenizer` struct comes from the `tokenizers` crate. `TokenOutputStream` owns it, owns every generated token id pushed into `tokens`, and mutates two token indices:

- `prev_index` is the start of the retained decoding window.
- `current_index` separates the already emitted prefix in that window from tokens whose text has not yet been emitted.

The indices count tokens. When the code compares or splits decoded strings, it uses byte offsets because Rust's `String::len` and `str::split_at` operate on UTF-8 bytes.

### Construction and decoding

`TokenOutputStream::new` takes ownership of a tokenizer and starts with no tokens and both indices at zero.

The private `decode` method delegates to `Tokenizer::decode`:

```rust
fn decode(&self, tokens: &[u32]) -> Result<String> {
    match self.tokenizer.decode(tokens, false) {
        Ok(str) => Ok(str),
        Err(err) => candle_core::bail!("cannot decode: {err}"),
    }
}
```

The second argument is `false`, meaning that special tokens are not skipped. This matters for reasoning models: a closing marker such as QwQ's `</think>` may itself be a special token. If decoding removed it, the later reasoning code would never see the point where the answer begins.

End-of-sequence tokens do not leak as text. The `Engine::generate_with` method, examined fully in Chapter 6, checks generated ids against the model's end-of-sequence set before passing them to `TokenOutputStream`. Reasoning markers, by contrast, must survive this stage because they carry information needed by the next file.

`Tokenizer::decode` can fail. The private method converts that failure into Candle's `Result` type, and `next_token` or `decode_rest` propagates it to the generation caller. No fragment is emitted after a decode error.

### Emitting one completed fragment

The ordinary operation is `next_token`:

```rust
pub fn next_token(&mut self, token: u32) -> Result<Option<String>>
```

It accepts one generated id and returns one of three outcomes:

- `Ok(Some(fragment))` means new complete text is ready.
- `Ok(None)` means the token was accepted but more tokens are needed.
- `Err(error)` means the tokenizer could not decode the retained token window.

Before appending the new id, `next_token` decodes `tokens[prev_index..current_index]` as `prev_text`. It then appends the id and decodes `tokens[prev_index..]` as `text`. If `text` has grown, the suffix after `prev_text` is the newly available fragment.

The suffix is emitted only when `text` does not end in the Unicode replacement character `U+FFFD`. A replacement character at the end usually means the available byte-fallback tokens contain only part of a UTF-8 character. Returning `None` preserves those tokens so the next id can complete the character. A replacement character in the middle is treated as actual decoded content and is not withheld.

After an emission, `current_index` advances to the end of `tokens`, while the just-emitted token span remains between `prev_index` and `current_index` as decoding context. Text before `prev_index` no longer needs to be decoded again.

This condition also allows completed punctuation to stream immediately. Quotes, braces, and closing tags are significant in structured model output; waiting for a later alphanumeric character could truncate or visibly delay a tool call.

### Finishing the token stream

`decode_rest` returns text still buffered after the last ordinary emission:

```rust
pub fn decode_rest(&self) -> Result<Option<String>>
```

The engine calls it once whenever generation ends, whether the cause was end-of-sequence, a token limit, cancellation, repetition detection, or a caller-requested stop. This final read prevents an incomplete tail from being silently omitted.

Despite the word "decode" being used for both operations, `decode_rest` does not consume or reset the stream: it borrows `&self`. Calling it twice would return the same remaining text twice. Its internal engine caller invokes it only once at the end of a generation.

The test-only `decode_all` method decodes every stored id in one operation. It provides a reference string against which tests compare the fragments accumulated from `next_token` and `decode_rest`.

## `reasoning.rs`: text to reasoning and answer

The private `reasoning` module receives decoded text. Unlike `token_output_stream`, its public types and functions are re-exported from `lib.rs`, so library users can apply the same reasoning rules to output from another model backend. The exported set is:

- the `Reasoned` struct;
- the `split_reasoning` and `strip_reasoning` functions;
- the `Channel` enum;
- the `ReasoningSplitter` struct.

The file provides two related interfaces:

- `split_reasoning` handles a complete model response.
- The `ReasoningSplitter` struct classifies fragments while a response is still arriving.

The complete-response function controls what is stored. The streaming struct controls what a live caller may display as reasoning or answer. Streaming is not the source of conversation history.

### Recognized marker forms

The private `Dialect` struct stores one opening and closing marker:

```rust
struct Dialect {
    open: &'static str,
    close: &'static str,
}
```

The private `DIALECTS` constant currently contains two entries:

- `<think>` and `</think>` for Qwen3, DeepSeek-R1 distills, and other models using that spelling;
- `◁think▷` and `◁/think▷` for Kimi.

Both the complete-response and streaming implementations derive their recognized markers from this one list.

### Splitting a complete response

The public `Reasoned` struct owns the result:

```rust
pub struct Reasoned {
    pub reasoning: Option<String>,
    pub answer: String,
}
```

The public `split_reasoning` function takes `&str` and returns `Reasoned`. It searches all dialects for the closing marker that occurs latest in the response. Everything after that marker becomes the trimmed answer. Text before it becomes the trimmed reasoning span, with an opening marker removed when one is present.

Searching for the close marker rather than requiring an opener handles the pre-seeded templates from Chapter 1. DeepSeek's prompt already contains `<think>`, so its generated output may contain reasoning followed by `</think>` without generating another opener.

If there is no recognized closing marker, `split_reasoning` makes no split: `reasoning` is `None`, and the whole trimmed input becomes `answer`. This is safe for ordinary non-reasoning models and avoids discarding content after a half-generated opening marker. It also means that a response truncated after `<think>` can be treated as an answer by the complete-response path.

The public `strip_reasoning` function is the smaller interface for callers that need only `split_reasoning(text).answer`.

A tool request after the reasoning close marker remains part of `answer`. That name means "non-reasoning model output" at this stage; the agent's tool codec, introduced in Chapter 8, may subsequently interpret the text as a tool call rather than user-facing prose.

### Classifying a live stream

The public `Channel` enum labels streamed text:

```rust
pub enum Channel {
    Reasoning,
    Answer,
}
```

The enum is deliberately about text presentation, not every event in an agent run. A tool request is handled by the agent's tool protocol rather than represented as a third `Channel` variant.

The public `ReasoningSplitter` struct owns the state needed while fragments arrive:

```rust
pub struct ReasoningSplitter {
    in_reasoning: bool,
    buf: String,
}
```

`in_reasoning` selects the channel for ordinary buffered text. `buf` holds text that has not yet been emitted, including a suffix that might become a marker when the next fragment arrives.

`ReasoningSplitter::new` starts in the answer channel. This is correct for models that generate their own opening marker: the splitter withholds a possible marker prefix, consumes the complete opener, and changes to reasoning before emitting the text inside it.

`ReasoningSplitter::seeded` starts in the reasoning channel. It is used when the prompt supplied the opening marker, as DeepSeek's template does, so the first generated text is already reasoning and the first generated marker may be `</think>`. The `ChatFormat` enum introduced in Chapter 8 records which configured formats require this constructor.

`Default::default` is the same as `ReasoningSplitter::new`.

### Pushing fragments

The public `push` method appends a fragment and calls a supplied closure for each piece it can classify:

```rust
pub fn push(&mut self, fragment: &str, emit: impl FnMut(Channel, &str))
```

One push may emit no text, one piece, or several pieces. The private `drain` method repeatedly finds the earliest complete opening or closing marker in `buf`. The private `all_markers` helper function supplies those markers by turning each `Dialect` into an opener and closer, so there is no second marker table to maintain. `drain` emits preceding text on the current channel, removes the marker itself, and sets `in_reasoning` according to whether the marker opens or closes reasoning.

Markers set the state rather than toggling it. A duplicated `<think>` keeps the splitter in reasoning, and a duplicated `</think>` keeps it in answer. In either case the marker is consumed rather than leaked into displayed text.

When there is no complete marker, the private `held_back_len` function finds the longest suffix of `buf` that is also the beginning of any known marker. `drain` retains that suffix and emits everything before it. Thus fragments ending in `<thi` or `◁/thi` wait for the next push rather than being displayed prematurely.

The public `finish` method consumes the splitter, drains complete markers once more, and emits any remaining partial-marker suffix as ordinary text on the current channel. Consuming `self` makes the end of this classification stream explicit and prevents a second finish. Streaming preserves whitespace exactly; unlike `split_reasoning`, neither `push` nor `finish` trims emitted text.

If a complete opening marker arrives but no closing marker follows, the streaming path remains in reasoning and emits the remaining text on `Channel::Reasoning`. This differs from `split_reasoning`, which treats a complete response with no closing marker as an unsplit answer.

## One output from beginning to end

The following schematic code has the same shape as the local engine and a streaming caller. It accepts generated ids, decodes complete fragments, and classifies each fragment immediately:

```rust
fn route_tokens(
    tokenizer: Tokenizer,
    non_eos_ids: impl IntoIterator<Item = u32>,
) -> candle_core::Result<(String, String)> {
    let mut text_stream = TokenOutputStream::new(tokenizer);
    let mut splitter = ReasoningSplitter::new();
    let mut reasoning = String::new();
    let mut answer = String::new();
    {
        let mut emit = |channel: Channel, text: &str| match channel {
            Channel::Reasoning => reasoning.push_str(text),
            Channel::Answer => answer.push_str(text),
        };

        for token in non_eos_ids {
            if let Some(fragment) = text_stream.next_token(token)? {
                splitter.push(&fragment, &mut emit);
            }
        }
        if let Some(fragment) = text_stream.decode_rest()? {
            splitter.push(&fragment, &mut emit);
        }
        splitter.finish(&mut emit);
    }

    Ok((reasoning, answer))
}
```

Suppose the ids decode to `<think>Check the earlier turn.</think>Ada`. Token boundaries and callback fragments may divide those words or markers anywhere. `TokenOutputStream` ensures the fragments concatenate to the original decoded text. `ReasoningSplitter` buffers across those fragment boundaries, discards the two markers, and produces `("Check the earlier turn.", "Ada")`.

There are two different finalization operations because the two structs protect different boundaries. `decode_rest` recovers text withheld while token bytes were incomplete. `ReasoningSplitter::finish` releases text withheld because it might have been the start of a reasoning marker.

After generation completes, the `Completion` struct returned by a model backend contains the assembled text. Chapter 4 introduces that struct and the `Completer` trait that returns it. In simplified `ChatSession` code, the complete response is split again before history is updated:

```rust
let Reasoned { reasoning, answer } = split_reasoning(&completion.text);
self.last_reasoning = reasoning;
self.turns.push(Turn {
    role: Role::Assistant,
    content: answer,
});
```

This is the point governed by [`REASON-1`](https://github.com/shayne-fletcher/yatima/blob/d724b9ed2f07b709dff29597ff91f24aff5ac8ad/lib/src/lib.rs#L259): reasoning may be delivered separately to a caller, but it must not be stored in the assistant turn that is rendered into the next prompt. The complete response is split again for storage rather than trusting whatever a particular UI did with the live fragments.

## What this does not parse

Both paths in `reasoning.rs` recognize marker-delimited text embedded in a completion. Adding another model with the same behavior means adding its opening and closing spellings to `DIALECTS` and checking both whole-response and split-fragment behavior.

Muse's ATEM format has a different structure. It labels separate messages with recipients such as `self` and `user`; reasoning is identified by the recipient rather than by delimiters inside one text stream. ATEM therefore needs a message-framing interpreter, not another `Dialect` entry. Chapter 8 determines where that interpreter belongs.

## Contract for callers

`TokenOutputStream` promises that emitted fragments, followed by its final remainder, reproduce the tokenizer's decoded output without exposing incomplete trailing UTF-8 during ordinary streaming. It assumes end-of-sequence ids have already been removed and deliberately preserves other special tokens.

`ReasoningSplitter` promises to remove recognized marker text and preserve the order of all other streamed text while assigning it to `Reasoning` or `Answer`. `split_reasoning` promises an answer-only value for complete responses carrying a recognized closing marker and acts as a trimmed identity when no closing marker exists.

## Maintainer checkpoint

- The engine creates one internal `TokenOutputStream` per generation. It owns a tokenizer clone, every non-EOS id passed to it, and the token indices needed for incremental decoding.
- A streaming caller creates one `ReasoningSplitter` per response. It owns the current channel and only the text suffix that cannot yet be classified safely.
- The normal path is generated id, completed text fragment, classified channel fragment, then a separate whole-response split before an assistant turn is stored.
- For a new marker-delimited reasoning model, add one `Dialect` and test complete, split-fragment, pre-seeded, and unterminated output. Do not represent an addressed-message protocol such as ATEM as a marker dialect.
- Preserve `REASON-1`: only the answer may enter a transcript used by a later model call.
