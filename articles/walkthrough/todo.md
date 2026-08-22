# Walkthrough todo

This is the backlog discovered while reading Yatima for the maintainer walkthrough. It is authoring state, not part of the reader-facing mdBook.

## Chapter 1: transcript and templates

- [ ] Correct the `templates_render_multi_turn_history_with_cue` comment in `lib/src/template.rs`. The test checks conversation history and generation cues, but its comment claims `TMPL-2`, whose declared obligation is folding system text for models without a system role. The dedicated Gemma and Mistral tests are the actual witnesses.
- [ ] Implement Mistral's native tool protocol. The built-in profile selects the function-calling-trained `Mistral-7B-Instruct-v0.3`, but Yatima implements only its plain `[INST]` chat format. Add native tool declarations and result rendering, a matching `ToolCallCodec` for `[AVAILABLE_TOOLS]` and `[TOOL_CALLS]`, protocol and agent tests, and then enable tools for `ChatFormat::Mistral`. Chapter 8 should refine this task while reading the existing codecs and format table.

## Chapter 2: token and reasoning streams

- [ ] Correct the final-flush comment in `Engine::generate_with`. It says `TokenOutputStream` withholds punctuation until a later alphanumeric token, but the current implementation emits completed punctuation immediately and withholds only a trailing `U+FFFD` incomplete sequence. Retain the requirement to call `decode_rest` after every non-error loop exit.
- [ ] Correct the `reasoning.rs` module documentation. Reasoning is exposed separately through `ChatSession::last_reasoning`, `AgentEvent`, and classified frontend fragments; the actual rule is that it must not be presented as the answer or committed to transcript history.
- [ ] Update the `Channel` documentation in `reasoning.rs`. It says the agent is non-streaming and only chat consumes channels, but the agent now uses `ReasoningSplitter` and emits `AgentEvent::Fragment` values while withholding tool-call markup separately.
- [ ] Reconcile unterminated reasoning between the two paths. `ReasoningSplitter` routes text after a complete opener with no closer to `Channel::Reasoning`, while `split_reasoning` returns the entire raw response as `answer`, allowing truncated reasoning to enter history despite `REASON-1`. Choose and test one storage policy for truncated reasoning in chat and agent turns.
- [ ] Give the core `TokenOutputStream` regressions a non-skipping test fixture. All current tests return successfully without assertions when the relevant cached tokenizer or GGUF is absent, so a fresh checkout does not exercise incremental punctuation, UTF-8, or special-marker preservation.
