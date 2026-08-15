# Walkthrough todo

This is the backlog discovered while reading Yatima for the maintainer walkthrough. It is authoring state, not part of the reader-facing mdBook.

## Chapter 1: transcript and templates

- [ ] Correct the `templates_render_multi_turn_history_with_cue` comment in `lib/src/template.rs`. The test checks conversation history and generation cues, but its comment claims `TMPL-2`, whose declared obligation is folding system text for models without a system role. The dedicated Gemma and Mistral tests are the actual witnesses.
- [ ] Implement Mistral's native tool protocol. The built-in profile selects the function-calling-trained `Mistral-7B-Instruct-v0.3`, but Yatima implements only its plain `[INST]` chat format. Add native tool declarations and result rendering, a matching `ToolCallCodec` for `[AVAILABLE_TOOLS]` and `[TOOL_CALLS]`, protocol and agent tests, and then enable tools for `ChatFormat::Mistral`. Chapter 8 should refine this task while reading the existing codecs and format table.
