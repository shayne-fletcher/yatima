//! Incremental, streaming detokenization.
//!
//! Adapted from candle's examples (we depend on candle as a library, not on
//! `candle-examples`), but using the canonical `prefix_offset`/`read_offset`
//! streaming-decode condition from HuggingFace text-generation-inference / vLLM:
//! a fragment is emitted as soon as the decode grows and does **not** end in the
//! U+FFFD replacement character (an unfinished multi-byte / byte-fallback
//! sequence). candle's stock example instead holds until the last char is
//! alphanumeric, which needlessly buffers trailing punctuation — corrupting
//! structured output such as tool-call JSON (`"`, `}`, `</tool_call>`).

use candle_core::Result;
use tokenizers::Tokenizer;

pub struct TokenOutputStream {
    tokenizer: Tokenizer,
    tokens: Vec<u32>,
    prev_index: usize,
    current_index: usize,
}

impl TokenOutputStream {
    pub fn new(tokenizer: Tokenizer) -> Self {
        Self {
            tokenizer,
            tokens: Vec::new(),
            prev_index: 0,
            current_index: 0,
        }
    }

    // Decode with `skip_special_tokens = false` (REASON-1). Reasoning models mark
    // their chain-of-thought boundary with a *special* token — QwQ's `</think>`
    // (id 151668) is flagged exactly like `<|im_end|>` — and
    // `skip_special_tokens = true` silently drops it, so the `ReasoningSplitter`
    // never sees the boundary and the answer is mislabeled as reasoning forever.
    // Keeping specials is safe here because end-of-turn / EOS ids are filtered by
    // id in the generation loop *before* a token reaches this stream, and the
    // splitter consumes the reasoning markers (they never leak to the UI).
    fn decode(&self, tokens: &[u32]) -> Result<String> {
        match self.tokenizer.decode(tokens, false) {
            Ok(str) => Ok(str),
            Err(err) => candle_core::bail!("cannot decode: {err}"),
        }
    }

    /// Push one token; return any newly-completed text fragment.
    pub fn next_token(&mut self, token: u32) -> Result<Option<String>> {
        let prev_text = if self.tokens.is_empty() {
            String::new()
        } else {
            let tokens = &self.tokens[self.prev_index..self.current_index];
            self.decode(tokens)?
        };
        self.tokens.push(token);
        let text = self.decode(&self.tokens[self.prev_index..])?;
        // Emit once the decode grows and is not a partial multi-byte sequence. A
        // trailing U+FFFD means an unfinished byte-fallback char — hold for more
        // tokens; a U+FFFD in the middle is a genuinely invalid id, so let it
        // through. (Canonical TGI/vLLM streaming-decode condition.)
        if text.len() > prev_text.len() && !text.ends_with('\u{fffd}') {
            let text = text.split_at(prev_text.len());
            self.prev_index = self.current_index;
            self.current_index = self.tokens.len();
            Ok(Some(text.1.to_string()))
        } else {
            Ok(None)
        }
    }

    /// Decode all pushed tokens at once — the faithful reference string. Use
    /// for correctness checks against the incremental [`next_token`] stream.
    #[cfg(test)]
    pub fn decode_all(&self) -> Result<String> {
        self.decode(&self.tokens)
    }

    /// Flush any text buffered after the last emitted fragment.
    pub fn decode_rest(&self) -> Result<Option<String>> {
        let prev_text = if self.tokens.is_empty() {
            String::new()
        } else {
            let tokens = &self.tokens[self.prev_index..self.current_index];
            self.decode(tokens)?
        };
        let text = self.decode(&self.tokens[self.prev_index..])?;
        if text.len() > prev_text.len() {
            let text = text.split_at(prev_text.len());
            Ok(Some(text.1.to_string()))
        } else {
            Ok(None)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Stream a known tool-call JSON through the incremental detokenizer and
    // check faithfulness, both as the agent's stop path sees it (next_token
    // fragments only, no decode_rest) and the full path (with decode_rest).
    // Gated on the Qwen tokenizer being present.
    #[test]
    fn incremental_detok_preserves_tool_call_json() {
        let dir = crate::model_dir(
            &crate::models_root(),
            &crate::ModelId::parse("Qwen/Qwen2.5-7B-Instruct").unwrap(),
        );
        let path = dir.join("tokenizer.json");
        if !path.exists() {
            eprintln!("skipping: Qwen tokenizer absent at {}", path.display());
            return;
        }
        let tk = Tokenizer::from_file(&path).unwrap();
        let input = "<tool_call>\n{\"name\": \"read_file\", \"arguments\": {\"path\": \"secret.txt\"}}\n</tool_call>";
        let ids = tk.encode(input, false).unwrap().get_ids().to_vec();

        let mut stream = TokenOutputStream::new(tk);
        let mut streamed = String::new();
        for id in ids {
            if let Some(f) = stream.next_token(id).unwrap() {
                streamed.push_str(&f);
            }
        }
        let reference = stream.decode_all().unwrap();
        let with_rest = {
            let mut s = streamed.clone();
            if let Some(f) = stream.decode_rest().unwrap() {
                s.push_str(&f);
            }
            s
        };

        eprintln!("reference : {reference:?}");
        eprintln!("streamed  : {streamed:?}");
        eprintln!("with_rest : {with_rest:?}");
        assert_eq!(reference, input, "tokenizer round-trip itself is faithful");
        // With the U+FFFD condition, complete UTF-8 (incl. quotes / `}` / `>`)
        // flushes immediately, so the incremental stream is faithful on its own —
        // no characters dropped mid-string or at the tail.
        assert_eq!(
            streamed, reference,
            "incremental stream must not drop characters"
        );
        assert_eq!(with_rest, reference);
    }

    // Prose round-trip with the words that doubled in live runs ("would would",
    // "have have", ...) — to tell a detokenizer double-emit from model
    // degeneration.
    #[test]
    fn incremental_detok_preserves_prose() {
        let dir = crate::model_dir(
            &crate::models_root(),
            &crate::ModelId::parse("Qwen/Qwen2.5-7B-Instruct").unwrap(),
        );
        let path = dir.join("tokenizer.json");
        if !path.exists() {
            eprintln!("skipping: Qwen tokenizer absent at {}", path.display());
            return;
        }
        let tk = Tokenizer::from_file(&path).unwrap();
        let input = "You would have to check the network configuration of the server, \
                     and you would need more of the runbook procedure to take the next action.";
        let ids = tk.encode(input, false).unwrap().get_ids().to_vec();

        let mut stream = TokenOutputStream::new(tk);
        let mut out = String::new();
        for id in ids {
            if let Some(f) = stream.next_token(id).unwrap() {
                out.push_str(&f);
            }
        }
        if let Some(f) = stream.decode_rest().unwrap() {
            out.push_str(&f);
        }
        eprintln!("out: {out:?}");
        assert_eq!(out, input, "prose must round-trip with no duplicated words");
    }

    // Regression for the "never green" bug (REASON-1): a reasoning model's close
    // marker is a *special* token (QwQ's `</think>` is flagged like `<|im_end|>`).
    // The stream must NOT skip it — if it does, the splitter never sees the
    // reasoning→answer boundary and the answer is mislabeled as reasoning forever.
    // Gated on the QwQ GGUF tokenizer being present.
    #[test]
    fn stream_preserves_special_reasoning_marker() {
        use candle_core::quantized::gguf_file;
        use candle_core::quantized::tokenizer::TokenizerFromGguf;

        let path =
            crate::models_root().join("bartowski/Qwen_QwQ-32B-GGUF/Qwen_QwQ-32B-Q4_K_M.gguf");
        if !path.exists() {
            eprintln!("skipping: QwQ GGUF absent at {}", path.display());
            return;
        }
        let mut f = std::fs::File::open(&path).unwrap();
        let content = gguf_file::Content::read(&mut f).unwrap();
        let tk = Tokenizer::from_gguf(&content).unwrap();
        assert_eq!(
            tk.encode("</think>", false).unwrap().get_ids(),
            &[151668],
            "QwQ encodes </think> as the special id this guards"
        );

        // reasoning tokens, then the special close marker, then answer tokens.
        let mut ids = tk.encode("weighing it", false).unwrap().get_ids().to_vec();
        ids.push(151668); // </think>
        ids.extend(tk.encode("the answer", false).unwrap().get_ids());

        let mut stream = TokenOutputStream::new(tk);
        let mut out = String::new();
        for id in ids {
            if let Some(frag) = stream.next_token(id).unwrap() {
                out.push_str(&frag);
            }
        }
        if let Some(frag) = stream.decode_rest().unwrap() {
            out.push_str(&frag);
        }
        eprintln!("streamed: {out:?}");
        assert!(
            out.contains("</think>"),
            "the reasoning close marker must survive the stream, got {out:?}"
        );
    }
}
