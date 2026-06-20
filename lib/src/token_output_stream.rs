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

    fn decode(&self, tokens: &[u32]) -> Result<String> {
        match self.tokenizer.decode(tokens, true) {
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
}
