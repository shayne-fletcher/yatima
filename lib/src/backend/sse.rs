//! Incremental framing of a server-sent-event byte stream into `data:`
//! payloads.
//!
//! llama-server's streaming `/completion` emits `data: {json}\n\n` events. The
//! network hands us arbitrary byte chunks: an event may span chunks, a chunk
//! may carry several events, and a split may land inside a multi-byte UTF-8
//! sequence — so the framer buffers *bytes* and converts only complete frames
//! to text. Shared by design with the future remote completer (see
//! plans/qwen-remote-completer.plan.md).

use anyhow::{Context, Result};

/// Accumulates raw stream bytes and yields each completed event's `data:`
/// payload. Non-`data:` lines (comments, other fields) are ignored.
#[derive(Default)]
pub struct SseFramer {
    buf: Vec<u8>,
}

impl SseFramer {
    pub fn new() -> SseFramer {
        SseFramer::default()
    }

    /// Feed one network chunk; returns the payload of every event the chunk
    /// completes, in order. An event is terminated by a blank line (`\n\n`,
    /// tolerating `\r\n` line endings); multiple `data:` lines in one event
    /// join with `\n` per the SSE spec.
    pub fn push(&mut self, chunk: &[u8]) -> Result<Vec<String>> {
        self.buf.extend_from_slice(chunk);
        let mut events = Vec::new();
        while let Some(end) = frame_end(&self.buf) {
            let frame: Vec<u8> = self.buf.drain(..end).collect();
            let frame = std::str::from_utf8(&frame)
                .context("llama-server stream frame is not valid UTF-8")?;
            let data: Vec<&str> = frame
                .lines()
                .filter_map(|line| {
                    line.strip_prefix("data: ")
                        .or_else(|| line.strip_prefix("data:"))
                })
                .collect();
            if !data.is_empty() {
                events.push(data.join("\n"));
            }
        }
        Ok(events)
    }
}

/// Byte offset one past the first blank-line frame terminator, if a complete
/// frame is buffered. Handles `\n\n` and the CRLF variants by treating `\r` as
/// ignorable before `\n`.
fn frame_end(buf: &[u8]) -> Option<usize> {
    let mut newlines = 0usize;
    for (i, &b) in buf.iter().enumerate() {
        match b {
            b'\n' => {
                newlines += 1;
                if newlines == 2 {
                    return Some(i + 1);
                }
            }
            b'\r' => {}
            _ => newlines = 0,
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frames_one_event_per_blank_line() {
        let mut f = SseFramer::new();
        let events = f.push(b"data: {\"a\":1}\n\ndata: {\"b\":2}\n\n").unwrap();
        assert_eq!(events, vec!["{\"a\":1}", "{\"b\":2}"]);
    }

    #[test]
    fn reassembles_events_split_across_chunks() {
        // The framer must be agnostic to where the network splits the stream —
        // including inside the `data: ` prefix and inside the payload.
        let mut f = SseFramer::new();
        assert!(f.push(b"da").unwrap().is_empty());
        assert!(f.push(b"ta: {\"conte").unwrap().is_empty());
        assert!(f.push(b"nt\":\"hi\"}\n").unwrap().is_empty());
        let events = f.push(b"\n").unwrap();
        assert_eq!(events, vec!["{\"content\":\"hi\"}"]);
    }

    #[test]
    fn tolerates_crlf_line_endings() {
        let mut f = SseFramer::new();
        let events = f.push(b"data: {\"a\":1}\r\n\r\n").unwrap();
        assert_eq!(events, vec!["{\"a\":1}"]);
    }

    #[test]
    fn survives_utf8_split_inside_payload() {
        // A chunk boundary may land mid-codepoint; only complete frames are
        // decoded, so this must not error.
        let mut f = SseFramer::new();
        let bytes = "data: {\"content\":\"caf\u{e9}\"}\n\n".as_bytes();
        let (a, b) = bytes.split_at(bytes.len() - 5); // splits the 2-byte é
        assert!(f.push(a).unwrap().is_empty());
        let events = f.push(b).unwrap();
        assert_eq!(events, vec!["{\"content\":\"caf\u{e9}\"}"]);
    }

    #[test]
    fn ignores_non_data_lines() {
        let mut f = SseFramer::new();
        let events = f.push(b": comment\nretry: 100\ndata: {}\n\n").unwrap();
        assert_eq!(events, vec!["{}"]);
    }
}
