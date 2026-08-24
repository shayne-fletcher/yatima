//! The reasoning channel: separating a model's chain-of-thought from its answer.
//!
//! Marker-based reasoning models (Kimi-Dev, Qwen3, the DeepSeek-R1 family) emit
//! an inline *thinking* span before their answer. Muse Glimmer instead emits a
//! sequence of assistant messages addressed to `self`, `user`, or a tool. In
//! either form, reasoning is **ephemeral**: callers may observe it separately,
//! but it must not enter the answer or transcript re-rendered into the next
//! prompt. [`split_reasoning`] handles complete marker-based replies,
//! [`ReasoningSplitter`] handles their streams, and [`AtemInterpreter`] handles
//! both complete and streamed Muse replies (REASON-1).
//!
//! Marker splitting is the identity when no marker is present. ATEM likewise
//! preserves a plain reply that never begins its addressed-message grammar.

/// One reasoning-marker dialect: the open/close pair a model wraps its
/// chain-of-thought in.
struct Dialect {
    open: &'static str,
    close: &'static str,
}

/// Every dialect we recognize. A model emits at most one; the spellings are
/// unambiguous and non-overlapping, so scanning all of them is safe.
const DIALECTS: &[Dialect] = &[
    // Qwen3, DeepSeek-R1 distills, and the de-facto generic spelling.
    Dialect {
        open: "<think>",
        close: "</think>",
    },
    // Kimi (Moonshot) — special tokens, not ASCII angle brackets.
    Dialect {
        open: "◁think▷",
        close: "◁/think▷",
    },
];

/// A completion split into its (optional) reasoning span and the answer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Reasoned {
    /// The chain-of-thought between the markers, trimmed — `None` when the
    /// completion carried no recognized reasoning span.
    pub reasoning: Option<String>,
    /// The surfaced answer: everything after the reasoning close marker,
    /// trimmed. Equal to the whole (trimmed) input when there is no marker.
    pub answer: String,
}

/// Split a raw completion into reasoning + answer (REASON-1).
///
/// The answer is everything after the **last** recognized close marker; the span
/// before it (minus the open marker, if present) is the reasoning. With no
/// marker at all this is the identity: the whole trimmed text is the answer and
/// `reasoning` is `None`. An **unterminated** open marker (no close — a
/// truncated chain-of-thought) classifies everything after the opener as
/// reasoning and only the prefix before it (usually empty) as answer: a
/// half-emitted think block must never leak into the committed answer
/// (REASON-1), and no content is lost — the span is surfaced as `reasoning`,
/// and an empty answer commits nothing at the chat/agent boundary.
///
/// A trailing tool call (`<tool_call>…`) sits after the close marker, so it
/// stays in `answer` and the agent codec still parses it.
pub fn split_reasoning(text: &str) -> Reasoned {
    // Pick the dialect whose close marker appears latest — a model uses one
    // dialect, and "latest close" matches the old strip-to-last-`</think>`
    // behavior when a span itself contains the marker text.
    let split = DIALECTS
        .iter()
        .filter_map(|d| text.rfind(d.close).map(|close_at| (d, close_at)))
        .max_by_key(|(_, close_at)| *close_at);

    match split {
        None => {
            // No close marker: either no reasoning at all (identity), or a
            // truncated span opened and never closed (earliest opener wins).
            let opened = DIALECTS
                .iter()
                .filter_map(|d| text.find(d.open).map(|open_at| (d, open_at)))
                .min_by_key(|(_, open_at)| *open_at);
            match opened {
                None => Reasoned {
                    reasoning: None,
                    answer: text.trim().to_string(),
                },
                Some((dialect, open_at)) => {
                    let reasoning = text[open_at + dialect.open.len()..].trim();
                    Reasoned {
                        reasoning: (!reasoning.is_empty()).then(|| reasoning.to_string()),
                        answer: text[..open_at].trim().to_string(),
                    }
                }
            }
        }
        Some((dialect, close_at)) => {
            let answer = text[close_at + dialect.close.len()..].trim().to_string();
            let before = &text[..close_at];
            // Drop the open marker if present; whatever precedes the close is the
            // reasoning, even if the open was never emitted.
            let reasoning = match before.find(dialect.open) {
                Some(open_at) => &before[open_at + dialect.open.len()..],
                None => before,
            }
            .trim();
            let reasoned = Reasoned {
                reasoning: (!reasoning.is_empty()).then(|| reasoning.to_string()),
                answer,
            };
            tracing::trace!(
                dialect = dialect.close,
                reasoning_chars = reasoned.reasoning.as_deref().map_or(0, str::len),
                answer_chars = reasoned.answer.len(),
                "reasoning split"
            );
            reasoned
        }
    }
}

/// [`split_reasoning`] for a **pre-seeded** cue (REASON-1): the prompt already
/// opened the reasoning block (`<think>` in the cue — DeepSeek, QwenThink), so
/// the model's output begins *inside* it and normally carries only the close
/// marker. With a close marker present this is the ordinary split; without
/// one, the reply never left the block — a truncated chain-of-thought — so the
/// whole text is reasoning and the answer is empty (and an empty answer
/// commits nothing at the chat/agent boundary). The streaming twin is
/// [`ReasoningSplitter::seeded`]; final and streaming classification agree.
pub(crate) fn split_seeded_reasoning(text: &str) -> Reasoned {
    if DIALECTS.iter().any(|d| text.contains(d.close)) {
        return split_reasoning(text);
    }
    let reasoning = text.trim();
    Reasoned {
        reasoning: (!reasoning.is_empty()).then(|| reasoning.to_string()),
        answer: String::new(),
    }
}

/// The answer only — `split_reasoning(text).answer`. The drop-in for callers
/// that don't need the reasoning trace.
pub fn strip_reasoning(text: &str) -> String {
    split_reasoning(text).answer
}

/// Which channel a streamed span belongs to.
///
/// Chat displays reasoning and answer directly. [`ToolCall`](Channel::ToolCall)
/// is internal protocol material: chat suppresses it and Agent converts it to
/// typed tool activity before any host/frontend event plane.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Channel {
    /// The model's chain-of-thought (between reasoning markers).
    Reasoning,
    /// The surfaced answer.
    Answer,
    /// A tool-directed ATEM message. Never user-facing completion text.
    ToolCall,
}

/// The streaming dual of [`split_reasoning`] (REASON-1): an incremental
/// classifier that routes each fragment of a *streamed* completion to
/// [`Channel::Reasoning`] or [`Channel::Answer`] as it arrives, so a live UI can
/// fold or dim the chain-of-thought. It recognizes the same dialects and handles
/// a marker that straddles fragment boundaries (split across two `push` calls).
/// Marker text itself is control, never emitted.
pub struct ReasoningSplitter {
    in_reasoning: bool,
    buf: String,
}

impl Default for ReasoningSplitter {
    fn default() -> ReasoningSplitter {
        ReasoningSplitter::new()
    }
}

impl ReasoningSplitter {
    /// A splitter for output that *begins in the answer* and enters reasoning on
    /// an open marker — the usual case (Kimi/Qwen3 emit `◁think▷`/`<think>`
    /// first).
    pub fn new() -> ReasoningSplitter {
        ReasoningSplitter {
            in_reasoning: false,
            buf: String::new(),
        }
    }

    /// A splitter for output that *begins inside the reasoning block* — used when
    /// the prompt pre-seeds the opener (DeepSeek's `<｜Assistant｜><think>` cue),
    /// so the stream's first marker is the close. See
    /// [`ChatFormat::pre_seeds_reasoning`](crate::ChatFormat::pre_seeds_reasoning).
    pub fn seeded() -> ReasoningSplitter {
        ReasoningSplitter {
            in_reasoning: true,
            buf: String::new(),
        }
    }

    /// Feed the next raw fragment; `emit(channel, text)` is called for each
    /// classified piece (zero or more times).
    pub fn push(&mut self, fragment: &str, mut emit: impl FnMut(Channel, &str)) {
        self.buf.push_str(fragment);
        self.drain(&mut emit);
    }

    /// Flush any buffered tail at end of stream. A partial marker that never
    /// completed is treated as content on the current channel.
    pub fn finish(mut self, mut emit: impl FnMut(Channel, &str)) {
        self.drain(&mut emit);
        if !self.buf.is_empty() {
            emit(self.channel(), &self.buf);
            self.buf.clear();
        }
    }

    fn channel(&self) -> Channel {
        if self.in_reasoning {
            Channel::Reasoning
        } else {
            Channel::Answer
        }
    }

    fn drain(&mut self, emit: &mut impl FnMut(Channel, &str)) {
        loop {
            // The earliest complete marker — open *or* close — controls the
            // channel. A marker *sets* state (open→reasoning, close→answer)
            // rather than toggling, so a stray or duplicated marker (e.g. a model
            // that emits `</think>` twice while degenerating) is always consumed,
            // never leaked into a channel.
            let hit = all_markers()
                .filter_map(|(text, opens)| self.buf.find(text).map(|i| (i, text, opens)))
                .min_by_key(|(i, ..)| *i);
            match hit {
                Some((i, text, opens)) => {
                    if i > 0 {
                        let ch = self.channel();
                        emit(ch, &self.buf[..i]);
                    }
                    self.buf = self.buf.split_off(i + text.len());
                    let was = self.in_reasoning;
                    self.in_reasoning = opens;
                    tracing::trace!(
                        marker = text,
                        opens,
                        was_reasoning = was,
                        now_reasoning = self.in_reasoning,
                        "reasoning channel marker"
                    );
                }
                None => {
                    // No complete marker: emit all but a tail that could be the
                    // start of one, so a boundary-straddling marker is caught on
                    // the next push.
                    let keep = held_back_len(&self.buf);
                    let upto = self.buf.len() - keep;
                    if upto > 0 {
                        let ch = self.channel();
                        emit(ch, &self.buf[..upto]);
                        self.buf.drain(..upto);
                    }
                    break;
                }
            }
        }
    }
}

/// Every marker the stream watches — both ends of every dialect, paired with
/// whether it *opens* a reasoning span — derived from the single [`DIALECTS`]
/// source so the batch and streaming splitters never drift.
fn all_markers() -> impl Iterator<Item = (&'static str, bool)> {
    DIALECTS
        .iter()
        .flat_map(|d| [(d.open, true), (d.close, false)])
}

/// Bytes to hold back at the tail of `buf`: the longest suffix that is a proper
/// prefix of any marker (at a marker char boundary, so the kept split is always
/// a valid `str` boundary), in case the marker completes in the next fragment.
/// No complete marker is present here (the caller already searched), so the
/// overlap is always shorter than the marker.
fn held_back_len(buf: &str) -> usize {
    let mut best = 0;
    for (m, _opens) in all_markers() {
        let mut k = m.len().min(buf.len());
        while k > best {
            if m.is_char_boundary(k) && buf.as_bytes().ends_with(&m.as_bytes()[..k]) {
                best = k;
                break;
            }
            k -= 1;
        }
    }
    best
}

pub(crate) const ATEM_START: &str = "<|start|>";
pub(crate) const ATEM_MESSAGE: &str = "<|message|>";
pub(crate) const ATEM_EOM: &str = "<|eom|>";
pub(crate) const ATEM_EOT: &str = "<|eot|>";
const ATEM_MAX_HEADER: usize = 256;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AtemState {
    TurnStart,
    Header,
    Body,
    Rejected,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum Recipient {
    User,
    Reasoning,
    Tool(String),
}

/// One tool-directed assistant message recovered from an ATEM response.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AtemToolMessage {
    pub recipient: String,
    pub body: String,
}

/// The complete interpretation of an ATEM response. Chat consumes the
/// `reasoned` projection; the Muse tool codec also consumes `tool_messages`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AtemResponse {
    pub reasoned: Reasoned,
    pub tool_messages: Vec<AtemToolMessage>,
    /// The framing machine entered its conservative rejection state. Retained
    /// tool messages are diagnostic attempts, never executable calls.
    pub rejected: bool,
}

/// Incrementally interprets Muse Glimmer's addressed assistant messages.
///
/// The prompt already supplied the first `<|start|>assistant` cue, so generated
/// text begins with the rest of that header. Later messages carry the full
/// header. The accepted text-completion subset is:
///
/// ```ebnf
/// completion      = first-message, { EOM, message }, [ EOT ] ;
/// first-message   = [ address ], MESSAGE, body ;
/// message         = START, "assistant", [ address ], MESSAGE, body ;
/// address         = " to=", recipient ;
/// recipient       = ? a nonempty recipient name containing no whitespace or "<" ? ;
/// body            = ? text containing none of START, MESSAGE, EOM, or EOT ? ;
/// START           = "<|start|>" ;
/// MESSAGE         = "<|message|>" ;
/// EOM             = "<|eom|>" ;
/// EOT             = "<|eot|>" ;
/// ```
///
/// An absent recipient and `to=user` select [`Channel::Answer`], `to=self`
/// selects [`Channel::Reasoning`], and every other recipient selects the
/// internal [`Channel::ToolCall`] stream retained for the Muse tool codec.
/// The grammar describes complete, well-formed replies. `push` additionally
/// handles arbitrary fragment boundaries, a plain-text fallback, and bounded
/// partial markers and headers. At end of stream, an incomplete ATEM header or
/// control marker is reasoning rather than answer. An invalid or overlong
/// header rejects the reply: all retained text is reasoning and the final
/// answer is empty, so the conversation boundary commits nothing (REASON-1).
/// Live fragments cannot be recalled, so a later rejection may invalidate text
/// already emitted on the answer channel even though no answer is committed.
pub struct AtemInterpreter {
    state: AtemState,
    recipient: Recipient,
    header: String,
    held: String,
    reasoning: String,
    answer: String,
    tool_messages: Vec<AtemToolMessage>,
    current_tool: Option<AtemToolMessage>,
    initial_header: bool,
    message_started: bool,
    ended: bool,
}

impl Default for AtemInterpreter {
    fn default() -> Self {
        Self::new()
    }
}

impl AtemInterpreter {
    /// Start at the continuation of the assistant cue already in the prompt.
    pub fn new() -> Self {
        Self {
            state: AtemState::Header,
            recipient: Recipient::User,
            header: String::new(),
            held: String::new(),
            reasoning: String::new(),
            answer: String::new(),
            tool_messages: Vec::new(),
            current_tool: None,
            initial_header: true,
            message_started: false,
            ended: false,
        }
    }

    /// Feed one decoded text fragment and emit every newly classified span.
    pub fn push(&mut self, fragment: &str, mut emit: impl FnMut(Channel, &str)) {
        let mut input = fragment;
        while !input.is_empty() {
            if self.ended {
                let tail = input.trim_start_matches(char::is_whitespace);
                if !tail.is_empty() {
                    self.reject(tail, &mut emit);
                }
                break;
            }
            match self.state {
                AtemState::Rejected => {
                    self.append_rejected(input, &mut emit);
                    break;
                }
                AtemState::TurnStart => {
                    let next = input.chars().next().expect("input is nonempty");
                    let len = next.len_utf8();
                    self.header.push(next);
                    input = &input[len..];
                    if self.header == ATEM_START {
                        self.header.clear();
                        self.state = AtemState::Header;
                        self.initial_header = false;
                    } else if !ATEM_START.starts_with(&self.header) {
                        let invalid = std::mem::take(&mut self.header);
                        self.reject(&invalid, &mut emit);
                    }
                }
                AtemState::Header => {
                    let next = input.chars().next().expect("input is nonempty");
                    let len = next.len_utf8();
                    self.header.push(next);
                    input = &input[len..];

                    if let Some(message_at) = self.header.find(ATEM_MESSAGE) {
                        if message_at > ATEM_MAX_HEADER {
                            let invalid = std::mem::take(&mut self.header);
                            self.reject(&invalid, &mut emit);
                            continue;
                        }
                        let recipient =
                            parse_atem_header(&self.header[..message_at], self.initial_header);
                        if let Some(recipient) = recipient {
                            self.header.clear();
                            self.state = AtemState::Body;
                            self.recipient = recipient;
                            self.initial_header = false;
                            self.message_started = false;
                        } else if self.initial_header && !looks_like_atem_header(&self.header) {
                            let plain = std::mem::take(&mut self.header);
                            self.state = AtemState::Body;
                            self.recipient = Recipient::User;
                            self.initial_header = false;
                            self.message_started = false;
                            self.emit_body(&plain, &mut emit);
                        } else {
                            let invalid = std::mem::take(&mut self.header);
                            self.reject(&invalid, &mut emit);
                        }
                    } else if !atem_header_prefix_possible(&self.header, self.initial_header) {
                        if self.initial_header && !looks_like_atem_header(&self.header) {
                            let plain = std::mem::take(&mut self.header);
                            self.state = AtemState::Body;
                            self.recipient = Recipient::User;
                            self.initial_header = false;
                            self.message_started = false;
                            self.emit_body(&plain, &mut emit);
                        } else {
                            let invalid = std::mem::take(&mut self.header);
                            self.reject(&invalid, &mut emit);
                        }
                    } else if self.header.len() > ATEM_MAX_HEADER {
                        let invalid = std::mem::take(&mut self.header);
                        self.reject(&invalid, &mut emit);
                    }
                }
                AtemState::Body => {
                    if !self.held.is_empty() {
                        let next = input.chars().next().expect("input is nonempty");
                        let len = next.len_utf8();
                        self.held.push(next);
                        input = &input[len..];
                        if let Some((at, marker)) = earliest_atem_control(&self.held) {
                            debug_assert_eq!(at + marker.len(), self.held.len());
                            if at > 0 {
                                let text = self.held[..at].to_string();
                                self.emit_body(&text, &mut emit);
                            }
                            self.held.clear();
                            self.consume_control(marker, &mut emit);
                        } else {
                            let keep = atem_held_back_len(&self.held);
                            let upto = self.held.len() - keep;
                            if upto > 0 {
                                let text = self.held[..upto].to_string();
                                let suffix = self.held[upto..].to_string();
                                self.held = suffix;
                                self.emit_body(&text, &mut emit);
                            }
                        }
                        continue;
                    }

                    if let Some((at, marker)) = earliest_atem_control(input) {
                        if at > 0 {
                            self.emit_body(&input[..at], &mut emit);
                        }
                        input = &input[at + marker.len()..];
                        self.consume_control(marker, &mut emit);
                    } else {
                        let keep = atem_held_back_len(input);
                        let upto = input.len() - keep;
                        if upto > 0 {
                            self.emit_body(&input[..upto], &mut emit);
                        }
                        if keep > 0 {
                            self.held.push_str(&input[upto..]);
                        }
                        break;
                    }
                }
            }
        }
    }

    /// Finish the response, returning the same answer/reasoning split used by
    /// final transcript commit. Streamed output preserves whitespace; these
    /// stored spans are trimmed like [`split_reasoning`].
    pub fn finish(mut self, mut emit: impl FnMut(Channel, &str)) -> AtemResponse {
        match self.state {
            AtemState::Body if !self.held.is_empty() => {
                let partial_marker = std::mem::take(&mut self.held);
                self.emit_reasoning_tail(&partial_marker, &mut emit);
            }
            AtemState::Header if !self.header.is_empty() => {
                let partial_header = std::mem::take(&mut self.header);
                if self.initial_header && !looks_like_atem_header(&partial_header) {
                    self.recipient = Recipient::User;
                    self.message_started = false;
                    self.emit_body(&partial_header, &mut emit);
                } else {
                    self.emit_reasoning_tail(&partial_header, &mut emit);
                }
            }
            AtemState::TurnStart if !self.header.is_empty() => {
                let partial_start = std::mem::take(&mut self.header);
                self.emit_reasoning_tail(&partial_start, &mut emit);
            }
            _ => {}
        }

        self.finish_tool_message();

        let reasoning = self.reasoning.trim().to_string();
        AtemResponse {
            reasoned: Reasoned {
                reasoning: (!reasoning.is_empty()).then_some(reasoning),
                answer: self.answer.trim().to_string(),
            },
            tool_messages: self.tool_messages,
            rejected: self.state == AtemState::Rejected,
        }
    }

    /// Interpret one complete response through the incremental machine.
    pub fn interpret(raw: &str) -> Reasoned {
        Self::interpret_full(raw).reasoned
    }

    /// Interpret one complete response, retaining tool-directed messages for
    /// the Muse codec.
    pub fn interpret_full(raw: &str) -> AtemResponse {
        let mut interpreter = Self::new();
        interpreter.push(raw, |_, _| {});
        interpreter.finish(|_, _| {})
    }

    fn emit_body(&mut self, text: &str, emit: &mut impl FnMut(Channel, &str)) {
        if text.is_empty() {
            return;
        }
        match self.recipient.clone() {
            Recipient::User => {
                if !self.message_started && !self.answer.is_empty() {
                    self.answer.push('\n');
                    emit(Channel::Answer, "\n");
                }
                self.answer.push_str(text);
                emit(Channel::Answer, text);
            }
            Recipient::Reasoning => {
                if !self.message_started && !self.reasoning.is_empty() {
                    self.reasoning.push('\n');
                    emit(Channel::Reasoning, "\n");
                }
                self.reasoning.push_str(text);
                emit(Channel::Reasoning, text);
            }
            Recipient::Tool(recipient) => {
                let message = self.current_tool.get_or_insert_with(|| AtemToolMessage {
                    recipient,
                    body: String::new(),
                });
                message.body.push_str(text);
                emit(Channel::ToolCall, text);
            }
        }
        self.message_started = true;
    }

    fn consume_control(&mut self, marker: &str, emit: &mut impl FnMut(Channel, &str)) {
        self.finish_tool_message();
        self.message_started = false;
        match marker {
            ATEM_EOM => {
                self.state = AtemState::TurnStart;
                self.header.clear();
            }
            ATEM_START => {
                self.state = AtemState::Header;
                self.header.clear();
                self.initial_header = false;
            }
            ATEM_EOT => {
                self.state = AtemState::TurnStart;
                self.header.clear();
                self.ended = true;
            }
            ATEM_MESSAGE => self.reject("", emit),
            _ => unreachable!("earliest_atem_control returned a known marker"),
        }
    }

    fn emit_reasoning_tail(&mut self, text: &str, emit: &mut impl FnMut(Channel, &str)) {
        if text.is_empty() {
            return;
        }
        self.reasoning.push_str(text);
        emit(Channel::Reasoning, text);
    }

    fn reject(&mut self, text: &str, emit: &mut impl FnMut(Channel, &str)) {
        self.finish_tool_message();
        if !self.answer.is_empty() {
            if !self.reasoning.is_empty() {
                self.reasoning.push('\n');
            }
            self.reasoning.push_str(&self.answer);
            self.answer.clear();
        }
        self.header.clear();
        self.held.clear();
        self.state = AtemState::Rejected;
        self.ended = false;
        self.append_rejected(text, emit);
    }

    fn append_rejected(&mut self, text: &str, emit: &mut impl FnMut(Channel, &str)) {
        if text.is_empty() {
            return;
        }
        self.reasoning.push_str(text);
        emit(Channel::Reasoning, text);
    }

    fn finish_tool_message(&mut self) {
        if let Some(message) = self.current_tool.take() {
            self.tool_messages.push(message);
        }
    }
}

/// The streaming response classifier selected by a prompt template.
pub enum ResponseClassifier {
    Markers(ReasoningSplitter),
    Atem(AtemInterpreter),
}

impl ResponseClassifier {
    pub fn markers() -> Self {
        Self::Markers(ReasoningSplitter::new())
    }

    pub fn seeded_markers() -> Self {
        Self::Markers(ReasoningSplitter::seeded())
    }

    pub fn atem() -> Self {
        Self::Atem(AtemInterpreter::new())
    }

    pub fn push(&mut self, fragment: &str, mut emit: impl FnMut(Channel, &str)) {
        match self {
            Self::Markers(splitter) => splitter.push(fragment, &mut emit),
            Self::Atem(interpreter) => interpreter.push(fragment, &mut emit),
        }
    }

    pub fn finish(self, mut emit: impl FnMut(Channel, &str)) {
        match self {
            Self::Markers(splitter) => splitter.finish(&mut emit),
            Self::Atem(interpreter) => {
                interpreter.finish(&mut emit);
            }
        }
    }
}

fn parse_atem_header(header: &str, initial: bool) -> Option<Recipient> {
    let suffix = if initial {
        header
    } else {
        header.strip_prefix("assistant")?
    };
    if suffix.is_empty() {
        return Some(Recipient::User);
    }
    let recipient = suffix.strip_prefix(" to=")?;
    valid_recipient(recipient).then(|| match recipient {
        "user" => Recipient::User,
        "self" => Recipient::Reasoning,
        tool => Recipient::Tool(tool.to_string()),
    })
}

fn valid_recipient(recipient: &str) -> bool {
    !recipient.is_empty() && !recipient.chars().any(|c| c.is_whitespace() || c == '<')
}

fn atem_header_prefix_possible(header: &str, initial: bool) -> bool {
    let suffix = if initial {
        header
    } else if "assistant".starts_with(header) {
        return true;
    } else if let Some(suffix) = header.strip_prefix("assistant") {
        suffix
    } else {
        return false;
    };

    if ATEM_MESSAGE.starts_with(suffix) {
        return true;
    }
    if " to=".starts_with(suffix) {
        return true;
    }
    let Some(recipient_and_marker) = suffix.strip_prefix(" to=") else {
        return false;
    };
    match recipient_and_marker.find('<') {
        Some(marker_at) => {
            valid_recipient(&recipient_and_marker[..marker_at])
                && ATEM_MESSAGE.starts_with(&recipient_and_marker[marker_at..])
        }
        None => valid_recipient(recipient_and_marker),
    }
}

fn looks_like_atem_header(header: &str) -> bool {
    !header.is_empty()
        && (" to=".starts_with(header)
            || header.starts_with(" to=")
            || header == "<"
            || header.starts_with("<|"))
}

fn earliest_atem_control(buf: &str) -> Option<(usize, &'static str)> {
    [ATEM_EOM, ATEM_EOT, ATEM_START, ATEM_MESSAGE]
        .into_iter()
        .filter_map(|marker| buf.find(marker).map(|at| (at, marker)))
        .min_by_key(|(at, _)| *at)
}

fn atem_held_back_len(buf: &str) -> usize {
    [ATEM_EOM, ATEM_EOT, ATEM_START, ATEM_MESSAGE]
        .into_iter()
        .map(|marker| {
            let max = marker.len().min(buf.len());
            (1..=max)
                .rev()
                .find(|&len| buf.as_bytes().ends_with(&marker.as_bytes()[..len]))
                .unwrap_or(0)
        })
        .max()
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_marker_is_the_identity() {
        let r = split_reasoning("no markers here");
        assert_eq!(r.reasoning, None);
        assert_eq!(r.answer, "no markers here");
        // trims like the old strip_think.
        assert_eq!(strip_reasoning("  padded  "), "padded");
    }

    #[test]
    fn splits_the_think_dialect() {
        let r = split_reasoning("<think>weighing it</think>\nthe answer");
        assert_eq!(r.reasoning.as_deref(), Some("weighing it"));
        assert_eq!(r.answer, "the answer");
    }

    #[test]
    fn splits_the_kimi_dialect() {
        let r = split_reasoning("◁think▷let me see◁/think▷ 4");
        assert_eq!(r.reasoning.as_deref(), Some("let me see"));
        assert_eq!(r.answer, "4");
    }

    #[test]
    fn keeps_text_after_the_last_close() {
        // Matches the old strip_think: split on the last close marker.
        let r = split_reasoning("a</think>b</think>final");
        assert_eq!(r.answer, "final");
        assert_eq!(r.reasoning.as_deref(), Some("a</think>b"));
    }

    #[test]
    fn unterminated_reasoning_never_reaches_the_answer() {
        // upholds: REASON-1 — a truncated chain-of-thought (opened, never
        // closed) is reasoning, not answer: committing it would replay the
        // think block into every later prompt. The span is surfaced, the
        // answer is the (empty) prefix, and an empty answer commits nothing
        // at the chat/agent boundary.
        let r = split_reasoning("<think>still thinking");
        assert_eq!(r.reasoning.as_deref(), Some("still thinking"));
        assert_eq!(r.answer, "");

        // A non-empty prefix before the opener survives as the answer.
        let r = split_reasoning("Sure.<think>truncated");
        assert_eq!(r.reasoning.as_deref(), Some("truncated"));
        assert_eq!(r.answer, "Sure.");
    }

    #[test]
    fn seeded_split_treats_a_closed_reply_normally() {
        let r = split_seeded_reasoning("thinking</think>the answer");
        assert_eq!(r.reasoning.as_deref(), Some("thinking"));
        assert_eq!(r.answer, "the answer");
    }

    #[test]
    fn seeded_truncation_is_all_reasoning() {
        // upholds: REASON-1 — under a pre-seeded cue a reply with no close
        // marker never left the think block: nothing may surface as answer,
        // and the empty answer commits nothing downstream.
        let r = split_seeded_reasoning("half a thought, then the budget");
        assert_eq!(
            r.reasoning.as_deref(),
            Some("half a thought, then the budget")
        );
        assert_eq!(r.answer, "");
    }

    #[test]
    fn close_without_open_still_yields_an_answer() {
        // Some setups suppress the open marker; treat everything before the
        // close as reasoning rather than leaking it into the answer.
        let r = split_reasoning("hidden reasoning</think>visible answer");
        assert_eq!(r.reasoning.as_deref(), Some("hidden reasoning"));
        assert_eq!(r.answer, "visible answer");
    }

    #[test]
    fn answer_retains_a_trailing_tool_call() {
        // The agent must still parse a tool call that follows the reasoning.
        let r = split_reasoning("<think>which file?</think>\n<tool_call>\n{}\n</tool_call>");
        assert_eq!(r.answer, "<tool_call>\n{}\n</tool_call>");
        assert!(r.answer.contains("<tool_call>"));
    }

    #[test]
    fn empty_reasoning_span_is_none() {
        let r = split_reasoning("<think></think>answer");
        assert_eq!(r.reasoning, None);
        assert_eq!(r.answer, "answer");
    }

    /// Run a splitter over `fragments`, collecting per-channel output.
    fn stream(mut s: ReasoningSplitter, fragments: &[&str]) -> (String, String) {
        let mut reasoning = String::new();
        let mut answer = String::new();
        let mut sink = |ch: Channel, t: &str| match ch {
            Channel::Reasoning => reasoning.push_str(t),
            Channel::Answer => answer.push_str(t),
            Channel::ToolCall => unreachable!("marker splitters do not emit tool calls"),
        };
        for f in fragments {
            s.push(f, &mut sink);
        }
        s.finish(&mut sink);
        (reasoning, answer)
    }

    #[test]
    fn splitter_classifies_a_single_fragment() {
        let (r, a) = stream(ReasoningSplitter::new(), &["<think>reason</think>answer"]);
        assert_eq!(r, "reason");
        assert_eq!(a, "answer");
    }

    #[test]
    fn splitter_handles_markers_across_boundaries() {
        // The open and close markers are each split across pushes.
        let (r, a) = stream(
            ReasoningSplitter::new(),
            &["<th", "ink>hi the", "re</thi", "nk>by", "e"],
        );
        assert_eq!(r, "hi there");
        assert_eq!(a, "bye");
    }

    #[test]
    fn splitter_seeded_starts_in_reasoning() {
        // DeepSeek pre-seeds `<think>`, so the stream opens mid-thought and the
        // first marker is the close.
        let (r, a) = stream(
            ReasoningSplitter::seeded(),
            &["thinking…", "</think>", "the answer"],
        );
        assert_eq!(r, "thinking…");
        assert_eq!(a, "the answer");
    }

    #[test]
    fn splitter_handles_the_kimi_dialect() {
        let (r, a) = stream(ReasoningSplitter::new(), &["◁think▷w◁/think▷4"]);
        assert_eq!(r, "w");
        assert_eq!(a, "4");
    }

    #[test]
    fn splitter_with_no_markers_is_all_answer() {
        let (r, a) = stream(ReasoningSplitter::new(), &["just ", "an ", "answer"]);
        assert_eq!(r, "");
        assert_eq!(a, "just an answer");
    }

    #[test]
    fn splitter_flushes_an_unterminated_partial_marker() {
        // A dangling `<thi` at end of stream is content, not a swallowed marker.
        let (r, a) = stream(ReasoningSplitter::new(), &["answer <thi"]);
        assert_eq!(r, "");
        assert_eq!(a, "answer <thi");
    }

    /// Drive the splitter one *character* at a time — the most adversarial
    /// fragmentation (every marker is split maximally) — and never leak a marker.
    fn stream_char_by_char(s: ReasoningSplitter, text: &str) -> (String, String) {
        let frags: Vec<String> = text.chars().map(|c| c.to_string()).collect();
        let refs: Vec<&str> = frags.iter().map(String::as_str).collect();
        stream(s, &refs)
    }

    #[test]
    fn splitter_seeded_consumes_close_amid_real_text() {
        // Regression: a DeepSeek-style stream (seeded, close marker after real
        // punctuation `]\n`) fed char-by-char must consume `</think>`, not leak
        // it into the answer.
        let raw = "reasoning\n\\boxed{3}\n]\n</think>\n\nThe answer is 3.";
        let (r, a) = stream_char_by_char(ReasoningSplitter::seeded(), raw);
        assert!(!a.contains("think"), "marker leaked into answer: {a:?}");
        assert!(!r.contains("think"), "marker leaked into reasoning: {r:?}");
        // The stream preserves whitespace (live display); trim for the compare.
        assert_eq!(a.trim(), "The answer is 3.");
        assert!(r.contains("\\boxed{3}"));
    }

    #[test]
    fn splitter_consumes_a_stray_or_duplicate_close() {
        // Regression for the live bug: a degenerating model emitted `</think>`
        // twice. With a toggle, the second close (seen while already in the
        // answer) leaked; set-semantics consume every marker. Reproduced
        // synthetically, no model needed.
        let raw = "think one</think>answer one</think>answer two";
        let (r, a) = stream_char_by_char(ReasoningSplitter::seeded(), raw);
        assert!(!a.contains("think"), "stray close leaked: {a:?}");
        assert_eq!(r, "think one");
        assert_eq!(a, "answer oneanswer two");
    }

    #[test]
    fn splitter_ignores_a_stray_open_while_reasoning() {
        // The dual: a second open while already reasoning is consumed, not leaked.
        let raw = "<think>a<think>b</think>done";
        let (r, a) = stream_char_by_char(ReasoningSplitter::new(), raw);
        assert!(!r.contains("think") && !a.contains("think"));
        assert_eq!(r, "ab");
        assert_eq!(a, "done");
    }

    #[test]
    fn splitter_open_then_close_char_by_char() {
        // The new() path under the same adversarial fragmentation.
        let raw = "<think>weigh it</think>final";
        let (r, a) = stream_char_by_char(ReasoningSplitter::new(), raw);
        assert_eq!(r, "weigh it");
        assert_eq!(a, "final");
    }

    fn stream_atem(fragments: &[&str]) -> (String, String, Reasoned) {
        let (reasoning, answer, _, response) = stream_atem_full(fragments);
        (reasoning, answer, response.reasoned)
    }

    fn stream_atem_full(fragments: &[&str]) -> (String, String, String, AtemResponse) {
        let mut interpreter = AtemInterpreter::new();
        let mut reasoning = String::new();
        let mut answer = String::new();
        let mut tool = String::new();
        let mut sink = |channel: Channel, text: &str| match channel {
            Channel::Reasoning => reasoning.push_str(text),
            Channel::Answer => answer.push_str(text),
            Channel::ToolCall => tool.push_str(text),
        };
        for fragment in fragments {
            interpreter.push(fragment, &mut sink);
        }
        let response = interpreter.finish(&mut sink);
        (reasoning, answer, tool, response)
    }

    const ATEM_CANONICAL: &str = " to=self<|message|>think café<|eom|>\
                                  <|start|>assistant to=user<|message|>answer<|eot|>";
    const ATEM_MULTI_MESSAGE: &str = " to=self<|message|>first<|eom|>\
                                      <|start|>assistant to=self<|message|>second<|eom|>\
                                      <|start|>assistant to=user<|message|>done<|eot|>";
    const ATEM_TOOL: &str = " to=self<|message|>inspect the file<|eom|>\
                            <|start|>assistant to=read_file<|message|>\
                            <atem:function_calls>\n<atem:invoke name=\"read_file\">\n\
                            <atem:parameter name=\"path\">README.md</atem:parameter>\n\
                            </atem:invoke>\n</atem:function_calls><|eot|>";

    #[test]
    fn atem_interprets_addressed_messages() {
        // upholds: REASON-1 — addressed reasoning and protocol framing never
        // enter the answer; streamed and final buckets agree.
        let (live_reasoning, live_answer, final_split) = stream_atem(&[ATEM_CANONICAL]);
        assert_eq!(live_reasoning, "think café");
        assert_eq!(live_answer, "answer");
        assert_eq!(final_split.reasoning.as_deref(), Some("think café"));
        assert_eq!(final_split.answer, "answer");
        assert!(!live_reasoning.contains("<|") && !live_answer.contains("<|"));
    }

    #[test]
    fn atem_preserves_a_plain_unframed_reply() {
        let (reasoning, answer, final_split) = stream_atem(&["Just a plain reply."]);
        assert_eq!(reasoning, "");
        assert_eq!(answer, "Just a plain reply.");
        assert_eq!(final_split.reasoning, None);
        assert_eq!(final_split.answer, "Just a plain reply.");
    }

    #[test]
    fn atem_preserves_angle_bracket_initial_unframed_replies() {
        for raw in ["<3 that idea!", "<div> is an HTML element"] {
            let (reasoning, answer, final_split) = stream_atem(&[raw]);
            assert_eq!(reasoning, "");
            assert_eq!(answer, raw);
            assert_eq!(final_split.reasoning, None);
            assert_eq!(final_split.answer, raw);
        }
    }

    #[test]
    fn atem_ignores_only_whitespace_after_eot() {
        let turn = " to=user<|message|>answer<|eot|>";
        for fragments in [
            vec![turn, "\n\t"],
            vec![" to=user<|message|>answer<|eot|>\n\t"],
        ] {
            let (reasoning, answer, final_split) = stream_atem(&fragments);
            assert_eq!(reasoning, "");
            assert_eq!(answer, "answer");
            assert_eq!(final_split.reasoning, None);
            assert_eq!(final_split.answer, "answer");
        }

        let (_, _, rejected) = stream_atem(&[" to=user<|message|>answer<|eot|>not whitespace"]);
        assert!(rejected.answer.is_empty());
        let reasoning = rejected.reasoning.expect("rejected tail is inspectable");
        assert!(reasoning.contains("answer"));
        assert!(reasoning.contains("not whitespace"));
    }

    #[test]
    fn atem_post_eot_rejection_is_chunk_invariant() {
        let whole = stream_atem(&[" to=user<|message|>answer<|eot|>\n\tnot whitespace"]);
        let fragmented =
            stream_atem(&[" to=user<|message|>answer<|eot|>", "\n\t", "not whitespace"]);
        assert_eq!(fragmented, whole);
    }

    #[test]
    fn atem_routes_tool_recipients_to_the_tool_bucket() {
        // upholds: REASON-1 — protocol payload is retained for the codec but
        // can never emerge as reasoning or user-facing answer text.
        let raw = " to=fs.stat_file<|message|><atem:function_calls>x</atem:function_calls>\
                   <|start|>assistant to=user<|message|>Done.";
        let (reasoning, answer, tool, response) = stream_atem_full(&[raw]);
        assert_eq!(reasoning, "");
        assert_eq!(answer, "Done.");
        assert!(tool.contains("atem:function_calls"));
        assert_eq!(response.reasoned.answer, "Done.");
        assert_eq!(response.tool_messages.len(), 1);
        assert_eq!(response.tool_messages[0].recipient, "fs.stat_file");
    }

    #[test]
    fn atem_accepts_the_observed_start_without_eom_recovery_shape() {
        // Stage 1 captured a generation with a fresh START directly after the
        // reasoning body. The formal grammar uses EOM; consuming START here is
        // the conservative recovery behavior pinned by that real trace.
        let raw = " to=self<|message|>think\
                   <|start|>assistant to=user<|message|>answer";
        let (_, _, final_split) = stream_atem(&[raw]);
        assert_eq!(final_split.reasoning.as_deref(), Some("think"));
        assert_eq!(final_split.answer, "answer");
    }

    #[test]
    fn atem_rejects_an_overlong_header_without_an_answer() {
        // upholds: REASON-1 — the header buffer is bounded and a reply that
        // exceeds it is not committable protocol text.
        let raw = format!(" to={}", "x".repeat(ATEM_MAX_HEADER + 1));
        let (_, answer, final_split) = stream_atem(&[&raw]);
        assert!(answer.is_empty());
        assert!(final_split.answer.is_empty());
        assert!(final_split.reasoning.is_some());
    }

    #[test]
    fn atem_rejects_an_invalid_later_header_and_clears_the_answer() {
        let raw = " to=user<|message|>partial answer<|start|>not-assistant";
        let (_, _, final_split) = stream_atem(&[raw]);
        assert!(final_split.answer.is_empty());
        let reasoning = final_split
            .reasoning
            .expect("rejected reply is inspectable");
        assert!(reasoning.contains("partial answer"));
        assert!(reasoning.contains("not-assistant"));
    }

    #[test]
    fn atem_partial_control_markers_never_reach_the_answer() {
        // Every incomplete header marker is framing; every incomplete body
        // control is withheld and conservatively finalized as reasoning.
        for marker in [ATEM_MESSAGE, ATEM_START, ATEM_EOM, ATEM_EOT] {
            for len in 1..marker.len() {
                let prefix = &marker[..len];
                let (_, _, header_split) = stream_atem(&[prefix]);
                assert!(
                    header_split.answer.is_empty(),
                    "header prefix leaked: {prefix:?}"
                );
                assert_eq!(header_split.reasoning.as_deref(), Some(prefix));
            }
        }
        for marker in [ATEM_MESSAGE, ATEM_START, ATEM_EOM, ATEM_EOT] {
            for len in 1..marker.len() {
                let prefix = &marker[..len];
                let raw = format!(" to=user<|message|>answer{prefix}");
                let (_, _, body_split) = stream_atem(&[&raw]);
                assert_eq!(body_split.answer, "answer");
                assert_eq!(body_split.reasoning.as_deref(), Some(prefix));
            }
        }
    }

    #[test]
    fn atem_rejects_a_message_delimiter_inside_a_body() {
        let raw = " to=user<|message|>answer<|message|>not-a-header";
        let (live_reasoning, _, final_split) = stream_atem(&[raw]);
        assert!(final_split.answer.is_empty());
        assert!(live_reasoning.contains("not-a-header"));
    }

    fn scalar_boundaries(text: &str) -> Vec<usize> {
        text.char_indices()
            .map(|(at, _)| at)
            .chain(std::iter::once(text.len()))
            .collect()
    }

    #[test]
    fn atem_is_invariant_under_every_single_scalar_split() {
        // upholds: REASON-1 — finite witnesses for the chunking law: every
        // single split of both canonical responses matches whole-input parsing.
        for raw in [ATEM_CANONICAL, ATEM_MULTI_MESSAGE] {
            let expected = stream_atem(&[raw]);
            for split in scalar_boundaries(raw) {
                assert_eq!(
                    stream_atem(&[&raw[..split], &raw[split..]]),
                    expected,
                    "split at byte {split}"
                );
            }
        }
    }

    #[test]
    fn atem_is_invariant_under_one_character_fragments() {
        for raw in [ATEM_CANONICAL, ATEM_MULTI_MESSAGE] {
            let boundaries = scalar_boundaries(raw);
            let fragments: Vec<&str> = boundaries
                .windows(2)
                .map(|pair| &raw[pair[0]..pair[1]])
                .collect();
            assert_eq!(stream_atem(&fragments), stream_atem(&[raw]));
        }
    }

    #[test]
    fn atem_tool_messages_are_invariant_under_fragmentation() {
        // upholds: REASON-1, PROTO-1 — the tool bucket, its recipient, and the
        // prose projections are independent of transport chunk boundaries.
        let expected = stream_atem_full(&[ATEM_TOOL]);
        for split in scalar_boundaries(ATEM_TOOL) {
            assert_eq!(
                stream_atem_full(&[&ATEM_TOOL[..split], &ATEM_TOOL[split..]]),
                expected,
                "tool transcript split at byte {split}"
            );
        }
        let boundaries = scalar_boundaries(ATEM_TOOL);
        let fragments: Vec<&str> = boundaries
            .windows(2)
            .map(|pair| &ATEM_TOOL[pair[0]..pair[1]])
            .collect();
        assert_eq!(stream_atem_full(&fragments), expected);
    }

    proptest::proptest! {
        #[test]
        fn atem_is_invariant_under_random_chunk_partitions(
            case in 0usize..2,
            cuts in proptest::collection::vec(proptest::bool::ANY, 0..160),
        ) {
            let raw = [ATEM_CANONICAL, ATEM_MULTI_MESSAGE][case];
            let boundaries = scalar_boundaries(raw);
            let mut fragments = Vec::new();
            let mut start = 0;
            for (index, &end) in boundaries.iter().enumerate().skip(1) {
                if end == raw.len() || cuts.get(index - 1).copied().unwrap_or(false) {
                    fragments.push(&raw[start..end]);
                    start = end;
                }
            }
            proptest::prop_assert_eq!(stream_atem(&fragments), stream_atem(&[raw]));
        }
    }
}
