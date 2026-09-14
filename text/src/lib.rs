//! Host-neutral prettification of model output text.
//!
//! LLMs write LaTeX (`\( 9\pi \)`, `\frac{a}{b}`, `x^2`) that no yatima
//! frontend can *typeset* — the TUI is a character grid, the GUI's real
//! math layout is a far-future slice. This crate is the shared best-effort
//! middle: drop the math delimiters, expand the brace commands models lean
//! on, map symbols and Greek to Unicode, convert super/subscripts — while
//! passing fenced code through untouched so code is never mangled.
//!
//! Pure std, no dependencies, WASM-clean by construction: the same pass
//! must run in the TUI, the GUI, and `yatima-serve`'s browser client.
//! (Extracted from `yatima-tui`'s renderer when the GUI became its second
//! consumer.)

// LaTeX command → symbol table for [`prettify_math`]. Longer names first so a
// prefix (e.g. `\le`) never shadows a longer one (`\leq`, `\leftarrow`).
const LATEX_SYMBOLS: &[(&str, &str)] = &[
    ("\\leftarrow", "←"),
    ("\\rightarrow", "→"),
    ("\\Rightarrow", "⇒"),
    ("\\Leftarrow", "⇐"),
    ("\\times", "×"),
    ("\\cdot", "·"),
    ("\\div", "÷"),
    ("\\pm", "±"),
    ("\\mp", "∓"),
    ("\\leq", "≤"),
    ("\\le", "≤"),
    ("\\geq", "≥"),
    ("\\ge", "≥"),
    ("\\neq", "≠"),
    ("\\ne", "≠"),
    ("\\approx", "≈"),
    ("\\equiv", "≡"),
    ("\\propto", "∝"),
    ("\\infty", "∞"),
    ("\\to", "→"),
    ("\\sum", "∑"),
    ("\\prod", "∏"),
    ("\\int", "∫"),
    ("\\partial", "∂"),
    ("\\nabla", "∇"),
    ("\\cdots", "⋯"),
    ("\\ldots", "…"),
    ("\\dots", "…"),
    ("\\angle", "∠"),
    ("\\circ", "°"),
    ("\\lfloor", "⌊"),
    ("\\rfloor", "⌋"),
    ("\\lceil", "⌈"),
    ("\\rceil", "⌉"),
    ("\\langle", "⟨"),
    ("\\rangle", "⟩"),
    ("\\bmod", "mod"),
    ("\\pmod", "mod"),
    ("\\mod", "mod"),
    ("\\in", "∈"),
    ("\\notin", "∉"),
    ("\\subseteq", "⊆"),
    ("\\subset", "⊂"),
    ("\\cup", "∪"),
    ("\\cap", "∩"),
    ("\\emptyset", "∅"),
    ("\\forall", "∀"),
    ("\\exists", "∃"),
    ("\\wedge", "∧"),
    ("\\vee", "∨"),
    ("\\oplus", "⊕"),
    ("\\otimes", "⊗"),
    ("\\mapsto", "↦"),
    ("\\%", "%"),
    ("\\Delta", "Δ"),
    ("\\Sigma", "Σ"),
    ("\\Omega", "Ω"),
    ("\\Theta", "Θ"),
    ("\\Lambda", "Λ"),
    ("\\Phi", "Φ"),
    ("\\Pi", "Π"),
    ("\\alpha", "α"),
    ("\\beta", "β"),
    ("\\gamma", "γ"),
    ("\\delta", "δ"),
    ("\\epsilon", "ε"),
    ("\\theta", "θ"),
    ("\\lambda", "λ"),
    ("\\mu", "μ"),
    ("\\pi", "π"),
    ("\\rho", "ρ"),
    ("\\sigma", "σ"),
    ("\\tau", "τ"),
    ("\\varphi", "φ"),
    ("\\phi", "φ"),
    ("\\psi", "ψ"),
    ("\\chi", "χ"),
    ("\\xi", "ξ"),
    ("\\eta", "η"),
    ("\\zeta", "ζ"),
    ("\\nu", "ν"),
    ("\\kappa", "κ"),
    ("\\omega", "ω"),
    // Function/operator names render as their plain word (no 2D layout needed).
    ("\\log", "log"),
    ("\\ln", "ln"),
    ("\\exp", "exp"),
    ("\\sin", "sin"),
    ("\\cos", "cos"),
    ("\\tan", "tan"),
    ("\\cot", "cot"),
    ("\\sec", "sec"),
    ("\\csc", "csc"),
    ("\\sinh", "sinh"),
    ("\\cosh", "cosh"),
    ("\\tanh", "tanh"),
    ("\\arg", "arg"),
    ("\\deg", "deg"),
    ("\\det", "det"),
    ("\\dim", "dim"),
    ("\\gcd", "gcd"),
    ("\\lim", "lim"),
    ("\\max", "max"),
    ("\\min", "min"),
    ("\\quad", "  "),
    ("\\,", " "),
    ("\\;", " "),
    ("\\!", ""),
];

/// The balanced `{...}` group at byte index `open` (which must be `{`): its inner
/// text and the index just past the closing `}`. `None` if unbalanced.
fn brace_group(s: &str, open: usize) -> Option<(String, usize)> {
    if s.as_bytes().get(open) != Some(&b'{') {
        return None;
    }
    let mut depth = 0usize;
    for (j, c) in s[open..].char_indices() {
        match c {
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    return Some((s[open + 1..open + j].to_string(), open + j + 1));
                }
            }
            _ => {}
        }
    }
    None
}

/// Expand the brace commands LLMs lean on: `\frac{a}{b}` → `a/b`, `\sqrt{x}` →
/// `√x`, and the wrappers (`\text`, `\boxed`, `\mathrm`, …) → their content.
/// Other `\name` sequences are left intact for the symbol table.
fn expand_latex_braces(s: &str) -> String {
    const WRAPPERS: &[&str] = &[
        "\\text",
        "\\boxed",
        "\\mathrm",
        "\\mathbf",
        "\\mathit",
        "\\operatorname",
    ];
    let mut out = String::with_capacity(s.len());
    let mut rest = s;
    loop {
        let Some(pos) = rest.find('\\') else {
            out.push_str(rest);
            return out;
        };
        out.push_str(&rest[..pos]);
        let after = &rest[pos..];
        if after.starts_with("\\frac") {
            if let Some((num, e1)) = brace_group(rest, pos + 5) {
                if let Some((den, e2)) = brace_group(rest, e1) {
                    out.push_str(&format!(
                        "{}/{}",
                        expand_latex_braces(&num),
                        expand_latex_braces(&den)
                    ));
                    rest = &rest[e2..];
                    continue;
                }
            }
        } else if after.starts_with("\\sqrt") {
            if let Some((inner, e)) = brace_group(rest, pos + 5) {
                out.push_str(&format!("√{}", expand_latex_braces(&inner)));
                rest = &rest[e..];
                continue;
            }
        } else if let Some(cmd) = WRAPPERS.iter().find(|c| after.starts_with(**c)) {
            if let Some((inner, e)) = brace_group(rest, pos + cmd.len()) {
                out.push_str(&expand_latex_braces(&inner));
                rest = &rest[e..];
                continue;
            }
        }
        // Not a brace command we expand: emit the backslash and move on (the
        // symbol table handles `\pi`, `\times`, … on the result).
        out.push('\\');
        rest = &rest[pos + 1..];
    }
}

/// Turn LLM LaTeX into readable Unicode (it cannot be *typeset* in a line-based
/// terminal, so this is best-effort prettifying, not layout): drop the math
/// delimiters, expand brace commands, map symbols/Greek, and split `\\` math
/// line-breaks. Fenced code blocks are passed through untouched so code is never
/// mangled.
/// How `x^2` / `s_j` render: as Unicode super/subscript glyphs (terminals
/// carry them) or as plain `^(…)`/`_(…)` (for hosts whose embedded fonts
/// lack the script blocks — egui shows tofu for `⁻ˣ`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ScriptStyle {
    Unicode,
    Plain,
}

/// Super/subscripts map to their Unicode forms (see [`prettify_math_plain_scripts`]).
pub fn prettify_math(src: &str) -> String {
    prettify(src, ScriptStyle::Unicode)
}

/// [`prettify_math`] with super/subscripts kept plain (`e^(-x)`, `s_(j)`)
/// — for hosts whose fonts would render the Unicode script glyphs as tofu.
pub fn prettify_math_plain_scripts(src: &str) -> String {
    prettify(src, ScriptStyle::Plain)
}

fn prettify(src: &str, scripts: ScriptStyle) -> String {
    let mut out = String::with_capacity(src.len());
    let mut in_fence = false;
    for line in src.split_inclusive('\n') {
        if line.trim_start().starts_with("```") {
            in_fence = !in_fence;
            out.push_str(line);
        } else if in_fence {
            out.push_str(line);
        } else {
            out.push_str(&prettify_math_line(line, scripts));
        }
    }
    out
}

/// Remove `\begin{env}` / `\end{env}` markers (tolerating a space before the
/// brace, as some models emit `\end {pmatrix}`). A line-based terminal can't lay
/// out the 2D environment they delimit (a matrix, an `aligned` block), but
/// dropping the wrappers and the `&` column separators leaves the cell contents
/// readable inline rather than as raw `\begin{…}` noise.
fn strip_environments(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut rest = s;
    loop {
        let begin = rest.find("\\begin");
        let end = rest.find("\\end");
        let (pos, cmd_len) = match (begin, end) {
            (Some(b), Some(e)) if b <= e => (b, 6),
            (Some(_), Some(e)) => (e, 4),
            (Some(b), None) => (b, 6),
            (None, Some(e)) => (e, 4),
            (None, None) => {
                out.push_str(rest);
                return out;
            }
        };
        out.push_str(&rest[..pos]);
        let mut j = pos + cmd_len;
        while rest[j..].starts_with(' ') {
            j += 1;
        }
        // Drop the `{env}` argument if present, else just the command token.
        rest = match brace_group(rest, j) {
            Some((_, after)) => &rest[after..],
            None => &rest[pos + cmd_len..],
        };
    }
}

fn prettify_math_line(line: &str, scripts: ScriptStyle) -> String {
    let mut s = strip_environments(line);
    for d in ["\\[", "\\]", "\\(", "\\)", "\\left", "\\right", "$$"] {
        s = s.replace(d, "");
    }
    s = s.replace("\\\\", "; "); // a row/line break becomes an inline separator
    s = s.replace(" & ", "  "); // matrix/align column separator → spacing
    s = expand_latex_braces(&s);
    for (from, to) in LATEX_SYMBOLS {
        s = s.replace(from, to);
    }
    sub_superscripts(&s, scripts)
}

/// The Unicode super/subscript for `c`, if one exists. Digits, signs and the
/// letters with established sub/superscript forms are covered; anything else
/// returns `None` so the caller can fall back to literal `^…`/`_…`.
fn script_char(c: char, sup: bool) -> Option<char> {
    // Capital letters have superscript forms (most of them) but no subscript
    // forms in Unicode, so they're only mapped when `sup`.
    const SUPER_CAPS: &[(char, char)] = &[
        ('A', 'ᴬ'),
        ('B', 'ᴮ'),
        ('D', 'ᴰ'),
        ('E', 'ᴱ'),
        ('G', 'ᴳ'),
        ('H', 'ᴴ'),
        ('I', 'ᴵ'),
        ('J', 'ᴶ'),
        ('K', 'ᴷ'),
        ('L', 'ᴸ'),
        ('M', 'ᴹ'),
        ('N', 'ᴺ'),
        ('O', 'ᴼ'),
        ('P', 'ᴾ'),
        ('R', 'ᴿ'),
        ('T', 'ᵀ'),
        ('U', 'ᵁ'),
        ('V', 'ⱽ'),
        ('W', 'ᵂ'),
    ];
    if sup {
        if let Some((_, hi)) = SUPER_CAPS.iter().find(|(base, _)| *base == c) {
            return Some(*hi);
        }
    }
    let table: &[(char, char, char)] = &[
        // (base, superscript, subscript)
        ('0', '⁰', '₀'),
        ('1', '¹', '₁'),
        ('2', '²', '₂'),
        ('3', '³', '₃'),
        ('4', '⁴', '₄'),
        ('5', '⁵', '₅'),
        ('6', '⁶', '₆'),
        ('7', '⁷', '₇'),
        ('8', '⁸', '₈'),
        ('9', '⁹', '₉'),
        ('+', '⁺', '₊'),
        ('-', '⁻', '₋'),
        ('=', '⁼', '₌'),
        ('(', '⁽', '₍'),
        (')', '⁾', '₎'),
        ('a', 'ᵃ', 'ₐ'),
        ('e', 'ᵉ', 'ₑ'),
        ('h', 'ʰ', 'ₕ'),
        ('i', 'ⁱ', 'ᵢ'),
        ('j', 'ʲ', 'ⱼ'),
        ('k', 'ᵏ', 'ₖ'),
        ('l', 'ˡ', 'ₗ'),
        ('m', 'ᵐ', 'ₘ'),
        ('n', 'ⁿ', 'ₙ'),
        ('o', 'ᵒ', 'ₒ'),
        ('p', 'ᵖ', 'ₚ'),
        ('r', 'ʳ', 'ᵣ'),
        ('s', 'ˢ', 'ₛ'),
        ('t', 'ᵗ', 'ₜ'),
        ('u', 'ᵘ', 'ᵤ'),
        ('v', 'ᵛ', 'ᵥ'),
        ('x', 'ˣ', 'ₓ'),
    ];
    table
        .iter()
        .find(|(base, _, _)| *base == c)
        .map(|(_, hi, lo)| if sup { *hi } else { *lo })
}

/// Render every char of `s` in super/subscript form, or `None` if any char has
/// no such form (so a mixed group stays legible rather than half-converted).
fn script_run(s: &str, sup: bool) -> Option<String> {
    s.chars().map(|c| script_char(c, sup)).collect()
}

/// Convert `x^2`, `s_j`, `^{N}`, `_{j=k+1}` to Unicode super/subscripts. A group
/// that can't be fully mapped is left as `^(…)` / `_(…)` so it stays readable.
fn sub_superscripts(s: &str, style: ScriptStyle) -> String {
    let mut out = String::with_capacity(s.len());
    let mut i = 0;
    while i < s.len() {
        let c = s[i..].chars().next().unwrap();
        if c == '^' || c == '_' {
            let sup = c == '^';
            let rest = &s[i + 1..];
            if rest.starts_with('{') {
                if let Some((inner, end)) = brace_group(s, i + 1) {
                    let mapped = match style {
                        ScriptStyle::Unicode => script_run(&inner, sup),
                        ScriptStyle::Plain => None,
                    };
                    match mapped {
                        Some(mapped) => out.push_str(&mapped),
                        None => {
                            out.push(c);
                            out.push('(');
                            out.push_str(&inner);
                            out.push(')');
                        }
                    }
                    i = end;
                    continue;
                }
            } else if let Some(first) = rest.chars().next() {
                // Identifier guard: `read_page`, `foo_bar` are snake_case, not
                // subscripts — leave `_x` alone when the run continues as a
                // word (the next char is alphanumeric or another underscore).
                let word_continues = !sup
                    && rest
                        .chars()
                        .nth(1)
                        .is_some_and(|next| next.is_ascii_alphanumeric() || next == '_');
                if !word_continues && style == ScriptStyle::Unicode {
                    if let Some(m) = script_char(first, sup) {
                        out.push(m);
                        i += 1 + first.len_utf8();
                        continue;
                    }
                }
            }
            out.push(c);
            i += 1;
        } else {
            out.push(c);
            i += c.len_utf8();
        }
    }
    out
}

/// The user's speaker label: their login name (`$USER`), falling back to
/// "you". Resolved once — it cannot change mid-session. Every frontend
/// labels the user's turns with it (one policy, each view's own
/// typography).
pub fn user_label() -> &'static str {
    static LABEL: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    LABEL.get_or_init(|| label_from(std::env::var("USER").ok()))
}

/// The pure half of [`user_label`], testable without touching the process
/// environment.
fn label_from(user: Option<String>) -> String {
    user.filter(|u| !u.trim().is_empty())
        .unwrap_or_else(|| "you".to_string())
}

/// Rewrite markdown image tags in a model answer for hosts that cannot (or
/// deliberately will not) load them. `![alt](file://…)` — a model echoing an
/// artifact the host already displays inline — is dropped. Any other
/// `![alt](url)` becomes the plain link `[alt](url)`: frontends never fetch
/// remote URLs the model wrote (that would bypass the capability doctrine),
/// so rendering a broken-image glyph would be noise where a clickable link
/// is honest.
/// Lift fenced code blocks out of list items to the top level: dedent the
/// fence and its body and surround the block with blank lines. Rendering
/// motive: egui_commonmark lays a fence nested in a bullet as a
/// continuation of the bullet's horizontal row, so the block lands to the
/// RIGHT of the text and inherits only leftover width — at zero leftover
/// it collapses to a one-token-per-line sliver (observed live). Splitting
/// the list around a top-level block is the accepted trade. Tolerant of
/// an unclosed (still-streaming) fence.
pub fn lift_fenced_code_blocks(text: &str) -> String {
    let mut out: Vec<String> = Vec::new();
    let mut fence_indent: Option<usize> = None;
    let mut blank_after_close = false;
    let mut in_quote = false;
    for line in text.lines() {
        let trimmed = line.trim_start();
        let indent = line.len() - trimmed.len();
        if blank_after_close {
            if !trimmed.is_empty() {
                out.push(String::new());
            }
            blank_after_close = false;
        }
        // A blockquote nested under a list item (indented `>`) hits the
        // same egui_commonmark right-shift as a nested fence — lift the
        // whole `>` run to the top level with blank-line separation.
        if fence_indent.is_none() {
            let nested_quote = indent > 0 && trimmed.starts_with('>');
            if nested_quote {
                if !in_quote && out.last().is_some_and(|prev| !prev.trim().is_empty()) {
                    out.push(String::new());
                }
                in_quote = true;
                out.push(trimmed.to_string());
                continue;
            }
            if in_quote {
                in_quote = false;
                if !trimmed.is_empty() {
                    out.push(String::new());
                }
            }
        }
        match fence_indent {
            None if trimmed.starts_with("```") => {
                if out.last().is_some_and(|prev| !prev.trim().is_empty()) {
                    out.push(String::new());
                }
                out.push(trimmed.to_string());
                fence_indent = Some(indent);
            }
            Some(opened) if trimmed.starts_with("```") => {
                let _ = opened;
                out.push(trimmed.to_string());
                blank_after_close = true;
                fence_indent = None;
            }
            Some(opened) => {
                let strip = opened.min(indent);
                out.push(line[strip..].to_string());
            }
            None => out.push(line.to_string()),
        }
    }
    out.join("\n")
}

/// Rewrite a fenced block's language tag to the file *extension* a syntax
/// highlighter resolves. egui_commonmark (and syntect beneath it) look a
/// fence's language up with `find_syntax_by_extension`, not by name: the
/// registered extension for Haskell is `hs`, for Rust `rs`, for Python `py`.
/// A model's natural ```haskell / ```rust / ```python tag therefore matches
/// nothing and the block renders unhighlighted. Only the *opening* fence's
/// first token is rewritten, and only for names whose extension differs from
/// the name; tags that already are extensions (`hs`, `rs`), tags syntect
/// accepts by name (`bash`, `json`, `cpp`, `yaml`, `go`, `java`, …), unknown
/// tags, bare fences, and every content line pass through byte-for-byte.
pub fn canonicalize_code_fence_langs(text: &str) -> String {
    // name → extension, only where the language *name* fails an extension
    // lookup but the extension resolves (verified against syntect's default
    // syntax set). Names that already work as extensions are absent by design.
    fn ext_for(lang: &str) -> Option<&'static str> {
        Some(match lang.to_ascii_lowercase().as_str() {
            "haskell" => "hs",
            "rust" => "rs",
            "python" | "python3" => "py",
            "javascript" | "node" => "js",
            "ruby" => "rb",
            "csharp" | "c#" => "cs",
            "shell" => "sh",
            "text" | "plaintext" => "txt",
            "golang" => "go",
            "ocaml" => "ml",
            "clojure" => "clj",
            "erlang" => "erl",
            "perl" => "pl",
            _ => return None,
        })
    }
    let mut out: Vec<String> = Vec::with_capacity(text.lines().count());
    let mut in_fence = false;
    for line in text.lines() {
        let trimmed = line.trim_start();
        if !trimmed.starts_with("```") {
            out.push(line.to_string());
            continue;
        }
        if in_fence {
            // A fence while open closes the block; never a language tag.
            in_fence = false;
            out.push(line.to_string());
            continue;
        }
        in_fence = true;
        // Opening fence: keep the indent and the exact backtick run, rewrite
        // only the leading language token when it maps to an extension.
        let indent = &line[..line.len() - trimmed.len()];
        let ticks = trimmed.len() - trimmed.trim_start_matches('`').len();
        let info = trimmed[ticks..].trim_start();
        let (lang, rest) = match info.find(char::is_whitespace) {
            Some(i) => (&info[..i], &info[i..]),
            None => (info, ""),
        };
        match ext_for(lang) {
            Some(ext) => out.push(format!("{indent}{}{ext}{rest}", "`".repeat(ticks))),
            None => out.push(line.to_string()),
        }
    }
    out.join("\n")
}

pub fn tame_markdown_images(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(bang) = rest.find("![") {
        let Some(mid) = rest[bang..].find("](") else {
            break;
        };
        let Some(close) = rest[bang + mid..].find(')') else {
            break;
        };
        let alt = &rest[bang + 2..bang + mid];
        let url = &rest[bang + mid + 2..bang + mid + close];
        out.push_str(&rest[..bang]);
        if !url.trim_start().starts_with("file://") {
            out.push('[');
            out.push_str(if alt.trim().is_empty() { "image" } else { alt });
            out.push_str("](");
            out.push_str(url);
            out.push(')');
        }
        rest = &rest[bang + mid + close + 1..];
    }
    out.push_str(rest);
    out
}

/// Rewrite markdown image tags for hosts that render *no* markdown. Where
/// [`tame_markdown_images`] link-ifies for markdown-rendering hosts, the
/// plain-text rules are:
///
/// - a **local/relative** target (`file://`, `./…`, anything schemeless):
///   the artifact the model is citing already arrived inline as bytes, so
///   the tag reduces to its alt text — or to nothing (an unclickable path
///   is noise);
/// - a **short remote** target (http/https, ≤ 100 chars): kept as literal
///   `[alt](url)` text — the model is pointing at an image it did *not*
///   fetch, and silently deleting the reference leaves its prose full of
///   empty promises (observed live: a numbered list of Sierpiński images,
///   every item blank);
/// - a **long remote** target: alt only — the four-hundred-character
///   signed-URL wall, also observed live, teaches nothing.
pub fn strip_markdown_images(text: &str) -> String {
    const KEEP_URL_MAX: usize = 100;
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    let mut stripped = false;
    while let Some(bang) = rest.find("![") {
        let Some(mid) = rest[bang..].find("](") else {
            break;
        };
        let Some(close) = rest[bang + mid..].find(')') else {
            break;
        };
        let alt = rest[bang + 2..bang + mid].trim();
        let url = rest[bang + mid + 2..bang + mid + close].trim();
        out.push_str(&rest[..bang]);
        let remote = url.starts_with("http://") || url.starts_with("https://");
        if remote && url.len() <= KEEP_URL_MAX {
            out.push('[');
            out.push_str(if alt.is_empty() { "image" } else { alt });
            out.push_str("](");
            out.push_str(url);
            out.push(')');
        } else {
            out.push_str(alt);
            stripped = true;
        }
        rest = &rest[bang + mid + close + 1..];
    }
    out.push_str(rest);
    if !stripped {
        return out;
    }
    // A tag that sat alone on its line leaves a blank hole where the image
    // "was" (observed live: an empty gap mid-answer) — collapse runs of
    // blank lines down to one, a paragraph break. Only a stripped answer is
    // reflowed; untouched text passes through byte-identical above.
    let mut result = String::with_capacity(out.len());
    let mut blanks = 0;
    for line in out.lines() {
        if line.trim().is_empty() {
            blanks += 1;
            if blanks > 1 {
                continue;
            }
        } else {
            blanks = 0;
        }
        result.push_str(line);
        result.push('\n');
    }
    if !out.ends_with('\n') {
        result.pop();
    }
    result
}

#[cfg(test)]
mod tests {
    #[test]
    fn fenced_blocks_lift_out_of_list_items() {
        let nested = "- bullet text\n  ```rust\n  fn f() {}\n  ```\n- next";
        assert_eq!(
            super::lift_fenced_code_blocks(nested),
            "- bullet text\n\n```rust\nfn f() {}\n```\n\n- next"
        );
        let top = "para\n\n```rust\nfn f() {}\n```\n\nafter";
        assert_eq!(super::lift_fenced_code_blocks(top), top);
        let streaming = "- item\n  ```rust\n  let x = 1;";
        assert_eq!(
            super::lift_fenced_code_blocks(streaming),
            "- item\n\n```rust\nlet x = 1;"
        );
        // A blockquote nested under a bullet lifts the same way; a
        // top-level quote is left alone.
        let quoted = "- point\n  > a cited line\n- next";
        assert_eq!(
            super::lift_fenced_code_blocks(quoted),
            "- point\n\n> a cited line\n\n- next"
        );
        let top_quote = "> already top\n\nafter";
        assert_eq!(super::lift_fenced_code_blocks(top_quote), top_quote);
    }

    use super::*;

    #[test]
    fn code_fence_langs_canonicalize_to_syntect_extensions() {
        // The names a model actually writes resolve only by extension, so the
        // opening tag is rewritten to the extension syntect registers.
        assert_eq!(
            canonicalize_code_fence_langs("```haskell\njoinSlot :: Slot\n```"),
            "```hs\njoinSlot :: Slot\n```"
        );
        assert_eq!(
            canonicalize_code_fence_langs("```rust\nfn f() {}\n```"),
            "```rs\nfn f() {}\n```"
        );
        assert_eq!(
            canonicalize_code_fence_langs("```python\nx = 1\n```"),
            "```py\nx = 1\n```"
        );
        // Case-insensitive on the name; the extension is lowercase.
        assert_eq!(
            canonicalize_code_fence_langs("```Haskell\nx\n```"),
            "```hs\nx\n```"
        );
        // Tags that already resolve by name pass through untouched, as do
        // unknown tags and bare fences.
        for keep in [
            "```bash\nls\n```",
            "```json\n{}\n```",
            "```hs\nx\n```",
            "```brainfuck\n+++\n```",
            "```\nplain\n```",
        ] {
            assert_eq!(canonicalize_code_fence_langs(keep), keep);
        }
        // A fence-like line *inside* a block is the closing fence, never a
        // language tag: a ```haskell that opens must not be re-treated on close.
        assert_eq!(
            canonicalize_code_fence_langs("```haskell\ncode\n```\n```rust\nmore\n```"),
            "```hs\ncode\n```\n```rs\nmore\n```"
        );
        // Trailing attributes after the language token survive.
        assert_eq!(
            canonicalize_code_fence_langs("```rust ignore\nfn f() {}\n```"),
            "```rs ignore\nfn f() {}\n```"
        );
    }

    #[test]
    fn stripped_images_reduce_to_their_alt_or_nothing() {
        // The plain-text twin of `tame_markdown_images`. Local tags vanish
        // (the artifact is already inline), prose around them untouched —
        // including the observed live case, an empty-alt image whose
        // signed-URL target ran past the keep cap.
        let wall = format!(
            "![](http://localhost:11111/plot.png?Expires=1&Signature={}) Here is the plot.",
            "X".repeat(200)
        );
        assert_eq!(strip_markdown_images(&wall), " Here is the plot.");
        assert_eq!(
            strip_markdown_images("see ![the chart](./plots/img-1.png), above"),
            "see the chart, above"
        );
        // A *short remote* target the model never fetched survives as
        // literal link text (observed live: a numbered list of images,
        // every item stripped blank — empty promises teach nothing).
        assert_eq!(
            strip_markdown_images("1. The basic one: ![basic](https://u.example/tri.gif)"),
            "1. The basic one: [basic](https://u.example/tri.gif)"
        );
        assert_eq!(strip_markdown_images("no images here"), "no images here");
        // A malformed tag (no closing paren) passes through untouched.
        assert_eq!(
            strip_markdown_images("dangling ![alt](oops"),
            "dangling ![alt](oops"
        );
        // A tag alone on its line takes its blank hole with it (observed
        // live as an empty gap mid-answer); a paragraph break survives.
        assert_eq!(
            strip_markdown_images("Here is the plot:\n\n![](./img/f.png)\n\nThe x-axis is n."),
            "Here is the plot:\n\nThe x-axis is n."
        );
    }

    #[test]
    fn plain_scripts_never_emit_script_glyphs() {
        // The GUI's fonts lack the Unicode script blocks (⁻ˣ is tofu), so
        // its variant renders exponents plainly — readable everywhere.
        assert_eq!(prettify_math_plain_scripts("\\( e^{-x} \\)"), " e^(-x) ");
        assert_eq!(prettify_math_plain_scripts("x^2"), "x^2");
        assert_eq!(prettify_math("\\( e^{-x} \\)"), " e⁻ˣ ");
    }

    #[test]
    fn user_label_is_the_login_name_or_you() {
        // The label policy shared by every frontend: $USER when it carries a
        // name, "you" when unset or blank.
        assert_eq!(label_from(Some("shayne".to_string())), "shayne");
        assert_eq!(label_from(Some("  ".to_string())), "you");
        assert_eq!(label_from(None), "you");
    }

    #[test]
    fn markdown_images_tame_to_links_or_vanish() {
        // A file:// echo of an inline artifact drops; a remote image becomes
        // an honest link (frontends never fetch model-written URLs); prose
        // around them survives byte-for-byte.
        assert_eq!(
            tame_markdown_images("see ![tri](file:///tmp/a.png) here"),
            "see  here"
        );
        assert_eq!(
            tame_markdown_images("see ![cube](https://x.example/c.svg) here"),
            "see [cube](https://x.example/c.svg) here"
        );
        assert_eq!(
            tame_markdown_images("![](https://x.example/c.svg)"),
            "[image](https://x.example/c.svg)"
        );
        assert_eq!(tame_markdown_images("no images at all"), "no images at all");
        // Malformed tags pass through rather than eating the answer.
        assert_eq!(tame_markdown_images("a ![dangling"), "a ![dangling");
    }

    #[test]
    fn subscripts_map_math_but_spare_snake_case_identifiers() {
        // Math subscripts still map…
        assert_eq!(sub_superscripts("x_i", ScriptStyle::Unicode), "xᵢ");
        assert_eq!(
            sub_superscripts("a_1 + a_2", ScriptStyle::Unicode),
            "a₁ + a₂"
        );
        // …but snake_case identifiers pass through untouched (a tool name in an
        // activity line must not become `readₚage`).
        assert_eq!(
            sub_superscripts("read_page", ScriptStyle::Unicode),
            "read_page"
        );
        assert_eq!(
            sub_superscripts("foo_bar_baz", ScriptStyle::Unicode),
            "foo_bar_baz"
        );
        assert_eq!(
            sub_superscripts("max_tokens=42", ScriptStyle::Unicode),
            "max_tokens=42"
        );
    }

    #[test]
    fn prettify_math_renders_llm_latex_readably() {
        // The constructs from a real QwQ answer: display delimiters, `\\` math
        // line-breaks, \text, \frac, \boxed, and an operator.
        let raw = "\\[ 60t + 60 = 80t \\\\ 60 = 20t \\\\ t = 3 \\]";
        let got = prettify_math(raw);
        assert!(
            !got.contains("\\["),
            "display delimiters are dropped: {got}"
        );
        assert!(!got.contains("\\\\"), "math line-breaks are gone: {got}");
        assert!(got.contains("60 = 20t"), "the math survives: {got}");

        assert_eq!(
            prettify_math("\\frac{60 \\text{ miles}}{20 \\text{ mph}}"),
            "60  miles/20  mph"
        );
        assert_eq!(prettify_math("\\boxed{7} PM"), "7 PM");
        assert_eq!(prettify_math("a \\times b \\leq c"), "a × b ≤ c");
        assert_eq!(prettify_math("\\sqrt{2} \\approx 1.41"), "√2 ≈ 1.41");
        assert_eq!(prettify_math("\\pi r^2 \\theta"), "π r² θ");
    }

    #[test]
    fn prettify_math_handles_scripts_floors_and_mod() {
        // The constructs from a reasoning trace: subscripts, superscripts, the
        // floor brackets, and \mod (with \left/\right dropped).
        assert_eq!(prettify_math("s_j and c_k"), "sⱼ and cₖ");
        assert_eq!(prettify_math("\\prod_{j=k+1}^N s_j"), "∏ⱼ₌ₖ₊₁ᴺ sⱼ");
        assert_eq!(
            prettify_math("\\left\\lfloor \\frac{r}{s_k} \\right\\rfloor"),
            "⌊ r/sₖ ⌋"
        );
        assert_eq!(prettify_math("r \\mod s_k"), "r mod sₖ");
        // A subscript with an unmappable char stays legible, not half-converted.
        assert_eq!(prettify_math("x_{QQ}"), "x_(QQ)");
    }

    #[test]
    fn prettify_math_strips_environments_and_functions() {
        // \begin/\end wrappers (even with a stray space) and `&` separators go;
        // the cell contents survive inline. Function names render as words.
        assert_eq!(
            prettify_math("\\begin{pmatrix} 1 & 1 ; 1 & 0 \\end {pmatrix}"),
            " 1  1 ; 1  0 "
        );
        assert_eq!(prettify_math("O(\\log n)"), "O(log n)");
        assert_eq!(prettify_math("\\psi^n / \\sqrt{5}"), "ψⁿ / √5");
    }

    #[test]
    fn prettify_math_leaves_fenced_code_untouched() {
        let raw = "before\n```\nlet x = a \\\\ b; // \\frac stays\n```\nafter";
        let got = prettify_math(raw);
        assert!(
            got.contains("a \\\\ b") && got.contains("\\frac stays"),
            "code fence is passed through verbatim: {got}"
        );
    }
}
