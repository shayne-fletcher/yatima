//! The plot tool's expression language: a **closed grammar** over functions
//! of one variable, parsed and evaluated host-side in Rust (PLOT-1 — the
//! model states symbolic intent like `sin(x) * exp(-x/10)`; no model-authored
//! code ever reaches an interpreter, and the sandbox's python still receives
//! only literal arrays).
//!
//! Grammar (whitespace-insensitive):
//!
//! ```text
//! expr   := term  (('+' | '-') term)*
//! term   := unary (('*' | '/') unary)*
//! unary  := '-' unary | power
//! power  := atom  ('^' unary)?               -- right-associative
//! atom   := number | 'x' | 'pi' | 'e'
//!         | func '(' expr ')' | '(' expr ')'
//! func   := sin | cos | tan | sinh | cosh | tanh | asin | acos | atan
//!         | exp | ln | log | log10 | log2 | sqrt | abs
//!         | floor | ceil | round | sign      -- 'log' is an alias for 'ln'
//! number := digits ('.' digits)?             -- no exponent form: 'e' is Euler
//! ```
//!
//! Precedence follows convention: `-2^2 = -(2^2)` and `2^3^2 = 2^(3^2)`.
//! Constants fold at parse (`pi` and `e` become numbers), so the AST is
//! closed under [`std::fmt::Display`]: `parse ∘ print = id`, property-tested
//! below. Input length and nesting depth are bounded, so the parser is total:
//! any string yields an [`Expr`] or an error naming the alphabet, never a
//! panic or a runaway.

use std::fmt;

use anyhow::{anyhow, bail, Result};

/// One line of teaching, embedded in every rejection: the model reads tool
/// errors as instructions, so a rejection must name the legal alphabet.
pub(crate) const ALPHABET: &str = "the grammar is numbers, x, pi, e, \
     + - * / ^, parentheses, and sin cos tan sinh cosh tanh asin acos atan \
     exp ln log log10 log2 sqrt abs floor ceil round sign";

/// Longest accepted source text. Expressions are short by nature; anything
/// longer is data trying to be code.
pub(crate) const MAX_LEN: usize = 256;

/// Deepest accepted nesting — bounds parser recursion (totality).
const MAX_DEPTH: usize = 64;

/// A binary operator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Op {
    Add,
    Sub,
    Mul,
    Div,
    Pow,
}

/// A whitelisted function — the entire function alphabet. (`log` lexes as an
/// alias for [`Func::Ln`]: python-trained models mean `math.log`.)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Func {
    Sin,
    Cos,
    Tan,
    Sinh,
    Cosh,
    Tanh,
    Asin,
    Acos,
    Atan,
    Exp,
    Ln,
    Log10,
    Log2,
    Sqrt,
    Abs,
    Floor,
    Ceil,
    Round,
    Sign,
}

impl Func {
    fn name(self) -> &'static str {
        match self {
            Func::Sin => "sin",
            Func::Cos => "cos",
            Func::Tan => "tan",
            Func::Sinh => "sinh",
            Func::Cosh => "cosh",
            Func::Tanh => "tanh",
            Func::Asin => "asin",
            Func::Acos => "acos",
            Func::Atan => "atan",
            Func::Exp => "exp",
            Func::Ln => "ln",
            Func::Log10 => "log10",
            Func::Log2 => "log2",
            Func::Sqrt => "sqrt",
            Func::Abs => "abs",
            Func::Floor => "floor",
            Func::Ceil => "ceil",
            Func::Round => "round",
            Func::Sign => "sign",
        }
    }
}

/// A parsed expression in one variable `x`.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Expr {
    Num(f64),
    X,
    Neg(Box<Expr>),
    Bin(Op, Box<Expr>, Box<Expr>),
    Call(Func, Box<Expr>),
}

impl Expr {
    /// Evaluate at `x`. Total over f64: domain edges yield NaN/±inf (e.g.
    /// `ln(0)`), which the caller rejects with a range-teaching error.
    pub(crate) fn eval(&self, x: f64) -> f64 {
        match self {
            Expr::Num(n) => *n,
            Expr::X => x,
            Expr::Neg(e) => -e.eval(x),
            Expr::Bin(op, l, r) => {
                let (l, r) = (l.eval(x), r.eval(x));
                match op {
                    Op::Add => l + r,
                    Op::Sub => l - r,
                    Op::Mul => l * r,
                    Op::Div => l / r,
                    Op::Pow => l.powf(r),
                }
            }
            Expr::Call(f, a) => {
                let a = a.eval(x);
                match f {
                    Func::Sin => a.sin(),
                    Func::Cos => a.cos(),
                    Func::Tan => a.tan(),
                    Func::Sinh => a.sinh(),
                    Func::Cosh => a.cosh(),
                    Func::Tanh => a.tanh(),
                    Func::Asin => a.asin(),
                    Func::Acos => a.acos(),
                    Func::Atan => a.atan(),
                    Func::Exp => a.exp(),
                    Func::Ln => a.ln(),
                    Func::Log10 => a.log10(),
                    Func::Log2 => a.log2(),
                    Func::Sqrt => a.sqrt(),
                    Func::Abs => a.abs(),
                    Func::Floor => a.floor(),
                    Func::Ceil => a.ceil(),
                    Func::Round => a.round(),
                    // Mathematical sgn: sign(0) = 0 (f64::signum gives ±1
                    // for signed zeros, which is not the function models
                    // mean).
                    Func::Sign => {
                        if a == 0.0 {
                            0.0
                        } else {
                            a.signum()
                        }
                    }
                }
            }
        }
    }

    /// Whether the expression mentions `x` — a range bound must not.
    pub(crate) fn references_x(&self) -> bool {
        match self {
            Expr::Num(_) => false,
            Expr::X => true,
            Expr::Neg(e) | Expr::Call(_, e) => e.references_x(),
            Expr::Bin(_, l, r) => l.references_x() || r.references_x(),
        }
    }
}

/// Fully parenthesized print — structure is explicit, so parsing it back
/// reproduces the AST exactly (the roundtrip law below).
impl fmt::Display for Expr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Expr::Num(n) => write!(f, "{n}"),
            Expr::X => write!(f, "x"),
            Expr::Neg(e) => write!(f, "(-{e})"),
            Expr::Bin(op, l, r) => {
                let op = match op {
                    Op::Add => "+",
                    Op::Sub => "-",
                    Op::Mul => "*",
                    Op::Div => "/",
                    Op::Pow => "^",
                };
                write!(f, "({l} {op} {r})")
            }
            Expr::Call(g, a) => write!(f, "{}({a})", g.name()),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum Tok {
    Num(f64),
    X,
    Func(Func),
    Plus,
    Minus,
    Star,
    Slash,
    Caret,
    Open,
    Close,
}

impl fmt::Display for Tok {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Tok::Num(n) => write!(f, "{n}"),
            Tok::X => write!(f, "x"),
            Tok::Func(g) => write!(f, "{}", g.name()),
            Tok::Plus => write!(f, "+"),
            Tok::Minus => write!(f, "-"),
            Tok::Star => write!(f, "*"),
            Tok::Slash => write!(f, "/"),
            Tok::Caret => write!(f, "^"),
            Tok::Open => write!(f, "("),
            Tok::Close => write!(f, ")"),
        }
    }
}

fn lex(src: &str) -> Result<Vec<Tok>> {
    let bytes = src.as_bytes();
    let mut toks = Vec::new();
    let mut i = 0;
    while i < bytes.len() {
        match bytes[i] {
            b' ' | b'\t' | b'\r' | b'\n' => i += 1,
            b'+' => {
                toks.push(Tok::Plus);
                i += 1;
            }
            b'-' => {
                toks.push(Tok::Minus);
                i += 1;
            }
            b'*' => {
                toks.push(Tok::Star);
                i += 1;
            }
            b'/' => {
                toks.push(Tok::Slash);
                i += 1;
            }
            b'^' => {
                toks.push(Tok::Caret);
                i += 1;
            }
            b'(' => {
                toks.push(Tok::Open);
                i += 1;
            }
            b')' => {
                toks.push(Tok::Close);
                i += 1;
            }
            b'0'..=b'9' | b'.' => {
                let start = i;
                while i < bytes.len() && (bytes[i].is_ascii_digit() || bytes[i] == b'.') {
                    i += 1;
                }
                let s = &src[start..i];
                let n: f64 = s
                    .parse()
                    .map_err(|_| anyhow!("plot expr: bad number {s:?} — {ALPHABET}"))?;
                if !n.is_finite() {
                    bail!("plot expr: number {s:?} overflows f64");
                }
                toks.push(Tok::Num(n));
            }
            b'a'..=b'z' => {
                // Names start with a letter and may end in digits (log10).
                let start = i;
                while i < bytes.len()
                    && (bytes[i].is_ascii_lowercase() || bytes[i].is_ascii_digit())
                {
                    i += 1;
                }
                toks.push(match &src[start..i] {
                    "x" => Tok::X,
                    "pi" => Tok::Num(std::f64::consts::PI),
                    "e" => Tok::Num(std::f64::consts::E),
                    "sin" => Tok::Func(Func::Sin),
                    "cos" => Tok::Func(Func::Cos),
                    "tan" => Tok::Func(Func::Tan),
                    "sinh" => Tok::Func(Func::Sinh),
                    "cosh" => Tok::Func(Func::Cosh),
                    "tanh" => Tok::Func(Func::Tanh),
                    "asin" => Tok::Func(Func::Asin),
                    "acos" => Tok::Func(Func::Acos),
                    "atan" => Tok::Func(Func::Atan),
                    "exp" => Tok::Func(Func::Exp),
                    "ln" | "log" => Tok::Func(Func::Ln),
                    "log10" => Tok::Func(Func::Log10),
                    "log2" => Tok::Func(Func::Log2),
                    "sqrt" => Tok::Func(Func::Sqrt),
                    "abs" => Tok::Func(Func::Abs),
                    "floor" => Tok::Func(Func::Floor),
                    "ceil" => Tok::Func(Func::Ceil),
                    "round" => Tok::Func(Func::Round),
                    "sign" => Tok::Func(Func::Sign),
                    name => bail!("plot expr: unknown name {name:?} — {ALPHABET}"),
                });
            }
            c => bail!("plot expr: unexpected {:?} — {ALPHABET}", char::from(c)),
        }
    }
    Ok(toks)
}

struct Parser<'a> {
    toks: &'a [Tok],
    pos: usize,
}

impl Parser<'_> {
    fn peek(&self) -> Option<Tok> {
        self.toks.get(self.pos).copied()
    }

    fn expect_close(&mut self) -> Result<()> {
        match self.peek() {
            Some(Tok::Close) => {
                self.pos += 1;
                Ok(())
            }
            Some(t) => bail!("plot expr: expected ')', found {t}"),
            None => bail!("plot expr: expected ')', found end of input"),
        }
    }

    fn expr(&mut self, depth: usize) -> Result<Expr> {
        let mut lhs = self.term(depth)?;
        while let Some(t) = self.peek() {
            let op = match t {
                Tok::Plus => Op::Add,
                Tok::Minus => Op::Sub,
                _ => break,
            };
            self.pos += 1;
            let rhs = self.term(depth)?;
            lhs = Expr::Bin(op, Box::new(lhs), Box::new(rhs));
        }
        Ok(lhs)
    }

    fn term(&mut self, depth: usize) -> Result<Expr> {
        let mut lhs = self.unary(depth)?;
        while let Some(t) = self.peek() {
            let op = match t {
                Tok::Star => Op::Mul,
                Tok::Slash => Op::Div,
                _ => break,
            };
            self.pos += 1;
            let rhs = self.unary(depth)?;
            lhs = Expr::Bin(op, Box::new(lhs), Box::new(rhs));
        }
        Ok(lhs)
    }

    fn unary(&mut self, depth: usize) -> Result<Expr> {
        if depth > MAX_DEPTH {
            bail!("plot expr: nested deeper than {MAX_DEPTH}");
        }
        if self.peek() == Some(Tok::Minus) {
            self.pos += 1;
            Ok(Expr::Neg(Box::new(self.unary(depth + 1)?)))
        } else {
            self.power(depth)
        }
    }

    fn power(&mut self, depth: usize) -> Result<Expr> {
        let base = self.atom(depth)?;
        if self.peek() == Some(Tok::Caret) {
            self.pos += 1;
            // Right-associative: 2^3^2 = 2^(3^2); exponent may be unary-negated.
            let exp = self.unary(depth + 1)?;
            Ok(Expr::Bin(Op::Pow, Box::new(base), Box::new(exp)))
        } else {
            Ok(base)
        }
    }

    fn atom(&mut self, depth: usize) -> Result<Expr> {
        let t = self.peek();
        self.pos += 1;
        match t {
            Some(Tok::Num(n)) => Ok(Expr::Num(n)),
            Some(Tok::X) => Ok(Expr::X),
            Some(Tok::Func(g)) => {
                match self.peek() {
                    Some(Tok::Open) => self.pos += 1,
                    _ => bail!(
                        "plot expr: {} needs parentheses, e.g. {}(x)",
                        g.name(),
                        g.name()
                    ),
                }
                let a = self.expr(depth + 1)?;
                self.expect_close()?;
                Ok(Expr::Call(g, Box::new(a)))
            }
            Some(Tok::Open) => {
                let e = self.expr(depth + 1)?;
                self.expect_close()?;
                Ok(e)
            }
            Some(t) => bail!("plot expr: unexpected {t} — {ALPHABET}"),
            None => bail!("plot expr: unexpected end of input — {ALPHABET}"),
        }
    }
}

/// Parse a source string against the closed grammar. Total: every input is
/// an [`Expr`] or a teaching error; length and depth are bounded.
pub(crate) fn parse(src: &str) -> Result<Expr> {
    if src.len() > MAX_LEN {
        bail!("plot expr: longer than {MAX_LEN} chars");
    }
    parse_inner(src)
}

/// [`parse`] without the length gate — the print/parse roundtrip law is about
/// grammar structure, and fully parenthesized prints of deep trees can
/// legitimately exceed the model-facing length cap.
fn parse_inner(src: &str) -> Result<Expr> {
    let toks = lex(src)?;
    let mut p = Parser {
        toks: &toks,
        pos: 0,
    };
    let e = p.expr(0)?;
    if p.pos != toks.len() {
        bail!(
            "plot expr: unexpected {} after a complete expression — {ALPHABET}",
            p.toks[p.pos]
        );
    }
    Ok(e)
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    use super::*;

    #[test]
    fn evaluates_the_conventional_algebra() {
        // upholds: PLOT-1 — the grammar means what mathematics means:
        // precedence, right-associative ^, unary minus below ^, constants.
        let cases = [
            ("sin(pi/2)", 0.0, 1.0),
            ("2+3*4^2", 0.0, 50.0),
            ("2^3^2", 0.0, 512.0),
            ("-2^2", 0.0, -4.0),
            ("2^-1", 0.0, 0.5),
            ("sqrt(abs(-9))", 0.0, 3.0),
            ("exp(0) + ln(e)", 0.0, 2.0),
            ("x*x - x", 3.0, 6.0),
            ("(1+2)*(3+4)", 0.0, 21.0),
            (".5 * 4", 0.0, 2.0),
            ("10 - 2 - 3", 0.0, 5.0),
            ("tanh(0) + cosh(0) + sinh(0)", 0.0, 1.0),
            ("asin(1)", 0.0, std::f64::consts::FRAC_PI_2),
            ("acos(1)", 0.0, 0.0),
            ("atan(1) * 4", 0.0, std::f64::consts::PI),
            ("log(e)", 0.0, 1.0),
            ("log10(1000)", 0.0, 3.0),
            ("log2(8)", 0.0, 3.0),
            ("floor(2.7) + ceil(2.2)", 0.0, 5.0),
            ("round(2.5)", 0.0, 3.0),
            ("sign(0.0 - 3) + sign(0)", 0.0, -1.0),
            ("sign(sin(x))", 4.0, -1.0),
        ];
        for (src, x, want) in cases {
            let got = parse(src).expect(src).eval(x);
            assert!((got - want).abs() < 1e-12, "{src} at {x}: {got} != {want}");
        }
    }

    #[test]
    fn rejects_code_shaped_input() {
        // upholds: PLOT-1 — anything outside the closed grammar (imports,
        // method calls, comprehensions, statements, applications, unknown
        // names) is a typed rejection carrying the alphabet.
        for src in [
            "__import__('os')",
            "Math.sin(x)",
            "sin(x); cos(x)",
            "[y for y in range(10)]",
            "x(1)",
            "foo(x)",
            "x y",
            "1 2",
            "()",
            "",
            "sin",
            "1..2",
            "x = 1",
        ] {
            let err = parse(src).unwrap_err().to_string();
            assert!(err.contains("plot expr"), "{src:?}: {err}");
        }
    }

    #[test]
    fn depth_and_length_are_bounded() {
        // upholds: PLOT-1 — the parser is total: pathological nesting and
        // length are refused, not recursed into.
        let deep = format!("{}x{}", "(".repeat(100), ")".repeat(100));
        assert!(parse(&deep).is_err());
        let long = "1+".repeat(200) + "1";
        assert!(parse(&long).is_err());
    }

    fn arb_expr() -> impl Strategy<Value = Expr> {
        let leaf = prop_oneof![(0.0f64..1e6).prop_map(Expr::Num), Just(Expr::X),];
        leaf.prop_recursive(5, 32, 2, |inner| {
            let op = prop_oneof![
                Just(Op::Add),
                Just(Op::Sub),
                Just(Op::Mul),
                Just(Op::Div),
                Just(Op::Pow),
            ];
            let func = proptest::sample::select(vec![
                Func::Sin,
                Func::Cos,
                Func::Tan,
                Func::Sinh,
                Func::Cosh,
                Func::Tanh,
                Func::Asin,
                Func::Acos,
                Func::Atan,
                Func::Exp,
                Func::Ln,
                Func::Log10,
                Func::Log2,
                Func::Sqrt,
                Func::Abs,
                Func::Floor,
                Func::Ceil,
                Func::Round,
                Func::Sign,
            ]);
            prop_oneof![
                inner.clone().prop_map(|e| Expr::Neg(Box::new(e))),
                (op, inner.clone(), inner.clone()).prop_map(|(o, l, r)| Expr::Bin(
                    o,
                    Box::new(l),
                    Box::new(r)
                )),
                (func, inner).prop_map(|(g, a)| Expr::Call(g, Box::new(a))),
            ]
        })
    }

    proptest! {
        // upholds: PLOT-1 — the parser is total: no input panics it.
        #[test]
        fn parse_never_panics(src in ".*") {
            let _ = parse(&src);
        }

        // The algebra closes: parse ∘ print = id, structurally.
        #[test]
        fn print_parse_roundtrip(e in arb_expr()) {
            prop_assert_eq!(parse_inner(&e.to_string()).unwrap(), e);
        }
    }
}
