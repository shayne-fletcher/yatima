# 3. Authority as values

The previous chapters described values that move toward and away from a model. This chapter looks at values that limit what model-requested tools may do. It reads [`capability.rs`](https://github.com/shayne-fletcher/yatima/blob/402d34a26bedd9d254e76a51be8c034961e28db1/lib/src/capability.rs) and [`expr.rs`](https://github.com/shayne-fletcher/yatima/blob/402d34a26bedd9d254e76a51be8c034961e28db1/lib/src/expr.rs) at commit `402d34a26bedd9d254e76a51be8c034961e28db1`.

The common idea is that the model supplies a request, while the program supplies the limits. A file tool receives a rooted directory rather than an unrestricted path API. A web reader receives approved origins rather than an unrestricted HTTP client. The plot tool accepts a small expression language rather than Python code.

## `capability.rs`: limits carried by values

The private `capability` module defines the values used to construct restricted tools. `lib.rs` re-exports the `Dir`, `WriteDir`, `WebOrigin`, `WebOrigins`, `PlotSandbox`, and `NtfyTopic` structs, along with the `origins_in` and `proposed_origins` functions.

The structs keep their authority-bearing fields private. A tool can clone or use a capability, but it cannot replace the capability's root, origin, interpreter, or topic by assigning a field. The important checks happen when a target is resolved or when a capability is constructed.

### Rooted filesystem paths

The `Dir` and `WriteDir` structs have the same representation but different meanings:

```rust
pub struct Dir {
    root: PathBuf,
}

pub struct WriteDir {
    root: PathBuf,
}
```

`Dir` is used by read and list tools. `WriteDir` is used by write and artifact-producing tools. Keeping them as separate Rust types prevents a function that requires write authority from accidentally accepting a read-only capability.

Both constructors take ownership of a root path. They do not open or canonicalize it. Their return types are spelled slightly differently but have the same concrete meaning:

```rust
pub fn new(root: impl Into<PathBuf>) -> Self
pub fn new(root: impl Into<PathBuf>) -> WriteDir
```

Both `resolve` methods call the crate-private `is_safe_relative` function and accept a string only when it is relative and every component Rust retains is normal. This rejects absolute paths, parent components such as `..`, a leading `./`, and platform prefixes. The accepted path is then joined to the stored root.

```rust
let reads = Dir::new("/srv/project");

let readme = reads.resolve("docs/README.md")?;
assert_eq!(readme, PathBuf::from("/srv/project/docs/README.md"));

assert!(reads.resolve("../secrets.txt").is_err());
assert!(reads.resolve("/etc/passwd").is_err());
```

An empty string is accepted and resolves to the root itself; `ListDir` uses that to list the root directory. The check is lexical: it examines the supplied path components, not the filesystem object eventually reached. At this commit, a symlink below the root can point outside it. The registry's stronger CAP-1 wording therefore exceeds what these types presently enforce; this is recorded in the walkthrough todo.

The `ReadFile`, `ListDir`, and `WriteFile` structs are defined later in `tool.rs`. Each owns the appropriate capability and calls its `resolve` method before touching the filesystem. The model supplies only the relative string.

### One web origin

The `WebOrigin` struct owns a parsed `reqwest::Url`:

```rust
pub struct WebOrigin {
    origin: Url,
}
```

`WebOrigin::new` accepts an HTTP or HTTPS origin. The shared private `parse_origin_url` function rejects a path, query, fragment, missing host, and non-web scheme. It also makes grants convenient to type: `en.wikipedia.org` becomes `https://en.wikipedia.org`, and surrounding quotation marks or sentence punctuation are removed. The `origin` method borrows the stored `Url`.

The constructor currently does not reject URL userinfo such as `https://name:secret@example.com`. The same-origin check also ignores userinfo on an absolute target URL, so a target carrying credentials can pass when its scheme, host, and port match. That gap is recorded in the todo. Callers should treat the intended value as a scheme, host, and optional port only.

`WebOrigin::resolve` accepts either a relative reference or an absolute URL. A relative reference is joined to the stored origin. An absolute URL must be covered by that origin. A fragment is removed because it is not sent to the server and should not create a separate cache entry.

Coverage normally means equal scheme, host, and effective port. There is one deliberate widening: a grant for `http://example.com` also covers `https://example.com` on the corresponding port. The reverse is refused, so an HTTPS grant cannot silently become a plaintext request.

```rust
let web = WebOrigin::new("https://example.com")?;

assert_eq!(
    web.resolve("/docs?page=2")?.as_str(),
    "https://example.com/docs?page=2"
);
assert!(web.resolve("https://other.example/docs").is_err());
assert!(web.resolve("http://example.com/docs").is_err());
```

This value does not perform a request. It decides whether a requested URL lies within the authority supplied to a web tool. The tool still has to check every redirect before following it; Chapter 9 follows that network path.

### A live set of web origins

Interactive sessions need grants to change while their tools remain alive. The `WebOrigins` struct therefore shares mutable state:

```rust
pub struct WebOrigins {
    origins: std::sync::Arc<std::sync::RwLock<Vec<WebOrigin>>>,
}
```

The source spells the `Arc` and `RwLock` with their full `std::sync` paths. `Arc` lets the host and several tools own clones referring to the same set. `RwLock` permits concurrent reads and exclusive changes. Cloning `WebOrigins` does not copy the origins into an independent set.

`WebOrigins::new` starts empty. `WebOrigins::one` constructs a set and grants one origin, which is useful for the one-shot CLI. The remaining methods operate on the shared set:

- `grant` validates an origin, inserts it only when absent, and reports whether the set grew.
- `revoke` validates an origin, removes it when present, and reports whether the set shrank.
- `list` returns display-ready origin strings in insertion order.
- `is_empty` reports whether any origin is granted.
- `resolve` checks a target against the current set.

An absolute target may match any member. A relative target is accepted only when the set has exactly one member, because otherwise there is no principled origin against which to resolve it. The error lists the possible absolute URLs instead of guessing.

```rust
let origins = WebOrigins::new();
origins.grant("https://a.example")?;

let reader_copy = origins.clone();
assert!(reader_copy.resolve("https://a.example/page").is_ok());

origins.revoke("https://a.example")?;
assert!(reader_copy.resolve("https://a.example/page").is_err());
```

The clone sees the revocation because both values point to the same lock and vector. This is how a host changes a running session's authority without rebuilding every tool.

`WebOrigins` stores origin grants only. The narrower CAP-4 authority for an exact image listed by an approved page is represented separately by the `ImageListing` struct in `tool.rs`. Selecting such an image does not insert its host into this set.

### Finding origins in text

The module defines two public functions that produce candidate origin strings. Neither function changes authority.

`origins_in` scans text supplied by the user for HTTP or HTTPS URLs. It removes paths and duplicates and preserves first-seen order. Frontends use the result to implement the rule that a URL typed by the user grants its origin.

`proposed_origins` scans model-influenced text before the host emits a typed `GrantProposal`. It is stricter: proposed origins must be canonical HTTP(S), have an ASCII host containing a dot, and contain no userinfo. The GUI, TUI, and browser render the resulting values, but only the user's approval sends a grant request.

The distinction is important. These functions recognize syntax; `WebOrigins::grant` is the operation that changes authority. Model output can propose an origin but cannot grant one.

### Plot output and interpreter choice

The `PlotSandbox` struct holds two choices made by the program:

```rust
pub struct PlotSandbox {
    out_dir: WriteDir,
    python: PathBuf,
}
```

`PlotSandbox::new` creates the output directory, runs the chosen interpreter with `-c "import matplotlib"`, and returns an error if the process cannot run or import the package. `PlotSandbox::system` chooses `python3`. A successful value therefore carries a confined output directory and an interpreter that passed the probe.

The `resolve` method delegates to `WriteDir::resolve`, and `python` borrows the fixed interpreter path. The model receives neither choice. Later, the `Plot` struct runs only Yatima's fixed generator script with literal data and writes to a filename selected by Yatima.

This is not an operating-system sandbox around arbitrary Python. Its safety claim is smaller: the model cannot supply Python code or choose an output path, and the fixed generator receives only a validated plot specification.

### One notification destination

The `NtfyTopic` struct fixes both halves of a notification destination:

```rust
pub struct NtfyTopic {
    server: Url,
    topic: String,
}
```

`NtfyTopic::new` uses `https://ntfy.sh`. `NtfyTopic::with_server` accepts another HTTP(S) origin. The topic must contain 1 to 64 ASCII letters, digits, hyphens, or underscores, so it cannot add another path segment or query.

The `server` and `topic` methods borrow the stored fields. `endpoint` clones the server URL and sets its path to the topic. A `SendNotification` tool constructed with this value can publish to that endpoint; model arguments may supply message text and presentation options, but not another server or topic.

## `expr.rs`: mathematical intent without model-written code

The private `expr` module implements the expression language used by function series in the `Plot` tool. Its types and `parse` function are visible only within `yatima-lib`; they are not re-exported to consumers.

A model can ask to plot `sin(x) * exp(-x/10)`. Yatima parses and samples that function in Rust. The Python renderer receives the resulting arrays, never the expression text as executable code.

### The accepted language

The source starts with a compact grammar. In that notation, `:=` means "is defined as," `|` separates alternatives, `*` means that the preceding parenthesized part may repeat zero or more times, and `?` means that the preceding part is optional. Parentheses group grammar elements, while quoted characters must occur literally. Read the first rule as: an expression is one term followed by any number of plus-or-minus terms.

```text
expr  := term  (('+' | '-') term)*
term  := unary (('*' | '/') unary)*
unary := '-' unary | power
power := atom ('^' unary)?
atom  := number | 'x' | 'pi' | 'e'
       | func '(' expr ')' | '(' expr ')'
```

The named functions are a fixed list: trigonometric and inverse trigonometric functions, hyperbolic functions, exponentials and logarithms, square root, absolute value, rounding functions, and `sign`. `log` is accepted as an alias for `ln`.

The lexer actually accepts any run of digits and decimal points as a candidate number and then asks Rust to parse it as `f64`. Consequently `.5` and `1.` are accepted, while malformed forms such as `1..2` are rejected. This is slightly broader than the number rule written in the source comment and is recorded as a documentation debt.

### The expression tree

Three crate-private enums represent a parsed expression. `Op` names the five binary operators. `Func` names the permitted functions. `Expr` describes the tree:

```rust
pub(crate) enum Expr {
    Num(f64),
    X,
    Neg(Box<Expr>),
    Bin(Op, Box<Expr>, Box<Expr>),
    Call(Func, Box<Expr>),
}
```

This is the complete set of executable meanings. There is no node for a statement, assignment, import, attribute lookup, loop, function definition, or arbitrary call. Once parsing has produced an `Expr`, later code can only evaluate this closed set.

`Expr::eval` recursively evaluates the tree for one supplied value of `x`. Floating-point domain failures such as `ln(0)` produce `NaN` or infinity rather than a Rust error. The plot caller samples the expression and rejects the whole series if any result is non-finite.

`Expr::references_x` recursively reports whether a tree contains `Expr::X`. The `PlotBound` enum defined later in `tool.rs` uses it to permit symbolic constants such as `2 * pi` for a sampling bound while rejecting `x + 1`, whose value would depend on the variable whose range it is supposed to define.

The `Display` implementation prints every operation with explicit parentheses. Parsing that output reconstructs the same tree. This property makes the expression representation closed under its own textual form and is checked over generated expression trees.

### Lexing and parsing

The private `Tok` enum is the lexer's smaller vocabulary: numbers, `x`, recognized functions, five operators, and parentheses. The `lex` function ignores ASCII whitespace, folds `pi` and `e` into numeric tokens, maps `log` to `Func::Ln`, and rejects every unknown byte or name with an error that repeats the legal alphabet.

The private `Parser` struct borrows the token slice and owns only its current position:

```rust
struct Parser<'a> {
    toks: &'a [Tok],
    pos: usize,
}
```

Its methods follow precedence from weakest to strongest: `expr` handles addition and subtraction, `term` handles multiplication and division, `unary` handles negation, `power` handles exponentiation, and `atom` handles values, calls, and parentheses. Exponentiation recurses through `unary` on its right, making `2^3^2` mean `2^(3^2)`. Negation delegates to `power`, making `-2^2` mean `-(2^2)`.

The crate-private `parse` function is the entry point. It rejects source longer than 256 bytes before lexing. Recursive parsing rejects nesting beyond 64 levels. After one expression has been parsed, any remaining token is an error rather than ignored trailing input.

### From expression text to a PNG

The `PlotSeries` struct and `Plot` tool live later in `tool.rs`, but their use of this module is short enough to follow here. A function series such as:

```json
{
  "expr": "sin(x) * exp(-x/10)",
  "from": 0,
  "to": "2 * pi",
  "samples": 512
}
```

takes this path:

1. Serde first checks that the surrounding plot request fits the closed `PlotSpec` and `PlotSeries` structs.
2. `PlotBound::resolve` parses the two bounds, rejects a bound that refers to `x`, evaluates it, and requires a finite result.
3. The private `resolve_plot_series` function parses the series expression, creates 512 evenly spaced `x` values, and calls `Expr::eval` for each one.
4. If every `y` value is finite, the function expression is replaced by literal `x` and `y` arrays.
5. The fixed Python generator receives only the resolved arrays and a Yatima-chosen output path inside `PlotSandbox`.

An input such as `__import__('os')`, `Math.sin(x)`, `x = 1`, or `sin(x); cos(x)` fails during lexing or parsing. It never reaches Python. This is the concrete work done by PLOT-1: a model can describe a mathematical function, but there is no representation for model-authored code in the accepted request.

## Contract for callers

The capability structs let callers choose authority before constructing a tool. Their private fields prevent later field replacement, and their resolution methods check each model-supplied target before the underlying effect. `WebOrigins` adds shared session mutation, but only its `grant` and `revoke` methods change the set.

The expression module accepts a bounded mathematical language and returns either a closed `Expr` tree or an explanatory error. It assumes the caller will reject non-finite evaluations and will pass only resolved literal data to any external renderer.

The [crate registry](https://github.com/shayne-fletcher/yatima/blob/402d34a26bedd9d254e76a51be8c034961e28db1/lib/src/lib.rs) names the obligations encountered here. CAP-1 names filesystem confinement, with the symlink limitation noted above. CAP-2 says an agent's effects must stay inside the capabilities held by its tools. CAP-3 separates user-granted origins from model proposals, while CAP-4 permits only exact public resources derived from an approved page. PLOT-1 forbids model-authored code. PLOT-2 intends to confine plot output through `PlotSandbox` and inherits the same unresolved filesystem-object caveat as CAP-1. PLOT-3's deterministic rendering behavior belongs to the `Plot` implementation and is deferred to Chapter 9.

## Maintainer checkpoint

- `Dir`, `WriteDir`, `WebOrigin`, `PlotSandbox`, and `NtfyTopic` own fixed limits chosen by the program. `WebOrigins` owns an `Arc<RwLock<_>>`, so the host and web tools share one mutable grant set.
- The normal effect path is: construct a capability, move or clone it into a tool, receive model arguments, resolve those arguments through the capability, then perform the effect.
- To add another kind of authority, put the checked value in a type with private fields, make the tool own that type, and ensure every effectful path uses it rather than ambient process access.
- `expr.rs` is the pattern for admitting useful model intent without admitting code: parse into a closed enum, evaluate in Rust, validate the result, and hand the external process only literal data.
- Changes here must preserve CAP-1 through CAP-4 and PLOT-1/PLOT-2. Do not claim a constructor prevents a state unless its fields and every construction path actually make that state unreachable.
