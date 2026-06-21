//! Ask a local model to review changes against Yatima's invariant registry.
//!
//! This is an early "Yatima improves Yatima" example: Rust gathers bounded,
//! cited repository context (registries, changed files, invariant references),
//! then an in-process chat model produces a review report. The model does not
//! edit files; it proposes missing invariants/tests for a human to judge.
//!
//! ```bash
//! cargo run -p yatima-lib --release --example invariant_reviewer --features metal -- \
//!   --profile qwen32b --diff
//! ```

use anyhow::{anyhow, bail, Context, Result};
use clap::builder::{PossibleValuesParser, TypedValueParser};
use clap::Parser;
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::process::Command;
use yatima_lib::{
    device, resolve_format, ChatFormat, ChatSession, Engine, GenOpts, ModelProfile, Sampling,
};

const SYSTEM: &str = "\
You are a senior Rust reviewer for Yatima. Review the supplied repository \
context against the invariant registry. Focus on correctness, missing laws, \
missing tests, wrong invariant citations, observability/security leaks, and \
documentation drift. Every finding must cite the supplied file path and line or \
section marker when available. Prefer a short report with actionable findings. \
If no blocking issue is apparent, say so and suggest at most three useful \
follow-ups. Do not invent files or invariants not present in the context.";

#[derive(Debug, Parser)]
#[command(about, long_about = None)]
struct Args {
    /// Repository root to inspect.
    #[arg(long, default_value = ".")]
    root: PathBuf,
    /// Review the current working-tree diff against HEAD.
    #[arg(long)]
    diff: bool,
    /// Review staged changes instead of the whole working-tree diff.
    #[arg(long)]
    staged: bool,
    /// Additional files to include, relative to --root.
    #[arg(long)]
    file: Vec<PathBuf>,
    /// Built-in profile name.
    #[arg(long, default_value = "qwen32b")]
    profile: String,
    /// Explicit model directory (overrides --profile's source).
    #[arg(long)]
    model: Option<PathBuf>,
    /// Repository id, resolved under the models root.
    #[arg(long)]
    repo: Option<String>,
    /// With --repo, the single GGUF quant to fetch.
    #[arg(long)]
    gguf: Option<String>,
    /// Chat format; omit to infer from the model's architecture.
    #[arg(long, value_parser = chat_format_parser())]
    format: Option<ChatFormat>,
    /// Don't auto-fetch a missing model; error instead.
    #[arg(long)]
    offline: bool,
    /// Force CPU instead of the GPU.
    #[arg(long)]
    cpu: bool,
    #[arg(long, default_value_t = 900)]
    max_tokens: usize,
    #[arg(long, default_value_t = 0.0)]
    temperature: f64,
    #[arg(long, default_value_t = 0)]
    seed: u64,
    /// Maximum bytes to include for any one file.
    #[arg(long, default_value_t = 24_000)]
    max_file_bytes: usize,
    /// Maximum bytes to include from git diff output.
    #[arg(long, default_value_t = 80_000)]
    max_diff_bytes: usize,
}

fn chat_format_parser() -> impl TypedValueParser<Value = ChatFormat> {
    PossibleValuesParser::new(ChatFormat::NAMES)
        .map(|s| s.parse::<ChatFormat>().expect("NAMES are valid formats"))
}

fn main() -> Result<()> {
    let args = Args::parse();
    let root = args.root.canonicalize().with_context(|| {
        format!(
            "canonicalizing repository root {}",
            args.root.as_path().display()
        )
    })?;
    let context = gather_context(&root, &args)?;
    eprintln!(
        "review context: {} files, {} invariant ids, {} bytes",
        context.files.len(),
        context.invariant_ids.len(),
        context.prompt_bytes()
    );

    let mut profile = ModelProfile::builtin(&args.profile).ok_or_else(|| {
        anyhow!(
            "unknown profile {:?}; built-ins: {:?}",
            args.profile,
            ModelProfile::BUILTIN_NAMES
        )
    })?;
    if let Some(model) = args.model.clone() {
        profile.dir = Some(model);
        profile.repo = None;
    }
    if let Some(repo) = args.repo.clone() {
        profile.repo = Some(repo);
        profile.dir = None;
    }
    if args.gguf.is_some() {
        profile.gguf = args.gguf.clone();
    }
    if args.format.is_some() {
        profile.format = args.format;
    }

    let dir = profile.to_source(args.offline)?.resolve()?;
    let mut engine = Engine::load(&dir, device(args.cpu)?)
        .with_context(|| format!("loading {}", dir.display()))?;
    let (format, mismatch) = resolve_format(engine.arch(), profile.format);
    if let Some(m) = mismatch {
        eprintln!("warning: {m}");
    }
    let opts = profile.apply_gen_overrides(GenOpts {
        max_tokens: args.max_tokens,
        sampling: Sampling::from_temperature(args.temperature, args.seed),
        ..Default::default()
    });
    eprintln!(
        "loaded {} [{:?} / {}]; format {}; max_tokens {}",
        dir.display(),
        engine.arch(),
        engine.backend(),
        format,
        opts.max_tokens
    );

    let prompt = build_prompt(&context);
    let prompt_tokens = engine.token_count(&prompt).unwrap_or(0);
    eprintln!("prompt tokens: {prompt_tokens}");
    let mut chat = ChatSession::new(&mut engine, format.template())
        .with_system(SYSTEM)
        .with_opts(opts);
    println!("{}", chat.turn(&prompt)?);
    Ok(())
}

#[derive(Debug)]
struct ReviewContext {
    root: PathBuf,
    diff: Option<String>,
    files: BTreeMap<PathBuf, String>,
    invariant_ids: BTreeSet<String>,
    upheld_refs: Vec<String>,
}

impl ReviewContext {
    fn prompt_bytes(&self) -> usize {
        self.diff.as_ref().map_or(0, String::len)
            + self.files.values().map(String::len).sum::<usize>()
            + self.upheld_refs.iter().map(String::len).sum::<usize>()
    }
}

fn gather_context(root: &Path, args: &Args) -> Result<ReviewContext> {
    let diff = if args.diff || args.staged {
        Some(truncate(
            &git_diff(root, args.staged)?,
            args.max_diff_bytes,
            "git diff",
        ))
    } else {
        None
    };

    let mut files = BTreeSet::from([
        PathBuf::from("lib/src/lib.rs"),
        PathBuf::from("cli/src/main.rs"),
        PathBuf::from("notes/design.md"),
    ]);
    files.extend(args.file.iter().cloned());
    if args.diff || args.staged {
        files.extend(git_changed_files(root, args.staged)?);
    }

    let mut file_text = BTreeMap::new();
    for rel in files {
        if !is_safe_relative(&rel) {
            bail!("refusing to read path outside root: {}", rel.display());
        }
        let path = root.join(&rel);
        if path.is_file() {
            let text = std::fs::read_to_string(&path)
                .with_context(|| format!("reading {}", path.display()))?;
            file_text.insert(
                rel,
                truncate(&with_line_numbers(&text), args.max_file_bytes, "file"),
            );
        }
    }

    let invariant_ids = extract_invariant_ids(
        file_text
            .get(Path::new("lib/src/lib.rs"))
            .map(String::as_str)
            .unwrap_or_default(),
    );
    let mut all_ids = invariant_ids;
    if let Some(cli) = file_text.get(Path::new("cli/src/main.rs")) {
        all_ids.extend(extract_invariant_ids(cli));
    }
    let upheld_refs = collect_upholds(root)?;

    Ok(ReviewContext {
        root: root.to_path_buf(),
        diff,
        files: file_text,
        invariant_ids: all_ids,
        upheld_refs,
    })
}

fn git_diff(root: &Path, staged: bool) -> Result<String> {
    let mut cmd = Command::new("git");
    cmd.arg("-C").arg(root).arg("diff");
    if staged {
        cmd.arg("--cached");
    }
    run_command(cmd, "git diff")
}

fn git_changed_files(root: &Path, staged: bool) -> Result<Vec<PathBuf>> {
    let mut cmd = Command::new("git");
    cmd.arg("-C").arg(root).arg("diff");
    if staged {
        cmd.arg("--cached");
    }
    cmd.arg("--name-only");
    let out = run_command(cmd, "git diff --name-only")?;
    Ok(out
        .lines()
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(PathBuf::from)
        .collect())
}

fn run_command(mut cmd: Command, label: &str) -> Result<String> {
    let output = cmd.output().with_context(|| format!("running {label}"))?;
    if !output.status.success() {
        bail!(
            "{label} failed: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        );
    }
    Ok(String::from_utf8_lossy(&output.stdout).into_owned())
}

fn collect_upholds(root: &Path) -> Result<Vec<String>> {
    let mut refs = Vec::new();
    for rel in [Path::new("lib/src"), Path::new("cli/src")] {
        collect_upholds_under(root, rel, &mut refs)?;
    }
    refs.sort();
    Ok(refs)
}

fn collect_upholds_under(root: &Path, rel: &Path, refs: &mut Vec<String>) -> Result<()> {
    let dir = root.join(rel);
    if !dir.exists() {
        return Ok(());
    }
    for entry in std::fs::read_dir(&dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() {
            let child = rel.join(entry.file_name());
            collect_upholds_under(root, &child, refs)?;
            continue;
        }
        if path.extension().is_some_and(|ext| ext == "rs") {
            let text = std::fs::read_to_string(&path)?;
            let display = rel.join(entry.file_name());
            for (i, line) in text.lines().enumerate() {
                if line.contains("upholds:") {
                    refs.push(format!("{}:{}: {}", display.display(), i + 1, line.trim()));
                }
            }
        }
    }
    Ok(())
}

fn build_prompt(context: &ReviewContext) -> String {
    let mut s = String::new();
    s.push_str("# Invariant Review Request\n\n");
    s.push_str(&format!("Repository root: {}\n\n", context.root.display()));
    s.push_str("Known invariant ids:\n");
    for id in &context.invariant_ids {
        s.push_str(&format!("- {id}\n"));
    }
    s.push_str("\nKnown test citations (`upholds:`):\n");
    for line in &context.upheld_refs {
        s.push_str("- ");
        s.push_str(line);
        s.push('\n');
    }
    if let Some(diff) = &context.diff {
        s.push_str("\n# Git Diff\n\n```diff\n");
        s.push_str(diff);
        s.push_str("\n```\n");
    }
    s.push_str("\n# Files\n");
    for (path, text) in &context.files {
        s.push_str(&format!("\n## {}\n\n```text\n", path.display()));
        s.push_str(text);
        s.push_str("\n```\n");
    }
    s.push_str(
        "\n# Requested Output\n\n\
         Return:\n\
         1. Findings, ordered by severity. Each finding must cite evidence.\n\
         2. Missing or weak invariants/tests.\n\
         3. Smallest next action.\n\
         If the change is sound, say so directly.\n",
    );
    s
}

fn with_line_numbers(text: &str) -> String {
    text.lines()
        .enumerate()
        .map(|(i, line)| format!("{:>4}: {line}", i + 1))
        .collect::<Vec<_>>()
        .join("\n")
}

fn truncate(text: &str, max: usize, label: &str) -> String {
    if text.len() <= max {
        return text.to_string();
    }
    let mut end = max.min(text.len());
    while !text.is_char_boundary(end) {
        end -= 1;
    }
    format!(
        "{}\n\n[truncated {label}: kept {} of {} bytes]",
        &text[..end],
        end,
        text.len()
    )
}

fn extract_invariant_ids(text: &str) -> BTreeSet<String> {
    let mut ids = BTreeSet::new();
    for line in text.lines() {
        let mut rest = line;
        while let Some(start) = rest.find("**") {
            rest = &rest[start + 2..];
            let Some(end) = rest.find("**") else { break };
            let candidate = &rest[..end];
            if candidate
                .chars()
                .all(|c| c.is_ascii_uppercase() || c.is_ascii_digit() || c == '-')
                && candidate.contains('-')
            {
                ids.insert(candidate.to_string());
            }
            rest = &rest[end + 2..];
        }
    }
    ids
}

fn is_safe_relative(path: &Path) -> bool {
    path.components().all(|component| {
        matches!(
            component,
            std::path::Component::Normal(_) | std::path::Component::CurDir
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extracts_invariant_ids_from_registry_markup() {
        let ids = extract_invariant_ids(
            "\
//! - **OBS-1** library emits tracing.
//! - **CAP-2** effects are scoped.
//! - **not-an-id** ignored.",
        );
        assert!(ids.contains("OBS-1"));
        assert!(ids.contains("CAP-2"));
        assert!(!ids.contains("not-an-id"));
    }

    #[test]
    fn safe_paths_cannot_escape_root() {
        assert!(is_safe_relative(Path::new("lib/src/lib.rs")));
        assert!(is_safe_relative(Path::new("./README.md")));
        assert!(!is_safe_relative(Path::new("../secret")));
        assert!(!is_safe_relative(Path::new("/tmp/secret")));
    }

    #[test]
    fn truncate_respects_utf8_boundaries() {
        let text = "abc🙂def";
        let truncated = truncate(text, 5, "test");
        assert!(truncated.starts_with("abc"));
        assert!(truncated.contains("truncated test"));
    }
}
