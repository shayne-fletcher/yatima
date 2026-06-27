//! `yatima` — a thin CLI over the in-process inference library.
//!
//! CLI invariants (part of the registry; see `yatima-lib`'s crate doc for the
//! rest). Protected by tests that cite the id (`// upholds: <id>`):
//! - **CLI-1** generation has exactly one model source (`--model` xor `--repo`).
//! - **CLI-2** `--offline` never fetches; an absent model is a clear error.
//! - **CLI-3** the CLI owns tracing subscriber setup and initializes it
//!   idempotently; `yatima-lib` only emits events.

use std::io::{Read, Write};
use std::path::PathBuf;

use anyhow::{bail, Result};
use clap::builder::{PossibleValuesParser, TypedValueParser};
use clap::{Parser, Subcommand};
use tracing_subscriber::EnvFilter;
use yatima_lib::{
    device, model_dir, models_root, resolve_format, run_blocking, Agent, ChatFormat,
    ChatMlTemplate, ChatSession, Completer, Dir, Engine, GenOpts, JsonToolCall, ListDir, ModelId,
    ModelSource, PlainTemplate, PromptTemplate, QwenToolCall, ReadFile, ReadPage, Sampling,
    ToolCallCodec, Tools, WebOrigin,
};

/// A clap value parser for [`ChatFormat`]: its names as `--help` possible values,
/// parsed back into the lib enum. (clap can't derive `ValueEnum` on a foreign
/// type, so we wrap `FromStr` over the published `NAMES`.)
fn chat_format_parser() -> impl TypedValueParser<Value = ChatFormat> {
    PossibleValuesParser::new(ChatFormat::NAMES)
        .map(|s| s.parse::<ChatFormat>().expect("NAMES are valid formats"))
}

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Generate a completion from local model weights.
    Generate(GenerateArgs),
    /// Chat with an instruction-tuned model (applies its chat template; no tools).
    Chat(ChatArgs),
    /// Run an agent loop: the model acts through capability-scoped tools.
    Agent(AgentArgs),
    /// Print the resolved models directory (or a repository's leaf dir).
    ModelsDir {
        /// Resolve to this repository's leaf directory under the models root.
        #[arg(long)]
        repo: Option<String>,
    },
}

#[derive(clap::Args)]
struct GenerateArgs {
    /// Explicit model directory.
    #[arg(long)]
    model: Option<PathBuf>,
    /// Repository id, resolved under the models root.
    #[arg(long)]
    repo: Option<String>,
    /// Override the models root (else $YATIMA_MODELS_DIR / XDG cache).
    #[arg(long)]
    models_dir: Option<PathBuf>,
    /// Prompt text; read from stdin when omitted.
    #[arg(long)]
    prompt: Option<String>,
    #[arg(long, default_value_t = 256)]
    max_tokens: usize,
    #[arg(long, default_value_t = 0.0)]
    temperature: f64,
    #[arg(long, default_value_t = 0)]
    seed: u64,
    /// Prompt prefill chunk size in tokens. Omit for model/backend default; use
    /// 0 to force one full-prompt prefill.
    #[arg(long)]
    prefill_chunk: Option<usize>,
    /// Force CPU instead of the GPU.
    #[arg(long)]
    cpu: bool,
    /// Don't auto-fetch a missing model; error instead.
    #[arg(long)]
    offline: bool,
    /// With `--repo`, fetch this single GGUF file (quantized) instead of
    /// safetensors shards.
    #[arg(long)]
    gguf: Option<String>,
}

#[derive(clap::Args)]
struct AgentArgs {
    /// Explicit model directory.
    #[arg(long)]
    model: Option<PathBuf>,
    /// Repository id, resolved under the models root.
    #[arg(long)]
    repo: Option<String>,
    /// Override the models root (else $YATIMA_MODELS_DIR / XDG cache).
    #[arg(long)]
    models_dir: Option<PathBuf>,
    /// The task / question for the agent.
    #[arg(long)]
    prompt: String,
    /// Capability root the file tools may read under (default: cwd).
    #[arg(long)]
    root: Option<PathBuf>,
    /// Grant the agent a `read_page` tool scoped to this HTTP(S) origin (e.g.
    /// `https://example.com`). Omit for no web access. One origin per run; the
    /// tool may only read same-origin URLs (CAP-2).
    #[arg(long = "web-origin")]
    web_origin: Option<String>,
    /// System prompt; a sensible default is used when omitted.
    #[arg(long)]
    system: Option<String>,
    /// Maximum tool rounds before giving up.
    #[arg(long, default_value_t = 6)]
    max_steps: usize,
    #[arg(long, default_value_t = 512)]
    max_tokens: usize,
    #[arg(long, default_value_t = 0.0)]
    temperature: f64,
    #[arg(long, default_value_t = 0)]
    seed: u64,
    /// Prompt prefill chunk size in tokens. Omit for model/backend default; use
    /// 0 to force one full-prompt prefill.
    #[arg(long)]
    prefill_chunk: Option<usize>,
    /// Force CPU instead of the GPU.
    #[arg(long)]
    cpu: bool,
    /// Don't auto-fetch a missing model; error instead.
    #[arg(long)]
    offline: bool,
    /// With `--repo`, fetch this single GGUF file (quantized) instead of
    /// safetensors shards.
    #[arg(long)]
    gguf: Option<String>,
    /// The model's chat / tool-call format. Omit to infer from the model's
    /// architecture; a value that contradicts the model is honored but warned.
    #[arg(long, value_parser = chat_format_parser())]
    format: Option<ChatFormat>,
    /// Print the full transcript (to stderr), not just the final answer.
    #[arg(long)]
    verbose: bool,
}

#[derive(clap::Args)]
struct ChatArgs {
    /// Explicit model directory.
    #[arg(long)]
    model: Option<PathBuf>,
    /// Repository id, resolved under the models root.
    #[arg(long)]
    repo: Option<String>,
    /// Override the models root (else $YATIMA_MODELS_DIR / XDG cache).
    #[arg(long)]
    models_dir: Option<PathBuf>,
    /// The user message. Omit for an interactive multi-turn session (reads
    /// stdin; `/exit` quits, `/reset` clears the conversation).
    #[arg(long)]
    prompt: Option<String>,
    /// Optional system instruction (applies for the whole session).
    #[arg(long)]
    system: Option<String>,
    /// The model's chat format. Omit to infer from the model's architecture; a
    /// value that contradicts the model is honored but warned.
    #[arg(long, value_parser = chat_format_parser())]
    format: Option<ChatFormat>,
    #[arg(long, default_value_t = 256)]
    max_tokens: usize,
    #[arg(long, default_value_t = 0.0)]
    temperature: f64,
    #[arg(long, default_value_t = 0)]
    seed: u64,
    /// Prompt prefill chunk size in tokens. Omit for model/backend default; use
    /// 0 to force one full-prompt prefill.
    #[arg(long)]
    prefill_chunk: Option<usize>,
    /// Force CPU instead of the GPU.
    #[arg(long)]
    cpu: bool,
    /// Don't auto-fetch a missing model; error instead.
    #[arg(long)]
    offline: bool,
    /// With `--repo`, fetch this single GGUF file (quantized).
    #[arg(long)]
    gguf: Option<String>,
}

const DEFAULT_AGENT_SYSTEM: &str =
    "You are a helpful assistant. You can read files under the working directory \
     using the provided tools. Call a tool when it helps, then answer.";

#[tokio::main(flavor = "multi_thread")]
async fn main() -> Result<()> {
    init_tracing();
    let cli = Cli::parse();
    match cli.command {
        Command::ModelsDir { repo } => {
            let root = models_root();
            let path = match repo {
                Some(r) => model_dir(&root, &ModelId::parse(&r)?),
                None => root,
            };
            println!("{}", path.display());
        }
        Command::Generate(args) => generate(args).await?,
        Command::Chat(args) => chat(args).await?,
        Command::Agent(args) => agent(args).await?,
    }
    Ok(())
}

fn init_tracing() {
    // upholds: CLI-3 / OBS-1 — the application edge owns collection. `try_init`
    // makes this tolerant of tests or embedders that installed a subscriber.
    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new("warn,yatima_lib=info,yatima_cli=info"));
    let _ = tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_writer(std::io::stderr)
        .try_init();
}

/// Chat: apply the model's chat template (no tools) — the layer between raw
/// `generate` and the `agent` tool loop. `--prompt` gives one shot; omitting it
/// opens an interactive multi-turn session that remembers the conversation.
async fn chat(args: ChatArgs) -> Result<()> {
    let dir = ModelSource::from_args(
        args.model,
        args.repo,
        args.models_dir,
        args.offline,
        args.gguf,
    )?
    .resolve()?;

    let dev = device(args.cpu)?;
    let mut engine = run_blocking(|| Engine::load(&dir, dev))?;
    eprintln!("loaded {} [{}]", dir.display(), engine.backend());

    // Infer the chat format from the model's architecture unless overridden
    // (HOST-1); warn on a contradicting override rather than mis-render (HOST-2).
    let (format, mismatch) = resolve_format(engine.arch(), args.format);
    if let Some(m) = mismatch {
        eprintln!("warning: {m}");
    }
    let template = format.template();
    let opts = GenOpts {
        max_tokens: args.max_tokens,
        sampling: Sampling::from_temperature(args.temperature, args.seed),
        prefill_chunk: args.prefill_chunk,
        ..Default::default()
    };

    let system = args.system;
    let mut session = ChatSession::new(&mut engine, template).with_opts(opts);
    if let Some(sys) = system {
        session = session.with_system(sys);
    }
    match args.prompt {
        // One-shot: dogfood the library `ChatSession` (batch). Memory isn't
        // needed for a single turn, but this exercises the public embedding API.
        Some(prompt) => {
            println!("{}", session.turn_async(&prompt).await?);
        }
        // Interactive: stream each turn through the same `ChatSession`, fully
        // dogfooding the library's streaming seam (`turn_streaming_async`).
        None => chat_repl(session).await?,
    }
    Ok(())
}

/// Interactive multi-turn loop over a library [`ChatSession`]: `you> ` prompt,
/// stdin line by line, the answer streamed to stdout token-by-token via
/// [`ChatSession::turn_streaming`]. EOF (Ctrl-D) or `/exit` ends the session;
/// `/reset` clears the conversation (keeping the system turn).
async fn chat_repl<C: Completer, T: PromptTemplate>(
    mut session: ChatSession<'_, C, T>,
) -> Result<()> {
    let stdin = std::io::stdin();
    eprintln!("entering chat — /exit to quit, /reset to clear history");
    loop {
        eprint!("\nyou> ");
        std::io::stderr().flush()?;
        let mut line = String::new();
        if stdin.read_line(&mut line)? == 0 {
            eprintln!("\nbye");
            break; // EOF / Ctrl-D
        }
        let line = line.trim();
        match line {
            "" => continue,
            "/exit" | "/quit" => {
                eprintln!("bye");
                break;
            }
            "/reset" => {
                session.reset();
                eprintln!("(history cleared)");
                continue;
            }
            _ => {}
        }
        let mut stdout = std::io::stdout();
        session
            .turn_streaming_async(line, &mut |piece| {
                // best-effort live echo; a write failure just drops the fragment.
                let _ = stdout.write_all(piece.as_bytes());
                let _ = stdout.flush();
            })
            .await?;
        println!();
    }
    Ok(())
}

/// Compose the agent's toolset: file tools under `root`, plus a `read_page` tool
/// when a web origin is granted. Factored out so the capability wiring is
/// unit-testable without loading a model.
fn agent_tools(root: &std::path::Path, web_origin: Option<&str>) -> Result<Tools> {
    let cap = Dir::new(root);
    let mut tools = Tools::new()
        .with(ReadFile::new(cap.clone()))
        .with(ListDir::new(cap));
    if let Some(origin) = web_origin {
        tools = tools.with(ReadPage::new(WebOrigin::new(origin)?)?);
    }
    Ok(tools)
}

async fn agent(args: AgentArgs) -> Result<()> {
    let dir = ModelSource::from_args(
        args.model,
        args.repo,
        args.models_dir,
        args.offline,
        args.gguf,
    )?
    .resolve()?;

    let root = match args.root {
        Some(r) => r,
        None => std::env::current_dir()?,
    };
    let tools = agent_tools(&root, args.web_origin.as_deref())?;

    let opts = GenOpts {
        max_tokens: args.max_tokens,
        sampling: Sampling::from_temperature(args.temperature, args.seed),
        prefill_chunk: args.prefill_chunk,
        // Keep the default repetition penalty: prose answers degenerate
        // (repeated words) without it. The penalty can mangle a tool call's JSON
        // punctuation, but the tolerant tool-call parser recovers those.
        ..Default::default()
    };

    let dev = device(args.cpu)?;
    let mut engine = run_blocking(|| Engine::load(&dir, dev))?;
    eprintln!(
        "loaded {} [{}]; tools rooted at {}",
        dir.display(),
        engine.backend(),
        root.display()
    );

    let system = args
        .system
        .unwrap_or_else(|| DEFAULT_AGENT_SYSTEM.to_string());

    // Infer the format from the model unless overridden (HOST-1/HOST-2), then
    // pick the codec/template pair. Chat-only formats can't enter the tool loop
    // (CAPS-1): the match's fallthrough rejects them.
    let (format, mismatch) = resolve_format(engine.arch(), args.format);
    if let Some(m) = mismatch {
        eprintln!("warning: {m}");
    }
    match format {
        ChatFormat::Qwen => {
            run_agent(
                &mut engine,
                &tools,
                QwenToolCall,
                ChatMlTemplate,
                system,
                args.max_steps,
                opts,
                &args.prompt,
                args.verbose,
            )
            .await
        }
        ChatFormat::Plain => {
            run_agent(
                &mut engine,
                &tools,
                JsonToolCall,
                PlainTemplate,
                system,
                args.max_steps,
                opts,
                &args.prompt,
                args.verbose,
            )
            .await
        }
        other => bail!(
            "--format {other} is chat-only (not tool-trained); use `yatima chat` for it, \
             or --format qwen for the agent"
        ),
    }
}

/// Build an agent for a given codec/template pair, run it, and print the answer
/// (full transcript to stderr when `verbose`). Generic so each `--format` arm
/// keeps the concrete, monomorphic `Agent` types.
#[allow(clippy::too_many_arguments)]
async fn run_agent<C: Completer, K: ToolCallCodec, T: PromptTemplate>(
    engine: &mut C,
    tools: &Tools,
    codec: K,
    template: T,
    system: String,
    max_steps: usize,
    opts: GenOpts,
    prompt: &str,
    verbose: bool,
) -> Result<()> {
    let mut agent = Agent::new(engine, tools, codec, template, system, max_steps).with_opts(opts);
    let run = agent.run_async(prompt).await?;

    if verbose {
        for turn in &run.transcript {
            eprintln!("── {:?} ──\n{}\n", turn.role, turn.content);
        }
    }
    println!("{}", run.answer);
    eprintln!("[{} steps, {:?}]", run.steps, run.stop);
    Ok(())
}

async fn generate(args: GenerateArgs) -> Result<()> {
    let dir = ModelSource::from_args(
        args.model,
        args.repo,
        args.models_dir,
        args.offline,
        args.gguf,
    )?
    .resolve()?;

    let prompt = match args.prompt {
        Some(p) => p,
        None => {
            let mut s = String::new();
            std::io::stdin().read_to_string(&mut s)?;
            s
        }
    };

    let dev = device(args.cpu)?;
    let mut engine = run_blocking(|| Engine::load(&dir, dev))?;
    eprintln!("loaded {} [{}]", dir.display(), engine.backend());

    let opts = GenOpts {
        max_tokens: args.max_tokens,
        sampling: Sampling::from_temperature(args.temperature, args.seed),
        prefill_chunk: args.prefill_chunk,
        ..Default::default()
    };
    let mut stdout = std::io::stdout();
    // Generation is the synchronous compute island (RT-1).
    let generation = run_blocking(|| {
        engine.generate(&prompt, &opts, |piece| {
            stdout.write_all(piece.as_bytes())?;
            stdout.flush()?;
            Ok(())
        })
    })?;
    println!();
    eprintln!("[{} tokens, {:?}]", generation.tokens, generation.stop);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tool_names(tools: &Tools) -> Vec<String> {
        tools.specs().into_iter().map(|s| s.name).collect()
    }

    #[test]
    fn agent_tools_have_no_web_access_without_origin() {
        let names = tool_names(&agent_tools(std::path::Path::new("."), None).unwrap());
        assert!(!names.iter().any(|n| n == "read_page"));
    }

    #[test]
    fn agent_tools_grant_read_page_for_a_web_origin() {
        let names = tool_names(
            &agent_tools(std::path::Path::new("."), Some("https://example.com")).unwrap(),
        );
        assert!(names.iter().any(|n| n == "read_page"));
    }

    #[test]
    fn agent_tools_reject_a_bad_web_origin() {
        assert!(agent_tools(std::path::Path::new("."), Some("not a url")).is_err());
    }

    #[test]
    fn chat_command_parses_optional_format() {
        let cli = Cli::try_parse_from([
            "yatima",
            "chat",
            "--repo",
            "google/gemma-2-2b-it",
            "--prompt",
            "explain rust",
            "--format",
            "gemma",
        ])
        .unwrap();
        let Command::Chat(args) = cli.command else {
            panic!("expected the chat subcommand");
        };
        assert_eq!(args.prompt.as_deref(), Some("explain rust"));
        assert_eq!(args.format, Some(ChatFormat::Gemma));
    }

    #[test]
    fn chat_format_is_optional_for_inference() {
        // Omitting --format is valid: the format is inferred from the model.
        let cli = Cli::try_parse_from(["yatima", "chat", "--repo", "x/y"]).unwrap();
        let Command::Chat(args) = cli.command else {
            panic!("expected the chat subcommand");
        };
        assert_eq!(args.format, None);
    }

    #[test]
    fn chat_without_prompt_is_repl_mode() {
        // Omitting --prompt is valid: it opens the interactive multi-turn REPL.
        let cli = Cli::try_parse_from(["yatima", "chat", "--repo", "x/y"]).unwrap();
        let Command::Chat(args) = cli.command else {
            panic!("expected the chat subcommand");
        };
        assert!(args.prompt.is_none());
    }

    #[test]
    fn chat_rejects_unknown_format() {
        // clap's PossibleValuesParser rejects a name outside ChatFormat::NAMES.
        assert!(
            Cli::try_parse_from(["yatima", "chat", "--repo", "x/y", "--format", "llama3"]).is_err()
        );
    }

    #[test]
    fn agent_command_parses_with_repo_and_prompt() {
        let cli = Cli::try_parse_from([
            "yatima",
            "agent",
            "--repo",
            "org/name",
            "--prompt",
            "do a thing",
            "--root",
            "/tmp",
            "--max-steps",
            "3",
        ])
        .unwrap();
        let Command::Agent(args) = cli.command else {
            panic!("expected the agent subcommand");
        };
        assert_eq!(args.prompt, "do a thing");
        assert_eq!(args.max_steps, 3);
        // the model source is parsed by the same checked library path as `generate`.
        assert!(ModelSource::from_args(
            args.model,
            args.repo,
            args.models_dir,
            args.offline,
            args.gguf
        )
        .is_ok());
    }

    #[test]
    fn agent_requires_a_prompt() {
        // clap rejects a missing required --prompt before any work happens.
        assert!(Cli::try_parse_from(["yatima", "agent", "--repo", "org/name"]).is_err());
    }

    #[test]
    fn tracing_init_is_idempotent() {
        // upholds: CLI-3 / OBS-1 — subscriber setup lives at the CLI edge and
        // repeated initialization is tolerated rather than panicking.
        init_tracing();
        init_tracing();
    }
}
