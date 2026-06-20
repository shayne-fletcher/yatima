//! `yatima` — a thin CLI over the in-process inference library.
//!
//! CLI invariants (part of the registry; see `yatima-lib`'s crate doc for the
//! rest). Protected by tests that cite the id (`// upholds: <id>`):
//! - **CLI-1** generation has exactly one model source (`--model` xor `--repo`).
//! - **CLI-2** `--offline` never fetches; an absent model is a clear error.

use std::io::{Read, Write};
use std::path::PathBuf;

use anyhow::{bail, Result};
use clap::{Parser, Subcommand, ValueEnum};
use yatima_lib::{
    device, model_dir, models_root, Agent, ChatMlTemplate, Completer, Dir, Engine, GenOpts,
    JsonToolCall, ListDir, ModelId, PlainTemplate, PromptTemplate, QwenToolCall, ReadFile,
    Sampling, ToolCallCodec, Tools,
};

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
    /// The model's chat / tool-call format.
    #[arg(long, value_enum, default_value_t = ChatFormat::Qwen)]
    format: ChatFormat,
    /// Print the full transcript (to stderr), not just the final answer.
    #[arg(long)]
    verbose: bool,
}

/// Which model-native chat + tool-call format to speak.
#[derive(Clone, Copy, ValueEnum)]
enum ChatFormat {
    /// Qwen2.5-Instruct: ChatML + `<tool_call>` (trained for tools).
    Qwen,
    /// Minimal `<|role|>` layout + `<tool_call>{json}</tool_call>` (fallback).
    Plain,
}

const DEFAULT_AGENT_SYSTEM: &str =
    "You are a helpful assistant. You can read files under the working directory \
     using the provided tools. Call a tool when it helps, then answer.";

fn main() -> Result<()> {
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
        Command::Generate(args) => generate(args)?,
        Command::Agent(args) => agent(args)?,
    }
    Ok(())
}

/// Map the CLI's `temperature`/`seed` flags to a [`Sampling`] policy: a
/// non-positive temperature means deterministic greedy (no seed).
fn sampling_of(temperature: f64, seed: u64) -> Sampling {
    if temperature <= 0.0 {
        Sampling::Greedy
    } else {
        Sampling::Sample { temperature, seed }
    }
}

fn agent(args: AgentArgs) -> Result<()> {
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
    let cap = Dir::new(&root);
    let tools = Tools::new()
        .with(ReadFile::new(cap.clone()))
        .with(ListDir::new(cap));

    let opts = GenOpts {
        max_tokens: args.max_tokens,
        sampling: sampling_of(args.temperature, args.seed),
        // Keep the default repetition penalty: prose answers degenerate
        // (repeated words) without it. The penalty can mangle a tool call's JSON
        // punctuation, but the tolerant tool-call parser recovers those.
        ..Default::default()
    };

    let mut engine = Engine::load(&dir, device(args.cpu)?)?;
    eprintln!(
        "loaded {} [{}]; tools rooted at {}",
        dir.display(),
        engine.backend(),
        root.display()
    );

    let system = args
        .system
        .unwrap_or_else(|| DEFAULT_AGENT_SYSTEM.to_string());

    // The codec/template pair is the model's native format, chosen by --format.
    match args.format {
        ChatFormat::Qwen => run_agent(
            &mut engine,
            &tools,
            QwenToolCall,
            ChatMlTemplate,
            system,
            args.max_steps,
            opts,
            &args.prompt,
            args.verbose,
        ),
        ChatFormat::Plain => run_agent(
            &mut engine,
            &tools,
            JsonToolCall,
            PlainTemplate,
            system,
            args.max_steps,
            opts,
            &args.prompt,
            args.verbose,
        ),
    }
}

/// Build an agent for a given codec/template pair, run it, and print the answer
/// (full transcript to stderr when `verbose`). Generic so each `--format` arm
/// keeps the concrete, monomorphic `Agent` types.
#[allow(clippy::too_many_arguments)]
fn run_agent<C: Completer, K: ToolCallCodec, T: PromptTemplate>(
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
    let run = agent.run(prompt)?;

    if verbose {
        for turn in &run.transcript {
            eprintln!("── {:?} ──\n{}\n", turn.role, turn.content);
        }
    }
    println!("{}", run.answer);
    eprintln!("[{} steps, {:?}]", run.steps, run.stop);
    Ok(())
}

fn generate(args: GenerateArgs) -> Result<()> {
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

    let mut engine = Engine::load(&dir, device(args.cpu)?)?;
    eprintln!("loaded {} [{}]", dir.display(), engine.backend());

    let opts = GenOpts {
        max_tokens: args.max_tokens,
        sampling: sampling_of(args.temperature, args.seed),
        ..Default::default()
    };
    let mut stdout = std::io::stdout();
    let generation = engine.generate(&prompt, &opts, |piece| {
        stdout.write_all(piece.as_bytes())?;
        stdout.flush()?;
        Ok(())
    })?;
    println!();
    eprintln!("[{} tokens, {:?}]", generation.tokens, generation.stop);
    Ok(())
}

/// Where a model's files come from — exactly one source, parsed at the edge so
/// the rest of the program never sees an invalid combination (CLI-1).
enum ModelSource {
    Directory(PathBuf),
    Repository {
        id: ModelId,
        root: PathBuf,
        fetch: FetchPolicy,
        /// A single GGUF file to fetch instead of safetensors shards.
        gguf: Option<String>,
    },
}

enum FetchPolicy {
    Online,
    Offline,
}

impl ModelSource {
    fn from_args(
        model: Option<PathBuf>,
        repo: Option<String>,
        models_dir: Option<PathBuf>,
        offline: bool,
        gguf: Option<String>,
    ) -> Result<ModelSource> {
        match (model, repo) {
            (Some(dir), None) => Ok(ModelSource::Directory(dir)),
            (None, Some(repo)) => Ok(ModelSource::Repository {
                id: ModelId::parse(&repo)?,
                gguf,
                root: models_dir.unwrap_or_else(models_root),
                fetch: if offline {
                    FetchPolicy::Offline
                } else {
                    FetchPolicy::Online
                },
            }),
            (Some(_), Some(_)) => bail!("pass only one of --model / --repo"),
            (None, None) => bail!("specify --model <dir> or --repo <id>"),
        }
    }

    /// Resolve to a concrete model directory, fetching on a cache miss when the
    /// policy is `Online` (CLI-2: `Offline` never touches the network).
    fn resolve(self) -> Result<PathBuf> {
        match self {
            ModelSource::Directory(dir) => Ok(dir),
            ModelSource::Repository {
                id,
                root,
                fetch,
                gguf,
            } => {
                let dir = model_dir(&root, &id);
                if yatima_lib::is_model_present(&dir) {
                    return Ok(dir);
                }
                match fetch {
                    FetchPolicy::Offline => bail!(
                        "model '{id}' not present at {} (drop --offline to fetch, or run: \
                         possum model download --repository {id} --to {})",
                        dir.display(),
                        root.display()
                    ),
                    FetchPolicy::Online => fetch_model(&id, &root, gguf.as_deref()),
                }
            }
        }
    }
}

#[cfg(feature = "fetch")]
fn fetch_model(id: &ModelId, root: &std::path::Path, gguf: Option<&str>) -> Result<PathBuf> {
    eprintln!("fetching {id} …");
    yatima_lib::ensure_model_blocking(id, root, gguf)
}

#[cfg(not(feature = "fetch"))]
fn fetch_model(id: &ModelId, _root: &std::path::Path, _gguf: Option<&str>) -> Result<PathBuf> {
    bail!("model '{id}' not present and yatima was built without the `fetch` feature")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn source_directory() {
        // upholds: CLI-1
        let s = ModelSource::from_args(Some(PathBuf::from("/m")), None, None, false, None).unwrap();
        assert!(matches!(s, ModelSource::Directory(_)));
    }

    #[test]
    fn source_repository_online_and_offline() {
        // upholds: CLI-1
        let on = ModelSource::from_args(None, Some("org/name".into()), None, false, None).unwrap();
        assert!(matches!(
            on,
            ModelSource::Repository {
                fetch: FetchPolicy::Online,
                ..
            }
        ));
        let off = ModelSource::from_args(None, Some("org/name".into()), None, true, None).unwrap();
        assert!(matches!(
            off,
            ModelSource::Repository {
                fetch: FetchPolicy::Offline,
                ..
            }
        ));
    }

    #[test]
    fn source_is_exclusive_and_required() {
        // upholds: CLI-1 — exactly one model source.
        assert!(ModelSource::from_args(
            Some(PathBuf::from("/m")),
            Some("org/name".into()),
            None,
            false,
            None
        )
        .is_err());
        assert!(ModelSource::from_args(None, None, None, false, None).is_err());
    }

    #[test]
    fn source_rejects_escaping_model_id() {
        // upholds: MS-3
        assert!(ModelSource::from_args(None, Some("../escape".into()), None, false, None).is_err());
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
        // the model source is parsed by the same checked path as `generate`.
        let src = ModelSource::from_args(
            args.model,
            args.repo,
            args.models_dir,
            args.offline,
            args.gguf,
        )
        .unwrap();
        assert!(matches!(src, ModelSource::Repository { .. }));
    }

    #[test]
    fn agent_requires_a_prompt() {
        // clap rejects a missing required --prompt before any work happens.
        assert!(Cli::try_parse_from(["yatima", "agent", "--repo", "org/name"]).is_err());
    }

    #[test]
    fn sampling_zero_temp_is_greedy() {
        assert_eq!(sampling_of(0.0, 7), Sampling::Greedy);
        assert!(matches!(
            sampling_of(0.8, 7),
            Sampling::Sample {
                temperature,
                seed: 7
            } if temperature == 0.8
        ));
    }

    #[test]
    fn offline_absent_errors_without_network() {
        // upholds: CLI-2 — offline + absent model errors, never fetches.
        let src = ModelSource::from_args(
            None,
            Some("org/name".into()),
            Some(PathBuf::from("/nonexistent-yatima-models-xyzzy")),
            true,
            None,
        )
        .unwrap();
        assert!(src.resolve().is_err());
    }
}
