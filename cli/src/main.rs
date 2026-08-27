//! `yatima` — a thin CLI over the in-process inference library.
//!
//! CLI invariants (part of the registry; see `yatima-lib`'s crate doc for the
//! rest). Protected by tests that cite the id (`// upholds: <id>`):
//! - **CLI-1** generation has exactly one model source (`--model` xor `--repo`).
//! - **CLI-2** `--offline` never fetches; an absent model is a clear error.
//! - **CLI-3** the CLI owns tracing subscriber setup and initializes it
//!   idempotently; `yatima-lib` only emits events.
//! - **CLI-4** an interactive agent renders classified `AgentEvent` fragments
//!   and tool status live on stderr while stdout remains the final answer for
//!   pipelines; a terminal never prints that final answer twice.

use std::io::{IsTerminal, Read, Write};
use std::path::PathBuf;
use std::process::ExitCode;

use anyhow::{bail, Context, Result};
use chrono::Local;
use clap::builder::{PossibleValuesParser, TypedValueParser};
use clap::{Parser, Subcommand};
use rustyline::error::ReadlineError;
use rustyline::DefaultEditor;
use tracing_subscriber::EnvFilter;
use yatima_lib::{
    device, model_dir, models_root, resolve_format, run_blocking, verify, Agent, AgentEvent,
    Cancel, Channel, ChatFormat, ChatSession, Completer, Dir, Engine, GenOpts, JsonToolCall,
    ListDir, LlamaServer, LlamaServerCompleter, LlamaServerConfig, LlamaServerProfile,
    LlamaServerSpawn, ModelId, ModelProfile, ModelSource, ProfileBackend, PromptTemplate,
    QwenToolCall, ReadFile, ReadPage, Sampling, ServerIdentity, ToolCallCodec, Tools, WebOrigins,
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
    /// Repetition penalty over the recent window; 1.0 disables it.
    #[arg(long, default_value_t = 1.1)]
    repeat_penalty: f32,
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
    /// Inference backend. Omit to use the profile's backend, or Candle when no
    /// profile selects another backend.
    #[arg(long, value_enum)]
    backend: Option<BackendArg>,
    /// Base URL of a running llama-server (e.g. `http://127.0.0.1:8080`), for
    /// `--backend llama-server`.
    #[arg(long)]
    server_url: Option<String>,
    /// A built-in model profile: sets the model, chat format, and generation
    /// defaults. Verified profiles reject model-source and conflicting format
    /// overrides; `--max-tokens` raises (never lowers) their budget.
    #[arg(long)]
    profile: Option<String>,
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

/// Which inference backend serves a chat session.
#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
enum BackendArg {
    /// The local Candle engine: resolves and loads the model in-process.
    Engine,
    /// llama-server over HTTP: attach via `--server-url`, or omit it to let
    /// Yatima resolve the GGUF and own the child process.
    LlamaServer,
}

#[derive(clap::Args)]
struct ChatArgs {
    /// Inference backend. Omit to use the profile's backend, or Candle when no
    /// profile selects another backend.
    #[arg(long, value_enum)]
    backend: Option<BackendArg>,
    /// Base URL of a running llama-server (e.g. `http://127.0.0.1:8080`), for
    /// `--backend llama-server`.
    #[arg(long)]
    server_url: Option<String>,
    /// A built-in model profile (e.g. `kimi-dev`, `deepseek-r1`): sets the model,
    /// chat format, and generation defaults. A reasoning profile raises the token
    /// budget and enables sampling. Verified profiles reject model-source and
    /// conflicting format overrides; `--max-tokens` raises (never lowers) their
    /// budget.
    #[arg(long)]
    profile: Option<String>,
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
    /// Nucleus (top-p) sampling cutoff; omit for the full distribution. A profile
    /// may set its own (e.g. reasoning profiles use 0.95).
    #[arg(long)]
    top_p: Option<f64>,
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

#[derive(Debug)]
struct Interrupted;

impl std::fmt::Display for Interrupted {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("interrupted")
    }
}

impl std::error::Error for Interrupted {}

#[tokio::main(flavor = "multi_thread")]
async fn main() -> ExitCode {
    init_tracing();
    match run_cli().await {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) if error.downcast_ref::<Interrupted>().is_some() => ExitCode::from(130),
        Err(error) => {
            eprintln!("Error: {error:#}");
            ExitCode::FAILURE
        }
    }
}

async fn run_cli() -> Result<()> {
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
    // A `--profile` names the model + format + generation defaults. Generation
    // options layer over the CLI base (PROFILE-1); backend/source consistency is
    // resolved separately before acquisition.
    let profile = match &args.profile {
        Some(name) => Some(ModelProfile::builtin(name).ok_or_else(|| {
            anyhow::anyhow!(
                "unknown profile {name:?}; built-ins: {:?}",
                ModelProfile::BUILTIN_NAMES
            )
        })?),
        None => None,
    };

    let base = GenOpts {
        max_tokens: args.max_tokens,
        sampling: Sampling::nucleus(args.temperature, args.top_p, args.seed),
        prefill_chunk: args.prefill_chunk,
        ..Default::default()
    };
    let opts = match &profile {
        Some(p) => p.apply_gen_overrides(base),
        None => base,
    };
    let current_date = Local::now().format("%Y-%m-%d").to_string();
    let cancel = Cancel::new();

    match chat_backend_config(&args, profile.as_ref())? {
        ChatBackendConfig::Attached { url, format } => {
            let config = LlamaServerConfig::new(url)?;
            let mut completer = LlamaServerCompleter::new(config)?;
            let props = completer.introspect().await?;
            eprintln!(
                "attached llama-server [{}]; unverified: {} ({}, {} ctx, {} slot)",
                format.name(),
                props.model_self_report,
                props.build,
                props.n_ctx,
                props.total_slots,
            );
            return run_chat(
                &mut completer,
                format.template_with_date(Some(current_date)),
                opts,
                args.system,
                args.prompt,
                &cancel,
            )
            .await;
        }
        ChatBackendConfig::Managed {
            format,
            llama_server_profile,
        } => {
            let interrupt = ManagedInterrupt::new(cancel.clone());
            let resolved = match &profile {
                Some(profile) => profile.to_source(args.offline)?.resolve_async().await?,
                None => {
                    ModelSource::from_args(
                        args.model,
                        args.repo,
                        args.models_dir,
                        args.offline,
                        args.gguf,
                    )?
                    .resolve_async()
                    .await?
                }
            };
            if cancel.is_cancelled() {
                return Err(Interrupted.into());
            }
            let mut server =
                launch_managed_server(resolved, llama_server_profile.as_ref(), format, &cancel)
                    .await?;
            let chat = run_chat(
                &mut server,
                format.template_with_date(Some(current_date)),
                opts,
                args.system,
                args.prompt,
                &cancel,
            )
            .await;
            let result = finish_managed_run(server, chat, &cancel).await;
            drop(interrupt);
            return result;
        }
        ChatBackendConfig::Engine => {}
    }

    let dir = match &profile {
        Some(p) => p
            .to_engine_source(args.offline)?
            .resolve_async()
            .await?
            .into_directory(),
        None => ModelSource::from_args(
            args.model,
            args.repo,
            args.models_dir,
            args.offline,
            args.gguf,
        )?
        .resolve_async()
        .await?
        .into_directory(),
    };

    let dev = device(args.cpu)?;
    let mut engine = run_blocking(|| Engine::load(&dir, dev))?;
    eprintln!("loaded {} [{}]", dir.display(), engine.backend());

    // Infer the chat format from the model's architecture unless overridden
    // (FMT-1) — an explicit `--format` wins, else the profile's, else inference;
    // warn on a contradicting override rather than mis-render (FMT-2).
    let explicit_format = args
        .format
        .or_else(|| profile.as_ref().and_then(ModelProfile::format));
    let (format, mismatch) = resolve_format(engine.arch(), explicit_format);
    if let Some(m) = mismatch {
        eprintln!("warning: {m}");
    }
    run_chat(
        &mut engine,
        format.template_with_date(Some(current_date)),
        opts,
        args.system,
        args.prompt,
        &cancel,
    )
    .await
}

#[derive(Debug, PartialEq, Eq)]
enum ChatBackendConfig {
    Engine,
    Attached {
        url: String,
        format: ChatFormat,
    },
    Managed {
        format: ChatFormat,
        llama_server_profile: Option<LlamaServerProfile>,
    },
}

/// The backend-relevant flags a subcommand feeds [`backend_config`]: one pure
/// resolver owns the mode matrix for both `chat` and `agent`, so the two
/// commands cannot drift into separate policies.
struct BackendFlags {
    backend: Option<BackendArg>,
    server_url: Option<String>,
    format: Option<ChatFormat>,
    has_model: bool,
    has_repo: bool,
    has_models_dir: bool,
    has_gguf: bool,
    cpu: bool,
    has_prefill_chunk: bool,
}

impl ChatArgs {
    fn backend_flags(&self) -> BackendFlags {
        BackendFlags {
            backend: self.backend,
            server_url: self.server_url.clone(),
            format: self.format,
            has_model: self.model.is_some(),
            has_repo: self.repo.is_some(),
            has_models_dir: self.models_dir.is_some(),
            has_gguf: self.gguf.is_some(),
            cpu: self.cpu,
            has_prefill_chunk: self.prefill_chunk.is_some(),
        }
    }
}

impl AgentArgs {
    fn backend_flags(&self) -> BackendFlags {
        BackendFlags {
            backend: self.backend,
            server_url: self.server_url.clone(),
            format: self.format,
            has_model: self.model.is_some(),
            has_repo: self.repo.is_some(),
            has_models_dir: self.models_dir.is_some(),
            has_gguf: self.gguf.is_some(),
            cpu: self.cpu,
            has_prefill_chunk: self.prefill_chunk.is_some(),
        }
    }
}

/// The `chat` entry to the shared resolver (kept so the matrix tests read at
/// the subcommand's own seam).
fn chat_backend_config(
    args: &ChatArgs,
    profile: Option<&ModelProfile>,
) -> Result<ChatBackendConfig> {
    backend_config(&args.backend_flags(), profile)
}

/// The `agent` entry to the shared resolver — the same policy, not a second one.
fn agent_backend_config(
    args: &AgentArgs,
    profile: Option<&ModelProfile>,
) -> Result<ChatBackendConfig> {
    backend_config(&args.backend_flags(), profile)
}

/// Resolve the three backend modes before model acquisition or process work.
/// Attached mode owns no model source; managed mode owns one; the Candle
/// engine rejects server-only flags, and the llama-server modes reject
/// Candle-only flags (`--cpu`, `--prefill-chunk`) rather than silently
/// ignoring them. Every profile's pinned source is authoritative, so any
/// model-source flag beside a profile is a rejected contradiction, never
/// silently ignored (PROFILE-2). Pure, so the matrix is unit-testable.
fn backend_config(
    flags: &BackendFlags,
    profile: Option<&ModelProfile>,
) -> Result<ChatBackendConfig> {
    if let Some(profile) = profile {
        let mut overrides = Vec::new();
        if flags.has_model {
            overrides.push("--model");
        }
        if flags.has_repo {
            overrides.push("--repo");
        }
        if flags.has_gguf {
            overrides.push("--gguf");
        }
        if flags.has_models_dir {
            overrides.push("--models-dir");
        }
        if !overrides.is_empty() {
            bail!(
                "profile {:?} rejects model-source overrides: {}",
                profile.name,
                overrides.join(", ")
            );
        }
    }
    let config = backend_mode(flags, profile)?;
    if !matches!(config, ChatBackendConfig::Engine) {
        let mut rejected = Vec::new();
        if flags.cpu {
            rejected.push("--cpu");
        }
        if flags.has_prefill_chunk {
            rejected.push("--prefill-chunk");
        }
        if !rejected.is_empty() {
            bail!(
                "Candle-only flags are not valid with the llama-server backend: {}",
                rejected.join(", ")
            );
        }
    }
    Ok(config)
}

fn backend_mode(flags: &BackendFlags, profile: Option<&ModelProfile>) -> Result<ChatBackendConfig> {
    if let Some(profile) = profile {
        if let ProfileBackend::LlamaServer(server) = &profile.backend {
            if flags.backend == Some(BackendArg::Engine) {
                bail!(
                    "profile {:?} requires managed llama-server and cannot use --backend engine",
                    profile.name
                );
            }
            if flags.server_url.is_some() {
                bail!(
                    "profile {:?} requires a verified managed llama-server and cannot attach with --server-url",
                    profile.name
                );
            }
            let format = profile.format.ok_or_else(|| {
                anyhow::anyhow!(
                    "verified llama-server profile {:?} does not pin a chat format",
                    profile.name
                )
            })?;
            if let Some(explicit) = flags.format {
                if explicit != format {
                    bail!(
                        "profile {:?} pins --format {}; cannot use {}",
                        profile.name,
                        format.name(),
                        explicit.name()
                    );
                }
            }
            return Ok(ChatBackendConfig::Managed {
                format,
                llama_server_profile: Some(server.clone()),
            });
        }
    }

    let backend = flags.backend.unwrap_or(BackendArg::Engine);
    if backend == BackendArg::Engine {
        if flags.server_url.is_some() {
            bail!("--server-url requires --backend llama-server");
        }
        return Ok(ChatBackendConfig::Engine);
    }

    if let Some(url) = flags.server_url.clone() {
        if profile.is_some()
            || flags.has_model
            || flags.has_repo
            || flags.has_models_dir
            || flags.has_gguf
        {
            bail!(
                "--profile/--model/--repo/--models-dir/--gguf are not used with an attached \
                 llama-server: the attached server owns the model"
            );
        }
        let format = flags.format.ok_or_else(|| {
            anyhow::anyhow!(
                "an attached llama-server requires --format: there is no local model to infer it"
            )
        })?;
        return Ok(ChatBackendConfig::Attached { url, format });
    }

    let format = flags
        .format
        .or_else(|| profile.and_then(ModelProfile::format))
        .ok_or_else(|| anyhow::anyhow!("managed llama-server requires --format or a profile"))?;
    if profile.is_none() && !flags.has_model && !flags.has_repo {
        bail!("managed llama-server requires --profile, --model, or --repo");
    }
    Ok(ChatBackendConfig::Managed {
        format,
        llama_server_profile: None,
    })
}

/// Await `fut` while a spawned ticker keeps one transient stderr line alive:
/// `{label} — you have been waiting {n}s`. The ticker is a separate task so it
/// keeps printing even while the awaited work holds its own thread inside the
/// runtime's blocking boundary (the sha256 hash under `run_blocking`); it
/// starts after one full second, so fast paths stay silent, and it only writes
/// to a terminal — piped stderr sees the ordinary begin/end lines and no
/// carriage-return animation. On completion the line is cleared and replaced
/// by `{label}: done in {n}s`.
async fn with_wait_timer<T>(label: &'static str, fut: impl std::future::Future<Output = T>) -> T {
    let started = std::time::Instant::now();
    let animate = std::io::stderr().is_terminal();
    let ticker = animate.then(|| {
        tokio::spawn(async move {
            let mut seconds = 0u64;
            loop {
                tokio::time::sleep(std::time::Duration::from_secs(1)).await;
                seconds += 1;
                eprint!("\r\x1b[2K{}", waiting_line(label, seconds));
                let _ = std::io::stderr().flush();
            }
        })
    });
    let out = fut.await;
    if let Some(ticker) = ticker {
        ticker.abort();
        // Reap the aborted task; a `JoinError::is_cancelled` result is the
        // expected outcome, not a failure.
        let _ = ticker.await;
        eprint!("\r\x1b[2K");
    }
    eprintln!("{label}: done in {:.1}s", started.elapsed().as_secs_f64());
    out
}

/// The transient wait line — pure so the copy is testable.
fn waiting_line(label: &str, seconds: u64) -> String {
    format!("{label} — you have been waiting {seconds}s")
}

/// Verify (when a profile pins a digest) and launch one managed llama-server —
/// the single composition `chat` and `agent` share, so no caller can omit
/// verification, the compatibility gates, or the pinned context and top_k
/// (LSRV-5). Prints the wait timers and the identity banner.
async fn launch_managed_server(
    resolved: yatima_lib::ResolvedModel,
    server_profile: Option<&LlamaServerProfile>,
    format: ChatFormat,
    cancel: &Cancel,
) -> Result<LlamaServer> {
    if cancel.is_cancelled() {
        return Err(Interrupted.into());
    }
    let artifact = resolved.into_gguf()?;
    let spawn = match server_profile {
        Some(server_profile) => {
            eprintln!("verifying {} (sha256)...", artifact.path().display());
            let verified = with_wait_timer(
                "verifying sha256",
                verify(artifact, &server_profile.expected_sha256),
            )
            .await?;
            // Hashing runs in the existing blocking island and cannot be
            // interrupted partway through. Do not launch a child after an
            // interrupt received while it was in progress.
            if cancel.is_cancelled() {
                return Err(Interrupted.into());
            }
            let mut spawn = LlamaServerSpawn::verified(verified, server_profile.server_gates());
            spawn.context = Some(server_profile.context);
            spawn.top_k = server_profile.top_k;
            spawn
        }
        None => LlamaServerSpawn::new(artifact),
    };
    if cancel.is_cancelled() {
        return Err(Interrupted.into());
    }
    let server = with_wait_timer("starting llama-server", LlamaServer::spawn(spawn)).await?;
    eprintln!("{}", managed_server_banner(format, &server));
    Ok(server)
}

/// One signal listener for the lifetime of a managed command. It translates
/// Ctrl-C into the same cooperative cancellation primitive used by chat,
/// agent, and llama-server transport instead of letting the process exit
/// before its owned child is reaped.
struct ManagedInterrupt {
    task: tokio::task::JoinHandle<()>,
}

impl ManagedInterrupt {
    fn new(cancel: Cancel) -> Self {
        let task = tokio::spawn(async move {
            match tokio::signal::ctrl_c().await {
                Ok(()) => {
                    eprintln!("interrupt received; stopping managed command");
                    cancel.cancel();
                }
                Err(error) => eprintln!("warning: cannot listen for Ctrl-C: {error}"),
            }
        });
        Self { task }
    }
}

impl Drop for ManagedInterrupt {
    fn drop(&mut self) {
        self.task.abort();
    }
}

/// Preserve the command result while making explicit child shutdown the only
/// normal exit from a managed run. An interrupt exits with status 130 only
/// after shutdown has reaped the child and joined its output drains.
async fn finish_managed_run(server: LlamaServer, run: Result<()>, cancel: &Cancel) -> Result<()> {
    let interrupted = cancel.is_cancelled();
    let shutdown = server.shutdown().await;
    if interrupted {
        shutdown.context("shut down managed llama-server after interrupt")?;
        eprintln!("managed llama-server stopped");
        return Err(Interrupted.into());
    }
    match (run, shutdown) {
        (Ok(()), Ok(_)) => Ok(()),
        (Ok(()), Err(error)) => Err(error).context("shut down managed llama-server"),
        (Err(error), Ok(_)) => Err(error),
        (Err(error), Err(shutdown)) => Err(error).context(format!(
            "managed llama-server also failed to shut down: {shutdown:#}"
        )),
    }
}

fn managed_server_banner(format: ChatFormat, server: &LlamaServer) -> String {
    format_managed_server_banner(
        format,
        server.identity(),
        server.launched_artifact(),
        &server.props().build,
    )
}

fn format_managed_server_banner(
    format: ChatFormat,
    identity: &ServerIdentity,
    artifact: &std::path::Path,
    build: &str,
) -> String {
    match identity {
        ServerIdentity::Verified { digest } => format!(
            "managed llama-server [{}]; verified sha256 {digest} ({}); build {}",
            format.name(),
            artifact.display(),
            build,
        ),
        ServerIdentity::Unverified { self_report } => format!(
            "managed llama-server [{}]; launched {} (unverified self-report: {}, {})",
            format.name(),
            artifact.display(),
            self_report,
            build,
        ),
    }
}

/// The shared chat tail: the local `Engine` and an attached llama-server
/// drive the same `ChatSession`; the arms differ only in who implements
/// [`Completer`]. One-shot `--prompt` dogfoods the batch embedding API;
/// omitting it opens the streaming REPL.
async fn run_chat<C: Completer>(
    completer: &mut C,
    template: Box<dyn PromptTemplate>,
    opts: GenOpts,
    system: Option<String>,
    prompt: Option<String>,
    cancel: &Cancel,
) -> Result<()> {
    let mut session = ChatSession::new(completer, template).with_opts(opts);
    if let Some(sys) = system {
        session = session.with_system(sys);
    }
    match prompt {
        Some(prompt) => {
            let answer = tokio::select! {
                biased;
                _ = cancel.cancelled() => return Err(Interrupted.into()),
                result = session.turn_async(&prompt) => result?,
            };
            println!("{answer}");
        }
        None => chat_repl(session, cancel).await?,
    }
    Ok(())
}

/// Interactive multi-turn loop over a library [`ChatSession`]: a terminal gets
/// editable input with in-memory Up/Down history; redirected stdin retains plain
/// line reads. Answers stream to stdout token-by-token via
/// [`ChatSession::turn_streaming`]. The session's template supplies the
/// response classifier, so marker-based and ATEM reasoning are dimmed while
/// answer text remains normal and protocol framing never prints (REASON-1).
/// EOF (Ctrl-D) or `/exit` ends the session; `/reset` clears model history but
/// leaves input recall intact.
async fn chat_repl<C: Completer, T: PromptTemplate>(
    mut session: ChatSession<'_, C, T>,
    cancel: &Cancel,
) -> Result<()> {
    let stdin = std::io::stdin();
    let mut editor = if stdin.is_terminal() {
        Some(DefaultEditor::new().context("initialize chat line editor")?)
    } else {
        None
    };
    let color = std::io::stdout().is_terminal();
    eprintln!("entering chat — /exit to quit, /reset to clear history");
    loop {
        let line = match editor.as_mut() {
            Some(editor) => {
                eprintln!();
                match editor.readline("you> ") {
                    Ok(line) => line,
                    Err(ReadlineError::Interrupted) => {
                        eprintln!("(input cancelled)");
                        continue;
                    }
                    Err(ReadlineError::Eof) => {
                        eprintln!("bye");
                        break;
                    }
                    Err(error) => return Err(error).context("read chat input"),
                }
            }
            None => {
                eprint!("\nyou> ");
                std::io::stderr().flush()?;
                let mut line = String::new();
                if stdin.read_line(&mut line)? == 0 {
                    eprintln!("\nbye");
                    break;
                }
                line
            }
        };
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
        if let Some(editor) = editor.as_mut() {
            editor
                .add_history_entry(line)
                .context("record chat input history")?;
        }
        let mut stdout = std::io::stdout();
        let mut classifier = session.classifier();
        let mut current: Option<Channel> = None;
        session
            .turn_streaming_cancellable_async(line, cancel, &mut |piece| {
                classifier.push(piece, |ch, text| {
                    write_channel(&mut stdout, ch, text, &mut current, color)
                });
            })
            .await?;
        classifier.finish(|ch, text| write_channel(&mut stdout, ch, text, &mut current, color));
        if color {
            let _ = stdout.write_all(b"\x1b[0m"); // ensure dim is cleared
        }
        println!();
        if cancel.is_cancelled() {
            return Err(Interrupted.into());
        }
    }
    Ok(())
}

/// Echo a classified streaming span: dim the reasoning channel (on a terminal),
/// normal for the answer, emitting the SGR escape only when the channel changes.
/// Best-effort — a write failure just drops the fragment.
fn write_channel(
    out: &mut impl Write,
    channel: Channel,
    text: &str,
    current: &mut Option<Channel>,
    color: bool,
) {
    if channel == Channel::ToolCall {
        return;
    }
    if color && *current != Some(channel) {
        let code: &[u8] = match channel {
            Channel::Reasoning => b"\x1b[2m", // dim
            Channel::Answer => b"\x1b[0m",    // reset
            Channel::ToolCall => unreachable!("handled above"),
        };
        let _ = out.write_all(code);
    }
    *current = Some(channel);
    let _ = out.write_all(text.as_bytes());
    let _ = out.flush();
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
        tools = tools.with(ReadPage::new(WebOrigins::one(origin)?)?);
    }
    Ok(tools)
}

async fn agent(mut args: AgentArgs) -> Result<()> {
    // The same profile lookup, generation layering, and backend matrix as
    // `chat` (one policy, two subcommands). PROFILE-1 layers the profile's
    // recipe — including top_p and the reasoning budget floor — over the CLI
    // base; the shared resolver rejects contradictory or Candle-only flags.
    let profile = match &args.profile {
        Some(name) => Some(ModelProfile::builtin(name).ok_or_else(|| {
            anyhow::anyhow!(
                "unknown profile {name:?}; built-ins: {:?}",
                ModelProfile::BUILTIN_NAMES
            )
        })?),
        None => None,
    };

    let base = GenOpts {
        max_tokens: args.max_tokens,
        sampling: Sampling::from_temperature(args.temperature, args.seed),
        prefill_chunk: args.prefill_chunk,
        // Keep the default repetition penalty: prose answers degenerate
        // (repeated words) without it. The penalty can mangle a tool call's JSON
        // punctuation, but the tolerant tool-call parser recovers those.
        ..Default::default()
    };
    let opts = match &profile {
        Some(p) => p.apply_gen_overrides(base),
        None => base,
    };
    let current_date = Local::now().format("%Y-%m-%d").to_string();
    let cancel = Cancel::new();

    // Resolve the backend before any field moves out of `args` (the resolver
    // borrows the whole struct).
    let config = agent_backend_config(&args, profile.as_ref())?;

    let root = match args.root.take() {
        Some(r) => r,
        None => std::env::current_dir()?,
    };
    let tools = agent_tools(&root, args.web_origin.as_deref())?;
    let system = args
        .system
        .take()
        .unwrap_or_else(|| DEFAULT_AGENT_SYSTEM.to_string());

    match config {
        ChatBackendConfig::Attached { url, format } => {
            let config = LlamaServerConfig::new(url)?;
            let mut completer = LlamaServerCompleter::new(config)?;
            let props = completer.introspect().await?;
            eprintln!(
                "attached llama-server [{}]; unverified: {} ({}, {} ctx, {} slot); \
                 tools rooted at {}",
                format.name(),
                props.model_self_report,
                props.build,
                props.n_ctx,
                props.total_slots,
                root.display(),
            );
            run_agent_for_format(
                &mut completer,
                format,
                &tools,
                system,
                args.max_steps,
                opts,
                &args.prompt,
                args.verbose,
                current_date,
                &cancel,
            )
            .await
        }
        ChatBackendConfig::Managed {
            format,
            llama_server_profile,
        } => {
            let interrupt = ManagedInterrupt::new(cancel.clone());
            let resolved = match &profile {
                Some(profile) => profile.to_source(args.offline)?.resolve_async().await?,
                None => {
                    ModelSource::from_args(
                        args.model,
                        args.repo,
                        args.models_dir,
                        args.offline,
                        args.gguf,
                    )?
                    .resolve_async()
                    .await?
                }
            };
            if cancel.is_cancelled() {
                return Err(Interrupted.into());
            }
            let mut server =
                launch_managed_server(resolved, llama_server_profile.as_ref(), format, &cancel)
                    .await?;
            eprintln!("tools rooted at {}", root.display());
            let run = run_agent_for_format(
                &mut server,
                format,
                &tools,
                system,
                args.max_steps,
                opts,
                &args.prompt,
                args.verbose,
                current_date,
                &cancel,
            )
            .await;
            let result = finish_managed_run(server, run, &cancel).await;
            drop(interrupt);
            result
        }
        ChatBackendConfig::Engine => {
            let dir = match &profile {
                Some(p) => p.to_engine_source(args.offline)?.resolve_async().await?,
                None => {
                    ModelSource::from_args(
                        args.model,
                        args.repo,
                        args.models_dir,
                        args.offline,
                        args.gguf,
                    )?
                    .resolve_async()
                    .await?
                }
            }
            .into_directory();

            let dev = device(args.cpu)?;
            let mut engine = run_blocking(|| Engine::load(&dir, dev))?;
            eprintln!(
                "loaded {} [{}]; tools rooted at {}",
                dir.display(),
                engine.backend(),
                root.display()
            );

            // Infer the format from the model unless overridden (FMT-1/FMT-2).
            let explicit = args
                .format
                .or_else(|| profile.as_ref().and_then(ModelProfile::format));
            let (format, mismatch) = resolve_format(engine.arch(), explicit);
            if let Some(m) = mismatch {
                eprintln!("warning: {m}");
            }
            if format == ChatFormat::MuseGlimmer {
                bail!(
                    "Muse Glimmer does not run on the Candle engine; \
                     use --profile muse-glimmer (managed llama-server)"
                );
            }
            run_agent_for_format(
                &mut engine,
                format,
                &tools,
                system,
                args.max_steps,
                opts,
                &args.prompt,
                args.verbose,
                current_date,
                &cancel,
            )
            .await
        }
    }
}

/// Dispatch the agent run by format: each arm pairs the format's codec with
/// its runtime-dated template (CAPS-1 — chat-only formats cannot enter the
/// tool loop; the fallthrough rejects them).
#[allow(clippy::too_many_arguments)]
async fn run_agent_for_format<C: Completer>(
    completer: &mut C,
    format: ChatFormat,
    tools: &Tools,
    system: String,
    max_steps: usize,
    opts: GenOpts,
    prompt: &str,
    verbose: bool,
    current_date: String,
    cancel: &Cancel,
) -> Result<()> {
    let template = format.template_with_date(Some(current_date));
    match format {
        ChatFormat::Qwen => {
            run_agent(
                completer,
                tools,
                QwenToolCall,
                template,
                system,
                max_steps,
                opts,
                prompt,
                verbose,
                cancel,
            )
            .await
        }
        ChatFormat::Plain => {
            run_agent(
                completer,
                tools,
                JsonToolCall,
                template,
                system,
                max_steps,
                opts,
                prompt,
                verbose,
                cancel,
            )
            .await
        }
        ChatFormat::MuseGlimmer => {
            run_agent(
                completer,
                tools,
                yatima_lib::MuseAtemCodec,
                template,
                system,
                max_steps,
                opts,
                prompt,
                verbose,
                cancel,
            )
            .await
        }
        other => bail!(
            "--format {other} is chat-only (not tool-trained); use `yatima chat` for it, \
             or --format qwen for the agent"
        ),
    }
}

/// Terminal-only projection of the agent event stream. stderr carries live
/// working state; stdout remains available for the final answer (CLI-4).
struct AgentDisplay<W> {
    out: W,
    enabled: bool,
    color: bool,
    current_channel: Option<Channel>,
    needs_newline: bool,
    current_tool: Option<String>,
    streamed_answer: bool,
}

impl<W: Write> AgentDisplay<W> {
    fn new(out: W, enabled: bool, color: bool) -> Self {
        Self {
            out,
            enabled,
            color,
            current_channel: None,
            needs_newline: false,
            current_tool: None,
            streamed_answer: false,
        }
    }

    fn observe(&mut self, event: AgentEvent) {
        if !self.enabled {
            return;
        }
        match event {
            AgentEvent::Fragment { channel, text } => {
                if channel == Channel::Answer && !text.is_empty() {
                    self.streamed_answer = true;
                }
                write_channel(
                    &mut self.out,
                    channel,
                    &text,
                    &mut self.current_channel,
                    self.color,
                );
                if !text.is_empty() {
                    self.needs_newline = !text.ends_with('\n');
                }
            }
            AgentEvent::ToolCall(call) => {
                self.end_fragments();
                let _ = writeln!(self.out, "[tool: {}]", call.name);
                self.current_tool = Some(call.name);
            }
            AgentEvent::ToolProgress(message) => {
                self.end_fragments();
                let name = self.current_tool.as_deref().unwrap_or("tool");
                let _ = writeln!(self.out, "[{name}: {message}]");
            }
            AgentEvent::ToolOutcome(outcome) => {
                self.end_fragments();
                let name = self.current_tool.take().unwrap_or_else(|| "tool".into());
                let _ = writeln!(self.out, "[{name}: {}]", outcome.kind());
            }
            AgentEvent::ToolArtifact(path) => {
                self.end_fragments();
                let _ = writeln!(self.out, "[artifact: {}]", path.display());
            }
            AgentEvent::Reasoning(_) | AgentEvent::Final(_) => self.end_fragments(),
            AgentEvent::ToolStarted(_) => {}
        }
    }

    fn end_fragments(&mut self) {
        if self.color && self.current_channel.is_some() {
            let _ = self.out.write_all(b"\x1b[0m");
        }
        if self.needs_newline {
            let _ = writeln!(self.out);
        }
        self.current_channel = None;
        self.needs_newline = false;
    }

    fn finish(&mut self) {
        self.end_fragments();
        let _ = self.out.flush();
    }
}

fn print_final_answer(stdout_is_terminal: bool, streamed_answer: bool) -> bool {
    !(stdout_is_terminal && streamed_answer)
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
    cancel: &Cancel,
) -> Result<()> {
    let mut agent = Agent::new(engine, tools, codec, template, system, max_steps).with_opts(opts);
    let stderr = std::io::stderr();
    let live = stderr.is_terminal();
    let display = AgentDisplay::new(stderr, live, live);
    let (mut display, run) = agent
        .run_with_cancellable_async(prompt, cancel, display, |mut display, event| {
            display.observe(event);
            Ok(std::ops::ControlFlow::Continue(display))
        })
        .await?;
    display.finish();
    if cancel.is_cancelled() {
        return Err(Interrupted.into());
    }

    if verbose {
        for turn in &run.transcript {
            eprintln!("── {:?} ──\n{turn}\n", turn.role());
        }
    }
    if print_final_answer(std::io::stdout().is_terminal(), display.streamed_answer) {
        println!("{}", run.answer);
    }
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
    .resolve_async()
    .await?
    .into_directory();

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
        repeat_penalty: args.repeat_penalty,
        prefill_chunk: args.prefill_chunk,
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
    fn chat_accepts_a_profile() {
        let cli = Cli::try_parse_from(["yatima", "chat", "--profile", "kimi-dev"]).unwrap();
        let Command::Chat(args) = cli.command else {
            panic!("expected the chat subcommand");
        };
        assert_eq!(args.profile.as_deref(), Some("kimi-dev"));
        // The named profile resolves to a reasoning model with its format pinned.
        let p = ModelProfile::builtin(args.profile.as_deref().unwrap()).unwrap();
        assert!(p.reasoning);
        assert_eq!(p.format(), Some(ChatFormat::Qwen));
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

    fn chat_args(argv: &[&str]) -> ChatArgs {
        let cli = Cli::try_parse_from(argv).unwrap();
        let Command::Chat(args) = cli.command else {
            panic!("expected the chat subcommand");
        };
        args
    }

    #[test]
    fn chat_backend_defaults_to_the_local_engine() {
        let args = chat_args(&["yatima", "chat", "--repo", "x/y"]);
        assert_eq!(args.backend, None);
        assert_eq!(
            chat_backend_config(&args, None).unwrap(),
            ChatBackendConfig::Engine
        );
    }

    #[test]
    fn chat_parses_an_attached_llama_server() {
        let args = chat_args(&[
            "yatima",
            "chat",
            "--backend",
            "llama-server",
            "--server-url",
            "http://127.0.0.1:8080",
            "--format",
            "muse-glimmer",
        ]);
        assert_eq!(args.backend, Some(BackendArg::LlamaServer));
        assert_eq!(
            chat_backend_config(&args, None).unwrap(),
            ChatBackendConfig::Attached {
                url: "http://127.0.0.1:8080".into(),
                format: ChatFormat::MuseGlimmer,
            }
        );
    }

    #[test]
    fn managed_backend_requires_a_source_and_format() {
        let args = chat_args(&["yatima", "chat", "--backend", "llama-server"]);
        let err = chat_backend_config(&args, None).unwrap_err();
        assert!(err.to_string().contains("--format"));

        let args = chat_args(&[
            "yatima",
            "chat",
            "--backend",
            "llama-server",
            "--format",
            "qwen",
        ]);
        let err = chat_backend_config(&args, None).unwrap_err();
        assert!(err.to_string().contains("--profile, --model, or --repo"));
    }

    #[test]
    fn attached_backend_requires_an_explicit_format() {
        let args = chat_args(&[
            "yatima",
            "chat",
            "--backend",
            "llama-server",
            "--server-url",
            "http://127.0.0.1:8080",
        ]);
        let err = chat_backend_config(&args, None).unwrap_err();
        assert!(err.to_string().contains("--format"));
    }

    #[test]
    fn attached_backend_rejects_local_model_sources() {
        // Attached mode must demonstrably never touch the model cache: local
        // source flags are contradictions, not silently ignored.
        let args = chat_args(&[
            "yatima",
            "chat",
            "--backend",
            "llama-server",
            "--server-url",
            "http://127.0.0.1:8080",
            "--format",
            "qwen",
            "--repo",
            "x/y",
        ]);
        let err = chat_backend_config(&args, None).unwrap_err();
        assert!(err.to_string().contains("not used with"));
    }

    #[test]
    fn managed_backend_reuses_model_source_flags() {
        let args = chat_args(&[
            "yatima",
            "chat",
            "--backend",
            "llama-server",
            "--repo",
            "org/name",
            "--gguf",
            "model.gguf",
            "--format",
            "qwen",
        ]);
        assert_eq!(
            chat_backend_config(&args, None).unwrap(),
            ChatBackendConfig::Managed {
                format: ChatFormat::Qwen,
                llama_server_profile: None,
            }
        );
    }

    #[test]
    fn muse_profile_selects_managed_llama_server_structurally() {
        let args = chat_args(&["yatima", "chat", "--profile", "muse-glimmer"]);
        let mut profile = ModelProfile::builtin("muse-glimmer").unwrap();
        profile.name = "renamed-without-changing-its-backend".into();
        let ChatBackendConfig::Managed {
            format,
            llama_server_profile,
        } = chat_backend_config(&args, Some(&profile)).unwrap()
        else {
            panic!("Muse must select managed llama-server");
        };
        assert_eq!(format, ChatFormat::MuseGlimmer);
        assert_eq!(
            llama_server_profile,
            match &profile.backend {
                ProfileBackend::LlamaServer(server) => Some(server.clone()),
                ProfileBackend::Engine => None,
            }
        );

        let explicit = chat_args(&[
            "yatima",
            "chat",
            "--profile",
            "muse-glimmer",
            "--backend",
            "llama-server",
            "--format",
            "muse-glimmer",
        ]);
        assert!(matches!(
            chat_backend_config(&explicit, Some(&profile)).unwrap(),
            ChatBackendConfig::Managed {
                format: ChatFormat::MuseGlimmer,
                llama_server_profile: Some(_),
            }
        ));
    }

    #[test]
    fn muse_profile_rejects_backend_and_attached_substitution() {
        let profile = ModelProfile::builtin("muse-glimmer").unwrap();
        let engine = chat_args(&[
            "yatima",
            "chat",
            "--profile",
            "muse-glimmer",
            "--backend",
            "engine",
        ]);
        let error = chat_backend_config(&engine, Some(&profile))
            .unwrap_err()
            .to_string();
        assert!(error.contains("requires managed llama-server"), "{error}");

        let attached = chat_args(&[
            "yatima",
            "chat",
            "--profile",
            "muse-glimmer",
            "--backend",
            "llama-server",
            "--server-url",
            "http://127.0.0.1:8080",
        ]);
        let error = chat_backend_config(&attached, Some(&profile))
            .unwrap_err()
            .to_string();
        assert!(error.contains("cannot attach"), "{error}");
    }

    #[test]
    fn muse_profile_rejects_every_identity_override() {
        let profile = ModelProfile::builtin("muse-glimmer").unwrap();
        for override_args in [
            &["--model", "/tmp/model"][..],
            &["--repo", "someone/else"][..],
            &["--gguf", "another.gguf"][..],
            &["--models-dir", "/tmp/models"][..],
        ] {
            let mut argv = vec!["yatima", "chat", "--profile", "muse-glimmer"];
            argv.extend_from_slice(override_args);
            let args = chat_args(&argv);
            let error = chat_backend_config(&args, Some(&profile))
                .unwrap_err()
                .to_string();
            assert!(error.contains("rejects model-source overrides"), "{error}");
            assert!(error.contains(override_args[0]), "{error}");
        }
    }

    #[test]
    fn muse_profile_accepts_only_its_pinned_format() {
        let profile = ModelProfile::builtin("muse-glimmer").unwrap();
        let matching = chat_args(&[
            "yatima",
            "chat",
            "--profile",
            "muse-glimmer",
            "--format",
            "muse-glimmer",
        ]);
        assert!(chat_backend_config(&matching, Some(&profile)).is_ok());

        let conflicting = chat_args(&[
            "yatima",
            "chat",
            "--profile",
            "muse-glimmer",
            "--format",
            "qwen",
        ]);
        let error = chat_backend_config(&conflicting, Some(&profile))
            .unwrap_err()
            .to_string();
        assert!(error.contains("pins --format muse-glimmer"), "{error}");
    }

    #[test]
    fn waiting_line_counts_in_the_requested_voice() {
        assert_eq!(
            waiting_line("verifying sha256", 34),
            "verifying sha256 — you have been waiting 34s"
        );
        assert_eq!(
            waiting_line("starting llama-server", 1),
            "starting llama-server — you have been waiting 1s"
        );
    }

    #[test]
    fn verified_managed_banner_names_its_evidence() {
        // upholds: LSRV-5 — the CLI's identity claim comes from the verified
        // state, while build remains a separately reported compatibility fact.
        let digest = "4cc57c0f51040a226e5a72cc47b7613f7772950e460a665f7083de89f183f60e"
            .parse()
            .unwrap();
        let banner = format_managed_server_banner(
            ChatFormat::MuseGlimmer,
            &ServerIdentity::Verified { digest },
            std::path::Path::new("/models/muse.gguf"),
            "b10520-cd644c395",
        );
        assert_eq!(
            banner,
            "managed llama-server [muse-glimmer]; verified sha256 \
             4cc57c0f51040a226e5a72cc47b7613f7772950e460a665f7083de89f183f60e \
             (/models/muse.gguf); build b10520-cd644c395"
        );
    }

    #[test]
    fn engine_backend_rejects_a_server_url() {
        let args = chat_args(&[
            "yatima",
            "chat",
            "--repo",
            "org/name",
            "--server-url",
            "http://127.0.0.1:8080",
        ]);
        let err = chat_backend_config(&args, None).unwrap_err();
        assert!(err.to_string().contains("--backend llama-server"));
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

    fn agent_args(argv: &[&str]) -> AgentArgs {
        let cli = Cli::try_parse_from(argv).unwrap();
        let Command::Agent(args) = cli.command else {
            panic!("expected the agent subcommand");
        };
        args
    }

    #[test]
    fn agent_muse_profile_selects_managed_llama_server() {
        // The agent resolves backends through the same core as chat: the Muse
        // profile structurally selects managed llama-server, format pinned.
        let args = agent_args(&[
            "yatima",
            "agent",
            "--profile",
            "muse-glimmer",
            "--prompt",
            "hi",
        ]);
        assert_eq!(args.profile.as_deref(), Some("muse-glimmer"));
        let profile = ModelProfile::builtin("muse-glimmer").unwrap();
        assert!(matches!(
            agent_backend_config(&args, Some(&profile)).unwrap(),
            ChatBackendConfig::Managed {
                format: ChatFormat::MuseGlimmer,
                llama_server_profile: Some(_),
            }
        ));
    }

    #[test]
    fn agent_parses_an_attached_llama_server() {
        let args = agent_args(&[
            "yatima",
            "agent",
            "--backend",
            "llama-server",
            "--server-url",
            "http://127.0.0.1:8080",
            "--format",
            "muse-glimmer",
            "--prompt",
            "hi",
        ]);
        assert_eq!(
            agent_backend_config(&args, None).unwrap(),
            ChatBackendConfig::Attached {
                url: "http://127.0.0.1:8080".into(),
                format: ChatFormat::MuseGlimmer,
            }
        );
    }

    /// Resolve either subcommand's argv through its shared-core wrapper.
    fn resolve_backend_for(
        argv: &[&str],
        profile: Option<&ModelProfile>,
    ) -> Result<ChatBackendConfig> {
        match Cli::try_parse_from(argv).unwrap().command {
            Command::Chat(args) => chat_backend_config(&args, profile),
            Command::Agent(args) => agent_backend_config(&args, profile),
            _ => panic!("expected the chat or agent subcommand"),
        }
    }

    fn resolve_backend(argv: &[&str]) -> Result<ChatBackendConfig> {
        resolve_backend_for(argv, None)
    }

    #[test]
    fn every_profile_rejects_model_source_overrides() {
        // upholds: PROFILE-2 — a profile's pinned source is authoritative in
        // both subcommands and on every backend; a model-source flag beside a
        // profile is a rejected contradiction, not silently ignored.
        let ordinary = ModelProfile::builtin("kimi-dev").unwrap();
        assert!(matches!(ordinary.backend, ProfileBackend::Engine));
        for subcommand in ["chat", "agent"] {
            for override_args in [
                &["--model", "/tmp/model"][..],
                &["--repo", "someone/else"][..],
                &["--gguf", "another.gguf"][..],
                &["--models-dir", "/tmp/models"][..],
            ] {
                let mut argv = vec!["yatima", subcommand, "--profile", "kimi-dev"];
                argv.extend_from_slice(override_args);
                if subcommand == "agent" {
                    argv.extend_from_slice(&["--prompt", "hi"]);
                }
                let error = resolve_backend_for(&argv, Some(&ordinary))
                    .unwrap_err()
                    .to_string();
                assert!(
                    error.contains("rejects model-source overrides"),
                    "{subcommand} {override_args:?}: {error}"
                );
                assert!(
                    error.contains(override_args[0]),
                    "{subcommand} {override_args:?}: {error}"
                );
            }
        }
    }

    #[test]
    fn candle_only_flags_are_rejected_by_every_llama_server_mode() {
        // The full matrix: {chat, agent} × {attached, managed} × {--cpu,
        // --prefill-chunk}. One shared resolver rejects them all by name.
        let attached = &[
            "--backend",
            "llama-server",
            "--server-url",
            "http://127.0.0.1:8080",
            "--format",
            "qwen",
        ][..];
        let managed = &[
            "--backend",
            "llama-server",
            "--repo",
            "org/name",
            "--format",
            "qwen",
        ][..];
        for subcommand in ["chat", "agent"] {
            for mode in [attached, managed] {
                for flag in [&["--cpu"][..], &["--prefill-chunk", "64"][..]] {
                    let mut argv = vec!["yatima", subcommand];
                    argv.extend_from_slice(mode);
                    argv.extend_from_slice(flag);
                    if subcommand == "agent" {
                        argv.extend_from_slice(&["--prompt", "hi"]);
                    }
                    let error = resolve_backend(&argv).unwrap_err().to_string();
                    assert!(
                        error.contains("Candle-only"),
                        "{subcommand} {flag:?}: {error}"
                    );
                    assert!(error.contains(flag[0]), "{subcommand} {flag:?}: {error}");
                }
            }
        }
        // Positive control: the engine backend keeps its own flags.
        assert_eq!(
            resolve_backend(&["yatima", "chat", "--repo", "org/name", "--cpu"]).unwrap(),
            ChatBackendConfig::Engine
        );
        assert_eq!(
            resolve_backend(&[
                "yatima",
                "agent",
                "--repo",
                "org/name",
                "--prompt",
                "hi",
                "--cpu",
                "--prefill-chunk",
                "64",
            ])
            .unwrap(),
            ChatBackendConfig::Engine
        );
    }

    #[test]
    fn cli_agent_display_projects_live_events_without_protocol_markup() {
        // upholds: CLI-4 / AGENT-4 — classified working state goes to the
        // terminal projection incrementally; the typed tool event, not ATEM or
        // marker text, supplies tool status.
        let mut bytes = Vec::new();
        let streamed_answer;
        {
            let mut display = AgentDisplay::new(&mut bytes, true, false);
            display.observe(AgentEvent::Fragment {
                channel: Channel::Reasoning,
                text: "Thinking".into(),
            });
            display.observe(AgentEvent::Reasoning("Thinking".into()));
            display.observe(AgentEvent::ToolCall(yatima_lib::ToolCall {
                name: "read_file".into(),
                args: Default::default(),
            }));
            display.observe(AgentEvent::ToolOutcome(yatima_lib::ToolOutcome::success(
                "contents",
            )));
            display.observe(AgentEvent::Fragment {
                channel: Channel::Answer,
                text: "Done.".into(),
            });
            display.observe(AgentEvent::Final("Done.".into()));
            display.finish();
            streamed_answer = display.streamed_answer;
        }
        assert_eq!(
            String::from_utf8(bytes).unwrap(),
            "Thinking\n[tool: read_file]\n[read_file: success]\nDone.\n"
        );
        assert!(streamed_answer);
    }

    #[test]
    fn final_answer_is_not_duplicated_on_a_terminal() {
        // upholds: CLI-4 — stdout remains populated for a pipeline, while an
        // interactive terminal does not repeat an answer already shown live.
        assert!(!print_final_answer(true, true));
        assert!(print_final_answer(true, false));
        assert!(print_final_answer(false, true));
        assert!(print_final_answer(false, false));
    }

    #[test]
    fn tracing_init_is_idempotent() {
        // upholds: CLI-3 / OBS-1 — subscriber setup lives at the CLI edge and
        // repeated initialization is tolerated rather than panicking.
        init_tracing();
        init_tracing();
    }
}
