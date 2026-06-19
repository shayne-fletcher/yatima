//! `yatima` — a thin CLI over the in-process inference library.

use std::io::{Read, Write};
use std::path::PathBuf;

use anyhow::{bail, Result};
use clap::{Parser, Subcommand};
use yatima_lib::{device, model_dir, models_root, Engine, GenOpts};

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
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::ModelsDir { repo } => {
            let root = models_root();
            let path = match repo {
                Some(r) => model_dir(&root, &r),
                None => root,
            };
            println!("{}", path.display());
        }
        Command::Generate(args) => generate(args)?,
    }
    Ok(())
}

fn generate(args: GenerateArgs) -> Result<()> {
    let dir = match (args.model, args.repo) {
        (Some(d), None) => d,
        (None, Some(repo)) => {
            let root = args.models_dir.unwrap_or_else(models_root);
            let dir = model_dir(&root, &repo);
            if !yatima_lib::is_model_present(&dir) {
                if args.offline {
                    bail!(
                        "model '{repo}' not present at {} (drop --offline to fetch, or run: \
                         possum model download --repository {repo} --to {})",
                        dir.display(),
                        root.display()
                    );
                }
                #[cfg(feature = "fetch")]
                {
                    eprintln!("fetching {repo} …");
                    yatima_lib::ensure_model_blocking(&repo, &root)?;
                }
                #[cfg(not(feature = "fetch"))]
                {
                    bail!(
                        "model '{repo}' not present and yatima was built without the \
                         `fetch` feature"
                    );
                }
            }
            dir
        }
        (Some(_), Some(_)) => bail!("pass only one of --model / --repo"),
        (None, None) => bail!("specify --model <dir> or --repo <id>"),
    };

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
        temperature: args.temperature,
        seed: args.seed,
    };
    let mut stdout = std::io::stdout();
    engine.generate(&prompt, &opts, |piece| {
        stdout.write_all(piece.as_bytes())?;
        stdout.flush()?;
        Ok(())
    })?;
    println!();
    Ok(())
}
