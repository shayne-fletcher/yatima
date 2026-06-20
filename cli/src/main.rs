//! `yatima` — a thin CLI over the in-process inference library.

use std::io::{Read, Write};
use std::path::PathBuf;

use anyhow::{bail, Result};
use clap::{Parser, Subcommand};
use yatima_lib::{device, model_dir, models_root, Engine, GenOpts, RepoId, Sampling};

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
                Some(r) => model_dir(&root, &RepoId::parse(&r)?),
                None => root,
            };
            println!("{}", path.display());
        }
        Command::Generate(args) => generate(args)?,
    }
    Ok(())
}

fn generate(args: GenerateArgs) -> Result<()> {
    let dir =
        ModelSource::from_args(args.model, args.repo, args.models_dir, args.offline)?.resolve()?;

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

    let sampling = if args.temperature <= 0.0 {
        Sampling::Greedy
    } else {
        Sampling::Sample {
            temperature: args.temperature,
            seed: args.seed,
        }
    };
    let opts = GenOpts {
        max_tokens: args.max_tokens,
        sampling,
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
        id: RepoId,
        root: PathBuf,
        fetch: FetchPolicy,
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
    ) -> Result<ModelSource> {
        match (model, repo) {
            (Some(dir), None) => Ok(ModelSource::Directory(dir)),
            (None, Some(repo)) => Ok(ModelSource::Repository {
                id: RepoId::parse(&repo)?,
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
            ModelSource::Repository { id, root, fetch } => {
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
                    FetchPolicy::Online => fetch_model(&id, &root),
                }
            }
        }
    }
}

#[cfg(feature = "fetch")]
fn fetch_model(id: &RepoId, root: &std::path::Path) -> Result<PathBuf> {
    eprintln!("fetching {id} …");
    yatima_lib::ensure_model_blocking(id, root)
}

#[cfg(not(feature = "fetch"))]
fn fetch_model(id: &RepoId, _root: &std::path::Path) -> Result<PathBuf> {
    bail!("model '{id}' not present and yatima was built without the `fetch` feature")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn source_directory() {
        let s = ModelSource::from_args(Some(PathBuf::from("/m")), None, None, false).unwrap();
        assert!(matches!(s, ModelSource::Directory(_)));
    }

    #[test]
    fn source_repository_online_and_offline() {
        let on = ModelSource::from_args(None, Some("org/name".into()), None, false).unwrap();
        assert!(matches!(
            on,
            ModelSource::Repository {
                fetch: FetchPolicy::Online,
                ..
            }
        ));
        let off = ModelSource::from_args(None, Some("org/name".into()), None, true).unwrap();
        assert!(matches!(
            off,
            ModelSource::Repository {
                fetch: FetchPolicy::Offline,
                ..
            }
        ));
    }

    // CLI-1: exactly one model source.
    #[test]
    fn source_is_exclusive_and_required() {
        assert!(ModelSource::from_args(
            Some(PathBuf::from("/m")),
            Some("org/name".into()),
            None,
            false
        )
        .is_err());
        assert!(ModelSource::from_args(None, None, None, false).is_err());
    }

    #[test]
    fn source_rejects_escaping_repo_id() {
        assert!(ModelSource::from_args(None, Some("../escape".into()), None, false).is_err());
    }
}
