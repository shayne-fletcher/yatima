//! Render a recorded run readably (headless-drive D3): fragments folded into
//! reasoning/answer spans, one row per turn with elapsed-to-first-call,
//! elapsed-to-first-image and elapsed-to-done. A projection only; the tape
//! stays the evidence.
//!
//! Usage: `yatima-report <run-dir | tape.jsonl> [--clip N]` (`--clip 0` for
//! full span text; default 1500 characters).

use std::path::PathBuf;
use std::process::ExitCode;

use clap::Parser;

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// A run directory (containing `tape.jsonl`) or the tape file itself.
    run: PathBuf,
    /// Characters of each span to show (0 = all).
    #[arg(long, default_value_t = 1500)]
    clip: usize,
}

fn main() -> ExitCode {
    let args = Args::parse();
    match yatima_drive::report::read_tape(&args.run) {
        Ok((header, lines)) => {
            print!(
                "{}",
                yatima_drive::report::render(&header, &lines, args.clip)
            );
            ExitCode::SUCCESS
        }
        Err(error) => {
            eprintln!("yatima-report: {error:#}");
            ExitCode::from(1)
        }
    }
}
