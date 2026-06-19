use clap::Parser;

/// a Rust runtime for language-integrated LLMs — inference as an in-process
/// library function.
#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Args {}

fn main() {
    let _args = Args::parse();
}
