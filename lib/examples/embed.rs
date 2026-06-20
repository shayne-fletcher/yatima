//! Embedding `yatima-lib` — the model as an ordinary in-process function.
//!
//! This is the thing the CLI can't show: a plain Rust program that loads a model
//! once and weaves inference into its own control flow, no service boundary. Run:
//!
//! ```bash
//! cargo run --example embed --features metal
//! ```
//!
//! It needs a cached instruct model (Qwen2.5-7B-Instruct here — fetch it once via
//! `yatima generate --repo Qwen/Qwen2.5-7B-Instruct --prompt hi`).

use yatima_lib::{
    device, is_model_present, model_dir, models_root, ChatMlTemplate, ChatSession, Engine, ModelId,
};

fn main() -> anyhow::Result<()> {
    // Resolve a cached model directory and load the engine once.
    let id = ModelId::parse("Qwen/Qwen2.5-7B-Instruct")?;
    let dir = model_dir(&models_root(), &id);
    if !is_model_present(&dir) {
        eprintln!(
            "model not cached at {} — fetch it first:\n  \
             yatima generate --repo Qwen/Qwen2.5-7B-Instruct --prompt hi",
            dir.display()
        );
        return Ok(());
    }
    let mut engine = Engine::load(&dir, device(false)?)?;

    // ── 1. A conversation with memory (ChatSession owns the transcript) ──
    // The session borrows the engine, so this scope releases it afterwards.
    {
        let mut chat = ChatSession::new(&mut engine, ChatMlTemplate).with_system("Be brief.");
        println!("# conversation");
        for user in [
            "My name is Ada and I work in Rust.",
            "What is my name and language?",
        ] {
            println!("you> {user}");
            println!("bot> {}\n", chat.turn(user)?);
        }
    }

    // ── 2. Inference woven into native control flow ──
    // The model returns a label; the program parses it into a Rust enum and
    // *branches/tallies* on it — exactly what a service-boundary API can't do
    // ergonomically. A fresh classification per item (reset() each time).
    let mut clf = ChatSession::new(&mut engine, ChatMlTemplate).with_system(
        "You are a ticket classifier. Reply with exactly one word: BUG, BILLING, or OTHER.",
    );
    let inbox = [
        "The app crashes whenever I click save.",
        "I was charged twice on my last invoice.",
        "Is there a dark mode setting?",
        "Login returns a 500 error every time.",
    ];

    println!("# triage ({} tickets)", inbox.len());
    let mut bugs = 0;
    let mut billing = 0;
    let mut other = 0;
    for ticket in inbox {
        clf.reset();
        let category = Category::parse(clf.turn(ticket)?);
        match category {
            Category::Bug => bugs += 1,
            Category::Billing => billing += 1,
            Category::Other => other += 1,
        }
        println!("  {category:<8?} <- {ticket}");
    }
    println!("\ntally: {bugs} bug, {billing} billing, {other} other");
    Ok(())
}

/// A typed classification result the program switches on.
#[derive(Debug, Clone, Copy)]
enum Category {
    Bug,
    Billing,
    Other,
}

impl Category {
    /// Map the model's free-text label onto the enum (first keyword wins).
    fn parse(answer: &str) -> Category {
        let a = answer.to_uppercase();
        if a.contains("BUG") {
            Category::Bug
        } else if a.contains("BILLING") {
            Category::Billing
        } else {
            Category::Other
        }
    }
}
