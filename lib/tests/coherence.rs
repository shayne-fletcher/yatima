//! Gated, table-driven coherence e2e.
//!
//! Each cached model follows a simple instruction and answers correctly — a
//! guard against quantized/loader gibberish (cf. the GLM-4 prefill bug) — and
//! the mechanical guard for REASON-1: a reasoning model is validated on its
//! *answer* ([`split_reasoning`]), not its scratchpad, and is asserted to emit a
//! reasoning span at all.
//!
//! Gated on `YATIMA_E2E=1`; every row skips if its weights aren't cached, so the
//! suite is a no-op on a fresh checkout. Heavy (Kimi ~45 GB); run with
//! `cargo test -p yatima-lib --features metal --test coherence -- --nocapture`.

use yatima_lib::{
    device, is_model_present, model_dir, models_root, split_reasoning, Channel, ChatFormat,
    ChatMlTemplate, ChatSession, Engine, GenOpts, ModelId, ModelProfile, ReasoningSplitter, Role,
    Turn,
};

/// One coherence row: a cached model, the format to speak, a prompt, and what a
/// coherent answer must contain.
struct Case {
    /// HuggingFace `org/name` (resolved under `models_root`).
    repo: &'static str,
    /// The chat format to render the prompt in (the model's native format).
    format: ChatFormat,
    /// A single user prompt.
    prompt: &'static str,
    /// A substring the *answer* must contain; `""` asserts only a non-empty,
    /// coherent answer (for open-ended prompts).
    expect: &'static str,
    /// Whether this is a reasoning model — if so, it must emit a reasoning span
    /// (REASON-1), proving the channel split fired on real output.
    expects_reasoning: bool,
    /// Token budget. Reasoning models need room to think *and* answer; too small
    /// truncates mid-thought (no closing marker → no answer).
    max_tokens: usize,
}

const CASES: &[Case] = &[
    // Non-reasoning: Gemma-2 follows an open-ended instruction.
    Case {
        repo: "google/gemma-2-2b-it",
        format: ChatFormat::Gemma,
        prompt: "Explain Rust in one sentence.",
        expect: "",
        expects_reasoning: false,
        max_tokens: 64,
    },
    // Non-reasoning: Qwen2.5 answers a factual prompt tersely.
    Case {
        repo: "Qwen/Qwen2.5-7B-Instruct",
        format: ChatFormat::Qwen,
        prompt: "What is 2 + 2? Reply with only the number.",
        expect: "4",
        expects_reasoning: false,
        max_tokens: 32,
    },
    // Reasoning, `<think>` dialect (cheap): a DeepSeek-R1 distill thinks, then
    // answers — exercises the channel split end-to-end without a 45 GB load.
    // Uses the native DeepSeek format (the distill is Qwen2 arch but trained on
    // DeepSeek's template; the cue pre-seeds <think> so output carries the close
    // marker only).
    Case {
        repo: "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        format: ChatFormat::DeepSeek,
        prompt: "What is 2 + 2? Reply with only the number.",
        expect: "4",
        expects_reasoning: true,
        max_tokens: 512,
    },
    // Reasoning, Kimi `◁think▷` dialect (heavy): the case that motivated
    // REASON-1 — its markers are special tokens the old `</think>` strip missed.
    Case {
        repo: "unsloth/Kimi-Dev-72B-GGUF",
        format: ChatFormat::Qwen,
        prompt: "What is 2 + 2? Reply with only the number.",
        expect: "4",
        expects_reasoning: true,
        max_tokens: 1024,
    },
];

fn gated() -> bool {
    std::env::var_os("YATIMA_E2E").is_some()
}

/// First `n` chars of a trace, for readable logs.
fn head(s: &str, n: usize) -> String {
    s.chars().take(n).collect()
}

fn run_case(c: &Case) -> anyhow::Result<()> {
    let dir = model_dir(&models_root(), &ModelId::parse(c.repo)?);
    if !is_model_present(&dir) {
        eprintln!("skip {}: not cached", c.repo);
        return Ok(());
    }
    let mut engine = Engine::load(&dir, device(false)?)?;
    let prompt = c.format.template().render(&[Turn {
        role: Role::User,
        content: c.prompt.to_string(),
    }]);
    let opts = GenOpts {
        max_tokens: c.max_tokens,
        ..Default::default()
    };
    let mut out = String::new();
    engine.generate(&prompt, &opts, |s| {
        out.push_str(s);
        Ok(())
    })?;

    // Validate on the answer, never the scratchpad (REASON-1).
    let split = split_reasoning(&out);
    eprintln!(
        "{} → reasoning={:?} answer={:?}",
        c.repo,
        split.reasoning.as_deref().map(|r| head(r, 60)),
        head(&split.answer, 120),
    );

    if c.expect.is_empty() {
        assert!(
            split.answer.trim().len() > 10,
            "{}: expected a coherent answer, got {:?}",
            c.repo,
            split.answer
        );
    } else {
        assert!(
            split.answer.contains(c.expect),
            "{}: answer must contain {:?}, got {:?}",
            c.repo,
            c.expect,
            split.answer
        );
    }
    if c.expects_reasoning {
        assert!(
            split.reasoning.is_some(),
            "{}: expected a reasoning span (REASON-1) — raise max_tokens if the \
             think block was truncated. raw head: {:?}",
            c.repo,
            head(&out, 200)
        );
    }
    Ok(())
}

#[test]
fn coherence_across_cached_models() -> anyhow::Result<()> {
    if !gated() {
        eprintln!("skipping e2e: set YATIMA_E2E=1 to run");
        return Ok(());
    }
    for case in CASES {
        run_case(case)?;
    }
    Ok(())
}

/// Every built-in **reasoning profile** actually surfaces a reasoning channel
/// when driven through the path the TUI uses: render with the profile's format,
/// pick the splitter by [`ChatFormat::pre_seeds_reasoning`], and classify the
/// output. This is the guard that catches a profile↔model format mismatch — the
/// QwQ bug, where a pre-seeding model on a plain-`Qwen` (non-seeding) format ran
/// the `new()` splitter and showed its reasoning as the answer (reasoning span
/// empty). Pre-seed-vs-emit is a model behavior, so this needs the real model;
/// gated, skips uncached profiles. (Kimi is skipped — MEM-1 would refuse it.)
#[test]
fn reasoning_profiles_surface_a_reasoning_channel() -> anyhow::Result<()> {
    if !gated() {
        eprintln!("skipping e2e: set YATIMA_E2E=1 to run");
        return Ok(());
    }
    for name in ["deepseek-r1", "qwq"] {
        let profile = ModelProfile::builtin(name).unwrap();
        assert!(profile.reasoning, "{name} should be a reasoning profile");
        // Resolve offline; skip if not cached.
        let dir = match profile.to_source(true).and_then(|s| s.resolve()) {
            Ok(dir) => dir,
            Err(_) => {
                eprintln!("skip {name}: not cached");
                continue;
            }
        };
        let format = profile.format().expect("a reasoning profile pins a format");
        let mut engine = Engine::load(&dir, device(false)?)?;
        let prompt = format.template().render(&[Turn {
            role: Role::User,
            content: "What is 2 + 2?".to_string(),
        }]);
        // A small budget suffices: with the right (seeded) splitter, the output so
        // far is reasoning even before `</think>`; with the wrong (new) splitter
        // it would all be answer.
        let opts = GenOpts {
            max_tokens: 64,
            ..Default::default()
        };
        let mut out = String::new();
        engine.generate(&prompt, &opts, |s| {
            out.push_str(s);
            Ok(())
        })?;
        // Classify exactly as the TUI does: splitter chosen by the format.
        let mut splitter = if format.pre_seeds_reasoning() {
            ReasoningSplitter::seeded()
        } else {
            ReasoningSplitter::new()
        };
        let mut reasoning = String::new();
        splitter.push(&out, |ch, t| {
            if ch == Channel::Reasoning {
                reasoning.push_str(t);
            }
        });
        splitter.finish(|ch, t| {
            if ch == Channel::Reasoning {
                reasoning.push_str(t);
            }
        });
        eprintln!("{name} ({format:?}): reasoning_chars={}", reasoning.len());
        assert!(
            !reasoning.trim().is_empty(),
            "{name} ({format:?}) surfaced NO reasoning — profile/format mismatch \
             (e.g. a pre-seeding model on a non-seeding format). raw: {:?}",
            out.chars().take(80).collect::<String>()
        );
    }
    Ok(())
}

/// The streaming reasoning channel is clean on a real reasoning model: routing
/// the live token stream through a [`ReasoningSplitter`] (exactly as the chat
/// REPL does) must never leak a marker into the answer channel (REASON-1). Dumps
/// the raw fragments for diagnosis.
#[test]
fn streaming_reasoning_channel_is_clean() -> anyhow::Result<()> {
    if !gated() {
        eprintln!("skipping e2e: set YATIMA_E2E=1 to run");
        return Ok(());
    }
    let dir = model_dir(
        &models_root(),
        &ModelId::parse("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")?,
    );
    if !is_model_present(&dir) {
        eprintln!("skip streaming-channel: DeepSeek-R1-Distill-Qwen-1.5B not cached");
        return Ok(());
    }
    let mut engine = Engine::load(&dir, device(false)?)?;
    let format = ChatFormat::DeepSeek;
    let mut chat = ChatSession::new(&mut engine, format.template()).with_opts(GenOpts {
        max_tokens: 700,
        ..Default::default()
    });

    let mut splitter = if format.pre_seeds_reasoning() {
        ReasoningSplitter::seeded()
    } else {
        ReasoningSplitter::new()
    };
    let mut reasoning = String::new();
    let mut answer = String::new();
    chat.turn_streaming("how many 'e's are there in 'greece'?", &mut |piece| {
        eprintln!("RAW FRAG: {piece:?}");
        splitter.push(piece, |ch, text| match ch {
            Channel::Reasoning => reasoning.push_str(text),
            Channel::Answer => answer.push_str(text),
        });
    })?;
    splitter.finish(|ch, text| match ch {
        Channel::Reasoning => reasoning.push_str(text),
        Channel::Answer => answer.push_str(text),
    });
    eprintln!("--- reasoning: {reasoning:?}");
    eprintln!("--- answer:    {answer:?}");
    assert!(
        !answer.contains("think"),
        "reasoning marker leaked into the answer channel: {answer:?}"
    );
    Ok(())
}

/// Multi-turn memory through the real chat path: push a fact, then ask it back,
/// re-rendering the whole transcript. Exercises [`ChatSession`] — including the
/// reasoning split on each reply (REASON-1) — over a reliable-recall model.
#[test]
fn chat_remembers_across_turns() -> anyhow::Result<()> {
    if !gated() {
        eprintln!("skipping e2e: set YATIMA_E2E=1 to run");
        return Ok(());
    }
    let dir = model_dir(&models_root(), &ModelId::parse("Qwen/Qwen2.5-7B-Instruct")?);
    if !is_model_present(&dir) {
        eprintln!("skip chat-memory: Qwen2.5-7B-Instruct not cached");
        return Ok(());
    }
    let mut engine = Engine::load(&dir, device(false)?)?;
    let mut chat = ChatSession::new(&mut engine, ChatMlTemplate).with_opts(GenOpts {
        max_tokens: 64,
        ..Default::default()
    });
    let a1 = chat
        .turn("My name is Ada. Please remember it.")?
        .to_string();
    let a2 = chat.turn("What is my name?")?.to_string();
    eprintln!("turn1 → {a1:?}\nturn2 → {a2:?}");
    assert!(
        a2.contains("Ada"),
        "second answer must recall the name from turn 1, got {a2:?}"
    );
    Ok(())
}
