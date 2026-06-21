//! Compare next-token logits after full-prompt prefill versus chunked prefill.
//!
//! This is a small reproducer for backend scheduling bugs: it loads one model,
//! renders one prompt, runs no generation, and prints the top-k next-token
//! candidates plus aggregate logit drift.
//!
//! ```bash
//! cargo run -p yatima-lib --release --example prefill_compare --features metal -- \
//!   ~/.cache/yatima/models/bartowski/THUDM_GLM-4-32B-0414-GGUF glm
//!
//! # smaller synthetic prompt, top-8 comparison
//! cargo run -p yatima-lib --release --example prefill_compare --features metal -- \
//!   ~/.cache/yatima/models/bartowski/THUDM_GLM-4-32B-0414-GGUF glm synthetic:64 64 8
//! ```

use anyhow::{bail, Context, Result};
use std::collections::HashSet;
use std::io::{Read, Write};
use std::path::PathBuf;
use yatima_lib::{
    device, ChatMlTemplate, Engine, GemmaTemplate, GlmTemplate, MistralTemplate, PlainTemplate,
    PrefillLogits, PrefillProgress, PromptTemplate, Role, Turn,
};

const SYSTEM: &str = "\
You are an investment research analyst producing an educational research note. \
Use only the supplied evidence. Every factual claim must cite a metric id. \
If evidence is insufficient, say so.";

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let model_dir = args
        .next()
        .map(PathBuf::from)
        .unwrap_or_else(default_glm_32b_dir);
    let format = args
        .next()
        .as_deref()
        .map(ChatFormat::parse)
        .transpose()?
        .unwrap_or(ChatFormat::Glm);
    let prompt_path = args.next().map(PathBuf::from);
    let chunk = args
        .next()
        .map(|s| s.parse::<usize>().context("parsing chunk size"))
        .transpose()?
        .unwrap_or(64);
    let top_k = args
        .next()
        .map(|s| s.parse::<usize>().context("parsing top-k"))
        .transpose()?
        .unwrap_or(12);

    let user_prompt = match prompt_path {
        Some(path) if path.as_os_str() == "-" => {
            let mut s = String::new();
            std::io::stdin().read_to_string(&mut s)?;
            s
        }
        Some(path) if path.to_string_lossy().starts_with("synthetic:") => {
            let rows = path
                .to_string_lossy()
                .trim_start_matches("synthetic:")
                .parse::<usize>()
                .context("parsing synthetic row count")?;
            synthetic_research_prompt(rows)
        }
        Some(path) => std::fs::read_to_string(&path)
            .with_context(|| format!("reading prompt file {}", path.display()))?,
        None => synthetic_research_prompt(96),
    };
    let prompt = format.render(&[
        Turn {
            role: Role::System,
            content: SYSTEM.to_string(),
        },
        Turn {
            role: Role::User,
            content: user_prompt,
        },
    ]);

    eprintln!(
        "loading {} with format {}; prompt chars {}; comparing full prefill vs chunk {chunk}",
        model_dir.display(),
        format.name(),
        prompt.len()
    );
    let mut engine = Engine::load(&model_dir, device(false)?)
        .with_context(|| format!("loading model {}", model_dir.display()))?;
    eprintln!(
        "loaded {} [{}]; running full prefill",
        model_dir.display(),
        engine.backend()
    );

    let full = run_prefill(&mut engine, &prompt, Some(0), "full prefill")?;
    eprintln!("full prefill complete; running chunked prefill ({chunk})");
    let chunked = run_prefill(
        &mut engine,
        &prompt,
        Some(chunk),
        &format!("chunked prefill ({chunk})"),
    )?;
    eprintln!("chunked prefill complete; printing comparison");
    if full.logits.len() != chunked.logits.len() {
        bail!(
            "vocab size mismatch: full={} chunked={}",
            full.logits.len(),
            chunked.logits.len()
        );
    }

    println!("prompt tokens: {}", full.token_count);
    println!("vocab size: {}", full.logits.len());
    print_logit_health("full prefill", &full.logits);
    print_logit_health(&format!("chunked prefill ({chunk})"), &chunked.logits);
    print_drift(&full.logits, &chunked.logits);
    println!();

    let full_top = engine.topk_from_logits(&full.logits, top_k);
    let chunked_top = engine.topk_from_logits(&chunked.logits, top_k);
    print_topk("full prefill", &full_top);
    println!();
    print_topk(&format!("chunked prefill ({chunk})"), &chunked_top);
    println!();
    print_overlap(&full_top, &chunked_top);

    Ok(())
}

fn run_prefill(
    engine: &mut Engine,
    prompt: &str,
    prefill_chunk: Option<usize>,
    label: &str,
) -> Result<PrefillLogits> {
    let mut last_line_len = 0usize;
    let result = engine.prefill_logits_with_progress(prompt, prefill_chunk, |p| {
        print_progress(label, p, &mut last_line_len);
    });
    if last_line_len > 0 {
        eprintln!();
    }
    result
}

fn print_progress(label: &str, p: PrefillProgress, last_line_len: &mut usize) {
    let status = if p.finished { "done" } else { "running" };
    let completed = p.chunk_index + usize::from(p.finished);
    let percent = (completed as f64 / p.chunk_count as f64) * 100.0;
    let width = 24usize;
    let filled = ((percent / 100.0) * width as f64).round() as usize;
    let bar = format!("{}{}", "#".repeat(filled), "-".repeat(width - filled));
    let line = format!(
        "{label}: [{bar}] {:>5.1}% chunk {}/{} {status} tokens {}..{} of {}",
        percent,
        p.chunk_index + 1,
        p.chunk_count,
        p.start_pos,
        p.end_pos,
        p.token_count
    );
    eprint!("\r{line}");
    if *last_line_len > line.len() {
        eprint!("{}", " ".repeat(*last_line_len - line.len()));
    }
    let _ = std::io::stderr().flush();
    *last_line_len = line.len();
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ChatFormat {
    Qwen,
    Gemma,
    Mistral,
    Glm,
    Plain,
    Raw,
}

impl ChatFormat {
    fn parse(s: &str) -> Result<ChatFormat> {
        match s {
            "qwen" => Ok(ChatFormat::Qwen),
            "gemma" => Ok(ChatFormat::Gemma),
            "mistral" => Ok(ChatFormat::Mistral),
            "glm" => Ok(ChatFormat::Glm),
            "plain" => Ok(ChatFormat::Plain),
            "raw" => Ok(ChatFormat::Raw),
            other => bail!("unknown format {other:?}; expected qwen|gemma|mistral|glm|plain|raw"),
        }
    }

    fn name(self) -> &'static str {
        match self {
            ChatFormat::Qwen => "qwen",
            ChatFormat::Gemma => "gemma",
            ChatFormat::Mistral => "mistral",
            ChatFormat::Glm => "glm",
            ChatFormat::Plain => "plain",
            ChatFormat::Raw => "raw",
        }
    }

    fn render(self, turns: &[Turn]) -> String {
        match self {
            ChatFormat::Qwen => ChatMlTemplate.render(turns),
            ChatFormat::Gemma => GemmaTemplate.render(turns),
            ChatFormat::Mistral => MistralTemplate.render(turns),
            ChatFormat::Glm => GlmTemplate.render(turns),
            ChatFormat::Plain => PlainTemplate.render(turns),
            ChatFormat::Raw => turns
                .iter()
                .map(|t| t.content.as_str())
                .collect::<Vec<_>>()
                .join("\n\n"),
        }
    }
}

fn print_drift(a: &[f32], b: &[f32]) {
    let mut max_abs = 0.0f32;
    let mut sum_sq = 0.0f64;
    let mut finite_pairs = 0usize;
    let mut non_finite_pairs = 0usize;
    let mut changed_finiteness = 0usize;
    for (&x, &y) in a.iter().zip(b) {
        if !x.is_finite() || !y.is_finite() {
            non_finite_pairs += 1;
            if x.is_finite() != y.is_finite() {
                changed_finiteness += 1;
            }
            continue;
        }
        let d = (x - y).abs();
        max_abs = max_abs.max(d);
        sum_sq += f64::from(d * d);
        finite_pairs += 1;
    }
    if finite_pairs == 0 {
        println!("finite-pair logit delta: unavailable (no finite pairs)");
    } else {
        let rms = (sum_sq / finite_pairs as f64).sqrt();
        println!("finite-pair max abs logit delta: {max_abs:.6}");
        println!("finite-pair rms logit delta: {rms:.6}");
    }
    println!("finite pairs: {finite_pairs}");
    println!("non-finite pairs: {non_finite_pairs}");
    println!("changed finiteness: {changed_finiteness}");
}

fn print_logit_health(label: &str, logits: &[f32]) {
    let mut finite = 0usize;
    let mut nan = 0usize;
    let mut pos_inf = 0usize;
    let mut neg_inf = 0usize;
    let mut zeros = 0usize;
    let mut finite_min = f32::INFINITY;
    let mut finite_max = f32::NEG_INFINITY;
    let mut first_bad = Vec::new();

    for (i, &logit) in logits.iter().enumerate() {
        if logit.is_finite() {
            finite += 1;
            zeros += usize::from(logit == 0.0);
            finite_min = finite_min.min(logit);
            finite_max = finite_max.max(logit);
        } else {
            if first_bad.len() < 8 {
                first_bad.push((i, logit));
            }
            if logit.is_nan() {
                nan += 1;
            } else if logit.is_sign_positive() {
                pos_inf += 1;
            } else {
                neg_inf += 1;
            }
        }
    }

    println!(
        "{label} logit health: finite={finite} nan={nan} +inf={pos_inf} -inf={neg_inf} zero={zeros}"
    );
    if finite > 0 {
        println!("{label} finite range: [{finite_min:.6}, {finite_max:.6}]");
    }
    if !first_bad.is_empty() {
        println!("{label} first non-finite logits:");
        for (id, logit) in first_bad {
            println!("  id={id:<7} value={logit:?}");
        }
    }
}

fn print_topk(label: &str, top: &[yatima_lib::TokenLogit]) {
    println!("{label} top-{}:", top.len());
    for (rank, item) in top.iter().enumerate() {
        println!(
            "{:>2}. id={:<7} logit={:>10.4} text={:?}",
            rank + 1,
            item.id,
            item.logit,
            item.text
        );
    }
}

fn print_overlap(a: &[yatima_lib::TokenLogit], b: &[yatima_lib::TokenLogit]) {
    let ids: HashSet<u32> = a.iter().map(|t| t.id).collect();
    let overlap = b.iter().filter(|t| ids.contains(&t.id)).count();
    println!("top-k overlap: {overlap}/{}", a.len().min(b.len()));
}

fn synthetic_research_prompt(rows: usize) -> String {
    let mut s = String::from(
        "Write a concise investment research note from this SEC-like evidence.\n\
         Required sections: thesis, evidence, risks, testable signals.\n\n\
         Evidence JSON:\n[\n",
    );
    for i in 0..rows {
        let metric = match i % 8 {
            0 => "Revenues",
            1 => "OperatingIncomeLoss",
            2 => "NetIncomeLoss",
            3 => "CapitalExpenditures",
            4 => "PaymentsForRepurchaseOfCommonStock",
            5 => "NetCashProvidedByUsedInOperatingActivities",
            6 => "CashAndCashEquivalentsAtCarryingValue",
            _ => "WeightedAverageNumberOfDilutedSharesOutstanding",
        };
        let value = 1_000_000_000i64 + (i as i64 * 37_000_000);
        s.push_str(&format!(
            "  {{\"id\":\"M{i:03}\",\"ticker\":\"META\",\"tag\":\"{metric}\",\
             \"period\":\"2025-Q{}\",\"filed\":\"2026-0{}-15\",\
             \"accession\":\"0001326801-26-{i:06}\",\"value_text\":\"{value} USD\"}},\n",
            (i % 4) + 1,
            (i % 9) + 1,
        ));
    }
    s.push_str("]\n");
    s
}

fn default_glm_32b_dir() -> PathBuf {
    let home = std::env::var_os("HOME").unwrap_or_else(|| ".".into());
    PathBuf::from(home).join(".cache/yatima/models/bartowski/THUDM_GLM-4-32B-0414-GGUF")
}
