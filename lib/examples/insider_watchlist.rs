//! Build an auditable insider-buy watchlist from real SEC Form 4 filings.
//!
//! Rust resolves tickers, fetches recent ownership filings from EDGAR, parses
//! Form 4 XML into typed transaction evidence, keeps only open-market `P`
//! purchases, sets aside everything else (awards, exercises, sales) as
//! disqualifiers, flags Rule 10b5-1 plan buys, and **assigns each issuer a
//! deterministic signal tier** (strong / moderate / weak / noise) from the
//! evidence. A local chat model then writes a note that may not claim more
//! conviction than Rust assigned; the example-local validator audits it.
//!
//! ```bash
//! SEC_USER_AGENT="your-name your-email@example.com" \
//! cargo run -p yatima-lib --release --example insider_watchlist --features metal -- \
//!   --ticker JPM --profile mistral --days 365
//!
//! SEC_USER_AGENT="your-name your-email@example.com" \
//! cargo run -p yatima-lib --release --example insider_watchlist --features metal -- \
//!   --tickers JPM,META,ABNB,NFLX,NVDA --no-model
//!
//! # deterministic model demo without touching SEC
//! cargo run -p yatima-lib --release --example insider_watchlist --features metal -- \
//!   --demo --profile mistral
//! ```
//!
//! Example-level invariants (cited in tests with `// upholds: <id>`):
//! - **IW-1** ticker planning is pure and normalizes `--ticker`/`--tickers` to
//!   a deduplicated uppercase list before any network or model work.
//! - **IW-2** every watchlist signal is derived from a parsed Form 4
//!   non-derivative transaction with code `P`, acquired/disposed code `A`, and
//!   positive shares.
//! - **IW-3** the issuer signal tier is assigned in Rust from the evidence
//!   before the model loads; the model note may not claim a stronger tier, and
//!   the validator flags overstatement, unknown cited accessions, claims that
//!   mischaracterize a non-discretionary row as an open-market purchase, and
//!   price/drawdown claims (no price evidence is supplied).
//! - **IW-4** Rule 10b5-1 plan buys (footnote heuristic) and non-`P`/`A`
//!   transactions are retained as evidence/disqualifiers but excluded from
//!   cluster, score, and tier.
//! - **IW-5** live SEC access is metered: the client cannot send faster than a
//!   hard spacing floor and cannot exceed a per-run request budget, and any 429
//!   stops the run rather than retrying into a longer block.

use anyhow::{anyhow, bail, Context, Result};
use chrono::{Duration as ChronoDuration, Utc};
use clap::builder::{PossibleValuesParser, TypedValueParser};
use clap::Parser;
use quick_xml::de::from_str as xml_from_str;
use reqwest::blocking::Client;
use reqwest::StatusCode;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::thread;
use std::time::{Duration, Instant};
use yatima_lib::{
    device, resolve_format, run_blocking, ChatFormat, ChatSession, Engine, GenOpts, ModelProfile,
    Sampling,
};

const COMPANY_TICKERS_URL: &str = "https://www.sec.gov/files/company_tickers.json";
const SUBMISSIONS_BASE_URL: &str = "https://data.sec.gov/submissions";
const ARCHIVES_BASE_URL: &str = "https://www.sec.gov/Archives/edgar/data";

/// Hard minimum spacing between live SEC requests, in milliseconds. SEC fair
/// access allows up to ~10 requests/second; this floor (~6.6/s) keeps a margin
/// and cannot be lowered by a flag, so no `--sec-delay-ms` value can provoke a
/// 429 (IW-5).
const SEC_FLOOR_MS: u64 = 150;

/// Signal-tier thresholds (USD), seeded from the insider-buy rubric. Discretionary
/// = an open-market `P`/`A` purchase that is not a Rule 10b5-1 plan transaction.
const STRONG_SINGLE_USD: f64 = 500_000.0;
const STRONG_TOTAL_USD: f64 = 1_000_000.0;
const MODERATE_SINGLE_USD: f64 = 250_000.0;
const WEAK_MAX_USD: f64 = 100_000.0;

const SYSTEM: &str = "\
You are ranking insider-buy signals for a stock watchlist. Use only the \
supplied SEC Form 4 evidence. Every factual claim about an insider purchase \
must cite ticker, owner, accession, transaction_date, shares, price, and \
value_text from the supplied JSON. Prefer open-market P purchases with large \
dollar value, cluster buying, and senior insider roles. Each company carries a \
Rust-assigned signal tier (strong, moderate, weak, or noise); treat that tier \
as a ceiling and never describe a company with stronger conviction than its \
assigned tier. Buys flagged is_10b5_1, and any row listed under disqualifiers, \
are NOT discretionary open-market purchases: never present them as conviction \
buys. No price history is supplied, so do not claim a purchase happened near \
lows, after a drawdown, or 'on the dip'. Separate ranked watchlist, evidence, \
risks, and follow-up research.";

/// Parse recent SEC Form 4 filings into insider-buy signals and optionally ask
/// a local model to rank the watchlist.
#[derive(Debug, Parser)]
#[command(about, long_about = None)]
struct Args {
    /// Single ticker symbol; may be repeated.
    #[arg(long)]
    ticker: Vec<String>,
    /// Comma-delimited ticker symbols.
    #[arg(long, value_delimiter = ',')]
    tickers: Vec<String>,
    /// Look back this many calendar days in the issuer's recent filing history.
    #[arg(long, default_value_t = 365)]
    days: i64,
    /// Maximum recent Form 4 filings to inspect per ticker.
    #[arg(long, default_value_t = 40)]
    limit_filings: usize,
    /// Minimum transaction value to keep, in USD.
    #[arg(long, default_value_t = 0.0)]
    min_value_usd: f64,
    /// Spacing between SEC requests, in ms. Clamped up to a hard floor
    /// (~6.6 req/s) that no value can cross, so lower values just run nearer the
    /// floor — they cannot provoke a 429. SEC allows ~10 req/s.
    #[arg(long, default_value_t = 300)]
    sec_delay_ms: u64,
    /// Per-run request budget: a backstop against runaway scans, not a usage
    /// cap. Raise it freely for wider discovery (each ticker costs ~1 + one per
    /// Form 4 fetched).
    #[arg(long, default_value_t = 300)]
    sec_max_requests: usize,
    /// Print evidence and skip the model pass.
    #[arg(long)]
    no_model: bool,
    /// Use a bundled deterministic evidence fixture instead of live SEC.
    #[arg(long)]
    demo: bool,
    /// Replay a previously captured watchlist JSON file instead of live SEC.
    #[arg(long)]
    evidence: Option<PathBuf>,
    /// Fetch explicit SEC filings as TICKER:CIK:ACCESSION, avoiding broad scans.
    #[arg(long)]
    filing: Vec<String>,
    /// Save the resolved evidence JSON for deterministic replay/backtesting.
    #[arg(long)]
    save_evidence: Option<PathBuf>,
    /// Print full JSON evidence before the model prompt.
    #[arg(long)]
    json: bool,
    /// Built-in profile name (one of `ModelProfile::BUILTIN_NAMES`). Defaults to
    /// `qwen32b` for real screens (strong enough to cite and respect the tier
    /// ceiling); `--demo` defaults to the lighter `mistral`.
    #[arg(long)]
    profile: Option<String>,
    /// Explicit model directory (overrides --profile's source).
    #[arg(long)]
    model: Option<PathBuf>,
    /// Repository id, resolved (and fetched on a miss) under the models root.
    #[arg(long)]
    repo: Option<String>,
    /// With --repo, the single GGUF quant to fetch.
    #[arg(long)]
    gguf: Option<String>,
    /// Chat format; omit to infer from the model's architecture.
    #[arg(long, value_parser = chat_format_parser())]
    format: Option<ChatFormat>,
    /// Prompt prefill chunk; omit for the model/backend default.
    #[arg(long)]
    prefill_chunk: Option<usize>,
    #[arg(long, default_value_t = 700)]
    max_tokens: usize,
    #[arg(long, default_value_t = 0.0)]
    temperature: f64,
    #[arg(long, default_value_t = 0)]
    seed: u64,
    /// Don't auto-fetch a missing model; error instead.
    #[arg(long)]
    offline: bool,
    /// Force CPU instead of the GPU.
    #[arg(long)]
    cpu: bool,
}

/// A clap value parser for [`ChatFormat`] (names -> enum).
fn chat_format_parser() -> impl TypedValueParser<Value = ChatFormat> {
    PossibleValuesParser::new(ChatFormat::NAMES)
        .map(|s| s.parse::<ChatFormat>().expect("NAMES are valid formats"))
}

#[tokio::main(flavor = "multi_thread")]
async fn main() -> Result<()> {
    let args = Args::parse();
    let watchlist = load_watchlist(&args)?;
    if let Some(path) = &args.save_evidence {
        fs::write(path, serde_json::to_string_pretty(&watchlist)?)
            .with_context(|| format!("writing {}", path.display()))?;
        eprintln!("saved evidence to {}", path.display());
    }

    eprintln!(
        "fetched {} insider-buy signals across {} tickers",
        watchlist.signals.len(),
        watchlist.companies.len()
    );
    if args.json || args.no_model {
        println!("{}", serde_json::to_string_pretty(&watchlist)?);
    } else {
        print_human_summary(&watchlist);
    }
    if args.no_model {
        return Ok(());
    }

    let prompt = build_prompt(&watchlist)?;
    let profile = model_profile(&args)?;
    let dir = profile.to_source(args.offline)?.resolve()?;
    let dev = device(args.cpu)?;
    let mut engine = run_blocking(|| Engine::load(&dir, dev))
        .with_context(|| format!("loading {}", dir.display()))?;
    let (format, mismatch) = resolve_format(engine.arch(), profile.format);
    if let Some(m) = mismatch {
        eprintln!("warning: {m}");
    }
    let opts = profile.apply_gen_overrides(GenOpts {
        max_tokens: args.max_tokens,
        sampling: Sampling::from_temperature(args.temperature, args.seed),
        prefill_chunk: args.prefill_chunk,
        ..Default::default()
    });
    let prompt_tokens = engine.token_count(&prompt).unwrap_or(0);
    print_run_metadata(&profile, &dir, &engine, format, &opts, prompt_tokens);

    let mut chat = ChatSession::new(&mut engine, format.template())
        .with_system(SYSTEM)
        .with_opts(opts);
    let mut stdout = std::io::stdout();
    let answer = chat
        .turn_streaming_async(&prompt, &mut |piece| {
            let _ = stdout.write_all(piece.as_bytes());
            let _ = stdout.flush();
        })
        .await?
        .to_string();
    println!();
    if let Some(stop) = chat.last_stop() {
        eprintln!("[stop: {stop:?}]");
    }

    let check = check_watchlist_note(&answer, &watchlist);
    if !check.is_clean() {
        eprintln!("\nvalidation warnings:");
        for warning in check.warnings {
            eprintln!("- {warning}");
        }
    }
    Ok(())
}

fn load_watchlist(args: &Args) -> Result<InsiderWatchlist> {
    let sources = usize::from(args.demo)
        + usize::from(args.evidence.is_some())
        + usize::from(!args.filing.is_empty());
    if sources > 1 {
        bail!("pass only one of --demo, --evidence, or --filing");
    }

    match (args.demo, args.evidence.as_ref(), args.filing.is_empty()) {
        (true, None, true) => Ok(demo_watchlist()),
        (false, Some(path), true) => {
            let json =
                fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
            serde_json::from_str(&json).with_context(|| format!("decoding {}", path.display()))
        }
        (false, None, false) => {
            let user_agent = require_user_agent()?;
            let filings = args
                .filing
                .iter()
                .map(|spec| parse_filing_spec(spec))
                .collect::<Result<Vec<_>>>()?;
            let (delay, max) = (args.sec_delay_ms, args.sec_max_requests);
            // Build, use, and drop the blocking client entirely inside the
            // blocking island; reqwest's internal runtime must not be dropped on
            // an async worker (RT-1).
            run_blocking(move || {
                let client = sec_client(user_agent, delay, max)?;
                fetch_explicit_filings(&client, &filings)
            })
        }
        (false, None, true) => {
            let tickers = plan_tickers(args)?;
            let user_agent = require_user_agent()?;
            let request = WatchlistRequest {
                tickers,
                days: args.days,
                limit_filings: args.limit_filings,
                min_value_usd: args.min_value_usd,
                sec_delay_ms: args.sec_delay_ms,
                sec_max_requests: args.sec_max_requests,
            };
            run_blocking(move || {
                let client =
                    sec_client(user_agent, request.sec_delay_ms, request.sec_max_requests)?;
                fetch_watchlist(&client, &request)
            })
        }
        _ => unreachable!("source count checked above"),
    }
}

fn require_user_agent() -> Result<String> {
    std::env::var("SEC_USER_AGENT").context(
        "set SEC_USER_AGENT to a descriptive value with contact info, \
         e.g. 'Your Name your.email@example.com'",
    )
}

fn sec_client(user_agent: String, delay_ms: u64, max_requests: usize) -> Result<SecClient> {
    Ok(SecClient::new(
        Client::builder()
            .user_agent(user_agent)
            .timeout(Duration::from_secs(30))
            .build()?,
        delay_ms,
        max_requests,
    ))
}

/// The default model profile when `--profile` is omitted: the strong `qwen32b`
/// for real screens, the lighter `mistral` for the offline `--demo`.
fn default_profile_name(demo: bool) -> &'static str {
    if demo {
        "mistral"
    } else {
        "qwen32b"
    }
}

fn model_profile(args: &Args) -> Result<ModelProfile> {
    let name = args
        .profile
        .clone()
        .unwrap_or_else(|| default_profile_name(args.demo).to_string());
    let mut profile = ModelProfile::builtin(&name).ok_or_else(|| {
        anyhow!(
            "unknown profile {name:?}; built-ins: {:?}",
            ModelProfile::BUILTIN_NAMES
        )
    })?;
    if args.model.is_some() {
        profile.dir = args.model.clone();
        profile.repo = None;
    }
    if args.repo.is_some() {
        profile.repo = args.repo.clone();
        profile.dir = None;
    }
    if args.gguf.is_some() {
        profile.gguf = args.gguf.clone();
    }
    if args.format.is_some() {
        profile.format = args.format;
    }
    if args.prefill_chunk.is_some() {
        profile.prefill_chunk = args.prefill_chunk;
    }
    Ok(profile)
}

fn plan_tickers(args: &Args) -> Result<Vec<String>> {
    let mut seen = BTreeSet::new();
    for ticker in args.ticker.iter().chain(args.tickers.iter()) {
        let normalized = ticker.trim().to_ascii_uppercase();
        if !normalized.is_empty() {
            seen.insert(normalized);
        }
    }
    if seen.is_empty() {
        bail!("pass at least one --ticker or --tickers value");
    }
    Ok(seen.into_iter().collect())
}

fn print_run_metadata(
    profile: &ModelProfile,
    dir: &Path,
    engine: &Engine,
    format: ChatFormat,
    opts: &GenOpts,
    prompt_tokens: usize,
) {
    let prefill = match opts.prefill_chunk.or(engine.default_prefill_chunk()) {
        Some(0) | None => "full-prompt".to_string(),
        Some(n) => format!("{n} tokens"),
    };
    let sampling = match opts.sampling {
        Sampling::Greedy => "greedy".to_string(),
        Sampling::Sample { temperature, seed } => format!("sample t={temperature} seed={seed}"),
    };
    eprintln!(
        "run: profile={} source={} arch={:?} format={} backend={} prefill={} max_tokens={} \
         sampling={} prompt_tokens={}",
        profile.name,
        dir.display(),
        engine.arch(),
        format,
        engine.backend(),
        prefill,
        opts.max_tokens,
        sampling,
        prompt_tokens,
    );
}

fn print_human_summary(watchlist: &InsiderWatchlist) {
    for company in &watchlist.companies {
        println!(
            "[{}] {} ({}) — {}",
            company.tier.label().to_uppercase(),
            company.ticker,
            company.company,
            company.tier_rationale.join("; "),
        );
    }
    for signal in &watchlist.signals {
        let plan = if signal.is_10b5_1 { " 10b5-1" } else { "" };
        println!(
            "{} {} {} {} {}{} {} score={} accession={}",
            signal.ticker,
            signal.transaction_date,
            signal.owner,
            signal.role,
            signal.value_text,
            plan,
            signal.reason_text,
            signal.score,
            signal.accession
        );
    }
    for disq in &watchlist.disqualifiers {
        println!(
            "excluded: {} {} {} code={} ({})",
            disq.ticker, disq.transaction_date, disq.owner, disq.transaction_code, disq.reason,
        );
    }
}

/// A compact, model-facing projection of the watchlist. The full
/// [`InsiderWatchlist`] stays the auditable artifact (`--json` / `--save-evidence`);
/// the model only needs what it must cite, so this drops the bulky internal
/// fields (`archive_url`, `reasons`, both CIKs, `form`, …). It is ~10x fewer
/// tokens, which keeps a wide screen inside a 32k context window.
#[derive(Serialize)]
struct PromptView<'a> {
    verdict: Vec<PromptVerdict<'a>>,
    buys: Vec<PromptBuy<'a>>,
    excluded: Vec<PromptExcluded<'a>>,
}

#[derive(Serialize)]
struct PromptVerdict<'a> {
    ticker: &'a str,
    company: &'a str,
    tier: &'static str,
    why: &'a [String],
}

#[derive(Serialize)]
struct PromptBuy<'a> {
    ticker: &'a str,
    owner: &'a str,
    role: &'a str,
    transaction_date: &'a str,
    value_text: &'a str,
    accession: &'a str,
    is_10b5_1: bool,
}

#[derive(Serialize)]
struct PromptExcluded<'a> {
    ticker: &'a str,
    owner: &'a str,
    transaction_code: &'a str,
    accession: &'a str,
    reason: &'a str,
}

fn prompt_view(watchlist: &InsiderWatchlist) -> PromptView<'_> {
    PromptView {
        verdict: watchlist
            .companies
            .iter()
            .map(|c| PromptVerdict {
                ticker: &c.ticker,
                company: &c.company,
                tier: c.tier.label(),
                why: &c.tier_rationale,
            })
            .collect(),
        buys: watchlist
            .signals
            .iter()
            .map(|s| PromptBuy {
                ticker: &s.ticker,
                owner: &s.owner,
                role: &s.role,
                transaction_date: &s.transaction_date,
                value_text: &s.value_text,
                accession: &s.accession,
                is_10b5_1: s.is_10b5_1,
            })
            .collect(),
        excluded: watchlist
            .disqualifiers
            .iter()
            .map(|d| PromptExcluded {
                ticker: &d.ticker,
                owner: &d.owner,
                transaction_code: &d.transaction_code,
                accession: &d.accession,
                reason: &d.reason,
            })
            .collect(),
    }
}

fn build_prompt(watchlist: &InsiderWatchlist) -> Result<String> {
    let evidence_json = serde_json::to_string_pretty(&prompt_view(watchlist))?;
    let tiers = watchlist
        .companies
        .iter()
        .map(|company| {
            format!(
                "- {} ({}): assigned tier = {} [{}]",
                company.ticker,
                company.company,
                company.tier.label(),
                company.tier_rationale.join("; "),
            )
        })
        .collect::<Vec<_>>()
        .join("\n");
    Ok(format!(
        "\
Rank these insider-buy signals as a stock-selection watchlist.

Rust-assigned signal tiers (a ceiling — do not describe any company with more
conviction than its assigned tier):
{tiers}

Required format:
- Ranked watchlist: 3-8 numbered bullets, each in EXACTLY this shape:
    N. TICKER — OWNER (ROLE) — value_text — transaction_date — accession ACCESSION
  Copy value_text and the accession verbatim from the evidence's `accession`
  field. A bullet that omits its accession is invalid and must not appear. Do
  not claim a stronger tier than the one assigned above.
- Signal strength: explain which mechanical reasons matter most.
- False-positive risks: explain which signals could be optics, compensation
  related, stale, or too small to matter. Rows under `excluded` and buys with
  `is_10b5_1: true` are not discretionary purchases — treat them as risks, never
  as conviction buys.
- Follow-up research: concrete public-data checks to run next.

SEC Form 4 evidence JSON (compact view of the audited watchlist):
```json
{evidence_json}
```"
    ))
}

fn demo_watchlist() -> InsiderWatchlist {
    let request = WatchlistRequest {
        tickers: vec!["DEMOA".into(), "DEMOB".into(), "DEMOC".into()],
        days: 365,
        limit_filings: 8,
        min_value_usd: 0.0,
        sec_delay_ms: 0,
        sec_max_requests: 0,
    };
    let mut companies = vec![
        CompanyEvidence {
            ticker: "DEMOA".into(),
            cik: "0001000001".into(),
            company: "Demo Applied Systems Inc.".into(),
            tier: Tier::Noise,
            tier_rationale: Vec::new(),
        },
        CompanyEvidence {
            ticker: "DEMOB".into(),
            cik: "0001000002".into(),
            company: "Demo Regional Bancorp".into(),
            tier: Tier::Noise,
            tier_rationale: Vec::new(),
        },
        CompanyEvidence {
            ticker: "DEMOC".into(),
            cik: "0001000003".into(),
            company: "Demo Consumer Platform Corp.".into(),
            tier: Tier::Noise,
            tier_rationale: Vec::new(),
        },
    ];
    // DEMOA: two insiders, one > $1M -> strong. DEMOB: single $484k -> moderate.
    // DEMOC: a $92k discretionary buy (weak) plus a $300k Rule 10b5-1 plan buy
    // that is shown but excluded, so DEMOC stays weak (IW-4).
    let mut signals = vec![
        demo_signal(DemoSignal {
            ticker: "DEMOA",
            company: "Demo Applied Systems Inc.",
            issuer_cik: "0001000001",
            owner: "Ada Founder",
            role: "Chief Executive Officer",
            accession: "0001000001-26-000101",
            transaction_date: "2026-05-14",
            shares: 25_000.0,
            price: 42.25,
            post_transaction_shares: Some(525_000.0),
            is_10b5_1: false,
        }),
        demo_signal(DemoSignal {
            ticker: "DEMOA",
            company: "Demo Applied Systems Inc.",
            issuer_cik: "0001000001",
            owner: "Grace Director",
            role: "director",
            accession: "0001000001-26-000102",
            transaction_date: "2026-05-16",
            shares: 8_000.0,
            price: 41.80,
            post_transaction_shares: Some(88_000.0),
            is_10b5_1: false,
        }),
        demo_signal(DemoSignal {
            ticker: "DEMOB",
            company: "Demo Regional Bancorp",
            issuer_cik: "0001000002",
            owner: "Edsger Chair",
            role: "director",
            accession: "0001000002-26-000055",
            transaction_date: "2026-04-29",
            shares: 40_000.0,
            price: 12.10,
            post_transaction_shares: Some(440_000.0),
            is_10b5_1: false,
        }),
        demo_signal(DemoSignal {
            ticker: "DEMOC",
            company: "Demo Consumer Platform Corp.",
            issuer_cik: "0001000003",
            owner: "Barbara Officer",
            role: "Chief Financial Officer",
            accession: "0001000003-26-000077",
            transaction_date: "2026-03-18",
            shares: 1_200.0,
            price: 77.00,
            post_transaction_shares: Some(91_200.0),
            is_10b5_1: false,
        }),
        demo_signal(DemoSignal {
            ticker: "DEMOC",
            company: "Demo Consumer Platform Corp.",
            issuer_cik: "0001000003",
            owner: "Carl Planner",
            role: "director",
            accession: "0001000003-26-000078",
            transaction_date: "2026-03-20",
            shares: 5_000.0,
            price: 60.00,
            post_transaction_shares: Some(55_000.0),
            is_10b5_1: true,
        }),
    ];
    let disqualifiers = vec![Disqualifier {
        ticker: "DEMOA".into(),
        owner: "Ada Founder".into(),
        role: "Chief Executive Officer".into(),
        accession: "0001000001-26-000103".into(),
        transaction_date: "2026-05-14".into(),
        transaction_code: "A".into(),
        acquired_disposed: "A".into(),
        shares: 12_000.0,
        value_text: dollars_text(0.0),
        is_10b5_1: false,
        reason: disqualifier_reason("A", "A"),
    }];
    apply_cluster_scores(&mut signals);
    rank_signals(&mut signals);
    assign_tiers(&mut companies, &signals);
    InsiderWatchlist {
        source: "bundled deterministic demo evidence for the insider-watchlist workflow".into(),
        request,
        companies,
        signals,
        disqualifiers,
    }
}

struct DemoSignal<'a> {
    ticker: &'a str,
    company: &'a str,
    issuer_cik: &'a str,
    owner: &'a str,
    role: &'a str,
    accession: &'a str,
    transaction_date: &'a str,
    shares: f64,
    price: f64,
    post_transaction_shares: Option<f64>,
    is_10b5_1: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ExplicitFiling {
    ticker: String,
    cik: u64,
    accession: String,
    primary_document: Option<String>,
    filing_date: Option<String>,
}

fn demo_signal(input: DemoSignal<'_>) -> InsiderSignal {
    let value_usd = input.shares * input.price;
    let ownership_increase_pct = input.post_transaction_shares.and_then(|after| {
        let before = after - input.shares;
        if before > 0.0 {
            Some((input.shares / before) * 100.0)
        } else {
            None
        }
    });
    let mut signal = InsiderSignal {
        ticker: input.ticker.into(),
        company: input.company.into(),
        issuer_cik: input.issuer_cik.into(),
        owner: input.owner.into(),
        owner_cik: None,
        role: input.role.into(),
        accession: input.accession.into(),
        form: "4".into(),
        filing_date: input.transaction_date.into(),
        report_date: Some(input.transaction_date.into()),
        transaction_date: input.transaction_date.into(),
        security_title: "Common Stock".into(),
        transaction_code: "P".into(),
        acquired_disposed: "A".into(),
        is_10b5_1: input.is_10b5_1,
        shares: input.shares,
        price: input.price,
        value_usd,
        value_text: dollars_text(value_usd),
        post_transaction_shares: input.post_transaction_shares,
        ownership_increase_pct,
        archive_url: format!(
            "https://www.sec.gov/Archives/edgar/data/{}/{}/doc4.xml",
            input.issuer_cik.trim_start_matches('0'),
            input.accession.replace('-', "")
        ),
        score: 0,
        reasons: Vec::new(),
        reason_text: String::new(),
    };
    score_signal(&mut signal);
    signal
}

fn fetch_explicit_filings(
    client: &SecClient,
    filings: &[ExplicitFiling],
) -> Result<InsiderWatchlist> {
    let request = WatchlistRequest {
        tickers: filings.iter().map(|filing| filing.ticker.clone()).collect(),
        days: 0,
        limit_filings: filings.len(),
        min_value_usd: 0.0,
        sec_delay_ms: client.delay_ms,
        sec_max_requests: client.max_requests,
    };
    let mut companies_by_ticker: BTreeMap<String, CompanyEvidence> = BTreeMap::new();
    let mut signals = Vec::new();
    let mut disqualifiers = Vec::new();

    for explicit in filings {
        let recent = RecentFiling {
            accession: explicit.accession.clone(),
            form: "4".into(),
            filing_date: explicit
                .filing_date
                .clone()
                .unwrap_or_else(|| "unknown".into()),
            report_date: None,
            primary_document: explicit
                .primary_document
                .clone()
                .unwrap_or_else(|| "doc4.xml".into()),
        };
        let xml_name = match &explicit.primary_document {
            Some(name) => name.clone(),
            None => fetch_ownership_xml_name(client, explicit.cik, &recent)?,
        };
        let url = recent.archive_file_url(explicit.cik, &xml_name);
        let xml = client
            .get(&url)?
            .text()
            .with_context(|| format!("reading {url}"))?;
        let document = parse_ownership_document(&xml)
            .with_context(|| format!("parsing ownership XML for {}", explicit.accession))?;
        let company = company_from_explicit(explicit, &document);
        companies_by_ticker
            .entry(explicit.ticker.clone())
            .or_insert_with(|| CompanyEvidence {
                ticker: explicit.ticker.clone(),
                cik: cik10(explicit.cik),
                company: company.title.clone(),
                tier: Tier::Noise,
                tier_rationale: Vec::new(),
            });
        let (sig, disq) =
            signals_from_document(&explicit.ticker, &company, &recent, &url, &document, 0.0);
        signals.extend(sig);
        disqualifiers.extend(disq);
    }

    apply_cluster_scores(&mut signals);
    rank_signals(&mut signals);
    let mut companies: Vec<CompanyEvidence> = companies_by_ticker.into_values().collect();
    assign_tiers(&mut companies, &signals);

    Ok(InsiderWatchlist {
        source: "SEC EDGAR explicit Form 4 ownership XML".into(),
        request,
        companies,
        signals,
        disqualifiers,
    })
}

fn company_from_explicit(explicit: &ExplicitFiling, document: &OwnershipDocument) -> TickerEntry {
    TickerEntry {
        cik_str: explicit.cik,
        ticker: explicit.ticker.clone(),
        title: document
            .issuer
            .as_ref()
            .and_then(|issuer| issuer.issuer_name.clone())
            .unwrap_or_else(|| explicit.ticker.clone()),
    }
}

fn parse_filing_spec(spec: &str) -> Result<ExplicitFiling> {
    let parts: Vec<_> = spec.split(':').collect();
    if !(3..=5).contains(&parts.len()) {
        bail!(
            "filing spec must be TICKER:CIK:ACCESSION[:PRIMARY_DOCUMENT[:FILING_DATE]], got {spec:?}"
        );
    }
    let ticker = parts[0].trim().to_ascii_uppercase();
    if ticker.is_empty() {
        bail!("filing spec has empty ticker: {spec:?}");
    }
    let cik = parts[1]
        .trim()
        .trim_start_matches('0')
        .parse::<u64>()
        .with_context(|| format!("invalid CIK in filing spec {spec:?}"))?;
    let accession = parts[2].trim().to_string();
    validate_accession(&accession)?;
    let primary_document = parts
        .get(3)
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .map(str::to_string);
    let filing_date = parts
        .get(4)
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .map(str::to_string);
    Ok(ExplicitFiling {
        ticker,
        cik,
        accession,
        primary_document,
        filing_date,
    })
}

fn validate_accession(accession: &str) -> Result<()> {
    let parts: Vec<_> = accession.split('-').collect();
    if parts.len() == 3
        && parts[0].len() == 10
        && parts[1].len() == 2
        && parts[2].len() == 6
        && parts
            .iter()
            .all(|part| part.chars().all(|c| c.is_ascii_digit()))
    {
        Ok(())
    } else {
        bail!("invalid accession {accession:?}; expected ##########-##-######")
    }
}

fn fetch_watchlist(client: &SecClient, request: &WatchlistRequest) -> Result<InsiderWatchlist> {
    let ticker_map = fetch_ticker_map(client)?;
    let cutoff = (Utc::now().date_naive() - ChronoDuration::days(request.days.max(0)))
        .format("%Y-%m-%d")
        .to_string();
    let mut companies = Vec::new();
    let mut signals = Vec::new();
    let mut disqualifiers = Vec::new();

    for ticker in &request.tickers {
        // A screen shouldn't die because one symbol is delisted, merged away, or
        // mistyped — warn and keep going so the rest of the run still produces a
        // watchlist.
        let Some(company) = ticker_map
            .values()
            .find(|entry| entry.ticker.eq_ignore_ascii_case(ticker))
            .cloned()
        else {
            eprintln!(
                "warning: ticker {ticker} not in SEC company_tickers.json \
                 (delisted/merged/typo?); skipping — use \
                 --filing {ticker}:CIK:ACCESSION to fetch a known filing directly"
            );
            continue;
        };
        let submissions = fetch_submissions(client, company.cik())?;
        companies.push(CompanyEvidence {
            ticker: ticker.clone(),
            cik: cik10(company.cik()),
            company: company.title.clone(),
            tier: Tier::Noise,
            tier_rationale: Vec::new(),
        });

        for filing in recent_form4_filings(&submissions, &cutoff)
            .into_iter()
            .take(request.limit_filings)
        {
            let (url, xml) = fetch_ownership_xml(client, company.cik(), &filing)?;
            let document = parse_ownership_document(&xml)
                .with_context(|| format!("parsing ownership XML for {}", filing.accession))?;
            let (sig, disq) = signals_from_document(
                ticker,
                &company,
                &filing,
                &url,
                &document,
                request.min_value_usd,
            );
            signals.extend(sig);
            disqualifiers.extend(disq);
        }
    }

    apply_cluster_scores(&mut signals);
    rank_signals(&mut signals);
    assign_tiers(&mut companies, &signals);

    Ok(InsiderWatchlist {
        source: "SEC EDGAR submissions + Form 4 ownership XML".into(),
        request: request.clone(),
        companies,
        signals,
        disqualifiers,
    })
}

/// Fetch the raw Form 4 ownership XML for a filing. To halve request volume
/// (IW-5) this derives the raw XML name from `primaryDocument` (dropping any
/// `xslF345X05/`-style styling prefix) and fetches it directly, falling back to
/// the filing-directory `index.json` only on a 404.
fn fetch_ownership_xml(
    client: &SecClient,
    cik: u64,
    filing: &RecentFiling,
) -> Result<(String, String)> {
    let basename = Path::new(&filing.primary_document)
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or(filing.primary_document.as_str());
    if basename.to_ascii_lowercase().ends_with(".xml") {
        let url = filing.archive_file_url(cik, basename);
        if let Some(resp) = client.get_allow_404(&url)? {
            let xml = resp.text().with_context(|| format!("reading {url}"))?;
            return Ok((url, xml));
        }
    }
    let xml_name = fetch_ownership_xml_name(client, cik, filing)?;
    let url = filing.archive_file_url(cik, &xml_name);
    let xml = client
        .get(&url)?
        .text()
        .with_context(|| format!("reading {url}"))?;
    Ok((url, xml))
}

fn fetch_ticker_map(client: &SecClient) -> Result<HashMap<String, TickerEntry>> {
    client
        .get(COMPANY_TICKERS_URL)?
        .json()
        .context("decoding SEC ticker map")
}

fn fetch_submissions(client: &SecClient, cik: u64) -> Result<Submissions> {
    let url = format!("{SUBMISSIONS_BASE_URL}/CIK{}.json", cik10(cik));
    client.get(&url)?.json().context("decoding SEC submissions")
}

fn fetch_ownership_xml_name(client: &SecClient, cik: u64, filing: &RecentFiling) -> Result<String> {
    let url = format!("{}/index.json", filing.archive_dir_url(cik));
    let index: ArchiveIndex = client
        .get(&url)?
        .json()
        .with_context(|| format!("decoding SEC filing index {url}"))?;

    let primary_basename = Path::new(&filing.primary_document)
        .file_name()
        .and_then(|name| name.to_str())
        .map(str::to_string);
    let mut xml_names: Vec<String> = index
        .directory
        .item
        .into_iter()
        .map(|item| item.name)
        .filter(|name| {
            let lower = name.to_ascii_lowercase();
            lower.ends_with(".xml") && lower != "filingsummary.xml"
        })
        .collect();
    xml_names.sort_by_key(|name| {
        if primary_basename.as_deref() == Some(name.as_str()) {
            0
        } else {
            1
        }
    });
    xml_names
        .into_iter()
        .next()
        .ok_or_else(|| anyhow!("no ownership XML found in SEC filing index {url}"))
}

struct SecClient {
    client: Client,
    delay_ms: u64,
    max_requests: usize,
    requests: std::cell::Cell<usize>,
    last_request: std::cell::RefCell<Option<Instant>>,
}

impl SecClient {
    /// Build a metered client. The spacing is clamped up to [`SEC_FLOOR_MS`] so
    /// no flag can drive the request rate above the floor (IW-5).
    fn new(client: Client, delay_ms: u64, max_requests: usize) -> SecClient {
        let delay_ms = delay_ms.max(SEC_FLOOR_MS);
        eprintln!(
            "SEC client: <= {:.1} req/s ({delay_ms} ms spacing), budget {max_requests} requests",
            1000.0 / delay_ms as f64,
        );
        SecClient {
            client,
            delay_ms,
            max_requests,
            requests: std::cell::Cell::new(0),
            last_request: std::cell::RefCell::new(None),
        }
    }

    /// Spend one request from the budget, honor the spacing floor, send, and
    /// hard-stop on the statuses that mean "back off" (429) or "you are blocked"
    /// (403). No automatic retry — one 429 ends the run (IW-5). Other statuses
    /// are returned unchecked for the caller to interpret.
    fn send_raw(&self, url: &str) -> Result<reqwest::blocking::Response> {
        let used = self.requests.get();
        if used >= self.max_requests {
            bail!(
                "SEC request budget exhausted ({}/{}) before {url}; narrow the run or raise --sec-max-requests deliberately",
                used,
                self.max_requests
            );
        }
        self.wait_for_slot();
        self.requests.set(used + 1);
        let response = self
            .client
            .get(url)
            .send()
            .with_context(|| format!("fetching {url}"))?;
        if response.status() == StatusCode::TOO_MANY_REQUESTS {
            let retry_after = response
                .headers()
                .get(reqwest::header::RETRY_AFTER)
                .and_then(|value| value.to_str().ok())
                .unwrap_or("unspecified");
            bail!(
                "SEC returned 429 Too Many Requests for {url} (Retry-After: {retry_after}); \
                 stop now and wait at least 10 minutes before retrying — do not loop"
            );
        }
        if response.status() == StatusCode::FORBIDDEN {
            bail!(
                "SEC returned 403 Forbidden for {url}; this usually means a missing or blocked \
                 User-Agent — set SEC_USER_AGENT to a real 'Name email' contact"
            );
        }
        Ok(response)
    }

    fn get(&self, url: &str) -> Result<reqwest::blocking::Response> {
        self.send_raw(url)?
            .error_for_status()
            .with_context(|| format!("SEC returned an error for {url}"))
    }

    /// Like [`SecClient::get`], but a 404 yields `Ok(None)` so a caller can fall
    /// back to another URL without masking a 429/403/budget stop.
    fn get_allow_404(&self, url: &str) -> Result<Option<reqwest::blocking::Response>> {
        let response = self.send_raw(url)?;
        if response.status() == StatusCode::NOT_FOUND {
            return Ok(None);
        }
        Ok(Some(response.error_for_status().with_context(|| {
            format!("SEC returned an error for {url}")
        })?))
    }

    fn wait_for_slot(&self) {
        let delay = Duration::from_millis(self.delay_ms);
        if let Some(last) = *self.last_request.borrow() {
            let elapsed = last.elapsed();
            if elapsed < delay {
                thread::sleep(delay - elapsed);
            }
        }
        *self.last_request.borrow_mut() = Some(Instant::now());
    }
}

fn recent_form4_filings(submissions: &Submissions, cutoff: &str) -> Vec<RecentFiling> {
    let recent = &submissions.filings.recent;
    let n = recent
        .accession_numbers
        .len()
        .min(recent.forms.len())
        .min(recent.filing_dates.len())
        .min(recent.primary_documents.len());
    (0..n)
        .filter_map(|i| {
            let form = recent.forms[i].as_str();
            let filing_date = recent.filing_dates[i].clone();
            if !matches!(form, "4" | "4/A") || filing_date.as_str() < cutoff {
                return None;
            }
            Some(RecentFiling {
                accession: recent.accession_numbers[i].clone(),
                form: recent.forms[i].clone(),
                filing_date,
                report_date: recent.report_dates.get(i).and_then(Clone::clone),
                primary_document: recent.primary_documents[i].clone(),
            })
        })
        .collect()
}

fn parse_ownership_document(xml: &str) -> Result<OwnershipDocument> {
    xml_from_str(xml).context("decoding Form 4 ownershipDocument XML")
}

fn signals_from_document(
    ticker: &str,
    company: &TickerEntry,
    filing: &RecentFiling,
    archive_url: &str,
    document: &OwnershipDocument,
    min_value_usd: f64,
) -> (Vec<InsiderSignal>, Vec<Disqualifier>) {
    let issuer_name = document
        .issuer
        .as_ref()
        .and_then(|issuer| issuer.issuer_name.clone())
        .unwrap_or_else(|| company.title.clone());
    let issuer_cik = document
        .issuer
        .as_ref()
        .and_then(|issuer| issuer.issuer_cik.clone())
        .unwrap_or_else(|| cik10(company.cik()));
    let owner = document
        .reporting_owner
        .first()
        .cloned()
        .unwrap_or_default();
    let owner_name = owner
        .reporting_owner_id
        .as_ref()
        .and_then(|id| id.rpt_owner_name.clone())
        .unwrap_or_else(|| "unknown owner".into());
    let owner_cik = owner
        .reporting_owner_id
        .as_ref()
        .and_then(|id| id.rpt_owner_cik.clone());
    let role = role_text(owner.reporting_owner_relationship.as_ref());
    let is_10b5_1 = document_is_10b5_1(document);

    let mut signals = Vec::new();
    let mut disqualifiers = Vec::new();

    for tx in document
        .non_derivative_table
        .as_ref()
        .map(NonDerivativeTable::non_derivative_transactions)
        .unwrap_or(&[])
    {
        let Some(code) = tx
            .transaction_coding
            .as_ref()
            .and_then(|coding| coding.transaction_code.clone())
        else {
            continue;
        };
        let acquired_disposed = tx
            .transaction_amounts
            .as_ref()
            .and_then(|amounts| amounts.transaction_acquired_disposed_code.as_ref())
            .and_then(|v| v.value.clone())
            .unwrap_or_default();
        let shares = tx
            .transaction_amounts
            .as_ref()
            .and_then(|amounts| amounts.transaction_shares.as_ref())
            .and_then(|v| v.value.as_deref())
            .and_then(parse_number)
            .unwrap_or(0.0);
        let price = tx
            .transaction_amounts
            .as_ref()
            .and_then(|amounts| amounts.transaction_price_per_share.as_ref())
            .and_then(|v| v.value.as_deref())
            .and_then(parse_number)
            .unwrap_or(0.0);
        let value_usd = shares * price;
        let transaction_date = tx
            .transaction_date
            .as_ref()
            .and_then(|v| v.value.clone())
            .or_else(|| filing.report_date.clone())
            .unwrap_or_else(|| filing.filing_date.clone());

        // Anything that is not a positive-share open-market P/A purchase is
        // retained as a disqualifier rather than dropped (IW-4).
        if code != "P" || acquired_disposed != "A" || shares <= 0.0 {
            disqualifiers.push(Disqualifier {
                ticker: ticker.to_string(),
                owner: owner_name.clone(),
                role: role.clone(),
                accession: filing.accession.clone(),
                transaction_date,
                transaction_code: code.clone(),
                acquired_disposed: acquired_disposed.clone(),
                shares,
                value_text: dollars_text(value_usd),
                is_10b5_1,
                reason: disqualifier_reason(&code, &acquired_disposed),
            });
            continue;
        }
        if value_usd < min_value_usd {
            continue;
        }

        let post_transaction_shares = tx
            .post_transaction_amounts
            .as_ref()
            .and_then(|amounts| amounts.shares_owned_following_transaction.as_ref())
            .and_then(|v| v.value.as_deref())
            .and_then(parse_number);
        let ownership_increase_pct = post_transaction_shares.and_then(|after| {
            let before = after - shares;
            if before > 0.0 {
                Some((shares / before) * 100.0)
            } else {
                None
            }
        });
        let security_title = tx
            .security_title
            .as_ref()
            .and_then(|v| v.value.clone())
            .unwrap_or_else(|| "common stock".into());

        let mut signal = InsiderSignal {
            ticker: ticker.to_string(),
            company: issuer_name.clone(),
            issuer_cik: issuer_cik.clone(),
            owner: owner_name.clone(),
            owner_cik: owner_cik.clone(),
            role: role.clone(),
            accession: filing.accession.clone(),
            form: filing.form.clone(),
            filing_date: filing.filing_date.clone(),
            report_date: filing.report_date.clone(),
            transaction_date,
            security_title,
            transaction_code: code,
            acquired_disposed,
            is_10b5_1,
            shares,
            price,
            value_usd,
            value_text: dollars_text(value_usd),
            post_transaction_shares,
            ownership_increase_pct,
            archive_url: archive_url.to_string(),
            score: 0,
            reasons: Vec::new(),
            reason_text: String::new(),
        };
        score_signal(&mut signal);
        signals.push(signal);
    }

    (signals, disqualifiers)
}

fn score_signal(signal: &mut InsiderSignal) {
    if signal.is_10b5_1 {
        signal
            .reasons
            .push("Rule 10b5-1 plan buy — excluded from conviction".into());
        signal.reason_text = signal.reasons.join("; ");
        return;
    }
    signal.score += 50;
    signal.reasons.push("open-market purchase code P".into());

    if signal.value_usd >= 1_000_000.0 {
        signal.score += 30;
        signal.reasons.push("purchase value >= $1M".into());
    } else if signal.value_usd >= 500_000.0 {
        signal.score += 20;
        signal.reasons.push("purchase value >= $500K".into());
    } else if signal.value_usd >= 100_000.0 {
        signal.score += 10;
        signal.reasons.push("purchase value >= $100K".into());
    }

    let role = signal.role.to_ascii_lowercase();
    if role.contains("ceo")
        || role.contains("chief")
        || role.contains("president")
        || role.contains("founder")
    {
        signal.score += 20;
        signal.reasons.push("senior officer role".into());
    } else if role.contains("director") || role.contains("officer") || role.contains("10% owner") {
        signal.score += 12;
        signal
            .reasons
            .push("director/officer/10% owner role".into());
    }

    if signal.ownership_increase_pct.unwrap_or(0.0) >= 5.0 {
        signal.score += 10;
        signal.reasons.push("ownership increase >= 5%".into());
    }

    signal.reason_text = signal.reasons.join("; ");
}

fn apply_cluster_scores(signals: &mut [InsiderSignal]) {
    // Cluster = >= 2 distinct *discretionary* owners (10b5-1 plan buys excluded,
    // IW-4).
    let mut owners_by_ticker: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    for signal in signals.iter().filter(|s| !s.is_10b5_1) {
        owners_by_ticker
            .entry(signal.ticker.clone())
            .or_default()
            .insert(signal.owner.clone());
    }
    for signal in signals {
        if signal.is_10b5_1 {
            continue;
        }
        if owners_by_ticker
            .get(&signal.ticker)
            .map(|owners| owners.len() >= 2)
            .unwrap_or(false)
        {
            signal.score += 15;
            signal.reasons.push("cluster buying across insiders".into());
            signal.reason_text = signal.reasons.join("; ");
        }
    }
}

/// Deterministic per-issuer signal tier, assigned in Rust from the evidence
/// before any model runs (IW-3). Only discretionary open-market buys
/// (`!is_10b5_1`) count toward conviction (IW-4).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
enum Tier {
    Strong,
    Moderate,
    Weak,
    Noise,
}

impl Tier {
    /// Higher = stronger; used to flag model overstatement and to rank issuers.
    fn rank(self) -> u8 {
        match self {
            Tier::Strong => 3,
            Tier::Moderate => 2,
            Tier::Weak => 1,
            Tier::Noise => 0,
        }
    }

    fn label(self) -> &'static str {
        match self {
            Tier::Strong => "strong",
            Tier::Moderate => "moderate",
            Tier::Weak => "weak",
            Tier::Noise => "noise",
        }
    }
}

/// Classify one issuer's signals into a tier plus the rubric facts the tier
/// rests on. `signals` may include 10b5-1 buys; they are filtered here.
fn tier_for(ticker: &str, signals: &[InsiderSignal]) -> (Tier, Vec<String>) {
    let buys: Vec<&InsiderSignal> = signals
        .iter()
        .filter(|s| s.ticker == ticker && !s.is_10b5_1)
        .collect();
    if buys.is_empty() {
        let plan_buys = signals.iter().any(|s| s.ticker == ticker && s.is_10b5_1);
        let why = if plan_buys {
            "only Rule 10b5-1 plan buys; no discretionary open-market purchase"
        } else {
            "no discretionary open-market purchase in window"
        };
        return (Tier::Noise, vec![why.to_string()]);
    }

    let owners: BTreeSet<&str> = buys.iter().map(|s| s.owner.as_str()).collect();
    let cluster = owners.len();
    let total: f64 = buys.iter().map(|s| s.value_usd).sum();
    let largest = buys.iter().map(|s| s.value_usd).fold(0.0_f64, f64::max);

    let mut why = vec![format!(
        "{} discretionary buy(s) by {} distinct insider(s), total {}, largest {}",
        buys.len(),
        cluster,
        dollars_text(total),
        dollars_text(largest),
    )];

    let tier = if cluster >= 2 && (largest >= STRONG_SINGLE_USD || total >= STRONG_TOTAL_USD) {
        why.push("cluster of >=2 insiders with material size".into());
        Tier::Strong
    } else if largest >= MODERATE_SINGLE_USD || (cluster >= 2 && total >= MODERATE_SINGLE_USD) {
        why.push("a single material buy or a smaller cluster".into());
        Tier::Moderate
    } else if largest < WEAK_MAX_USD {
        why.push("only small purchases in the optics range".into());
        Tier::Weak
    } else {
        why.push("mid-size single-insider purchase".into());
        Tier::Moderate
    };
    (tier, why)
}

/// Assign a tier to every company and rank companies/signals strongest-first.
fn assign_tiers(companies: &mut [CompanyEvidence], signals: &[InsiderSignal]) {
    for company in companies.iter_mut() {
        let (tier, rationale) = tier_for(&company.ticker, signals);
        company.tier = tier;
        company.tier_rationale = rationale;
    }
    companies.sort_by(|a, b| {
        b.tier
            .rank()
            .cmp(&a.tier.rank())
            .then_with(|| a.ticker.cmp(&b.ticker))
    });
}

/// Order signals strongest-first: score, then dollar value, then recency.
fn rank_signals(signals: &mut [InsiderSignal]) {
    signals.sort_by(|a, b| {
        b.score
            .cmp(&a.score)
            .then_with(|| b.value_usd.total_cmp(&a.value_usd))
            .then_with(|| b.transaction_date.cmp(&a.transaction_date))
    });
}

/// Map a non-qualifying transaction code to a plain-language exclusion reason.
fn disqualifier_reason(code: &str, acquired_disposed: &str) -> String {
    match code {
        "A" => "award/grant under Rule 16b-3, not an open-market purchase".into(),
        "M" => "option/derivative exercise or conversion".into(),
        "F" => "shares withheld for taxes".into(),
        "G" => "gift".into(),
        "S" => "open-market or private sale".into(),
        "P" => format!("purchase code P but disposed ({acquired_disposed}), not an acquisition"),
        other => format!("transaction code {other}, not a qualifying open-market purchase"),
    }
}

fn role_text(rel: Option<&ReportingOwnerRelationship>) -> String {
    let Some(rel) = rel else {
        return "unknown role".into();
    };
    let mut parts = Vec::new();
    if is_true(rel.is_director.as_deref()) {
        parts.push("director".to_string());
    }
    if is_true(rel.is_officer.as_deref()) {
        if let Some(title) = &rel.officer_title {
            parts.push(title.clone());
        } else {
            parts.push("officer".into());
        }
    }
    if is_true(rel.is_ten_percent_owner.as_deref()) {
        parts.push("10% owner".to_string());
    }
    if is_true(rel.is_other.as_deref()) {
        parts.push("other".to_string());
    }
    if parts.is_empty() {
        "unknown role".into()
    } else {
        parts.join(", ")
    }
}

fn is_true(value: Option<&str>) -> bool {
    matches!(value, Some("1" | "true" | "TRUE" | "True"))
}

fn parse_number(value: &str) -> Option<f64> {
    value.replace(',', "").parse().ok()
}

fn dollars_text(n: f64) -> String {
    if n.abs() >= 1_000_000_000.0 {
        format!("${:.3} billion", n / 1_000_000_000.0)
    } else if n.abs() >= 1_000_000.0 {
        format!("${:.3} million", n / 1_000_000.0)
    } else if n.abs() >= 1_000.0 {
        format!("${:.1} thousand", n / 1_000.0)
    } else {
        format!("${n:.0}")
    }
}

fn cik10(cik: u64) -> String {
    format!("{cik:010}")
}

#[derive(Debug, Default, PartialEq, Eq)]
struct WatchlistCheck {
    warnings: Vec<String>,
}

impl WatchlistCheck {
    fn warn(&mut self, warning: impl Into<String>) {
        self.warnings.push(warning.into());
    }

    fn is_clean(&self) -> bool {
        self.warnings.is_empty()
    }
}

fn check_watchlist_note(answer: &str, watchlist: &InsiderWatchlist) -> WatchlistCheck {
    let mut check = WatchlistCheck::default();

    // Accessions that may legitimately appear: discretionary signals plus the
    // retained disqualifiers/10b5-1 rows (the model may cite the latter as
    // risks).
    let known_accessions: HashSet<&str> = watchlist
        .signals
        .iter()
        .map(|signal| signal.accession.as_str())
        .chain(watchlist.disqualifiers.iter().map(|d| d.accession.as_str()))
        .collect();
    for accession in cited_accessions(answer) {
        if !known_accessions.contains(accession.as_str()) {
            check.warn(format!("unknown accession cited: {accession}"));
        }
    }

    // Non-discretionary rows: 10b5-1 plan buys and disqualifiers. Describing
    // these with open-market/conviction language is a mischaracterization.
    let non_discretionary: HashSet<&str> = watchlist
        .signals
        .iter()
        .filter(|signal| signal.is_10b5_1)
        .map(|signal| signal.accession.as_str())
        .chain(watchlist.disqualifiers.iter().map(|d| d.accession.as_str()))
        .collect();

    let strong_words = [
        "strong",
        "high-conviction",
        "high conviction",
        "smart money",
        "slam dunk",
        "table-pounding",
    ];
    let discretionary_words = ["open-market", "open market", "discretionary", "conviction"];
    let price_words = [
        "52-week low",
        "52 week low",
        "all-time low",
        "record low",
        "drawdown",
        "bought the dip",
        "buying the dip",
        "near its low",
        "near lows",
        "near 52-week",
        "oversold",
        "the sell-off",
        "the selloff",
    ];

    for line in answer.lines().filter(|line| !line.trim().is_empty()) {
        let lower = line.to_ascii_lowercase();
        let trimmed = line.trim();

        // Existing: a quantity-bearing insider line must cite a known accession.
        let mentions_signal = watchlist
            .signals
            .iter()
            .any(|signal| line.contains(&signal.ticker) || line.contains(&signal.owner));
        if mentions_signal && mentions_quantity(line) {
            let has_accession = known_accessions.iter().any(|accn| line.contains(*accn));
            if !has_accession {
                check.warn(format!(
                    "quantity-bearing insider line lacks a known accession: {line}"
                ));
            }
        }

        // Overstatement (IW-3): a weaker-tier company in stronger-tier terms.
        if strong_words.iter().any(|word| lower.contains(word)) {
            for company in &watchlist.companies {
                if company.tier.rank() < Tier::Strong.rank() && line.contains(&company.ticker) {
                    check.warn(format!(
                        "overstatement: {} is tier {} but the note uses stronger-conviction \
                         language: {trimmed}",
                        company.ticker,
                        company.tier.label(),
                    ));
                }
            }
        }

        // Mischaracterization (IW-3/IW-4): a 10b5-1 / disqualifier row presented
        // as a discretionary open-market purchase.
        if discretionary_words.iter().any(|word| lower.contains(word)) {
            if let Some(accn) = non_discretionary.iter().find(|accn| line.contains(**accn)) {
                check.warn(format!(
                    "mischaracterization: accession {accn} is a Rule 10b5-1 or non-purchase row \
                     described as a discretionary open-market buy: {trimmed}"
                ));
            }
        }

        // No-price guard (IW-3): no price history is supplied as evidence.
        if price_words.iter().any(|word| lower.contains(word)) {
            check.warn(format!(
                "price/drawdown claim with no price evidence supplied: {trimmed}"
            ));
        }
    }
    check
}

fn cited_accessions(answer: &str) -> Vec<String> {
    answer
        .split(|c: char| c.is_whitespace() || matches!(c, '(' | ')' | ';' | ',' | '.'))
        .filter(|token| {
            let parts: Vec<_> = token.split('-').collect();
            parts.len() == 3
                && parts[0].len() == 10
                && parts[1].len() == 2
                && parts[2].len() == 6
                && parts.iter().all(|p| p.chars().all(|c| c.is_ascii_digit()))
        })
        .map(str::to_string)
        .collect()
}

fn mentions_quantity(line: &str) -> bool {
    line.contains('$') || line.to_ascii_lowercase().contains(" shares")
}

#[derive(Debug, Clone)]
struct RecentFiling {
    accession: String,
    form: String,
    filing_date: String,
    report_date: Option<String>,
    primary_document: String,
}

impl RecentFiling {
    fn archive_dir_url(&self, cik: u64) -> String {
        format!(
            "{ARCHIVES_BASE_URL}/{}/{}",
            cik,
            self.accession.replace('-', "")
        )
    }

    fn archive_file_url(&self, cik: u64, filename: &str) -> String {
        format!("{}/{}", self.archive_dir_url(cik), filename)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct WatchlistRequest {
    tickers: Vec<String>,
    days: i64,
    limit_filings: usize,
    min_value_usd: f64,
    sec_delay_ms: u64,
    sec_max_requests: usize,
}

#[derive(Debug, Serialize, Deserialize)]
struct InsiderWatchlist {
    source: String,
    request: WatchlistRequest,
    companies: Vec<CompanyEvidence>,
    signals: Vec<InsiderSignal>,
    #[serde(default)]
    disqualifiers: Vec<Disqualifier>,
}

#[derive(Debug, Serialize, Deserialize)]
struct CompanyEvidence {
    ticker: String,
    cik: String,
    company: String,
    #[serde(default = "default_tier")]
    tier: Tier,
    #[serde(default)]
    tier_rationale: Vec<String>,
}

fn default_tier() -> Tier {
    Tier::Noise
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct InsiderSignal {
    ticker: String,
    company: String,
    issuer_cik: String,
    owner: String,
    owner_cik: Option<String>,
    role: String,
    accession: String,
    form: String,
    filing_date: String,
    report_date: Option<String>,
    transaction_date: String,
    security_title: String,
    transaction_code: String,
    acquired_disposed: String,
    is_10b5_1: bool,
    shares: f64,
    price: f64,
    value_usd: f64,
    value_text: String,
    post_transaction_shares: Option<f64>,
    ownership_increase_pct: Option<f64>,
    archive_url: String,
    score: i32,
    reasons: Vec<String>,
    reason_text: String,
}

/// A Form 4 transaction kept as evidence but excluded from conviction: awards,
/// exercises, sales, gifts, tax withholding (IW-4). Retained so the model can
/// cite "excluded because…" and so the validator can flag mischaracterization.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct Disqualifier {
    ticker: String,
    owner: String,
    role: String,
    accession: String,
    transaction_date: String,
    transaction_code: String,
    acquired_disposed: String,
    shares: f64,
    value_text: String,
    is_10b5_1: bool,
    reason: String,
}

#[derive(Debug, Clone, Deserialize)]
struct TickerEntry {
    cik_str: u64,
    ticker: String,
    title: String,
}

impl TickerEntry {
    fn cik(&self) -> u64 {
        self.cik_str
    }
}

#[derive(Debug, Deserialize)]
struct Submissions {
    filings: Filings,
}

#[derive(Debug, Deserialize)]
struct Filings {
    recent: RecentFilings,
}

#[derive(Debug, Deserialize)]
struct RecentFilings {
    #[serde(rename = "accessionNumber")]
    accession_numbers: Vec<String>,
    #[serde(rename = "filingDate")]
    filing_dates: Vec<String>,
    #[serde(rename = "reportDate")]
    report_dates: Vec<Option<String>>,
    #[serde(rename = "form")]
    forms: Vec<String>,
    #[serde(rename = "primaryDocument")]
    primary_documents: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct ArchiveIndex {
    directory: ArchiveDirectory,
}

#[derive(Debug, Deserialize)]
struct ArchiveDirectory {
    item: Vec<ArchiveItem>,
}

#[derive(Debug, Deserialize)]
struct ArchiveItem {
    name: String,
}

#[derive(Debug, Default, Deserialize)]
#[serde(rename_all = "camelCase")]
struct OwnershipDocument {
    issuer: Option<Issuer>,
    #[serde(default)]
    reporting_owner: Vec<ReportingOwner>,
    non_derivative_table: Option<NonDerivativeTable>,
    footnotes: Option<Footnotes>,
}

#[derive(Debug, Default, Deserialize)]
struct Footnotes {
    #[serde(default)]
    footnote: Vec<Footnote>,
}

#[derive(Debug, Default, Deserialize)]
struct Footnote {
    #[serde(rename = "$text", default)]
    text: String,
}

/// Heuristic Rule 10b5-1 detection: a Form 4 has no first-class discretionary
/// flag pre-2023, so a plan transaction is surfaced by a footnote mentioning
/// "10b5-1". Document-level (all rows in the filing share the flag) is the
/// practical signal; it errs toward caution by excluding flagged buys from
/// conviction (IW-4).
fn document_is_10b5_1(document: &OwnershipDocument) -> bool {
    document
        .footnotes
        .as_ref()
        .map(|notes| {
            notes
                .footnote
                .iter()
                .any(|note| note.text.to_ascii_lowercase().contains("10b5-1"))
        })
        .unwrap_or(false)
}

#[derive(Debug, Default, Deserialize)]
#[serde(rename_all = "camelCase")]
struct Issuer {
    issuer_cik: Option<String>,
    issuer_name: Option<String>,
}

#[derive(Debug, Default, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ReportingOwner {
    reporting_owner_id: Option<ReportingOwnerId>,
    reporting_owner_relationship: Option<ReportingOwnerRelationship>,
}

#[derive(Debug, Default, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ReportingOwnerId {
    rpt_owner_cik: Option<String>,
    rpt_owner_name: Option<String>,
}

#[derive(Debug, Default, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ReportingOwnerRelationship {
    is_director: Option<String>,
    is_officer: Option<String>,
    is_ten_percent_owner: Option<String>,
    is_other: Option<String>,
    officer_title: Option<String>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(rename_all = "camelCase")]
struct NonDerivativeTable {
    #[serde(default)]
    non_derivative_transaction: Vec<NonDerivativeTransaction>,
}

impl NonDerivativeTable {
    fn non_derivative_transactions(&self) -> &[NonDerivativeTransaction] {
        &self.non_derivative_transaction
    }
}

#[derive(Debug, Default, Deserialize)]
#[serde(rename_all = "camelCase")]
struct NonDerivativeTransaction {
    security_title: Option<ValueField>,
    transaction_date: Option<ValueField>,
    transaction_coding: Option<TransactionCoding>,
    transaction_amounts: Option<TransactionAmounts>,
    post_transaction_amounts: Option<PostTransactionAmounts>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(rename_all = "camelCase")]
struct TransactionCoding {
    transaction_code: Option<String>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(rename_all = "camelCase")]
struct TransactionAmounts {
    transaction_shares: Option<ValueField>,
    transaction_price_per_share: Option<ValueField>,
    transaction_acquired_disposed_code: Option<ValueField>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(rename_all = "camelCase")]
struct PostTransactionAmounts {
    shares_owned_following_transaction: Option<ValueField>,
}

#[derive(Debug, Default, Deserialize)]
struct ValueField {
    value: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(ticker: Vec<&str>, tickers: Vec<&str>) -> Args {
        Args {
            ticker: ticker.into_iter().map(str::to_string).collect(),
            tickers: tickers.into_iter().map(str::to_string).collect(),
            days: 365,
            limit_filings: 40,
            min_value_usd: 0.0,
            sec_delay_ms: 0,
            sec_max_requests: 25,
            no_model: true,
            demo: false,
            evidence: None,
            filing: Vec::new(),
            save_evidence: None,
            json: false,
            profile: None,
            model: None,
            repo: None,
            gguf: None,
            format: None,
            prefill_chunk: None,
            max_tokens: 700,
            temperature: 0.0,
            seed: 0,
            offline: true,
            cpu: true,
        }
    }

    #[test]
    fn plan_tickers_deduplicates_and_uppercases() {
        // upholds: IW-1 — ticker planning is pure normalization before effects.
        let planned = plan_tickers(&args(vec!["jpm", "META"], vec!["meta", " abnb "])).unwrap();
        assert_eq!(planned, vec!["ABNB", "JPM", "META"]);
    }

    #[test]
    fn archive_url_strips_accession_dashes_and_not_cik_padding() {
        let filing = RecentFiling {
            accession: "0001127602-24-015433".into(),
            form: "4".into(),
            filing_date: "2024-05-01".into(),
            report_date: Some("2024-04-30".into()),
            primary_document: "xslF345X05/form4.xml".into(),
        };
        assert_eq!(
            filing.archive_file_url(19617, &filing.primary_document),
            "https://www.sec.gov/Archives/edgar/data/19617/000112760224015433/xslF345X05/form4.xml"
        );
    }

    #[test]
    fn recent_form4_filings_filters_form_and_cutoff() {
        let submissions = Submissions {
            filings: Filings {
                recent: RecentFilings {
                    accession_numbers: vec!["a".into(), "b".into(), "c".into()],
                    filing_dates: vec![
                        "2024-01-02".into(),
                        "2023-01-02".into(),
                        "2024-01-03".into(),
                    ],
                    report_dates: vec![None, None, None],
                    forms: vec!["4".into(), "4".into(), "10-K".into()],
                    primary_documents: vec!["a.xml".into(), "b.xml".into(), "c.htm".into()],
                },
            },
        };
        let filings = recent_form4_filings(&submissions, "2024-01-01");
        assert_eq!(filings.len(), 1);
        assert_eq!(filings[0].accession, "a");
    }

    #[test]
    fn parse_explicit_filing_spec_accepts_optional_document_and_date() {
        let filing =
            parse_filing_spec("jpm:0000019617:0001225208-26-006142:doc4.xml:2026-06-22").unwrap();
        assert_eq!(
            filing,
            ExplicitFiling {
                ticker: "JPM".into(),
                cik: 19617,
                accession: "0001225208-26-006142".into(),
                primary_document: Some("doc4.xml".into()),
                filing_date: Some("2026-06-22".into()),
            }
        );
        assert!(parse_filing_spec("JPM:19617:not-an-accession").is_err());
    }

    #[test]
    fn sec_client_enforces_request_budget_before_network() {
        let client = SecClient::new(Client::new(), 0, 0);
        let err = client.get("http://127.0.0.1:9/nope").unwrap_err();
        assert!(err.to_string().contains("SEC request budget exhausted"));
    }

    #[test]
    fn parses_form4_purchase_into_signal() {
        // upholds: IW-2 — a signal comes from a P/A positive-share transaction.
        let document = parse_ownership_document(SAMPLE_FORM4).unwrap();
        let company = TickerEntry {
            cik_str: 12345,
            ticker: "TST".into(),
            title: "Test Corp".into(),
        };
        let filing = RecentFiling {
            accession: "0000000000-24-000001".into(),
            form: "4".into(),
            filing_date: "2024-05-03".into(),
            report_date: Some("2024-05-02".into()),
            primary_document: "form4.xml".into(),
        };
        let (signals, disqualifiers) = signals_from_document(
            "TST",
            &company,
            &filing,
            "https://example.test/form4.xml",
            &document,
            0.0,
        );
        assert_eq!(signals.len(), 1);
        let signal = &signals[0];
        assert_eq!(signal.owner, "Ada Founder");
        assert_eq!(signal.transaction_code, "P");
        assert_eq!(signal.acquired_disposed, "A");
        assert_eq!(signal.shares, 1_000.0);
        assert_eq!(signal.price, 12.5);
        assert_eq!(signal.value_usd, 12_500.0);
        assert!(signal.score >= 50);
        assert!(!signal.is_10b5_1);
        // upholds: IW-4 — the S sale is retained as a disqualifier, not dropped.
        assert_eq!(disqualifiers.len(), 1);
        assert_eq!(disqualifiers[0].transaction_code, "S");
    }

    #[test]
    fn validator_flags_unknown_accession() {
        // upholds: IW-3 — cited accessions are checked against supplied evidence.
        let watchlist = InsiderWatchlist {
            source: "test".into(),
            request: WatchlistRequest {
                tickers: vec!["TST".into()],
                days: 365,
                limit_filings: 1,
                min_value_usd: 0.0,
                sec_delay_ms: 0,
                sec_max_requests: 0,
            },
            companies: vec![],
            signals: vec![InsiderSignal {
                ticker: "TST".into(),
                company: "Test Corp".into(),
                issuer_cik: "0000012345".into(),
                owner: "Ada Founder".into(),
                owner_cik: None,
                role: "CEO".into(),
                accession: "0000000000-24-000001".into(),
                form: "4".into(),
                filing_date: "2024-05-03".into(),
                report_date: None,
                transaction_date: "2024-05-02".into(),
                security_title: "Common Stock".into(),
                transaction_code: "P".into(),
                acquired_disposed: "A".into(),
                is_10b5_1: false,
                shares: 100.0,
                price: 10.0,
                value_usd: 1_000.0,
                value_text: "$1000".into(),
                post_transaction_shares: None,
                ownership_increase_pct: None,
                archive_url: "https://example.test".into(),
                score: 50,
                reasons: vec![],
                reason_text: String::new(),
            }],
            disqualifiers: vec![],
        };
        let check = check_watchlist_note(
            "TST Ada Founder bought 100 shares for $1000, accession 9999999999-24-000001.",
            &watchlist,
        );
        assert_eq!(check.warnings.len(), 2);
        assert_eq!(
            check.warnings[0],
            "unknown accession cited: 9999999999-24-000001"
        );
        assert!(check.warnings[1].contains("lacks a known accession"));
    }

    #[test]
    fn demo_watchlist_has_rankable_signals() {
        let watchlist = demo_watchlist();
        assert!(watchlist.signals.len() >= 3);
        assert!(watchlist
            .signals
            .iter()
            .all(|signal| signal.transaction_code == "P"
                && signal.acquired_disposed == "A"
                && signal.shares > 0.0
                && signal.accession.len() == 20));
        assert!(watchlist.signals.iter().any(|signal| signal
            .reasons
            .iter()
            .any(|r| r == "cluster buying across insiders")));

        // upholds: IW-3/IW-4 — tiers assigned in Rust; DEMOC stays weak because
        // its only sizeable buy is a Rule 10b5-1 plan transaction (excluded).
        let tier = |t: &str| {
            watchlist
                .companies
                .iter()
                .find(|c| c.ticker == t)
                .unwrap()
                .tier
        };
        assert_eq!(tier("DEMOA"), Tier::Strong);
        assert_eq!(tier("DEMOB"), Tier::Moderate);
        assert_eq!(tier("DEMOC"), Tier::Weak);
        // companies are ranked strongest-first.
        assert_eq!(watchlist.companies[0].ticker, "DEMOA");
        // the 10b5-1 buy is present as evidence but contributes no score.
        assert!(watchlist
            .signals
            .iter()
            .any(|s| s.is_10b5_1 && s.score == 0));
    }

    fn sample_signal(ticker: &str, owner: &str, value_usd: f64, is_10b5_1: bool) -> InsiderSignal {
        let mut signal = InsiderSignal {
            ticker: ticker.into(),
            company: "Co".into(),
            issuer_cik: "0000000000".into(),
            owner: owner.into(),
            owner_cik: None,
            role: "director".into(),
            accession: "0000000000-26-000001".into(),
            form: "4".into(),
            filing_date: "2026-01-01".into(),
            report_date: None,
            transaction_date: "2026-01-01".into(),
            security_title: "Common Stock".into(),
            transaction_code: "P".into(),
            acquired_disposed: "A".into(),
            is_10b5_1,
            shares: 1.0,
            price: value_usd,
            value_usd,
            value_text: dollars_text(value_usd),
            post_transaction_shares: None,
            ownership_increase_pct: None,
            archive_url: String::new(),
            score: 0,
            reasons: vec![],
            reason_text: String::new(),
        };
        score_signal(&mut signal);
        signal
    }

    #[test]
    fn tier_for_classifies_rubric_cases() {
        // upholds: IW-3 — discrete tier assigned in Rust from the rubric.
        let strong = vec![
            sample_signal("X", "Ada", 600_000.0, false),
            sample_signal("X", "Ben", 450_000.0, false),
        ];
        assert_eq!(tier_for("X", &strong).0, Tier::Strong);

        let moderate = vec![sample_signal("X", "Ada", 300_000.0, false)];
        assert_eq!(tier_for("X", &moderate).0, Tier::Moderate);

        let weak = vec![sample_signal("X", "Ada", 30_000.0, false)];
        assert_eq!(tier_for("X", &weak).0, Tier::Weak);

        let noise = vec![sample_signal("X", "Ada", 600_000.0, true)];
        assert_eq!(tier_for("X", &noise).0, Tier::Noise);
    }

    #[test]
    fn ten_b5_1_buys_are_excluded_from_conviction() {
        // upholds: IW-4 — a 10b5-1 plan buy is evidence but never counts toward
        // cluster or tier; only the small discretionary buy remains.
        let mut signals = vec![
            sample_signal("X", "Ada", 90_000.0, false),
            sample_signal("X", "Ben", 300_000.0, true),
        ];
        apply_cluster_scores(&mut signals);
        assert!(!signals.iter().any(|s| s
            .reasons
            .iter()
            .any(|r| r == "cluster buying across insiders")));
        assert_eq!(tier_for("X", &signals).0, Tier::Weak);
    }

    #[test]
    fn detects_10b5_1_from_footnote() {
        // upholds: IW-4 — the footnote heuristic flags a plan buy.
        let document = parse_ownership_document(SAMPLE_FORM4_10B51).unwrap();
        let company = TickerEntry {
            cik_str: 12345,
            ticker: "TST".into(),
            title: "Test Corp".into(),
        };
        let filing = RecentFiling {
            accession: "0000000000-24-000002".into(),
            form: "4".into(),
            filing_date: "2024-05-03".into(),
            report_date: Some("2024-05-02".into()),
            primary_document: "form4.xml".into(),
        };
        let (signals, _) = signals_from_document(
            "TST",
            &company,
            &filing,
            "https://example.test",
            &document,
            0.0,
        );
        assert_eq!(signals.len(), 1);
        assert!(signals[0].is_10b5_1);
        assert_eq!(signals[0].score, 0);
    }

    #[test]
    fn validator_flags_overstatement_mischaracterization_and_price() {
        // upholds: IW-3 — the model may not exceed the Rust tier, mislabel a
        // non-discretionary row, or assert price context with no price evidence.
        let mut plan_buy = sample_signal("WEAK", "Plan Owner", 300_000.0, true);
        plan_buy.accession = "0000000000-26-000010".into();
        let watchlist = InsiderWatchlist {
            source: "test".into(),
            request: WatchlistRequest {
                tickers: vec!["WEAK".into()],
                days: 365,
                limit_filings: 1,
                min_value_usd: 0.0,
                sec_delay_ms: 0,
                sec_max_requests: 0,
            },
            companies: vec![CompanyEvidence {
                ticker: "WEAK".into(),
                cik: "0000000000".into(),
                company: "Weak Signal Co".into(),
                tier: Tier::Weak,
                tier_rationale: vec![],
            }],
            signals: vec![plan_buy],
            disqualifiers: vec![],
        };
        let answer = "- WEAK Plan Owner made a strong high-conviction open-market purchase \
                      (accession 0000000000-26-000010) near the 52-week low.";
        let check = check_watchlist_note(answer, &watchlist);
        assert!(check.warnings.iter().any(|w| w.contains("overstatement")));
        assert!(check
            .warnings
            .iter()
            .any(|w| w.contains("mischaracterization")));
        assert!(check
            .warnings
            .iter()
            .any(|w| w.contains("no price evidence")));
    }

    #[test]
    fn sec_client_clamps_delay_to_floor() {
        // upholds: IW-5 — no flag can drive the rate above the spacing floor.
        let client = SecClient::new(Client::new(), 0, 5);
        assert!(client.delay_ms >= SEC_FLOOR_MS);
    }

    #[test]
    fn default_profile_is_qwen_for_screens_and_mistral_for_demo() {
        assert_eq!(default_profile_name(false), "qwen32b");
        assert_eq!(default_profile_name(true), "mistral");
        // both must be real built-ins.
        assert!(ModelProfile::builtin(default_profile_name(false)).is_some());
        assert!(ModelProfile::builtin(default_profile_name(true)).is_some());
    }

    #[test]
    fn prompt_view_is_leaner_but_keeps_citations() {
        let watchlist = demo_watchlist();
        let full = serde_json::to_string(&watchlist).unwrap();
        let lean = serde_json::to_string(&prompt_view(&watchlist)).unwrap();
        assert!(
            lean.len() * 2 < full.len(),
            "lean {} full {}",
            lean.len(),
            full.len()
        );
        // every accession the model is asked to cite survives the projection.
        for signal in &watchlist.signals {
            assert!(lean.contains(&signal.accession));
        }
        // 10b5-1 flag is preserved so the model can be held to it.
        assert!(lean.contains("\"is_10b5_1\":true"));
    }

    const SAMPLE_FORM4: &str = r#"
<ownershipDocument>
  <issuer>
    <issuerCik>0000012345</issuerCik>
    <issuerName>Test Corp</issuerName>
    <issuerTradingSymbol>TST</issuerTradingSymbol>
  </issuer>
  <reportingOwner>
    <reportingOwnerId>
      <rptOwnerCik>0000099999</rptOwnerCik>
      <rptOwnerName>Ada Founder</rptOwnerName>
    </reportingOwnerId>
    <reportingOwnerRelationship>
      <isDirector>1</isDirector>
      <isOfficer>1</isOfficer>
      <isTenPercentOwner>0</isTenPercentOwner>
      <officerTitle>Chief Executive Officer</officerTitle>
    </reportingOwnerRelationship>
  </reportingOwner>
  <nonDerivativeTable>
    <nonDerivativeTransaction>
      <securityTitle><value>Common Stock</value></securityTitle>
      <transactionDate><value>2024-05-02</value></transactionDate>
      <transactionCoding><transactionCode>P</transactionCode></transactionCoding>
      <transactionAmounts>
        <transactionShares><value>1000</value></transactionShares>
        <transactionPricePerShare><value>12.50</value></transactionPricePerShare>
        <transactionAcquiredDisposedCode><value>A</value></transactionAcquiredDisposedCode>
      </transactionAmounts>
      <postTransactionAmounts>
        <sharesOwnedFollowingTransaction><value>11000</value></sharesOwnedFollowingTransaction>
      </postTransactionAmounts>
    </nonDerivativeTransaction>
    <nonDerivativeTransaction>
      <securityTitle><value>Common Stock</value></securityTitle>
      <transactionDate><value>2024-05-02</value></transactionDate>
      <transactionCoding><transactionCode>S</transactionCode></transactionCoding>
      <transactionAmounts>
        <transactionShares><value>50</value></transactionShares>
        <transactionPricePerShare><value>13.00</value></transactionPricePerShare>
        <transactionAcquiredDisposedCode><value>D</value></transactionAcquiredDisposedCode>
      </transactionAmounts>
    </nonDerivativeTransaction>
  </nonDerivativeTable>
</ownershipDocument>
"#;

    const SAMPLE_FORM4_10B51: &str = r#"
<ownershipDocument>
  <issuer>
    <issuerCik>0000012345</issuerCik>
    <issuerName>Test Corp</issuerName>
    <issuerTradingSymbol>TST</issuerTradingSymbol>
  </issuer>
  <reportingOwner>
    <reportingOwnerId>
      <rptOwnerName>Plan Owner</rptOwnerName>
    </reportingOwnerId>
    <reportingOwnerRelationship>
      <isDirector>1</isDirector>
    </reportingOwnerRelationship>
  </reportingOwner>
  <nonDerivativeTable>
    <nonDerivativeTransaction>
      <securityTitle><value>Common Stock</value></securityTitle>
      <transactionDate><value>2024-05-02</value></transactionDate>
      <transactionCoding><transactionCode>P</transactionCode></transactionCoding>
      <transactionAmounts>
        <transactionShares><value>2000</value></transactionShares>
        <transactionPricePerShare><value>20.00</value></transactionPricePerShare>
        <transactionAcquiredDisposedCode><value>A</value></transactionAcquiredDisposedCode>
      </transactionAmounts>
    </nonDerivativeTransaction>
  </nonDerivativeTable>
  <footnotes>
    <footnote id="F1">This purchase was made pursuant to a Rule 10b5-1 trading plan adopted on 2024-01-15.</footnote>
  </footnotes>
</ownershipDocument>
"#;
}
