//! Build an auditable insider-buy watchlist from real SEC Form 4 filings.
//!
//! Rust resolves tickers, fetches recent ownership filings from EDGAR, parses
//! Form 4 XML into typed transaction evidence, scores open-market insider buys,
//! and optionally asks a local chat model to rank the resulting watchlist.
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
//! Example-level invariants:
//! - **IW-1** ticker planning is pure and normalizes `--ticker`/`--tickers` to
//!   a deduplicated uppercase list before any network or model work.
//! - **IW-2** every watchlist signal is derived from a parsed Form 4
//!   non-derivative transaction with code `P`, acquired/disposed code `A`, and
//!   positive shares.
//! - **IW-3** model output is advisory over supplied evidence only; cited SEC
//!   accessions are checked against the watchlist evidence.

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

const SYSTEM: &str = "\
You are ranking insider-buy signals for a stock watchlist. Use only the \
supplied SEC Form 4 evidence. Every factual claim about an insider purchase \
must cite ticker, owner, accession, transaction_date, shares, price, and \
value_text from the supplied JSON. Prefer open-market P purchases with large \
dollar value, cluster buying, and senior insider roles. Distinguish strong \
signals from weak or optical purchases. Separate ranked watchlist, evidence, \
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
    /// Delay between SEC filing-directory/document requests.
    #[arg(long, default_value_t = 1000)]
    sec_delay_ms: u64,
    /// Hard cap on SEC HTTP requests for this run.
    #[arg(long, default_value_t = 25)]
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
    /// Built-in profile name (one of `ModelProfile::BUILTIN_NAMES`).
    #[arg(long, default_value = "mistral")]
    profile: String,
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
            let user_agent = std::env::var("SEC_USER_AGENT").context(
                "set SEC_USER_AGENT to a descriptive value with contact info, \
                 e.g. 'Your Name your.email@example.com'",
            )?;
            let client = SecClient::new(
                Client::builder()
                    .user_agent(user_agent)
                    .timeout(Duration::from_secs(30))
                    .build()?,
                args.sec_delay_ms,
                args.sec_max_requests,
            );
            let filings = args
                .filing
                .iter()
                .map(|spec| parse_filing_spec(spec))
                .collect::<Result<Vec<_>>>()?;
            run_blocking(|| fetch_explicit_filings(&client, &filings))
        }
        (false, None, true) => {
            let tickers = plan_tickers(args)?;
            let user_agent = std::env::var("SEC_USER_AGENT").context(
                "set SEC_USER_AGENT to a descriptive value with contact info, \
                 e.g. 'Your Name your.email@example.com'",
            )?;
            let client = SecClient::new(
                Client::builder()
                    .user_agent(user_agent)
                    .timeout(Duration::from_secs(30))
                    .build()?,
                args.sec_delay_ms,
                args.sec_max_requests,
            );

            let request = WatchlistRequest {
                tickers,
                days: args.days,
                limit_filings: args.limit_filings,
                min_value_usd: args.min_value_usd,
                sec_delay_ms: args.sec_delay_ms,
                sec_max_requests: args.sec_max_requests,
            };
            run_blocking(|| fetch_watchlist(&client, &request))
        }
        _ => unreachable!("source count checked above"),
    }
}

fn model_profile(args: &Args) -> Result<ModelProfile> {
    let mut profile = ModelProfile::builtin(&args.profile).ok_or_else(|| {
        anyhow!(
            "unknown profile {:?}; built-ins: {:?}",
            args.profile,
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
    for signal in &watchlist.signals {
        println!(
            "{} {} {} {} {} {} score={} accession={}",
            signal.ticker,
            signal.transaction_date,
            signal.owner,
            signal.role,
            signal.value_text,
            signal.reason_text,
            signal.score,
            signal.accession
        );
    }
}

fn build_prompt(watchlist: &InsiderWatchlist) -> Result<String> {
    let evidence_json = serde_json::to_string_pretty(watchlist)?;
    Ok(format!(
        "\
Rank these insider-buy signals as a stock-selection watchlist.

Required format:
- Ranked watchlist: 3-8 bullets. Each bullet must cite ticker, owner, accession,
  transaction_date, shares, price, and value_text exactly as supplied.
- Signal strength: explain which mechanical reasons matter most.
- False-positive risks: explain which signals could be optics, compensation
  related, stale, or too small to matter.
- Follow-up research: concrete public-data checks to run next.

SEC Form 4 evidence JSON:
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
    let companies = vec![
        CompanyEvidence {
            ticker: "DEMOA".into(),
            cik: "0001000001".into(),
            company: "Demo Applied Systems Inc.".into(),
        },
        CompanyEvidence {
            ticker: "DEMOB".into(),
            cik: "0001000002".into(),
            company: "Demo Regional Bancorp".into(),
        },
        CompanyEvidence {
            ticker: "DEMOC".into(),
            cik: "0001000003".into(),
            company: "Demo Consumer Platform Corp.".into(),
        },
    ];
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
        }),
    ];
    apply_cluster_scores(&mut signals);
    signals.sort_by(|a, b| {
        b.score
            .cmp(&a.score)
            .then_with(|| b.value_usd.total_cmp(&a.value_usd))
            .then_with(|| b.transaction_date.cmp(&a.transaction_date))
    });
    InsiderWatchlist {
        source: "bundled deterministic demo evidence for the insider-watchlist workflow".into(),
        request,
        companies,
        signals,
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
            });
        signals.extend(signals_from_document(
            &explicit.ticker,
            &company,
            &recent,
            &url,
            &document,
            0.0,
        ));
    }

    apply_cluster_scores(&mut signals);
    signals.sort_by(|a, b| {
        b.score
            .cmp(&a.score)
            .then_with(|| b.value_usd.total_cmp(&a.value_usd))
            .then_with(|| b.transaction_date.cmp(&a.transaction_date))
    });

    Ok(InsiderWatchlist {
        source: "SEC EDGAR explicit Form 4 ownership XML".into(),
        request,
        companies: companies_by_ticker.into_values().collect(),
        signals,
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

    for ticker in &request.tickers {
        let company = ticker_map
            .values()
            .find(|entry| entry.ticker.eq_ignore_ascii_case(ticker))
            .cloned()
            .ok_or_else(|| anyhow!("ticker {ticker} not found in SEC company_tickers.json"))?;
        let submissions = fetch_submissions(client, company.cik())?;
        companies.push(CompanyEvidence {
            ticker: ticker.clone(),
            cik: cik10(company.cik()),
            company: company.title.clone(),
        });

        for filing in recent_form4_filings(&submissions, &cutoff)
            .into_iter()
            .take(request.limit_filings)
        {
            let xml_name = fetch_ownership_xml_name(client, company.cik(), &filing)?;
            let url = filing.archive_file_url(company.cik(), &xml_name);
            let xml = client
                .get(&url)?
                .text()
                .with_context(|| format!("reading {url}"))?;
            let document = parse_ownership_document(&xml)
                .with_context(|| format!("parsing ownership XML for {}", filing.accession))?;
            signals.extend(signals_from_document(
                ticker,
                &company,
                &filing,
                &url,
                &document,
                request.min_value_usd,
            ));
        }
    }

    apply_cluster_scores(&mut signals);
    signals.sort_by(|a, b| {
        b.score
            .cmp(&a.score)
            .then_with(|| b.value_usd.total_cmp(&a.value_usd))
            .then_with(|| b.transaction_date.cmp(&a.transaction_date))
    });

    Ok(InsiderWatchlist {
        source: "SEC EDGAR submissions + Form 4 ownership XML".into(),
        request: request.clone(),
        companies,
        signals,
    })
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
    fn new(client: Client, delay_ms: u64, max_requests: usize) -> SecClient {
        SecClient {
            client,
            delay_ms,
            max_requests,
            requests: std::cell::Cell::new(0),
            last_request: std::cell::RefCell::new(None),
        }
    }

    fn get(&self, url: &str) -> Result<reqwest::blocking::Response> {
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
            bail!(
                "SEC returned 429 Too Many Requests for {url}; stop now and wait at least 10 minutes before retrying"
            );
        }
        response
            .error_for_status()
            .with_context(|| format!("SEC returned an error for {url}"))
    }

    fn wait_for_slot(&self) {
        if self.delay_ms == 0 {
            *self.last_request.borrow_mut() = Some(Instant::now());
            return;
        }
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
) -> Vec<InsiderSignal> {
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

    document
        .non_derivative_table
        .as_ref()
        .map(NonDerivativeTable::non_derivative_transactions)
        .unwrap_or(&[])
        .iter()
        .filter_map(|tx| {
            let code = tx
                .transaction_coding
                .as_ref()
                .and_then(|coding| coding.transaction_code.clone())?;
            let acquired = tx
                .transaction_amounts
                .as_ref()
                .and_then(|amounts| amounts.transaction_acquired_disposed_code.as_ref())
                .and_then(|v| v.value.as_deref())
                == Some("A");
            let shares = tx
                .transaction_amounts
                .as_ref()
                .and_then(|amounts| amounts.transaction_shares.as_ref())
                .and_then(|v| v.value.as_deref())
                .and_then(parse_number)?;
            if code != "P" || !acquired || shares <= 0.0 {
                return None;
            }
            let price = tx
                .transaction_amounts
                .as_ref()
                .and_then(|amounts| amounts.transaction_price_per_share.as_ref())
                .and_then(|v| v.value.as_deref())
                .and_then(parse_number)
                .unwrap_or(0.0);
            let value_usd = shares * price;
            if value_usd < min_value_usd {
                return None;
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
            let transaction_date = tx
                .transaction_date
                .as_ref()
                .and_then(|v| v.value.clone())
                .or_else(|| filing.report_date.clone())
                .unwrap_or_else(|| filing.filing_date.clone());
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
                acquired_disposed: "A".into(),
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
            Some(signal)
        })
        .collect()
}

fn score_signal(signal: &mut InsiderSignal) {
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
    let mut owners_by_ticker: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    for signal in signals.iter() {
        owners_by_ticker
            .entry(signal.ticker.clone())
            .or_default()
            .insert(signal.owner.clone());
    }
    for signal in signals {
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
    let known_accessions: HashSet<&str> = watchlist
        .signals
        .iter()
        .map(|signal| signal.accession.as_str())
        .collect();
    for accession in cited_accessions(answer) {
        if !known_accessions.contains(accession.as_str()) {
            check.warn(format!("unknown accession cited: {accession}"));
        }
    }

    for line in answer.lines().filter(|line| !line.trim().is_empty()) {
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
}

#[derive(Debug, Serialize, Deserialize)]
struct CompanyEvidence {
    ticker: String,
    cik: String,
    company: String,
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
            profile: "mistral".into(),
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
        let signals = signals_from_document(
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
}
