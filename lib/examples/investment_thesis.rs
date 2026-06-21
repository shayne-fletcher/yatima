//! Embed a local model to turn public SEC facts into a cited research note.
//!
//! Rust fetches and normalizes real public evidence, an in-process chat model
//! writes a thesis constrained to that evidence, and Rust audits the result for
//! unsupported citations and claims.
//!
//! ```bash
//! SEC_USER_AGENT="your-name your-email@example.com" \
//!   cargo run -p yatima-lib --release --example investment_thesis --features metal -- AAPL
//! ```
//!
//! Pass an explicit model directory as the second argument and chat format as
//! the third (`qwen`, `gemma`, `mistral`, `glm`, or `plain`) if you do not want the
//! default Qwen 32B GGUF path:
//!
//! ```bash
//! SEC_USER_AGENT="your-name your-email@example.com" \
//!   cargo run -p yatima-lib --release --example investment_thesis --features metal -- \
//!     MSFT ~/.cache/yatima/models/google/gemma-2-2b-it gemma
//! ```

use anyhow::{anyhow, bail, Context, Result};
use reqwest::blocking::Client;
use serde::{Deserialize, Serialize};
use serde_json::Number;
use std::collections::{HashMap, HashSet};
use std::io::Write;
use std::path::PathBuf;
use std::time::Duration;
use yatima_lib::{
    device, ChatMlTemplate, ChatSession, Engine, GemmaTemplate, GenOpts, GlmTemplate,
    MistralTemplate, PlainTemplate, PromptTemplate, Sampling,
};

const COMPANY_TICKERS_URL: &str = "https://www.sec.gov/files/company_tickers.json";
const COMPANYFACTS_BASE_URL: &str = "https://data.sec.gov/api/xbrl/companyfacts";

const SYSTEM: &str = "\
You are an investment research analyst producing an educational research note, \
not investment advice. Use only the supplied SEC facts. Every factual claim \
about company performance, balance sheet strength, cash generation, or share \
count must cite the relevant metric with its filed date, accession, and XBRL \
tag. Cite only XBRL tags that appear in the supplied JSON. Do not infer fiscal \
quarter names; use the supplied period field. Do not rescale numeric values; \
copy value_text exactly. Do not describe a single-period metric as growth unless \
the supplied evidence includes a prior comparison period. If the evidence is \
insufficient, say so. Separate thesis, evidence, risks, and testable signals.";

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let ticker = args
        .next()
        .unwrap_or_else(|| "AAPL".to_string())
        .to_uppercase();
    let model_dir = args
        .next()
        .map(PathBuf::from)
        .unwrap_or_else(default_qwen_32b_dir);
    let format = args
        .next()
        .as_deref()
        .map(ChatFormat::parse)
        .transpose()?
        .unwrap_or(ChatFormat::Qwen);
    let prefill_chunk = args
        .next()
        .map(|s| s.parse::<usize>().context("parsing prefill chunk"))
        .transpose()?;

    let user_agent = std::env::var("SEC_USER_AGENT").context(
        "set SEC_USER_AGENT to a descriptive value with contact info, \
         e.g. 'Your Name your.email@example.com'",
    )?;

    let client = Client::builder()
        .user_agent(user_agent)
        .timeout(Duration::from_secs(30))
        .build()?;

    let report = fetch_metrics_report(&client, &ticker)?;
    let evidence_json = serde_json::to_string_pretty(&report)?;
    eprintln!(
        "fetched {} SEC metrics for {} / CIK {}",
        report.metrics.len(),
        report.ticker,
        report.cik
    );

    let mut engine = Engine::load(&model_dir, device(false)?)
        .with_context(|| format!("loading model {}", model_dir.display()))?;
    eprintln!(
        "loaded {} [{}]; format {}",
        model_dir.display(),
        engine.backend(),
        format.name()
    );

    let prompt = format!(
        "\
Write a concise investment research note from this SEC evidence.

Required format:
- Thesis: one paragraph.
- Evidence: 3-5 bullets, each using value_text and citing period, filed date, accession, and XBRL tag.
- Risks / counterpoints: 2-3 bullets, only from gaps or weaknesses visible in the supplied facts.
- Testable signals: 2-3 measurable follow-ups, citing only supplied metric names and XBRL tags. Do not cite unsupplied tags.

SEC evidence JSON:
```json
{evidence_json}
```"
    );

    let opts = GenOpts {
        max_tokens: 768,
        sampling: Sampling::Greedy,
        prefill_chunk,
        ..Default::default()
    };
    let mut chat = ChatSession::new(&mut engine, format.template())
        .with_system(SYSTEM)
        .with_opts(opts);
    let mut stdout = std::io::stdout();
    let answer = chat
        .turn_streaming(&prompt, &mut |piece| {
            let _ = stdout.write_all(piece.as_bytes());
            let _ = stdout.flush();
        })?
        .to_string();
    println!();
    let check = check_thesis(&answer, &report);
    if !check.is_clean() {
        eprintln!("\nvalidation warnings:");
        for warning in &check.warnings {
            eprintln!("- {warning}");
        }
    }
    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ChatFormat {
    Qwen,
    Gemma,
    Mistral,
    Glm,
    Plain,
}

impl ChatFormat {
    fn parse(s: &str) -> Result<ChatFormat> {
        match s {
            "qwen" => Ok(ChatFormat::Qwen),
            "gemma" => Ok(ChatFormat::Gemma),
            "mistral" => Ok(ChatFormat::Mistral),
            "glm" => Ok(ChatFormat::Glm),
            "plain" => Ok(ChatFormat::Plain),
            other => bail!("unknown chat format {other:?}; expected qwen|gemma|mistral|glm|plain"),
        }
    }

    fn name(self) -> &'static str {
        match self {
            ChatFormat::Qwen => "qwen",
            ChatFormat::Gemma => "gemma",
            ChatFormat::Mistral => "mistral",
            ChatFormat::Glm => "glm",
            ChatFormat::Plain => "plain",
        }
    }

    fn template(self) -> Box<dyn PromptTemplate> {
        match self {
            ChatFormat::Qwen => Box::new(ChatMlTemplate),
            ChatFormat::Gemma => Box::new(GemmaTemplate),
            ChatFormat::Mistral => Box::new(MistralTemplate),
            ChatFormat::Glm => Box::new(GlmTemplate),
            ChatFormat::Plain => Box::new(PlainTemplate),
        }
    }
}

fn default_qwen_32b_dir() -> PathBuf {
    let home = std::env::var_os("HOME").unwrap_or_default();
    PathBuf::from(home)
        .join(".cache")
        .join("yatima")
        .join("models")
        .join("bartowski")
        .join("Qwen2.5-32B-Instruct-GGUF")
}

fn fetch_metrics_report(client: &Client, ticker: &str) -> Result<MetricsReport> {
    let company = resolve_ticker(client, ticker)?;
    let facts = fetch_company_facts(client, company.cik())?;
    let metrics = extract_metrics(&facts);
    if metrics.is_empty() {
        bail!(
            "no supported metrics found for {ticker} / CIK {}",
            cik10(company.cik())
        );
    }
    Ok(MetricsReport {
        ticker: ticker.to_string(),
        cik: cik10(company.cik()),
        company: company.title,
        entity_name: facts.entity_name,
        source: "SEC EDGAR companyfacts".into(),
        metrics,
    })
}

fn resolve_ticker(client: &Client, ticker: &str) -> Result<TickerEntry> {
    let entries: HashMap<String, TickerEntry> = client
        .get(COMPANY_TICKERS_URL)
        .send()
        .context("fetching SEC ticker map")?
        .error_for_status()
        .context("SEC ticker map returned an error")?
        .json()
        .context("decoding SEC ticker map")?;

    entries
        .into_values()
        .find(|entry| entry.ticker.eq_ignore_ascii_case(ticker))
        .ok_or_else(|| anyhow!("ticker {ticker} not found in SEC company_tickers.json"))
}

fn fetch_company_facts(client: &Client, cik: u64) -> Result<CompanyFacts> {
    let url = format!("{COMPANYFACTS_BASE_URL}/CIK{}.json", cik10(cik));
    client
        .get(url)
        .send()
        .context("fetching SEC companyfacts")?
        .error_for_status()
        .context("SEC companyfacts returned an error")?
        .json()
        .context("decoding SEC companyfacts")
}

fn extract_metrics(facts: &CompanyFacts) -> Vec<MetricFact> {
    metric_specs()
        .iter()
        .filter_map(|spec| extract_metric(facts, spec))
        .collect()
}

fn extract_metric(facts: &CompanyFacts, spec: &MetricSpec) -> Option<MetricFact> {
    let taxonomy = facts.facts.get(spec.taxonomy)?;
    for tag in spec.tags {
        let Some(concept) = taxonomy.get(*tag) else {
            continue;
        };
        for unit in spec.units {
            let Some(candidates) = concept.units.get(*unit) else {
                continue;
            };
            if let Some(fact) = latest_filed(candidates) {
                return Some(MetricFact {
                    metric: spec.name.to_string(),
                    value: fact.val.clone(),
                    value_text: value_text(&fact.val, unit),
                    unit: unit.to_string(),
                    fy: fact.fy,
                    fp: fact.fp.clone(),
                    form: fact.form.clone(),
                    filed: fact.filed.clone(),
                    end: fact.end.clone(),
                    period: period_text(fact),
                    frame: fact.frame.clone(),
                    accession: fact.accn.clone(),
                    taxonomy: spec.taxonomy.to_string(),
                    xbrl_tag: (*tag).to_string(),
                    label: concept.label.clone(),
                });
            }
        }
    }
    None
}

fn latest_filed(facts: &[UnitFact]) -> Option<&UnitFact> {
    facts
        .iter()
        .filter(|fact| {
            matches!(
                fact.form.as_deref(),
                Some("10-K" | "10-K/A" | "10-Q" | "10-Q/A")
            )
        })
        .max_by_key(|fact| (fact.filed.as_deref(), fact.end.as_deref()))
}

fn cik10(cik: u64) -> String {
    format!("{cik:010}")
}

fn period_text(fact: &UnitFact) -> String {
    let mut parts = Vec::new();
    if let Some(fy) = fact.fy {
        parts.push(format!("fy {fy}"));
    }
    if let Some(fp) = &fact.fp {
        parts.push(format!("fp {fp}"));
    }
    if let Some(end) = &fact.end {
        parts.push(format!("end {end}"));
    }
    if let Some(filed) = &fact.filed {
        parts.push(format!("filed {filed}"));
    }
    parts.join(", ")
}

fn value_text(value: &Number, unit: &str) -> String {
    let Some(n) = value.as_f64() else {
        return format!("{value} {unit}");
    };
    match unit {
        "USD" => dollars_text(n),
        "shares" => scaled_text(n, "shares"),
        _ => format!("{value} {unit}"),
    }
}

fn dollars_text(n: f64) -> String {
    if n.abs() >= 1_000_000_000.0 {
        format!("${:.3} billion", n / 1_000_000_000.0)
    } else if n.abs() >= 1_000_000.0 {
        format!("${:.3} million", n / 1_000_000.0)
    } else {
        format!("${n:.0}")
    }
}

fn scaled_text(n: f64, unit: &str) -> String {
    if n.abs() >= 1_000_000_000.0 {
        format!("{:.3} billion {unit}", n / 1_000_000_000.0)
    } else if n.abs() >= 1_000_000.0 {
        format!("{:.3} million {unit}", n / 1_000_000.0)
    } else {
        format!("{n:.0} {unit}")
    }
}

#[derive(Debug, Default, PartialEq, Eq)]
struct ThesisCheck {
    warnings: Vec<String>,
}

impl ThesisCheck {
    fn warn(&mut self, warning: impl Into<String>) {
        self.warnings.push(warning.into());
    }

    fn is_clean(&self) -> bool {
        self.warnings.is_empty()
    }
}

fn check_thesis(answer: &str, report: &MetricsReport) -> ThesisCheck {
    let mut check = ThesisCheck::default();
    let known_tags: HashSet<&str> = report.metrics.iter().map(|m| m.xbrl_tag.as_str()).collect();
    let known_accessions: HashSet<&str> = report
        .metrics
        .iter()
        .filter_map(|m| m.accession.as_deref())
        .collect();

    for accession in cited_accessions(answer) {
        if !known_accessions.contains(accession.as_str()) {
            check.warn(format!("unknown accession cited: {accession}"));
        }
    }

    for tag in cited_xbrl_tags(answer) {
        if !known_tags.contains(tag.as_str()) {
            check.warn(format!("unknown XBRL tag cited: {tag}"));
        }
    }

    for line in answer.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        check_value_line(trimmed, report, &mut check);
        check_missing_citation(trimmed, &known_tags, &known_accessions, &mut check);
    }

    if has_single_period_per_metric(report) && contains_trend_language(answer) {
        check.warn(
            "trend language appears, but the supplied evidence has only one period per metric",
        );
    }

    check
}

fn check_value_line(line: &str, report: &MetricsReport, check: &mut ThesisCheck) {
    let lower = line.to_ascii_lowercase();
    for metric in &report.metrics {
        if lower.contains(&metric.metric.to_ascii_lowercase())
            && mentions_quantity(line)
            && !line.contains(&metric.value_text)
        {
            check.warn(format!(
                "line mentions {} with a quantity but not exact value_text `{}`: {}",
                metric.metric, metric.value_text, line
            ));
        }
    }
}

fn check_missing_citation(
    line: &str,
    known_tags: &HashSet<&str>,
    known_accessions: &HashSet<&str>,
    check: &mut ThesisCheck,
) {
    if !mentions_quantity(line) {
        return;
    }
    let has_tag = known_tags.iter().any(|tag| line.contains(*tag));
    let has_accession = known_accessions.iter().any(|accn| line.contains(*accn));
    if !(has_tag && has_accession) {
        check.warn(format!(
            "quantity-bearing line lacks a known accession and XBRL tag: {line}"
        ));
    }
}

fn mentions_quantity(line: &str) -> bool {
    line.contains('$') || line.to_ascii_lowercase().contains(" shares")
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

fn cited_xbrl_tags(answer: &str) -> Vec<String> {
    let mut tags = Vec::new();
    for line in answer.lines() {
        let Some((_, after)) = line.split_once("XBRL tag") else {
            continue;
        };
        let after = after.trim_start_matches([':', 's', ' ', '(', '[']);
        for token in after.split(|c: char| {
            c.is_whitespace()
                || matches!(c, ',' | ';' | ')' | '(' | '.' | ']' | '[' | ':' | '`' | '*')
        }) {
            let token = token.trim();
            if looks_like_xbrl_tag(token) {
                tags.push(token.to_string());
            }
        }
    }
    tags
}

fn looks_like_xbrl_tag(token: &str) -> bool {
    token.len() > 5
        && token.chars().all(|c| c.is_ascii_alphanumeric())
        && token.chars().any(|c| c.is_ascii_uppercase())
        && token != "XBRL"
        && token != "Tag"
        && token != "Tags"
}

fn has_single_period_per_metric(report: &MetricsReport) -> bool {
    let mut periods_by_metric: HashMap<&str, HashSet<&str>> = HashMap::new();
    for metric in &report.metrics {
        periods_by_metric
            .entry(metric.metric.as_str())
            .or_default()
            .insert(metric.period.as_str());
    }
    periods_by_metric.values().all(|periods| periods.len() <= 1)
}

fn contains_trend_language(answer: &str) -> bool {
    let lower = answer.to_ascii_lowercase();
    [
        " growth",
        " grew",
        " increase",
        " increased",
        " decline",
        " declined",
        " improve",
        " improved",
        " deceleration",
    ]
    .iter()
    .any(|needle| lower.contains(needle))
}

fn metric_specs() -> &'static [MetricSpec] {
    &[
        MetricSpec {
            name: "Revenue",
            taxonomy: "us-gaap",
            tags: &[
                "RevenueFromContractWithCustomerExcludingAssessedTax",
                "Revenues",
                "SalesRevenueNet",
            ],
            units: &["USD"],
        },
        MetricSpec {
            name: "NetIncome",
            taxonomy: "us-gaap",
            tags: &["NetIncomeLoss"],
            units: &["USD"],
        },
        MetricSpec {
            name: "Assets",
            taxonomy: "us-gaap",
            tags: &["Assets"],
            units: &["USD"],
        },
        MetricSpec {
            name: "Liabilities",
            taxonomy: "us-gaap",
            tags: &["Liabilities"],
            units: &["USD"],
        },
        MetricSpec {
            name: "StockholdersEquity",
            taxonomy: "us-gaap",
            tags: &[
                "StockholdersEquity",
                "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
            ],
            units: &["USD"],
        },
        MetricSpec {
            name: "CashAndEquivalents",
            taxonomy: "us-gaap",
            tags: &[
                "CashAndCashEquivalentsAtCarryingValue",
                "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents",
            ],
            units: &["USD"],
        },
        MetricSpec {
            name: "OperatingCashFlow",
            taxonomy: "us-gaap",
            tags: &["NetCashProvidedByUsedInOperatingActivities"],
            units: &["USD"],
        },
        MetricSpec {
            name: "CapitalExpenditures",
            taxonomy: "us-gaap",
            tags: &[
                "PaymentsToAcquirePropertyPlantAndEquipment",
                "PaymentsToAcquireProductiveAssets",
                "CapitalExpendituresIncurredButNotYetPaid",
            ],
            units: &["USD"],
        },
        MetricSpec {
            name: "SharesOutstanding",
            taxonomy: "dei",
            tags: &["EntityCommonStockSharesOutstanding"],
            units: &["shares"],
        },
    ]
}

struct MetricSpec {
    name: &'static str,
    taxonomy: &'static str,
    tags: &'static [&'static str],
    units: &'static [&'static str],
}

#[derive(Debug, Deserialize)]
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
struct CompanyFacts {
    #[serde(rename = "entityName")]
    entity_name: String,
    facts: HashMap<String, HashMap<String, Concept>>,
}

#[derive(Debug, Deserialize)]
struct Concept {
    label: Option<String>,
    units: HashMap<String, Vec<UnitFact>>,
}

#[derive(Debug, Deserialize)]
struct UnitFact {
    val: Number,
    accn: Option<String>,
    fy: Option<i64>,
    fp: Option<String>,
    form: Option<String>,
    filed: Option<String>,
    end: Option<String>,
    frame: Option<String>,
}

#[derive(Debug, Serialize)]
struct MetricsReport {
    ticker: String,
    cik: String,
    company: String,
    entity_name: String,
    source: String,
    metrics: Vec<MetricFact>,
}

#[derive(Debug, Serialize, PartialEq, Eq)]
struct MetricFact {
    metric: String,
    value: Number,
    value_text: String,
    unit: String,
    fy: Option<i64>,
    fp: Option<String>,
    form: Option<String>,
    filed: Option<String>,
    end: Option<String>,
    period: String,
    frame: Option<String>,
    accession: Option<String>,
    taxonomy: String,
    xbrl_tag: String,
    label: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn pads_cik_to_ten_digits() {
        assert_eq!(cik10(320193), "0000320193");
    }

    #[test]
    fn formats_values_for_model_copying() {
        assert_eq!(
            value_text(&Number::from(82_627_000_000_i64), "USD"),
            "$82.627 billion"
        );
        assert_eq!(
            value_text(&Number::from(14_687_356_000_i64), "shares"),
            "14.687 billion shares"
        );
    }

    #[test]
    fn extracts_dei_and_us_gaap_metrics() {
        let facts: CompanyFacts = serde_json::from_value(json!({
            "entityName": "Example Corp",
            "facts": {
                "us-gaap": {
                    "Revenues": {
                        "label": "Revenue",
                        "units": {
                            "USD": [{
                                "val": 125,
                                "accn": "rev-accn",
                                "fy": 2024,
                                "fp": "FY",
                                "form": "10-K",
                                "filed": "2025-01-01",
                                "end": "2024-12-31"
                            }]
                        }
                    }
                },
                "dei": {
                    "EntityCommonStockSharesOutstanding": {
                        "label": null,
                        "units": {
                            "shares": [{
                                "val": 42,
                                "accn": "shares-accn",
                                "fy": 2024,
                                "fp": "FY",
                                "form": "10-K",
                                "filed": "2025-01-01",
                                "end": "2024-12-31"
                            }]
                        }
                    }
                }
            }
        }))
        .unwrap();

        let metrics = extract_metrics(&facts);
        assert!(metrics.iter().any(|m| {
            m.metric == "Revenue"
                && m.taxonomy == "us-gaap"
                && m.value_text == "$125"
                && m.period == "fy 2024, fp FY, end 2024-12-31, filed 2025-01-01"
                && m.accession.as_deref() == Some("rev-accn")
        }));
        assert!(metrics.iter().any(|m| {
            m.metric == "SharesOutstanding"
                && m.taxonomy == "dei"
                && m.label.is_none()
                && m.accession.as_deref() == Some("shares-accn")
        }));
    }

    #[test]
    fn validator_accepts_grounded_citations() {
        let report = sample_report();
        let answer = "\
- Revenue was $111.184 billion (period: fy 2026, fp Q2, end 2026-03-28, filed 2026-05-01; accession: 0000320193-26-000013; XBRL tag: RevenueFromContractWithCustomerExcludingAssessedTax).
- NetIncome was $29.578 billion (period: fy 2026, fp Q2, end 2026-03-28, filed 2026-05-01; accession: 0000320193-26-000013; XBRL tag: NetIncomeLoss).";

        assert!(check_thesis(answer, &report).is_clean());
    }

    #[test]
    fn validator_warns_on_unknown_citation_and_value_drift() {
        let report = sample_report();
        let answer = "\
- Revenue growth was $11.184 billion (accession: 0000000000-00-000000; XBRL tag: MadeUpRevenueTag).
- NetIncome was $29.578 billion without a citation.";

        let check = check_thesis(answer, &report);
        assert!(check
            .warnings
            .iter()
            .any(|w| w.contains("unknown accession")));
        assert!(check
            .warnings
            .iter()
            .any(|w| w.contains("unknown XBRL tag")));
        assert!(check
            .warnings
            .iter()
            .any(|w| w.contains("not exact value_text")));
        assert!(check.warnings.iter().any(|w| w.contains("trend language")));
    }

    fn sample_report() -> MetricsReport {
        MetricsReport {
            ticker: "AAPL".into(),
            cik: "0000320193".into(),
            company: "Apple Inc.".into(),
            entity_name: "Apple Inc.".into(),
            source: "SEC EDGAR companyfacts".into(),
            metrics: vec![
                MetricFact {
                    metric: "Revenue".into(),
                    value: Number::from(111_184_000_000_i64),
                    value_text: "$111.184 billion".into(),
                    unit: "USD".into(),
                    fy: Some(2026),
                    fp: Some("Q2".into()),
                    form: Some("10-Q".into()),
                    filed: Some("2026-05-01".into()),
                    end: Some("2026-03-28".into()),
                    period: "fy 2026, fp Q2, end 2026-03-28, filed 2026-05-01".into(),
                    frame: Some("CY2026Q1".into()),
                    accession: Some("0000320193-26-000013".into()),
                    taxonomy: "us-gaap".into(),
                    xbrl_tag: "RevenueFromContractWithCustomerExcludingAssessedTax".into(),
                    label: Some("Revenue".into()),
                },
                MetricFact {
                    metric: "NetIncome".into(),
                    value: Number::from(29_578_000_000_i64),
                    value_text: "$29.578 billion".into(),
                    unit: "USD".into(),
                    fy: Some(2026),
                    fp: Some("Q2".into()),
                    form: Some("10-Q".into()),
                    filed: Some("2026-05-01".into()),
                    end: Some("2026-03-28".into()),
                    period: "fy 2026, fp Q2, end 2026-03-28, filed 2026-05-01".into(),
                    frame: Some("CY2026Q1".into()),
                    accession: Some("0000320193-26-000013".into()),
                    taxonomy: "us-gaap".into(),
                    xbrl_tag: "NetIncomeLoss".into(),
                    label: Some("Net Income".into()),
                },
            ],
        }
    }
}
