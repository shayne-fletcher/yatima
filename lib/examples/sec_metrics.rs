//! Fetch a ticker's public SEC metrics and normalize them into cited facts.
//!
//! This is a small data-side example for auditable investment research: Rust
//! resolves a ticker, pulls SEC XBRL company facts, and emits typed JSON with
//! provenance the model can cite later.
//!
//! ```bash
//! SEC_USER_AGENT="your-name your-email@example.com" \
//!   cargo run -p yatima-lib --example sec_metrics -- AAPL
//! ```

use anyhow::{anyhow, bail, Context, Result};
use reqwest::blocking::Client;
use serde::{Deserialize, Serialize};
use serde_json::Number;
use std::collections::HashMap;
use std::time::Duration;

const COMPANY_TICKERS_URL: &str = "https://www.sec.gov/files/company_tickers.json";
const COMPANYFACTS_BASE_URL: &str = "https://data.sec.gov/api/xbrl/companyfacts";

fn main() -> Result<()> {
    let ticker = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "AAPL".to_string())
        .to_uppercase();
    let user_agent = std::env::var("SEC_USER_AGENT").context(
        "set SEC_USER_AGENT to a descriptive value with contact info, \
         e.g. 'Your Name your.email@example.com'",
    )?;

    let client = Client::builder()
        .user_agent(user_agent)
        .timeout(Duration::from_secs(30))
        .build()?;

    let company = resolve_ticker(&client, &ticker)?;
    let facts = fetch_company_facts(&client, company.cik())?;
    let metrics = extract_metrics(&facts);
    if metrics.is_empty() {
        bail!(
            "no supported metrics found for {ticker} / CIK {}",
            cik10(company.cik())
        );
    }

    let report = MetricsReport {
        ticker,
        cik: cik10(company.cik()),
        company: company.title,
        entity_name: facts.entity_name,
        source: "SEC EDGAR companyfacts".into(),
        metrics,
    };
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
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
                    unit: unit.to_string(),
                    fy: fact.fy,
                    fp: fact.fp.clone(),
                    form: fact.form.clone(),
                    filed: fact.filed.clone(),
                    end: fact.end.clone(),
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
    unit: String,
    fy: Option<i64>,
    fp: Option<String>,
    form: Option<String>,
    filed: Option<String>,
    end: Option<String>,
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
    fn extracts_latest_supported_filing_with_provenance() {
        let facts: CompanyFacts = serde_json::from_value(json!({
            "entityName": "Example Corp",
            "facts": {
                "us-gaap": {
                    "Revenues": {
                        "label": "Revenue",
                        "units": {
                            "USD": [
                                {
                                    "val": 100,
                                    "accn": "old",
                                    "fy": 2023,
                                    "fp": "FY",
                                    "form": "10-K",
                                    "filed": "2024-01-01",
                                    "end": "2023-12-31"
                                },
                                {
                                    "val": 125,
                                    "accn": "new",
                                    "fy": 2024,
                                    "fp": "FY",
                                    "form": "10-K",
                                    "filed": "2025-01-01",
                                    "end": "2024-12-31",
                                    "frame": "CY2024"
                                }
                            ]
                        }
                    }
                }
            }
        }))
        .unwrap();

        let metrics = extract_metrics(&facts);
        assert_eq!(metrics.len(), 1);
        assert_eq!(metrics[0].metric, "Revenue");
        assert_eq!(metrics[0].value, Number::from(125));
        assert_eq!(metrics[0].accession.as_deref(), Some("new"));
        assert_eq!(metrics[0].taxonomy, "us-gaap");
        assert_eq!(metrics[0].xbrl_tag, "Revenues");
        assert_eq!(metrics[0].frame.as_deref(), Some("CY2024"));
    }
}
