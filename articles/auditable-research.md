# Auditable research

The investment-research example is the clearest demonstration of why this shape
is interesting. Rust resolves a ticker through SEC EDGAR, fetches public XBRL
company facts, normalizes them into cited evidence records, embeds a local chat
model, asks for a concise research note, then audits the generated thesis against
the evidence it supplied.

```bash
SEC_USER_AGENT="your-name your-email@example.com" \
  cargo run -p yatima-lib --release --example investment_thesis --features metal -- \
    --ticker AAPL
```

Compare several models on the same evidence:

```bash
SEC_USER_AGENT="yatima research example shayne@shayne-fletcher.org" \
cargo run -p yatima-lib --release --example investment_thesis --features metal -- \
  --ticker META \
  --compare qwen32b,gemma2,mistral \
  --temperature 0 \
  --max-tokens 900
```

The generated note is not investment advice. It is a grounded-output demo:
quantity-bearing claims are expected to cite the SEC accession, period/filed date,
and XBRL tag they came from. The example-local validator warns when the model
cites unknown tags or accessions, drifts from normalized `value_text`, omits
citation fields, or uses trend language when only one period was supplied.

The same shape powers **`sieve`**, a private project built on `yatima-lib` that
screens SEC Form 4 filings for insider buys: it keeps only open-market `P`
purchases and assigns each issuer a deterministic Rust signal tier — strong,
moderate, weak, or noise — that caps what the model may claim. Available to
invited collaborators on request.
