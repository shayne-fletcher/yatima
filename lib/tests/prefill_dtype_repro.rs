//! Regression (gated e2e): chunked prefill must work on a BF16 safetensors model.
//!
//! Before the dtype gate (`prefill_chunk_for`), a BF16 model on Metal chunked its
//! prefill and Candle's KV-cache `cat` failed with "dtype mismatch in cat, lhs:
//! BF16, rhs: F32" on any prompt longer than one chunk (~64 tokens) — surfacing
//! mid-conversation once history grew. This drives a >1-chunk prompt through the
//! real model and asserts it generates. The pure gate is unit-tested model-free
//! in `engine.rs` (`prefill_chunking_is_gated_on_f32_dtype`); this is the
//! end-to-end guard on real weights. `YATIMA_E2E=1`, Metal, skips if uncached.

use yatima_lib::{
    device, is_model_present, model_dir, models_root, Engine, GenOpts, ModelId, Sampling,
};

#[test]
fn chunked_prefill_on_bf16_safetensors_generates() -> anyhow::Result<()> {
    if std::env::var_os("YATIMA_E2E").is_none() {
        eprintln!("skipping e2e: set YATIMA_E2E=1 to run");
        return Ok(());
    }
    // A small BF16 safetensors Qwen2 model — same arch/dtype as the 7B that hit
    // the bug, but fast.
    let dir = model_dir(
        &models_root(),
        &ModelId::parse("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")?,
    );
    if !is_model_present(&dir) {
        eprintln!("skip: model not cached");
        return Ok(());
    }
    let mut engine = Engine::load(&dir, device(false)?)?;
    eprintln!(
        "backend={} default_prefill_chunk={:?}",
        engine.backend(),
        engine.default_prefill_chunk()
    );
    // Well over one prefill chunk (~64), to force the multi-chunk path that used
    // to crash.
    let long_prompt = "word ".repeat(200);
    let opts = GenOpts {
        max_tokens: 8,
        sampling: Sampling::Greedy,
        ..Default::default()
    };
    let mut out = String::new();
    engine.generate(&long_prompt, &opts, |s| {
        out.push_str(s);
        Ok(())
    })?; // must not error with the cat dtype mismatch
    eprintln!("generated: {out:?}");
    assert!(
        !out.is_empty(),
        "expected tokens from a chunked-prefill prompt"
    );
    Ok(())
}
