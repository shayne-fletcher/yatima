//! Gated guard: seeded sampling is reproducible (the seed is actually wired
//! through to the sampler), and a different seed gives different output.
use yatima_lib::{
    device, is_model_present, model_dir, models_root, Engine, GenOpts, ModelId, Sampling,
};

fn gen(engine: &mut Engine, seed: u64) -> String {
    let opts = GenOpts {
        max_tokens: 48,
        sampling: Sampling::nucleus(0.8, Some(0.95), seed),
        ..Default::default()
    };
    let mut out = String::new();
    engine
        .generate("Tell me a short story about a fox.", &opts, |s| {
            out.push_str(s);
            Ok(())
        })
        .unwrap();
    out
}

#[test]
fn seeded_sampling_is_reproducible() -> anyhow::Result<()> {
    if std::env::var_os("YATIMA_E2E").is_none() {
        eprintln!("skip: set YATIMA_E2E=1");
        return Ok(());
    }
    let dir = model_dir(
        &models_root(),
        &ModelId::parse("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")?,
    );
    if !is_model_present(&dir) {
        eprintln!("skip: not cached");
        return Ok(());
    }
    let mut engine = Engine::load(&dir, device(false)?)?;
    let a = gen(&mut engine, 7);
    let b = gen(&mut engine, 7);
    let c = gen(&mut engine, 99);
    assert_eq!(
        a, b,
        "same seed must be byte-identical — the seed is wired through"
    );
    assert_ne!(a, c, "a different seed should change sampled output");
    Ok(())
}
