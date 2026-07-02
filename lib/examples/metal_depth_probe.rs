//! Metal-vs-CPU numerics probe for the attention ops at long KV depths.
//!
//! Quantized 32B GGUF generation on Metal corrupts deterministically once the
//! prompt passes ~8.2k tokens (clean at 8,100, garbage at 8,300; prefill
//! chunking does not help). The sequence-length-dependent ops in the quantized
//! attention path are exactly: q·kᵀ, masked fill, softmax over the KV row,
//! probs·v, and the KV-cache cat. This example runs each op on identical
//! inputs on CPU and Metal at depths bracketing the cliff and reports the
//! divergence, isolating which kernel breaks without loading a model.
//!
//! Run: cargo run -p yatima-lib --release --example metal_depth_probe \
//!   --features metal

use anyhow::Result;
use candle_core::{DType, Device, Tensor};

// Qwen2.5-32B attention geometry (post repeat_kv: all 40 heads carry KV).
const HEADS: usize = 40;
const HEAD_DIM: usize = 128;
const PREFILL_CHUNK: usize = 64;

/// Deterministic pseudo-random f32s in [-scale, scale] (xorshift; same bits on
/// every run and both devices — the tensors are built once on CPU and copied).
fn pseudo(n: usize, seed: u64, scale: f32) -> Vec<f32> {
    let mut s = seed | 1;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s as f64 / u64::MAX as f64) as f32 * 2.0 * scale - scale
        })
        .collect()
}

fn max_abs_diff(a: &Tensor, b: &Tensor) -> Result<f32> {
    let d = (a - b)?.abs()?.flatten_all()?.max(0)?.to_scalar::<f32>()?;
    Ok(d)
}

/// Compare one op across devices; print max |Δ| and flag non-finite outputs.
fn report(name: &str, kv: usize, cpu_out: &Tensor, metal_out: &Tensor) -> Result<()> {
    let metal_on_cpu = metal_out.to_device(&Device::Cpu)?;
    let diff = max_abs_diff(cpu_out, &metal_on_cpu)?;
    let finite = metal_on_cpu
        .flatten_all()?
        .to_vec1::<f32>()?
        .iter()
        .all(|v| v.is_finite());
    let flag = if !finite {
        "  <-- NON-FINITE"
    } else if diff > 1e-2 {
        "  <-- DIVERGED"
    } else {
        ""
    };
    println!("kv={kv:>6}  {name:<22} max|Δ| = {diff:.3e}{flag}");
    Ok(())
}

fn probe(metal: &Device, kv: usize, q_rows: usize) -> Result<()> {
    let cpu = Device::Cpu;
    // Attention-scale inputs: q/k/v activations are O(1); scores are scaled by
    // 1/sqrt(head_dim) in the model, applied here too.
    let q = Tensor::from_vec(
        pseudo(HEADS * q_rows * HEAD_DIM, 3, 1.0),
        (HEADS, q_rows, HEAD_DIM),
        &cpu,
    )?;
    let k = Tensor::from_vec(
        pseudo(HEADS * kv * HEAD_DIM, 5, 1.0),
        (HEADS, kv, HEAD_DIM),
        &cpu,
    )?;
    let v = Tensor::from_vec(
        pseudo(HEADS * kv * HEAD_DIM, 7, 1.0),
        (HEADS, kv, HEAD_DIM),
        &cpu,
    )?;

    let (q_m, k_m, v_m) = (
        q.to_device(metal)?,
        k.to_device(metal)?,
        v.to_device(metal)?,
    );

    // q·kᵀ — decode (q_rows=1) and prefill-chunk (q_rows=64) shapes.
    let att = (q.matmul(&k.t()?)? / (HEAD_DIM as f64).sqrt())?;
    let att_m = (q_m.matmul(&k_m.t()?)? / (HEAD_DIM as f64).sqrt())?;
    report(&format!("q@k.t (rows={q_rows})"), kv, &att, &att_m)?;

    // softmax over the KV row — computed on each device from ITS own scores to
    // mirror the model, then also from identical inputs to isolate the op.
    let sm = candle_nn::ops::softmax_last_dim(&att)?;
    let sm_m = candle_nn::ops::softmax_last_dim(&att.to_device(metal)?)?;
    report("softmax_last_dim", kv, &sm, &sm_m)?;

    // probs·v — probabilities from the CPU softmax on both devices.
    let y = sm.matmul(&v)?;
    let y_m = sm.to_device(metal)?.matmul(&v_m)?;
    report(&format!("probs@v (rows={q_rows})"), kv, &y, &y_m)?;

    // KV cat at the sequence dim (cache append), then a strided read.
    let one = Tensor::from_vec(
        pseudo(HEADS * HEAD_DIM, 11, 1.0),
        (HEADS, 1, HEAD_DIM),
        &cpu,
    )?;
    let cat = Tensor::cat(&[&k, &one], 1)?.contiguous()?;
    let cat_m = Tensor::cat(&[&k_m, &one.to_device(metal)?], 1)?.contiguous()?;
    report("cat+contiguous", kv, &cat, &cat_m)?;

    // Causal mask fill on the score matrix (prefill path only).
    if q_rows > 1 {
        let mask_bits: Vec<u8> = (0..q_rows)
            .flat_map(|i| (0..kv).map(move |j| u8::from(j > kv - q_rows + i)))
            .collect();
        let mask =
            Tensor::from_vec(mask_bits, (q_rows, kv), &cpu)?.broadcast_as((HEADS, q_rows, kv))?;
        let neg = Tensor::new(f32::NEG_INFINITY, &cpu)?.broadcast_as((HEADS, q_rows, kv))?;
        let filled = mask.where_cond(&neg, &att)?;
        let filled_m = mask
            .to_device(metal)?
            .where_cond(&neg.to_device(metal)?, &att.to_device(metal)?)?;
        // Compare post-softmax (the -inf entries defeat a raw abs diff).
        report(
            "mask+softmax",
            kv,
            &candle_nn::ops::softmax_last_dim(&filled)?,
            &candle_nn::ops::softmax_last_dim(&filled_m)?,
        )?;
    }
    Ok(())
}

fn main() -> Result<()> {
    let metal = Device::new_metal(0)?;
    println!("dtype=F32 heads={HEADS} head_dim={HEAD_DIM}");
    for &kv in &[4096usize, 8000, 8100, 8192, 8193, 8300, 9000, 11605] {
        println!("-- decode shape (1 query row) --");
        probe(&metal, kv, 1)?;
        println!("-- prefill-chunk shape ({PREFILL_CHUNK} query rows) --");
        probe(&metal, kv, PREFILL_CHUNK)?;
    }
    let _ = DType::F32; // keep the import honest if shapes change
    Ok(())
}
