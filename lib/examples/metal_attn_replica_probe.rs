//! Full-fidelity replica of one quantized-qwen2 decode attention step,
//! CPU vs Metal, at KV depths bracketing the 8,192 corruption cliff.
//!
//! The earlier probes (`metal_depth_probe`, `metal_rope_probe`,
//! `metal_trig_probe`) test each op in isolation on simplified 3D contiguous
//! tensors and are clean at these shapes — yet the composed model corrupts
//! deterministically at kv = 8,192 (`notes/metal-kv-cliff.md`). This probe
//! closes the provenance gap: it replays a decode step *exactly* as
//! `candle_transformers::models::quantized_qwen2::LayerWeights::forward_attn`
//! does — 4D tensors with the batch dim, an 8-head KV cache grown by `cat`,
//! `repeat_kv` (cat-of-n + reshape), RoPE applied through narrowed table
//! views at the true positions — and diffs every intermediate across
//! devices. No model weights; tensors are tens of MB; runs in seconds.
//!
//! Run: cargo run -p yatima-lib --release --example metal_attn_replica_probe \
//!   --features metal

use anyhow::Result;
use candle_core::{DType, Device, Tensor, D};

// Qwen2.5-32B geometry.
const N_HEAD: usize = 40;
const N_KV_HEAD: usize = 8;
const HEAD_DIM: usize = 128;
const CONTEXT: usize = 32768;
const FREQ_BASE: f32 = 1_000_000.0;

/// The exact table construction from candle's quantized qwen2 (f32).
fn freqs_cis(device: &Device) -> Result<(Tensor, Tensor)> {
    let theta: Vec<_> = (0..HEAD_DIM)
        .step_by(2)
        .map(|i| 1f32 / FREQ_BASE.powf(i as f32 / HEAD_DIM as f32))
        .collect();
    let theta = Tensor::new(theta.as_slice(), device)?;
    let idx_theta = Tensor::arange(0, CONTEXT as u32, device)?
        .to_dtype(DType::F32)?
        .reshape((CONTEXT, 1))?
        .matmul(&theta.reshape((1, theta.elem_count()))?)?;
    Ok((idx_theta.cos()?, idx_theta.sin()?))
}

/// The exact `repeat_kv` from candle-transformers utils (cat-of-n + reshape).
fn repeat_kv(xs: Tensor, n_rep: usize) -> Result<Tensor> {
    if n_rep == 1 {
        return Ok(xs);
    }
    let (b_sz, n_kv_head, seq_len, head_dim) = xs.dims4()?;
    Ok(Tensor::cat(&vec![&xs; n_rep], 2)?.reshape((b_sz, n_kv_head * n_rep, seq_len, head_dim))?)
}

/// Deterministic pseudo-random f32s in [-1, 1] (same bits every run).
fn pseudo(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed | 1;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s as f64 / u64::MAX as f64) as f32 * 2.0 - 1.0
        })
        .collect()
}

fn max_abs_diff(a: &Tensor, b: &Tensor) -> Result<f32> {
    Ok((a - b.to_device(a.device())?)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?)
}

/// One decode step at position `pos` (cache holds `pos` entries, attends over
/// `pos + 1`): returns every intermediate for cross-device diffing.
struct Step {
    q_rot: Tensor,
    k_rot: Tensor,
    k_cache: Tensor,
    k_full: Tensor,
    att: Tensor,
    probs: Tensor,
    out: Tensor,
}

fn decode_step(
    device: &Device,
    cos: &Tensor,
    sin: &Tensor,
    cache_k: &Tensor, // (1, 8, pos, 128) on `device`
    cache_v: &Tensor,
    q_new: &Tensor, // (1, 40, 1, 128) on `device`
    k_new: &Tensor, // (1, 8, 1, 128)
    v_new: &Tensor,
    pos: usize,
) -> Result<Step> {
    // apply_rotary_emb: narrow the table at index_pos, rope the new q/k.
    let (c, s) = (cos.narrow(0, pos, 1)?, sin.narrow(0, pos, 1)?);
    let q_rot = candle_nn::rotary_emb::rope(&q_new.contiguous()?, &c, &s)?;
    let k_rot = candle_nn::rotary_emb::rope(&k_new.contiguous()?, &c, &s)?;

    // KV cache append, exactly as the model does mid-generation.
    let k_cache = Tensor::cat(&[cache_k, &k_rot], 2)?;
    let v_cache = Tensor::cat(&[cache_v, v_new], 2)?;

    // GQA expansion.
    let k_full = repeat_kv(k_cache.clone(), N_HEAD / N_KV_HEAD)?;
    let v_full = repeat_kv(v_cache, N_HEAD / N_KV_HEAD)?;

    // Attention; decode (seq_len 1) carries no mask in the model.
    let att = (q_rot.matmul(&k_full.t()?)? / (HEAD_DIM as f64).sqrt())?;
    let probs = candle_nn::ops::softmax_last_dim(&att)?;
    let out = probs.matmul(&v_full.contiguous()?)?;
    let out = out
        .transpose(1, 2)?
        .reshape(&[1usize, 1, N_HEAD * HEAD_DIM])?;
    let _ = device;
    Ok(Step {
        q_rot,
        k_rot,
        k_cache,
        att,
        k_full,
        probs,
        out,
    })
}

fn main() -> Result<()> {
    let cpu = Device::Cpu;
    let metal = Device::new_metal(0)?;
    let (cos_c, sin_c) = freqs_cis(&cpu)?;
    let (cos_m, sin_m) = (cos_c.to_device(&metal)?, sin_c.to_device(&metal)?);

    for &pos in &[8_189usize, 8_190, 8_191, 8_192, 8_193, 8_194] {
        // A cache of `pos` rope'd-looking entries plus this step's new q/k/v,
        // identical bits on both devices.
        let cache_k = Tensor::from_vec(
            pseudo(N_KV_HEAD * pos * HEAD_DIM, 3),
            (1, N_KV_HEAD, pos, HEAD_DIM),
            &cpu,
        )?;
        let cache_v = Tensor::from_vec(
            pseudo(N_KV_HEAD * pos * HEAD_DIM, 5),
            (1, N_KV_HEAD, pos, HEAD_DIM),
            &cpu,
        )?;
        let q_new = Tensor::from_vec(pseudo(N_HEAD * HEAD_DIM, 7), (1, N_HEAD, 1, HEAD_DIM), &cpu)?;
        let k_new = Tensor::from_vec(
            pseudo(N_KV_HEAD * HEAD_DIM, 11),
            (1, N_KV_HEAD, 1, HEAD_DIM),
            &cpu,
        )?;
        let v_new = Tensor::from_vec(
            pseudo(N_KV_HEAD * HEAD_DIM, 13),
            (1, N_KV_HEAD, 1, HEAD_DIM),
            &cpu,
        )?;

        let ref_step = decode_step(
            &cpu, &cos_c, &sin_c, &cache_k, &cache_v, &q_new, &k_new, &v_new, pos,
        )?;
        let dev_step = decode_step(
            &metal,
            &cos_m,
            &sin_m,
            &cache_k.to_device(&metal)?,
            &cache_v.to_device(&metal)?,
            &q_new.to_device(&metal)?,
            &k_new.to_device(&metal)?,
            &v_new.to_device(&metal)?,
            pos,
        )?;

        println!("== pos={pos} (kv after cat = {}) ==", pos + 1);
        for (name, a, b) in [
            ("q_rot", &ref_step.q_rot, &dev_step.q_rot),
            ("k_rot", &ref_step.k_rot, &dev_step.k_rot),
            ("k_cache(cat)", &ref_step.k_cache, &dev_step.k_cache),
            ("k_full(repeat_kv)", &ref_step.k_full, &dev_step.k_full),
            ("att(q@kT)", &ref_step.att, &dev_step.att),
            ("probs(softmax)", &ref_step.probs, &dev_step.probs),
            ("out(probs@v)", &ref_step.out, &dev_step.out),
        ] {
            let d = max_abs_diff(a, b)?;
            let flag = if d > 1e-2 { "  <-- DIVERGED" } else { "" };
            println!("  {name:<18} max|Δ| = {d:.3e}{flag}");
        }
        // The scalar that actually decides the next token's fate: is the new
        // (last) position's attention weight sane on Metal?
        let last_w = dev_step
            .probs
            .to_device(&cpu)?
            .narrow(D::Minus1, pos, 1)?
            .flatten_all()?
            .max(0)?
            .to_scalar::<f32>()?;
        println!("  metal max attn weight at new position: {last_w:.3e}");
    }
    Ok(())
}
