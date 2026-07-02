//! Metal-vs-CPU probe for `candle_nn::rotary_emb::rope` at large positions.
//!
//! The model narrows the precomputed cos/sin tables at `index_pos`, handing
//! the rope kernel *offset views* into the table buffer. This probes the
//! kernel across positions bracketing the observed ~8.2k generation cliff,
//! with the tables passed both as narrowed views (as the model does) and as
//! contiguous copies (control), for decode (seq=1) and prefill-chunk (seq=64)
//! shapes.
//!
//! Run: cargo run -p yatima-lib --release --example metal_rope_probe \
//!   --features metal

use anyhow::Result;
use candle_core::{DType, Device, Tensor};

const HEADS: usize = 40;
const HEAD_DIM: usize = 128;
const CONTEXT: usize = 32768;
const FREQ_BASE: f32 = 1_000_000.0;

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
    let d = (a - b)?.abs()?.flatten_all()?.max(0)?.to_scalar::<f32>()?;
    Ok(d)
}

fn main() -> Result<()> {
    let cpu = Device::Cpu;
    let metal = Device::new_metal(0)?;
    let (cos_c, sin_c) = freqs_cis(&cpu)?;
    // The model computes the table on its own device; both variants were shown
    // equal by metal_trig_probe, so build Metal's from the CPU one for exact
    // input parity and probe only the rope kernel here.
    let (cos_m, sin_m) = (cos_c.to_device(&metal)?, sin_c.to_device(&metal)?);

    for &seq in &[1usize, 64] {
        let x = Tensor::from_vec(
            pseudo(HEADS * seq * HEAD_DIM, 3),
            (1, HEADS, seq, HEAD_DIM),
            &cpu,
        )?;
        let x_m = x.to_device(&metal)?;
        println!("-- seq={seq} --");
        for &p in &[
            0usize, 4096, 8000, 8191, 8192, 8193, 8300, 9000, 11605, 16384,
        ] {
            let (cc, sc) = (cos_c.narrow(0, p, seq)?, sin_c.narrow(0, p, seq)?);
            let (cm, sm) = (cos_m.narrow(0, p, seq)?, sin_m.narrow(0, p, seq)?);
            let r_cpu = candle_nn::rotary_emb::rope(&x, &cc, &sc)?;
            // As the model does it: offset views straight into the kernel.
            let r_view = candle_nn::rotary_emb::rope(&x_m, &cm, &sm)?.to_device(&cpu)?;
            // Control: contiguous copies of the same rows.
            let r_copy = candle_nn::rotary_emb::rope(&x_m, &cm.contiguous()?, &sm.contiguous()?)?
                .to_device(&cpu)?;
            let dv = max_abs_diff(&r_cpu, &r_view)?;
            let dc = max_abs_diff(&r_cpu, &r_copy)?;
            let flag = if dv.max(dc) > 1e-2 {
                "  <-- DIVERGED"
            } else {
                ""
            };
            println!("pos={p:>6}  view max|Δ| = {dv:.3e}   contig max|Δ| = {dc:.3e}{flag}");
        }
    }
    let _ = DType::F32;
    Ok(())
}
