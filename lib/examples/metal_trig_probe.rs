//! Metal-vs-CPU probe for the RoPE cos/sin table at large positions.
//!
//! `precomput_freqs_cis` computes `cos(p·θ)`/`sin(p·θ)` on the load device;
//! for θ₀ = 1 the argument reaches the position index itself. Metal fast-math
//! trig is only accurate on a bounded domain — if the table corrupts past a
//! position threshold, RoPE scrambles q/k there and generation degrades at
//! exactly that prompt depth.
//!
//! Run: cargo run -p yatima-lib --release --example metal_trig_probe \
//!   --features metal

use anyhow::Result;
use candle_core::{DType, Device, Tensor};

const HEAD_DIM: usize = 128;
const CONTEXT: usize = 32768;
const FREQ_BASE: f32 = 1_000_000.0;

/// The exact table construction from candle's quantized qwen2.
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

fn row_max_diff(a: &Tensor, b: &Tensor, row: usize) -> Result<f32> {
    let ra = a.narrow(0, row, 1)?.flatten_all()?.to_vec1::<f32>()?;
    let rb = b.narrow(0, row, 1)?.flatten_all()?.to_vec1::<f32>()?;
    Ok(ra
        .iter()
        .zip(&rb)
        .map(|(x, y)| (x - y).abs())
        .fold(0f32, f32::max))
}

fn main() -> Result<()> {
    let cpu = Device::Cpu;
    let metal = Device::new_metal(0)?;
    let (cos_c, sin_c) = freqs_cis(&cpu)?;
    let (cos_m, sin_m) = freqs_cis(&metal)?;
    let (cos_m, sin_m) = (cos_m.to_device(&cpu)?, sin_m.to_device(&cpu)?);

    println!("row (position)   max|Δcos|    max|Δsin|");
    for &p in &[
        0usize, 1024, 4096, 8000, 8100, 8191, 8192, 8193, 8300, 9000, 11605, 16384, 32767,
    ] {
        let dc = row_max_diff(&cos_c, &cos_m, p)?;
        let ds = row_max_diff(&sin_c, &sin_m, p)?;
        let flag = if dc.max(ds) > 1e-2 {
            "  <-- DIVERGED"
        } else {
            ""
        };
        println!("{p:>8}       {dc:.3e}    {ds:.3e}{flag}");
    }
    Ok(())
}
