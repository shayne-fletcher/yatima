//! Decode image artifacts to raw RGBA and step animations on a clock.
//!
//! One frame for a still; every frame with its hold time for an animated
//! GIF or WebP; `None` for a format this build does not render (the caller
//! decides what a placeholder looks like). Nothing here is a view type, so the
//! stepping is unit-tested natively and the same code paints in the native
//! GUI and in the browser.

/// An artifact decoded to raw RGBA — everything a view needs to make
/// textures. One frame for a still; many, each with its hold time, for an
/// animated GIF.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecodedImage {
    pub size: [usize; 2],
    pub frames: Vec<AnimationFrame>,
}

/// One frame of a decoded image: raw RGBA plus how long it holds before
/// the next (0 for a still).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AnimationFrame {
    pub rgba: Vec<u8>,
    pub delay_ms: u32,
}

/// Animations keep at most this many frames — a runaway file (the fetch
/// is already byte-capped upstream) degrades to a long-but-finite loop,
/// never an unbounded texture upload.
pub const MAX_GIF_FRAMES: usize = 128;

/// Browsers clamp near-zero frame delays to about this; a raw 0 would spin
/// a repaint loop for no visible motion.
const MIN_GIF_DELAY_MS: u32 = 20;

/// Decode PNG/JPEG/still WebP (one frame) or an animated GIF/WebP (every
/// frame, with delays) to RGBA. `None` is not an error: an unknown format
/// (SVG, …) is the caller's placeholder.
pub fn decode_rgba(bytes: &[u8]) -> Option<DecodedImage> {
    use image::AnimationDecoder;
    let cursor = std::io::Cursor::new(bytes);
    if bytes.starts_with(b"GIF8") {
        let decoder = image::codecs::gif::GifDecoder::new(cursor).ok()?;
        return animation(decoder.into_frames());
    }
    if bytes.len() >= 12 && &bytes[..4] == b"RIFF" && &bytes[8..12] == b"WEBP" {
        let decoder = image::codecs::webp::WebPDecoder::new(cursor).ok()?;
        if decoder.has_animation() {
            return animation(decoder.into_frames());
        }
        // A still WebP falls through to the one-frame path below.
    }
    let rgba = image::load_from_memory(bytes).ok()?.to_rgba8();
    let (w, h) = rgba.dimensions();
    Some(DecodedImage {
        size: [w as usize, h as usize],
        frames: vec![AnimationFrame {
            rgba: rgba.into_raw(),
            delay_ms: 0,
        }],
    })
}

/// Every frame of an animation, capped and delay-clamped; `None` when the
/// stream yields no decodable frame.
fn animation(frames: image::Frames<'_>) -> Option<DecodedImage> {
    let mut out = Vec::new();
    let mut size = [0usize; 2];
    for frame in frames.take(MAX_GIF_FRAMES) {
        let frame = frame.ok()?;
        let (numer, denom) = frame.delay().numer_denom_ms();
        let delay_ms = (numer / denom.max(1)).max(MIN_GIF_DELAY_MS);
        let buf = frame.into_buffer();
        size = [buf.width() as usize, buf.height() as usize];
        out.push(AnimationFrame {
            rgba: buf.into_raw(),
            delay_ms,
        });
    }
    if out.is_empty() {
        return None;
    }
    Some(DecodedImage { size, frames: out })
}

/// Which frame an animation shows at time `t` (seconds, any monotonic
/// clock), and how long until the next flip (`None` for a still — no
/// repaint needed). Pure, so the stepping is testable off-screen; the view
/// feeds it its clock and schedules a repaint for the flip.
pub fn frame_at(frames: &[AnimationFrame], t: f64) -> (usize, Option<f64>) {
    let total_ms: u64 = frames.iter().map(|f| f.delay_ms as u64).sum();
    if frames.len() < 2 || total_ms == 0 {
        return (0, None);
    }
    let mut into = ((t * 1000.0) as u64) % total_ms;
    for (i, frame) in frames.iter().enumerate() {
        let hold = frame.delay_ms as u64;
        if into < hold {
            return (i, Some((hold - into) as f64 / 1000.0));
        }
        into -= hold;
    }
    (0, Some(frames[0].delay_ms as f64 / 1000.0)) // unreachable by arithmetic
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::codecs::gif::{GifEncoder, Repeat};
    use image::{Delay, Frame, Rgba, RgbaImage};

    fn gif(frames: usize, delay_ms: u32) -> Vec<u8> {
        let mut bytes = Vec::new();
        {
            let mut enc = GifEncoder::new(&mut bytes);
            enc.set_repeat(Repeat::Infinite).unwrap();
            for i in 0..frames {
                enc.encode_frame(Frame::from_parts(
                    RgbaImage::from_pixel(1, 1, Rgba([i as u8, 0, 0, 255])),
                    0,
                    0,
                    Delay::from_numer_denom_ms(delay_ms, 1),
                ))
                .unwrap();
            }
        }
        bytes
    }

    #[test]
    fn gif_decodes_every_frame_with_clamped_delays() {
        let decoded = decode_rgba(&gif(3, 100)).expect("gif decodes");
        assert_eq!(decoded.frames.len(), 3);
        assert_eq!(decoded.size, [1, 1]);
        assert!(decoded.frames.iter().all(|f| f.delay_ms == 100));
        // A zero delay clamps up: no repaint spin.
        let fast = decode_rgba(&gif(2, 0)).expect("gif decodes");
        assert!(fast.frames.iter().all(|f| f.delay_ms == MIN_GIF_DELAY_MS));
    }

    /// A lossless still WebP from the crate's own encoder.
    fn webp_still(w: u32, h: u32, color: [u8; 4]) -> Vec<u8> {
        let mut bytes = Vec::new();
        image::codecs::webp::WebPEncoder::new_lossless(&mut bytes)
            .encode(
                RgbaImage::from_pixel(w, h, Rgba(color)).as_raw(),
                w,
                h,
                image::ExtendedColorType::Rgba8,
            )
            .unwrap();
        bytes
    }

    /// An animated WebP assembled by hand: the crate encodes stills only,
    /// so this wraps each still's `VP8L` chunk in an `ANMF` frame under a
    /// `VP8X`+`ANIM` header (the container is a plain RIFF chunk list).
    fn webp_animated(colors: &[[u8; 4]], delay_ms: u32) -> Vec<u8> {
        fn chunk(fourcc: &[u8; 4], payload: &[u8]) -> Vec<u8> {
            let mut c = fourcc.to_vec();
            c.extend((payload.len() as u32).to_le_bytes());
            c.extend(payload);
            if payload.len() % 2 == 1 {
                c.push(0);
            }
            c
        }
        fn u24(v: u32) -> [u8; 3] {
            [v as u8, (v >> 8) as u8, (v >> 16) as u8]
        }
        let mut body = Vec::new();
        // VP8X: animation + alpha flags, canvas 1×1.
        let mut vp8x = vec![0x02 | 0x10, 0, 0, 0];
        vp8x.extend(u24(0));
        vp8x.extend(u24(0));
        body.extend(chunk(b"VP8X", &vp8x));
        // ANIM: background colour, loop forever.
        body.extend(chunk(b"ANIM", &[0, 0, 0, 0, 0, 0]));
        for color in colors {
            let still = webp_still(1, 1, *color);
            // The still is RIFF(4) size(4) WEBP(4) then its VP8L chunk.
            let vp8l = &still[12..];
            let mut anmf = Vec::new();
            anmf.extend(u24(0)); // x/2
            anmf.extend(u24(0)); // y/2
            anmf.extend(u24(0)); // width-1
            anmf.extend(u24(0)); // height-1
            anmf.extend(u24(delay_ms));
            anmf.push(0); // blend + dispose flags
            anmf.extend(vp8l);
            body.extend(chunk(b"ANMF", &anmf));
        }
        let mut file = b"RIFF".to_vec();
        file.extend(((body.len() + 4) as u32).to_le_bytes());
        file.extend(b"WEBP");
        file.extend(body);
        file
    }

    #[test]
    fn webp_still_is_one_frame_and_animated_webp_is_every_frame() {
        let still = decode_rgba(&webp_still(2, 1, [7, 8, 9, 255])).expect("webp decodes");
        assert_eq!(still.size, [2, 1]);
        assert_eq!(still.frames.len(), 1);
        assert_eq!(still.frames[0].delay_ms, 0);
        assert_eq!(&still.frames[0].rgba[..4], &[7, 8, 9, 255]);

        let anim = decode_rgba(&webp_animated(
            &[[255, 0, 0, 255], [0, 255, 0, 255], [0, 0, 255, 255]],
            90,
        ))
        .expect("animated webp decodes");
        assert_eq!(anim.frames.len(), 3);
        assert!(anim.frames.iter().all(|f| f.delay_ms == 90));
        // The animation compositor may round a channel by one; the frame's
        // identity is its dominant channel.
        let px = &anim.frames[2].rgba[..4];
        assert!(
            px[2] >= 254 && px[0] <= 1 && px[1] <= 1 && px[3] == 255,
            "{px:?}"
        );
        assert_eq!(
            frame_at(&anim.frames, 0.1).0,
            1,
            "the stepper walks webp frames too"
        );
    }

    #[test]
    fn frame_cap_bounds_a_runaway_gif() {
        let decoded = decode_rgba(&gif(MAX_GIF_FRAMES + 40, 50)).expect("gif decodes");
        assert_eq!(decoded.frames.len(), MAX_GIF_FRAMES);
    }

    #[test]
    fn png_is_one_zero_delay_frame_and_unknown_is_none() {
        let mut png = Vec::new();
        image::DynamicImage::ImageRgba8(
            RgbaImage::from_raw(2, 1, vec![255, 0, 0, 255, 0, 255, 0, 255]).unwrap(),
        )
        .write_to(&mut std::io::Cursor::new(&mut png), image::ImageFormat::Png)
        .unwrap();
        let decoded = decode_rgba(&png).expect("png decodes");
        assert_eq!(decoded.size, [2, 1]);
        assert_eq!(decoded.frames.len(), 1);
        assert_eq!(decoded.frames[0].delay_ms, 0);
        assert_eq!(decoded.frames[0].rgba.len(), 8);
        assert!(decode_rgba(b"<svg xmlns='http://www.w3.org/2000/svg'/>").is_none());
        assert!(decode_rgba(b"").is_none());
    }

    #[test]
    fn stepper_loops_and_its_wait_lands_inside_the_next_hold() {
        let d = decode_rgba(&gif(2, 100)).unwrap().frames;
        assert_eq!(frame_at(&d, 0.05).0, 0);
        assert_eq!(frame_at(&d, 0.15).0, 1);
        assert_eq!(frame_at(&d, 0.25).0, 0, "wraps");
        // The wait returned always ends exactly at the next frame boundary.
        for t in [0.0, 0.05, 0.099, 0.1, 0.15, 0.199, 0.2, 1.05] {
            let (i, wait) = frame_at(&d, t);
            let wait = wait.expect("animations schedule repaints");
            assert!(wait > 0.0 && wait <= 0.1, "t={t}: wait {wait}");
            let (next, _) = frame_at(&d, t + wait + 1e-6);
            assert_eq!(next, (i + 1) % 2, "t={t}: the wait reaches the next frame");
        }
        // Stills and zero-total animations never ask for a repaint.
        let still = vec![AnimationFrame {
            rgba: vec![0; 4],
            delay_ms: 0,
        }];
        assert_eq!(frame_at(&still, 123.0), (0, None));
        let zero = vec![
            AnimationFrame {
                rgba: vec![0; 4],
                delay_ms: 0,
            },
            AnimationFrame {
                rgba: vec![0; 4],
                delay_ms: 0,
            },
        ];
        assert_eq!(frame_at(&zero, 1.0), (0, None));
    }
}
