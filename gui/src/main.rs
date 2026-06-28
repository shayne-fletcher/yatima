//! `yatima-gui` — the GPU frontend's first slice: a "hello, engine" window.
//!
//! An egui/eframe app (wgpu → Metal on macOS) that loads a local model and
//! streams one chat turn at a time. It is deliberately minimal — input box,
//! scrolling transcript, live token streaming — to prove the toolchain, egui's
//! native text shaping, and that a *second* frontend can drive the engine. No
//! docking, reasoning split, markdown, math, cancel, or live-tweak yet; those
//! are later slices (see `plans/text-rendering.plan.md`).
//!
//! The engine is `!Send`, so — exactly as `yatima-tui`'s actor does — it is
//! created and owned on a dedicated background thread (`runner`), which serves
//! prompts over a channel and streams fragments back. Rendering is immediate
//! mode: `update` is a pure projection of the accumulated state, redrawn each
//! frame (the same discipline as the TUI's `ui(frame, &App)`).

use std::path::PathBuf;
use std::sync::mpsc::{Receiver, Sender};
use std::thread;

use anyhow::Result;
use clap::Parser;
use eframe::egui;

use yatima_lib::{
    device, resolve_format, Cancel, Channel, ChatFormat, ChatSession, Engine, GenOpts,
    ModelProfile, ModelSource, ReasoningSplitter, Sampling,
};

/// Interactive GUI chat over a local model.
#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// A built-in model profile (e.g. `qwq`, `deepseek-r1`).
    #[arg(long)]
    profile: Option<String>,
    /// Explicit model directory.
    #[arg(long)]
    model: Option<PathBuf>,
    /// Repository id, resolved under the models root.
    #[arg(long)]
    repo: Option<String>,
    /// Override the models root (else $YATIMA_MODELS_DIR / XDG cache).
    #[arg(long)]
    models_dir: Option<PathBuf>,
    /// With `--repo`, fetch this single GGUF file (quantized).
    #[arg(long)]
    gguf: Option<String>,
    /// Optional system instruction (applies for the whole session).
    #[arg(long)]
    system: Option<String>,
    #[arg(long, default_value_t = 1024)]
    max_tokens: usize,
    #[arg(long, default_value_t = 0.0)]
    temperature: f64,
    /// Nucleus (top-p) sampling cutoff; omit for the full distribution.
    #[arg(long)]
    top_p: Option<f64>,
    #[arg(long, default_value_t = 0)]
    seed: u64,
    /// Force CPU instead of the GPU.
    #[arg(long)]
    cpu: bool,
    /// Don't auto-fetch a missing model; error instead.
    #[arg(long)]
    offline: bool,
}

/// Everything the background runner needs to load a model — all `Send`, so it
/// crosses into the thread; the `!Send` `Engine` is then *created* inside it.
struct RunConfig {
    dir: PathBuf,
    cpu: bool,
    opts: GenOpts,
    format: Option<ChatFormat>,
    system: Option<String>,
    label: String,
}

fn resolve(args: &Args) -> Result<RunConfig> {
    let profile = match &args.profile {
        Some(name) => Some(ModelProfile::builtin(name).ok_or_else(|| {
            anyhow::anyhow!(
                "unknown profile {name:?}; built-ins: {:?}",
                ModelProfile::BUILTIN_NAMES
            )
        })?),
        None => None,
    };

    let (dir, label) = match &profile {
        Some(p) => (p.to_source(args.offline)?.resolve()?, p.name.clone()),
        None => {
            let dir = ModelSource::from_args(
                args.model.clone(),
                args.repo.clone(),
                args.models_dir.clone(),
                args.offline,
                args.gguf.clone(),
            )?
            .resolve()?;
            (dir.clone(), dir.display().to_string())
        }
    };

    let base = GenOpts {
        max_tokens: args.max_tokens,
        sampling: Sampling::nucleus(args.temperature, args.top_p, args.seed),
        ..Default::default()
    };
    let opts = match &profile {
        Some(p) => p.apply_gen_overrides(base),
        None => base,
    };
    let format = profile.as_ref().and_then(ModelProfile::format);

    Ok(RunConfig {
        dir,
        cpu: args.cpu,
        opts,
        format,
        system: args.system.clone(),
        label,
    })
}

/// What's running — surfaced in the status rail beside the receded mark. Built
/// in the runner (where the engine and config live) and shipped to the UI on
/// `Ready`. All fields are pre-formatted strings to keep the UI a pure view.
#[derive(Clone)]
struct ModelInfo {
    label: String,
    arch: String,
    device: String,
    format: String,
    sampling: String,
    max_tokens: usize,
    /// The model's context window in tokens (the meter's denominator), if known.
    context_length: Option<usize>,
}

/// Runner → UI events (over `std::sync::mpsc`; the UI drains them each frame).
enum Ev {
    /// Model loaded and ready to chat (carries what's running).
    Ready(ModelInfo),
    /// A streamed slice of the surfaced answer.
    Answer(String),
    /// A streamed slice of the chain-of-thought (shown only when toggled on).
    Reasoning(String),
    /// Context tokens used (the most recent rendered prompt) — for the meter.
    Context(usize),
    /// An image artifact (PNG bytes) produced by the turn — the artifact plane,
    /// spiked here with a matplotlib chart from `/plot`.
    Image(Vec<u8>),
    /// The current turn finished.
    Done,
    /// The current turn failed.
    Error(String),
    /// The model could not be loaded; the session never starts.
    Fatal(String),
}

/// The background engine thread: load the model, then serve prompts one turn at
/// a time, streaming fragments back. Owns the `!Send` `Engine` for its whole
/// life (created here, never crossed over a thread boundary).
fn runner(cfg: RunConfig, req_rx: Receiver<String>, ev_tx: Sender<Ev>, ctx: egui::Context) {
    let dev = match device(cfg.cpu) {
        Ok(d) => d,
        Err(e) => return fatal(&ev_tx, &ctx, e),
    };
    let mut engine = match Engine::load(&cfg.dir, dev) {
        Ok(e) => e,
        Err(e) => return fatal(&ev_tx, &ctx, e),
    };
    let (format, _mismatch) = resolve_format(engine.arch(), cfg.format);
    let pre_seeds = format.pre_seeds_reasoning();

    // Snapshot what's running for the status rail before `cfg.opts` is moved.
    let info = ModelInfo {
        label: cfg.label.clone(),
        arch: format!("{:?}", engine.arch()),
        device: if cfg.cpu { "cpu" } else { "gpu" }.to_string(),
        format: format!("{format:?}"),
        sampling: match cfg.opts.sampling {
            Sampling::Greedy => "greedy".to_string(),
            Sampling::Sample {
                temperature,
                top_p,
                seed,
            } => match top_p {
                Some(p) => format!("temp {temperature:.2} · top-p {p:.2} · seed {seed}"),
                None => format!("temp {temperature:.2} · seed {seed}"),
            },
        },
        max_tokens: cfg.opts.max_tokens,
        context_length: engine.context_length(),
    };

    let template = format.template();
    let mut session = ChatSession::new(&mut engine, template).with_opts(cfg.opts);
    if let Some(system) = cfg.system {
        session = session.with_system(system);
    }
    let _ = ev_tx.send(Ev::Ready(info));
    ctx.request_repaint();

    while let Ok(prompt) = req_rx.recv() {
        // Spike: `/plot` exercises the artifact plane without the model — shell
        // out to matplotlib, send the PNG back as an Image artifact.
        if prompt.trim() == "/plot" {
            let _ = ev_tx.send(match render_demo_chart() {
                Ok(bytes) => Ev::Image(bytes),
                Err(e) => Ev::Error(e.to_string()),
            });
            let _ = ev_tx.send(Ev::Done);
            ctx.request_repaint();
            continue;
        }

        let cancel = Cancel::new();
        // Classify the stream into reasoning vs answer as it arrives, so the UI
        // can fold the chain-of-thought away unless the user toggles it on.
        let mut splitter = if pre_seeds {
            ReasoningSplitter::seeded()
        } else {
            ReasoningSplitter::new()
        };
        let to_ev = |ch: Channel, text: &str| match ch {
            Channel::Reasoning => Ev::Reasoning(text.to_string()),
            Channel::Answer => Ev::Answer(text.to_string()),
        };
        let result = {
            let tx = ev_tx.clone();
            let ctx = ctx.clone();
            let mut on_token = |frag: &str| {
                splitter.push(frag, |ch, text| {
                    let _ = tx.send(to_ev(ch, text));
                });
                ctx.request_repaint();
            };
            session
                .turn_streaming_cancellable(&prompt, &cancel, &mut on_token)
                .map(|answer| answer.to_string())
        };
        // Flush any tail buffered against a straddling marker.
        splitter.finish(|ch, text| {
            let _ = ev_tx.send(to_ev(ch, text));
        });
        if let Some(used) = session.last_prompt_tokens() {
            let _ = ev_tx.send(Ev::Context(used));
        }
        let _ = ev_tx.send(match result {
            Ok(_) => Ev::Done,
            Err(e) => Ev::Error(e.to_string()),
        });
        ctx.request_repaint();
    }
}

fn fatal(ev_tx: &Sender<Ev>, ctx: &egui::Context, e: impl std::fmt::Display) {
    let _ = ev_tx.send(Ev::Fatal(e.to_string()));
    ctx.request_repaint();
}

/// The Python interpreter to use — the seam that, in a real `RunPython` tool,
/// *is* the capability. Prefers `$YATIMA_PYTHON`, then a project `.venv` (the
/// pinned environment the plan calls for), then system `python3`.
fn python_interpreter() -> std::path::PathBuf {
    if let Ok(p) = std::env::var("YATIMA_PYTHON") {
        return p.into();
    }
    if let Ok(cwd) = std::env::current_dir() {
        let venv = cwd.join(".venv/bin/python3");
        if venv.exists() {
            return venv;
        }
    }
    "python3".into()
}

/// Spike: produce a matplotlib chart as PNG bytes by shelling out to Python.
/// A stand-in for a real capability-scoped `RunPython` tool — enough to prove the
/// artifact → egui-image path. Needs `numpy` + `matplotlib` (see the `.venv`).
fn render_demo_chart() -> Result<Vec<u8>> {
    let out = std::env::temp_dir().join("yatima-gui-plot.png");
    let code = r#"import sys, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
x = np.linspace(0, 2 * np.pi, 400)
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(x, np.sin(x), label="sin")
ax.plot(x, np.cos(x), label="cos")
ax.set_title("yatima-gui artifact spike")
ax.legend()
ax.grid(True, alpha=0.3)
fig.savefig(sys.argv[1], dpi=120, bbox_inches="tight")
"#;
    let python = python_interpreter();
    let output = std::process::Command::new(&python)
        .arg("-c")
        .arg(code)
        .arg(&out)
        .output()
        .map_err(|e| anyhow::anyhow!("could not run {}: {e}", python.display()))?;
    if !output.status.success() {
        anyhow::bail!(
            "{} (numpy/matplotlib) failed: {}",
            python.display(),
            String::from_utf8_lossy(&output.stderr).trim()
        );
    }
    Ok(std::fs::read(&out)?)
}

/// The `/help` listing.
const HELP_TEXT: &str = "\
commands
  /plot      render a demo matplotlib chart (artifact)
  /stats     toggle the system panel — state + controls   (alias /control)
  /cls       clear the screen                              (Ctrl+L)
  /about     about yatima
  /help      this message
  /quit      exit

reasoning is hidden by default — turn on \"show reasoning\" in /stats.";

/// A rendered transcript entry (the UI mirror; the runner's session is truth).
enum Msg {
    User(String),
    Assistant {
        answer: String,
        /// The chain-of-thought, kept so `/stats`' reasoning toggle can reveal
        /// it after the fact. Never re-enters the prompt (the lib drops it).
        reasoning: Option<String>,
    },
    /// A decoded image artifact, uploaded as a GPU texture.
    Image(egui::TextureHandle),
    /// An app message (e.g. `/help`, `/about`) — not from the model.
    Note(String),
    Error(String),
}

/// Where the session is in its lifecycle (drives the status line / input gating).
enum Status {
    Loading,
    Ready(String),
    Failed(String),
}

struct GuiApp {
    req_tx: Sender<String>,
    ev_rx: Receiver<Ev>,
    /// A handle to the egui context, for uploading image artifacts as textures
    /// off the render path (in `drain_events`).
    ctx: egui::Context,
    /// When the splash animation began (engine time), so the sigil draws on
    /// once. Reset to `None` on submit so it replays on a return to the splash.
    splash_anim_start: Option<f32>,
    /// What's running, for the status rail (set once the model is ready).
    info: Option<ModelInfo>,
    /// Whether the `/stats` panel (state + controls) is open.
    show_stats: bool,
    input: String,
    transcript: Vec<Msg>,
    /// The answer currently streaming in, if a turn is in flight.
    streaming: Option<String>,
    /// The chain-of-thought streaming in alongside it (this turn).
    streaming_reasoning: String,
    /// Whether to surface reasoning. Off by default — the answer is what matters;
    /// the chain-of-thought is opt-in via `/stats`.
    show_reasoning: bool,
    status: Status,
    /// Opacity applied to image artifacts and the logo splash (live slider).
    opacity: f32,
    /// Set when the model becomes ready, so the next frame hands focus to the
    /// input — the box activates the moment loading finishes.
    focus_input: bool,
    /// Engine time until which the avatar registers surprise (e.g. an artifact
    /// just popped in). `0.0` = not surprised.
    surprise_until: f32,
    /// `/help` overlay: whether it's showing, and when its drop began.
    help_open: bool,
    help_start: f32,
    /// Per-turn telemetry for the live status readout: engine time of the first
    /// token, and the count of streamed tokens (reasoning + answer).
    turn_start: Option<f32>,
    gen_tokens: usize,
    /// Context tokens used (last rendered prompt), for the meter / ticker.
    context_used: Option<usize>,
    /// Whether the idle status ticker scrolls (off by default — keeps the bar
    /// calm; opt in via `/stats`).
    show_ticker: bool,
    /// Engine time a `/do-a-barrel-roll` began (the avatar spins once). Secret.
    roll_start: Option<f32>,
    /// When `strawberry fields` mode began (reality dissolves into drifting
    /// particles), or `None` when off. Esc recovers reality. Secret.
    strawberry_start: Option<f32>,
}

impl GuiApp {
    fn new(cc: &eframe::CreationContext<'_>, cfg: RunConfig) -> GuiApp {
        let (req_tx, req_rx) = std::sync::mpsc::channel::<String>();
        let (ev_tx, ev_rx) = std::sync::mpsc::channel::<Ev>();
        let ctx = cc.egui_ctx.clone();
        let runner_ctx = ctx.clone();
        thread::spawn(move || runner(cfg, req_rx, ev_tx, runner_ctx));
        GuiApp {
            req_tx,
            ev_rx,
            ctx,
            splash_anim_start: None,
            info: None,
            show_stats: false,
            input: String::new(),
            transcript: Vec::new(),
            streaming: None,
            streaming_reasoning: String::new(),
            show_reasoning: false,
            status: Status::Loading,
            opacity: 0.85,
            focus_input: false,
            surprise_until: 0.0,
            help_open: false,
            help_start: 0.0,
            turn_start: None,
            gen_tokens: 0,
            context_used: None,
            show_ticker: false,
            roll_start: None,
            strawberry_start: None,
        }
    }

    /// The tint applied to images at the current opacity.
    fn image_tint(&self) -> egui::Color32 {
        egui::Color32::from_white_alpha((self.opacity.clamp(0.0, 1.0) * 255.0).round() as u8)
    }

    /// The welcome splash shown while the transcript is empty: the sigil drawn
    /// on stroke by stroke as built-in vector graphics (no raster), shimmering
    /// through the aurora palette, then the wordmark and a status caption. After
    /// it plays and holds, the whole mark *recedes* — shrinking and docking to
    /// the top-left, ceding the stage to the prompt. A failed load shows the
    /// same mark, static, dim, and centered (status bar has the why).
    fn show_splash(&mut self, ui: &mut egui::Ui) {
        let t = ui.input(|i| i.time) as f32;
        let start = *self.splash_anim_start.get_or_insert(t);
        let failed = matches!(self.status, Status::Failed(_));

        // The sigil shimmers through the aurora ramp; a failed load is dim and
        // fully drawn (no animation, no shimmer, no recede).
        let (color, animate) = if failed {
            (egui::Color32::from_gray(90), false)
        } else {
            let a = aurora_at(t * 0.35);
            (egui::Color32::from_rgb(a.r(), a.g(), a.b()), true)
        };
        let elapsed = if animate { t - start } else { 99.0 };
        let recede = if failed {
            0.0
        } else {
            smoothstep(time_ease(elapsed, 4.5, 6.2))
        };

        let panel = ui.max_rect();
        let painter = ui.painter().clone();
        let lerp = |a: f32, b: f32, s: f32| a + (b - a) * s;

        // Size the big (centered) mark so the whole composition fits the panel
        // with a margin. Below the center it overhangs by ~1.513 r + 32 (the
        // wordmark at 0.27 r cap height, then the caption); above, by r. Width
        // needs ~2.05 r (the sigil diameter, a touch wider than the wordmark).
        let margin = 16.0;
        let r_by_w = (panel.width() - 2.0 * margin) / 2.05;
        let r_by_h = (panel.height() - 32.0 - 2.0 * margin) / 2.513;
        let big_r = r_by_w.min(r_by_h).clamp(1.0, 160.0);
        let small_r = big_r * 0.30;
        let r = lerp(big_r, small_r, recede);
        let stroke_w = (r * 0.015).max(1.0);

        // Big: vertically centered (offset for the below-center overhang). Small:
        // docked to the top-left corner.
        let big_c = egui::pos2(
            panel.center().x,
            panel.center().y - (0.513 * big_r + 32.0) / 2.0,
        );
        let small_c = egui::pos2(panel.left() + 18.0 + small_r, panel.top() + 18.0 + small_r);
        let center = egui::pos2(
            lerp(big_c.x, small_c.x, recede),
            lerp(big_c.y, small_c.y, recede),
        );
        draw_sigil(&painter, center, r, elapsed, color, stroke_w);

        // Wordmark below the mark, scaled with it: centered when large, then
        // left-aligned under the mark as it recedes.
        let cap_h = r * 0.27;
        let wm_y = center.y + r + cap_h * 0.9;
        let wm_x = lerp(panel.center().x, center.x - r, recede);
        let align = lerp(0.5, 0.0, recede);
        draw_wordmark(
            &painter,
            egui::pos2(wm_x, wm_y),
            cap_h,
            color,
            stroke_w * 0.9,
            elapsed,
            align,
        );

        // Status caption, centered under the wordmark; it fades as we recede.
        let cap_fade = 1.0 - recede;
        let caption = match self.status {
            Status::Loading => "loading\u{2026}",
            Status::Failed(_) => "",
            _ => "ready",
        };
        if !caption.is_empty() && cap_fade > 0.01 {
            painter.text(
                egui::pos2(panel.center().x, wm_y + cap_h + 16.0),
                egui::Align2::CENTER_TOP,
                caption,
                egui::FontId::proportional(14.0),
                with_alpha(color, (170.0 * cap_fade) as u8),
            );
        }

        // System-status rail: once the mark has receded to the corner, the freed
        // space to its right reports what's running. Fades in with the recede.
        if recede > 0.01 {
            if let Some(info) = &self.info {
                let x = small_c.x + small_r + 28.0;
                let mut y = panel.top() + 20.0;
                let val = with_alpha(color, (recede * 230.0) as u8);
                let key = with_alpha(color, (recede * 110.0) as u8);
                let font = egui::FontId::monospace(12.0);
                let rows: [(&str, String); 6] = [
                    ("model", info.label.clone()),
                    ("arch", info.arch.clone()),
                    ("device", info.device.clone()),
                    ("format", info.format.clone()),
                    ("sampling", info.sampling.clone()),
                    ("max tokens", info.max_tokens.to_string()),
                ];
                for (k, v) in rows {
                    painter.text(
                        egui::pos2(x, y),
                        egui::Align2::LEFT_TOP,
                        k,
                        font.clone(),
                        key,
                    );
                    painter.text(
                        egui::pos2(x + 92.0, y),
                        egui::Align2::LEFT_TOP,
                        v,
                        font.clone(),
                        val,
                    );
                    y += 18.0;
                }
            }
        }

        if animate {
            ui.ctx().request_repaint(); // drive the draw-on, recede, and shimmer
        }
    }

    /// The `/help` overlay: a dimmed scrim over which the help lines drop in
    /// from the top one at a time and bounce-settle near the bottom — Tetris
    /// style. Dismissed by Esc or a click.
    fn draw_help(&mut self, ui: &mut egui::Ui, now: f32) {
        let screen = ui.ctx().content_rect();
        let p = ui.ctx().layer_painter(egui::LayerId::new(
            egui::Order::Foreground,
            egui::Id::new("help_overlay"),
        ));
        p.rect_filled(screen, 0.0, egui::Color32::from_black_alpha(180));

        let acc = aurora_at(now * 0.35);
        let color = egui::Color32::from_rgb(acc.r(), acc.g(), acc.b());
        let font = egui::FontId::monospace(14.0);
        let line_h = 22.0;

        let lines: Vec<&str> = HELP_TEXT.lines().collect();
        let block_h = lines.len() as f32 * line_h;
        let rest_top = screen.bottom() - 96.0 - block_h;
        let x = screen.center().x - 235.0;
        let y_from = screen.top() - line_h;

        for (i, line) in lines.iter().enumerate() {
            let t0 = self.help_start + i as f32 * 0.10;
            if now < t0 {
                continue; // hasn't dropped yet
            }
            let prog = ((now - t0) / 0.6).clamp(0.0, 1.0);
            let e = ease_out_bounce(prog);
            let y = y_from + (rest_top + i as f32 * line_h - y_from) * e;
            p.text(
                egui::pos2(x, y),
                egui::Align2::LEFT_TOP,
                line,
                font.clone(),
                color,
            );
        }

        // A dismiss hint, faded in once the stack has landed.
        let settled = self.help_start + lines.len() as f32 * 0.10 + 0.6;
        let hint = ((now - settled) / 0.4).clamp(0.0, 1.0);
        if hint > 0.0 {
            p.text(
                egui::pos2(screen.center().x, rest_top + block_h + 16.0),
                egui::Align2::CENTER_TOP,
                "esc / click to close",
                egui::FontId::proportional(12.0),
                with_alpha(color, (hint * 150.0) as u8),
            );
        }

        if ui.input(|i| i.key_pressed(egui::Key::Escape) || i.pointer.any_click()) {
            self.help_open = false;
        }
        ui.ctx().request_repaint();
    }

    /// 🍓 Strawberry fields: reality dissolves into a drift of twinkling
    /// particles. Esc recovers reality. A foreground overlay; the UI is still
    /// there, just behind the haze.
    fn draw_strawberry(&mut self, ui: &mut egui::Ui, now: f32) {
        let Some(start) = self.strawberry_start else {
            return;
        };
        let screen = ui.ctx().content_rect();
        let p = ui.ctx().layer_painter(egui::LayerId::new(
            egui::Order::Foreground,
            egui::Id::new("strawberry"),
        ));
        // Dissolve in over ~1.2s: the haze deepens and the particles bloom.
        let dissolve = smoothstep(time_ease(now, start, start + 1.2));
        p.rect_filled(
            screen,
            0.0,
            egui::Color32::from_black_alpha((dissolve * 225.0) as u8),
        );

        let count = 170;
        for i in 0..count {
            let fi = i as f32;
            let ph = hash01(fi * 3.1) * std::f32::consts::TAU;
            let spd = 0.15 + hash01(fi * 0.7) * 0.5;
            let bx = hash01(fi * 1.3);
            let by = hash01(fi * 2.7 + 1.1);
            let x = (bx + 0.07 * (now * spd + ph).sin()).rem_euclid(1.0);
            // a slow upward drift, wrapping — the field is forever
            let y = (by - now * 0.02 * spd + 0.05 * (now * spd * 1.3 + ph).cos()).rem_euclid(1.0);
            let pos = egui::pos2(
                screen.left() + x * screen.width(),
                screen.top() + y * screen.height(),
            );
            let r = (1.5 + hash01(fi * 5.0) * 3.5) * dissolve;
            let twinkle = 0.45 + 0.55 * (now * 1.3 + ph).sin();
            let col = aurora_at(now * 0.25 + fi * 0.11);
            p.circle_filled(pos, r, with_alpha(col, (dissolve * twinkle * 210.0) as u8));
        }

        // A faint way back, once the haze has settled.
        let hint = smoothstep(time_ease(now, start + 1.4, start + 2.2));
        if hint > 0.0 {
            p.text(
                egui::pos2(screen.center().x, screen.bottom() - 40.0),
                egui::Align2::CENTER_BOTTOM,
                "esc to return",
                egui::FontId::proportional(12.0),
                egui::Color32::from_white_alpha((hint * 110.0) as u8),
            );
        }

        if ui.input(|i| i.key_pressed(egui::Key::Escape)) {
            self.strawberry_start = None; // …living is easy with eyes closed
        }
        ui.ctx().request_repaint();
    }

    fn in_flight(&self) -> bool {
        self.streaming.is_some()
    }

    /// The idle ticker content: uptime and context usage.
    fn ticker_text(&self, now: f32) -> String {
        let mut parts = vec![format!("uptime {}", fmt_uptime(now))];
        let cap = self.info.as_ref().and_then(|i| i.context_length);
        match (self.context_used, cap) {
            (Some(u), Some(c)) => {
                parts.push(format!(
                    "context {}/{} ({}%)",
                    k(u),
                    k(c),
                    u * 100 / c.max(1)
                ));
            }
            (None, Some(c)) => parts.push(format!("context –/{}", k(c))),
            (Some(u), None) => parts.push(format!("context {}", k(u))),
            _ => {}
        }
        parts.join("      ·      ")
    }

    /// Clear the screen (Ctrl+L / `/cls`): drop the transcript and any in-flight
    /// stream, dismiss the help, and replay the splash. The session/model stays.
    fn clear(&mut self) {
        self.transcript.clear();
        self.streaming = None;
        self.streaming_reasoning.clear();
        self.help_open = false;
        self.splash_anim_start = None;
    }

    /// Fold runner events into the UI mirror. `now` is the engine clock, used to
    /// stamp transient reactions (e.g. the avatar's surprise).
    fn drain_events(&mut self, now: f32) {
        while let Ok(ev) = self.ev_rx.try_recv() {
            match ev {
                Ev::Ready(info) => {
                    self.status = Status::Ready(info.label.clone());
                    self.info = Some(info);
                    self.focus_input = true; // activate the input on transition
                }
                Ev::Answer(text) => {
                    if self.turn_start.is_none() {
                        self.turn_start = Some(now);
                    }
                    self.gen_tokens += 1;
                    if let Some(buf) = self.streaming.as_mut() {
                        buf.push_str(&text);
                    }
                }
                Ev::Reasoning(text) => {
                    if self.turn_start.is_none() {
                        self.turn_start = Some(now);
                    }
                    self.gen_tokens += 1;
                    self.streaming_reasoning.push_str(&text);
                }
                Ev::Context(used) => self.context_used = Some(used),
                Ev::Image(bytes) => match decode_texture(&self.ctx, &bytes) {
                    Ok(tex) => {
                        self.transcript.push(Msg::Image(tex));
                        self.surprise_until = now + 1.4; // an artifact! oh!
                    }
                    Err(e) => self
                        .transcript
                        .push(Msg::Error(format!("image decode: {e}"))),
                },
                Ev::Done => {
                    // Drop the streaming buffers; only commit a reply if the
                    // answer carried text (a `/plot` turn streams none).
                    let reasoning = std::mem::take(&mut self.streaming_reasoning);
                    if let Some(buf) = self.streaming.take() {
                        if !buf.trim().is_empty() {
                            let reasoning = (!reasoning.trim().is_empty())
                                .then(|| reasoning.trim().to_string());
                            self.transcript.push(Msg::Assistant {
                                answer: buf,
                                reasoning,
                            });
                        }
                    }
                }
                Ev::Error(message) => {
                    self.streaming = None;
                    self.streaming_reasoning.clear();
                    self.transcript.push(Msg::Error(message));
                }
                Ev::Fatal(message) => {
                    self.streaming = None;
                    self.streaming_reasoning.clear();
                    self.status = Status::Failed(message);
                }
            }
        }
    }

    /// Submit the current input as a turn — unless empty, not ready, or a turn
    /// is already in flight (single-in-flight, as in the TUI).
    fn submit(&mut self, now: f32) {
        let prompt = self.input.trim().to_string();
        self.help_open = false; // any submit dismisses the help overlay
        if prompt == "/quit" {
            self.ctx.send_viewport_cmd(egui::ViewportCommand::Close);
            return;
        }
        if prompt == "/stats" || prompt == "/control" {
            self.show_stats = !self.show_stats;
            self.input.clear();
            return;
        }
        if prompt == "/help" {
            self.help_open = true; // drop the help in, Tetris-style
            self.help_start = now;
            self.input.clear();
            return;
        }
        if prompt == "/cls" {
            self.clear();
            self.input.clear();
            return;
        }
        if prompt == "/do-a-barrel-roll" {
            self.roll_start = Some(now); // 🦊
            self.input.clear();
            return;
        }
        if prompt == "/strawberry-fields" {
            self.strawberry_start = Some(now); // 🍓 let me take you down…
            self.input.clear();
            return;
        }
        if prompt == "/about" {
            let about = match &self.info {
                Some(i) => format!(
                    "yatima — a local-LLM runtime; this is her GPU frontend \
                     (egui · wgpu/Metal).\nrunning {} · {} · {}.",
                    i.label, i.arch, i.device
                ),
                None => "yatima — a local-LLM runtime; this is her GPU frontend \
                         (egui · wgpu/Metal)."
                    .to_string(),
            };
            self.transcript.push(Msg::Note(about));
            self.input.clear();
            return;
        }
        if prompt.is_empty() || self.in_flight() || !matches!(self.status, Status::Ready(_)) {
            return;
        }
        self.transcript.push(Msg::User(prompt.clone()));
        self.streaming = Some(String::new());
        self.streaming_reasoning.clear();
        self.turn_start = None;
        self.gen_tokens = 0;
        let _ = self.req_tx.send(prompt);
        self.input.clear();
        self.splash_anim_start = None; // replay the draw-on if we return to it
    }
}

impl eframe::App for GuiApp {
    // eframe 0.35: the app draws into a root `Ui` (panels operate on a `&mut Ui`,
    // and `TopBottomPanel`/`SidePanel` are unified into `egui::Panel`).
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let now = ui.input(|i| i.time) as f32;
        self.drain_events(now);

        // Ctrl+L clears the screen, emacs-style (same as `/cls`).
        if ui.input(|i| i.modifiers.ctrl && i.key_pressed(egui::Key::L)) {
            self.clear();
        }

        egui::Panel::top("status").show(ui, |ui| {
            let t = ui.input(|i| i.time) as f32;
            let acc = aurora_at(t * 0.35);
            let face_col = egui::Color32::from_rgb(acc.r(), acc.g(), acc.b());
            // yatima's mood follows her state: surprised when an artifact just
            // landed, asleep while loading, sad on failure; mid-turn she thinks
            // hard while reasoning and talks once the answer flows; else calm.
            let answering = self.streaming.as_deref().is_some_and(|s| !s.is_empty());
            let expr = if t < self.surprise_until {
                Face::Surprised
            } else if matches!(self.status, Status::Loading) {
                Face::Sleeping
            } else if matches!(self.status, Status::Failed(_)) {
                Face::Sad
            } else if self.in_flight() {
                if answering {
                    Face::Talking
                } else {
                    // A brief thinking-hard burst at the start of reasoning, then
                    // settle to calm — the strain shouldn't hold for a long span.
                    let bursting = self.turn_start.is_none_or(|s| now - s < 1.3);
                    if bursting {
                        Face::Thinking
                    } else {
                        Face::Idle
                    }
                }
            } else {
                Face::Idle
            };
            // Identity at rest; a live readout during a turn — phase + elapsed
            // while thinking, then tokens + tok/s + elapsed once answering.
            let (text, color) = match &self.status {
                Status::Loading => ("loading model…".to_string(), egui::Color32::GRAY),
                Status::Failed(msg) => (format!("failed: {msg}"), egui::Color32::LIGHT_RED),
                Status::Ready(label) => {
                    let base = format!("yatima · {label}");
                    let text = if self.in_flight() {
                        let elapsed = self.turn_start.map(|s| now - s).unwrap_or(0.0);
                        let clock = fmt_clock(elapsed);
                        let answering = self.streaming.as_deref().is_some_and(|s| !s.is_empty());
                        if answering {
                            let tps = if elapsed > 0.1 {
                                self.gen_tokens as f32 / elapsed
                            } else {
                                0.0
                            };
                            format!(
                                "{base} · {} tok · {tps:.0} tok/s · {clock}",
                                self.gen_tokens
                            )
                        } else {
                            format!("{base} · thinking · {clock}")
                        }
                    } else {
                        base
                    };
                    (text, egui::Color32::LIGHT_GREEN)
                }
            };
            ui.horizontal(|ui| {
                let (frect, _) =
                    ui.allocate_exact_size(egui::vec2(30.0, 26.0), egui::Sense::hover());
                // Barrel roll: one full turn over ~0.8s when triggered.
                let roll = self
                    .roll_start
                    .map(|s| {
                        let p = (now - s) / 0.8;
                        if p < 1.0 {
                            p * std::f32::consts::TAU
                        } else {
                            0.0
                        }
                    })
                    .unwrap_or(0.0);
                draw_face(
                    &ui.painter_at(frect),
                    frect,
                    expr,
                    t,
                    face_col,
                    warp_at(t, 14.0),       // horizontal-axis teleport
                    warp_at(t + 9.0, 19.0), // vertical-axis teleport (offset cadence)
                    roll,
                );
                ui.colored_label(color, text);
                // An idle, opt-in status ticker scrolls in the space between the
                // identity and /stats — uptime and context usage, drifting by.
                let idle = matches!(self.status, Status::Ready(_)) && !self.in_flight();
                if self.show_ticker && idle {
                    let avail = ui.available_width() - 56.0; // leave room for /stats
                    if avail > 70.0 {
                        let (trect, _) = ui.allocate_exact_size(
                            egui::vec2(avail, ui.available_height()),
                            egui::Sense::hover(),
                        );
                        let content = self.ticker_text(t);
                        draw_ticker(
                            &ui.painter_at(trect),
                            trect,
                            &content,
                            t,
                            with_alpha(face_col, 90),
                        );
                    }
                }
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    ui.weak("/stats");
                });
            });
            ui.ctx().request_repaint(); // keep the avatar breathing/blinking
        });

        // `/stats` (alias `/control`): a right rail of state + controls, usable
        // during a chat (the splash has its own receding rail).
        if self.show_stats {
            egui::Panel::right("stats").show(ui, |ui| {
                ui.add_space(8.0);
                ui.horizontal(|ui| {
                    ui.label(egui::RichText::new("system").strong());
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if ui.button("✕").on_hover_text("close (/stats)").clicked() {
                            self.show_stats = false;
                        }
                    });
                });
                ui.add_space(4.0);
                match &self.info {
                    Some(info) => {
                        egui::Grid::new("stats_grid")
                            .num_columns(2)
                            .spacing([12.0, 4.0])
                            .show(ui, |ui| {
                                let max_tokens = info.max_tokens.to_string();
                                let ctx = match (self.context_used, info.context_length) {
                                    (Some(u), Some(c)) => {
                                        format!("{}/{} ({}%)", k(u), k(c), u * 100 / c.max(1))
                                    }
                                    (None, Some(c)) => format!("–/{}", k(c)),
                                    (Some(u), None) => k(u),
                                    _ => "–".to_string(),
                                };
                                let rows = [
                                    ("model", info.label.as_str()),
                                    ("arch", info.arch.as_str()),
                                    ("device", info.device.as_str()),
                                    ("format", info.format.as_str()),
                                    ("sampling", info.sampling.as_str()),
                                    ("max tokens", max_tokens.as_str()),
                                    ("context", ctx.as_str()),
                                ];
                                for (key, v) in rows {
                                    ui.weak(key);
                                    ui.monospace(v);
                                    ui.end_row();
                                }
                            });
                    }
                    None => {
                        ui.weak("loading…");
                    }
                }
                ui.add_space(10.0);
                ui.separator();
                ui.label(egui::RichText::new("controls").strong());
                ui.add_space(4.0);
                ui.checkbox(&mut self.show_reasoning, "show reasoning");
                ui.checkbox(&mut self.show_ticker, "status ticker");
                ui.add(
                    egui::Slider::new(&mut self.opacity, 0.0..=1.0)
                        .text("image opacity")
                        .fixed_decimals(2),
                );
            });
        }

        egui::Panel::bottom("input").show(ui, |ui| {
            ui.add_space(4.0);
            let ready = matches!(self.status, Status::Ready(_)) && !self.in_flight();
            ui.horizontal(|ui| {
                let send = ui.add_enabled(ready, egui::Button::new("send")).clicked();
                let edit = ui.add_enabled(
                    ready,
                    egui::TextEdit::singleline(&mut self.input)
                        .hint_text("message — try /plot")
                        .desired_width(f32::INFINITY),
                );
                let entered = edit.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter));
                // As a rule: whenever idle and ready, the input holds focus —
                // unless the user has deliberately focused something else (e.g.
                // the opacity slider). So we only grab when nothing is focused.
                let nothing_focused = ui.memory(|m| m.focused().is_none());
                if send || entered {
                    self.submit(now);
                    edit.request_focus();
                } else if ready && (self.focus_input || nothing_focused) {
                    edit.request_focus();
                    if self.focus_input {
                        // Place the cursor at the end so the field is genuinely in
                        // edit mode (accepting keystrokes), not just ringed.
                        if let Some(mut state) = egui::TextEdit::load_state(ui.ctx(), edit.id) {
                            let end = egui::text::CCursor::new(self.input.chars().count());
                            state
                                .cursor
                                .set_char_range(Some(egui::text::CCursorRange::one(end)));
                            state.store(ui.ctx(), edit.id);
                        }
                    }
                    self.focus_input = false;
                }
            });
            ui.add_space(4.0);
        });

        egui::CentralPanel::default().show(ui, |ui| {
            // Empty transcript (loading / pre-first-message): an animated splash.
            if self.transcript.is_empty() && self.streaming.is_none() {
                self.show_splash(ui);
                return;
            }
            egui::ScrollArea::vertical()
                .stick_to_bottom(true)
                .show(ui, |ui| {
                    let tint = self.image_tint();
                    for msg in &self.transcript {
                        render_msg(ui, msg, tint, self.show_reasoning);
                    }
                    if let Some(buf) = &self.streaming {
                        speaker(ui, "yatima", egui::Color32::LIGHT_GREEN);
                        if self.show_reasoning && !self.streaming_reasoning.is_empty() {
                            ui.label(
                                egui::RichText::new(&self.streaming_reasoning)
                                    .weak()
                                    .italics(),
                            );
                            ui.add_space(4.0);
                        }
                        if buf.is_empty() && !self.streaming_reasoning.is_empty() {
                            // Reasoning is flowing but the answer hasn't begun.
                            ui.label(egui::RichText::new("thinking…").weak());
                        } else {
                            ui.label(buf);
                        }
                        ui.add_space(8.0);
                    }
                });
        });

        if self.help_open {
            self.draw_help(ui, now);
        }
        if self.strawberry_start.is_some() {
            self.draw_strawberry(ui, now);
        }

        // Keep redrawing while a turn streams (the runner also pokes us per
        // fragment; this is the belt-and-braces fallback).
        if self.in_flight() {
            ui.ctx().request_repaint();
        }
    }
}

fn speaker(ui: &mut egui::Ui, who: &str, color: egui::Color32) {
    ui.label(egui::RichText::new(who).strong().color(color));
}

/// Format an elapsed-seconds duration as `M:SS` for the status readout.
fn fmt_clock(secs: f32) -> String {
    let s = secs.max(0.0) as u32;
    format!("{}:{:02}", s / 60, s % 60)
}

/// Format uptime as `H:MM:SS` (dropping the hours when zero).
fn fmt_uptime(secs: f32) -> String {
    let s = secs.max(0.0) as u32;
    let (h, m, sec) = (s / 3600, (s % 3600) / 60, s % 60);
    if h > 0 {
        format!("{h}:{m:02}:{sec:02}")
    } else {
        format!("{m}:{sec:02}")
    }
}

/// A cheap deterministic hash to `[0,1)` — pseudo-random scatter without an RNG
/// (used to place the strawberry-fields particles).
fn hash01(n: f32) -> f32 {
    let x = (n * 127.1).sin() * 43758.547;
    x - x.floor()
}

/// Compact token count: `2.1k`, `32.0k`, or the bare number under 1000.
fn k(n: usize) -> String {
    if n >= 1000 {
        format!("{:.1}k", n as f32 / 1000.0)
    } else {
        n.to_string()
    }
}

/// A scrolling marquee within `rect` (already clipped): the text drifts left and
/// loops seamlessly. Drives the idle status ticker.
fn draw_ticker(
    painter: &egui::Painter,
    rect: egui::Rect,
    text: &str,
    t: f32,
    color: egui::Color32,
) {
    let galley = painter.layout_no_wrap(text.to_owned(), egui::FontId::proportional(12.0), color);
    let tw = galley.size().x.max(1.0);
    let period = tw + 90.0; // text plus a gap before it repeats
    let off = (t * 32.0).rem_euclid(period);
    let y = rect.center().y - galley.size().y / 2.0;
    painter.galley(egui::pos2(rect.left() - off, y), galley.clone(), color);
    painter.galley(egui::pos2(rect.left() - off + period, y), galley, color);
}

/// `color` with its alpha replaced (it keeps the RGB, takes a new opacity).
fn with_alpha(color: egui::Color32, a: u8) -> egui::Color32 {
    egui::Color32::from_rgba_unmultiplied(color.r(), color.g(), color.b(), a)
}

/// The classic ease-out-bounce on `[0,1]` — a falling object that hits the floor
/// and bounces a couple of times before settling. Drives the `/help` drop.
fn ease_out_bounce(x: f32) -> f32 {
    let n1 = 7.5625;
    let d1 = 2.75;
    if x < 1.0 / d1 {
        n1 * x * x
    } else if x < 2.0 / d1 {
        let x = x - 1.5 / d1;
        n1 * x * x + 0.75
    } else if x < 2.5 / d1 {
        let x = x - 2.25 / d1;
        n1 * x * x + 0.9375
    } else {
        let x = x - 2.625 / d1;
        n1 * x * x + 0.984375
    }
}

/// Hermite smoothstep on a clamped `[0,1]` input — eases the draw-on so strokes
/// arrive and depart gently rather than linearly.
fn smoothstep(x: f32) -> f32 {
    let x = x.clamp(0.0, 1.0);
    x * x * (3.0 - 2.0 * x)
}

/// Map an `elapsed` time to `[0,1]` over the window `[t0, t1]` — the raw phase
/// of one animation step before easing.
fn time_ease(elapsed: f32, t0: f32, t1: f32) -> f32 {
    ((elapsed - t0) / (t1 - t0)).clamp(0.0, 1.0)
}

/// The shortest angular distance between two angles (degrees).
fn ang_dist(a: f32, b: f32) -> f32 {
    (((a - b) + 540.0) % 360.0 - 180.0).abs()
}

/// Draw a circular arc (degrees in egui's y-down frame: 0=right, 90=bottom,
/// 270=top) from `a0` to `a1`, revealed up to fraction `reveal` of its length,
/// leaving any angular `gaps` (center, half-width in degrees) unstroked. The
/// building block of the broken ring.
#[allow(clippy::too_many_arguments)]
fn draw_arc(
    painter: &egui::Painter,
    center: egui::Pos2,
    r: f32,
    a0: f32,
    a1: f32,
    reveal: f32,
    gaps: &[(f32, f32)],
    stroke: egui::Stroke,
) {
    let segments = 180;
    let (a0, a1) = (a0.to_radians(), a1.to_radians());
    let mut prev: Option<egui::Pos2> = None;
    for i in 0..=segments {
        let f = i as f32 / segments as f32;
        if f > reveal {
            break;
        }
        let ang = a0 + (a1 - a0) * f;
        let deg = ang.to_degrees().rem_euclid(360.0);
        if gaps.iter().any(|(c, half)| ang_dist(deg, *c) < *half) {
            prev = None;
            continue;
        }
        let pt = center + egui::vec2(ang.cos(), ang.sin()) * r;
        if let Some(pp) = prev {
            painter.line_segment([pp, pt], stroke);
        }
        prev = Some(pt);
    }
}

/// Draw the yatima sigil — the tree rune inside its broken ring — as vector
/// geometry with egui's painter, revealed stroke by stroke over `elapsed`
/// seconds: ring sweeps in, the stem descends (broken by a vertical ellipsis),
/// the branches fork, the nodes pop. `center`/`r` place and size the ring (in
/// pixels); coordinates are normalized to the ring radius. This is built-in
/// graphics — drawn live on the GPU, not a raster — so it can animate and
/// shimmer with the aurora `color`.
fn draw_sigil(
    painter: &egui::Painter,
    center: egui::Pos2,
    r: f32,
    elapsed: f32,
    color: egui::Color32,
    stroke_w: f32,
) {
    let stroke = egui::Stroke::new(stroke_w, color);
    let p = |nx: f32, ny: f32| center + egui::vec2(nx, ny) * r;

    // Broken ring: two arcs — a long one (lower-right around the bottom and
    // left up to the crown, with a dash-dot break in the lower left) and a short
    // one down the upper right — leaving the signature gaps at the crown and at
    // ~4 o'clock. Drawn progressively along each arc.
    let ring_t = smoothstep(time_ease(elapsed, 0.0, 0.9));
    draw_arc(
        painter,
        center,
        r,
        50.0,
        262.0,
        ring_t,
        &[(152.0, 7.0)],
        stroke,
    );
    draw_arc(painter, center, r, -80.0, 10.0, ring_t, &[], stroke);
    // The lone dot sitting inside the lower-left break.
    let break_dot = smoothstep(time_ease(elapsed, 0.5, 0.9));
    if break_dot > 0.0 {
        let a = 152f32.to_radians();
        painter.circle_filled(
            center + egui::vec2(a.cos(), a.sin()) * r,
            stroke_w * break_dot,
            color,
        );
    }

    // Central stem, drawn crown -> root, but BROKEN in the upper section by a
    // vertical ellipsis: the three dots sit on the line itself, replacing a
    // segment, rather than floating beside it.
    let crown = p(0.0, -1.0);
    let root = p(0.0, 1.0);
    let gap_top = -0.57; // stem stops here above the dots
    let gap_bot = -0.27; // stem resumes here below the dots

    let upper_t = smoothstep(time_ease(elapsed, 0.6, 0.9));
    let top_b = p(0.0, gap_top);
    painter.line_segment([crown, crown + (top_b - crown) * upper_t], stroke);

    let dot_ys = [-0.50f32, -0.42, -0.34];
    for (k, dy) in dot_ys.iter().enumerate() {
        let t0 = 0.9 + k as f32 * 0.12;
        let dot_t = smoothstep(time_ease(elapsed, t0, t0 + 0.3));
        if dot_t > 0.0 {
            painter.circle_filled(p(0.0, *dy), stroke_w * 1.1 * dot_t, color);
        }
    }

    let lower_t = smoothstep(time_ease(elapsed, 0.95, 1.6));
    let bot_a = p(0.0, gap_bot);
    painter.line_segment([bot_a, bot_a + (root - bot_a) * lower_t], stroke);

    // The two branches fork upward and outward.
    let branch_t = smoothstep(time_ease(elapsed, 1.1, 1.7));
    let fork = p(0.0, 0.08);
    let left = p(-0.46, -0.30);
    let right = p(0.46, -0.30);
    painter.line_segment([fork, fork + (left - fork) * branch_t], stroke);
    painter.line_segment([fork, fork + (right - fork) * branch_t], stroke);

    // Open-circle nodes pop in after the line each terminates.
    let node_r = r * 0.05;
    let nodes = [(crown, 0.7f32), (root, 1.3), (left, 1.5), (right, 1.5)];
    for (pos, t0) in nodes {
        let node_t = smoothstep(time_ease(elapsed, t0, t0 + 0.35));
        if node_t > 0.0 {
            painter.circle_stroke(pos, node_r * node_t, stroke);
        }
    }
}

/// Draw the "YATIMA" wordmark as vector strokes in the thin geometric face of
/// the logo — chevron `A`s, a pointed `M`, no serifs — squat and widely tracked.
/// `anchor.y` is the vertical center; `align` places it horizontally (0.5 =
/// centered on `anchor.x`, 0.0 = left edge at `anchor.x`). Letter geometry is
/// normalized to cap height (y down, 0 = top); letters fade in left to right.
#[allow(clippy::too_many_arguments)]
fn draw_wordmark(
    painter: &egui::Painter,
    anchor: egui::Pos2,
    cap_h: f32,
    color: egui::Color32,
    stroke_w: f32,
    elapsed: f32,
    align: f32,
) {
    // (advance width, polylines) per glyph, widths and points in cap heights.
    type Glyph = (f32, &'static [&'static [(f32, f32)]]);
    let letters: [Glyph; 6] = [
        // Y
        (
            0.62,
            &[
                &[(0.0, 0.0), (0.31, 0.5)],
                &[(0.62, 0.0), (0.31, 0.5)],
                &[(0.31, 0.5), (0.31, 1.0)],
            ],
        ),
        // A — a clean chevron, no crossbar
        (0.62, &[&[(0.0, 1.0), (0.31, 0.0), (0.62, 1.0)]]),
        // T
        (
            0.58,
            &[&[(0.0, 0.0), (0.58, 0.0)], &[(0.29, 0.0), (0.29, 1.0)]],
        ),
        // I
        (0.10, &[&[(0.05, 0.0), (0.05, 1.0)]]),
        // M — pointed top corners, a V dipping to mid height
        (
            0.74,
            &[&[
                (0.0, 1.0),
                (0.0, 0.0),
                (0.37, 0.62),
                (0.74, 0.0),
                (0.74, 1.0),
            ]],
        ),
        // A
        (0.62, &[&[(0.0, 1.0), (0.31, 0.0), (0.62, 1.0)]]),
    ];
    let track = 0.62; // inter-letter gap, in cap heights (spaced out)
    let xw = 1.22; // horizontal widen, so the squat letters don't read cramped

    let total: f32 =
        letters.iter().map(|(w, _)| w * xw).sum::<f32>() + track * (letters.len() - 1) as f32;
    let mut x = anchor.x - total * cap_h * align;
    let top = anchor.y - cap_h / 2.0;

    for (i, (w, polylines)) in letters.iter().enumerate() {
        let appear = smoothstep(time_ease(
            elapsed,
            2.0 + i as f32 * 0.1,
            2.45 + i as f32 * 0.1,
        ));
        if appear > 0.0 {
            let stroke = egui::Stroke::new(stroke_w, with_alpha(color, (appear * 235.0) as u8));
            for poly in *polylines {
                for seg in poly.windows(2) {
                    let a = egui::pos2(x + seg[0].0 * cap_h * xw, top + seg[0].1 * cap_h);
                    let b = egui::pos2(x + seg[1].0 * cap_h * xw, top + seg[1].1 * cap_h);
                    painter.line_segment([a, b], stroke);
                }
            }
        }
        x += (w * xw + track) * cap_h;
    }
}

/// yatima's moods — a tiny, coarse set for v1. Refined (thinking vs explaining)
/// once the reasoning channel lands.
#[derive(Clone, Copy)]
enum Face {
    Sleeping,
    Idle,
    Thinking,
    Talking,
    Sad,
    Surprised,
}

/// yatima's avatar — a rounded screen-head with glowing eyes (and a mouth) that
/// shape-shift to express state. Minimal and cute: the emotion is all in the
/// eyes and mouth. The gaze wanders, and she periodically *warps* on each axis:
/// `warp` is a horizontal scale (collapse to a vertical sliver) and `warp_y` a
/// vertical one (collapse to a horizontal sliver) — the teleport flourishes.
/// Geometry is center-relative so both scales apply uniformly. Built-in vectors.
#[allow(clippy::too_many_arguments)]
fn draw_face(
    painter: &egui::Painter,
    rect: egui::Rect,
    expr: Face,
    t: f32,
    color: egui::Color32,
    warp: f32,
    warp_y: f32,
    roll: f32,
) {
    let line = (rect.width() * 0.055).max(1.0);
    let stroke = egui::Stroke::new(line, color);
    let full = rect.shrink(rect.width() * 0.08);
    let cx = full.center().x;
    let cy = full.center().y;
    let (sx, sy) = (warp, warp_y);
    let h2 = full.height();
    let (rs, rc) = (roll.sin(), roll.cos());
    let rolling = roll.abs() > 1e-4;

    // Center-relative point: scale by (sx, sy) then rotate by `roll` about the
    // center. So the teleports collapse the face and a barrel roll spins it.
    let pt = |dx: f32, dy: f32| {
        let (px, py) = (dx * sx, dy * sy);
        egui::pos2(cx + px * rc - py * rs, cy + px * rs + py * rc)
    };

    // Head: a rounded screen at rest; a crisp rotated square while rolling.
    if rolling {
        let (hw, hh) = (full.width() * 0.5, full.height() * 0.5);
        let c = [pt(-hw, -hh), pt(hw, -hh), pt(hw, hh), pt(-hw, hh)];
        for i in 0..4 {
            painter.line_segment([c[i], c[(i + 1) % 4]], stroke);
        }
    } else {
        let head = egui::Rect::from_center_size(
            full.center(),
            egui::vec2(full.width() * sx, full.height() * sy),
        );
        painter.rect_stroke(head, full.width() * 0.30, stroke, egui::StrokeKind::Inside);
    }

    let pill = |dx: f32, dy: f32, w: f32, h: f32| {
        if rolling {
            let (hw, hh) = (w * 0.5, h * 0.5);
            let pts = vec![
                pt(dx - hw, dy - hh),
                pt(dx + hw, dy - hh),
                pt(dx + hw, dy + hh),
                pt(dx - hw, dy + hh),
            ];
            painter.add(egui::Shape::convex_polygon(pts, color, egui::Stroke::NONE));
        } else {
            let r = egui::Rect::from_center_size(
                pt(dx, dy),
                egui::vec2((w * sx).max(line * 0.5), (h * sy).max(line * 0.5)),
            );
            painter.rect_filled(r, (w * sx) * 0.5, color);
        }
    };
    // `dip > 0` smiles (corners up), `dip < 0` frowns.
    let mouth = |dy: f32, width: f32, dip: f32| {
        let n = 10;
        let mut prev: Option<egui::Pos2> = None;
        for i in 0..=n {
            let f = i as f32 / n as f32;
            let u = 2.0 * f - 1.0;
            let p = pt(-width / 2.0 + width * f, dy + dip * (1.0 - u * u));
            if let Some(pp) = prev {
                painter.line_segment([pp, p], stroke);
            }
            prev = Some(p);
        }
    };

    let ex = full.width() * 0.22; // eye x offset
    let ew = full.width() * 0.15; // eye width

    let bp = t.rem_euclid(3.0);
    let blink = if bp < 0.14 {
        ((bp / 0.14) - 0.5).abs() * 2.0
    } else {
        1.0
    };

    // Wandering gaze (left/right + up/down) — lively idle, subtle talking.
    let gaze = match expr {
        Face::Idle => 1.0,
        Face::Talking => 0.35,
        _ => 0.0,
    };
    let lx = (t * 0.7).sin() * (t * 0.31).cos() * ew * 0.5 * gaze;
    let ly = (t * 0.5).cos() * (t * 0.23).sin() * h2 * 0.07 * gaze;

    match expr {
        Face::Sleeping => {
            pill(-ex, 0.0, ew * 1.3, line);
            pill(ex, 0.0, ew * 1.3, line);
        }
        Face::Idle => {
            let h = ew * 1.5 * blink;
            pill(-ex + lx, -h2 * 0.04 + ly, ew, h);
            pill(ex + lx, -h2 * 0.04 + ly, ew, h);
            mouth(h2 * 0.20, ew * 2.0, h2 * 0.05);
        }
        Face::Thinking => {
            // A soft, cute ponder: round eyes glancing up and slowly to the
            // side, with a tiny neutral mouth. No strain.
            let glance = (t * 1.1).sin() * ew * 0.35;
            let edy = -h2 * 0.10; // looking up
            let h = ew * 1.4 * blink;
            pill(-ex + glance, edy, ew, h);
            pill(ex + glance, edy, ew, h);
            pill(0.0, h2 * 0.20, ew * 0.5, line); // tiny mouth dot
        }
        Face::Talking => {
            let pulse = 1.0 + 0.10 * (t * 6.0).sin();
            let h = ew * 1.5 * pulse * blink;
            pill(-ex + lx, -h2 * 0.05 + ly, ew, h);
            pill(ex + lx, -h2 * 0.05 + ly, ew, h);
            let open = (h2 * 0.10 * (0.5 + 0.5 * (t * 9.0).sin())).max(line);
            pill(0.0, h2 * 0.20, ew * 1.5, open); // animated talking mouth
        }
        Face::Sad => {
            pill(-ex, h2 * 0.02, ew, ew * 1.2);
            pill(ex, h2 * 0.02, ew, ew * 1.2);
            mouth(h2 * 0.26, ew * 1.9, -h2 * 0.05);
        }
        Face::Surprised => {
            // Wide round eyes raised high, and a small "o" of a mouth.
            pill(-ex, -h2 * 0.08, ew * 1.35, ew * 2.1);
            pill(ex, -h2 * 0.08, ew * 1.35, ew * 2.1);
            painter.circle_stroke(pt(0.0, h2 * 0.22), full.width() * 0.09 * sx.min(sy), stroke);
        }
    }
}

/// The teleport flourish: every `period` seconds yatima warps — collapsing to a
/// vertical sliver and snapping back over a short window. Returns a horizontal
/// scale in `[~0.05, 1]` for [`draw_face`].
fn warp_at(t: f32, period: f32) -> f32 {
    let wp = t.rem_euclid(period);
    if wp < 0.5 {
        let v = (2.0 * (wp / 0.5) - 1.0).abs(); // 1 at the ends, 0 at the middle
        0.05 + 0.95 * v
    } else {
        1.0
    }
}

/// An aurora color (northern-lights green -> teal -> cyan -> blue -> violet ->
/// pink), ping-ponged and sampled at `phase`. The GUI cousin of the TUI's
/// aurora ramp; truecolor here rather than the 256-color cube.
fn aurora_at(phase: f32) -> egui::Color32 {
    const STOPS: [(u8, u8, u8); 7] = [
        (72, 210, 160),
        (64, 200, 200),
        (80, 180, 230),
        (110, 140, 235),
        (150, 120, 225),
        (200, 120, 205),
        (225, 140, 180),
    ];
    let span = (STOPS.len() - 1) as f32;
    let p = phase.rem_euclid(2.0 * span);
    let p = if p > span { 2.0 * span - p } else { p };
    let i = p.floor() as usize;
    let f = p - i as f32;
    let j = (i + 1).min(STOPS.len() - 1);
    let lerp = |a: u8, b: u8| (a as f32 + (b as f32 - a as f32) * f).round() as u8;
    egui::Color32::from_rgb(
        lerp(STOPS[i].0, STOPS[j].0),
        lerp(STOPS[i].1, STOPS[j].1),
        lerp(STOPS[i].2, STOPS[j].2),
    )
}

fn render_msg(ui: &mut egui::Ui, msg: &Msg, image_tint: egui::Color32, show_reasoning: bool) {
    match msg {
        Msg::User(text) => {
            speaker(ui, "you", egui::Color32::LIGHT_BLUE);
            ui.label(text);
        }
        Msg::Assistant { answer, reasoning } => {
            speaker(ui, "yatima", egui::Color32::LIGHT_GREEN);
            if show_reasoning {
                if let Some(r) = reasoning {
                    ui.label(egui::RichText::new(r).weak().italics());
                    ui.add_space(4.0);
                }
            }
            ui.label(answer);
        }
        Msg::Image(tex) => {
            speaker(ui, "yatima", egui::Color32::LIGHT_GREEN);
            // Centered, tinted to settle the chart's white panel into the dark
            // UI, and clamped to the available width so an over-wide artifact
            // never pushes the scroll content off the left edge.
            let max_w = (ui.available_width() - 8.0).clamp(64.0, 640.0);
            ui.vertical_centered(|ui| {
                ui.add(
                    egui::Image::new(egui::load::SizedTexture::from_handle(tex))
                        .max_width(max_w)
                        .tint(image_tint),
                );
            });
        }
        Msg::Note(text) => {
            ui.label(egui::RichText::new(text).weak());
        }
        Msg::Error(text) => {
            ui.colored_label(egui::Color32::LIGHT_RED, format!("error: {text}"));
        }
    }
    ui.add_space(8.0);
}

/// Decode PNG bytes and upload them as an egui texture.
fn decode_texture(ctx: &egui::Context, bytes: &[u8]) -> Result<egui::TextureHandle> {
    let rgba = image::load_from_memory(bytes)?.to_rgba8();
    let (w, h) = rgba.dimensions();
    let color = egui::ColorImage::from_rgba_unmultiplied([w as usize, h as usize], rgba.as_raw());
    Ok(ctx.load_texture("artifact", color, egui::TextureOptions::default()))
}

fn main() -> Result<()> {
    let args = Args::parse();
    let cfg = resolve(&args)?;
    let title = format!("yatima — {}", cfg.label);

    eprintln!("loading model… (first run may fetch weights)");
    let native = eframe::NativeOptions {
        renderer: eframe::Renderer::Wgpu,
        viewport: egui::ViewportBuilder::default()
            .with_title(title.clone())
            .with_inner_size([580.0, 410.0]),
        ..Default::default()
    };
    eframe::run_native(
        &title,
        native,
        Box::new(move |cc| Ok(Box::new(GuiApp::new(cc, cfg)))),
    )
    .map_err(|e| anyhow::anyhow!("eframe error: {e}"))
}
