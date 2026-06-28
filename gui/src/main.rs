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
    device, resolve_format, Cancel, ChatFormat, ChatSession, Engine, GenOpts, ModelProfile,
    ModelSource, Sampling,
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

/// Runner → UI events (over `std::sync::mpsc`; the UI drains them each frame).
enum Ev {
    /// Model loaded and ready to chat (carries the model label).
    Ready(String),
    /// A streamed slice of the current reply (raw — reasoning not yet split).
    Fragment(String),
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
    let template = format.template();
    let mut session = ChatSession::new(&mut engine, template).with_opts(cfg.opts);
    if let Some(system) = cfg.system {
        session = session.with_system(system);
    }
    let _ = ev_tx.send(Ev::Ready(cfg.label));
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
        let result = {
            let tx = ev_tx.clone();
            let ctx = ctx.clone();
            let mut on_token = move |frag: &str| {
                let _ = tx.send(Ev::Fragment(frag.to_string()));
                ctx.request_repaint();
            };
            session
                .turn_streaming_cancellable(&prompt, &cancel, &mut on_token)
                .map(|answer| answer.to_string())
        };
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

/// A rendered transcript entry (the UI mirror; the runner's session is truth).
enum Msg {
    User(String),
    Assistant(String),
    /// A decoded image artifact, uploaded as a GPU texture.
    Image(egui::TextureHandle),
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
    input: String,
    transcript: Vec<Msg>,
    /// The reply currently streaming in, if a turn is in flight.
    streaming: Option<String>,
    status: Status,
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
            input: String::new(),
            transcript: Vec::new(),
            streaming: None,
            status: Status::Loading,
        }
    }

    fn in_flight(&self) -> bool {
        self.streaming.is_some()
    }

    /// Fold runner events into the UI mirror.
    fn drain_events(&mut self) {
        while let Ok(ev) = self.ev_rx.try_recv() {
            match ev {
                Ev::Ready(label) => self.status = Status::Ready(label),
                Ev::Fragment(text) => {
                    if let Some(buf) = self.streaming.as_mut() {
                        buf.push_str(&text);
                    }
                }
                Ev::Image(bytes) => match decode_texture(&self.ctx, &bytes) {
                    Ok(tex) => self.transcript.push(Msg::Image(tex)),
                    Err(e) => self
                        .transcript
                        .push(Msg::Error(format!("image decode: {e}"))),
                },
                Ev::Done => {
                    // Drop the streaming buffer; only commit it as a reply if it
                    // actually carried text (a `/plot` turn streams no text).
                    if let Some(buf) = self.streaming.take() {
                        if !buf.trim().is_empty() {
                            self.transcript.push(Msg::Assistant(buf));
                        }
                    }
                }
                Ev::Error(message) => {
                    self.streaming = None;
                    self.transcript.push(Msg::Error(message));
                }
                Ev::Fatal(message) => {
                    self.streaming = None;
                    self.status = Status::Failed(message);
                }
            }
        }
    }

    /// Submit the current input as a turn — unless empty, not ready, or a turn
    /// is already in flight (single-in-flight, as in the TUI).
    fn submit(&mut self) {
        let prompt = self.input.trim().to_string();
        if prompt.is_empty() || self.in_flight() || !matches!(self.status, Status::Ready(_)) {
            return;
        }
        self.transcript.push(Msg::User(prompt.clone()));
        self.streaming = Some(String::new());
        let _ = self.req_tx.send(prompt);
        self.input.clear();
    }
}

impl eframe::App for GuiApp {
    // eframe 0.35: the app draws into a root `Ui` (panels operate on a `&mut Ui`,
    // and `TopBottomPanel`/`SidePanel` are unified into `egui::Panel`).
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        self.drain_events();

        egui::Panel::top("status").show(ui, |ui| {
            let (text, color) = match &self.status {
                Status::Loading => ("loading model…".to_string(), egui::Color32::GRAY),
                Status::Ready(label) => (format!("yatima · {label}"), egui::Color32::LIGHT_GREEN),
                Status::Failed(msg) => (format!("failed: {msg}"), egui::Color32::LIGHT_RED),
            };
            ui.horizontal(|ui| {
                ui.colored_label(color, text);
                if self.in_flight() {
                    ui.spinner();
                    ui.label("answering…");
                }
            });
        });

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
                if send || entered {
                    self.submit();
                    edit.request_focus();
                }
            });
            ui.add_space(4.0);
        });

        egui::CentralPanel::default().show(ui, |ui| {
            egui::ScrollArea::vertical()
                .stick_to_bottom(true)
                .show(ui, |ui| {
                    for msg in &self.transcript {
                        render_msg(ui, msg);
                    }
                    if let Some(buf) = &self.streaming {
                        speaker(ui, "yatima", egui::Color32::LIGHT_GREEN);
                        ui.label(buf);
                        ui.add_space(8.0);
                    }
                });
        });

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

fn render_msg(ui: &mut egui::Ui, msg: &Msg) {
    match msg {
        Msg::User(text) => {
            speaker(ui, "you", egui::Color32::LIGHT_BLUE);
            ui.label(text);
        }
        Msg::Assistant(text) => {
            speaker(ui, "yatima", egui::Color32::LIGHT_GREEN);
            ui.label(text);
        }
        Msg::Image(tex) => {
            speaker(ui, "yatima", egui::Color32::LIGHT_GREEN);
            // Centered, and tinted to ~85% opacity so the chart's white panel
            // settles into the dark UI rather than glaring against it.
            ui.vertical_centered(|ui| {
                ui.add(
                    egui::Image::new(egui::load::SizedTexture::from_handle(tex))
                        .max_width(640.0)
                        .tint(egui::Color32::from_white_alpha(217)),
                );
            });
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
            .with_inner_size([900.0, 700.0]),
        ..Default::default()
    };
    eframe::run_native(
        &title,
        native,
        Box::new(move |cc| Ok(Box::new(GuiApp::new(cc, cfg)))),
    )
    .map_err(|e| anyhow::anyhow!("eframe error: {e}"))
}
