//! The browser viewer: a thin egui view over [`yatima_web::Transcript`],
//! speaking yatima-protocol JSON over one WebSocket to yatima-serve (SRV-2:
//! the wire is the protocol crate, so this client speaks serve by
//! construction). Compiles only for wasm32 — the native side of this crate
//! is the transcript model and its tests (src/lib.rs).

#[cfg(target_arch = "wasm32")]
mod app {
    use std::collections::HashMap;

    use eframe::egui;
    use yatima_protocol::{HostEvent, HostRequest};
    use yatima_web::{Entry, Transcript};

    /// The WebSocket URL for the serve session this bundle was loaded from:
    /// same origin, `/ws` route (serve's static route hands out this app, so
    /// the socket goes back to the same host:port).
    fn ws_url() -> String {
        let location = web_sys::window().expect("browser window").location();
        let scheme = match location.protocol().as_deref() {
            Ok("https:") => "wss",
            _ => "ws",
        };
        let host = location.host().expect("location host");
        format!("{scheme}://{host}/ws")
    }

    /// Open the socket, waking egui on every socket event — without the
    /// wakeup an idle frame never notices an arriving fragment.
    fn connect(ctx: &egui::Context) -> (ewebsock::WsSender, ewebsock::WsReceiver) {
        let ctx = ctx.clone();
        ewebsock::connect_with_wakeup(ws_url(), ewebsock::Options::default(), move || {
            ctx.request_repaint()
        })
        .expect("websocket connect")
    }

    pub struct WebApp {
        transcript: Transcript,
        ws_tx: ewebsock::WsSender,
        ws_rx: ewebsock::WsReceiver,
        /// The egui context, kept so a reconnect can wire its wakeup.
        ctx: egui::Context,
        /// Socket state, for the status line ("connecting…" until Opened).
        connected: bool,
        /// The socket died (idle phone, network blip): offer the reconnect
        /// button. A reconnect here — no page reload — keeps this app
        /// instance, so the transcript and its textures stay on screen and
        /// the resumed stream (SRV-3) appends to them; a browser refresh
        /// would wipe the mirror and replay only what queued while away.
        dropped: bool,
        input: String,
        /// Client-local turn counter (the id space is this client's own;
        /// serve relays, the host arms its gate per turn).
        next_turn_id: u64,
        /// Reasoning fold: collapsed by default, one checkbox to reveal.
        show_reasoning: bool,
        /// Textures for decoded images, keyed by entry index (uploaded once,
        /// on first render).
        textures: HashMap<usize, egui::TextureHandle>,
    }

    impl WebApp {
        pub fn new(ctx: &egui::Context) -> Self {
            let (ws_tx, ws_rx) = connect(ctx);
            WebApp {
                transcript: Transcript::default(),
                ws_tx,
                ws_rx,
                ctx: ctx.clone(),
                connected: false,
                dropped: false,
                input: String::new(),
                next_turn_id: 0,
                show_reasoning: false,
                textures: HashMap::new(),
            }
        }

        /// Replace the dead socket with a fresh one; serve returns the same
        /// event stream to the next connection, so the session continues
        /// where it left off (`in_flight` is deliberately not cleared: if a
        /// turn was running, its remaining events — or its Done — arrive on
        /// the resumed stream and settle it).
        fn reconnect(&mut self) {
            let (ws_tx, ws_rx) = connect(&self.ctx);
            self.ws_tx = ws_tx;
            self.ws_rx = ws_rx;
            self.connected = false;
            self.dropped = false;
        }

        fn send(&mut self, request: &HostRequest) {
            match serde_json::to_string(request) {
                Ok(frame) => self.ws_tx.send(ewebsock::WsMessage::Text(frame)),
                Err(e) => log::warn!("could not encode request: {e}"),
            }
        }

        fn drain_socket(&mut self) {
            while let Some(event) = self.ws_rx.try_recv() {
                match event {
                    ewebsock::WsEvent::Opened => self.connected = true,
                    ewebsock::WsEvent::Message(ewebsock::WsMessage::Text(frame)) => {
                        match serde_json::from_str::<HostEvent>(&frame) {
                            Ok(ev) => self.transcript.apply(ev),
                            Err(e) => log::warn!("unintelligible frame dropped: {e}"),
                        }
                    }
                    ewebsock::WsEvent::Message(_) => {} // ping/pong/binary: not this wire
                    ewebsock::WsEvent::Error(e) => {
                        self.connected = false;
                        self.dropped = true;
                        self.transcript
                            .apply(HostEvent::Note(format!("[socket error: {e}]")));
                    }
                    ewebsock::WsEvent::Closed => {
                        self.connected = false;
                        self.dropped = true;
                        self.transcript
                            .apply(HostEvent::Note("[connection closed]".into()));
                    }
                }
            }
        }

        fn submit(&mut self) {
            let text = self.input.trim().to_string();
            if text.is_empty() || self.transcript.in_flight.is_some() {
                return; // single-in-flight, as in the GUI/TUI
            }
            let turn_id = self.next_turn_id;
            self.next_turn_id += 1;
            self.transcript.push_user(turn_id, &text);
            self.send(&HostRequest::Submit { turn_id, text });
            self.input.clear();
        }

        fn status_line(&self) -> String {
            if let Some(fatal) = &self.transcript.fatal {
                return format!("failed: {fatal}");
            }
            if self.dropped {
                return "disconnected".into();
            }
            let mut parts = Vec::new();
            parts.push(match (&self.transcript.model, self.connected) {
                (Some(info), _) => info.label.clone(),
                (None, true) => "loading…".into(),
                (None, false) => "connecting…".into(),
            });
            if let Some(tokens) = self.transcript.prompt_tokens {
                parts.push(format!("ctx {tokens} tok"));
            }
            parts.join("  ·  ")
        }

        fn texture_for(
            ctx: &egui::Context,
            textures: &mut HashMap<usize, egui::TextureHandle>,
            index: usize,
            img: &yatima_web::DecodedImage,
        ) -> egui::TextureHandle {
            textures
                .entry(index)
                .or_insert_with(|| {
                    let color = egui::ColorImage::from_rgba_unmultiplied(img.size, &img.rgba);
                    ctx.load_texture(&img.name, color, egui::TextureOptions::default())
                })
                .clone()
        }
    }

    impl eframe::App for WebApp {
        // egui 0.35 shape: the root callback receives a `Ui` and panels are
        // unified into `egui::Panel` (same as the GUI).
        fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
            self.drain_socket();

            egui::Panel::top("status").show(ui, |ui| {
                ui.horizontal(|ui| {
                    ui.label(self.status_line());
                    if self.transcript.in_flight.is_some() {
                        ui.spinner();
                    }
                    ui.checkbox(&mut self.show_reasoning, "show reasoning");
                    if self.dropped && ui.button("reconnect").clicked() {
                        self.reconnect();
                    }
                });
            });

            egui::Panel::bottom("input").show(ui, |ui| {
                let can_submit = self.connected && self.transcript.fatal.is_none();
                ui.horizontal(|ui| {
                    let edit = ui.add_enabled(
                        can_submit,
                        egui::TextEdit::singleline(&mut self.input)
                            .desired_width(ui.available_width() - 140.0)
                            .hint_text("ask…"),
                    );
                    let submitted =
                        edit.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter));
                    if ui
                        .add_enabled(
                            can_submit && self.transcript.in_flight.is_none(),
                            egui::Button::new("send"),
                        )
                        .clicked()
                        || submitted
                    {
                        self.submit();
                        edit.request_focus();
                    }
                    if let Some(turn_id) = self.transcript.in_flight {
                        if ui.button("stop").clicked() {
                            self.send(&HostRequest::Cancel { turn_id });
                        }
                    }
                });
            });

            egui::CentralPanel::default().show(ui, |ui| {
                egui::ScrollArea::vertical()
                    .stick_to_bottom(true)
                    .show(ui, |ui| {
                        let ctx = ui.ctx().clone();
                        for index in 0..self.transcript.entries.len() {
                            match &self.transcript.entries[index] {
                                Entry::User(text) => {
                                    ui.strong(format!("you: {text}"));
                                }
                                Entry::Assistant { answer, reasoning } => {
                                    if self.show_reasoning {
                                        if let Some(r) = reasoning {
                                            ui.weak(r);
                                        }
                                    }
                                    ui.label(answer);
                                }
                                Entry::Image(img) => {
                                    let tex =
                                        Self::texture_for(&ctx, &mut self.textures, index, img);
                                    // Clamped to the available width (the
                                    // GUI's exact clamp): an over-wide chart
                                    // fits a phone screen instead of pushing
                                    // past its edge, aspect preserved.
                                    let max_w = (ui.available_width() - 8.0).clamp(64.0, 640.0);
                                    ui.add(
                                        egui::Image::new(egui::load::SizedTexture::from_handle(
                                            &tex,
                                        ))
                                        .max_width(max_w),
                                    );
                                }
                                Entry::Note(text) => {
                                    ui.weak(text);
                                }
                                Entry::Error(text) => {
                                    ui.colored_label(
                                        egui::Color32::LIGHT_RED,
                                        format!("error: {text}"),
                                    );
                                }
                            }
                            ui.add_space(6.0);
                        }
                        // The turn in flight: reasoning fold first (opt-in),
                        // then the streaming answer.
                        if self.show_reasoning && !self.transcript.streaming_reasoning().is_empty()
                        {
                            ui.weak(self.transcript.streaming_reasoning());
                        }
                        if let Some(answer) = self.transcript.streaming_answer() {
                            if !answer.is_empty() {
                                ui.label(answer);
                            }
                        }
                    });
            });
        }
    }
}

#[cfg(target_arch = "wasm32")]
fn main() {
    use eframe::wasm_bindgen::JsCast as _;
    wasm_bindgen_futures::spawn_local(async {
        let document = web_sys::window()
            .expect("browser window")
            .document()
            .expect("document");
        let canvas = document
            .get_element_by_id("yatima_canvas")
            .expect("canvas element (index.html defines it)")
            .dyn_into::<web_sys::HtmlCanvasElement>()
            .expect("#yatima_canvas is a canvas");
        eframe::WebRunner::new()
            .start(
                canvas,
                eframe::WebOptions::default(),
                Box::new(|cc| Ok(Box::new(app::WebApp::new(&cc.egui_ctx)))),
            )
            .await
            .expect("eframe start");
    });
}

#[cfg(not(target_arch = "wasm32"))]
fn main() {
    // The app targets the browser; natively this crate is its transcript
    // model and tests. `trunk build` (or `trunk serve`) produces the bundle.
    eprintln!("yatima-web is a wasm32 app: build with `trunk build` in web/");
    std::process::exit(2);
}
