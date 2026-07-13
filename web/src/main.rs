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

    /// Render `text` as a label — except when it quotes the grant
    /// suggestion's origin, in which case the url itself renders as a
    /// raised, hyperlink-styled button in the flow of the sentence (WEB-7:
    /// the ask appears in prose; the control lives where the eye already
    /// is). While the suggestion is *offered* the button is live and a
    /// click lands in `clicked` (the caller sends outside the borrow);
    /// once *sent* it renders disabled — visible state change, no
    /// double-clicks into duplicate queued grants (the host services
    /// requests between turns, so the acknowledging report can lag).
    /// Returns whether a button rendered, so the caller can fall back to a
    /// standalone one when the prose never quotes the origin.
    fn label_maybe_grant(
        ui: &mut egui::Ui,
        text: &str,
        offered: Option<&str>,
        sent: Option<&str>,
        muted: bool,
        clicked: &mut Option<String>,
    ) -> bool {
        let plain = |ui: &mut egui::Ui, s: &str| {
            if s.trim().is_empty() {
                return;
            }
            if muted {
                ui.weak(s);
            } else {
                ui.label(s);
            }
        };
        let hit = offered
            .and_then(|o| text.find(o).map(|i| (o, i, true)))
            .or_else(|| sent.and_then(|o| text.find(o).map(|i| (o, i, false))));
        let Some((origin, at, live)) = hit else {
            plain(ui, text);
            return false;
        };
        let mut rendered = false;
        ui.horizontal_wrapped(|ui| {
            ui.spacing_mut().item_spacing.x = 4.0;
            plain(ui, &text[..at]);
            let link = egui::RichText::new(origin)
                .underline()
                .color(ui.visuals().hyperlink_color);
            let hover = if live {
                "grant read access to this origin"
            } else {
                "grant sent — applies when the turn yields"
            };
            if ui
                .add_enabled(live, egui::Button::new(link))
                .on_hover_text(hover)
                .clicked()
            {
                *clicked = Some(origin.to_string());
            }
            rendered = true;
            plain(ui, &text[at + origin.len()..]);
        });
        rendered
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
        /// where it left off. `in_flight` is deliberately kept: if a turn was
        /// running, its remaining events re-arm the mirror (the Fragment arms
        /// arm on demand) and its Done settles it. Should that Done be dropped
        /// at the seam, the spinner stays lit but the stop button — shown
        /// whenever `in_flight` is set — settles the turn locally.
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
                            Ok(ev) => self.transcript.fold(ev),
                            Err(e) => log::warn!("unintelligible frame dropped: {e}"),
                        }
                    }
                    ewebsock::WsEvent::Message(_) => {} // ping/pong/binary: not this wire
                    ewebsock::WsEvent::Error(e) => {
                        self.connected = false;
                        self.dropped = true;
                        self.transcript
                            .fold(HostEvent::Note(format!("[socket error: {e}]")));
                    }
                    ewebsock::WsEvent::Closed => {
                        self.connected = false;
                        self.dropped = true;
                        self.transcript
                            .fold(HostEvent::Note("[connection closed]".into()));
                    }
                }
            }
        }

        fn submit(&mut self) {
            // upholds: WEB-2 — a submit is refused while a turn is in flight.
            let text = self.input.trim().to_string();
            // Grant commands work any time, even mid-turn (the GUI's rule):
            // they are requests, not turns. The reports come back as notes.
            if let Some(request) = yatima_web::parse_grant_command(&text) {
                self.send(&request);
                self.input.clear();
                return;
            }
            if text.is_empty() || self.transcript.in_flight().is_some() {
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
                    if self.transcript.in_flight().is_some() {
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
                            can_submit && self.transcript.in_flight().is_none(),
                            egui::Button::new("send"),
                        )
                        .clicked()
                        || submitted
                    {
                        self.submit();
                        edit.request_focus();
                    }
                    if let Some(turn_id) = self.transcript.in_flight() {
                        if ui.button("stop").clicked() {
                            // Tell the host to cancel, and settle the mirror
                            // locally now — don't wait for a Done that the
                            // reconnect seam may have dropped. This is also the
                            // escape hatch out of a wedged spinner.
                            self.send(&HostRequest::Cancel { turn_id });
                            self.transcript.abort();
                        }
                    }
                });
            });

            // The grant suggestion, pre-cloned so the entry loop below can
            // borrow entries freely; a click lands in `grant_click` and is
            // sent after the panel releases its borrows.
            let offered = self.transcript.pending_grant().map(str::to_string);
            let sent = self.transcript.sent_grant().map(str::to_string);
            let mut grant_click: Option<String> = None;
            let mut grant_inline = false;
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
                                    grant_inline |= label_maybe_grant(
                                        ui,
                                        answer,
                                        offered.as_deref(),
                                        sent.as_deref(),
                                        false,
                                        &mut grant_click,
                                    );
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
                                    grant_inline |= label_maybe_grant(
                                        ui,
                                        text,
                                        offered.as_deref(),
                                        sent.as_deref(),
                                        true,
                                        &mut grant_click,
                                    );
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
                                grant_inline |= label_maybe_grant(
                                    ui,
                                    answer,
                                    offered.as_deref(),
                                    sent.as_deref(),
                                    false,
                                    &mut grant_click,
                                );
                            }
                        }
                        // WEB-7 fallback: the suggestion exists but no
                        // rendered prose quoted the origin (the model
                        // paraphrased) — a standalone button at the
                        // conversation's tail, live while offered, disabled
                        // once sent, so the affordance never goes missing
                        // and the state change is visible either way.
                        if !grant_inline {
                            let (origin, live) = match (&offered, &sent) {
                                (Some(o), _) => (Some(o), true),
                                (None, Some(o)) => (Some(o), false),
                                (None, None) => (None, false),
                            };
                            if let Some(origin) = origin {
                                ui.add_space(6.0);
                                let label = if live {
                                    format!("grant {origin}")
                                } else {
                                    format!("grant sent: {origin}")
                                };
                                let link = egui::RichText::new(label)
                                    .underline()
                                    .color(ui.visuals().hyperlink_color);
                                if ui
                                    .add_enabled(live, egui::Button::new(link))
                                    .on_hover_text(if live {
                                        "grant read access to this origin"
                                    } else {
                                        "grant sent — applies when the turn yields"
                                    })
                                    .clicked()
                                {
                                    grant_click = Some(origin.clone());
                                }
                                ui.add_space(4.0);
                            }
                        }
                    });
            });
            // The tap is the user's grant (WEB-7) — sent here, outside the
            // panel's borrows. The mirror marks it Sent at once (the button
            // disables this same frame; no repeat clicks), and the landing
            // grant report clears the suggestion.
            if let Some(origin) = grant_click {
                self.transcript.mark_grant_sent();
                self.send(&HostRequest::Grant { origin });
            }
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
