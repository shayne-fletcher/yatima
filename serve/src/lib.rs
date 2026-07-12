//! The serve bridge: yatima's event/request planes over one WebSocket.
//!
//! `yatima-serve` is the third frontend — except it draws nothing. It owns
//! a [`yatima_host::HostHandle`] exactly as the GUI does, and bridges the
//! two planes to a browser: every [`HostEvent`] goes out as one JSON text
//! frame; every inbound text frame is a [`HostRequest`]. The browser client
//! (`web/`, a wasm32 build over yatima-protocol) is a *viewer* in the same
//! sense the TUI and GUI are — the vision's rung 2: the event plane over a
//! websocket.
//!
//! # Invariant & law registry
//!
//! - **SRV-1** serve binds only an explicitly supplied, specific address:
//!   there is no default bind, and the unspecified addresses are refused
//!   with the private-network rule named — exposure beyond the tailnet is a
//!   decision nobody makes by accident. The refusal is on the *canonical*
//!   address, so the IPv4-mapped wildcard (`[::ffff:0.0.0.0]`, which binds
//!   every IPv4 interface just as `0.0.0.0` does) is refused too, not only
//!   the bare `0.0.0.0` / `[::]` forms. Cited by
//!   `bind_law_refuses_unspecified_and_requires_explicit`.
//! - **SRV-2** the wire is exactly the yatima-protocol enums as
//!   externally-tagged JSON — serve defines no message types of its own,
//!   and a client that speaks the protocol crate speaks serve. Cited by
//!   `wire_is_the_protocol_round_tripped`.
//! - **SRV-3** one client at a time: the host emits one event stream, and
//!   splitting it across sockets would silently corrupt both readers. A
//!   second concurrent connection is refused (409); when a client
//!   disconnects, the stream is returned intact and the next connection
//!   resumes it. Nothing is dropped. Events emitted while nobody is
//!   connected wait in the channel; the one event a session had already
//!   pulled and attempted to send rides the carry slot
//!   ([`EventStream::pending`]) to the next session. Delivery at the seam
//!   is *at-least-once*, not exactly-once: a successful `socket.send` means
//!   only that the frame was buffered locally, never that the peer read it,
//!   so a client that closes right after we send may never have seen that
//!   frame. Rather than guess, a session carries its last-attempted event
//!   forward and the next client receives it first — a viewer tolerates a
//!   repeated final fragment far better than a hole. Cited by
//!   `second_client_refused_and_stream_survives_reconnect` and
//!   `carry_slot_redelivers_the_last_attempted_event`.
//!
//!   The stream always comes back: a session hands it over through a
//!   [`StreamLease`] whose `Drop` restores it, so a WebSocket upgrade that
//!   fails after the stream is claimed, or a session that panics, cannot
//!   strand it and wedge serve at 409. And a session is always able to end:
//!   the outbound send is capped ([`SEND_STALL_CAP`]) and the peer is pinged
//!   on an idle timer ([`KEEPALIVE_INTERVAL`]) so a half-open client that
//!   stopped answering — even while the host is idle and no send is
//!   attempted — is reaped rather than holding the one stream forever.
//!
//! Mid-decode cancellation rides the out-of-band gate, not the request
//! queue: a wire [`HostRequest::Cancel`] maps to [`CancelGate::cancel`],
//! exactly as the host crate's module doc specifies for serve — the
//! request channel is unserviced while the actor decodes, so a queued
//! cancel would arrive after the turn it meant to stop.

use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::mpsc::Sender;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use axum::extract::ws::{Message, WebSocket};
use axum::extract::{State, WebSocketUpgrade};
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::routing::get;
use axum::Router;
use tokio::sync::mpsc::UnboundedReceiver;
use yatima_host::{CancelGate, HostHandle};
use yatima_protocol::{HostEvent, HostRequest};

/// Parse and validate a bind address under SRV-1: explicit and specific.
pub fn validate_bind(addr: &str) -> anyhow::Result<SocketAddr> {
    let addr: SocketAddr = addr
        .parse()
        .map_err(|e| anyhow::anyhow!("--bind {addr:?} is not a socket address: {e}"))?;
    // Canonicalize first: `::ffff:0.0.0.0` binds every IPv4 interface exactly
    // as `0.0.0.0` does, but its raw bytes are not all-zero, so a bare
    // `is_unspecified()` would wave it through (SRV-1's one real bypass).
    if addr.ip().to_canonical().is_unspecified() {
        anyhow::bail!(
            "--bind {addr} refused (SRV-1): binding every interface exposes \
             the session beyond the private network; bind the loopback or \
             the tailnet address explicitly (e.g. 100.x.y.z:PORT)"
        );
    }
    Ok(addr)
}

/// One [`HostEvent`], as the one wire frame it becomes (SRV-2).
pub fn encode_event(event: &HostEvent) -> serde_json::Result<String> {
    serde_json::to_string(event)
}

/// One inbound text frame, as the [`HostRequest`] it must be (SRV-2).
pub fn decode_request(frame: &str) -> serde_json::Result<HostRequest> {
    serde_json::from_str(frame)
}

/// The event stream a client session borrows, with its carry slot: the
/// last event a session attempted to send. A buffered `socket.send`
/// success is not proof the peer read the frame, so on handoff that one
/// event rides here to the next session and is delivered before the
/// channel resumes — at-least-once at the seam, never a drop (SRV-3). At
/// most one event sits in the slot: the most recent one attempted.
pub struct EventStream {
    rx: UnboundedReceiver<HostEvent>,
    pending: Option<HostEvent>,
}

/// The bridge's shared state: the host's request plane and cancel gate
/// (cloneable), the single event stream a connected client borrows (SRV-3 —
/// `None` while a client holds it), and the two liveness caps a session
/// obeys (fields so tests can shrink them from seconds to milliseconds).
pub struct Bridge {
    req_tx: Sender<HostRequest>,
    cancel: CancelGate,
    event_rx: Mutex<Option<EventStream>>,
    send_stall_cap: Duration,
    keepalive_interval: Duration,
}

impl Bridge {
    pub fn new(handle: HostHandle) -> Arc<Bridge> {
        Bridge::with_timing(handle, SEND_STALL_CAP, KEEPALIVE_INTERVAL)
    }

    fn with_timing(
        handle: HostHandle,
        send_stall_cap: Duration,
        keepalive_interval: Duration,
    ) -> Arc<Bridge> {
        Arc::new(Bridge {
            req_tx: handle.req_tx,
            cancel: handle.cancel,
            event_rx: Mutex::new(Some(EventStream {
                rx: handle.event_rx,
                pending: None,
            })),
            send_stall_cap,
            keepalive_interval,
        })
    }

    /// The serve router: the WebSocket route plus, when a client bundle
    /// directory is supplied, static serving for it. `ServeDir` rejects `..`
    /// traversal but follows symlinks (tower-http 0.6 exposes no toggle), so
    /// the bundle directory must be trusted operator build output (trunk's
    /// `dist/`) — a symlink inside it escapes to its target. Not attacker
    /// input on the tailnet posture; noted so it stays that way.
    pub fn router(self: Arc<Bridge>, static_dir: Option<PathBuf>) -> Router {
        let router = Router::new().route("/ws", get(ws_upgrade)).with_state(self);
        match static_dir {
            Some(dir) => router.fallback_service(tower_http::services::ServeDir::new(dir)),
            None => router,
        }
    }
}

/// A session's borrow of the one event stream, with return guaranteed
/// (SRV-3). The stream is taken *before* the upgrade so a second client is
/// refused at the handshake — but that split means the caller that takes it
/// (the HTTP handler) is not the code that hands it back (the upgraded
/// session), and axum runs the `on_upgrade` callback only on a *successful*
/// upgrade. A lease closes the gap: whoever holds it restores the stream on
/// `Drop`, so a failed upgrade (axum drops the callback with the lease
/// inside) or a panicking session returns the stream instead of stranding it
/// and wedging serve at 409 forever.
struct StreamLease {
    bridge: Arc<Bridge>,
    stream: Option<EventStream>,
}

impl StreamLease {
    /// Borrow the stream if it is free; `None` if a client already holds it.
    fn acquire(bridge: Arc<Bridge>) -> Option<StreamLease> {
        let stream = bridge.event_rx.lock().expect("event stream lock").take()?;
        Some(StreamLease {
            bridge,
            stream: Some(stream),
        })
    }

    fn stream(&mut self) -> &mut EventStream {
        self.stream
            .as_mut()
            .expect("lease holds its stream until Drop")
    }
}

impl Drop for StreamLease {
    fn drop(&mut self) {
        if let Some(stream) = self.stream.take() {
            *self.bridge.event_rx.lock().expect("event stream lock") = Some(stream);
            tracing::info!("client session ended; event stream returned");
        }
    }
}

async fn ws_upgrade(
    State(bridge): State<Arc<Bridge>>,
    upgrade: WebSocketUpgrade,
) -> impl IntoResponse {
    // SRV-3: lease the event stream before upgrading; a second concurrent
    // client finds it borrowed and is refused at the handshake. The lease
    // rides into the callback and returns the stream on Drop even if the
    // upgrade never completes.
    let Some(lease) = StreamLease::acquire(bridge) else {
        tracing::info!("second concurrent client refused 409 (SRV-3)");
        return (
            StatusCode::CONFLICT,
            "another client is connected (serve is single-client — SRV-3)",
        )
            .into_response();
    };
    tracing::info!("client connected; session begins");
    upgrade
        .on_upgrade(move |socket| client_session(lease, socket))
        .into_response()
}

/// The longest a single outbound frame may take to send. A client that has
/// stopped reading (dead peer, suspended laptop) would otherwise park
/// `send().await` forever — and since that session holds the one event
/// stream, an unbounded send would wedge serve permanently (every future
/// client refused under SRV-3). On the cap: the client is declared gone and
/// the stream returns. Generous, because image frames are large JSON in the
/// spike.
const SEND_STALL_CAP: Duration = Duration::from_secs(30);

/// How often an otherwise-idle session pings the client. The send cap only
/// reaps a peer there is a frame to send *to*; between turns nothing flows,
/// so a client that half-opens while the host is idle would hold the stream
/// until the OS TCP keepalive (~2h) noticed. A server ping on this timer with
/// a one-interval pong deadline reaps it in seconds instead — a live client's
/// WebSocket stack answers automatically, so only a gone one fails to.
const KEEPALIVE_INTERVAL: Duration = Duration::from_secs(20);

/// Serve one connected client until it hangs up (or a send/keepalive marks it
/// gone), then return the event stream — including the last event this session
/// attempted to send — for the next one (SRV-3, via [`StreamLease`]'s Drop).
/// Every wait here is bounded or cancels with the peer: the send is capped,
/// the idle read is bounded by the keepalive timer, and the channel/socket
/// reads legitimately wait for input — so the session can always end and the
/// stream can always come back.
async fn client_session(mut lease: StreamLease, mut socket: WebSocket) {
    let bridge = Arc::clone(&lease.bridge);
    let stream = lease.stream();
    // The event the previous session last attempted goes out before the
    // channel resumes: its send may have been buffered to a peer that then
    // vanished without reading it, so this new client receives it first
    // (at-least-once at the seam — SRV-3). Done once, before the loop; the
    // slot is a hand-off carry, not a per-iteration retry.
    if let Some(event) = stream.pending.take() {
        match send_event(&mut socket, &event, bridge.send_stall_cap).await {
            // A buffered send to this new client is no more proof of receipt
            // than the original was: keep the event in the carry slot so it
            // still rides to the next session if this one ends before a fresh
            // event supersedes it — the same rule the loop's `Sent` arm obeys.
            SendOutcome::Sent => stream.pending = Some(event),
            // The lease's Drop returns the stream (with the carried event) on
            // every early return below — no manual restore needed.
            SendOutcome::PeerGone => {
                stream.pending = Some(event);
                return;
            }
            SendOutcome::Unencodable => {}
        }
    }
    let mut keepalive = tokio::time::interval(bridge.keepalive_interval);
    keepalive.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
    keepalive.tick().await; // the first tick is immediate; spend it now
    let mut awaiting_pong = false;
    loop {
        // `biased` with the socket arm first so a waiting Close/Pong/request is
        // handled before another event is pulled — the sooner a dead peer is
        // seen, the sooner the stream returns. Correctness does not hinge on
        // it: whichever event this session last pulled is carried forward
        // regardless of who wins the race, so nothing is lost either way.
        tokio::select! {
            biased;
            msg = socket.recv() => match msg {
                Some(Ok(Message::Text(frame))) => match decode_request(&frame) {
                    // Mid-decode cancel must bypass the (unserviced) request
                    // queue and flip the out-of-band gate.
                    Ok(HostRequest::Cancel { turn_id }) => {
                        tracing::info!("wire Cancel for turn {turn_id} — flipping the gate");
                        bridge.cancel.cancel(turn_id);
                    }
                    Ok(request) => {
                        if let HostRequest::Submit { turn_id, .. } = &request {
                            // The id only: prompt text is the user's, and the
                            // console is not the transcript.
                            tracing::info!("Submit for turn {turn_id}");
                        }
                        if bridge.req_tx.send(request).is_err() {
                            return; // host gone
                        }
                    }
                    Err(e) => {
                        tracing::warn!("unintelligible frame dropped: {e}");
                    }
                },
                Some(Ok(Message::Pong(_))) => awaiting_pong = false,
                Some(Ok(Message::Close(_))) | None => return,
                Some(Ok(_)) => {} // ping/binary: nothing on this wire
                Some(Err(_)) => return,
            },
            _ = keepalive.tick() => {
                // A ping unanswered since the last tick means the peer is gone.
                if awaiting_pong {
                    tracing::warn!("client missed keepalive pong; dropping it");
                    return;
                }
                match send_ping(&mut socket, bridge.send_stall_cap).await {
                    SendOutcome::Sent => awaiting_pong = true,
                    _ => return,
                }
            }
            event = stream.rx.recv() => match event {
                Some(event) => match send_event(&mut socket, &event, bridge.send_stall_cap).await {
                    // A buffered send is not proof of receipt: carry this
                    // event forward so a peer that closes right after still
                    // costs no event — the next client re-receives it (SRV-3).
                    SendOutcome::Sent => stream.pending = Some(event),
                    SendOutcome::PeerGone => {
                        stream.pending = Some(event);
                        return;
                    }
                    SendOutcome::Unencodable => {}
                },
                None => return, // host gone; nothing left to serve
            },
        }
    }
}

enum SendOutcome {
    Sent,
    /// Send failed or stalled past the send cap — the peer is gone.
    PeerGone,
    /// The event could not be encoded — a protocol-crate bug (PROTO-2 says
    /// this cannot happen); dropping the frame beats dropping the session,
    /// and it must not ride the carry slot (it would wedge every future
    /// session on the same undeliverable event).
    Unencodable,
}

async fn send_event(socket: &mut WebSocket, event: &HostEvent, cap: Duration) -> SendOutcome {
    let frame = match encode_event(event) {
        Ok(frame) => frame,
        Err(e) => {
            tracing::warn!("could not encode event: {e}");
            return SendOutcome::Unencodable;
        }
    };
    match tokio::time::timeout(cap, socket.send(Message::Text(frame.into()))).await {
        Ok(Ok(())) => SendOutcome::Sent,
        Ok(Err(_)) => SendOutcome::PeerGone,
        Err(_) => {
            tracing::warn!("client stalled past {cap:?}; dropping it");
            SendOutcome::PeerGone
        }
    }
}

/// Send a keepalive ping under the same stall cap as an event; a stalled or
/// failed ping means the peer is gone. `Unencodable` cannot arise (empty
/// payload), so only [`SendOutcome::Sent`]/[`SendOutcome::PeerGone`] matter.
async fn send_ping(socket: &mut WebSocket, cap: Duration) -> SendOutcome {
    match tokio::time::timeout(cap, socket.send(Message::Ping(Vec::new().into()))).await {
        Ok(Ok(())) => SendOutcome::Sent,
        _ => SendOutcome::PeerGone,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures_util::SinkExt;
    use futures_util::StreamExt;
    use yatima_protocol::Channel;

    /// Every await in these tests is bounded: a broken bridge must be a
    /// red test in seconds, never a silent hang (the first version of this
    /// suite deadlocked CI by blocking the single-threaded test runtime —
    /// the integration tests below run multi_thread and never call a
    /// blocking std primitive on a runtime worker).
    const TEST_CAP: std::time::Duration = std::time::Duration::from_secs(10);

    async fn within<T>(what: &str, fut: impl std::future::Future<Output = T>) -> T {
        match tokio::time::timeout(TEST_CAP, fut).await {
            Ok(value) => value,
            Err(_) => panic!("timed out after {TEST_CAP:?}: {what}"),
        }
    }

    #[test]
    fn bind_law_refuses_unspecified_and_requires_explicit() {
        // upholds: SRV-1 — the unspecified addresses are refused with the
        // private-network rule named, including the IPv4-mapped wildcard that
        // binds every IPv4 interface but is not literally `0.0.0.0`; loopback
        // and specific addresses (mapped or not) pass.
        for refused in ["0.0.0.0:8080", "[::]:8080", "[::ffff:0.0.0.0]:8080"] {
            let err = validate_bind(refused).unwrap_err().to_string();
            assert!(err.contains("SRV-1"), "{err}");
            assert!(err.contains("private network"), "{err}");
        }
        assert!(validate_bind("127.0.0.1:0").is_ok());
        assert!(validate_bind("100.112.4.109:8080").is_ok());
        assert!(validate_bind("[::ffff:127.0.0.1]:0").is_ok());
        assert!(validate_bind("not-an-addr").is_err());
    }

    #[test]
    fn wire_is_the_protocol_round_tripped() {
        // upholds: SRV-2 — the frame is the protocol enum and nothing else:
        // encode → decode over the protocol types is identity, including a
        // bytes-bearing event and a multibyte fragment.
        let events = [
            HostEvent::Fragment {
                turn_id: 7,
                channel: Channel::Answer,
                text: "héllo — ≥ 3".into(),
            },
            HostEvent::Image {
                turn_id: 7,
                bytes: vec![0x89, b'P', b'N', b'G'],
                name: "img-abc.png".into(),
            },
            HostEvent::Note("compacted: dropped 2".into()),
        ];
        for event in events {
            let frame = encode_event(&event).unwrap();
            let back: HostEvent = serde_json::from_str(&frame).unwrap();
            assert_eq!(back, event);
        }
        let requests = [
            HostRequest::Submit {
                turn_id: 1,
                text: "hi".into(),
            },
            HostRequest::Cancel { turn_id: 1 },
            HostRequest::Reset,
        ];
        for request in requests {
            let frame = serde_json::to_string(&request).unwrap();
            assert_eq!(decode_request(&frame).unwrap(), request);
        }
    }

    type FakeHost = (
        Arc<Bridge>,
        tokio::sync::mpsc::UnboundedSender<HostEvent>,
        std::sync::mpsc::Receiver<HostRequest>,
        CancelGate,
    );

    /// A fake host: hand-built channels shaped like a `HostHandle`.
    fn fake_host() -> FakeHost {
        fake_host_timed(SEND_STALL_CAP, KEEPALIVE_INTERVAL)
    }

    /// A fake host with the liveness caps shrunk so a test can observe a
    /// keepalive reap or a send stall in milliseconds, not seconds.
    fn fake_host_timed(send_stall_cap: Duration, keepalive_interval: Duration) -> FakeHost {
        let (event_tx, event_rx) = tokio::sync::mpsc::unbounded_channel();
        let (req_tx, req_rx) = std::sync::mpsc::channel();
        let cancel = CancelGate::new();
        let bridge = Bridge::with_timing(
            HostHandle {
                req_tx,
                event_rx,
                cancel: cancel.clone(),
            },
            send_stall_cap,
            keepalive_interval,
        );
        (bridge, event_tx, req_rx, cancel)
    }

    async fn serve_on_ephemeral(bridge: Arc<Bridge>) -> SocketAddr {
        serve_on_ephemeral_dir(bridge, None).await
    }

    async fn serve_on_ephemeral_dir(
        bridge: Arc<Bridge>,
        static_dir: Option<PathBuf>,
    ) -> SocketAddr {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            axum::serve(listener, bridge.router(static_dir))
                .await
                .unwrap();
        });
        addr
    }

    fn ws_url(addr: SocketAddr) -> String {
        format!("ws://{addr}/ws")
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn bridge_pumps_events_out_requests_in_and_cancel_hits_the_gate() {
        // upholds: SRV-2 (the live wire carries the protocol types) and the
        // cancel-to-gate mapping: a wire Cancel flips the armed gate
        // out-of-band instead of queueing behind the in-flight turn.
        let (bridge, event_tx, req_rx, gate) = fake_host();
        let addr = serve_on_ephemeral(bridge).await;
        let (mut ws, _) = within(
            "client handshake",
            tokio_tungstenite::connect_async(format!("ws://{addr}/ws")),
        )
        .await
        .unwrap();

        event_tx.send(HostEvent::Note("ready-ish".into())).unwrap();
        let frame = within("first event frame", ws.next())
            .await
            .unwrap()
            .unwrap()
            .into_text()
            .unwrap();
        assert_eq!(
            serde_json::from_str::<HostEvent>(&frame).unwrap(),
            HostEvent::Note("ready-ish".into())
        );

        within(
            "submit send",
            ws.send(tokio_tungstenite::tungstenite::Message::Text(
                serde_json::to_string(&HostRequest::Submit {
                    turn_id: 3,
                    text: "run".into(),
                })
                .unwrap(),
            )),
        )
        .await
        .unwrap();
        // The request plane is a std (blocking) receiver; never block a
        // runtime worker on it — park the wait on the blocking pool.
        let received = within(
            "request reaches the host plane",
            tokio::task::spawn_blocking(move || req_rx.recv_timeout(TEST_CAP)),
        )
        .await
        .unwrap()
        .unwrap();
        assert_eq!(
            received,
            HostRequest::Submit {
                turn_id: 3,
                text: "run".into()
            }
        );

        let in_flight = yatima_lib::Cancel::new();
        gate.arm(3, in_flight.clone());
        within(
            "cancel send",
            ws.send(tokio_tungstenite::tungstenite::Message::Text(
                serde_json::to_string(&HostRequest::Cancel { turn_id: 3 }).unwrap(),
            )),
        )
        .await
        .unwrap();
        // The gate flips out-of-band; poll (bounded) rather than sleep blind.
        within("gate observes the cancel", async {
            while !in_flight.is_cancelled() {
                tokio::time::sleep(std::time::Duration::from_millis(10)).await;
            }
        })
        .await;
        assert!(in_flight.is_cancelled(), "wire Cancel must reach the gate");
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn second_client_refused_and_stream_survives_reconnect() {
        // upholds: SRV-3 — a concurrent second client is refused at the
        // handshake; after the first disconnects, a new client resumes the
        // same stream and events queued while disconnected arrive.
        let (bridge, event_tx, _req_rx, _gate) = fake_host();
        let addr = serve_on_ephemeral(bridge).await;

        let (mut first, _) = within(
            "first handshake",
            tokio_tungstenite::connect_async(ws_url(addr)),
        )
        .await
        .unwrap();
        // upholds: SRV-3 — the refusal is specifically HTTP 409, not any
        // stray non-101 (a regression to 500/reset would still "err").
        let refused = within(
            "second (refused) handshake",
            tokio_tungstenite::connect_async(ws_url(addr)),
        )
        .await;
        match refused {
            Err(tokio_tungstenite::tungstenite::Error::Http(resp)) => {
                assert_eq!(resp.status(), 409, "second client must be refused 409");
            }
            other => panic!("expected a 409 Http error, got {other:?}"),
        }

        within("first close", first.close(None)).await.unwrap();
        // Events sent while nobody is connected wait in the channel.
        event_tx.send(HostEvent::Note("queued".into())).unwrap();
        // The stream is returned on disconnect; poll the handshake (bounded)
        // until the returning session releases it.
        let mut second = within("reconnect after disconnect", async {
            loop {
                match tokio_tungstenite::connect_async(format!("ws://{addr}/ws")).await {
                    Ok((ws, _)) => break ws,
                    Err(_) => tokio::time::sleep(std::time::Duration::from_millis(10)).await,
                }
            }
        })
        .await;
        let frame = within("queued event arrives", second.next())
            .await
            .unwrap()
            .unwrap()
            .into_text()
            .unwrap();
        assert_eq!(
            serde_json::from_str::<HostEvent>(&frame).unwrap(),
            HostEvent::Note("queued".into())
        );
    }

    #[test]
    fn stream_lease_restores_on_drop() {
        // upholds: SRV-3 — the stream is returned even when the session code
        // that normally returns it never runs (a failed WS upgrade drops the
        // callback with the lease inside) or unwinds (a panic): the lease's
        // Drop is the single guarantee. A leaked stream would wedge serve at
        // 409 for every future client.
        let (bridge, _event_tx, _req_rx, _gate) = fake_host();
        {
            let lease = StreamLease::acquire(Arc::clone(&bridge)).expect("first lease");
            assert!(
                StreamLease::acquire(Arc::clone(&bridge)).is_none(),
                "the stream is single-borrow while a lease holds it"
            );
            drop(lease);
        }
        assert!(
            StreamLease::acquire(bridge).is_some(),
            "dropping the lease must return the stream to the bridge"
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn carry_slot_redelivers_the_last_attempted_event() {
        // upholds: SRV-3 — at-least-once at the seam. A client that receives an
        // event then drops without reading further may never have rendered it,
        // so the next client re-receives that last-attempted event before the
        // channel resumes. (The reconnect test covers channel-queued delivery;
        // this covers the carry slot the send path fills.)
        let (bridge, event_tx, _req_rx, _gate) = fake_host();
        let addr = serve_on_ephemeral(bridge).await;

        let (mut first, _) = within(
            "first handshake",
            tokio_tungstenite::connect_async(ws_url(addr)),
        )
        .await
        .unwrap();
        event_tx.send(HostEvent::Note("seam".into())).unwrap();
        let got = within("first receives the event", first.next())
            .await
            .unwrap()
            .unwrap()
            .into_text()
            .unwrap();
        assert_eq!(
            serde_json::from_str::<HostEvent>(&got).unwrap(),
            HostEvent::Note("seam".into())
        );
        // Drop without reading more: the send buffered, but this client is
        // treated as if it may never have shown the frame.
        within("first close", first.close(None)).await.unwrap();

        let mut second = within("reconnect", async {
            loop {
                match tokio_tungstenite::connect_async(ws_url(addr)).await {
                    Ok((ws, _)) => break ws,
                    Err(_) => tokio::time::sleep(Duration::from_millis(10)).await,
                }
            }
        })
        .await;
        let again = within("second re-receives the carried event", second.next())
            .await
            .unwrap()
            .unwrap()
            .into_text()
            .unwrap();
        assert_eq!(
            serde_json::from_str::<HostEvent>(&again).unwrap(),
            HostEvent::Note("seam".into()),
            "the last-attempted event rides the carry slot to the next client"
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn unintelligible_frame_is_dropped_and_the_session_survives() {
        // A frame that is not a HostRequest is warned and dropped, never fatal:
        // a following valid request still reaches the host plane.
        let (bridge, _event_tx, req_rx, _gate) = fake_host();
        let addr = serve_on_ephemeral(bridge).await;
        let (mut ws, _) = within("handshake", tokio_tungstenite::connect_async(ws_url(addr)))
            .await
            .unwrap();
        within(
            "garbage send",
            ws.send(tokio_tungstenite::tungstenite::Message::Text(
                "{not a request}".to_string(),
            )),
        )
        .await
        .unwrap();
        within(
            "valid submit send",
            ws.send(tokio_tungstenite::tungstenite::Message::Text(
                serde_json::to_string(&HostRequest::Submit {
                    turn_id: 1,
                    text: "after garbage".into(),
                })
                .unwrap(),
            )),
        )
        .await
        .unwrap();
        let received = within(
            "submit still reaches the host after garbage",
            tokio::task::spawn_blocking(move || req_rx.recv_timeout(TEST_CAP)),
        )
        .await
        .unwrap()
        .unwrap();
        assert_eq!(
            received,
            HostRequest::Submit {
                turn_id: 1,
                text: "after garbage".into()
            }
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn host_gone_ends_the_session_and_frees_the_stream() {
        // When the host drops its event sender, the session's rx.recv() returns
        // None, the session ends, and the stream is freed for the next client
        // rather than stranded.
        let (bridge, event_tx, _req_rx, _gate) = fake_host();
        let addr = serve_on_ephemeral(bridge).await;
        let (_first, _) = within(
            "first handshake",
            tokio_tungstenite::connect_async(ws_url(addr)),
        )
        .await
        .unwrap();
        drop(event_tx); // host thread gone: the event channel closes
        let _second = within("reconnect after host gone", async {
            loop {
                match tokio_tungstenite::connect_async(ws_url(addr)).await {
                    Ok((ws, _)) => break ws,
                    Err(_) => tokio::time::sleep(Duration::from_millis(10)).await,
                }
            }
        })
        .await;
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn idle_unresponsive_peer_is_reaped_by_keepalive() {
        // upholds: SRV-3 liveness — a peer that stops answering while the host
        // is idle (no event to send, so the send cap never fires) is pinged and,
        // on a missed pong, dropped, freeing the stream. Without the keepalive
        // this session would hold the one stream until the OS TCP keepalive.
        let (bridge, _event_tx, _req_rx, _gate) =
            fake_host_timed(Duration::from_secs(30), Duration::from_millis(50));
        let addr = serve_on_ephemeral(bridge).await;
        // Connect but never poll: tungstenite answers pings only on read, so
        // this client never pongs — a half-open peer by construction.
        let _silent = within(
            "silent handshake",
            tokio_tungstenite::connect_async(ws_url(addr)),
        )
        .await
        .unwrap();
        let _second = within("reconnect after keepalive reap", async {
            loop {
                match tokio_tungstenite::connect_async(ws_url(addr)).await {
                    Ok((ws, _)) => break ws,
                    Err(_) => tokio::time::sleep(Duration::from_millis(20)).await,
                }
            }
        })
        .await;
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn static_dir_serves_the_client_bundle() {
        // The --static-dir branch (router(Some(dir))) serves the bundle at `/`,
        // distinct from the /ws route.
        let dir = std::env::temp_dir().join(format!("yatima-serve-static-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("index.html"), "<h1>hello from dist</h1>").unwrap();

        let (bridge, _event_tx, _req_rx, _gate) = fake_host();
        let addr = serve_on_ephemeral_dir(bridge, Some(dir.clone())).await;
        let response = within("static GET", http_get(addr, "/index.html")).await;
        assert!(response.contains("200 OK"), "{response}");
        assert!(response.contains("hello from dist"), "{response}");

        let _ = std::fs::remove_dir_all(&dir);
    }

    async fn http_get(addr: SocketAddr, path: &str) -> String {
        use tokio::io::AsyncReadExt;
        use tokio::io::AsyncWriteExt;
        let mut stream = tokio::net::TcpStream::connect(addr).await.unwrap();
        let request =
            format!("GET {path} HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n");
        stream.write_all(request.as_bytes()).await.unwrap();
        let mut buf = Vec::new();
        stream.read_to_end(&mut buf).await.unwrap();
        String::from_utf8_lossy(&buf).into_owned()
    }
}
