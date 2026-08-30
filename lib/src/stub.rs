//! Hermetic process fixture for managed llama-server integration tests —
//! the single source of the stub's protocol behavior (the plan's rule: reuse,
//! never duplicate). Two thin bins wrap [`run`]: yatima-lib's own
//! `llama-server-stub` (its `CARGO_BIN_EXE` serves the lib battery) and
//! yatima-host's `llama-server-stub-host` (the same for the host battery —
//! `CARGO_BIN_EXE` is only set for bins of the crate under test).
//!
//! The `-m` filename selects behavior; all other arguments are the real
//! llama-server arguments the supervisor must supply.

use std::env;
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::Path;
use std::process;

/// The stub's entire behavior; never returns.
#[doc(hidden)]
pub fn run() -> ! {
    let args: Vec<String> = env::args().collect();
    let model = value_after(&args, "-m").unwrap_or_else(|| fail("missing -m"));
    let port: u16 = value_after(&args, "--port")
        .unwrap_or_else(|| fail("missing --port"))
        .parse()
        .unwrap_or_else(|_| fail("bad --port"));
    if value_after(&args, "--host") != Some("127.0.0.1") {
        fail("--host was not pinned to 127.0.0.1");
    }
    if value_after(&args, "-np") != Some("1") || !args.iter().any(|arg| arg == "--jinja") {
        fail("-np 1 and --jinja are required");
    }

    let behavior = Path::new(model)
        .file_stem()
        .and_then(|name| name.to_str())
        .unwrap_or_else(|| fail("model has no UTF-8 file stem"));
    eprintln!("stub behavior={behavior} args={args:?}");

    if behavior.starts_with("leak-pipe") {
        // Spawn a descendant that inherits this process's stdout/stderr
        // pipes and outlives it: after the supervisor kills the stub, the
        // drains never see EOF, so the bounded cleanup fails — the
        // epilogue-failure witness. `exec -a` pins the model path into the
        // sleeper's argv[0] (a plain `exec` would erase it), so the test
        // can find and kill it afterward; the 45 s is its own backstop.
        let _ = std::process::Command::new("/bin/bash")
            .arg("-c")
            .arg("exec -a \"$1\" sleep 45")
            .arg("_")
            .arg(model)
            .spawn();
    }
    if behavior == "exit-immediately" {
        eprintln!("exit-immediately marker");
        process::exit(23);
    }
    flood_if_requested(behavior);

    let listener = TcpListener::bind(("127.0.0.1", port)).unwrap_or_else(|error| {
        eprintln!("bind failed: {error}");
        process::exit(24);
    });
    let mut health_polls = 0usize;
    let mut completion_hits = 0usize;
    for stream in listener.incoming() {
        let mut stream = stream.unwrap_or_else(|error| fail(&format!("accept: {error}")));
        let path = request_path(&mut stream);
        if path == "/completion" {
            completion_hits += 1;
            std::fs::write(completion_sentinel(model), b"completion requested")
                .unwrap_or_else(|error| fail(&format!("write completion sentinel: {error}")));
        }
        match path.as_str() {
            "/health" => {
                health_polls += 1;
                let ready_after = behavior
                    .strip_prefix("ready-after-")
                    .and_then(|count| count.parse::<usize>().ok())
                    .unwrap_or(0);
                if behavior == "never-ready" || health_polls <= ready_after {
                    respond(&mut stream, 503, "text/plain", "loading");
                } else {
                    respond(&mut stream, 200, "application/json", r#"{"status":"ok"}"#);
                }
            }
            "/props" if behavior == "introspection-fail" => {
                respond(&mut stream, 200, "application/json", r#"{"model_path":7}"#);
            }
            "/props" => respond(
                &mut stream,
                200,
                "application/json",
                &props(behavior, model),
            ),
            "/completion" if behavior == "slow-stream" => {
                // Hold the stream open, dribbling fragments: the active-turn
                // shutdown witness cancels mid-stream. Write errors mean the
                // client hung up (the cancel landed) — they must not kill the
                // stub, whose child process must stay alive to be reaped.
                dribble(&mut stream);
            }
            "/completion" if behavior == "die-after-ready" || behavior == "leak-pipe-die" => {
                eprintln!("die-after-ready marker");
                process::exit(25);
            }
            "/completion" if behavior == "incomplete-stream" => respond(
                &mut stream,
                200,
                "text/event-stream",
                "data: {\"content\":\"partial\",\"stop\":false}\n\n",
            ),
            // First turn fails mid-stream, later turns answer: the
            // recoverable-turn-failure fixture (the host must emit Error,
            // keep the session, and serve the retry).
            "/completion" if behavior == "error-then-answer" && completion_hits == 1 => respond(
                &mut stream,
                200,
                "text/event-stream",
                "data: {\"content\":\"partial\",\"stop\":false}\n\n",
            ),
            "/completion" => {
                let body = match behavior {
                    // One complete ATEM chat turn: addressed reasoning, then
                    // the answer, split mid-marker to exercise chunked
                    // classification.
                    "muse-chat" => sse(&[
                        " to=self<|message|>consider the question",
                        "<|eom|><|start|>assistant to=user<|mess",
                        "age|>a clean answer<|eot|>",
                    ]),
                    // A complete ATEM tool round across two turns: reasoning
                    // plus a read_url invocation, then (next /completion)
                    // reasoning plus the final answer.
                    "muse-tool-round" if completion_hits == 1 => sse(&[
                        " to=self<|message|>fetch the page first<|eom|>",
                        "<|start|>assistant to=read_url<|message|><atem:function_calls>\n",
                        "<atem:invoke name=\"read_url\">\n<atem:parameter name=\"url\">",
                        "https://other.example/</atem:parameter>\n</atem:invoke>\n",
                        "</atem:function_calls><|eot|>",
                    ]),
                    "muse-tool-round" => sse(&[
                        " to=self<|message|>the fetch was refused; answer from memory<|eom|>",
                        "<|start|>assistant to=user<|message|>the final grounded answer<|eot|>",
                    ]),
                    _ => sse(&["stub answer"]),
                };
                respond(&mut stream, 200, "text/event-stream", &body);
            }
            _ => respond(&mut stream, 404, "text/plain", "not found"),
        }
    }
    process::exit(0)
}

/// Stream an unending completion slowly: headers, one immediate fragment
/// (so the observer can prove the turn is live), then a dribble with no stop
/// event. Ends only when the peer disconnects; up to ~60 s as a backstop.
fn dribble(stream: &mut TcpStream) {
    let head = "HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\nconnection: close\r\n\r\n";
    if stream.write_all(head.as_bytes()).is_err() {
        return;
    }
    let fragment = format!(
        "data: {}\n\n",
        serde_json::json!({ "content": "streaming answer text ", "stop": false })
    );
    for _ in 0..600 {
        if stream.write_all(fragment.as_bytes()).is_err() || stream.flush().is_err() {
            return;
        }
        std::thread::sleep(std::time::Duration::from_millis(100));
    }
}

/// Render fragments as a llama-server SSE completion stream: one `data:` line
/// per fragment (JSON-escaped), closed by an EOS stop event.
fn sse(fragments: &[&str]) -> String {
    let mut body = String::new();
    for fragment in fragments {
        body.push_str(&format!(
            "data: {}\n\n",
            serde_json::json!({ "content": fragment, "stop": false })
        ));
    }
    body.push_str("data: {\"content\":\"\",\"stop\":true,\"stop_type\":\"eos\"}\n\n");
    body
}

fn props(behavior: &str, model: &str) -> String {
    let build = match behavior {
        "gate-old-build" => "b10000-stub",
        "gate-unparseable-build" => "stub-b10520",
        _ => "b10520-stub",
    };
    let context = if behavior == "gate-wrong-context" || behavior == "leak-pipe-bad-gate" {
        2048
    } else {
        4096
    };
    let slots = if behavior == "gate-wrong-slots" { 2 } else { 1 };
    let template = if behavior == "gate-wrong-template" {
        Some("wrong-template")
    } else if behavior == "gate-missing-template" {
        None
    } else {
        Some("stub-template")
    };
    let mut value = serde_json::json!({
        "build_info": build,
        "model_path": model,
        "total_slots": slots,
        "default_generation_settings": { "n_ctx": context },
    });
    if let Some(template) = template {
        value["chat_template"] = serde_json::Value::String(template.to_string());
    }
    value.to_string()
}

fn completion_sentinel(model: &str) -> std::path::PathBuf {
    Path::new(model).with_extension("completion-hit")
}

fn value_after<'a>(args: &'a [String], flag: &str) -> Option<&'a str> {
    args.iter()
        .position(|arg| arg == flag)
        .and_then(|index| args.get(index + 1))
        .map(String::as_str)
}

fn request_path(stream: &mut TcpStream) -> String {
    let mut request = [0u8; 16 * 1024];
    let count = stream
        .read(&mut request)
        .unwrap_or_else(|error| fail(&format!("read request: {error}")));
    let first = String::from_utf8_lossy(&request[..count]);
    first
        .lines()
        .next()
        .and_then(|line| line.split_whitespace().nth(1))
        .unwrap_or("/")
        .to_string()
}

fn respond(stream: &mut TcpStream, status: u16, content_type: &str, body: &str) {
    let reason = match status {
        200 => "OK",
        404 => "Not Found",
        503 => "Service Unavailable",
        _ => "Response",
    };
    write!(
        stream,
        "HTTP/1.1 {status} {reason}\r\ncontent-type: {content_type}\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}",
        body.len()
    )
    .unwrap_or_else(|error| fail(&format!("write response: {error}")));
    stream.flush().ok();
}

fn flood_if_requested(behavior: &str) {
    const SIZE: usize = 256 * 1024;
    let stdout = behavior == "flood-stdout" || behavior == "flood-both";
    let stderr = behavior == "flood-stderr" || behavior == "flood-both";
    let out = stdout.then(|| {
        std::thread::spawn(|| {
            std::io::stdout().write_all(&vec![b'o'; SIZE]).unwrap();
            println!("stdout-flood-complete");
        })
    });
    let err = stderr.then(|| {
        std::thread::spawn(|| {
            std::io::stderr().write_all(&vec![b'e'; SIZE]).unwrap();
            eprintln!("stderr-flood-complete");
        })
    });
    if let Some(task) = out {
        task.join().unwrap();
    }
    if let Some(task) = err {
        task.join().unwrap();
    }
}

fn fail(message: &str) -> ! {
    eprintln!("llama-server-stub: {message}");
    process::exit(22)
}
