//! The driver's hermetic battery: the real `yatima-drive` binary over the
//! one llama-server protocol stub — scenarios in, tapes and exit codes out,
//! every exit converging on the joined reap. No network, no models.
//!
//! Each test's unique tempdir path doubles as the process marker the pgrep
//! oracle greps for (the lifecycle battery's discipline); the doc-hidden
//! `YATIMA_DRIVE_TEST_STUB_*` seam points the binary's resolver at the
//! wrapper stub, mirroring `with_managed_launcher`.

use std::path::{Path, PathBuf};
use std::time::Duration;

use tempfile::TempDir;

const WITHIN: Duration = Duration::from_secs(30);

/// One battery at a time: every test spawns and reaps real OS children in
/// one process, and coalesced SIGCHLD under churn fails bystanders (the
/// host battery's finding). tokio's Mutex: never a std lock across an await.
static SESSION: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

struct Run {
    status: std::process::ExitStatus,
    stderr: String,
    run_dir: PathBuf,
}

/// A driver invocation over the stub: `{behavior}.gguf` in a fresh tempdir,
/// the scenario written beside it, the tape under it too.
async fn run_driver(dir: &TempDir, behavior: &str, scenario: &str, extra: &[&str]) -> Run {
    let child = spawn_driver(dir, behavior, scenario, extra);
    let output = tokio::time::timeout(WITHIN, child.wait_with_output())
        .await
        .expect("the driver must exit within the bound")
        .expect("collect driver output");
    let stdout = String::from_utf8_lossy(&output.stdout).into_owned();
    let stderr = String::from_utf8_lossy(&output.stderr).into_owned();
    let run_dir = PathBuf::from(stdout.lines().next().unwrap_or_default());
    Run {
        status: output.status,
        stderr,
        run_dir,
    }
}

fn spawn_driver(
    dir: &TempDir,
    behavior: &str,
    scenario: &str,
    extra: &[&str],
) -> tokio::process::Child {
    std::fs::write(dir.path().join(format!("{behavior}.gguf")), b"stub").unwrap();
    let scenario_path = dir.path().join("scenario.txt");
    std::fs::write(&scenario_path, scenario).unwrap();
    let tape = dir.path().join("run");
    tokio::process::Command::new(env!("CARGO_BIN_EXE_yatima-drive"))
        .arg(scenario_path)
        .arg("--tape")
        .arg(&tape)
        .args(extra)
        .env("YATIMA_DRIVE_TEST_STUB_DIR", dir.path())
        .env(
            "YATIMA_DRIVE_TEST_STUB_BIN",
            env!("CARGO_BIN_EXE_llama-server-stub-drive"),
        )
        .env("YATIMA_GIT_DESCRIBE", "battery-provenance")
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .stdin(std::process::Stdio::null())
        .kill_on_drop(true)
        .spawn()
        .expect("spawn yatima-drive")
}

/// The reap oracle. pgrep's exit semantics: 0 = matched, 1 = no match; the
/// marker is the tempdir path baked into the stub child's argv.
async fn stub_children_alive(marker: &Path) -> bool {
    let output = tokio::process::Command::new("pgrep")
        .args(["-f", &marker.display().to_string()])
        .output()
        .await
        .expect("run pgrep");
    match output.status.code() {
        Some(0) => true,
        Some(1) => false,
        other => panic!(
            "pgrep could not answer (exit {other:?}): {}",
            String::from_utf8_lossy(&output.stderr)
        ),
    }
}

fn tape_lines(run_dir: &Path) -> Vec<serde_json::Value> {
    std::fs::read_to_string(run_dir.join("tape.jsonl"))
        .expect("tape exists")
        .lines()
        .map(|line| serde_json::from_str(line).expect("every complete line parses"))
        .collect()
}

fn summary(run_dir: &Path) -> serde_json::Value {
    serde_json::from_str(&std::fs::read_to_string(run_dir.join("summary.json")).expect("summary"))
        .expect("summary parses")
}

/// Records whose payload is a request of the named variant, by tape order.
fn request_seqs(lines: &[serde_json::Value], variant: &str) -> Vec<u64> {
    lines
        .iter()
        .skip(1)
        .filter(|line| {
            let request = &line["request"];
            request == variant || !request[variant].is_null()
        })
        .map(|line| line["seq"].as_u64().unwrap())
        .collect()
}

fn event_seqs(lines: &[serde_json::Value], variant: &str) -> Vec<u64> {
    lines
        .iter()
        .skip(1)
        .filter(|line| !line["event"][variant].is_null() || line["event"] == variant)
        .map(|line| line["seq"].as_u64().unwrap())
        .collect()
}

#[tokio::test(flavor = "multi_thread")]
async fn happy_path_orders_both_planes_and_accounts_the_run() {
    // upholds: TAPE-1 / HOST-3 — one observation-ordered tape, both
    // provenance layers distinct, a completed summary, exit 0, no child.
    let _serial = SESSION.lock().await;
    let dir = TempDir::new().unwrap();
    let run = run_driver(&dir, "muse-chat", "first question\nsecond question\n", &[]).await;
    assert_eq!(run.status.code(), Some(0), "stderr: {}", run.stderr);

    let lines = tape_lines(&run.run_dir);
    // Header: requested provenance only.
    let header = &lines[0];
    let origin = header["meta"]["origin"].as_str().unwrap();
    assert!(origin.starts_with("yatima-drive "), "{origin}");
    assert_eq!(header["meta"]["model"], "stub-muse");
    assert_eq!(header["meta"]["notes"]["agent_max_steps"], "16");
    assert_eq!(
        header["meta"]["notes"]["git_describe"], "battery-provenance",
        "invoker-supplied provenance rides the header verbatim"
    );
    // The observed layer: Ready carries the verified digest of the stub
    // bytes, byte for byte — never a header claim.
    use sha2::Digest;
    let expected = format!("{:x}", sha2::Sha256::digest(b"stub"));
    let ready: Vec<&serde_json::Value> = lines
        .iter()
        .filter(|line| !line["event"]["Ready"].is_null())
        .collect();
    assert_eq!(ready.len(), 1);
    assert_eq!(
        ready[0]["event"]["Ready"]["identity"]["VerifiedSha256"],
        serde_json::Value::String(expected)
    );

    // Sequence is monotone from 0 and each Submit precedes its settlement.
    let seqs: Vec<u64> = lines
        .iter()
        .skip(1)
        .map(|line| line["seq"].as_u64().unwrap())
        .collect();
    assert_eq!(seqs, (0..seqs.len() as u64).collect::<Vec<_>>());
    let submits = request_seqs(&lines, "Submit");
    let dones = event_seqs(&lines, "Done");
    assert_eq!(submits.len(), 2);
    assert_eq!(dones.len(), 2);
    assert!(submits[0] < dones[0] && submits[1] < dones[1]);
    assert!(dones[0] < submits[1], "turns are sequential");
    let shutdowns = request_seqs(&lines, "Shutdown");
    assert_eq!(shutdowns.len(), 1);
    let all_requests: Vec<u64> = ["Submit", "Grant", "Revoke", "Cancel", "Shutdown"]
        .iter()
        .flat_map(|variant| request_seqs(&lines, variant))
        .collect();
    assert_eq!(
        shutdowns[0],
        *all_requests.iter().max().unwrap(),
        "the semantic Shutdown is the final request (tail events may follow)"
    );

    let summary = summary(&run.run_dir);
    assert_eq!(summary["disposition"], "completed");
    assert_eq!(summary["requests"], 3, "two submits and the shutdown");
    assert_eq!(
        summary["records"].as_u64().unwrap(),
        summary["requests"].as_u64().unwrap() + summary["events"].as_u64().unwrap()
    );
    assert!(!stub_children_alive(dir.path()).await, "child reaped");
}

#[tokio::test(flavor = "multi_thread")]
async fn a_grant_line_precedes_its_dependent_submit_on_the_tape() {
    // upholds: CAP-3 / TAPE-1 — the ordered request plane is visible on the
    // tape: the scenario's grant rides ahead of the turn that needs it.
    let _serial = SESSION.lock().await;
    let dir = TempDir::new().unwrap();
    let run = run_driver(
        &dir,
        "muse-chat",
        "/grant https://example.com/\nuse the page\n",
        &[],
    )
    .await;
    assert_eq!(run.status.code(), Some(0), "stderr: {}", run.stderr);
    let lines = tape_lines(&run.run_dir);
    let grants = request_seqs(&lines, "Grant");
    let submits = request_seqs(&lines, "Submit");
    assert_eq!(grants.len(), 1);
    assert!(grants[0] < submits[0], "grant precedes its Submit");
}

#[tokio::test(flavor = "multi_thread")]
async fn a_timed_out_turn_cancels_and_the_next_line_still_runs() {
    // upholds: CANCEL-1 / TAPE-1 — the recorded semantic Cancel precedes
    // settlement, the second turn completes, and the code reports timeout.
    let _serial = SESSION.lock().await;
    let dir = TempDir::new().unwrap();
    let run = run_driver(
        &dir,
        "stall-then-answer",
        "first stalls\nsecond answers\n",
        &["--turn-within", "1"],
    )
    .await;
    assert_eq!(run.status.code(), Some(4), "stderr: {}", run.stderr);
    let lines = tape_lines(&run.run_dir);
    let cancels = request_seqs(&lines, "Cancel");
    let submits = request_seqs(&lines, "Submit");
    assert_eq!(cancels.len(), 1, "one cancel, for the stalled turn");
    assert_eq!(submits.len(), 2, "the run continued to the second line");
    assert_eq!(summary(&run.run_dir)["disposition"], "timeout");
    assert!(!stub_children_alive(dir.path()).await, "child reaped");
}

#[tokio::test(flavor = "multi_thread")]
async fn stop_on_error_ends_the_run_at_the_first_errored_turn() {
    let _serial = SESSION.lock().await;
    let dir = TempDir::new().unwrap();
    let run = run_driver(
        &dir,
        "error-then-answer",
        "first errors\nnever submitted\n",
        &["--stop-on-error"],
    )
    .await;
    assert_eq!(run.status.code(), Some(2), "stderr: {}", run.stderr);
    let lines = tape_lines(&run.run_dir);
    assert_eq!(request_seqs(&lines, "Submit").len(), 1, "one turn only");
    assert_eq!(summary(&run.run_dir)["disposition"], "errored");
}

#[tokio::test(flavor = "multi_thread")]
async fn a_fatal_run_exits_3_with_the_child_reaped() {
    // upholds: HOST-3 / LSRV-1 — the dead backend still converges on the
    // joined reap, and the tape closes with the honest disposition.
    let _serial = SESSION.lock().await;
    let dir = TempDir::new().unwrap();
    // The child's death settles the in-flight turn as an Error and the
    // host's synchronous Fatal follows it; the epilogue's tail drain must
    // record that Fatal and fold it into the verdict — one prompt, no
    // sacrificial next turn.
    let run = run_driver(&dir, "die-after-ready", "one question\n", &[]).await;
    assert_eq!(run.status.code(), Some(3), "stderr: {}", run.stderr);
    let lines = tape_lines(&run.run_dir);
    assert_eq!(
        event_seqs(&lines, "Fatal").len(),
        1,
        "the terminal Fatal rides the tape"
    );
    let s = summary(&run.run_dir);
    assert_eq!(s["disposition"], "fatal");
    assert_eq!(s["fatal"], true, "the accounting sees the Fatal too");
    assert!(!stub_children_alive(dir.path()).await, "child reaped");
}

#[tokio::test(flavor = "multi_thread")]
async fn a_trailing_grant_still_tapes_the_hosts_reply() {
    // upholds: TAPE-1 / CAP-3 — a grant-only scenario has no turn wait to
    // absorb the host's Grants reply; the epilogue tail drain records it.
    let _serial = SESSION.lock().await;
    let dir = TempDir::new().unwrap();
    let run = run_driver(&dir, "muse-chat", "/grant https://example.com/\n", &[]).await;
    assert_eq!(run.status.code(), Some(0), "stderr: {}", run.stderr);
    let lines = tape_lines(&run.run_dir);
    assert!(
        !event_seqs(&lines, "Grants").is_empty(),
        "the Grants reply is on the tape"
    );
    let s = summary(&run.run_dir);
    assert_eq!(s["disposition"], "completed");
    assert!(
        s["grants_events"].as_u64().unwrap() >= 1,
        "the accounting sees the reply"
    );
    assert!(!stub_children_alive(dir.path()).await, "child reaped");
}

#[tokio::test(flavor = "multi_thread")]
async fn startup_failure_after_the_recorder_exits_1_with_a_summary() {
    // upholds: HOST-3 / TAPE-1 — the recorder exists before the host, so a
    // startup failure still consumes it with the honest disposition.
    let _serial = SESSION.lock().await;
    let dir = TempDir::new().unwrap();
    let child = {
        std::fs::write(dir.path().join("never-ready.gguf"), b"stub").unwrap();
        let scenario_path = dir.path().join("scenario.txt");
        std::fs::write(&scenario_path, "unreached\n").unwrap();
        tokio::process::Command::new(env!("CARGO_BIN_EXE_yatima-drive"))
            .arg(scenario_path)
            .arg("--tape")
            .arg(dir.path().join("run"))
            .env("YATIMA_DRIVE_TEST_STUB_DIR", dir.path())
            .env(
                "YATIMA_DRIVE_TEST_STUB_BIN",
                env!("CARGO_BIN_EXE_llama-server-stub-drive"),
            )
            .env("YATIMA_DRIVE_TEST_STUB_READY_MS", "300")
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .stdin(std::process::Stdio::null())
            .kill_on_drop(true)
            .spawn()
            .expect("spawn yatima-drive")
    };
    let output = tokio::time::timeout(WITHIN, child.wait_with_output())
        .await
        .expect("bounded exit")
        .expect("collect output");
    assert_eq!(output.status.code(), Some(1));
    let run_dir = PathBuf::from(
        String::from_utf8_lossy(&output.stdout)
            .lines()
            .next()
            .expect("the run directory is on stdout"),
    );
    assert_eq!(summary(&run_dir)["disposition"], "startup-error");
    assert!(!stub_children_alive(dir.path()).await, "child reaped");
}

/// A one-shot hermetic SearXNG-shaped endpoint: std TCP, no dependencies —
/// answers every GET with one fixed JSON result set, then keeps serving
/// until dropped via the returned shutdown flag.
fn searxng_stub() -> (String, std::sync::Arc<std::sync::atomic::AtomicBool>) {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind searxng stub");
    let addr = listener.local_addr().unwrap();
    let stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let stop_thread = stop.clone();
    listener.set_nonblocking(true).ok();
    std::thread::spawn(move || {
        use std::io::{Read, Write};
        let body = r#"{"results":[{"title":"Antikythera mechanism","url":"https://en.example/wiki/Antikythera","content":"an ancient analog computer"}]}"#;
        let response = format!(
            "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{}",
            body.len(),
            body
        );
        while !stop_thread.load(std::sync::atomic::Ordering::Acquire) {
            match listener.accept() {
                Ok((mut stream, _)) => {
                    let mut buf = [0u8; 2048];
                    let _ = stream.read(&mut buf);
                    let _ = stream.write_all(response.as_bytes());
                }
                Err(_) => std::thread::sleep(Duration::from_millis(20)),
            }
        }
    });
    (format!("http://{addr}/search"), stop)
}

#[tokio::test(flavor = "multi_thread")]
async fn the_shipped_binary_composes_search_without_minting_authority() {
    // upholds: CAP-2/CAP-3 (R1a) — the real binary, an env-injected
    // hermetic endpoint, and a scripted model: the tape shows the search
    // call succeed, and no result page is ever requested (nothing was
    // granted, and the found origin resolves nowhere in this sandbox).
    let _serial = SESSION.lock().await;
    let (endpoint, stop) = searxng_stub();
    let dir = TempDir::new().unwrap();
    std::fs::write(dir.path().join("search-round.gguf"), b"stub").unwrap();
    let scenario_path = dir.path().join("scenario.txt");
    std::fs::write(
        &scenario_path,
        "find me writeups on the antikythera mechanism\n",
    )
    .unwrap();
    let child = tokio::process::Command::new(env!("CARGO_BIN_EXE_yatima-drive"))
        .arg(&scenario_path)
        .arg("--tape")
        .arg(dir.path().join("run"))
        .env("YATIMA_DRIVE_TEST_STUB_DIR", dir.path())
        .env(
            "YATIMA_DRIVE_TEST_STUB_BIN",
            env!("CARGO_BIN_EXE_llama-server-stub-drive"),
        )
        .env("YATIMA_GIT_DESCRIBE", "battery-provenance")
        .env("YATIMA_SEARCH_URL", &endpoint)
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .stdin(std::process::Stdio::null())
        .kill_on_drop(true)
        .spawn()
        .expect("spawn yatima-drive");
    let output = tokio::time::timeout(WITHIN, child.wait_with_output())
        .await
        .expect("bounded exit")
        .expect("collect output");
    stop.store(true, std::sync::atomic::Ordering::Release);
    assert_eq!(
        output.status.code(),
        Some(0),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let run_dir = PathBuf::from(
        String::from_utf8_lossy(&output.stdout)
            .lines()
            .next()
            .expect("run dir on stdout"),
    );
    let lines = tape_lines(&run_dir);
    let notes: Vec<(String, String)> = lines
        .iter()
        .filter_map(|l| {
            let note = &l["event"]["ToolNote"];
            Some((
                note["kind"].as_str()?.to_string(),
                note["text"].as_str()?.to_string(),
            ))
        })
        .collect();
    let call_at = notes
        .iter()
        .position(|(k, t)| k == "Call" && t.contains("web_search"))
        .unwrap_or_else(|| panic!("the search call is on the tape: {notes:?}"));
    assert!(
        matches!(notes.get(call_at + 1), Some((k, _)) if k == "Success"),
        "the search succeeded (long results summarize as a char count): {notes:?}"
    );
    assert!(
        !notes.iter().any(|(k, _)| k == "Failure"),
        "no tool failed: {notes:?}"
    );
    assert!(
        !notes.iter().any(|(_, t)| t.contains("read_page")),
        "no page fetch followed — searching mints no authority: {notes:?}"
    );
    assert!(!stub_children_alive(dir.path()).await, "child reaped");
}

/// The journey's whole hermetic web: search endpoint, a 403-serving page,
/// a readable article with one image, and the image itself — one origin,
/// one listener, ephemeral port. std TCP, no dependencies.
fn journey_web_stub() -> (String, std::sync::Arc<std::sync::atomic::AtomicBool>) {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind journey stub");
    let origin = format!("http://{}", listener.local_addr().unwrap());
    let stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let stop_thread = stop.clone();
    let search_body = format!(
        r#"{{"results":[
            {{"title":"Antikythera overview","url":"{origin}/forbidden","content":"a refusing source"}},
            {{"title":"Antikythera fragments","url":"{origin}/wiki/Antikythera","content":"an ancient analog computer"}}
        ]}}"#
    );
    listener.set_nonblocking(true).ok();
    std::thread::spawn(move || {
        use std::io::{Read, Write};
        while !stop_thread.load(std::sync::atomic::Ordering::Acquire) {
            match listener.accept() {
                Ok((mut stream, _)) => {
                    // The accepted stream inherits the listener's
                    // non-blocking mode (observed: an early read saw zero
                    // bytes and routed a real request to 404); force
                    // blocking and read until the request line is whole.
                    stream.set_nonblocking(false).ok();
                    let mut buf = Vec::new();
                    let mut chunk = [0u8; 4096];
                    while !buf.windows(4).any(|w| w == b"\r\n\r\n") {
                        match stream.read(&mut chunk) {
                            Ok(0) | Err(_) => break,
                            Ok(count) => buf.extend_from_slice(&chunk[..count]),
                        }
                    }
                    let path = String::from_utf8_lossy(&buf)
                        .lines()
                        .next()
                        .and_then(|l| l.split_whitespace().nth(1).map(str::to_string))
                        .unwrap_or_default();
                    let (status, content_type, body) = match path.split('?').next() {
                        Some("/search") => ("200 OK", "application/json", search_body.clone()),
                        Some("/forbidden") => {
                            ("403 Forbidden", "text/plain", "bots begone".to_string())
                        }
                        Some("/wiki/Antikythera") => (
                            "200 OK",
                            "text/html",
                            "<html><body><article><h1>Antikythera mechanism</h1>\
                             <p>An ancient Greek analog computer recovered from a wreck.</p>\
                             <img src=\"/img/fragment.svg\" alt=\"the largest fragment\">\
                             </article></body></html>"
                                .to_string(),
                        ),
                        Some("/img/fragment.svg") => (
                            "200 OK",
                            "image/svg+xml",
                            "<svg xmlns=\"http://www.w3.org/2000/svg\" \
                             width=\"4\" height=\"4\"/>"
                                .to_string(),
                        ),
                        _ => ("404 Not Found", "text/plain", "no such page".to_string()),
                    };
                    let response = format!(
                        "HTTP/1.1 {status}\r\ncontent-type: {content_type}\r\n\
                         content-length: {}\r\nconnection: close\r\n\r\n{body}",
                        body.len()
                    );
                    let _ = stream.write_all(response.as_bytes());
                }
                Err(_) => std::thread::sleep(Duration::from_millis(10)),
            }
        }
    });
    (origin, stop)
}

#[tokio::test(flavor = "multi_thread")]
async fn the_research_journey_survives_a_403_and_displays_an_image() {
    // The acceptance harness for the whole web-research journey (fast and
    // hermetic): search → report → user grants → a 403'd fetch survived
    // IN-TURN with the redirecting ERR-1 error (never a grant beg) → the
    // good page's images listed → one displayed → grounded close. Also
    // witnesses HOST-6 end to end: the grant note rides the model's next
    // prompt through the shipped binary.
    let _serial = SESSION.lock().await;
    let (origin, stop) = journey_web_stub();
    let dir = TempDir::new().unwrap();
    std::fs::write(dir.path().join("research-journey.gguf"), b"stub").unwrap();
    let scenario_path = dir.path().join("scenario.txt");
    std::fs::write(
        &scenario_path,
        format!(
            "find me good writeups on the antikythera mechanism\n\
             /grant {origin}\n\
             read the first writeup and show me an image from it\n"
        ),
    )
    .unwrap();
    let child = tokio::process::Command::new(env!("CARGO_BIN_EXE_yatima-drive"))
        .arg(&scenario_path)
        .arg("--tape")
        .arg(dir.path().join("run"))
        .env("YATIMA_DRIVE_TEST_STUB_DIR", dir.path())
        .env(
            "YATIMA_DRIVE_TEST_STUB_BIN",
            env!("CARGO_BIN_EXE_llama-server-stub-drive"),
        )
        .env("YATIMA_GIT_DESCRIBE", "battery-provenance")
        .env("YATIMA_SEARCH_URL", format!("{origin}/search"))
        .env("YATIMA_STUB_FORBIDDEN_URL", format!("{origin}/forbidden"))
        .env("YATIMA_STUB_PAGE_URL", format!("{origin}/wiki/Antikythera"))
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .stdin(std::process::Stdio::null())
        .kill_on_drop(true)
        .spawn()
        .expect("spawn yatima-drive");
    let output = tokio::time::timeout(WITHIN, child.wait_with_output())
        .await
        .expect("bounded exit")
        .expect("collect output");
    stop.store(true, std::sync::atomic::Ordering::Release);
    assert_eq!(
        output.status.code(),
        Some(0),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let run_dir = PathBuf::from(
        String::from_utf8_lossy(&output.stdout)
            .lines()
            .next()
            .expect("run dir on stdout"),
    );
    let lines = tape_lines(&run_dir);
    let notes: Vec<(String, String)> = lines
        .iter()
        .filter_map(|l| {
            let note = &l["event"]["ToolNote"];
            Some((
                note["kind"].as_str()?.to_string(),
                note["text"].as_str()?.to_string(),
            ))
        })
        .collect();
    let position = |pred: &dyn Fn(&(String, String)) -> bool| {
        notes
            .iter()
            .position(pred)
            .unwrap_or_else(|| panic!("expected note missing: {notes:?}"))
    };
    let search = position(&|(k, t)| k == "Call" && t.contains("web_search"));
    assert!(matches!(&notes[search + 1], (k, _) if k == "Success"));
    // The 403 leg: the failure is the redirecting ERR-1 error, and it
    // never suggests granting.
    let refused = position(&|(k, t)| k == "Failure" && t.contains("/forbidden"));
    assert!(
        notes[refused]
            .1
            .contains("the origin is granted, but the server refused")
            && notes[refused].1.contains("HTTP 403"),
        "{}",
        notes[refused].1
    );
    assert!(
        !notes[refused].1.contains("/grant"),
        "a server refusal must not beg: {}",
        notes[refused].1
    );
    // The journey continues in the same turn: article images, then the
    // display.
    let listing =
        position(&|(k, t)| k == "Call" && t.contains("read_page") && t.contains("images_only"));
    assert!(matches!(&notes[listing + 1], (k, _) if k == "Success"));
    let display = position(&|(k, t)| k == "Call" && t.contains("read_image"));
    assert!(
        matches!(&notes[display + 1], (k, _) if k == "Success"),
        "{notes:?}"
    );
    assert!(search < refused && refused < listing && listing < display);
    assert_eq!(
        notes.iter().filter(|(k, _)| k == "Failure").count(),
        1,
        "the 403 is the only failure: {notes:?}"
    );
    assert!(
        !event_seqs(&lines, "Image").is_empty(),
        "the displayed image rides the tape as a typed artifact event"
    );
    // R2, end to end: the settled search turn's answer named the page
    // URL, so the host emitted its typed GrantProposal (canonical origin,
    // no prose parse) ahead of the user's Grant request on the tape.
    let proposals = event_seqs(&lines, "GrantProposal");
    let grants = request_seqs(&lines, "Grant");
    assert!(
        !proposals.is_empty(),
        "the typed proposal rides the tape: {lines:?}"
    );
    assert!(
        proposals[0] < grants[0],
        "propose, then the user's tap/command grants"
    );
    let proposed_origin = lines
        .iter()
        .find_map(|l| l["event"]["GrantProposal"]["origins"][0].as_str())
        .expect("proposal carries origins");
    assert!(
        origin.starts_with(proposed_origin),
        "the proposal names the page origin: {proposed_origin} vs {origin}"
    );

    // R4's deterministic acceptance order, on the shipped binary's tape:
    // search → GrantProposal(A) → Grant(A) → read_page(A) → derived
    // image(B) → displayed — with no request to A before its grant, no
    // grant for B ever, and the tape's Image record carrying the CAP-4
    // derivation edge back to the granted page.
    let search_seq = lines
        .iter()
        .find(|l| {
            l["event"]["ToolNote"]["text"]
                .as_str()
                .is_some_and(|t| t.contains("web_search"))
        })
        .and_then(|l| l["seq"].as_u64())
        .expect("search on tape");
    let first_read_seq = lines
        .iter()
        .find(|l| {
            l["event"]["ToolNote"]["text"]
                .as_str()
                .is_some_and(|t| t.contains("read_page"))
        })
        .and_then(|l| l["seq"].as_u64())
        .expect("read_page on tape");
    assert!(
        search_seq < proposals[0] && proposals[0] < grants[0] && grants[0] < first_read_seq,
        "search → proposal → grant → read, in tape order"
    );
    // (Cross-origin derivation — an image host that is never granted —
    // is witnessed at the lib layer, where the loopback test seam is
    // cfg(test)-only; no shipped binary carries a CAP-4 switch.)
    let image_record = lines
        .iter()
        .find(|l| !l["event"]["Image"].is_null())
        .expect("the displayed image rides the tape");
    assert_eq!(
        image_record["event"]["Image"]["derived_from"]
            .as_str()
            .unwrap_or_default(),
        format!("{origin}/wiki/Antikythera"),
        "the CAP-4 derivation edge is recorded beside the image identity"
    );

    // HOST-6, end to end: the second turn's first completion prompt
    // carries the grant note the host queued for the model.
    let prompt3 = std::fs::read_to_string(dir.path().join("research-journey.prompt3"))
        .expect("the stub captured turn 2's prompt");
    assert!(
        prompt3.contains("the user granted read access to"),
        "the grant note reaches the model's next prompt: {prompt3}"
    );
    assert!(!stub_children_alive(dir.path()).await, "child reaped");
}

#[tokio::test(flavor = "multi_thread")]
async fn sigint_mid_turn_interrupts_cleanly() {
    // upholds: CANCEL-1 / HOST-3 / TAPE-1 — the cooperative interrupt
    // cancels the recorded way, the epilogue runs whole, exit is 130, and
    // no managed child survives (the CLI battery's oracle, hermetic here).
    let _serial = SESSION.lock().await;
    let dir = TempDir::new().unwrap();
    let child = spawn_driver(
        &dir,
        "stall-then-answer",
        "this turn dribbles until interrupted\n",
        &["--turn-within", "600"],
    );
    let pid = child.id().expect("child pid");

    // Wait until the turn is provably mid-flight: its Submit is on the tape.
    let tape = dir.path().join("run").join("tape.jsonl");
    tokio::time::timeout(WITHIN, async {
        loop {
            if let Ok(text) = std::fs::read_to_string(&tape) {
                if text.contains("\"Submit\"") || text.contains("Submit") {
                    break;
                }
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
    })
    .await
    .expect("the Submit record must appear");
    // And give the stub a beat to be mid-dribble.
    tokio::time::sleep(Duration::from_millis(200)).await;

    let signalled = tokio::process::Command::new("kill")
        .args(["-INT", &pid.to_string()])
        .status()
        .await
        .expect("send SIGINT");
    assert!(signalled.success());

    let output = tokio::time::timeout(WITHIN, child.wait_with_output())
        .await
        .expect("bounded exit after SIGINT")
        .expect("collect output");
    assert_eq!(output.status.code(), Some(130), "the Unix convention");

    let run_dir = dir.path().join("run");
    assert_eq!(summary(&run_dir)["disposition"], "interrupted");
    let lines = tape_lines(&run_dir);
    assert_eq!(request_seqs(&lines, "Cancel").len(), 1, "recorded cancel");
    assert_eq!(
        request_seqs(&lines, "Shutdown").len(),
        1,
        "recorded shutdown"
    );
    assert!(!stub_children_alive(dir.path()).await, "child reaped");
}

/// The R4 hermetic two-origin web: page host A (search, 403 route,
/// article) and a separate image host B the article embeds — B is never
/// granted, so a displayed image proves CAP-4 derivation through the
/// shipped binary. Feature-gated with the seam it needs.
#[cfg(feature = "hermetic-acceptance")]
fn r4_web_stub() -> (
    String,
    String,
    std::sync::Arc<std::sync::atomic::AtomicBool>,
) {
    use std::io::{Read, Write};
    let serve = |listener: std::net::TcpListener,
                 stop: std::sync::Arc<std::sync::atomic::AtomicBool>,
                 route: Box<dyn Fn(&str) -> (String, String, String) + Send>| {
        listener.set_nonblocking(true).ok();
        std::thread::spawn(move || {
            while !stop.load(std::sync::atomic::Ordering::Acquire) {
                match listener.accept() {
                    Ok((mut stream, _)) => {
                        stream.set_nonblocking(false).ok();
                        let mut buf = Vec::new();
                        let mut chunk = [0u8; 4096];
                        while !buf.windows(4).any(|w| w == b"\r\n\r\n") {
                            match stream.read(&mut chunk) {
                                Ok(0) | Err(_) => break,
                                Ok(count) => buf.extend_from_slice(&chunk[..count]),
                            }
                        }
                        let path = String::from_utf8_lossy(&buf)
                            .lines()
                            .next()
                            .and_then(|l| l.split_whitespace().nth(1).map(str::to_string))
                            .unwrap_or_default();
                        let (status, content_type, body) =
                            route(path.split('?').next().unwrap_or(""));
                        let _ = stream.write_all(
                            format!(
                                "HTTP/1.1 {status}\r\ncontent-type: {content_type}\r\n\
                                 content-length: {}\r\nconnection: close\r\n\r\n{body}",
                                body.len()
                            )
                            .as_bytes(),
                        );
                    }
                    Err(_) => std::thread::sleep(Duration::from_millis(10)),
                }
            }
        });
    };
    let page_listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind page stub");
    let origin = format!("http://{}", page_listener.local_addr().unwrap());
    let image_listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind image stub");
    let image_origin = format!("http://{}", image_listener.local_addr().unwrap());
    let stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let search_body = format!(
        r#"{{"results":[{{"title":"Antikythera fragments","url":"{origin}/wiki/Antikythera","content":"an ancient analog computer"}}]}}"#
    );
    let page_html = format!(
        "<html><body><article><h1>Antikythera mechanism</h1>\
         <p>An ancient Greek analog computer recovered from a wreck.</p>\
         <img src=\"{image_origin}/img/fragment.svg\" alt=\"the largest fragment\">\
         </article></body></html>"
    );
    serve(
        page_listener,
        stop.clone(),
        Box::new(move |path| match path {
            "/search" => (
                "200 OK".into(),
                "application/json".into(),
                search_body.clone(),
            ),
            "/forbidden" => (
                "403 Forbidden".into(),
                "text/plain".into(),
                "bots begone".into(),
            ),
            "/wiki/Antikythera" => ("200 OK".into(), "text/html".into(), page_html.clone()),
            _ => (
                "404 Not Found".into(),
                "text/plain".into(),
                "no such page".into(),
            ),
        }),
    );
    serve(
        image_listener,
        stop.clone(),
        Box::new(|_| {
            (
                "200 OK".into(),
                "image/svg+xml".into(),
                "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"4\" height=\"4\"/>".into(),
            )
        }),
    );
    (origin, image_origin, stop)
}

/// R4's deterministic cross-origin acceptance (feature-built, per the
/// contract): the shipped binary composes `web_search → GrantProposal(A)
/// → Grant(A) → read_page(A) → derived image from ungranted origin B →
/// displayed`, in tape order, with no grant for B ever and the CAP-4
/// derivation edge recorded. The seam admitting loopback derivation is
/// compiled in ONLY under this feature chain; an ordinary build has no
/// such branch (`environment_cannot_weaken_cap4_in_the_shipped_toolset`).
/// Run: cargo test -p yatima-drive --features hermetic-acceptance
#[cfg(feature = "hermetic-acceptance")]
#[tokio::test(flavor = "multi_thread")]
async fn the_r4_deterministic_acceptance_composes_cross_origin_derivation() {
    let _serial = SESSION.lock().await;
    let (origin, image_origin, stop) = r4_web_stub();
    let dir = TempDir::new().unwrap();
    std::fs::write(dir.path().join("research-journey.gguf"), b"stub").unwrap();
    let scenario_path = dir.path().join("scenario.txt");
    std::fs::write(
        &scenario_path,
        format!(
            "find me good writeups on the antikythera mechanism\n\
             /grant {origin}\n\
             read the first writeup and show me an image from it\n"
        ),
    )
    .unwrap();
    let child = tokio::process::Command::new(env!("CARGO_BIN_EXE_yatima-drive"))
        .arg(&scenario_path)
        .arg("--tape")
        .arg(dir.path().join("run"))
        .env("YATIMA_DRIVE_TEST_STUB_DIR", dir.path())
        .env(
            "YATIMA_DRIVE_TEST_STUB_BIN",
            env!("CARGO_BIN_EXE_llama-server-stub-drive"),
        )
        .env("YATIMA_GIT_DESCRIBE", "battery-provenance")
        .env("YATIMA_SEARCH_URL", format!("{origin}/search"))
        .env("YATIMA_STUB_FORBIDDEN_URL", format!("{origin}/forbidden"))
        .env("YATIMA_STUB_PAGE_URL", format!("{origin}/wiki/Antikythera"))
        .env("YATIMA_TEST_ALLOW_LOOPBACK_DERIVATION", "1")
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .stdin(std::process::Stdio::null())
        .kill_on_drop(true)
        .spawn()
        .expect("spawn yatima-drive");
    let output = tokio::time::timeout(WITHIN, child.wait_with_output())
        .await
        .expect("bounded exit")
        .expect("collect output");
    stop.store(true, std::sync::atomic::Ordering::Release);
    assert_eq!(
        output.status.code(),
        Some(0),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let run_dir = PathBuf::from(
        String::from_utf8_lossy(&output.stdout)
            .lines()
            .next()
            .expect("run dir on stdout"),
    );
    let lines = tape_lines(&run_dir);
    let search_seq = lines
        .iter()
        .find(|l| {
            l["event"]["ToolNote"]["text"]
                .as_str()
                .is_some_and(|t| t.contains("web_search"))
        })
        .and_then(|l| l["seq"].as_u64())
        .expect("search on tape");
    let proposals = event_seqs(&lines, "GrantProposal");
    let grants = request_seqs(&lines, "Grant");
    let first_read_seq = lines
        .iter()
        .find(|l| {
            l["event"]["ToolNote"]["text"]
                .as_str()
                .is_some_and(|t| t.contains("read_page"))
        })
        .and_then(|l| l["seq"].as_u64())
        .expect("read_page on tape");
    assert!(
        search_seq < proposals[0] && proposals[0] < grants[0] && grants[0] < first_read_seq,
        "search → proposal → grant → read, in tape order"
    );
    assert!(
        !lines.iter().any(|l| {
            l["request"]["Grant"]["origin"]
                .as_str()
                .is_some_and(|o| o.starts_with(&image_origin))
        }),
        "the image host is never granted — derivation, not authority"
    );
    let image_record = lines
        .iter()
        .find(|l| !l["event"]["Image"].is_null())
        .expect("the cross-origin derived image rides the tape");
    assert!(
        image_record["event"]["Image"]["source"]
            .as_str()
            .is_some_and(|s| s.starts_with(&image_origin)),
        "the displayed bytes came from ungranted origin B"
    );
    assert_eq!(
        image_record["event"]["Image"]["derived_from"]
            .as_str()
            .unwrap_or_default(),
        format!("{origin}/wiki/Antikythera"),
        "the CAP-4 derivation edge is recorded beside the image identity"
    );
    assert!(!stub_children_alive(dir.path()).await, "child reaped");
}
