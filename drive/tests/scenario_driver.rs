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
    assert_eq!(header["meta"]["notes"]["agent_max_steps"], "12");
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
