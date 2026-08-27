//! Live CLI acceptance for the managed Muse agent milestone: the real
//! `yatima` binary, end to end. Ignored in the offline gate; run by exact name
//! under `cargo test --release` (debug SHA-256 over 17 GB is prohibitively
//! slow), and only when no other Muse / llama-server session is using the
//! machine.

use anyhow::{bail, Context};
use std::process::Stdio;
use std::time::Duration;
use tokio::io::AsyncReadExt;

const LIVE_WITHIN: Duration = Duration::from_secs(15 * 60);
const INTERRUPT_WITHIN: Duration = Duration::from_secs(20);
const STARTUP_WITHIN: Duration = Duration::from_secs(2 * 60);

async fn matching_pids(pattern: &str) -> anyhow::Result<Vec<u32>> {
    let output = tokio::process::Command::new("pgrep")
        .args(["-f", pattern])
        .output()
        .await?;
    if !output.status.success() {
        return Ok(Vec::new());
    }
    String::from_utf8(output.stdout)?
        .lines()
        .map(|line| line.parse().context("parse pgrep pid"))
        .collect()
}

async fn signal(pid: u32, name: &str) -> anyhow::Result<bool> {
    Ok(tokio::process::Command::new("/bin/kill")
        .args([name, &pid.to_string()])
        .status()
        .await?
        .success())
}

#[tokio::test]
#[ignore = "run with --release; requires llama-server and the local Muse Glimmer GGUF"]
async fn live_cli_managed_muse_agent() -> anyhow::Result<()> {
    // upholds: LSRV-1 / LSRV-5 / PROFILE-1 / PROFILE-2 — the complete Stage 4b CLI
    // composition, unreachable from library tests: the shared backend
    // resolver, the verified managed spawn (banner and all), the ATEM tool
    // round under --root, the framing-free answer on stdout, and the child
    // reap, all through the shipped binary.
    let existing = matching_pids("llama-server").await?;
    if !existing.is_empty() {
        bail!("live test requires an idle machine; found llama-server pids {existing:?}");
    }

    let root = tempfile::tempdir()?;
    std::fs::write(
        root.path().join("README.md"),
        "yatima is a Rust runtime for language-integrated LLM inference: local \
         Candle engines and managed llama-server children behind one Completer.",
    )?;

    let mut command = tokio::process::Command::new(env!("CARGO_BIN_EXE_yatima"));
    command
        .args([
            "agent",
            "--profile",
            "muse-glimmer",
            "--offline",
            "--verbose",
        ])
        .arg("--root")
        .arg(root.path())
        .args([
            "--prompt",
            "Read README.md and say what yatima is in two sentences.",
        ])
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true);
    let mut child = command.spawn().context("spawn yatima CLI")?;
    let pid = child.id().context("yatima CLI has no process id")?;
    let mut child_stdout = child.stdout.take().context("capture yatima stdout")?;
    let mut child_stderr = child.stderr.take().context("capture yatima stderr")?;
    let stdout_task = tokio::spawn(async move {
        let mut bytes = Vec::new();
        child_stdout.read_to_end(&mut bytes).await?;
        Ok::<_, std::io::Error>(bytes)
    });
    let stderr_task = tokio::spawn(async move {
        let mut bytes = Vec::new();
        child_stderr.read_to_end(&mut bytes).await?;
        Ok::<_, std::io::Error>(bytes)
    });

    let (status, timed_out) = match tokio::time::timeout(LIVE_WITHIN, child.wait()).await {
        Ok(status) => (status?, false),
        Err(_) => {
            // Exercise the product's graceful interrupt path before resorting
            // to force. Retaining `child` here is essential: dropping an
            // `output()` future would lose the only reap handle and could
            // orphan the managed llama-server.
            let _ = signal(pid, "-INT").await?;
            match tokio::time::timeout(INTERRUPT_WITHIN, child.wait()).await {
                Ok(status) => (status?, true),
                Err(_) => {
                    // A regression in the interrupt handler must still leave
                    // the developer's machine clean. This test starts only
                    // when no llama-server exists, so every match is ours.
                    for server_pid in matching_pids("llama-server").await? {
                        let _ = signal(server_pid, "-KILL").await?;
                    }
                    let _ = child.start_kill();
                    let status = tokio::time::timeout(INTERRUPT_WITHIN, child.wait())
                        .await
                        .context("force-cleanup of timed-out yatima CLI also timed out")??;
                    (status, true)
                }
            }
        }
    };

    let stdout = String::from_utf8(stdout_task.await.context("join stdout reader")??)?;
    let stderr = String::from_utf8(stderr_task.await.context("join stderr reader")??)?;
    let leaked = matching_pids("llama-server").await?;
    if !leaked.is_empty() {
        bail!("llama-server survived the CLI run: {leaked:?}");
    }
    if timed_out {
        bail!(
            "live CLI agent run exceeded {} seconds; interrupt cleanup exited {status:?}\nstderr:\n{stderr}",
            LIVE_WITHIN.as_secs()
        );
    }
    assert!(status.success(), "exit {:?}\nstderr:\n{stderr}", status);

    // The verified banner proves this went through the managed composition —
    // neither attached nor the Candle engine prints it.
    assert!(
        stderr.contains("managed llama-server [muse-glimmer]; verified sha256"),
        "stderr:\n{stderr}"
    );
    assert!(
        stderr.contains(&format!("tools rooted at {}", root.path().display())),
        "stderr:\n{stderr}"
    );
    // --verbose surfaces both structured halves of the tool round from the
    // run transcript, rather than merely mentioning the tool in the prompt.
    assert!(
        stderr.contains("── Assistant ──\nread_file {")
            && stderr.contains("── Tool ──\n[read_file ok]"),
        "stderr:\n{stderr}"
    );
    assert!(stderr.contains(", Final]"), "stderr:\n{stderr}");

    let answer = stdout.trim();
    assert!(!answer.is_empty(), "stdout empty; stderr:\n{stderr}");
    for framing in ["<|start|>", "<|message|>", "<|eom|>", "<|eot|>", "atem:"] {
        assert!(
            !answer.contains(framing),
            "answer leaked {framing}: {answer}"
        );
    }

    Ok(())
}

#[tokio::test]
#[ignore = "run with --release; requires llama-server and the local Muse Glimmer GGUF"]
async fn live_cli_interrupt_reaps_managed_server() -> anyhow::Result<()> {
    // upholds: CANCEL-1 / LSRV-1 — SIGINT becomes cooperative cancellation;
    // the CLI returns 130 only after its managed child has been reaped.
    let existing = matching_pids("llama-server").await?;
    if !existing.is_empty() {
        bail!("live test requires an idle machine; found llama-server pids {existing:?}");
    }

    let root = tempfile::tempdir()?;
    std::fs::write(
        root.path().join("README.md"),
        "Yatima is a Rust LLM runtime.",
    )?;
    let mut command = tokio::process::Command::new(env!("CARGO_BIN_EXE_yatima"));
    // This lifecycle witness deliberately uses ordinary managed mode over the
    // same exact cached GGUF. The verified-profile composition is covered by
    // `live_cli_managed_muse_agent`; skipping its debug-build SHA-256 pass here
    // keeps this test focused on signal-to-reap behavior.
    command
        .args([
            "agent",
            "--backend",
            "llama-server",
            "--repo",
            "meta-models/Muse-Glimmer-30B-GGUF",
            "--gguf",
            "Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf",
            "--format",
            "muse-glimmer",
            "--offline",
        ])
        .arg("--root")
        .arg(root.path())
        .args(["--prompt", "Read README.md and summarize it."])
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true);
    let mut child = command.spawn().context("spawn yatima CLI")?;
    let pid = child.id().context("yatima CLI has no process id")?;
    let mut child_stdout = child.stdout.take().context("capture yatima stdout")?;
    let mut child_stderr = child.stderr.take().context("capture yatima stderr")?;
    let stdout_task = tokio::spawn(async move {
        let mut bytes = Vec::new();
        child_stdout.read_to_end(&mut bytes).await?;
        Ok::<_, std::io::Error>(bytes)
    });
    let (banner_tx, banner_rx) = tokio::sync::oneshot::channel();
    let stderr_task = tokio::spawn(async move {
        let mut bytes = Vec::new();
        let mut chunk = [0u8; 4096];
        let mut banner_tx = Some(banner_tx);
        loop {
            let count = child_stderr.read(&mut chunk).await?;
            if count == 0 {
                break;
            }
            bytes.extend_from_slice(&chunk[..count]);
            if banner_tx.is_some()
                && String::from_utf8_lossy(&bytes).contains("managed llama-server [")
            {
                let _ = banner_tx.take().expect("checked above").send(());
            }
        }
        Ok::<_, std::io::Error>(bytes)
    });

    let reached_server = tokio::time::timeout(STARTUP_WITHIN, banner_rx).await;
    if !matches!(reached_server, Ok(Ok(()))) {
        for server_pid in matching_pids("llama-server").await? {
            let _ = signal(server_pid, "-KILL").await?;
        }
        let _ = child.start_kill();
        let _ = tokio::time::timeout(INTERRUPT_WITHIN, child.wait()).await;
        let stderr = String::from_utf8(stderr_task.await.context("join stderr reader")??)?;
        let _ = stdout_task.await;
        bail!("managed server did not become ready within the startup bound\nstderr:\n{stderr}");
    }

    assert!(signal(pid, "-INT").await?, "send SIGINT to yatima CLI");
    let status = match tokio::time::timeout(INTERRUPT_WITHIN, child.wait()).await {
        Ok(status) => status?,
        Err(_) => {
            for server_pid in matching_pids("llama-server").await? {
                let _ = signal(server_pid, "-KILL").await?;
            }
            let _ = child.start_kill();
            let _ = tokio::time::timeout(INTERRUPT_WITHIN, child.wait()).await;
            bail!("yatima CLI did not finish bounded interrupt cleanup");
        }
    };
    let stdout = String::from_utf8(stdout_task.await.context("join stdout reader")??)?;
    let stderr = String::from_utf8(stderr_task.await.context("join stderr reader")??)?;

    assert_eq!(
        status.code(),
        Some(130),
        "stderr:\n{stderr}\nstdout:\n{stdout}"
    );
    assert!(
        stderr.contains("interrupt received; stopping managed command")
            && stderr.contains("managed llama-server stopped"),
        "stderr:\n{stderr}"
    );
    let leaked = matching_pids("llama-server").await?;
    assert!(
        leaked.is_empty(),
        "llama-server survived SIGINT: {leaked:?}"
    );
    Ok(())
}
