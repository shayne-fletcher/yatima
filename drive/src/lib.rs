//! Record one host session as a durable, replayable tape.
//!
//! A tape is a run directory:
//!
//! - `tape.jsonl` begins with a versioned header. Every later complete line is
//!   exactly one host request or host event, with one sequence number and the
//!   elapsed time at which the recorder observed it.
//! - `artifacts/NNNN-<name>` holds image bytes out of line. The event line
//!   carries the relative path, SHA-256 digest, and byte count.
//! - `summary.json` closes a clean run with request, event, and turn counts.
//!
//! One Tokio task owns the writer and assigns every sequence number. Async
//! callers await [`RecorderHandle::enqueue`]; synchronous view code may use
//! [`RecorderHandle::enqueue_blocking`] only from an established blocking
//! region. Yatima owns no recorder thread.
//!
//! # Invariant & law registry
//!
//! - **TAPE-1** complete newline-terminated records are one strictly
//!   increasing, parseable, flushed prefix of the recorder's command order.
//!   At most one incomplete final line may follow after abrupt process death;
//!   recovery discards it. Artifact bytes are flushed before their referencing
//!   line, an acknowledged barrier makes every earlier command durable, and a
//!   clean consuming finish drains every earlier command before writing the
//!   summary. The capacity-one queue bounds the recoverable omitted suffix
//!   after abrupt death to the in-progress record and one queued record. Cited
//!   by the tests in this crate.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use anyhow::{anyhow, Context, Result};
use serde::ser::SerializeMap;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use sha2::{Digest, Sha256};
use tokio::fs::File;
use tokio::io::{AsyncWrite, AsyncWriteExt, BufWriter};
use tokio::sync::{mpsc, oneshot};
use tokio::task::JoinHandle;
use yatima_protocol::{HostEvent, HostRequest};

/// Tape schema version, bumped on an incompatible line or summary change.
pub const TAPE_SCHEMA: u32 = 1;

const RECORDER_QUEUE_CAPACITY: usize = 1;
const HASH_BLOCKING_THRESHOLD: usize = 1024 * 1024;
const FINISH_WITHIN: Duration = Duration::from_secs(10);

/// Caller-supplied run metadata. The recorder does not invent provenance.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct TapeMeta {
    pub origin: String,
    pub model: String,
    pub notes: BTreeMap<String, String>,
}

/// The header, line one of every tape.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TapeHeader {
    pub schema: u32,
    pub meta: TapeMeta,
}

/// A semantic input to the recorder. The worker turns an image event into its
/// out-of-line tape representation before writing the corresponding line.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TapeRecord {
    Request(HostRequest),
    Event(HostEvent),
}

/// The exactly-one-kind payload carried by a [`TapeLine`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TapePayload {
    Request(HostRequest),
    Event(serde_json::Value),
}

/// One recorded request or event in the worker's single sequence.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TapeLine {
    pub seq: u64,
    pub t_ms: u64,
    pub turn: Option<u64>,
    pub payload: TapePayload,
}

impl Serialize for TapeLine {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut map = serializer.serialize_map(Some(if self.turn.is_some() { 4 } else { 3 }))?;
        map.serialize_entry("seq", &self.seq)?;
        map.serialize_entry("t_ms", &self.t_ms)?;
        if let Some(turn) = self.turn {
            map.serialize_entry("turn", &turn)?;
        }
        match &self.payload {
            TapePayload::Request(request) => map.serialize_entry("request", request)?,
            TapePayload::Event(event) => map.serialize_entry("event", event)?,
        }
        map.end()
    }
}

impl<'de> Deserialize<'de> for TapeLine {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        #[serde(deny_unknown_fields)]
        struct Wire {
            seq: u64,
            t_ms: u64,
            turn: Option<u64>,
            request: Option<HostRequest>,
            event: Option<serde_json::Value>,
        }

        let wire = Wire::deserialize(deserializer)?;
        let payload = match (wire.request, wire.event) {
            (Some(request), None) => TapePayload::Request(request),
            (None, Some(event)) => TapePayload::Event(event),
            (Some(_), Some(_)) => {
                return Err(serde::de::Error::custom(
                    "a tape line cannot contain both request and event",
                ));
            }
            (None, None) => {
                return Err(serde::de::Error::custom(
                    "a tape line must contain one request or event",
                ));
            }
        };
        Ok(TapeLine {
            seq: wire.seq,
            t_ms: wire.t_ms,
            turn: wire.turn,
            payload,
        })
    }
}

/// The run's closing accounting, written to `summary.json` on clean finish.
#[derive(Debug, Default, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Summary {
    pub schema: u32,
    pub disposition: String,
    pub duration_ms: u64,
    pub records: u64,
    pub events: u64,
    pub requests: u64,
    pub turns_started: u64,
    pub turns_done: u64,
    pub turns_errored: u64,
    pub fragments: u64,
    pub tool_notes: u64,
    pub images: u64,
    pub grants_events: u64,
    pub fatal: bool,
    pub time_to_first_artifact_ms: Option<u64>,
}

#[derive(Debug)]
enum Command {
    Write(TapeRecord),
    Barrier(oneshot::Sender<()>),
    Finish(String),
}

/// Cloneable producer for the recorder's capacity-one command queue.
#[derive(Clone, Debug)]
pub struct RecorderHandle {
    tx: mpsc::Sender<Command>,
}

impl RecorderHandle {
    /// Queue one record without blocking a Tokio worker.
    pub async fn enqueue(&self, record: TapeRecord) -> Result<()> {
        self.tx
            .send(Command::Write(record))
            .await
            .map_err(|_| anyhow!("flight recorder is unavailable"))
    }

    /// Queue one record from synchronous code. This may wait for the one-slot
    /// queue and must therefore be called only from a blocking region.
    pub fn enqueue_blocking(&self, record: TapeRecord) -> Result<()> {
        self.tx
            .blocking_send(Command::Write(record))
            .map_err(|_| anyhow!("flight recorder is unavailable"))
    }

    /// Wait until every command queued before this call has been flushed.
    pub async fn barrier(&self) -> Result<()> {
        let (tx, rx) = oneshot::channel();
        self.tx
            .send(Command::Barrier(tx))
            .await
            .map_err(|_| anyhow!("flight recorder is unavailable"))?;
        rx.await
            .map_err(|_| anyhow!("flight recorder failed before the barrier"))
    }
}

/// The recorder task's sole finishing authority.
#[must_use = "a recorder owner must be finished and joined"]
pub struct RecorderOwner {
    tx: mpsc::Sender<Command>,
    task: JoinHandle<Result<Summary>>,
}

impl RecorderOwner {
    /// Drain all earlier commands, write the summary, and join the task.
    /// A stalled task is aborted and joined before a bounded error is returned.
    pub async fn finish(self, disposition: impl Into<String>) -> Result<Summary> {
        self.finish_within(disposition.into(), FINISH_WITHIN).await
    }

    async fn finish_within(self, disposition: String, within: Duration) -> Result<Summary> {
        let RecorderOwner { tx, mut task } = self;
        let completion = tokio::time::timeout(within, async {
            let finish_sent = tx.send(Command::Finish(disposition)).await.is_ok();
            drop(tx);
            (finish_sent, (&mut task).await)
        })
        .await;

        match completion {
            Ok((finish_sent, joined)) => finish_result(finish_sent, joined),
            Err(_) => {
                task.abort();
                let joined = task.await;
                match joined {
                    Err(error) if error.is_cancelled() => Err(anyhow!(
                        "flight recorder finish timed out after {within:?}; task aborted and joined"
                    )),
                    Ok(Err(error)) => Err(error.context(format!(
                        "flight recorder finish timed out after {within:?}"
                    ))),
                    Ok(Ok(_)) => Err(anyhow!(
                        "flight recorder finish exceeded its {within:?} deadline"
                    )),
                    Err(error) => Err(anyhow!(
                        "flight recorder finish timed out after {within:?}; task join failed: {error}"
                    )),
                }
            }
        }
    }
}

fn finish_result(
    finish_sent: bool,
    joined: std::result::Result<Result<Summary>, tokio::task::JoinError>,
) -> Result<Summary> {
    match joined {
        Ok(result) => match (finish_sent, result) {
            (_, Err(error)) => Err(error),
            (true, Ok(summary)) => Ok(summary),
            (false, Ok(_)) => Err(anyhow!("flight recorder stopped before finish")),
        },
        Err(error) => Err(anyhow!("flight recorder task failed: {error}")),
    }
}

/// Create a new run, flush its header, and start its sole writer task.
pub async fn start_recorder(dir: &Path, meta: TapeMeta) -> Result<(RecorderHandle, RecorderOwner)> {
    let recorder = RecorderCore::create(dir, meta).await?;
    Ok(start_worker(recorder))
}

struct RecorderCore<W> {
    dir: PathBuf,
    tape: W,
    started: Instant,
    seq: u64,
    artifact_seq: u64,
    summary: Summary,
}

impl RecorderCore<BufWriter<File>> {
    async fn create(dir: &Path, meta: TapeMeta) -> Result<Self> {
        tokio::fs::create_dir_all(dir.parent().unwrap_or(Path::new(".")))
            .await
            .with_context(|| format!("create parent of {}", dir.display()))?;
        tokio::fs::create_dir(dir)
            .await
            .with_context(|| format!("create run dir {}", dir.display()))?;
        tokio::fs::create_dir(dir.join("artifacts"))
            .await
            .with_context(|| format!("create artifacts dir under {}", dir.display()))?;
        let file = File::create(dir.join("tape.jsonl"))
            .await
            .with_context(|| format!("create tape under {}", dir.display()))?;
        let mut recorder = Self::from_writer(dir.to_path_buf(), BufWriter::new(file));
        write_json_line(
            &mut recorder.tape,
            &TapeHeader {
                schema: TAPE_SCHEMA,
                meta,
            },
        )
        .await
        .context("write tape header")?;
        Ok(recorder)
    }
}

impl<W> RecorderCore<W>
where
    W: AsyncWrite + Unpin,
{
    fn from_writer(dir: PathBuf, tape: W) -> Self {
        Self {
            dir,
            tape,
            started: Instant::now(),
            seq: 0,
            artifact_seq: 0,
            summary: Summary {
                schema: TAPE_SCHEMA,
                ..Summary::default()
            },
        }
    }

    async fn record(&mut self, record: TapeRecord) -> Result<()> {
        let t_ms = self.elapsed_ms();
        let turn = turn_of(&record);
        let payload = match record {
            TapeRecord::Request(request) => {
                self.summary.requests += 1;
                TapePayload::Request(request)
            }
            TapeRecord::Event(event) => {
                self.summary.events += 1;
                self.account_event(&event, t_ms);
                TapePayload::Event(self.event_value(event).await?)
            }
        };
        let line = TapeLine {
            seq: self.seq,
            t_ms,
            turn,
            payload,
        };
        write_json_line(&mut self.tape, &line)
            .await
            .context("write tape record")?;
        self.seq += 1;
        Ok(())
    }

    async fn finish(mut self, disposition: String) -> Result<Summary> {
        self.summary.disposition = disposition;
        self.summary.duration_ms = self.elapsed_ms();
        self.summary.records = self.seq;
        self.tape.flush().await.context("flush tape at finish")?;

        let json = serde_json::to_vec_pretty(&self.summary)?;
        let path = self.dir.join("summary.json");
        let mut summary_file = File::create(&path)
            .await
            .with_context(|| format!("create {}", path.display()))?;
        summary_file
            .write_all(&json)
            .await
            .with_context(|| format!("write {}", path.display()))?;
        summary_file
            .flush()
            .await
            .with_context(|| format!("flush {}", path.display()))?;
        Ok(self.summary)
    }

    fn elapsed_ms(&self) -> u64 {
        u64::try_from(self.started.elapsed().as_millis()).unwrap_or(u64::MAX)
    }

    fn account_event(&mut self, event: &HostEvent, t_ms: u64) {
        match event {
            HostEvent::Started { .. } => self.summary.turns_started += 1,
            HostEvent::Done { .. } => self.summary.turns_done += 1,
            HostEvent::Error { .. } => self.summary.turns_errored += 1,
            HostEvent::Fragment { .. } => self.summary.fragments += 1,
            HostEvent::ToolNote { .. } => self.summary.tool_notes += 1,
            HostEvent::Image { .. } => {
                self.summary.images += 1;
                self.summary.time_to_first_artifact_ms.get_or_insert(t_ms);
            }
            HostEvent::Grants { .. } => self.summary.grants_events += 1,
            HostEvent::Fatal(_) => self.summary.fatal = true,
            _ => {}
        }
    }

    async fn event_value(&mut self, event: HostEvent) -> Result<serde_json::Value> {
        let HostEvent::Image {
            turn_id,
            bytes,
            name,
        } = event
        else {
            return Ok(serde_json::to_value(event)?);
        };

        let (bytes, digest) = hash_artifact(bytes).await?;
        let file = format!("{:04}-{}", self.artifact_seq, safe_name(&name));
        self.artifact_seq += 1;
        let path = self.dir.join("artifacts").join(&file);
        let mut artifact = File::create(&path)
            .await
            .with_context(|| format!("create {}", path.display()))?;
        artifact
            .write_all(&bytes)
            .await
            .with_context(|| format!("write {}", path.display()))?;
        artifact
            .flush()
            .await
            .with_context(|| format!("flush {}", path.display()))?;

        Ok(serde_json::json!({
            "Image": {
                "turn_id": turn_id,
                "name": name,
                "artifact": format!("artifacts/{file}"),
                "sha256": digest,
                "bytes_len": bytes.len(),
            }
        }))
    }
}

fn start_worker<W>(recorder: RecorderCore<W>) -> (RecorderHandle, RecorderOwner)
where
    W: AsyncWrite + Unpin + Send + 'static,
{
    let (tx, rx) = mpsc::channel(RECORDER_QUEUE_CAPACITY);
    let task = tokio::spawn(run_recorder(recorder, rx));
    (
        RecorderHandle { tx: tx.clone() },
        RecorderOwner { tx, task },
    )
}

async fn run_recorder<W>(
    mut recorder: RecorderCore<W>,
    mut rx: mpsc::Receiver<Command>,
) -> Result<Summary>
where
    W: AsyncWrite + Unpin,
{
    while let Some(command) = rx.recv().await {
        match command {
            Command::Write(record) => recorder.record(record).await?,
            Command::Barrier(reply) => {
                let _ = reply.send(());
            }
            Command::Finish(disposition) => return recorder.finish(disposition).await,
        }
    }
    Err(anyhow!("flight recorder owner dropped without finish"))
}

async fn write_json_line<W, T>(writer: &mut W, value: &T) -> Result<()>
where
    W: AsyncWrite + Unpin,
    T: Serialize,
{
    let mut bytes = serde_json::to_vec(value)?;
    bytes.push(b'\n');
    writer.write_all(&bytes).await?;
    writer.flush().await?;
    Ok(())
}

async fn hash_artifact(bytes: Vec<u8>) -> Result<(Vec<u8>, String)> {
    if bytes.len() < HASH_BLOCKING_THRESHOLD {
        let digest = hex(&Sha256::digest(&bytes));
        return Ok((bytes, digest));
    }
    tokio::task::spawn_blocking(move || {
        let digest = hex(&Sha256::digest(&bytes));
        (bytes, digest)
    })
    .await
    .context("join artifact digest task")
}

fn turn_of(record: &TapeRecord) -> Option<u64> {
    match record {
        TapeRecord::Event(event) => match event {
            HostEvent::Started { turn_id }
            | HostEvent::Fragment { turn_id, .. }
            | HostEvent::RetractAnswer { turn_id, .. }
            | HostEvent::ToolNote { turn_id, .. }
            | HostEvent::Image { turn_id, .. }
            | HostEvent::Done { turn_id, .. }
            | HostEvent::Error { turn_id, .. } => Some(*turn_id),
            _ => None,
        },
        TapeRecord::Request(request) => match request {
            HostRequest::Submit { turn_id, .. } | HostRequest::Cancel { turn_id } => Some(*turn_id),
            _ => None,
        },
    }
}

fn safe_name(name: &str) -> String {
    let base = name.rsplit(['/', '\\']).next().unwrap_or(name);
    let safe: String = base
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-') {
                c
            } else {
                '_'
            }
        })
        .collect();
    if safe.is_empty() {
        "artifact".to_string()
    } else {
        safe
    }
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

#[cfg(test)]
mod tests {
    use std::future::Future;
    use std::io;
    use std::pin::Pin;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex};
    use std::task::{Context as TaskContext, Poll, Waker};

    use super::*;
    use tokio::sync::mpsc::error::TrySendError;
    use yatima_protocol::StartupPhase;

    // Longer than the production owner's 10-second bound, so a broken
    // finalizer reports and joins before the scenario guard can fire.
    const TEST_WITHIN: Duration = Duration::from_secs(15);

    async fn within<T>(what: &str, future: impl Future<Output = T>) -> T {
        tokio::time::timeout(TEST_WITHIN, future)
            .await
            .unwrap_or_else(|_| panic!("{what} did not finish within {TEST_WITHIN:?}"))
    }

    fn meta() -> TapeMeta {
        TapeMeta {
            origin: "test".into(),
            model: "stub-muse".into(),
            notes: BTreeMap::from([("agent_max_steps".into(), "12".into())]),
        }
    }

    async fn lines(dir: &Path) -> Vec<String> {
        tokio::fs::read_to_string(dir.join("tape.jsonl"))
            .await
            .unwrap()
            .lines()
            .map(str::to_string)
            .collect()
    }

    #[tokio::test]
    async fn requests_and_events_share_one_exact_sequence() {
        within(
            "requests and events share one exact sequence",
            requests_and_events_share_one_exact_sequence_case(),
        )
        .await;
    }

    async fn requests_and_events_share_one_exact_sequence_case() {
        // upholds: TAPE-1 — both planes share the worker's sequence, retain
        // their exact wire values, and serialize as exactly one payload kind.
        let dir = tempfile::tempdir().unwrap();
        let run = dir.path().join("run");
        let records = vec![
            TapeRecord::Request(HostRequest::Grant {
                origin: "https://example.com".into(),
            }),
            TapeRecord::Event(HostEvent::Grants {
                origins: vec!["https://example.com".into()],
                message: "granted".into(),
            }),
            TapeRecord::Request(HostRequest::Submit {
                turn_id: 7,
                text: "hello".into(),
            }),
            TapeRecord::Event(HostEvent::Started { turn_id: 7 }),
        ];
        let (handle, owner) = start_recorder(&run, meta()).await.unwrap();
        for record in &records {
            handle.enqueue(record.clone()).await.unwrap();
        }
        let summary = owner.finish("completed").await.unwrap();

        let lines = lines(&run).await;
        assert_eq!(lines.len(), 1 + records.len());
        let header: TapeHeader = serde_json::from_str(&lines[0]).unwrap();
        assert_eq!(header.meta, meta());
        for (index, record) in records.iter().enumerate() {
            let line: TapeLine = serde_json::from_str(&lines[index + 1]).unwrap();
            assert_eq!(line.seq, index as u64);
            assert_eq!(line.turn, turn_of(record));
            match (record, &line.payload) {
                (TapeRecord::Request(expected), TapePayload::Request(actual)) => {
                    assert_eq!(actual, expected)
                }
                (TapeRecord::Event(expected), TapePayload::Event(actual)) => {
                    assert_eq!(actual, &serde_json::to_value(expected).unwrap())
                }
                pair => panic!("payload kind changed: {pair:?}"),
            }
            let wire: serde_json::Value = serde_json::from_str(&lines[index + 1]).unwrap();
            assert_eq!(
                wire.get("request").is_some() as u8 + wire.get("event").is_some() as u8,
                1
            );
        }
        assert_eq!(summary.records, 4);
        assert_eq!(summary.requests, 2);
        assert_eq!(summary.events, 2);
        assert!(
            serde_json::from_value::<TapeLine>(serde_json::json!({
                "seq": 0,
                "t_ms": 0,
                "request": "Reset",
                "event": {"Note": "contradiction"}
            }))
            .is_err(),
            "both payload kinds are rejected"
        );
        assert!(
            serde_json::from_value::<TapeLine>(serde_json::json!({
                "seq": 0,
                "t_ms": 0
            }))
            .is_err(),
            "a missing payload kind is rejected"
        );
    }

    #[tokio::test]
    async fn barrier_exposes_a_complete_flushed_prefix() {
        within(
            "barrier exposes a complete flushed prefix",
            barrier_exposes_a_complete_flushed_prefix_case(),
        )
        .await;
    }

    async fn barrier_exposes_a_complete_flushed_prefix_case() {
        // upholds: TAPE-1 — a barrier acknowledges every earlier record while
        // the worker remains alive; each visible line is complete JSON.
        let dir = tempfile::tempdir().unwrap();
        let run = dir.path().join("run");
        let (handle, owner) = start_recorder(&run, meta()).await.unwrap();
        handle
            .enqueue(TapeRecord::Event(HostEvent::Startup {
                phase: StartupPhase::ResolvingModel,
            }))
            .await
            .unwrap();
        handle.barrier().await.unwrap();

        let visible = lines(&run).await;
        assert_eq!(visible.len(), 2);
        serde_json::from_str::<TapeHeader>(&visible[0]).unwrap();
        serde_json::from_str::<TapeLine>(&visible[1]).unwrap();
        assert!(!run.join("summary.json").exists(), "worker is still live");
        owner.finish("completed").await.unwrap();
    }

    #[tokio::test]
    async fn image_bytes_are_flushed_before_their_line() {
        within(
            "image bytes are flushed before their line",
            image_bytes_are_flushed_before_their_line_case(),
        )
        .await;
    }

    async fn image_bytes_are_flushed_before_their_line_case() {
        // upholds: TAPE-1 — the tape writer stays pending at its first line
        // while this task observes the already-complete artifact bytes.
        let dir = tempfile::tempdir().unwrap();
        let run = dir.path().join("run");
        tokio::fs::create_dir_all(run.join("artifacts"))
            .await
            .unwrap();
        let artifact = run.join("artifacts/0000-img.png");
        let (line_started, line_waiting) = oneshot::channel();
        let gate = Arc::new(WriteGate::default());
        let tape_bytes = Arc::new(Mutex::new(Vec::new()));
        let writer = GatedTapeWriter {
            started: Some(line_started),
            gate: gate.clone(),
            bytes: tape_bytes.clone(),
        };
        let (handle, owner) = start_worker(RecorderCore::from_writer(run, writer));
        let bytes = b"image-bytes".to_vec();
        handle
            .enqueue(TapeRecord::Event(HostEvent::Image {
                turn_id: 3,
                bytes: bytes.clone(),
                name: "img.png".into(),
            }))
            .await
            .unwrap();
        line_waiting.await.unwrap();
        assert_eq!(tokio::fs::read(&artifact).await.unwrap(), bytes);
        gate.release();
        let summary = owner.finish("completed").await.unwrap();

        assert_eq!(summary.images, 1);
        assert!(summary.time_to_first_artifact_ms.is_some());
        let tape = String::from_utf8(tape_bytes.lock().unwrap().clone()).unwrap();
        let lines: Vec<_> = tape.lines().collect();
        assert_eq!(lines.len(), 1);
        let wire: serde_json::Value = serde_json::from_str(lines[0]).unwrap();
        let image = &wire["event"]["Image"];
        assert_eq!(image["artifact"], "artifacts/0000-img.png");
        assert_eq!(image["sha256"], hex(&Sha256::digest(&bytes)));
        assert_eq!(image["bytes_len"], bytes.len());
        assert!(image.get("bytes").is_none(), "image bytes stay out of line");
    }

    #[tokio::test]
    async fn clean_finish_drains_work_and_closes_the_summary() {
        within(
            "clean finish drains work and closes the summary",
            clean_finish_drains_work_and_closes_the_summary_case(),
        )
        .await;
    }

    async fn clean_finish_drains_work_and_closes_the_summary_case() {
        // upholds: TAPE-1 — Finish follows queued writes and accounts the
        // durable records before summary.json becomes visible.
        let dir = tempfile::tempdir().unwrap();
        let run = dir.path().join("run");
        let (handle, owner) = start_recorder(&run, meta()).await.unwrap();
        for record in [
            TapeRecord::Event(HostEvent::Started { turn_id: 1 }),
            TapeRecord::Event(HostEvent::Error {
                turn_id: 1,
                message: "backend lost".into(),
            }),
            TapeRecord::Event(HostEvent::Fatal("child died".into())),
            TapeRecord::Request(HostRequest::Shutdown),
        ] {
            handle.enqueue(record).await.unwrap();
        }
        let summary = owner.finish("fatal").await.unwrap();

        assert_eq!(summary.records, 4);
        assert_eq!(summary.events, 3);
        assert_eq!(summary.requests, 1);
        assert_eq!(summary.turns_started, 1);
        assert_eq!(summary.turns_errored, 1);
        assert!(summary.fatal);
        assert_eq!(summary.disposition, "fatal");
        let reread: Summary =
            serde_json::from_slice(&tokio::fs::read(run.join("summary.json")).await.unwrap())
                .unwrap();
        assert_eq!(reread, summary);
        assert_eq!(lines(&run).await.len(), 5);
    }

    #[tokio::test]
    async fn a_run_directory_is_never_reused() {
        within(
            "a run directory is never reused",
            a_run_directory_is_never_reused_case(),
        )
        .await;
    }

    async fn a_run_directory_is_never_reused_case() {
        let dir = tempfile::tempdir().unwrap();
        let run = dir.path().join("run");
        let (_handle, owner) = start_recorder(&run, meta()).await.unwrap();
        assert!(start_recorder(&run, meta()).await.is_err());
        owner.finish("completed").await.unwrap();
    }

    #[tokio::test]
    async fn artifact_names_cannot_escape_or_collide() {
        within(
            "artifact names cannot escape or collide",
            artifact_names_cannot_escape_or_collide_case(),
        )
        .await;
    }

    async fn artifact_names_cannot_escape_or_collide_case() {
        let dir = tempfile::tempdir().unwrap();
        let run = dir.path().join("run");
        let (handle, owner) = start_recorder(&run, meta()).await.unwrap();
        for name in ["../../escape.png", "a b/c:d.png"] {
            handle
                .enqueue(TapeRecord::Event(HostEvent::Image {
                    turn_id: 1,
                    bytes: vec![1, 2, 3],
                    name: name.into(),
                }))
                .await
                .unwrap();
        }
        owner.finish("completed").await.unwrap();
        let mut files = Vec::new();
        let mut entries = tokio::fs::read_dir(run.join("artifacts")).await.unwrap();
        while let Some(entry) = entries.next_entry().await.unwrap() {
            files.push(entry.file_name().to_string_lossy().into_owned());
        }
        files.sort();
        assert_eq!(files, ["0000-escape.png", "0001-c_d.png"]);
    }

    #[tokio::test]
    async fn a_partial_failed_line_is_the_only_unparseable_tail() {
        within(
            "a partial failed line is the only unparseable tail",
            a_partial_failed_line_is_the_only_unparseable_tail_case(),
        )
        .await;
    }

    async fn a_partial_failed_line_is_the_only_unparseable_tail_case() {
        // upholds: TAPE-1 — a failed write retains its original error and may
        // leave one partial tail, but every newline-terminated line remains a
        // valid prefix record.
        let dir = tempfile::tempdir().unwrap();
        let run = dir.path().join("run");
        tokio::fs::create_dir_all(run.join("artifacts"))
            .await
            .unwrap();
        let mut initial = serde_json::to_vec(&TapeHeader {
            schema: TAPE_SCHEMA,
            meta: meta(),
        })
        .unwrap();
        initial.push(b'\n');
        let complete = TapeLine {
            seq: 0,
            t_ms: 0,
            turn: None,
            payload: TapePayload::Event(
                serde_json::to_value(HostEvent::Note("kept".into())).unwrap(),
            ),
        };
        initial.extend(serde_json::to_vec(&complete).unwrap());
        initial.push(b'\n');
        let bytes = Arc::new(Mutex::new(initial));
        let writer = FailingWriter {
            bytes: bytes.clone(),
            remaining: 12,
        };
        let mut recorder = RecorderCore::from_writer(run, writer);
        recorder.seq = 1;
        recorder.summary.events = 1;
        let (handle, owner) = start_worker(recorder);
        handle
            .enqueue(TapeRecord::Event(HostEvent::Note("evidence".into())))
            .await
            .unwrap();
        let error = owner.finish("completed").await.unwrap_err();
        assert!(format!("{error:#}").contains("injected tape write failure"));

        let bytes = bytes.lock().unwrap();
        let last_newline = bytes.iter().rposition(|byte| *byte == b'\n').unwrap();
        let prefix = &bytes[..=last_newline];
        let tail = &bytes[last_newline + 1..];
        let complete_lines: Vec<_> = prefix
            .split(|byte| *byte == b'\n')
            .filter(|line| !line.is_empty())
            .collect();
        assert_eq!(complete_lines.len(), 2, "header plus one complete record");
        for line in complete_lines {
            serde_json::from_slice::<serde_json::Value>(line).unwrap();
        }
        assert!(!tail.is_empty());
        assert!(serde_json::from_slice::<TapeLine>(tail).is_err());
    }

    #[tokio::test]
    async fn capacity_one_bounds_the_waiting_suffix() {
        within(
            "capacity one bounds the waiting suffix",
            capacity_one_bounds_the_waiting_suffix_case(),
        )
        .await;
    }

    async fn capacity_one_bounds_the_waiting_suffix_case() {
        // upholds: TAPE-1 — while one record is stalled in the writer, exactly
        // one more can wait and a third is refused by the bounded queue.
        assert_eq!(RECORDER_QUEUE_CAPACITY, 1);
        let dir = tempfile::tempdir().unwrap();
        let run = dir.path().join("run");
        tokio::fs::create_dir_all(run.join("artifacts"))
            .await
            .unwrap();
        let (writer, reader) = tokio::io::duplex(1);
        let (handle, owner) = start_worker(RecorderCore::from_writer(run, writer));
        handle
            .enqueue(TapeRecord::Event(HostEvent::Note("writing".into())))
            .await
            .unwrap();
        for _ in 0..100 {
            if handle.tx.capacity() == RECORDER_QUEUE_CAPACITY {
                break;
            }
            tokio::task::yield_now().await;
        }
        assert_eq!(
            handle.tx.capacity(),
            1,
            "worker took the in-progress record"
        );
        handle
            .tx
            .try_send(Command::Write(TapeRecord::Event(HostEvent::Note(
                "queued".into(),
            ))))
            .unwrap();
        assert!(matches!(
            handle
                .tx
                .try_send(Command::Write(TapeRecord::Event(HostEvent::Note(
                    "too many".into()
                )))),
            Err(TrySendError::Full(_))
        ));
        drop(reader);
        assert!(owner.finish("failed").await.is_err());
    }

    #[tokio::test]
    async fn timed_out_finish_aborts_and_joins_the_recorder_task() {
        within(
            "timed out finish aborts and joins the recorder task",
            timed_out_finish_aborts_and_joins_the_recorder_task_case(),
        )
        .await;
    }

    async fn timed_out_finish_aborts_and_joins_the_recorder_task_case() {
        // upholds: TAPE-1 / HOST-3 — a wedged writer cannot detach its owner
        // or keep running after bounded finalization returns.
        let dir = tempfile::tempdir().unwrap();
        let run = dir.path().join("run");
        tokio::fs::create_dir_all(run.join("artifacts"))
            .await
            .unwrap();
        let polls = Arc::new(AtomicUsize::new(0));
        let writer = NeverWriter {
            polls: polls.clone(),
        };
        let (handle, owner) = start_worker(RecorderCore::from_writer(run, writer));
        handle
            .enqueue(TapeRecord::Event(HostEvent::Note("wedged".into())))
            .await
            .unwrap();
        while polls.load(Ordering::Acquire) == 0 {
            tokio::task::yield_now().await;
        }

        let error = owner
            .finish_within("failed".into(), Duration::from_millis(20))
            .await
            .unwrap_err();
        assert!(format!("{error:#}").contains("task aborted and joined"));
        let polls_after_join = polls.load(Ordering::Acquire);
        tokio::time::sleep(Duration::from_millis(20)).await;
        assert_eq!(polls.load(Ordering::Acquire), polls_after_join);
    }

    struct FailingWriter {
        bytes: Arc<Mutex<Vec<u8>>>,
        remaining: usize,
    }

    struct NeverWriter {
        polls: Arc<AtomicUsize>,
    }

    impl AsyncWrite for NeverWriter {
        fn poll_write(
            self: Pin<&mut Self>,
            _cx: &mut TaskContext<'_>,
            _input: &[u8],
        ) -> Poll<io::Result<usize>> {
            self.polls.fetch_add(1, Ordering::AcqRel);
            Poll::Pending
        }

        fn poll_flush(self: Pin<&mut Self>, _cx: &mut TaskContext<'_>) -> Poll<io::Result<()>> {
            Poll::Ready(Ok(()))
        }

        fn poll_shutdown(self: Pin<&mut Self>, _cx: &mut TaskContext<'_>) -> Poll<io::Result<()>> {
            Poll::Ready(Ok(()))
        }
    }

    impl AsyncWrite for FailingWriter {
        fn poll_write(
            mut self: Pin<&mut Self>,
            _cx: &mut TaskContext<'_>,
            input: &[u8],
        ) -> Poll<io::Result<usize>> {
            if self.remaining == 0 {
                return Poll::Ready(Err(io::Error::other("injected tape write failure")));
            }
            let written = input.len().min(self.remaining);
            self.bytes
                .lock()
                .unwrap()
                .extend_from_slice(&input[..written]);
            self.remaining -= written;
            Poll::Ready(Ok(written))
        }

        fn poll_flush(self: Pin<&mut Self>, _cx: &mut TaskContext<'_>) -> Poll<io::Result<()>> {
            Poll::Ready(Ok(()))
        }

        fn poll_shutdown(self: Pin<&mut Self>, _cx: &mut TaskContext<'_>) -> Poll<io::Result<()>> {
            Poll::Ready(Ok(()))
        }
    }

    #[derive(Default)]
    struct WriteGate {
        released: AtomicBool,
        waker: Mutex<Option<Waker>>,
    }

    impl WriteGate {
        fn release(&self) {
            self.released.store(true, Ordering::Release);
            if let Some(waker) = self.waker.lock().unwrap().take() {
                waker.wake();
            }
        }
    }

    struct GatedTapeWriter {
        started: Option<oneshot::Sender<()>>,
        gate: Arc<WriteGate>,
        bytes: Arc<Mutex<Vec<u8>>>,
    }

    impl AsyncWrite for GatedTapeWriter {
        fn poll_write(
            mut self: Pin<&mut Self>,
            cx: &mut TaskContext<'_>,
            input: &[u8],
        ) -> Poll<io::Result<usize>> {
            if let Some(started) = self.started.take() {
                let _ = started.send(());
            }
            if !self.gate.released.load(Ordering::Acquire) {
                *self.gate.waker.lock().unwrap() = Some(cx.waker().clone());
                return Poll::Pending;
            }
            self.bytes.lock().unwrap().extend_from_slice(input);
            Poll::Ready(Ok(input.len()))
        }

        fn poll_flush(self: Pin<&mut Self>, _cx: &mut TaskContext<'_>) -> Poll<io::Result<()>> {
            Poll::Ready(Ok(()))
        }

        fn poll_shutdown(self: Pin<&mut Self>, _cx: &mut TaskContext<'_>) -> Poll<io::Result<()>> {
            Poll::Ready(Ok(()))
        }
    }
}
