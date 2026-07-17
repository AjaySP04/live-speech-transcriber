# Live Transcriber Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A Rust web app that listens through the browser mic, VAD-gates recording to human speech only, transcribes Hindi/Urdu/Arabic (any Whisper language) with live English translation and `Person N` speaker labels, and persists conversations to SQLite. Runs via Docker; targets ARM IoT boards later.

**Architecture:** Single `axum` binary serves static UI + REST history API + a WebSocket. Browser streams 16 kHz mono i16 PCM up the socket; a per-connection pipeline thread runs Silero VAD → utterance segmenter → Whisper (transcribe + translate) → speaker embedding + online cosine clustering → SQLite, and streams JSON transcript events back.

**Tech Stack:** Rust 2021, axum 0.8, tokio, whisper-rs 0.16 (whisper.cpp), voice_activity_detector 0.2 (Silero V5, model bundled in crate), sherpa-rs 0.6 (speaker embeddings, TitaNet-small ONNX), rusqlite (bundled SQLite), vanilla JS frontend, Docker multi-stage build.

## Global Constraints

- Audio format everywhere in the backend: **16 kHz, mono, i16 PCM**. VAD chunk size: **512 samples (32 ms)**.
- Whisper models are ggml files named `models/ggml-<size>.bin`; speaker model is `models/nemo_en_titanet_small.onnx`. Never commit models; `models/` and `data/` are gitignored.
- Tests that need models use **ggml-tiny.bin** and must panic with message `run scripts/fetch-models.sh tiny` if it is missing.
- Frontend: no frameworks, no build step — plain HTML/JS/CSS in `static/`.
- All whisper decoding: greedy sampling, language auto-detect. Pad audio shorter than 1.1 s (17,600 samples) with trailing zeros before decoding (whisper.cpp misbehaves on very short input).
- Speaker labels are 1-based (`Person 1`, …), per-session.
- Commit after every task with the message given in the task's final step. Commit messages end with `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.
- Build prerequisites on macOS: `cmake` (for sherpa-rs) — `brew install cmake` if missing. First `cargo build` compiles whisper.cpp + sherpa-onnx + downloads ONNX Runtime; it can take 10–20 min. Run long builds with `run_in_background` and poll rather than hitting Bash timeouts.
- Crate APIs were verified against docs 2026-07-16. If a method signature fails to compile (whisper-rs and sherpa-rs evolve), check `docs.rs` for the resolved version and adapt the call site only — do not change the module's public interface defined in this plan.

---

### Task 1: Project scaffold, config module, minimal server

**Files:**
- Create: `Cargo.toml`, `.gitignore`, `config.toml`, `src/main.rs`, `src/config.rs`, `static/index.html` (placeholder)

**Interfaces:**
- Produces: `config::Config` (all fields below), `Config::load(path: &str) -> anyhow::Result<Config>`, `Config::whisper_model_path(&self) -> PathBuf`, `Config::speaker_model_path(&self) -> PathBuf`. Later tasks read `cfg.<field>` exactly as named here.

- [ ] **Step 1: Scaffold cargo project and gitignore**

```bash
cd /Users/ajaysinghparmar/Desktop/Projects/Personal/live-transcriber
cargo init --name live-transcriber
```

`.gitignore`:
```
/target
/models
/data
```

`Cargo.toml`:
```toml
[package]
name = "live-transcriber"
version = "0.1.0"
edition = "2021"

[dependencies]
anyhow = "1"
axum = { version = "0.8", features = ["ws"] }
axum-server = { version = "0.7", features = ["tls-rustls"] }
futures-util = "0.3"
rcgen = "0.13"
rusqlite = { version = "0.32", features = ["bundled"] }
serde = { version = "1", features = ["derive"] }
serde_json = "1"
sherpa-rs = "0.6"
tokio = { version = "1", features = ["full"] }
toml = "0.8"
tower-http = { version = "0.6", features = ["fs"] }
tracing = "0.1"
tracing-subscriber = "0.3"
voice_activity_detector = "0.2"
whisper-rs = "0.16"

[dev-dependencies]
hound = "3.5"
tokio-tungstenite = "0.24"

[features]
metal = ["whisper-rs/metal"]
```

`config.toml`:
```toml
port = 8080
models_dir = "models"
data_dir = "data"
whisper_model = "medium"
english_only = false
vad_threshold = 0.5
silence_end_ms = 700
pre_roll_ms = 300
min_speech_ms = 250
max_utterance_s = 25
speaker_similarity_threshold = 0.65
whisper_threads = 4
tls = false
```

- [ ] **Step 2: Write failing config test**

`src/config.rs`:
```rust
use anyhow::Result;
use serde::Deserialize;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct Config {
    pub port: u16,
    pub models_dir: String,
    pub data_dir: String,
    pub whisper_model: String,
    pub english_only: bool,
    pub vad_threshold: f32,
    pub silence_end_ms: u32,
    pub pre_roll_ms: u32,
    pub min_speech_ms: u32,
    pub max_utterance_s: u32,
    pub speaker_similarity_threshold: f32,
    pub whisper_threads: i32,
    pub tls: bool,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            port: 8080,
            models_dir: "models".into(),
            data_dir: "data".into(),
            whisper_model: "medium".into(),
            english_only: false,
            vad_threshold: 0.5,
            silence_end_ms: 700,
            pre_roll_ms: 300,
            min_speech_ms: 250,
            max_utterance_s: 25,
            speaker_similarity_threshold: 0.65,
            whisper_threads: 4,
            tls: false,
        }
    }
}

impl Config {
    /// Loads config from a TOML file; missing file = all defaults.
    /// Env overrides: LT_PORT, LT_WHISPER_MODEL, LT_ENGLISH_ONLY.
    pub fn load(path: &str) -> Result<Config> {
        let mut cfg: Config = match std::fs::read_to_string(path) {
            Ok(s) => toml::from_str(&s)?,
            Err(_) => Config::default(),
        };
        if let Ok(v) = std::env::var("LT_PORT") {
            cfg.port = v.parse()?;
        }
        if let Ok(v) = std::env::var("LT_WHISPER_MODEL") {
            cfg.whisper_model = v;
        }
        if let Ok(v) = std::env::var("LT_ENGLISH_ONLY") {
            cfg.english_only = v == "1" || v.eq_ignore_ascii_case("true");
        }
        Ok(cfg)
    }

    pub fn whisper_model_path(&self) -> PathBuf {
        Path::new(&self.models_dir).join(format!("ggml-{}.bin", self.whisper_model))
    }

    pub fn speaker_model_path(&self) -> PathBuf {
        Path::new(&self.models_dir).join("nemo_en_titanet_small.onnx")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn missing_file_gives_defaults() {
        let cfg = Config::load("/nonexistent/config.toml").unwrap();
        assert_eq!(cfg.port, 8080);
        assert_eq!(cfg.whisper_model, "medium");
        assert!(!cfg.english_only);
        assert_eq!(cfg.whisper_model_path().to_str().unwrap(), "models/ggml-medium.bin");
    }

    #[test]
    fn partial_toml_overrides_defaults() {
        let dir = std::env::temp_dir().join("lt-config-test");
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("config.toml");
        std::fs::write(&p, "whisper_model = \"tiny\"\nport = 9999\n").unwrap();
        let cfg = Config::load(p.to_str().unwrap()).unwrap();
        assert_eq!(cfg.port, 9999);
        assert_eq!(cfg.whisper_model, "tiny");
        assert_eq!(cfg.silence_end_ms, 700); // untouched default
    }
}
```

- [ ] **Step 3: Minimal main.rs + placeholder page** (rewritten fully in Task 9)

`src/main.rs`:
```rust
mod config;

use tower_http::services::ServeDir;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt().init();
    let cfg = config::Config::load("config.toml")?;
    let app = axum::Router::new()
        .route("/healthz", axum::routing::get(|| async { "ok" }))
        .fallback_service(ServeDir::new("static"));
    let addr = format!("0.0.0.0:{}", cfg.port);
    tracing::info!("listening on http://{addr}");
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    axum::serve(listener, app).await?;
    Ok(())
}
```

`static/index.html`:
```html
<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Live Transcriber</title></head>
<body><h1>Live Transcriber — coming soon</h1></body></html>
```

- [ ] **Step 4: Build and test** (first build compiles whisper.cpp/sherpa-onnx — run in background, expect 10–20 min)

Run: `cargo test` (background, poll until done)
Expected: PASS — 2 tests in `config::tests`.

Run: `cargo run` in background, then `curl -s localhost:8080/healthz`
Expected: `ok`. Kill the server after.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: scaffold axum server with config module"
```

---

### Task 2: SQLite persistence layer

**Files:**
- Create: `src/db.rs`
- Modify: `src/main.rs` (add `mod db;`)

**Interfaces:**
- Produces:
  - `db::Db` with `open(path: &str) -> Result<Db>`, `open_in_memory() -> Result<Db>`
  - `create_session(&self) -> Result<i64>`, `end_session(&self, id: i64) -> Result<()>`, `set_session_title(&self, id: i64, title: &str) -> Result<()>`, `session_exists(&self, id: i64) -> Result<bool>`
  - `insert_utterance(&self, session_id: i64, u: &UtteranceRow) -> Result<i64>`
  - `upsert_speaker(&self, session_id: i64, speaker_num: i64, centroid: &[f32]) -> Result<()>`, `load_speakers(&self, session_id: i64) -> Result<Vec<(i64, Vec<f32>)>>` (ordered by speaker_num)
  - `list_sessions(&self) -> Result<Vec<SessionSummary>>`, `get_utterances(&self, session_id: i64) -> Result<Vec<UtteranceRow>>`
  - `pub struct UtteranceRow { pub speaker_num: i64, pub lang: String, pub original_text: String, pub english_text: String, pub start_ms: i64, pub duration_ms: i64 }` (derives `Serialize, Clone, Debug`)
  - `pub struct SessionSummary { pub id: i64, pub started_at: String, pub ended_at: Option<String>, pub title: String, pub utterance_count: i64 }` (derives `Serialize, Debug`)

- [ ] **Step 1: Write failing tests** (bottom of new `src/db.rs`, module skeleton with `todo!()` bodies so it compiles, or write tests first referencing the API)

```rust
#[cfg(test)]
mod tests {
    use super::*;

    fn mem() -> Db {
        Db::open_in_memory().unwrap()
    }

    #[test]
    fn session_lifecycle_and_utterances() {
        let db = mem();
        let sid = db.create_session().unwrap();
        assert!(db.session_exists(sid).unwrap());
        assert!(!db.session_exists(sid + 99).unwrap());

        let u = UtteranceRow {
            speaker_num: 1,
            lang: "hi".into(),
            original_text: "आप कैसे हैं?".into(),
            english_text: "How are you?".into(),
            start_ms: 1200,
            duration_ms: 900,
        };
        db.insert_utterance(sid, &u).unwrap();
        db.set_session_title(sid, "Test chat").unwrap();
        db.end_session(sid).unwrap();

        let sessions = db.list_sessions().unwrap();
        assert_eq!(sessions.len(), 1);
        assert_eq!(sessions[0].title, "Test chat");
        assert_eq!(sessions[0].utterance_count, 1);
        assert!(sessions[0].ended_at.is_some());

        let utts = db.get_utterances(sid).unwrap();
        assert_eq!(utts.len(), 1);
        assert_eq!(utts[0].english_text, "How are you?");
        assert_eq!(utts[0].lang, "hi");
    }

    #[test]
    fn speaker_centroids_roundtrip() {
        let db = mem();
        let sid = db.create_session().unwrap();
        db.upsert_speaker(sid, 1, &[0.1, 0.2, 0.3]).unwrap();
        db.upsert_speaker(sid, 2, &[0.9, 0.8, 0.7]).unwrap();
        db.upsert_speaker(sid, 1, &[0.5, 0.5, 0.5]).unwrap(); // update in place
        let spk = db.load_speakers(sid).unwrap();
        assert_eq!(spk.len(), 2);
        assert_eq!(spk[0].0, 1);
        assert_eq!(spk[0].1, vec![0.5, 0.5, 0.5]);
        assert_eq!(spk[1].0, 2);
    }
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test db::`
Expected: FAIL (unresolved names / todo panics).

- [ ] **Step 3: Implement `src/db.rs`**

```rust
use anyhow::Result;
use rusqlite::{params, Connection};
use serde::Serialize;

const SCHEMA: &str = "
CREATE TABLE IF NOT EXISTS sessions (
    id INTEGER PRIMARY KEY,
    started_at TEXT NOT NULL DEFAULT (datetime('now')),
    ended_at TEXT,
    title TEXT NOT NULL DEFAULT ''
);
CREATE TABLE IF NOT EXISTS utterances (
    id INTEGER PRIMARY KEY,
    session_id INTEGER NOT NULL REFERENCES sessions(id),
    speaker_num INTEGER NOT NULL,
    lang TEXT NOT NULL,
    original_text TEXT NOT NULL,
    english_text TEXT NOT NULL,
    start_ms INTEGER NOT NULL,
    duration_ms INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_utt_session ON utterances(session_id);
CREATE TABLE IF NOT EXISTS speakers (
    session_id INTEGER NOT NULL REFERENCES sessions(id),
    speaker_num INTEGER NOT NULL,
    centroid BLOB NOT NULL,
    PRIMARY KEY (session_id, speaker_num)
);
";

#[derive(Debug, Clone, Serialize)]
pub struct UtteranceRow {
    pub speaker_num: i64,
    pub lang: String,
    pub original_text: String,
    pub english_text: String,
    pub start_ms: i64,
    pub duration_ms: i64,
}

#[derive(Debug, Serialize)]
pub struct SessionSummary {
    pub id: i64,
    pub started_at: String,
    pub ended_at: Option<String>,
    pub title: String,
    pub utterance_count: i64,
}

pub struct Db {
    conn: Connection,
}

fn f32s_to_blob(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|f| f.to_le_bytes()).collect()
}

fn blob_to_f32s(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

impl Db {
    pub fn open(path: &str) -> Result<Self> {
        let conn = Connection::open(path)?;
        conn.pragma_update(None, "journal_mode", "WAL")?;
        conn.execute_batch(SCHEMA)?;
        Ok(Db { conn })
    }

    pub fn open_in_memory() -> Result<Self> {
        let conn = Connection::open_in_memory()?;
        conn.execute_batch(SCHEMA)?;
        Ok(Db { conn })
    }

    pub fn create_session(&self) -> Result<i64> {
        self.conn.execute("INSERT INTO sessions DEFAULT VALUES", [])?;
        Ok(self.conn.last_insert_rowid())
    }

    pub fn session_exists(&self, id: i64) -> Result<bool> {
        let n: i64 = self.conn.query_row(
            "SELECT COUNT(*) FROM sessions WHERE id = ?1",
            params![id],
            |r| r.get(0),
        )?;
        Ok(n > 0)
    }

    pub fn end_session(&self, id: i64) -> Result<()> {
        self.conn.execute(
            "UPDATE sessions SET ended_at = datetime('now') WHERE id = ?1",
            params![id],
        )?;
        Ok(())
    }

    pub fn set_session_title(&self, id: i64, title: &str) -> Result<()> {
        self.conn.execute(
            "UPDATE sessions SET title = ?2 WHERE id = ?1",
            params![id, title],
        )?;
        Ok(())
    }

    pub fn insert_utterance(&self, session_id: i64, u: &UtteranceRow) -> Result<i64> {
        self.conn.execute(
            "INSERT INTO utterances (session_id, speaker_num, lang, original_text, english_text, start_ms, duration_ms)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            params![session_id, u.speaker_num, u.lang, u.original_text, u.english_text, u.start_ms, u.duration_ms],
        )?;
        Ok(self.conn.last_insert_rowid())
    }

    pub fn upsert_speaker(&self, session_id: i64, speaker_num: i64, centroid: &[f32]) -> Result<()> {
        self.conn.execute(
            "INSERT INTO speakers (session_id, speaker_num, centroid) VALUES (?1, ?2, ?3)
             ON CONFLICT(session_id, speaker_num) DO UPDATE SET centroid = excluded.centroid",
            params![session_id, speaker_num, f32s_to_blob(centroid)],
        )?;
        Ok(())
    }

    pub fn load_speakers(&self, session_id: i64) -> Result<Vec<(i64, Vec<f32>)>> {
        let mut stmt = self.conn.prepare(
            "SELECT speaker_num, centroid FROM speakers WHERE session_id = ?1 ORDER BY speaker_num",
        )?;
        let rows = stmt.query_map(params![session_id], |r| {
            Ok((r.get::<_, i64>(0)?, r.get::<_, Vec<u8>>(1)?))
        })?;
        let mut out = Vec::new();
        for row in rows {
            let (num, blob) = row?;
            out.push((num, blob_to_f32s(&blob)));
        }
        Ok(out)
    }

    pub fn list_sessions(&self) -> Result<Vec<SessionSummary>> {
        let mut stmt = self.conn.prepare(
            "SELECT s.id, s.started_at, s.ended_at, s.title,
                    (SELECT COUNT(*) FROM utterances u WHERE u.session_id = s.id)
             FROM sessions s ORDER BY s.id DESC",
        )?;
        let rows = stmt.query_map([], |r| {
            Ok(SessionSummary {
                id: r.get(0)?,
                started_at: r.get(1)?,
                ended_at: r.get(2)?,
                title: r.get(3)?,
                utterance_count: r.get(4)?,
            })
        })?;
        Ok(rows.collect::<std::result::Result<Vec<_>, _>>()?)
    }

    pub fn get_utterances(&self, session_id: i64) -> Result<Vec<UtteranceRow>> {
        let mut stmt = self.conn.prepare(
            "SELECT speaker_num, lang, original_text, english_text, start_ms, duration_ms
             FROM utterances WHERE session_id = ?1 ORDER BY id",
        )?;
        let rows = stmt.query_map(params![session_id], |r| {
            Ok(UtteranceRow {
                speaker_num: r.get(0)?,
                lang: r.get(1)?,
                original_text: r.get(2)?,
                english_text: r.get(3)?,
                start_ms: r.get(4)?,
                duration_ms: r.get(5)?,
            })
        })?;
        Ok(rows.collect::<std::result::Result<Vec<_>, _>>()?)
    }
}
```

Add `mod db;` to `src/main.rs` (above `mod config;` alphabetically; add `#[allow(dead_code)]` on the module if warnings block, but do not silence the whole crate).

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test db::`
Expected: PASS — 2 tests.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: SQLite persistence layer (sessions, utterances, speaker centroids)"
```

---

### Task 3: Model fetch script and test fixture

**Files:**
- Create: `scripts/fetch-models.sh`, `tests/fixtures/jfk.wav` (committed)

**Interfaces:**
- Produces: `models/ggml-<size>.bin`, `models/nemo_en_titanet_small.onnx` on disk; fixture `tests/fixtures/jfk.wav` (16 kHz mono 16-bit, ~11 s JFK speech — used by every audio test in later tasks).

- [ ] **Step 1: Write `scripts/fetch-models.sh`**

```bash
#!/usr/bin/env bash
# Usage: scripts/fetch-models.sh [whisper-size] [models-dir]
set -euo pipefail
MODEL="${1:-medium}"
DIR="${2:-models}"
mkdir -p "$DIR"

WHISPER="$DIR/ggml-${MODEL}.bin"
if [ ! -f "$WHISPER" ]; then
  echo "Downloading whisper ${MODEL} model..."
  curl -fL --progress-bar -o "$WHISPER.part" \
    "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-${MODEL}.bin"
  mv "$WHISPER.part" "$WHISPER"
fi

SPEAKER="$DIR/nemo_en_titanet_small.onnx"
if [ ! -f "$SPEAKER" ]; then
  echo "Downloading speaker embedding model..."
  curl -fL --progress-bar -o "$SPEAKER.part" \
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/nemo_en_titanet_small.onnx"
  mv "$SPEAKER.part" "$SPEAKER"
fi

echo "Models ready in $DIR"
```

(Note: `speaker-recongition-models` is the literal upstream release tag — the typo is theirs; do not "fix" it.)

- [ ] **Step 2: Fetch tiny model + fixture, verify**

```bash
chmod +x scripts/fetch-models.sh
./scripts/fetch-models.sh tiny
mkdir -p tests/fixtures
curl -fL -o tests/fixtures/jfk.wav "https://github.com/ggml-org/whisper.cpp/raw/master/samples/jfk.wav"
ls -la models tests/fixtures
```

Expected: `models/ggml-tiny.bin` (~75 MB), `models/nemo_en_titanet_small.onnx` (~25 MB), `tests/fixtures/jfk.wav` (~350 KB).

- [ ] **Step 3: Commit** (fixture yes, models no — gitignored)

```bash
git add scripts tests/fixtures && git commit -m "feat: model fetch script and jfk.wav test fixture"
```

---

### Task 4: Utterance segmenter state machine (pure logic)

**Files:**
- Create: `src/segmenter.rs`
- Modify: `src/main.rs` (add `mod segmenter;`)

**Interfaces:**
- Produces:
  - `segmenter::CHUNK: usize = 512` and `segmenter::CHUNK_MS: u64 = 32`
  - `pub enum SegmentEvent { SpeechStart, Utterance { samples: Vec<i16>, start_ms: u64, duration_ms: u64 } }` (derives `Debug, PartialEq, Clone`)
  - `pub struct SegmenterConfig { pub vad_threshold: f32, pub silence_end_ms: u32, pub pre_roll_ms: u32, pub min_speech_ms: u32, pub max_utterance_ms: u32 }`
  - `Segmenter::new(cfg: SegmenterConfig) -> Segmenter`
  - `Segmenter::push(&mut self, chunk: &[i16], prob: f32) -> Option<SegmentEvent>` — call once per 512-sample chunk with its VAD probability.
- Consumes: nothing (pure; VAD is injected by the caller).

Semantics to implement:
- **Idle:** chunks accumulate in a pre-roll ring (capacity `pre_roll_ms / 32` chunks). When `prob >= vad_threshold`, switch to Speaking, seed the utterance buffer with the pre-roll contents (which include the current chunk), record `start_ms` as the timestamp of the earliest pre-roll chunk, and return `Some(SpeechStart)`.
- **Speaking:** append every chunk to the buffer. `prob >= threshold` resets the silence counter and increments the speech-chunk count; below-threshold increments silence. When silence reaches `silence_end_ms / 32` chunks: if speech-chunk count ≥ `min_speech_ms / 32`, emit `Utterance` (buffer, start, duration = buffered-chunks × 32 ms) and go Idle; otherwise discard silently and go Idle (false trigger — this is how non-speech blips die). When the buffer reaches `max_utterance_ms / 32` chunks, emit `Utterance` immediately and stay Speaking with a fresh buffer starting at the current timestamp.
- Track a global chunk counter across the whole call history for timestamps (`chunk_index * 32` ms).

- [ ] **Step 1: Write failing tests** (in `src/segmenter.rs` `#[cfg(test)] mod tests`)

```rust
#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> SegmenterConfig {
        SegmenterConfig {
            vad_threshold: 0.5,
            silence_end_ms: 320,  // 10 chunks
            pre_roll_ms: 96,      // 3 chunks
            min_speech_ms: 160,   // 5 chunks
            max_utterance_ms: 3200, // 100 chunks
        }
    }

    fn push_n(s: &mut Segmenter, n: usize, prob: f32) -> Vec<SegmentEvent> {
        let chunk = [100i16; CHUNK];
        (0..n).filter_map(|_| s.push(&chunk, prob)).collect()
    }

    #[test]
    fn silence_produces_no_events() {
        let mut s = Segmenter::new(cfg());
        assert!(push_n(&mut s, 50, 0.1).is_empty());
    }

    #[test]
    fn speech_then_silence_emits_utterance_with_preroll() {
        let mut s = Segmenter::new(cfg());
        push_n(&mut s, 3, 0.1); // fills pre-roll
        let ev = push_n(&mut s, 20, 0.9); // speech
        assert_eq!(ev, vec![SegmentEvent::SpeechStart]);
        let ev = push_n(&mut s, 10, 0.1); // silence closes it
        match &ev[0] {
            SegmentEvent::Utterance { samples, start_ms, duration_ms } => {
                // 3 pre-roll (incl. current trigger chunk counted once) + 19 more speech + 10 silence chunks buffered
                assert_eq!(samples.len() % CHUNK, 0);
                let chunks = samples.len() / CHUNK;
                assert_eq!(chunks, 3 + 19 + 10);
                // pre-roll cap is 3: trigger chunk pushes out chunk 0, so buffer starts at chunk 1
                assert_eq!(*start_ms, CHUNK_MS);
                assert_eq!(*duration_ms, (chunks as u64) * CHUNK_MS);
            }
            other => panic!("expected Utterance, got {other:?}"),
        }
    }

    #[test]
    fn short_blip_is_discarded() {
        let mut s = Segmenter::new(cfg());
        let ev = push_n(&mut s, 2, 0.9); // only 2 speech chunks < min 5
        assert_eq!(ev, vec![SegmentEvent::SpeechStart]);
        let ev = push_n(&mut s, 10, 0.1);
        assert!(ev.is_empty(), "blip must be discarded, got {ev:?}");
    }

    #[test]
    fn long_speech_force_flushes_at_max() {
        let mut s = Segmenter::new(cfg());
        let ev = push_n(&mut s, 150, 0.9); // exceeds 100-chunk cap
        let utts: Vec<_> = ev.iter().filter(|e| matches!(e, SegmentEvent::Utterance { .. })).collect();
        assert_eq!(utts.len(), 1);
        if let SegmentEvent::Utterance { samples, .. } = utts[0] {
            assert_eq!(samples.len() / CHUNK, 100);
        }
    }
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test segmenter::`
Expected: FAIL — module/types don't exist.

- [ ] **Step 3: Implement**

```rust
use std::collections::VecDeque;

pub const CHUNK: usize = 512;
pub const CHUNK_MS: u64 = 32;

#[derive(Debug, PartialEq, Clone)]
pub enum SegmentEvent {
    SpeechStart,
    Utterance {
        samples: Vec<i16>,
        start_ms: u64,
        duration_ms: u64,
    },
}

#[derive(Debug, Clone)]
pub struct SegmenterConfig {
    pub vad_threshold: f32,
    pub silence_end_ms: u32,
    pub pre_roll_ms: u32,
    pub min_speech_ms: u32,
    pub max_utterance_ms: u32,
}

enum State {
    Idle,
    Speaking,
}

pub struct Segmenter {
    cfg: SegmenterConfig,
    state: State,
    pre_roll: VecDeque<Vec<i16>>,
    buf: Vec<i16>,
    buf_start_chunk: u64,
    silence_chunks: u32,
    speech_chunks: u32,
    chunk_index: u64,
}

impl Segmenter {
    pub fn new(cfg: SegmenterConfig) -> Self {
        Self {
            cfg,
            state: State::Idle,
            pre_roll: VecDeque::new(),
            buf: Vec::new(),
            buf_start_chunk: 0,
            silence_chunks: 0,
            speech_chunks: 0,
            chunk_index: 0,
        }
    }

    fn pre_roll_cap(&self) -> usize {
        (self.cfg.pre_roll_ms as u64 / CHUNK_MS).max(1) as usize
    }

    fn silence_limit(&self) -> u32 {
        (self.cfg.silence_end_ms as u64 / CHUNK_MS).max(1) as u32
    }

    fn min_speech(&self) -> u32 {
        (self.cfg.min_speech_ms as u64 / CHUNK_MS).max(1) as u32
    }

    fn max_chunks(&self) -> usize {
        (self.cfg.max_utterance_ms as u64 / CHUNK_MS).max(2) as usize
    }

    fn take_utterance(&mut self) -> SegmentEvent {
        let samples = std::mem::take(&mut self.buf);
        let chunks = (samples.len() / CHUNK) as u64;
        SegmentEvent::Utterance {
            samples,
            start_ms: self.buf_start_chunk * CHUNK_MS,
            duration_ms: chunks * CHUNK_MS,
        }
    }

    /// Feed exactly one 512-sample chunk with its VAD probability.
    pub fn push(&mut self, chunk: &[i16], prob: f32) -> Option<SegmentEvent> {
        debug_assert_eq!(chunk.len(), CHUNK);
        let idx = self.chunk_index;
        self.chunk_index += 1;
        let speech = prob >= self.cfg.vad_threshold;

        match self.state {
            State::Idle => {
                self.pre_roll.push_back(chunk.to_vec());
                while self.pre_roll.len() > self.pre_roll_cap() {
                    self.pre_roll.pop_front();
                }
                if speech {
                    self.state = State::Speaking;
                    self.buf_start_chunk = idx + 1 - self.pre_roll.len() as u64;
                    self.buf = self.pre_roll.drain(..).flatten().collect();
                    self.silence_chunks = 0;
                    self.speech_chunks = 1;
                    Some(SegmentEvent::SpeechStart)
                } else {
                    None
                }
            }
            State::Speaking => {
                self.buf.extend_from_slice(chunk);
                if speech {
                    self.silence_chunks = 0;
                    self.speech_chunks += 1;
                } else {
                    self.silence_chunks += 1;
                }

                if self.buf.len() / CHUNK >= self.max_chunks() {
                    let ev = self.take_utterance();
                    // stay Speaking; next utterance continues from here
                    self.buf_start_chunk = idx + 1;
                    self.silence_chunks = 0;
                    self.speech_chunks = 0;
                    return Some(ev);
                }

                if self.silence_chunks >= self.silence_limit() {
                    let enough = self.speech_chunks >= self.min_speech();
                    let ev = if enough { Some(self.take_utterance()) } else { None };
                    self.buf.clear();
                    self.state = State::Idle;
                    self.pre_roll.clear();
                    self.silence_chunks = 0;
                    self.speech_chunks = 0;
                    return ev;
                }
                None
            }
        }
    }
}
```

Add `mod segmenter;` to `src/main.rs`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test segmenter::`
Expected: PASS — 4 tests.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: VAD-gated utterance segmenter state machine"
```

---

### Task 5: Silero VAD wrapper

**Files:**
- Create: `src/vad.rs`
- Modify: `src/main.rs` (add `mod vad;`)

**Interfaces:**
- Produces: `vad::Vad` with `new() -> anyhow::Result<Vad>` and `predict(&mut self, chunk: &[i16]) -> f32` (chunk must be 512 samples @ 16 kHz). Model is bundled inside the `voice_activity_detector` crate — no file needed.
- Consumes: fixture `tests/fixtures/jfk.wav` from Task 3.

- [ ] **Step 1: Write failing test** (`#[cfg(test)]` in `src/vad.rs`)

```rust
#[cfg(test)]
mod tests {
    use super::*;

    fn read_fixture() -> Vec<i16> {
        let mut r = hound::WavReader::open("tests/fixtures/jfk.wav").unwrap();
        assert_eq!(r.spec().sample_rate, 16000);
        r.samples::<i16>().map(|s| s.unwrap()).collect()
    }

    #[test]
    fn speech_scores_high_silence_scores_low() {
        let mut vad = Vad::new().unwrap();
        let samples = read_fixture();
        let max_speech = samples
            .chunks_exact(512)
            .map(|c| vad.predict(c))
            .fold(0.0f32, f32::max);
        assert!(max_speech > 0.8, "expected speech in jfk.wav, max prob {max_speech}");

        let mut vad2 = Vad::new().unwrap();
        let silence = vec![0i16; 512];
        let p: f32 = (0..10).map(|_| vad2.predict(&silence)).fold(0.0, f32::max);
        assert!(p < 0.3, "silence scored {p}");
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test vad::`
Expected: FAIL — `Vad` not defined.

- [ ] **Step 3: Implement `src/vad.rs`**

```rust
use anyhow::Result;
use voice_activity_detector::VoiceActivityDetector;

/// Silero V5 VAD. 16 kHz, 512-sample chunks.
pub struct Vad {
    inner: VoiceActivityDetector,
}

impl Vad {
    pub fn new() -> Result<Self> {
        let inner = VoiceActivityDetector::builder()
            .sample_rate(16000)
            .chunk_size(512usize)
            .build()?;
        Ok(Self { inner })
    }

    pub fn predict(&mut self, chunk: &[i16]) -> f32 {
        self.inner.predict(chunk.iter().copied())
    }
}
```

Add `mod vad;` to `src/main.rs`.

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test vad::`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: Silero VAD wrapper"
```

---

### Task 6: Whisper engine (transcribe + translate)

**Files:**
- Create: `src/whisper.rs`
- Modify: `src/main.rs` (add `mod whisper;`)

**Interfaces:**
- Produces:
  - `whisper::WhisperEngine` with `new(model_path: &Path, threads: i32) -> anyhow::Result<WhisperEngine>` (Send + Sync; share via `Arc`)
  - `transcribe(&self, audio: &[f32]) -> anyhow::Result<Transcription>` — original language text, auto-detected language
  - `translate(&self, audio: &[f32]) -> anyhow::Result<Transcription>` — English text (lang field = detected source language)
  - `pub struct Transcription { pub lang: String, pub text: String }` (derives `Debug, Clone`)
  - `whisper::i16_to_f32(samples: &[i16]) -> Vec<f32>`
- Consumes: `models/ggml-tiny.bin` (tests), fixture jfk.wav.

- [ ] **Step 1: Write failing test**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    fn engine() -> WhisperEngine {
        let p = Path::new("models/ggml-tiny.bin");
        assert!(p.exists(), "run scripts/fetch-models.sh tiny");
        WhisperEngine::new(p, 4).unwrap()
    }

    fn fixture_f32() -> Vec<f32> {
        let mut r = hound::WavReader::open("tests/fixtures/jfk.wav").unwrap();
        let s: Vec<i16> = r.samples::<i16>().map(|s| s.unwrap()).collect();
        i16_to_f32(&s)
    }

    #[test]
    fn transcribes_and_translates_jfk() {
        let e = engine();
        let audio = fixture_f32();

        let t = e.transcribe(&audio).unwrap();
        assert_eq!(t.lang, "en");
        assert!(t.text.to_lowercase().contains("country"), "got: {}", t.text);

        let tr = e.translate(&audio).unwrap();
        assert!(tr.text.to_lowercase().contains("country"), "got: {}", tr.text);
    }

    #[test]
    fn short_audio_is_padded_not_crashing() {
        let e = engine();
        let audio = vec![0.0f32; 4000]; // 0.25 s of silence
        let t = e.transcribe(&audio).unwrap();
        assert!(t.text.len() < 200); // whatever it hallucinates, it must not crash
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test whisper:: -- --test-threads=1`
Expected: FAIL — module doesn't exist.

- [ ] **Step 3: Implement `src/whisper.rs`**

```rust
use anyhow::Result;
use std::path::Path;
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

/// Minimum samples whisper.cpp handles well: 1.1 s @ 16 kHz.
const MIN_SAMPLES: usize = 17_600;

#[derive(Debug, Clone)]
pub struct Transcription {
    pub lang: String,
    pub text: String,
}

pub struct WhisperEngine {
    ctx: WhisperContext,
    threads: i32,
}

pub fn i16_to_f32(samples: &[i16]) -> Vec<f32> {
    samples.iter().map(|s| *s as f32 / 32768.0).collect()
}

impl WhisperEngine {
    pub fn new(model_path: &Path, threads: i32) -> Result<Self> {
        let ctx = WhisperContext::new_with_params(
            model_path.to_str().expect("model path must be utf-8"),
            WhisperContextParameters::default(),
        )?;
        Ok(Self { ctx, threads })
    }

    fn run(&self, audio: &[f32], translate: bool) -> Result<Transcription> {
        let mut padded;
        let audio = if audio.len() < MIN_SAMPLES {
            padded = audio.to_vec();
            padded.resize(MIN_SAMPLES, 0.0);
            &padded[..]
        } else {
            audio
        };

        let mut state = self.ctx.create_state()?;
        let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
        params.set_language(Some("auto"));
        params.set_translate(translate);
        params.set_n_threads(self.threads);
        params.set_print_special(false);
        params.set_print_progress(false);
        params.set_print_realtime(false);
        params.set_print_timestamps(false);
        params.set_suppress_blank(true);
        params.set_no_context(true);
        state.full(params, audio)?;

        let lang = whisper_rs::get_lang_str(state.full_lang_id_from_state()?)
            .unwrap_or("unknown")
            .to_string();
        let n = state.full_n_segments()?;
        let mut text = String::new();
        for i in 0..n {
            text.push_str(&state.full_get_segment_text(i)?);
        }
        Ok(Transcription {
            lang,
            text: text.trim().to_string(),
        })
    }

    pub fn transcribe(&self, audio: &[f32]) -> Result<Transcription> {
        self.run(audio, false)
    }

    pub fn translate(&self, audio: &[f32]) -> Result<Transcription> {
        self.run(audio, true)
    }
}
```

Add `mod whisper;` to `src/main.rs`.

(whisper-rs 0.16 API notes: if `full_lang_id_from_state` doesn't exist, it's `state.full_lang_id()`; if `full_n_segments` returns plain `c_int`, drop the `?`. Adapt call sites only.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test whisper:: -- --test-threads=1` (background; tiny model decode of 11 s audio ×3 ≈ under a minute on M4 Pro)
Expected: PASS — 2 tests.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: whisper engine with transcribe and translate"
```

---

### Task 7: Speaker embeddings and online clustering

**Files:**
- Create: `src/speakers.rs`
- Modify: `src/main.rs` (add `mod speakers;`)

**Interfaces:**
- Produces:
  - `speakers::SpeakerEmbedder` with `new(model_path: &Path) -> anyhow::Result<SpeakerEmbedder>` and `embed(&mut self, samples: &[f32]) -> anyhow::Result<Vec<f32>>` (16 kHz assumed internally)
  - `speakers::SpeakerRegistry` with `new(threshold: f32) -> SpeakerRegistry`, `restore(threshold: f32, centroids: Vec<Vec<f32>>) -> SpeakerRegistry` (ordered by speaker number), `assign(&mut self, emb: &[f32]) -> usize` (1-based person number; updates centroid running mean), `centroid(&self, num: usize) -> &[f32]`, `len(&self) -> usize`
- Consumes: `models/nemo_en_titanet_small.onnx` (Task 3), fixture jfk.wav.

- [ ] **Step 1: Write failing tests**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    #[test]
    fn registry_clusters_and_updates_centroids() {
        let mut reg = SpeakerRegistry::new(0.65);
        // two nearly-orthogonal directions
        let a = vec![1.0, 0.0, 0.05];
        let b = vec![0.0, 1.0, 0.05];
        assert_eq!(reg.assign(&a), 1);
        assert_eq!(reg.assign(&b), 2);
        // close to a -> speaker 1 again
        assert_eq!(reg.assign(&[0.95, 0.05, 0.0]), 1);
        assert_eq!(reg.len(), 2);
        // centroid moved toward the new sample
        assert!(reg.centroid(1)[0] < 1.0 && reg.centroid(1)[0] > 0.9);
    }

    #[test]
    fn restore_keeps_numbering() {
        let mut reg = SpeakerRegistry::restore(0.65, vec![vec![1.0, 0.0], vec![0.0, 1.0]]);
        assert_eq!(reg.assign(&[0.0, 0.9]), 2);
        assert_eq!(reg.assign(&[0.7, 0.7]), 3); // 45° off both, below threshold
    }

    #[test]
    fn same_voice_embeds_similarly() {
        let p = Path::new("models/nemo_en_titanet_small.onnx");
        assert!(p.exists(), "run scripts/fetch-models.sh tiny");
        let mut e = SpeakerEmbedder::new(p).unwrap();
        let mut r = hound::WavReader::open("tests/fixtures/jfk.wav").unwrap();
        let s: Vec<f32> = r
            .samples::<i16>()
            .map(|s| s.unwrap() as f32 / 32768.0)
            .collect();
        let half = s.len() / 2;
        let e1 = e.embed(&s[..half]).unwrap();
        let e2 = e.embed(&s[half..]).unwrap();
        assert!(!e1.is_empty());
        let sim = cosine(&e1, &e2);
        assert!(sim > 0.5, "same speaker halves scored {sim}");
    }
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test speakers::`
Expected: FAIL — module doesn't exist.

- [ ] **Step 3: Implement `src/speakers.rs`**

```rust
use anyhow::Result;
use sherpa_rs::speaker_id::{EmbeddingExtractor, ExtractorConfig};
use std::path::Path;

pub struct SpeakerEmbedder {
    inner: EmbeddingExtractor,
}

impl SpeakerEmbedder {
    pub fn new(model_path: &Path) -> Result<Self> {
        let config = ExtractorConfig {
            model: model_path.to_string_lossy().into_owned(),
            provider: None,
            num_threads: Some(2),
            debug: false,
        };
        let inner = EmbeddingExtractor::new(config)?;
        Ok(Self { inner })
    }

    pub fn embed(&mut self, samples: &[f32]) -> Result<Vec<f32>> {
        self.inner.compute_speaker_embedding(samples.to_vec(), 16000)
    }
}

pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na == 0.0 || nb == 0.0 {
        0.0
    } else {
        dot / (na * nb)
    }
}

/// Online speaker clustering: cosine similarity against running centroids.
pub struct SpeakerRegistry {
    threshold: f32,
    speakers: Vec<(Vec<f32>, u32)>, // (centroid, sample count)
}

impl SpeakerRegistry {
    pub fn new(threshold: f32) -> Self {
        Self { threshold, speakers: Vec::new() }
    }

    /// Restore from persisted centroids ordered by speaker number.
    pub fn restore(threshold: f32, centroids: Vec<Vec<f32>>) -> Self {
        Self {
            threshold,
            speakers: centroids.into_iter().map(|c| (c, 1)).collect(),
        }
    }

    pub fn len(&self) -> usize {
        self.speakers.len()
    }

    pub fn centroid(&self, num: usize) -> &[f32] {
        &self.speakers[num - 1].0
    }

    /// Returns the 1-based speaker number, creating a new speaker if nothing matches.
    pub fn assign(&mut self, emb: &[f32]) -> usize {
        let mut best: Option<(usize, f32)> = None;
        for (i, (c, _)) in self.speakers.iter().enumerate() {
            let sim = cosine(c, emb);
            if sim >= self.threshold && best.map_or(true, |(_, b)| sim > b) {
                best = Some((i, sim));
            }
        }
        match best {
            Some((i, _)) => {
                let (c, n) = &mut self.speakers[i];
                *n += 1;
                let k = 1.0 / *n as f32;
                for (cv, ev) in c.iter_mut().zip(emb) {
                    *cv += (ev - *cv) * k;
                }
                i + 1
            }
            None => {
                self.speakers.push((emb.to_vec(), 1));
                self.speakers.len()
            }
        }
    }
}
```

Add `mod speakers;` to `src/main.rs`.

(sherpa-rs note: if `ExtractorConfig` field names differ in the resolved 0.6.x, check `docs.rs/sherpa-rs` — the constructor may be `ExtractorConfig::new(model, provider, num_threads, debug)` in older patches. Keep `SpeakerEmbedder`'s public interface unchanged.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test speakers:: -- --test-threads=1`
Expected: PASS — 3 tests.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: speaker embeddings and online clustering registry"
```

---

### Task 8: Pipeline orchestrator

**Files:**
- Create: `src/pipeline.rs`
- Modify: `src/main.rs` (add `mod pipeline;`)

**Interfaces:**
- Consumes: everything from Tasks 2, 4, 5, 6, 7 exactly as their Interfaces blocks define.
- Produces:
  - `pipeline::PipelineEvent` — `#[derive(Serialize, Clone, Debug)] #[serde(tag = "type", rename_all = "snake_case")] pub enum PipelineEvent { SessionStarted { session_id: i64 }, SpeechStart, Transcribing, Utterance { speaker: usize, lang: String, original_text: String, english_text: String, start_ms: u64, duration_ms: u64 } }`
  - `pipeline::PipelineDeps { pub whisper: Arc<WhisperEngine>, pub db: Arc<Mutex<Db>>, pub cfg: Config }` (all `pub`, struct is `Clone`)
  - `pipeline::run_pipeline(deps: PipelineDeps, session_id: i64, audio_rx: std::sync::mpsc::Receiver<Vec<i16>>, event_tx: tokio::sync::mpsc::UnboundedSender<PipelineEvent>)` — blocking; run on a dedicated `std::thread`. Exits when `audio_rx` closes.

Behavior:
1. On start: send `SessionStarted`. Load persisted centroids via `db.load_speakers(session_id)` into `SpeakerRegistry::restore` (enables reconnect-resume with stable numbering). Create `Vad` and `SpeakerEmbedder` inside the thread.
2. Accumulate incoming `Vec<i16>` into a pending buffer; while ≥ 512 samples, drain a chunk, run VAD, feed segmenter, forward `SpeechStart` events.
3. On `Utterance`: send `Transcribing`; convert to f32; if `cfg.english_only`, run only `translate` and use its text for `english_text` with `original_text = ""`; else run `transcribe` (original + lang) then `translate` (english). Embed → `registry.assign` → persist centroid via `upsert_speaker`. Insert `UtteranceRow`. If it's the session's first utterance, `set_session_title` to first 8 words of english text. Send `Utterance` event.
4. Skip utterances whose translate text is empty after trim (whisper returned nothing).

- [ ] **Step 1: Write failing integration-style test** (`#[cfg(test)]` in `src/pipeline.rs`)

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;
    use crate::db::Db;
    use crate::whisper::WhisperEngine;
    use std::path::Path;
    use std::sync::{Arc, Mutex};

    #[test]
    fn jfk_through_pipeline_yields_english_utterance() {
        assert!(Path::new("models/ggml-tiny.bin").exists(), "run scripts/fetch-models.sh tiny");
        let mut cfg = Config::default();
        cfg.whisper_model = "tiny".into();

        let whisper = Arc::new(WhisperEngine::new(&cfg.whisper_model_path(), 4).unwrap());
        let db = Arc::new(Mutex::new(Db::open_in_memory().unwrap()));
        let session_id = db.lock().unwrap().create_session().unwrap();

        let (audio_tx, audio_rx) = std::sync::mpsc::channel::<Vec<i16>>();
        let (event_tx, mut event_rx) = tokio::sync::mpsc::unbounded_channel();

        let deps = PipelineDeps { whisper, db: db.clone(), cfg };
        let handle = std::thread::spawn(move || run_pipeline(deps, session_id, audio_rx, event_tx));

        let mut r = hound::WavReader::open("tests/fixtures/jfk.wav").unwrap();
        let samples: Vec<i16> = r.samples::<i16>().map(|s| s.unwrap()).collect();
        for chunk in samples.chunks(4096) {
            audio_tx.send(chunk.to_vec()).unwrap();
        }
        // a second of silence so the segmenter closes the last utterance
        for _ in 0..8 {
            audio_tx.send(vec![0i16; 4096]).unwrap();
        }
        drop(audio_tx);
        handle.join().unwrap();

        let mut utterances = Vec::new();
        while let Ok(ev) = event_rx.try_recv() {
            if let PipelineEvent::Utterance { english_text, speaker, .. } = ev {
                utterances.push((speaker, english_text));
            }
        }
        assert!(!utterances.is_empty(), "no utterances produced");
        let all_text: String = utterances.iter().map(|(_, t)| t.to_lowercase()).collect();
        assert!(all_text.contains("country"), "got: {all_text}");
        assert!(utterances.iter().all(|(s, _)| *s == 1), "one voice must be one speaker");

        let rows = db.lock().unwrap().get_utterances(session_id).unwrap();
        assert_eq!(rows.len(), utterances.len());
        let sessions = db.lock().unwrap().list_sessions().unwrap();
        assert!(!sessions[0].title.is_empty(), "title should be auto-set");
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test pipeline:: -- --test-threads=1`
Expected: FAIL — module doesn't exist.

- [ ] **Step 3: Implement `src/pipeline.rs`**

```rust
use crate::config::Config;
use crate::db::{Db, UtteranceRow};
use crate::segmenter::{SegmentEvent, Segmenter, SegmenterConfig, CHUNK};
use crate::speakers::{SpeakerEmbedder, SpeakerRegistry};
use crate::vad::Vad;
use crate::whisper::{i16_to_f32, WhisperEngine};
use serde::Serialize;
use std::sync::{Arc, Mutex};

#[derive(Serialize, Clone, Debug)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum PipelineEvent {
    SessionStarted {
        session_id: i64,
    },
    SpeechStart,
    Transcribing,
    Utterance {
        speaker: usize,
        lang: String,
        original_text: String,
        english_text: String,
        start_ms: u64,
        duration_ms: u64,
    },
}

#[derive(Clone)]
pub struct PipelineDeps {
    pub whisper: Arc<WhisperEngine>,
    pub db: Arc<Mutex<Db>>,
    pub cfg: Config,
}

pub fn run_pipeline(
    deps: PipelineDeps,
    session_id: i64,
    audio_rx: std::sync::mpsc::Receiver<Vec<i16>>,
    event_tx: tokio::sync::mpsc::UnboundedSender<PipelineEvent>,
) {
    let _ = event_tx.send(PipelineEvent::SessionStarted { session_id });

    let mut vad = match Vad::new() {
        Ok(v) => v,
        Err(e) => {
            tracing::error!("VAD init failed: {e:#}");
            return;
        }
    };
    let mut embedder = match SpeakerEmbedder::new(&deps.cfg.speaker_model_path()) {
        Ok(e) => e,
        Err(e) => {
            tracing::error!("speaker embedder init failed: {e:#}");
            return;
        }
    };

    let persisted = deps
        .db
        .lock()
        .unwrap()
        .load_speakers(session_id)
        .unwrap_or_default();
    let mut registry = SpeakerRegistry::restore(
        deps.cfg.speaker_similarity_threshold,
        persisted.into_iter().map(|(_, c)| c).collect(),
    );
    let mut first_utterance = deps
        .db
        .lock()
        .unwrap()
        .get_utterances(session_id)
        .map(|u| u.is_empty())
        .unwrap_or(true);

    let mut segmenter = Segmenter::new(SegmenterConfig {
        vad_threshold: deps.cfg.vad_threshold,
        silence_end_ms: deps.cfg.silence_end_ms,
        pre_roll_ms: deps.cfg.pre_roll_ms,
        min_speech_ms: deps.cfg.min_speech_ms,
        max_utterance_ms: deps.cfg.max_utterance_s * 1000,
    });

    let mut pending: Vec<i16> = Vec::new();

    while let Ok(samples) = audio_rx.recv() {
        pending.extend_from_slice(&samples);
        let mut offset = 0;
        while pending.len() - offset >= CHUNK {
            let chunk = &pending[offset..offset + CHUNK];
            let prob = vad.predict(chunk);
            if let Some(ev) = segmenter.push(chunk, prob) {
                match ev {
                    SegmentEvent::SpeechStart => {
                        let _ = event_tx.send(PipelineEvent::SpeechStart);
                    }
                    SegmentEvent::Utterance { samples, start_ms, duration_ms } => {
                        let _ = event_tx.send(PipelineEvent::Transcribing);
                        handle_utterance(
                            &deps, session_id, &samples, start_ms, duration_ms,
                            &mut embedder, &mut registry, &mut first_utterance, &event_tx,
                        );
                    }
                }
            }
            offset += CHUNK;
        }
        pending.drain(..offset);
    }
    tracing::info!("pipeline for session {session_id} finished");
}

#[allow(clippy::too_many_arguments)]
fn handle_utterance(
    deps: &PipelineDeps,
    session_id: i64,
    samples: &[i16],
    start_ms: u64,
    duration_ms: u64,
    embedder: &mut SpeakerEmbedder,
    registry: &mut SpeakerRegistry,
    first_utterance: &mut bool,
    event_tx: &tokio::sync::mpsc::UnboundedSender<PipelineEvent>,
) {
    let audio = i16_to_f32(samples);

    let (lang, original_text, english_text) = if deps.cfg.english_only {
        match deps.whisper.translate(&audio) {
            Ok(t) => (t.lang, String::new(), t.text),
            Err(e) => {
                tracing::error!("translate failed: {e:#}");
                return;
            }
        }
    } else {
        let orig = match deps.whisper.transcribe(&audio) {
            Ok(t) => t,
            Err(e) => {
                tracing::error!("transcribe failed: {e:#}");
                return;
            }
        };
        let eng = match deps.whisper.translate(&audio) {
            Ok(t) => t.text,
            Err(e) => {
                tracing::error!("translate failed: {e:#}");
                return;
            }
        };
        (orig.lang, orig.text, eng)
    };

    if english_text.trim().is_empty() && original_text.trim().is_empty() {
        return; // whisper heard nothing worth keeping
    }

    let speaker = match embedder.embed(&audio) {
        Ok(emb) => {
            let num = registry.assign(&emb);
            let _ = deps.db.lock().unwrap().upsert_speaker(
                session_id,
                num as i64,
                registry.centroid(num),
            );
            num
        }
        Err(e) => {
            tracing::warn!("embedding failed, defaulting speaker 1: {e:#}");
            1
        }
    };

    let row = UtteranceRow {
        speaker_num: speaker as i64,
        lang: lang.clone(),
        original_text: original_text.clone(),
        english_text: english_text.clone(),
        start_ms: start_ms as i64,
        duration_ms: duration_ms as i64,
    };
    if let Err(e) = deps.db.lock().unwrap().insert_utterance(session_id, &row) {
        tracing::error!("db insert failed: {e:#}");
    }

    if *first_utterance {
        let title: String = english_text.split_whitespace().take(8).collect::<Vec<_>>().join(" ");
        let _ = deps.db.lock().unwrap().set_session_title(session_id, &title);
        *first_utterance = false;
    }

    let _ = event_tx.send(PipelineEvent::Utterance {
        speaker,
        lang,
        original_text,
        english_text,
        start_ms,
        duration_ms,
    });
}
```

Add `mod pipeline;` to `src/main.rs`.

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test pipeline:: -- --test-threads=1` (background; expect ~1–3 min)
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: pipeline orchestrator (VAD -> segmenter -> whisper -> speakers -> db)"
```

---

### Task 9: HTTP/WebSocket server and full main.rs

**Files:**
- Create: `src/server.rs`, `tests/server.rs`
- Modify: `src/main.rs` (full rewrite below), `src/lib.rs` (new — re-export modules so the integration test can use them)

**Interfaces:**
- Consumes: `PipelineDeps`, `run_pipeline`, `PipelineEvent`, `Db`, `WhisperEngine`, `Config`.
- Produces:
  - `server::AppState { pub whisper: Arc<WhisperEngine>, pub db: Arc<Mutex<Db>>, pub cfg: Config }`
  - `server::router(state: Arc<AppState>) -> axum::Router`
  - Routes: `GET /healthz` → `"ok"`; `GET /api/sessions` → JSON array of `SessionSummary`; `GET /api/sessions/{id}` → `{"id":…,"started_at":…,"title":…,"utterances":[UtteranceRow…]}` (404 if unknown); `GET /ws?session=<id optional>` → WebSocket (binary frames in = LE i16 16 kHz PCM; text frames out = `PipelineEvent` JSON); static files from `static/` as fallback.

- [ ] **Step 1: Create `src/lib.rs` and slim `src/main.rs`**

`src/lib.rs`:
```rust
pub mod config;
pub mod db;
pub mod pipeline;
pub mod segmenter;
pub mod server;
pub mod speakers;
pub mod vad;
pub mod whisper;
```

`src/main.rs` (full replacement — module decls move to lib.rs):
```rust
use live_transcriber::{config::Config, db::Db, server, whisper::WhisperEngine};
use std::sync::{Arc, Mutex};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt().init();
    let cfg = Config::load("config.toml")?;
    std::fs::create_dir_all(&cfg.data_dir)?;

    let whisper_path = cfg.whisper_model_path();
    anyhow::ensure!(
        whisper_path.exists(),
        "whisper model missing at {whisper_path:?} — run: scripts/fetch-models.sh {}",
        cfg.whisper_model
    );
    anyhow::ensure!(
        cfg.speaker_model_path().exists(),
        "speaker model missing — run: scripts/fetch-models.sh"
    );

    tracing::info!("loading whisper model {:?}…", whisper_path);
    let whisper = Arc::new(WhisperEngine::new(&whisper_path, cfg.whisper_threads)?);
    let db = Arc::new(Mutex::new(Db::open(&format!("{}/transcriber.db", cfg.data_dir))?));

    let state = Arc::new(server::AppState { whisper, db, cfg: cfg.clone() });
    let app = server::router(state);
    let addr: std::net::SocketAddr = format!("0.0.0.0:{}", cfg.port).parse()?;

    if cfg.tls {
        let cert_path = format!("{}/cert.pem", cfg.data_dir);
        let key_path = format!("{}/key.pem", cfg.data_dir);
        if !std::path::Path::new(&cert_path).exists() {
            let cert = rcgen::generate_simple_self_signed(vec!["localhost".into()])?;
            std::fs::write(&cert_path, cert.cert.pem())?;
            std::fs::write(&key_path, cert.key_pair.serialize_pem())?;
            tracing::info!("generated self-signed cert in {}", cfg.data_dir);
        }
        let tls = axum_server::tls_rustls::RustlsConfig::from_pem_file(&cert_path, &key_path).await?;
        tracing::info!("listening on https://{addr}");
        axum_server::bind_rustls(addr, tls)
            .serve(app.into_make_service())
            .await?;
    } else {
        tracing::info!("listening on http://{addr}");
        let listener = tokio::net::TcpListener::bind(addr).await?;
        axum::serve(listener, app).await?;
    }
    Ok(())
}
```

- [ ] **Step 2: Write failing integration test `tests/server.rs`**

```rust
use futures_util::{SinkExt, StreamExt};
use live_transcriber::{config::Config, db::Db, server, whisper::WhisperEngine};
use std::sync::{Arc, Mutex};
use tokio_tungstenite::tungstenite::Message;

async fn spawn_server() -> (String, Arc<server::AppState>) {
    assert!(
        std::path::Path::new("models/ggml-tiny.bin").exists(),
        "run scripts/fetch-models.sh tiny"
    );
    let mut cfg = Config::default();
    cfg.whisper_model = "tiny".into();
    let whisper = Arc::new(WhisperEngine::new(&cfg.whisper_model_path(), 4).unwrap());
    let db = Arc::new(Mutex::new(Db::open_in_memory().unwrap()));
    let state = Arc::new(server::AppState { whisper, db, cfg });
    let app = server::router(state.clone());
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    (format!("127.0.0.1:{}", addr.port()), state)
}

#[tokio::test]
async fn healthz_and_empty_sessions() {
    let (addr, _state) = spawn_server().await;
    let body = http_get(&format!("http://{addr}/healthz")).await;
    assert_eq!(body, "ok");
    let body = http_get(&format!("http://{addr}/api/sessions")).await;
    assert_eq!(body.trim(), "[]");
}

#[tokio::test]
async fn ws_creates_session_and_reports_it() {
    let (addr, state) = spawn_server().await;
    let (mut ws, _) = tokio_tungstenite::connect_async(format!("ws://{addr}/ws"))
        .await
        .unwrap();
    // first message must be session_started
    let msg = tokio::time::timeout(std::time::Duration::from_secs(10), ws.next())
        .await
        .unwrap()
        .unwrap()
        .unwrap();
    let v: serde_json::Value = serde_json::from_str(msg.to_text().unwrap()).unwrap();
    assert_eq!(v["type"], "session_started");
    let sid = v["session_id"].as_i64().unwrap();

    // stream a little silence, then close; session must exist and be ended
    ws.send(Message::Binary(vec![0u8; 8192].into())).await.unwrap();
    ws.close(None).await.unwrap();
    tokio::time::sleep(std::time::Duration::from_millis(500)).await;
    assert!(state.db.lock().unwrap().session_exists(sid).unwrap());
}

// Minimal HTTP/1.0 GET over std TcpStream — avoids adding an HTTP client dependency.
async fn http_get(url: &str) -> String {
    let url = url.strip_prefix("http://").unwrap().to_string();
    let (host, path) = url.split_once('/').map(|(h, p)| (h.to_string(), format!("/{p}"))).unwrap();
    tokio::task::spawn_blocking(move || {
        use std::io::{Read, Write};
        let mut s = std::net::TcpStream::connect(&host).unwrap();
        write!(s, "GET {path} HTTP/1.0\r\nHost: {host}\r\n\r\n").unwrap();
        let mut buf = String::new();
        s.read_to_string(&mut buf).unwrap();
        buf.split("\r\n\r\n").nth(1).unwrap_or("").to_string()
    })
    .await
    .unwrap()
}
```

- [ ] **Step 3: Run test to verify it fails**

Run: `cargo test --test server -- --test-threads=1`
Expected: FAIL — `server` module doesn't exist.

- [ ] **Step 4: Implement `src/server.rs`**

```rust
use crate::config::Config;
use crate::db::Db;
use crate::pipeline::{run_pipeline, PipelineDeps, PipelineEvent};
use crate::whisper::WhisperEngine;
use axum::{
    extract::{
        ws::{Message, WebSocket, WebSocketUpgrade},
        Path, Query, State,
    },
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::get,
    Json, Router,
};
use futures_util::{SinkExt, StreamExt};
use serde::Deserialize;
use serde_json::json;
use std::sync::{Arc, Mutex};
use tower_http::services::ServeDir;

pub struct AppState {
    pub whisper: Arc<WhisperEngine>,
    pub db: Arc<Mutex<Db>>,
    pub cfg: Config,
}

pub fn router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/healthz", get(|| async { "ok" }))
        .route("/api/sessions", get(list_sessions))
        .route("/api/sessions/{id}", get(get_session))
        .route("/ws", get(ws_handler))
        .fallback_service(ServeDir::new("static"))
        .with_state(state)
}

async fn list_sessions(State(state): State<Arc<AppState>>) -> Response {
    match state.db.lock().unwrap().list_sessions() {
        Ok(s) => Json(s).into_response(),
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()).into_response(),
    }
}

async fn get_session(
    State(state): State<Arc<AppState>>,
    Path(id): Path<i64>,
) -> Response {
    let db = state.db.lock().unwrap();
    match (db.session_exists(id), db.get_utterances(id)) {
        (Ok(true), Ok(utts)) => {
            let summary = db
                .list_sessions()
                .unwrap_or_default()
                .into_iter()
                .find(|s| s.id == id);
            Json(json!({
                "id": id,
                "started_at": summary.as_ref().map(|s| s.started_at.clone()),
                "title": summary.as_ref().map(|s| s.title.clone()),
                "utterances": utts,
            }))
            .into_response()
        }
        (Ok(false), _) => (StatusCode::NOT_FOUND, "no such session").into_response(),
        (Err(e), _) | (_, Err(e)) => {
            (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()).into_response()
        }
    }
}

#[derive(Deserialize)]
struct WsQuery {
    session: Option<i64>,
}

async fn ws_handler(
    ws: WebSocketUpgrade,
    Query(q): Query<WsQuery>,
    State(state): State<Arc<AppState>>,
) -> Response {
    ws.on_upgrade(move |socket| handle_socket(socket, state, q.session))
}

async fn handle_socket(socket: WebSocket, state: Arc<AppState>, resume: Option<i64>) {
    let session_id = {
        let db = state.db.lock().unwrap();
        match resume {
            Some(id) if db.session_exists(id).unwrap_or(false) => id,
            _ => match db.create_session() {
                Ok(id) => id,
                Err(e) => {
                    tracing::error!("create_session failed: {e:#}");
                    return;
                }
            },
        }
    };

    let (mut sink, mut stream) = socket.split();
    let (audio_tx, audio_rx) = std::sync::mpsc::channel::<Vec<i16>>();
    let (event_tx, mut event_rx) = tokio::sync::mpsc::unbounded_channel::<PipelineEvent>();

    let deps = PipelineDeps {
        whisper: state.whisper.clone(),
        db: state.db.clone(),
        cfg: state.cfg.clone(),
    };
    std::thread::spawn(move || run_pipeline(deps, session_id, audio_rx, event_tx));

    let send_task = tokio::spawn(async move {
        while let Some(ev) = event_rx.recv().await {
            let msg = match serde_json::to_string(&ev) {
                Ok(m) => m,
                Err(_) => continue,
            };
            if sink.send(Message::Text(msg.into())).await.is_err() {
                break;
            }
        }
    });

    while let Some(Ok(msg)) = stream.next().await {
        match msg {
            Message::Binary(b) => {
                let samples: Vec<i16> = b
                    .chunks_exact(2)
                    .map(|c| i16::from_le_bytes([c[0], c[1]]))
                    .collect();
                if audio_tx.send(samples).is_err() {
                    break;
                }
            }
            Message::Close(_) => break,
            _ => {}
        }
    }

    drop(audio_tx); // pipeline thread drains and exits
    let _ = send_task.await;
    let _ = state.db.lock().unwrap().end_session(session_id);
    tracing::info!("session {session_id} socket closed");
}
```

- [ ] **Step 5: Run all tests**

Run: `cargo test -- --test-threads=1` (background)
Expected: PASS — all unit tests plus 2 integration tests.

- [ ] **Step 6: Commit**

```bash
git add -A && git commit -m "feat: websocket + REST server, TLS option, full startup wiring"
```

---

### Task 10: Web UI (live view + history)

**Files:**
- Create: `static/app.js`, `static/worklet.js`, `static/style.css`
- Modify: `static/index.html` (full replacement)

**Interfaces:**
- Consumes: WS `PipelineEvent` JSON (`session_started`, `speech_start`, `transcribing`, `utterance` with fields exactly as Task 8 defines), REST `/api/sessions` and `/api/sessions/{id}` shapes from Task 9. Sends binary LE i16 16 kHz PCM.

- [ ] **Step 1: `static/index.html` (full replacement)**

```html
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Live Transcriber</title>
<link rel="stylesheet" href="style.css">
</head>
<body>
<header>
  <h1>Live Transcriber</h1>
  <nav>
    <button id="tab-live" class="tab active">Live</button>
    <button id="tab-history" class="tab">History</button>
  </nav>
</header>

<main id="view-live">
  <div class="controls">
    <button id="btn-mic" class="mic">🎤 Start listening</button>
    <span id="status" class="status idle">idle</span>
    <label class="toggle"><input type="checkbox" id="show-original" checked> show original</label>
  </div>
  <div id="banner" class="banner hidden"></div>
  <div id="transcript" class="transcript"></div>
</main>

<main id="view-history" class="hidden">
  <div id="session-list"></div>
  <div id="session-detail" class="transcript"></div>
</main>

<script src="app.js"></script>
</body>
</html>
```

- [ ] **Step 2: `static/worklet.js`**

```js
// Posts raw Float32Array frames (at the AudioContext's native rate) to the main thread.
class CaptureProcessor extends AudioWorkletProcessor {
  process(inputs) {
    const ch = inputs[0] && inputs[0][0];
    if (ch) this.port.postMessage(ch.slice(0));
    return true;
  }
}
registerProcessor('capture', CaptureProcessor);
```

- [ ] **Step 3: `static/app.js`**

```js
const $ = (id) => document.getElementById(id);
const TARGET_RATE = 16000;
const SEND_BATCH = 2048; // samples (~128 ms)

let ws = null;
let audioCtx = null;
let workletNode = null;
let mediaStream = null;
let sessionId = null;
let running = false;
let sendBuf = new Int16Array(0);

// ---------- audio ----------
function downsample(f32, fromRate) {
  const ratio = fromRate / TARGET_RATE;
  const outLen = Math.floor(f32.length / ratio);
  const out = new Int16Array(outLen);
  for (let i = 0; i < outLen; i++) {
    const pos = i * ratio;
    const i0 = Math.floor(pos);
    const i1 = Math.min(i0 + 1, f32.length - 1);
    const s = f32[i0] + (f32[i1] - f32[i0]) * (pos - i0);
    out[i] = Math.max(-32768, Math.min(32767, Math.round(s * 32767)));
  }
  return out;
}

function queueSamples(int16) {
  const merged = new Int16Array(sendBuf.length + int16.length);
  merged.set(sendBuf); merged.set(int16, sendBuf.length);
  sendBuf = merged;
  while (sendBuf.length >= SEND_BATCH) {
    const out = sendBuf.slice(0, SEND_BATCH);
    sendBuf = sendBuf.slice(SEND_BATCH);
    if (ws && ws.readyState === WebSocket.OPEN) ws.send(out.buffer);
  }
}

async function startMic() {
  try {
    mediaStream = await navigator.mediaDevices.getUserMedia({
      audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true },
    });
  } catch (e) {
    banner(`Microphone access denied or unavailable (${e.name}). ` +
      `Allow mic access in your browser settings. On non-localhost addresses the page must be HTTPS.`);
    return false;
  }
  audioCtx = new AudioContext();
  await audioCtx.audioWorklet.addModule('worklet.js');
  const src = audioCtx.createMediaStreamSource(mediaStream);
  workletNode = new AudioWorkletNode(audioCtx, 'capture');
  workletNode.port.onmessage = (e) => queueSamples(downsample(e.data, audioCtx.sampleRate));
  src.connect(workletNode);
  return true;
}

function stopMic() {
  if (workletNode) workletNode.disconnect();
  if (audioCtx) audioCtx.close();
  if (mediaStream) mediaStream.getTracks().forEach((t) => t.stop());
  workletNode = audioCtx = mediaStream = null;
}

// ---------- websocket ----------
function connect() {
  const proto = location.protocol === 'https:' ? 'wss' : 'ws';
  const q = sessionId ? `?session=${sessionId}` : '';
  ws = new WebSocket(`${proto}://${location.host}/ws${q}`);
  ws.binaryType = 'arraybuffer';
  ws.onmessage = (e) => handleEvent(JSON.parse(e.data));
  ws.onclose = () => {
    if (running) {
      setStatus('reconnecting', 'reconnecting…');
      setTimeout(connect, 1500); // resume same session
    }
  };
}

function handleEvent(ev) {
  switch (ev.type) {
    case 'session_started':
      sessionId = ev.session_id;
      setStatus('listening', 'listening');
      break;
    case 'speech_start':
      setStatus('speech', 'hearing speech…');
      break;
    case 'transcribing':
      setStatus('transcribing', 'transcribing…');
      break;
    case 'utterance':
      appendUtterance($('transcript'), ev);
      setStatus('listening', 'listening');
      break;
  }
}

// ---------- rendering ----------
function speakerColor(n) {
  return `hsl(${(n * 67) % 360} 60% 45%)`;
}

function appendUtterance(container, u) {
  const div = document.createElement('div');
  div.className = 'utterance';
  const showOrig = $('show-original').checked && u.original_text;
  div.innerHTML = `
    <span class="speaker" style="color:${speakerColor(u.speaker ?? u.speaker_num)}">
      Person ${u.speaker ?? u.speaker_num}</span>
    <span class="lang">${u.lang}</span>
    ${showOrig ? `<div class="original">${escapeHtml(u.original_text)}</div>` : ''}
    <div class="english">${u.original_text ? '↳ ' : ''}${escapeHtml(u.english_text)}</div>`;
  container.appendChild(div);
  container.scrollTop = container.scrollHeight;
}

function escapeHtml(s) {
  const d = document.createElement('div');
  d.textContent = s ?? '';
  return d.innerHTML;
}

function setStatus(cls, text) {
  const el = $('status');
  el.className = `status ${cls}`;
  el.textContent = text;
}

function banner(msg) {
  const b = $('banner');
  b.textContent = msg;
  b.classList.remove('hidden');
}

// ---------- controls ----------
$('btn-mic').onclick = async () => {
  if (!running) {
    if (!(await startMic())) return;
    running = true;
    sessionId = null; // new session on manual start
    connect();
    $('btn-mic').textContent = '⏹ Stop';
  } else {
    running = false;
    stopMic();
    if (ws) ws.close();
    setStatus('idle', 'idle');
    $('btn-mic').textContent = '🎤 Start listening';
  }
};

// ---------- history ----------
async function loadSessions() {
  const res = await fetch('/api/sessions');
  const sessions = await res.json();
  const list = $('session-list');
  list.innerHTML = sessions.length ? '' : '<p>No conversations yet.</p>';
  for (const s of sessions) {
    const item = document.createElement('button');
    item.className = 'session-item';
    item.textContent = `#${s.id} ${s.started_at} — ${s.title || '(untitled)'} (${s.utterance_count})`;
    item.onclick = async () => {
      const d = await (await fetch(`/api/sessions/${s.id}`)).json();
      const box = $('session-detail');
      box.innerHTML = `<h3>${escapeHtml(d.title || 'Untitled')}</h3>`;
      for (const u of d.utterances) appendUtterance(box, u);
    };
    list.appendChild(item);
  }
}

$('tab-live').onclick = () => switchTab(true);
$('tab-history').onclick = () => { switchTab(false); loadSessions(); };
function switchTab(live) {
  $('view-live').classList.toggle('hidden', !live);
  $('view-history').classList.toggle('hidden', live);
  $('tab-live').classList.toggle('active', live);
  $('tab-history').classList.toggle('active', !live);
}
```

- [ ] **Step 4: `static/style.css`**

```css
* { box-sizing: border-box; margin: 0; }
body { font-family: system-ui, sans-serif; background: #101418; color: #e8eaed; min-height: 100vh; }
header { display: flex; align-items: center; justify-content: space-between; padding: 0.8rem 1.2rem; border-bottom: 1px solid #2a2f36; }
h1 { font-size: 1.1rem; font-weight: 600; }
.tab { background: none; border: 1px solid #2a2f36; color: #aaa; padding: 0.4rem 1rem; border-radius: 999px; margin-left: 0.5rem; cursor: pointer; }
.tab.active { color: #fff; border-color: #4f8cff; }
main { max-width: 720px; margin: 0 auto; padding: 1rem; }
.controls { display: flex; align-items: center; gap: 1rem; flex-wrap: wrap; }
.mic { font-size: 1rem; padding: 0.6rem 1.4rem; border-radius: 999px; border: none; background: #4f8cff; color: #fff; cursor: pointer; }
.status { padding: 0.25rem 0.8rem; border-radius: 999px; font-size: 0.85rem; background: #2a2f36; }
.status.speech { background: #1d4ed8; }
.status.transcribing { background: #b45309; }
.status.listening { background: #166534; }
.status.reconnecting { background: #991b1b; }
.toggle { font-size: 0.85rem; color: #aaa; }
.banner { margin-top: 1rem; padding: 0.8rem 1rem; background: #7f1d1d; border-radius: 8px; }
.hidden { display: none; }
.transcript { margin-top: 1rem; display: flex; flex-direction: column; gap: 0.9rem; max-height: 70vh; overflow-y: auto; }
.utterance { background: #1a1f26; border-radius: 10px; padding: 0.7rem 0.9rem; }
.speaker { font-weight: 700; margin-right: 0.5rem; }
.lang { font-size: 0.7rem; color: #888; text-transform: uppercase; }
.original { margin-top: 0.3rem; font-size: 1.05rem; }
.english { margin-top: 0.2rem; color: #9fc0ff; }
.session-item { display: block; width: 100%; text-align: left; background: #1a1f26; color: #e8eaed; border: none; border-radius: 8px; padding: 0.7rem 0.9rem; margin-bottom: 0.5rem; cursor: pointer; }
.session-item:hover { background: #232a33; }
```

- [ ] **Step 5: Manual verification**

```bash
./scripts/fetch-models.sh tiny   # if models absent; use tiny for the smoke test
LT_WHISPER_MODEL=tiny cargo run  # background
```

Open `http://localhost:8080` in a browser. Verify: Start listening → speak → status cycles idle→listening→hearing speech→transcribing→line appears with `Person 1`, original + English; History tab lists the session. Stop the server after.

If a browser can't be driven in this environment, verify instead with `curl -s localhost:8080/ | grep -q 'Live Transcriber' && curl -s localhost:8080/app.js | head -1` and flag manual browser testing for the user.

- [ ] **Step 6: Commit**

```bash
git add -A && git commit -m "feat: live + history web UI with mic capture worklet"
```

---

### Task 11: Docker packaging and README

**Files:**
- Create: `Dockerfile`, `docker-compose.yml`, `.dockerignore`, `scripts/entrypoint.sh`, `README.md`

**Interfaces:**
- Consumes: the finished binary, `scripts/fetch-models.sh`, `config.toml`, `static/`.

- [ ] **Step 1: `.dockerignore`**

```
target
models
data
.git
docs
tests/fixtures
```

- [ ] **Step 2: `Dockerfile`**

```dockerfile
FROM rust:1.83-bookworm AS build
RUN apt-get update && apt-get install -y --no-install-recommends cmake clang git && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY . .
RUN cargo build --release
# collect the binary and any dynamic libs ort/sherpa produced next to it
RUN mkdir /out \
 && cp target/release/live-transcriber /out/ \
 && (cp target/release/*.so* /out/ 2>/dev/null || true) \
 && (cp target/release/deps/*.so* /out/ 2>/dev/null || true)

FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl bash && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY --from=build /out/ /app/bin/
COPY static ./static
COPY config.toml ./config.toml
COPY scripts/fetch-models.sh scripts/entrypoint.sh ./scripts/
RUN chmod +x scripts/*.sh
ENV LD_LIBRARY_PATH=/app/bin
EXPOSE 8080
ENTRYPOINT ["/app/scripts/entrypoint.sh"]
```

- [ ] **Step 3: `scripts/entrypoint.sh`**

```bash
#!/usr/bin/env bash
set -euo pipefail
MODEL="${LT_WHISPER_MODEL:-medium}"
./scripts/fetch-models.sh "$MODEL" models
exec /app/bin/live-transcriber
```

- [ ] **Step 4: `docker-compose.yml`**

```yaml
services:
  transcriber:
    build: .
    ports:
      - "8080:8080"
    volumes:
      - ./models:/app/models
      - ./data:/app/data
    environment:
      - LT_WHISPER_MODEL=${LT_WHISPER_MODEL:-medium}
```

- [ ] **Step 5: Build and smoke-test the container**

```bash
docker compose build          # background — 20-40 min cold (compiles whisper.cpp, sherpa-onnx)
LT_WHISPER_MODEL=tiny docker compose up -d
sleep 20 && curl -s localhost:8080/healthz   # allow model download+load time; poll until "ok"
docker compose logs --tail 20
docker compose down
```

Expected: `ok` from healthz; logs show "listening on http://0.0.0.0:8080".

- [ ] **Step 6: `README.md`**

Write a README covering exactly:
- What it is (one paragraph: VAD-gated live transcription of Hindi/Urdu/Arabic → English with speaker labels, offline, SQLite history).
- Quick start: `docker compose up` (first run downloads the whisper `medium` model ≈1.5 GB into `./models`), open `http://localhost:8080`.
- Choosing a model: `LT_WHISPER_MODEL=small docker compose up` — table of tiny/base/small/medium/large-v3 with rough RAM and quality notes.
- Phone/other-device usage: mic requires HTTPS off-localhost; set `tls = true` in `config.toml`, restart, open `https://<laptop-ip>:8080`, accept the self-signed cert warning.
- Native (Metal) mode: `brew install cmake`, `./scripts/fetch-models.sh medium`, `cargo run --release --features metal`.
- IoT deployment: `docker buildx build --platform linux/arm64 .` for Pi-class boards; recommend `LT_WHISPER_MODEL=tiny` + `english_only = true`.
- Config reference: every `config.toml` key with its default and effect.
- Architecture sketch: the pipeline diagram from the spec.

- [ ] **Step 7: Final full-test pass and commit**

```bash
cargo test -- --test-threads=1   # background; all green
git add -A && git commit -m "feat: docker packaging, compose file, and README"
```

---

## Post-plan verification (after all tasks)

1. `cargo test -- --test-threads=1` — everything green.
2. `LT_WHISPER_MODEL=tiny cargo run` + browser smoke test (speak in Hindi/Urdu/Arabic if possible; verify original script + English + Person N).
3. `docker compose up` end-to-end once with `tiny`.
4. Update the spec's status line if any interface drifted during implementation.
