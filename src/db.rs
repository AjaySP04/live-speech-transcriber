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
        assert_eq!(spk[1].1, vec![0.9, 0.8, 0.7]);
    }

    #[test]
    fn file_backed_open_uses_wal() {
        use std::fs;
        let temp_dir = std::env::temp_dir().join("lt-db-test");
        let _ = fs::create_dir_all(&temp_dir);
        let db_path = temp_dir.join("test-wal.db");

        // Clean up from previous runs
        let _ = fs::remove_file(&db_path);
        let _ = fs::remove_file(format!("{}-wal", db_path.display()));
        let _ = fs::remove_file(format!("{}-shm", db_path.display()));

        let db = Db::open(db_path.to_str().unwrap()).unwrap();

        // Verify WAL mode is enabled
        let mode: String = db.conn.query_row(
            "PRAGMA journal_mode",
            [],
            |r| r.get(0),
        ).unwrap();
        assert_eq!(mode.to_lowercase(), "wal");

        // Clean up after test
        let _ = fs::remove_file(&db_path);
        let _ = fs::remove_file(format!("{}-wal", db_path.display()));
        let _ = fs::remove_file(format!("{}-shm", db_path.display()));
    }
}
