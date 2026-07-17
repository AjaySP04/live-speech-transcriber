# Live Transcriber — Design Spec

**Date:** 2026-07-16
**Status:** Approved by user

## Purpose

A self-hosted web app that listens through a device microphone, detects when humans are speaking (and only then records), transcribes speech in Hindi/Urdu/Arabic (or any Whisper-supported language), translates it to English live, and attributes each utterance to a speaker (`Person 1`, `Person 2`, …). Conversations are persisted to a database. Long-term target: the same software runs on a small ARM IoT device (e.g., Raspberry Pi 5).

## Decisions made with user

| Decision | Choice |
|---|---|
| Processing location | Local backend server (no cloud APIs, fully open source/offline) |
| Backend language | Rust |
| Transcript display | Original script + English translation per line |
| Persistence | SQLite conversation database |
| Runtime | Docker (backend + UI in one container via docker compose) |
| Dev machine | Apple M4 Pro, 24 GB RAM (CPU-only inside Docker — macOS Docker has no Metal access; accepted) |

## Architecture

A single Rust binary (`axum`) serves the static web UI, a REST API for history, and a WebSocket endpoint for live audio. The browser captures mic audio, downsamples to 16 kHz mono 16-bit PCM in an AudioWorklet, and streams ~128 ms binary chunks over the WebSocket. The server runs the speech pipeline and streams transcript events (JSON) back on the same socket.

```
Browser (any device on LAN)                    Docker container (Rust binary)
┌──────────────────────────┐                  ┌─────────────────────────────────────┐
│ mic → AudioWorklet       │  16kHz PCM       │ VAD (Silero) → utterance segmenter  │
│ downsample to 16 kHz ────┼──WebSocket──────▶│  → Whisper transcribe (orig text)   │
│                          │                  │  → Whisper translate (English)      │
│ live transcript UI ◀─────┼──JSON events─────│  → speaker embedding + clustering   │
│ session history view     │                  │  → SQLite (sessions, utterances)    │
└──────────────────────────┘                  └─────────────────────────────────────┘
```

## Audio pipeline

1. **VAD gate:** Silero VAD (small ONNX model via the `ort` crate) scores every 32 ms frame. Audio is discarded until speech probability crosses a threshold — silence and non-speech noise never enter the pipeline and are never stored.
2. **Utterance segmenter:** a state machine buffers speech with 300 ms pre-roll; ~700 ms of continuous silence ends the utterance; a 25 s hard cap force-flushes at the nearest silence dip. Thresholds configurable.
3. **Transcription (whisper-rs → whisper.cpp):** per utterance, pass 1 `transcribe` with auto language detection → original-script text + language code; pass 2 `translate` → English text. Default model `medium`; configurable (`tiny`…`large-v3`). An `english_only` config flag skips pass 1 for constrained devices.
4. **Diarization:** an ECAPA-TDNN speaker-embedding ONNX model (~20 MB, via `ort`) embeds each utterance. Online clustering: cosine similarity against per-session speaker centroids; similarity ≥ threshold (default 0.65, configurable) assigns the existing speaker and updates the centroid (running mean); otherwise a new `Person N` is created. Labels are stable within a session.
5. **Output:** each completed utterance is inserted into SQLite and pushed to the client as a JSON event: `{speaker_num, lang, original_text, english_text, start_ms, duration_ms}`.

**Latency expectation:** segment-level live — text appears ~1–3 s after the speaker pauses (Whisper is not word-streaming). Accepted trade-off.

## Data model (SQLite, `rusqlite`, WAL mode, file `data/transcriber.db`)

- `sessions(id INTEGER PK, started_at, ended_at, title)` — title auto-generated from date + first utterance words.
- `utterances(id INTEGER PK, session_id FK, speaker_num, lang, original_text, english_text, start_ms, duration_ms)`
- `speakers(session_id FK, speaker_num, centroid BLOB, PRIMARY KEY(session_id, speaker_num))` — persisted centroids so a reconnect within the same session keeps speaker numbering.

REST API: `GET /api/sessions`, `GET /api/sessions/:id` (session with utterances). WebSocket: `WS /ws` (binary = audio in, text = JSON events out).

## Web UI

No framework: one HTML file + vanilla JS + small CSS, served by axum. Two screens:

- **Live:** Start/Stop mic button; status indicator (idle / listening / hearing speech / transcribing); transcript feed with color-coded `Person N:` lines, original text with `↳ English` beneath, and a toggle to hide originals.
- **History:** list of past sessions → full stored transcript view.

Must render acceptably on a phone browser.

## Error handling

- Mic permission denied / absent → clear banner with instructions.
- Mic on non-localhost origins: browsers require a secure context. README documents phone-testing options (self-signed cert config flag).
- WebSocket drop → client auto-reconnect with backoff; server holds the session open 60 s and resumes it.
- Transcription backlog → utterances queue in order; UI shows placeholder rows.
- Missing models at startup → downloaded automatically to the models volume with progress logs; `scripts/fetch-models.sh` supports offline prep.

## Packaging

- Multi-stage `Dockerfile` (Rust build stage → slim Debian runtime). `docker-compose.yml` mounts `./models` and `./data`, exposes port 8080. CPU-only inference inside Docker.
- Native escape hatch: `cargo run --features metal` outside Docker uses Apple GPU for larger models.
- **IoT path:** `docker buildx` arm64 image for Raspberry Pi-class boards; config flips to `small`/`tiny` model + `english_only` mode. The binary is self-contained, so bare-metal deployment (no Docker) also works.

## Configuration (`config.toml`, env-overridable)

`model` (whisper size), `english_only` (bool), `vad_threshold`, `silence_end_ms`, `max_utterance_s`, `speaker_similarity_threshold`, `port`, `tls` (off | self-signed).

## Testing

- **Unit:** segmenter state machine on synthetic VAD probability sequences; clustering on synthetic embeddings; DB layer round-trips.
- **Integration:** WAV fixtures (short Hindi/Urdu/Arabic clips, incl. one two-speaker clip) through the full pipeline; assert detected language, non-empty original + English text, expected speaker count.
- **Manual:** browser smoke test on laptop and phone.

## Out of scope (YAGNI)

Word-by-word streaming partials, cross-session speaker identity ("this is Ajay"), translation to languages other than English, user accounts/auth, audio recording storage (only text is persisted).
