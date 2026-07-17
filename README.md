# Tarjuman — ترجمان

**Self-hosted, fully offline live speech transcription, translation, and speaker diarization — in a single Rust binary.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/Rust-2021-orange.svg)](https://www.rust-lang.org/)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)

*Tarjumān (ترجمان) — "interpreter" in Urdu, Arabic, and Persian.*

Tarjuman listens through any device's microphone in the browser, detects when humans are actually speaking, transcribes speech in its original language (built with Hindi, Urdu, and Arabic in mind — any Whisper-supported language works), translates every utterance to English live, and labels who said what (`Person 1`, `Person 2`, …). Conversations are stored locally in SQLite and browsable from a built-in history view.

Everything runs on your own hardware. No cloud APIs, no telemetry, no audio ever leaving your machine — and no audio stored at all: only text is persisted.

---

## Table of contents

- [Features](#features)
- [How it works](#how-it-works)
- [Quick start](#quick-start)
- [Choosing a model](#choosing-a-model)
- [Using it from a phone or another device](#using-it-from-a-phone-or-another-device)
- [Native (Metal) mode on macOS](#native-metal-mode-on-macos)
- [IoT deployment (Raspberry Pi / ARM boards)](#iot-deployment-raspberry-pi--arm-boards)
- [Configuration reference](#configuration-reference)
- [HTTP & WebSocket API](#http--websocket-api)
- [Project layout](#project-layout)
- [Development](#development)
- [Roadmap](#roadmap)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Features

- **Voice-activity gated** — Silero VAD scores every 32 ms frame; silence, music, and background noise never enter the pipeline and are never stored.
- **Live transcription + translation** — each utterance is decoded twice by whisper.cpp: once for the original-script text with automatic language detection, once for the English translation. An `english_only` mode halves the work for constrained devices.
- **Speaker diarization** — a NeMo TitaNet speaker-embedding model plus online cosine clustering assigns stable `Person N` labels within a session, with a short-utterance guard that prevents "yeah"/"okay" blips from spawning phantom speakers.
- **Conversation history** — sessions and utterances (speaker, language, original text, English text, timestamps) persist to a single SQLite file; a history tab replays any past conversation.
- **Zero-dependency frontend** — one HTML file, vanilla JS, an `AudioWorklet` for capture, and a WebSocket. No build step, no framework, renders fine on a phone.
- **One small binary** — Rust + `axum` serves the UI, REST API, and WebSocket; whisper.cpp, Silero VAD, and sherpa-onnx run in-process. Deploys via Docker Compose or bare `cargo run`.
- **ARM/IoT ready** — the whole stack cross-compiles for `linux/arm64`; the long-term target is always-on Pi-class listening devices.

## How it works

```
Browser (any device on LAN)                    Tarjuman server (single Rust binary)
┌──────────────────────────┐                  ┌─────────────────────────────────────┐
│ mic → AudioWorklet       │  16kHz PCM       │ VAD (Silero) → utterance segmenter  │
│ downsample to 16 kHz ────┼──WebSocket──────▶│  → Whisper transcribe (orig text)   │
│                          │                  │  → Whisper translate (English)      │
│ live transcript UI ◀─────┼──JSON events─────│  → speaker embedding + clustering   │
│ session history view     │                  │  → SQLite (sessions, utterances)    │
└──────────────────────────┘                  └─────────────────────────────────────┘
```

The browser captures microphone audio in an `AudioWorklet`, downsamples it to 16 kHz mono 16-bit PCM, and streams it in ~128 ms binary chunks over a WebSocket. Server-side, Silero VAD gates the stream; a segmenter state machine groups speech into utterances (with 300 ms pre-roll so first syllables aren't clipped); whisper.cpp transcribes and translates each utterance; a speaker-embedding model + online clustering assigns the `Person N` label; the result is written to SQLite and pushed back to the browser as a JSON event — typically 1–3 seconds after the speaker pauses.

## Quick start

Requires [Docker](https://docs.docker.com/get-docker/) with Compose.

```bash
git clone <your-fork-or-clone-url>
cd tarjuman
docker compose up
```

The first run downloads the Whisper `medium` model (~1.5 GB) and a small speaker-embedding model (~40 MB) into `./models`; subsequent starts reuse them. When the log shows `listening on http://0.0.0.0:8080`, open:

```
http://localhost:8080
```

Click **Start listening**, allow microphone access, and speak. The live view shows each utterance as:

```
Person 1  HI   आप कैसे हैं?
               ↳ How are you?
```

Conversations land in `./data/transcriber.db` and are browsable from the **History** tab.

## Choosing a model

Override the default model with the `LT_WHISPER_MODEL` environment variable:

```bash
LT_WHISPER_MODEL=small docker compose up
```

| Model | Approx. RAM | Quality / speed notes |
|---|---|---|
| `tiny` | ~1 GB | Fastest, lowest accuracy. Smoke tests and very constrained hardware only — noticeably poor for multilingual use. |
| `base` | ~1 GB | Slightly better than `tiny`, still very fast. |
| `small` | ~2 GB | Good accuracy/speed balance; recommended minimum for real conversational use. |
| `medium` | ~5 GB | **Default.** Strong accuracy across Hindi/Urdu/Arabic; a few seconds per utterance on CPU. |
| `large-v3` | ~10 GB | Best available accuracy; slow on CPU-only Docker — pair it with the native Metal build below. |

Model files live in `./models` and are shared across restarts and model switches (each size gets its own `ggml-<size>.bin`, so nothing re-downloads).

## Using it from a phone or another device

Browsers only allow microphone access on a secure context — `localhost` is exempt, but LAN IPs are not. To use the app from another device on your network:

1. Set `tls = true` in `config.toml`.
2. Recreate the container: `docker compose up -d --force-recreate` (the host `config.toml` is volume-mounted). On first TLS start a self-signed certificate is generated into `./data/cert.pem` / `./data/key.pem`.
3. On the other device, open `https://<host-ip>:8080`.
4. Accept the browser's self-signed-certificate warning — expected for a local cert.

## Native (Metal) mode on macOS

Docker on macOS cannot reach the Apple GPU. Running natively lets whisper.cpp use Metal, which makes even `large-v3` responsive:

```bash
brew install cmake
./scripts/fetch-models.sh large-v3
LT_WHISPER_MODEL=large-v3 cargo run --release --features metal
```

The server behaves identically, reading `config.toml`, `./models`, and `./data` from the working directory.

## IoT deployment (Raspberry Pi / ARM boards)

The binary is self-contained and the full native stack (whisper.cpp, sherpa-onnx, ONNX Runtime) builds for ARM:

```bash
docker buildx build --platform linux/arm64 .
```

For constrained hardware, use the smallest practical model and skip the original-script pass:

```bash
LT_WHISPER_MODEL=tiny LT_ENGLISH_ONLY=true docker compose up
```

(`english_only = true` in `config.toml` works too — it skips the original-language transcription pass and produces only the English translation, roughly halving per-utterance CPU work.)

## Configuration reference

All keys in `config.toml` are optional; defaults below apply to omitted keys. `port`, `whisper_model`, and `english_only` can also be set via the `LT_PORT`, `LT_WHISPER_MODEL`, and `LT_ENGLISH_ONLY` environment variables (env wins over file).

| Key | Default | Effect |
|---|---|---|
| `port` | `8080` | TCP port the server listens on. |
| `models_dir` | `"models"` | Directory containing the Whisper and speaker-embedding models. |
| `data_dir` | `"data"` | Directory for the SQLite database and (with TLS) the generated cert/key. |
| `whisper_model` | `"medium"` | Whisper model size (`tiny`, `base`, `small`, `medium`, `large-v3`). |
| `english_only` | `false` | Skip the original-language pass; produce only English. Saves CPU on constrained devices. |
| `vad_threshold` | `0.5` | Silero VAD speech-probability threshold (0–1). Higher = stricter about what counts as speech. |
| `silence_end_ms` | `700` | Continuous silence (ms) that ends the current utterance. |
| `pre_roll_ms` | `300` | Audio (ms) buffered before the VAD trigger and prepended, so word onsets aren't clipped. |
| `min_speech_ms` | `250` | Minimum speech duration for a segment to count as an utterance (filters blips). |
| `max_utterance_s` | `25` | Hard cap on utterance length; force-flushed immediately at the cap. |
| `speaker_similarity_threshold` | `0.65` | Cosine similarity vs existing speaker centroids. At/above: assigned + centroid updated. Below: new `Person N`. |
| `min_new_speaker_ms` | `2000` | Utterances shorter than this can never create a *new* speaker — short clips embed unreliably, so they're attributed to the nearest existing speaker instead. |
| `whisper_threads` | `8` | CPU threads for Whisper inference. |
| `tls` | `false` | Serve HTTPS with a self-signed cert (generated into `data_dir` on first start). Required for mic access from non-localhost origins. |

## HTTP & WebSocket API

The frontend is just a client of this API — anything else (a CLI, an OBS overlay, home automation) can consume it too.

### REST

| Endpoint | Response |
|---|---|
| `GET /healthz` | `ok` |
| `GET /api/sessions` | `[{ "id", "started_at", "ended_at", "title", "utterance_count" }, …]` (newest first) |
| `GET /api/sessions/{id}` | `{ "id", "started_at", "title", "utterances": [{ "speaker_num", "lang", "original_text", "english_text", "start_ms", "duration_ms" }, …] }` — `404` if unknown |

### WebSocket — `GET /ws`

- **Client → server:** binary frames of 16 kHz mono 16-bit little-endian PCM, any chunk size.
- **Server → client:** JSON text events:

```jsonc
{ "type": "session_started", "session_id": 42 }
{ "type": "speech_start" }                        // VAD heard a human
{ "type": "transcribing" }                        // utterance closed, decoding
{ "type": "utterance", "speaker": 1, "lang": "hi",
  "original_text": "आप कैसे हैं?", "english_text": "How are you?",
  "start_ms": 1234, "duration_ms": 2100 }
```

- **Resuming:** reconnect with `GET /ws?session=<id>` to append to an existing session; persisted speaker centroids are reloaded so `Person N` numbering stays stable.

## Project layout

```
src/
├── main.rs        # startup wiring: config, model checks, DB, HTTP/TLS serve
├── lib.rs         # module exports
├── config.rs      # config.toml + env overrides
├── server.rs      # axum router: REST, WebSocket, static files
├── pipeline.rs    # per-connection orchestrator thread (audio in → events out)
├── segmenter.rs   # pure utterance-segmentation state machine
├── vad.rs         # Silero VAD wrapper (model bundled in-crate)
├── whisper.rs     # whisper.cpp wrapper: transcribe + translate
├── speakers.rs    # speaker embeddings + online cosine clustering
└── db.rs          # SQLite: sessions, utterances, speaker centroids
static/            # index.html, app.js, worklet.js, style.css — no build step
scripts/           # fetch-models.sh, entrypoint.sh
tests/             # integration tests + audio fixture
docs/              # design spec and implementation plan
```

## Development

Prerequisites: Rust (stable, 1.85+), `cmake` (for sherpa-onnx), and ~10 min for the first build (it compiles whisper.cpp and sherpa-onnx from source).

```bash
./scripts/fetch-models.sh tiny        # test models (~120 MB total)
cargo test -- --test-threads=1        # full suite (single-threaded: heavy native inference)
LT_WHISPER_MODEL=tiny cargo run       # dev server on :8080
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for the workflow, test conventions, and architecture notes.

## Roadmap

- [ ] Session-resume timestamp continuity (resumed sessions currently restart `start_ms` at 0)
- [ ] WebSocket ping/pong liveness so vanished clients release their pipeline promptly
- [ ] Error events surfaced to the UI when the pipeline fails to start
- [ ] Bounded audio queue with drop-oldest under sustained decode backlog
- [ ] Hindi/Urdu/Arabic and two-speaker test fixtures for end-to-end multilingual coverage
- [ ] "Show original" toggle applied retroactively to already-rendered lines
- [ ] Word-level streaming partials (utterance-level today)
- [ ] Cross-session speaker identity ("this is Ajay") — currently per-session labels by design

## Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| Poor transcription/translation accuracy | You're on a small model. Use `medium` (default) or `large-v3` with the native Metal build. |
| "Microphone access denied" banner | Browser blocked mic. On non-localhost origins you must use HTTPS — see the phone section. |
| Long delay before text appears | CPU decode of two passes per utterance. Try a smaller model, `english_only = true`, or native Metal mode. |
| Same voice becomes `Person 2` | Raise `min_new_speaker_ms` or lower `speaker_similarity_threshold` slightly (e.g. 0.6). |
| Two people merged into one speaker | Raise `speaker_similarity_threshold` (e.g. 0.7). |
| Container starts then exits | Check `docker compose logs` — usually a failed model download; re-run, the fetch resumes cleanly. |

## Contributing

Contributions are welcome — bug reports, test fixtures in more languages, docs, and code. Please read [CONTRIBUTING.md](CONTRIBUTING.md) first; it covers setup, test expectations, and the review conventions this repo uses.

## License

[MIT](LICENSE)

## Acknowledgements

This project stands on excellent open-source work:

- [whisper.cpp](https://github.com/ggml-org/whisper.cpp) and [whisper-rs](https://github.com/tazz4843/whisper-rs) — speech recognition & translation (OpenAI Whisper models)
- [Silero VAD](https://github.com/snakers4/silero-vad) via [voice_activity_detector](https://github.com/nkeenan38/voice_activity_detector) — voice activity detection
- [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) and [sherpa-rs](https://github.com/thewh1teagle/sherpa-rs) — speaker embeddings (NVIDIA NeMo TitaNet model)
- [axum](https://github.com/tokio-rs/axum), [tokio](https://tokio.rs), [rusqlite](https://github.com/rusqlite/rusqlite)
