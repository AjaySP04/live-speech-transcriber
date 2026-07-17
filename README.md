# live-transcriber

## What it is

`live-transcriber` is a self-hosted, offline live transcription and translation server. It listens through a device microphone, uses voice activity detection (VAD) to gate on speech only (silence and background noise are never transcribed or stored), transcribes Hindi, Urdu, Arabic (or any Whisper-supported language) in the original script, translates each utterance to English, and attributes it to a speaker (`Person 1`, `Person 2`, …) using on-the-fly speaker embeddings and clustering. Everything runs locally — no cloud APIs — and conversation history is persisted to a SQLite database so past sessions can be reviewed later.

## Quick start

```bash
docker compose up
```

The first run downloads the Whisper `medium` model (~1.5 GB) and a small speaker-embedding model into `./models` (this can take a few minutes depending on connection speed; subsequent starts skip the download since the files are already on the volume). Once the server logs `listening on http://0.0.0.0:8080`, open:

```
http://localhost:8080
```

Click Start, allow microphone access, and speak. Conversations are stored under `./data/transcriber.db` and are browsable from the History tab.

## Choosing a model

Override the default model with the `LT_WHISPER_MODEL` environment variable:

```bash
LT_WHISPER_MODEL=small docker compose up
```

| Model | Approx. RAM | Quality / speed notes |
|---|---|---|
| `tiny` | ~1 GB | Fastest, lowest accuracy. Good for quick smoke tests or very constrained hardware (e.g. Raspberry Pi). |
| `base` | ~1 GB | Slightly better accuracy than `tiny`, still very fast. |
| `small` | ~2 GB | Good accuracy/speed balance for most laptops; recommended minimum for real conversational use. |
| `medium` | ~5 GB | Default. Strong accuracy across Hindi/Urdu/Arabic, noticeably slower per utterance on CPU. |
| `large-v3` | ~10 GB | Best accuracy, significantly slower on CPU-only Docker; best paired with the native Metal build below. |

Model files live in `./models` and are shared across restarts and model-size changes (each size gets its own `ggml-<size>.bin` file, so switching models doesn't re-download ones you've already fetched).

## Phone / other-device usage

Browsers only allow microphone access on a "secure context" — `localhost` is exempt, but any other device on your LAN accessing the server by IP is not. To use the app from a phone or another machine on your network:

1. Set `tls = true` in `config.toml`.
2. Recreate the container so it picks up the change: `docker compose up -d --force-recreate` (the host `config.toml` is volume-mounted into the container). On first TLS start it generates a self-signed certificate into `./data/cert.pem` / `./data/key.pem`.
3. From the other device, open `https://<laptop-ip>:8080`.
4. Accept the browser's self-signed certificate warning (it will look untrusted — that's expected for a local self-signed cert).

## Native (Metal) mode

Running natively on macOS lets Whisper use the Apple GPU (Metal) instead of CPU-only inference inside Docker, which is much faster for the larger models:

```bash
brew install cmake
./scripts/fetch-models.sh medium
cargo run --release --features metal
```

The app then serves on `http://localhost:8080` as usual, reading `config.toml` and `./models` / `./data` from the current directory.

## IoT deployment (Raspberry Pi / ARM boards)

The binary is self-contained and the whole stack (whisper.cpp, sherpa-onnx) cross-compiles for ARM, so this same image runs on Pi-class boards:

```bash
docker buildx build --platform linux/arm64 .
```

For constrained IoT hardware, use the smallest practical model and skip the original-language transcription pass to save CPU:

```bash
LT_WHISPER_MODEL=tiny docker compose up
```

and set `english_only = true` in `config.toml`, then `docker compose up -d --force-recreate` (the file is volume-mounted into the container). This skips the original-script transcription pass and only produces the English translation, roughly halving Whisper's per-utterance work. `LT_ENGLISH_ONLY=true` in the environment works too.

## Configuration reference (`config.toml`)

All keys are optional; the defaults below are used for any key you omit. `port`, `whisper_model` (via `LT_WHISPER_MODEL`), and `english_only` (via `LT_ENGLISH_ONLY`) can also be overridden with environment variables.

| Key | Default | Effect |
|---|---|---|
| `port` | `8080` | TCP port the server listens on. |
| `models_dir` | `"models"` | Directory containing the Whisper and speaker-embedding model files. |
| `data_dir` | `"data"` | Directory for the SQLite database and (if TLS is enabled) the generated cert/key. |
| `whisper_model` | `"medium"` | Whisper model size to load (`tiny`, `base`, `small`, `medium`, `large-v3`). |
| `english_only` | `false` | If `true`, skips the original-language transcription pass and only produces the English translation — saves CPU on constrained devices at the cost of not showing the original script. |
| `vad_threshold` | `0.5` | Silero VAD speech-probability threshold (0–1). Higher = stricter about what counts as speech; fewer false triggers on noise but may clip quiet speech. |
| `silence_end_ms` | `700` | Milliseconds of continuous silence after speech that ends the current utterance. |
| `pre_roll_ms` | `300` | Milliseconds of audio buffered before the VAD trigger and prepended to the utterance, so the start of a word isn't clipped. |
| `min_speech_ms` | `250` | Minimum speech duration for a detected segment to be treated as a real utterance (filters out very short blips/noise). |
| `max_utterance_s` | `25` | Hard cap on utterance length in seconds; a long continuous utterance is force-flushed immediately at the cap rather than growing unbounded. |
| `speaker_similarity_threshold` | `0.65` | Cosine-similarity threshold against existing per-session speaker centroids. At or above this, the utterance is assigned to that speaker (and its centroid is updated); below it, a new `Person N` is created. |
| `min_new_speaker_ms` | `2000` | Minimum utterance duration required to spawn a brand-new speaker. Short utterances (e.g. "yeah", "okay") produce embeddings too unreliable to confidently identify a new speaker, so anything shorter than this is instead attributed to the nearest existing speaker even if the raw similarity is below `speaker_similarity_threshold`. |
| `whisper_threads` | `4` | Number of CPU threads Whisper uses for inference. |
| `tls` | `false` | If `true`, serves over HTTPS with a self-signed certificate (generated on first start into `data_dir`), required for microphone access from non-localhost origins. |

## Architecture

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

A single Rust binary (built on `axum`) serves the static web UI, a REST API for session history, and a WebSocket endpoint for live audio. The browser captures microphone audio in an `AudioWorklet`, downsamples it to 16 kHz mono 16-bit PCM, and streams it in small chunks over the WebSocket. The server runs it through Silero VAD to gate speech, segments it into utterances, transcribes and translates each with whisper.cpp, embeds and clusters speakers with a sherpa-onnx speaker-ID model, persists the result to SQLite, and pushes a JSON transcript event back over the same socket.
