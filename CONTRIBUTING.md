# Contributing to Tarjuman

Thanks for your interest in contributing! Tarjuman is a young project and welcomes bug reports, feature discussion, documentation improvements, test fixtures in more languages, and code.

## Getting set up

**Prerequisites**

- Rust stable **1.85+** (`rustup` recommended)
- Node **20+** (frontend build)
- `cmake` — required to build sherpa-onnx (`brew install cmake` on macOS, `apt install cmake` on Debian/Ubuntu)
- A C/C++ toolchain (clang preferred; see the Dockerfile notes about gcc-12 on ARM)
- Docker (only if you're touching the container packaging)

**First build**

```bash
git clone <repo-url>
cd tarjuman
./scripts/fetch-models.sh tiny   # downloads the small test models (~120 MB)
cargo build
(cd web && npm install && npm run build)   # builds the React UI into static/
```

The first Rust build compiles whisper.cpp and sherpa-onnx from source and downloads ONNX Runtime — expect 10–20 minutes. Subsequent builds are incremental and fast.

**Run the dev servers**

```bash
LT_WHISPER_MODEL=tiny cargo run   # backend + built UI on http://localhost:8080
```

For frontend work with hot reload, run the Vite dev server alongside (it proxies `/api` and `/ws` to the backend — override the target with `VITE_BACKEND=http://localhost:<port>`):

```bash
cd web && npm run dev             # http://localhost:5173
```

On macOS, add `--release --features metal` to `cargo run` for GPU-accelerated inference.

## Running tests

```bash
cargo test -- --test-threads=1   # backend (Rust)
cd web && npm test               # frontend (vitest)
```

- **Fetch the test models first** (`./scripts/fetch-models.sh tiny`) — the whisper, speaker, and pipeline tests panic with a `run scripts/fetch-models.sh tiny` message when `models/` is missing. Run from the repo root; model paths are relative.
- `--test-threads=1` is recommended: several tests run real whisper/ONNX inference and are memory- and CPU-heavy.
- The audio fixture `tests/fixtures/jfk.wav` (16 kHz mono 16-bit PCM) is committed; backend tests decode it end-to-end — nothing is mocked.
- Frontend tests cover the platform-neutral core (WebSocket batching + session-resume reconnect, formatting, type bridging) and the downsampler. They are plain-logic tests — no browser or jsdom needed.

Please make sure the full suite is green before opening a PR, and add tests for any behavior you add or change. Bug fixes should come with a test that fails without the fix.

## Architecture in five sentences

The browser (React + Vite app in `web/`, built into `static/`) captures mic audio in an `AudioWorklet`, downsamples to 16 kHz i16 PCM, and streams it over a WebSocket. `src/server.rs` (axum) accepts the socket and hands audio to a per-connection pipeline thread (`src/pipeline.rs`). The pipeline runs Silero VAD (`src/vad.rs`) → a pure segmentation state machine (`src/segmenter.rs`) → whisper.cpp transcribe + translate (`src/whisper.rs`) → speaker embedding + online clustering (`src/speakers.rs`) → SQLite (`src/db.rs`), then pushes a JSON event back over the socket. `src/config.rs` maps `config.toml` + `LT_*` env vars. The design spec that drove the implementation lives in the repo history if you want the deeper rationale.

A few interface contracts worth knowing before you change things:

- **Audio is 16 kHz mono i16 everywhere in the backend.** VAD consumes exactly 512-sample (32 ms) chunks.
- **`segmenter.rs` is pure** (no I/O, no models) — keep it that way; it's the most heavily unit-tested logic in the repo.
- **Event JSON shapes** (`PipelineEvent` in `src/pipeline.rs`) and REST shapes (`src/db.rs` structs) are mirrored by `web/src/core/types.ts` — change them in lockstep.
- **`web/src/core/` must stay platform-neutral** — no DOM, no `window`/`document`/`location`. It is the module a future React Native app imports; DOM-specific code lives in `web/src/audio/` and `web/src/ui/`.
- **Never use `dangerouslySetInnerHTML`** — all transcript text is server-originated; React's default text rendering is the XSS boundary.
- **Content policy and post-transcript actions belong in the harness** (`src/harness/`), not inline in the pipeline. Implement the `Guardrail` trait (inspect/rewrite/block an utterance) or the `Tool` trait (side-effectful actions after guardrails pass) and register it in `Harness::from_config` — the pipeline itself should never grow policy logic.

## Code style

- `cargo fmt` before committing; `cargo clippy` should introduce no new warnings.
- Match the existing error-handling patterns: `anyhow::Result` at module boundaries; the pipeline logs-and-continues rather than crashing the session.
- Frontend: React + TypeScript, strict mode, function components and hooks only; plain CSS with the design tokens in `web/src/styles.css` (no CSS-in-JS, no UI libraries).
- Keep files single-responsibility; if a module is outgrowing its purpose, raise it in the PR rather than restructuring unilaterally.

## Pull requests

1. Fork and create a topic branch.
2. Keep PRs focused — one logical change per PR.
3. Include: what changed, why, and how you tested it (paste test output for non-trivial changes).
4. Update the README when you change user-facing behavior (config keys, API shapes, CLI).
5. CI-less for now: reviewers will run the suite, so state clearly which tests cover your change.

## Good first contributions

The [Roadmap](README.md#roadmap) lists known gaps. Especially valuable right now:

- **Language test fixtures** — short (5–15 s), freely-licensed 16 kHz mono WAV clips of Hindi, Urdu, or Arabic speech (and a two-speaker clip) with expected transcripts, wired into the existing pipeline integration test.
- **Session-resume timestamp continuity** and **WebSocket ping/pong liveness** — both are contained changes in `src/server.rs` / `src/pipeline.rs`.
- Diarization tuning reports: real-world accuracy notes with different `speaker_similarity_threshold` / `min_new_speaker_ms` values help calibrate defaults.

## Reporting bugs

Open an issue with: what you did, what you expected, what happened, your platform (OS/arch, Docker vs native), the model size, and relevant server logs (`docker compose logs`). For transcription-quality issues, note the language and, if possible, attach a short audio sample.

## License

By contributing, you agree that your contributions will be licensed under the [MIT License](LICENSE).
