# Contributing to Tarjuman

Thanks for your interest in contributing! Tarjuman is a young project and welcomes bug reports, feature discussion, documentation improvements, test fixtures in more languages, and code.

## Getting set up

**Prerequisites**

- Rust stable **1.85+** (`rustup` recommended)
- `cmake` — required to build sherpa-onnx (`brew install cmake` on macOS, `apt install cmake` on Debian/Ubuntu)
- A C/C++ toolchain (clang preferred; see the Dockerfile notes about gcc-12 on ARM)
- Docker (only if you're touching the container packaging)

**First build**

```bash
git clone <repo-url>
cd tarjuman
./scripts/fetch-models.sh tiny   # downloads the small test models (~120 MB)
cargo build
```

The first build compiles whisper.cpp and sherpa-onnx from source and downloads ONNX Runtime — expect 10–20 minutes. Subsequent builds are incremental and fast.

**Run the dev server**

```bash
LT_WHISPER_MODEL=tiny cargo run
# open http://localhost:8080
```

On macOS, add `--release --features metal` for GPU-accelerated inference.

## Running tests

```bash
cargo test -- --test-threads=1
```

- `--test-threads=1` is required: several tests run real whisper/ONNX inference and are memory- and CPU-heavy.
- Tests that need models will panic with `run scripts/fetch-models.sh tiny` if you haven't fetched them.
- The audio fixture `tests/fixtures/jfk.wav` (16 kHz mono 16-bit PCM) is committed; tests decode it end-to-end — nothing is mocked.

Please make sure the full suite is green before opening a PR, and add tests for any behavior you add or change. Bug fixes should come with a test that fails without the fix.

## Architecture in five sentences

The browser (static/, no build step) captures mic audio in an `AudioWorklet`, downsamples to 16 kHz i16 PCM, and streams it over a WebSocket. `src/server.rs` (axum) accepts the socket and hands audio to a per-connection pipeline thread (`src/pipeline.rs`). The pipeline runs Silero VAD (`src/vad.rs`) → a pure segmentation state machine (`src/segmenter.rs`) → whisper.cpp transcribe + translate (`src/whisper.rs`) → speaker embedding + online clustering (`src/speakers.rs`) → SQLite (`src/db.rs`), then pushes a JSON event back over the socket. `src/config.rs` maps `config.toml` + `LT_*` env vars. The design spec that drove the implementation lives in the repo history if you want the deeper rationale.

A few interface contracts worth knowing before you change things:

- **Audio is 16 kHz mono i16 everywhere in the backend.** VAD consumes exactly 512-sample (32 ms) chunks.
- **`segmenter.rs` is pure** (no I/O, no models) — keep it that way; it's the most heavily unit-tested logic in the repo.
- **Event JSON shapes** (`PipelineEvent` in `src/pipeline.rs`) and REST shapes (`src/db.rs` structs) are consumed by `static/app.js` — change them in lockstep.
- **Everything rendered via `innerHTML` in `app.js` must pass through `escapeHtml`.**

## Code style

- `cargo fmt` before committing; `cargo clippy` should introduce no new warnings.
- Match the existing error-handling patterns: `anyhow::Result` at module boundaries; the pipeline logs-and-continues rather than crashing the session.
- Frontend stays framework-free and build-step-free by design — vanilla JS only.
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
