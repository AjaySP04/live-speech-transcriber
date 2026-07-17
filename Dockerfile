# rust:1.83 fails: Cargo.lock's home v0.5.12 requires edition2024 (stable since Rust 1.85)
FROM rust:1.90-bookworm AS build
RUN apt-get update && apt-get install -y --no-install-recommends cmake clang git && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY . .
# gcc-12 on arm64/bookworm miscompiles ggml's NEON fp16 intrinsics (always_inline
# target-mismatch errors); use clang for the whisper.cpp/sherpa-onnx C/C++ builds.
ENV CC=clang CXX=clang++
RUN cargo build --release
# collect the binary and any dynamic libs ort/sherpa produced next to it
RUN mkdir /out \
 && cp target/release/live-transcriber /out/ \
 && (cp target/release/*.so* /out/ 2>/dev/null || true) \
 && (cp target/release/deps/*.so* /out/ 2>/dev/null || true) \
 && (find / -xdev -name 'libonnxruntime*.so*' -exec cp {} /out/ \; 2>/dev/null || true)

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
