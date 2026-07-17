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
