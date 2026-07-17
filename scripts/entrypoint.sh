#!/usr/bin/env bash
set -euo pipefail
MODEL="${LT_WHISPER_MODEL:-medium}"
./scripts/fetch-models.sh "$MODEL" models
exec /app/bin/tarjuman
