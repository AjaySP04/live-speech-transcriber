#!/usr/bin/env bash
# Expose the local Tarjuman server on a free public HTTPS URL (Cloudflare quick tunnel).
# Usage: scripts/tunnel.sh [port]   (default 8080)
# The URL is printed once the tunnel is up; it changes on every run.
# For a permanent URL, create a named tunnel with your own domain:
#   https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/
set -euo pipefail
PORT="${1:-8080}"
command -v cloudflared >/dev/null || {
  echo "cloudflared not installed. macOS: brew install cloudflared" >&2
  exit 1
}
exec cloudflared tunnel --url "http://localhost:${PORT}"
