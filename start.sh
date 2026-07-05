#!/usr/bin/env bash
set -e

PORT="${1:-8501}"
DIR="$(cd "$(dirname "$0")" && pwd)"

echo "==> Installing deps..."
pip install -r "$DIR/requirements.txt"

echo "==> Starting streamlit on 0.0.0.0:$PORT (Tailscale mobile)"
echo "    Access via: http://$(hostname):$PORT or http://<tailscale-ip>:$PORT"
streamlit run "$DIR/app.py" \
  --server.address 0.0.0.0 \
  --server.port "$PORT" \
  --server.headless true \
  --server.enableWebsocketCompression false
