#!/usr/bin/env bash
# TrajectoryRL Validator — All-in-One Entrypoint
#
# Manages three processes inside a single container:
#   1. mock-tools   (Python/FastAPI on port 3001)
#   2. OpenClaw     (Node.js gateway on port 18789)
#   3. validator    (Python, foreground — PID 1 via exec)
#
# On startup, runs init_workspace.py once to set up fixtures + config.

set -euo pipefail

echo "[entrypoint] TrajectoryRL Validator (all-in-one)"

# ── 1. Environment setup ────────────────────────────────────────
# Paths for init_workspace.py (override legacy Docker volume mount defaults)
export SCENARIOS_DIR="${SCENARIOS_DIR:-/app/clawbench/scenarios}"
export FIXTURES_DIR="${FIXTURES_DIR:-/app/clawbench/fixtures}"
export WORKSPACE_DIR="${WORKSPACE_DIR:-/workspace}"
export WORKSPACE_PATH="${WORKSPACE_PATH:-$WORKSPACE_DIR}"
export CONFIG_DIR="${CONFIG_DIR:-/app/clawbench/config}"
export OPENCLAW_HOME="${OPENCLAW_HOME:-/openclaw-home}"

# OpenClaw reads OPENAI_* env vars for LLM routing
export OPENAI_API_KEY="${CLAWBENCH_LLM_API_KEY:-}"
export OPENAI_BASE_URL="${CLAWBENCH_LLM_BASE_URL:-https://open.bigmodel.cn/api/paas/v4}"
export OPENCLAW_GATEWAY_TOKEN="${OPENCLAW_GATEWAY_TOKEN:-sandbox-token-12345}"

# All-in-one defaults (no Docker port mapping)
export OPENCLAW_URL="${OPENCLAW_URL:-http://localhost:18789}"
export MOCK_TOOLS_URL="${MOCK_TOOLS_URL:-http://localhost:3001}"

# ── 2. Init workspace (one-shot) ────────────────────────────────
echo "[entrypoint] Initializing workspace..."
mkdir -p "$WORKSPACE_DIR" "$OPENCLAW_HOME"
python /app/clawbench/scripts/init_workspace.py

chmod -R 777 "$WORKSPACE_DIR"
echo "[entrypoint] Workspace ready"

# ── 3. Start mock-tools server (background) ─────────────────────
echo "[entrypoint] Starting mock-tools on port 3001..."
python -m mock_tools.server &
MOCK_PID=$!

for i in $(seq 1 30); do
    if python -c "import urllib.request; urllib.request.urlopen('http://localhost:3001/health')" 2>/dev/null; then
        echo "[entrypoint] mock-tools ready"
        break
    fi
    if [ "$i" -eq 30 ]; then
        echo "[entrypoint] ERROR: mock-tools failed to start within 30s"
        exit 1
    fi
    sleep 1
done

# ── 4. Start OpenClaw gateway (background) ──────────────────────
echo "[entrypoint] Starting OpenClaw gateway on port 18789..."
cd /app/openclaw
node dist/index.js gateway --allow-unconfigured --bind loopback &
OPENCLAW_PID=$!
cd /app

for i in $(seq 1 60); do
    if curl -sf http://localhost:18789/health >/dev/null 2>&1; then
        echo "[entrypoint] OpenClaw gateway ready"
        break
    fi
    if [ "$i" -eq 60 ]; then
        echo "[entrypoint] ERROR: OpenClaw gateway failed to start within 60s"
        exit 1
    fi
    sleep 1
done

# ── 5. Signal handling ──────────────────────────────────────────
cleanup() {
    echo "[entrypoint] Shutting down..."
    kill "$MOCK_PID" "$OPENCLAW_PID" 2>/dev/null || true
    wait "$MOCK_PID" "$OPENCLAW_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# ── 6. Start validator (foreground) ─────────────────────────────
echo "[entrypoint] Starting validator..."
exec python -u neurons/validator.py "$@"
