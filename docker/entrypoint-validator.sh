#!/usr/bin/env bash
# TrajectoryRL Validator — Entrypoint
#
# Season 1: the validator orchestrates trajrl-bench sandbox containers
# via Docker API.  No embedded services needed.

set -euo pipefail

log() { echo "$(date -u '+%Y-%m-%dT%H:%M:%S%z') [entrypoint] $*"; }

log "TrajectoryRL Validator (Season 1 — trajrl-bench)"

# ── Start validator (foreground) ─────────────────────────────
log "Starting validator..."
exec python -u neurons/validator.py "$@"
