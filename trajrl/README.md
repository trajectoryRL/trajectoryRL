# trajrl

CLI for the [TrajectoryRL subnet](https://trajrl.com) (Bittensor SN11). Query live validator, miner, and evaluation data from the terminal.

Designed for AI agents (Claude Code, Cursor, Codex, OpenClaw, Manus) and humans alike — outputs JSON when piped, Rich tables when interactive.

## Install

```bash
pip install trajrl
```

## Commands

```
trajrl status                          # Network health overview
trajrl validators                      # List all validators
trajrl scores                          # Per-miner scores (picks first active validator)
trajrl scores --uid <uid>              # Per-miner scores from a specific validator
trajrl miner --uid <uid>               # Miner detail + diagnostics
trajrl miner <hotkey>                  # Query by hotkey
trajrl download -u <uid>               # Download miner's current pack + eval results
trajrl download <hotkey> <pack_hash>   # Download a specific pack version
trajrl submissions [--failed]          # Recent pack submissions
trajrl logs                            # List eval log archives
trajrl logs --type cycle               # List cycle logs only
trajrl logs --show                     # Download and display the latest cycle log
trajrl --version                       # Show CLI version
```

### Global Options

Every command accepts:

| Option | Description |
|--------|-------------|
| `--json` / `-j` | Force JSON output (auto-enabled when stdout is piped) |
| `--base-url URL` | Override API base (default: `https://trajrl.com`, env: `TRAJRL_BASE_URL`) |
| `--version` / `-v` | Show CLI version and exit |

### New in v0.2.0

- **UID support everywhere**: Query miners by UID instead of hotkey
  ```bash
  trajrl miner --uid 65
  trajrl download -u 104
  trajrl scores --uid 221
  ```

- **`download` command**: Replaces `pack`. Downloads a miner's pack and evaluation results. Resolves hotkey and pack hash automatically from UID.
  ```bash
  trajrl download -u 104   # Just the UID — resolves everything else
  ```

- **Unified `logs` command**: Replaces `eval-history`, `cycle-log`, and old `logs`. One command for all log operations.
  ```bash
  trajrl logs                            # List archives
  trajrl logs --type cycle               # Filter by type
  trajrl logs --show                     # Download and display latest log
  trajrl logs --show --validator 5Cd6h...  # Specific validator
  ```

- **`scores` works with no arguments**: Automatically picks the first active validator instead of crashing.

## Usage Examples

### Quick network check

```bash
trajrl status
```
```
╭──────────────────── Network Status ────────────────────╮
│   Validators: 7 total, 7 active (seen <1h)             │
│   LLM Models: zhipu/glm-5 (3), chutes/GLM-5-TEE (3)    │
│   Latest Eval: 7h ago                                  │
│   Submissions: 65 passed, 35 failed (last batch)       │
╰────────────────────────────────────────────────────────╯
```

### Inspect a miner

```bash
trajrl miner --uid 65
```

Shows rank, qualification status, cost, scenario breakdown, per-validator reports, recent submissions, and ban records.

### Download a miner's pack

```bash
trajrl download -u 104
```

Returns the pack's cached content, eval status, per-validator scenario breakdowns, and the `gcsPackUrl` for downloading the verified pack JSON.

### Check scores

```bash
trajrl scores                  # Any validator
trajrl scores --uid 221        # Specific validator
```

Returns per-miner entries with `qualified`, `costUsd`, `score`, `weight`, `scenarioScores`, and rejection info.

### View failed submissions

```bash
trajrl submissions --failed
```

Shows recent packs that failed pre-eval integrity checks, with rejection reason.

### View eval logs

```bash
trajrl logs                              # List all log archives
trajrl logs --type cycle --limit 5       # Recent cycle logs
trajrl logs --show                       # Download and display latest cycle log
trajrl logs --miner 5HMgR6...           # Logs for a specific miner
```

### JSON output for agents

Pipe to any tool — JSON is automatic:

```bash
trajrl validators | jq '.validators[].hotkey'
trajrl scores | jq '.entries[] | select(.qualified) | {uid, costUsd, weight}'
trajrl miner --uid 65 | jq '.scenarioSummary'
trajrl download -u 104 | jq '.gcsPackUrl'
```

Force JSON in an interactive terminal:

```bash
trajrl miner --uid 65 --json
```

## API Reference

All data comes from the [TrajectoryRL Public API](https://trajrl.com) — read-only, no authentication required. See [PUBLIC_API.md](PUBLIC_API.md) for full endpoint documentation.

| Endpoint | CLI Command |
|----------|-------------|
| `GET /api/validators` | `trajrl validators` |
| `GET /api/scores/by-validator?validator=` | `trajrl scores [--uid <uid>]` |
| `GET /api/miners/:hotkey` | `trajrl miner [--uid <uid>]` |
| `GET /api/miners/:hotkey/packs/:hash` | `trajrl download [-u <uid>]` |
| `GET /api/submissions` | `trajrl submissions` |
| `GET /api/eval-logs` | `trajrl logs` |
| `GET /api/eval-logs` + GCS download | `trajrl logs --show` |
