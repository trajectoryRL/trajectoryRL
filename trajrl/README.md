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
trajrl scores <validator_hotkey>       # Per-miner scores from a validator
trajrl scores --uid <uid>              # Query by validator UID instead
trajrl miner <hotkey>                  # Miner detail + diagnostics
trajrl miner --uid <uid>               # Query by miner UID instead
trajrl pack <hotkey> <pack_hash>       # Pack evaluation detail
trajrl submissions [--failed]          # Recent pack submissions
trajrl eval-history <validator>        # List eval cycle IDs for a validator
trajrl eval-history <v> --from <date>  # Filter by date range
trajrl cycle-log <validator>           # Download and display a cycle log
trajrl cycle-log <v> --format summary  # Show parsed summary tables
trajrl logs [--type cycle|miner]       # Eval log archives
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

- **UID support**: Query validators and miners by UID instead of hotkey
  ```bash
  trajrl miner --uid 65      # Instead of full hotkey
  trajrl scores --uid 221    # Query validator by UID
  ```

- **Date filtering**: Filter eval history by date range
  ```bash
  trajrl eval-history 5Cd6h... --from 2026-03-25 --to 2026-03-26
  ```

- **Cycle log summary**: Parse cycle logs into structured tables
  ```bash
  trajrl cycle-log 5Cd6h... --format summary
  ```
  Shows: eval metrics, winner info, top qualified miners in tables instead of raw text

- **Version command**: Check your CLI version
  ```bash
  trajrl --version
  ```

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

### List validators

```bash
trajrl validators
```
```
 Hotkey         UID  Version  LLM Model              Last Eval   Last Seen
 5Cd6h…sn11     29  0.2.7    chutes/zai-org/GLM-5…   7h ago      2m ago
 5EcgNd…797f   221  0.2.7    zhipu/glm-5             10h ago     6m ago
 ...
```

### Inspect a miner

By hotkey:
```bash
trajrl miner 5HMgR6LnNqUAtaKRwa6bLF4Vy4KBf7TaxCLehyff9mWPhSHt
```

Or by UID:
```bash
trajrl miner --uid 65
```

Shows rank, qualification status, cost, scenario breakdown, per-validator reports, recent submissions, and ban records.

### Check a validator's scores

See how a specific validator scored all miners:

```bash
trajrl scores --uid 221
```

Returns per-miner entries with `qualified`, `costUsd`, `score`, `weight`, `scenarioScores`, and rejection info. Useful for comparing validator behavior or debugging why a miner was rejected.

### View failed submissions

```bash
trajrl submissions --failed
```

Shows recent packs that failed pre-eval integrity checks, with rejection stage and reason.

### Investigate a validator's eval cycle

First, list recent eval cycles for a validator:

```bash
trajrl eval-history 5Cd6h...
```
```
            Eval IDs (5) — 5Cd6h…sn11
 Eval ID            Validator    Block  Logs  Created
 20260325_060012    5Cd6h…sn11  421890    12  3h ago
 20260324_060008    5Cd6h…sn11  421530    11  1d ago
 20260323_060015    5Cd6h…sn11  421170    13  2d ago
 ...
```

Then fetch the full cycle log for a specific eval:

```bash
trajrl cycle-log 5Cd6h... --eval-id 20260325_060012
```

Or just grab the latest one:

```bash
trajrl cycle-log 5Cd6h...
```

Use `--format summary` to parse the raw log into structured tables — eval metrics, winner info, and top qualified miners:

```bash
trajrl cycle-log 5Cd6h... --format summary
```

The cycle log contains the complete eval cycle output: metagraph sync, miner enumeration, per-miner eval timing, WEIGHT RESULTS, and set_weights status.

> **Note:** Eval IDs are defined by each validator independently (typically a timestamp like `20260325_060012`). They are **not** globally unique — the same eval ID from different validators refers to different evaluation cycles. Always pair an eval ID with a specific validator hotkey.

### Inspect a specific pack

Check how a miner's pack was evaluated across all validators:

```bash
trajrl pack 5HMgR6... abc123def456...
```

Returns the pack's aggregated qualification status, best/average cost, and per-validator scenario breakdowns.

### Filter eval logs

```bash
trajrl logs --type cycle --limit 5
trajrl logs --validator 5Cd6h... --type miner
trajrl logs --eval-id 20260324_000340
trajrl logs --miner 5HMgR6... --pack-hash abc123...
```

### JSON output for agents

Pipe to any tool — JSON is automatic:

```bash
trajrl validators | jq '.validators[].hotkey'
trajrl scores --uid 221 | jq '.entries[] | select(.qualified) | {uid, costUsd, weight}'
trajrl miner --uid 65 | jq '.scenarioSummary'
```

Parse with Python:

```bash
trajrl scores 5Cd6h... | python3 -c "
  import sys, json
  d = json.load(sys.stdin)
  for e in d['entries'][:5]:
      print(f\"{e['minerHotkey'][:12]}  qual={e['qualified']}  cost={e['costUsd']}\")
"
```

Force JSON in an interactive terminal:

```bash
trajrl miner 5HMgR6... --json
```

## API Reference

All data comes from the [TrajectoryRL Public API](https://trajrl.com) — read-only, no authentication required. See [PUBLIC_API.md](PUBLIC_API.md) for full endpoint documentation.

| Endpoint | CLI Command |
|----------|-------------|
| `GET /api/validators` | `trajrl validators` |
| `GET /api/scores/by-validator?validator=` | `trajrl scores <hotkey>` |
| `GET /api/miners/:hotkey` | `trajrl miner <hotkey>` |
| `GET /api/miners/:hotkey/packs/:hash` | `trajrl pack <hotkey> <hash>` |
| `GET /api/submissions` | `trajrl submissions` |
| `GET /api/eval-logs` | `trajrl logs` |
| `GET /api/eval-logs?log_type=cycle` | `trajrl eval-history <hotkey>` |
| `GET /api/eval-logs` + GCS download | `trajrl cycle-log <hotkey>` |

