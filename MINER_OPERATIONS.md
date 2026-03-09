# Miner Operations Guide

**Subnet**: SN11 (TrajectoryRL)
**Date**: 2026-03-04

> For mechanism design and scoring rules, see [INCENTIVE_MECHANISM.md](INCENTIVE_MECHANISM.md).

---

## What Is Mining on TrajectoryRL?

Mining means writing **policy packs** — system prompts, tool usage rules, and stop conditions — that make AI agents perform better on ClawBench scenarios. You're not running GPU workloads. You're doing policy optimization.

The best pack wins 100% of miner emissions each epoch (or top-3 split 70/20/10 in bootstrap phase). Your ongoing cost is effectively zero — all evaluation is done by validators.

---

## Prerequisites

| Requirement | Details |
|-------------|---------|
| **Bittensor wallet** | `btcli wallet create --wallet.name miner --wallet.hotkey default` |
| **Registration** | `btcli subnet register --netuid 11 --wallet.name miner` (dynamic cost, check CLI before registering) |
| **Python** | 3.10+ |
| **HTTP hosting** | Any public HTTP(S) endpoint for pack hosting (or S3-compatible bucket for `--mode default`) |
| **LLM API key** | OpenAI-compatible API key required for `--mode default`; any LLM for manual local testing |

---

## Pack Format (OPP v1)

A PolicyBundle is a JSON object. Full schema: [INCENTIVE_MECHANISM.md — Pack Schema](INCENTIVE_MECHANISM.md#pack-schema-opp-v1).

```json
{
  "schema_version": 1,
  "files": {
    "AGENTS.md": "# Your Policy\n...",
    "SOUL.md": "(optional) personality guidance..."
  },
  "tool_policy": {
    "allow": ["exec", "slack", "memory_search", "memory_get", "read"],
    "deny": ["admin_*", "shell"]
  },
  "metadata": {
    "pack_name": "my-pack",
    "pack_version": "1.0.0",
    "target_suite": "clawbench_v1"
  }
}
```

Constraints: `AGENTS.md` required, total JSON ≤ 32KB, valid semver, content-addressed via SHA256. Write AGENTS.md as a **generic policy** — avoid hardcoding specific names, companies, or dates, since the evaluation fixtures define the agent's identity context.

---

## Quick Start

### Docker (recommended)

```bash
git clone https://github.com/trajectoryRL/trajectoryRL.git
cd trajectoryRL
cp .env.miner.example .env.miner
# Edit .env.miner — set CLAWBENCH_LLM_API_KEY (+ storage config or PACK_URL)

docker compose -f docker/docker-compose.miner.yml up -d
docker compose -f docker/docker-compose.miner.yml logs -f miner
```

### Bare metal

```bash
git clone https://github.com/trajectoryRL/trajectoryRL.git
cd trajectoryRL
pip install -e .
cp .env.miner.example .env.miner
# Edit .env.miner — set CLAWBENCH_LLM_API_KEY (+ storage config or PACK_URL)

python neurons/miner.py run --mode default
```

---

## Run Modes

### Default Mode (Recommended)

Fully automated: an LLM generates AGENTS.md, builds a pack, uploads to S3-compatible storage, and submits on-chain. Each cycle improves the policy by feeding the previous AGENTS.md back to the LLM.

```
┌─────────────────────────────────────────────────────────┐
│  Default Mode Loop                                      │
│                                                         │
│  1. Generate (or improve) AGENTS.md via OpenAI-compatible API │
│  2. Build OPP v1 pack                                   │
│  3. Validate locally (schema + size)                    │
│  4. Skip if pack hash matches on-chain (no-op)          │
│  5. Upload to S3-compatible storage (GCS, AWS, R2, …)   │
│  6. Submit on-chain commitment (hash|url ≤ 128 bytes)   │
│  7. Sleep interval, then repeat with improved policy    │
└─────────────────────────────────────────────────────────┘
```

```bash
python neurons/miner.py run --mode default
python neurons/miner.py run --mode default --interval 1800  # every 30 min
```

**Requirements**: `CLAWBENCH_LLM_API_KEY` + either `S3_BUCKET` (auto-upload) or `PACK_URL` (you upload manually).

The generator prompt includes all 5 ClawBench scenario descriptions, available tool surface, rubric check categories, scoring formula (`weighted_mean - 0.1 * variance`), and policy constraints (<28K chars, no hardcoded names/dates). If a previous AGENTS.md exists from the last cycle, it's fed back with an improvement instruction.

### Demo Mode

Fetches and submits a sample pack from `trajrl.com`. Useful for verifying wallet setup and on-chain submission without needing an LLM API key or S3 bucket.

```bash
python neurons/miner.py run --mode demo
```

Policy optimization is fundamentally a language task — understanding what rubric checks test, then writing instructions that cause an agent to pass them. Miners are free to use any approach: manual prompt engineering, different LLMs, evolutionary search, or hybrid strategies. The default mode is a starting point, not a ceiling.

---

## Configuration (`.env.miner`)

```bash
cp .env.miner.example .env.miner
```

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `WALLET_NAME` | yes | `miner` | Bittensor wallet name |
| `WALLET_HOTKEY` | yes | `default` | Bittensor hotkey |
| `NETUID` | yes | `11` | Subnet ID |
| `NETWORK` | yes | `finney` | Bittensor network |
| `CLAWBENCH_LLM_API_KEY` | default mode | — | API key for AGENTS.md generation |
| `CLAWBENCH_LLM_BASE_URL` | no | `https://open.bigmodel.cn/api/paas/v4` | OpenAI-compatible API base URL |
| `CLAWBENCH_DEFAULT_MODEL` | no | `glm-5` | Model for AGENTS.md generation |
| `S3_BUCKET` | default mode* | — | S3-compatible bucket name |
| `S3_ENDPOINT_URL` | no | — | Custom endpoint for GCS/R2/MinIO |
| `S3_REGION` | no | `us-east-1` | Bucket region |
| `AWS_ACCESS_KEY_ID` | default mode* | — | S3/GCS HMAC access key |
| `AWS_SECRET_ACCESS_KEY` | default mode* | — | S3/GCS HMAC secret key |
| `PACK_URL` | no | — | Skip S3 upload; use this fixed URL instead |
| `CHECK_INTERVAL` | no | `3600` | Seconds between cycles |
| `LOG_LEVEL` | no | `INFO` | Logging level |

\* Required unless `PACK_URL` is set (in which case you upload the pack yourself).

### S3-Compatible Storage

Default mode uploads packs via presigned URLs. Works with any S3-compatible service:

| Service | `S3_ENDPOINT_URL` | Credentials |
|---------|-------------------|-------------|
| **AWS S3** | _(leave empty)_ | `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` |
| **Google Cloud Storage** | `https://storage.googleapis.com` | GCS HMAC keys |
| **Cloudflare R2** | `https://<account>.r2.cloudflarestorage.com` | R2 API tokens |
| **MinIO** | `https://your-minio-host:9000` | MinIO access/secret keys |

---

## CLI Reference

```bash
# Daemon modes
python neurons/miner.py run --mode default    # LLM generate → upload → submit (loop)
python neurons/miner.py run --mode demo       # submit sample pack (loop)

# One-shot commands
python neurons/miner.py build --agents-md ./AGENTS.md -o pack.json
python neurons/miner.py validate pack.json
python neurons/miner.py submit https://example.com/pack.json
python neurons/miner.py status
```

### Docker equivalents

```bash
# Shorthand
alias miner="docker compose -f docker/docker-compose.miner.yml"

# Daemon
miner up -d                                                      # default mode
MODE=demo miner up -d                                            # demo mode
miner logs -f miner                                              # follow logs
miner down                                                       # stop

# One-shot
miner run --rm miner status
miner run --rm miner build --agents-md /app/packs/AGENTS.md -o /app/packs/pack.json
miner run --rm miner validate /app/packs/pack.json
miner run --rm miner submit https://example.com/pack.json
```

---

## Local Testing

```bash
cd clawbench
pip install -e .
cp .env.example .env
# Edit .env — set CLAWBENCH_DEFAULT_MODEL (e.g., zhipu/glm-5)
```

Validators use the model configured by `CLAWBENCH_DEFAULT_MODEL` (commonly `zhipu/glm-5`). Miners can use any model for local testing since scoring is regex-based. Use a cheaper model for rapid iteration, then validate final results with the same model configuration used by validators.

```bash
# Single scenario
python scripts/run_episode.py --scenario inbox_triage --variant optimized --json

# All scenarios
python scripts/run_batch.py

# Test your own pack
cp /path/to/your/AGENTS.md clawbench/fixtures/inbox_triage/AGENTS.md.optimized
python scripts/run_episode.py --scenario inbox_triage --variant optimized --json
```

The `--json` output shows per-check pass/fail results. Focus on failed checks with the highest point values.

---

## Manual Submission Workflow

If you prefer writing your own AGENTS.md instead of using `--mode default`:

```bash
# 1. Build pack
python neurons/miner.py build --agents-md ./AGENTS.md -o pack.json

# 2. Upload pack.json to any public HTTP(S) URL, then submit
python neurons/miner.py submit https://your-server.com/pack.json

# 3. Verify
python neurons/miner.py status
```

The commitment format is `{pack_hash}|{pack_url}` (≤128 bytes). The **on-chain block number** establishes first-mover precedence.

> **Rate limit**: One commitment per ~100 blocks (~20 min) per hotkey — sufficient for daily epochs.

Epochs run every 24 hours. Upload improved packs and submit a new commitment — validators only re-evaluate when your `pack_hash` changes.

---

## Score Targets

```
0.90-1.00: Top-tier — competitive for winning
0.80-0.90: Strong — viable in bootstrap phase (top-3 rewards)
0.70-0.80: Good — needs iteration
0.50-0.70: Weak — missing key safety/correctness checks
0.00-0.50: Failed — likely missing tool usage or stop rules
```

Target **0.85+** to be competitive. You need `current_best + δ` (δ=0.05) to dethrone the leader.

---

## References

- **Incentive Mechanism**: [INCENTIVE_MECHANISM.md](INCENTIVE_MECHANISM.md) — scoring rules, anti-gaming, winner selection
- **ClawBench**: [clawbench/README.md](clawbench/README.md) — scenario details, fixture data
- **Validator Operations**: [VALIDATOR_OPERATIONS.md](VALIDATOR_OPERATIONS.md) — how validators evaluate your pack
