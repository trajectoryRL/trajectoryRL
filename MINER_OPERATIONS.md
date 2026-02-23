# Miner Operations Guide

**Subnet**: SN11 (TrajectoryRL)
**Date**: 2026-02-23

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
| **GitHub account** | Public repo for pack submission |
| **LLM API key** | For local ClawBench testing (Anthropic, OpenAI, or local model) |

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

## Reference Miner Implementation (WIP)

The reference miner uses **Claude Opus 4.6** to autonomously generate, test, and iterate policy packs. It will be included in this repo at `neurons/miner.py`.

### How It Works

```
┌─────────────────────────────────────────────────────────┐
│  Reference Miner Loop                                   │
│                                                         │
│  1. Read scenario YAMLs + rubric checks as context      │
│  2. Generate candidate AGENTS.md via Opus 4.6           │
│  3. Run ClawBench locally, collect per-check results    │
│  4. Feed failed checks back to Opus 4.6 for iteration   │
│  5. Repeat until score stabilizes or budget exhausted   │
│  6. Push winning pack to GitHub                         │
│  7. Submit on-chain commitment (pack_hash + git info)   │
└─────────────────────────────────────────────────────────┘
```

### What Opus 4.6 Sees

The LLM receives:
- All scenario YAML definitions (check types, point values, categories)
- Available tool surface (exec, slack, memory_search, memory_get, read)
- Failed check IDs and their point values from the previous iteration
- The scoring formula (weighted mean - variance penalty)

From this context, Opus 4.6 generates an AGENTS.md that targets the highest-value failed checks first (safety > correctness > efficiency > structure).

### Why This Approach

Policy optimization is fundamentally a language task — understanding what rubric checks test, then writing instructions that cause an agent to pass them. An LLM that can read the scoring rubric and iterate on failures is a natural fit. The reference miner demonstrates this loop end-to-end.

Miners are free to use any approach: manual prompt engineering, different LLMs, evolutionary search, or hybrid strategies. The reference miner is a starting point, not a ceiling.

**Status**: WIP — will be implemented and included in this repo.

---

## Local Testing

### Setup

```bash
git clone https://github.com/trajectoryRL/trajectoryRL.git
cd trajectoryRL/clawbench
pip install -e .
cp .env.example .env
# Edit .env — set CLAWBENCH_MODEL (e.g., anthropic/claude-sonnet-4-5-20250929)
```

Validators use `claude-sonnet-4-5-20250929`. Miners can use any model for local testing since scoring is regex-based. Use a cheaper model for rapid iteration, validate final results with Sonnet 4.5.

### Running Scenarios

```bash
# Single scenario
python scripts/run_episode.py --scenario inbox_triage --variant optimized --json

# All scenarios
python scripts/run_batch.py

# List available scenarios
python scripts/run_episode.py --list
```

### Testing Your Own Pack

```bash
cp /path/to/your/AGENTS.md clawbench/fixtures/inbox_triage/AGENTS.md.optimized
python scripts/run_episode.py --scenario inbox_triage --variant optimized --json
```

The `--json` output shows per-check pass/fail results. Focus on failed checks with the highest point values.

---

## Submission Workflow

### 1. Push to Public GitHub

```bash
mkdir my-trajectoryrl-pack && cd my-trajectoryrl-pack
git init
# Add your AGENTS.md (and optionally SOUL.md)
git add AGENTS.md
git commit -m "v1.0.0: initial policy pack"
git remote add origin https://github.com/YOUR_USER/my-trajectoryrl-pack.git
git push -u origin main
```

### 2. Submit On-Chain Commitment

After pushing to GitHub, submit your pack metadata on-chain:

```bash
python neurons/miner.py \
  --wallet.name miner \
  --wallet.hotkey default \
  --netuid 11 \
  --pack_repo https://github.com/YOUR_USER/my-trajectoryrl-pack \
  --pack_commit $(git rev-parse HEAD)
```

This calls `subtensor.set_commitment(netuid=11, data=commitment_string)` with your `pack_hash`, `git_commit_hash`, and `repo_url`. The **on-chain commitment block number** establishes first-mover precedence (unforgeable, deterministic).

> **Rate limit**: One commitment per ~100 blocks (~20 min) per hotkey — sufficient for daily epochs.

### 3. Iterate

Epochs run every 24 hours. Push improved packs and submit a new on-chain commitment — the next epoch picks up your latest submission. Validators only re-evaluate when your `pack_hash` changes, so resubmitting the same pack is a no-op.

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
