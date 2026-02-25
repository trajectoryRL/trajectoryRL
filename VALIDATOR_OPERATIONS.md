# Validator Operations Guide

**Subnet**: SN11 (TrajectoryRL)
**Date**: 2026-02-24

> Operational guidance for running a TrajectoryRL validator. For mechanism design and scoring rules, see [INCENTIVE_MECHANISM.md](INCENTIVE_MECHANISM.md).

---

## Current Phase: Cold-Start

The subnet is bootstrapping. Validators set consensus weights to anchor the network while the full ClawBench evaluation pipeline is finalized. **Mainnet launch with live evaluation is expected in ~1 week.** Your validator will automatically transition to the production pipeline via Watchtower — no manual action required.

During cold-start, the validator:
- Sets weights every ~25 minutes
- Requires no LLM API key (no inference costs)
- Auto-updates when we push new code to `main`

---

## Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/trajectoryRL/trajectoryRL.git
cd trajectoryRL

# 2. Create your .env file
cat > .env.validator <<'EOF'
WALLET_NAME=your-wallet-name
WALLET_HOTKEY=default
NETUID=11
NETWORK=finney
EOF

# 3. Start the validator + Watchtower (auto-updates from GHCR)
docker compose -f docker/docker-compose.validator.yml --env-file .env.validator up -d

# 4. Check logs
docker logs -f trajectoryrl_validator
docker logs -f trajectoryrl_watchtower
```

Requirements: Docker, a registered SN11 hotkey. That's it.

---

## Automatic Updates (Watchtower + GHCR)

The validator image is hosted on GitHub Container Registry (`ghcr.io/trajectoryrl/trajectoryrl:latest`). A Watchtower sidecar polls GHCR every 5 minutes and automatically pulls new images when we push code to `main`.

**Update flow:**
1. Team pushes code to `main`
2. GitHub Actions builds and pushes a new image to GHCR (~1 min)
3. Watchtower detects the new image (~5 min poll interval)
4. Watchtower stops the old container, starts a new one with the updated image
5. Total latency: **~5 minutes from push to running new code**

**No git pull, no manual rebuilds, no downtime management.**

### Verify auto-updates are working

```bash
# Watchtower should show "Found new image" entries
docker logs trajectoryrl_watchtower

# Validator should show a recent start time
docker inspect trajectoryrl_validator --format '{{.Created}}'
```

### Running without auto-update

If you prefer manual control, pull the image yourself:

```bash
docker compose -f docker/docker-compose.validator.yml --env-file .env.validator pull
docker compose -f docker/docker-compose.validator.yml --env-file .env.validator up -d
```

---

## Cost Asymmetry: Validators Pay, Miners Don't

Validators bear **all LLM inference costs** — they run ClawBench episodes against each miner's policy pack. Miners submit static policy packs (JSON) and pay zero inference cost per epoch; their only costs are registration and R&D iteration.

This is by design: miners compete on *intelligence* (better prompts/policies), not on compute.

## Validator Cost Model

Each epoch, a validator evaluates every active miner whose pack has changed since last evaluation:

```
episodes_per_new_miner = scenarios(5) × 1 run each = 5 per miner
epochs_per_day         = 24h / epoch_interval(24h) = 1
episodes_per_day       = new_or_changed_miners × 5
```

> **Score persistence**: Validators only re-evaluate a miner when their `pack_hash` changes. Unchanged packs carry forward their cached score at zero cost. In steady state, most epochs evaluate only 0-2 new submissions.

**Per-episode token estimate** (averaged across 5 scenarios):

| Component | Tokens |
|-----------|--------|
| System prompt (miner's AGENTS.md) | ~300 |
| User message | ~80 |
| Workspace context (USER.md) | ~220 |
| Fixture data (emails, calendar, tasks) | ~1,600 |
| **Total input** | **~2,200** |
| **Output (agent response)** | **~900** |

## Daily Cost Projections

Designated model: `anthropic/claude-sonnet-4-5-20250929` ($3/M input, $15/M output). Cost per episode ≈ **$0.020**. Will switch to `claude-sonnet-4-6` once OpenClaw supports it (same pricing).

**All validators must use the designated model.** This is a consensus requirement: if validators use different models, agents produce different tool-call sequences, leading to different rubric outcomes and validator disagreement on scores. Using the wrong model puts your validator out of consensus and risks down-weighting by Yuma.

### Worst-case: all miners submit new packs every epoch

| Active Miners | Episodes/day | Daily Cost | Monthly Cost |
|:-------------:|:------------:|:----------:|:------------:|
| 5 | 25 | **$0.50** | **$15** |
| 14 | 70 | **$1.40** | **$42** |
| 30 | 150 | **$3.00** | **$90** |
| 64 | 320 | **$6.40** | **$192** |
| 128 | 640 | **$12.80** | **$384** |
| 256 | 1,280 | **$25.60** | **$768** |

**Worst-case formula**: `daily_cost ≈ miners × $0.10/day` (5 scenarios × $0.02/episode).

In practice, most epochs only re-evaluate a handful of new/changed packs. A typical day with 30 miners and 2 new submissions costs ~$0.20, not $3.00.

## Miner Cost Model

| Cost Item | Estimate |
|-----------|----------|
| Policy iteration (prompt tuning) | Engineer time only |
| Local testing via ClawBench | ~$0.02/episode × ~50 test runs ≈ **$1/iteration** |
| GitHub repo hosting | Free |
| Bittensor registration | ~200 TAO (one-time) |
| **Ongoing operational cost** | **~$0/month** |

## Cost Reduction Levers

1. **Score persistence** (built-in): Validators only re-evaluate when `pack_hash` changes — unchanged packs cost zero. This is the primary cost control mechanism
2. **24h epoch interval** (built-in): Caps evaluation to once per miner per day, even if a miner submits multiple times
3. **Prompt caching**: Anthropic prompt caching saves ~80% on input tokens (fixture data is identical across runs for the same scenario)

## Sustainability

Validator economics depend on alpha earnings (convertible to TAO) exceeding LLM costs.

Validators earn **subnet alpha**, not TAO directly. Alpha can be swapped for TAO via the subnet's liquidity pool at a market-determined rate. Current SN11 alpha price: ~$2.64 (≈0.015 TAO at ~$180/TAO).

```
Estimated alpha earnings (medium stake ~5k TAO, ~10% validator weight):
  ~295 alpha/day ≈ 4 TAO-equivalent at current pool rate ≈ $720/day

Example (30 miners, worst-case all submit new packs):
  Daily costs:   30 × $0.10 = $3.00/day
  Daily revenue: ~$720/day (alpha, at current pool rate)
  Net profit:    ~$717/day (~99% margin)

Example (30 miners, typical day with 2 new submissions):
  Daily costs:   2 × $0.10 = $0.20/day
  Daily revenue: ~$720/day
  Net profit:    ~$720/day
```

**At current rates**, TrajectoryRL validators are highly profitable:

| Scenario | Daily Cost (worst-case) | Daily Revenue (~$720 alpha) | Monthly Profit |
|----------|:----------:|:---------------------------:|:--------------:|
| 30 miners | $3.00 | $720 | **$21,510** |
| 64 miners | $6.40 | $720 | **$21,408** |
| 128 miners | $12.80 | $720 | **$21,216** |
| 256 miners | $25.60 | $720 | **$20,832** |

Even at 256 miners (worst case, all submitting new packs every day), LLM costs are only **~4%** of validator alpha revenue. In practice, costs are far lower due to score persistence.

**Break-even analysis**: At 256 miners ($25.60/day worst-case cost), the alpha-TAO pool rate would need to drop ~28x from current levels before validators become unprofitable. Note: these figures fluctuate with pool exchange rates and subnet demand.

## Score Publishing

Validators publish per-UID scores to the shared `trajectoryRL/validator-scores` GitHub repo after each evaluation. This is required for stake-weighted consensus: validators that don't publish are excluded from the aggregation.

Validators do not have direct write access to the repo. Scores are submitted via PRs that a CI pipeline auto-merges after verifying the sr25519 payload signature inside the JSON.

Each epoch, the validator:
1. Evaluates new/changed packs via ClawBench
2. Creates a score file with an sr25519 signature over the payload
3. Commits to its fork of `validator-scores`, opens a PR with the score file at `epoch-{N}/{hotkey}.json`
4. CI verifies the payload signature, hotkey registration, stake, and JSON schema, then auto-merges
5. Pulls all merged scores, computes stake-weighted mean
6. Sets on-chain weights based on the consensus winner

Every tempo (~72 min), the validator re-pulls scores, re-computes consensus, and re-submits weights. This allows consensus to converge as more validators publish their results.

For full details on the shared score bucket, signed score files, and stake-weighted aggregation, see [INCENTIVE_MECHANISM.md — Validator Consensus](INCENTIVE_MECHANISM.md#validator-consensus).

