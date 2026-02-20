# Validator Operations Guide

**Subnet**: SN11 (TrajectoryRL)
**Date**: 2026-02-19

> Operational guidance for running a TrajectoryRL validator. For mechanism design and scoring rules, see [INCENTIVE_MECHANISM.md](INCENTIVE_MECHANISM.md).

---

## Cost Asymmetry: Validators Pay, Miners Don't

Validators bear **all LLM inference costs** — they run ClawBench episodes against each miner's policy pack. Miners submit static policy packs (JSON) and pay zero inference cost per epoch; their only costs are registration and R&D iteration.

This is by design: miners compete on *intelligence* (better prompts/policies), not on compute.

## Validator Cost Model

Each epoch, a validator evaluates every active miner:

```
episodes_per_epoch = scenarios_per_epoch(4) × seeds_per_task(3) = 12 per miner
epochs_per_day     = 24h / epoch_interval(24h) = 1
episodes_per_day   = miners × 12 × 1 = miners × 12
```

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

| Active Miners | Episodes/day | Daily Cost | Monthly Cost |
|:-------------:|:------------:|:----------:|:------------:|
| 5 | 60 | **$1** | **$36** |
| 14 | 168 | **$3** | **$101** |
| 30 | 360 | **$7** | **$216** |
| 64 | 768 | **$15** | **$461** |
| 128 | 1,536 | **$31** | **$922** |
| 256 | 3,072 | **$61** | **$1,843** |

**Formula**: `daily_cost ≈ miners × $0.24/day` (at Sonnet pricing).

## Miner Cost Model

| Cost Item | Estimate |
|-----------|----------|
| Policy iteration (prompt tuning) | Engineer time only |
| Local testing via ClawBench | ~$0.02/episode × ~50 test runs ≈ **$1/iteration** |
| GitHub repo hosting | Free |
| Bittensor registration | ~200 TAO (one-time) |
| **Ongoing operational cost** | **~$0/month** |

## Cost Reduction Levers

1. **Prompt caching**: Anthropic prompt caching saves ~80% on input tokens (fixture data is identical across seeds for the same scenario)
2. **Fewer seeds**: `seeds_per_task=1` instead of 3, 3x cheaper (but weaker consensus)
3. **Fewer scenarios**: 2 per epoch instead of 4, 2x cheaper
4. **Skip unchanged packs**: don't re-evaluate miners whose pack hash hasn't changed since last epoch

## Sustainability

Validator economics depend on alpha earnings (convertible to TAO) exceeding LLM costs.

Validators earn **subnet alpha**, not TAO directly. Alpha can be swapped for TAO via the subnet's liquidity pool at a market-determined rate. Current SN11 alpha price: ~$2.64 (≈0.015 TAO at ~$180/TAO).

```
Estimated alpha earnings (medium stake ~5k TAO, ~10% validator weight):
  ~295 alpha/day ≈ 4 TAO-equivalent at current pool rate ≈ $720/day

Example (30 miners):
  Daily costs:   30 × $0.24 = $7.20/day
  Daily revenue: ~$720/day (alpha, at current pool rate)
  Net profit:    ~$713/day (~99% margin)
```

**At current rates**, TrajectoryRL validators are highly profitable:

| Scenario | Daily Cost | Daily Revenue (~$720 alpha) | Monthly Profit |
|----------|:----------:|:---------------------------:|:--------------:|
| 30 miners | $7 | $720 | **$21,390** |
| 64 miners | $15 | $720 | **$21,150** |
| 128 miners | $31 | $720 | **$20,670** |
| 256 miners | $61 | $720 | **$19,770** |

Even at 256 miners (worst case), LLM costs are only **~8%** of validator alpha revenue. Sonnet 4.5 remains economically viable at any realistic scale.

**Break-even analysis**: At 256 miners ($61/day cost), the alpha-TAO pool rate would need to drop ~12x from current levels before validators become unprofitable. Note: these figures fluctuate with pool exchange rates and subnet demand.

## Automatic Updates

Validators should run with `docker compose watch` to automatically pick up new scenarios, scoring updates, and code changes without manual container rebuilds.

### Starting with auto-update

```bash
# Start all services with file watching enabled
docker compose up --watch
```

### How it works

When the team releases updates (new scenarios, scoring fixes, validator code), pull the changes:

```bash
git pull --recurse-submodules
```

`docker compose watch` detects the file changes and automatically applies them:

| Change type | Action | Downtime |
|-------------|--------|----------|
| Validator source code (`trajectoryrl/`, `neurons/`) | Sync + restart | Seconds |
| ClawBench scenarios, fixtures, scoring (`clawbench/`) | Sync + restart | Seconds |
| Mock tools server (`clawbench/clawbench/mock_tools/`) | Sync + restart | Seconds |
| Dependencies (`requirements.txt`, `pyproject.toml`) | Full rebuild | Minutes |
| Dockerfile changes | Full rebuild | Minutes |

### Optional: automated git pull

Set up a cron job to pull updates periodically:

```bash
# Pull every 6 hours (add to crontab -e)
0 */6 * * * cd /path/to/trajectoryrl && git pull --recurse-submodules >> /var/log/trajectoryrl-pull.log 2>&1
```

### Running without auto-update

If you prefer manual control, run detached without watch:

```bash
docker compose up -d

# After pulling updates, manually rebuild:
git pull --recurse-submodules
docker compose up -d --build
```
