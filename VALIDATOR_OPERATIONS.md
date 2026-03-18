# Validator Operations Guide

**Subnet**: SN11 (TrajectoryRL)
**Date**: 2026-03-06

> Operational guidance for running a TrajectoryRL validator. For mechanism design and scoring rules, see [INCENTIVE_MECHANISM.md](INCENTIVE_MECHANISM.md).

---

## Cost Asymmetry: Validators Pay, Miners Don't

Validators bear **all LLM inference costs** — they run ClawBench episodes against each miner's policy pack. Miners submit static policy packs (JSON) and pay zero inference cost per epoch; their only costs are registration and R&D iteration.

This is by design: miners compete on *intelligence* (better prompts/policies), not on compute.

## Validator Cost Model

Each eval_interval (24h), a validator evaluates active miners marked for re-evaluation — either their `pack_hash` changed or `eval_interval` has elapsed since last eval:

```
episodes_per_eval      = scenarios(5) × 1 run each = 5 per miner
max_evals_per_day      = 24h / eval_interval(24h) = 1
episodes_per_day       = marked_miners × 5
```

> **EMA accumulation**: Validators re-evaluate packs periodically (even if unchanged) to accumulate per-scenario EMA samples and smooth out LLM non-determinism. Rate-limited to at most 1 eval per miner per eval_interval.

**Observed cost per episode** (median across 5 scenarios, multi-turn agent conversations):

Agents typically make 2–8 LLM calls per episode depending on scenario complexity. Observed cost range: **$0.05–$0.13** per episode, median **~$0.08**.

## Daily Cost Projections

Designated model: `GLM-5` via OpenAI-compatible API. Cost per episode ≈ **$0.08** (observed median).

**All validators must use the designated model.** This is a consensus requirement: if validators use different models, agents produce different tool-call sequences, leading to different rubric outcomes and validator disagreement on scores. Using the wrong model puts your validator out of consensus and risks down-weighting by Yuma.

### Worst-case: all miners re-evaluated every eval_interval

| Active Miners | Episodes/day | Daily Cost | Monthly Cost |
|:-------------:|:------------:|:----------:|:------------:|
| 5 | 25 | **$2** | **$60** |
| 14 | 70 | **$6** | **$168** |
| 30 | 150 | **$12** | **$360** |
| 64 | 320 | **$26** | **$768** |
| 128 | 640 | **$51** | **$1,536** |
| 256 | 1,280 | **$102** | **$3,072** |

**Worst-case formula**: `daily_cost ≈ miners × 1 eval × 5 episodes × $0.08 = miners × $0.40/day`.

In practice, only miners whose `pack_hash` changed are re-evaluated immediately. Stable packs are re-evaluated once per eval_interval (24h) for EMA accumulation.

## Miner Cost Model

| Cost Item | Estimate |
|-----------|----------|
| Policy iteration (prompt tuning) | Engineer time only |
| Local testing via ClawBench | ~$0.08/episode × ~50 test runs ≈ **$4/iteration** |
| GitHub repo hosting | Free |
| Bittensor registration | ~200 TAO (one-time) |
| **Ongoing operational cost** | **~$0/month** |

## Cost Reduction Levers

1. **Rate limiting** (built-in): At most 1 eval per miner per eval_interval (24h), regardless of how often the miner updates their commitment. Prevents API budget drain from rapid submissions
2. **EMA convergence**: Once a pack's EMA scores stabilize, re-evaluation adds diminishing value. Future optimization: skip re-eval when EMA variance is below threshold
3. **Prompt caching**: Anthropic prompt caching saves ~80% on input tokens (fixture data is identical across runs for the same scenario)

## Sustainability

Validator economics depend on alpha earnings (convertible to TAO) exceeding LLM costs.

Validators earn **subnet alpha**, not TAO directly. Alpha can be swapped for TAO via the subnet's liquidity pool at a market-determined rate. Current SN11 alpha price: ~$2.64 (≈0.015 TAO at ~$180/TAO).

```
Estimated alpha earnings (medium stake ~5k TAO, ~10% validator weight):
  ~295 alpha/day ≈ 4 TAO-equivalent at current pool rate ≈ $720/day

Example (30 miners, worst-case all re-evaluated every interval):
  Daily costs:   30 × $0.40 = $12/day
  Daily revenue: ~$720/day (alpha, at current pool rate)
  Net profit:    ~$708/day (~98% margin)

Example (30 miners, typical day):
  Daily costs:   ~$5-10/day (most packs unchanged)
  Daily revenue: ~$720/day
  Net profit:    ~$710-715/day
```

**At current rates**, TrajectoryRL validators are highly profitable:

| Scenario | Daily Cost (worst-case) | Daily Revenue (~$720 alpha) | Monthly Profit |
|----------|:----------:|:---------------------------:|:--------------:|
| 30 miners | $12 | $720 | **$21,240** |
| 64 miners | $26 | $720 | **$20,820** |
| 128 miners | $51 | $720 | **$20,070** |
| 256 miners | $102 | $720 | **$18,540** |

Even at 256 miners (worst case, all re-evaluated every eval_interval), LLM costs are only **~14%** of validator alpha revenue.

**Break-even analysis**: At 256 miners ($102/day worst-case cost), validators are extremely profitable at current alpha rates. The alpha-TAO pool rate would need to drop ~86% before validators become unprofitable. Note: these figures fluctuate with pool exchange rates and subnet demand.

## Weight Setting

Each validator sets weights independently based on its own evaluation data. There is no shared score repo or off-chain consensus mechanism.

Every tempo (~72 min), the validator:
1. Checks qualification: each miner must pass all safety and correctness checks across all scenarios
2. Ranks qualified miners by cost (lowest $/episode wins), using EMA-smoothed costs
3. Applies first-mover protection: a challenger must be ≥10% cheaper than the incumbent to dethrone
4. Maps miner hotkeys to UIDs via the current metagraph
5. Sets weights on-chain via commit-reveal

Cross-validator consensus is handled entirely on-chain by **YC3 (Yuma Consensus 3)** with **Liquid Alpha**, which dynamically adjusts per-bond learning rates based on how well validators agree.

For full details on qualification gate, cost ranking, and YC3 configuration, see [INCENTIVE_MECHANISM.md — Validator Consensus](INCENTIVE_MECHANISM.md#validator-consensus).

---

## Startup Options

### `EVAL_ON_STARTUP`

Triggers one full evaluation cycle immediately when the validator starts, without waiting for the next scheduled UTC 00:00 window. Useful after a restart or manual intervention when you don't want to wait until midnight.

```bash
docker compose -f docker/docker-compose.validator.yml --env-file .env.validator \
  up -d -e EVAL_ON_STARTUP=1 validator
```

After the startup eval completes, the validator resumes its normal daily schedule (UTC 00:00). If the startup eval finishes before midnight, the daily eval will still fire at 00:00 as usual.

---

## Automatic Updates

The validator image is hosted on GHCR (GitHub Container Registry). Watchtower runs alongside the validator and automatically pulls new images when updates are pushed to `prod`, typically within 5 minutes.

No manual action is needed — Watchtower handles image pull, container restart, and cleanup.

### Verifying auto-update

```bash
# Check Watchtower logs
docker compose -f docker/docker-compose.validator.yml logs watchtower

# Check current validator image
docker inspect trajectoryrl_validator --format '{{.Image}}'
```

### Manual update (if not using Watchtower)

```bash
docker compose -f docker/docker-compose.validator.yml --env-file .env.validator pull validator
docker compose -f docker/docker-compose.validator.yml --env-file .env.validator up -d validator
```
