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

Default model: Claude Sonnet 4.5 ($3/M input, $15/M output). Cost per episode ≈ **$0.020**.

| Active Miners | Episodes/day | Daily Cost | Monthly Cost |
|:-------------:|:------------:|:----------:|:------------:|
| 5 | 60 | **$1** | **$36** |
| 14 | 168 | **$3** | **$101** |
| 30 | 360 | **$7** | **$216** |
| 64 | 768 | **$15** | **$461** |
| 128 | 1,536 | **$31** | **$922** |
| 256 | 3,072 | **$61** | **$1,843** |

**Formula**: `daily_cost ≈ miners × $0.24/day` (at Sonnet pricing).

## Model Alternatives

Validators can use cheaper models to reduce costs. Scoring is regex-based (no LLM judge), so model choice only affects the agent's task execution quality — not scoring fidelity.

| Model | Input $/M | Output $/M | Cost/episode | 30 miners/day | 128 miners/day |
|-------|:---------:|:----------:|:------------:|:--------------:|:--------------:|
| Claude Sonnet 4.5 | $3 | $15 | $0.020 | $7 | $31 |
| Claude Haiku 4.5 | $0.80 | $4 | $0.005 | $2 | $8 |
| Local (Llama 3.3) | $0 | $0 | ~$0* | ~$0 | ~$0 |

*Local models: hardware cost instead (~$1.50/hr for A100, handles ~100 episodes/hr).

## Miner Cost Model

| Cost Item | Estimate |
|-----------|----------|
| Policy iteration (prompt tuning) | Engineer time only |
| Local testing via ClawBench | ~$0.02/episode × ~50 test runs ≈ **$1/iteration** |
| GitHub repo hosting | Free |
| Bittensor registration | ~200 TAO (one-time) |
| **Ongoing operational cost** | **~$0/month** |

## Cost Reduction Levers

1. **Cheaper model** — Haiku 4.5 cuts costs **4x** with likely acceptable evaluation fidelity
2. **Prompt caching** — Anthropic prompt caching saves ~80% on input tokens (fixture data is identical across seeds for the same scenario)
3. **Fewer seeds** — `seeds_per_task=1` instead of 3 → **3x** cheaper (but weaker consensus)
4. **Fewer scenarios** — 2 per epoch instead of 4 → **2x** cheaper
5. **Skip unchanged packs** — Don't re-evaluate miners whose pack hash hasn't changed since last epoch

## Sustainability

Validator economics depend on TAO emissions exceeding LLM costs:

```
Break-even: daily_TAO_earnings × TAO_price > miners × $0.24/day

Example (30 miners, ~6 TAO/day validator earnings):
  Break-even TAO price = (30 × $0.24) / 6 = $1.20/TAO
```

With 24h epochs, Sonnet 4.5 remains economically viable even at scale. At 128 miners and ~10 TAO/day validator earnings, break-even TAO price is only $3.07/TAO.
