# Scoring & Evaluation

**Applies to**: Season 1 (Self-Learning Agents)

**Version**: 2.0

**Date**: 2026-05-14

**Parent document**: [INCENTIVE_MECHANISM.md](INCENTIVE_MECHANISM.md)

**Full design document**: [seasons/self_learning_s1.md](seasons/self_learning_s1.md) — architecture, scoring algorithm, scenario set, versioning, anti-gaming

---

## Overview

This is the **concise spec** for Season 1's evaluation business logic: how packs are scored, what schema they must follow, what the evaluation benchmark looks like, and which pre-eval / runtime checks gate weight assignment. For the full design rationale, sandbox architecture, and scenario list, see the [Season 1 Design Document](seasons/self_learning_s1.md).

The underlying consensus protocol, reward distribution, and winner-take-all mechanics are defined in [INCENTIVE_MECHANISM.md](INCENTIVE_MECHANISM.md) and remain stable across seasons. This document may change when the competition format evolves.

> **2026-05 architecture pivot.** Season 1 v1.0 of this doc described a 3-container (sandbox + testee + judge) agent-judge architecture with mock services. That design was deprecated 2026-05-03 and removed 2026-05-11. Current evaluation is one **unified scenario container** per scenario, scored by a fresh verifier container running `tests/test.sh` → `ctrf.json`. No agent judge, no mock services, no per-rep fixture variation.

---

## Competition Metric

Season 1 uses **quality-based competition**: per-scenario `passed/total` from a programmatic verifier, summed across a fixed scenario set. Higher is better.

```
score_direction = higher_is_better
better(challenger, winner, δ) = challenger > winner × (1 + δ)
```

Winner protection: `δ = 0.10` (challenger must score at least 10% higher than the reigning winner to displace).

---

## Scoring

### Per scenario

The validator runs the testee agent inside a fresh scenario container (`FROM ghcr.io/trajectoryrl/sandbox-agent:tag` + scenario-specific deps), then runs a **fresh verifier container of the same image** with the agent's output (`agent_output_path` from `task.toml`) injected. The verifier executes `tests/test.sh`, which writes a pytest-json-ctrf-style `ctrf.json` and a binary `reward.txt`.

```
quality_scenario = passed / total          (from ctrf.json, ∈ [0, 1])
quality_scenario = float(reward.txt)        (binary fallback when ctrf is missing)
```

### Per session

A miner's session runs every scenario in `SANDBOX_SCENARIOS`. The final score is the equal-weighted sum across scenarios:

```
final_score   = Σ quality_scenario           (∈ [0, N] for N active scenarios)
mean_quality  = final_score / N              (∈ [0, 1], convenience aggregate)
```

Consensus uses `final_score`; `mean_quality` is reported alongside for human readability.

**Cost** (USD, summed from per-turn cost in Hermes's `turns.jsonl`) is reported per scenario and per session but **never folded into the score** — it's a separate axis on the leaderboard.

No learning bonus, no split-half delta, no early-mean floor. With one episode per scenario the within-scenario delta concept doesn't apply.

### Eval-result schema (per miner per session)

```json
{
  "miner_uid": 42,
  "pack_hash": "abc123…",
  "spec_number": 11,
  "final_score": 5.5,
  "mean_quality": 0.611,
  "qualified": true,
  "scenario_qualities": {
    "break-filter-js-from-html": 1.0,
    "cancel-async-tasks": 0.5,
    "configure-git-webserver": 0.5,
    "db-wal-recovery": 1.0,
    "fix-git": 1.0,
    "log-summary-date-ranges": 0.0,
    "nginx-request-logging": 0.5,
    "path-tracing": 0.0,
    "vulnerable-secret": 1.0
  },
  "scenario_costs_usd": { "...": 0.04 },
  "total_cost_usd": 0.42
}
```

**`qualified`** is the any-pass flag: `True` iff at least one scenario produced `quality > 0`. It is **not** an all-or-nothing rubric gate — a session that scored on any scenario is qualified for consensus. The all-episodes qualification gate from the v1.0 agent-judge era no longer exists.

---

## Pack Schema

A `pack.json` is a JSON object containing the files needed to evaluate the SKILL.md. Validators validate every submission against the schema before evaluation.

### Required fields

```json
{
  "schema_version": 1,
  "files": {
    "SKILL.md": "# Domain Knowledge\n..."
  }
}
```

| Field | Type | Required | Description |
|-------|------|:--------:|-------------|
| `schema_version` | int | Yes | Must be `1` |
| `files` | dict | Yes | Filename → content string. **Must include `SKILL.md`** |

### Validation rules

- **`SKILL.md` required.** The `files` dict must contain `SKILL.md`.
- **`SKILL.md` must not be empty.** Empty or whitespace-only SKILL.md is rejected.
- **Size limit**: total pack JSON ≤ **32 KB** (`json.dumps(pack)` byte length). Prevents token bombs and scenario-stuffing.
- **File content must be strings.** Every value in `files` is a string (no nested objects).
- **Content-addressed.** `sha256(json.dumps(pack, sort_keys=True))` must match the `pack_hash` in the on-chain commitment.

For the reference miner workflow and local testing, see [MINER_OPERATIONS.md](MINER_OPERATIONS.md).

---

## Evaluation Dataset

The benchmark is defined by **`SANDBOX_SCENARIOS`** in `trajectoryrl/utils/sandbox_harness.py`, paired with a manually maintained **`SPEC_NUMBER`** constant in `trajectoryrl/utils/config.py`. Each scenario is a Terminal-Bench-style task that ships at `trajectoryRL/trajrl-bench:scenarios/<name>/`:

```
scenarios/<name>/
  task.toml             # metadata, verifier_timeout, agent_output_path, docker_image
  instruction.md        # static task statement (drop into /workspace each session)
  environment/
    Dockerfile          # FROM ghcr.io/trajectoryrl/sandbox-agent:tag + scenario deps
  tests/
    test.sh             # verifier — exits 0/1, writes reward.txt + ctrf.json
    test_outputs.py     # pytest assertions
  solution/
    solve.sh            # reference solution (not used at eval time)
  DESIGN.md             # provenance + license (for adopted scenarios)
```

Adding a scenario means publishing a new scenario image to GHCR, appending the name to `SANDBOX_SCENARIOS`, and bumping `SPEC_NUMBER`. **All scenarios run every evaluation cycle.** No subset selection or rotation within a cycle; rotation happens across `SPEC_NUMBER` bumps.

Aggregation picks the target `spec_number` from on-chain commitments (stake-weighted majority), so upgrades are self-coordinating — see [INCENTIVE_MECHANISM.md § SPEC_NUMBER & target spec selection](INCENTIVE_MECHANISM.md#spec_number-and-target-spec-selection).

For the current active set, see [seasons/self_learning_s1.md § Current Scenarios](seasons/self_learning_s1.md#current-scenarios).

---

## Pack Requirements

### Minimum quality thresholds

| Requirement | Threshold |
|-------------|-----------|
| Schema validation | MUST pass |
| Size limit | ≤ 32 KB |
| Pack integrity (pre-eval filter) | No critical flags |
| Qualification gate | `final_score > 0` (≥ 1 scenario passed any test) |

Packs failing schema or size validation receive **weight = 0**. Packs failing integrity analysis are **disqualified before evaluation runs**. Packs that pass integrity but score `final_score = 0` are **non-winners** (weight = 0); they still count as active commitments.

### Pack rejection flow

| Outcome | `qualified` | Weight | Counts as active? | Scenarios run? |
|---------|:-----------:|:------:|:-----------------:|:--------------:|
| **No commitment** on-chain (or unparseable) | N/A | 0.0 | No | Skipped |
| **Pack URL inaccessible** (404, timeout, hash mismatch) | N/A | 0.0 | No | Skipped |
| **Schema validation failure** (missing/empty SKILL.md, > 32 KB) | N/A | 0.0 | No | Skipped |
| **Pack integrity failed** (gaming patterns detected) | N/A | 0.0 | No | Skipped |
| **NCD similarity ≥ 0.80** (later submitter excluded) | N/A | 0.0 | No | May run* |
| **All scenarios scored 0** | `False` | 0.0 | Yes | Full |
| **`final_score > 0`, not Winner** | `True` | 0.0 | Yes | Full |
| **`final_score > 0`, Winner (WTA + winner protection)** | `True` | 1.0 | Yes | Full |

*\* NCD copies are filtered before weight assignment; whether their session ran depends on the validator's per-cycle NCD pre-filter.*

### Key rules

1. **Fail-fast pre-eval.** Schema validation, pack integrity, and verification run *before* sandboxed evaluation. Integrity-failed packs never run scenarios. Exact-copy miners (same `pack_hash`) are skipped during evaluation.

2. **"Active" means valid commitment + completed session.** A miner counts as active only if their on-chain commitment passes pre-evaluation and at least one scenario session completes. Used for the bootstrap threshold for winner-take-all.

3. **Qualification is any-pass, not all-pass.** The all-episodes qualification gate from the v1.0 agent-judge era is gone. Any non-zero `final_score` is qualified.

4. **Weight = 0.0 vs. omitted.** Non-winning miners still appear in the weight vector with `weight = 0.0`. Required by Bittensor's `set_weights` which covers all UIDs in the metagraph.

---

## References

- **Parent protocol**: [INCENTIVE_MECHANISM.md](INCENTIVE_MECHANISM.md) — consensus, rewards, anti-gaming core
- **Season 1 design**: [seasons/self_learning_s1.md](seasons/self_learning_s1.md) — architecture, scenarios, versioning, risks
- **Miner Guide**: [MINER_GUIDE.md](MINER_GUIDE.md) — SKILL.md writing patterns, anti-gaming rules
- **Miner Operations**: [MINER_OPERATIONS.md](MINER_OPERATIONS.md) — CLI workflow, local testing, eval-log retrieval
- **Validator Operations**: [VALIDATOR_OPERATIONS.md](VALIDATOR_OPERATIONS.md) — cost projections, hosting requirements
- **Benchmark repo**: [trajrl-bench](https://github.com/trajectoryRL/trajrl-bench) — scenarios, Dockerfiles, verifier code

---

**Version**: 2.0

**Date**: 2026-05-14
