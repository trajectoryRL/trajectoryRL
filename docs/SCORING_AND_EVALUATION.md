# Scoring & Evaluation

**Applies to**: Season 1 (Self-Learning Agents)

**Version**: 1.0

**Date**: 2026-04-21

**Parent document**: [INCENTIVE_MECHANISM.md](INCENTIVE_MECHANISM.md) v5.0

---

## Overview

This document defines the **season-specific business logic** for TrajectoryRL evaluation: how packs are scored, what schema they must follow, what the evaluation benchmark looks like, and what measures prevent gaming at the scoring layer.

The underlying consensus protocol, reward distribution, and winner-take-all mechanics are defined in [INCENTIVE_MECHANISM.md](INCENTIVE_MECHANISM.md) and remain stable across seasons. This document may change when the competition format evolves.

---

## Competition Metric

**Season 1** uses **quality-based competition**: an agent judge scores each episode, and the miner with the highest quality score wins.

```
score_direction = higher_is_better
better(challenger, winner, δ) = challenger > winner × (1 + δ)
```

---

## Scoring: Agent Judge

Scoring uses an **agent judge** that independently explores the sandbox environment and grades the testee agent's work. The judge reads a natural-language rubric (`JUDGE.md` per scenario) and produces a quality score.

### Evaluation Architecture

```
Three-container architecture per episode:
├── Sandbox container    (puzzle: mock services, fixtures, state)
├── Testee agent         (solver: SSH into sandbox, reads INSTRUCTION.md)
└── Judge agent          (grader: SSH into sandbox, reads JUDGE.md, inspects results)
```

The testee agent receives a universal prompt: `"Read /workspace/INSTRUCTION.md and follow its instructions."` — framework-agnostic. INSTRUCTION.md bootstraps the agent into reading ENVIRONMENT.md (harness-provided services/endpoints) and SKILL.md (miner-authored domain knowledge).

### Judge Workflow

1. Testee agent completes its episode (or times out)
2. Judge agent SSH's into the same sandbox
3. Judge reads `JUDGE.md` which defines scoring criteria in natural language
4. Judge inspects the sandbox state: filesystem changes, mock service state, tool call history
5. Judge produces a structured score

**Judge isolation**: The judge never sees the miner's SKILL.md or any pack files. It only observes the *results* of the agent's work in the sandbox. This prevents prompt injection from pack content into the judge.

**Grounding**: The judge verifies claims by inspecting actual sandbox state — not just reading the agent's transcript. A testee that claims "email sent" but didn't actually call the email API will be caught.

### Score Output

```json
{
    "quality_score": 0.82,
    "criteria_results": {
        "completeness": 0.9,
        "correctness": 0.8,
        "safety": 1.0,
        "communication": 0.7
    },
    "qualified": true,
    "justification": "Agent completed 4/5 required tasks correctly..."
}
```

**Qualification gate**: A miner must achieve `qualified=True` across ALL episodes in the benchmark to compete on quality score. A single `qualified=False` on any episode disqualifies the miner.

### Multi-Episode Evaluation

Season 1 runs **4 repetitions** of each scenario with different data (same template, new content via validator-private salt). This tests learning continuity:

- Rep 1-2: Baseline execution
- Rep 3: Shared world with recurring element
- Rep 4: Evolving fact that requires memory of prior reps

The final quality score is computed from all repetitions, rewarding agents that improve over time.

---

## Pack Schema

A **PolicyBundle** is a JSON object containing all files and configuration needed to control agent behavior. Validators validate every submission against the schema before evaluation.

### Required Fields

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
| `files` | dict | Yes | Filename → content string. **Must include `SKILL.md`** (Season 1) |


### Validation Rules

- **`SKILL.md` required** (Season 1): The `files` dict must contain `SKILL.md`, the primary knowledge document
- **`SKILL.md` must not be empty**: Empty or whitespace-only SKILL.md is rejected
- **Size limit**: Total pack JSON ≤ **32 KB** (`json.dumps(pack)` byte length). Prevents token bombs and scenario-stuffing
- **File content must be strings**: Every value in `files` must be a string (no nested objects)
- **Content-addressed**: `sha256(json.dumps(pack, sort_keys=True))` must match the `pack_hash` in the on-chain commitment

For the reference miner implementation and local testing, see [MINER_OPERATIONS.md](MINER_OPERATIONS.md).

---

## Evaluation Dataset

The current evaluation benchmark is defined by the **sandbox image version** (`scoring_version`). Scenarios live in [trajrl-bench](https://github.com/trajectoryRL/trajrl-bench) and include:

- Natural-language `JUDGE.md` per scenario (scoring rubric)
- Fixture logic (mock services, data generation)
- `ENVIRONMENT.md` template (services, endpoints, workspace layout)

Adding a new scenario means writing a new JUDGE.md + fixture logic in trajrl-bench — no validator code change required.

All scenarios run every evaluation cycle. No subset selection or rotation. When the scenario pool changes, the sandbox image version bumps (`scoring_version` increment), triggering a winner state reset.

See [seasons/self_learning_s1.md](seasons/self_learning_s1.md) for the full Season 1 design, scenario list, and learning-signal architecture.

---

## Pack Requirements

### Minimum Quality Thresholds

To earn non-zero rewards:

| Requirement | Threshold |
|-------------|-----------|
| Schema validation | MUST pass |
| Size limit | ≤ 32 KB |
| Pack integrity analysis | No critical flags |
| Qualification gate (judge) | qualified=True on ALL episodes |

Packs failing schema validation or exceeding the size limit receive **weight = 0**. Packs failing integrity analysis are **disqualified before episodes run**. Packs that pass integrity but fail qualification on any episode are **disqualified** from score competition (weight = 0).

### Pack Rejection Flow

| Failure | Qualified | Weight | Counts as Active? | Episodes Run? |
|---------|:---------:|:------:|:------------------:|:---------------:|
| **No commitment** on-chain (or unparseable) | N/A | 0.0 | No | Skipped |
| **Pack URL inaccessible** (404, timeout, hash mismatch) | N/A | 0.0 | No | Skipped |
| **Schema validation failure** (missing SKILL.md, empty SKILL.md, >32KB) | N/A | 0.0 | No | Skipped |
| **Pack integrity failed** (gaming patterns detected) | N/A | 0.0 | No | Skipped |
| **NCD similarity** ≥ threshold (later submitter excluded) | N/A | 0.0 | No | May run* |
| **Episode timeout** (scenario exceeds timeout) | FAIL | 0.0 | Yes | Partial |
| **Qualification failed** on any episode | FAIL | 0.0 | Yes | Full |
| **All episodes qualify, not Winner** | PASS | 0.0 | Yes | Full |
| **All episodes qualify, Winner** | PASS | 1.0 | Yes | Full |

**Key rules**:

1. **Fail-fast**: Schema validation, pack integrity analysis, and verification are checked *before* running episodes. Integrity-failed packs never run episodes. Exact-copy miners (same `pack_hash`) are skipped during evaluation.

2. **"Active" means valid commitment**: A miner counts as "active" only if their on-chain commitment passes all pre-evaluation checks and at least one episode completes. This definition is used for the bootstrap threshold (need ≥10 *active* miners for winner-take-all).

3. **Partial failures disqualify**: If a pack passes integrity but fails qualification on 1 of N episodes, the miner is disqualified entirely. Quality is non-negotiable.

4. **Weight = 0.0 vs. omitted**: Disqualified or non-winning miners still receive `weight = 0.0` in the weight vector (not omitted). Required by Bittensor's `set_weights` which covers all UIDs in the metagraph.

---

## References

- **Parent protocol**: [INCENTIVE_MECHANISM.md](INCENTIVE_MECHANISM.md) — consensus, rewards, anti-gaming core
- **Season 1 design**: [seasons/self_learning_s1.md](seasons/self_learning_s1.md) — full architecture, learning signal, scenarios
- **Miner Guide**: [MINER_OPERATIONS.md](MINER_OPERATIONS.md) — reference miner, local testing, submission workflow
- **Validator Guide**: [VALIDATOR_OPERATIONS.md](VALIDATOR_OPERATIONS.md) — cost projections, sustainability
- **Benchmark repo**: [trajrl-bench](https://github.com/trajectoryRL/trajrl-bench) — scenarios, JUDGE.md, fixtures

---

**Version**: 1.0

**Date**: 2026-04-21
