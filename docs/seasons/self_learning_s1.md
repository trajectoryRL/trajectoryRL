# Season 1: Self-Learning Agents

> **v0.30 (2026-05-14): Architecture pivot from agent-judge to shell_verifier.** The 3-container (sandbox + testee + judge), SSH-bounded, mock-services design described in pre-v0.30 revisions of this doc was deprecated 2026-05-03 and removed 2026-05-11. Current architecture: one **unified sandbox-agent container** per scenario, Terminal-Bench-style scenarios scored by `tests/test.sh` → `ctrf.json`. No mock services, no agent judge, no SSH boundary, no per-rep fixture variation. See `project_legacy_scenarios_deprecated.md` in TrajOS for the rollback rationale.

---

## Document Positioning

This is the **Season 1 Design Document** — the architectural reference for TrajectoryRL's first competition season. It explains how the evaluation system works: scenario format, sandbox architecture, scoring, versioning, anti-gaming, and the bindings into the core incentive protocol.

**Related documents:**

| Document | Audience | Content |
|----------|----------|---------|
| [EVALUATION_S1.md](../EVALUATION_S1.md) | Developers / validators | Concise spec: pack schema, scoring method, rejection flow |
| [MINER_GUIDE.md](../MINER_GUIDE.md) | Miners | SKILL.md writing guide, anti-gaming rules |
| [MINER_OPERATIONS.md](../MINER_OPERATIONS.md) | Miners | CLI toolbox reference |
| [INCENTIVE_MECHANISM.md](../INCENTIVE_MECHANISM.md) | All | Season-agnostic core: consensus, WTA, winner protection |

---

## Design Principles

The eval produces a single signal per miner per session: **a continuous quality score per scenario, summed across a fixed scenario set.** No learning bonus, no within-scenario delta, no judge LLM.

**Key properties:**

1. **One container per scenario.** The scenario image bundles the puzzle, the agent runtime (Hermes), and the verifier. The validator runs the agent *inside* the scenario container and then runs a fresh verifier container of the same image against the agent's output. Adding a scenario = new Dockerfile + `tests/test.sh`; no validator code change beyond appending to `SANDBOX_SCENARIOS`.

2. **Programmatic verification.** Each scenario ships `tests/test.sh` that exits 0/1 and writes a pytest-json-ctrf-style `ctrf.json`. Quality = passed / total ∈ [0, 1]. No LLM judge variance, no prose-bias attack surface.

3. **Framework-agnostic testee.** The agent only needs to be docker-execable and able to read `/workspace/SKILL.md` + `/workspace/INSTRUCTION.md`. Hermes is the default; any agent that fits the contract works.

4. **Resistant to gaming.** Tests are hidden (run in a fresh container the agent never sees), the scenario set rotates over time, and the pre-eval filter rejects packs that hardcode scenario playbooks. Programmatic scoring eliminates the prose-bias / metaphor-essay class of attacks that plagued the agent-judge era.

5. **Scenarios are public, third-party where possible.** Most scenarios are adopted from [Terminal-Bench](https://github.com/laude-institute/terminal-bench) (89-task pool, third-party-vetted). Provenance + licensing tracked in `NOTICE` + per-scenario `DESIGN.md`. See the adoption playbook in TrajOS `feedback_third_party_license_playbook.md`.

---

## Sandbox Architecture

Each scenario image **layers on the `sandbox-agent` base** (Hermes + non-root user `hermes` uid 10000 + `/workspace` + `/app` perms + the `trajrl_bench` CLI):

```
                     ┌────────────────────────────────────────┐
                     │ ghcr.io/trajectoryrl/sandbox-agent:tag │   base image
                     │  • Hermes runtime + agent user         │
                     │  • /workspace + /app                   │
                     │  • CLI: scenarios, scenario-info       │
                     └────────────────────────────────────────┘
                                       ↑ FROM
                              ┌────────┼────────┐
   ┌──────────────────────────────┐  ┌──────────────────────────────┐  ┌──────────────────────────────┐
   │ scenario-cancel-async-tasks  │  │ scenario-log-summary-...     │  │ scenario-break-filter-...    │
   │ + scenario deps              │  │ + log generator output       │  │ + chromium + selenium        │
   └──────────────────────────────┘  └──────────────────────────────┘  └──────────────────────────────┘
```

Per session, the validator iterates `SANDBOX_SCENARIOS` and, for each scenario:

1. `docker run scenario-<name>:<tag>` — container starts with `tail -f /dev/null`.
2. Drop `SKILL.md` + `INSTRUCTION.md` into `/workspace`.
3. `docker exec -u hermes` runs `hermes chat -q "<prompt>" -m <model> --quiet --yolo --max-turns 30`.
4. Extract `agent_output_path` (e.g. `/app/run.py`, `/app/setup.sh`, `/app/recovered.json`).
5. Run a **fresh verifier container** of the same image with the agent's output injected; parse `passed/total` from `ctrf.json`.
6. Stop + remove both containers.

**No mock services.** No FastAPI, no Slack/Notion/Gitea, no SQLite state store, no `/state` endpoint. The scenario is whatever filesystem + tooling the Dockerfile builds in.

**No SSH boundary.** The agent runs *inside* the scenario container as user `hermes`. Hermes's built-in `terminal` / `file` / `execute_code` tools operate directly on `/workspace` and `/app`.

**No fixture randomization per validator.** Scenarios are deterministic. Cross-validator score variation comes from testee LLM non-determinism (sampling, timeouts), not from different inputs. Stake-weighted aggregation across validators absorbs the noise.

**Privilege boundary.** `tests/test.sh` is bundled in the scenario image but never exposed to the agent — it only runs in the fresh verifier container the validator spawns after the agent exits.

---

## Pack Format

Miners submit a `pack.json` containing their `SKILL.md`. For pack schema, validation, and size limits, see [EVALUATION_S1.md](../EVALUATION_S1.md#pack-schema). For writing guidance, see [MINER_GUIDE.md](../MINER_GUIDE.md).

**SKILL.md** is a plain markdown document the validator drops at `/workspace/SKILL.md` (read-only to the agent). It contains the miner's product: judgment, process, safety rules, tactics. The harness prompt instructs the agent to consult `SKILL.md` and the per-scenario `INSTRUCTION.md` before acting.

**Properties:**
- **Static.** SKILL.md never changes during evaluation. Same file used for every scenario in the session.
- **Scenario-agnostic encouraged.** The pre-eval filter rejects packs that name specific scenarios or hardcode benchmark-specific playbooks. The winning strategy is general instruction quality, not scenario-by-scenario rules.
- **Framework-agnostic.** Any agent that reads `/workspace/SKILL.md` and runs shell commands can use it.

Each scenario also ships a static `INSTRUCTION.md` (the task statement) which the validator copies into `/workspace/INSTRUCTION.md` alongside SKILL.md.

---

## Scoring

For a concise spec view, see [EVALUATION_S1.md](../EVALUATION_S1.md#scoring).

**Per scenario:** `quality = passed / total` from `ctrf.json` ∈ [0, 1]. Binary fallback `float(reward.txt)` when `ctrf.json` is missing or unparseable.

**Per session:** `final_score = sum(per_scenario_quality)` ∈ [0, N] for N scenarios. `mean_quality = final_score / N` ∈ [0, 1] is reported as the convenience aggregate; consensus uses `final_score`.

**Cost** (USD, from `turns.jsonl` exported by Hermes) is reported alongside the score but **never folded into the score** — it's a separate axis on the leaderboard.

No split-half delta, no learning bonus, no early-mean floor. With one episode per scenario the within-scenario delta concept doesn't apply.

---

## Current Scenarios

`SANDBOX_SCENARIOS` (in `trajectoryrl/utils/sandbox_harness.py`), alphabetical order:

| Scenario | Category | Difficulty | Verifier output |
|---|---|---|---|
| `break-filter-js-from-html` | security | medium | `/app/out.html` |
| `cancel-async-tasks` | async/concurrency | hard | `/app/run.py` |
| `configure-git-webserver` | sysadmin / git | hard | `/app/setup.sh` |
| `db-wal-recovery` | file ops / database | medium | `/app/recovered.json` |
| `fix-git` | git / version control | easy | `/app/recovery.sh` |
| `log-summary-date-ranges` | data processing | medium | (runs in `/app`) |
| `nginx-request-logging` | sysadmin / web server | medium | `/app/setup.sh` |
| `path-tracing` | graphics / C | hard | `/app/image.c` |
| `vulnerable-secret` | security / binary RE | medium | `/app/results.txt` |

Per-scenario sources live at `scenarios/<name>/` in [trajrl-bench](https://github.com/trajectoryRL/trajrl-bench):

```
scenarios/<name>/
  task.toml             # metadata, verifier timeout, agent_output_path, docker_image
  instruction.md        # static task statement
  environment/
    Dockerfile          # FROM sandbox-agent + scenario deps
  tests/
    test.sh             # verifier — exits 0/1, writes reward.txt + ctrf.json
    test_outputs.py     # pytest assertions
  solution/
    solve.sh            # reference solution (not used at eval time)
  DESIGN.md             # provenance + license (when adopted from elsewhere)
```

Scenarios from Terminal-Bench retain upstream attribution in `NOTICE` and `THIRD_PARTY_LICENSES`. Adoption rules: TrajOS `feedback_third_party_license_playbook.md`.

`swe-bench-astropy-2` lives in the bench repo but is **not** in the validator's active list yet.

---

## Evaluation Flow

```
1. Validator pulls scenario images (per-cycle, idempotent — PR #252).
2. Validator queries scenario metadata:
   docker run sandbox-agent python -m trajrl_bench.cli scenarios
   docker run sandbox-agent python -m trajrl_bench.cli scenario-info --scenario X
3. For each scenario in SANDBOX_SCENARIOS:
   a. docker run scenario-<name>:<tag>  (container CMD = tail -f /dev/null)
   b. Drop SKILL.md + INSTRUCTION.md into /workspace.
   c. docker exec -u hermes hermes chat -q "<universal prompt>" -m <model>
      --quiet --yolo --max-turns 30
   d. On agent exit (or timeout), extract agent_output_path from the container.
   e. docker run scenario-<name>:<tag> /tests/test.sh  — fresh verifier container
      with the agent's output injected.
   f. Read /reward.txt + /ctrf.json from the verifier container.
   g. quality = passed/total (ctrf) or float(reward) (fallback).
   h. Stop + remove both containers.
4. final_score = sum(qualities), mean_quality = sum/N.
5. Publish (validator_hotkey, final_score, per_scenario_qualities, ctrf payloads,
   transcripts) to dashboard; consensus aggregates stake-weighted final_score.
```

A full 9-scenario session typically takes ~30-45 min wall-clock (LLM latency-bound). Validator capacity at this cadence is the rate-limiter on eval cycles per epoch.

---

## What the Validator Sees

```json
{
  "miner_uid": 42,
  "pack_hash": "abc123...",
  "spec_number": 11,
  "mean_quality": 0.611,
  "final_score": 5.5,
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
  "cost_usd": 0.42
}
```

Full per-episode artifacts (SKILL.md, INSTRUCTION.md, `turns.jsonl` transcript, `ctrf.json`, `reward.txt`) are tarred and uploaded to the dashboard. Retrievable via:

```
trajrl logs --eval-id <id> --show
trajrl logs --eval-id <id> --dump-to <dir>
```

See [MINER_OPERATIONS.md § Viewing Evaluation Results](../MINER_OPERATIONS.md#viewing-evaluation-results) for the archive layout.

---

## Score → Weights

Quality flows into the consensus protocol as a higher-is-better score. The consensus pipeline (stake-weighted aggregation, winner protection with δ=10%, WTA weight setting) is defined in [INCENTIVE_MECHANISM.md](../INCENTIVE_MECHANISM.md#winner-take-all-with-winner-protection).

---

## Anti-Gaming

| Mechanism | What it prevents |
|---|---|
| Hidden tests (`tests/test.sh` runs only in a fresh verifier container the agent never sees) | Agent pattern-matching on visible criteria |
| Programmatic verification (`ctrf.json`) | Prose-bias, metaphor-essay packs, judge-LLM hallucination |
| Scenario rotation + new-scenario additions, communicated publicly | Overfitting to a fixed scenario set |
| Pre-eval filter (rejects packs naming specific scenarios or mirroring rubric criteria) | Hardcoded benchmark playbooks |
| Adopted scenarios from Terminal-Bench (large third-party pool) | Validator-author bias in scenario design |
| Cross-validator stake-weighted aggregation | Per-validator non-determinism / outliers |

For miner-facing anti-gaming rules, see [MINER_GUIDE.md § Pre-Eval Compliance](../MINER_GUIDE.md#pre-eval-compliance-anti-gaming).

---

## Incentive Mechanism: Season 1 Bindings

Season 1 implements [INCENTIVE_MECHANISM.md](../INCENTIVE_MECHANISM.md) with the following season-specific bindings. WTA, NCD, consensus, winner protection are inherited from the core protocol.

| Component | Season 1 |
|-----------|----------|
| **Score direction** | Higher is better (`challenger_score > winner_score × (1 + δ)`) |
| **Score source** | `consensus_score` = stake-weighted average of validator `final_score` |
| **Qualification gate** | `final_score > 0` |
| **Pack format** | SKILL.md only (one pack per contest) |
| **NCD target** | SKILL.md content (threshold 0.80) |
| **Default testee harness** | [Hermes Agent](https://github.com/NousResearch/hermes-agent) 0.13.x — built-in `terminal` / `file` / `execute_code` tools |
| **Default testee LLM** | OpenRouter `qwen/qwen3.5-35b-a3b` (per-validator configurable via `LLM_*` env) |
| **Winner protection δ** | 10% |
| **Inactivity** | 14400 blocks |

Harness + LLM identity per eval is reported to the dashboard (bench v4.0.6+) — see TrajOS `project_harness_llm_identity.md`.

---

## Versioning

Score comparability across validators is governed by `SPEC_NUMBER` — a manually maintained constant in `trajectoryrl/utils/config.py`, decoupled from the `trajrl-bench` image version. Aggregation derives the active `spec_number` from on-chain stake distribution: the stake-weighted dominant value wins if it holds >50% stake; otherwise validators fall back to their local `SPEC_NUMBER`. See [INCENTIVE_MECHANISM.md](../INCENTIVE_MECHANISM.md#spec_number-and-target-spec-selection).

**Current**: `SPEC_NUMBER = 11` (bumped in v0.6.15 when `configure-git-webserver` was added; the previous bump 9→10 was for the Hermes 0.13 cutover).

| Change | Bench image bump | `SPEC_NUMBER` bump? |
|--------|-------------------|---------------------|
| New scenario added to `SANDBOX_SCENARIOS` | Minor | **Yes** — score range changes from N to N+1 |
| Scenario `test.sh` / `instruction.md` / `Dockerfile` changed | Minor | **Yes** — rerun, cached scores invalidate |
| Harness runtime upgrade (Hermes major) | Minor or major bench bump | **Yes** — testee behavior changes |
| Bench-infra bug fix preserving scoring | Patch | No |
| Validator-side refactor preserving scoring | n/a | No |

Bench-image-only releases hot-propagate to validators on their next eval cycle (PR #252, per-cycle pull). Validator-code releases require a `git tag v*` push and watchtower restart — see TrajOS `feedback_validator_release_image_pull.md` for timing rules.

---

## Known Risks

### 1. Scenario pool size

Nine active scenarios is small enough that a sufficiently-tuned SKILL.md can plausibly target each one. Mitigation: scenario rotation + periodic additions (public changelog), pre-eval filter, eventual growth to 10–15+ scenarios over the season. See TrajOS `project_tb_scenario_diversity_thesis.md`.

### 2. Validator non-determinism

Same SKILL.md → different per-scenario qualities across validators because the testee LLM is non-deterministic (sampling, timeouts, occasional silent agent stalls). Mitigation: cross-validator stake-weighted aggregation absorbs the noise. Operator-level: validators report harness/LLM identity per eval so consumers can read consensus with awareness of the testee mix.

### 3. Evaluation cost and time

Per miner per session: ~30-45 min wall-clock, dominated by LLM latency (Qwen3.5-A3B at ~10-30s per turn × ~30 turns × 9 scenarios). Validators bear all inference cost (testee only — no judge). At 200 miners × stable epoch length, full eval cycles take several hours; some validators stagger or skip epochs when behind.

### 4. The "already good" problem

A SKILL.md that produces high quality from the first attempt has no within-session room to grow — but there is no learning bonus in current scoring, so this is *fine*. Quality dominates by design. Miners compete on getting as close to `final_score = N` as possible.

### 5. Miner meta-game evolution

Differentiation comes from SKILL.md instruction quality, not model choice. Top miners discover this within weeks. Patterns observed: lean generalist (UID 25, 224 in 2026-04-17 first eval) beat scripted playbooks; tight-directive SKILL.md lifts Qwen3.5-A3B from a satisficing-0 floor to near-ceiling on scenarios it has the capability for. See TrajOS `project_qwen35_skill_design_findings.md` and `feedback_qwen35_satisficing_floor.md`.

---

## §6a: Prior Art — ClawsBench

[ClawsBench](https://clawsbench.benchflow.ai/) (arXiv 2604.05172, April 2026) evaluates LLM productivity agents across 5 mock services, 44 tasks, 6 models, 4 harnesses, 7,224 trials. Its headline finding — **SKILL.md-style scaffolding adds +39–63 percentage points** to task success rate, and after scaffolding the top 5 models are statistically indistinguishable — directly validates the Season 1 thesis: **miners compete on instruction quality, not model selection.**

Season 1 originally adopted ClawsBench's mock-services + agent-judge architecture (pre-v0.30 of this doc). After the 2026-05 architecture pivot, the surface similarity is gone — current S1 is closer to [Terminal-Bench](https://github.com/laude-institute/terminal-bench) than to ClawsBench — but the underlying finding (scaffolding > model) still motivates the SKILL.md-as-product framing.

**References:** [ClawsBench](https://clawsbench.benchflow.ai/) (arXiv 2604.05172), [Terminal-Bench](https://github.com/laude-institute/terminal-bench).

---

## Minimal Viable Implementation

Four components, all live today:

1. **Sandbox base** (`trajrl-bench/docker/Dockerfile.sandbox-agent`): Hermes runtime, non-root `hermes` user, `/workspace` + `/app` perms, `trajrl_bench` CLI.
2. **Scenario images** (`trajrl-bench/scenarios/<name>/environment/Dockerfile`): `FROM sandbox-agent` + scenario-specific deps + `tests/`.
3. **Validator harness** (`trajectoryrl/utils/sandbox_harness.py`): orchestrates `docker run` + `docker exec hermes` + verifier container per scenario.
4. **CLI probes** (`trajrl_bench.cli`): `scenarios` (lists names) and `scenario-info --scenario X` (returns image repo, instruction.md, agent_output_path, verifier_timeout, base64 `tests/`).

Current versions: `trajrl-bench` v4.0.6+, `trajectoryRL` v0.6.15, `SPEC_NUMBER` 11. The competition is on SKILL.md instruction quality and the scenario set's diversity.
