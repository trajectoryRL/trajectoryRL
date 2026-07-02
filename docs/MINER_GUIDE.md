# Season 1 Miner Guide

**Subnet**: SN11 (TrajectoryRL)
**Scoring**: Programmatic verifier, quality-based, summed across the active scenario set

> Mining means writing a **SKILL.md** — a scaffold that teaches a small open-source LLM how to solve scenario tasks. The best agent behavior wins; the artifact you ship *is* the deployed skill.

> **Architecture note (2026-05).** This guide describes the current unified-container, shell-verifier architecture. Earlier revisions of this document described a 3-container (sandbox + testee + judge) agent-judge design with mock services — that design was removed 2026-05-11. If you find old guides referencing `JUDGE.md`, `/workspace/learned/`, 4-rep split-half delta, or `localhost:8090` mock services, they are stale.

---

## How It Works

One Docker container per scenario per miner. The scenario image bundles the puzzle, the testee runtime (Hermes), and the verifier:

1. The validator runs `docker run scenario-<name>:<tag>` (CMD = `tail -f /dev/null`), drops your `SKILL.md` and the scenario's `INSTRUCTION.md` into `/workspace`, then `docker exec -u hermes` runs Hermes against the scenario.
2. Hermes uses its built-in `terminal` / `file` / `execute_code` tools to read `/workspace/SKILL.md` + `/workspace/INSTRUCTION.md` and produce a deliverable at the scenario's `agent_output_path` (e.g. `/app/run.py`, `/app/setup.sh`, `/app/recovered.json`).
3. The validator stops the agent container, then runs a **fresh verifier container of the same image** with your output injected. The verifier executes `tests/test.sh`, which writes `ctrf.json` (pytest results) and `reward.txt` (binary).
4. Your scenario score = `passed / total` from `ctrf.json`. Fallback: `float(reward.txt)` (0 or 1).

Each session runs every scenario in `SANDBOX_SCENARIOS`. Final score = sum of per-scenario qualities.

```
final_score   = Σ (passed_i / total_i)         ∈ [0, N]
mean_quality  = final_score / N                ∈ [0, 1]
```

No mock services, no SSH boundary, no per-rep fixtures, no agent judge. The agent runs inside the scenario container; the verifier scores its output programmatically.

---

## Pack Format

Your pack is a JSON file containing your `SKILL.md`. One pack = one contest.

```json
{
  "schema_version": 1,
  "files": {
    "SKILL.md": "# Your skill content here..."
  }
}
```

Season 1 requires `SKILL.md` only.

- `SKILL.md` must not be empty.
- Total pack JSON ≤ 32 KB.
- Content-addressed: `sha256(json.dumps(pack, sort_keys=True))` matches the `pack_hash` carried in your submission.

---

## Writing SKILL.md

Your `SKILL.md` is a static instruction document the agent reads before working on every scenario. The validator drops it at `/workspace/SKILL.md` (read-only). The agent has no other persistent memory — there is no `/workspace/learned/`, no cross-scenario state, no carryover between sessions.

The competition rewards SKILL.md content that lifts the **Qwen3.5-35B-A3B testee** (default LLM) above its vanilla satisficing floor across a diverse scenario set. Vanilla Qwen3.5 tends to read the task, run one or two inspection commands, then stop without producing a deliverable. A well-written SKILL.md fixes that.

### What a winning SKILL.md looks like

**1. Directive operating procedure.** Tell the agent what to produce, in what order, and where to write it. Qwen3.5-A3B does not improvise the "write the answer to the deliverable file" step reliably — the SKILL.md must spell it out.

```markdown
## Procedure
1. Read INSTRUCTION.md to identify the deliverable path and the success criterion.
2. Investigate the environment with `ls -la /app`, `cat` relevant files.
3. Draft your solution.
4. Write the deliverable to the path INSTRUCTION.md specifies. Do not stop earlier.
5. If the deliverable is a script, verify it runs without error before declaring done.
```

**2. Domain-specific tactics, not scenario-specific playbooks.** General patterns that apply to the broad class of work (e.g. "debugging async Python", "fixing git history", "configuring a webserver") help. Hardcoded answers to specific benchmark scenarios are caught by the pre-eval filter.

```markdown
## Async Python tactics
- `asyncio.gather` with `return_exceptions=False` cancels siblings on the first
  exception. To run-and-collect, wrap each coro in `asyncio.shield` or use
  `asyncio.gather(..., return_exceptions=True)`.
- When cancelling a task group, propagate `asyncio.CancelledError` after cleanup —
  swallowing it leaves the parent's cancellation in an indeterminate state.
```

**3. Disciplined verification.** Tell the agent to check its output against the success criterion before exiting. Most "0 / N" scenarios are agents that produced something but never confirmed it satisfied the task.

```markdown
## Self-check before exiting
- Re-read INSTRUCTION.md. Did you produce the exact file at the exact path it asked for?
- If the task is "fix the failing test", actually run the test and confirm it passes.
- If you can't get to a passing state, leave the best partial solution and explain
  what failed and where in a comment at the top of the deliverable.
```

**4. Tight and focused.** Long prescriptive packs cause Qwen3.5 to satisfice on instruction-following and stop before producing real work. Empirically: tight ~3–5 KB packs with verified payloads outperform long playbooks. The 32 KB schema cap is a ceiling, not a target.

### What NOT to include

- **Scenario-specific playbooks** ("for `cancel-async-tasks`, do X; for `fix-git`, do Y"). Caught by the pre-eval filter.
- **Hardcoded test names, file paths from the benchmark internals, or verifier output schemas.** The agent sees `INSTRUCTION.md`; the verifier and its tests are not exposed to it.
- **Mirroring the rubric.** There is no judge LLM and no natural-language criteria list to mirror. Scoring is pure `passed/total` — tactics that "look thorough to a judge" do nothing.
- **Cross-session memory hooks.** There is no persistent `/workspace/learned/`. SKILL.md is the only persistent context.

---

## Pre-Eval Compliance (Anti-Gaming)

> Validators run a pre-eval filter that rejects packs that look like benchmark-specific cheats. If an author could not have written your SKILL.md without knowing the exact scenario set — it gets flagged.

Write a **genuinely general SKILL.md** that guides an agent to reason dynamically. The pre-eval gate checks for these patterns:

### 1. Hardcoded benchmark answers

Do not embed solutions to specific scenarios:

- Pre-written shell scripts that solve a specific scenario verbatim
- Specific file paths from inside scenario containers
- Test names or assertions copied from a scenario's `test.sh`

**Instead:** Provide general tactics for the category of task; let the agent investigate and solve at runtime.

### 2. Scenario-name dispatch

Do not branch behavior by scenario name or task title:

```
If task mentions "git" → run this exact recovery script
If task mentions "nginx" → drop this exact config
```

**Instead:** Provide domain knowledge applicable to the broad category.

### 3. Tool avoidance

Do not disable or discourage tools needed to investigate:

- "Do not use `terminal`"
- "Limit yourself to N tool calls"
- "Avoid reading files larger than X bytes"

**Instead:** Suggest efficient tool use without forbidding necessary ones.

### 4. Per-scenario playbook collections

Do not include N scenario-specific sections matching the active scenario count:

- Red flag: exactly 12 (current scenario-count) detailed playbooks
- Red flag: per-scenario output templates with exact field lists
- Red flag: instructions that only make sense if the author knew the eval scenarios

**Instead:** One unified procedure that handles diverse tasks.

### 5. Verifier reverse-engineering

Do not embed rules that mirror specific verifier checks:

- Regex/pattern gaming targeting exact strings the verifier looks for
- Output formatting that targets known `test_outputs.py` assertions
- "Always emit this exact JSON structure" templates that match a particular scenario's expected schema

**Instead:** General quality guidance, not encoded verifier logic.

### 6. Benchmark-coverage overfitting

Do not tailor the skill to cover exactly and only the benchmark scenarios:

- Addressing only known scenarios with zero generalization beyond them
- Suspiciously precise knowledge of what the verifier grades on

**Instead:** Write a genuinely general skill that handles tasks beyond the current scenario set.

### Self-check before submitting

1. **Generality** — Does this SKILL.md still make sense for a brand-new, never-seen scenario in the same domain?
2. **Knowledge source** — Could every instruction be written without knowing the active scenario list?
3. **Tool freedom** — Can the agent still use all necessary tools?
4. **Scope** — Does the skill cover task types beyond the current benchmark?

---

## Sandbox Environment

The agent runs inside the scenario container as the non-root user `hermes` (uid 10000). It has a full Linux shell with whatever the scenario's Dockerfile installed (Python, git, curl, jq, chromium, etc. — varies by scenario).

```
/workspace/
  SKILL.md          # Your pack (read-only, dropped by validator)
  INSTRUCTION.md    # Scenario task statement (read-only, scenario-static)
/app/
  <scenario files>  # Scenario-specific working dir; agent writes deliverable here
```

The agent has **no internet** (`eval_net` is internal-only) except for the LLM API egress that Hermes itself uses to talk to the testee model. Standard Linux tools and any deps the scenario installed are available. The verifier (`tests/test.sh`) is bundled in the image but lives at a path the `hermes` user cannot read.

### Per-scenario specifics

Each scenario has its own:

- **`INSTRUCTION.md`** — the task statement. The agent must read this every session.
- **`agent_output_path`** — where the agent's deliverable goes (e.g. `/app/run.py`, `/app/setup.sh`). Defined in `task.toml`.
- **`agent_timeout_s`** — wall-clock limit for the agent. Varies per scenario (default ~900s). Defined in `task.toml`.
- **Pre-installed tooling** — Dockerfile installs whatever the task needs (selenium + chromium for HTML scenarios, postgres tools for db scenarios, etc.).

For the live active list, definitions, and source code, see [trajrl-bench](https://github.com/trajectoryRL/trajrl-bench).

---

## Scoring Recap

```
quality_scenario = passed / total            (from ctrf.json)
final_score      = Σ quality_scenario        ∈ [0, N]
mean_quality     = final_score / N           ∈ [0, 1]
```

Cost (USD from Hermes's `turns.jsonl`) is reported alongside but never folded into the score. The fastest cheapest 0-scoring agent still gets weight 0; quality dominates.

No learning bonus, no split-half delta, no early-mean floor.

`qualified = True` whenever `final_score > 0` (at least one scenario passed any tests).

---

## Submission

Use the `trajectoryrl-miner` CLI (full reference in [MINER_OPERATIONS.md](MINER_OPERATIONS.md)):

```bash
git clone https://github.com/trajectoryRL/trajectoryRL.git
cd trajectoryRL && pip install -e .

trajectoryrl-miner build SKILL.md -o pack.json

# Pay the submission fee + submit. web-submit is the only channel: the
# CLI recycles SUBMISSION_FEE_ALPHA (default 50) on chain via
# recycle_alpha, then submits the pack referencing that receipt.
trajectoryrl-miner web-submit pack.json

trajectoryrl-miner status
```

`web-submit` is the sole submission channel: the platform stores your `pack.json`, verifies the recycle receipt, runs pre-eval, and writes the challenger-queue entry. Validators then fetch your pack, verify the hash, extract `SKILL.md`, and run the full scenario session. (On-chain `set_commitment` is no longer ingested as a submission.)

---

## Submission Fee

Every submission is backed by an on-chain `recycle_alpha` burn, paid before the pack is accepted. Recycling is irreversible — it is the economic cost that deters spam and low-quality packs (it replaces the old owner-ban system).

- **Amount** — `SUBMISSION_FEE_ALPHA` (default `50` α), signed by your **coldkey** via `recycle_alpha(hotkey, amount, netuid=11)`. Recycle at least the network fee; surplus is burned too, too little is rejected.
- **Receipt** — the CLI passes the recycle's `(block, extrinsic_index)` with the submission; the server verifies it on chain (your hotkey, `amount ≥ fee`, within 24 h) before queuing the pack.
- **Reuse** — the fee is consumed only once your pack enters the eval queue. A submission that fails a technical check (bad format, duplicate, similarity, …) does not consume the receipt, so the same recycle can back another submission within its 24 h window. Re-submitting the same `pack_hash` that already failed terminally does not re-evaluate it.
- **Cooldown** — one submission per hotkey every ~20 min (server-enforced, `MINER_SUBMIT_COOLDOWN_SECONDS`), on top of the fee.

Set the amount in `.env.miner` (`SUBMISSION_FEE_ALPHA=50`) or inline; full reference in [MINER_OPERATIONS.md](MINER_OPERATIONS.md).

---

## Local Testing

```bash
git clone https://github.com/trajectoryRL/trajectoryRL.git
cd trajectoryRL && pip install -e .

cp .env.validator.example .env.validator
# Edit: set LLM_API_KEY (default: qwen3.6-35b-a3b via api.engy.ai)

python scripts/eval_pack.py --skill-md path/to/SKILL.md
# or: python scripts/eval_pack.py --pack pack.json --json results.json -o ./eval_output
```

The script auto-pulls `sandbox-agent` and every per-scenario image in `SANDBOX_SCENARIOS` on first invocation. Output: per-scenario `passed/total`, per-scenario cost, final `Σ` score. Use `--json` for machine-readable results, `-o ./eval_output` to save full transcripts and verifier artifacts.

Prereqs: Docker daemon, an LLM API key (e.g. OpenRouter), ~10 GB free disk for images on first run.

---

## Tips

1. **Spell out the deliverable.** Qwen3.5-A3B does not reliably figure out "write to /app/foo.sh" without being told. The most common 0-score failure is "agent produced text in chat but never wrote the file."
2. **Verify before declaring done.** Run the deliverable. Check it satisfies the task. Many partial credits come from "the script exists but errors on first invocation."
3. **Keep it tight.** Empirically, ~3–5 KB focused packs beat long playbooks. Qwen3.5 stops earlier when given more prescriptive text.
4. **Diversify across domains.** The active scenario set covers async Python, git, sysadmin, database recovery, HTML/JS, log analysis, C/graphics, binary RE. A SKILL.md that helps in two domains beats one that's deep in one.
5. **Don't game the scenario list.** Scenarios rotate. Each `SPEC_NUMBER` bump can add/remove scenarios. SKILL.md targeted at exactly today's set will drop when the set changes.
6. **Don't reverse-engineer the verifier.** It's hidden, and scoring is pure `passed/total` from pytest. The way to score higher is to write better tactics for the agent, not to guess what `test_outputs.py` asserts.

---

## FAQ

**Q: What agent framework runs my SKILL.md?**
A: Hermes (NousResearch) 0.13.x is the default testee. It runs inside the scenario container with built-in `terminal`, `file`, and `execute_code` tools. The validator controls the testee image; miners do not configure it.

**Q: What LLM does the testee use?**
A: Configurable per validator. Default: `qwen3.6-35b-a3b` via engy (`api.engy.ai`), the subnet's own inference API. Validators may run other models; the harness/LLM identity is published per-eval on the dashboard.

**Q: Is there a judge LLM?**
A: No. Scoring is fully programmatic from `tests/test.sh` → `ctrf.json`. The earlier agent-judge architecture was removed 2026-05-11.

**Q: Can I see the scenario tests?**
A: No. `tests/test.sh` is in the scenario image but at a path the `hermes` user cannot read. You can read `INSTRUCTION.md` and any other files the scenario's Dockerfile exposed to `/workspace` or `/app`. For development, you can clone [trajrl-bench](https://github.com/trajectoryRL/trajrl-bench) locally and inspect the tests — but if your SKILL.md hardcodes test names or expected assertions, the pre-eval filter will catch it.

**Q: Does my SKILL.md get to learn between scenarios in a session?**
A: No. SKILL.md is static — same file used for every scenario. There is no persistent `/workspace/learned/` between scenarios.

**Q: Can I submit multiple files?**
A: The schema permits multiple files in `files`, but Season 1 only reads `SKILL.md`. Other files are placed in `/workspace` but the testee prompt only points to SKILL.md + INSTRUCTION.md.

**Q: What if I don't submit a SKILL.md?**
A: Pack rejected at schema validation.

**Q: How often can I resubmit?**
A: One submission per hotkey every ~20 min (server cooldown, `MINER_SUBMIT_COOLDOWN_SECONDS`). Each submission also costs the `recycle_alpha` fee — see [Submission Fee](#submission-fee).
