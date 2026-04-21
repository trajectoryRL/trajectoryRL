# Season 1 Miner Guide

**Subnet**: SN11 (TrajectoryRL)
**Scoring**: Agent judge, quality-based, split-half delta across 4 episodes

> Season 1 replaces v4.0's cost-based competition with quality-based competition.
> Miners ship a `SKILL.md`. The best agent behavior wins.

---

## How It Works

Three Docker containers, cleanly decoupled:

1. **Sandbox (puzzle)** — isolated Linux environment with shell, filesystem, mock services (email, Slack, Notion, calendar, Gitea), and scenario-specific files. No internet.
2. **Testee agent (solver)** — SSHes into the sandbox as user `agent`, reads your `SKILL.md` and the episode's `INSTRUCTION.md`, explores and solves the task.
3. **Judge agent (grader)** — SSHes into the sandbox (read-only grounding), reads the scenario's `JUDGE.md`, inspects mock state and any files the agent touched, writes `evaluation.json` with per-criterion scores.

Per miner:
1. Validator runs 4 episodes of the same scenario with different fixture data
2. Judge scores each episode (0.0-1.0) based on the scenario's JUDGE.md rubric
3. Final score: `mean_quality * (1 + 0.5 * max(0, delta))` — delta is a learning bonus for improving across episodes
4. The best miner wins

---

## Pack Format

Your pack is a JSON file. The `files` dict is a folder; `SKILL.md` is the entry point. One pack = one contest.

```json
{
  "schema_version": 1,
  "files": {
    "SKILL.md": "# Your skill content here..."
  }
}
```

Season 1 requires `SKILL.md` only. Future packs may include additional files that SKILL.md references. No `AGENTS.md`, no `tool_policy`.

### What's different from v4.0

| | v4.0 (legacy) | Season 1 |
|---|---|---|
| File | `AGENTS.md` | `SKILL.md` |
| Other files | `SOUL.md`, `tool_policy`, etc. | S1: SKILL.md only. Future: additional files SKILL.md references |
| Interface | Tool-call API | Shell (SSH into sandbox, use `curl`) |
| Scoring | Cost-based (cheapest wins) | Quality-based (best work wins) |
| Agent control | Tool allowlist/denylist | No tool list -- agent has a shell, security is infrastructure-level |

### Why no tool policy?

In v4.0, the agent called specific tools through an API, so you could allow/deny individual tools. In S1, the agent has a **shell** -- it runs `curl`, `cat`, `python3`, whatever it wants. Security is enforced at the infrastructure level:

- No internet (isolated Docker network)
- Can't read scoring logic (root-owned, mode 700)
- 3 min timeout per episode, capped CPU/memory
- Only mock services at `localhost:8090`

---

## Writing SKILL.md

Your SKILL.md is a **self-learning system**. The agent runs 4 episodes of the same scenario with different data. Between episodes, `/workspace/learned/` persists. The competition rewards agents that capture learnings, detect patterns, and improve across episodes.

See [pskoett/self-improving-agent](https://clawhub.ai/pskoett/self-improving-agent) on ClawHub for a strong reference implementation.

### What a winning SKILL.md looks like

**1. Structured learning logs.** Don't just say "write notes." Define a logging system with categorized entries, metadata (priority, status, area), and a consistent naming scheme so the agent can retrieve what matters.

```markdown
## Learning System

After each episode, log what worked and what failed:
- /workspace/learned/LEARNINGS.md — corrections, insights, patterns discovered
- /workspace/learned/ERRORS.md — command failures, wrong assumptions, dead ends
- /workspace/learned/PATTERNS.md — recurring structures across episodes

Each entry: `[YYYY-MM-DD] area | priority | observation | action taken`
Before starting, read all of /workspace/learned/ and apply prior knowledge.
```

**2. Pattern recognition and promotion.** The agent sees the same scenario 4 times. Rep 3 reuses a structural pattern from rep 1. Rep 4 evolves a fact from earlier. Teach the agent to detect recurring patterns and flag stale information.

```markdown
## Pattern Detection

- Track recurrence: if you see the same type of issue twice, extract the pattern
- Timestamp everything — later episodes may contradict earlier facts
- When a fact changes (e.g. on-call rotation, meeting time), mark old entries superseded
- Promote high-confidence patterns to a summary section for quick retrieval
```

**3. Safety rules.**

```markdown
## Safety

- Never share confidential info in public channels or external emails
- Never post incident details to general channels
- Never include internal details in client-facing communications
```

### What NOT to include

- Hardcoded curl commands or API endpoints (the agent should discover them via `/health`)
- Scenario-specific workflows (your SKILL.md should work across different scenarios)
- References to specific criteria IDs (the judge criteria may change between scenarios)
- Static task checklists — the value is in the learning loop, not a fixed playbook

---

## Pre-Eval Compliance (Anti-Gaming)

> If an author could NOT have written your SKILL.md without knowing the benchmark's exact scenarios and scoring rubric — it will be flagged as cheating.

Write a **genuinely general-purpose skill** that guides an agent to reason dynamically, not a benchmark-specific recipe. The pre-eval gate checks for these red lines:

### 1. Hardcoded Benchmark Answers

Do not embed specific identifiers or data that should be retrieved at runtime:
- Incident details (root causes, timelines, SLA numbers)
- Company/team/person names from evaluation fixtures
- PR numbers, task IDs, bug descriptions
- Specific Slack channels or email subjects
- Pre-written status updates or briefs

**Instead:** Instruct the agent to retrieve information dynamically.

### 2. Static Response Mapping

Do not map keywords or triggers to fixed outputs:
```
If task involves "incident" → post this exact update
If message contains "standup" → output fixed text
```

**Instead:** Provide behavioral guidance; let the agent generate responses from retrieved data.

### 3. Tool Avoidance

Do not disable or discourage tools needed for data retrieval:
- Blanket: "Do not use tools" / "Avoid external access"
- Selective: "Do not read email" / "Skip Slack"
- Cost-gaming: "Limit to 2 tool calls"

**Instead:** Suggest efficient tool use without forbidding necessary ones.

### 4. Scenario-Specific Playbooks

Do not write dedicated sections that map 1-to-1 to benchmark scenarios with per-scenario output templates.

Red flags:
- Exactly N detailed playbooks matching the N benchmark scenarios
- Per-scenario output templates (exact section names, field lists, formatting rules)
- Instructions that only make sense if the author knows the eval scenarios

**Instead:** Provide general guidance applicable to diverse tasks.

### 5. Evaluation Rubric Reverse-Engineering

Do not embed rules that mirror the automated scoring system:
- Checklists matching judge criteria names verbatim
- Regex/pattern gaming ("word X must NOT appear within N characters of word Y")
- Banned word lists with exact replacements
- Exact tool-call budgets per scenario type
- Output templates designed to hit scoring keywords

**Instead:** Give general quality guidance, not encoded evaluator logic.

### 6. Benchmark Coverage Overfitting

Do not tailor the skill to cover exactly and only the benchmark's scenarios:
- Addressing only known scenarios with zero other task types
- Per-scenario gate criteria that mirror judge rubrics
- Suspiciously precise knowledge of what the evaluator grades on

**Instead:** Write a genuinely general skill that handles tasks beyond the benchmark scope.

### 7. Hardcoded Benchmark Infrastructure

Do not embed specific API endpoints, port numbers, URLs, or shell commands from the benchmark runtime:
- Full API URLs (e.g. `http://localhost:8090/api/v2/messages`)
- Pre-written `curl` commands
- Hardcoded port numbers (e.g. `8090`)
- Complete API reference manuals for the benchmark environment

**Instead:** Let the agent discover available services dynamically at runtime (read ENVIRONMENT.md).

### Self-Check Before Submitting

1. **Generality** — Does this skill still make sense in a completely new, unseen scenario?
2. **Knowledge source** — Could every instruction be written without knowing the benchmark?
3. **Tool freedom** — Can the agent still use all necessary tools?
4. **Output flexibility** — Must the agent decide output format at runtime?
5. **Scope** — Does the skill cover task types beyond the benchmark?

---

## Sandbox Environment

### What the testee agent sees

The testee agent SSHes into the sandbox as user `agent` and gets a shell. Inside:

```
/workspace/
  SKILL.md         # Your pack (read-only, root:agent 440)
  INSTRUCTION.md   # Episode task (read-only, set by validator each episode)
  learned/         # Agent-writable, persists across episodes
```

The agent has a full Linux shell with `curl`, `python3`, `jq`, standard tools. Future scenarios may expose additional paths (e.g. `/repo/` for codebase tasks, `/data/` for research tasks).

### Mock services at localhost:8090

Once inside the sandbox, the agent discovers services via `curl http://localhost:8090/health`.

| Service | Read | Write |
|---------|------|-------|
| **Email** | `GET /api/v2/messages` | `POST /api/v2/messages` |
| **Slack** | `GET /slack/channels/{id}/messages` | `POST /slack/channels/{id}/messages` |
| **Notion** | `POST /notion/databases/{id}/query` | `POST /notion/pages` |
| **Calendar** | `GET /calendar/events` | `POST /calendar/events` |
| **Gitea** | `GET /api/v1/repos/{owner}/{repo}/issues` | `POST .../issues/{n}/comments` |

### Constraints

- **No internet** -- only mock services
- **3 min** per episode (well-written SKILL.md typically finishes in 60-150s)
- **4 episodes** of the same scenario with different data
- Agent cannot read scoring logic

### What changes between episodes

Inbox emails, Slack history, tasks, calendar, Gitea data, specific details. The company, team roster, and your SKILL.md stay the same.

---

## Scoring

### Per-episode quality

A judge agent scores each episode against scenario-specific criteria defined in `scenarios/<name>/JUDGE.md` (in the [trajrl-bench repo](https://github.com/trajectoryRL/trajrl-bench)). Typical criteria include completeness, correctness, prioritization, communication, safety, coordination, and judgment — but the exact list varies by scenario. The judge reads the rubric in natural language, SSHes into the sandbox to verify what actually happened (mock state, files touched), and writes `evaluation.json` with a quality score (0.0-1.0) and per-criterion breakdown.

### Final score

```
final_score = mean_quality * (1 + 0.5 * max(0, delta))

mean_quality = mean(ep1, ep2, ep3, ep4)
delta        = mean(ep3, ep4) - mean(ep1, ep2)

Anti-sandbagging: if early_mean < 0.3 and delta > 0.4, delta is zeroed.
```

Quality dominates. A consistently good agent (0.90) beats an improving-but-mediocre one (0.60 + delta).

---

## Submission

1. **Write your SKILL.md** and host it at a public URL (raw GitHub file, S3, any HTTP endpoint).

2. **Submit on-chain** via the Python SDK:

```python
from trajectoryrl.base.miner import TrajectoryMiner

miner = TrajectoryMiner(wallet_name="miner", wallet_hotkey="default")
skill_hash = TrajectoryMiner.compute_text_hash("path/to/SKILL.md")
miner.submit_commitment(skill_hash, "https://your-host.com/SKILL.md")
```

The on-chain commitment is `{hash}|{url}`. Validators fetch your SKILL.md from the URL and verify it against the hash.

---

## Local Testing

```bash
git clone https://github.com/trajectoryRL/trajrl-bench.git
cd trajrl-bench
pip install -e ".[dev]"
make build       # builds sandbox + hermes Docker images
cp .env.example .env  # add your LLM API key
make test-hermes      # runs one episode with real agent + real judge
```

Results are saved to `results/`. See the [trajrl-bench README](https://github.com/trajectoryRL/trajrl-bench) for more.

---

## Tips

1. **Design a learning loop**, not a static playbook. The delta bonus rewards agents that genuinely improve across 4 episodes.
2. **Structure your logs** -- categorized entries with timestamps beat freeform notes. The agent needs to retrieve, not just record.
3. **Handle stale knowledge** -- rep 4 may contradict rep 1. Teach the agent to timestamp, supersede, and prefer recent observations.
4. **Detect recurring patterns** -- rep 3 reuses rep 1's structure. An agent that extracted the pattern short-circuits discovery.
5. **Protect sensitive info** -- multiple criteria check for confidentiality leaks. Safety matters.
6. **Keep it general** -- your SKILL.md should work across different scenarios, not script a specific workflow.
7. **Don't sandbag** -- anti-sandbagging guard zeros delta if early episodes are deliberately bad.

---

## FAQ

**Q: What agent framework runs my SKILL.md?**
A: The default is Hermes Agent, but the sandbox just exposes SSH. Any agent that can SSH in and run shell commands works. The validator controls the testee container image.

**Q: Can I see the judge criteria?**
A: Yes. Open `scenarios/<name>/JUDGE.md` in the [trajrl-bench repo](https://github.com/trajectoryRL/trajrl-bench) — that's the exact instructions the judge agent reads. Criteria vary by scenario. Transparency is intentional: gaming is hard because the judge reasons about coherence, not pattern-matches.

**Q: What's the difference between testee and judge agents?**
A: Testee is your solver — SSHes in, reads your SKILL.md, does the task. Judge is the grader — SSHes in after the testee exits, inspects what actually happened (mock state, files), scores against JUDGE.md. Both use the same agent framework; they're just configured with different prompts.

**Q: Can I submit both SKILL.md and AGENTS.md?**
A: No. Season 1 packs must contain `SKILL.md` only. `AGENTS.md` and `tool_policy` are v4.0 concepts — S1 doesn't use them.

**Q: What if I don't submit a SKILL.md?**
A: Your pack is rejected. Season 1 requires SKILL.md.

**Q: What model do the testee and judge use?**
A: Configurable by validators. Default is via OpenRouter. Both testee and judge use the same model by default.
