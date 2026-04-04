# Season 1: Self-Learning Agents

> v0.2 — Docker sandbox evaluation with persistent SKILL.md and LLM judge scoring for self-learning agents.

---

## Design Principles

The evaluation uses a single mechanism: **the LLM judge.**

```
Quality (judge score)
  |                     *
  |              *  *
  |          *
  |      *  *
  |    *
  |  *
  +----------------------→ Episode
  1  2  3  4  5  ... N

Score = the upward trend. Steeper = faster learner.
```

Run a sequence of tasks, judge each trajectory, score the trend. The same LLM judge used in v1 evaluates every episode. An agent that learns produces higher-quality trajectories over time.

**Key properties:**

1. **Single mechanism.** The LLM judge scores trajectories — same approach across any task type, any domain, any agent framework. No custom scoring infrastructure needed.

2. **Agent-harness-agnostic.** The interface is: SSH into a sandbox, read SKILL.md, execute. The validator only sees a quality score per episode.

   | Framework | How it consumes SKILL.md |
   |-----------|-------------------------|
   | Claude Code | Reads as `CLAUDE.md` |
   | Cursor | Reads as `.cursor/rules` |
   | OpenClaw | Reads as `AGENTS.md` |
   | Custom harness | `cat /workspace/SKILL.md` |
   | Raw LLM + bash | System prompt includes file |

   Miners compete on which agent framework learns best, not which prompt is cleverest. A miner running Claude Code competes directly against a miner running a custom Python harness.

3. **Resistant to gaming.** A single scenario result can be hacked. A consistent upward quality trend across 4-8 repetitions of the same scenario cannot, because:
   - Episode count **varies** each epoch (N=8-16)
   - Data is **different** each attempt (same template, new content)
   - Two scenario domains prevent single-task overfitting

The only viable strategy is to build an agent that genuinely learns.

---

## What's Wrong with v1

### 1. Stateless Mock Tools — The Fundamental Lie

Tools don't reflect mutations:

- Agent sends email → `himalaya envelope list` still returns the original inbox
- Agent creates a Notion task → next `databases/query` returns the same tasks
- Agent deletes calendar event → event is still there on next read

We can only score the agent's *intent* (did it try to send?), not its *competence* at handling state transitions. **Multi-step workflows where step 2 depends on step 1 are fundamentally untestable.**

### 2. `exec` Is a God-Function with Brittle Regex

The exec handler is a ~170-line chain of regex patterns covering 4 completely different systems (email, tasks, calendar, GitHub):

- **Command variation kills agents**: `himalaya envelope list` works, `himalaya list envelope` doesn't.
- **Creative agents are punished**: An agent that uses `curl` to hit the email API directly instead of `himalaya` gets a generic fallback.

This creates a **narrow corridor of "correct" commands** that rewards pattern memorization over genuine capability.

### 3. Fixture-Scenario Tight Coupling

Each scenario is an isolated island with manually crafted fixture files:

- **No composability**: Can't mix fixtures across scenarios.
- **Manual data maintenance**: Adding a scenario means hand-writing 5-10 JSON fixture files. Slow and error-prone.
- **No parameterized variation**: Can't test "same scenario but 50 emails instead of 5."

### 4. Single-Turn, Single-Episode Limitation

Can't express approval flows, clarification dialogues, follow-up evaluation, or long-running tasks. Miners have zero incentive to handle multi-turn interactions.

### 5. No Error Simulation

Tools always succeed. No rate limits, auth failures, timeouts. We can't evaluate agent **robustness** — only happy-path behavior.

### 6. Knowledge-Worker Monoculture

All 7 scenarios are office-worker email/calendar/tasks/Slack. The "cost competition" is really "who can write the shortest AGENTS.md that passes these 7 office scenarios."

### 7. Fixed Fixture Pool Enables Memorization

Miners can read every fixture email/task/event (open-source ClawBench). With only 7 scenarios and static data, miners can and do memorize the benchmark. Subtle optimization-through-memorization is structurally indistinguishable from genuine capability.

### 8. No Self-Learning Evaluation

The current system evaluates a single snapshot in time. There's no mechanism to test whether an agent can improve across tasks, retain corrections, or transfer learnings. Every episode is independent — the agent has no memory of what it did before.

---

## The Solution: Docker Sandbox

Instead of mock tool handlers that regex-match commands and return static fixtures, **the agent SSHs/execs into a prepared Docker sandbox** where real (mock) services run with real protocols and stateful behavior.

### v1 → v2

```
v1: Agent → OpenClaw API → mock handler → regex match → static fixture → canned response
v2: Agent → SSH/exec into Docker → real shell → real (mock) services → stateful environment
```

### What Changes

**No exec god-function.** The agent runs real commands in a real shell. `curl localhost:1080/api/v2/messages`, `python3 -c "import imaplib; ..."`, raw `telnet localhost 1025` — all valid, all produce real results.

**Stateful by default.** Agent sends email → it appears in the mock SMTP server's mailbox. Scoring inspects the final state of the environment, not the commands used.

**Procedural data.** Fixture generation seeds different data each eval. Same structure, completely different content. Memorization is no longer a viable strategy.

**Protocol-level interface.** Mock services respond to real protocols, not pattern-matched strings. Agents are free to use any tool or method that speaks the protocol.

---

## Sandbox Container Architecture

### Universal Interface: Shell + Filesystem + HTTP

The sandbox requires **no framework-specific tools**. Every agent framework — Claude Code, Cursor, Hermes, OpenClaw, or a custom harness — has access to the same three primitives:

| Primitive | What it does | How agents use it |
|-----------|-------------|-------------------|
| **Shell** (bash via SSH/exec) | Run any command | `curl`, `git`, `python3`, pipe commands, etc. |
| **Filesystem** (read/write/edit) | Persist data, read configs | SKILL.md, workspace files, code repos |
| **HTTP** (localhost services) | Talk to mock services | Any HTTP client speaks the same protocol |

Any method that speaks the protocol works: `curl localhost:1080/api/v2/messages`, `python3 -c "import smtplib; ..."`, or raw socket connections. The mock services expose **standard protocols**, not framework-specific APIs.

```
Docker Container ("eval sandbox")
├── Mock Services (stateful, standard protocols — all deterministic)
│   ├── MailHog/MailPit       (SMTP :1025, HTTP API :1080) — email
│   ├── Mock Notion API       (HTTP :8080) — tasks / databases
│   ├── Mock Calendar API     (CalDAV :5232 or HTTP :8081)
│   ├── Mock Slack API        (HTTP :8082) — channels, messages
│   └── Gitea                 (HTTP :3000, git SSH :2222) — repos, PRs, issues
│
├── Standard Tools (pre-installed, all optional — agent can use any method)
│   ├── curl, jq, python3, git, node, gh — universal
│   └── ~/.config/ pre-configured to point at local mock services
│
├── Workspace
│   ├── /workspace/SKILL.md    (miner's pack — PERSISTS across episodes)
│   ├── /workspace/learned/    (agent's learning store — PERSISTS)
│   ├── /workspace/...         (pack files)
│   └── /workspace/docs/       (scenario-specific reference docs)
│
├── Seed Data (LLM-generated from scenario template + epoch_seed)
│   ├── Pre-loaded emails in MailHog
│   ├── Pre-loaded tasks in mock Notion
│   ├── Pre-loaded calendar events
│   ├── Pre-loaded Slack channel history
│   └── Pre-loaded web search results + memory entries (static fixtures)
│
└── Security
    ├── Network: all egress blocked (fully offline sandbox)
    ├── CPU / memory / disk limits
    └── Hard timeout per episode
```

**Key point:** The sandbox is tool-agnostic. It doesn't know or care which agent framework is running. It exposes standard protocols and inspects final state. A Claude Code agent using `bash` and a custom Python harness using `requests` are evaluated identically.

### Sandbox Lifecycle

```
Container lifecycle:

  ┌──────────────────────────────────────────────┐
  │  Docker Sandbox (persistent across episodes) │
  │                                              │
  │  /workspace/SKILL.md  ← PERSISTS             │
  │  /workspace/learned/  ← PERSISTS             │
  │                                              │
  │  Mock services        ← DATA RESETS each ep  │
  │  Shell history        ← CAPTURED each ep     │
  │  Agent process        ← RESTARTS each ep     │
  └──────────────────────────────────────────────┘

Between episodes:
  1. Capture: shell transcript, cost, service state
  2. Score: LLM judge → quality score (0.0–1.0)
  3. Reset: reload mock services with new fixtures
  4. Preserve: /workspace/SKILL.md, /workspace/learned/
  5. Start next episode
```

The container never stops. Only the "world" resets. The agent's brain (SKILL.md + any files it creates) persists.

---

## SKILL.md: Agent-Harness-Agnostic Pack Format

Rename AGENTS.md → **SKILL.md**. A skill file is a plain markdown document that any agent framework can consume. The sandbox places it at `/workspace/SKILL.md`. The agent harness — whatever it is — reads it.

```markdown
# SKILL.md (miner-authored)

## Instructions
[How to approach tasks, tool usage patterns, safety rules, etc.]

## Learned Patterns
[Initially empty. Agent appends here as it learns.]

## Project Context
[Accumulated knowledge about the workspace, codebase, team, etc.]
```

The miner authors the initial `Instructions` section. The `Learned Patterns` and `Project Context` sections start empty and grow as the agent self-improves across episodes.

**Why SKILL.md works:**
- It's just a file. No special protocol, no API, no framework dependency.
- The agent reads it at episode start, writes to it at episode end.
- The validator doesn't need to understand the format — it just doesn't delete it between episodes.
- Miners compete on how well their SKILL.md teaches the agent to learn, not on memorizing scenarios.

**Reference implementation:** [ivangdavila/self-improving](https://clawhub.ai/ivangdavila/self-improving) is an existing skill that uses three-tier memory (HOT/WARM/COLD) with auto-promotion of patterns after repeated use. It is instruction-only, framework-agnostic, and requires no external dependencies — exactly the kind of approach this evaluation is designed to test. A well-engineered SKILL.md should outperform it by optimizing specifically for cost reduction across episodes rather than general-purpose memory management.

### Harness Adapters

The miner declares which agent framework to use in `pack.yaml`. The validator has **predefined harness adapters** — no miner code executes, only whitelisted binaries.

```yaml
# pack.yaml (miner-provided)
harness: claude-code    # from whitelist
```

| Harness | Validator copies SKILL.md to | Launches |
|---------|------------------------------|----------|
| `claude-code` | `/workspace/CLAUDE.md` | `claude --task "$PROMPT"` |
| `cursor` | `/workspace/.cursor/rules` | `cursor-agent --task "$PROMPT"` |
| `openclaw` | `/workspace/AGENTS.md` | `openclaw run --task "$PROMPT"` |
| `raw-bash` | (reads directly) | `bash -c "cat SKILL.md && $PROMPT"` |

After each episode, the adapter syncs back: e.g., `cp CLAUDE.md SKILL.md` so learned patterns persist in the canonical location.

**Security:** Miners control SKILL.md content only — not execution. Adding a new framework = adding one adapter to the validator (a few lines of shell). The harness whitelist is part of the validator release, not configurable by miners.

---

## Mock Strategy

### Two-Tier Architecture

```
Mock Services (stateful, scoring inspects state)
  → Email (MailHog/MailPit), Tasks, Calendar, Slack, GitHub/Gitea
  → Web search, web fetch, memory (read-only, pre-generated fixtures)
  → Agent mutations are real, state is inspectable after episode

Fixture Factory (build-time generation)
  → Generates ALL seed data — stateful services AND read-only fixtures
  → Replaces hand-crafted JSON fixture files
  → Deterministic distribution via hash consensus
```

All services are deterministic. All data is pre-generated before the episode starts. No LLM calls during episode runtime. No egress from the sandbox.

### Service Table

| Service | Mutations? | Scoring Method |
|---------|-----------|----------------|
| Email | Yes (send, delete, move) | State inspection: "is there an email to Dana?" |
| Tasks/Notion | Yes (create, update, close) | State inspection: "are there 3 new tasks?" |
| Calendar | Yes (create, delete events) | State inspection: "is the conflict resolved?" |
| Slack | Yes (send messages, react) | State inspection: "was P0 posted to #engineering?" |
| GitHub/Gitea | Yes (commits, PRs, issues) | State inspection: "do tests pass? Is PR merged?" |
| Web search | No (read-only, fixture) | Response quality: "did agent use relevant results?" |
| Web fetch | No (read-only, fixture) | Response quality: "did agent use page content?" |
| Memory | No (read-only, fixture) | Response quality: "did agent leverage past context?" |
| Filesystem | Yes (create, edit files) | File diff: deterministic |

**Stateful services** accept mutations — scoring inspects final state.
**Read-only services** return pre-generated fixtures — scoring evaluates what the agent did with the information.

### Read-Only Services: File-Based Mocks

Web search, web fetch, and memory are served as static fixture files by lightweight HTTP endpoints:

```
/fixtures/web_search/results.json    → keyed by topic/keyword
/fixtures/web_pages/                 → full page content (markdown/HTML)
/fixtures/memory/entries.json        → keyed by query pattern
```

```
Agent: curl localhost:8083/search?q=notion+batch+operations
  → Mock service matches keywords against results.json
  → Returns pre-generated search results

Agent: curl localhost:8083/fetch?url=https://docs.notion.so/api
  → Mock service returns matching page from /fixtures/web_pages/
  → Returns pre-generated page content

Agent: curl localhost:8084/memory?q=meeting+notes+dana
  → Mock service matches against entries.json
  → Returns pre-generated memory entries
```

The fixture factory generates these alongside email/tasks/calendar data — same `epoch_seed`, same generation flow, same `fixture_hash`. The search fixtures cover the relevant information space for each scenario. Queries that don't match any fixture return an empty result set.

### Fixture Factory

Replace hand-crafted fixture JSON with LLM-generated fixtures from scenario templates:

```yaml
# scenario_template: morning_brief
fixture_generation:
  email:
    prompt: |
      Generate {n_emails} work emails for {persona.name}, {persona.role} at {persona.company}.
      Requirements:
      - {n_urgent} marked urgent
      - At least one mentions a calendar conflict
      - At least one contains confidential information ({confidential_topic})
      Cross-reference: use the same project names as the task fixtures.
    schema: schemas/email.json
    params:
      n_emails: "rng.randint(8, 15)"
      n_urgent: "rng.randint(2, 4)"
      confidential_topic: "rng.choice(['SOC 2 audit', 'acquisition talks', 'layoff planning'])"
```

**Generation flow:**

```
epoch_seed
  → PRNG determines structural params (counts, urgency distribution, topics)
  → LLM generates content within those constraints
  → Output validated against JSON schema
  → Cached by hash(epoch_seed + scenario_id + template_version)
  → Loaded into mock services at container start
```

**Scenario authoring becomes:**
- Current: write scenario YAML + hand-craft 5-10 fixture JSON files (days)
- Proposed: write scenario YAML + fixture generation prompt + scoring spec (hours)

### Determinism Across Validators

**Solution: Hash-locked consensus.**

```
1. Validator generates fixtures from epoch_seed + scenario template
2. Hashes the complete fixture bundle → fixture_hash
3. Reports fixture_hash alongside evaluation results
4. Consensus: if >50% of validators report the same fixture_hash → canonical
5. Outlier validators re-download canonical fixtures and re-evaluate
```

---

## Scoring: LLM Judge

### Per-Episode: LLM Judge Evaluation

The same LLM-as-judge approach from v1, applied to each episode's trajectory. The judge receives the shell transcript and mock service state, evaluates against scenario criteria.

```
v1: "Did the agent call `exec` with arg matching `/himalaya.*send/`?"  (regex)
v2: "Query MailHog API — is there an email to dana@acme.com with      (state-based)
     subject containing 'incident update'?"
v3: LLM judge evaluates the full trajectory against criteria           (universal)
```

The judge produces a quality score per episode (0.0–1.0) covering correctness, completeness, and safety. State-based checks (mock service inspection) serve as grounding evidence for the judge — not as the scoring mechanism itself.

### Across Episodes: Quality Trajectory (Learning Signal)

The learning signal is whether quality scores improve over episodes:

- Flat at low quality = not learning (low score)
- Flat at high quality = already capable, not improving (high score, no learning bonus)
- Upward trend = learning from experience (high score + learning bonus)
- Steep upward trend = fast learner (maximum score)

No separate gate. A bad episode scores low (e.g. 0.1) and drags down `mean(quality)`. The formula handles it naturally — no binary disqualification needed.

```
# Per-scenario learning curves (4-8 data points each — statistically meaningful)
for scenario in scenarios:
    scores = [judge_quality(ep) for ep in episodes if ep.scenario == scenario]
    learning_rate[scenario] = linear_regression_slope(scores)

overall_learning = mean(learning_rate.values())
overall_quality  = mean(all_episode_scores)
final_score      = overall_quality * (1 + max(0, overall_learning))
```

Per-scenario regression over 4-8 repetitions. High quality AND improving = win.

---

## Evaluation Flow

```
1. Build sandbox (mock services + CLI tools)
2. Load miner's SKILL.md into /workspace/
3. Derive sequence from epoch_seed:
   - N = rng.randint(8, 16)                        ← variable length
   - sequence = interleave(2 scenarios, N)          ← 4-8 reps each
4. For episode i = 1..N:
   a. Reset mock service data (new fixtures from epoch_seed + i)
   b. Deliver task prompt: sequence[i]
   c. Agent runs: reads SKILL.md → does task → updates SKILL.md
   d. Capture: shell transcript + mock service state
   e. Judge: LLM evaluates trajectory → quality score (0.0–1.0)
5. Tear down sandbox
6. Score:
   - Per-scenario learning curves (regression slope over 4-8 reps)
   - final_score = mean(quality) * (1 + mean(learning_rates))
```

One scoring mechanism, one formula. No gates, no thresholds.

---

## Episode Sequence Design

### Few Scenarios, Many Repetitions

Instead of spreading N episodes across 7 scenario types (~2 repetitions each, weak signal), use **2 deep scenarios repeated 4-8 times each** (strong signal).

```python
rng = Random(epoch_seed)

scenarios = ["client_escalation", "code_bug_fix"]   # 2 deep scenarios
N = rng.randint(8, 16)
sequence = [rng.choice(scenarios) for _ in range(N)]
rng.shuffle(sequence)

# Each episode gets unique data from the same scenario template
for i, scenario in enumerate(sequence):
    fixtures = generate(epoch_seed + i, scenario)
```

**Example: Epoch A (N=10):**

```
 E1:  client_escalation   (data_seed_1)    ← 1st attempt
 E2:  code_bug_fix        (data_seed_2)    ← 1st attempt
 E3:  client_escalation   (data_seed_3)    ← 2nd attempt, should improve
 E4:  client_escalation   (data_seed_4)    ← 3rd attempt
 E5:  code_bug_fix        (data_seed_5)    ← 2nd attempt, should improve
 E6:  code_bug_fix        (data_seed_6)    ← 3rd attempt
 E7:  client_escalation   (data_seed_7)    ← 4th attempt
 E8:  code_bug_fix        (data_seed_8)    ← 4th attempt
 E9:  client_escalation   (data_seed_9)    ← 5th attempt, should be best
E10:  code_bug_fix        (data_seed_10)   ← 5th attempt, should be best
```

Each scenario appears 4-8 times with different data. The learning curve per scenario is the signal — not a noisy global regression across unrelated tasks.

**Why 2 scenarios, not 1 or 7:**

| Count | Pros | Cons |
|-------|------|------|
| 1 | Maximum repetitions, cleanest signal | No breadth testing, over-fits to one task type |
| 2 | Strong signal (4-8 reps each), tests two domains | Moderate breadth |
| 7 | Maximum breadth | ~2 reps each, noisy signal, hard to measure learning |

Two is the sweet spot: enough repetitions for a clear learning curve, enough variety to prevent single-scenario overfitting.

**Scenario selection criteria:**
- **Deep**: many sub-tasks, decision points, safety constraints (room to improve)
- **Different domains**: one knowledge-worker, one code/technical
- **Complex enough** that a first attempt is genuinely harder than a fifth attempt with accumulated patterns

### Key Properties

- **Repeated scenarios**: 4-8 attempts per scenario = clear per-scenario learning curve
- **Variable N**: can't predict when eval ends
- **Different data each attempt**: same scenario template, completely different content
- **Two domains**: learning must work for both knowledge work and code tasks

---

## What the Validator Sees

```json
{
  "miner_uid": 42,
  "pack_hash": "abc123...",
  "episodes": [
    {"id": 1,  "scenario": "client_escalation", "quality": 0.45},
    {"id": 2,  "scenario": "code_bug_fix",       "quality": 0.40},
    {"id": 3,  "scenario": "client_escalation", "quality": 0.62},
    {"id": 4,  "scenario": "code_bug_fix",       "quality": 0.58},
    {"id": 5,  "scenario": "client_escalation", "quality": 0.71},
    {"id": 6,  "scenario": "code_bug_fix",       "quality": 0.65},
    {"id": 7,  "scenario": "client_escalation", "quality": 0.78},
    {"id": 8,  "scenario": "code_bug_fix",       "quality": 0.72},
    {"id": 9,  "scenario": "client_escalation", "quality": 0.82},
    {"id": 10, "scenario": "code_bug_fix",       "quality": 0.79}
  ],
  "per_scenario": {
    "client_escalation": {"mean": 0.676, "learning_rate": 0.088},
    "code_bug_fix":       {"mean": 0.628, "learning_rate": 0.094}
  },
  "overall_quality": 0.652,
  "overall_learning": 0.091,
  "final_score": 0.711
}
```

Per-scenario learning curves — each with 5 data points — are the evaluation output.

---

## Anti-Gaming Analysis

Four mechanisms work together:

| Mechanism | What it prevents |
|-----------|-----------------|
| Varying data (generated fixtures) | Memorization of specific answers |
| Per-scenario regression (4-8 reps) | Noise-based gaming (too few data points to fake a trend) |
| Variable episode count (N=8-16) | Endpoint targeting (inflate early, deflate late) |
| LLM judge (not cost-based) | Scheduled efficiency tricks (e.g. "be verbose early, concise later") |

A successful gaming strategy would need to defeat all four simultaneously.

Using the LLM judge as the sole scoring mechanism eliminates a class of gaming strategies that cost-based scoring is vulnerable to. A miner cannot trick a judge into seeing quality improvement — the judge evaluates the actual trajectory outcome, not a proxy metric.

**What miners must actually build:** A SKILL.md that teaches the agent to reflect after each task, identify effective patterns, store them compactly, and retrieve them in new contexts. This is an agent engineering problem, not a benchmark optimization problem.

---

## High-Confidence Components

These parts of the design solve real v1 problems regardless of how scoring evolves:

1. **Docker sandbox.** Replacing regex mocks with real stateful services is a strict upgrade. Agents run real commands, mutations are real, scoring inspects actual state. This alone eliminates the exec god-function, the narrow command corridor, and the intent-vs-competence gap.

2. **SKILL.md as pack format.** A plain markdown file any framework can read. No protocol, no API, no framework lock-in. The abstraction is minimal and correct.

3. **Procedural data generation (Fixture Factory).** LLM-generated fixtures from seed + template eliminates memorization and removes the fixture maintenance burden. Deployable independently — even without the sandbox, this improves v1.

4. **State-based scoring evidence.** Checking MailHog API for "is there an email to Dana?" instead of regex-matching `himalaya send` is a qualitative leap. Agents are free to use any tool or method. The scoring sees outcomes, not commands.

5. **LLM judge as universal scorer.** Already proven in v1. Extending it to continuous quality scoring (0.0–1.0) is incremental, not architectural. No separate gate mechanism — low quality episodes simply score low. The judge is the one component that doesn't need to be built from scratch.

The primary unknowns are in the **multi-episode learning signal** (how to reliably measure improvement with small N) and **cross-validator determinism** (judge variance). These are addressed in the risks section below.

---

## Known Risks & Mitigations

### 1. ~~Weak learning signal with small N~~ — Resolved

~~With many scenarios and few repetitions, the learning signal was noisy.~~

**Resolution:** Reduced to 2 deep scenarios repeated 4-8 times each. Per-scenario regression over 4-8 data points is statistically meaningful. This is a direct fix, not a mitigation.

### 2. SKILL.md growth vs. quality trade-off

As SKILL.md accumulates patterns, it grows. A larger SKILL.md means the agent spends more tokens reading context before each task. Without pruning, later episodes may degrade as context bloat reduces agent effectiveness.

**Mitigation:** This is intentionally part of the competition — miners must engineer SKILL.md with pruning and compression strategies. A size cap (Open Question #3) provides a hard bound.

### 3. LLM judge variance across validators

The LLM judge is probabilistic. Validator A and Validator B may score the same trajectory differently. With N=12 episodes per miner, small per-episode variance compounds into meaningful disagreement on `mean_quality` and `learning_rate`.

**Mitigation:** Use structured rubrics with binary sub-criteria per dimension (e.g., correctness, completeness, safety) rather than a single numeric score. Binary judgments are more reproducible across LLM calls. Quality score = fraction of criteria passed. Alternatively, use median-of-validators scoring.

### 4. Evaluation cost and time

Each miner requires 8-16 episodes × (agent runtime + judge call). At ~3 min/episode, a 12-episode eval takes ~36 min per miner. With 50 miners and 10 parallel containers: ~3 hours per epoch.

**Mitigation:** This is within the 24h epoch window. Parallel containers scale linearly. Judge calls can be batched. LLM cost per epoch (~$30-50 at current rates) is manageable for validators earning TAO emissions.

### 5. The "already good" problem

A miner whose SKILL.md produces high-quality trajectories from episode 1 shows no improvement slope (`learning_rate ≈ 0`). The formula `mean(quality) * (1 + learning_rate)` still rewards high absolute quality, but does not distinguish "consistently excellent" from "mediocre but slightly improving."

**Mitigation:** The formula handles this — `mean(quality)` dominates when `learning_rate` is near zero. A miner scoring 0.95 across all episodes gets `final_score = 0.95`, which beats a miner improving from 0.4 to 0.7 (`mean=0.55, rate=0.05, final=0.578`). The learning bonus rewards improvement but doesn't override raw quality.

### 6. Miner meta-game evolution

**Month 1-2:** Basic SKILL.md files ("reflect after each task"). Low differentiation. Most miners produce similar quality trajectories.

**Month 3-4:** Top miners discover that SKILL.md structure matters — concise pattern storage, good categorization, pruning instructions. Separation emerges.

**Long-term:** Competition converges on optimal memory management strategies. The eval must add new scenario types (Phase 6) to maintain competitive pressure.

**Mitigation:** The season model is designed for this — Season 1 runs until the meta stabilizes, then Season 2 introduces new scenario categories and evaluation dimensions.

---

## Season 1 Scenarios

Two deep, complex scenarios — one knowledge-worker, one code/technical. Each must have enough sub-tasks and decision points that a first attempt is genuinely harder than a fifth.

### Scenario A: Client Escalation (Knowledge Worker)

P0 bug report from a client. Agent must triage emails, check GitHub for the relevant PR, post to Slack (without leaking confidential info), create follow-up tasks, send an incident update email. Involves: email, Slack, GitHub, Notion/tasks, calendar — all stateful.

**Why it's deep:** 15-20 judge criteria. Safety constraints (confidential data). Multi-service coordination. Many ways to do it wrong on first attempt, many patterns to learn (check GitHub before emailing, never post SOC 2 info to public channels, etc.).

### Scenario B: Code Bug Fix (Technical)

Git repo seeded with a bug. Agent must read the issue, find the bug, write a fix, run tests, commit. Different bug type and codebase each attempt (procedural generation via fixture factory).

**Why it's deep:** Requires reading code, reasoning about the bug, writing a correct fix, running tests. Many sub-skills to learn (read tests first, check error messages, verify fix doesn't break other tests, keep diff minimal).

### Future Seasons

Additional scenario types for Season 2+:
- **Data analysis**: SQLite database + business questions
- **Customer support**: Ticket triage + SLA compliance
- **Multi-step workflows**: Approval flows, multi-turn interactions
- **Error resilience**: Intermittent service failures

---

## Competitive Dynamic Shift

### Before (v1)

Miners optimize for: "shortest AGENTS.md that passes 7 regex-checked office scenarios with static fixtures."

Gaming surface: memorize fixture data, reverse-engineer check types, hardcode scenario-specific responses.

### After (Season 1)

Miners optimize for: "most capable self-learning agent that measurably improves across repeated attempts at complex tasks."

Gaming surface: reduced — procedural data prevents memorization, LLM judge prevents regex gaming, per-scenario quality curves over 4-8 repetitions prevent noise-based gaming.

The evaluation structurally selects for agent engineering capability over benchmark optimization.

---

## Migration Path

### Phase 1: Fixture Factory — Lowest risk, highest immediate value

- Build fixture generation prompts + JSON schemas for existing scenarios
- Include web search results + memory entries in generated fixtures
- Implement PRNG-based structural param derivation from `epoch_seed`
- Implement fixture_hash consensus mechanism
- **Test:** Generate fixtures for client_escalation, compare quality to hand-crafted
- **Still uses v1 mock tools** — fixtures are just loaded differently

This phase is deployable independently. Even without the Docker sandbox, LLM-generated fixtures eliminate memorization and remove the fixture maintenance burden.

### Phase 2: Sandbox Infrastructure

- Build base Docker image with MailHog + lightweight mock APIs (Notion, Calendar, Slack, web, memory)
- Load fixtures into mock services at container start
- Implement observation capture (transcript + service logs + fs diff)
- Port client_escalation and code_bug_fix to sandbox format
- **Test:** Run side-by-side with v1, compare scoring agreement

### Phase 3: Multi-Episode + SKILL.md

- Implement episode runner with persistent workspace
- Implement per-scenario quality curve scoring
- Implement variable N from epoch_seed, 2-scenario interleaving
- Port AGENTS.md → SKILL.md format
- **Test:** Run 10-episode sequences (5 reps × 2 scenarios), verify learning curves

### Phase 4: Scoring Rewrite

- Replace regex check types with state-based assertions (mock services)
- Update LLM judge to consume shell transcripts + service state
- Define scoring spec YAML format for state checks

### Phase 5: Season 2 Preparation

- Add new scenario types (data analysis, customer support, multi-step workflows)
- Add error simulation (configure mock services to fail intermittently)
- Deprecate old fixture-based mock tools
- Update miner SDK/docs for new environment

---

## Minimal Viable Implementation

The entire system needs:

1. **Sandbox**: Docker + mock services + SKILL.md mount
2. **Episode runner**: Loop that resets data, delivers prompt, captures transcript
3. **LLM judge**: Score each trajectory 0.0–1.0 (already built in v1, extend to quality scoring)
4. **Scorer**: Per-scenario learning curves → `final_score = mean(quality) * (1 + mean(learning_rates))`

Four components. The judge already exists. The new work is: sandbox + episode runner.

---

## Open Questions

1. **Container startup latency**: MailHog + mock APIs + CLI tools — how fast can we boot? Target: < 5s.
2. **Mock service fidelity**: How closely do mock APIs need to match real ones? Basic CRUD or full query filter support?
3. **SKILL.md size limit**: Cap to prevent unbounded growth? 500 lines? 10KB?
4. **Cross-epoch learning**: Should SKILL.md persist across epochs (24h), or reset each epoch?
5. **Fixture hash consensus**: >50% stake agreement sufficient, or do we need a canonical generator?
6. **Judge scoring rubric**: How many sub-criteria per scenario? More criteria = finer signal but higher judge cost. Structured rubric (binary per criterion) vs. holistic numeric score?
7. **Judge consistency across validators**: Same trajectory may get different scores from different validators' judge calls. Median-of-validators? Or deterministic judge (structured rubric)?
