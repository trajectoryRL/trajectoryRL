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

2. **Agent-harness-agnostic.** The interface is: SSH into a sandbox, read SKILL.md, execute. The validator only sees cost per episode and PASS/FAIL.

   | Framework | How it consumes SKILL.md |
   |-----------|-------------------------|
   | Claude Code | Reads as `CLAUDE.md` |
   | Cursor | Reads as `.cursor/rules` |
   | OpenClaw | Reads as `AGENTS.md` |
   | Custom harness | `cat /workspace/SKILL.md` |
   | Raw LLM + bash | System prompt includes file |

   Miners compete on which agent framework learns best, not which prompt is cleverest. A miner running Claude Code competes directly against a miner running a custom Python harness.

3. **Resistant to gaming.** A single scenario result can be hacked. A consistent upward quality trend across N episodes cannot, because:
   - Task order is **permuted** each epoch
   - Episode count **varies** each epoch
   - Data is **different** each episode
   - Task types are **interleaved** (learning must transfer)

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

**No exec god-function.** The agent runs real commands in a real shell. `himalaya envelope list`, `curl localhost:1080/api/v2/messages`, `python3 -c "import imaplib; ..."` — all valid, all produce real results.

**Stateful by default.** Agent sends email → it appears in the mock SMTP server's mailbox. Scoring inspects the final state of the environment, not the commands used.

**Procedural data.** Fixture generation seeds different data each eval. Same structure, completely different content. Memorization is no longer a viable strategy.

**Protocol-level interface.** Mock services respond to real protocols, not pattern-matched strings. Agents are free to use any tool or method that speaks the protocol.

---

## Sandbox Container Architecture

```
Docker Container ("eval sandbox")
├── Tier 1: Deterministic Mock Services (stateful, real protocols)
│   ├── MailHog/MailPit       (SMTP :1025, HTTP API :1080) — email
│   ├── Mock Notion API       (HTTP :8080) — tasks / databases
│   ├── Mock Calendar API     (CalDAV :5232 or HTTP :8081)
│   ├── Mock Slack API        (HTTP :8082) — channels, messages
│   └── Mock GitHub / Gitea   (HTTP :3000) — repos, PRs, issues
│
├── Tier 2: LLM-Backed Runtime Mocks (read-only, on-the-fly generation)
│   ├── Web search/fetch proxy (HTTP :8083) — LLM generates search results & pages
│   └── Memory service         (HTTP :8084) — LLM generates memory entries
│   (Requires outbound access to LLM API only — all other egress blocked)
│
├── CLI Tools (pre-installed)
│   ├── himalaya, gh, curl, jq, python3, git, etc.
│   └── ~/.config/ pre-configured to point at local mock services
│
├── Workspace
│   ├── /workspace/SKILL.md    (miner's pack — PERSISTS across episodes)
│   ├── /workspace/learned/    (agent's learning store — PERSISTS)
│   ├── /workspace/...         (pack files)
│   └── /workspace/docs/       (scenario-specific reference docs)
│
├── Seed Data (LLM-generated from scenario template + epoch_seed, Tier 3)
│   ├── Pre-loaded emails in MailHog
│   ├── Pre-loaded tasks in mock Notion
│   ├── Pre-loaded calendar events
│   └── Pre-loaded Slack channel history
│
└── Security
    ├── Network: egress blocked EXCEPT validator's LLM API (for Tier 2 mocks)
    ├── CPU / memory / disk limits
    └── Hard timeout per episode
```

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
  2. Score: LLM judge PASS/FAIL
  3. Reset: reload Tier 1 services with new fixtures
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

---

## LLM-Hybrid Mock Strategy

### The Three-Tier Architecture

```
Tier 1: Deterministic Mock Services (stateful, scoring inspects state)
        → Email (MailHog/MailPit), Tasks, Calendar, Slack, GitHub/Gitea
        → Agent mutations are real, state is inspectable after episode
        → Seed data: LLM-generated (Tier 3), loaded before episode

Tier 2: LLM-Backed Runtime Mocks (read-only, scoring checks outcomes)
        → web_search, web_fetch, memory_search
        → LLM generates realistic responses on the fly during episode
        → No state to inspect — scoring evaluates agent's final output

Tier 3: LLM Fixture Factory (build-time generation)
        → Generates all seed data for Tier 1 services
        → Replaces hand-crafted JSON fixture files
        → Deterministic distribution via hash consensus
```

### Why Three Tiers?

| Service | Mutations? | Scoring Method | Tier |
|---------|-----------|----------------|------|
| Email | Yes (send, delete, move) | State inspection: "is there an email to Dana?" | 1 |
| Tasks/Notion | Yes (create, update, close) | State inspection: "are there 3 new tasks?" | 1 |
| Calendar | Yes (create, delete events) | State inspection: "is the conflict resolved?" | 1 |
| Slack | Yes (send messages, react) | State inspection: "was P0 posted to #engineering?" | 1 |
| GitHub/Gitea | Yes (commits, PRs, issues) | State inspection: "do tests pass? Is PR merged?" | 1 |
| Web search | No (read-only) | Response quality: "did agent find the right info?" | 2 |
| Web fetch | No (read-only) | Response quality: "did agent use page content?" | 2 |
| Memory | No (read-only) | Response quality: "did agent leverage past context?" | 2 |
| Filesystem | Yes (create, edit files) | File diff: deterministic | 1 |

**Stateful services where scoring inspects state → deterministic mock (Tier 1).**
**Read-only information services where scoring checks agent output → LLM-backed (Tier 2).**

### Tier 2: LLM-Backed Runtime Mocks

During an episode, the agent can freely query web/memory tools. Instead of matching against fixture files, an LLM generates realistic responses:

```
Agent: web_search("notion API batch operations")
  → Sandbox web service intercepts
  → LLM generates 5 realistic search results
  → Returns to agent

Agent: memory_search("previous meeting notes with Dana")
  → LLM generates relevant memory entries consistent with scenario context
  → Returns to agent
```

**What this unlocks:** agent can search for *anything*, no fixture files to maintain, results are contextually appropriate, new scenarios don't need new web fixtures.

**Anti-gaming:** The mock LLM has a hard system prompt constraining it to generate realistic service responses, not direct answers. It never sees scoring criteria or scenario requirements.

### Tier 3: LLM Fixture Factory

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
  → Loaded into Tier 1 mock services at container start
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

The judge produces a quality score per episode (0.0–1.0) covering correctness, completeness, and safety. State-based checks (Tier 1 service inspection) serve as grounding evidence for the judge — not as the scoring mechanism itself.

### Across Episodes: Quality Trajectory (Learning Signal)

The learning signal is whether quality scores improve over episodes:

- Flat at low quality = not learning (no reward)
- Flat at high quality = already capable but not improving (moderate reward)
- Upward trend = genuinely learning from experience (high reward)
- Steep upward trend = fast learner (maximum reward)

```
episode_scores = [judge_quality(trajectory_i) for i in 1..N]
learning_rate  = linear_regression_slope(episode_scores)
final_score    = mean(episode_scores) * (1 + max(0, learning_rate))
```

High quality AND improving = win.

---

## Evaluation Flow

```
1. Build sandbox (mock services + CLI tools)
2. Load miner's SKILL.md into /workspace/
3. Derive sequence from epoch_seed:
   - N = rng.randint(8, 16)            ← variable length
   - sequence = rng.sample(pool, N)    ← permuted order
4. For episode i = 1..N:
   a. Reset mock service data (new Tier 3 fixtures from epoch_seed + i)
   b. Deliver task prompt: sequence[i]
   c. Agent runs: reads SKILL.md → does task → updates SKILL.md
   d. Capture: shell transcript + mock service state
   e. Judge: LLM evaluates trajectory → quality score (0.0–1.0)
5. Tear down sandbox
6. Score:
   - final_score = mean(quality_scores) * (1 + learning_rate)
```

**Quality:** LLM judge scores each trajectory. **Learning:** quality trend across episodes.

---

## Episode Sequence Design

### Permuted Order + Variable Length

The episode sequence is **shuffled from `epoch_seed`** and the episode count **varies**. A miner cannot predict what task comes at which position, or how many episodes there will be.

```python
rng = Random(epoch_seed)

# Variable episode count: 8-16
task_pool = scenarios * 2  # each scenario appears ~2x
N = rng.randint(8, 16)
sequence = rng.sample(task_pool, N)

# Each episode gets unique data
for i, scenario in enumerate(sequence):
    fixtures = generate(epoch_seed + i, scenario)
```

**Example: Epoch A might produce (N=10):**

```
 E1:  client_escalation   (data_seed_1)
 E2:  morning_brief       (data_seed_2)
 E3:  inbox_triage        (data_seed_3)
 E4:  team_standup        (data_seed_4)
 E5:  morning_brief       (data_seed_5)    ← 2nd time, should improve
 E6:  client_escalation   (data_seed_6)    ← 2nd time, should improve
 E7:  hiring_debrief      (data_seed_7)
 E8:  inbox_triage        (data_seed_8)    ← 2nd time, should improve
 E9:  post_incident       (data_seed_9)
E10:  morning_brief       (data_seed_10)   ← 3rd time, should be cheapest
```

**Epoch B produces a completely different sequence (N=13):**

```
 E1:  inbox_triage        (data_seed_1)
 E2:  post_incident       (data_seed_2)
 E3:  morning_brief       (data_seed_3)
 ...different order, different length, different data...
```

### Key Properties

- **Permuted order**: can't optimize for "task X always comes at position Y"
- **Variable N**: can't optimize for "be cheap on the last 4 episodes"
- **Each task type appears 2-3 times** with different data
- **Interleaving**: learning must persist across different task types
- **Transfer**: learning from inbox_triage might help morning_brief

---

## What the Validator Sees

```json
{
  "miner_uid": 42,
  "pack_hash": "abc123...",
  "episodes": [
    {"id": 1, "scenario": "morning_brief",    "quality": 0.55},
    {"id": 2, "scenario": "inbox_triage",      "quality": 0.60},
    {"id": 3, "scenario": "client_escalation", "quality": 0.52},
    {"id": 4, "scenario": "morning_brief",     "quality": 0.78},
    {"id": 5, "scenario": "team_standup",      "quality": 0.81},
    {"id": 6, "scenario": "inbox_triage",      "quality": 0.85}
  ],
  "mean_quality": 0.685,
  "learning_rate": 0.054,
  "final_score": 0.722
}
```

The quality trajectory is the complete evaluation output.

---

## Anti-Gaming Analysis

Four mechanisms work together:

| Mechanism | What it prevents |
|-----------|-----------------|
| Varying data (Tier 3 fixtures) | Memorization of specific answers |
| Permuted episode order | Position-based optimization |
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

3. **Procedural data generation (Tier 3).** LLM-generated fixtures from seed + template eliminates memorization and removes the fixture maintenance burden. Deployable independently — even without the sandbox, this improves v1.

4. **State-based scoring evidence.** Checking MailHog API for "is there an email to Dana?" instead of regex-matching `himalaya send` is a qualitative leap. Agents are free to use any tool or method. The scoring sees outcomes, not commands.

5. **LLM judge as universal scorer.** Already proven in v1. Extending it from PASS/FAIL to quality scoring is incremental, not architectural. The judge is the one component that doesn't need to be built from scratch.

The primary unknowns are in the **multi-episode learning signal** (how to reliably measure improvement with small N) and **cross-validator determinism** (Tier 2 mocks, judge variance). These are addressed in the risks section below.

---

## Known Risks & Mitigations

### 1. Weak learning signal with small N

With N=8-16 and each scenario appearing ~2 times, the quality trajectory has few data points. A genuine learning agent may show improvement on only 3-4 repeated scenarios — the rest are first encounters that contribute noise.

**Mitigation:** Increase the minimum N, or compute per-scenario quality deltas (compare same scenario across repeats) rather than relying on global regression slope. Per-scenario comparison is a stronger signal with less noise.

### 2. SKILL.md growth vs. quality trade-off

As SKILL.md accumulates patterns, it grows. A larger SKILL.md means the agent spends more tokens reading context before each task. Without pruning, later episodes may degrade as context bloat reduces agent effectiveness.

**Mitigation:** This is intentionally part of the competition — miners must engineer SKILL.md with pruning and compression strategies. A size cap (Open Question #3) provides a hard bound.

### 3. LLM judge variance across validators

The LLM judge is probabilistic. Validator A and Validator B may score the same trajectory differently. With N=12 episodes per miner, small per-episode variance compounds into meaningful disagreement on `mean_quality` and `learning_rate`.

**Mitigation:** Use structured rubrics with binary sub-criteria (PASS/FAIL per criterion) rather than a single numeric score. Binary judgments are more reproducible across LLM calls. Quality score = fraction of criteria passed. Alternatively, use median-of-validators scoring.

### 4. Tier 2 non-determinism across validators

LLM-backed runtime mocks (web_search, memory) produce different responses for different validators. The agent takes different paths, leading to different trajectories and potentially different PASS/FAIL outcomes.

**Mitigation:** Defer Tier 2 to a later phase. Start with Tier 1 (deterministic) + Tier 3 (fixture factory) only. Or pre-generate Tier 2 responses at build time (making them effectively Tier 3) and cache by fixture_hash.

### 5. Evaluation cost and time

Each miner requires 8-16 episodes × (agent runtime + judge call). At ~3 min/episode, a 12-episode eval takes ~36 min per miner. With 50 miners and 10 parallel containers: ~3 hours per epoch.

**Mitigation:** This is within the 24h epoch window. Parallel containers scale linearly. Judge calls can be batched. LLM cost per epoch (~$30-50 at current rates) is manageable for validators earning TAO emissions.

### 6. The "already good" problem

A miner whose SKILL.md produces high-quality trajectories from episode 1 shows no improvement slope (`learning_rate ≈ 0`). The formula `mean(quality) * (1 + learning_rate)` still rewards high absolute quality, but does not distinguish "consistently excellent" from "mediocre but slightly improving."

**Mitigation:** The formula handles this — `mean(quality)` dominates when `learning_rate` is near zero. A miner scoring 0.95 across all episodes gets `final_score = 0.95`, which beats a miner improving from 0.4 to 0.7 (`mean=0.55, rate=0.05, final=0.578`). The learning bonus rewards improvement but doesn't override raw quality.

### 7. Miner meta-game evolution

**Month 1-2:** Basic SKILL.md files ("reflect after each task"). Low differentiation. Most miners produce similar quality trajectories.

**Month 3-4:** Top miners discover that SKILL.md structure matters — concise pattern storage, good categorization, pruning instructions. Separation emerges.

**Long-term:** Competition converges on optimal memory management strategies. The eval must add new scenario types (Phase 6) to maintain competitive pressure.

**Mitigation:** The season model is designed for this — Season 1 runs until the meta stabilizes, then Season 2 introduces new scenario categories and evaluation dimensions.

---

## What This Unlocks: New Scenario Categories

### Code Tasks

Seed the sandbox with a git repo containing a bug. Agent must find the bug, fix it, run tests, commit.

**Score**: do tests pass? Is the diff minimal and correct?

### Data Analysis

Seed a SQLite database with business data. Agent must query, analyze, produce a summary.

**Score**: are the numbers accurate? Did the query return correct results?

### Customer Support

Seed with a ticket system (mock Zendesk/Linear). Agent must triage, respond, escalate.

**Score**: check ticket states + response quality + SLA compliance.

### Multi-Step Workflows

Agent reads email → asks user for approval (multi-turn) → creates draft → user approves → agent sends. The sandbox holds real state throughout.

### Error Resilience

Configure mock services to fail intermittently. Email service returns 503 on first attempt. Calendar API has 2-second latency.

**Score**: did the task eventually succeed despite errors?

---

## Competitive Dynamic Shift

### Before (v1)

Miners optimize for: "shortest AGENTS.md that passes 7 regex-checked office scenarios with static fixtures."

Gaming surface: memorize fixture data, reverse-engineer check types, hardcode scenario-specific responses.

### After (Season 1)

Miners optimize for: "most capable self-learning agent that improves across diverse tasks in a real environment."

Gaming surface: reduced — procedural data prevents memorization, outcome-based scoring prevents regex gaming, cost curve prevents snapshot optimization, diverse scenarios prevent over-specialization.

The evaluation structurally selects for agent engineering capability over benchmark optimization.

---

## Migration Path

### Phase 1: LLM Fixture Factory (Tier 3) — Lowest risk, highest immediate value

- Build fixture generation prompts + JSON schemas for existing scenarios
- Implement PRNG-based structural param derivation from `epoch_seed`
- Implement fixture_hash consensus mechanism
- **Test:** Generate fixtures for morning_brief, compare quality to hand-crafted
- **Still uses v1 mock tools** — fixtures are just loaded differently

This phase is deployable independently. Even without the Docker sandbox, LLM-generated fixtures eliminate memorization and remove the fixture maintenance burden.

### Phase 2: Sandbox Infrastructure (Tier 1)

- Build base Docker image with MailHog + lightweight mock APIs (Notion, Calendar, Slack)
- Load Tier 3 fixtures into mock services at container start
- Implement observation capture (transcript + service logs + fs diff)
- Port morning_brief and client_escalation to sandbox format
- **Test:** Run side-by-side with v1, compare scoring agreement

### Phase 3: LLM Runtime Mocks (Tier 2)

- Build web search/fetch proxy with LLM backend
- Build memory service with LLM backend + session cache
- Harden system prompts against prompt injection
- **Test:** Measure mock quality, latency, cost per episode

### Phase 4: Multi-Episode + SKILL.md

- Implement episode runner with persistent workspace
- Implement cost curve scoring (learning_efficiency + final_score)
- Implement permuted sequence + variable N from epoch_seed
- Port AGENTS.md → SKILL.md format
- **Test:** Run 10-episode sequences, verify cost curve measurement

### Phase 5: Scoring Rewrite

- Replace regex check types with state-based assertions (Tier 1 services)
- Update LLM judge to consume shell transcripts + service state
- Define scoring spec YAML format for state checks

### Phase 6: Scenario Expansion + Full Cutover

- Add code tasks (git repo + bug fix) — leverages Gitea (Tier 1)
- Add data analysis (SQLite + LLM-generated business data)
- Add error simulation (configure Tier 1 services to fail intermittently)
- Deprecate old fixture-based mock tools
- Update miner SDK/docs for new environment

---

## Minimal Viable Implementation

The entire system needs:

1. **Sandbox**: Docker + mock services + SKILL.md mount
2. **Episode runner**: Loop that resets data, delivers prompt, captures transcript
3. **LLM judge**: Score each trajectory 0.0–1.0 (already built in v1, extend to quality scoring)
4. **Scorer**: `final_score = mean(quality_scores) * (1 + learning_rate)`

Four components. The judge already exists. The new work is: sandbox + episode runner.

---

## Open Questions

1. **Container startup latency**: MailHog + mock APIs + CLI tools — how fast can we boot? Target: < 5s.
2. **Mock service fidelity**: How closely do mock APIs need to match real ones? Basic CRUD or full query filter support?
3. **SKILL.md size limit**: Cap to prevent unbounded growth? 500 lines? 10KB?
4. **Cross-epoch learning**: Should SKILL.md persist across epochs (24h), or reset each epoch?
5. **Harness specification**: How does the validator know which agent harness to run? Miner specifies in pack metadata?
6. **Tier 2 LLM model choice**: Same model as the judge? Smaller/cheaper?
7. **Fixture hash consensus**: >50% stake agreement sufficient, or do we need a canonical generator?
8. **Tier 2 session consistency**: How to prevent contradictions across multiple LLM-mock calls? Cache-only, or session context?
9. **Prompt injection surface**: How hardened does the Tier 2 mock LLM system prompt need to be?
10. **Judge scoring granularity**: Binary PASS/FAIL vs. continuous 0.0–1.0? Continuous is richer but noisier across validators.
11. **Judge consistency across validators**: Same trajectory may get different scores from different validators' judge calls. Median-of-validators? Or deterministic judge (structured rubric)?
