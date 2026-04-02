# ClawBench v2: Sandbox Architecture

> Internal design document — next-season redesign of ClawBench and mock tooling.

## Status Quo: What's Wrong with v1

### 1. Stateless Mock Tools — The Fundamental Lie

The biggest architectural problem. Tools don't reflect mutations:

- Agent sends email → `himalaya envelope list` still returns the original inbox
- Agent creates a Notion task → next `databases/query` returns the same tasks
- Agent deletes calendar event → event is still there on next read

We can only score the agent's *intent* (did it try to send?), not its *competence* at handling state transitions. A real agent that sends an email and then confirms delivery would look like a hallucinator because the mock never changes state. **Multi-step workflows where step 2 depends on step 1 are fundamentally untestable.**

### 2. `exec` Is a God-Function with Brittle Regex

The exec handler is a ~170-line chain of regex patterns covering 4 completely different systems (email, tasks, calendar, GitHub):

- **Command variation kills agents**: `himalaya envelope list` works, `himalaya list envelope` doesn't. Flags in unexpected positions break the match.
- **Pattern order is the API contract**: First match wins. Undocumented, untested priority ordering.
- **No argument parsing**: Real CLIs have complex flag/option semantics; the mock just regex-matches the whole string.
- **Creative agents are punished**: An agent that uses `curl` to hit the email API directly instead of `himalaya` gets a generic fallback, even though the approach is equally valid.

This creates a **narrow corridor of "correct" commands** that rewards pattern memorization over genuine capability. Miners learn which exact command strings the mock accepts, not how to be good agents.

### 3. Fixture-Scenario Tight Coupling

Each scenario is an isolated island with manually crafted fixture files:

- **No composability**: Can't mix fixtures across scenarios to create new ones.
- **Manual data maintenance**: Adding a scenario means hand-writing 5-10 JSON fixture files with realistic, cross-referenced data. Slow and error-prone.
- **No parameterized variation**: Can't test "same scenario but 50 emails instead of 5." Only variation is the user persona (name/role/company) — superficial.
- **No schema enforcement**: A typo in a fixture (`"subjct"` instead of `"subject"`) silently breaks a scenario.

### 4. Single-Turn, Single-Episode Limitation

Every episode: one user message → agent tool-calls → one final response. Can't express:

- **Approval flows**: "Draft this email, wait for my approval, then send"
- **Clarification dialogues**: Agent asks follow-up, user responds
- **Long-running tasks**: Agent hits a blocker, reports back, user provides more info
- **Follow-up evaluation**: "Good brief, now add the budget numbers"

These are core knowledge-worker patterns the benchmark can't test. Miners have zero incentive to handle multi-turn interactions.

### 5. No Error Simulation

Tools always succeed (or return generic 404 for missing fixtures). No rate limits, auth failures, timeouts, partial responses, service degradation. We can't evaluate agent **robustness** — only happy-path behavior.

### 6. Knowledge-Worker Monoculture

All 7 scenarios are office-worker email/calendar/tasks/Slack. No code tasks, data analysis, customer support, research, creative work. Miners over-specialize for one narrow domain. The "cost competition" is really "who can write the shortest AGENTS.md that passes these 7 office scenarios."

### 7. Validator-ClawBench Coupling Is Fragile

- Subprocess-based with implicit JSON contract (parses **last line** as JSON)
- Any stray print/log to stdout breaks parsing
- Judge has hardcoded formatting for each tool type — adding a new tool requires judge code changes
- Workspace path, env var names, flag contracts are all implicit, no schema

### 8. Fixed Fixture Pool Enables Memorization

Miners can read every fixture email/task/event (open-source ClawBench). They see exactly which criteria exist. With only 7 scenarios and static data, miners can and do memorize the benchmark. The LLM integrity judge catches blatant gaming, but subtle optimization-through-memorization is structurally indistinguishable from genuine capability.

### 9. No Cost Measurement for Tool Calls

Cost only tracks LLM token usage. Real deployments have API rate limits, network latency, and service charges. A pack making 20 tool calls at $0.02 LLM cost isn't necessarily cheaper than 3 tool calls at $0.03 when you factor in real API overhead.

### 10. Toy Memory Implementation

`memory_search` does line-by-line keyword matching on markdown files. Agents that build sophisticated retrieval strategies get no benefit — the mock always returns the same keyword-matched lines.

---

## The Core Idea: Docker Sandbox Evaluation

Instead of mock tool handlers that regex-match commands and return static fixtures, **the agent SSHs/execs into a prepared Docker sandbox** where real (mock) services run with real protocols and stateful behavior.

### Current Architecture

```
Agent → OpenClaw API → mock handler → regex match → static fixture → canned response
```

### v2 Architecture

```
Agent → SSH/exec into Docker → real shell → real (mock) services → stateful environment
```

---

## Why This Changes Everything

### The Exec God-Function Dies

No more regex matching. The agent runs **real commands in a real shell**. Want to check email?

- `himalaya envelope list` (CLI)
- `curl localhost:1080/api/v2/messages` (HTTP API)
- `python3 -c "import imaplib; ..."` (programmatic)
- `cat /var/mail/user/new/*` (raw maildir)
- Chain commands with pipes, write scripts, use `jq`

All valid. All produce real results. The benchmark tests **can the agent accomplish the task**, not **does the agent know our exact mock API**.

### Statefulness Comes for Free

Agent sends email → it actually appears in the mock SMTP server's mailbox. Agent creates a task → the mock API's database has a new row. Agent deletes a calendar event → it's gone.

Scoring becomes: **inspect the final state of the environment**. Not "did the agent call the right regex pattern," but "is there actually an email in the sent folder addressed to Dana with the right subject?"

### Memorization Becomes Nearly Impossible

With real services and procedural fixture generation:

- Seed different data each eval (deterministic from `epoch_seed`)
- Same scenario structure, completely different emails/tasks/people
- Agent can't hardcode "read msg_003" because msg_003 doesn't exist this time
- There might be 15 messages with different IDs, different senders, different urgency

### The "Narrow Corridor" Opens Wide

Current: only ONE valid way to read email (exact `himalaya` command string).

Sandbox: agent picks its own approach. The mock services respond to **real protocols**, not pattern-matched strings. Creative, efficient agents are rewarded instead of punished.

---

## Sandbox Container Architecture

```
Docker Container ("eval sandbox")
├── Mock Services (stateful, real protocols)
│   ├── MailHog/MailPit       (SMTP :1025, HTTP API :1080) — email
│   ├── Mock Notion API       (HTTP :8080) — tasks / databases
│   ├── Mock Calendar API     (CalDAV :5232 or HTTP :8081)
│   ├── Mock Slack API        (HTTP :8082) — channels, messages
│   ├── Mock GitHub / Gitea   (HTTP :3000) — repos, PRs, issues
│   └── Mock Web Server       (HTTP :8083) — web_search / web_fetch targets
│
├── CLI Tools (pre-installed)
│   ├── himalaya, gh, curl, jq, python3, git, etc.
│   └── ~/.config/ pre-configured to point at local mock services
│
├── Workspace
│   ├── /workspace/AGENTS.md   (miner's pack)
│   ├── /workspace/...         (pack files)
│   └── /workspace/docs/       (scenario-specific reference docs)
│
├── Seed Data (scenario-specific, procedurally generated)
│   ├── Pre-loaded emails in MailHog
│   ├── Pre-loaded tasks in mock Notion
│   ├── Pre-loaded calendar events
│   └── Pre-loaded Slack channel history
│
└── Security
    ├── No external network access (iptables DROP egress)
    ├── CPU / memory / disk limits
    └── Hard timeout per episode
```

---

## Evaluation Flow

```
1. Build sandbox image
   - Base image: mock services + CLI tools
   - Layer: scenario seed data (procedurally generated from epoch_seed)
   - Layer: miner's pack files into /workspace/

2. Start container (isolated network, resource limits)

3. Agent gets exec access (SSH or docker exec)

4. Deliver user prompt: "Give me my morning brief"

5. Agent explores environment, runs commands, interacts with services
   - Full shell transcript captured (script/typescript)
   - Mock service request logs captured (every HTTP call)
   - File system changes tracked (before/after snapshot)

6. Agent outputs final response

7. Validator inspects:
   a. Shell transcript          → trajectory for LLM judge
   b. Service state snapshots   → outcome-based scoring
   c. Workspace file changes    → any artifacts created?
   d. Final response quality    → judge evaluates

8. LLM judge scores trajectory + final state

9. Tear down container, collect cost metrics
```

---

## Scoring: State-Based Instead of Intent-Based

### Current (Intent-Based)

> "Did the agent call `exec` with arg matching `/himalaya.*send/`?"

### v2 (Outcome-Based)

> "Query MailHog API — is there an email from `user@company.com` to `dana@acme.com` with subject containing 'incident update'?"

### Example Scoring Spec

```yaml
scoring:
  state_checks:
    - service: email
      query: "GET /api/v2/search?kind=to&query=dana@acme.com"
      assert:
        count: ">= 1"
        items[0].Content.Headers.Subject: contains("incident")

    - service: notion
      query: "GET /databases/tasks/query"
      assert:
        results: "length >= 3"
        results[*].properties.Status: not_contains("Duplicate")

    - service: slack
      query: "GET /channels/engineering/messages"
      assert:
        latest.text: contains("P0")
        latest.text: not_contains("SOC 2")   # safety: no confidential data leaked

  response_checks:
    - type: contains_all
      values: ["calendar conflict at 4pm", "auth migration blocked"]

  trajectory_checks:
    - type: no_bruteforce
      description: "Agent should not blindly iterate all message IDs"
      max_similar_commands: 10
```

This is far more expressive than the current 13 check types and much harder to game.

---

## What This Unlocks: New Scenario Categories

### Code Tasks

Seed the sandbox with a git repo containing a bug. Agent must find the bug, fix it, run tests, commit.

**Score**: do tests pass? Is the diff minimal and correct?

### Data Analysis

Seed a SQLite/Postgres database with business data. Agent must query, analyze, produce a summary.

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

## Key Design Decisions

### 1. Agent Access Model

| Option | Description | Trade-off |
|--------|-------------|-----------|
| **Pure shell** | Agent gets SSH/exec, does everything via commands | Most powerful, hardest for LLMs, most realistic |
| **Structured tools + shell escape** | Agent has high-level tools (email.list, slack.send) that internally call sandbox services, plus a `shell` tool for arbitrary commands | Easier for LLMs, still flexible |
| **Hybrid** | Agent chooses: use structured API OR drop to shell | Best of both, more complex to implement |

### 2. Sandbox Lifecycle

| Option | Description | Trade-off |
|--------|-------------|-----------|
| **Per-scenario** | Fresh container per scenario | Clean isolation, slower (startup overhead) |
| **Per-miner** | One container, all scenarios run sequentially | Faster, but state leaks between scenarios |
| **Snapshot-based** | One base container, checkpoint/restore per scenario | Fast + isolated, requires CRIU or similar |

### 3. Observation Capture

Three data sources to feed the judge:

1. **Shell transcript** — full command history + outputs (`script` / `typescript` capture)
2. **Service request logs** — every HTTP call to mock services (structured JSON)
3. **Filesystem diff** — before/after snapshot of workspace

All three combined give the judge a rich, complete picture of what the agent did and what resulted.

### 4. Procedural Seed Generation

```
epoch_seed (deterministic)
    ↓
Scenario template + seed
    ↓
Generate: 12 emails (3 urgent, 2 mention {client}, 1 confidential)
Generate: 8 calendar events (1 conflict, 2 past, 5 future)
Generate: 15 Notion tasks (3 overdue, 2 blocked, 10 in-progress)
Generate: Slack history (2 channels, ~20 messages each)
    ↓
Load into mock services at container start
```

All validators derive the same data from the same `epoch_seed`. Different epochs test different data. Same structure, different content. Memorization structurally eliminated.

### 5. Security Isolation

- **Network**: `iptables -A OUTPUT -j DROP` — no external access from sandbox
- **Resources**: `--cpus=2 --memory=2g --storage-opt size=1g`
- **Time**: hard timeout per episode (configurable, e.g. 5 min)
- **Filesystem**: no access outside `/workspace` + service data directories
- **Secrets**: no cloud credentials, API keys, or host mounts in sandbox

---

## Competitive Dynamic Shift

### Before (v1)

Miners optimize for: "shortest AGENTS.md that passes 7 regex-checked office scenarios with static fixtures."

Gaming surface: memorize fixture data, reverse-engineer check types, hardcode scenario-specific responses.

### After (v2)

Miners optimize for: "most capable agent that can navigate a real environment with dynamic data across diverse task types."

Gaming surface: dramatically reduced — procedural data kills memorization, outcome-based scoring kills regex gaming, diverse scenarios kill over-specialization.

The miner population separates into **genuinely capable agent builders** vs. **memorization optimizers** — and the latter are structurally eliminated.

---

## Migration Path

### Phase 1: Sandbox Infrastructure

- Build base Docker image with mock services (MailHog, mock APIs)
- Implement procedural seed generation from `epoch_seed`
- Implement observation capture (transcript + service logs + fs diff)
- Port 2-3 existing scenarios (morning_brief, client_escalation) to sandbox format

### Phase 2: Scoring Rewrite

- Replace regex check types with state-based assertions
- Update LLM judge to consume shell transcripts + service state
- Define scoring spec YAML format for state checks

### Phase 3: Scenario Expansion

- Add code tasks (git repo + bug fix)
- Add data analysis (SQL database)
- Add multi-turn episodes
- Add error simulation

### Phase 4: Full Cutover

- Deprecate old fixture-based mock tools
- All scenarios run in sandbox
- Update miner SDK/docs for new environment

---

## Open Questions

1. **Container startup latency**: MailHog + mock APIs + CLI tools — how fast can we boot? Target: < 5s.
2. **Mock service fidelity**: How closely do mock APIs need to match real ones? (e.g., does mock Notion need full query filter support, or just basic CRUD?)
3. **Multi-turn protocol**: How does the sandbox deliver follow-up user messages? (File watch? Stdin pipe? HTTP callback?)
4. **Cost model**: Include sandbox compute time in miner cost, or keep it LLM-token-only?
5. **Backward compatibility**: Can existing packs (AGENTS.md targeting OpenClaw tool-call API) work in the sandbox with a compatibility shim?
