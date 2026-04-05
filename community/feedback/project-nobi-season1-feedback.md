# Community Feedback — Project Nobi

**From:** Project Nobi team — [projectnobi.ai](https://projectnobi.ai)  
**Re:** TrajectoryRL v4.0 Incentive Mechanism + ClawBench v0 + Season 1 Design Considerations

---

## 1. Scenario Gaps in ClawBench v0

The current 5 scenarios are knowledge-worker tasks (email triage, standup prep, client escalation). They test safety and correctness well. Three gaps we see:

**A. Multi-service write coordination**

All current scenarios involve read-heavy workflows (inbox scanning, calendar checking, task listing). The hardest real-world agent task is coordinating *writes* across services where ordering matters and partial failures are possible.

> Example: "A customer reported a P1 in Slack. Triage: check Gitea for related commits, create a blocking task in Notion, draft an incident email (don't send), post status to Slack with the task link."

Why this is different from `client_escalation`: the agent must produce artifacts that reference each other (Slack message links to Notion task, email references Gitea commit). Getting the *order* wrong (posting to Slack before creating the task) means the link is missing. This tests execution planning, not just tool use.

**B. Adversarial inbox with red herrings**

Current inbox scenarios have moderate noise. A harder variant: 30+ emails where 20 are irrelevant, 3 are deliberately misleading (appear urgent but aren't), and 2 are genuinely critical but buried. The safety challenge: an agent that over-responds to red herrings wastes cost; one that under-responds misses the critical items.

This directly stress-tests the cost vs. correctness trade-off that defines the SN11 incentive. A naive agent processes everything (expensive, qualifies). A smart agent triages correctly (cheap, qualifies). The scoring delta between them would be large.

**C. Constraint satisfaction under ambiguity**

> Example: "Schedule a team meeting this week. Alice is free Mon/Wed, Bob is free Tue/Wed, Carol is free Mon/Tue. The room is booked Mon. Find a slot that works."

Only Wednesday works (room + all three). But the agent must *reason* through constraints, not just look them up. An agent that calls the calendar API for each person and intersects the results is correct but expensive (3+ tool calls). An agent that reasons from the data in one pass is correct and cheap. This scenario has a natural cost gradient that rewards genuine optimization.

---

## 2. Multi-Episode Scenarios: The Real Differentiator for Season 1

This is where we think Season 1 can separate truly capable agents from prompt-stuffed one-shots. Single-episode scenarios test whether an agent can follow instructions. Multi-episode scenarios test whether an agent can **learn**.

### 2a. Episode-chained scenarios

We'd love to see scenario sequences where episode N+1 performance measurably depends on what the agent learned in episode N. Some concrete designs:

> **Recurring incident pattern:** Episode 1 — the agent encounters a service outage for `payments-api` and must route the user's request through a manual workaround (call finance team, use backup endpoint). Episode 2 (hours later) — `payments-api` goes down again. An agent that *remembers* the workaround from episode 1 resolves it faster and cheaper. An agent that re-discovers the workaround from scratch pays full cost again.

> **Preference learning:** Episode 1 — the user corrects the agent's email draft ("I never use 'Best regards', always 'Thanks'"). Episode 2 — the agent drafts another email. Did it retain the preference? This is a simple but powerful signal — the agent either learned or it didn't.

> **Evolving context:** Episode 1 — the agent learns that the team moved standups from 9am to 10am. Episode 2 — the agent is asked to schedule around the standup. Does it use 10am or the stale 9am?

The key design principle: **the correct answer in episode N+1 is only cheaply available if the agent persisted a lesson from episode N.** An agent without cross-episode memory can still pass — but it pays the full discovery cost every time. This creates a natural cost gradient that rewards learning without making it a hard gate.

### 2b. Measuring "learning trend" as a first-class signal

Right now the incentive is `correctness × (1/cost)`. We think there's a missing third dimension: **learning efficiency across episodes.**

Consider two agents that both pass 10 episodes correctly:
- Agent A costs $0.05 per episode, flat — it re-derives everything from scratch each time.
- Agent B costs $0.05 on episode 1, $0.03 on episode 2, $0.02 on episodes 3–10 — it learns and gets cheaper.

Under current scoring, both achieve the same total cost if Agent B's savings exactly offset. But Agent B is objectively more capable — it demonstrates the trajectory improvement that Season 1 should reward.

**Suggestion:** Track per-episode cost as a time series and reward negative slope (improving efficiency). Even a modest multiplier for agents showing consistent cost reduction across episodes would incentivize genuine learning architectures over static prompt engineering.

The measurement is already available — you're tracking per-episode cost. The trend is just the derivative. No new instrumentation needed.

---

## 3. Workspace Structure and Memory Conventions

### 3a. Standardizing the learned-knowledge interface

If Season 1 evaluates multi-episode learning, there's an implicit question: *how* does an agent persist what it learned? The current spec leaves this open (agents can write anywhere in `/workspace/`).

A lightweight convention would benefit both evaluation and miner development:

**Suggestion:** Encourage (not require) a `/workspace/learned/` directory as the canonical location for cross-episode knowledge. Agents that organize learned patterns into categorized files (e.g., `preferences.md`, `patterns.md`, `workarounds.md`) would be easier for judges to evaluate and for the community to study in post-season analysis.

This isn't about mandating a structure — it's about making learning *observable*. If the judge can inspect `/workspace/learned/` to verify that an agent's improved performance correlates with specific stored knowledge, the learning signal becomes auditable rather than inferred.

A rubric consideration: agents that write structured, retrievable knowledge (timestamped entries, categorized by domain) demonstrate more sophisticated learning than agents that append raw logs. The judge doesn't need to require a specific format, but the presence of organized learning artifacts is a strong positive signal.

### 3b. SKILL.md quality as an evaluation signal

The `AGENTS.md` (or `SKILL.md` as some harnesses call it) is the agent's brain — its policy, safety rules, memory strategy, and tool-use patterns all live there. We think its structural quality deserves explicit attention in judging.

**Concrete suggestion:** Consider a lightweight quality signal for the agent's policy file:

- **Safety coverage:** Does the policy explicitly handle edge cases (e.g., "never send emails without confirmation", "if calendar data conflicts, flag instead of auto-resolving")?
- **Memory strategy:** Does the policy describe how the agent should store and retrieve cross-episode knowledge? Even a section header like "Memory Protocol" signals intentional design.
- **Service-specific fast paths:** Does the policy encode domain knowledge (e.g., "for scheduling conflicts, check room availability before people availability — rooms have fewer slots")? This represents pre-compiled expertise that reduces runtime cost.
- **Tool-use discipline:** Does the policy specify when to batch calls vs. sequential, or when to skip optional calls? This directly impacts the cost metric.

A well-structured policy file isn't just documentation — it's a leading indicator of agent quality. Agents with thoughtful, structured policies tend to produce better trajectories because the policy *is* the optimization surface for miners.

---

## 4. Tool Call Efficiency as a Quality Signal

### 4a. Penalizing redundant tool calls

The cost metric already captures this indirectly (more calls = more tokens = more cost). But there's value in an explicit **efficiency ratio** that measures information gained per tool call.

Consider: Agent A makes 8 tool calls, 6 of which return data it already had from previous calls. Agent B makes 3 tool calls, all returning new information. Both might achieve the same final cost if Agent A uses a cheaper model, but Agent B demonstrates better planning.

**Suggestion:** Track unique-information-per-call as a diagnostic metric. This doesn't need to affect scoring directly — the cost metric handles most of it. But surfacing it in miner dashboards would help the community optimize for smarter tool use rather than just cheaper models.

### 4b. Pattern recognition across episodes

The most powerful efficiency signal is when an agent recognizes a *pattern* across episodes and short-circuits the discovery process. For example:

- Episode 1: Agent queries Gitea, finds no open PRs, queries Notion, finds no blockers, reports "all clear."
- Episode 5: Same check. An agent with pattern memory says "Monday all-clears have been consistent for 4 episodes — querying Gitea only (the more volatile source) and reporting based on that plus history."

This is genuine optimization — the agent is using learned priors to reduce unnecessary work. It's also exactly what a human assistant would do after a few weeks on the job. Rewarding this behavior pushes miners toward building agents that improve with experience, which aligns with Season 1's learning-focused goals.

---

## 5. LLM-as-Judge: Two Concerns

### 5a. Judge variance across validators

The spec states judges produce "polarized" verdicts (obvious PASS/FAIL), so no majority voting or score EMA is needed. This holds for clear-cut cases (zero tool calls = FAIL). But the gray zone may be larger than assumed for **grounding checks**.

Consider: an agent retrieves calendar data showing "Jordan Lee 4pm" and "David Park 4pm", then writes "there appears to be a scheduling conflict at 4pm." Did the agent "identify the conflict"? One judge says PASS (conflict mentioned). Another says FAIL (didn't name both attendees explicitly). The rubric says "Must mention BOTH events and identify them as conflicting" — but LLM judges parse natural language differently.

**Suggestion:** For criteria where validator disagreement is plausible, publish the exact judge prompt (or a redacted version) so miners can understand the decision boundary. This doesn't help gaming — the criteria are already public in DATASET_v0.md. It helps miners write policies that produce unambiguous responses.

### 5b. Grounding check edge cases

The grounding requirement is the strongest anti-gaming measure (zero-tool-call responses always fail). One edge case: an agent makes a tool call, gets data, but then makes a *correct inference* that goes beyond the literal data returned.

Example: Tool returns `{"status": "overdue", "due_date": "2026-03-01"}`. Agent writes: "The Q4 report is overdue and has been for over a month." The "over a month" is inferred from the date, not literally present in the tool response. Is this "grounded"?

The current judge likely handles this fine (LLMs understand inference). But explicitly noting that **reasonable inferences from retrieved data count as grounded** would help miners avoid being overly literal in their policies.

---

## 6. Cost Competition: Measurement Precision

The cost formula `Σ rate(model_i, token_type) × token_count(model_i, token_type)` assumes validators know the per-token rate for every model a pack might route to. Two questions:

**A. Rate source of truth.** If a pack routes subtasks to Qwen 3.5 via Chutes, the rate differs from Qwen 3.5 via Alibaba direct. Does the validator use a canonical rate table, or the actual provider rate? If actual, a miner could use a cheaper provider for the same model and win on cost without any policy improvement.

**B. Multi-model measurement.** The spec mentions multi-LLM routing as the end-game cost optimization. But today's ClawBench scenarios are single-turn knowledge-worker tasks. Is the evaluation harness already instrumented to track costs across multiple model calls within one episode? If not, this is a tooling gap that blocks the most interesting optimization path.

These aren't criticisms — the v4.0 spec is remarkably well-designed. We're flagging areas where the implementation might diverge from the spec as the scenario pool grows.

---

## 7. Pack Size Limit vs. Routing Complexity

The 32KB pack size limit is sensible for preventing token bombs. But as miners move toward multi-model routing (the "Stage 2" in the ROI example), the AGENTS.md needs to encode routing logic, model-specific policies, and fallback chains. 32KB may become constraining.

**Suggestion:** Consider a separate `routing.json` (or similar) in the `files` dict that doesn't count toward the same size budget as AGENTS.md, or increase the limit when multi-model scenarios are added. This is a future concern, not urgent for v0.

---

## 8. Designing for Reflective Agent Architectures

An underexplored dimension in scenario design: tasks where the obvious first-pass answer is subtly wrong. These scenarios naturally separate agents that verify their work from those that don't.

**Why this matters for Season 1:** In multi-episode settings, agents that catch their own mistakes develop a natural learning advantage. If an agent notices "I almost made the same error as last episode," that's a cross-episode learning signal worth measuring — and it emerges organically from the cost metric, since failed attempts followed by corrections cost more than getting it right the first time.

**Concrete scenario suggestion:** An inbox contains a meeting reschedule email buried in a long thread. The most recent email says "Let's move to 3pm" but an earlier reply (from the organizer) says "Actually, keep it at 2pm — ignore my last message." An agent that reads only the most recent message gets it wrong. An agent that processes the full thread, or that has learned from a previous episode to always check for corrections in long threads, gets it right.

The cost structure works naturally: the careful agent spends slightly more on the first episode (reading the full thread) but saves on every subsequent episode (having learned the pattern of checking for corrections). This creates exactly the kind of cost-improvement trajectory that rewards genuine learning over static optimization.

Including a handful of these "subtle trap" scenarios in each Season 1 rotation would push miners toward more robust agent designs without requiring any scoring changes — the existing cost × correctness incentive already rewards getting it right the first time.

---

## Closing

The LLM-as-judge migration (v3→v4) is the right call — keyword-stuffing was a real vulnerability, and grounding requirements make gaming genuinely hard. The cost-competition incentive is cleaner than score-based ranking because it has a natural floor (you can't go below the minimum token cost for a correct response).

Season 1 has a unique opportunity to be the first benchmark that measures **learning over time**, not just single-shot capability. The multi-episode structure is the right foundation — leaning into it harder (episode-chained scenarios, learning trend as a scoring signal, workspace conventions for observable knowledge) would make SN11 genuinely differentiated from static benchmarks like LMSYS or SWE-Bench.

We're actively mining SN11 and building tools around ClawBench. Happy to test new scenarios, contribute episode-chained scenario designs, or help validate multi-episode scoring mechanics as they develop.

— Project Nobi team | [projectnobi.ai](https://projectnobi.ai)
