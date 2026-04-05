# Community Feedback — Project Nobi

**From:** Project Nobi team — [projectnobi.ai](https://projectnobi.ai)  
**Re:** TrajectoryRL v4.0 Incentive Mechanism + ClawBench v0

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

## 2. LLM-as-Judge: Two Concerns

### 2a. Judge variance across validators

The spec states judges produce "polarized" verdicts (obvious PASS/FAIL), so no majority voting or score EMA is needed. This holds for clear-cut cases (zero tool calls = FAIL). But we think the gray zone is larger than assumed for **grounding checks**.

Consider: an agent retrieves calendar data showing "Jordan Lee 4pm" and "David Park 4pm", then writes "there appears to be a scheduling conflict at 4pm." Did the agent "identify the conflict"? One judge says PASS (conflict mentioned). Another says FAIL (didn't name both attendees explicitly). The rubric says "Must mention BOTH events and identify them as conflicting" — but LLM judges parse natural language differently.

**Suggestion:** For criteria where validator disagreement is plausible, publish the exact judge prompt (or a redacted version) so miners can understand the decision boundary. This doesn't help gaming — the criteria are already public in DATASET_v0.md. It helps miners write policies that produce unambiguous responses.

### 2b. Grounding check edge cases

The grounding requirement is the strongest anti-gaming measure (zero-tool-call responses always fail). One edge case: an agent makes a tool call, gets data, but then makes a *correct inference* that goes beyond the literal data returned.

Example: Tool returns `{"status": "overdue", "due_date": "2026-03-01"}`. Agent writes: "The Q4 report is overdue and has been for over a month." The "over a month" is inferred from the date, not literally present in the tool response. Is this "grounded"?

We suspect the current judge handles this fine (LLMs understand inference). But explicitly noting that **reasonable inferences from retrieved data count as grounded** would help miners avoid being overly literal in their policies.

---

## 3. Cost Competition: Measurement Precision

The cost formula `Σ rate(model_i, token_type) × token_count(model_i, token_type)` assumes validators know the per-token rate for every model a pack might route to. Two questions:

**A. Rate source of truth.** If a pack routes subtasks to Qwen 3.5 via Chutes, the rate differs from Qwen 3.5 via Alibaba direct. Does the validator use a canonical rate table, or the actual provider rate? If actual, a miner could use a cheaper provider for the same model and win on cost without any policy improvement.

**B. Multi-model measurement.** The spec mentions multi-LLM routing as the end-game cost optimization. But today's ClawBench scenarios are single-turn knowledge-worker tasks. Is the evaluation harness already instrumented to track costs across multiple model calls within one episode? If not, this is a tooling gap that blocks the most interesting optimization path.

These aren't criticisms — the v4.0 spec is remarkably well-designed. We're flagging areas where the implementation might diverge from the spec as the scenario pool grows.

---

## 4. Pack Size Limit vs. Routing Complexity

The 32KB pack size limit is sensible for preventing token bombs. But as miners move toward multi-model routing (the "Stage 2" in the ROI example), the AGENTS.md needs to encode routing logic, model-specific policies, and fallback chains. 32KB may become constraining.

**Suggestion:** Consider a separate `routing.json` (or similar) in the `files` dict that doesn't count toward the same size budget as AGENTS.md, or increase the limit when multi-model scenarios are added. This is a future concern, not urgent for v0.

---

## Closing

The LLM-as-judge migration (v3→v4) is the right call — keyword-stuffing was a real vulnerability, and grounding requirements make gaming genuinely hard. The cost-competition incentive is cleaner than score-based ranking because it has a natural floor (you can't go below the minimum token cost for a correct response).

We're actively mining SN11 and building tools around ClawBench. Happy to test new scenarios or provide more detailed feedback as the dataset evolves.

— Project Nobi team | [projectnobi.ai](https://projectnobi.ai)
