# Season 1 Community Feedback — Project Nobi

**From:** Project Nobi team — Bittensor agent platform
**Context:** 

---

## Scenario Ideas

**1. Cross-service workflow coordination**
> "A customer reported an issue in Slack. Triage it: check Gitea for related recent commits, create a Notion task, send a summary email, post a Slack status update."

Tests inter-service chaining across 4 services. Learning signal: does the agent discover better sequencing patterns (e.g., checking Gitea *before* writing the email vs after)? Each episode uses different customer, channel, repo — no memorization.

**2. Research + synthesis**
> "Prepare a briefing on [topic] from available sources. Store in Notion, send to stakeholders."

Tests information quality, not just tool use. Learning signal: are summaries more actionable each episode?

**3. Refactor under constraint**
> "Refactor this function to pass the failing tests. Tests cannot be modified."

Different from "bug fix" — teaches test-first patterns. A miner that doesn't learn to read tests before touching code will score low on episode 1 and (if SKILL.md is well-designed) much higher by episode 4.

---

## Scoring: Split-Half Delta Concern

The mechanism is elegant. One concern at **N=4**:

If episode 1 is an outlier (bad fixture, judge variance), a genuinely good agent scores: `low → high → high → high`. Split-half delta rewards this — but the agent didn't "learn," it just recovered from bad luck.

**Suggestion:** Weighted linear trend fit across all 4 episodes + penalize high variance:
```
score = slope(episode_scores) - λ * variance(episode_scores)
```
This is more robust to single-episode noise than raw split-half delta and harder to game (intentional bad episode-1 increases variance, reducing score).

---

## Sandbox: Three Gaps We Found

**1. `/workspace/learned/` needs a spec**
- Size limit? (unbounded writes = potential abuse)
- Cleared between miners? (should be yes — otherwise miner A contaminates miner B)
- Suggestion: 1MB cap, cleared per-miner, persists within a miner's 4-episode run.

**2. Judge score transparency**
Miners can't improve SKILL.md if they don't know why episode 1 scored 0.6 vs episode 4's 0.9. We'd love a sanitized judge reasoning log post-evaluation (no fixture data, just quality reasoning).

**3. Memory format flexibility**
Is `/workspace/learned/` format standardized? The reference uses markdown, but structured JSON might work better for some memory patterns. Suggest explicitly stating any file structure under `/workspace/learned/` is valid.

---

## What We're Building

Project Nobi is building a SKILL.md focused on:
- **Three-tier memory** (HOT/WARM/COLD patterns with auto-promotion)
- **Post-task reflection discipline** — append learnings before declaring done
- **Failure mode documentation** — mistakes.md prevents repeating errors
- **Service-specific fast paths** — discovered patterns for MailHog, Notion, Gitea

We're excited about the learning trend signal — it rewards genuine intelligence improvement, not just raw performance. This is the most interesting incentive design in Bittensor.

Looking forward to Season 1. Happy to test or provide feedback on the sandbox.

— Project Nobi team | [projectnobi.ai](https://projectnobi.ai)
