# Proposal: Dynamic Scenario Generation for Season 1+

> Status: Draft
> Author: Claude Code (Ning driving)
> Date: 2026-04-14
> Context: Current scenarios are hardcoded in trajectory-sandbox (open source). Miners can reverse-engineer criteria and write walkthrough SKILL.md packs that game the judge.

---

## Problem

The current eval pipeline has three hardcoded layers per scenario:

```
fixture_factory.py  → generates data (emails, Slack, Gitea, etc.)
evidence.py         → extracts grounding evidence (A1-A10)
judge.py            → scores on criteria (C1-C22)
```

All three are open source. A miner who reads the code knows:
- There will always be a monitoring alert, client complaint, CTO escalation
- There will always be a confidential email trap
- The judge checks exactly 22 criteria in a known order
- Evidence A1 checks `slack_channels.incidents.messages`

**A walkthrough SKILL.md that maps to these criteria scores 22/22 on episode 1.** The split-half delta bonus (0.5x) can't overcome this — a static walkthrough beats a genuine learner.

This is fine for launch (the integrity judge rejects obvious walkthroughs). But post-launch, sophisticated miners will write "soft walkthroughs" that encode the criteria without looking like a cheat sheet.

---

## Proposed Solution: LLM-Generated Scenario Templates

Instead of hardcoding scenarios in code, validators generate them at eval time using an LLM. The scenario itself becomes a **private validator artifact**, not public code.

### Architecture

```
                    CURRENT                              PROPOSED
                    
 trajectory-sandbox (public)              trajectory-sandbox (public)
 ┌─────────────────────────┐              ┌─────────────────────────┐
 │ fixture_factory.py      │              │ scenario_engine.py      │
 │   incident_response()   │              │   generate_scenario()   │
 │   morning_brief()       │              │   - takes: domain, seed │
 │                         │              │   - returns: ScenarioDef│
 │ evidence.py             │              │     .fixtures           │
 │   IncidentResponseEvid. │              │     .instruction_md     │
 │   MorningBriefEvidence  │              │     .criteria (LLM gen) │
 │                         │              │     .evidence_spec      │
 │ judge.py                │              │                         │
 │   CRITERIA_INCIDENT_... │              │ judge.py                │
 │   CRITERIA_MORNING_...  │              │   score(criteria=...)   │
 └─────────────────────────┘              └─────────────────────────┘
                                          
         Fixed criteria                   Criteria generated per-epoch
         Miners can read                  Miners see only the domain
```

### How It Works

**Step 1: Validator generates a scenario definition (private)**

```python
scenario_def = scenario_engine.generate(
    domain="operations",           # broad category (public)
    epoch_seed=epoch_seed,         # from on-chain
    validator_salt=validator_salt,  # private
    llm=judge_llm,                 # same LLM used for judging
)
```

The LLM generates:
- A **situation** (what happened, who's involved, what services are relevant)
- An **instruction** (what the agent should do)
- **Fixtures** (emails, Slack messages, tasks, calendar events, Gitea data)
- **Criteria** (8-25 judge criteria specific to this situation)
- **Evidence spec** (what to check in mock state to ground each criterion)

**Step 2: Scenario definition is deterministic from seed + salt**

```
scenario_hash = SHA-256(epoch_seed || validator_salt || domain || template_version)
```

Same seed + salt = same scenario. Different validators = different scenarios.
After scoring, validators publish `(validator_salt, scenario_hash, scores)` for verification.

**Step 3: Miners never see the criteria**

Miners see only:
- SKILL.md (their own pack)
- INSTRUCTION.md (generated task description, not criteria)
- The mock service APIs (same as now)

The criteria, evidence spec, and fixture structure are **never exposed to the miner**.

### Example

**Current (hardcoded):**
```
Scenario: incident_response
Criteria: C1 (Slack #incidents), C2 (no confidential), C3 (client email), ...
Every validator runs this exact scenario. Miners know all 22 criteria.
```

**Proposed (dynamic):**
```
Validator A (epoch 100, salt "abc"):
  LLM generates: "Product launch coordination" scenario
  Criteria: posted launch plan to #marketing, notified press contacts,
            scheduled rehearsal, didn't leak pricing before embargo...
  
Validator B (epoch 100, salt "xyz"):
  LLM generates: "Customer data breach" scenario
  Criteria: notified legal within 1h, sent affected user emails,
            preserved audit log, didn't discuss in public Slack...
```

Both are "operations" domain. Both test communication, prioritization, safety. But a walkthrough for one doesn't work on the other.

### Constraints

**The scenario LLM call must be:**
1. **Deterministic** — same seed+salt = same scenario (for verification)
2. **Cheap** — one LLM call per epoch, not per miner (cached)
3. **Grounded** — the generated criteria must be verifiable from mock service state
4. **Bounded** — scenario complexity within the fixture API surface (email, Slack, Notion, calendar, Gitea)

**What stays the same:**
- Mock service API (email, Slack, Notion, calendar, Gitea) — the interface is fixed
- Split-half delta scoring formula
- 4 episodes per eval
- Private salt for cross-validator variation
- Pack format (SKILL.md in OPP v1)

**What changes:**
- Scenario definition: code → LLM-generated per-epoch
- Criteria: hardcoded 22 → dynamic 8-25 per scenario
- Evidence checks: hardcoded A1-A10 → generated from criteria
- Fixture factory: template-based → LLM-generated (validated against schema)

---

## Phased Rollout

### Phase 1: Scenario Rotation (now, no LLM changes)

Use existing hardcoded scenarios but rotate them unpredictably per epoch:

```python
# In sandbox_harness.py evaluate_miner():
available = ["incident_response", "morning_brief"]  # add more over time
scenario_idx = epoch_seed % len(available)
scenario = available[scenario_idx]
```

Miners can't know which scenario runs until the epoch starts. Adding a new scenario to the list (server-side validator update) immediately disrupts miners who over-specialized.

**Cost: zero.** Just change one line in the harness.

### Phase 2: Parameterized Scenarios (weeks)

Same code-based scenarios but with LLM-parameterized variation:

```python
# Generate scenario parameters from LLM (one call per epoch, cached)
params = llm.generate(
    "Generate parameters for an incident response scenario: "
    "what kind of company, what services are affected, "
    "what's the confidential topic, what's the twist..."
)

# Feed params into existing fixture factory
factory = FixtureFactory(
    epoch_seed=seed, validator_salt=salt,
    scenario="incident_response",
    params=params,  # NEW: overrides default pools
)
```

The **structure** is still code-defined (22 criteria, A1-A10 evidence). But the **content** varies beyond the PRNG pools — the LLM might generate a scenario where the confidential topic is a patent filing, not an acquisition. Walkthroughs that pattern-match on "acquisition" keywords fail.

### Phase 3: Fully Dynamic Scenarios (months)

The full proposal above. The LLM generates the criteria themselves. Requires:
- Criteria → evidence mapping validation (can the evidence extractor actually check this?)
- Schema validation for generated fixtures
- Consensus on criteria quality (bad LLM criteria = unfair scoring)
- Template library of "scenario archetypes" to constrain generation

---

## Why This Fundamentally Works

The insight from the spec (§Cross-Validator Variation as Monte Carlo Sampling) already applies: different validators test different samples. Extending this from "different fixture data" to "different scenario structure" makes the sampling space exponentially larger.

A miner who builds a genuinely capable agent (reads all channels, assesses, acts appropriately, protects sensitive info, follows up) will score well on ANY operations scenario. A miner who walkthroughs one specific scenario will fail on variants.

**The competitive moat becomes: general operational competence, not scenario-specific scripts.**

---

## Relationship to Existing Codebase

| Component | Current | Phase 1 | Phase 2 | Phase 3 |
|-----------|---------|---------|---------|---------|
| `sandbox_harness.py` | hardcoded `incident_response` | epoch-based rotation | params from LLM | `ScenarioDef` from LLM |
| `fixture_factory.py` | 2 scenario generators | unchanged | accepts `params` override | replaced by `scenario_engine.py` |
| `evidence.py` | 2 extractors (A1-A10, B1-B10) | unchanged | unchanged | generic extractor from `evidence_spec` |
| `judge.py` | 2 criteria sets (22, 18) | unchanged | unchanged | dynamic criteria from `ScenarioDef` |
| `episode_scorer.py` | `for_scenario()` factory | unchanged | unchanged | takes `ScenarioDef` directly |
| Validator config | `evaluation_harness` | + `scenario_rotation: true` | + `scenario_params_llm: true` | + `dynamic_scenarios: true` |
