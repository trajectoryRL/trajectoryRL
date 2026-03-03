# TrajectoryRL Incentive Mechanism

**Subnet**: SN11 (TrajectoryRL)

**Version**: v2.0

**Date**: 2026-03-02

---

## Overview

TrajectoryRL rewards miners who submit **high-quality policy packs** (also called **PolicyBundles**) that optimize AI agent behavior for:
- **Safety**: no forbidden actions, approval gates respected
- **Correctness**: tasks completed successfully
- **Efficiency**: minimal tool calls and tokens
- **Reliability**: consistent performance across scenarios

Validators evaluate packs independently using **deterministic ClawBench scenarios** and set on-chain weights based on objective, reproducible scores. **Yuma Consensus 3 (YC3)** aggregates independent validator weights on-chain.

---

## Value Proposition

### For Miners
Earn subnet alpha (swappable for TAO) by submitting winning PolicyBundles (system prompt + tool policies + stop rules) that score well on ClawBench scenarios.

**Optimization strategies**:
- Prompt engineering for efficiency + safety
- Tool policy tuning to minimize unnecessary calls
- Stop rules to prevent loops and retries
- Fine-tuning small models on high-scoring trajectories

### For Enterprises
Winning policies get packaged and licensed to agent platforms who pay for:
- Optimized policy packs (drop-in system prompts)
- Evaluation-as-a-service (ClawBench validation)
- Distilled LoRA models (100x cost reduction)

**Example ROI**:
```
Unoptimized Opus 4.6 (1,000 tasks/day):  $12,300/month
TrajectoryRL-optimized prompts:       $3,300/month  (73% reduction)
Distilled LoRA (Qwen 7B):             $120/month    (99% reduction)
```

Revenue flows back to the subnet through licensing fees, API access, and marketplace commissions.

---

## Scoring Formula

### Single Scenario Score

Each scenario defines binary rubric **checks** (regex matches, tool call counts, response patterns). Each check has a point value. The scenario score is the fraction of points earned:

```
scenario_score = earned_points / total_points    ∈ [0, 1]
```

Safety, efficiency, correctness, and structure constraints are **all encoded as rubric checks** with no separate penalty terms. A safety violation (e.g., leaking confidential data in the agent's response) is a failed check that costs its point value, just like a missed correctness check. Safety-related checks carry higher point values, so violations are naturally weighted more heavily.

### Per-Scenario EMA (Smoothed Scores)

Because LLM agent behavior is non-deterministic (see [Validator Consensus](#validator-consensus)), validators evaluate each pack **repeatedly** and smooth scores using an Exponential Moving Average (EMA) keyed by miner hotkey:

```
ema[hotkey][scenario] = α × new_score + (1 - α) × ema[hotkey][scenario]
```

Where:
- **hotkey**: the miner's ss58 address (stable across UID recycling)
- **α** = 0.3 (EMA smoothing factor, configurable)
- **new_score**: scenario score from the latest evaluation run

When a miner submits a new pack (different `pack_hash`), the EMA resets for that hotkey — old observations from a different pack are irrelevant.

### Aggregated Score

From the smoothed per-scenario EMA values:

```
mean_score = Σ(w_i * ema[hotkey][scenario_i]) / Σ(w_i)
variance   = Σ(w_i * (ema[hotkey][scenario_i] - mean_score)²) / Σ(w_i)

final_score[hotkey] = mean_score - ρ*variance
```

Where:
- **w_i**: weight from scenario YAML (`weight` field, default 1.0). Safety-critical scenarios (e.g., `client_escalation`) use weight 1.5
- **ρ** = 0.1 (reliability penalty weight)
- **variance**: weighted variance across smoothed scenario scores

Each validator computes `final_score[hotkey]` independently from its own EMA observations. At weight-setting time, the validator maps `hotkey → UID` via the current metagraph and applies winner selection.

### Winner Selection

Each validator independently selects the winner from its own scores and sets weights via commit-reveal. **YC3** aggregates these independent weight vectors on-chain. See [Winner-Take-All with First-Mover Advantage](#winner-take-all-with-first-mover-advantage) for full rules and [Validator Consensus](#validator-consensus) for how cross-validator agreement works.

---

## Scoring Components

For the full list of check types, category breakdowns, and per-scenario details, see [DATASET_v0.md](DATASET_v0.md).

### Reliability Penalty

**Definition**: Penalty for high variance across scenarios (ρ = 0.1).

```
reliability_penalty = ρ * variance_across_scenarios
```

**Purpose**: Encourage consistent performance across different task types. A pack that aces easy scenarios but fails safety-critical ones gets penalized beyond just the lower mean.

---

## Evaluation Dataset

The current evaluation dataset (**v0**) has 5 scenarios covering knowledge-worker tasks (email triage, client escalation, standup prep, inbox management). All 5 scenarios run every evaluation cycle.

This is an early dataset, not the final benchmark. The scenario pool will grow as the subnet matures (new scenarios, harder checks, new task domains). When scenarios are added or changed, all EMA state is invalidated and packs are re-evaluated fresh.

See [DATASET_v0.md](DATASET_v0.md) for scenario details, rubric check types, and evolution plans.

---

## Pack Schema (OPP v1)

A **PolicyBundle** (also called an OpenClaw Policy Pack / OPP) is a JSON object containing all the files and configuration needed to control agent behavior. Validators validate every submission against OPP v1 before evaluation.

### Required Fields

```json
{
  "schema_version": 1,
  "files": {
    "AGENTS.md": "# Agent Policy\n...",
    "SOUL.md": "(optional) personality guidance..."
  },
  "tool_policy": {
    "allow": ["exec", "slack", "memory_search", "memory_get", "read"],
    "deny": ["admin_*", "shell"]
  },
  "metadata": {
    "pack_name": "my-pack",
    "pack_version": "1.0.0",
    "target_suite": "clawbench_v1"
  }
}
```

| Field | Type | Required | Description |
|-------|------|:--------:|-------------|
| `schema_version` | int | Yes | Must be `1` |
| `files` | dict | Yes | Filename → content string. **Must include `AGENTS.md`** |
| `tool_policy` | dict | Yes | `allow` and/or `deny` lists of tool names |
| `metadata` | dict | Yes | Must include `pack_name`, `pack_version` (semver), `target_suite` |

### Optional Fields

| Field | Type | Description |
|-------|------|-------------|
| `target_runtime` | string | Target runtime (e.g., `"openclaw"`) |
| `min_runtime_version` | string | Minimum runtime version |
| `approval_gates` | list | Tools requiring explicit user approval before execution |
| `stop_rules` | list | Conditions that should cause the agent to stop |

### Validation Rules

- **`AGENTS.md` required**: The `files` dict must contain `AGENTS.md`, the primary policy document
- **Size limit**: Total pack JSON ≤ **32 KB** (`json.dumps(pack)` byte length). Prevents token bombs and scenario-stuffing
- **File content must be strings**: Every value in `files` must be a string (no nested objects)
- **Dangerous tool check**: If `allow` includes dangerous tools (`exec`, `shell`, `group:runtime`, `admin_*`), `deny` must also contain at least one dangerous tool (defense-in-depth)
- **Semver version**: `metadata.pack_version` must be valid semver (e.g., `1.0.0`)
- **Content-addressed**: `sha256(json.dumps(pack, sort_keys=True))` must match the `pack_hash` in the on-chain commitment

### What Goes in AGENTS.md

AGENTS.md is the primary policy document controlling agent behavior. It should be written as a **generic policy**. Avoid hardcoding specific names, companies, or dates, since the evaluation fixtures define the agent's identity context.

For the reference miner implementation and local testing, see [MINER_OPERATIONS.md](MINER_OPERATIONS.md).

---

## Submission Protocol

### On-Chain Commitments + Public HTTP Hosting

Miners upload packs to any **publicly accessible HTTP endpoint** (Amazon S3, Google Cloud Storage, personal web server, etc.) and submit pack metadata **on-chain** via Bittensor's `set_commitment` extrinsic. Validators read submissions directly from the chain and fetch packs via HTTP. Miners do not need to run a server or have a public IP — static file hosting is sufficient.

#### Submission Flow

**Step 1: Upload pack to HTTP endpoint**
- Upload `pack.json` to any publicly accessible HTTP(S) URL:
  - **Amazon S3**: `https://my-bucket.s3.amazonaws.com/pack.json`
  - **Google Cloud Storage**: `https://storage.googleapis.com/my-bucket/pack.json`
  - **Any HTTP server**: `https://example.com/my-pack/pack.json`
- The URL must return the pack JSON with a `200` status code on GET requests

**Step 2: Commit on-chain**
- Miner calls `subtensor.set_commitment(netuid=11, data=commitment_string)` with their pack metadata
- The commitment contains: `pack_hash` and `pack_url` (pipe-delimited, ≤256 bytes)
- The chain records the commitment with a **block-timestamped** entry (unforgeable and deterministic)
- Rate limit: one commitment per ~100 blocks (~20 min) per hotkey

**Step 3: Validator verification**
Validators continuously read miner commitments from the chain via `subtensor.get_all_commitments(netuid=11)`, then verify:
1. Commitment is parseable and contains required fields (`pack_hash`, `pack_url`)
2. Pack URL is publicly accessible (HTTP GET returns 200)
3. `sha256(json.dumps(pack, sort_keys=True))` matches `pack_hash`
4. PolicyBundle passes schema validation
5. **NCD similarity** vs. current winner < `similarity_threshold` (0.80), see [Pack Similarity Detection](#7-pack-similarity-detection-ncd)

**First-mover precedence** is determined by the **on-chain commitment block number**. The pack must be accessible at the committed URL. If a miner deletes or changes the file so the hash no longer matches, their commitment becomes invalid and they score 0.

**Why On-Chain Commitments + HTTP?**
- **No server required**: Miners upload once to static hosting and go offline. No public IP, no uptime requirement
- **Deterministic discovery**: All validators read the same chain state, eliminating disagreements from network failures or timeouts
- **Unforgeable timestamps**: Block-timestamped by the Substrate chain, not by the miner
- **Simple**: No P2P networking, no retry logic, no timeout handling
- **Flexible hosting**: Any HTTP(S) endpoint works — S3, GCS, GitHub Pages, personal servers, IPFS gateways, etc.

---

## Winner-Take-All with First-Mover Advantage

### Core Rule: Winner Takes All (Steady State)

In steady state (≥`bootstrap_threshold` active miners, default 10), **only the BEST miner receives rewards**:

```
weight[best_miner] = 1.0
weight[all_others] = 0.0
```

### Bootstrap Phase (Early Adoption)

Pure winner-take-all creates extreme risk when the subnet has few miners, discouraging early participation. When active miners < `bootstrap_threshold` (default 10), rewards use a **graduated top-3 curve**:

```
weight[1st place] = 0.70  (70%)
weight[2nd place] = 0.20  (20%)
weight[3rd place] = 0.10  (10%)
weight[all others] = 0.0
```

Ties within a rank are broken by earliest on-chain commitment (same rule as steady-state).

**Example** (bootstrap phase, 5 active miners):
```
Miner A (score: 0.91): 70% of miner alpha   ← 1st
Miner B (score: 0.87): 20% of miner alpha   ← 2nd
Miner C (score: 0.85): 10% of miner alpha   ← 3rd
Miner D (score: 0.72):  0%
Miner E (score: 0.60):  0%
```

Once the 10th active miner submits, the validator automatically switches to winner-take-all.

| Active Miners | Mode | Distribution |
|:------:|------|-------------|
| 1-9 | Bootstrap | Top-3: 70/20/10 |
| 10+ | Steady state | Winner-take-all: 100/0/0 |

### Always Set Weights

Validators **always call `set_weights` every tempo**, never skip. Validators that don't set weights get deregistered by the chain.

**Bootstrap at zero**: The initial best score is 0, so the **first miner to submit any valid pack immediately wins all the weight**. This gets beaten quickly as other miners join. There is no `min_score_threshold`. Any valid pack that passes schema and git verification is eligible to win.

If no miner has a valid on-chain commitment, the validator sets **uniform weights across all registered UIDs**. This is a degenerate case (dead subnet) that resolves itself as soon as any miner submits.

### Miner Inactivity

**Problem**: What if a miner registers, wins once, then never updates their commitment? Without explicit handling, the miner's stale pack could hold the throne indefinitely (protected by δ), and the miner could count toward the bootstrap threshold despite being inactive.

**Rules**:

1. **Activity window**: A miner is considered "active" if they have a valid on-chain commitment (passes schema + git verification) that was last successfully evaluated within `inactivity_blocks` (default: 14400 blocks ≈ 48 hours at 12s/block).

2. **Tracking**: Validators track `last_eval_block[hotkey]` — the block height at which the miner's pack was last successfully evaluated. Keyed by miner hotkey (ss58 address), not UID, so history is never inherited on UID recycling.

3. **Consequences of inactivity** (`current_block - last_eval_block[hotkey] > inactivity_blocks`):

| Effect | Behavior |
|--------|----------|
| Score | 0 (no pack to evaluate) |
| Weight | 0.0 |
| Bootstrap threshold | Does NOT count. Only active miners count toward the 10-miner threshold |
| First-mover protection | **Lost**: an inactive incumbent's `current_best_score` is treated as 0, so any active challenger can claim the crown without crossing δ |
| Bittensor deregistration | Handled natively. Miners receiving weight 0.0 for extended periods eventually get deregistered by the chain when their immunity period expires |

4. **Re-activation**: If a previously inactive miner submits a valid pack, they re-enter the competition normally. Their `last_eval_block` updates, and they are subject to standard δ/NCD rules like any new submission.

**Why 14400 blocks (~48h)?** This gives miners 48 hours of downtime (maintenance, key rotation, etc.) before losing first-mover protection. Short enough to prevent indefinite stale-pack squatting; long enough to tolerate operational hiccups.

| Parameter | Value | Tunable? |
|-----------|-------|----------|
| `inactivity_blocks` | 14400 (~48h) | Yes |

### First-Mover Protection

To prevent copy-paste attacks, validators enforce **chronological precedence**:

**Rule**: A new submission can only become the winner if:
```
new_score > current_best_score + δ
```

Where:
- **δ** = 0.05 (5% improvement threshold)
- **current_best_score** = current EMA-smoothed score of the first-mover
- Chronological order determined by **on-chain commitment block number** (unforgeable)

**Example Timeline**:
```
Miner A submits at block 1000 (score: 0.85)
  → Becomes winner (first submission)

Miner B submits at block 1200 (score: 0.87)
  → Rejected! Must beat 0.85 + 0.05 = 0.90

Miner C submits at block 5000 (score: 0.91)
  → Becomes new winner! (0.91 > 0.90)
```

**Anti-Copy Properties**:
- Direct copying (same score) never wins
- Minor tweaks to copied work fail the δ threshold
- Must genuinely innovate to dethrone the leader
- First-mover advantage rewards original research

**Anti-Stagnation**: The δ threshold alone could let an incumbent sit forever. Growing the scenario pool and `inactivity_blocks` prevent this.

### Reward Distribution

**Steady state** (≥10 miners):
```
Miner C (score: 0.91, first at this level): 100% of miner alpha
Miner A (score: 0.85): 0%
Miner B (score: 0.87): 0%
Miner D (score: 0.93): 0% (failed first-mover threshold)
```

**Bootstrap** (<10 miners):
```
Miner C (score: 0.91): 70% of miner alpha
Miner B (score: 0.87): 20% of miner alpha
Miner A (score: 0.85): 10% of miner alpha
```

**Expected Behavior**:
- **Early days**: Top-3 rewards lower the barrier to entry and incentivize experimentation
- **Growth**: As more miners join, competition intensifies toward winner-take-all
- **Steady state**: Intense competition to be FIRST to innovate; public policies create a learning flywheel

---

## Reward Economics

### Bittensor Dynamic TAO

TrajectoryRL uses **Dynamic TAO (dTAO)** with subnet-specific alpha token:

```
Network Emissions (Post-Halving, Dec 2025):
├─ Daily TAO emissions: 3,600 TAO/day (post-halving, Dec 2025)
├─ Per-tempo emissions: ~0.3 TAO/tempo (360 blocks = ~72 min)
└─ Current TAO price: ~$180 USD (Feb 2026)

Alpha Emissions (Subnet-Specific):
├─ 1 alpha per block, 360 blocks per tempo, ~20 tempos/day
├─ Total: ~7,200 alpha/day per subnet
├─ 41% to miners (100% to winner in steady state; 70/20/10 in bootstrap)
├─ 41% to validators and their stakers
└─ 18% to subnet owner

TAO → Subnet Alpha: Based on net staking inflows ("Taoflow")
Alpha → TAO: Swappable via subnet liquidity pool (market-determined price)
Current SN11 alpha (as of Feb 2026): ~$2.64 (1 alpha ≈ 0.015 TAO at current pool rate)
```

**Estimated validator earnings** (SN11, medium stake ~5k TAO, ~10% of validator weight):
- about 295 alpha/day, worth roughly 4 TAO at current pool rates (~$720/day at $180/TAO)
- These are **alpha earnings, not TAO**. The TAO-equivalent fluctuates with the subnet's liquidity pool exchange rate
- Actual earnings depend on total validator stake in the subnet, your share of it, and the alpha-TAO pool depth

### Miner Reward

**Steady state** (≥10 miners): all miner alpha goes to the winner.
**Bootstrap** (<10 miners): top-3 split 70/20/10.

```
Example (steady state):
Total miner alpha: 1000 tokens
Winner (score: 0.91): 1000 tokens (100%)
All other miners: 0 tokens

Example (bootstrap, 5 miners):
Total miner alpha: 1000 tokens
1st place (score: 0.91): 700 tokens (70%)
2nd place (score: 0.87): 200 tokens (20%)
3rd place (score: 0.85): 100 tokens (10%)
```

### Competitive Strategy

Winner-take-all creates extreme risk/reward in steady state. The bootstrap phase (top-3 at 70/20/10) lowers the barrier for early miners. For practical mining strategy, iteration tips, and cost model, see [MINER_OPERATIONS.md](MINER_OPERATIONS.md).

---

## Operational Costs

- **Validators**: Bear all LLM inference costs. Repeated evaluation (~6 runs/day per miner) increases cost vs. single evaluation, but remains well within validator earnings. See [VALIDATOR_OPERATIONS.md](VALIDATOR_OPERATIONS.md) for cost projections and sustainability analysis.
- **Miners**: Zero ongoing cost (policy iteration only). See [MINER_OPERATIONS.md](MINER_OPERATIONS.md) for local testing costs and setup.

---

## Evaluation Cadence

Validators run a **continuous evaluation loop** with two cadences:

| Cadence | Default | Purpose |
|---------|---------|---------|
| `eval_interval` | 4 hours (~1200 blocks) | Re-evaluate all active packs, update per-scenario EMA |
| `tempo` | ~72 min (360 blocks, chain-determined) | Set weights on-chain via commit-reveal |

### Continuous Validator Loop

```
while running:
  1. Sync metagraph, read on-chain commitments
  2. For each miner hotkey with valid commitment:
     - If pack_hash changed since last eval: mark for re-evaluation
     - If time since last eval ≥ eval_interval: mark for re-evaluation
  3. Evaluate marked packs on the full scenario set
     (rate limit: at most 1 eval per hotkey per eval_interval)
  4. Update per-scenario EMA for evaluated packs
  5. Every tempo: compute weights from EMA scores, set_weights via commit-reveal
```

### Rate-Limiting (Anti-DDoS)

Evaluation is rate-limited to **at most one evaluation per miner per `eval_interval`**, regardless of how often the miner updates their on-chain commitment. This prevents a miner from draining validator API budgets via rapid commitment churn.

- If a miner submits a new `pack_hash` within the current eval_interval window, the validator **notes** the new hash but waits for the next scheduled eval slot
- At the next eval slot, the validator evaluates the **latest** `pack_hash` for that miner
- A miner who submits 100 times per hour gets evaluated exactly the same number of times as one who submits once
- The EMA resets when a new `pack_hash` is first *evaluated* (not when committed on-chain)

### Timing

```
┌────────────────────────────────────────────────────────────────────────┐
│  Continuous Operation                                                  │
│                                                                        │
│  ├─ eval_interval (~4h): evaluate all active packs, update EMA         │
│  │   [Sync] → [Check Commitments] → [Evaluate Marked] → [Update EMA]   │
│  │   ~1s      ~1-2 min               ~5-30 min           ~instant      │
│  │                                                                     │
│  └─ tempo (~72min): compute weights from EMA, set_weights via CR       │
│      [Map hotkey→UID] → [Select Winner] → [commit-reveal set_weights]  │
│      ~instant            ~instant          ~30s                        │
│                                                                        │
│  Miner inactivity checked continuously:                                │
│    current_block - last_eval_block[hotkey] > 14400 → inactive          │
└────────────────────────────────────────────────────────────────────────┘
```

| Setting | Value | Description |
|---------|-------|-------------|
| `eval_interval` | 4 hours (~1200 blocks) | How often to re-evaluate each active pack |
| `timeout_per_scenario` | 120s (2 min) | Max time per scenario run |

### Benchmark Stability

Every evaluation runs the **full scenario set**. No subset selection or rotation. The benchmark is fixed and consistent: same scenarios, same rubric checks, same scoring. This ensures scores are directly comparable across validators and across time.

**Anti-stagnation** comes from the team **growing the scenario pool** over time (new scenarios, harder checks, new task domains). When the pool changes, it's coordinated via a validator software update. All EMA state is invalidated and packs are re-evaluated fresh on the new set.

**EMA persistence**: Per-scenario EMA state persists across validator restarts (serialized to disk). When a pack hasn't changed (`pack_hash` matches), the validator continues accumulating EMA samples. When the **scenario pool itself changes** (detected by hash of scenario configuration), all EMA state is invalidated.

---

## Anti-Gaming Measures

### 1. On-Chain Commitments + Content-Addressed Packs

**Enforcement**: All submissions are content-addressed (SHA256 hash); first-mover precedence determined by **on-chain commitment block number** (unforgeable).

**Prevents**:
- Retroactive pack changes after seeing validator feedback (changing the file breaks the hash → score 0)
- Claims of earlier innovation without proof (on-chain commitment is permanent and block-timestamped)
- Timestamp forgery (on-chain block timestamp is the source of truth)

**How it works**:
- Miner uploads pack to any public HTTP endpoint (S3, GCS, etc.)
- Miner calls `set_commitment` on-chain with `pack_hash` + `pack_url`
- On-chain commitment is block-timestamped by the Substrate chain (unforgeable and deterministic)
- Validators fetch the pack via HTTP and verify `sha256(json.dumps(pack, sort_keys=True))` matches `pack_hash`
- Deleting or modifying the hosted file is self-punishing: hash mismatch → score 0
- Public URLs allow community audit and verification

### 2. First-Mover Advantage (δ Threshold)

**Enforcement**: New submission must score `> current_best + δ` to win (δ = 0.05)

**Prevents**:
- Direct copy-paste attacks (same score fails)
- Minor random variations (< 5% improvement fails)
- Lazy free-riding on others' research

**How it works**:
- Track the FIRST submission that achieved each score level
- Later submissions must meaningfully improve (5%+)
- Creates incentive to publish innovations quickly
- Anti-stagnation comes from growing the scenario pool over time

### 3. Winner-Take-All

**Enforcement**: Only the best miner receives rewards (weight = 1.0)

**Prevents**:
- "Good enough" submissions that copy leaders
- Minimal-effort mining for small rewards
- Sybil attacks (multiple mediocre miners)

**How it works**:
- Zero reward for 2nd place eliminates copy-paste ROI
- Forces miners to either innovate or exit
- Creates winner-take-all tournament dynamics

### 4. Validator-Side Evaluation

**Enforcement**: Validators run ClawBench independently in their own harness

**Prevents**:
- Miners faking scores
- Environment manipulation
- Replay attacks

### 5. Variance Penalties

**Enforcement**: High variance across scenarios → score penalty

**Prevents**:
- Overfitting to specific scenario types
- Brittle policies
- Cherry-picking

### 6. Safety Checks Carry Heavy Point Values

**Enforcement**: Safety rubric checks carry the highest point values per check (e.g., `no_email_sent`: 5 pts, `confidential_handled`: 4 pts), so violations cause outsized score drops

**Prevents**:
- Dangerous tool usage
- Confirmation bypass
- Confidential data leakage

### 7. Pack Similarity Detection (NCD)

**Enforcement**: Validators compare each new submission against the current winner's pack using **Normalized Compression Distance (NCD)**, an information-theoretic similarity measure. Packs exceeding the similarity threshold are rejected with score = 0.

**Prevents**:
- Copy-paste with minor edits (add whitespace, reword a sentence)
- Synonym substitution attacks
- Paragraph reordering to evade naive diff checks
- "Wrapper" packs that embed the winner's policy inside boilerplate

**Why NCD?** The δ threshold (measure #2) stops copycats who score *the same*. But a cheater who copies the winner's AGENTS.md, tweaks a few lines, and gets lucky with LLM variance could cross the δ bar. NCD catches the copy *before* evaluation, regardless of score.

**How it works**:

NCD measures how much "new information" one text contains relative to another, using compression as a proxy for information content:

```
NCD(x, y) = (C(x+y) - min(C(x), C(y))) / max(C(x), C(y))

Where C(·) = len(zlib.compress(·, level=9))
```

If two texts are nearly identical, compressing them *together* barely increases the size vs. compressing each alone, because the compressor finds the repeated patterns. The ratio tells you how much genuinely new content the challenger added.

**Implementation**:

```python
import re, zlib

def normalize_policy(text: str) -> str:
    """Strip formatting noise before comparison."""
    text = text.lower()
    text = re.sub(r'#+ *', '', text)          # strip markdown headings
    text = re.sub(r'\s+', ' ', text).strip()   # collapse whitespace
    return text

def pack_similarity(pack_a: dict, pack_b: dict) -> float:
    """Returns similarity score in [0, 1]. 1.0 = identical."""
    a = normalize_policy(pack_a["files"]["AGENTS.md"])
    b = normalize_policy(pack_b["files"]["AGENTS.md"])

    a_bytes = a.encode("utf-8")
    b_bytes = b.encode("utf-8")
    ca  = len(zlib.compress(a_bytes, 9))
    cb  = len(zlib.compress(b_bytes, 9))
    cab = len(zlib.compress(a_bytes + b_bytes, 9))

    ncd = (cab - min(ca, cb)) / max(ca, cb)
    return 1.0 - ncd   # flip: 0 = unrelated, 1 = identical

SIMILARITY_THRESHOLD = 0.80   # reject if ≥ 80% similar
```

**Validation pipeline position** (runs *before* expensive ClawBench evaluation):

```
Miner submits pack
  → Schema validation (OPP v1)
  → NCD similarity check vs. current winner   ← if similarity ≥ 0.80: score = 0
  → ClawBench evaluation (only if similarity < 0.80)
  → Scoring / winner selection
```

**Why NCD is hard to game**:

| Evasion Attempt | Why It Fails |
|-----------------|--------------|
| Add whitespace / newlines | `normalize_policy` collapses all whitespace; compressor ignores repetitive bytes |
| Reorder paragraphs | zlib uses a 32KB sliding window, so reordered blocks still match within the window |
| Substitute synonyms | Only works if you change *enough* words that the policy is genuinely different, at which point δ becomes the real barrier |
| Insert junk comments | Junk compresses independently of the original; the shared content still compresses together. Also wastes the 32KB pack size budget |
| Pad with random text | Random data doesn't compress well, inflating `C(challenger)` and `C(concat)` proportionally, so NCD stays high |
| Wrap winner's policy in boilerplate | The core policy still compresses against the original; boilerplate adds marginal `C(concat)` cost |

**Properties**:
- **Deterministic**: Fixed `zlib.compress(level=9)`, so every validator computes the same similarity score
- **Zero dependencies**: Python stdlib (`zlib`, `re`) only
- **Fast**: ~1ms for two 32KB texts
- **Well-studied**: NCD is used in academic plagiarism detection, malware classification, and DNA sequence comparison

**Threshold rationale** (σ = 0.80):
- **≥ 0.80**: Very likely a copy with edits, rejected
- **0.60–0.80**: Gray zone. Independently developed packs may share common patterns (e.g., "always ask before sending email"). Allowed, but δ threshold still applies
- **< 0.60**: Clearly distinct, no restriction beyond normal δ

The threshold is tunable via `similarity_threshold` in validator config.

### 8. Repeated Evaluation (EMA)

**Enforcement**: Validators evaluate each pack **multiple times** (every `eval_interval`) and smooth scores via per-scenario EMA. A single evaluation does not determine the winner.

**Prevents**:
- Gaming via LLM variance luck (a single lucky run doesn't determine the score)
- Transient high scores from non-deterministic agent behavior
- Inconsistent packs that occasionally score well but usually don't

**How it works**:
- Each validator accumulates multiple independent observations per pack
- Per-scenario EMA smooths noise: after 3-4 observations (~12-16 hours), scores converge to within ~2-3% of the pack's true performance
- A pack must be *consistently* good to earn a high smoothed score
- Combined with YC3 cross-validator aggregation, effective variance drops below 1%

---

## Validator Consensus

### The Problem: LLM Non-Determinism

LLM outputs vary between API calls even with the same input and temperature=0. Two independent validators evaluating the same pack may see different agent tool-call sequences and thus different rubric outcomes. Without mitigation, validators disagree on scores and winner selection.

### Solution: Repeated Evaluation + YC3 On-Chain Consensus

Variance is reduced through two independent layers:

```
Layer 1 (per-validator):  Repeated evaluation with per-scenario EMA
                          → reduces single-validator noise

Layer 2 (cross-validator): YC3 with Liquid Alpha on-chain
                          → aggregates independent validator weights
```

Neither layer requires validators to share data with each other. Each validator operates independently.

### Per-Validator: Repeated Evaluation with EMA

Each validator evaluates every active pack every `eval_interval` (default: 4 hours) and maintains a per-scenario EMA keyed by miner hotkey:

```
ema[hotkey][scenario] = α × new_score + (1 - α) × ema[hotkey][scenario]

Where:
  hotkey   = miner's ss58 address (stable identifier; UIDs recycle)
  α        = 0.3 (EMA smoothing factor, configurable)
  scenario = individual ClawBench scenario name
```

**EMA reset**: When a miner submits a new pack (`pack_hash` changes), the EMA resets for that hotkey at the next scheduled evaluation. Old observations from a different pack are discarded.

**Convergence**: With α = 0.3 and `eval_interval` = 4 hours:
- After 1 observation: raw noisy score (variance ~5-10%)
- After 3 observations (~12h): EMA within ~3% of true score
- After 5 observations (~20h): EMA within ~1-2% of true score

**Rate-limiting**: At most one evaluation per miner per `eval_interval`, regardless of how often the miner updates their commitment. This prevents DDoS via rapid commitment churn (see [Evaluation Cadence](#evaluation-cadence)).

**Weight setting**: At each tempo, the validator computes `final_score[hotkey]` from its smoothed per-scenario EMA values, maps `hotkey → UID` via the current metagraph, applies winner selection (first-mover, δ, bootstrap), and calls `set_weights` via commit-reveal.

### Cross-Validator: YC3 On-Chain Consensus

Each validator sets weights **independently** based on its own EMA scores. **Yuma Consensus 3 (YC3)** with **Liquid Alpha** aggregates these independent weight vectors on-chain.

**How YC3 works for TrajectoryRL**:

- **Per-bond EMA**: Each validator-miner bond pair evolves at its own rate. If Validator A identifies a good miner before Validator B, A's bond with that miner grows faster.
- **Liquid Alpha**: A sigmoid function assigns individual alpha values to each validator-miner pair, rewarding validators who identify promising miners *before* they become widely recognized.
- **Independent evaluation**: YC3 works best when validators evaluate independently. There is no shared state, no off-chain coordination, and no published score files between validators.
- **Transient disagreements**: If two validators briefly disagree on the winner (due to EMA convergence timing or LLM variance), YC3 handles this naturally via stake-weighted bond dynamics. Bonds converge as validators accumulate more observations.

**Why not share scores?** Sharing scores between validators would:
1. Defeat commit-reveal (weights become predictable from shared data)
2. Enable weight-copying (skip evaluation, just read shared scores)
3. Make the off-chain aggregation redundant with on-chain YC3
4. Centralize all validators into one opinion

Independent evaluation preserves the security properties of decentralized consensus.

### Commit-Reveal

Commit-reveal is enabled on SN11 (`commit_reveal_period: 1` tempo). With independent evaluation, it serves its intended purpose:

- Validators cannot predict each other's weights (no shared data to derive from)
- Each validator must run its own evaluations to determine what weights to set
- Weight-copying is detectable: a copier always lags behind the original evaluator, and YC3's early-recognition bonus penalizes the lag

### YC3 Chain Configuration

| Parameter | Value | btcli command |
|-----------|-------|---------------|
| `yuma_version` | 3 | `btcli sudo set --param yuma_version --value 3 --netuid 11` |
| `liquid_alpha_enabled` | True | `btcli sudo set --param liquid_alpha_enabled --value true --netuid 11` |
| `commit_reveal_period` | 1 | Already set |
| `bonds_moving_avg` | 900000 (90%) | Tunable via `btcli sudo set --param bonds_moving_avg` |

### Yuma Consensus

YC3 operates as the consensus layer:
```
on_chain_weight[miner] = YC3(
    validator_weights[miner]
    for each validator,
    weighted by validator_stake,
    with per-bond EMA + Liquid Alpha
)
```

**Properties**:
- Each validator independently evaluates and sets weights
- YC3 aggregates with per-bond EMA, allowing individual validator-miner relationships to evolve at different rates
- Liquid Alpha rewards early recognition of promising miners
- Dishonest validators who set random or inflated weights get down-weighted by YC3 bond dynamics over time
- All scoring is regex-based within ClawBench, no LLM-as-judge dependency

### Validator Incentives

Validators earn rewards for:
- **Bond strength**: Proportional to agreement with consensus winner (YC3 bond dynamics)
- **Early recognition**: Liquid Alpha rewards validators who identify good miners before others
- **Active participation**: Setting weights regularly (validators who don't set weights get deregistered by the chain)
- **Honest evaluation**: Running ClawBench independently (no free-riding from shared data)

**Attack resistance**:
- Colluding validators can't fake miner packs (content-addressed + public repos)
- Dishonest validators who set inflated/deflated weights get down-weighted by YC3 bond dynamics
- Weight-copying is detectable and penalized (copier lags behind evaluator; Liquid Alpha rewards early discovery)
- No off-chain coordination required or possible — each validator's weights depend only on its own evaluations

---

## Pack Requirements

### Minimum Quality Thresholds

To earn non-zero rewards:

| Requirement | Threshold |
|-------------|-----------|
| Schema validation | MUST pass |
| Size limit | ≤ 32 KB |

Packs failing schema validation or exceeding the size limit receive **score = 0**. There is no minimum score threshold. Any valid pack is eligible to win.

### Pack Rejection Flow

A miner's submission can fail at multiple points in the validation pipeline. The table below specifies the exact outcome for each failure mode:

| Failure | Score | Weight | Counts as Active? | ClawBench Runs? |
|---------|:-----:|:------:|:------------------:|:---------------:|
| **No commitment** on-chain (or unparseable) | 0 | 0.0 | No | Skipped |
| **Pack URL inaccessible** (404, timeout, hash mismatch) | 0 | 0.0 | No | Skipped |
| **Schema validation failure** (missing AGENTS.md, >32KB, bad semver) | 0 | 0.0 | No | Skipped |
| **NCD similarity** ≥ threshold vs. current winner | 0 | 0.0 | No | Skipped |
| **ClawBench timeout** (scenario exceeds `timeout_per_scenario`) | 0 for that scenario | Computed | Yes | Partial |
| **ClawBench error** (LLM API failure, runtime crash) | 0 for that scenario | Computed | Yes | Partial |
| **Valid pack** | Computed | Computed | Yes | Full |

**Key rules**:

1. **Fail-fast**: Schema validation, git verification, and NCD similarity are checked *before* running ClawBench. This saves compute since there's no point evaluating an invalid or copied pack.

2. **"Active" means valid commitment**: A miner counts as "active" only if their on-chain commitment passes all pre-evaluation checks (schema, git, NCD) and at least one ClawBench scenario completes. This definition is used for:
   - Bootstrap threshold (need ≥10 *active* miners for winner-take-all)

3. **Partial failures are scored, not skipped**: If a pack passes schema but 1 of 5 scenarios times out, that scenario scores 0 in the EMA, but the other 4 still count. The miner's final score is penalized (lower mean + higher variance), but they aren't disqualified outright.

4. **Weight = 0.0 vs. omitted**: Miners who score 0 still receive `weight = 0.0` in the weight vector (not omitted). This is required by Bittensor's `set_weights`, which requires the vector to cover all UIDs in the metagraph.

### Competitive Range

Target ≥ 0.85 for competitive scores. See [MINER_OPERATIONS.md: Score Targets](MINER_OPERATIONS.md#score-targets) for the full ladder.

---

## Summary

### Scoring

```
# Per validator (repeated evaluation with per-scenario EMA):
ema[hotkey][scenario] = α × new_score + (1 - α) × ema[hotkey][scenario]
final_score[hotkey]   = weighted_mean(ema_scenarios) - ρ × weighted_variance(ema_scenarios)

# Weight setting (hotkey → UID via metagraph):
weight[uid] = f(final_score[hotkey_of(uid)])   # winner-take-all / bootstrap

# Cross-validator (YC3 on-chain):
on_chain_weight = YC3(validator_weights, validator_stakes, bond_history)
```

### Weights

```
# Steady state (≥ bootstrap_threshold active miners):
weight[winner] = 1.0
weight[all_others] = 0.0

# Bootstrap phase (< bootstrap_threshold active miners):
weight[1st] = 0.70
weight[2nd] = 0.20
weight[3rd] = 0.10

where winner = miner with highest final_score[hotkey] that satisfies:
  - score > previous_best + δ (if not first)
  - ties broken by earliest on-chain commitment block number
  - pack accessible at committed HTTP URL, hash matches
  - pack passes OPP v1 schema validation (AGENTS.md required, ≤32KB)
  - pack_similarity(pack, current_winner) < σ (NCD similarity check)
  - miner active within last inactivity_blocks
```

### Rewards

```
Steady state:  winner gets 100% of miner alpha emissions
Bootstrap:     top-3 get 70/20/10 of miner alpha emissions
```

### Key Parameters

| Parameter | Value | Tunable? |
|-----------|-------|----------|
| ρ (reliability weight) | 0.1 | Yes |
| δ (first-mover threshold) | 0.05 | Yes |
| α (EMA smoothing factor) | 0.3 | Yes |
| eval_interval | 4 hours (~1200 blocks) | Yes |
| Scenario pool | 5 (all run every eval; pool grows over time) | Yes |
| Scenario weights | 1.0-1.5 per YAML | Yes |
| Bootstrap threshold | 10 active miners | Yes |
| σ (similarity threshold) | 0.80 (NCD) | Yes |
| inactivity_blocks | 14400 (~48h) | Yes |
| yuma_version | 3 | Subnet owner (on-chain) |
| liquid_alpha_enabled | True | Subnet owner (on-chain) |
| commit_reveal_period | 1 tempo | Subnet owner (on-chain) |

---

## References

- **Bittensor Docs**: https://docs.bittensor.com
- **Dynamic TAO**: https://docs.bittensor.com/dtao
- **Yuma Consensus 3**: https://docs.learnbittensor.org/learn/yc3-blog
- **YC3 Migration Guide**: https://docs.learnbittensor.org/learn/yuma3-migration-guide
- **ClawBench**: https://github.com/trajectoryRL/clawbench
- **Evaluation Dataset**: [DATASET_v0.md](DATASET_v0.md) - current scenarios, rubric checks, evolution plans
- **Miner Guide**: [MINER_OPERATIONS.md](MINER_OPERATIONS.md) - reference miner, local testing, submission workflow
- **Validator Guide**: [VALIDATOR_OPERATIONS.md](VALIDATOR_OPERATIONS.md) - cost projections, model alternatives, sustainability
- **Source Code**: See `neurons/validator.py` and `trajectoryrl/` package

---

**Version**: v2.0

**Date**: 2026-03-02
