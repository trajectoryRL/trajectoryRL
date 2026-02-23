# TrajectoryRL Incentive Mechanism

**Subnet**: SN11 (TrajectoryRL)

**Version**: v1.06

**Date**: 2026-02-23

---

## Overview

TrajectoryRL rewards miners who submit **high-quality policy packs** (also called **PolicyBundles**) that optimize AI agent behavior for:
- ✅ **Safety**: no forbidden actions, approval gates respected
- ✅ **Correctness**: tasks completed successfully
- ✅ **Efficiency**: minimal tool calls and tokens
- ✅ **Reliability**: consistent performance across scenarios

Validators evaluate packs using **deterministic ClawBench scenarios** and set on-chain weights based on objective, reproducible scores.

> **Current Status**: Validator and ClawBench scoring are implemented. On-chain commitment submission and miner implementation are in progress, see [Status](#summary) for details.

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

### Aggregated Score

Across all scenarios (weighted average):

```
mean_score = Σ(w_i * scenario_score_i) / Σ(w_i)
variance   = Σ(w_i * (scenario_score_i - mean_score)²) / Σ(w_i)

final_score = mean_score - ρ*variance
```

Where:
- **w_i**: weight from scenario YAML (`weight` field, default 1.0). Safety-critical scenarios (e.g., `client_escalation`) use weight 1.5
- **ρ** = 0.1 (reliability penalty weight)
- **variance**: weighted variance across scenarios

Each validator runs each scenario **once** and publishes their raw `final_score` per UID to the shared score bucket (see [Validator Consensus](#validator-consensus)). The **consensus score** used for winner selection is a stake-weighted mean across all validators' published scores.

### Winner Selection

The miner with the highest `consensus_score` wins, subject to first-mover protection (δ = 0.05) and on-chain commitment block tie-breaking. See [Winner-Take-All with First-Mover Advantage](#winner-take-all-with-first-mover-advantage) for full rules.

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

The current evaluation dataset (**v0**) has 5 scenarios covering knowledge-worker tasks (email triage, client escalation, standup prep, inbox management). All 5 scenarios run every epoch.

This is an early dataset, not the final benchmark. The scenario pool will grow as the subnet matures (new scenarios, harder checks, new task domains). When scenarios are added or changed, all packs are re-evaluated.

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
- **Content-addressed**: `sha256(json.dumps(pack, sort_keys=True))` must match the `pack_hash` submitted on-chain

### What Goes in AGENTS.md

AGENTS.md is the primary policy document controlling agent behavior. It should be written as a **generic policy**. Avoid hardcoding specific names, companies, or dates, since the evaluation fixtures define the agent's identity context.

For the reference miner implementation and local testing, see [MINER_OPERATIONS.md](MINER_OPERATIONS.md).

---

## Submission Protocol

### On-Chain Commitments + Public GitHub

Miners publish packs to their own **public GitHub repository** and submit pack metadata **on-chain** via Bittensor's `set_commitment` extrinsic. Validators read submissions directly from the chain and fetch packs from the miner's repo. Miners do not need to run a server or have a public IP.

#### Submission Flow

**Step 1: Publish to GitHub**
- Create a public GitHub repository
- Commit PolicyBundle (`pack.json`) to the repo
- Push to GitHub

**Step 2: Commit on-chain**
- Miner calls `subtensor.set_commitment(netuid=11, data=commitment_string)` with their pack metadata
- The commitment contains: `pack_hash`, `git_commit_hash`, and `repo_url` (compact-encoded to fit 128-byte limit)
- The chain records the commitment with a **block-timestamped** entry (unforgeable and deterministic)
- Commit hash must be reachable from HEAD of the repo
- Rate limit: one commitment per ~100 blocks (~20 min) per hotkey, sufficient for daily epochs

**Step 3: Validator verification**
Each epoch, validators read all miner commitments from the chain via `subtensor.get_all_commitments(netuid=11)`, then verify:
1. Commitment is parseable and contains required fields (`pack_hash`, `git_commit_hash`, `repo_url`)
2. Repository is publicly accessible
3. Git commit hash exists and is valid
4. Pack content at that commit matches `pack_hash`
5. PolicyBundle passes schema validation
6. **NCD similarity** vs. current winner < `similarity_threshold` (0.80), see [Pack Similarity Detection](#7-pack-similarity-detection-ncd)

**First-mover precedence** is determined by the **on-chain commitment block number**. The pack must exist in the miner's repo at the referenced commit hash. If a miner force-pushes and the commit disappears, their commitment becomes invalid and they score 0.

**Why On-Chain Commitments?**
- **No server required**: Miners commit once and go offline. No public IP, no uptime requirement
- **Deterministic discovery**: All validators read the same chain state, eliminating disagreements from network failures or timeouts
- **Unforgeable timestamps**: Block-timestamped by the Substrate chain, not by the miner
- **Simple**: No P2P networking, no retry logic, no timeout handling

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

Once the 10th miner registers and submits, the next epoch automatically switches to winner-take-all. This is **deterministic**: every validator computes the same miner count from the metagraph, so they agree on which reward mode to use.

| Miners | Mode | Distribution |
|:------:|------|-------------|
| 1-9 | Bootstrap | Top-3: 70/20/10 |
| 10+ | Steady state | Winner-take-all: 100/0/0 |

### Always Set Weights

Validators **always call `set_weights` every epoch**, never skip. Validators that don't set weights get deregistered by the chain.

**Bootstrap at zero**: The initial best score is 0, so the **first miner to submit any valid pack immediately wins all the weight**. This gets beaten quickly as other miners join. There is no `min_score_threshold`. Any valid pack that passes schema and git verification is eligible to win.

If no miner has a valid on-chain commitment, the validator sets **uniform weights across all registered UIDs**. This is a degenerate case (dead subnet) that resolves itself as soon as any miner submits.

### Miner Inactivity

**Problem**: What if a miner registers, wins once, then never updates their commitment? Without explicit handling, the miner's stale pack could hold the throne indefinitely (protected by δ), and the miner could count toward the bootstrap threshold despite being inactive.

**Rules**:

1. **Activity window**: A miner is considered "active" if they have a valid on-chain commitment (passes schema + git verification) that was updated within the last `inactivity_window` epochs (default: 2 epochs = ~48 hours).

2. **Tracking**: Validators track `last_valid_epoch[miner_uid]`, the most recent epoch in which the miner's on-chain commitment passed pre-evaluation checks.

3. **Consequences of inactivity** (no valid submission for > `inactivity_window` epochs):

| Effect | Behavior |
|--------|----------|
| Score | 0 (no pack to evaluate) |
| Weight | 0.0 |
| Bootstrap threshold | Does NOT count. Only active miners count toward the 10-miner threshold |
| First-mover protection | **Lost**: an inactive incumbent's `current_best_score` is treated as 0, so any active challenger can claim the crown without crossing δ |
| Bittensor deregistration | Handled natively. Miners receiving weight 0.0 for extended periods eventually get deregistered by the chain when their immunity period expires |

4. **Re-activation**: If a previously inactive miner responds again with a valid pack, they re-enter the competition normally. Their `last_valid_epoch` updates, and they are subject to standard δ/NCD rules like any new submission.

**Why `inactivity_window = 2`?** At one epoch per 24 hours, this gives miners 48 hours (2 days) of downtime (maintenance, key rotation, etc.) before losing first-mover protection. Short enough to prevent indefinite stale-pack squatting; long enough to tolerate operational hiccups.

| Parameter | Value | Tunable? |
|-----------|-------|----------|
| `inactivity_window` | 2 epochs (~48h) | Yes |

### First-Mover Protection

To prevent copy-paste attacks, validators enforce **chronological precedence**:

**Rule**: A new submission can only become the winner if:
```
new_score > current_best_score + δ
```

Where:
- **δ** = 0.05 (5% improvement threshold)
- **current_best_score** = current epoch score of the first-mover
- Chronological order determined by **on-chain commitment block number** (unforgeable)

**Example Timeline**:
```
Epoch 1 - Miner A submits (score: 0.85)
  → Becomes winner (first submission)

Epoch 1 - Miner B submits (score: 0.87)
  → Rejected! Must beat 0.85 + 0.05 = 0.90

Epoch 3 - Miner C submits (score: 0.91)
  → Becomes new winner! (0.91 > 0.90)
```

**Anti-Copy Properties**:
- Direct copying (same score) never wins
- Minor tweaks to copied work fail the δ threshold
- Must genuinely innovate to dethrone the leader
- First-mover advantage rewards original research

**Anti-Stagnation**: The δ threshold alone could let an incumbent sit forever. See [Benchmark Stability](#benchmark-stability) for how growing the scenario pool and `inactivity_window` prevent this.

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
Example epoch (steady state):
Total miner alpha: 1000 tokens
Winner (score: 0.91): 1000 tokens (100%)
All other miners: 0 tokens

Example epoch (bootstrap, 5 miners):
Total miner alpha: 1000 tokens
1st place (score: 0.91): 700 tokens (70%)
2nd place (score: 0.87): 200 tokens (20%)
3rd place (score: 0.85): 100 tokens (10%)
```

### Competitive Strategy

Winner-take-all creates extreme risk/reward in steady state. The bootstrap phase (top-3 at 70/20/10) lowers the barrier for early miners. For practical mining strategy, iteration tips, and cost model, see [MINER_OPERATIONS.md](MINER_OPERATIONS.md).

---

## Operational Costs

- **Validators**: Bear all LLM inference costs. See [VALIDATOR_OPERATIONS.md](VALIDATOR_OPERATIONS.md) for cost projections and sustainability analysis.
- **Miners**: Zero ongoing cost (policy iteration only). See [MINER_OPERATIONS.md](MINER_OPERATIONS.md) for local testing costs and setup.

---

## Epochs

### What Is an Epoch?

An **epoch** is one complete evaluation cycle. The epoch number is derived from the **Bittensor block height** (`epoch_number = current_block // blocks_per_epoch`), so all validators agree on the same epoch regardless of when they started. Each epoch:

1. Syncs the Bittensor metagraph
2. Reads on-chain commitments, identifies new/changed packs (compare `pack_hash` to cache)
3. Fetches and verifies new packs from GitHub
4. Runs ClawBench **once** per new/changed pack on the **full scenario set**
5. Submits per-UID scores to the shared **validator-scores** repo via PR (see [Validator Consensus](#validator-consensus))
6. Pulls all validators' published scores, computes **stake-weighted mean** per UID
7. Selects winner and sets on-chain weights

### Epoch Timing

```
┌──────────────────────────────────────────────────────────────────────┐
│  Epoch N                                                             │
│                                                                      │
│  [Sync] → [Fetch Packs] → [Evaluate New] → [Publish] → [Aggregate]   │
│  ~1s      ~1-5 min         ~5-30 min        ~30s        ~30s         │
│                                                                      │
│  Every tempo (~72 min): re-pull scores, re-compute, set_weights      │
│                                                                      │
│  ──── epoch_interval (86400s / 24 hours) ────────────────────────    │
│                                                                      │
│  Epoch N+1 starts                                                    │
└──────────────────────────────────────────────────────────────────────┘
```

| Setting | Value | Description |
|---------|-------|-------------|
| `epoch_interval` | 86400s (24 hours) | Epoch length (~7200 blocks at 12s/block) |
| `timeout_per_scenario` | 120s (2 min) | Max time per scenario run |

**Typical cadence**: 1 epoch per day. Validators only run ClawBench on **new or changed** packs (detected by comparing `pack_hash` to cache). Unchanged packs carry forward their cached score. If no miner submits a new pack, the validator skips evaluation entirely and re-publishes cached scores.

**Weight cadence**: Validators `set_weights` on-chain every tempo (~72 minutes), not just once per epoch. Each time, they re-pull the latest scores from the shared validator-scores repo, re-compute the stake-weighted consensus, and re-submit. This keeps the validator active on-chain and allows consensus to converge as more validators publish their scores (see [Validator Consensus](#validator-consensus)).

### Benchmark Stability

Every epoch runs the **full scenario set**. No per-epoch subset selection or rotation. The benchmark is fixed and consistent: same scenarios, same rubric checks, same scoring. This ensures scores are directly comparable across validators and across time.

**Anti-stagnation** comes from the team **growing the scenario pool** over time (new scenarios, harder checks, new task domains). When the pool changes, it's coordinated via a validator software update. All packs are re-evaluated on the new set, and previously winning packs may lose their edge.

**Score persistence**: If a miner's pack hasn't changed (`pack_hash` matches cache), their cached score carries forward. Validators only run ClawBench on new or changed packs. When the **scenario pool itself changes**, all packs are re-evaluated.

---

## Anti-Gaming Measures

### 1. On-Chain Commitments + Content-Addressed Packs

**Enforcement**: All submissions must be git commits in public repos; first-mover precedence determined by **on-chain commitment block number** (unforgeable).

**Prevents**:
- `git commit --date` / `GIT_COMMITTER_DATE` timestamp forgery (on-chain block timestamp is the source of truth, git dates are ignored)
- Retroactive pack changes after seeing validator feedback (force-push is self-punishing: referenced commit disappears, score 0)
- Claims of earlier innovation without proof (on-chain commitment is permanent and block-timestamped)

**How it works**:
- Miner pushes pack to their public GitHub repo
- Miner calls `set_commitment` on-chain with `pack_hash` + `git_commit_hash` + `repo_url`
- On-chain commitment is block-timestamped by the Substrate chain (unforgeable and deterministic)
- Validators verify that the referenced commit exists and `sha256(pack.json)` matches `pack_hash`
- Force-push is self-punishing: if the commit disappears, the miner's commitment becomes invalid (score 0)
- Public repos allow community audit and verification

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

**Enforcement**: Validators run ClawBench in their own harness

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

---

## Validator Consensus

### The Problem: LLM Non-Determinism

LLM outputs vary between API calls even with the same input and temperature=0. Two independent validators evaluating the same pack may see different agent tool-call sequences and thus different rubric outcomes. Without mitigation, validators disagree on scores and winner selection, breaking Yuma consensus.

### Solution: Shared Score Buckets + Stake-Weighted Mean

Validators **share raw scores** and compute a **deterministic consensus** via stake-weighted aggregation.

#### Shared Score Repository

Validator scores are published to a team-managed GitHub repo: `trajectoryRL/validator-scores`. The repo is **append-only** for validators: they can only add score files via signed commits and pull requests. A CI pipeline verifies and auto-merges valid submissions.

```
validator-scores/
├── epoch-42/
│   ├── 5F3sa...hotkey_A.json     # scores from validator A
│   ├── 5Gw2p...hotkey_B.json     # scores from validator B
│   └── ...
└── epoch-43/
    └── ...
```

#### Submission Flow

Validators do **not** have direct write access to the repo. Instead, each validator:

1. **Forks** the `validator-scores` repo (one-time setup)
2. **Creates a score file** (`epoch-{N}/{hotkey}.json`) containing an sr25519 signature over the payload (see [Score File Schema](#score-file-schema))
3. **Commits and pushes** to their fork
4. **Opens a PR** against `trajectoryRL/validator-scores`

A **GitHub Actions CI pipeline** on the repo automatically validates each PR:
1. The PR only adds/updates a single file matching the pattern `epoch-{N}/{hotkey}.json`
2. The JSON passes schema validation (required fields: `validator_hotkey`, `epoch`, `block_height`, `scores`, `signature`)
3. The `signature` field is a valid sr25519 signature over the payload, signed by the `validator_hotkey`
4. The `validator_hotkey` matches a registered validator in the current metagraph
5. The validator has non-zero stake

If all checks pass, the CI **auto-merges** the PR. If any check fails, the PR is rejected with a comment explaining why.

**Why this design?**
- **No direct push access**: Validators cannot modify each other's files, delete history, or force-push. Only the CI bot merges.
- **Payload signatures**: The sr25519 signature inside the JSON is the authentication mechanism. Even if someone opens a PR, they cannot produce a valid signature without the validator's private key. Git commit signing is not used (sr25519 is incompatible with git's GPG/SSH signing).
- **Audit trail**: Every score submission is a PR, visible in GitHub's merge history.
- **Deterministic**: All validators pull from the same `main` branch and see the same merged scores.

#### Score File Schema

Each validator publishes a JSON file after completing evaluation:

```json
{
  "validator_hotkey": "5F3sa...",
  "epoch": 42,
  "block_height": 302400,
  "scores": {
    "uid_0": {
      "final_score": 0.87,
      "per_scenario": {
        "client_escalation": 0.92,
        "morning_brief": 0.85,
        "inbox_to_action": 0.88,
        "team_standup": 0.83
      }
    },
    "uid_1": {
      "final_score": 0.91,
      "per_scenario": { "..." : "..." }
    }
  },
  "signature": "0x3a1b..."
}
```

The `signature` field contains the sr25519 signature over the canonical JSON payload (all fields except `signature`, serialized with sorted keys and no extra whitespace). Per-scenario breakdowns are included for **auditability**: the community and team can investigate scoring patterns, identify overfitting, and debug disagreements.

#### Verification

Before including a score file in the stake-weighted aggregation, validators verify:
1. The `validator_hotkey` matches a registered validator in the current metagraph
2. The `signature` is a valid sr25519 signature over the payload, signed by that hotkey
3. The hotkey has non-zero stake (prevents deregistered validators from lingering)

These checks are redundant with the CI pipeline (defense-in-depth). Even if the CI is compromised, each validator independently rejects invalid score files.

#### Stake-Weighted Aggregation

All validators pull scores from the shared repo and compute:

```
consensus_score[uid] = Σ(stake_i * score_i[uid]) / Σ(stake_i)

where stake_i = validator i's TAO stake from metagraph
```

The winner is the miner with the highest `consensus_score`, subject to the δ first-mover rule. Ties are broken by earliest on-chain commitment block number (deterministic).

Since every validator reads the **same repo** and the **same metagraph stakes**, they all compute the **same consensus scores** → same winner → 100% agreement on `set_weights`.

#### Timing: Rolling Async Convergence

Validators operate **asynchronously** with no synchronized phases or deadlines:

1. **Evaluate**: Run ClawBench once per new/changed pack
2. **Publish**: Submit PR to validator-scores repo (CI verifies payload signature, auto-merges)
3. **Aggregate**: Pull all available scores, compute stake-weighted mean
4. **Set weights**: Call `set_weights` on-chain
5. **Repeat**: Every tempo (~72 min), re-pull scores, re-compute, re-submit weights

```
Example timeline (epoch 42, 3 validators):

Hour 0.0:  Epoch 42 starts
Hour 0.5:  Val A (fast) finishes eval → publishes scores
           Val A pulls (sees only A) → set_weights based on own scores
Hour 1.0:  Val B finishes → publishes
           Val A re-pulls (sees A+B) → re-computes → set_weights
           Val B pulls (sees A+B) → set_weights
Hour 1.5:  Val C finishes → publishes
           All pull (see A+B+C) → all compute same consensus → set_weights
Hour 2-24: All validators re-submit same converged weights every tempo
```

**Convergence**: Early `set_weights` calls may briefly disagree (each validator only sees a subset of scores), but as more validators publish, they all see the same data and converge. By the time all validators have published (typically within 1-2 hours), consensus is 100%. Bittensor's Yuma consensus handles the brief transient disagreement naturally, running every tempo and stake-weighting validator opinions.

#### Score Persistence

Scores **persist across epochs**. Validators only re-evaluate a UID when its `pack_hash` changes. See [Benchmark Stability](#benchmark-stability) for details.

#### Resilience and Recovery

The validator-scores repo is a **coordination layer**, not the ultimate source of truth. On-chain weights (set via `set_weights`) are immutable and determine emissions. The repo only helps validators converge on which weights to set.

**Why the repo is not a single point of failure:**

1. **Git is distributed.** Every validator maintains a fork with the full commit history. If the upstream repo is lost or corrupted, it can be reconstructed from any validator's fork.
2. **Scores are reconstructable.** ClawBench scoring is deterministic (regex-based, no LLM judge). Given the same pack (content-addressed, stored on miners' public GitHub repos) and the same model, any validator can re-run evaluation and produce identical scores.
3. **On-chain weights survive.** Previous `set_weights` calls are recorded on the Bittensor chain and cannot be altered. Only the current epoch's consensus process would be disrupted during recovery.
4. **Local retention.** Validators store their own computed scores locally before publishing. A repo outage does not erase data that validators already hold.

**Recovery procedure** (if the upstream repo is lost or compromised):

1. Team designates a validator fork with the most complete history as the new upstream
2. Other validators re-point their forks and resume PR submissions
3. Any missing epochs can be backfilled from validators' local score caches
4. If a full rebuild is needed, validators re-evaluate current packs (unchanged `pack_hash` values are still on-chain via `set_commitments`)

**What cannot be recovered from the repo alone:** The repo does not store on-chain data (miner commitments, validator stakes, weight history). That data lives on the Bittensor chain and is independently preserved by the substrate network.

### Yuma Consensus

Bittensor's Yuma Consensus still operates as the final layer:
```
on_chain_weight[miner] = yuma_consensus(
    validator_weights[miner]
    for each validator,
    weighted by validator_stake
)
```

**With shared score buckets**:
- Validators converge on the same winner because they compute from the same shared data
- Brief transient disagreements (before all scores are published) are handled by Yuma's stake-weighting
- Dishonest validators who publish fake scores get down-weighted by Yuma consensus over time
- All scoring is regex-based within ClawBench, no LLM-as-judge dependency
- Published scores are publicly auditable in the validator-scores repo

### Validator Incentives

Validators earn rewards for:
- Agreement with consensus winner (validator bonding)
- Setting weights regularly (not idle)
- Running valid evaluations (not random)
- Publishing scores to the shared repo (required for others to include your scores)

**Attack resistance**:
- Colluding validators can't fake miner packs (content-addressed + public repos)
- Dishonest validators who publish inflated/deflated scores get down-weighted by Yuma consensus
- **No direct write access**: Validators submit scores via PRs, auto-merged by CI after verification
- **Signed score files** prevent impersonation: each file contains an sr25519 payload signature, verified both by CI and by every validator on pull
- Community can audit every validator's per-UID, per-scenario scores in the public validator-scores repo
- Stake-weighted mean ensures high-stake validators have proportionally more influence

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
| **Invalid git repo** (404, private, bad commit) | 0 | 0.0 | No | Skipped |
| **Schema validation failure** (missing AGENTS.md, >32KB, bad semver) | 0 | 0.0 | No | Skipped |
| **NCD similarity** ≥ threshold vs. current winner | 0 | 0.0 | No | Skipped |
| **ClawBench timeout** (scenario exceeds `timeout_per_scenario`) | 0 for that scenario | Computed | Yes | Partial |
| **ClawBench error** (LLM API failure, runtime crash) | 0 for that scenario | Computed | Yes | Partial |
| **Valid pack** | Computed | Computed | Yes | Full |

**Key rules**:

1. **Fail-fast**: Schema validation, git verification, and NCD similarity are checked *before* running ClawBench. This saves compute since there's no point evaluating an invalid or copied pack.

2. **"Active" means valid commitment**: A miner counts as "active" only if their on-chain commitment passes all pre-evaluation checks (schema, git, NCD) and at least one ClawBench scenario completes. This definition is used for:
   - Bootstrap threshold (need ≥10 *active* miners for winner-take-all)

3. **Partial failures are scored, not skipped**: If a pack passes schema but 1 of 5 scenarios times out, that scenario scores 0, but the other 4 still count. The miner's final score is penalized (lower mean + higher variance), but they aren't disqualified outright.

4. **Weight = 0.0 vs. omitted**: Miners who score 0 still receive `weight = 0.0` in the weight vector (not omitted). This is required by Bittensor's `set_weights`, which requires the vector to cover all UIDs in the metagraph.

### Competitive Range

Target ≥ 0.85 for competitive scores. See [MINER_OPERATIONS.md: Score Targets](MINER_OPERATIONS.md#score-targets) for the full ladder.

---

## Summary

### Scoring

```
# Per validator (single run per scenario):
scenario_score = earned_points / total_points
final_score    = weighted_mean(scenario_scores) - ρ*variance

# Across validators (shared score bucket):
consensus_score[uid] = Σ(stake_i * score_i[uid]) / Σ(stake_i)
```

### Weights

```
# Steady state (≥ bootstrap_threshold miners):
weight[winner] = 1.0
weight[all_others] = 0.0

# Bootstrap phase (< bootstrap_threshold miners):
weight[1st] = 0.70
weight[2nd] = 0.20
weight[3rd] = 0.10

where winner = miner with highest consensus_score that satisfies:
  - score > previous_best + δ (if not first)
  - ties broken by earliest on-chain commitment block number
  - public GitHub repo with valid commit
  - pack passes OPP v1 schema validation (AGENTS.md required, ≤32KB)
  - pack_similarity(pack, current_winner) < σ (NCD similarity check)
  - miner active within last inactivity_window epochs
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
| Scenario pool | 5 (all run every epoch; pool grows over time) | Yes |
| Scenario weights | 1.0-1.5 per YAML | Yes |
| Bootstrap threshold | 10 miners | Yes |
| Epoch interval | 86400s (24h) | Yes |
| σ (similarity threshold) | 0.80 (NCD) | Yes |
| Inactivity window | 2 epochs (~48h) | Yes |

---

## References

- **Bittensor Docs**: https://docs.bittensor.com
- **Dynamic TAO**: https://docs.bittensor.com/dtao
- **Yuma Consensus**: https://docs.bittensor.com/yuma-consensus
- **ClawBench**: https://github.com/trajectoryRL/clawbench
- **Evaluation Dataset**: [DATASET_v0.md](DATASET_v0.md) - current scenarios, rubric checks, evolution plans
- **Miner Guide**: [MINER_OPERATIONS.md](MINER_OPERATIONS.md) - reference miner, local testing, submission workflow
- **Validator Guide**: [VALIDATOR_OPERATIONS.md](VALIDATOR_OPERATIONS.md) - cost projections, model alternatives, sustainability
- **Source Code**: See `neurons/validator.py` and `trajectoryrl/` package

---

**Version**: v1.06

**Date**: 2026-02-23

**Status**: Implemented (validator + ClawBench scoring). Pending: on-chain commitment submission, shared score bucket, miner implementation.
