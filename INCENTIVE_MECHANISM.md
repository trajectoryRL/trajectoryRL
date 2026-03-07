# TrajectoryRL Incentive Mechanism

**Subnet**: SN11 (TrajectoryRL)

**Version**: v3.1

**Date**: 2026-03-07

---

## Overview

TrajectoryRL rewards miners who submit **policy packs** (PolicyBundles) that complete AI agent tasks at the **lowest cost** while passing all safety and correctness checks.

The incentive is simple: **pass the gate, then compete on cost**.

1. **Qualification gate**: Every safety and correctness check must pass (binary PASS/FAIL per scenario)
2. **Cost competition**: Among qualified miners, the one with the lowest cost wins

Validators evaluate packs independently using **deterministic ClawBench scenarios** and set on-chain weights based on objective, reproducible results. **Yuma Consensus 3 (YC3)** aggregates independent validator weights on-chain.

---

## Value Proposition

### For Miners
Earn subnet alpha (swappable for TAO) by submitting winning PolicyBundles (system prompt + tool policies + stop rules) that pass all safety/correctness checks at the lowest inference cost.

**Optimization strategies**:
- Prompt engineering to reduce token usage while maintaining correctness
- Tool policy tuning to minimize unnecessary tool calls
- Stop rules to prevent loops and retries
- Multi-LLM routing via AGENTS.md + injected skills: dispatch sub-tasks to the lowest-cost capable model

### For Enterprises
Winning policies get packaged and licensed to agent platforms who pay for:
- Optimized policy packs (drop-in AGENTS.md + skills)
- Evaluation-as-a-service (ClawBench validation)
- Hybrid routing configurations (multi-LLM orchestration for dramatic cost reduction)

**Example ROI** (1,000 tasks/day):
```
Unoptimized Claude Opus 4.6:             $12,300/month

Stage 1 — Prompt optimization (AGENTS.md tuning):
  Optimized prompts + stop rules:         $3,300/month  (73% reduction)

Stage 2 — Hybrid routing (AGENTS.md + injected skills):
  Multi-LLM dynamic routing:               $900/month  (93% reduction)
    ├─ Qwen 3.5 (Alibaba) handles 40% of sub-tasks (tool calls, lookups)
    ├─ GLM-5 (Z.ai) handles 25% (structured extraction, formatting)
    ├─ Gemini 3 Flash (Google) handles 20% (search, summarization)
    ├─ GPT-5.2 (OpenAI) handles 10% (reasoning, drafting)
    └─ Claude Opus 4.6 (Anthropic) handles 5% (complex judgment calls)
```

Both stages use the same PolicyBundle format — AGENTS.md controls routing logic, injected skills handle model dispatch. The pack system is general enough to express prompt-only optimizations *and* multi-model orchestration.

Revenue flows back to the subnet through licensing fees, API access, and marketplace commissions.

---

## Scoring: Qualification Gate + Cost Ranking

### Step 1: Qualification Gate (Binary PASS/FAIL)

Each scenario defines rubric **checks** categorized as `safety`, `correctness`, `efficiency`, or `structure`. The qualification gate requires that **every safety and correctness check passes**:

```
qualified[scenario] = all safety checks passed AND all correctness checks passed
qualified[miner]    = qualified on ALL scenarios
```

Efficiency and structure checks are informational only — they do not affect qualification. A single failed safety or correctness check on any scenario disqualifies the miner from cost competition.

**Examples of gate checks**:
- **Safety**: no unauthorized emails sent, confidential data not leaked, approval gates respected
- **Correctness**: required actions completed, correct recipients addressed, proper tool usage

### Step 2: Cost Measurement

For each scenario evaluation, the validator measures the total inference cost in USD:

```
cost_usd = Σ  rate(model_i, token_type) × token_count(model_i, token_type)
           i
```

Cost is captured from the LLM provider's usage data after each episode. It includes all tokens consumed during the agent's tool-calling loop across **all models used** — a pack that routes sub-tasks to Haiku, Sonnet, and Opus accumulates cost from each model at its respective rate.

### Step 3: Per-Scenario Cost EMA

Validators smooth per-scenario costs using an Exponential Moving Average (EMA):

```
ema_cost[hotkey][scenario] = α × new_cost + (1 - α) × ema_cost[hotkey][scenario]
```

Where:
- **hotkey**: the miner's ss58 address (stable across UID recycling)
- **α** = 0.3 (cost EMA smoothing factor, configurable)
- **new_cost**: measured cost from the latest evaluation run

When a miner submits a new pack (different `pack_hash`), the cost EMA resets for that hotkey — old cost observations from a different pack are irrelevant.

### Step 4: Aggregated Cost

From the smoothed per-scenario cost EMA values:

```
total_cost[hotkey] = Σ(w_i × ema_cost[hotkey][scenario_i]) / Σ(w_i)
```

Where **w_i** is the weight from each scenario YAML (`weight` field, default 1.0).

### Step 5: Winner Selection

Among **qualified** miners, the one with the lowest `total_cost` wins. See [Winner-Take-All with First-Mover Advantage](#winner-take-all-with-first-mover-advantage) for full rules.

### Informational Score EMA

Validators also maintain a per-scenario score EMA for logging and debugging:

```
ema_score[hotkey][scenario] = α × new_score + (1 - α) × ema_score[hotkey][scenario]
final_score[hotkey] = weighted_mean(ema_scores) - ρ × weighted_variance(ema_scores)
```

This score is **not used for winner selection** — it exists purely for monitoring. The score represents the fraction of gate checks passed and provides a diagnostic signal.

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
5. **NCD similarity** pairwise dedup among all active miners (see [Pack Similarity Detection](#6-pack-similarity-detection-ncd))

**First-mover precedence** is determined by the **on-chain commitment block number**. The pack must be accessible at the committed URL. If a miner deletes or changes the file so the hash no longer matches, their commitment becomes invalid and they receive weight 0.

**Why On-Chain Commitments + HTTP?**
- **No server required**: Miners upload once to static hosting and go offline. No public IP, no uptime requirement
- **Deterministic discovery**: All validators read the same chain state, eliminating disagreements from network failures or timeouts
- **Unforgeable timestamps**: Block-timestamped by the Substrate chain, not by the miner
- **Simple**: No P2P networking, no retry logic, no timeout handling
- **Flexible hosting**: Any HTTP(S) endpoint works — S3, GCS, GitHub Pages, personal servers, IPFS gateways, etc.

---

## Winner-Take-All with First-Mover Advantage

### Core Rule: Winner Takes All (Steady State)

**Winner** = lowest-cost qualified miner. In steady state (≥`bootstrap_threshold` active miners, default 10), **only the Winner receives rewards**:

```
weight[winner] = 1.0
weight[others] = 0.0
```

Disqualified miners (any failed safety or correctness check) receive weight 0 regardless of cost.

### Bootstrap Phase (Early Adoption)

When active miners < `bootstrap_threshold` (default 10), rewards use a **graduated top-3 curve** among qualified miners, ranked by lowest cost:

```
weight[1st] = 0.70  (70%)
weight[2nd] = 0.20  (20%)
weight[3rd] = 0.10  (10%)
weight[others] = 0.0
```

Ties within a rank are broken by earliest on-chain commitment (same rule as steady-state).

**Example** (bootstrap phase, 5 active miners):
```
Miner A ($2.30/episode, qualified): 70% of miner alpha  ← 1st
Miner B ($3.10/episode, qualified): 20% of miner alpha  ← 2nd
Miner C ($4.50/episode, qualified): 10% of miner alpha  ← 3rd
Miner D ($1.80/episode, DISQUALIFIED): 0%  ← failed safety check
Miner E ($5.60/episode, qualified):  0%
```

Once the 10th active miner submits, the validator automatically switches to winner-take-all.

| Active Miners | Mode | Distribution |
|:------:|------|-------------|
| 1-9 | Bootstrap | Top-3 qualified: 70/20/10 |
| 10+ | Steady state | Winner-take-all: 100/0/0 |

### Always Set Weights

Validators **always call `set_weights` every tempo**, never skip. Validators that don't set weights get deregistered by the chain.

**Bootstrap at zero**: The **first miner to submit any valid pack that passes the qualification gate immediately wins all the weight**. This gets beaten quickly as other miners optimize costs. There is no minimum cost threshold. Any qualified pack is eligible to win.

If no miner has a valid on-chain commitment, the validator sets **uniform weights across all registered UIDs**. This is a degenerate case (dead subnet) that resolves itself as soon as any miner submits.

### Miner Inactivity

**Problem**: What if a miner registers, wins once, then never updates their commitment? Without explicit handling, the miner's stale pack could hold the throne indefinitely (protected by first-mover), and the miner could count toward the bootstrap threshold despite being inactive.

**Rules**:

1. **Activity window**: A miner is considered "active" if they have a valid on-chain commitment (passes schema + verification) that was last successfully evaluated within `inactivity_blocks` (default: 14400 blocks ≈ 48 hours at 12s/block).

2. **Tracking**: Validators track `last_eval_block[hotkey]` — the block height at which the miner's pack was last successfully evaluated. Keyed by miner hotkey (ss58 address), not UID, so history is never inherited on UID recycling.

3. **Consequences of inactivity** (`current_block - last_eval_block[hotkey] > inactivity_blocks`):

| Effect | Behavior |
|--------|----------|
| Cost | N/A (no pack to evaluate) |
| Weight | 0.0 |
| Bootstrap threshold | Does NOT count. Only active miners count toward the 10-miner threshold |
| First-mover protection | **Lost**: an inactive incumbent is removed from cost competition, so any active challenger can claim the crown |
| Bittensor deregistration | Handled natively. Miners receiving weight 0.0 for extended periods eventually get deregistered by the chain when their immunity period expires |

4. **Re-activation**: If a previously inactive miner submits a valid pack, they re-enter the competition normally. Their `last_eval_block` updates, and they are subject to standard first-mover/NCD rules like any new submission.

**Why 14400 blocks (~48h)?** This gives miners 48 hours of downtime (maintenance, key rotation, etc.) before losing first-mover protection. Short enough to prevent indefinite stale-pack squatting; long enough to tolerate operational hiccups.

| Parameter | Value | Tunable? |
|-----------|-------|----------|
| `inactivity_blocks` | 14400 (~48h) | Yes |

### First-Mover Protection (Cost-Based)

To prevent copy-paste attacks and cheap-by-epsilon undercutting, validators enforce **chronological precedence with a multiplicative cost threshold**:

**Rule**: A later submission can only dethrone the current champion if:
```
new_cost < current_best_cost × (1 - δ)
```

Where:
- **δ** = 0.10 (10% cost improvement threshold)
- **current_best_cost** = current EMA-smoothed cost of the first-mover
- Chronological order determined by **on-chain commitment block number** (unforgeable)

**Example Timeline**:
```
Miner A submits at block 1000 (cost: $5.00/episode)
  → Becomes winner (first qualified submission)

Miner B submits at block 1200 (cost: $4.80/episode)
  → Rejected! Must beat $5.00 × 0.90 = $4.50

Miner C submits at block 5000 (cost: $3.80/episode)
  → Becomes new winner! ($3.80 < $4.50)
```

**Why multiplicative (not additive)?** Cost is measured in dollars, not a [0,1] score. A fixed additive threshold (e.g., $0.50) would be meaningless for a $50 episode but prohibitive for a $1 episode. A 10% multiplicative threshold scales naturally with the cost range.

**Anti-Copy Properties**:
- Direct copying (same cost) never wins
- Minor optimizations (< 10% cheaper) fail the threshold
- Must genuinely reduce cost to dethrone the leader
- First-mover advantage rewards original research

**Anti-Stagnation**: The δ threshold alone could let an incumbent sit forever. Growing the scenario pool and `inactivity_blocks` prevent this.

### Reward Distribution

**Steady state** (≥10 miners):
```
Miner C ($3.80, qualified):  100% of miner alpha  ← Winner
Miner A ($5.00, qualified):           0%
Miner B ($4.80, qualified):           0%  (not 10% cheaper than Winner)
Miner D ($1.50, DISQUALIFIED):        0%  (failed safety check)
```

**Bootstrap** (<10 miners):
```
Miner C ($3.80, qualified): 70% of miner alpha
Miner A ($5.00, qualified): 20% of miner alpha
Miner B ($4.80, qualified): 10% of miner alpha
```

**Expected Behavior**:
- **Early days**: Top-3 rewards lower the barrier to entry and incentivize experimentation
- **Growth**: As more miners join, competition intensifies toward winner-take-all
- **Steady state**: Intense competition to be FIRST to reduce cost; public policies create a learning flywheel
- **End game**: Multi-LLM routing packs dramatically undercut single-model costs by dispatching each sub-task to the lowest-cost capable model

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

**Steady state** (≥10 miners): all miner alpha goes to the Winner.
**Bootstrap** (<10 miners): top-3 qualified split 70/20/10.

```
Example (steady state):
Total miner alpha: 1000 tokens
Winner ($3.80/episode, qualified): 1000 tokens (100%)
All other miners: 0 tokens

Example (bootstrap, 5 miners):
Total miner alpha: 1000 tokens
1st ($3.80, qualified): 700 tokens (70%)
2nd ($5.00, qualified): 200 tokens (20%)
3rd ($5.60, qualified): 100 tokens (10%)
```

### Competitive Strategy

Winner-take-all creates extreme risk/reward in steady state. The bootstrap phase (top-3 at 70/20/10) lowers the barrier for early miners. For practical mining strategy, iteration tips, and cost model, see [MINER_OPERATIONS.md](MINER_OPERATIONS.md).

---

## Operational Costs

- **Validators**: Bear all LLM inference costs. Daily evaluation (~1 run/day per miner) remains well within validator earnings. See [VALIDATOR_OPERATIONS.md](VALIDATOR_OPERATIONS.md) for cost projections and sustainability analysis.
- **Miners**: Zero ongoing cost (policy iteration only). See [MINER_OPERATIONS.md](MINER_OPERATIONS.md) for local testing costs and setup.

---

## Evaluation Cadence

Validators run a **continuous evaluation loop** with two cadences:

| Cadence | Default | Purpose |
|---------|---------|---------|
| `eval_interval` | 24 hours (~7200 blocks) | Re-evaluate all active packs, update per-scenario EMA |
| `tempo` | ~72 min (360 blocks, chain-determined) | Set weights on-chain via commit-reveal |

### Continuous Validator Loop

```
while running:
  1. Sync metagraph, read on-chain commitments
  2. Pack-hash pre-dedup: group miners by pack_hash, skip evaluation
     for exact copies (only evaluate the first mover per pack_hash)
  3. For each remaining miner hotkey with valid commitment:
     - If pack_hash changed since last eval: mark for re-evaluation
     - If time since last eval ≥ eval_interval: mark for re-evaluation
  4. Evaluate marked packs on the full scenario set
     (rate limit: at most 1 eval per hotkey per eval_interval)
  5. Update per-scenario cost EMA and qualification status
  6. Every tempo: pairwise NCD dedup among all active miners,
     then compute weights from cost + qualification, set_weights via commit-reveal
```

### Rate-Limiting (Anti-DDoS)

Evaluation is rate-limited to **at most one evaluation per miner per `eval_interval`**, regardless of how often the miner updates their on-chain commitment. This prevents a miner from draining validator API budgets via rapid commitment churn.

- If a miner submits a new `pack_hash` within the current eval_interval window, the validator **notes** the new hash but waits for the next scheduled eval slot
- At the next eval slot, the validator evaluates the **latest** `pack_hash` for that miner
- A miner who submits 100 times per hour gets evaluated exactly the same number of times as one who submits once
- The cost EMA resets when a new `pack_hash` is first *evaluated* (not when committed on-chain)

### Timing

```
┌────────────────────────────────────────────────────────────────────────┐
│  Continuous Operation                                                  │
│                                                                        │
│  ├─ eval_interval (~24h): evaluate all active packs, update EMA        │
│  │   [Sync] → [Check Commitments] → [Evaluate Marked] → [Update EMA]   │
│  │   ~1s      ~1-2 min               ~5-30 min           ~instant      │
│  │                                                                     │
│  └─ tempo (~72min): compute weights from cost EMA, set_weights via CR  │
│      [Map hotkey→UID] → [Select Winner] → [commit-reveal set_weights]│
│      ~instant            ~instant            ~30s                      │
│                                                                        │
│  Miner inactivity checked continuously:                                │
│    current_block - last_eval_block[hotkey] > 14400 → inactive          │
└────────────────────────────────────────────────────────────────────────┘
```

| Setting | Value | Description |
|---------|-------|-------------|
| `eval_interval` | 24 hours (~7200 blocks) | How often to re-evaluate each active pack |
| `timeout_per_scenario` | 120s (2 min) | Max time per scenario run |

### Benchmark Stability

Every evaluation runs the **full scenario set**. No subset selection or rotation. The benchmark is fixed and consistent: same scenarios, same rubric checks, same scoring. This ensures costs and qualification are directly comparable across validators and across time.

**Anti-stagnation** comes from the team **growing the scenario pool** over time (new scenarios, harder checks, new task domains). When the pool changes, it's coordinated via a validator software update. All EMA state is invalidated and packs are re-evaluated fresh on the new set.

**EMA persistence**: Per-scenario cost and score EMA state persists across validator restarts (serialized to disk). When a pack hasn't changed (`pack_hash` matches), the validator continues accumulating EMA samples. When the **scenario pool itself changes** (detected by hash of scenario configuration), all EMA state is invalidated.

---

## Anti-Gaming Measures

### 1. On-Chain Commitments + Content-Addressed Packs

**Enforcement**: All submissions are content-addressed (SHA256 hash); first-mover precedence determined by **on-chain commitment block number** (unforgeable).

**Prevents**:
- Retroactive pack changes after seeing validator feedback (changing the file breaks the hash → weight 0)
- Claims of earlier innovation without proof (on-chain commitment is permanent and block-timestamped)
- Timestamp forgery (on-chain block timestamp is the source of truth)

**How it works**:
- Miner uploads pack to any public HTTP endpoint (S3, GCS, etc.)
- Miner calls `set_commitment` on-chain with `pack_hash` + `pack_url`
- On-chain commitment is block-timestamped by the Substrate chain (unforgeable and deterministic)
- Validators fetch the pack via HTTP and verify `sha256(json.dumps(pack, sort_keys=True))` matches `pack_hash`
- Deleting or modifying the hosted file is self-punishing: hash mismatch → weight 0
- Public URLs allow community audit and verification

### 2. First-Mover Advantage (Multiplicative δ)

**Enforcement**: New submission must cost `< current_best × (1 - δ)` to win (δ = 0.10, i.e. 10% cheaper)

**Prevents**:
- Direct copy-paste attacks (same cost fails)
- Cheap-by-epsilon undercutting (< 10% cheaper fails)
- Lazy free-riding on others' research

**How it works**:
- Track the FIRST qualified submission and its cost
- Later submissions must meaningfully reduce cost (10%+)
- Creates incentive to publish cost optimizations quickly
- Anti-stagnation comes from growing the scenario pool over time

### 3. Qualification Gate

**Enforcement**: Binary PASS/FAIL on all safety and correctness checks. Disqualified miners receive weight 0 regardless of cost.

**Prevents**:
- Racing to the bottom by cutting corners on safety
- Cheap-but-dangerous policies that skip approval gates
- Cost optimization at the expense of correctness

**How it works**:
- Each scenario has categorized rubric checks (safety, correctness, efficiency, structure)
- All `safety` and `correctness` checks must pass for the scenario to be qualified
- A miner must be qualified on ALL scenarios to compete on cost
- One failed safety check on one scenario → disqualified from cost competition

### 4. Winner-Take-All

**Enforcement**: Only the Winner receives rewards (weight = 1.0)

**Prevents**:
- "Good enough" submissions that copy leaders
- Minimal-effort mining for small rewards
- Sybil attacks (multiple mediocre miners)

**How it works**:
- Zero reward for 2nd place eliminates copy-paste ROI
- Forces miners to either innovate on cost or exit
- Creates winner-take-all tournament dynamics

### 5. Validator-Side Evaluation

**Enforcement**: Validators run ClawBench independently in their own harness

**Prevents**:
- Miners faking costs or qualification
- Environment manipulation
- Replay attacks

### 6. Pack Similarity Detection (NCD)

**Enforcement**: Validators perform **pairwise** similarity comparison among **all** active miners' packs using **Normalized Compression Distance (NCD)**. For each pair of similar packs (similarity ≥ threshold), the later submitter (higher on-chain `block_number`) is excluded with weight = 0. The first mover (lower `block_number`) is always preserved.

**Prevents**:
- Copy-paste with minor edits (add whitespace, reword a sentence)
- Synonym substitution attacks
- Paragraph reordering to evade naive diff checks
- "Wrapper" packs that embed another miner's policy inside boilerplate
- Sybil attacks where one entity runs multiple miners with the same or similar pack
- Multiple miners using the same pack URL or hosting identical content on different URLs

**Why NCD?** The δ threshold (measure #2) stops copycats who achieve *the same cost*. But a cheater who copies another miner's AGENTS.md, tweaks a few lines, and gets lucky with LLM variance could cross the δ bar. NCD catches the copy regardless of cost.

**How it works**:

NCD measures how much "new information" one text contains relative to another, using compression as a proxy for information content:

```
NCD(x, y) = (C(x+y) - min(C(x), C(y))) / max(C(x), C(y))

Where C(·) = len(zlib.compress(·, level=9))
```

If two texts are nearly identical, compressing them *together* barely increases the size vs. compressing each alone, because the compressor finds the repeated patterns. The ratio tells you how much genuinely new content exists between the two packs.

**Two-layer dedup**:

Deduplication runs in the **weight-setting phase** (before winner selection) with two layers:

```
Layer 1: Pack-hash grouping (O(N), catches exact copies)
  → Group miners by pack_hash
  → For each group with multiple miners, keep only the one
    with the lowest on-chain block_number (first mover)
  → All others get weight = 0

Layer 2: NCD pairwise among unique packs (catches paraphrases)
  → Pre-compute normalized AGENTS.md + compression for each unique pack
  → Compare all unique pairs
  → For similar pairs (similarity ≥ threshold), the later submitter
    (higher block_number) gets weight = 0
```

Additionally, during the **evaluation phase**, miners with duplicate `pack_hash` are skipped to save LLM API costs. Only the first mover is evaluated; the rest are excluded before ClawBench runs.

**Priority determination**: On-chain `block_number` from `subtensor.get_commitment_metadata()` determines who submitted first. This is unforgeable and consistent across validators and across evaluation cycles — a miner who committed at block 100 will always be recognized as earlier than one who committed at block 200, regardless of evaluation order or which epoch the comparison happens in.

**Why pairwise instead of winner-only?** A previous design only compared each miner against the current winner. This left a gap: two miners submitting identical packs would both pass the NCD check as long as neither was the winner. The pairwise approach eliminates this gap — every pair of similar packs is detected, and the later submitter is always excluded.

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

**Performance**: With 200 miners and ~80 unique packs, Layer 1 is O(N) and Layer 2 requires C(80,2) = 3,160 NCD comparisons. Each comparison is one `zlib.compress` call (~50μs), totaling ~0.15 seconds. This is negligible compared to the hours spent on LLM evaluation.

**Why NCD is hard to game**:

| Evasion Attempt | Why It Fails |
|-----------------|--------------|
| Add whitespace / newlines | `normalize_policy` collapses all whitespace; compressor ignores repetitive bytes |
| Reorder paragraphs | zlib uses a 32KB sliding window, so reordered blocks still match within the window |
| Substitute synonyms | Only works if you change *enough* words that the policy is genuinely different, at which point δ becomes the real barrier |
| Insert junk comments | Junk compresses independently of the original; the shared content still compresses together. Also wastes the 32KB pack size budget |
| Pad with random text | Random data doesn't compress well, inflating `C(challenger)` and `C(concat)` proportionally, so NCD stays high |
| Wrap winner's policy in boilerplate | The core policy still compresses against the original; boilerplate adds marginal `C(concat)` cost |
| Host same content on a different URL | NCD compares AGENTS.md content, not URLs. Changing the URL has zero effect on similarity |

**Properties**:
- **Deterministic**: Fixed `zlib.compress(level=9)`, so every validator computes the same similarity score
- **Zero dependencies**: Python stdlib (`zlib`, `re`) only
- **Fast**: ~1ms per pair for two 32KB texts; full pairwise under 1 second
- **Well-studied**: NCD is used in academic plagiarism detection, malware classification, and DNA sequence comparison

**Threshold rationale** (σ = 0.80):
- **≥ 0.80**: Very likely a copy with edits, excluded (later submitter gets weight 0)
- **0.60–0.80**: Gray zone. Independently developed packs may share common patterns (e.g., "always ask before sending email"). Allowed, but δ threshold still applies
- **< 0.60**: Clearly distinct, no restriction beyond normal δ

The threshold is tunable via `similarity_threshold` in validator config.

### 7. Repeated Evaluation (EMA)

**Enforcement**: Validators evaluate each pack **multiple times** (every `eval_interval`) and smooth costs via per-scenario EMA. A single evaluation does not determine the winner.

**Prevents**:
- Gaming via LLM variance luck (a single cheap run doesn't determine the cost)
- Transient low costs from non-deterministic agent behavior
- Inconsistent packs that occasionally complete cheaply but usually don't

**How it works**:
- Each validator accumulates multiple independent cost observations per pack
- Per-scenario cost EMA smooths noise: after 3-4 observations (~12-16 hours), costs converge to within ~2-3% of the pack's true cost
- A pack must be *consistently* cheap to earn a low smoothed cost
- Combined with YC3 cross-validator aggregation, effective variance drops below 1%

---

## Validator Consensus

### The Problem: LLM Non-Determinism

LLM outputs vary between API calls even with the same input and temperature=0. Two independent validators evaluating the same pack may see different agent tool-call sequences, different token counts, and thus different costs and rubric outcomes. Without mitigation, validators disagree on costs and winner selection.

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

Each validator evaluates every active pack every `eval_interval` (default: 24 hours) and maintains per-scenario EMAs keyed by miner hotkey:

```
ema_cost[hotkey][scenario] = α × new_cost + (1 - α) × ema_cost[hotkey][scenario]

Where:
  hotkey   = miner's ss58 address (stable identifier; UIDs recycle)
  α        = 0.3 (cost EMA smoothing factor, configurable)
  scenario = individual ClawBench scenario name
```

**EMA reset**: When a miner submits a new pack (`pack_hash` changes), the cost EMA resets for that hotkey at the next scheduled evaluation. Old observations from a different pack are discarded.

**Convergence**: With α = 0.3 and `eval_interval` = 24 hours:
- After 1 observation: raw noisy cost (variance ~5-10%)
- After 3 observations (~3 days): EMA within ~3% of true cost
- After 5 observations (~5 days): EMA within ~1-2% of true cost

**Rate-limiting**: At most one evaluation per miner per `eval_interval`, regardless of how often the miner updates their commitment. This prevents DDoS via rapid commitment churn (see [Evaluation Cadence](#evaluation-cadence)).

**Weight setting**: At each tempo, the validator computes `total_cost[hotkey]` and `qualified[hotkey]` from its smoothed per-scenario EMA values, maps `hotkey → UID` via the current metagraph, applies winner selection (qualification gate, cost ranking, first-mover, bootstrap), and calls `set_weights` via commit-reveal.

### Cross-Validator: YC3 On-Chain Consensus

Each validator sets weights **independently** based on its own cost and qualification data. **Yuma Consensus 3 (YC3)** with **Liquid Alpha** aggregates these independent weight vectors on-chain.

**How YC3 works for TrajectoryRL**:

- **Per-bond EMA**: Each validator-miner bond pair evolves at its own rate. If Validator A identifies a cheap qualified miner before Validator B, A's bond with that miner grows faster.
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
- **Early recognition**: Liquid Alpha rewards validators who identify cheap qualified miners before others
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
| Qualification gate | ALL safety + correctness checks MUST pass |

Packs failing schema validation or exceeding the size limit receive **weight = 0**. Packs that pass schema but fail any safety or correctness check are **disqualified** from cost competition (weight = 0).

### Pack Rejection Flow

A miner's submission can fail at multiple points in the validation pipeline. The table below specifies the exact outcome for each failure mode:

| Failure | Qualified | Weight | Counts as Active? | ClawBench Runs? |
|---------|:---------:|:------:|:------------------:|:---------------:|
| **No commitment** on-chain (or unparseable) | N/A | 0.0 | No | Skipped |
| **Pack URL inaccessible** (404, timeout, hash mismatch) | N/A | 0.0 | No | Skipped |
| **Schema validation failure** (missing AGENTS.md, >32KB, bad semver) | N/A | 0.0 | No | Skipped |
| **NCD similarity** ≥ threshold (pairwise dedup, later submitter excluded) | N/A | 0.0 | No | May run* |
| **ClawBench timeout** (scenario exceeds `timeout_per_scenario`) | FAIL | 0.0 | Yes | Partial |
| **Safety/correctness check failed** on any scenario | FAIL | 0.0 | Yes | Full |
| **All gate checks pass, not Winner** | PASS | 0.0 | Yes | Full |
| **All gate checks pass, Winner** | PASS | 1.0 | Yes | Full |

**Key rules**:

1. **Fail-fast**: Schema validation and verification are checked *before* running ClawBench. Exact-copy miners (same `pack_hash`) are skipped during evaluation to save LLM costs. Paraphrased copies are caught by pairwise NCD dedup in the weight-setting phase — their evaluation still runs, but they receive weight 0.

2. **"Active" means valid commitment**: A miner counts as "active" only if their on-chain commitment passes all pre-evaluation checks (schema) and at least one ClawBench scenario completes. This definition is used for:
   - Bootstrap threshold (need ≥10 *active* miners for winner-take-all)

3. **Partial failures disqualify**: If a pack passes schema but 1 of 5 scenarios has a failed safety check, the miner is disqualified from cost competition entirely. This is intentional — safety is non-negotiable.

4. **Weight = 0.0 vs. omitted**: Miners who are disqualified or not the Winner still receive `weight = 0.0` in the weight vector (not omitted). This is required by Bittensor's `set_weights`, which requires the vector to cover all UIDs in the metagraph.

---

## Summary

### Evaluation

```
# Per validator (repeated evaluation with per-scenario EMA):
ema_cost[hotkey][scenario]  = α × new_cost + (1 - α) × ema_cost[hotkey][scenario]
qualified[hotkey][scenario] = all safety + correctness checks passed (latest result)
total_cost[hotkey]          = weighted_mean(ema_cost_scenarios)
fully_qualified[hotkey]     = qualified on ALL scenarios

# Weight setting (hotkey → UID via metagraph):
weight[uid] = f(total_cost, fully_qualified)   # Winner = lowest-cost qualified

# Cross-validator (YC3 on-chain):
on_chain_weight = YC3(validator_weights, validator_stakes, bond_history)
```

### Weights

```
# Steady state (≥ bootstrap_threshold active miners):
weight[winner] = 1.0
weight[others] = 0.0

# Bootstrap phase (< bootstrap_threshold active miners):
weight[1st] = 0.70
weight[2nd] = 0.20
weight[3rd] = 0.10

where Winner = lowest-cost qualified miner that satisfies:
  - all safety + correctness checks pass on every scenario
  - cost < previous_best × (1 - δ) (if not first)
  - ties broken by earliest on-chain commitment block number
  - pack accessible at committed HTTP URL, hash matches
  - pack passes OPP v1 schema validation (AGENTS.md required, ≤32KB)
  - pairwise NCD: no other earlier-submitted pack with similarity ≥ σ
  - miner active within last inactivity_blocks
```

### Rewards

```
Steady state:  Winner gets 100% of miner alpha emissions
Bootstrap:     top-3 qualified get 70/20/10 of miner alpha emissions
```

### Key Parameters

| Parameter | Value | Tunable? |
|-----------|-------|----------|
| δ (cost first-mover threshold) | 0.10 (10% cheaper) | Yes |
| α (cost EMA smoothing factor) | 0.3 | Yes |
| Required categories | safety, correctness | Yes |
| eval_interval | 24 hours (~7200 blocks) | Yes |
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

**Version**: v3.1

**Date**: 2026-03-07
