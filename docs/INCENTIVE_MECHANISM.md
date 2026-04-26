# TrajectoryRL Incentive Mechanism

**Subnet**: SN11 (TrajectoryRL)

**Version**: v5.2

**Date**: 2026-04-26

---

## Overview

TrajectoryRL rewards miners who submit **packs** (PolicyBundles) that are evaluated against a **season-defined benchmark** and compete on a **season-defined score**. The score direction (lower-is-better or higher-is-better) and evaluation method are defined per season; the consensus protocol, reward distribution, and anti-gaming machinery below remain unchanged.

The core loop:

1. **Qualification gate**: A miner is disqualified (receives weight 0) if any of the following conditions are met:
   - Pack missing `SKILL.md`
   - `SKILL.md` is empty or whitespace-only
   - Evaluation runtime error (sandbox failure)
   - Pre-eval anti-gaming rejection (platform API detects hardcoded answers, benchmark overfitting, etc.)
2. **Score competition**: Among qualified miners, the one with the best score wins
3. **Consensus**: Validators independently evaluate, share results via off-chain protocol, and compute stake-weighted consensus before setting on-chain weights

For the current season's scoring method, pack schema, and evaluation specifics, see [EVALUATION_S1.md](EVALUATION_S1.md).

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
4. PolicyBundle passes schema validation (season-defined)
5. **Pack ownership lock** (`pack_first_seen`): each validator records the first hotkey it observes for every `pack_hash`; later submitters of the same hash are treated as copies and receive weight 0 (see [Pack Ownership Lock](#3-pack-ownership-lock-pack_first_seen))

**Pack ownership** is determined by **first observation per validator**, not by the on-chain commitment block number. Once a validator records `pack_first_seen[pack_hash] = (hotkey, block)`, that mapping is permanent for the lifetime of the entry — it is not refreshed when the original owner re-commits, and it is not transferred when the original owner goes inactive (see "no succession" below). The pack must remain accessible at the committed URL; if a miner deletes or changes the file so the hash no longer matches, their commitment becomes invalid and they receive weight 0.

**Why On-Chain Commitments + HTTP?**
- **No server required**: Miners upload once to static hosting and go offline. No public IP, no uptime requirement
- **Deterministic discovery**: All validators read the same chain state, eliminating disagreements from network failures or timeouts
- **Unforgeable timestamps**: Block-timestamped by the Substrate chain, not by the miner
- **Simple**: No P2P networking, no retry logic, no timeout handling
- **Flexible hosting**: Any HTTP(S) endpoint works — S3, GCS, GitHub Pages, personal servers, IPFS gateways, etc.

---

## Winner-Take-All with Winner Protection

### Core Rule: Winner Takes All (Steady State)

**Winner** = best-score qualified miner. In steady state (≥`bootstrap_threshold` active miners, default 10), **only the Winner receives rewards**:

```
weight[winner] = 1.0
weight[others] = 0.0
```

Disqualified miners (failed qualification gate) receive weight 0 regardless of score.

### Bootstrap Phase (Early Adoption)

When active miners < `bootstrap_threshold` (default 10), rewards use a **graduated top-3 curve** among qualified miners, ranked by best score:

```
weight[1st] = 0.70  (70%)
weight[2nd] = 0.20  (20%)
weight[3rd] = 0.10  (10%)
weight[others] = 0.0
```

Ties within a rank are broken by earliest on-chain commitment (same rule as steady-state).

Once the 10th active miner submits, the validator automatically switches to winner-take-all.

| Active Miners | Mode | Distribution |
|:------:|------|-------------|
| 1-9 | Bootstrap | Top-3 qualified: 70/20/10 |
| 10+ | Steady state | Winner-take-all: 100/0/0 |

### Always Set Weights

Validators **always call `set_weights` every tempo**, never skip. Validators that don't set weights get deregistered by the chain.

**Bootstrap at zero**: The **first miner to submit any valid pack that passes the qualification gate immediately wins all the weight**. There is no minimum score threshold. Any qualified pack is eligible to win.

If no miner has a valid on-chain commitment (or no miner has score data), the validator sets **all weight to the subnet owner UID**. This ensures the validator always calls `set_weights` (avoiding deregistration). Note: **miner incentive directed to the owner hotkey is burned** by the chain (not paid to the owner), so this fallback effectively burns miner emissions until a qualifying miner submits.

### Submission Staleness Filter

Validators only consider **submissions committed within the last 14400 blocks (~48 hours)**. If a miner's on-chain commitment is older than this threshold, it is skipped during evaluation and the miner receives weight 0.

```python
age = current_block - commitment.block_number
if age > inactivity_blocks:  # default 14400 (~48h at 12s/block)
    skip  # stale submission, not evaluated
```

This prevents indefinite squatting with a stale pack while tolerating normal operational hiccups (maintenance, key rotation). A miner re-enters competition immediately upon submitting a fresh commitment.

| Parameter | Value | Tunable? |
|-----------|-------|----------|
| `inactivity_blocks` | 14400 (~48h) | Yes |

### Winner Protection (Score-Based)

To prevent copy-paste attacks and trivial undercutting, and to stabilize winner selection against evaluation variance, validators enforce **Winner Protection with a multiplicative threshold**:

**Rule**: The current winner defends with their **winning score** (the consensus score at the time they became winner). A challenger can only dethrone the winner if:
```
better(challenger_score, winner_score, δ)
```

Where:
- **δ** = 0.10 (10% improvement threshold)
- **winner_score** = the consensus score recorded when the winner was elected (frozen, not the winner's latest score)
- **better()** is season-defined (for cost-based: `challenger < winner × (1 - δ)`; for quality-based: `challenger > winner × (1 + δ)`)

**Winner self-update**: The winner can update their own record by the same rule — if the winner's new consensus score passes `better()` against their winning score, their record is updated. This makes their defense stronger going forward.

**Winner disqualified**: If the current winner is disqualified (consensus-disqualified via stake-weighted majority, or post-consensus pre-eval rejection), the winner is removed and the best-score eligible miner takes over.

**Cross-spec transition**: Winner Protection only compares scores within a single `spec_number`. During the aggregation round in which the chain-derived `target_spec_number` first differs from `WinnerState.spec_number`, the validator bypasses the δ threshold and elects the highest-scoring eligible miner under the new spec. The returned `WinnerState` is then stamped with `spec_number = target_spec_number`, so subsequent rounds resume normal δ-protected comparisons inside the new spec. This makes the winner handover happen on the same round in which stake-weighted majority flips to the new spec, with no extra burn delay.

**Validator local state**: Each validator persists a `WinnerState` containing:
- `winner_hotkey` — current winner's hotkey
- `winner_pack_hash` — the pack hash when they won
- `winner_score` — the consensus score when they won
- `spec_number` — the `SPEC_NUMBER` under which they won. Records audit context **and** gates Winner Protection: when it differs from `target_spec_number`, the δ threshold is bypassed for that round (see "SPEC_NUMBER and target spec selection")


---

## Reward Distribution

**Steady state** (≥10 active miners): Winner takes 100% of miner alpha.

**Bootstrap** (<10 active miners): top-3 qualified miners split 70/20/10.

For practical mining strategy, see [MINER_OPERATIONS.md](MINER_OPERATIONS.md).

---

## Evaluation Cadence

An **epoch** is TrajectoryRL's validation cycle — a fixed-length period (measured in chain blocks) that encompasses the full evaluate → submit → aggregate → settle lifecycle. Each epoch is subdivided into three **windows** — evaluation, propagation, and aggregation — representing the different stages within the cycle. Unlike Bittensor's chain-level **tempo** (360 blocks, ~72 min) which governs on-chain weight setting, epochs and their windows are defined by the subnet itself.

```
Epoch (7200 blocks, ~24h)
├── Evaluation window    (block 0 → 5760, 80%)    Run season benchmark for target_window
├── Propagation window   (block 5760 → 6480, 10%) Dissemination buffer (not a hard publish gate)
└── Aggregation window   (block 6480 → 7200, 10%) Quorum-gated consensus for target_window
```

Validators run a **continuous evaluation loop** synchronized by chain block height:

| Cadence | Default | Purpose |
|---------|---------|---------|
| `eval_interval` | 7200 blocks (~24h at 12s/block) | Epoch length, block-aligned |
| `tempo` | 360 blocks (~72 min, chain-determined) | Set weights on-chain via commit-reveal |

### Continuous Validator Loop (Dual-Window)

```
while running:
  1. Sync metagraph and compute physical_window from block height
  2. Ensure persisted target_window (logical consensus window)
  3. Build/load active-set snapshot for target_window:
       active_set_window_N = commitments with commitment.block_number < window_start(N)
       + inactivity filter at window_start reference
  4. If target_window == physical_window and target not evaluated:
       evaluate all active miners in snapshot (no hard deadline)
  5. Once evaluation is complete:
       submit full payload immediately (phase-decoupled from T_publish)
  6. At each physical aggregation phase:
       check quorum = submitted_stake(target_window, effective_spec) / total_validator_stake
       if quorum > quorum_threshold (default 0.5):
         aggregate and update winner, then jump target_window to physical_window
       else:
         keep target_window anchored and retry next aggregation phase
  7. Every tempo:
       always call set_weights; burn weights while waiting for quorum
```

### Deterministic Active-Set Snapshot

Each target window persists a frozen miner set as `active_set_window_{N}.json`:

```
active_set_window_N = { commitment | commitment.block_number < window_start(N) }
```

This makes evaluation inputs restart-safe and cross-validator consistent (within accepted polling-granularity overwrite races). Runtime-only filters such as validator permit and blacklist are still applied live, outside snapshot materialization.

### Evaluation Rate-Limiting

Evaluation is rate-limited to **at most one evaluation per miner per epoch**, regardless of how often the miner updates their on-chain commitment.

- If a miner submits a new `pack_hash` within the current epoch, the validator **notes** the new hash but waits for the next epoch
- At the next epoch, the validator evaluates the **latest** `pack_hash` for that miner
- A miner who submits 100 times per hour gets evaluated exactly the same number of times as one who submits once

### Timing Parameters

```
epoch_number  = floor((current_block - global_anchor) / eval_interval)
block_offset  = (current_block - global_anchor) % eval_interval

Evaluation:   block 0    → 5760  (80%)
Propagation:  block 5760 → 6480  (10%)
Aggregation:  block 6480 → 7200  (10%)

set_weights:  every tempo (360 blocks), independent of epoch
```

| Setting | Value | Description |
|---------|-------|-------------|
| `eval_interval` | 7200 blocks (~24h) | Epoch length, block-aligned |

### Benchmark Stability

Every evaluation runs the **full benchmark set** defined by the season. No subset selection or rotation. The benchmark is fixed within a `spec_number`: same scenarios, same criteria, same evaluation method. This ensures scores are directly comparable across validators and across time.

**Anti-stagnation** comes from **growing the benchmark** over time (new scenarios, harder criteria, new domains). When the benchmark changes, it's coordinated via a validator software update and a `SPEC_NUMBER` bump (see "SPEC_NUMBER and target spec selection"). Packs are re-evaluated fresh on the new set.

**State persistence**: Winner state (hotkey, pack_hash, score) persists across validator restarts (serialized to disk as JSON). Cached evaluation results persist across restarts where applicable.

### SPEC_NUMBER and target spec selection

`SPEC_NUMBER` is an integer constant in validator code (`trajectoryrl/utils/config.py`). It identifies a "scoring specification" — the combination of scenario set, scoring methodology, and judge prompt that determines whether two evaluations produce comparable scores. Maintainers bump it whenever a change makes new scores incomparable with old ones (adding/removing scenarios, changing weights, modifying judge prompts). Bench-image patch releases that preserve scoring semantics do **not** bump it. The constant is decoupled from the `trajrl-bench` image version (now used purely for audit / log fields).

A validator always **writes** its commitments using its locally configured `SPEC_NUMBER`. During aggregation, the **target spec_number used to filter incoming commitments** is derived from on-chain stake distribution:

```
1. Read all consensus commitments
2. Apply basic filters (protocol / window / per-validator min stake)
3. Group survivors by spec_number, sum validator stake per group
4. dominant = group with the largest total stake
5. if dominant.stake > 0.5 * total_participating_stake:
       target_spec_number = dominant.spec_number
   else:
       target_spec_number = local SPEC_NUMBER
6. Filter pipeline keeps only commitments whose payload spec_number == target
```

This makes upgrades self-coordinating. While stake-weighted majority remains on the previous spec, every validator (upgraded or not) computes weights against that spec, so the previous winner keeps receiving emissions instead of getting burned. Once stake-weighted majority migrates to the new spec, target flips automatically and the new winner takes over. When no group reaches majority (split network, partial outage), validators fall back to their local `SPEC_NUMBER` and proceed with whatever data they have, again avoiding burn. The 50% threshold reuses the existing `disqualify_stake_threshold` semantic; no new parameter.

`WinnerState.spec_number` records which spec the current winner was selected under. It does **not** trigger a state reset on its own, but it gates Winner Protection: whenever it differs from the round's `target_spec_number`, the δ threshold is bypassed and the new spec's best miner takes over immediately, after which `WinnerState.spec_number` is overwritten to the new target. This guarantees `winner_score` comparisons never cross spec boundaries.

---

## Anti-Gaming Measures

### 1. On-Chain Commitments + Content-Addressed Packs

**Enforcement**: All submissions are content-addressed (SHA256 hash). On-chain commitments give every submission a tamper-proof integrity record; pack ownership for reward attribution is handled separately by the per-validator `pack_first_seen` table (see [Pack Ownership Lock](#3-pack-ownership-lock-pack_first_seen)).

**Prevents**:
- Retroactive pack changes after seeing validator feedback (changing the file breaks the hash → weight 0)
- Pack URL tampering (hash mismatch → invalid commitment)
- Forged pack contents (validators verify `sha256(canonical_json) == pack_hash`)

### 2. Winner Protection (Multiplicative δ)

**Enforcement**: Challenger must improve score by > δ (10%) to dethrone the current winner.

**Prevents**:
- Direct copy-paste attacks (same score fails)
- Trivial undercutting (< 10% improvement fails)
- Lazy free-riding on others' research
- Winner oscillation from evaluation variance (winner defends with frozen winning score)

### 3. Pack Ownership Lock (`pack_first_seen`)

Each validator maintains a per-validator persistent table `pack_first_seen[pack_hash] = (first_hotkey, first_block)`. The first time a validator observes a given `pack_hash`, the submitting hotkey is recorded as the owner. Subsequent commitments with the same `pack_hash` from a different hotkey are treated as **copies**: skipped before evaluation, marked `rejected=True` with `rejection_stage="integrity_check"`, and assigned weight 0.

**Properties**:
- **First observation, not on-chain block**: ownership is anchored on the first time the validator sees the hash, not on the rate-limited `commitment.block_number` (which refreshes whenever a miner re-commits to stay inside `inactivity_blocks`). This eliminates first-mover rotation between identical-pack submitters.
- **No succession**: if the original owner goes inactive, no other miner inherits ownership. Copies always receive weight 0.
- **Eviction (by-active with 7-window grace)**: at the end of each cycle, entries whose `pack_hash` appears in any active commitment have their grace clock refreshed; entries whose `pack_hash` is absent are kept until they have been continuously inactive for `EVICTION_GRACE_WINDOWS = 7` consecutive wall-clock windows (~7 days at 1 window/24h). Any re-activation inside the grace window resets the clock. This bounds the table's size while shielding the original author from losing ownership during a single short outage. Once both the original owner and every copy have been silent for the full grace period, a new miner can re-introduce the pack and become its new owner. "Wall-clock" means the grace span is measured against `window_number` directly — validator downtime still counts against the 7-window budget.
- **Restart-safe**: `pack_first_seen` is persisted to its own dedicated file (`pack_first_seen.json`, default `/var/lib/trajectoryrl/pack_first_seen.json`, configurable via `PACK_FIRST_SEEN_PATH`), separate from the per-hotkey `scenario_scores` / `_eval_pack_hash` / `last_eval_block` / `last_eval_window` caches in `eval_state.json`. Splitting the file means ownership locks survive `spec_number` bumps that invalidate score caches, and the table can be inspected or reset independently. The file carries both the ownership table (`pack_first_seen`) and the per-hash `pack_last_seen_window` tracker that drives grace eviction.
- **Per-validator**: not shared on-chain. Stake-weighted consensus naturally aligns the resulting per-miner scores — a copy scores 0 from every validator that locked the pack to a different hotkey.

**Paraphrase defense**: pre-v5.1 the validator ran a pairwise NCD comparison among all active packs. NCD has known false positives/negatives near the threshold and shared the on-chain `block_number` rotation problem. As of v5.1 paraphrase defense is delegated to **Winner Protection's δ threshold + score-based competition**: a paraphrased copy must beat the current winner by at least δ to take over emissions, the same bar a genuine improvement faces. The NCD library functions in `trajectoryrl/utils/ncd.py` are kept for tooling but no longer gate evaluation.

### 4. Pre-Eval Gate (Platform API)

**Enforcement**: Before evaluating a miner's pack (and again during aggregation), validators call the platform pre-eval API to check whether the submission is allowed. Miners flagged by the platform (banned, hardcoded packs detected server-side) are disqualified before spending evaluation resources.

**Prevents**:
- Known-bad packs from consuming validator evaluation budget
- Banned miners from participating after being flagged
- Pack-switch escapes (aggregation re-checks using epoch_number to resolve the pack that was active during the evaluation window)

**How it works**:
- **Evaluation phase**: Each miner is checked via `pre_eval(hotkey, pack_hash, pack_url)` before episodes run. Rejected miners are marked disqualified and skipped.
- **Aggregation phase**: All miners in the consensus set are re-checked via `pre_eval(hotkey, epoch_number)`. Miners rejected since evaluation are disqualified before winner selection.
- **Fail-open**: If the API is unreachable, validators fall back to cached results or proceed without gating (validators remain self-sufficient).
- **Caching**: Responses are cached by `(hotkey, identifier)` to handle transient API failures.

Controlled by `TRAJECTORYRL_PRE_EVAL_ENABLED` (default: enabled).

### 5. Validator-Side Evaluation

**Enforcement**: Validators run the season benchmark independently in their own harness.

**Prevents**:
- Miners faking scores or qualification
- Environment manipulation
- Replay attacks

### 6. Cross-Validator Consensus (Stake-Weighted Aggregation)

**Enforcement**: Each miner's score is computed as a stake-weighted average across validators that did NOT disqualify that miner. Disqualification requires >50% of reporting stake to agree (stake-weighted majority).

**Prevents**:
- Gaming via evaluation variance luck (consensus across many validators suppresses noise)
- Single-validator manipulation (one validator cannot unilaterally disqualify a miner)
- Transient anomalous scores from non-deterministic agent behavior
- Artificially favorable scores from incomplete evaluations (fail-fast partial results excluded from consensus)

### 7. Season-Specific Measures

Each season defines additional anti-gaming measures specific to its evaluation method. See [EVALUATION_S1.md](EVALUATION_S1.md) for current-season measures (e.g., pack integrity analysis, grounding requirements, judge isolation).

---

## Validator Consensus

### The Problem: Evaluation Non-Determinism

Evaluation outputs vary between runs even with identical inputs. Two independent validators evaluating the same pack may see different results and thus different scores. Without mitigation, validators disagree on scores and winner selection, causing the winner to oscillate between epochs.

### Solution: Two-Phase Evaluation Consensus + YC3

Variance is managed at two layers:

```
Layer 1 (cross-validator):    Two-phase off-chain consensus protocol
                              → validators share raw evaluation results and compute
                                stake-weighted consensus scores before setting weights
                              → disqualification uses stake-weighted majority (>50% stake)

Layer 2 (on-chain):           YC3 with Liquid Alpha
                              → aggregates weight vectors on-chain
                              → handles residual disagreement after off-chain consensus
```

**Disqualification** uses stake-weighted majority: each validator reports a `disqualified` dict (hotkey → reason) for miners that failed evaluation (pre-eval rejected, schema failure, eval error, etc.). A miner is consensus-disqualified only if >50% of reporting stake included that miner in their `disqualified` set. This prevents any single validator from unilaterally disqualifying a miner.

**Score** benefits from cross-validator consensus: each validator's raw score measurement is one noisy estimate. Aggregating estimates from multiple validators using stake-weighted averaging produces a more accurate consensus score. Only scores from validators that did NOT disqualify a miner are included — this prevents artificially favorable results from fail-fast partial evaluations from skewing the consensus.

### Epochs and Windows

All validators operate on synchronized **epochs** derived from chain block height. Each epoch is subdivided into **windows** that structure the evaluate → submit → aggregate → settle workflow. Any validator can independently compute the current epoch number and window — no central coordination needed.

**Block-based epoch computation**:

```
epoch_length  = 7200 blocks (~24h at 12s/block)
global_anchor  = genesis block or a fixed agreed-upon block height
epoch_number  = floor((current_block - global_anchor) / epoch_length)
epoch_start   = global_anchor + epoch_number × epoch_length
```

Every validator reads `current_block` from the chain and arrives at the same `epoch_number`. Wall-clock time is never used for epoch alignment — block height is the single source of truth.

**Windows within an epoch** (block offsets relative to `epoch_start`):

```
Epoch N (7200 blocks, ~20 tempos)
├── Evaluation   [block 0 ── 5760]    (80%)
├── Propagation  [block 5760 ── 6480] (10%)
└── Aggregation  [block 6480 ── 7200] (10%)

set_weights: every tempo (360 blocks), uses latest available consensus
```

**Relationship between epochs, windows, and tempo**: An epoch (7200 blocks, ~24h) and a tempo (360 blocks, ~72 min) are **independent cadences**. The tempo is Bittensor's chain-level cycle for weight setting; the epoch is the subnet's own validation cycle. Validators call `set_weights` via commit-reveal at **every tempo** regardless of which window the epoch is in — this is required by the chain to avoid deregistration. The epoch only determines **when the consensus data gets updated**:

```
latest_consensus persists across epochs and restarts:

  Epoch N-1 T_aggregate → latest_consensus = Epoch N-1 results
  Epoch N   block 0–6480 → still using Epoch N-1 results
  Epoch N   T_aggregate  → latest_consensus = Epoch N results (overwritten)
  Epoch N+1 block 0–6480 → still using Epoch N results
  ...

  set_weights(latest_consensus)          (called every 360 blocks, always)
  If no consensus ever computed:         set_fallback_weights()
```

**Timing rationale**: The 80/10/10 split remains useful as a phase rhythm for evaluation, dissemination, and aggregation checkpoints. `T_publish` is no longer a hard submit deadline.

**Submission rule**: Validators submit only after full target-window evaluation is complete (no partial payload at `T_publish`).

**Quorum gate rule**: Aggregation is phase-aligned but stake-gated:

```
quorum_ratio = submitted_stake(target_window, effective_spec) / total_validator_stake
aggregate iff quorum_ratio > quorum_threshold  # default 0.5
```

If quorum is not met, validators keep `target_window` unchanged, retry next physical aggregation phase, and set burn weights while waiting. Once quorum succeeds, `target_window` jumps to current `physical_window`.

### Payload Externalization + On-Chain Pointer Registration

Evaluation payloads are too large for direct on-chain storage.

**Solution**: Two-layer storage with on-chain pointer registration.

1. **Content-Addressed Storage (CAS)**: Upload the full evaluation payload. IPFS is the primary backend; GCS proxy fallback stores payload and returns a public URL. The content address (IPFS CID or sha256 hash) serves as an integrity proof.
2. **On-chain pointer**: Write a lightweight commitment via `subtensor.set_commitment()` with format: `consensus:{protocol_version}|{epoch_number}|{spec_number}|{content_address}`.

Validator consensus commitments share the same commitment channel as miner pack commitments (`pack_hash|pack_url`). They are distinguished by the `consensus:` prefix. During aggregation, each validator reads `get_all_commitments(netuid)` and filters for entries starting with `consensus:`.

**Backward compatibility**: The on-chain commitment string is positional, so the integer at field 3 is read whether it was written as `scoring_version` or `spec_number`. Older 3-field commitments (`consensus:{pv}|{epoch}|{content_address}`) parse with `spec_number` defaulting to 1. Inside CAS payloads, JSON deserialization accepts either `spec_number` or the legacy `scoring_version` key.

**Verification**: Any validator can independently verify a submission: read on-chain pointer → decode address → download payload from CAS (try IPFS, fall back to GCS) → verify content hash matches.

### Submission Filter Pipeline

Before aggregation, each validator filters incoming submissions:

| Layer | Filter | Discards |
|-------|--------|----------|
| 1 | Protocol version | Mismatched protocol |
| 2 | Epoch number | Wrong epoch |
| 3 | Stake threshold | Below minimum stake |
| 4 | Data integrity | CAS hash mismatch |
| 5 | spec_number target | Payload spec_number mismatches chain-derived target spec |
| 6 | Zero-signal | All-zero scores when others report non-zero |

Valid submissions → stake-weighted aggregation.

### Stake-Weighted Aggregation

Each validator's `ConsensusPayload` contains two key fields:
- `scores`: Dict[miner_hotkey → quality_score] — miners that completed evaluation
- `disqualified`: Dict[miner_hotkey → reason] — miners rejected before or during evaluation (pre-eval rejected, schema failure, eval error, etc.)

**Consensus disqualification** uses stake-weighted majority:

```
disq_stake[miner]      = Σ(stake_i for validators that included miner in disqualified)
reporting_stake[miner] = Σ(stake_i for all validators reporting on this miner)
consensus_disqualified[miner] = disq_stake / reporting_stake > 0.50
```

This prevents any single validator from unilaterally disqualifying a miner. A malicious validator with 5% stake can only shift the disqualification ratio by 5% — controlling >50% of stake is required, consistent with Bittensor's security assumptions.

**Consensus score** uses stake-weighted average across ONLY validators that did NOT disqualify the miner:

```
consensus_score[miner] = Σ(stake_i × score_i) / Σ(stake_i)
                          where i ∈ {validators that did NOT disqualify miner}
```

Scores from validators that disqualified a miner are excluded because fail-fast evaluation may produce incomplete results that are not trustworthy.

**Post-aggregation pre-eval gate**: After stake-weighted aggregation but before winner selection, the pre-eval gate re-checks all miners in the consensus set (using epoch_number). Miners flagged since evaluation are added to the disqualified set:

```
eligible_scores = { hk: score for hk, score in consensus_scores if hk not in disqualified }
```

**Fallback**: When all submissions are filtered out (e.g., storage outage), the consensus from the previous epoch is retained. If no consensus has ever been computed, fallback weights are set (owner UID burn).

### Winner Protection (Post-Consensus)

After computing consensus scores and disqualification, each validator applies **Winner Protection** locally:

```
If no current winner OR winner is disqualified:
  → best-score eligible miner becomes winner
  → record (winner_hotkey, winner_pack_hash, winner_score)

If winner exists and is not disqualified:
  → find best-score eligible miner (including winner)
  → if better(best_score, winner_score, δ):
      → that miner becomes the new winner (or winner self-updates)
      → record new (winner_hotkey, winner_pack_hash, winner_score)
  → else:
      → winner retains, no change
```

**Key properties**:
- Winner always defends with their **winning score** (frozen at time of winning), not their latest score
- Winner's pack can degrade without losing the title — only disqualification removes them
- Self-update follows the same δ rule — winner must beat their own winning score by δ to update
- No automatic season reset — winner persists until beaten or disqualified. Manual reset available for operational control.

**Validator local state** (`WinnerState`): Each validator persists `winner_hotkey`, `winner_pack_hash`, `winner_score`, and `spec_number` to a local JSON file. The `spec_number` records the spec under which the current winner was selected; it does **not** reset state on its own, but it gates Winner Protection — when it differs from the round's chain-derived `target_spec_number`, the δ threshold is bypassed for that round and the field is then overwritten to the new target (see "SPEC_NUMBER and target spec selection"). Since all validators process the same consensus data with the same deterministic algorithm, they converge on the same winner.

**Rate-limiting**: At most one evaluation per miner per `eval_interval`, regardless of how often the miner updates their commitment.

### Degradation Strategies

| Scenario | Behavior |
|----------|----------|
| **CAS upload failure** | Validator skips submission for this epoch; continues using previous epoch's consensus for weight setting. Logged as degraded state. |
| **CAS download failure** (aggregation) | Skip that validator's submission; aggregate from the remaining valid subset. Log failure statistics. |
| **Zero valid submissions** | Previous epoch's consensus is retained for weight setting. If no consensus has ever been computed, fallback weights are set. |
| **Slow evaluator** | Continue evaluating without hard deadline and submit only after full target-window coverage. |
| **Aggregation quorum miss** | Keep `target_window` anchored, retry at each future physical aggregation phase, and burn weights while waiting. |
| **Mid-window restart** | Reload `active_set_window_{N}.json` and continue the same frozen target set. |

### Cross-Validator: YC3 On-Chain Consensus

After off-chain consensus, each validator sets weights based on consensus scores. **Bittensor YC3 with Liquid Alpha** aggregates these weight vectors on-chain. Because validators converge on scores before setting weights, YC3 sees minimal disagreement.

### Validator Incentives

- Validators must set weights every tempo (otherwise deregistered by chain)
- Validators who submit evaluation results to off-chain consensus produce more accurate weights → stronger YC3 bonds → more rewards
- Free-riding validators (no evaluations) are filtered by zero-signal exclusion

---

## Summary

### Evaluation Pipeline

```
# Per validator, per epoch:

# ── Evaluation window (0% → 80%) ──

# Season-defined evaluation pipeline
scores[hotkey] = season_evaluate(pack)        # quality score (0.0–1.0)
disqualified[hotkey] = reason                 # miners that failed (pre-eval, schema, eval error)

# ── Propagation window (80% → 90%) ──

payload = { scores, disqualified, spec_number, metadata }
content_address = cas_upload(payload)
subtensor.set_commitment("consensus:{version}|{epoch}|{spec_number}|{content_address}")

# ── Aggregation window (90% → 100%) ──

submissions = subtensor.get_all_commitments()  # filter for "consensus:" prefix
valid = filter_pipeline(submissions)
consensus_disqualified[hotkey] = disq_stake / reporting_stake > 0.50
consensus_score[hotkey] = Σ(stake_i × score_i) / Σ(stake_i)  # non-disqualified votes only
eligible_scores = consensus_scores - disqualified             # post-consensus pre-eval pass

# ── Winner Protection ──

winner = select_winner_with_protection(eligible_scores, winner_state, delta)

# ── Weight setting (hotkey → UID via metagraph, every tempo) ──

weight[uid] = f(eligible_scores, winner)

# ── Cross-validator (YC3 on-chain) ──

on_chain_weight = YC3(validator_weights, validator_stakes, bond_history)
```

### Weights

```
# Steady state (≥ bootstrap_threshold active miners):
weight[winner] = 1.0
weight[others] = 0.0

# Bootstrap phase (< bootstrap_threshold active miners):
weight[1st] = 0.70, weight[2nd] = 0.20, weight[3rd] = 0.10

where Winner = best-score eligible miner that satisfies:
  - not consensus-disqualified (>50% reporting stake must disqualify to exclude)
  - not disqualified by post-consensus pre-eval gate
  - better(consensus_score, winner_score, δ) to dethrone current winner
  - pack accessible at committed HTTP URL, hash matches
  - hotkey is the recorded owner in `pack_first_seen[pack_hash]`
    (any later submitter of the same hash is treated as a copy, weight 0)
  - submission within last inactivity_blocks
```

### Rewards

```
Steady state:  Winner gets 100% of miner alpha emissions
Bootstrap:     top-3 qualified get 70/20/10
```

### Key Parameters

| Parameter | Value | Tunable? |
|-----------|-------|----------|
| δ (score_delta) | 0.10 (10%) — Winner Protection threshold | Yes |
| disqualify_stake_threshold | 0.50 (>50% stake majority to disqualify) | Yes |
| eval_interval | 7200 blocks (~24h at 12s/block) | Yes |
| T_publish (propagation window start) | 80% of epoch (block 5760) | Yes (phase boundary only; no hard submit deadline) |
| T_aggregate (aggregation window start) | 90% of epoch (block 6480) | Yes |
| quorum_threshold | 0.50 (aggregate only when submitted stake share > threshold) | Yes |
| min_validator_stake | minimum stake for consensus participation | Yes |
| `target_window` catch-up policy | On success, jump directly to current `physical_window` | No (protocol behavior) |
| Bootstrap threshold | 10 active miners | Yes |
| `pack_first_seen` eviction | by-active with grace (drop entries inactive for `EVICTION_GRACE_WINDOWS` consecutive windows) | No |
| `EVICTION_GRACE_WINDOWS` | 7 windows (~7 days; clock resets on any active reference) | No (validator-side constant) |
| active-set snapshot persistence | `active_set_window_{N}.json` | Yes (`ACTIVE_SET_DIR`) |
| inactivity_blocks | 14400 (~48h) | Yes |
| yuma_version | 3 | Subnet owner (on-chain) |
| liquid_alpha_enabled | True | Subnet owner (on-chain) |
| commit_reveal_period | 1 tempo | Subnet owner (on-chain) |

---

## Version History

| Version | Date | Summary |
|---------|------|---------|
| v5.2 | 2026-04-26 | Added deterministic window-start active-set snapshots (`active_set_window_{N}.json`) for restart-safe, cross-validator-stable evaluation sets. Replaced hard `T_publish` cutoff with submit-when-fully-done, introduced stake-quorum-gated aggregation (`quorum_threshold`), dual-window semantics (`physical_window` vs `target_window`), burn-while-waiting on quorum miss, and jump-to-current catch-up after delayed aggregation succeeds. |
| v5.1 | 2026-04-24 | Replaced on-chain `block_number`-based first-mover priority + pairwise NCD pre-eval gate with a per-validator `pack_first_seen` ownership lock (no succession; by-active eviction with `EVICTION_GRACE_WINDOWS = 7` wall-clock-window grace period that resets on any active reference). Lock state persists to its own `pack_first_seen.json` (separate from `eval_state.json`) alongside the `pack_last_seen_window` tracker that drives grace eviction. Paraphrase defense delegated to Winner Protection's δ threshold. NCD library kept for tooling, no longer gates evaluation. |
| v5.0 | 2026-04-21 | Refactored into season-agnostic core. Extracted scoring, pack schema, and evaluation details to EVALUATION_S1.md. Abstracted "cost" to "score" (direction defined per season). |
| v4.2 | 2026-03-29 | Simplified winner selection: removed EMA, unified Winner Protection (δ=10%), stake-weighted majority qualification. |
| v4.1 | 2026-03-15 | Added two-phase off-chain consensus protocol (CAS + pointer registration). |
| v4.0 | 2026-03-01 | Replaced regex-based scoring with LLM-as-judge. |

---

## References

- **Bittensor Docs**: https://docs.bittensor.com
- **Dynamic TAO**: https://docs.bittensor.com/dtao
- **Yuma Consensus 3**: https://docs.learnbittensor.org/learn/yc3-blog
- **YC3 Migration Guide**: https://docs.learnbittensor.org/learn/yuma3-migration-guide
- **Current Season Scoring**: [EVALUATION_S1.md](EVALUATION_S1.md) - pack schema, evaluation method, scoring details
- **Miner Guide**: [MINER_OPERATIONS.md](MINER_OPERATIONS.md) - reference miner, local testing, submission workflow
- **Validator Guide**: [VALIDATOR_OPERATIONS.md](VALIDATOR_OPERATIONS.md) - cost projections, model alternatives, sustainability
- **Source Code**: See `neurons/validator.py` and `trajectoryrl/` package

---

**Version**: v5.2

**Date**: 2026-04-26
