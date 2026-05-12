# TrajectoryRL Incentive Mechanism

**Subnet**: SN11 (TrajectoryRL)

**Version**: v6.0

**Date**: 2026-05-07

---

## Overview

TrajectoryRL rewards miners who submit **packs** (PolicyBundles) that are evaluated against a **season-defined benchmark** and compete on a **season-defined score**. The score direction (lower-is-better or higher-is-better) and evaluation method are defined per season; the consensus protocol, reward distribution, and anti-gaming machinery below remain unchanged.

The core loop:

1. **Submission**: Miners deliver packs via either the on-chain commitment slot or the platform's web submit API (both first-class).
2. **Pre-Eval Gate**: The platform server runs the integrity LLM-as-judge check once per `pack_hash` and admits passing submissions to the **challenger queue**.
3. **Challenge Epoch**: Each epoch the queue head is dispatched to validators as the active challenger. Validators evaluate it independently and post stake-signed scores back to the platform.
4. **Stake-Weighted Aggregation**: The platform finalizes the epoch by computing the consensus score (stake-weighted average) and consensus qualified flag (stake-weighted majority), gated by a stake quorum.
5. **Winner Protection**: A challenger replaces the seated winner only if it qualifies and beats the seated winner's score by ≥ δ.
6. **Weight Setting**: Validators read the canonical winner from the platform and call `set_weights` every tempo. The seated winner takes 100% of miner alpha; when no seat exists (cold start, deregistered or banned winner) weight is set to the subnet owner UID and the chain burns the emission.

For the current season's scoring method, pack schema, and evaluation specifics, see [SCORING_AND_EVALUATION.md](SCORING_AND_EVALUATION.md).

> **v6.0 architectural change**: The validator consensus is now coordinated by the platform server rather than reached via an off-chain CAS protocol. Each challenge epoch evaluates exactly one challenger pack, replacing the previous all-miners-every-window cycle. See [Migration from v5.2](#migration-from-v52) and [Validator Consensus](#validator-consensus).

---

## Submission Protocol

Miners deliver packs via one of two **first-class** channels. Both feed the same `miner_submissions` row, the same pre-eval pipeline, and the same challenger queue.

### Channel A — On-Chain Commitment

Miners upload `pack.json` to any publicly accessible HTTP(S) URL (S3, GCS, IPFS gateway, personal server, …) and submit metadata on-chain via Bittensor's `set_commitment` extrinsic.

```
1. Upload  pack.json  →  any HTTP(S) endpoint that returns 200 on GET
2. Commit  subtensor.set_commitment(netuid=11, data="<pack_hash>|<pack_url>")
3. Server sync worker observes the new commitment, fetches the pack,
   verifies sha256(canonical_json) == pack_hash, and inserts a row
   with eval_status = 'pending_pre_eval'.
```

- Commit content: `pack_hash` and `pack_url`, pipe-delimited, ≤ 256 bytes
- Rate limit: one commitment per ~100 blocks (~20 min) per hotkey (chain-enforced)
- The chain timestamp is unforgeable (block-level)

### Channel B — Web Submit API (Platform-Hosted)

Miners POST a signed payload to the platform; the server uploads the pack to GCS and inserts the row directly.

```
POST /api/v2/miners/submit
{
  miner_hotkey, timestamp, signature,
  pack_hash, pack_content   // canonical pack JSON, ≤ 32 KB
}
```

- Hotkey signature verified server-side
- Pack canonicalized + hashed server-side; mismatch → `400`
- Per-miner cooldown: 60 min (configurable via `MINER_SUBMIT_COOLDOWN_SECONDS`)
- Owner ban check (see [Anti-Gaming](#anti-gaming-measures))
- On success, server uploads the canonical bytes to GCS and inserts a row with `eval_status = 'pending_pre_eval'`

### Why Two Channels?

| Property | Channel A (on-chain) | Channel B (web submit) |
|----------|----------------------|------------------------|
| Hosting requirement | Self-hosted HTTP | None (platform-hosted GCS) |
| Discovery | All validators read chain | Server publishes via queue API |
| Timestamp authority | Chain block | Server clock + signature timestamp |
| Latency to queue entry | ~5 min sync interval | Immediate (next pre-eval tick) |
| Use case | Fully self-sovereign miners | Convenience + lower ops burden |

Both channels converge through the same **pre-eval pipeline** (see [Anti-Gaming → Pre-Eval Gate](#4-pre-eval-gate-server-side)). When pre-eval passes, the row advances to `eval_status = 'pending_eval'` and is eligible for the challenger queue.

### Pack Verification (both channels)

Validators and the server enforce:

1. Pack JSON parseable, schema-valid (season-defined)
2. `sha256(canonical_json) == pack_hash`
3. **Pack ownership lock** (`pack_first_seen`): the platform server records the first hotkey it sees for every `pack_hash` and rejects later submissions of the same `pack_hash` from a different hotkey at queue-admission time (`eval_status = 'failed'`, `rejection_stage = 'integrity_check'`). Copies never reach validators (see [Pack Ownership Lock](#3-pack-ownership-lock-pack_first_seen))

---

## Winner-Take-All with Winner Protection

### Core Rule: Winner Takes All

**Winner** = seated winner in `winner_state`, refreshed only at challenge-epoch finalize. There is one mode:

```
weight[winner] = 1.0
weight[others] = 0.0
```

The v5.x bootstrap top-3 split (70/20/10 below a `bootstrap_threshold`) has been retired in v6.0. The challenge-epoch model evaluates exactly one challenger per epoch, so there is no per-epoch ranking to distribute weight across; "top-3" had no well-defined meaning under the new aggregation. The cold-start case (no seat yet) is handled by the burn rule below, not by a separate bootstrap reward curve.

### Always Set Weights

Validators **always call `set_weights` every tempo**, never skip — failure to set weights leads to deregistration.

If `winner_state` is empty (no one has ever won), or the winner hotkey has been deregistered/banned, validators set **all weight to the subnet owner UID**, which the chain burns. This guarantees `set_weights` always runs and emissions are paused (not lost) until a qualifying winner exists.

### Winner Protection (Score-Based, δ = 3%)

To resist copy-paste attacks, trivial undercutting, and evaluation variance, the platform enforces **multiplicative Winner Protection** on every challenge-epoch finalize:

**Rule**: The seated winner defends with their **winning score** (the consensus score recorded when they took the seat). A challenger dethrones only if:

```
better(challenger_score, winner_score, δ)
```

Where:

- **δ = 0.03** (3% improvement threshold — tightened from 10% in v6.0 because each challenge epoch already aggregates stake-weighted across all participating validators, suppressing evaluation variance more directly than the older two-phase off-chain protocol)
- **winner_score** = the consensus score frozen at seat acquisition (not the winner's latest evaluation)
- **better()** is season-defined:
  - higher-is-better seasons: `challenger > winner × (1 + δ)`
  - lower-is-better seasons: `challenger < winner × (1 - δ)`

**Winner self-update**: If a `challenge_epoch` evaluates a new pack from the seated winner and that new pack's consensus score passes `better()` against the winner_score, `winner_state` advances to the new pack and a higher defense bar. A *worse* new pack from the seated winner is harmless — the seat references the previous winning pack, not the latest one.

**Winner disqualification on the seat**: If a separate periodic reconciliation (deregistration check, owner ban) marks the seated hotkey ineligible, `winner_state` is cleared. The next finalized epoch with a qualifying challenger seats the new winner without δ being applied.

**Canonical state — published, not dictated**: The server stores a single `winner_state` row, but in v6 that row is **a published aggregate, not a directive**. What binds the network's `set_weights` is convergent **local derivation** on each validator: the server publishes the per-vote inputs (with each vote's `validator_stake` frozen at score-submission time) via `GET /api/v2/winner/current`, every honest daemon runs its own copy of the consensus algorithm against those inputs plus its deterministic on-chain metagraph view, and the locally-derived winner is what drives `set_weights`. The server's `winner_state` row is shown alongside as an advisory cross-check; a divergence between local derivation and the server's claim is a server-bug or attack signal, not a network-consensus problem — honest validators stay convergent regardless. Daemons may cache the *raw response* up to `WINNER_FALLBACK_TTL` (default 24 h) on server unreachability and re-derive each tick from the cached inputs; beyond TTL the daemon refuses to set weights and emits an alert.

### Decentralized Winner Derivation in v6

The split between server-side and validator-side responsibilities is deliberate. The server cannot reproduce the votes (only the validator-signed score POST creates them) and the validator cannot reproduce the off-chain pack-fetch / pre-eval pipeline (only the server runs that). So:

- **The server uniquely owns**: receiving signed score POSTs, freezing each vote's `validator_stake` at receipt, exposing the immutable input set via `GET /api/v2/winner/current`, running its own copy of `finalize.ts` to publish `winner_state` for observability.
- **Each validator uniquely owns**: running the off-chain eval, signing score POSTs, fetching its own deterministic on-chain metagraph view at the relevant block, and **running its own copy of the same aggregation algorithm** the server runs in `finalize.ts` to derive the seated winner. `set_weights` is driven by the validator's local result — never the server's claim.

The decentralization guarantee follows from these two non-overlapping responsibilities: tampering with the server's published `winner_state` row does not change `set_weights` (validators ignore that field for authority); tampering with submission contents requires forging a validator signature (infeasible). The only attack surface left is the published input set itself — which is verifiable: validators can compare the `submissions[]` they fetch with their own record of what they submitted, and any structural divergence (vote dropped, vote modified, stake snapshot altered) is observable.

**Inputs the daemon needs locally** for the aggregation:

1. The `submissions[]` array from `/api/v2/winner/current` — votes with frozen `validator_stake`. Authoritative for per-vote weight; the daemon must not substitute the live metagraph value for that validator.
2. The deterministic on-chain metagraph snapshot at the finalized epoch's `start_block` — for the global denominators (total active stake, deregistration, eligibility). Every honest validator reads the same chain and reaches the same snapshot.
3. The consensus config (`MIN_VALIDATOR_STAKE`, `QUORUM_FRACTION`, `WINNER_PROTECTION_DELTA`, …) — pinned in the validator's release; not fetched from the server.

When the daemon's locally-derived winner matches the server's published `winner_state` (the steady state), they agree silently. When they diverge, the daemon proceeds with its local result and emits an alarm; widespread divergence across independent validators is an operational signal that the server-side `finalize.ts` or the server's input feed has drifted — but `set_weights` continues to track the convergent local result.

### Reward Distribution

```
Seated winner:        100% of miner alpha
All other miners:       0%
No seated winner:     burned to subnet owner UID
```

---

## Reward Economics

### Bittensor Dynamic TAO

TrajectoryRL uses **Dynamic TAO (dTAO)** with subnet-specific alpha:

```
Network Emissions (post-halving Dec 2025):
├─ Daily TAO emissions: 3,600 TAO/day
├─ Per-tempo emissions: ~0.3 TAO/tempo (360 blocks ≈ 72 min)
└─ Current TAO price: ~$180 USD (Feb 2026)

Alpha Emissions (subnet-specific):
├─ 1 alpha/block, 360 blocks/tempo, ~20 tempos/day
├─ Total ~7,200 alpha/day per subnet
├─ 41% to miners (winner-take-all; burned to owner UID when no seat)
├─ 41% to validators and stakers
└─ 18% to subnet owner
```

Alpha is swappable to TAO via the subnet liquidity pool.

### Competitive Strategy

Winner-take-all creates extreme risk/reward — there is no second place. For mining strategy details see [MINER_OPERATIONS.md](MINER_OPERATIONS.md).

---

## Evaluation Cadence

The validation cycle in v6.0 is the **challenge epoch** — a configurable-length period (in chain blocks) during which exactly **one** challenger pack is evaluated by all participating validators. This replaces the v5.x epoch-with-three-windows structure.

### Challenge Epoch

```
Challenge Epoch  (length = EPOCH_LENGTH_BLOCKS, default 150 ≈ 30 min)
├── start_block (B)       Server picks queue head, opens epoch,
│                         exposes via GET /api/v2/epoch/current
├── B → B+L               Validators fetch pack, run season eval,
│                         POST /api/v2/epoch/{challenge_epoch_id}/score
└── Two finalise paths:
    ├── all whitelisted validators submitted AND
    │   elapsed >= EPOCH_MIN_BLOCKS    Early finalise (dynamic epoch)
    └── end_block (B+L)                Hard deadline (always)
```

`EPOCH_LENGTH_BLOCKS` is bench-driven — set it long enough that a typical validator finishes the full evaluation under realistic LLM latency, with margin. Acceptable range: ~30 min to several days.

### Dynamic Epoch (Early Finalise)

An epoch can finalise before its `end_block` deadline once every operator-curated whitelist validator has submitted, provided at least `EPOCH_MIN_BLOCKS` (default 30) blocks have elapsed since `start_block`. The hard deadline still applies as the upper bound.

The whitelist source is a boolean column `nodes.whitelisted` on the platform side, manually managed by the operator. The effective gate set is `whitelisted = true AND is_validator = true AND deregistered = false`, so a deregistered hotkey drops out automatically. An empty effective whitelist disables early finalise — the epoch runs to its hard deadline.

The trigger state is server-internal — `GET /api/v2/epoch/current` does not expose whitelist size or submission counts. Daemons see early finalise only as "the next `/api/v2/epoch/current` poll returns a new epoch id" or as a `409 epoch is finalized` response on `POST /score`. The required daemon adjustment is to cap the poll sleep at a daemon-side maximum (e.g. 10 s) so the new epoch is observed within that bound; do not derive long sleeps from `remaining_blocks * 12s`. See [API.md → Dynamic Epoch (early finalise)](API.md#dynamic-epoch-early-finalise).

### Empty Queue

If no eligible challenger exists at scheduler tick, the server sleeps `EMPTY_QUEUE_RECHECK_BLOCKS` (default 60) and re-checks. During empty periods `winner_state` is unchanged and validators continue setting weights for the seated winner.

### Catch-Up

A validator that was offline during an epoch simply does not submit; the epoch finalizes with whatever stake reported (provided quorum). Missing a submission counts toward the `INACTIVE_THRESHOLD_EPOCHS` inactivity flag (see [Validator Inactive Tracking](#validator-inactive-tracking)). There is no per-validator catch-up — the epoch result is final once finalized server-side.

### Bittensor Tempo Alignment

Bittensor's tempo (~360 blocks, ~72 min) governs `set_weights` cadence and is independent of `EPOCH_LENGTH_BLOCKS`. A challenge epoch may span one or many tempos. Within a tempo with no finalize, validators set weights for the same seated winner; a finalize that lands mid-tempo affects the next tempo's `set_weights`.

### SPEC_NUMBER and Target Spec Selection

When `SPEC_NUMBER` (formerly `scoring_version`) bumps, the active scoring contract changes. Behavior:

- Existing `pending_eval` submissions remain pickable but are evaluated against the new spec
- `winner_state` is **not** automatically reset on `SPEC_NUMBER` change (operator decision; same as v5.x)
- `score_submit_log` rows carry their `spec_number` so historical comparisons are well-defined

---

## Anti-Gaming Measures

### 1. Content-Addressed Packs + Submission Verification

**Enforcement**: All packs are content-addressed (SHA256). Channel A's chain commitment provides a tamper-proof timestamp; Channel B's signed POST provides hotkey-signed authenticity. Validators verify `sha256(canonical_json) == pack_hash` independently.

**Prevents**:

- Retroactive pack changes (different bytes → different hash → invalid)
- URL tampering (Channel A: hash mismatch invalidates commitment)
- Forged submissions (Channel B: invalid signature is rejected at the API)

### 2. Winner Protection (Multiplicative δ)

**Enforcement**: Challenger must improve score by > δ (3%) to dethrone. See [Winner Protection](#winner-protection-score-based-δ--3) above.

**Prevents**:

- Exact copy-paste (same score fails)
- Trivial undercutting (< 3% improvement fails)
- Free-riding on others' research
- Winner oscillation from evaluation variance

### 3. Pack Ownership Lock (`pack_first_seen`)

**Enforcement**: The platform server persistently maintains `pack_first_seen[pack_hash] = (first_hotkey, first_seen_at)`. When a submission arrives via either channel and the canonical pack bytes are recorded, the server claims ownership for that hotkey if no entry exists. A later submission with the same `pack_hash` but a *different* hotkey is treated as a **copy** and rejected before queue admission: `eval_status = 'failed'` with `rejection_stage = 'integrity_check'`. Copies never enter the challenger queue and are never dispatched to validators, so the validator daemon carries no ownership state of its own.

**No succession**: ownership is permanent until eviction. If the original owner goes inactive, the entry is **not** transferred — any other submitter of that pack who is not the original owner is rejected. Trade-off: prevents "outliving the original" attacks, accepts that an abandoned popular pack can become orphaned.

**Eviction (by-active with grace)**: The server periodically runs an eviction sweep. Entries whose `pack_hash` is referenced by any active row (a `miner_submissions` row in `pending_pre_eval` / `pending_eval`, or the seated winner's pack) refresh their grace clock. Entries absent from all active references for `EVICTION_GRACE_WINDOWS = 7` consecutive wall-clock windows are evicted, after which a new miner may re-introduce and own the pack ("resurrection path").

**Persistence**: `pack_first_seen` is part of the platform database alongside `miner_submissions`; it survives `spec_number` bumps. There is no per-validator JSON file or `PACK_FIRST_SEEN_PATH` environment variable in v6.0 — ownership is single-source-of-truth on the server.

### 4. Pre-Eval Gate (Server-Side)

**Enforcement**: Before any pack enters the challenger queue, the platform server runs a **single** Phase-1 LLM-as-judge integrity analysis (the same gate v4.x ran inside each validator). The judge looks for `hardcoded_response`, `instruction_override`, `tool_avoidance`, `keyword_stuffing`, `scenario_gaming`, `prompt_injection`. Any critical flag → `eval_status = 'failed'`, never enters the queue.

Centralizing Phase-1 server-side eliminates N redundant LLM calls per pack (one per validator) while keeping the gate strict — the judge is hardened with its own system prompt that pack content cannot override.

**Caching**: Results are keyed by `pack_hash`. A pack is analyzed at most once.

**Owner ban**: If a hotkey's owner has been banned (see [Owner Ban](#owner-ban-server-side)), every submission from that hotkey is rejected at the gate.

**Prevents**:

- Known-bad packs from consuming validator evaluation budget
- Banned miners from re-entering after being flagged
- Wasted N×LLM cost on the same pack across all validators

### 5. Owner Ban (Server-Side)

The platform tracks per-ownerkey ban state (`miner_bans` table). Each ownerkey accumulates a `failed_pack_count`. When `failed_pack_count > 3`, the ownerkey is banned for 30 days. A second over-threshold event after the 30-day ban (`served_one_timed_ban = true`) extends the ban to ~99 years. Bans propagate to all hotkeys controlled by the same ownerkey.

**Prevents**:

- Sybil "burn-the-hotkey" attacks where one operator throws away hotkeys to flood the queue with bad packs

### 6. Validator-Side Evaluation

**Enforcement**: Validators run the season benchmark independently in their own harness against the challenger pack. The platform never substitutes a single validator's result for the consensus.

**Prevents**:

- Faked scores or qualification claims
- Environment manipulation
- Replay attacks

### 7. Server-Coordinated Stake-Weighted Aggregation

**Enforcement**: For each finalized challenge epoch, the platform aggregates per-validator submissions:

```
consensus_qualified = (Σ stake of validators reporting qualified=true)
                       > 0.50 × (Σ stake of all reporting validators)

consensus_score     = Σ(stake_i × score_i) / Σ(stake_i),
                      summed over qualified-reporting validators only
                      (mirrors v5.x: only non-disqualifying votes contribute
                       to the score, preventing fail-fast partials from
                       skewing the score)
```

Aggregation runs server-side, but every input (per-validator `score_submit_log` row, hotkey signature, stake snapshot at start_block) is publicly readable and the computation is pure-function deterministic — any auditor can replay it.

**Prevents**:

- Single-validator manipulation (one validator cannot unilaterally disqualify or shift the score)
- Evaluation-variance noise (averaging across many validators)
- Partial-evaluation gaming (fail-fast scores excluded from the score average)

### 8. Quorum Gate

**Enforcement**: An epoch finalizes only if reporting stake ≥ `QUORUM_FRACTION` (default 0.5) of the start_block-snapshot active stake. Below quorum, the epoch is `aborted_quorum` and the challenger submission is moved to `eval_status = 'exhausted'` immediately — there is no retry. The miner can submit a different `pack_hash` to re-enter the queue (see [Challenge Epoch Lifecycle](#challenge-epoch-lifecycle)).

**Prevents**:

- A small subset of validators from finalizing decisions when most of the network is offline
- "Fast challenger" gaming: submitting at a moment when known-friendly validators are online and most others are not

### 9. Season-Specific Measures

Each season defines additional anti-gaming measures. See [SCORING_AND_EVALUATION.md](SCORING_AND_EVALUATION.md).

---

## Validator Consensus

### The Problem (Unchanged)

Evaluation outputs vary between runs even with identical inputs. Without mitigation, validators disagree on scores and winner selection.

### Solution: Server-Coordinated Challenge Epochs

v6.0 replaces the v5.x off-chain CAS + on-chain pointer protocol with a server-coordinated model:

```
Layer 1 (per epoch, off-chain):
  Server publishes one challenger
  → Validators evaluate independently and POST signed scores
  → Server aggregates stake-weighted, applies Winner Protection,
    publishes canonical winner_state.

Layer 2 (on-chain, every tempo):
  Validators read winner_state via API and call set_weights
  → YC3 with Liquid Alpha aggregates weight vectors on-chain
  → Handles residual disagreement and validator weight-copying.
```

The on-chain layer (YC3 + commit-reveal + bond dynamics) is unchanged from v5.x.

### Challenge Epoch Lifecycle

```
[scheduler tick]
   pick queue head:
     SELECT FROM miner_submissions
       WHERE eval_status = 'pending_eval'
         AND no OTHER row of the same miner_hotkey is also in pending_eval
       ORDER BY submitted_at ASC
       LIMIT 1

   if empty: sleep EMPTY_QUEUE_RECHECK_BLOCKS, retry

   else: INSERT INTO challenge_epochs (
           challenger_hotkey, challenger_pack_hash,
           start_block = B, end_block = B+L,
           status = 'in_progress'
         )
         expose via GET /api/v2/epoch/current

[B → B+L]  Validator independent evaluation
   GET /api/v2/epoch/current   (returns epoch + seated winner)
   Fetch pack from CAS / GCS
   Run season-defined eval (pre-eval already done server-side)
   POST /api/v2/epoch/{challenge_epoch_id}/score {
     validator_hotkey, timestamp, signature, version, spec_number,
     challenger: { score, qualified, rejected?, rejection_detail?,
                   scenario_results? },
     // optional `winner` block when dual-eval lands
     llm_base_url, llm_model, bench_image_hash, harness_image_hash,
     bench_version
   }
   // Signed prefix: trajectoryrl-challenge-score:
   //   {validator_hotkey}:{timestamp}:{challenge_epoch_id}
   // challenger_hotkey/pack_hash are server-stamped from
   // challenge_epochs(id), never read from the request body.

[block B+L]  Hard deadline → finalize
   Snapshot stake from metagraph at start_block (B), not B+L —
   locks the eligible-validator set so mid-epoch register/deregister
   churn cannot retroactively change the quorum denominator.

   Apply submission filter pipeline:
     - drop rows with rejected = true
     - drop submissions from validators below MIN_VALIDATOR_STAKE
     - drop submissions from validators marked inactive
     - drop submissions from validators absent in B-snapshot

   reporting_stake / total_active_stake < QUORUM_FRACTION:
     → status = aborted_quorum
            eval_status = 'blacklisted'
       else:
            row stays pending_eval, will be repicked next epoch

   else: aggregate + Winner Protection
     consensus_qualified := stake-weighted majority (>50% reporting stake true)
     consensus_score     := stake-weighted average over qualified votes
     if seated winner empty AND consensus_qualified:
        seat (challenger_hotkey, challenger_pack_hash, consensus_score)
     elif consensus_qualified AND
          better(consensus_score, winner_score, δ=0.03):
        seat (challenger_hotkey, challenger_pack_hash, consensus_score)
        record winner_history row with changed_from_prev = true
     else:
        winner_state unchanged, outcome = 'winner_held'

     status = finalized
     miner_submissions.eval_status: pending_eval → completed
```

### Submission Filter Pipeline

Server-side, per finalize:

```
raw challenge_scores rows for challenge_epoch_id
  (one row per validator, indexed UNIQUE on (challenge_epoch_id, validator_hotkey),
   challenger_* columns always present, winner_* columns NULL in single-eval)
  │
  ├─ [1] reject if challenger.rejected = true
  ├─ [2] reject if validator stake < MIN_VALIDATOR_STAKE at B-snapshot
  ├─ [3] reject if validator marked inactive at B
  ├─ [4] reject if validator missing from B-snapshot active set
  ├─ [5] reject if signature invalid (re-checked at finalize)
  └─ [6] reject if version mismatch (validator version < min_supported)
              → counted toward inactivity, prompts upgrade

valid rows → aggregation (over challenger_* columns; dual-eval will add
             a parallel pass over winner_* columns in a future release)
```

Each rejection is logged to a per-epoch diagnostic record so operators can investigate participation drops.

### Validator Inactive Tracking

After every terminal epoch (both `finalized` and `aborted_quorum`), the server records one `validator_activity` row per active validator: `participated = true` if a non-rejected submission from that validator exists for the epoch, else `false`.

A validator is **marked inactive** if `participated = false` for the most recent `INACTIVE_THRESHOLD_EPOCHS` (default 3) consecutive epochs. Inactive validators have stake **excluded** from quorum and aggregation until they participate again (one participated row clears the flag). This is non-punitive: it prevents disconnected validators from dragging quorum below threshold.

### Winner Protection (Post-Aggregation)

See [Winner Protection](#winner-protection-score-based-δ--3) above for the full rule. Three concrete states result from each finalized epoch:

| Outcome | Meaning |
|---------|---------|
| `winner_replaced` | Seat advanced (challenger won, or self-update raised defense) |
| `winner_held` | Challenger failed `better()` (or failed qualification); seat unchanged |
| `aborted_quorum` | Quorum gate failed; epoch had no aggregation; seat unchanged; challenger retried |

`winner_state` updates are persisted with the epoch id; `winner_history` records all `winner_replaced` transitions for audit.

### Degradation Strategies

| Failure | Behavior |
|---------|----------|
| Server transient outage | Validators wait. Daemon falls back to cached `winner_state` for `set_weights` until `WINNER_FALLBACK_TTL`. |
| Server outage > TTL | Daemons refuse to set weights; alert operators. No corrupt state written. |
| Validator timeout | Submission absent → counted as non-participation. After `INACTIVE_THRESHOLD_EPOCHS` consecutive misses, marked inactive. |
| Pre-Eval LLM unavailable | New rows stay `pending_pre_eval`; server retries on next tick. Queue does not move on these rows; previously-eligible rows continue normally. |
| Chain RPC unavailable | Channel A sync stops; Channel B unaffected. Previously-queued rows continue. |
| CAS / GCS read failure on validator | Validator skips the epoch. Counted as non-participation. |
| Quorum miss | Epoch `aborted_quorum`; challenger submission moves to `eval_status='exhausted'` immediately — no retry. Miner can submit a different `pack_hash`. |
| DB outage on server | Write APIs return 5xx; validators retry. Read APIs serve last cached state if available. |
| Single validator submitting malicious score | Diluted by stake-weighted average. Cannot dethrone unless > 50% stake collusion (already a network-level trust assumption). |

### Cross-Validator: YC3 On-Chain Consensus

Unchanged from v5.x. Validators set weights every tempo via commit-reveal; YC3 with Liquid Alpha aggregates on-chain.

| Parameter | Value | Notes |
|-----------|-------|-------|
| `yuma_version` | 3 | `btcli sudo set --param yuma_version --value 3 --netuid 11` |
| `liquid_alpha_enabled` | True | `btcli sudo set --param liquid_alpha_enabled --value true --netuid 11` |
| `commit_reveal_period` | 1 tempo | Already set |
| `bonds_moving_avg` | 900000 (90%) | Tunable |

### Validator Incentives

Validators earn rewards for:

- **Bond strength** — proportional to agreement with consensus winner (YC3 bond dynamics)
- **Early recognition** — Liquid Alpha rewards validators who recognize winners ahead of others
- **Active participation** — submitting scores within the epoch window keeps the inactivity flag clear and stake counted in quorum
- **Setting weights regularly** — chain deregisters validators that don't

**Attack resistance**:

- Colluding validators cannot fake packs (content-addressed + verifiable bytes)
- Dishonest validators submitting inflated/deflated scores are diluted by stake-weighted aggregation from honest validators
- Free-riding validators (no submissions, just reading `winner_state`) are filtered by participation tracking and YC3's zero-signal exclusion
- Weight-copying detectable by YC3; copier lags behind on bond dynamics

---

## Validator Daemon Loop (v6.0)

The v6.0 validator daemon is intentionally thin. It carries no per-miner evaluation cache, no client-side integrity gate, and no ownership state. Its responsibilities reduce to **three** concurrent loops — eval, winner-derivation, and weights — split because the underlying server-published reads have different semantics:

- `GET /api/v2/epoch/current` answers "what's the active challenge **right now**?" — used by the eval loop to find the pack to score.
- `GET /api/v2/winner/current` answers "what evidence has been finalized into the **current seat**?" — used by the winner-derivation loop, which **re-runs the consensus aggregation locally** from the returned `submissions[]` and uses the local result (not the server's `winner_state` claim) to drive `set_weights`.

```
# Eval loop — drives the per-epoch challenger evaluation
loop every ~10 s:
    resp = GET /api/v2/epoch/current
    if resp.epoch and not already_scored(resp.epoch.challenge_epoch_id):
        pack = fetch_and_verify(resp.epoch.challenger_pack_hash)
        result = run_season_eval(pack)
        POST /api/v2/epoch/{challenge_epoch_id}/score   # signed
        mark already_scored(resp.epoch.challenge_epoch_id)

# Winner-derivation loop — produces the locally-derived seated winner used by set_weights
loop every ~10 s (can share a tick with the eval loop):
    w = GET /api/v2/winner/current
    if w.finalized_epoch is null:
        cache.winner = null                            # cold start, no finalized epoch yet
    else:
        derived = local_aggregate(
                    w.submissions,                     # votes + frozen validator_stake
                    on_chain_metagraph_at(w.finalized_epoch.start_block),
                    CONSENSUS_CONFIG)                  # pinned in this release
        if derived.uid != w.winner?.uid:
            log_alarm("server winner ≠ locally-derived"); proceed with derived
        cache.winner = derived

# Weight loop — drives on-chain weight-setting
loop every ~5 min:
    if it_is_time_to_set_weights():                    # tempo-gated
        if cache.winner is null:
            skip                                       # cold start
        elif cache_age > WINNER_FALLBACK_TTL:
            refuse + alarm                             # server been unreachable too long
        else:
            set_weights(uid = cache.winner.uid)        # winner-take-all
```

The eval loop produces at most one signed score per `challenge_epoch_id`. The winner-derivation loop re-derives on every successful poll — it does **not** cache the derived winner across polls (otherwise a later finalize-time fix on the server would not propagate); what it caches for fault-tolerance is the *raw* `/api/v2/winner/current` response. The weight loop is independent and tempo-gated. A separate heartbeat task (~10 min) reports liveness, version, and bench/harness image digests via `POST /api/v2/validators/heartbeat`.

The server's `winner_state` row, surfaced as `w.winner` on `/api/v2/winner/current` (and as the non-authoritative `winner` block on `/api/v2/epoch/current`), is **advisory** — used only for cross-checking. `set_weights` follows the locally-derived result. See [Decentralized Winner Derivation in v6](#decentralized-winner-derivation-in-v6) for the security argument.

## What Validators No Longer Do (v6.0)

Compared with v5.x, the following responsibilities have been removed from the validator daemon and either centralized server-side or retired:

- **Client-side LLM-as-judge / Phase-1 integrity analysis** — moved server-side as the [Pre-Eval Gate](#4-pre-eval-gate-server-side); no validator-resident `PackIntegrityJudge` or per-pack judge cache.
- **Pre-eval / NCD / similarity self-checks** — none. Validators trust that anything dispatched as a challenger has cleared server gates.
- **`pack_first_seen` ownership maintenance** — moved server-side (see [Pack Ownership Lock](#3-pack-ownership-lock-pack_first_seen)); no per-validator JSON file, no `PACK_FIRST_SEEN_PATH`.
- **Cross-epoch per-hotkey evaluation cache** (the v5.x `scenario_scores` accumulator and friends) — gone. Each challenge epoch produces a single signed score, posted and forgotten.
- **CAS consensus payload upload + on-chain commitment-pointer registration** — replaced by direct `POST /api/v2/epoch/{id}/score` to the platform.
- **`epoch_snapshot` polling for an active-set evaluation list** — v6.0 evaluates exactly one challenger per epoch, retrieved via `GET /api/v2/epoch/current`.
- **Calls to legacy `POST /api/v2/scores/submit`** — replaced by `POST /api/v2/epoch/{challenge_epoch_id}/score` (distinct signing prefix).

What validators newly **gain** in v6.0:

- **Local Winner Protection (δ) computation** — explicitly **not** removed. v6.0 validators run their own copy of the consensus aggregation against `/api/v2/winner/current.submissions[]` and use the locally-derived winner to drive `set_weights`. The server's `winner_state` row is advisory only. See [Decentralized Winner Derivation in v6](#decentralized-winner-derivation-in-v6).

---

## API Surface

All read endpoints are public; write endpoints require hotkey signature. Validator critical-path endpoints are marked **(critical)** — the v6 daemon must call these. Other endpoints are observability-only.

| Method | Path | Purpose |
|--------|------|---------|
| `GET` | `/api/v2/epoch/current` | **(critical)** In-progress challenge: `{ epoch: { challenge_epoch_id, challenger_hotkey, challenger_pack_hash, start_block, end_block, status } }`. Also returns a non-authoritative `winner` block (mirror of `winner_state`) for observability — daemons must **not** drive `set_weights` from this field; use `/api/v2/winner/current` instead. 404 when no epoch is in progress. Polling cadence ~10 s. |
| `GET` | `/api/v2/winner/current` | **(critical)** Latest finalized epoch's per-validator votes (`challenger_score`, `challenger_qualified`, `winner_*` for dual-eval) with each vote's `validator_stake` snapshotted at score POST. Also returns the server's `winner_state` claim as `winner` (advisory only). v6 daemons consume `submissions[]`, run a local copy of the consensus aggregation, and use the locally-derived winner to drive `set_weights`. This is the endpoint that makes v6 winner derivation decentralized — see [Decentralized Winner Derivation in v6](#decentralized-winner-derivation-in-v6). |
| `POST` | `/api/v2/epoch/{challenge_epoch_id}/score` | **(critical)** Validator v6 score submission. Path carries the epoch id; body carries a `challenger` block (required) and a `winner` block (optional, dual-eval). Signed prefix: `trajectoryrl-challenge-score:{validator_hotkey}:{timestamp}:{challenge_epoch_id}` — replay-safe across epochs. Server stamps `challenger_hotkey`/`challenger_pack_hash` from `challenge_epochs(id)` and `validator_stake` from `nodes` at receipt; writes to `challenge_scores`. Distinct path / prefix from the legacy v5.2 `/api/v2/scores/submit`. |
| `POST` | `/api/v2/validators/heartbeat` | **(critical)** Validator liveness + running version + bench/harness image digests. |
| `GET` | `/api/queue` | FIFO challenger queue snapshot (validators do not call this). |
| `GET` | `/api/epoch/{id}` | Epoch metadata + per-validator submissions + outcome (validators do not call this). |
| `GET` | `/api/winner/history?limit=N` | Recent `winner_replaced`/reaffirmed transitions (validators do not call this). |
| `GET` | `/api/validator/{hotkey}/activity?limit=N` | Recent per-epoch participation records (validators do not call this). |
| `POST` | `/api/v2/miners/submit` | Channel B miner pack submission (signed). |
| `POST` | `/api/validators/logs/upload` | Per-miner eval log archive (fire-and-forget, debugging only). |
| `POST` | `/api/validators/logs/cycle` | Cycle-level eval log archive (fire-and-forget, debugging only). |

> **Note on the legacy v5.2 score-submit path.** `POST /api/v2/scores/submit` (signed prefix `trajectoryrl-submit`) is **not** used by v6 daemons. It is retained only for the v5.2 daemon during the cutover window; new code must target `POST /api/v2/epoch/{challenge_epoch_id}/score`. The two paths are independent in code, schema, and signature prefix.

---

## Summary

### Submission → Winner

```
miner submits (Channel A: chain commitment, or Channel B: web POST)
   │
   ▼
miner_submissions row inserted with eval_status = 'pending_pre_eval'
   │
   ▼ pre-eval pipeline (server, 5 min cadence)
   │  Phase-1 LLM judge + 7 *_check steps
   │
   ├─ failed                    → eval_status = 'failed'    (terminal)
   └─ all checks pass           → eval_status = 'pending_eval'  (queue)

[scheduler picks one challenger from the queue every EPOCH_LENGTH_BLOCKS]
   │
   ▼ challenge epoch opens
   │  validators evaluate independently, POST stake-signed score
   │
   ▼ epoch end (hard block deadline)
   │
   ├─ quorum < QUORUM_FRACTION  → aborted_quorum, eval_status='exhausted' (no retry)
   │
   └─ quorum ≥ QUORUM_FRACTION  → aggregate (stake-weighted)
                                  → apply Winner Protection (δ = 0.03)
                                  → seat winner / hold
                                  → eval_status = 'completed'

every ~10 s (winner-derivation loop, can share a tick with the eval loop):
   validator GET /api/v2/winner/current  (returns latest finalized_epoch + per-validator submissions)
   → local_aggregate(submissions, on_chain_metagraph_at(finalized_epoch.start_block), CONSENSUS_CONFIG)
   → cache.winner = derived          (server's response.winner is advisory only)
   → if derived.uid != response.winner?.uid: emit divergence alarm

every Bittensor tempo (set_weights cadence):
   validator uses cache.winner from the loop above
   → if cache age > WINNER_FALLBACK_TTL: refuse + alarm
   → set_weights(winner_uid = 1.0, or burn to subnet owner UID when no seat)
   → YC3 aggregates on-chain
```

### Weights

```
weight[seated_winner] = 1.0
weight[others]        = 0.0

No seated winner → all weight to subnet owner UID (burned).
```

Where seated_winner satisfies:

- Server-canonical winner_state is non-empty and not deregistered
- consensus_qualified = stake-weighted majority of qualified votes
- consensus_score passes `better(score, winner_score, δ = 0.03)` against the prior seat
- pack passes schema validation and content-address check
- hotkey is the recorded owner in the server's `pack_first_seen[pack_hash]` (copies are filtered before queue admission and never reach validators)
- miner active within `inactivity_blocks`

### Rewards

```
Seated winner:     100% of miner alpha
No seated winner:  burned to subnet owner UID
```

### Key Parameters

| Parameter | Default | Tunable? |
|-----------|---------|----------|
| `EPOCH_LENGTH_BLOCKS` | 150 (≈ 30 min) | Yes (bench-driven; 30 min – several days) |
| `QUORUM_FRACTION` | 0.50 | Yes |
| `WINNER_PROTECTION_MARGIN` (δ) | 0.03 (3%) | Yes |
| `INACTIVE_THRESHOLD_EPOCHS` | 3 | Yes |
| `MIN_VALIDATOR_STAKE` | 10000.0 | Yes |
| `EMPTY_QUEUE_RECHECK_BLOCKS` | 60 | Yes |
| `WINNER_FALLBACK_TTL` | 24 h | Yes (validator daemon side) |
| `MINER_SUBMIT_COOLDOWN_SECONDS` (Channel B) | 3600 (60 min) | Yes |
| `inactivity_blocks` | 14400 (~48 h) | Yes |
| `EVICTION_GRACE_WINDOWS` (`pack_first_seen`) | 7 windows (~7 d) | No (server constant) |
| `yuma_version` | 3 | Subnet owner (chain) |
| `liquid_alpha_enabled` | True | Subnet owner (chain) |
| `commit_reveal_period` | 1 tempo | Subnet owner (chain) |

---

## Migration from v5.2

v6.0 ships in three phases gated by an `INCENTIVE_MECHANISM` major-version bump.

**Phase 1 — Shadow run (v6.0-rc):**

- Schema additions deployed (additive only — no destructive change to existing tables).
- Server runs the full new pipeline end-to-end and populates `challenge_epochs`, `winner_state`, `winner_history`.
- Validator daemons upgrade to a build that **continues to drive `set_weights` from v5.2 consensus** but additionally posts the active-challenger score to the v6 path `POST /api/v2/epoch/{challenge_epoch_id}/score` (signed prefix `trajectoryrl-challenge-score`). The v5.2 `POST /api/v2/scores/submit` write path remains in use for legacy scoring during this phase.
- Operators compare v5.2 winner output to `winner_state` over a window of weeks. Discrepancies are investigated; pipeline iterated.

**Phase 2 — Cutover (v6.0):**

- Validator daemons flip to driving `set_weights` from the locally-derived winner (`GET /api/v2/winner/current` → `local_aggregate(submissions, on_chain_metagraph_at(finalized_epoch.start_block), CONSENSUS_CONFIG)`) and to writing scores via `POST /api/v2/epoch/{challenge_epoch_id}/score`.
- v5.2 off-chain CAS + chain commitment-pointer path is disabled (code retained, gated by an env flag).
- `consensus_payload_log` writes stop, and `POST /api/v2/scores/submit` traffic from v6 daemons is expected to drop to zero. Existing rows retained as historical record.

**Phase 3 — Cleanup (v6.x):**

- Remove disabled v5.2 consensus code path.
- Optionally archive and drop `consensus_payload_log` (separate spec).

No data migration is required for `miner_submissions` rows that landed under v5.x — existing rows remain in their existing `eval_status`. The new `'exhausted'` value applies to v6.0 onward (terminal on first quorum-abort under the no-retry policy).

The full design rationale lives in [`docs/superpowers/specs/2026-05-07-per-epoch-per-miner-design.md`](https://github.com/trajectoryRL/trajectoryrl.web/blob/main/docs/superpowers/specs/2026-05-07-per-epoch-per-miner-design.md) (in the web repo).

---

## Version History

| Version | Date | Summary |
|---------|------|---------|
| **v6.0** | **2026-05-07** | **Winner-challenger model.** Replaced decentralized off-chain CAS + chain-commitment-pointer consensus with server-coordinated challenge epochs (one challenger per epoch, evaluated against the seated winner). Centralized Phase-1 pre-eval as queue gate. Added `web submit` as a first-class submission channel alongside on-chain commitments. `winner_state` is now server-canonical (replaces per-validator local `WinnerState` JSON). Tightened Winner Protection δ from 10% to 3%. New tables: `challenge_epochs`, `winner_state`, `winner_history`, `validator_activity`.  New `eval_status = 'exhausted'` for `pack_hash` whose first scheduled epoch ended in `aborted_quorum` (no-retry policy — superseded the earlier retry-up-to-3 design). |
| v5.2 | 2026-04-26 | Deterministic window-start active-set snapshots; submit-when-fully-done with stake-quorum-gated aggregation; dual-window `physical_window` vs `target_window`; burn-while-waiting on quorum miss. |
| v5.1 | 2026-04-24 | Replaced on-chain `block_number` first-mover + NCD pre-eval with `pack_first_seen` ownership lock (no succession; 7-window grace). |
| v5.0 | 2026-04-21 | Refactored into season-agnostic core. Extracted scoring + pack schema to `SCORING_AND_EVALUATION.md`. Abstracted "cost" to "score". |
| v4.2 | 2026-03-29 | Simplified winner selection: removed EMA, unified Winner Protection (δ = 10%), stake-weighted majority qualification. |
| v4.1 | 2026-03-15 | Added two-phase off-chain consensus protocol (CAS + pointer registration). |
| v4.0 | 2026-03-01 | Replaced regex-based scoring with LLM-as-judge. |

Earlier versions of this document are preserved in `legacy/` (e.g. `legacy/INCENTIVE_MECHANISM_v5.2.md`).

---

## References

- **Bittensor Docs**: https://docs.bittensor.com
- **Dynamic TAO**: https://docs.bittensor.com/dtao
- **Yuma Consensus 3**: https://docs.learnbittensor.org/learn/yc3-blog
- **YC3 Migration Guide**: https://docs.learnbittensor.org/learn/yuma3-migration-guide
- **Current Season Scoring**: [SCORING_AND_EVALUATION.md](SCORING_AND_EVALUATION.md)
- **Miner Guide**: [MINER_OPERATIONS.md](MINER_OPERATIONS.md)
- **Validator Guide**: [VALIDATOR_OPERATIONS.md](VALIDATOR_OPERATIONS.md)
- **v6.0 Design Spec**: [`docs/superpowers/specs/2026-05-07-per-epoch-per-miner-design.md`](https://github.com/trajectoryRL/trajectoryrl.web/blob/main/docs/superpowers/specs/2026-05-07-per-epoch-per-miner-design.md) (web repo)
- **Source Code**: see `neurons/validator.py` and `trajectoryrl/` package

---

**Version**: v6.0
