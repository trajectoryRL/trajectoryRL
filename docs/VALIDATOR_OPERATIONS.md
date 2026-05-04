# Validator Operations Guide

**Subnet**: SN11 (TrajectoryRL)
**Season**: 1 (Self-Learning Agents)

> For consensus protocol, winner selection, and reward distribution, see [INCENTIVE_MECHANISM.md](INCENTIVE_MECHANISM.md).
> For miner-side flow (pack hosting, submission), see [MINER_OPERATIONS.md](MINER_OPERATIONS.md).

---

## What you do as a validator

Each epoch (~24 h, 7200 chain blocks):

1. **Pull the eval target set** from the subnet's snapshot endpoint
2. **Evaluate** each `(miner_hotkey, pack_hash)` whose `pre_eval_status = 'passed'` — run the trajrl-bench harness against the pack
3. **Submit scores** for the epoch
4. **Set on-chain weights** at the epoch boundary

Steps 1 and 3 are HTTP calls to the subnet's web service. The actual evaluation in step 2 runs locally on your hardware (sandbox + LLM + judge).

This guide covers step 1 — the **epoch_snapshot** API.

---

## The single source of truth: `/api/v2/validators/epoch_snapshot`

Validators **no longer query Bittensor commitments directly** at epoch start, and **no longer call `/api/v2/miners/pre-eval` per miner**. Both responsibilities are absorbed into the `epoch_snapshot` endpoint.

```
POST https://trajrl.com/api/v2/validators/epoch_snapshot
Content-Type: application/json

{
  "epoch_number": 1234,
  "validator_hotkey": "5FFA…",
  "timestamp": 1714000000,
  "signature": "0xabc…"
}
```

`signature` is sr25519 over `"trajectoryrl-snapshot:{validator_hotkey}:{timestamp}"` (5-min drift tolerance). **Signature is required** — the response includes `pack_url` for each entry, and for web-submitted packs that URL is sensitive (random GCS key) until its 48 h reveal gate.

### Response

```json
{
  "epoch_number": 1234,
  "built_at": "2026-05-04T12:05:00.000Z",
  "window_start": 8092800,
  "cutoff_block": 8092080,
  "cutoff_time": "2026-05-04T12:00:00.000Z",
  "eligible_start_time": "2026-05-02T12:00:00.000Z",
  "inactivity_window_hours": 48,
  "snapshot_block": 8092105,
  "entries": [
    {
      "uid": 42,
      "hotkey": "5GrwvaEF…",
      "pack_hash": "abc123…",
      "pack_url": "https://storage.googleapis.com/.../pack.json",
      "refresh_time": "2026-05-04T08:00:00.000Z",
      "pre_eval_status": "passed",
      "pre_eval_reason": null
    },
    {
      "uid": 88,
      "hotkey": "5HBE…",
      "pack_hash": "deadbeef…",
      "pack_url": "https://…",
      "refresh_time": "2026-05-04T09:30:00.000Z",
      "pre_eval_status": "failed",
      "pre_eval_reason": "hardcoded"
    }
  ]
}
```

### What each entry tells you

- `pre_eval_status: "passed"` → run the full eval on this `(hotkey, pack_hash)`
- `pre_eval_status: "failed"` → submit a score row with `rejected: true`, `rejection_stage: "integrity_check"`, `rejection_detail: <pre_eval_reason>`. **Don't run the eval.**

Pipeline failure reasons you may see in `pre_eval_reason`:

| Reason | Cause |
|---|---|
| `download failed after 3 retries: <err>` | miner's self-hosted URL is unreachable |
| `invalid_pack_json: missing or empty files.SKILL.md` | malformed pack |
| `hash_mismatch` / `invalid_pack_json` | claimed `pack_hash` doesn't match content |
| `pack_hash owned by <hotkey>` | uniqueness check (legacy multi-miner same pack) |
| `coldkey banned until <iso>` | miner's ownerkey accumulated 4+ failed packs in 30 days |
| `too similar (sim=X) to <hotkey>` | NCD copy detection (currently disabled by default) |
| `hardcoded` / 7 LLM red-line categories | LLM detected benchmark/runtime overfitting |

### Determinism guarantees

- The snapshot is **immutable per epoch**: every validator gets byte-identical bytes, regardless of when they call.
- The snapshot is **precomputed** by the subnet's sync worker as soon as `cutoff_time` passes (≈ 2.4 h before `window_start` of the requested epoch). The endpoint is a pure DB read.
- Sort order is `(refresh_time ASC, hotkey ASC)`.

---

## Cutoff and eligibility window

For epoch `N` the snapshot includes rows whose `refresh_time` falls in:

```
[ cutoff_time − 48h ,   cutoff_time )

cutoff_time = block→time(window_start(N) − 720)
            = aggregation_start(N−1)
            = window_start(N) − 2.4h
```

- **Upper bound** (`cutoff_time`): rows refreshed at or after this point are deferred to the next epoch's snapshot. The 2.4 h gap is the contract with the sync worker — every row that lands before cutoff is guaranteed to have its full check pipeline complete before epoch N's eval phase opens.
- **Lower bound** (`eligible_start_time = cutoff_time − 48h`): rows refreshed earlier than this are considered abandoned and excluded.

A miner stays in the eval set as long as either (a) their on-chain commitment is still on chain and the sync worker is bumping their `refresh_time` every 5 min, or (b) they re-submit via `/api/v2/miners/submit` within the 48 h window.

---

## Signed-request example

`signature` is sr25519 over `trajectoryrl-snapshot:{validator_hotkey}:{timestamp}`. This endpoint uses its own dedicated prefix (distinct from the `trajectoryrl-report` prefix that `/api/v2/miners/pre-eval` and `/api/v2/scores/submit` share).

Python (using `bittensor` wallet API):

```python
import time, json, requests
from bittensor import wallet

w = wallet(name="validator", hotkey="default")
hotkey = w.hotkey.ss58_address
timestamp = int(time.time())
message = f"trajectoryrl-snapshot:{hotkey}:{timestamp}"
signature = "0x" + w.hotkey.sign(message.encode()).hex()

r = requests.post(
    "https://trajrl.com/api/v2/validators/epoch_snapshot",
    json={
        "epoch_number": 1234,
        "validator_hotkey": hotkey,
        "timestamp": timestamp,
        "signature": signature,
    },
    timeout=15,
)
r.raise_for_status()
snapshot = r.json()
for e in snapshot["entries"]:
    if e["pre_eval_status"] == "passed":
        # run eval against e["pack_hash"] / e["pack_url"]
        ...
    else:
        # submit a rejected score row
        ...
```

---

## Failure modes & retries

| Status | Meaning | What to do |
|---|---|---|
| 200 | snapshot returned | proceed to eval phase |
| 400 | `epoch_number` missing/invalid | check request body |
| 401 | auth fields missing | include validator_hotkey + timestamp + signature |
| 403 | hotkey not on-chain validator OR signature invalid | re-derive signature; check hotkey registration |
| 404 | `Snapshot for epoch N is not available yet` | sync worker hasn't built it yet — retry on next cycle (~5 min) |

The endpoint is the **only** source for the eval set, so on any non-200 the validator should wait and retry on its next iteration. The eval phase covers roughly the first 80 % of the window (≈ 19 h of a 24 h window), giving wide retry headroom. There is no client-side fallback to a chain query.

A validator that cannot reach this endpoint for an entire epoch will not eval that epoch and falls through to its existing fallback weights behavior (set_weights to subnet-owner UID, miner emissions burned).

---

## Recommended validator loop

```python
import time

while True:
    target_epoch = next_unevaluated_epoch()       # local state

    while True:
        try:
            snap = fetch_epoch_snapshot(target_epoch)   # HTTP call
            break
        except SnapshotNotReady:
            time.sleep(300)                              # ~5 min then retry
        except (NetworkError, AuthError) as e:
            log_and_alert(e); time.sleep(60); continue

    for entry in snap["entries"]:
        if entry["pre_eval_status"] == "passed":
            score = run_full_eval(entry)
            submit_score(entry, score)
        else:
            submit_rejection(entry, reason=entry["pre_eval_reason"])

    set_on_chain_weights(target_epoch)            # at end of epoch
```

The full request/response spec for `/api/v2/validators/epoch_snapshot` (including processing flow, determinism guarantees, and error codes) is in [`trajectoryrl.web/API.md`](https://github.com/trajectoryRL/trajectoryrl.web/blob/main/API.md).

---

## References

- **API spec**: [API.md (POST /api/v2/validators/epoch_snapshot)](https://github.com/trajectoryRL/trajectoryrl.web/blob/main/API.md)
- **Incentive Mechanism**: [INCENTIVE_MECHANISM.md](INCENTIVE_MECHANISM.md) — consensus protocol, winner selection
- **Score submission**: [API.md (POST /api/v2/scores/submit)](https://github.com/trajectoryRL/trajectoryrl.web/blob/main/API.md) — how to report eval results
- **Heartbeat & log upload**: [API.md (POST /api/v2/validators/heartbeat, POST /api/validators/logs/upload)](https://github.com/trajectoryRL/trajectoryrl.web/blob/main/API.md)
