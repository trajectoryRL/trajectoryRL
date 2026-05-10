# API Documentation

This document focuses on **validator integration** (pre-eval, score submit, consensus payload, etc.). Website-only monitoring endpoints (for example per-miner epoch breakdown) live in [docs/web-api-epoch-summary.md](docs/web-api-epoch-summary.md).

## POST /api/v2/miners/pre-eval

Read-only query for validators to check a miner's eval status before spending resources on a full evaluation. All actual pack evaluation is handled by the background `syncMinerSubmissions` process using on-chain commitment data — this endpoint never triggers evaluation itself.

See [docs/pre-eval.md](docs/pre-eval.md) for full architecture and integration guide.

### Method
`POST`

### Headers
- `Content-Type`: `application/json`

### Request Body

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `miner_hotkey` | string | Yes | Hotkey of the miner to check |
| `pack_hash` | string | Conditional | Pack hash to query. Required if `epoch_number` is not provided. |
| `epoch_number` | number | Conditional | Epoch number to query. Required if `pack_hash` is not provided. When used, returns the eval result for the most recently submitted pack in this epoch. |
| `validator_hotkey` | string | No | Hotkey of the calling validator. Required for signed requests. |
| `timestamp` | number | No | Unix timestamp in seconds. Required for signed requests. |
| `signature` | string | No | Validator's signature over `"trajectoryrl-report:{validator_hotkey}:{timestamp}"`. Required for signed requests. |

> **Signing is optional.** If all three signing fields (`validator_hotkey`, `timestamp`, `signature`) are provided, the server verifies the validator's identity and signature. If omitted, the request is processed without authentication. This is safe because the endpoint is purely read-only.

### Request Examples

**Minimal request (by pack_hash, unsigned):**
```json
{
  "miner_hotkey": "5FFApaS75bvpgP9gQ5hTUdZHiTc6LB2VPP9gvHN6VQCNug6e",
  "pack_hash": "abc123def456"
}
```

**By epoch_number (unsigned):**
```json
{
  "miner_hotkey": "5FFApaS75bvpgP9gQ5hTUdZHiTc6LB2VPP9gvHN6VQCNug6e",
  "epoch_number": 1234
}
```

**Signed request (backward compatible):**
```json
{
  "validator_hotkey": "5FFApaS75bvpgP9gQ5hTUdZHiTc6LB2VPP9gvHN6VQCNug6f",
  "timestamp": 1710000000,
  "signature": "0xabc123...",
  "miner_hotkey": "5FFApaS75bvpgP9gQ5hTUdZHiTc6LB2VPP9gvHN6VQCNug6e",
  "pack_hash": "abc123def456"
}
```

### Success Responses

**Allowed — pack passed verification:**
```json
{ "allowed": true, "verified": true, "pack_hash": "abc123def456" }
```

**Allowed — pack not yet evaluated:**
```json
{ "allowed": true, "verified": false, "pack_hash": "abc123def456" }
```

**Blocked — miner's owner is banned:**
```json
{
  "allowed": false,
  "reason": "banned",
  "ownerkey": "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY",
  "banned_until": "2025-04-01T00:00:00.000Z"
}
```

**Blocked — hardcoding detected:**
```json
{
  "allowed": false,
  "reason": "hardcoded",
  "verified": true,
  "pack_hash": "abc123def456"
}
```

> **Note:** When querying by `epoch_number`, the response `pack_hash` is resolved from the most recently submitted pack in that epoch. This lets callers discover which pack was matched.

### Error Responses

| Status | Error | Message |
|--------|-------|---------|
| 400 | Missing/invalid fields | Field-specific error message (e.g. `"Either pack_hash or epoch_number is required"`) |
| 403 | Validator not on-chain | `"Hotkey is not a registered validator on-chain"` (signed requests only) |
| 403 | Miner not on-chain | `"miner_hotkey is not a registered miner on-chain"` |
| 403 | Invalid signature | Verification error message (signed requests only) |
| 500 | Server error | `"Internal server error"` |

### Processing Flow

1. Validate request body fields (`miner_hotkey` required; at least one of `pack_hash` or `epoch_number`)
2. If signing fields are present (`validator_hotkey` + `timestamp` + `signature`):
   a. Verify `validator_hotkey` is a registered on-chain validator
   b. Verify validator Sr25519 signature (5-minute drift tolerance)
3. Verify `miner_hotkey` is a registered, non-deregistered miner
4. Check if the miner's ownerkey is banned → return `allowed: false, reason: "banned"`
5. Look up cached eval result:
   - If `pack_hash` provided → query by `(miner_hotkey, pack_hash)`
   - If `epoch_number` provided → query by `(miner_hotkey, epoch_number)`, returning the latest submission in that epoch
6. Return result:
   - `passed` → return `allowed: true, verified: true, pack_hash`
   - `failed` → return `allowed: false, reason, verified: true, pack_hash`
   - Not found / `pending` → return `allowed: true, verified: false`

---


## POST /api/v2/miners/submit

Miner-facing endpoint for submitting a `pack.json` directly to the web service. The web stores the pack in GCS under an unguessable random filename (so competitors can't enumerate packs by hotkey/uid/hash), immediately runs pre-eval asynchronously, and persists a `miner_submissions` row tagged `source='web'`.

This **complements** — does not replace — the existing self-hosted flow. Validators still discover miners via on-chain commitments; this endpoint gives miners a managed hosting option and a faster pre-eval feedback loop.

### Method
`POST`

### Headers
- `Content-Type`: `application/json`

### Request Body

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `miner_hotkey` | string | Yes | Submitting miner's ss58 hotkey. |
| `timestamp` | number | Yes | Unix timestamp in **seconds**. Drift > 5 min is rejected. |
| `signature` | string | Yes | Sr25519 signature (hex, optional `0x` prefix) over `"trajectoryrl-miner-submit:{miner_hotkey}:{timestamp}"`. |
| `pack_hash` | string | Yes | 64-char lowercase hex SHA-256 of the canonical pack JSON. Server recomputes and rejects on mismatch. |
| `pack_content` | string | Yes | The pack JSON serialized in **canonical** form (Python `json.dumps(pack, sort_keys=True)` with non-ASCII characters escaped as `\uXXXX`). Must hash to `pack_hash`. Maximum **32 KB**. |

> **Identity validation** can be disabled in development by setting `VALIDATE_MINER_IDENTITY=false`. Production must leave it enabled.

### Request Example

```json
{
  "miner_hotkey": "5FFApaS75bvpgP9gQ5hTUdZHiTc6LB2VPP9gvHN6VQCNug6e",
  "timestamp": 1714000000,
  "signature": "0xabc123...",
  "pack_hash": "ab12...64hex",
  "pack_content": "{\"files\": {\"SKILL.md\": \"...\"}, \"schema_version\": 1}"
}
```

### Success Response (200)

```json
{
  "success": true,
  "pack_hash": "ab12...64hex",
  "pack_url": "https://storage.googleapis.com/<bucket>/<prefix>/<uid>/<random_key>.json",
  "submission_id": 12345,
  "next_upload_allowed_at": "2026-05-04T15:00:00.000Z",
  "cooldown_seconds": 1200,
  "pre_eval_status": "pending"
}
```

| Field | Meaning |
|-------|---------|
| `pack_hash` | Echoed back (lowercased, normalized). |
| `pack_url` | GCS URL of the stored pack, hosted under `<S3_PREFIX>/<uid>/<random_key>.json`. **Returned only to the submitting miner** — this URL is intentionally not exposed by any other public endpoint, so leak risk is bounded to the miner's own response. The miner uses this URL to verify upload contents and commits `<pack_hash>\|<pack_url>` on-chain so validators can pick it up via the existing chain-sync flow. |
| `submission_id` | Row id in `miner_submissions`. |
| `next_upload_allowed_at` | When the per-miner cooldown lifts. |
| `cooldown_seconds` | Configured cooldown window (default 1200 = 20 min, matching the on-chain commitment rate limit). |
| `pre_eval_status` | `"pending"` / `"passed"` / `"failed"`. On a first-time `(hotkey, pack_hash)` insert this is always `"pending"` — pre-eval runs asynchronously after the response. On a refresh of an existing pair, the row's current lifecycle state is collapsed via `toLegacyPreEvalStatus()`: internal `pending_pre_eval` → `"pending"`, `pending_eval` and `completed` → `"passed"`, `failed` → `"failed"`. The wire value set is the same as before — see `docs/eval-status-lifecycle.md`. Poll `/api/v2/miners/pre-eval` to check the outcome of an async run. |

### Error Responses

| Status | Error | Notes |
|--------|-------|-------|
| 400 | Field-specific message | Missing/invalid required field, body not JSON, or `pack_hash` does not equal server-recomputed hash of `pack_content`. |
| 400 | `pack_content must be ≤32768 bytes` | `pack_content` size cap. Total request body is also capped at 64 KB. |
| 400 | `pack_content must be valid pack.json containing files.SKILL.md` | Pack shape check. |
| 403 | Signature error | Timestamp drift > 5 min or invalid signature. |
| 403 | `miner_hotkey is not a registered miner on-chain` | Hotkey absent from metagraph or deregistered. |
| 403 | `ownerkey is currently banned` | Owner ban active. Response also includes `ownerkey` and `banned_until`. |
| 429 | `cooldown` | Per-miner cooldown not yet elapsed. Response includes `last_upload_at`, `next_upload_allowed_at`, `cooldown_seconds`. |
| 500 | `Failed to upload pack content to storage` | GCS write failed. Cooldown is **not** consumed (the row is written only after a successful GCS upload). Safe to retry. |
| 500 | `Failed to persist submission record` | DB write failure. |

### Cooldown Semantics

- **Per-miner**, **strict counting**: 1 successful submission per `cooldown_seconds` (default **1200 = 20 min**, matching the Bittensor on-chain commitment rate limit so neither submit channel offers a faster path; overridable via `MINER_SUBMIT_COOLDOWN_SECONDS`), regardless of whether pre-eval ultimately passes or fails.
- The cooldown clock starts when the `miner_submissions` row is persisted (which is **after** GCS upload succeeds). Field-validation errors, signature errors, and GCS upload failures do not consume the cooldown.
- Resubmitting the same `pack_hash` within the cooldown window returns `429`. The previously-issued `pack_url` remains valid.

### Idempotency

If the same `(miner_hotkey, pack_hash)` is submitted again **after** the cooldown elapses, the existing `pack_url` on the upload row is reused — the hash check guarantees the new body is byte-identical to the stored object, so no re-upload is performed and the URL returned to the miner is unchanged.

### On-chain Commit (recommended workflow)

After a successful submission, miners should still commit `<pack_hash>|<pack_url>` to Subtensor `commitments` so validators pick the pack up through the existing chain-sync pipeline (preserves first-mover precedence and keeps the discovery layer decentralized).

### Processing Flow

1. Read body (≤ 64 KB), parse as JSON object
2. Validate `miner_hotkey`, `timestamp`, `signature`, `pack_hash`, `pack_content` shape
3. Verify `pack_content` size (≤ 32 KB) and that `sha256(canonical_json(pack_content))` equals `pack_hash`
4. Verify pack shape: `files["SKILL.md"]` is present
5. Verify Sr25519 signature over `"trajectoryrl-miner-submit:{miner_hotkey}:{timestamp}"` (5-min drift)
6. Verify miner is registered on-chain and not deregistered (also reads `uid` for the GCS path)
7. Verify ownerkey is not banned
8. Enforce per-miner cooldown (latest `created_at` for `source='web'` rows of this hotkey)
9. Look up an existing upload-source `pack_url` for `(miner_hotkey, pack_hash)`; if present, skip the GCS write (the hash check guarantees the stored object is byte-identical), otherwise generate a fresh 16-char base64url filename
10. If a fresh upload is needed, write the body to GCS at `<S3_PREFIX>/<uid>/<random_key>.json` (`Content-Type: application/json`). The `<S3_PREFIX>` segment matches the existing `sync-packs` convention (`traj-prod` in production, `traj-dev` in development, overridable via `S3_PREFIX`); the random filename is what makes the URL unguessable. The random key is a transient value — only the full URL is persisted (in `pack_url`)
11. Upsert the `miner_submissions` row with `source='web'` and the GCS URL stored in `pack_url`. **`gcs_pack_url` is intentionally left empty at this point** — the `sync-packs` job promotes `pack_url` into `gcs_pack_url` only after the row is older than 24 hours, deferring the publicly-visible mirror URL by that window
12. Fire-and-forget pre-eval (`evaluateSubmissionNow`) — validators / the miner can poll `/api/v2/miners/pre-eval` for the result a few seconds later
13. Return the success payload

---


## GET /api/previous-epoch

Returns the computed summary for the previous epoch from `epoch_summary`.

### Method
`GET`

### Success Response (200)

| Field | Type | Description |
|-------|------|-------------|
| `blockHeight` | number | Latest synced chain block height |
| `epochNumber` | number \| null | Previous epoch number; `null` when unavailable |
| `status` | `"winner"` \| `"burn"` \| `"no_quorum"` \| null | Epoch resolution status |
| `winner` | object \| null | Winner details when `status="winner"` |
| `diagnostics` | object \| null | Quorum and aggregation diagnostics |
| `finalMinerResults` | array \| null | Full final miner list used by epoch summary winner resolution |
| `stats` | object | Operational counters for validators/miners/submissions |

### `finalMinerResults` Item Schema

| Field | Type | Description |
|-------|------|-------------|
| `hotkey` | string | Miner hotkey |
| `uid` | number \| null | Miner UID |
| `score` | number \| null | Final aggregated score for this epoch (`0` is valid) |
| `rank` | number \| null | Rank among **eligible** miners only; `null` for disqualified rows |
| `disqualified` | boolean | Whether this miner was filtered out from winner candidacy |

### Semantics

- `disqualified=true` means the miner was excluded by consensus/pre-eval filters and is not ranked (`rank=null`).
- `disqualified=false` with `score=0` means the miner remained eligible, was not filtered, and is still represented separately from disqualified rows.

For per-miner stake-weighted validator breakdown (`GET /api/previous-epoch/miner-breakdown`), used by the website leaderboard, see [docs/web-api-epoch-summary.md](docs/web-api-epoch-summary.md).

---


## POST /api/v2/scores/submit

> **Legacy (v5.2).** v6.0 daemons must use [`POST /api/v2/epoch/{challenge_epoch_id}/score`](#post-apiv2epochchallenge_epoch_idscore) instead. Retained for the v5.2 daemon during cutover; new code must not target this path.

**Purpose: statistics and auditing only. This endpoint is not in the validator's critical path.**

Validators submit eval results — including pre-eval rejections — through this endpoint. Tracks the evaluation environment (Docker image digests and trajrl-bench version) alongside the score itself.

After completing an evaluation of one miner, a validator asynchronously reports the locally-computed results to this endpoint for monitoring and audit purposes. The validator's core eval loop (EMA calculation, weight setting) must not depend on this endpoint — submission failures must be silently ignored and must never block or affect the validator's operation.

### Method
`POST`

### Headers
- `Content-Type`: `application/json`

### Two Submission Modes

This endpoint handles two types of submissions:

- **Normal eval result** (`rejected` absent or `false`): full evaluation completed, includes score and per-scenario results.
- **Pre-eval rejection** (`rejected: true`): evaluation was aborted before any episode ran. `score` and `weight` must be `0`, `qualified` must be `false`, and `scenario_results` must be omitted. `rejection_stage` and `rejection_detail` describe the failure.

### Request Body

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `validator_hotkey` | string | Yes | Hotkey of the submitting validator |
| `miner_hotkey` | string | Yes | Hotkey of the evaluated miner |
| `miner_uid` | number | Yes | UID of the evaluated miner |
| `block_height` | number | Yes | Current block height |
| `epoch_number` | number | No | Optional epoch number this submission belongs to. When supplied (non-negative integer), the server stores it as-is. When omitted/null, the server derives it from the latest synced `block_height` in `sync_state`. Use this if your eval pipeline already knows the chain epoch and you want to avoid drift caused by sync lag. |
| `timestamp` | number | Yes | Unix timestamp in seconds |
| `signature` | string | Yes | Validator's signature over `"trajectoryrl-submit:{validator_hotkey}:{timestamp}"` |
| `version` | string | Yes | Current running `trajectoryrl` package version (e.g. `"0.4.14"`) |
| `score` | number | Yes | Raw score from this single eval; `0` for rejections |
| `weight` | number | Yes | On-chain weight assigned to this miner at last set_weights; `0` for rejections |
| `qualified` | boolean | Yes | Whether the miner is qualified; always `false` for rejections |
| `pack_url` | string | No | Data pack URL submitted by the miner |
| `pack_hash` | string | No | Data pack hash |
| `eval_count` | number | No | Number of evals accumulated for the current pack; omitted for rejections |
| `scenario_results` | object | No | Per-scenario eval results keyed by scenario name (see below); omitted for rejections |
| `llm_base_url` | string | No | LLM API base URL used by the validator |
| `llm_model` | string | No | LLM model identifier used by the validator (e.g. `"claude-sonnet-4-6"`) |
| `rejected` | boolean | No | `true` if the eval was aborted before any episode ran (pre-eval rejection). Omit or set `false` for normal eval results |
| `rejection_stage` | string | No | Required when `rejected` is `true`. Stage at which the eval was rejected: `"pack_fetch"` \| `"schema_validation"` \| `"integrity_check"` |
| `rejection_detail` | string | No | Human-readable description of the rejection reason (e.g. `"hard-coded responses detected"`) |
| `spec_number` | number | No | Major version number of `bench_version` (e.g. `bench_version` `"v3.0.1"` → `spec_number` `3`). Used to isolate consensus aggregation across incompatible bench releases. **New canonical name**: clients should send this field. |
| `scoring_version` | number | No | **Legacy alias for `spec_number`**. Accepted for backward compatibility during the migration window. If both are present they must agree (server prefers `spec_number` and logs a warning on mismatch). New clients should send `spec_number` instead. |
| `bench_image_hash` | string | No | Docker image digest of the `trajrl-bench` sandbox image (e.g. `"sha256:a1b2c3..."`) |
| `harness_image_hash` | string | No | Docker image digest of the `hermes-agent` harness image (e.g. `"sha256:d4e5f6..."`) |
| `bench_version` | string | No | Version string reported by the trajrl-bench CLI inside the sandbox container (e.g. `"v1.2.0"`) |

#### scenario_results Object Structure

Keyed by scenario name. Each value contains:

| Field | Type | Description |
|-------|------|-------------|
| `score` | number | Binary score: `1.0` if qualified, `0.0` otherwise |
| `weight` | number | Scenario weight used in weighted aggregation |
| `qualified` | boolean | Whether the LLM judge passed this scenario |
| `token_usage` | object | Token consumption for this scenario (omitted if unavailable) |
| `token_usage.input_tokens` | number | Input tokens consumed |
| `token_usage.output_tokens` | number | Output tokens consumed |
| `token_usage.cache_read_tokens` | number | Cache read tokens consumed |
| `token_usage.cache_write_tokens` | number | Cache write tokens consumed |
| `model_usage` | array | Per-model breakdown (omitted if unavailable) |
| `model_usage[].model` | string | Model identifier |
| `model_usage[].count` | number | Number of calls to this model |
| `judge` | object | LLM trajectory judge details (omitted if unavailable) |

#### scenario_results[].judge Object Structure

| Field | Type | Description |
|-------|------|-------------|
| `overall_score` | number | Weighted judge score (0.0–1.0) |
| `safety_passed` | boolean | Whether all safety criteria passed |
| `correctness_passed` | boolean | Whether all correctness criteria passed |
| `qualification_gate` | boolean | Final pass/fail verdict (AND of safety + correctness) |
| `verdict` | string | Criteria pass summary in `"x/y"` format (e.g. `"3/4"` means 3 of 4 criteria passed) |
| `grounded` | string | Grounded verdict summary in `"x/y"` format (e.g. `"2/4"` means 2 of 4 criteria were grounded in the trajectory) |
| `error` | string \| null | Error message if the judge encountered an error, otherwise `null` |

### Request Example

```json
{
  "validator_hotkey": "5FFApaS75bvpgP9gQ5hTUdZHiTc6LB2VPP9gvHN6VQCNug6f",
  "miner_hotkey": "5FFApaS75bvpgP9gQ5hTUdZHiTc6LB2VPP9gvHN6VQCNug6e",
  "miner_uid": 1,
  "block_height": 12345,
  "timestamp": 1710000000,
  "signature": "0xabc123...",
  "version": "0.4.14",
  "score": 0.93,
  "weight": 1.0,
  "qualified": true,
  "pack_url": "https://example.com/pack1.zip",
  "pack_hash": "abc123def456...",
  "eval_count": 5,
  "llm_base_url": "https://api.anthropic.com",
  "llm_model": "claude-sonnet-4-6",
  "spec_number": 1,
  "scoring_version": 1,
  "bench_image_hash": "sha256:a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2",
  "harness_image_hash": "sha256:f6e5d4c3b2a1f6e5d4c3b2a1f6e5d4c3b2a1f6e5d4c3b2a1f6e5d4c3b2a1f6e5",
  "bench_version": "v1.2.0",
  "scenario_results": {
    "client_escalation": {
      "score": 1.0,
      "weight": 1.0,
      "qualified": true,
      "judge": {
        "overall_score": 0.9167,
        "safety_passed": true,
        "correctness_passed": true,
        "qualification_gate": true,
        "verdict": "2/2",
        "grounded": "2/2",
        "error": null
      }
    }
  }
}
```

### Success Response

```json
{
  "success": true,
  "result": {
    "validator_hotkey": "5FFApaS75bvpgP9gQ5hTUdZHiTc6LB2VPP9gvHN6VQCNug6f",
    "miner_hotkey": "5FFApaS75bvpgP9gQ5hTUdZHiTc6LB2VPP9gvHN6VQCNug6e",
    "miner_uid": 1,
    "block_height": 12345,
    "score": 0.93,
    "qualified": true,
    "submittedAt": "2025-03-09T10:30:00.000Z"
  }
}
```

### Error Responses

| Error | Message |
|-------|---------|
| Invalid request body | `"Request body must be a JSON object"` |
| Invalid validator_hotkey | `"validator_hotkey is required and must be a non-empty string"` |
| Invalid miner_hotkey | `"miner_hotkey is required and must be a non-empty string"` |
| Invalid miner_uid | `"miner_uid is required and must be a non-negative number"` |
| Invalid block_height | `"block_height is required and must be a positive number"` |
| Invalid timestamp | `"timestamp is required (unix seconds)"` |
| Invalid signature | `"signature is required"` |
| Invalid score | `"score is required and must be a number"` |
| Invalid qualified | `"qualified is required and must be a boolean"` |
| Signature verification failed | Verification error message |
| Server error | `"Internal server error"` |

### Processing Flow

1. Validate request body fields
2. If `rejected` is `true`, verify that `rejection_stage` is one of the allowed values and that `score` and `weight` are both `0` and `qualified` is `false`
3. Verify validator signature
4. Upsert the miner eval result to the database (unique key: `validator_hotkey + miner_hotkey + block_height`), including `bench_image_hash`, `harness_image_hash`, and `bench_version` if provided
5. Return success response

> **Validator-side requirement**: This call must be fire-and-forget. Any network error, timeout, or non-2xx response must be logged and discarded. The validator must never retry in a blocking manner or halt its eval loop waiting for this response.

---

## POST /api/validators/logs/upload

**Purpose: debugging and auditing only. This endpoint is not in the validator's critical path.**

After completing (or failing) a single miner evaluation, the validator uploads all log files from that eval run. Each upload corresponds to exactly one miner and one pack.

Like `/api/scores/submit`, this is fire-and-forget — failures must be silently ignored and must never block or affect the validator's operation.

### Method
`POST`

### Headers
- `Content-Type`: `multipart/form-data`

### Request Body (multipart form fields)

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `validator_hotkey` | string | Yes | Hotkey of the submitting validator |
| `eval_id` | string | Yes | Eval cycle identifier in `YYYYMMDD_HHMMSS` format (e.g. `"20260320_143025"`). All per-miner uploads and the cycle-level upload within the same eval cycle share the same `eval_id`. |
| `miner_hotkey` | string | Yes | Hotkey of the evaluated miner |
| `miner_uid` | string | Yes | UID of the evaluated miner (string-encoded number) |
| `block_height` | string | Yes | Block height of this eval (string-encoded number; links to the corresponding `submit_eval` record) |
| `pack_hash` | string | Yes | SHA-256 hash of the pack being evaluated. Each upload must correspond to exactly one pack. |
| `epoch_number` | string | No | Optional epoch number (string-encoded non-negative integer) this log belongs to. When supplied, the server stores it as-is on the `eval_logs` row. When omitted/empty, the server derives it from the latest synced `block_height` in `sync_state`. |
| `timestamp` | string | Yes | Unix timestamp in seconds (string-encoded number) |
| `signature` | string | Yes | Validator's signature over `"trajectoryrl-logs:{validator_hotkey}:{timestamp}"` |
| `log_archive` | file | Yes | `.tar.gz` archive containing all log files for this eval run. Max 10 MB. |

### Archive Structure

The `log_archive` file must be a valid gzip-compressed tar archive. Expected contents:

```
validator.log                            # Main validator log segment for this eval
miner.log                               # Per-miner log segment for this eval
client_escalation_calls.jsonl            # Tool call log per scenario (JSONL)
client_escalation_all_requests.jsonl     # All HTTP requests per scenario (JSONL)
morning_brief_calls.jsonl
morning_brief_all_requests.jsonl
...                                      # One pair per scenario evaluated
```

- `validator.log`: Main validator process log segment covering this eval run — includes overall eval flow (miner selection, skip reasons, timing, errors).
- `miner.log`: Per-miner log segment covering this eval run — includes per-scenario judge scores, cost breakdowns, tool call counts, and criterion-level pass/fail details.
- `{scenario}_calls.jsonl`: One JSON object per line, each recording a tool call made during the episode.
- `{scenario}_all_requests.jsonl`: One JSON object per line, each recording an HTTP request to mock-tools during the episode.

All log segments are excerpts covering **only the current eval run**, not full history.

### Request Example (curl)

```bash
curl -X POST https://trajrl.com/api/validators/logs/upload \
  -F "validator_hotkey=5FFApaS75bvpgP9gQ5hTUdZHiTc6LB2VPP9gvHN6VQCNug6f" \
  -F "eval_id=20260320_143025" \
  -F "miner_hotkey=5FFApaS75bvpgP9gQ5hTUdZHiTc6LB2VPP9gvHN6VQCNug6e" \
  -F "miner_uid=1" \
  -F "block_height=12345" \
  -F "pack_hash=abc123def456..." \
  -F "epoch_number=1234" \
  -F "timestamp=1710000000" \
  -F "signature=0xabc123..." \
  -F "log_archive=@/app/logs/eval_logs.tar.gz;type=application/gzip"
```

### Success Response

```json
{
  "success": true,
  "log_id": "abc123def456",
  "size_bytes": 245760
}
```

### Error Responses

| Status | Error | Message |
|--------|-------|---------|
| 400 | Missing/invalid fields | Field-specific error message |
| 400 | Archive too large | `"log_archive exceeds 10MB limit"` |
| 400 | Invalid archive | `"log_archive must be a valid .tar.gz file"` |
| 403 | Validator not on-chain | `"Hotkey is not a registered validator on-chain"` |
| 403 | Invalid signature | Verification error message |
| 404 | No matching eval | `"No eval record found for this validator/miner/block combination"` |
| 500 | Server error | `"Internal server error"` |

### Processing Flow

1. Validate form fields (all required fields present, `miner_uid` and `block_height` are valid numbers, `pack_hash` is non-empty)
2. Verify `validator_hotkey` is a registered on-chain validator
3. Verify validator signature over `"trajectoryrl-logs:{validator_hotkey}:{timestamp}"`
4. Validate archive size (<= 10 MB) and format (valid `.tar.gz`)
5. Look up the corresponding eval record by `validator_hotkey + miner_hotkey + block_height + pack_hash`
6. Store the archive in object storage (keyed by `{validator_hotkey}/{pack_hash}/{block_height}.tar.gz`)
7. Update the eval record with a reference to the stored log archive
8. Return success with `log_id` and `size_bytes`

> **Validator-side requirement**: Same fire-and-forget contract as `/api/scores/submit`. Any network error, timeout, or non-2xx response must be logged and discarded. The validator must never retry in a blocking manner or halt its eval loop waiting for this response.

---

## POST /api/validators/logs/cycle

**Purpose: debugging and auditing only. This endpoint is not in the validator's critical path.**

At the end of a full eval cycle, the validator uploads the overall cycle log covering all miners. This log captures the complete picture of the cycle — which miners were evaluated, skipped, rejected, cached, and the final summary.

Like `/api/scores/submit`, this is fire-and-forget.

### Method
`POST`

### Headers
- `Content-Type`: `multipart/form-data`

### Request Body (multipart form fields)

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `validator_hotkey` | string | Yes | Hotkey of the submitting validator |
| `eval_id` | string | Yes | Eval cycle identifier in `YYYYMMDD_HHMMSS` format (same value shared with per-miner uploads of this cycle) |
| `block_height` | string | Yes | Block height at cycle start (string-encoded number) |
| `epoch_number` | string | No | Optional epoch number (string-encoded non-negative integer) this cycle belongs to. When supplied, the server stores it as-is on the `eval_logs` row. When omitted/empty, the server derives it from the latest synced `block_height` in `sync_state`. |
| `timestamp` | string | Yes | Unix timestamp in seconds (string-encoded number) |
| `signature` | string | Yes | Validator's signature over `"trajectoryrl-logs:{validator_hotkey}:{timestamp}"` |
| `log_archive` | file | Yes | `.tar.gz` archive containing the cycle log. Max 10 MB. |

### Archive Structure

```
validator.log                            # Full eval cycle log
```

- `validator.log`: Complete validator log from cycle start to end — includes metagraph sync, miner enumeration, pre-eval rejections, cache hits, interval skips, per-miner eval timing, error summaries, and the cycle completion summary.

### Request Example (curl)

```bash
curl -X POST https://trajrl.com/api/validators/logs/cycle \
  -F "validator_hotkey=5FFApaS75bvpgP9gQ5hTUdZHiTc6LB2VPP9gvHN6VQCNug6f" \
  -F "eval_id=20260320_143025" \
  -F "block_height=12345" \
  -F "epoch_number=1234" \
  -F "timestamp=1710000000" \
  -F "signature=0xabc123..." \
  -F "log_archive=@/app/logs/cycle_log.tar.gz;type=application/gzip"
```

### Success Response

```json
{
  "success": true,
  "log_id": "abc123def456",
  "size_bytes": 51200
}
```

### Error Responses

| Status | Error | Message |
|--------|-------|---------|
| 400 | Missing/invalid fields | Field-specific error message |
| 400 | Archive too large | `"log_archive exceeds 10MB limit"` |
| 400 | Invalid archive | `"log_archive must be a valid .tar.gz file"` |
| 403 | Validator not on-chain | `"Hotkey is not a registered validator on-chain"` |
| 403 | Invalid signature | Verification error message |
| 500 | Server error | `"Internal server error"` |

### Processing Flow

1. Validate form fields (`validator_hotkey`, `block_height`, `timestamp`, `signature` present and valid)
2. Verify `validator_hotkey` is a registered on-chain validator
3. Verify validator signature over `"trajectoryrl-logs:{validator_hotkey}:{timestamp}"`
4. Validate archive size (<= 10 MB) and format (valid `.tar.gz`)
5. Store the archive in object storage (keyed by `{validator_hotkey}/__cycle__/{block_height}.tar.gz`)
6. Return success with `log_id` and `size_bytes`

> **Validator-side requirement**: Same fire-and-forget contract as `/api/scores/submit`.

### Validator-Side Local Storage

Both per-miner and cycle-level logs are organized locally under `{log_dir}/evals/`:

```
{log_dir}/evals/
├── 20260320_143025/                     # eval timestamp (YYYYMMDD_HHMMSS)
│   ├── validator.log                    # cycle-level log (uploaded via /logs/cycle)
│   ├── 5FFApaS75bvpgP9g/               # miner hotkey[:16]
│   │   ├── validator.log
│   │   ├── miner.log
│   │   ├── client_escalation_calls.jsonl
│   │   ├── client_escalation_all_requests.jsonl
│   │   └── morning_brief_calls.jsonl
│   └── 5GrwvaEF5zXb26Fz/               # another miner in the same eval cycle
│       └── ...
├── 20260320_162510/                     # next eval cycle
│   └── ...
```

The timestamp serves as the eval cycle identifier. The cycle-level `validator.log` sits directly under the timestamp directory, while per-miner logs are partitioned into subdirectories.

---

## POST /api/v2/validators/heartbeat

Submit a validator heartbeat with running version, Docker image digests, and sandbox version. Validators call this endpoint periodically to report liveness and the eval environment they're running.

### Method
`POST`

### Headers
- `Content-Type`: `application/json`

### Request Body

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `hotkey` | string | Yes | Validator hotkey, must be a non-empty string |
| `version` | string | Yes | Current running `trajectoryrl` package version (e.g. `"0.4.14"`) |
| `timestamp` | number | Yes | Unix timestamp in seconds, must be positive |
| `signature` | string | Yes | Validator's signature over `"trajectoryrl-heartbeat:{hotkey}:{timestamp}"` |
| `last_set_weights_at` | number | No | Unix timestamp (seconds) of the most recent set-weight operation |
| `last_eval_at` | number | No | Unix timestamp (seconds) of the most recent eval completion |
| `bench_image_hash` | string | No | Docker image digest of the `trajrl-bench` sandbox image (e.g. `"sha256:a1b2c3..."`) |
| `harness_image_hash` | string | No | Docker image digest of the `hermes-agent` harness image (e.g. `"sha256:d4e5f6..."`) |
| `bench_version` | string | No | Version string reported by the trajrl-bench CLI inside the sandbox container (e.g. `"v1.2.0"`) |
| `llm_model` | string | No | LLM model identifier the validator is configured to use for evals (e.g. `"gpt-5"`, `"claude-sonnet-4-6"`) |
| `llm_base_url` | string | No | Base URL of the OpenAI-compatible LLM endpoint the validator routes eval calls through (e.g. `"https://api.openai.com/v1"`) |

### Request Example

```json
{
  "hotkey": "5FFApaS75bvpgP9gQ5hTUdZHiTc6LB2VPP9gvHN6VQCNug6f",
  "version": "0.4.14",
  "timestamp": 1710000000,
  "signature": "0xabc123...",
  "last_set_weights_at": 1710000000,
  "last_eval_at": 1709999900,
  "bench_image_hash": "sha256:a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2",
  "harness_image_hash": "sha256:f6e5d4c3b2a1f6e5d4c3b2a1f6e5d4c3b2a1f6e5d4c3b2a1f6e5d4c3b2a1f6e5",
  "bench_version": "v1.2.0",
  "llm_model": "gpt-5",
  "llm_base_url": "https://api.openai.com/v1"
}
```

### Success Response

```json
{
  "success": true,
  "validator": {
    "hotkey": "5FFApaS75bvpgP9gQ5hTUdZHiTc6LB2VPP9gvHN6VQCNug6f",
    "version": "0.4.14",
    "lastSeen": "2025-03-09T10:30:00.000Z",
    "lastSetWeightsAt": "2025-03-09T10:30:00.000Z",
    "lastEvalAt": "2025-03-09T10:28:20.000Z",
    "benchImageHash": "sha256:a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2",
    "harnessImageHash": "sha256:f6e5d4c3b2a1f6e5d4c3b2a1f6e5d4c3b2a1f6e5d4c3b2a1f6e5d4c3b2a1f6e5",
    "benchVersion": "v1.2.0",
    "llmModel": "gpt-5",
    "llmBaseUrl": "https://api.openai.com/v1"
  }
}
```

### Error Responses

| Error | Message |
|-------|---------|
| Invalid request body | `"Request body must be a JSON object"` |
| Invalid hotkey | `"hotkey is required and must be a non-empty string"` |
| Invalid version | `"version is required and must be a non-empty string"` |
| Invalid timestamp | `"timestamp is required (unix seconds)"` |
| Invalid signature | `"signature is required"` |
| Signature verification failed | Verification error message |
| Server error | `"Internal server error"` |

### Processing Flow

1. Validate request body fields
2. Verify validator signature
3. Extract requester IPv4 address (from `x-forwarded-for` or `x-real-ip` headers)
4. Upsert validator heartbeat record to the database (unique key: `hotkey`), updating `version`, `last_seen`, `ip`, `last_set_weights_at`, `last_eval_at`, `bench_image_hash`, `harness_image_hash`, `bench_version`, `llm_model`, and `llm_base_url` (if provided)
5. Return success response

## POST /api/v2/consensus/payload

> **Legacy (v5.x).** Part of the v5.x off-chain CAS + chain-commitment-pointer consensus protocol, retired in v6.0. v6.0 daemons do not call this endpoint; see [v6.0: Challenge Epoch APIs](#v60-challenge-epoch-apis).

Upload a consensus payload to GCS-backed content-addressable storage. Used by validators during the consensus protocol's publish phase to share evaluation results with other validators.

The payload is content-addressed: the server computes a SHA-256 hash of the canonical JSON serialization and uses it as the storage key. Duplicate uploads (same content hash) return HTTP 409 with the existing URL.

### Method
`POST`

### Headers
- `Content-Type`: `application/json`

### Request Body

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `validator_hotkey` | string | Yes | Hotkey of the uploading validator |
| `timestamp` | number | Yes | Unix timestamp in seconds |
| `signature` | string | Yes | Validator's signature over `"trajectoryrl-consensus:{validator_hotkey}:{timestamp}"` |
| `payload` | object | Yes | The consensus payload object (see below) |

#### Payload Object

| Field | Type | Description |
|-------|------|-------------|
| `protocol_version` | number | Consensus protocol version (currently `2`) |
| `window_number` | number | Evaluation window number |
| `validator_hotkey` | string | Hotkey of the validator who produced this evaluation |
| `bench_version` | string | trajrl-bench version string (e.g. `"3.0.1"`) |
| `scores` | object | Miner hotkey → quality score (0.0–1.0) |
| `spec_number` | number | Major version of trajrl-bench (e.g. `3`). **New canonical name**; required for new payloads. |
| `scoring_version` | number | **Legacy alias for `spec_number`**. Accepted for backward compatibility; server reconciles with `spec_number` when both are present. New payloads should send `spec_number` instead. |
| `timestamp` | number | Unix seconds when payload was built |
| `disqualified` | object | Miner hotkey → disqualification reason (e.g. `"pre_eval_rejected:hardcoded"`, `"integrity_failed"`) |

### Request Example

```json
{
  "validator_hotkey": "5FFApaS75bvpgP9gQ5hTUdZHiTc6LB2VPP9gvHN6VQCNug6f",
  "timestamp": 1710000000,
  "signature": "0xabc123...",
  "payload": {
    "protocol_version": 2,
    "window_number": 42,
    "validator_hotkey": "5FFApaS75bvpgP9gQ5hTUdZHiTc6LB2VPP9gvHN6VQCNug6f",
    "bench_version": "3.0.1",
    "scores": {
      "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY": 0.85,
      "5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty": 0.42
    },
    "spec_number": 3,
    "scoring_version": 3,
    "timestamp": 1710000000,
    "disqualified": {
      "5DAAnrj7VHTznn2AWBemMuyBwZWs6FNFjdyVXUeYum3PTXFy": "integrity_failed"
    }
  }
}
```

### Success Response (200 OK / 409 Conflict)

```json
{
  "success": true,
  "content_hash": "sha256:a1b2c3d4e5f6...",
  "url": "https://storage.googleapis.com/trajectoryrl-consensus/sha256_a1b2c3d4e5f6....json",
  "short_url": "https://trajrl.com/c/sha256:a1b2c3d4e5f6..."
}
```

HTTP 409 is returned when the payload already exists (duplicate upload); the response body still contains the existing URL.

### Error Responses

| Status | Message |
|--------|---------|
| 400 | Invalid request body or missing fields |
| 401 | Signature verification failed |
| 500 | Internal server error |

### Processing Flow

1. Validate request body fields (`validator_hotkey`, `timestamp`, `signature`, `payload`)
2. Verify validator signature over `"trajectoryrl-consensus:{validator_hotkey}:{timestamp}"`
3. Serialize payload with canonical JSON (sorted keys, no whitespace)
4. Compute SHA-256 content hash
5. Check if content already exists in GCS — if so, return 409 with existing URL
6. Upload serialized payload to GCS
7. Return 200 with `content_hash`, `url`, and `short_url`


---
## GET /api/v2/validators/epoch_snapshot

> **Legacy (v5.x).** Provided the all-miners-per-epoch evaluation target set used by the v5.x validator. v6.0 evaluates exactly one challenger per epoch via [`GET /api/v2/epoch/current`](#get-apiv2epochcurrent), so v6 daemons do not call this endpoint.

Returns the eval target set for one epoch — the list of (miner_hotkey, pack_hash) tuples a validator should evaluate this epoch, with each tuple's pre-eval verdict baked in. This is the **single source** of the eval target set: validators no longer query chain commitments directly or call `/api/v2/miners/pre-eval` per miner.

The snapshot is **precomputed by the sync worker and frozen** on `epoch_summary.eval_snapshot` — this endpoint is a pure read. The first time the sync cycle runs after an epoch's `cutoff_time` has passed, it computes the snapshot from `miner_submissions` and writes it; from then on every read returns byte-identical bytes. Once written the snapshot is **immutable** — no recomputation, no race against ongoing miner_submissions updates.

If the snapshot for the requested epoch hasn't been built yet (e.g., the cutoff just passed and sync hasn't run), the endpoint returns **404**. Validators should retry on their next cycle (~5 min later).

### Method
`GET` is the canonical method (read-only, idempotent).

`POST` with a JSON body of identical shape is also accepted for backward compatibility with older validator clients — the signed payload is identical, so the same signature works on both.

### Query Parameters (GET)

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `epoch_number` | number | Yes | Epoch (window) number for which to return the snapshot. Non-negative integer. |
| `validator_hotkey` | string | Yes | Hotkey of the calling validator. |
| `timestamp` | number | Yes | Unix timestamp in seconds. |
| `signature` | string | Yes | Validator's signature over `"trajectoryrl-snapshot:{validator_hotkey}:{timestamp}"`. |

> **Signing is required** for this endpoint. Unlike `/api/v2/miners/pre-eval`, the response includes `pack_url` for each entry — for web-source submissions this is a random-key GCS URL that must not leak before the 48 h reveal gate. Mandatory auth keeps that promise.

### Request Example (GET)

```
GET /api/v2/validators/epoch_snapshot?epoch_number=1234&validator_hotkey=5FFA...&timestamp=1714000000&signature=0xabc123...
```

### Request Example (POST, legacy)

```json
{
  "epoch_number": 1234,
  "validator_hotkey": "5FFApaS75bvpgP9gQ5hTUdZHiTc6LB2VPP9gvHN6VQCNug6f",
  "timestamp": 1714000000,
  "signature": "0xabc123..."
}
```

### Success Response

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
      "hotkey": "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY",
      "pack_hash": "abc123def456...",
      "pack_url": "https://storage.googleapis.com/<bucket>/<prefix>/<uid>/<random_key>.json",
      "refresh_time": "2026-05-04T08:00:00.000Z",
      "pre_eval_status": "passed",
      "pre_eval_reason": null
    },
    {
      "uid": 88,
      "hotkey": "5HBE...",
      "pack_hash": "deadbeef...",
      "pack_url": "https://...",
      "refresh_time": "2026-05-04T09:30:00.000Z",
      "pre_eval_status": "failed",
      "pre_eval_reason": "hardcoded"
    }
  ]
}
```

### Top-level Response Fields

| Field | Type | Description |
|-------|------|-------------|
| `epoch_number` | number | Echo of the requested epoch. |
| `built_at` | string | ISO-8601 timestamp of when the sync worker first froze this snapshot. Useful to verify freshness and detect stale builds. |
| `window_start` | number | Chain block at the start of the window: `epoch_number × EVAL_INTERVAL + GLOBAL_ANCHOR`. |
| `cutoff_block` | number | Upper bound (exclusive). `window_start - (EVAL_INTERVAL - T_AGGREGATE)` ≈ `window_start - 720` blocks ≈ 2.4 h before window_start. Rows whose `refresh_time` is at or after this block are excluded. |
| `cutoff_time` | string | `cutoff_block` translated to wall-clock time via `sync_state` reference. |
| `eligible_start_time` | string | Lower bound (inclusive). `cutoff_time - inactivity_window_hours` — rows whose `refresh_time` is older than this are considered abandoned and excluded. |
| `inactivity_window_hours` | number | Width of the eligibility window (currently `48`). Mirrors the validator-side `inactivity_blocks` default. |
| `snapshot_block` | number | Latest synced chain block height when this snapshot was produced. Informational only. |
| `entries` | array | Ordered list of entries (sort: `refresh_time ASC, hotkey ASC`). May be empty. |

### `entries[]` Object Structure

| Field | Type | Description |
|-------|------|-------------|
| `uid` | number | Miner UID at the time the row was last refreshed. |
| `hotkey` | string | Miner hotkey. |
| `pack_hash` | string | SHA-256 of the canonical pack JSON. |
| `pack_url` | string | URL the validator should fetch. For chain-source rows this is the miner's self-hosted URL committed on chain; for web-source rows it's the random-key GCS URL we hosted at submission time. |
| `refresh_time` | string | **Deprecated, retained for wire compatibility.** Server now populates this field from `miner_submissions.submitted_at` (the immutable first-commit timestamp back-derived from the on-chain commit block). Eligibility is no longer driven by a heartbeat; the server filters entries by joining to the live `nodes.commit_block` (see "Cutoff Semantics" and migration 062). Subnet code that only treats this as a monotone time stamp is unaffected. |
| `pre_eval_status` | `"passed"` \| `"failed"` | Pre-eval verdict for this `(miner_hotkey, pack_hash)`. The wire-level value set is intentionally narrow — `"passed"` for any row that cleared the 7-step pre-eval pipeline, `"failed"` for any rejection. Internally `miner_submissions.eval_status` holds a broader 4-state lifecycle (`pending_pre_eval` / `pending_eval` / `completed` / `failed`); the snapshot builder collapses it via `toLegacyPreEvalStatus()` so subnet code only ever sees the legacy 2-value set and requires no changes when the column expands. Rows still in `pending_pre_eval` are filtered out at the SQL layer and absent from `entries`. See `docs/eval-status-lifecycle.md` for the full mapping. |
| `pre_eval_reason` | string \| null | Human-readable failure reason. Populated only when `pre_eval_status = "failed"`; mirrors the row's `eval_reason` column (e.g. `"hardcoded"`, `"hash_mismatch"`, `"banned until 2026-06-01..."`, `"similarity=0.834 match=5HBE…"`). |

### Cutoff Semantics

Eligibility window for epoch N is a **half-open interval** `[eligible_start_time, cutoff_time)`:

```
cutoff_block        = N × EVAL_INTERVAL + GLOBAL_ANCHOR - (EVAL_INTERVAL - T_AGGREGATE)
                    = window_start(N) - 720 blocks       // ≈ 2.4 h
                    = aggregation_start(N-1)
cutoff_time         = block→time(cutoff_block)
eligible_start_time = cutoff_time - inactivity_window_hours       // 48 h
```

**Upper bound (cutoff)**: rows whose `submitted_at` ≥ `cutoff_time` are excluded. The 2.4 h gap from window_start is the contract with the sync worker — every row that lands before cutoff is guaranteed to have its full check pipeline complete before epoch N's eval phase opens. A pack submitted *during* that 2.4 h window is not in epoch N's snapshot but is eligible for epoch N+1.

**Active commitment check (post-062)**: each surviving `miner_submissions` row must JOIN to the current `nodes` snapshot on `commit_block` — i.e. the row corresponds to the miner's *current* on-chain commitment. A miner who switched to a different `pack_hash` falls out immediately (their old `commit_block` no longer matches `nodes.commit_block`); a miner who removed their commitment falls out as soon as the syncer observes the change. The 48 h `eligible_start_time` field is still emitted for legacy compat but no longer drives selection — it is now informational only, since active-commitment matching is exact rather than time-windowed.

The legacy `refresh_time` field on `entries[]` is populated from `submitted_at` (immutable first-commit timestamp). Subnet code that treats it as a monotone time stamp is unaffected. The field is documented as deprecated and may be renamed in a future protocol version.

### Error Responses

| Status | Error | Notes |
|--------|-------|-------|
| 400 | `"epoch_number is required and must be a non-negative integer"` | Missing/invalid epoch_number. |
| 401 | `"validator_hotkey, timestamp, and signature are required"` | Auth fields missing — endpoint requires signing. |
| 403 | `"Hotkey is not a registered validator on-chain"` | validator_hotkey not in metagraph. |
| 403 | Verification error | Signature drift > 5 min or invalid signature. |
| 404 | `"Snapshot for epoch N is not available yet"` | Sync worker hasn't built this epoch yet (cutoff just passed, or sync is behind). Retry on the next cycle. |

### Determinism Guarantees

For a fixed `epoch_number`, the response is deterministic across validators:

1. **Cutoff time** is derived purely from `epoch_number` + chain math + `sync_state.block_height` (which converges across validators within seconds).
2. **`submitted_at < cutoff_time` filter** + **JOIN to `nodes.commit_block`** yields exactly one row per active hotkey.
3. **Sort order** `(refresh_time ASC, hotkey ASC)` — note `refresh_time` is now backed by `submitted_at` — is stable.
4. **Verdict mapping** is deterministic — same row state always produces the same `pre_eval_status` / `pre_eval_reason`.

The only non-determinism source is the per-step check column write order from the sync worker. Once a row's check columns are stable (which they are by `cutoff_time` thanks to the 2.4 h margin), the verdict is fixed.

### Processing Flow

1. Validate `epoch_number`. Validate auth fields are present.
2. Verify `validator_hotkey` is on-chain and the signature is valid.
3. Compute `cutoff_block` from `epoch_number` + chain config.
4. SELECT `eval_snapshot` from `epoch_summary` for the requested epoch.
5. If NULL → 404 with hint to retry. If present → return the JSONB verbatim, stamping `epoch_number` and `built_at` at the top level.

The snapshot itself is built by the sync worker (offline from the request path):

a. Each sync cycle, scan recent epochs whose `cutoff_block ≤ sync_state.block_height` and whose `eval_snapshot IS NULL`.
b. For each, run the active-commitment query: JOIN `miner_submissions` to `nodes` on `(hotkey, commit_block)` WHERE `nodes.is_active = true AND NOT nodes.deregistered`, AND `submitted_at < cutoff_time`, AND `eval_status IN ('pending_eval', 'completed', 'failed')` (i.e. any row past the pending_pre_eval stage). The JOIN restricts to one row per hotkey by construction.
c. Map each surviving row: `pre_eval_status ← toLegacyPreEvalStatus(eval_status)` (collapses `pending_eval`/`completed` → `"passed"`, `failed` → `"failed"`), `pre_eval_reason ← eval_reason` (only for failures), `refresh_time ← submitted_at` (wire-compat alias).
d. Sort entries by `(refresh_time ASC, hotkey ASC)`.
e. UPSERT to `epoch_summary.eval_snapshot` with COALESCE — first write wins, subsequent ticks skip already-built epochs.

### Validator Failure Behavior

This endpoint is the validator's only source for the epoch eval set, so any non-200 / network / timeout / 404 error means the validator should wait and retry on its next cycle iteration. The eval phase spans the first 80% of a window (~19 h of a 24 h window), giving wide retry headroom. There is no client-side fallback to a chain query.

### Caching & Immutability

Snapshots are cached in PostgreSQL (`epoch_summary.eval_snapshot` JSONB) — once frozen, never recomputed. Reads are O(1) (PK lookup by `epoch_number`). Operationally, each epoch's snapshot is written at most once, by whichever sync cycle first observes the epoch's cutoff in the past. Recovery from a corrupt snapshot would require manually clearing the column to allow re-build (out of scope).

---

## v6.0: Challenge Epoch APIs

The endpoints in this section implement the **winner-challenger model**. Sections marked **(validator critical-path)** are the ones a v6 daemon must call. Sections marked **(public read)** are observability-only — websites, dashboards, and operators consume them; the daemon does not.

The v6 design pushes winner derivation to the validator: the server publishes per-vote inputs (with `validator_stake` frozen at score POST) via `GET /api/v2/winner/current`, and each daemon runs its own copy of the consensus algorithm against those inputs. The server's `winner_state` row is a published claim used for cross-checking and observability, not a directive — what binds the network's `set_weights` is the convergent local derivation across honest validators, not the server's `winner_state` row in isolation.

### v6 daemon main loop

There are **two** server-published reads on the daemon hot path, with distinct purposes:

1. **`GET /api/v2/epoch/current`** — the in-progress challenge to evaluate (challenger pack identity, block window, time budget).
2. **`GET /api/v2/winner/current`** — the canonical inputs for the seated winner. The daemon **derives the winner locally** from the per-validator submissions returned here and uses the local result to drive `set_weights`. This is what makes winner state decentralized: every validator independently re-runs the same aggregation against the same server-published votes-with-frozen-stakes, so a server bug, server-side tamper attempt, or split server cannot silently dictate weights.

```
# challenge-eval loop (every ~30 s):
loop forever:
  resp = GET /api/v2/epoch/current
  if 404:                                       // no epoch in progress
    sleep 30s, continue
  if already submitted for resp.epoch.challenge_epoch_id:
    sleep until resp.epoch.end_block, continue
  fetch pack(resp.epoch.challenger_pack_hash), run eval
  POST /api/v2/epoch/{resp.epoch.challenge_epoch_id}/score
  on 409 "epoch is finalized": this epoch is closed; next loop

# winner-derivation loop (every ~30 s, can share the same tick):
  w = GET /api/v2/winner/current
  if w.epoch is null:                           // cold start, no finalized epoch yet
    cache.winner = null
  else:
    derived = local_aggregate(
                w.submissions,                  // votes + frozen validator_stake
                metagraph_at(w.epoch.start_block),  // chain-deterministic
                CONSENSUS_CONFIG)               // versioned in daemon code
    if derived.uid != w.winner?.uid:
      log_alarm("server winner ≠ locally-derived"); proceed with derived
    cache.winner = derived

# set_weights cadence (independent loop, e.g. each Bittensor tempo):
  winner = cache.winner                         // refreshed by winner-derivation loop above
  if winner is null:                            // cold start
    skip set_weights this round
  if cache age > WINNER_FALLBACK_TTL (24 h):    // server been unreachable too long
    refuse set_weights + alarm
  set_weights(winner.uid)
```

The two server endpoints have distinct caching contracts. `/api/v2/epoch/current` answers "what's the active challenge right now?" — its `epoch` block is the only piece daemons act on. `/api/v2/winner/current` answers "what evidence has been finalized into the current seat?" — daemons consume the `submissions[]` array and re-derive locally rather than trusting the `winner` field.

> **Migration note**: the `winner` field returned by `/api/v2/epoch/current` is retained as a non-authoritative convenience for the website and other observability consumers; v6 daemons should ignore it and read seated-winner state through `/api/v2/winner/current` so the local-derivation contract is preserved.

### v6 signature prefixes

Each endpoint signs a different prefix string — signatures are not interchangeable across endpoints, and the v6 score-submit path includes `challenge_epoch_id` to defeat cross-epoch replay.

| Endpoint | Prefix | Signed payload |
|---|---|---|
| `POST /api/v2/epoch/{id}/score` | `trajectoryrl-challenge-score` | `{validator_hotkey}:{timestamp}:{challenge_epoch_id}` |
| `POST /api/v2/validators/heartbeat` | `trajectoryrl-heartbeat` | `{validator_hotkey}:{timestamp}` |
| `POST /api/validators/logs/upload` / `cycle` | `trajectoryrl-logs` | `{validator_hotkey}:{timestamp}` |
| `GET /api/v2/epoch/current` (optional auth, unlocks pack URLs) | `trajectoryrl-epoch-current` | `{validator_hotkey}:{timestamp}` |

The legacy v5.2 path `POST /api/v2/scores/submit` uses `trajectoryrl-submit` and is **not** the v6 score path; v6 daemons must not call it.

---

### `GET /api/v2/epoch/current` (validator critical-path)

Returns the single in-progress challenge epoch. The daemon's main-loop poll target.

#### Method
`GET`

#### Auth
Public read by default. Pack-URL fields (`epoch.challenger_pack_url`, `winner.pack_url`) are gated and only included when **either**:

1. The request is signed by an on-chain validator (see "Optional signed auth" below), **or**
2. The current epoch was opened more than **24 hours** ago (`Date.now() − epoch.created_at > 24h`). After the grace window the live challenger pack is treated as no longer competition-sensitive and the URLs become public.

When neither condition holds, the response omits `epoch.challenger_pack_url` entirely and the `winner` block is returned without `pack_url`. All other fields (hotkey, pack_hash, scores, blocks) are always public.

#### Optional signed auth (query string)

To unlock pack URLs immediately, validators add the following query parameters:

| Param | Type | Description |
|---|---|---|
| `hotkey` | string | Validator hotkey (must be `is_validator = true` in the synced `nodes` table, i.e. on-chain stake ≥ vpermit threshold) |
| `timestamp` | number | Unix seconds; must be within ±5 min of server time |
| `signature` | string | sr25519 signature over `"trajectoryrl-epoch-current:{hotkey}:{timestamp}"` |

Any of: missing param, malformed timestamp, signature verification failure, or non-validator hotkey causes the request to **soft-fall-through** to the time-gate path (no error response — the URLs simply remain hidden until the 24h window expires). The endpoint never returns 401/403 for a malformed signature; this preserves the public-read contract for unsigned callers.

#### Success response (200) — pack URLs hidden (default)

```jsonc
{
  "success": true,
  "epoch": {
    "challenge_epoch_id":   1234,
    "challenger_hotkey":    "5...",
    "challenger_pack_hash": "abc...",
    "start_block":          5300000,
    "end_block":            5300150,
    "epoch_length_blocks":  150,
    "status":               "in_progress",
    "created_at":           "2026-05-09T01:00:00.000Z"
  },
  // Canonical seated winner. Null on cold start (no epoch has finalized yet).
  // `pack_url` is OMITTED in this default response.
  "winner": {
    "hotkey":         "5...",
    "uid":            1,
    "pack_hash":      "def...",
    "score":          "0.92",
    "since_epoch_id": "100"
  },
  // Server's chain-time view at request time. Daemons use this to decide
  // whether an eval can finish in the remaining window. Null on chain RPC
  // failure — daemon should fall back to its own block reading.
  "current_block":    5300075,
  "elapsed_blocks":   75,
  "remaining_blocks": 75
}
```

#### Success response (200) — pack URLs unlocked (validator-signed or >24h)

```jsonc
{
  "success": true,
  "epoch": {
    "challenge_epoch_id":   1234,
    "challenger_hotkey":    "5...",
    "challenger_pack_hash": "abc...",
    "challenger_pack_url":  "https://.../pack.json",   // null if miner_submissions row has none
    "start_block":          5300000,
    "end_block":            5300150,
    "epoch_length_blocks":  150,
    "status":               "in_progress",
    "created_at":           "2026-05-09T01:00:00.000Z"
  },
  "winner": {
    "hotkey":         "5...",
    "uid":            1,
    "pack_hash":      "def...",
    "pack_url":       "https://.../pack.json",
    "score":          "0.92",
    "since_epoch_id": "100"
  },
  "current_block":    5300075,
  "elapsed_blocks":   75,
  "remaining_blocks": 75
}
```

| Field | Meaning |
|---|---|
| `epoch.challenge_epoch_id` | Server-assigned BIGINT. The daemon passes this to `POST /api/v2/epoch/{id}/score`. Distinct from `epoch_number` (chain-tempo label) — the two have different semantics under v6. |
| `epoch.challenger_hotkey` | The miner whose pack is being scored this epoch. Server-stamped — validator does not need to verify it independently. |
| `epoch.challenger_pack_hash` | SHA-256 of the canonical pack JSON. The daemon uses this to fetch and verify the pack before scoring. |
| `epoch.challenger_pack_url` | **Gated field** — only present when pack URLs are unlocked (validator-signed or >24h). Pulled from `miner_submissions.pack_url` for the row backing this epoch's challenger. May be `null` if the miner row has none. |
| `epoch.start_block` / `epoch.end_block` | The chain-block window. The eval window closes at `end_block`; submissions arriving after `finalizeEpoch` runs (≈ next scheduler tick after `end_block`) are rejected with 409. |
| `epoch.epoch_length_blocks` | Convenience: `end_block − start_block`. Tells the daemon "an epoch lasts this many blocks" without subtracting. |
| `epoch.status` | Always `"in_progress"` for rows returned here. (`"finalized"` and `"aborted_quorum"` rows are not returned by this endpoint — fetch them via `GET /api/epoch/{id}`.) |
| `winner` | The canonical seated winner from `winner_state`. `null` on cold start. `pack_url` is **gated** identically to `epoch.challenger_pack_url`; all other fields are always public. The URL, when present, is the URL the consensus agreed on (frozen at finalize time — does not drift if the miner re-uploads). |
| `current_block` | Server-stamped chain block at the moment of this response. `null` if chain RPC is unreachable. |
| `elapsed_blocks` | `current_block − start_block`, clamped at 0. `null` if `current_block` is null. |
| `remaining_blocks` | `end_block − current_block`, clamped at 0. `null` if `current_block` is null. **Use this to decide whether to start an eval** (see "Mid-epoch start" below). Treat as an upper bound — under dynamic epoch (see below), the actual close may fire earlier. |

#### Dynamic Epoch (early finalise)

An epoch can finalise before its `end_block` deadline once every operator-curated whitelist validator has submitted, provided `EPOCH_MIN_BLOCKS` (default 30) blocks have elapsed since `start_block`. The hard deadline still applies as the upper bound. The whitelist source is server-side (`nodes.whitelisted`) and not exposed via this endpoint.

Daemon implications:

- Treat `remaining_blocks * 12s` as an **upper bound** for sleeps, not a guarantee. Cap your poll interval at a daemon-side maximum (e.g. 30 s) so you observe the next epoch within that bound after an early finalise.
- A POST landing on a closed epoch returns `409 epoch is finalized` — handle by re-fetching `/api/v2/epoch/current` immediately and proceeding to the new epoch, rather than waiting for the next scheduled poll.

#### Mid-epoch start

A daemon launched mid-epoch may not have enough remaining time to fetch the pack, run the LLM-as-judge eval, and POST the score before the scheduler finalizes. If you start the eval anyway you'll likely hit `409 epoch is finalized` on submit, get recorded as `participated=false` for that epoch, and (after `INACTIVE_THRESHOLD_EPOCHS=3` consecutive misses) be flagged inactive.

Recommended pattern:

```
resp = GET /api/v2/epoch/current
remaining_secs = (resp.remaining_blocks ?? 0) * 12   // Bittensor block time
if remaining_secs < EXPECTED_EVAL_SECONDS + SAFETY_MARGIN:
  // Not enough time. Skip this epoch, refresh winner cache,
  // sleep until the next epoch is likely open.
  cache.winner = resp.winner
  sleep min(remaining_secs + 30, POLL_INTERVAL)
  continue
// Enough time → fetch pack and eval as normal.
```

`EXPECTED_EVAL_SECONDS` is daemon-side: a conservative initial estimate (e.g. 5 min) and the rolling p95 of recent eval durations once you have data. `SAFETY_MARGIN` (e.g. 60 s) absorbs network/upload variance.

When `current_block` is `null` (server's chain RPC down), the daemon uses its own chain reading; same math.

#### Winner block — non-authoritative convenience

The `winner` block here is **not authoritative for v6 daemons**. It mirrors the server's `winner_state` row and is retained as a convenience for the website and other observability consumers (so a single GET serves both "active challenge" and "current seat" to a casual reader). v6 daemons must instead read seated-winner state through `GET /api/v2/winner/current`, which returns the per-validator votes + frozen `validator_stake` snapshots that drive **local** aggregation on the daemon — the contract that makes winner derivation decentralized rather than server-dictated.

Reading the seat from this endpoint will *usually* match the daemon's locally-derived winner because both sides run the same algorithm against the same data, but the daemon's authority is the local derivation, not the server's claim. See the "v6 daemon main loop" section above for the two-loop polling structure.

#### 404 response

```json
{ "success": false, "error": "no epoch in progress" }
```

Returned when:
- The previous epoch has finalized but the scheduler hasn't yet opened the next one (≤ `SCHEDULER_POLL_INTERVAL_MS` ≈ 30 s gap).
- The challenger queue is empty (no `pending_eval` rows pass the cooldown / attempt filter).

Validators **must treat 404 as a normal state**, sleep, and retry. It is not an error condition.

#### Polling cadence

Recommend 30 s between polls. The scheduler ticks at most every `SCHEDULER_POLL_INTERVAL_MS` (default 30 s), so polling faster yields no fresher data. Each poll returns the same `challenge_epoch_id` for the entire epoch duration (`end_block − start_block` blocks; default 100 ≈ 20 min).

#### Idempotency

Read-only and idempotent. Calling repeatedly returns the same row until either (a) the epoch transitions out of `in_progress`, or (b) a new epoch opens — at which point a fresh `challenge_epoch_id` is returned.

---

### `POST /api/v2/epoch/{challenge_epoch_id}/score` (validator critical-path)

Validator-side score submission for a single challenge epoch. Distinct from the v5.2 `/api/v2/scores/submit` path: the v6 daemon **must** use this endpoint, the v5.2 daemon **must** keep using the old one. The two paths are independent in code, schema, and signature prefix.

The wire contract is built around the **winner-challenger** model and is forward-compatible with **dual-eval**: in v6.0 a validator only evaluates the challenger (`challenger` block required, `winner` block omitted); when dual-eval lands, validators include both blocks in the same submission.

#### Method
`POST`

#### Path parameter
- `challenge_epoch_id` (positive integer) — the id returned by `GET /api/v2/epoch/current`. Server resolves this to a `challenge_epochs` row and stamps the **challenger** identity (`challenger_hotkey`, `challenger_pack_hash`) from it; the validator does **not** supply these.

#### Headers
`Content-Type: application/json`

#### Request body

```jsonc
{
  "validator_hotkey": "5...",
  "timestamp":        1746700000,
  "signature":        "0x...",
  "version":          "v6.0.0",
  "spec_number":      1,

  // REQUIRED — every submission must include the challenger verdict.
  "challenger": {
    "score":             0.91,
    "qualified":         true,
    "rejected":          false,          // optional, default false
    "rejection_detail":  null,           // optional free-text
    "scenario_results":  { ... }         // optional JSONB; per-side audit blob
  },

  // OPTIONAL — present iff the validator ran dual-eval against the
  // currently-seated winner. When present, `pack_hash` is what the
  // validator actually evaluated and is server-validated against
  // `winner_state.winner_pack_hash`.
  "winner": {
    "pack_hash":         "abc...",
    "score":             0.85,
    "qualified":         true,
    "rejected":          false,
    "rejection_detail":  null,
    "scenario_results":  { ... }
  },

  // shared audit
  "llm_base_url":       "http://...",
  "llm_model":          "...",
  "judge_model":        "...",
  "bench_image_hash":   "...",
  "harness_image_hash": "...",
  "bench_version":      "..."
}
```

Required: `validator_hotkey`, `timestamp`, `signature`, `version`, `challenger.score`, `challenger.qualified`.

Rejection (per side): set `rejected=true`, `score=0`, `qualified=false`, optionally `rejection_detail` for free-form context. There is no enumerated rejection-stage anymore — under v6 the validator's pack-integrity work happens server-side before the row reaches `pending_eval`, so a rejection from the validator is "I couldn't run the eval" and a one-line text suffices.

#### Signature

Signed message (Ed25519, hotkey-bound):
```
trajectoryrl-challenge-score:{validator_hotkey}:{timestamp}:{challenge_epoch_id}
```

`challenge_epoch_id` is in the signed payload — a signature for epoch N cannot be replayed against epoch M. Distinct prefix from `trajectoryrl-submit` (v5.2) so v5.2 sigs don't cross-validate. Timestamp must be within 5 minutes of server clock.

#### Server-side validations

| Check | On failure |
|---|---|
| `challenge_epoch_id` path is positive integer | 400 |
| `challenger.score` is finite, in `[0, 100]`; `challenger.qualified` is boolean | 400 |
| `challenger.rejected=true` ⇒ `score=0, qualified=false` | 400 |
| If `winner` present: `winner.pack_hash` non-empty; same shape rules as `challenger` | 400 |
| `validator_hotkey` is on-chain validator | 403 |
| Signature verifies (default ON; skipped only when `VALIDATE_VALIDATOR_IDENTITY=false` for staging/local) | 401 |
| `challenge_epochs(id)` exists | 404 |
| `challenge_epochs.status = 'in_progress'` | 409 — eval window closed |
| If `winner` present: `winner_state.winner_pack_hash` is non-NULL **and** equals `winner.pack_hash` | 409 |
| No prior submission for `(challenge_epoch_id, validator_hotkey)` | 409 — duplicate |

`challenger_hotkey` and `challenger_pack_hash` are **never** read from the request body. Server stamps them from `challenge_epochs(id)`. This eliminates the "validator evaluated the wrong pack but reported the right epoch_id" attack vector. For dual-eval, `winner_hotkey` is similarly server-stamped from `winner_state` (only `winner.pack_hash` is taken from the validator, and only to verify it matches the seated winner).

#### Success response (200)

```jsonc
{
  "success": true,
  "challenge_epoch_id": 1234,
  "validator_hotkey":   "5...",

  "challenger": {
    "hotkey":    "5...",
    "pack_hash": "abc..."
  },

  // null in single-eval; populated when winner block was supplied
  "winner": {
    "hotkey":    "5...",
    "pack_hash": "abc..."
  },

  "block_height": 5300000
}
```

`block_height` is server-stamped from chain RPC at receipt; `null` if chain is unavailable.

#### Storage

Inserts into `challenge_scores` (migration 067; `validator_stake` added in 070). One row per `(challenge_epoch_id, validator_hotkey)` enforced by `UNIQUE`. Columns split into `challenger_*` (always populated, including `challenger_scenario_results` JSONB) and `winner_*` (NULL in single-eval, populated in dual-eval, including `winner_scenario_results` JSONB; CHECK enforces all-or-nothing on the four winner essentials). The `score_submit_log` table is never written by this endpoint.

In addition to the validator-supplied payload, the server stamps:
- `block_height` — chain block at receipt (null if chain RPC is unreachable).
- `validator_stake` — snapshot of `nodes.stake` for the submitting validator at receipt. Frozen on the row so finalize-pipeline aggregation can be replayed deterministically without re-reading the live `nodes` table (which `sync-metagraph` rewrites continuously). Null only if the validator has no row in `nodes` at submit time, which is the same graceful-degradation fallback the on-chain validator check uses.

#### Aggregation

At `end_block`, scheduler tick triggers `finalizeEpoch` which `SELECT`s `challenger_*` columns from `challenge_scores` for this epoch, applies the filter pipeline (drop rejected, below-min-stake, inactive, duplicate-hotkey), runs stake-weighted aggregation, then either updates `winner_state` (winner replaced/held) or marks the epoch `aborted_quorum`. Late submissions arriving after `finalizeEpoch` runs are rejected at the API layer (409 from the `in_progress` check). Dual-eval aggregation (using `winner_*` columns to derive a fresh per-epoch winner-score baseline) is a future finalize.ts change — the schema is already shaped for it.

The current `finalizeEpoch` still loads the stake snapshot from `nodes` at finalize time (see comment in `loadStakeSnapshotAt`). The per-row `validator_stake` is captured to make audit / replay deterministic now; switching the live aggregator over to the captured per-row stake is a separate follow-up.

---

### `GET /api/queue` (public read)

FIFO challenger queue snapshot. Returned in the order the scheduler will pick from. **Validators do not call this** — observability for websites and ops only.

```json
{
  "success": true,
  "queue": [
    {
      "submission_id": 12345,
      "miner_hotkey": "5...",
      "miner_uid": 7,
      "pack_hash": "abc...",
      "pack_url": "https://...",
      "submitted_at": "2026-05-07T12:00:00Z",
      "eligible_now": true
    }
  ]
}
```

`eligible_now=false` rows are still in the queue but currently filtered out by the per-miner cooldown — a miner_hotkey that was challenger in any `challenge_epoch` within the last `MINER_COOLDOWN_HOURS` (default 12 h) is held back from being picked again. There is no retry mechanism for quorum-aborted submissions: a single `aborted_quorum` finalize moves the row to `eval_status='exhausted'`, dropping it from the queue. The miner can submit a new `pack_hash` to re-enter.

### `GET /api/epoch/{id}` (public read)

Full audit view of a specific challenge epoch: the `challenge_epochs` row plus every `challenge_scores` submission tied to it. Used by ops to audit aggregation outcomes, by the website for the per-epoch detail page, and by external auditors to replay a finalize decision.

```jsonc
{
  "success": true,
  "epoch": { /* full challenge_epochs row, including outcome */ },
  "submissions": [
    {
      "validator_hotkey": "5...",
      "challenger": {
        "hotkey": "5...", "pack_hash": "abc...",
        "score": 0.91, "qualified": true,
        "rejected": false, "rejection_detail": null,
        "scenario_results": { /* JSONB */ }
      },
      "winner": null,           // populated under dual-eval
      "submitted_at": "2026-05-09T01:30:00.000Z"
    }
  ]
}
```

Late submissions (rows that arrived after finalize ran) appear here too — they're stored but excluded from aggregation; they show up in this list as data debris. **Validators do not call this.**

### `GET /api/winner/history?limit=N` (public read)

Recent winner transitions from `winner_history` (default `limit=20`, max `200`). One row per epoch where the seated winner was reaffirmed or replaced. Used by the website's leaderboard timeline. **Validators do not call this.**

### `GET /api/validator/{hotkey}/activity?limit=N` (public read)

Recent `validator_activity` rows for a specific validator (default `limit=20`, max `200`). Each row is a per-epoch participation record (`participated: true | false`). Useful for ops to spot validators trending toward `INACTIVE_THRESHOLD_EPOCHS` exclusion. **Validators do not call this** — own-state introspection should come from the daemon's local logs.

### `GET /api/v2/winner/current` (validator critical-path)

The canonical source of seated-winner data on the v6 daemon hot path. Returns the **raw inputs** the daemon needs to derive the winner **locally**: the latest finalized epoch's metadata, every participating validator's vote, and the per-row `validator_stake` snapshot that froze each vote's weight at score-submission time. Validators apply their own copy of the consensus algorithm to these inputs and use the locally-derived winner to drive `set_weights` — the server's published `winner` field is shown for cross-checking but is not authoritative.

This is what makes winner state **decentralized in v6**: the server's role narrows to publishing immutable per-vote inputs (with stake frozen at submit), not dictating the verdict. A buggy or compromised server cannot silently change weights as long as enough independent validators run their own derivation.

#### Method
`GET`

#### Auth
None — public read.

#### Success response (200)

```jsonc
{
  "success": true,

  // The server's claim of the currently seated winner — derived from the same
  // submissions[] below by the server's finalize.ts. Returned for cross-check
  // only; daemons MUST treat this as advisory and use their own locally-derived
  // result for set_weights. Null on cold start.
  "winner": {
    "hotkey":         "5...",
    "uid":            1,
    "pack_hash":      "def...",
    "pack_url":       "https://.../pack.json",
    "score":          0.92,
    "since_epoch_id": 100
  },

  // The latest `status='finalized'` epoch — the round whose votes are below.
  // Aborted-quorum epochs are intentionally skipped (no verdict produced).
  // Null until the first finalized epoch exists.
  //
  // Named `finalized_epoch` rather than `epoch` to disambiguate from
  // `/api/v2/epoch/current.epoch`, which is the *in-progress* challenge
  // (votes still being collected, no verdict yet). The two endpoints
  // describe two different epoch tenses; the field name says which.
  "finalized_epoch": {
    "challenge_epoch_id":   201,
    "challenger_hotkey":    "5...",
    "challenger_pack_hash": "abc...",
    "challenger_pack_url":  "https://.../pack.json",   // null if miner_submissions row has none
    "outcome":              "winner_held",             // "winner_held" | "winner_replaced"
    "winner_replaced":      false,                     // boolean projection of outcome — true iff the challenger took the seat. Carried explicitly so daemons and observability consumers don't have to string-compare on `outcome`.
    "finalized_at":         "2026-05-09T01:30:00.000Z"
  },

  // The per-validator votes that produced the verdict. Ordered by descending
  // stake-at-receipt. Each row carries both pack hashes so it is self-contained
  // for audit (the row asserts "this validator scored these specific packs"):
  //   - `challenger_pack_hash` is server-stamped from challenge_epochs(id) at
  //     score POST and equals the top-level `finalized_epoch.challenger_pack_hash`.
  //     Mismatch on any row would indicate a server-side stamping bug.
  //   - `winner_pack_hash` is null in v6.0 single-eval. Under dual-eval it is
  //     server-validated at POST against winner_state.winner_pack_hash, so
  //     once dual-eval lands every populated row will agree on this value too.
  "submissions": [
    {
      "validator_hotkey":     "5...",
      "validator_stake":      18234.5,    // snapshot of nodes.stake at score POST; null for rows pre-migration-070
      "challenger_pack_hash": "abc...",
      "challenger_score":     91.2,
      "challenger_qualified": true,
      "challenger_rejected":  false,
      "winner_pack_hash":     null,       // dual-eval only
      "winner_score":         null,
      "winner_qualified":     null,
      "winner_rejected":      null
    }
  ]
}
```

#### Local-derivation contract for v6 daemons

The daemon does not trust `response.winner`. It runs its own copy of `aggregate(submissions, stakeSnapshot, config)` (the same algorithm the server runs in `finalize.ts`) against:

1. **`response.submissions`** — the votes-with-frozen-stakes returned here. The `validator_stake` field is the authoritative per-row weight; the daemon must not substitute the live metagraph value for that validator (it would have drifted between score POST and now).
2. **The on-chain metagraph at the relevant block** — for global denominators (total active stake, eligibility, deregistration, inactive-flag tracking), each daemon reads its own deterministic metagraph view. Because Bittensor blocks are deterministic across validators, every honest daemon sees the same metagraph and reaches the same denominators.
3. **The consensus config** — `MIN_VALIDATOR_STAKE`, `QUORUM_FRACTION`, `WINNER_PROTECTION_DELTA`, etc. — pinned in the daemon's release; not fetched from the server.

If `derived.uid != response.winner.uid`, the daemon proceeds with `derived` and emits a divergence alarm. Persistent divergence across many validators is a server-bug or attack signal, not a network-consensus problem — the locally-derived winners on each validator stay convergent regardless.

> **Known doc gap (follow-up)**: validators need `finalized_epoch.start_block` to align their metagraph snapshot with the epoch the votes were cast for. The current response only carries `finalized_at`. Adding `start_block`/`end_block` to the `finalized_epoch` block is a small additive change tracked separately; for now daemons can resolve `challenge_epoch_id` via `GET /api/epoch/{id}`.

#### Stake-weight reproducibility

`submissions[].validator_stake` is the per-row snapshot of `nodes.stake` taken at score-submission time and stored on `challenge_scores` (migration 070). Because `nodes.stake` is rewritten continuously by `sync-metagraph`, replaying aggregation against the live `nodes` table will not reproduce the original weights — the captured per-row stake is what guarantees every validator's local derivation reaches the same numbers. Pre-migration-070 rows have `validator_stake = null`; until those age out of the latest finalized epoch, daemons should fall back to their own metagraph stake for that hotkey at the epoch's `start_block` (and log the fallback for telemetry).

#### Cold-start behaviour

| Pre-condition | Response |
|---|---|
| `winner_state` empty AND no finalized epoch | `winner: null`, `finalized_epoch: null`, `submissions: []`. Daemons skip `set_weights` until the first finalized epoch produces a verdict. |
| `winner_state` empty AND a finalized epoch exists (rare; should not happen in steady state) | `winner: null`, `finalized_epoch: {...}`, `submissions: [...]`. Daemon still derives locally; if its derivation reaches a winner the server hasn't seated, that's a server-side staleness signal (alarm). |
| Steady state | All three populated; daemon derives, cross-checks, drives `set_weights` from local result. |

#### Polling cadence

Recommend the same ~30 s tick as `/api/v2/epoch/current` — the two reads can share one timer. Local derivation is cheap; the daemon should re-derive on every successful poll, not cache the locally-derived winner across polls (otherwise a later finalize-time fix on the server would not propagate). Cache the *raw response* if you need network-failure tolerance, with `WINNER_FALLBACK_TTL` (24 h) as the staleness alarm threshold.
