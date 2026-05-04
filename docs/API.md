# API Documentation

This document covers the **validator and miner integration** endpoints (miner pack submission, score submit, log upload, heartbeat, consensus payload, epoch snapshot). Website-only monitoring endpoints (e.g. epoch summary / leaderboard) live in [docs/web-api-epoch-summary.md](docs/web-api-epoch-summary.md).


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
  "cooldown_seconds": 3600,
  "pre_eval_status": "pending"
}
```

| Field | Meaning |
|-------|---------|
| `pack_hash` | Echoed back (lowercased, normalized). |
| `pack_url` | GCS URL of the stored pack, hosted under `<S3_PREFIX>/<uid>/<random_key>.json`. **Returned only to the submitting miner** — this URL is intentionally not exposed by any other public endpoint, so leak risk is bounded to the miner's own response. The miner uses this URL to verify upload contents and commits `<pack_hash>\|<pack_url>` on-chain so validators can pick it up via the existing chain-sync flow. |
| `submission_id` | Row id in `miner_submissions`. |
| `next_upload_allowed_at` | When the per-miner cooldown lifts. |
| `cooldown_seconds` | Configured cooldown window (default 3600). |
| `pre_eval_status` | Always `"pending"` at submit time — pre-eval runs asynchronously after this response is sent. Poll `/api/v2/miners/pre-eval` to check whether it ended up `passed` or `failed`. |

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

- **Per-miner**, **strict counting**: 1 successful submission per `cooldown_seconds` (default 3600 — overridable via `MINER_SUBMIT_COOLDOWN_SECONDS`), regardless of whether pre-eval ultimately passes or fails.
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


## POST /api/v2/scores/submit

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
  "bench_version": "v1.2.0"
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
    "benchVersion": "v1.2.0"
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
4. Upsert validator heartbeat record to the database (unique key: `hotkey`), updating `version`, `last_seen`, `ip`, `last_set_weights_at`, `last_eval_at`, `bench_image_hash`, `harness_image_hash`, and `bench_version` (if provided)
5. Return success response

## POST /api/v2/consensus/payload

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
## POST /api/v2/validators/epoch_snapshot

Returns the eval target set for one epoch — the list of (miner_hotkey, pack_hash) tuples a validator should evaluate this epoch, with each tuple's pre-eval verdict baked in. This is the **single source** of the eval target set: validators no longer query chain commitments directly or call `/api/v2/miners/pre-eval` per miner.

The snapshot is **precomputed by the sync worker and frozen** on `epoch_summary.eval_snapshot` — this endpoint is a pure read. The first time the sync cycle runs after an epoch's `cutoff_time` has passed, it computes the snapshot from `miner_submissions` and writes it; from then on every read returns byte-identical bytes. Once written the snapshot is **immutable** — no recomputation, no race against ongoing miner_submissions updates.

If the snapshot for the requested epoch hasn't been built yet (e.g., the cutoff just passed and sync hasn't run), the endpoint returns **404**. Validators should retry on their next cycle (~5 min later).

### Method
`POST`

### Headers
- `Content-Type`: `application/json`

### Request Body

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `epoch_number` | number | Yes | Epoch (window) number for which to return the snapshot. Non-negative integer. |
| `validator_hotkey` | string | Yes | Hotkey of the calling validator. |
| `timestamp` | number | Yes | Unix timestamp in seconds. |
| `signature` | string | Yes | Validator's signature over `"trajectoryrl-snapshot:{validator_hotkey}:{timestamp}"`. |

> **Signing is required** for this endpoint. Unlike `/api/v2/miners/pre-eval`, the response includes `pack_url` for each entry — for web-source submissions this is a random-key GCS URL that must not leak before the 48 h reveal gate. Mandatory auth keeps that promise.

### Request Example

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
| `refresh_time` | string | ISO-8601 timestamp of the most recent commitment refresh — set whenever sync sees the chain commitment OR the miner re-calls `/api/v2/miners/submit`. |
| `pre_eval_status` | `"passed"` \| `"failed"` | Direct mirror of the row's `eval_status` column on `miner_submissions`. Rows whose `eval_status` is still `'pending'` are filtered out at the SQL layer and absent from `entries`. Reading the summary status (instead of the 7 per-step columns) keeps the endpoint backward-compatible with rows evaluated by the legacy pipeline. |
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

**Upper bound (cutoff)**: rows whose `refresh_time` ≥ `cutoff_time` are excluded. The 2.4 h gap from window_start is the contract with the sync worker — every row that lands before cutoff is guaranteed to have its full check pipeline complete before epoch N's eval phase opens. A pack submitted *during* that 2.4 h window is not in epoch N's snapshot but is eligible for epoch N+1.

**Lower bound (eligible_start_time)**: rows whose `refresh_time` is older than 48 h before cutoff are considered abandoned and excluded. With `sync-metagraph` bumping `refresh_time` every 5 min for any commitment still on chain, only genuinely inactive rows (web miner who stopped calling submit, chain miner who removed their commitment) age out of the window.

For each surviving miner, **DISTINCT ON (miner_hotkey)** picks the row with the latest `refresh_time` in the window. Then entries are sorted by `(refresh_time ASC, hotkey ASC)` — earliest active miner first.

`refresh_time` is bumped on every upsert touch — both `sync-metagraph` re-seeing a chain commitment each cycle and `/api/v2/miners/submit` being re-called for the same `(hotkey, pack_hash)`. A miner who stops refreshing naturally falls out of the eval set.

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
2. **`refresh_time < cutoff` filter** + **latest-per-hotkey** grouping yields the same row set.
3. **Sort order** `(refresh_time ASC, hotkey ASC)` is stable.
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
b. For each, run the latest-per-miner query: `eligible_start_time <= refresh_time < cutoff_time` AND `eval_status IN ('passed', 'failed')`.
c. Map each surviving row: `pre_eval_status ← eval_status`, `pre_eval_reason ← eval_reason` (only for failures).
d. Sort entries by `(refresh_time ASC, hotkey ASC)`.
e. UPSERT to `epoch_summary.eval_snapshot` with COALESCE — first write wins, subsequent ticks skip already-built epochs.

### Validator Failure Behavior

This endpoint is the validator's only source for the epoch eval set, so any non-200 / network / timeout / 404 error means the validator should wait and retry on its next cycle iteration. The eval phase spans the first 80% of a window (~19 h of a 24 h window), giving wide retry headroom. There is no client-side fallback to a chain query.

### Caching & Immutability

Snapshots are cached in PostgreSQL (`epoch_summary.eval_snapshot` JSONB) — once frozen, never recomputed. Reads are O(1) (PK lookup by `epoch_number`). Operationally, each epoch's snapshot is written at most once, by whichever sync cycle first observes the epoch's cutoff in the past. Recovery from a corrupt snapshot would require manually clearing the column to allow re-build (out of scope).

---
