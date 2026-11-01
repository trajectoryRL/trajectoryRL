# Miner Operations Guide

**Subnet**: SN11 (TrajectoryRL)

> New public web submissions pause from challenge epoch **4650 / SPEC 28**.
> Building and validating packs locally remain available.

> Live eval mechanics, active scenarios, and mission framing: <https://trajrl.com/rules>.
> For mechanism design (consensus, winner selection), see [INCENTIVE_MECHANISM.md](INCENTIVE_MECHANISM.md).

---

## What Is Mining on TrajectoryRL?

Mining means shipping a **fusion policy** next to your `SKILL.md`: a program that decides, per request, which open-weight models on Engy answer the agent and in what combination (see [FUSION_POLICY.md](FUSION_POLICY.md)). Scenarios span coding, sysadmin, file ops and debugging, mostly adapted from [Terminal-Bench](https://github.com/laude-institute/terminal-bench). You're not running GPU workloads or a long-running daemon.

For each scenario, validators run the Hermes agent in a fresh container with your `SKILL.md` and the scenario's `INSTRUCTION.md`, talking to your policy in a sidecar; the policy's model calls go through the validator's meter. The agent produces a deliverable file. A separate verifier container runs `pytest` against the deliverable and emits `passed/total` from a continuous CTRF report. Your pack's score is `Σ passed_i / total_i` across all active scenarios — range `[0, N]` for `N` scenarios.

The miner CLI (`trajectoryrl-miner`) is a **toolbox**: independent commands you compose however you want. Write your SKILL.md and your policy (manually, with an LLM, with your own automation), then use the CLI to build and submit via the web endpoint.

---

## Prerequisites

| Requirement | Details |
|-------------|---------|
| **Bittensor wallet** | `btcli wallet create --wallet.name miner --wallet.hotkey default` |
| **Registration** | `btcli subnet register --netuid 11 --wallet.name miner` (dynamic cost) |
| **Python** | 3.10+ |

---

## Quick Start

Submission is web-only — two steps: build, then web-submit (the platform handles queue admission; no on-chain commit needed).

```bash
git clone https://github.com/trajectoryRL/trajectoryRL.git
cd trajectoryRL && pip install -e .

# 1. Write your SKILL.md (use any method — manual, LLM, your own scripts)
vim SKILL.md

# 2. Build pack
trajectoryrl-miner build SKILL.md -o pack.json

# Season 2: add your fusion policy directory (policy.py or policy.json + helper files)
trajectoryrl-miner build SKILL.md --policy ./my_policy -o pack.json

# 3. Submit to a service accepting new submissions.
#    The CLI signs the HTTP request with your hotkey.
#    It does not burn or recycle alpha.
trajectoryrl-miner web-submit pack.json
# → prints pack_url + cooldown info
```

Then check status:

```bash
trajectoryrl-miner status
```

That's it. Repeat whenever you improve your pack.

> The CLI is also runnable as `python neurons/miner.py <command>` if you'd rather not install the entry point. Behaviour is identical.

---

## Web Submit — the sole submission channel

`web-submit` is the **only** way to enter the challenger queue. On-chain `set_commitment` is **no longer ingested as a submission** (`CHAIN_SUBMIT_INGEST_ENABLED=false`).

```bash
trajectoryrl-miner build SKILL.md -o pack.json
trajectoryrl-miner web-submit pack.json
# → Response: { pack_url: "https://storage.googleapis.com/.../pack.json", ... }
```

The `web-submit` command:
1. Validates and hashes your pack locally.
2. POSTs the pack content to `POST /api/v2/miners/submit`, signed with your hotkey.
3. Reports the server response. On acceptance, the platform handles pre-eval and queue admission without a follow-up `set_commitment`.

The sample CLI performs no alpha burn/recycle and sends no fee receipt. Legacy
`SUBMISSION_FEE_ALPHA` settings are ignored. A service requiring fee receipts
will reject this sample request; it will not trigger a payment.

The platform queues the pack for the next eligible challenger epoch on its own. Pack URLs are not exposed in any other API for **48 hours** (the "reveal gate"), so a competitor can't scrape the dashboard to harvest your fresh pack.

Full request/response spec for `/api/v2/miners/submit` is in [`trajectoryrl.web/API.md`](https://github.com/trajectoryRL/trajectoryrl.web/blob/main/API.md).

## Submission Availability

Public web submissions pause from **SPEC 28 / challenge epoch 4650**. The
server returns **HTTP 503** while submissions are closed; the CLI reports
failure without making any on-chain payment. Existing queued evaluations and
validator score reporting continue. Use `--api-base-url` with a compatible
staging or self-hosted service for submission development.

---

## CLI Reference

```bash
trajectoryrl-miner build       <skill_md_path> [--policy DIR] [-o pack.json]
trajectoryrl-miner validate    <pack.json>
trajectoryrl-miner web-submit  <pack.json> [--api-base-url ...]
trajectoryrl-miner status
trajectoryrl-miner upload      <pack.json> [--bucket ...] [--endpoint-url ...]  # legacy — upload only, not ingested as submission
trajectoryrl-miner submit      <pack_url>                                        # legacy — on-chain commit, no longer ingested as submission
```

### build

Build a pack from a SKILL.md file and, with `--policy DIR`, the policy files in that directory (`policy.py` or `policy.json` plus helper text files; `.pyc` and `__pycache__` skipped, binary files rejected).

```bash
trajectoryrl-miner build ./SKILL.md -o pack.json
```

Example output from a service accepting submissions:
```
Pack built: pack.json
  Hash:  a3f8c2...
  Size:  4521 bytes (limit: 32768)
  Valid: PASSED
```

Automatically validates schema + size. Fails if SKILL.md is empty or pack exceeds 32 KB.

### validate

Validate an existing pack.json locally (without submitting).

```bash
trajectoryrl-miner validate pack.json
```

### web-submit

POSTs the pack to `POST /api/v2/miners/submit`, signed with your hotkey. The
sample performs no alpha burn/recycle and does not access your coldkey. When
submissions are enabled and accepted, the server stores your pack, creates the
submission row and starts pre-eval. No S3 / R2 / bucket configuration is needed.
Public submissions pause at epoch 4650; HTTP 503 is reported as a failure.

This command is **self-contained** — you do **not** need to follow up with anything. The platform queues the pack for evaluation on its own.

```bash
trajectoryrl-miner web-submit pack.json
trajectoryrl-miner web-submit pack.json --api-base-url https://stage.trajrl.com
```

Example output from a service accepting submissions:

```
Pack hash: a3f8c2...
Endpoint: https://trajrl.com/api/v2/miners/submit
Submitting pack content to web service...
Submitted! pack_url = https://storage.googleapis.com/<bucket>/<prefix>/<uid>/<random_key>.json
  submission_id:    12345
  cooldown_seconds: 1200
  next_upload_at:   2026-05-04T14:20:00.000Z
  pre_eval_status:  pending

Done — the platform will pre-eval and admit your pack to the
challenger queue automatically. No on-chain commit required.
```

**Cooldown**: The endpoint enforces a per-miner cooldown (default **~20 min**, set server-side via `MINER_SUBMIT_COOLDOWN_SECONDS=1200`). Submitting again before `next_upload_allowed_at` returns HTTP 429 and the CLI prints the cooldown info to stderr. Field-validation, signature, and GCS-upload errors do **not** consume the cooldown — only successfully persisted submissions do.

**Idempotency**: Resubmitting the **same** `(hotkey, pack_hash)` after the cooldown lifts is safe — the server reuses the existing `pack_url` (no re-upload), so the URL you got the first time stays valid.

**Hash & signing**: The CLI canonicalises the pack as `json.dumps(pack, sort_keys=True)` (matches the validator's hash convention) and signs `trajectoryrl-miner-submit:{hotkey}:{timestamp}` with sr25519. The server recomputes the hash from `pack_content` and rejects on mismatch.

> **`pack_url` is sensitive until the 48 h reveal gate.** The URL is intentionally unguessable so competitors can't enumerate packs by hotkey/uid/hash. Don't paste it into Discord, public dashboards, or shared logs — leaking the URL lets other miners hash-match and submit a copy of your pack.

### upload (legacy — no longer ingested as a submission)

Upload pack.json to S3-compatible storage. Prints the public URL.

```bash
trajectoryrl-miner upload pack.json
trajectoryrl-miner upload pack.json --bucket my-bucket --endpoint-url https://storage.googleapis.com
```

Reads S3 config from environment or CLI flags. Returns the public URL. **This command uploads only — it does not submit to the challenger queue.** Use `web-submit` instead.

### submit (legacy — no longer ingested as a submission)

Submit an on-chain commitment pointing to a hosted pack via `set_commitment`.

```bash
trajectoryrl-miner submit https://your-bucket.s3.amazonaws.com/pack.json
```

Fetches the pack from the URL, verifies the hash, and calls `set_commitment` on-chain. **On-chain commitments are no longer ingested as submissions** (`CHAIN_SUBMIT_INGEST_ENABLED=false`). This command still exists in the CLI but will not queue your pack for evaluation. Use `web-submit` instead.

### status

Show your current on-chain commitment.

```bash
trajectoryrl-miner status
```

Example output from a service accepting submissions:
```
On-chain commitment:
  Pack hash: a3f8c2...
  Pack URL:  https://your-bucket.s3.amazonaws.com/pack.json
```

---

## Configuration

The CLI reads wallet and storage config from environment variables. Create a `.env.miner` or export them directly.

### Wallet & Network

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `WALLET_NAME` | yes | `miner` | Bittensor wallet name |
| `WALLET_HOTKEY` | yes | `default` | Bittensor hotkey |
| `NETUID` | yes | `11` | Subnet ID |
| `NETWORK` | yes | `finney` | Bittensor network |

### Web Submit (for `web-submit` command)

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `TRAJECTORYRL_API_BASE_URL` | no | `https://trajrl.com` | Override the web service base URL (for staging / self-hosted). The CLI also accepts `--api-base-url` per-invocation. |

No HMAC keys / bucket credentials needed — the request is signed with the miner's bittensor hotkey. The sample does not make any on-chain payment.

### S3-Compatible Storage (for legacy `upload` command only)

> **Legacy** — these variables are only used by the `upload` CLI command. The `upload` command uploads a pack to your own bucket but does **not** submit it to the challenger queue. Use `web-submit` instead.

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `S3_BUCKET` | for upload | — | S3-compatible bucket name |
| `S3_ENDPOINT_URL` | no | — | Custom endpoint for GCS / R2 / MinIO |
| `S3_REGION` | no | `us-east-1` | Bucket region |
| `AWS_ACCESS_KEY_ID` | for upload | — | S3 / GCS HMAC access key |
| `AWS_SECRET_ACCESS_KEY` | for upload | — | S3 / GCS HMAC secret key |

| Service | `S3_ENDPOINT_URL` | Credentials |
|---------|-------------------|-------------|
| **AWS S3** | _(leave empty)_ | `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` |
| **Google Cloud Storage** | `https://storage.googleapis.com` | GCS HMAC keys |
| **Cloudflare R2** | `https://<account>.r2.cloudflarestorage.com` | R2 API tokens |
| **MinIO** | `https://your-minio-host:9000` | MinIO access/secret keys |

---

## Pack Format

```json
{
  "schema_version": 1,
  "files": {
    "SKILL.md": "# Your skill content here..."
  }
}
```

- `SKILL.md` required, must not be empty
- Total pack JSON ≤ 32 KB
- Content-addressed via SHA256: `sha256(json.dumps(pack, sort_keys=True))`

For SKILL.md authoring patterns and what makes a pack score well, the canonical reference is the live eval mechanics page: <https://trajrl.com/rules>. The active scenario list, per-scenario test counts, and the scoring formula all live there.

---

## Typical Workflow

```
┌─────────────────────────────────────────────────────────┐
│  You (human or your own automation)                     │
│                                                         │
│  1. Write / iterate on SKILL.md                         │
│  2. build → pack.json                                   │
│  3. web-submit → hotkey-signed HTTP request              │
│                  (when submissions are enabled)         │
│  4. Wait for validator evaluation (one epoch)           │
│  5. Check results, iterate                              │
└─────────────────────────────────────────────────────────┘
```

Validators only re-evaluate when your `pack_hash` changes. No need to resubmit if your SKILL.md hasn't changed — once your pack is in the queue via `web-submit`, it carries forward into every future epoch's eval set automatically. Eligibility is driven by the latest `miner_submissions` row for your hotkey, not by an off-chain heartbeat.

### Submission timing — the freeze gap

The eval set for an epoch is **frozen ahead of the epoch's start block** at `cutoff_time`, after which a new web-submit is no longer eligible for that epoch's snapshot. Submissions that land inside the freeze window get evaluated in the **next** epoch instead.

```
... ← previous epoch ──┬───── freeze gap ─────┐ epoch N starts ... eval runs ...
                       ↑                      ↑
                    cutoff for               window_start(N)
                    epoch N's snapshot
```

Practical rules of thumb:

- **Want to be evaluated this epoch?** Get your `web-submit` confirmed at least a few hours before the next epoch boundary. Anything later goes into the epoch after.
- **Pack already queued, no `pack_hash` change?** You don't need to do anything — your submission carries forward automatically.
- **Switched packs late?** Pack swaps that miss the cutoff don't dodge anything; they simply land in the next snapshot. Validators evaluate the pack you had at cutoff, not the one you swap in mid-epoch.

For the full window math (`cutoff_time`, `window_start`, `commit_block` semantics), see [INCENTIVE_MECHANISM.md § Snapshot eligibility window](INCENTIVE_MECHANISM.md#snapshot-eligibility-window).

---

## Local Testing

Before submitting, run the full evaluation harness against your SKILL.md locally. This is the same harness the production validator runs.

```bash
git clone https://github.com/trajectoryRL/trajectoryRL.git
cd trajectoryRL && pip install -e .

cp .env.validator.example .env.validator
# Edit .env.validator: set LLM_API_KEY (default: qwen3.8-27b via api.engy.ai)

# Evaluate a SKILL.md directly
python scripts/eval_pack.py --skill-md path/to/SKILL.md

# Or evaluate a built pack.json
python scripts/eval_pack.py --pack pack.json --json results.json -o ./eval_output
```

The script auto-pulls the `sandbox-agent` base image and every per-scenario image in `SANDBOX_SCENARIOS` on first invocation. Output: per-scenario `passed/total` quality, per-scenario cost, final `Σ` score in `[0, N]`. Use `--json` to capture machine-readable results, `-o ./eval_output` to save full transcripts and verifier artifacts.

Prereqs: Docker daemon running, an LLM API key (e.g. OpenRouter), ~10 GB free disk for the per-scenario images on first run.

For scenario-level definitions (test files, fixture data, verifier code), see the [trajrl-bench](https://github.com/trajectoryRL/trajrl-bench) repo.

---

## Viewing Evaluation Results

Validators upload per-miner eval logs after each epoch. Each log archive contains:

```
SKILL.md                                      # the pack content that was evaluated
metadata.json                                 # final_score, mean_quality,
                                              # scenario_qualities (per-scenario passed/total),
                                              # scenario_costs_usd, total_cost_usd, pack_hash
episodes/
  scenario_<name>/                            # one directory per scenario
    testee_transcript.txt                     # agent's session output
    turns.jsonl                               # full hermes turn log
    evaluation.json                           # verifier output (passed/total + ctrf payload)
    ctrf.json                                 # standalone CTRF report from the verifier
```

Retrieve eval logs via the `trajrl` CLI:

```bash
pip install trajrl
trajrl logs --miner <hotkey> --limit 5     # list recent evals
trajrl logs --eval-id <id> --show          # pretty-print summary
trajrl logs --eval-id <id> --dump-to ./    # extract full archive
```

The dashboard surfaces the same data interactively:

- **Per-pack breakdown**: `https://trajrl.com/miner/<hotkey>/pack/<pack_hash>`
- **Live eval streaming**: `https://trajrl.com/live`
- **Active scenarios + scoring formula**: `https://trajrl.com/rules`

Use the `testee_transcript.txt` + the per-scenario `ctrf.json` together to debug failure modes and iterate on your SKILL.md.

---

## References

- **Live eval mechanics**: <https://trajrl.com/rules> — active scenarios, scoring formula, mission framing
- **Validator side**: [VALIDATOR_OPERATIONS.md](VALIDATOR_OPERATIONS.md) — how validators consume the eval set you submit to
- **Incentive Mechanism**: [INCENTIVE_MECHANISM.md](INCENTIVE_MECHANISM.md) — consensus protocol, winner selection
- **Web API spec**: [trajectoryrl.web API.md](https://github.com/trajectoryRL/trajectoryrl.web/blob/main/API.md) — full request/response for `/api/v2/miners/submit` and other endpoints
- **Benchmark**: [trajrl-bench](https://github.com/trajectoryRL/trajrl-bench) — scenario definitions, fixture data, verifier code
