# Miner Operations Guide

**Subnet**: SN11 (TrajectoryRL)
**Season**: 1 (Self-Learning Agents)

> For scoring, sandbox environment, SKILL.md writing, and anti-gaming rules, see [MINER_GUIDE.md](MINER_GUIDE.md).
> For mechanism design, see [INCENTIVE_MECHANISM.md](INCENTIVE_MECHANISM.md).

---

## What Is Mining on TrajectoryRL?

Mining means writing a **SKILL.md** — a knowledge document that teaches an AI agent how to explore, learn, and solve tasks in sandboxed environments. You're not running GPU workloads or a long-running daemon. You're doing agent instruction engineering.

The miner CLI is a **toolbox**: independent commands you compose however you want. Write your SKILL.md (manually, with an LLM, with your own automation), then use the CLI to build, upload, and submit.

---

## Prerequisites

| Requirement | Details |
|-------------|---------|
| **Bittensor wallet** | `btcli wallet create --wallet.name miner --wallet.hotkey default` |
| **Registration** | `btcli subnet register --netuid 11 --wallet.name miner` (dynamic cost) |
| **Python** | 3.10+ |
| **HTTP hosting** | Any public HTTP(S) endpoint for pack hosting (S3, GCS, GitHub raw, etc.) |

---

## Quick Start

```bash
git clone https://github.com/trajectoryRL/trajectoryRL.git
cd trajectoryRL

# 1. Write your SKILL.md (use any method — manual, LLM, your own scripts)
vim SKILL.md

# 2. Build pack
python neurons/miner.py build SKILL.md -o pack.json

# 3a. Self-host: upload to your own S3-compatible storage
python neurons/miner.py upload pack.json
#    OR
# 3b. Web submit: POST to the subnet's hosted endpoint (no hosting required)
#    See "Path B" below for the curl recipe; returns a pack_url to use in step 4

# 4. Submit on-chain
python neurons/miner.py submit <pack_url>

# 5. Check status
python neurons/miner.py status
```

That's it. Repeat steps 1-4 whenever you improve your SKILL.md.

---

## Two submission paths

| Path | What you operate | Where the pack lives | Reveal | Use this when |
|------|-----------------|----------------------|--------|---------------|
| **A. Self-host + chain commit** | An HTTP host (S3/GCS/GH/own server) for `pack.json` | Your bucket | Public on chain immediately (commit URL is on-chain) | You already have hosting and want full control |
| **B. Web submit (recommended for new miners)** | Just the miner CLI, no hosting | Subnet's GCS at an unguessable random key | URL hidden 48 h; commit on-chain still required for discovery | You want the easiest path; you don't want to run hosting |

Both paths produce the same on-chain artifact: `set_commitment(pack_hash | url)`. The difference is who hosts the pack content.

### Path A — chain commit (classic)

```bash
python neurons/miner.py build  SKILL.md -o pack.json
python neurons/miner.py upload pack.json                       # to your S3/GCS
python neurons/miner.py submit https://your-bucket/pack.json   # set_commitment on chain
```

### Path B — web submit (preferred)

```bash
# 1. Build pack locally (same as Path A)
python neurons/miner.py build SKILL.md -o pack.json

# 2. POST pack to the subnet's hosted submit endpoint (signed with your hotkey).
#    The endpoint stores the pack at an unguessable random GCS path and
#    returns the `pack_url` to use on-chain.
curl -X POST https://trajrl.com/api/v2/miners/submit \
  -H 'content-type: application/json' \
  -d "$(jq -n --slurpfile pack pack.json \
              --arg miner_hotkey "$MINER_HOTKEY" \
              --arg ts "$(date +%s)" \
              --arg sig "$(sign 'trajectoryrl-miner-submit:'"$MINER_HOTKEY"':'"$ts")" \
              --arg hash "$(sha256_canonical pack.json)" \
        '{ miner_hotkey: $miner_hotkey, timestamp: ($ts | tonumber),
           signature: $sig, pack_hash: $hash, pack_content: ($pack[0] | tostring) }')"

# Response → { pack_url: "https://storage.googleapis.com/.../pack.json", ... }

# 3. Commit on-chain with the returned pack_url
python neurons/miner.py submit <pack_url_from_response>
```

The `submit` endpoint immediately validates your signature, hash and pack format, kicks off pre-eval asynchronously, and returns the `pack_url`. Web-source URLs are not exposed in any other API for **48 hours** (the "reveal gate"), so a competitor can't simply scrape the dashboard to harvest your fresh SKILL.md.

Full request/response spec for `/api/v2/miners/submit` is in [`trajectoryrl.web/API.md`](https://github.com/trajectoryRL/trajectoryrl.web/blob/main/API.md).

---

## CLI Reference

```bash
python neurons/miner.py build     <skill_md_path> [-o pack.json]
python neurons/miner.py validate  <pack.json>
python neurons/miner.py upload    <pack.json> [--bucket ...] [--endpoint-url ...]
python neurons/miner.py submit    <pack_url>
python neurons/miner.py status
```

### build

Build a Season 1 pack from a SKILL.md file.

```bash
python neurons/miner.py build ./SKILL.md -o pack.json
```

Output:
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
python neurons/miner.py validate pack.json
```

### upload

Upload pack.json to S3-compatible storage. Prints the public URL.

```bash
python neurons/miner.py upload pack.json
python neurons/miner.py upload pack.json --bucket my-bucket --endpoint-url https://storage.googleapis.com
```

Reads S3 config from environment or CLI flags. Returns the public URL for use with `submit`.

### submit

Submit an on-chain commitment pointing to your hosted pack.

```bash
python neurons/miner.py submit https://your-bucket.s3.amazonaws.com/pack.json
```

Fetches the pack from the URL, verifies the hash, and calls `set_commitment` on-chain. The on-chain block number establishes first-mover precedence.

> **Rate limit**: One commitment per ~100 blocks (~20 min) per hotkey.

### status

Show your current on-chain commitment.

```bash
python neurons/miner.py status
```

Output:
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

### S3-Compatible Storage (for `upload` command)

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `S3_BUCKET` | for upload | — | S3-compatible bucket name |
| `S3_ENDPOINT_URL` | no | — | Custom endpoint for GCS/R2/MinIO |
| `S3_REGION` | no | `us-east-1` | Bucket region |
| `AWS_ACCESS_KEY_ID` | for upload | — | S3/GCS HMAC access key |
| `AWS_SECRET_ACCESS_KEY` | for upload | — | S3/GCS HMAC secret key |

| Service | `S3_ENDPOINT_URL` | Credentials |
|---------|-------------------|-------------|
| **AWS S3** | _(leave empty)_ | `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` |
| **Google Cloud Storage** | `https://storage.googleapis.com` | GCS HMAC keys |
| **Cloudflare R2** | `https://<account>.r2.cloudflarestorage.com` | R2 API tokens |
| **MinIO** | `https://your-minio-host:9000` | MinIO access/secret keys |

---

## Pack Format (Season 1)

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

For what to write in SKILL.md, see [MINER_GUIDE.md § Writing SKILL.md](MINER_GUIDE.md#writing-skillmd).

---

## Typical Workflow

```
┌─────────────────────────────────────────────────────────┐
│  You (human or your own automation)                     │
│                                                         │
│  1. Write / iterate on SKILL.md                         │
│  2. build → pack.json                                   │
│  3. either upload (Path A) or POST web-submit (Path B)  │
│     → public pack_url                                   │
│  4. submit → on-chain commitment (pack_hash | pack_url) │
│  5. Wait for validator evaluation (~24h epoch)           │
│  6. Check results, iterate                              │
└─────────────────────────────────────────────────────────┘
```

Validators only re-evaluate when your `pack_hash` changes. No need to resubmit if your SKILL.md hasn't changed. **However**, if you want to remain in the active eval set you must keep your on-chain commitment alive — the snapshot endpoint excludes packs whose `refresh_time` (the rolling activity stamp) is older than 48 hours. The sync service refreshes this stamp every 5 min for any commitment still on chain, so as long as you don't deregister you stay active.

Re-submitting the same `(hotkey, pack_hash)` to `/api/v2/miners/submit` (Path B) within 1h is rate-limited by the cooldown; outside the cooldown it just bumps `refresh_time` without re-running the pipeline.

---

## Local Testing

Before submitting, test your SKILL.md locally with the evaluation harness:

```bash
git clone https://github.com/trajectoryRL/trajrl-bench.git
cd trajrl-bench
pip install -e ".[dev]"
make build       # builds sandbox + hermes Docker images
cp .env.example .env  # add your LLM API key
make test-hermes      # runs one episode with real agent + real judge
```

Results are saved to `results/`. See the [trajrl-bench README](https://github.com/trajectoryRL/trajrl-bench) for more.

---

## Viewing Evaluation Results

After each evaluation epoch (~24h), validators upload per-miner eval logs to the dashboard. Each log is a tar.gz containing:

```
SKILL.md                                 # miner's product
JUDGE.md                                 # scoring rubric used
metadata.json                            # final_score, mean_quality, delta, episode qualities
world.json                               # company context + validator salt
episodes/episode_N/
  testee_transcript.txt                  # testee's session output
  judge_transcript.txt                   # judge agent's grading session
  evaluation.json                        # per-criterion scores + summary + strengths/weaknesses
  episode.json                           # fixtures + instruction
```

Retrieve your eval logs via `trajrl subnet logs` (see [trajrl CLI docs](https://pypi.org/project/trajrl/)):

```bash
trajrl subnet logs --miner <hotkey> --limit 5     # list recent evals
trajrl subnet logs --eval-id <id> --show          # pretty-print summary
trajrl subnet logs --eval-id <id> --dump-to ./    # extract full archive
```

Use the testee transcript + judge feedback together to debug failure modes and iterate on your SKILL.md.

---

## References

- **Season 1 Guide**: [MINER_GUIDE.md](MINER_GUIDE.md) — scoring, sandbox, SKILL.md writing, anti-gaming rules
- **Validator side**: [VALIDATOR_OPERATIONS.md](VALIDATOR_OPERATIONS.md) — how validators consume the eval set you submit to
- **Incentive Mechanism**: [INCENTIVE_MECHANISM.md](INCENTIVE_MECHANISM.md) — consensus protocol, winner selection
- **Scoring & Evaluation**: [EVALUATION_S1.md](EVALUATION_S1.md) — pack schema, scoring formula
- **Web API spec**: [API.md](https://github.com/trajectoryRL/trajectoryrl.web/blob/main/API.md) — full request/response for `/api/v2/miners/submit` and other endpoints
- **Benchmark**: [trajrl-bench](https://github.com/trajectoryRL/trajrl-bench) — scenarios, JUDGE.md, local testing
