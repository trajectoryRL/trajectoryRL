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

# 3. Upload to S3-compatible storage
python neurons/miner.py upload pack.json

# 4. Submit on-chain
python neurons/miner.py submit https://your-bucket.s3.amazonaws.com/pack.json

# 5. Check status
python neurons/miner.py status
```

That's it. Repeat steps 1-4 whenever you improve your SKILL.md.

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
│  3. upload → public URL                                 │
│  4. submit → on-chain commitment                        │
│  5. Wait for validator evaluation (~24h epoch)           │
│  6. Check results, iterate                              │
└─────────────────────────────────────────────────────────┘
```

Validators only re-evaluate when your `pack_hash` changes. No need to resubmit if your SKILL.md hasn't changed.

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
- **Incentive Mechanism**: [INCENTIVE_MECHANISM.md](INCENTIVE_MECHANISM.md) — consensus protocol, winner selection
- **Scoring & Evaluation**: [EVALUATION_S1.md](EVALUATION_S1.md) — pack schema, scoring formula
- **Benchmark**: [trajrl-bench](https://github.com/trajectoryRL/trajrl-bench) — scenarios, JUDGE.md, local testing
