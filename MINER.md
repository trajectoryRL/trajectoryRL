# Miner Submission Pipeline

Quick-start guide for building and submitting policy packs on SN11.

> For scoring rules and mechanism design, see [INCENTIVE_MECHANISM.md](INCENTIVE_MECHANISM.md).
> For operational context (local testing, score targets), see [MINER_OPERATIONS.md](MINER_OPERATIONS.md).

---

## Prerequisites

```bash
# 1. Install
pip install -e .

# 2. Create wallet (if needed)
btcli wallet create --wallet.name miner --wallet.hotkey default

# 3. Register on subnet
btcli subnet register --netuid 11 --wallet.name miner --wallet.hotkey default

# 4. Create a public GitHub repo for your pack
```

---

## Workflow

### Step 1: Write Your Policy

Create an `AGENTS.md` file with your agent policy. This is the core of your pack — instructions that guide the agent's behavior during ClawBench evaluation.

```markdown
# AGENTS.md
You are a helpful assistant. Follow these rules:
- Always check tool outputs before proceeding
- Never execute destructive commands
- Prioritize safety checks over speed
```

### Step 2: Build the Pack

```bash
python neurons/miner.py build \
  --agents-md ./AGENTS.md \
  --pack-name my-pack \
  --pack-version 1.0.0 \
  --output pack.json
```

Optional flags:
- `--soul-md ./SOUL.md` — personality guidance file
- `--pack-name NAME` — pack name for metadata (default: `my-pack`)
- `--pack-version X.Y.Z` — semver version (default: `1.0.0`)

Output shows pack hash, size, file list, and schema validation result.

### Step 3: Validate Locally

```bash
python neurons/miner.py validate pack.json
```

Checks: schema version, required files (`AGENTS.md`), size limit (32KB), valid semver.

### Step 4: Push to GitHub

```bash
cd /path/to/your-pack-repo
cp /path/to/pack.json .
git add pack.json AGENTS.md
git commit -m "v1.0.0: initial policy pack"
git push origin main
```

Note the commit hash — you'll need it for submission.

### Step 5: Submit On-Chain

**Option A** — Pack already pushed, provide commit hash:

```bash
python neurons/miner.py submit pack.json \
  --repo youruser/your-pack-repo \
  --git-commit $(git rev-parse HEAD) \
  --wallet.name miner \
  --wallet.hotkey default \
  --netuid 11 \
  --network finney
```

**Option B** — Auto-push from local repo:

```bash
python neurons/miner.py submit pack.json \
  --repo youruser/your-pack-repo \
  --repo-path /path/to/local/repo \
  --wallet.name miner \
  --wallet.hotkey default
```

### Step 6: Verify

```bash
python neurons/miner.py status \
  --wallet.name miner \
  --wallet.hotkey default
```

Shows your current on-chain commitment: pack hash, git commit, and repo URL.

---

## On-Chain Commitment Format

The commitment stored on-chain is a pipe-delimited string (max 128 bytes):

```
{pack_hash_hex}|{git_commit_hash}|{owner/repo}
```

- `pack_hash_hex` — SHA256 of canonical `json.dumps(pack, sort_keys=True)` (64 chars)
- `git_commit_hash` — full 40-char git commit hash
- `owner/repo` — GitHub repository shorthand

The block number of the commitment establishes first-mover precedence.

---

## Troubleshooting

| Error | Cause | Fix |
|-------|-------|-----|
| "not registered on subnet" | Hotkey not registered | `btcli subnet register --netuid 11 --wallet.name miner --wallet.hotkey default` |
| "Cannot connect to network" | Chain endpoint unreachable | Check internet connection; try `--network test` for testnet |
| "Wallet key file error" | Missing or corrupted wallet | `btcli wallet list` to check; recreate if needed |
| "Chain transaction failed" | On-chain tx rejected | Check balance, wait ~100 blocks between submissions |
| "Schema: FAILED" | Invalid pack format | Run `validate` subcommand; check AGENTS.md exists, size < 32KB, valid semver |
| "git commit" fails silently | No git user identity | Fixed in current version — git identity is set automatically |

---

## Python API

For programmatic use:

```python
from trajectoryrl.base.miner import TrajectoryMiner

# Build
pack = TrajectoryMiner.build_pack(agents_md="path/to/AGENTS.md")

# Validate
result = TrajectoryMiner.validate(pack)
assert result.passed, result.issues

# Save and hash
pack_hash = TrajectoryMiner.save_pack(pack, "pack.json")

# Submit
miner = TrajectoryMiner(wallet_name="miner", wallet_hotkey="default")
miner.submit(pack=pack, repo="youruser/your-pack", git_commit="b" * 40)
```

---

## File Reference

| File | Purpose |
|------|---------|
| `neurons/miner.py` | CLI entry point (build, validate, submit, status) |
| `trajectoryrl/base/miner.py` | `TrajectoryMiner` class — core logic |
| `trajectoryrl/utils/opp_schema.py` | OPP v1 schema validation |
| `trajectoryrl/utils/commitments.py` | Commitment string parsing |
| `tests/test_miner.py` | Test suite (43 tests) |
