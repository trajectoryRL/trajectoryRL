# TrajectoryRL

> **Bittensor Subnet 11** — Optimize AI agent policies through decentralized competition

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Bittensor](https://img.shields.io/badge/bittensor-7.0+-green.svg)](https://github.com/opentensor/bittensor)

> **Status: Cold-Start Phase** — The subnet is bootstrapping. Validators are setting weights to anchor consensus. Full ClawBench evaluation is coming soon. Miners can register and prepare packs now.

TrajectoryRL is a Bittensor subnet where miners compete to optimize AI agent policies for real-world tasks. Validators evaluate policy packs using deterministic scenarios, rewarding agents that are **safe**, **efficient**, and **reliable**.

## Overview

```
┌─────────────────────────────────────────────────────┐
│              TRAJECTORYRL SUBNET (SN11)              │
│                                                      │
│  MINERS                        VALIDATORS            │
│  ┌──────────────┐              ┌──────────────┐    │
│  │ Submit       │  PackRequest │ Query miners │    │
│  │ AGENTS.md    │─────────────▶│              │    │
│  │ policy packs │              │ Evaluate via │    │
│  │              │◀─────────────│ ClawBench    │    │
│  └──────────────┘  PackResponse└──────────────┘    │
│        │                              │             │
│        │  Optimized policies          │ Set weights │
│        ▼                              ▼             │
│  ┌──────────────────────────────────────────────┐  │
│  │         BITTENSOR BLOCKCHAIN                 │  │
│  │  TAO rewards based on performance            │  │
│  └──────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────┘
```

### What Makes TrajectoryRL Unique

- **Deterministic Evaluation** — Uses [ClawBench](https://github.com/trajectoryRL/clawbench) scenarios with fixed fixtures and regex scoring (no LLM-as-judge randomness)
- **Content-Addressed Packs** — Miners submit policy bundles (OPP v1, ≤32KB) identified by SHA256 hash
- **Real-World Tasks** — 5 scenarios testing email triage, calendar conflicts, task delegation, incident response
- **Safety-First Scoring** — Critical safety violations = immediate score of 0
- **Weighted Scenarios** — Safety-critical scenarios (e.g., client escalation) carry higher weight in aggregation
- **Winner-Take-All** — Best miner gets 100% of rewards; first-mover advantage protects early innovators

### Anti-Copy Incentive Mechanism

TrajectoryRL employs three layers of protection against copy-paste attacks:

1. **GitHub-Based Submission**
   - Miners publish packs to public repositories FIRST
   - Validators verify **server-side push timestamps** via GitHub API (not forgeable git dates)
   - Uses Events API + Compare API (public, no auth required)
   - Validators independently clone repos and verify hashes

2. **Winner-Take-All**
   - Only the best miner receives rewards (100% of epoch emissions)
   - No participation rewards for mediocre submissions
   - Forces miners to innovate, not copy

3. **First-Mover Advantage**
   - Later submissions must beat `first_best_score + 0.05` to win
   - Protects early innovators from marginal improvements
   - Prevents "copy then tweak" strategies

## Quick Start

### For Validators

Validators run via Docker with automatic updates from GHCR via Watchtower. When new code is pushed to `main`, GitHub Actions builds a new image and Watchtower auto-pulls and restarts within 5 minutes.

```bash
# 1. Create .env.validator
cat > .env.validator <<'EOF'
WALLET_NAME=your-wallet
WALLET_HOTKEY=default
NETUID=11
NETWORK=finney
EOF

# 2. Start validator + Watchtower (auto-updates from GHCR)
docker compose -f docker/docker-compose.validator.yml --env-file .env.validator up -d

# 3. View logs
docker compose -f docker/docker-compose.validator.yml logs -f validator
```

**Prerequisites** (one-time setup on the host, before starting Docker):

```bash
# Install btcli
pip install bittensor-cli

# Create or import your wallet
btcli wallet create --wallet-name my-validator

# Register hotkey on SN11 (~0.2 TAO burn fee)
btcli subnets register --wallet-name my-validator --hotkey default --netuid 11

# Stake alpha so your weights count (must be top 64 by stake for validator permit)
btcli stake add --wallet-name my-validator --hotkey default --netuid 11 --amount 100
```

The Docker container uses the bittensor Python SDK to set weights — it reads wallet keyfiles from the mounted `~/.bittensor/wallets/` directory. No btcli is needed inside the container.

### For Miners

Miners compete by creating **policy packs** — instruction bundles (AGENTS.md + optional SOUL.md) that guide an AI agent through real-world tasks. The best pack wins 100% of epoch rewards.

**Step 1: Register on SN11**

```bash
pip install bittensor

# Create wallet
btcli wallet create --wallet-name miner

# Register (costs ~0.2 TAO burn fee)
btcli subnets register --wallet-name miner --hotkey default --netuid 11
```

**Step 2: Build your policy pack**

```bash
# Clone the repo (includes miner CLI tools)
git clone --recursive https://github.com/trajectoryRL/trajectoryRL.git
cd trajectoryRL
pip install -e .

# Write your AGENTS.md — this is where your strategy lives
# See "Policy Pack Format" below for the full spec

# Build pack.json from your AGENTS.md
python neurons/miner.py build --agents-md ./AGENTS.md --output pack.json

# Validate locally
python neurons/miner.py validate pack.json
```

**Step 3: Publish to GitHub**

Your pack must be in a **public** GitHub repo before submitting. Validators verify server-side push timestamps to prevent copy-paste attacks.

```bash
# Create a public repo, e.g. github.com/yourname/my-sn11-pack
cd /path/to/your-pack-repo
cp /path/to/pack.json .
git add pack.json
git commit -m "my policy pack v1"
git push origin main
```

**Step 4: Submit on-chain**

```bash
python neurons/miner.py submit pack.json \
    --repo yourname/my-sn11-pack \
    --git-commit $(git rev-parse HEAD) \
    --wallet.name miner --wallet.hotkey default

# Check your submission
python neurons/miner.py status --wallet.name miner --wallet.hotkey default
```

**Step 5: Iterate**

Improve your AGENTS.md, rebuild, push, and resubmit. The winner-take-all mechanism means you need to beat the current best score + 0.05 to take the lead.

## Policy Pack Format

A policy pack is an OPP v1 (OpenClaw Policy Pack) JSON file, max 32KB:

```json
{
  "schema_version": 1,
  "files": {
    "AGENTS.md": "# Your agent instructions here...\n\nTell the agent how to handle emails, calendar, tasks, incidents, etc.",
    "SOUL.md": "# Optional tone and boundaries..."
  },
  "tool_policy": {
    "allow": ["exec", "slack", "memory_search", "web_search", "web_fetch", "read"],
    "deny": ["group:runtime"]
  },
  "metadata": {
    "pack_name": "my-pack",
    "pack_version": "1.0.0"
  }
}
```

The `AGENTS.md` file is the core of your strategy. It gets injected into the AI agent's system prompt. A good AGENTS.md teaches the agent how to:
- Triage and respond to emails efficiently
- Handle calendar conflicts and scheduling
- Synthesize information for standups and briefs
- Escalate incidents safely (never leak confidential data)
- Minimize tool calls while maximizing task completion

### Scenarios Your Pack Will Be Evaluated On

| Scenario | Weight | Description |
|---|---|---|
| `client_escalation` | 1.5x | P0 incident response — safety critical |
| `inbox_to_action` | 1.5x | Email batch processing — efficiency critical |
| `morning_brief` | 1.0x | Daily operating picture synthesis |
| `team_standup` | 1.0x | Sprint status synthesis |
| `inbox_triage` | 1.0x | Inbox review smoke test |

### Miner CLI Commands

```bash
python neurons/miner.py build    --agents-md ./AGENTS.md -o pack.json   # Build pack
python neurons/miner.py validate pack.json                               # Validate schema
python neurons/miner.py submit   pack.json --repo user/repo             # Submit on-chain
python neurons/miner.py status   --wallet.name miner                    # Check commitment
```

## Project Structure

```
trajectoryrl/
├── docker-compose.yml                  # Dev: validator + ClawBench services
├── docker/
│   ├── Dockerfile.validator            # Validator Docker image
│   ├── docker-compose.validator.yml    # Production: validator + Watchtower
│   └── Dockerfile.miner
├── .github/workflows/
│   └── build-validator.yml             # CI: build & push to GHCR on push to main
├── .env.example                        # Configuration template
│
├── trajectoryrl/                       # Main package
│   ├── protocol/                       # Bittensor synapses (PackRequest/PackResponse)
│   ├── base/                           # Core classes (TrajectoryValidator, TrajectoryMiner)
│   ├── utils/                          # ClawBench harness, config, GitHub verification
│   └── scoring/                        # Score aggregation, winner-take-all
│
├── neurons/                            # Entry points
│   ├── validator.py                    # Validator (cold-start → production)
│   └── miner.py                        # Miner CLI (build/validate/submit/status)
│
├── clawbench/                          # Git submodule — scenarios, fixtures, scoring
├── openclaw/                           # Git submodule — AI gateway
├── tests/                              # Test suite
├── pyproject.toml                      # Package definition
└── README.md                           # This file
```

## How It Works

### Evaluation Pipeline (coming soon)

1. Validators read miner commitments from on-chain
2. Fetch packs from public GitHub repos, verify hashes and push timestamps
3. Run all 5 ClawBench scenarios with deterministic fixtures and regex scoring
4. Score: `final = quantize(weighted_mean - 0.1 * variance, step=0.05)`
5. Winner-take-all: best score gets 100% of epoch rewards
6. First-mover advantage: later submissions must beat `best + 0.05` to win
7. Bootstrap phase (< 10 miners): top 3 get 70% / 20% / 10%

## Scoring

### Performance Evaluation

ClawBench scenarios evaluate packs on multiple dimensions:

- **Success** — Task completion (did it work?)
- **Safety** — No forbidden actions, confidential data protected
- **Efficiency** — Minimize tool calls and tokens
- **Reliability** — Low variance across seeds (ρ = 0.1)
- **Consensus** — Each scenario runs N=3 seeds; rubric checks are majority-voted for stable scoring across validators

Scenarios are **weighted** — safety-critical scenarios (client_escalation, inbox_to_action) carry 1.5x weight:

```python
# Weighted aggregation across 5 scenarios
# client_escalation=0.90 (w=1.5), morning_brief=0.85 (w=1.0),
# inbox_to_action=0.88 (w=1.5), team_standup=0.80 (w=1.0), inbox_triage=0.95 (w=1.0)
weighted_mean = (0.90*1.5 + 0.85*1.0 + 0.88*1.5 + 0.80*1.0 + 0.95*1.0) / 6.0
# = 0.877

# Variance penalty + quantization
final_score = quantize(0.877 - 0.1 * variance, step=0.05)
# = 0.85 (snapped to 0.05 grid)
```

### Winner-Take-All Mechanism

TrajectoryRL uses **winner-take-all** reward distribution with **first-mover advantage**:

1. **Best score wins** — Miner with highest quantized score receives 100% of rewards
2. **Consensus epsilon** — Scores within ε=0.02 are tied; tie goes to earliest push timestamp
3. **First-mover protection** — Later submissions must beat `first_best_score + δ` (δ = 0.05)
4. **All others receive zero** — No participation rewards

### Bootstrap Phase

When the subnet has **fewer than 10 active miners**, rewards are distributed using a graduated curve to encourage early adoption:

| Rank | Share |
|------|-------|
| 1st  | 70%   |
| 2nd  | 20%   |
| 3rd  | 10%   |

Once ≥ 10 miners are active, the subnet transitions to pure winner-take-all.

## ClawBench Integration

TrajectoryRL uses [ClawBench](https://github.com/trajectoryRL/clawbench) for deterministic evaluation:

- **Fixed fixtures** — Same inbox, calendar, tasks for every evaluation
- **5 scenarios** — client_escalation, morning_brief, inbox_to_action, team_standup, inbox_triage
- **Identity variation** — `{{PLACEHOLDER}}` templates in USER.md are filled per epoch (name, role, company)
- **Regex scoring** — Deterministic rubric checks, fully reproducible (LLM-as-judge planned for v0.3.0)
- **Mock tools** — No real API calls, sandboxed execution

### Version Pinning

Both ClawBench and OpenClaw are **pinned as git submodules** to ensure validator consensus:

- **ClawBench**: v0.3.0+22 (commit `e50824d`) — scenarios, fixtures, scoring
- **OpenClaw**: commit `b5ffec1` — AI gateway with ClawBench tools plugin
- **Automatic**: Cloned with `git clone --recursive`
- **Update**: `git submodule update --init --recursive`

This ensures all validators evaluate packs with identical scoring logic and gateway behavior, preventing consensus failures from version drift.

## Development

### Installation

```bash
# Clone with submodules
git clone --recursive https://github.com/trajectoryRL/trajectoryrl.git
cd trajectoryrl

# If you already cloned without --recursive:
git submodule update --init --recursive

# Install in editable mode with dev dependencies
pip install -e ".[dev]"
```

### Running Tests

```bash
pytest tests/
```

### Code Formatting

```bash
# Format code
black trajectoryrl/ neurons/

# Lint
ruff check trajectoryrl/ neurons/
```

### Local Testing

```bash
# 1. Start local Bittensor subtensor
docker run -p 9944:9944 opentensor/subtensor:latest

# 2. Create test wallets
btcli wallet new_coldkey --wallet.name test_miner --no_password
btcli wallet new_coldkey --wallet.name test_validator --no_password

# 3. Run locally
NETWORK=local python neurons/miner.py --wallet.name test_miner
NETWORK=local python neurons/validator.py --wallet.name test_validator
```

## Documentation

- **[Incentive Mechanism](INCENTIVE_MECHANISM.md)** — Scoring, rewards, and anti-copy protection
- **[ClawBench](https://github.com/trajectoryRL/clawbench)** — Evaluation framework (scenarios, fixtures, scoring)

## Roadmap

### Cold-Start Phase (current)
- [x] Validator and miner implementation
- [x] Docker + Watchtower auto-deploy pipeline
- [x] Consensus anchoring via owner validators
- [ ] Onboard first external miners and validators

### v1.0 (next)
- [ ] Full ClawBench evaluation enabled
- [ ] Example policy packs (baseline + optimized)
- [ ] Pack cache with LRU eviction

### v2.0 (planned)
- [ ] Hybrid LLM-as-judge scoring alongside regex checks
- [ ] Anti-gaming defenses (keyword blocklist, fixture name variation)
- [ ] Hidden held-out scenarios (scored but not published)
- [ ] Web dashboard (scores, leaderboard, epoch history)

## Community

- **GitHub**: https://github.com/trajectoryRL/trajectoryRL
- **Website**: https://trajrl.com

## License

MIT © 2026 TrajectoryRL Team

---

**Built on [Bittensor](https://bittensor.com)** | **Powered by [ClawBench](https://github.com/trajectoryRL/clawbench)**
