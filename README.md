# TrajectoryRL

> **Bittensor Subnet 11** — Optimize AI agent policies through decentralized competition

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Bittensor](https://img.shields.io/badge/bittensor-7.0+-green.svg)](https://github.com/opentensor/bittensor)

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

- **Deterministic Evaluation** — Uses [ClawBench](https://github.com/trajectoryRL/clawbench) scenarios with fixed fixtures (no LLM-as-judge randomness)
- **Content-Addressed Packs** — Miners submit policy bundles identified by SHA256 hash
- **Real-World Tasks** — Scenarios test email triage, calendar conflicts, task delegation, incident response
- **Safety-First Scoring** — Critical safety violations = immediate score of 0

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

```bash
# 1. Clone repo with submodules
git clone --recursive https://github.com/trajectoryRL/trajectoryrl.git
cd trajectoryrl

# 2. Configure
cp .env.example .env
# Edit .env: add your ANTHROPIC_API_KEY and wallet info

# 3. Start (launches validator + ClawBench services automatically)
docker compose up -d

# 4. View logs
docker compose logs -f validator
```

One command starts everything: the validator, ClawBench mock-tools, and the OpenClaw AI gateway. ClawBench is pinned to v0.3.0 (`b718230`) for validator consensus.

### Model Selection

Set `CLAWBENCH_MODEL` to change the LLM used for evaluation:

```bash
# In .env
CLAWBENCH_MODEL=anthropic/claude-sonnet-4-5-20250929  # default
CLAWBENCH_MODEL=ollama/llama3.3                        # local Ollama
CLAWBENCH_MODEL=openai/gpt-4o                          # OpenAI
```

See [ClawBench Model Configuration](https://github.com/trajectoryRL/clawbench#model-configuration) for local LLM setup details.

### For Miners

```bash
# 1. Clone repo
git clone --recursive https://github.com/trajectoryRL/trajectoryrl.git
cd trajectoryrl

# 2. Create your policy pack
mkdir -p packs
# Write packs/pack.json (see "How It Works" below)

# 3. Publish to GitHub
# In your pack repo: git add pack.json && git commit && git push

# 4. Configure
cp .env.example .env
# Edit .env: add WALLET_NAME, PACK_REPO, PACK_COMMIT

# 5. Start miner
docker compose --profile miner up -d

# 6. View logs
docker compose logs -f miner
```

## Project Structure

```
trajectoryrl/
├── docker-compose.yml         # All-in-one: validator + ClawBench services
├── .env.example               # Configuration template
├── trajectoryrl/              # Main package
│   ├── protocol/              # Bittensor synapses (PackRequest/PackResponse)
│   ├── base/                  # Core classes (TrajectoryValidator)
│   ├── utils/                 # ClawBench harness, config, GitHub verification
│   └── scoring/               # Score aggregation, winner-take-all
│
├── neurons/                   # Entry points
│   ├── validator.py           # python neurons/validator.py
│   └── miner.py               # python neurons/miner.py
│
├── docker/                    # Dockerfiles
│   ├── Dockerfile.validator
│   └── Dockerfile.miner
│
├── tests/                     # Test suite
├── pyproject.toml             # Package definition
└── README.md                  # This file
```

The `docker-compose.yml` automatically starts all required services:
- **validator** — Evaluates miner packs and sets on-chain weights
- **mock-tools** — Serves deterministic fixture data for ClawBench scenarios
- **openclaw** — AI gateway that runs Claude with each pack's AGENTS.md
- **miner** — (optional, `--profile miner`) Serves a policy pack via Bittensor

## How It Works

### For Miners

1. **Create a policy pack** (OpenClaw Policy Pack - OPP v1 format):
   ```json
   {
     "schema_version": 1,
     "files": {
       "AGENTS.md": "# Rules for the agent...",
       "SOUL.md": "# Tone and boundaries..."
     },
     "tool_policy": {
       "allow": ["exec", "slack", "memory_search"],
       "deny": ["group:runtime"]
     },
     "metadata": {
       "pack_name": "efficient_safe_ops",
       "pack_version": "1.0.0"
     }
   }
   ```

2. **Publish to public GitHub repository**
   - Commit your pack as `pack.json` in your repo
   - Note the git commit hash (40-char SHA)

3. **Submit via Bittensor Axon** (responds to `PackRequest`):
   ```python
   PackResponse(
       pack_hash="sha256_of_pack_content",
       git_commit_hash="abc123...",  # 40-char git SHA
       repo_url="https://github.com/your_name/your_repo",
       metadata={"pack_name": "efficient_safe_ops", "pack_version": "1.0.0"}
   )
   ```

4. **Validators verify and evaluate** your pack via GitHub
5. **Earn TAO** if you achieve the best score (winner-take-all)

### For Validators

1. **Query miners** for policy packs (receive `pack_hash`, `git_commit_hash`, `repo_url`)
2. **Verify GitHub submission**:
   - Clone/update miner's public repository
   - Verify commit exists and extract pack
   - Verify server-side push timestamp (via GitHub API) < on-chain submission time
   - Verify `sha256(pack_json) == pack_hash`
3. **Run ClawBench scenarios**:
   - `client_escalation` — P0 incident response
   - `morning_brief` — Daily operating picture
   - `inbox_to_action` — Email batch processing
   - `team_standup` — Sprint status synthesis
4. **Score results**:
   ```
   score = success - ρ*variance
   ```
   (ClawBench already includes safety and efficiency in scenario scoring)
5. **Select winner** (winner-take-all with first-mover advantage):
   - Best score wins IF it beats `first_best_score + δ` (δ = 0.05)
   - Protects early high-quality submissions from copy-paste attacks
6. **Set on-chain weights** (winner = 1.0, others = 0.0)

## Scoring

### Performance Evaluation

ClawBench scenarios evaluate packs on multiple dimensions:

- **Success** — Task completion (did it work?)
- **Safety** — No forbidden actions, confirmations required
- **Efficiency** — Minimize tool calls and tokens
- **Reliability** — Low variance across seeds (ρ = 0.1)
- **Consensus** — Each scenario runs N=3 times; rubric checks are majority-voted for stable scoring across validators

**Example scoring**:
```python
# Scenario: client_escalation (3 majority-voted runs)
clawbench_score = 0.92  # From majority-voted rubric checks
variance_penalty = 0.1 * 0.02  # ρ * variance across scenarios

final_score = 0.92 - 0.002 = 0.918
```

### Winner-Take-All Mechanism

TrajectoryRL uses **winner-take-all** reward distribution with **first-mover advantage**:

1. **Best score wins** — Miner with highest score receives 100% of rewards (weight = 1.0)
2. **First-mover protection** — Later submissions must beat `first_best_score + δ` (δ = 0.05)
3. **All others receive zero** — No participation rewards

**Example**:
```python
# Epoch 1: Miner A submits (score=0.85, timestamp=100)
weights = {A: 1.0}  # A wins

# Epoch 2: Miner B copies A's pack (score=0.85, timestamp=200)
# Required: 0.85 + 0.05 = 0.90
# B's score (0.85) < 0.90 → A retains win
weights = {A: 1.0, B: 0.0}

# Epoch 3: Miner C improves (score=0.91, timestamp=300)
# C's score (0.91) > 0.90 → C wins
weights = {A: 0.0, B: 0.0, C: 1.0}
```

This mechanism prevents copy-paste attacks by requiring meaningful improvement (5%) to overtake early submissions.

## ClawBench Integration

TrajectoryRL uses [ClawBench](https://github.com/trajectoryRL/clawbench) for deterministic evaluation:

- **Fixed fixtures** — Same inbox, calendar, tasks for every evaluation
- **Identity variation** — `{{PLACEHOLDER}}` templates in USER.md are filled per epoch (name, role, company)
- **Regex scoring** — No LLM judge, fully reproducible
- **Mock tools** — No real API calls, sandboxed execution

### Version Pinning

ClawBench is **pinned as a git submodule** to ensure validator consensus:

- **Version**: v0.3.0 (commit `b718230`)
- **Automatic**: Cloned with `git clone --recursive`
- **Update**: `git submodule update --init --recursive`

This ensures all validators evaluate packs with identical scoring logic, preventing consensus failures from version drift.

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

- **[Validator Guide](docs/validator_guide.md)** — Setup and operation
- **[Miner Guide](docs/miner_guide.md)** — Creating and optimizing packs
- **[OPP Spec](docs/opp_spec.md)** — Policy pack format
- **[Scoring](docs/scoring.md)** — How rewards are calculated
- **[Architecture](docs/architecture.md)** — Technical deep-dive

## Roadmap

- [x] ~~Validator implementation~~ (v0.1.0)
- [x] ~~Consensus scoring (majority-vote N=3)~~
- [x] ~~Epoch-seeded identity variation (35M+ contexts)~~
- [ ] Miner implementation (v0.2.0)
- [ ] Example policy packs
- [ ] Pack optimizer tools
- [ ] Prometheus metrics
- [ ] Web dashboard

## Community

- **GitHub**: https://github.com/trajectoryRL/trajectoryRL
- **Website**: https://trajrl.com

## License

MIT © 2026 TrajectoryRL Team

---

**Built on [Bittensor](https://bittensor.com)** | **Powered by [ClawBench](https://github.com/trajectoryRL/clawbench)**
