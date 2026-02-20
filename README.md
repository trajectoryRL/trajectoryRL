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

```bash
# 1. Clone repo with submodules
git clone --recursive https://github.com/trajectoryRL/trajectoryrl.git
cd trajectoryrl

# 2. Configure
cp .env.example .env
# Edit .env: add your ANTHROPIC_API_KEY and wallet info

# 3. Start with auto-update (watches for code changes)
docker compose up --watch

# Or run detached (no auto-update)
docker compose up -d

# 4. View logs
docker compose logs -f validator
```

One command starts everything: the validator, ClawBench mock-tools, and the OpenClaw AI gateway. `--watch` monitors the local repo for changes — after `git pull`, the validator automatically picks up new scenarios, scoring updates, and code fixes without manual container rebuilds. ClawBench is pinned to v0.3.0 (`b718230`) for validator consensus.

### Model Selection

All validators **must** use the designated model for consensus:

```bash
# In .env (default — do not change)
CLAWBENCH_MODEL=anthropic/claude-sonnet-4-5-20250929
```

Using a different model will produce different tool-call sequences and scoring outcomes, putting your validator out of consensus. See [VALIDATOR_OPERATIONS.md](VALIDATOR_OPERATIONS.md) for details.

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
3. **Run ClawBench scenarios** (5 scenarios, weighted):
   - `client_escalation` — P0 incident response (weight 1.5)
   - `morning_brief` — Daily operating picture (weight 1.0)
   - `inbox_to_action` — Email batch processing (weight 1.5)
   - `team_standup` — Sprint status synthesis (weight 1.0)
   - `inbox_triage` — Inbox review smoke test (weight 1.0)
4. **Score results** (weighted aggregation with consensus):
   ```
   weighted_mean = Σ(w_i * s_i) / Σ(w_i)
   final_score = quantize(weighted_mean - ρ * variance, step=0.05)
   ```
   Each scenario runs N=3 seeds with majority-vote consensus. Quantization to 0.05 grid ensures validator agreement.
5. **Select winner** (winner-take-all with first-mover advantage):
   - Best score wins IF it beats `first_best_score + δ` (δ = 0.05)
   - Scores within ε = 0.02 are tied; tie goes to earliest push timestamp
   - Protects early high-quality submissions from copy-paste attacks
   - **Bootstrap phase** (< 10 miners): top 3 get 70% / 20% / 10% instead of WTA
6. **Set on-chain weights** (steady state: winner = 1.0, others = 0.0)

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

### v0.1.0 (complete)
- [x] Validator implementation
- [x] Consensus scoring (majority-vote N=3, quantization, epsilon tie-break)
- [x] Epoch-seeded identity variation (35M+ contexts)
- [x] Weighted scenario scoring (safety-critical scenarios = 1.5x)
- [x] Bootstrap graduated rewards (70/20/10 when < 10 miners)
- [x] OPP v1 pack schema with validation (32KB limit)
- [x] GitHub push timestamp verification (server-side, not forgeable)

### v0.2.0 (next)
- [ ] Miner implementation
- [ ] Example policy packs (baseline + optimized)
- [ ] URL-based pack fetching (for packs > inline base64)
- [ ] Pack cache with LRU eviction

### v0.3.0 (planned)
- [ ] **Hybrid LLM-as-judge scoring** — Use a fixed LLM model to evaluate semantic correctness and response quality alongside regex checks. Regex handles objective checks (safety, efficiency); LLM judge handles subjective checks (correctness, structure). Consensus maintained via: pinned model + binary pass/fail output + majority-vote over N=3 seeds + score quantization.
- [ ] Anti-gaming defenses (AGENTS.md keyword blocklist, fixture name variation)
- [ ] Hidden held-out scenarios (scored but not published)

### Future
- [ ] Pack optimizer tools
- [ ] Prometheus metrics and alerting
- [ ] Web dashboard (scores, leaderboard, epoch history)

## Community

- **GitHub**: https://github.com/trajectoryRL/trajectoryRL
- **Website**: https://trajrl.com

## License

MIT © 2026 TrajectoryRL Team

---

**Built on [Bittensor](https://bittensor.com)** | **Powered by [ClawBench](https://github.com/trajectoryRL/clawbench)**
