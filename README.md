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

## Quick Start

### For Validators

```bash
# 1. Clone repo
git clone https://github.com/trajectoryRL/trajectoryrl.git
cd trajectoryrl

# 2. Install
pip install -e .

# 3. Configure
cp .env.example .env
# Edit: add ANTHROPIC_API_KEY, WALLET_NAME

# 4. Run validator
python neurons/validator.py
```

### For Miners

```bash
# 1. Install
pip install -e .

# 2. Create policy pack
# See: docs/creating_packs.md

# 3. Run miner
python neurons/miner.py --pack.path ./my_pack.json
```

## Project Structure

```
trajectoryrl/
├── trajectoryrl/              # Main package
│   ├── protocol/              # Bittensor synapses
│   │   └── synapse.py         # PackRequest/PackResponse
│   ├── base/                  # Core classes
│   │   ├── miner.py           # TrajectoryMiner (TODO)
│   │   └── validator.py       # TrajectoryValidator
│   ├── utils/                 # Shared utilities
│   │   ├── config.py          # Configuration
│   │   ├── clawbench.py       # ClawBench integration
│   │   └── opp_schema.py      # OPP v1 validation
│   └── scoring/               # Scoring logic
│       └── __init__.py        # Score aggregation
│
├── neurons/                   # Entry points
│   ├── validator.py           # Validator node
│   └── miner.py               # Miner node
│
├── docker/                    # Docker deployment
│   ├── Dockerfile.validator
│   └── docker-compose.yml     # Includes ClawBench
│
├── tests/                     # Test suite
├── docs/                      # Documentation
├── pyproject.toml             # Package definition
└── README.md                  # This file
```

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

2. **Serve pack** via Bittensor Axon (responds to `PackRequest`)
3. **Validators fetch and evaluate** your pack
4. **Earn TAO** based on performance

### For Validators

1. **Query miners** for policy packs
2. **Verify pack hash** (`sha256(pack_json)`)
3. **Run ClawBench scenarios**:
   - `client_escalation` — P0 incident response
   - `morning_brief` — Daily operating picture
   - `inbox_to_action` — Email batch processing
   - `team_standup` — Sprint status synthesis
4. **Score results**:
   ```
   score = success - λ*cost - μ*safety - ρ*variance
   ```
5. **Set on-chain weights**

## Scoring

Validators score packs on four dimensions:

| Dimension | Weight | Description |
|-----------|--------|-------------|
| **Success** | 50% | Task completion (did it work?) |
| **Safety** | 40% | No forbidden actions, confirmations required |
| **Efficiency** | 30% | Minimize tool calls and tokens |
| **Reliability** | 10% | Low variance across seeds |

**Example scoring**:
```python
# Scenario: client_escalation
success_rate = 0.95  # 95% task completion
safety_score = 1.0   # No violations
efficiency = 0.8     # Used 20% more tools than baseline
variance = 0.02      # Low variance

final_score = 0.95 - 0.3*(1-0.8) - 0.4*(1-1.0) - 0.1*0.02
            = 0.95 - 0.06 - 0 - 0.002
            = 0.888
```

## ClawBench Integration

TrajectoryRL uses [ClawBench](https://github.com/trajectoryRL/clawbench) for deterministic evaluation:

- **Fixed fixtures** — Same inbox, calendar, tasks for every evaluation
- **Regex scoring** — No LLM judge, fully reproducible
- **Mock tools** — No real API calls, sandboxed execution

Validators must have ClawBench cloned as a sibling directory:
```bash
git clone https://github.com/trajectoryRL/clawbench.git ../clawbench
```

## Development

### Installation

```bash
# Clone with submodules
git clone https://github.com/trajectoryRL/trajectoryrl.git
cd trajectoryrl

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
- [ ] Miner implementation (v0.2.0)
- [ ] Example policy packs
- [ ] Pack optimizer tools
- [ ] Multi-seed variance testing
- [ ] Prometheus metrics
- [ ] Web dashboard

## Community

- **GitHub**: https://github.com/trajectoryRL/trajectoryRL
- **Website**: https://trajrl.com

## License

MIT © 2026 TrajectoryRL Team

---

**Built on [Bittensor](https://bittensor.com)** | **Powered by [ClawBench](https://github.com/trajectoryRL/clawbench)**
