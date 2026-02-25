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

> **WIP** — Mining is not live yet. The evaluation pipeline is being activated. Miners can register and prepare packs now, but submissions are not being scored during the cold-start phase.

See [INCENTIVE_MECHANISM.md](INCENTIVE_MECHANISM.md) for full details on how packs are evaluated and scored.

## Documentation

- **[Incentive Mechanism](INCENTIVE_MECHANISM.md)** — Scoring, rewards, winner-take-all, and anti-copy protection
- **[ClawBench](https://github.com/trajectoryRL/clawbench)** — Evaluation framework (scenarios, fixtures, scoring)

## Community

- **GitHub**: https://github.com/trajectoryRL/trajectoryRL
- **Website**: https://trajrl.com

## License

MIT © 2026 TrajectoryRL Team

---

**Built on [Bittensor](https://bittensor.com)** | **Powered by [ClawBench](https://github.com/trajectoryRL/clawbench)**
