# TrajectoryRL Docker Deployment

Docker is the **recommended way** to run TrajectoryRL validators and miners. A single all-in-one image contains everything needed: validator, mock-tools server, and OpenClaw gateway. Watchtower auto-updates the image from GHCR.

## Quick Start — Validator

```bash
# 1. Clone repo
git clone https://github.com/trajectoryRL/trajectoryRL.git
cd trajectoryRL

# 2. Configure
cp .env.example .env.validator
# Edit .env.validator: set WALLET_NAME, WALLET_HOTKEY, CLAWBENCH_LLM_API_KEY

# 3. Start
docker compose -f docker/docker-compose.validator.yml --env-file .env.validator up -d

# 4. View logs
docker compose -f docker/docker-compose.validator.yml logs -f validator
```

## Configuration

Create `.env.validator` in the repo root:

```bash
# Required
WALLET_NAME=validator
WALLET_HOTKEY=default
NETUID=11
NETWORK=finney
CLAWBENCH_LLM_API_KEY=...
CLAWBENCH_LLM_BASE_URL=https://open.bigmodel.cn/api/paas/v4
CLAWBENCH_DEFAULT_MODEL=zhipu/glm-5.1

# Optional
LOG_LEVEL=INFO                    # DEBUG for development
EVAL_INTERVAL_BLOCKS=7200         # ~24h (use 100 for dev)
WEIGHT_INTERVAL_BLOCKS=360        # ~72min
EMA_ALPHA=0.3
INACTIVITY_BLOCKS=14400           # ~48h
```

### Wallet Setup

Wallets must exist at `~/.bittensor/wallets/` (mounted read-only):

```
~/.bittensor/wallets/{WALLET_NAME}/
  ├── coldkey
  ├── coldkeypub.txt
  └── hotkeys/
      └── {WALLET_HOTKEY}
```

## Quick Start — Miner

```bash
# Configure
cp .env.miner.example .env.miner
# Edit: set WALLET_NAME, WALLET_HOTKEY, ANTHROPIC_API_KEY

# Run miner CLI
docker compose -f docker/docker-compose.miner.yml --env-file .env.miner run miner build
docker compose -f docker/docker-compose.miner.yml --env-file .env.miner run miner validate
docker compose -f docker/docker-compose.miner.yml --env-file .env.miner run miner submit
```

## Operations

```bash
# Start
docker compose -f docker/docker-compose.validator.yml --env-file .env.validator up -d

# View logs
docker compose -f docker/docker-compose.validator.yml logs -f validator
docker compose -f docker/docker-compose.validator.yml logs --tail=100 validator

# Stop
docker compose -f docker/docker-compose.validator.yml down

# Force update (Watchtower handles this automatically)
docker compose -f docker/docker-compose.validator.yml pull validator
docker compose -f docker/docker-compose.validator.yml up -d validator
```

## Architecture

The all-in-one validator image runs three processes managed by a bash entrypoint:

```
┌─────────────────────────────────────────────┐
│  trajectoryrl-validator container           │
│                                             │
│  1. mock-tools   (port 3001)  ─ background  │
│  2. OpenClaw     (port 18789) ─ background  │
│  3. validator    (PID 1)      ─ foreground  │
│                                             │
│  Shared: /workspace (AGENTS.md, fixtures)   │
└─────────────────────────────────────────────┘
```

- **mock-tools**: Deterministic tool responses from fixtures (email, calendar, Slack, tasks)
- **OpenClaw gateway**: LLM agent orchestrator with clawbench-tools plugin
- **validator**: Reads on-chain commitments, fetches packs, evaluates via ClawBench, sets weights

Watchtower polls GHCR every 5 minutes and auto-updates the entire image.

### Compose Variants

| File | Use Case |
|------|----------|
| `docker-compose.validator.yml` | Production (pulls `ghcr.io/.../trajectoryrl:latest`) |
| `docker-compose.validator-staging.yml` | Staging (pulls `:staging` tag) |
| `docker-compose.validator-dev.yml` | Development (builds locally, bind-mounts source) |

## Troubleshooting

### Validator won't start
```bash
# Check logs for startup sequence
docker compose -f docker/docker-compose.validator.yml logs validator

# Look for:
#   [entrypoint] mock-tools ready
#   [entrypoint] OpenClaw gateway ready
#   [entrypoint] Starting validator...

# Common issues:
# 1. Missing API key → Add CLAWBENCH_LLM_API_KEY to .env.validator
# 2. Wallet not found → Ensure ~/.bittensor/wallets exists
```

### Scores are all identical / evaluations complete instantly
OpenClaw may be failing silently. Check the container logs for OpenClaw errors:
```bash
docker compose -f docker/docker-compose.validator.yml logs validator | grep -i "openclaw\|error"
```

### Network issues
```bash
# Test chain connectivity
docker exec trajectoryrl_validator python -c "import bittensor as bt; print(bt.Subtensor(network='finney'))"
```

## Security

1. **Wallet**: Store on encrypted filesystem, mounted read-only
2. **API Keys**: Use `.env.validator` file (never commit), rotate periodically
3. **Network**: Validator uses host networking; firewall restrict inbound as needed

## Support

- GitHub: https://github.com/trajectoryRL/trajectoryRL/issues
- Website: https://trajrl.com
