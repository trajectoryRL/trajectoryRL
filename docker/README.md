# TrajectoryRL Docker Deployment

Docker is the **recommended way** to run TrajectoryRL validators and miners. A single `docker compose up` starts all required services (validator, OpenClaw gateway, mock-tools).

## Quick Start — Validator

```bash
# 1. Clone repo with submodules
git clone --recurse-submodules https://github.com/trajectoryRL/trajectoryRL.git
cd trajectoryRL

# 2. Configure
cp .env.example .env.validator
# Edit .env.validator: set ANTHROPIC_API_KEY, WALLET_NAME, WALLET_HOTKEY

# 3. Start (all services in one command)
docker compose -f docker/docker-compose.validator.yml --env-file .env.validator up -d

# 4. View logs
docker compose -f docker/docker-compose.validator.yml logs -f validator
```

This starts 5 services automatically:
1. **init** — sets up workspace files (runs once, exits)
2. **init-perms** — fixes workspace permissions (runs once, exits)
3. **mock-tools** — deterministic tool responses from fixtures (port 3001)
4. **openclaw-gateway** — OpenClaw with clawbench-tools plugin (port 18790)
5. **validator** — reads on-chain commitments, evaluates packs, sets weights

## Configuration

Create `.env.validator` in the repo root:

```bash
# Required
WALLET_NAME=validator
WALLET_HOTKEY=default
NETUID=11
NETWORK=finney
ANTHROPIC_API_KEY=sk-ant-...

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

# View all service logs
docker compose -f docker/docker-compose.validator.yml logs -f

# Stop
docker compose -f docker/docker-compose.validator.yml down

# Rebuild after updates
git pull && git submodule update --init --recursive
docker compose -f docker/docker-compose.validator.yml --env-file .env.validator up -d --build
```

## Architecture

### Service Dependencies

```
init → init-perms → mock-tools (healthy) → openclaw-gateway (healthy) → validator
```

- **mock-tools + openclaw-gateway** run on an internal `sandbox-net` Docker network
- **validator** uses `network_mode: host` to reach mock-tools (localhost:3001) and OpenClaw (localhost:18790) via published ports
- A shared `workspace` Docker volume connects init → openclaw-gateway → validator

### Validator Container
- Base: Python 3.10-slim
- ClawBench: Baked in via submodule (version-pinned)
- Volumes:
  - `~/.bittensor/wallets` → Wallet access (read-only)
  - `workspace` → Shared with OpenClaw for AGENTS.md

### OpenClaw Gateway
- Runs the ClawBench evaluation sandbox
- Port: 18790

### Mock Tools Server
- Serves deterministic tool responses from fixtures
- Port: 3001

## Troubleshooting

### Validator won't start
```bash
# Check all service logs
docker compose -f docker/docker-compose.validator.yml logs

# Common issues:
# 1. Missing API key → Add ANTHROPIC_API_KEY to .env.validator
# 2. Wallet not found → Ensure ~/.bittensor/wallets exists
# 3. OpenClaw permission error → Remove workspace volume and restart:
#    docker volume rm docker_workspace && docker compose ... up -d
```

### OpenClaw permission denied (SOUL.md / AGENTS.md)
The init-perms container fixes this automatically. If it persists:
```bash
docker compose -f docker/docker-compose.validator.yml down
docker volume rm docker_workspace
docker compose -f docker/docker-compose.validator.yml --env-file .env.validator up -d
```

### Scores are all identical / evaluations complete instantly
This means OpenClaw is failing silently. Check its logs:
```bash
docker compose -f docker/docker-compose.validator.yml logs openclaw-gateway
```

### Network issues
```bash
# Test chain connectivity
docker exec trajectoryrl_validator python -c "import bittensor as bt; print(bt.Subtensor(network='finney'))"
```

## Updating ClawBench

ClawBench is pinned as a git submodule. The validator config also hardcodes
the commit hash as a safety check. Both must be updated together.

```bash
# 1. Update the submodule
cd clawbench && git fetch origin && git checkout origin/main && cd ..

# 2. Update the hash in trajectoryrl/utils/config.py (clawbench_commit)
git -C clawbench rev-parse HEAD
# Edit config.py with the new hash

# 3. Stage and commit
git add clawbench trajectoryrl/utils/config.py
git commit -m "Update ClawBench submodule"

# 4. Rebuild
docker compose -f docker/docker-compose.validator.yml --env-file .env.validator up -d --build
```

## Security

1. **Wallet**: Store on encrypted filesystem, mounted read-only
2. **API Keys**: Use `.env.validator` file (never commit), rotate periodically
3. **Network**: Validator uses host networking; firewall restrict inbound as needed

## Support

- GitHub: https://github.com/trajectoryRL/trajectoryRL/issues
- Website: https://trajrl.com
