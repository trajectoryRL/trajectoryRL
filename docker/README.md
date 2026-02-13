# TrajectoryRL Docker Deployment

Docker is the **recommended way** to run TrajectoryRL validators and miners. It ensures consistent environments, automatic ClawBench version pinning (v0.3.0), and simplified deployment.

## Quick Start

### Validator

```bash
# 1. Clone repo
git clone --recursive https://github.com/trajectoryRL/trajectoryrl.git
cd trajectoryrl

# 2. Configure
cp .env.example .env
# Edit: add ANTHROPIC_API_KEY, WALLET_NAME

# 3. Start
docker-compose up -d validator

# 4. View logs
docker-compose logs -f validator
```

### Miner

```bash
# 1. Prepare pack
mkdir -p packs
cp /path/to/pack.json packs/

# 2. Configure
cp .env.example .env
# Edit: add PACK_REPO, PACK_COMMIT

# 3. Start
docker-compose --profile miner up -d miner

# 4. View logs
docker-compose logs -f miner
```

## Configuration

Edit `.env` file:

```bash
# Bittensor
WALLET_NAME=validator
WALLET_HOTKEY=default
NETUID=11
NETWORK=finney

# Validator
ANTHROPIC_API_KEY=sk-ant-...
EPOCH_INTERVAL=720
LOG_LEVEL=INFO

# Miner
PACK_REPO=https://github.com/user/repo
PACK_COMMIT=abc123def456...
```

## Operations

```bash
# Start services
docker-compose up -d validator                    # Validator only
docker-compose --profile miner up -d              # Both

# View logs
docker-compose logs -f validator
docker-compose logs --tail=100 validator

# Stop services
docker-compose stop validator
docker-compose down

# Rebuild after updates
git pull && git submodule update --init --recursive
docker-compose build validator
docker-compose up -d validator
```

## Architecture

### Validator Container
- Base: Python 3.10-slim
- ClawBench: Pinned to v0.3.0 (b718230) via submodule
- Volumes:
  - `~/.bittensor/wallets` → Wallet access (ro)
  - `./logs` → Persistent logs
  - `./git_cache` → GitHub verification cache

### Miner Container
- Base: Python 3.10-slim
- Volumes:
  - `~/.bittensor/wallets` → Wallet access (ro)
  - `./packs` → Policy packs (ro)
  - `./logs` → Persistent logs

## Troubleshooting

### Validator won't start
```bash
# Check logs
docker-compose logs validator

# Common fixes:
# 1. Missing API key → Add ANTHROPIC_API_KEY to .env
# 2. Wallet not found → Ensure ~/.bittensor/wallets exists
# 3. Version mismatch → Rebuild: docker-compose build --no-cache validator
```

### Miner submission fails
```bash
# Check environment
docker exec trajectoryrl_miner env | grep PACK

# Verify pack file
docker exec trajectoryrl_miner ls -la /app/packs/

# Ensure PACK_REPO and PACK_COMMIT are set in .env
```

### Network issues
```bash
# Test connectivity
docker exec trajectoryrl_validator ping -c 3 api.bittensor.com

# Check subtensor
docker exec trajectoryrl_validator python -c "import bittensor as bt; print(bt.subtensor(network='finney'))"
```

## Production

### Systemd service

```bash
# /etc/systemd/system/trajectoryrl-validator.service
[Unit]
Description=TrajectoryRL Validator
After=docker.service
Requires=docker.service

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=/path/to/trajectoryrl
ExecStart=/usr/local/bin/docker-compose up -d validator
ExecStop=/usr/local/bin/docker-compose down

[Install]
WantedBy=multi-user.target
```

### Log rotation

```bash
# /etc/logrotate.d/trajectoryrl
/path/to/trajectoryrl/logs/*.log {
    daily
    rotate 7
    compress
    missingok
    notifempty
}
```

## Security

1. **Wallet**: Store on encrypted filesystem, use read-only mounts
2. **API Keys**: Use `.env` file (never commit), rotate periodically
3. **Network**: Run on isolated network, firewall restrict inbound

## Updating ClawBench

```bash
# When new ClawBench version released:
git pull && git submodule update --remote clawbench
# Update trajectoryrl/utils/config.py → clawbench_commit
docker-compose build --no-cache validator
docker-compose up -d validator
```

## Support

- GitHub: https://github.com/trajectoryRL/trajectoryRL/issues
- Website: https://trajrl.com

---

**Recommendation**: Always use Docker for production. Manual installation is for development only.
