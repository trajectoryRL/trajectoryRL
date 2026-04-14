# TrajectoryRL

> **Bittensor Subnet 11** — A reinforcement learning playground that continuously produces state-of-the-art skills for AI agents

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Bittensor](https://img.shields.io/badge/bittensor-7.0+-green.svg)](https://github.com/opentensor/bittensor)

Every platform shift creates a new software category. PCs gave us desktop apps. Smartphones gave us mobile apps. Agents are the next platform, and **skills are the software that runs on them**. The volume of skills the world needs is far beyond what human developers can produce. Agents will write skills for other agents. TrajectoryRL is the RL playground where that happens.

The competition runs 24/7 on Bittensor. Miners compete every epoch to produce the best agent skills, validators evaluate them in real sandboxes with real protocols, and the winning skills surface automatically. Every season the bar rises. You don't bring us your prompt. **Skills come out, you install them.**

```bash
pip install trajrl
```

One install gives any agent (Claude Code, Cursor, Codex, OpenClaw, Hermes, Manus, …) access to every skill the subnet has shipped. Source, catalog, and docs: [`trajrl`](https://github.com/trajectoryRL/trajrl).

## Overview

```
┌──────────────────────────────────────────────────────────────┐
│                   TRAJECTORYRL SUBNET (SN11)                 │
│                                                              │
│  MINERS                              VALIDATORS              │
│  ┌───────────────┐                   ┌───────────────────┐   │
│  │ Upload        │   on-chain        │ Read commitments  │   │
│  │ pack.json to  │   commitment      │ from chain        │   │
│  │ public HTTP   │─────────────────> │                   │   │
│  │ endpoint      │                   │ Fetch packs via   │   │
│  └───────────────┘                   │ HTTP, verify      │   │
│        │                             │ hash + timestamp  │   │
│        │                             │                   │   │
│        │                             │ Evaluate via      │   │
│        │                             │ ClawBench         │   │
│        │                             └───────────────────┘   │
│        │                                      │              │
│        │                                      │ set_weights  │
│        ▼                                      ▼              │
│  ┌──────────────────────────────────────────────────────┐    │
│  │              BITTENSOR BLOCKCHAIN                    │    │
│  │   Commitments, weights, TAO rewards                  │    │
│  └──────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────┘
```

- **No server required** — Miners upload packs to any HTTP endpoint and commit metadata on-chain. No public IP, no uptime needed.
- **Season 1 evaluation** — Agent SSHes into isolated sandbox with mock services (email, Slack, Notion, calendar, Gitea). 100% LLM judge scoring across 4 episodes. Quality-based competition.
- **Content-addressed** — Packs identified by SHA256 hash, verified against on-chain commitment
- **Winner-take-all** — Best miner gets 100% of rewards; first-mover advantage protects early innovators
- **Anti-copy** — On-chain block timestamps + NCD similarity detection + first-mover threshold (delta=0.10)

See [INCENTIVE_MECHANISM.md](INCENTIVE_MECHANISM.md) for full scoring, rewards, and anti-gaming details.

### Example ROI (1,000 tasks/day)

```
Unoptimized GLM-5.1:                       $12,300/month

Stage 1 — Prompt optimization (AGENTS.md tuning):
  Optimized prompts + stop rules:         $3,300/month  (73% reduction)

Stage 2 — Hybrid routing (AGENTS.md + injected skills):
  Multi-LLM dynamic routing:               $900/month  (93% reduction)
    ├─ Qwen 3.5 (Alibaba) handles 40% of sub-tasks (tool calls, lookups)
    ├─ GLM-5.1 (Z.ai) handles 25% (structured extraction, formatting)
    ├─ Gemini 3 Flash (Google) handles 20% (search, summarization)
    ├─ GPT-5.2 (OpenAI) handles 10% (reasoning, drafting)
    └─ Claude Opus 4.6 (Anthropic) handles 5% (complex judgment calls)
```

## Quick Start

### For Validators

Validators run via Docker with automatic updates from GHCR via Watchtower. When new code is pushed to `prod`, GitHub Actions builds a new image and Watchtower auto-pulls and restarts within 5 minutes.

#### 1. Prerequisites (one-time)

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

#### 2. Configure environment

```bash
cat > .env.validator <<'EOF'
WALLET_NAME=my-validator
WALLET_HOTKEY=default
NETUID=11
NETWORK=finney
CLAWBENCH_LLM_API_KEY=your-api-key
CLAWBENCH_LLM_BASE_URL=https://open.bigmodel.cn/api/paas/v4/
CLAWBENCH_DEFAULT_MODEL=zhipu/glm-5.1
EOF
```

**Supported providers** (any OpenAI-compatible API works):

| Provider | `CLAWBENCH_LLM_BASE_URL` | `CLAWBENCH_DEFAULT_MODEL` |
|----------|--------------------------|---------------------------|
| [Zhipu AI](https://bigmodel.cn) (default) | `https://open.bigmodel.cn/api/paas/v4` | `zhipu/glm-5.1` |
| [Chutes](https://chutes.ai) | `https://llm.chutes.ai/v1` | `chutes/zai-org/GLM-5.1-TEE` |
| [OpenRouter](https://openrouter.ai) | `https://openrouter.ai/api/v1` | `openrouter/z-ai/glm-5.1` |

| Variable | Required | Description |
|----------|:--------:|-------------|
| `WALLET_NAME` | Yes | Bittensor wallet name |
| `WALLET_HOTKEY` | Yes | Hotkey name (usually `default`) |
| `NETUID` | Yes | Subnet UID (`11`) |
| `NETWORK` | Yes | `finney`, `test`, or `local` |
| `CLAWBENCH_LLM_API_KEY` | Yes | API key for the LLM provider (e.g. [Zhipu AI](https://bigmodel.cn), [Chutes](https://chutes.ai), [OpenRouter](https://openrouter.ai)) |
| `CLAWBENCH_LLM_BASE_URL` | Yes | Base URL for the OpenAI-compatible API |
| `CLAWBENCH_DEFAULT_MODEL` | Yes | LLM model for evaluation (default: `zhipu/glm-5.1`) |
| `JUDGE_MODEL` | No | LLM model for judge (defaults to `CLAWBENCH_DEFAULT_MODEL`) |
| `JUDGE_API_KEY` | No | API key for judge (defaults to `CLAWBENCH_LLM_API_KEY`) |
| `JUDGE_BASE_URL` | No | Base URL for judge (defaults to `CLAWBENCH_LLM_BASE_URL`) |

#### 3. Start validator

```bash
# Start validator + Watchtower (auto-updates from GHCR)
docker compose -f docker/docker-compose.validator.yml --env-file .env.validator up -d

# View logs
docker compose -f docker/docker-compose.validator.yml logs -f validator
```

The Docker container reads wallet keyfiles from the mounted `~/.bittensor/wallets/` directory. No btcli is needed inside the container.

> **Tip:** Watchtower checks for new images every 5 minutes. To update immediately:
> ```bash
> docker compose -f docker/docker-compose.validator.yml pull
> docker compose -f docker/docker-compose.validator.yml --env-file .env.validator up -d
> ```

See [VALIDATOR_OPERATIONS.md](VALIDATOR_OPERATIONS.md) for cost model, auto-update details, and operational guidance.

### For Miners (Season 1)

Mining means writing a **SKILL.md** — instructions and strategies that teach an AI agent how to handle operational scenarios. The agent SSHes into an isolated sandbox with mock services, executes tasks, and gets scored by an LLM judge on quality. No GPU, no server, no uptime required.

> **IP Notice:** All policy packs submitted to TrajectoryRL are published to public repositories and licensed under the [MIT License](LICENSE). By submitting a pack, you agree that your submission is freely available for anyone — including TrajectoryRL, other miners, and third parties — to use, modify, and redistribute. Do not submit content you are not willing to release publicly under MIT.

#### 1. Prerequisites (one-time)

```bash
pip install bittensor-cli

btcli wallet create --wallet-name my-miner
btcli subnets register --wallet-name my-miner --hotkey default --netuid 11
```

#### 2. Write your SKILL.md and submit

```bash
pip install trajrl

# Build pack from your SKILL.md
trajrl pack build --skill-md ./SKILL.md

# Submit on-chain
trajrl pack submit --url https://your-host.com/pack.json
```

#### 3. Test locally (optional)

```bash
git clone https://github.com/trajectoryRL/trajectory-sandbox.git
cd trajectory-sandbox
pip install -e ".[dev]"
make build                # build sandbox + agent Docker images
cp .env.example .env      # add your LLM API key
make test-hermes          # run one episode with real agent + real judge
```

See [MINER_GUIDE_S1.md](MINER_GUIDE_S1.md) for the full guide: SKILL.md authoring, sandbox environment, scoring, and tips.

> **v4.0 miners**: The previous AGENTS.md + ClawBench flow is documented in [MINER_OPERATIONS.md](MINER_OPERATIONS.md). Season 1 replaces this with SKILL.md + trajectory-sandbox.

## trajrl — consume what the playground produces

You don't interact with the competition. You consume its output. `trajrl` is the CLI that delivers battle-tested skills to your agent.

```bash
pip install trajrl
trajrl subnet status                       # Network overview
trajrl subnet analyze <validator-hotkey>   # Full validator analysis
trajrl subnet analyze <hotkey> --deep      # Drill into top miners
```

Each skill is a self-contained `SKILL.md` that any agent can discover and follow directly. CLI output is JSON when piped, Rich tables when interactive.

Source, skill catalog, and full documentation: https://github.com/trajectoryRL/trajrl

## Documentation

- **[Season 1 Miner Guide](MINER_GUIDE_S1.md)** — SKILL.md authoring, sandbox environment, scoring, and submission
- **[Season 1 Spec](seasons/self_learning_s1.md)** — Design doc: sandbox architecture, scoring formula, scenarios
- **[trajectory-sandbox](https://github.com/trajectoryRL/trajectory-sandbox)** — SSH sandbox for S1 evaluations (mock services, LLM judge, Docker)
- **[Incentive Mechanism](INCENTIVE_MECHANISM.md)** — Scoring, rewards, winner-take-all, and anti-copy protection
- **[Validator Operations](VALIDATOR_OPERATIONS.md)** — Cost model, auto-updates, and operational guidance
- **[Miner Operations (v4.0)](MINER_OPERATIONS.md)** — Legacy: AGENTS.md pack format, ClawBench testing
- **[ClawBench (v4.0)](https://github.com/trajectoryRL/clawbench)** — Legacy evaluation framework
- **[trajrl](https://github.com/trajectoryRL/trajrl)** — Official CLI delivering the subnet's skills to end users

## Community

- **GitHub**: https://github.com/trajectoryRL/trajectoryRL
- **Website**: https://trajrl.com

## License

This project is licensed under the [MIT License](LICENSE).

All miner-submitted policy packs are public and released under the same MIT License. By participating as a miner, you acknowledge that your submissions become open-source contributions available to everyone.

---

**Built on [Bittensor](https://bittensor.com)** | **Season 1: [trajectory-sandbox](https://github.com/trajectoryRL/trajectory-sandbox)**
