# TrajectoryRL

> **Bittensor Subnet 11** — An open skill factory for AI agents

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Bittensor](https://img.shields.io/badge/bittensor-7.0+-green.svg)](https://github.com/opentensor/bittensor)

TrajectoryRL leverages Bittensor's distributed compute and incentive layer with reinforcement learning to produce state-of-the-art **agent skills**. Miners compete every epoch to submit policies that pass strict safety and correctness gates at the lowest cost, validators evaluate them on real knowledge-worker tasks, and the winning policies are released as skills any AI agent can use.

End users consume those skills through the official CLI, [`trajrl`](https://github.com/trajectoryRL/trajrl) — `pip install trajrl` and any agent (Claude Code, Cursor, Codex, OpenClaw, Hermes, Manus, …) gets immediate access to the latest catalog.

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
- **Two-phase evaluation** — [ClawBench](https://github.com/trajectoryRL/clawbench) scenarios with fixed fixtures; LLM-as-judge scores trajectories against natural-language criteria (Phase 1: pack integrity, Phase 2: trajectory quality)
- **Content-addressed** — Packs identified by SHA256 hash, verified against on-chain commitment
- **Winner-take-all** — Best miner gets 100% of rewards; first-mover advantage protects early innovators
- **Anti-copy** — On-chain block timestamps + NCD similarity detection + first-mover threshold (delta=0.10)

See [INCENTIVE_MECHANISM.md](INCENTIVE_MECHANISM.md) for full scoring, rewards, and anti-gaming details.

### Example ROI (1,000 tasks/day)

```
Unoptimized GLM-5:                       $12,300/month

Stage 1 — Prompt optimization (AGENTS.md tuning):
  Optimized prompts + stop rules:         $3,300/month  (73% reduction)

Stage 2 — Hybrid routing (AGENTS.md + injected skills):
  Multi-LLM dynamic routing:               $900/month  (93% reduction)
    ├─ Qwen 3.5 (Alibaba) handles 40% of sub-tasks (tool calls, lookups)
    ├─ GLM-5 (Z.ai) handles 25% (structured extraction, formatting)
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
CLAWBENCH_DEFAULT_MODEL=zhipu/glm-5
EOF
```

**Supported providers** (any OpenAI-compatible API works):

| Provider | `CLAWBENCH_LLM_BASE_URL` | `CLAWBENCH_DEFAULT_MODEL` |
|----------|--------------------------|---------------------------|
| [Zhipu AI](https://bigmodel.cn) (default) | `https://open.bigmodel.cn/api/paas/v4` | `zhipu/glm-5` |
| [Chutes](https://chutes.ai) | `https://llm.chutes.ai/v1` | `chutes/zai-org/GLM-5-TEE` |
| [OpenRouter](https://openrouter.ai) | `https://openrouter.ai/api/v1` | `openrouter/zhipu/glm-5` |

| Variable | Required | Description |
|----------|:--------:|-------------|
| `WALLET_NAME` | Yes | Bittensor wallet name |
| `WALLET_HOTKEY` | Yes | Hotkey name (usually `default`) |
| `NETUID` | Yes | Subnet UID (`11`) |
| `NETWORK` | Yes | `finney`, `test`, or `local` |
| `CLAWBENCH_LLM_API_KEY` | Yes | API key for the LLM provider (e.g. [Zhipu AI](https://bigmodel.cn), [Chutes](https://chutes.ai), [OpenRouter](https://openrouter.ai)) |
| `CLAWBENCH_LLM_BASE_URL` | Yes | Base URL for the OpenAI-compatible API |
| `CLAWBENCH_DEFAULT_MODEL` | Yes | LLM model for evaluation (default: `zhipu/glm-5`) |
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

### For Miners

Mining means writing **policy packs** — system prompts, tool usage rules, and stop conditions — that make AI agents perform tasks safely and cheaply. No GPU, no server, no uptime required.

> **IP Notice:** All policy packs submitted to TrajectoryRL are published to public repositories and licensed under the [MIT License](LICENSE). By submitting a pack, you agree that your submission is freely available for anyone — including TrajectoryRL, other miners, and third parties — to use, modify, and redistribute. Do not submit content you are not willing to release publicly under MIT.

#### 1. Prerequisites (one-time)

```bash
pip install bittensor-cli

btcli wallet create --wallet-name my-miner
btcli subnets register --wallet-name my-miner --hotkey default --netuid 11
```

#### 2. Configure environment

```bash
cat > .env.miner <<'EOF'
WALLET_NAME=my-miner
WALLET_HOTKEY=default
NETUID=11
NETWORK=finney
LLM_API_KEY=your-api-key
LLM_BASE_URL=https://open.bigmodel.cn/api/paas/v4/
LLM_MODEL=zhipu/glm-5
EOF
```

> **Tip:** Any OpenAI-compatible provider works. For OpenRouter, use `LLM_BASE_URL=https://openrouter.ai/api/v1` and `LLM_MODEL=zhipu/glm-5`.

#### 3. Start mining

```bash
git clone https://github.com/trajectoryRL/trajectoryRL.git
cd trajectoryRL
pip install -e .

# Run in default mode: generates AGENTS.md → builds pack → uploads → submits
python neurons/miner.py run --mode default
```

> **Note**: Simply letting the LLM randomly generate AGENTS.md may not get you a good score. You need to actively optimize and improve your policy pack — study the ClawBench scenarios, understand what makes an agent perform well, and iteratively refine your prompts, tool rules, and stop conditions.

#### 4. Manual operations (optional)

```bash
# Build pack from your own AGENTS.md
python neurons/miner.py build --agents-md ./AGENTS.md -o pack.json

# Validate pack locally
python neurons/miner.py validate pack.json

# Check on-chain status
python neurons/miner.py status
```

#### 5. Local testing with ClawBench

```bash
cd clawbench
pip install -e .
# Set CLAWBENCH_LLM_API_KEY, CLAWBENCH_LLM_BASE_URL, CLAWBENCH_DEFAULT_MODEL in .env
# Example Zhipu:      CLAWBENCH_LLM_BASE_URL=https://open.bigmodel.cn/api/paas/v4/, CLAWBENCH_DEFAULT_MODEL=zhipu/glm-5
# Example Chutes:     CLAWBENCH_LLM_BASE_URL=https://llm.chutes.ai/v1,              CLAWBENCH_DEFAULT_MODEL=chutes/zai-org/GLM-5-TEE
# Example OpenRouter: CLAWBENCH_LLM_BASE_URL=https://openrouter.ai/api/v1,           CLAWBENCH_DEFAULT_MODEL=openrouter/zhipu/glm-5

# Test a single scenario
python scripts/run_episode.py --scenario inbox_triage --variant optimized --json

# Test all scenarios
python scripts/run_batch.py
```

See [MINER_OPERATIONS.md](MINER_OPERATIONS.md) for full details: automated mode, S3 upload, pack format, and scoring targets.

## trajrl — the official CLI for TrajectoryRL skills

TrajectoryRL is a skill factory: miners compete every epoch to produce policy packs that pass safety and correctness gates at the lowest cost, and the winning packs become **skills** — the subnet's product. `trajrl` is the official command-line tool that delivers those skills to end users.

```bash
pip install trajrl
```

One install gives any human or AI agent (Claude Code, Cursor, Codex, OpenClaw, Hermes, Manus, …) access to every skill the subnet has shipped. Each skill is a self-contained `SKILL.md` that agents can discover and follow directly. CLI output is JSON when piped, Rich tables when interactive.

```bash
trajrl subnet status                       # Network overview
trajrl subnet analyze <validator-hotkey>   # Full validator analysis
trajrl subnet analyze <hotkey> --deep      # Drill into top miners
```

Source, skill catalog, and full documentation: https://github.com/trajectoryRL/trajrl

## Documentation

- **[Incentive Mechanism](INCENTIVE_MECHANISM.md)** — Scoring, rewards, winner-take-all, and anti-copy protection
- **[Validator Operations](VALIDATOR_OPERATIONS.md)** — Cost model, auto-updates, and operational guidance
- **[Miner Operations](MINER_OPERATIONS.md)** — Pack format, run modes, local testing, and submission workflow
- **[ClawBench](https://github.com/trajectoryRL/clawbench)** — Evaluation framework (scenarios, fixtures, scoring)
- **[trajrl](https://github.com/trajectoryRL/trajrl)** — Official CLI delivering the subnet's skills to end users

## Community

- **GitHub**: https://github.com/trajectoryRL/trajectoryRL
- **Website**: https://trajrl.com

## License

This project is licensed under the [MIT License](LICENSE).

All miner-submitted policy packs are public and released under the same MIT License. By participating as a miner, you acknowledge that your submissions become open-source contributions available to everyone.

---

**Built on [Bittensor](https://bittensor.com)** | **Powered by [ClawBench](https://github.com/trajectoryRL/clawbench)**
