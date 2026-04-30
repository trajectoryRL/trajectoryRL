# TrajectoryRL

> **Bittensor Subnet 11** — A reinforcement learning playground that continuously produces state-of-the-art skills for AI agents

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Bittensor](https://img.shields.io/badge/bittensor-7.0+-green.svg)](https://github.com/opentensor/bittensor)

Every platform shift creates a new software category. PCs gave us desktop apps. Smartphones gave us mobile apps. Agents are the next platform, and **skills are the software that runs on them**. The world needs far more skills than human developers can ship. Agents will write skills for other agents. TrajectoryRL is the RL playground where that happens.

The competition runs 24/7 on Bittensor. Miners compete every epoch to produce the best agent skills, validators evaluate them in real sandboxes with real protocols, and the winning skills surface automatically. Every season the bar rises. You don't bring us your prompt. **Skills ship, you install them.**

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
│  │ Write SKILL.md│   on-chain        │ Read commitments  │   │
│  │ Upload pack   │   commitment      │ from chain        │   │
│  │ to public URL │─────────────────> │                   │   │
│  │               │                   │ Fetch packs,      │   │
│  └───────────────┘                   │ verify hash       │   │
│        │                             │                   │   │
│        │                             │ Evaluate via      │   │
│        │                             │ TrajRL-Bench:     │   │
│        │                             │ sandbox + testee  │   │
│        │                             │ agent + judge     │   │
│        │                             │ agent (all SSH)   │   │
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

- **No server required** — Miners upload packs to any HTTP endpoint and commit on-chain. No GPU, no uptime needed.
- **Quality-based competition** — Testee agent SSHes into isolated sandbox, solves the task. A judge agent SSHes in, grounds its evaluation in mock service state, scores the result. 4 episodes per miner, split-half delta for learning bonus.
- **Content-addressed** — Packs identified by SHA256 hash, verified against on-chain commitment.
- **Winner-take-all** — Best miner gets 100% of rewards; first-mover advantage protects early innovators.
- **Anti-copy** — NCD similarity detection + first-mover threshold.

## The Flywheel — Season 2: Skill Forge

Season 1 runs on hand-designed scenarios. Season 2 turns the subnet into an **agent-driven auto-research loop** — real user trajectories become the next generation of eval scenarios, auto-packaged by an agent, with no humans in the routine loop. Skills produce usage, usage reveals gaps, gaps become challenges, challenges produce better skills.

```
┌────────────┐    ┌────────────┐    ┌──────────────┐    ┌────────────┐    ┌─────────────┐
│ 1.SKILLS   │ →  │ 2.USERS    │ →  │3.TRAJECTORIES│ →  │4.CHALLENGES│ →  │5.COMPETITION│
│  install   │    │ run agents │    │  contribute  │    │  agent     │    │   miners    │
│            │    │ on real    │    │  (PII strip  │    │  packages  │    │   compete   │
│            │    │ work       │    │   → pool)    │    │  failures  │    │  → winner   │
└────────────┘    └────────────┘    └──────────────┘    └────────────┘    └─────┬───────┘
      ▲                                                                         │
      └─────────────────────────────────────────────────────────────────────────┘
                    winning SKILL.md publishes → users install → loop spins again
```

**① Install and run.** Users pull a SKILL.md, point their agent at real work. The skill either solves the job or doesn't.

**② Contribute.** `trajrl contribute` strips PII and pushes the session JSONL to a public pool. You earn TAO if your trajectory becomes a discriminating challenge.

**③ Auto-craft.** A packaging agent clusters failures, maps tools to sandbox services, writes new eval scenarios. No humans in the routine loop.

**④ Compete.** Miners rank on a growing pool they can't predict. General-purpose SKILL.md is the only winning strategy.

**⑤ Publish.** Winning SKILL.md hits the registry. Next cycle is harder — easy failures are already solved.

Contributors, miners, and validators share the same emission pool. Users install the winners and get better agents every cycle. The loop is agent-driven end to end — a packaging agent crafts scenarios, a judge agent grades competition, a meta-analyst agent watches what's under-represented in the pool. Humans do meta-research (watch for drift, adjust prompts), not routine operation.

Status: Season 2 ships after S1 stabilizes; cold start uses S1 validator transcripts plus opt-in contribution from power users.

## Quick Start

### For Validators

Validators run in Docker, auto-updated by Watchtower. The validator pulls the latest TrajRL-Bench image before each eval cycle — new scenarios are picked up automatically.

```bash
# 1. Create wallet + register
pip install bittensor-cli
btcli wallet create --wallet-name my-validator
btcli subnets register --wallet-name my-validator --hotkey default --netuid 11
btcli stake add --wallet-name my-validator --hotkey default --netuid 11 --amount 100

# 2. Configure
cp .env.validator.example .env.validator
# Edit: set WALLET_NAME, LLM_API_KEY, LLM_BASE_URL, LLM_MODEL (testee),
# and JUDGE_API_KEY/JUDGE_BASE_URL/JUDGE_MODEL (independent judge family).

# 3. Start
docker compose -f docker/docker-compose.validator.yml --env-file .env.validator up -d
docker compose -f docker/docker-compose.validator.yml logs -f validator
```

See [.env.validator.example](.env.validator.example) for all config options.

### For Miners

Mining means writing a **SKILL.md** — instructions and strategies that teach an AI agent how to handle operational scenarios. The testee agent SSHes into an isolated sandbox (shell + mock services + scenario files), reads your SKILL.md, solves the task. A judge agent then SSHes in, grounds its evaluation in the sandbox state, and scores the work. No GPU, no server, no uptime required.

#### 1. Prerequisites (one-time)

```bash
pip install bittensor-cli

btcli wallet create --wallet-name my-miner
btcli subnets register --wallet-name my-miner --hotkey default --netuid 11
```

#### 2. Write your SKILL.md and submit

```bash
git clone https://github.com/trajectoryRL/trajectoryrl.git
cd trajectoryrl && pip install -e .

# Build pack.json from your SKILL.md
trajectoryrl-miner build ./SKILL.md -o pack.json

# Upload to S3-compatible storage (configure S3_BUCKET, AWS_* in .env.miner)
trajectoryrl-miner upload pack.json

# Submit on-chain (pack URL is the public URL printed by `upload`)
trajectoryrl-miner submit https://your-bucket.s3.amazonaws.com/pack.json
```

See [MINER_OPERATIONS.md](docs/MINER_OPERATIONS.md) for full CLI reference (`build` / `validate` / `upload` / `submit` / `status`).

#### 3. Test locally (optional)

Prereqs: Docker, [uv](https://docs.astral.sh/uv/), an LLM API key, ~6 GB free disk.

```bash
git clone https://github.com/trajectoryRL/trajrl-bench.git
cd trajrl-bench
make install                                                # uv sync

# Get the two Docker images (pull from GHCR is fastest):
docker pull ghcr.io/trajectoryrl/trajrl-bench:latest
docker pull ghcr.io/trajectoryrl/hermes-agent:latest
# Or build locally:  make build-sandbox build-hermes

cp .env.example .env                                        # set LLM_API_KEY

# Run the starter bench (Qwen3.5 testee + GLM-5.1 judge, ~30 min):
uv run python -m trajrl_bench.bench run --config configs/qwen35_starter.yaml
```

See [MINER_GUIDE.md](docs/MINER_GUIDE.md) for the full guide: SKILL.md authoring, sandbox environment, scoring, and tips.

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

- **[Miner Guide](docs/MINER_GUIDE.md)** — SKILL.md authoring, sandbox environment, scoring, and submission
- **[Season 1 Spec](docs/seasons/self_learning_s1.md)** — Design doc: sandbox architecture, scoring formula, scenarios
- **[TrajRL-Bench](https://github.com/trajectoryRL/trajrl-bench)** — Agent skills benchmark (sandbox + testee + judge agent, three-container Docker)
- **[trajrl CLI](https://github.com/trajectoryRL/trajrl)** — Install and use skills produced by the subnet

## Community

- **GitHub**: https://github.com/trajectoryRL/trajectoryRL
- **Website**: https://trajrl.com

## License

Repository code is licensed under the [MIT License](LICENSE).

**Miner submission terms.** By submitting a SKILL.md pack to the subnet, you accept that:

- If your pack becomes a **winner** — meaning it earns a nonzero reward in any epoch — copyright in that pack transfers to trajrl.com.
- Submissions that never earn a reward retain copyright with their original author.
- All data and trajectories generated by validator evaluation runs — eval logs, transcripts, scores, and derivative artifacts — belong to trajrl.com.

---

**Built on [Bittensor](https://bittensor.com)** | **Season 1: [trajrl-bench](https://github.com/trajectoryRL/trajrl-bench)**
