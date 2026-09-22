# TrajectoryRL

> **Bittensor Subnet 11.** Season 2 (the transition season) is live: miners compete on **fusion policies**, programs
> that decide, per request, which open-weight models on Engy answer an agent and in what combination. Where SN11 goes
> next is an open protocol for verifiable environments: [docs/ROADMAP.md](docs/ROADMAP.md).

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Bittensor](https://img.shields.io/badge/bittensor-7.0+-green.svg)](https://github.com/opentensor/bittensor)

TrajectoryRL runs a continuous competition on Bittensor. Validators evaluate every submission in real sandboxes
against tasks with hidden tests, consensus is stake-weighted, and the best submission holds the seat until a better
one takes it. What the competition produces is deployable: the seated fusion policy will be served on Engy as its
auto mode (`engy/auto`), for any customer and any harness.

## How Season 2 works

```
 miner                          validator (per scenario)                              Engy
 ┌────────────┐   web-submit    ┌─────────────────────────────────────────────┐
 │ SKILL.md   │ ─────────────►  │  scenario container: Hermes + the task      │
 │ policy.py  │                 │        │  OpenAI-compatible calls           │
 │ (or .json) │                 │        ▼                                    │
 └────────────┘                 │  policy sidecar: the miner's fusion policy  │  metered calls
                                │        │  only route: the meter             │ ─────────────►  8 open-weight models
                                │        ▼                                    │   receipts
                                │  meter: allowlist, $1 cap, cost, provenance │ ◄─────────────
                                │        │                                    │
                                │  verifier container: hidden tests → score   │
                                └─────────────────────────────────────────────┘
```

- **What you submit**: a pack with your `SKILL.md` and a fusion policy (`policy.json` with no code, `policy.py` on a
  small SDK, or a raw OpenAI-compatible server). The policy runs in a sidecar next to the scenario container and can
  call any of the eight allowlisted models, in sequence or in parallel, rewrite what they see, keep session state,
  and stream the answer back.
- **How it is scored**: the sum over 26 scenarios of tests passed over tests total, from a fresh verifier container.
  Cost is metered on every model call at a frozen price table and shown per scenario and per model; it is not in the
  score. A $1 per-scenario safety cap bounds spending.
- **Consensus**: stake-weighted, Winsorized across validators; the seat changes hands only on a real margin.
- **No GPU, no server, no uptime**: the platform stores your pack; validators run everything.

Full guide, with a quick start that was run end to end before launch: **[docs/FUSION_POLICY.md](docs/FUSION_POLICY.md)**.

## Quick start for miners

```bash
git clone https://github.com/trajectoryRL/trajectoryRL && cd trajectoryRL && pip install -e .

# a directory with policy.json or policy.py (examples with measured scores: examples/policies/)
trajectoryrl-miner build SKILL.md --policy ./my_policy -o pack.json
trajectoryrl-miner validate pack.json

# test with the exact validator code (Docker + your own Engy key; local runs are on your account)
LLM_API_KEY=$ENGY_API_KEY python scripts/eval_pack.py --pack pack.json -o ./out --scenarios git-leak-recovery,postgres-csv-clean

# submit (signed with your hotkey; the platform runs pre-eval and queues the pack)
trajectoryrl-miner web-submit pack.json
```

Baselines measured through the validator code on the 26 scenarios: SKILL.md only (pin qwen3.8-27b) 19.9;
pin glm-5.3-flash 22.6 at $0.56 a session; pin kimi-k3 22.8 at $8.50; kimi-k3 with three cheap advisers 23.6 at $10.02.

## Quick start for validators

Validators run in Docker, auto-updated by Watchtower. Models and endpoints are fixed in code so every validator
scores against the same catalog; the only secret is your Engy key.

```bash
pip install bittensor-cli
btcli wallet create --wallet-name my-validator
btcli subnets register --wallet-name my-validator --hotkey default --netuid 11
btcli stake add --wallet-name my-validator --hotkey default --netuid 11 --amount 100

cp .env.validator.example .env.validator      # set WALLET_NAME and LLM_API_KEY
docker compose -f docker/docker-compose.validator.yml --env-file .env.validator up -d
docker compose -f docker/docker-compose.validator.yml logs -f validator
```

Operations, the Season 2 network model, and spend expectations: [docs/VALIDATOR_OPERATIONS.md](docs/VALIDATOR_OPERATIONS.md).

## Where SN11 is going

Season 2 removes Season 1's single-model ceiling; the rest needs a new platform. An objective becomes an environment
plus a verifier: a reproducible world where every action yields a content-addressed version, and a deterministic
program that scores a version. SN11 becomes the registry and host of such environments: it admits problems that fit,
records every attempt as evidence, verifies by execution and replay, pays for verified progress, and lets the market
grow the checks. Miners bring their own agents; the models are Engy's; the environments and the verdicts are the
subnet's. The order of steps and the reasoning are in [docs/ROADMAP.md](docs/ROADMAP.md).

## Documentation

- **[Fusion policy guide](docs/FUSION_POLICY.md)**: quick start, policy shapes and SDK, allowlist and prices, pack
  format, local testing, anti-gaming
- **[Roadmap](docs/ROADMAP.md)**: Season 1, Season 2, the protocol, what comes next
- **[Miner operations](docs/MINER_OPERATIONS.md)**: CLI reference (`build`, `validate`, `web-submit`, `status`)
- **[Validator operations](docs/VALIDATOR_OPERATIONS.md)**: deployment, Season 2 notes
- **[Scoring and evaluation](docs/EVALUATION_S1.md)** and **[Incentive mechanism](docs/INCENTIVE_MECHANISM.md)**:
  the scoring spec and the consensus protocol
- **[trajrl-bench](https://github.com/trajectoryRL/trajrl-bench)**: the scenario images and verifiers
- **[trajrl CLI](https://github.com/trajectoryRL/trajrl)**: subnet status and analysis

## Community

- **GitHub**: https://github.com/trajectoryRL/trajectoryRL
- **Website**: https://trajrl.com

## License

Repository code is licensed under the [MIT License](LICENSE).

**Miner submission terms.** By submitting a pack to the subnet, you accept that:

- If your pack becomes a **winner**, meaning it earns a nonzero reward in any epoch, copyright in that pack transfers
  to trajrl.com.
- Submissions that never earn a reward retain copyright with their original author.
- All data and trajectories generated by validator evaluation runs, eval logs, transcripts, scores, and derivative
  artifacts, belong to trajrl.com.

---

**Built on [Bittensor](https://bittensor.com)**
