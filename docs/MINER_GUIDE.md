# Miner Guide (Season 2, the transition season)

**Subnet**: SN11 (TrajectoryRL)
**Scoring**: programmatic verifier, quality summed across the active scenario set
**Spec**: 25 (26 scenarios)

> Mining means shipping a **fusion policy** next to your `SKILL.md`: a program that sits between the agent harness
> and the Engy model catalog and decides, per request, which open-weight models answer, in what combination (route,
> escalate, ask advisers, ensemble), and what they are sent. The best policy wins the seat, and we will serve the
> seated policy on Engy as its auto mode. The detailed guide, with the SDK, prices and examples, is
> [FUSION_POLICY.md](FUSION_POLICY.md); this page is the shorter operational view.

---

## How It Works

One Docker container per scenario per miner, plus a sidecar for your policy:

1. The validator starts the scenario container (the task, the Hermes agent runtime, the hidden verifier) on a private
   network with no internet, drops your `SKILL.md` and the scenario's `INSTRUCTION.md` into `/workspace`, and starts
   your policy in a sidecar on the same network.
2. Hermes talks to your policy as if it were the model (`model: auto`). Your policy calls Engy models through the
   validator's meter, the only route out of the sidecar. The meter enforces the allowlist and a $1 per-scenario cap,
   and records every call.
3. When Hermes finishes or the scenario's deadline hits, the validator runs a **fresh verifier container** with your
   deliverable injected. `tests/test.sh` writes `ctrf.json`; your scenario score is `passed / total`.
4. Session score = sum over the 26 scenarios. Cost per scenario and per model is reported with it, not scored.

## Quick start

```bash
git clone https://github.com/trajectoryRL/trajectoryRL && cd trajectoryRL && pip install -e .
# also: Docker, and an Engy API key (https://api.engy.ai) for local runs, which are on your account

trajectoryrl-miner build SKILL.md --policy ./my_policy -o pack.json      # my_policy/ holds policy.json or policy.py
trajectoryrl-miner validate pack.json                                    # same rules the validator applies
LLM_API_KEY=$ENGY_API_KEY python scripts/eval_pack.py --pack pack.json -o ./out --scenarios git-leak-recovery
trajectoryrl-miner web-submit pack.json                                  # signed with your hotkey
```

The step-by-step version, including the Docker-free inner loop, is the quick start in
[FUSION_POLICY.md](FUSION_POLICY.md#quick-start).

## Pack Format

```json
{
  "schema_version": 1,
  "files": {
    "SKILL.md": "# Your skill ...",
    "policy.json": "{\"kind\": \"advisers\", \"writer\": \"kimi-k3\", \"advisers\": [\"deepseek-v4.1-flash\", \"qwen3.8-27b\", \"glm-5.3-flash\"]}"
  }
}
```

- `SKILL.md` is required and must not be empty. It is dropped read-only into `/workspace` as in Season 1.
- Every other file is part of the policy and is copied into the sidecar's `/policy` directory: `policy.py` or
  `policy.json`, plus any helper text files. Binary files are rejected.
- A pack with only `SKILL.md` is evaluated with the default policy: a pin on qwen3.8-27b, the Season 1 testee.
- Total pack JSON at most 32 KB; content-addressed by `sha256(json.dumps(pack, sort_keys=True))`.

## Writing a policy

Three shapes, evaluated identically:

| shape | what you write | when |
|---|---|---|
| `policy.json` | `{"kind": "pin", "model": ...}`, `{"kind": "escalate", "cheap": ..., "strong": ..., "signals": [...]}`, `{"kind": "advisers", "writer": ..., "advisers": [...]}` | no code; every configuration we measured is expressible |
| `policy.py` on the SDK | a `Policy` subclass with `async def handle(self, req, ctx)` that returns `await ctx.upstream(req)` or `ctx.render(message)`; `ctx.call(...)` for extra model calls, `ctx.session` for state, `ctx.signals()` for stall / tool-error signals | most policies |
| raw `policy.py` | your own server on `$POLICY_PORT` speaking `/v1/chat/completions` and `/v1/models`, calling `$UPSTREAM_URL` with `Authorization: Bearer $EPISODE_TOKEN` | full control |

Allowlisted models (frozen prices in [FUSION_POLICY.md](FUSION_POLICY.md#allowlist-and-prices-spec-25-frozen)):
`deepseek-v4.1-flash`, `deepseek-v4-flash-0731`, `qwen3.8-27b`, `qwen3.6-35b-a3b`, `glm-5.3-flash`, `glm-5.2`,
`glm-5.3`, `kimi-k3`. Any other model name is refused.

Examples with measured scores: [`examples/policies/`](../examples/policies/).

## Writing SKILL.md

Your `SKILL.md` is still the static instruction file the agent reads before every task, and it still matters: it
tells the agent what to produce, in what order, where to write it, and to verify before finishing. What changed is
the model reading it: no longer one fixed small model, but whatever your policy routes to. Keep it general, tight
(3 to 5 KB works best), and free of anything scenario-specific.

## Anti-gaming

The Season 1 rules apply to **every file in the pack**, not only `SKILL.md`: no hardcoded scenario solutions, no
dispatch on scenario names or on phrases from a scenario's `INSTRUCTION.md`, no verifier internals, no obfuscated or
encoded code. Routing on what a task looks like (language, file types, failing tests, turn count, tool errors) is the
competition; a lookup table keyed on the identity of the known scenarios is not. In addition:

- **Provenance** (shadow at launch): assistant messages the harness receives must come from model calls the meter
  saw. Selecting, truncating and combining model output is allowed; authoring it is not.
- **Isolation**: the sidecar has no filesystem access to the task and no internet; the agent container has no
  internet either (from v0.7.1). Off-allowlist models are impossible by construction.
- **Cap**: the $1 per-scenario cap ends an episode's model calls; whatever the agent had written is verified.
- **Copying**: duplicate packs fail the uniqueness check and the copycat audit of `SKILL.md`; extending that audit to policy files is next.
- **Rotation**: the scenario set changes at spec bumps, as it did through Season 1.

## Sandbox Environment

The agent runs inside the scenario container as the non-root user `hermes`. It has a full Linux shell with whatever
the scenario's Dockerfile installed, `/workspace/SKILL.md` and `/workspace/INSTRUCTION.md` read-only, `/app` as
the working directory, and no internet. The verifier (`tests/test.sh`) is bundled in the image at a path the agent
cannot read and runs afterwards in a fresh container. Each scenario has its own `agent_output_path` and time budget
(`task.toml`); the live list is in [trajrl-bench](https://github.com/trajectoryRL/trajrl-bench).

## Scoring Recap

```
quality_scenario = passed / total            (from ctrf.json)
final_score      = sum(quality_scenario)     in [0, N]
```

Cost (USD at the frozen price table, from the meter) is reported per scenario and per model and never folded into
the score in this spec. `qualified = True` whenever `final_score > 0`. Consensus, the seat and the takeover margin
are in [INCENTIVE_MECHANISM.md](INCENTIVE_MECHANISM.md).

## Submission

`trajectoryrl-miner web-submit pack.json` is the only channel. The platform stores the pack, runs pre-eval, and
queues it; validators fetch it, verify the hash, and run the full session. One submission per hotkey every 20
minutes. The submission fee (`recycle_alpha`) is configured server-side; the CLI handles it when it is on.

## Local Testing

```bash
# the exact validator code: sidecar, meter, verifier
LLM_API_KEY=$ENGY_API_KEY python scripts/eval_pack.py --pack pack.json -o ./out
LLM_API_KEY=$ENGY_API_KEY python scripts/eval_pack.py --pack pack.json -o ./out --scenarios git-leak-recovery,postgres-csv-clean

# fast inner loop, no Docker: your policy as a local OpenAI-compatible endpoint
POLICY_PORT=8800 POLICY_DIR=./my_policy UPSTREAM_URL=https://api.engy.ai/v1 EPISODE_TOKEN=$ENGY_API_KEY \
  python trajectoryrl/policy/runtime/trajrl_policy.py
```

Per scenario the eval leaves `policy.log`, `meter.json` (every call, cost, clamps, provenance), the agent
transcript, `evaluation.json`, and `error.txt` if the policy failed to start. Requires Docker and about 10 GB of
disk for the scenario images on first run. Two overlapping local runs on one machine need
`TRAJRL_SKIP_ORPHAN_SCAN=1` (never on a validator).

## FAQ

**Which model does the agent use?** Whatever your policy calls. A `SKILL.md`-only pack runs a pin on qwen3.8-27b.

**Which harness?** Hermes, the version baked into the current sandbox-agent image, with `terminal`, `file` and
`execute_code` tools. Validators control it; miners do not.

**Is there a judge LLM?** No. Scoring is `tests/test.sh` in a fresh container.

**Can I see the scenario tests?** No. You can clone trajrl-bench and read them locally, but a pack that encodes them
is rejected by the audit and worth nothing when the set rotates.

**Can my policy hang or crash?** It costs you that scenario, not the epoch: a crash at start fails the scenario in
seconds with the traceback in the artifacts; a policy that completes no model call within two minutes is stopped.

**Does my policy see the task files?** No. It sees the request stream Hermes sends, which includes the tool results
Hermes chose to fetch.
