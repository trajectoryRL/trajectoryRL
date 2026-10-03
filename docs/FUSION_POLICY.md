# Season 2 (transition): Fusion Policies

**Subnet**: SN11 (TrajectoryRL)
**Status**: TRANSITION SEASON. Season 2 is a bridge, not the destination.

> **Where SN11 is going.** SN11 will become **RSI infrastructure**: a platform where emission pays for
> verifiable agent runs on Engy, miners author and fund the agents, and validators run a replayable
> orchestrator that dispatches work against automatically verified objectives (formalized mathematics,
> benchmark tasks, Engy's own serving stack). That rebuild takes longer than we are willing to leave the
> current SKILL.md contest running, so Season 2 reuses the Season 1 harness, scenarios, verifier and
> consensus unchanged and changes only the object of competition: the miner's **fusion policy** over
> the Engy model catalog. Expect Season 2 to be superseded when the RSI platform launches; the policies
> written for it remain useful, because a fusion policy is exactly the per-request "auto" model Engy
> serves: we will serve the policy that holds the seat as Engy's auto mode (`engy/auto`) for every customer
> and every harness. The full sequence of seasons is in [ROADMAP.md](ROADMAP.md).

**Scoring**: programmatic verifier, quality summed across the active scenario set (unchanged from Season 1)
**Spec**: 25 (same 26 scenarios as SPEC 24; new testee: your fusion policy)

> Mining means writing a **fusion policy**: a program that sits between the agent harness (Hermes) and the
> Engy model catalog and decides, per request, which models to call, in what combination (route, escalate,
> ask advisers, ensemble), and what to send them. The best policy
> wins, and it will ship: we will serve the seated policy as Engy's auto mode.

---

## What changed from Season 1

| | Season 1 | Season 2 |
|---|---|---|
| testee model | qwen3.8-27b, pinned by the validator | whatever your policy calls, from the Engy allowlist |
| what you submit | SKILL.md | SKILL.md (optional but recommended) + `policy.py` or `policy.json` (+ helper text files) |
| where your code runs | nowhere; SKILL.md is a prompt | a **policy sidecar** container next to the scenario container |
| score | sum of per-scenario quality | same |
| cost | reported | reported per episode and per model (not scored at launch; a safety cap of $0.60 per episode applies) |

Everything else is the same: one container per scenario, Hermes as the agent, hidden verifier, 600 s per
scenario, Winsorized consensus across validators, winner-take-all seat with the takeover margin.

---

## Quick start

**1. Install.** `git clone https://github.com/trajectoryRL/trajectoryRL && cd trajectoryRL && pip install -e .`
You also need Docker and an Engy API key (https://api.engy.ai). Local runs use your key; validators pay for the
real evaluations.

**2. Write a policy**: a directory with `policy.json` (no code) or `policy.py`.

```json
{"kind": "advisers", "writer": "kimi-k3", "advisers": ["deepseek-v4.1-flash", "qwen3.8-27b", "glm-5.3-flash"]}
```

```python
from trajrl_policy import Policy, serve

class Mine(Policy):
    async def handle(self, req, ctx):
        req["model"] = "kimi-k3" if "toolerr" in ctx.signals() else "glm-5.3-flash"
        return await ctx.upstream(req)   # streams through

serve(Mine())
```

Or a raw `policy.py` that serves `/v1/chat/completions` itself. Examples with measured scores are in
`examples/policies/`; the three shapes are described under "Writing a policy" below.

**3. Test fast, no Docker**: your policy as a local endpoint for any OpenAI client, Hermes or OpenCode.

```bash
POLICY_PORT=8800 POLICY_DIR=./my_policy UPSTREAM_URL=https://api.engy.ai/v1 EPISODE_TOKEN=$ENGY_API_KEY \
  python trajectoryrl/policy/runtime/trajrl_policy.py
# base URL http://localhost:8800/v1, model "auto"; every turn is logged as JSON on stdout
```

**4. Test for real**: the exact validator code, with the sidecar, the meter and the verifier, on scenarios you pick.

```bash
trajectoryrl-miner build SKILL.md --policy ./my_policy -o pack.json
trajectoryrl-miner validate pack.json
LLM_API_KEY=$ENGY_API_KEY python scripts/eval_pack.py --pack pack.json -o ./out --scenarios git-leak-recovery,postgres-csv-clean
```

Per scenario you get `policy.log`, `meter.json` (every call, cost, clamps, provenance), the agent transcript,
and `error.txt` if the policy failed to start. Details under "Local testing" below.

**5. Submit.** `trajectoryrl-miner web-submit pack.json`, same as Season 1. Packs under 32 KB, SKILL.md still
required. The first pack through pre-eval starts the next epoch.

## How an episode runs

```
 validator process                          docker
 ┌──────────────────────┐   internal net    ┌────────────────────────────┐
 │ meter :8790          │◄──────────────────│ policy sidecar (YOUR CODE)  │◄──┐  LLM_BASE_URL=http://policy:8800/v1
 │ operator's Engy key  │                   │ no volumes, 1 CPU, 1 GB     │   │  LLM_MODEL=auto
 │ allowlist, prices,   │                   └────────────────────────────┘   │
 │ $0.60 cap, call log  │                   ┌────────────────────────────┐   │
 └──────────┬───────────┘                   │ scenario container (Hermes) │───┘
            ▼ api.engy.ai                   │ /app, /workspace, verifier  │
                                            └────────────────────────────┘
```

1. The validator creates a private internal network for the episode, starts your policy in a sidecar on it, and
   attaches the scenario container. The sidecar's only route out is the validator's **meter**.
2. Hermes runs exactly as in Season 1, but its LLM endpoint is your sidecar (`model: auto`).
3. Every model call your policy makes goes through the meter, which forwards it to Engy with the validator's
   key, checks the model against the allowlist, records tokens and cost at a frozen price table, and stops
   forwarding once the episode has spent the safety cap.
4. When Hermes finishes (or the 600 s deadline hits), the deliverable is verified as before. The episode's
   artifacts include your policy's log and the meter's per-call log.

Your policy sees only the request stream (the full conversation Hermes sends on every turn). It cannot read
`/app`, `/workspace` or the tests, and it cannot reach the internet.

---

## Allowlist and prices (SPEC 25, frozen)

| model | $/M prompt | $/M completion | $/M cache read |
|---|---|---|---|
| deepseek-v4-flash-0731 | 0.045 | 0.09 | 0.009 |
| deepseek-v4.1-flash | 0.04 | 0.08 | 0.008 |
| qwen3.6-35b-a3b | 0.045 | 0.30 | 0.015 |
| qwen3.8-27b | 0.045 | 0.32 | 0.015 |
| glm-5.3-flash | 0.135 | 0.45 | 0.027 |
| glm-5.2 | 0.68 | 1.50 | 0.18 |
| glm-5.3 | 0.98 | 3.08 | 0.18 |
| kimi-k3 | 1.95 | 9.75 | 0.195 |

Any other model name is refused (HTTP 400). The $0.60 safety cap is enforced **before** a call is forwarded:
the meter reserves the worst case for the call (every billed input field, messages and tools included, at
list price with no cache assumed, ASCII at 3 characters per token and non-ASCII at one token per character with
a 1.25x safety factor, plus `max_tokens` at the completion price) against what is left after the other calls in
flight have reserved theirs. If a call still bills more input than reserved, the excess is booked as
`overshoot_usd` and the episode is closed to further calls.
If the requested completion length does not fit, `max_tokens` is reduced to what fits (the call row records
`clamped`); if fewer than 64 completion tokens fit, the call is refused (HTTP 402). A request without
`max_tokens` is treated as 8192. So near the cap a policy sees shorter answers first and refusals last, and the
episode can never overspend. `x-trajrl-budget-remaining-usd` on every response is the budget left after the
call's reservation; for streamed calls it is computed before the tokens flow, so the authoritative number is
`GET /v1/budget` (same bearer token), which the SDK queries after every streamed call to keep
`ctx.remaining_usd` current.

---

## Writing a policy

Three levels. All three are evaluated identically.

### Level 1: `policy.json` (no code)

```json
{"kind": "advisers", "writer": "kimi-k3", "advisers": ["deepseek-v4.1-flash", "qwen3.8-27b", "glm-5.3-flash"]}
```

Kinds shipped in the runtime:

| kind | fields | behaviour |
|---|---|---|
| `pin` | `model` | every call goes to `model` |
| `escalate` | `cheap`, `strong`, `signals` (subset of `stall`, `len`, `toolerr`, `noprog`, `repeat`), `stall_s` | start cheap; on a signal switch the session to `strong` and stay |
| `advisers` | `writer`, `advisers`, `max_tokens`, `temperature`, `every` | advisers read the conversation and write short notes; the writer answers with the notes appended; advisers engage once tool results exist |

### Level 2: `policy.py` on the SDK

```python
from trajrl_policy import Policy, serve

class Mine(Policy):
    async def handle(self, req, ctx):
        # req: the OpenAI chat request Hermes sent (full conversation every turn)
        # ctx.session: dict that persists across the turns of one agent session
        # ctx.signals(): ["stall", "toolerr", ...] read off the transcript
        # ctx.remaining_usd: what the meter says is left for this episode
        req["model"] = "kimi-k3" if "toolerr" in ctx.signals() else "glm-5.3-flash"
        return await ctx.upstream(req)          # streams the model's tokens through unchanged

serve(Mine())
```

Extra calls: `await ctx.call(model, messages, max_tokens=..., temperature=...)` returns a completed message
(advisers, judges, samplers). To answer with a message you assembled yourself, return `ctx.render(message)`;
it is delivered to Hermes as a stream when Hermes asked for one.

See `examples/policies/sdk_custom/policy.py` for a complete example.

### Level 3: raw

`policy.py` is executed as `python policy.py` inside the sidecar with these environment variables:

| var | meaning |
|---|---|
| `POLICY_PORT` | port to serve on (8800) |
| `UPSTREAM_URL` | the meter's base URL, e.g. `http://meter:8790/v1` |
| `EPISODE_TOKEN` | bearer token for the meter |
| `POLICY_DIR` | where your files are (`/policy`) |
| `DEFAULT_MODEL` | the Season 1 testee model, for reference |

The meter also serves `GET /v1/budget` (bearer `EPISODE_TOKEN`): `{cap_usd, spent_usd, reserved_usd, remaining_usd}`.

Serve `POST /v1/chat/completions` (streaming and non-streaming) and `GET /v1/models` (health) on
`POLICY_PORT`; call `UPSTREAM_URL/chat/completions` with `Authorization: Bearer $EPISODE_TOKEN`. Python 3.13
with `aiohttp` and `httpx` is available; nothing can be installed at run time (no network).

### Runtime facts

- Hermes resends the whole conversation on every turn. Its system prompt and first user turn are the same for
  every scenario; the two tool results that read SKILL.md and INSTRUCTION.md make a session unique. The SDK's
  `session_key` uses the first five messages.
- Prefix caching on Engy is per model: switching models mid-session costs one cold prefill of the new model,
  then it is warm. Switching every turn is expensive; switching a few times per session is not.
- Adviser fan-out adds 10 to 17 s per turn with three long-context calls in parallel. The 600 s deadline is
  per scenario, so latency is quality.
- The sidecar has 1 CPU and 1 GB. It is started fresh for every scenario; there is no state across scenarios.
- Watchdog: if no model call has completed through the meter within 120 s of the agent starting, or none in
  the last 600 s, the validator kills the agent for that scenario and verifies whatever was written. A policy
  that hangs, crashes, or refuses every call therefore costs at most two minutes per scenario, not the whole
  budget. A policy that crashes before answering its health check fails the scenario immediately, with the
  traceback in the episode's artifacts.

---

## Pack format

```json
{
  "schema_version": 1,
  "files": {
    "SKILL.md": "...",
    "policy.py": "...",
    "prompts/adviser.txt": "..."
  }
}
```

- `SKILL.md` is still required by the submit path (it is dropped read-only into `/workspace` as in Season 1;
  a short one is fine). Every other file is copied into the sidecar's `/policy` directory.
- One of `policy.py` or `policy.json` must exist; a SKILL.md-only pack is evaluated with `pin` on the Season 1
  testee model.
- Total pack JSON ≤ 32 KB, content-addressed as before.

Build and test locally:

```bash
trajectoryrl-miner build SKILL.md --policy ./my_policy -o pack.json   # text files only; .pyc and __pycache__ skipped
trajectoryrl-miner validate pack.json                                  # same policy-file rules the validator applies
LLM_API_KEY=<your engy key> python scripts/eval_pack.py --pack pack.json -o ./eval_output
trajectoryrl-miner web-submit pack.json
```

Local runs use your own Engy key through the same meter, so what you see is what the validators see, at your
cost.

Two local runs on one machine: the harness removes every trajectoryrl container on the host when a session
starts. Set `TRAJRL_SKIP_ORPHAN_SCAN=1` for local runs that overlap (never on a validator).

### Fast inner loop without Docker

The runtime that the validator copies into the sidecar is one file and runs anywhere. Point it at Engy with
your own key as the token and it becomes a local OpenAI-compatible endpoint serving your policy:

```bash
cd my_policy   # contains policy.py or policy.json
POLICY_PORT=8800 POLICY_DIR=. UPSTREAM_URL=https://api.engy.ai/v1 EPISODE_TOKEN=$ENGY_API_KEY \
  python /path/to/trajectoryRL/trajectoryrl/policy/runtime/trajrl_policy.py
# any OpenAI client, or Hermes / OpenCode with base URL http://localhost:8800/v1 and model "auto"
curl -s localhost:8800/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"auto","messages":[{"role":"user","content":"hello"}],"max_tokens":200}'
```

Every turn is logged as JSON on stdout (model chosen, tokens, latency). No cap and no meter in this mode; it
is for iterating on prompts and routing logic. When it behaves, run `scripts/eval_pack.py` for the real
thing: the sidecar, the meter, the scenarios and the verifier exactly as validators run them.

### What a local eval leaves behind

`eval_output/episodes/<scenario>/`: `policy.log` (your policy's own log), `meter.json` (every model call
with tokens, cost, clamps, refusals, and the provenance result), `testee_transcript.txt` (what Hermes did),
`evaluation.json` (verifier result, cost by model, setup time, stall flag), and `error.txt` when the sidecar
failed to start (the traceback is in it).

---

## Baselines (26 SPEC-25 scenarios, one trial each, run through this exact harness on 2026-09-18)

| pack | policy | score /26 | $ per session | note |
|---|---|---|---|---|
| SKILL.md only | default pin on qwen3.8-27b (the Season 1 testee) | 19.92 | 0.64 | the Season 1 opening position |
| `examples/policies/pin` | pin glm-5.3-flash | 22.60 | 0.56 | the cheap floor to beat |
| pin kimi-k3 | pin kimi-k3 | 22.84 | 8.50 | strongest single model, 15x the price of the floor |
| `examples/policies/escalate` | glm-5.3-flash, switch to kimi-k3 on signals | 20.99 | 4.68 | naive escalation loses: it escalates the wrong sessions |
| `examples/policies/sdk_custom` | cheap, then kimi-k3 + one adviser after turn 12 or a signal | 22.93 | 5.78 | zero scenarios at 0 |
| `examples/policies/advisers` | kimi-k3 writes, three cheap advisers | 23.55 | 10.02 | best measured; one trial |

Trial-to-trial movement of a single policy is about one point on this set (pinned kimi-k3 ranged 19.7 to 23.3
over four lab trials). Validators run one trial each and consensus averages them. Every run above was made
with `scripts/eval_pack.py --pack ...` on the same code the validators ship, so a miner reproducing them at home
should land within noise.

## Anti-gaming

The Season 1 rules apply to every string in the pack, not only SKILL.md: no hardcoded scenario solutions, no
scenario-name dispatch, no verifier internals, no obfuscated code. In addition:

- **Provenance.** Assistant messages the harness receives must come from model calls the meter saw. A policy
  may select, truncate, combine and reorder model output; it may not author it. The meter fingerprints every
  model answer (text and tool calls) and the validator compares them with what Hermes recorded; the per-episode
  coverage travels with the score. Shadow mode at launch: recorded and visible, not scored. Measured on the
  baseline runs above (156 episodes): pass-through and adviser policies score 1.0 on all but 2 episodes, where
  Hermes itself repaired one malformed tool-call JSON or normalised one message (coverage 0.91 and 0.93). The
  review threshold is coverage below 0.8, not below 1.0.
- Off-allowlist models are impossible by construction (network isolation + allowlist).
- The sidecar shares the episode network with the scenario container, so it can reach any port a scenario
  happens to open (a scenario's own nginx, for example). It cannot read the filesystem, run commands there, or
  see the verifier, which runs afterwards in a fresh container; treat anything learned that way as covered by
  the same hardcoding rules.
- The safety cap ends an episode's model calls at $0.60; whatever the agent had written is verified.
