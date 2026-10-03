# Example fusion policies (Season 2, the transition season)

Season 2 is a transition: same harness, scenarios and consensus as Season 1, new competition object (fusion policies). SN11's next stage is RSI infrastructure (see docs/FUSION_POLICY.md).

Each directory is the non-SKILL.md part of a pack. Build a pack with your SKILL.md plus these files:

    trajectoryrl-miner build SKILL.md --policy examples/policies/advisers -o pack.json
    python scripts/eval_pack.py --pack pack.json -o ./eval_output      # local run, your own Engy key

| dir | what it does | 26 SPEC-25 scenarios, one trial, real harness (2026-09-18) |
|---|---|---|
| pin | glm-5.3-flash for every call | 22.60 / 26 at $0.56 per session |
| escalate | glm-5.3-flash, switch to kimi-k3 on stall / tool errors / repeats | 20.99 / 26 at $4.68 |
| advisers | kimi-k3 writes; three cheap models write notes before every turn | 23.55 / 26 at $10.02 |
| sdk_custom | policy.py against the SDK: cheap first, strong + one adviser after turn 12 or a signal | 22.93 / 26 at $5.78 |

Reference: pinned kimi-k3 22.84 at $8.50; SKILL.md only (pin qwen3.8-27b, the Season 1 testee) 19.92 at $0.64.

All of these runs used the launch cap of $1.00 per scenario. The cap is now $0.30, and they have not been
re-measured under it: a policy that keeps kimi-k3 as the writer for a whole long episode will be cut off earlier.
