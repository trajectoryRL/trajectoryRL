#!/usr/bin/env python3
"""TrajectoryRL Validator entry point.

Runs the TrajectoryValidator daemon (see trajectoryrl/base/validator.py):
  - Weight-only (default): mirror the server-canonical winner into
    on-chain weights, tempo-gated.
  - EVAL_ENABLED=1: additionally evaluate each epoch's challenger in the
    trajrl-bench sandbox and submit the score.

Each validator operates independently. Yuma Consensus aggregates weights on-chain.

Environment variables:
    WALLET_NAME             Bittensor wallet name          (default: validator)
    WALLET_HOTKEY           Hotkey name inside wallet      (default: default)
    NETUID                  Subnet UID                     (default: 11)
    NETWORK                 Subtensor network              (default: finney)
    EVAL_ENABLED            1 = evaluate challengers       (default: 0, weight-only)
    LLM_API_KEY             engy key, needed with EVAL_ENABLED=1
    LOG_LEVEL               Logging level                  (default: INFO)
"""

import asyncio

from trajectoryrl.base.validator import main

if __name__ == "__main__":
    asyncio.run(main())
