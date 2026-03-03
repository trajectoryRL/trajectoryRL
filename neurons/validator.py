#!/usr/bin/env python3
"""TrajectoryRL Validator entry point.

Runs the TrajectoryValidator with continuous evaluation loop:
  - eval_interval (~4h): evaluate all active packs, update per-scenario EMA
  - tempo (~72 min): compute weights from EMA scores, set_weights on-chain

Each validator operates independently. YC3 aggregates weights on-chain.

Environment variables:
    WALLET_NAME             Bittensor wallet name         (default: validator)
    WALLET_HOTKEY           Hotkey name inside wallet      (default: default)
    NETUID                  Subnet UID                     (default: 11)
    NETWORK                 Subtensor network              (default: finney)
    EVAL_INTERVAL_BLOCKS    Blocks between evaluations     (default: 1200, ~4h)
    WEIGHT_INTERVAL_BLOCKS  Blocks between set_weights     (default: 360, ~72min)
    EMA_ALPHA               EMA smoothing factor           (default: 0.3)
    SIMILARITY_THRESHOLD    NCD similarity threshold       (default: 0.80)
    INACTIVITY_BLOCKS       Blocks before inactive         (default: 14400, ~48h)
    LOG_LEVEL               Logging level                  (default: INFO)
"""

import asyncio

from trajectoryrl.base.validator import main

if __name__ == "__main__":
    asyncio.run(main())
