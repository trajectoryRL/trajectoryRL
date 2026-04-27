#!/usr/bin/env python3
"""TrajectoryRL Validator entry point.

Runs the TrajectoryValidator with continuous evaluation loop:
  - Daily eval at UTC eval_utc_hour: evaluate all active packs, score with split-half delta
  - tempo (~72 min): compute weights from eval scores, set_weights on-chain

Each validator operates independently. Yuma Consensus aggregates weights on-chain.

Environment variables:
    WALLET_NAME             Bittensor wallet name         (default: validator)
    WALLET_HOTKEY           Hotkey name inside wallet      (default: default)
    NETUID                  Subnet UID                     (default: 11)
    NETWORK                 Subtensor network              (default: finney)
    WEIGHT_INTERVAL_BLOCKS  Blocks between set_weights     (default: 360, ~72min)
    SIMILARITY_THRESHOLD    NCD similarity threshold       (default: 0.80)
    INACTIVITY_BLOCKS       Blocks before inactive         (default: 14400, ~48h)
    LOG_LEVEL               Logging level                  (default: INFO)
    EVAL_ON_STARTUP         Run eval immediately on startup (1=enable, default: 0)
"""

import asyncio

from trajectoryrl.base.validator import main

if __name__ == "__main__":
    asyncio.run(main())
