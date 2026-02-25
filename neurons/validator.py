#!/usr/bin/env python3
"""TrajectoryRL Validator — Cold-start phase.

Sets weights to a single target UID to anchor consensus while the subnet
bootstraps.  Designed to be hot-swapped to the full production validator
by pushing new code to main — Watchtower auto-deploys.

Environment variables:
    WALLET_NAME     Bittensor wallet name         (default: validator)
    WALLET_HOTKEY   Hotkey name inside wallet      (default: default)
    NETUID          Subnet UID                     (default: 11)
    NETWORK         Subtensor network              (default: finney)
    TARGET_UID      Miner UID to receive weight    (default: 74)
    WEIGHT_INTERVAL Seconds between set_weights    (default: 1500)
"""

import asyncio
import logging
import os
import sys
import time

import bittensor as bt

# ---------------------------------------------------------------------------
# Config from environment
# ---------------------------------------------------------------------------
WALLET_NAME = os.environ.get("WALLET_NAME", "validator")
WALLET_HOTKEY = os.environ.get("WALLET_HOTKEY", "default")
NETUID = int(os.environ.get("NETUID", "11"))
NETWORK = os.environ.get("NETWORK", "finney")
TARGET_UID = int(os.environ.get("TARGET_UID", "74"))
WEIGHT_INTERVAL = int(os.environ.get("WEIGHT_INTERVAL", "1500"))  # ~25 min

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger("coldstart-validator")

# ---------------------------------------------------------------------------
# Version — bump this when you want validators to signal a new code version
# ---------------------------------------------------------------------------
VERSION = 1


async def main():
    logger.info("=== TrajectoryRL Cold-Start Validator ===")
    logger.info(f"  wallet  : {WALLET_NAME}/{WALLET_HOTKEY}")
    logger.info(f"  netuid  : {NETUID}")
    logger.info(f"  network : {NETWORK}")
    logger.info(f"  target  : UID {TARGET_UID}")
    logger.info(f"  interval: {WEIGHT_INTERVAL}s")

    wallet = bt.Wallet(name=WALLET_NAME, hotkey=WALLET_HOTKEY)
    subtensor = bt.Subtensor(network=NETWORK)

    logger.info(f"Connected to {NETWORK} | block {subtensor.block}")

    while True:
        try:
            t0 = time.time()
            logger.info(f"Setting weights: UID {TARGET_UID} = 1.0 ...")

            result = subtensor.set_weights(
                wallet=wallet,
                netuid=NETUID,
                uids=[TARGET_UID],
                weights=[1.0],
                wait_for_inclusion=True,
                wait_for_finalization=False,
            )

            elapsed = time.time() - t0

            if result.success:
                logger.info(f"set_weights OK  ({elapsed:.1f}s)")
            else:
                logger.warning(
                    f"set_weights returned success=False  ({elapsed:.1f}s) — "
                    f"error={result.error}, message={result.message}, "
                    f"extrinsic={result.extrinsic_function}"
                )

        except Exception as e:
            logger.error(f"set_weights failed: {e}", exc_info=True)

        logger.info(f"Sleeping {WEIGHT_INTERVAL}s until next round ...")
        await asyncio.sleep(WEIGHT_INTERVAL)


if __name__ == "__main__":
    asyncio.run(main())
