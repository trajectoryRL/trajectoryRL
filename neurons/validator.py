#!/usr/bin/env python3
"""TrajectoryRL Validator — Cold-start phase.

Sets weights to a single target UID to anchor consensus while the subnet
bootstraps.  Before setting on-chain weights, publishes scores to the
shared validator-scores repo and computes stake-weighted consensus.

Environment variables:
    WALLET_NAME               Bittensor wallet name         (default: validator)
    WALLET_HOTKEY             Hotkey name inside wallet      (default: default)
    NETUID                    Subnet UID                     (default: 11)
    NETWORK                   Subtensor network              (default: finney)
    TARGET_UID                Miner UID to receive weight    (default: 74)
    WEIGHT_INTERVAL           Seconds between set_weights    (default: 1500)
    GITHUB_TOKEN              GitHub PAT for score publishing
    VALIDATOR_SCORES_FORK_URL Fork of trajectoryRL/validator-scores
"""

import asyncio
import logging
import os
import sys
import time
from pathlib import Path

import bittensor as bt

from trajectoryrl.utils.score_publisher import ScorePublisher

# ---------------------------------------------------------------------------
# Config from environment
# ---------------------------------------------------------------------------
WALLET_NAME = os.environ.get("WALLET_NAME", "validator")
WALLET_HOTKEY = os.environ.get("WALLET_HOTKEY", "default")
NETUID = int(os.environ.get("NETUID", "11"))
NETWORK = os.environ.get("NETWORK", "finney")
TARGET_UID = int(os.environ.get("TARGET_UID", "74"))
WEIGHT_INTERVAL = int(os.environ.get("WEIGHT_INTERVAL", "1500"))  # ~25 min
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "")
VALIDATOR_SCORES_FORK_URL = os.environ.get("VALIDATOR_SCORES_FORK_URL", "")
EPOCH_INTERVAL = int(os.environ.get("EPOCH_INTERVAL", "86400"))  # 24h

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

    publisher = None
    if GITHUB_TOKEN and VALIDATOR_SCORES_FORK_URL:
        publisher = ScorePublisher(
            wallet_name=WALLET_NAME,
            wallet_hotkey=WALLET_HOTKEY,
            fork_repo_url=VALIDATOR_SCORES_FORK_URL,
            local_path=Path("/tmp/trajectoryrl_validator_scores"),
            github_token=GITHUB_TOKEN,
        )
        logger.info("Score publisher initialized")
    else:
        logger.warning(
            "GITHUB_TOKEN or VALIDATOR_SCORES_FORK_URL not set — "
            "running in solo mode (no score publishing)"
        )

    logger.info(f"Connected to {NETWORK} | block {subtensor.block}")

    while True:
        try:
            t0 = time.time()
            block = subtensor.block
            epoch = int(time.time()) // EPOCH_INTERVAL
            metagraph = subtensor.metagraph(netuid=NETUID)

            scores = {
                str(TARGET_UID): {"final_score": 1.0, "per_scenario": {}},
            }

            # Publish scores → pull consensus → derive weights
            uids = [TARGET_UID]
            weights = [1.0]

            if publisher:
                try:
                    ok = await publisher.publish_scores(
                        epoch=epoch, block_height=block, scores=scores,
                    )
                    if ok:
                        logger.info(f"Scores published for epoch {epoch}")
                    else:
                        logger.warning(f"Score publish returned False for epoch {epoch}")

                    all_scores = await publisher.pull_all_scores(epoch=epoch)
                    if all_scores:
                        consensus = ScorePublisher.compute_consensus(all_scores, metagraph)
                        logger.info(
                            f"Consensus: {consensus.num_validators} validators, "
                            f"stake={consensus.total_stake:.2f}"
                        )
                        if consensus.consensus_scores:
                            uids = list(consensus.consensus_scores.keys())
                            weights = list(consensus.consensus_scores.values())
                except Exception as e:
                    logger.error(f"Score publishing/consensus failed: {e}", exc_info=True)

            logger.info(f"Setting weights: {dict(zip(uids, weights))}")
            result = subtensor.set_weights(
                wallet=wallet,
                netuid=NETUID,
                uids=uids,
                weights=weights,
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
            logger.error(f"Round failed: {e}", exc_info=True)

        logger.info(f"Sleeping {WEIGHT_INTERVAL}s until next round ...")
        await asyncio.sleep(WEIGHT_INTERVAL)


if __name__ == "__main__":
    asyncio.run(main())
