"""Example: sign and publish validator scores via ScorePublisher.

Prerequisites:
    - `gh` CLI installed and authenticated (`gh auth login`)
    - You have forked `trajectoryRL/validator-scores` to your GitHub account
    - Bittensor wallet created and registered on the subnet
    - .env.validator configured (or equivalent env vars exported)
"""

import asyncio
import logging

import bittensor as bt

from trajectoryrl.utils.config import ValidatorConfig
from trajectoryrl.utils.score_publisher import ScorePublisher


async def main():
    logging.basicConfig(level=logging.DEBUG)
    logging.info("Starting score publish example")
    # 1. Load config from environment / .env.validator
    config = ValidatorConfig.from_env()

    # 2. Initialize ScorePublisher (all params come from config)
    publisher = ScorePublisher(
        wallet_name=config.wallet_name,
        wallet_hotkey=config.wallet_hotkey,
        fork_repo_url=config.validator_scores_fork_url,
        local_path=config.validator_scores_local_path,
        github_token=config.github_token,
        git_email=config.git_email,
        git_name=config.git_name,
    )

    # 3. Build per-UID scores (typically from ClawBench evaluation results)
    scores = {
        "uid_0": {"final_score": 0.85, "per_scenario": {"client_escalation": 0.92}},
        "uid_1": {"final_score": 0.72, "per_scenario": {"client_escalation": 0.72}},
    }

    # 4. Sign + commit + push + open PR (single call)
    success = await publisher.publish_scores(
        epoch=42,
        block_height=150000,
        scores=scores,
    )
    print(f"Publish: {'OK' if success else 'FAILED'}")

    # 5. Pull all validators' scores for consensus (with signature verification)
    all_files = await publisher.pull_all_scores(epoch=42)
    print(f"Pulled {len(all_files)} verified score files")

    # 6. Compute stake-weighted consensus
    subtensor = bt.Subtensor(network=config.network)
    metagraph = subtensor.metagraph(netuid=config.netuid)

    consensus = ScorePublisher.compute_consensus(all_files, metagraph)
    print(f"Consensus from {consensus.num_validators} validators:")
    for uid, score in sorted(consensus.consensus_scores.items()):
        print(f"  UID {uid}: {score:.4f}")


if __name__ == "__main__":
    asyncio.run(main())
