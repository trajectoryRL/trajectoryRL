#!/usr/bin/env python3
"""Evaluate one or more miners by UID using the S1 trajrl-bench sandbox.

exec python -u scripts/eval_miners.py --miner-uid 225 "$@"

Reads miner commitments from the chain, fetches & verifies packs,
runs the trajrl-bench 3-container evaluation (sandbox + testee + judge),
and prints results with split-half delta scoring.

No side effects: no weight setting, no eval state persistence, no on-chain writes.

Usage:
    # Evaluate miner UID 42 on finney:
    python scripts/eval_miners.py --miner-uid 42

    # Evaluate multiple miners:
    python scripts/eval_miners.py --miner-uid 42 43 44

    # Evaluate on testnet:
    python scripts/eval_miners.py --miner-uid 42 --network test

    # Override LLM settings:
    python scripts/eval_miners.py --miner-uid 42 \
        --model openai/gpt-4o --api-key sk-xxx --base-url https://api.openai.com/v1

    # Custom episode count:
    python scripts/eval_miners.py --miner-uid 42 --num-episodes 2

    # Save eval artifacts (transcripts, evaluation.json, etc.):
    python scripts/eval_miners.py --miner-uid 42 -o ./eval_output

    # Save results as JSON:
    python scripts/eval_miners.py --miner-uid 42 43 --json results.json

Environment variables (also read from .env.validator):
    NETUID                    Subnet UID                   (default: 11)
    NETWORK                   Subtensor network            (default: finney)
    LLM_MODEL                 LLM model                   (default: glm-5.1)
    LLM_API_KEY               API key
    LLM_BASE_URL              Base URL
"""

import argparse
import asyncio
import hashlib
import json
import logging
import os
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from trajectoryrl.utils.sandbox_harness import (
    TrajectorySandboxHarness,
    SandboxEvaluationResult,
)
from trajectoryrl.utils.config import ValidatorConfig
from trajectoryrl.base.miner import TrajectoryMiner
from trajectoryrl.utils.github import PackFetcher
from trajectoryrl.utils.commitments import (
    MinerCommitment,
    parse_commitment,
)

logger = logging.getLogger("eval_miners")


def compute_epoch_seed(epoch: int, netuid: int = 11) -> int:
    raw = f"trajectoryrl-{netuid}-epoch-{epoch}".encode()
    return int(hashlib.sha256(raw).hexdigest()[:8], 16)


def compute_validator_salt(netuid: int) -> str:
    data = f"eval_miners_cli:{netuid}"
    return hashlib.sha256(data.encode()).hexdigest()[:16]


async def evaluate_single_miner(
    miner_uid: int,
    commitments,
    harness: TrajectorySandboxHarness,
    epoch_seed: int,
    validator_salt: str,
    *,
    force: bool = False,
) -> Optional[SandboxEvaluationResult]:
    """Evaluate a single miner via the S1 sandbox pipeline.

    Returns SandboxEvaluationResult on success, None on setup failure.
    """
    if miner_uid not in commitments:
        logger.error("No valid commitment found for miner UID %d", miner_uid)
        if commitments:
            logger.info(
                "Available UIDs with commitments: %s",
                sorted(commitments.keys()),
            )
        return None

    commitment = commitments[miner_uid]
    logger.info(
        "Commitment: hash=%s..., url=%s",
        commitment.pack_hash[:16], commitment.pack_url,
    )
    logger.info("Commitment block: %d", commitment.block_number)

    # Fetch and verify pack
    logger.info("Fetching and verifying pack...")
    fetcher = PackFetcher()
    verification = await fetcher.verify_submission(
        pack_url=commitment.pack_url,
        pack_hash=commitment.pack_hash,
    )

    if not verification.valid or verification.pack_content is None:
        logger.error("Pack verification failed: %s", verification.error)
        return None

    pack = verification.pack_content
    logger.info("Pack verified: %d files", len(pack.get("files", {})))
    for fname in pack.get("files", {}):
        logger.info("  - %s", fname)

    # S1 schema validation (schema_version + SKILL.md + size limit)
    issues = TrajectoryMiner.validate_s1(pack)
    if issues:
        logger.error("S1 validation failed: %s", issues)
        if not force:
            return None
        logger.warning("--force specified, continuing despite validation errors")
    else:
        logger.info("S1 validation passed")

    # Extract SKILL.md
    files = pack.get("files", {})
    skill_md = files.get("SKILL.md")
    if not skill_md or not skill_md.strip():
        logger.error("Pack missing or empty SKILL.md")
        return None

    extra_files = [f for f in files if f.lower() != "skill.md"]
    if extra_files:
        logger.warning("S1 pack contains unexpected files: %s", extra_files)

    logger.info("SKILL.md: %d chars", len(skill_md))

    # Run S1 sandbox evaluation (3-container: sandbox + testee + judge)
    logger.info("Starting S1 sandbox evaluation...")
    try:
        result = await harness.evaluate_miner(
            skill_md=skill_md,
            epoch_seed=epoch_seed,
            pack_hash=commitment.pack_hash,
            validator_salt=validator_salt,
        )
    except Exception as e:
        logger.error("S1 evaluation failed: %s", e, exc_info=True)
        return None

    if result.error:
        logger.error("S1 evaluation error: %s", result.error)
        return None

    # Log per-episode details
    for ep in result.session_result.episodes:
        idx = ep.episode_index
        logger.info(
            "  Episode %d: quality=%.3f, duration=%.1fs%s",
            idx, ep.quality, ep.duration_s,
            " (TIMEOUT)" if ep.timed_out else "",
        )
        if ep.error:
            logger.warning("  Episode %d error: %s", idx, ep.error)

    logger.info(
        "S1 result: final_score=%.3f, mean_quality=%.3f, delta=%.3f, "
        "episodes=%s, scenario=%s",
        result.score, result.mean_quality, result.delta,
        result.episode_qualities, result.scenario_name,
    )

    return result


def print_miner_summary(
    miner_uid: int,
    miner_hotkey: str,
    commitment,
    result: SandboxEvaluationResult,
    sandbox_version: str,
):
    """Print S1 evaluation summary for a single miner."""
    W = 70
    print("\n")
    print("=" * W)
    print(f"EVALUATION SUMMARY — Miner UID {miner_uid} ({miner_hotkey[:16]}...)")
    print(f"Pack hash:       {commitment.pack_hash[:16]}...")
    print(f"Pack URL:        {commitment.pack_url}")
    print(f"Sandbox version: {sandbox_version}")
    print(f"Scenario:        {result.scenario_name}")
    print("=" * W)

    sr = result.session_result
    print(f"\n  {'Episode':<12} {'Quality':>8}")
    print(f"  {'-' * 22}")
    for ep in sr.episodes:
        timeout_mark = " (TIMEOUT)" if ep.timed_out else ""
        error_mark = " (ERROR)" if ep.error else ""
        print(
            f"  ep_{ep.episode_index:<8} {ep.quality:>8.3f}"
            f"{timeout_mark}{error_mark}"
        )

    print(f"\n  Early mean (ep 0-1):  {sr.early_mean:.3f}")
    print(f"  Late mean  (ep 2-3):  {sr.late_mean:.3f}")
    print(f"  Delta (late - early): {sr.delta:+.3f}")
    print(f"  Learning bonus:       {sr.learning_bonus:+.3f}")
    print(f"  Mean quality:         {sr.mean_quality:.3f}")
    print(f"  Final score:          {sr.final_score:.3f}")

    qualified = result.success
    print(f"\n  Qualification: {'QUALIFIED' if qualified else 'NOT QUALIFIED'}")
    print("=" * W)

    return sr.final_score, qualified


async def run_evaluation(args):
    import bittensor as bt

    # Re-apply logging after bittensor import (it overrides root logger)
    log_level = getattr(logging, args.log_level)
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    ))
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(log_level)
    logging.getLogger("eval_miners").setLevel(log_level)

    # --- 1. Connect to chain (read-only, no full metagraph) ---
    logger.info("Connecting to %s network...", args.network)
    subtensor = bt.Subtensor(network=args.network)
    current_block = subtensor.get_current_block()
    logger.info(
        "Network: %s, Netuid: %d, Block: %d",
        args.network, args.netuid, current_block,
    )

    # --- 2. Read on-chain commitments (per-UID, no metagraph needed) ---
    miner_uids: List[int] = args.miner_uid
    commitments: Dict[int, MinerCommitment] = {}
    logger.info("Reading commitments for UIDs: %s", miner_uids)

    for uid in miner_uids:
        logger.info("Querying UID %d...", uid)
        try:
            neuron = subtensor.neuron_for_uid(uid=uid, netuid=args.netuid)
        except Exception as e:
            logger.warning("UID %d: failed to query neuron: %s", uid, e)
            continue
        if not neuron or not neuron.hotkey:
            logger.warning("UID %d: not registered on subnet %d", uid, args.netuid)
            continue
        hotkey = neuron.hotkey
        try:
            raw = subtensor.get_commitment(netuid=args.netuid, uid=uid)
        except Exception as e:
            logger.warning("UID %d: failed to read commitment: %s", uid, e)
            continue
        if not raw:
            logger.warning("UID %d: no commitment on chain", uid)
            continue
        parsed = parse_commitment(raw)
        if parsed is None:
            logger.warning("UID %d: unparseable commitment, skipping", uid)
            continue
        pack_hash, pack_url = parsed
        commitments[uid] = MinerCommitment(
            uid=uid, hotkey=hotkey,
            pack_hash=pack_hash, pack_url=pack_url,
            block_number=current_block, raw=raw,
        )
        logger.info(
            "UID %d: hotkey=%s..., hash=%s..., url=%s",
            uid, hotkey[:16], pack_hash[:16], pack_url,
        )

    # --- 3. Prepare evaluation harness via ValidatorConfig ---
    model = (
        args.model
        or os.getenv("LLM_MODEL")
        or os.getenv("CLAWBENCH_DEFAULT_MODEL", "glm-5.1")
    )
    api_key = (
        args.api_key
        or os.getenv("LLM_API_KEY")
        or os.getenv("CLAWBENCH_LLM_API_KEY", "")
    )
    base_url = (
        args.base_url
        or os.getenv("LLM_BASE_URL")
        or os.getenv("CLAWBENCH_LLM_BASE_URL", "https://open.bigmodel.cn/api/paas/v4")
    )

    if not api_key:
        logger.error("No LLM API key configured. Set LLM_API_KEY or --api-key")
        return 1

    tmp_dir = Path(tempfile.mkdtemp(prefix="eval_miners_"))
    config = ValidatorConfig(
        netuid=args.netuid,
        network=args.network,
        llm_model=model,
        llm_api_key=api_key,
        llm_base_url=base_url,
        sandbox_num_episodes=args.num_episodes,
        pack_cache_dir=tmp_dir / "packs",
        log_dir=tmp_dir / "logs",
    )

    harness = TrajectorySandboxHarness(config)

    logger.info("Model:           %s", model)
    logger.info("Base URL:        %s", base_url)
    logger.info("Sandbox image:   %s", config.sandbox_image)
    logger.info("Harness image:   %s", config.harness_image)
    logger.info("Episodes:        %d", config.sandbox_num_episodes)
    logger.info("Timeout/episode: %ds", config.sandbox_timeout_per_episode)

    # --- 4. Pull latest sandbox + harness images ---
    logger.info("Pulling latest images...")
    await harness.pull_latest()
    logger.info(
        "Sandbox version: %s, scoring_version: %d",
        harness.sandbox_version, harness.scoring_version,
    )
    if harness.sandbox_scenarios:
        logger.info("Available scenarios: %s", harness.sandbox_scenarios)

    # --- 5. Generate epoch context ---
    epoch = current_block // config.eval_interval_blocks
    epoch_seed = (
        args.seed if args.seed is not None
        else compute_epoch_seed(epoch, args.netuid)
    )
    validator_salt = compute_validator_salt(args.netuid)

    logger.info("Epoch seed: %d", epoch_seed)

    # --- 6. Evaluate each miner ---
    logger.info("Miner UIDs to evaluate: %s", miner_uids)

    all_results: Dict[int, SandboxEvaluationResult] = {}
    any_failure = False

    for miner_uid in miner_uids:
        logger.info("\n%s", "#" * 70)
        logger.info("# Evaluating Miner UID %d", miner_uid)
        logger.info("%s", "#" * 70)

        result = await evaluate_single_miner(
            miner_uid=miner_uid,
            commitments=commitments,
            harness=harness,
            epoch_seed=epoch_seed,
            validator_salt=validator_salt,
            force=args.force,
        )

        if result is None:
            logger.error("Miner UID %d failed, skipping.", miner_uid)
            any_failure = True
            continue

        all_results[miner_uid] = result

    # --- 7. Print summaries ---
    all_output_data = []
    for miner_uid, result in all_results.items():
        commitment = commitments[miner_uid]
        miner_hotkey = commitment.hotkey
        final_score, qualified = print_miner_summary(
            miner_uid, miner_hotkey, commitment, result,
            sandbox_version=harness.sandbox_version,
        )
        if not qualified:
            any_failure = True

        sr = result.session_result
        all_output_data.append({
            "miner_uid": miner_uid,
            "miner_hotkey": miner_hotkey,
            "pack_hash": commitment.pack_hash,
            "pack_url": commitment.pack_url,
            "commitment_block": commitment.block_number,
            "eval_block": current_block,
            "epoch_seed": epoch_seed,
            "sandbox_version": harness.sandbox_version,
            "scoring_version": harness.scoring_version,
            "scenario": result.scenario_name,
            "evaluation": {
                "final_score": round(sr.final_score, 4),
                "mean_quality": round(sr.mean_quality, 4),
                "early_mean": round(sr.early_mean, 4),
                "late_mean": round(sr.late_mean, 4),
                "delta": round(sr.delta, 4),
                "learning_bonus": round(sr.learning_bonus, 4),
                "episode_qualities": [
                    round(q, 4) for q in result.episode_qualities
                ],
                "qualified": qualified,
            },
        })

    # --- 8. Write eval artifacts ---
    if args.output:
        out_base = Path(args.output)
        out_base.mkdir(parents=True, exist_ok=True)
        for miner_uid, result in all_results.items():
            miner_dir = out_base / f"uid_{miner_uid}"
            result.write_artifacts(miner_dir)
            logger.info("Artifacts written to %s", miner_dir)
        print(f"\nEval artifacts written to: {out_base}")

    # --- 9. Optional JSON output ---
    if args.json_output:
        if len(all_output_data) == 1:
            output = all_output_data[0]
        else:
            output = {"miners": all_output_data}
        Path(args.json_output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.json_output, "w") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        print(f"\nJSON results written to: {args.json_output}")

    return 1 if any_failure else 0


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate one or more miners by UID using the S1 trajrl-bench sandbox. "
            "Read-only: no weight setting, no eval state persistence, no on-chain writes."
        ),
    )

    parser.add_argument(
        "--miner-uid", type=int, nargs="+", required=True,
        help="Miner UID(s) to evaluate (space-separated)",
    )

    # Network config
    parser.add_argument(
        "--network", type=str, default=None,
        help="Bittensor network (default: from env or finney)",
    )
    parser.add_argument(
        "--netuid", type=int, default=None,
        help="Subnet UID (default: from env or 11)",
    )
    # Evaluation config
    parser.add_argument(
        "--num-episodes", type=int, default=4,
        help="Number of episodes per evaluation (default: 4)",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Override epoch seed (default: auto from chain block)",
    )

    # LLM config
    parser.add_argument(
        "--model", type=str, default=None,
        help="LLM model (e.g. glm-5.1)",
    )
    parser.add_argument(
        "--api-key", type=str, default=None,
        help="API key for the LLM provider",
    )
    parser.add_argument(
        "--base-url", type=str, default=None,
        help="Base URL for the LLM API",
    )

    # Output
    parser.add_argument(
        "--output", "-o", type=str, default=None,
        help="Write eval artifacts (transcripts, evaluation.json) to directory",
    )
    parser.add_argument(
        "--json", dest="json_output", type=str, default=None,
        help="Write summary results to JSON file",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Show detailed episode logs",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Continue even if schema validation fails",
    )
    parser.add_argument(
        "--log-level", type=str, default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )

    # Load .env.validator before parsing args so env defaults work
    env_path = PROJECT_ROOT / ".env.validator"
    if env_path.exists():
        try:
            from dotenv import load_dotenv
            load_dotenv(env_path)
            print(f"Loaded config from {env_path}")
        except ImportError:
            pass

    args = parser.parse_args()

    # Apply env defaults for network config
    if args.network is None:
        args.network = os.getenv("NETWORK", "finney")
    if args.netuid is None:
        args.netuid = int(os.getenv("NETUID", "11"))

    try:
        exit_code = asyncio.run(run_evaluation(args))
    except Exception as e:
        print(f"FATAL: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        exit_code = 1
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
