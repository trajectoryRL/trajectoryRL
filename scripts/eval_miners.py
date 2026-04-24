#!/usr/bin/env python3
"""Evaluate one or more miners by UID using the S1 trajrl-bench sandbox.

Reads miner commitments from the chain, fetches & verifies packs, and
runs the same Season 1 evaluation function the production validator
uses (``trajectoryrl.utils.miner_eval.evaluate_miner_s1``). No on-chain
writes, no eval-state persistence, no weight setting.

Configuration is taken from ``.env.validator`` via ``ValidatorConfig.from_env``.
The CLI exposes only the few flags that are unique to one-off debugging:
which miner(s) to evaluate, how to render output, and an optional epoch
seed override for reproducibility.

Usage:
    python scripts/eval_miners.py --miner-uid 42
    python scripts/eval_miners.py --miner-uid 42 43 44
    python scripts/eval_miners.py --miner-uid 42 -o ./eval_output
    python scripts/eval_miners.py --miner-uid 42 --json results.json

To override env values temporarily:
    NETUID=12 LLM_MODEL=openai/gpt-4o python scripts/eval_miners.py --miner-uid 42
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from trajectoryrl.utils.config import SPEC_NUMBER, ValidatorConfig
from trajectoryrl.utils.commitments import MinerCommitment, parse_commitment
from trajectoryrl.utils.github import PackFetcher
from trajectoryrl.utils.miner_eval import evaluate_miner_s1
from trajectoryrl.utils.sandbox_harness import (
    SandboxEvaluationResult,
    TrajectorySandboxHarness,
)

logger = logging.getLogger("eval_miners")


def _setup_logging(level_name: str) -> None:
    """Configure stderr logging after bittensor (re-)applied its handlers.

    Bittensor's import resets the root logger, so this must be called
    after ``import bittensor`` and explicitly re-set our project loggers.
    """
    level = getattr(logging, level_name)
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    ))
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)
    for name in (
        "eval_miners",
        "trajectoryrl",
        "trajectoryrl.utils.miner_eval",
        "trajectoryrl.utils.sandbox_harness",
        "trajectoryrl.utils.github",
    ):
        lg = logging.getLogger(name)
        lg.setLevel(level)
        lg.propagate = True


def _close_subtensor(subtensor) -> None:
    """Shut down bittensor's background websocket / RPC threads.

    ``async_substrate_interface`` keeps the asyncio loop alive after a
    one-shot CLI run unless we close the substrate connection
    explicitly.
    """
    for closer in (
        lambda: subtensor.substrate.close(),
        lambda: subtensor.close(),
    ):
        try:
            closer()
        except Exception as e:
            logger.debug("subtensor close path failed (ignored): %s", e)


def _fetch_commitments(
    subtensor,
    netuid: int,
    miner_uids: List[int],
    current_block: int,
) -> Dict[int, MinerCommitment]:
    """Fetch on-chain commitments for the requested UIDs only.

    Skips the full metagraph + ``get_all_commitments`` round-trip used
    by the production validator: a CLI run typically targets one or two
    UIDs, and per-UID RPC is faster than scanning the whole subnet.
    """
    commitments: Dict[int, MinerCommitment] = {}
    for uid in miner_uids:
        logger.info("Querying UID %d...", uid)
        try:
            neuron = subtensor.neuron_for_uid(uid=uid, netuid=netuid)
        except Exception as e:
            logger.warning("UID %d: failed to query neuron: %s", uid, e)
            continue
        if not neuron or not neuron.hotkey:
            logger.warning("UID %d: not registered on subnet %d", uid, netuid)
            continue
        hotkey = neuron.hotkey
        try:
            raw = subtensor.get_commitment(netuid=netuid, uid=uid)
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
    return commitments


def _print_summary(
    miner_uid: int,
    commitment: MinerCommitment,
    result: SandboxEvaluationResult,
    sandbox_version: str,
) -> bool:
    """Render a per-miner human-readable summary; return qualified flag."""
    W = 70
    sr = result.session_result
    print("\n" + "=" * W)
    print(f"EVALUATION SUMMARY — Miner UID {miner_uid} ({commitment.hotkey[:16]}...)")
    print(f"Pack hash:       {commitment.pack_hash[:16]}...")
    print(f"Pack URL:        {commitment.pack_url}")
    print(f"Sandbox version: {sandbox_version}")
    print(f"Scenario:        {result.scenario_name}")
    print("=" * W)

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
    return qualified


async def run_evaluation(args) -> int:
    import bittensor as bt
    _setup_logging(args.log_level)

    # Redirect cache + logs to a temp dir BEFORE from_env runs, so
    # ValidatorConfig.__post_init__ doesn't try to mkdir under
    # /var/lib/trajectoryrl (which the CLI lacks permission for).
    tmp_dir = Path(tempfile.mkdtemp(prefix="eval_miners_"))
    os.environ.setdefault("PACK_CACHE_DIR", str(tmp_dir / "packs"))
    os.environ.setdefault("LOG_DIR", str(tmp_dir / "logs"))

    config = ValidatorConfig.from_env(dotenv_path=PROJECT_ROOT / ".env.validator")

    if not config.llm_api_key:
        logger.error("No LLM API key configured. Set LLM_API_KEY in .env.validator")
        return 1

    miner_uids: List[int] = args.miner_uid
    logger.info("Connecting to %s (netuid %d)...", config.network, config.netuid)
    subtensor = bt.Subtensor(network=config.network)
    try:
        current_block = subtensor.get_current_block()
        logger.info("Block: %d", current_block)
        commitments = _fetch_commitments(
            subtensor, config.netuid, miner_uids, current_block,
        )
    finally:
        _close_subtensor(subtensor)

    if not commitments:
        logger.error("No valid commitments found for any requested UID.")
        return 1

    harness = TrajectorySandboxHarness(config)
    pack_fetcher = PackFetcher(cache_dir=config.pack_cache_dir)

    logger.info("Pulling latest sandbox + harness images...")
    await harness.pull_latest()
    logger.info(
        "Sandbox %s, spec_number %d, scenarios=%s",
        harness.sandbox_version, SPEC_NUMBER,
        harness.sandbox_scenarios or "?",
    )

    epoch = current_block // config.eval_interval_blocks
    epoch_seed = (
        args.seed if args.seed is not None
        # Lazy import: TrajectoryValidator's import triggers bittensor's
        # argv parsing at module load, which would intercept --help.
        else _epoch_seed_from_validator(epoch, config.netuid)
    )
    logger.info("Epoch %d (block %d), seed %d", epoch, current_block, epoch_seed)

    all_results: Dict[int, SandboxEvaluationResult] = {}
    any_failure = False

    for uid in miner_uids:
        if uid not in commitments:
            any_failure = True
            continue
        logger.info("\n%s\n# Evaluating Miner UID %d\n%s", "#" * 70, uid, "#" * 70)
        outcome = await evaluate_miner_s1(
            harness=harness,
            pack_fetcher=pack_fetcher,
            commitment=commitments[uid],
            epoch_seed=epoch_seed,
            # Empty salt -> harness falls back to its built-in default
            # (see TrajectorySandboxHarness._default_salt). CLI does not
            # need to match a specific validator's fixtures.
            validator_salt="",
            mlog=logger,
        )
        if outcome.success and outcome.sandbox_result is not None:
            all_results[uid] = outcome.sandbox_result
        else:
            logger.error(
                "Miner UID %d skipped: %s%s",
                uid, outcome.skip_reason or "unknown",
                f" ({outcome.skip_detail})" if outcome.skip_detail else "",
            )
            any_failure = True

    json_payloads = []
    for uid, result in all_results.items():
        commitment = commitments[uid]
        qualified = _print_summary(
            uid, commitment, result, harness.sandbox_version,
        )
        if not qualified:
            any_failure = True

        sr = result.session_result
        json_payloads.append({
            "miner_uid": uid,
            "miner_hotkey": commitment.hotkey,
            "pack_hash": commitment.pack_hash,
            "pack_url": commitment.pack_url,
            "commitment_block": commitment.block_number,
            "eval_block": current_block,
            "epoch_seed": epoch_seed,
            "sandbox_version": harness.sandbox_version,
            "spec_number": SPEC_NUMBER,
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

    if args.output:
        out_base = Path(args.output)
        out_base.mkdir(parents=True, exist_ok=True)
        for uid, result in all_results.items():
            miner_dir = out_base / f"uid_{uid}"
            result.write_artifacts(miner_dir)
            logger.info("Artifacts written to %s", miner_dir)
        print(f"\nEval artifacts written to: {out_base}")

    if args.json_output:
        payload = json_payloads[0] if len(json_payloads) == 1 else {"miners": json_payloads}
        Path(args.json_output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.json_output, "w") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print(f"\nJSON results written to: {args.json_output}")

    return 1 if any_failure else 0


def _epoch_seed_from_validator(epoch: int, netuid: int) -> int:
    """Lazy wrapper around ``TrajectoryValidator.compute_epoch_seed``.

    Importing the validator module triggers bittensor's argv parsing at
    module load (which would intercept ``--help`` before our argparse
    runs), so we defer the import until ``run_evaluation`` is executing.
    """
    from trajectoryrl.base.validator import TrajectoryValidator
    return TrajectoryValidator.compute_epoch_seed(epoch, netuid)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate one or more miners by UID using the same Season 1 "
            "function the production validator uses. Read-only: no weight "
            "setting, no eval state persistence, no on-chain writes. All "
            "configuration (network, LLM, episodes, ...) is taken from "
            ".env.validator; override via env vars when running."
        ),
    )
    parser.add_argument(
        "--miner-uid", type=int, nargs="+", required=True,
        help="Miner UID(s) to evaluate (space-separated)",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Override epoch seed (default: derived from current chain block)",
    )
    parser.add_argument(
        "--output", "-o", type=str, default=None,
        help="Write per-miner eval artifacts (transcripts, evaluation.json) to this directory",
    )
    parser.add_argument(
        "--json", dest="json_output", type=str, default=None,
        help="Write summary results to a JSON file",
    )
    parser.add_argument(
        "--log-level", type=str, default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )

    args = parser.parse_args()

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
