#!/usr/bin/env python3
"""Evaluate a single miner by UID using on-chain commitment data.

Reads the miner's commitment from the chain, fetches & verifies the pack,
runs ClawBench evaluation on all scenarios, and prints results.
No side effects: no weight setting, no EMA persistence, no on-chain writes.

Can be used as a Docker entrypoint (like neurons/validator.py) or standalone.

Usage:
    # Evaluate miner UID 42 on finney:
    python scripts/eval_single_miner.py --miner-uid 42

    # Evaluate on testnet with custom wallet:
    python scripts/eval_single_miner.py --miner-uid 42 --network test --wallet-name myval

    # Run specific scenarios only:
    python scripts/eval_single_miner.py --miner-uid 42 --scenarios client_escalation morning_brief

    # Multiple consensus runs per scenario:
    python scripts/eval_single_miner.py --miner-uid 42 --num-runs 3

    # Override LLM settings:
    python scripts/eval_single_miner.py --miner-uid 42 \
        --model openai/gpt-4o --api-key sk-xxx --base-url https://api.openai.com/v1

    # Save full results to JSON:
    python scripts/eval_single_miner.py --miner-uid 42 -o results.json

Environment variables (also read from .env.validator):
    WALLET_NAME               Bittensor wallet name        (default: validator)
    WALLET_HOTKEY             Hotkey name                  (default: default)
    NETUID                    Subnet UID                   (default: 11)
    NETWORK                   Subtensor network            (default: finney)
    CLAWBENCH_DEFAULT_MODEL   LLM model                   (default: zhipu/glm-5)
    CLAWBENCH_LLM_API_KEY     API key
    CLAWBENCH_LLM_BASE_URL    Base URL
    CLAWBENCH_PATH            Path to clawbench directory
"""

import argparse
import asyncio
import hashlib
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Optional

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from trajectoryrl.utils.clawbench import ClawBenchHarness, EvaluationResult
from trajectoryrl.utils.opp_schema import validate_opp_schema
from trajectoryrl.utils.github import PackFetcher
from trajectoryrl.utils.commitments import fetch_all_commitments
from trajectoryrl.utils.epoch_context import generate_epoch_context, render_context_preamble

logger = logging.getLogger("eval_single_miner")

DEFAULT_SCENARIOS = [
    "client_escalation",
    "morning_brief",
    "inbox_to_action",
    "team_standup",
    "inbox_triage",
]


def compute_epoch_seed(epoch: int, netuid: int = 11) -> int:
    raw = f"trajectoryrl-{netuid}-epoch-{epoch}".encode()
    return int(hashlib.sha256(raw).hexdigest()[:8], 16)


async def run_evaluation(args):
    import bittensor as bt

    # --- 1. Connect to chain (read-only) ---
    logger.info(f"Connecting to {args.network} network...")
    subtensor = bt.Subtensor(network=args.network)
    metagraph = subtensor.metagraph(args.netuid)
    current_block = subtensor.get_current_block()

    logger.info(f"Network: {args.network}, Netuid: {args.netuid}, Block: {current_block}")
    logger.info(f"Metagraph: {len(metagraph.hotkeys)} UIDs")

    # --- 2. Validate miner UID ---
    miner_uid = args.miner_uid
    if miner_uid < 0 or miner_uid >= len(metagraph.hotkeys):
        logger.error(f"Invalid miner UID {miner_uid} (metagraph has {len(metagraph.hotkeys)} UIDs)")
        return 1

    miner_hotkey = metagraph.hotkeys[miner_uid]
    logger.info(f"Miner UID: {miner_uid}, Hotkey: {miner_hotkey}")

    # --- 3. Read on-chain commitment ---
    logger.info("Reading on-chain commitments...")
    commitments = fetch_all_commitments(subtensor, args.netuid, metagraph)

    if miner_uid not in commitments:
        logger.error(f"No valid commitment found for miner UID {miner_uid}")
        # Show available commitments for debugging
        if commitments:
            logger.info(f"Available UIDs with commitments: {sorted(commitments.keys())}")
        return 1

    commitment = commitments[miner_uid]
    logger.info(f"Commitment: hash={commitment.pack_hash[:16]}..., url={commitment.pack_url}")
    logger.info(f"Commitment block: {commitment.block_number}")

    # --- 4. Fetch and verify pack ---
    logger.info("Fetching and verifying pack...")
    fetcher = PackFetcher()
    verification = await fetcher.verify_submission(
        pack_url=commitment.pack_url,
        pack_hash=commitment.pack_hash,
    )

    if not verification.valid:
        logger.error(f"Pack verification failed: {verification.error}")
        return 1

    pack = verification.pack_content
    logger.info(f"Pack verified: {len(pack.get('files', {}))} files")
    for fname in pack.get("files", {}):
        logger.info(f"  - {fname}")

    # --- 5. Schema validation ---
    lint_result = validate_opp_schema(pack)
    if not lint_result.passed:
        logger.error(f"Schema validation failed: {lint_result.issues}")
        if not args.force:
            return 1
        logger.warning("--force specified, continuing despite schema errors")
    else:
        logger.info("Schema validation passed")

    # --- 6. Prepare ClawBench harness ---
    clawbench_path = Path(args.clawbench_path)
    model = args.model or os.getenv("CLAWBENCH_DEFAULT_MODEL", "zhipu/glm-5")
    api_key = args.api_key or os.getenv("CLAWBENCH_LLM_API_KEY", "")
    base_url = args.base_url or os.getenv(
        "CLAWBENCH_LLM_BASE_URL", "https://open.bigmodel.cn/api/paas/v4"
    )

    if not api_key:
        logger.error("No LLM API key configured. Set CLAWBENCH_LLM_API_KEY or --api-key")
        return 1

    logger.info(f"ClawBench path: {clawbench_path}")
    logger.info(f"Model: {model}")
    logger.info(f"Base URL: {base_url}")
    logger.info(f"Timeout: {args.timeout}s per scenario")

    workspace_path = Path(args.workspace_path) if args.workspace_path else None

    harness = ClawBenchHarness(
        clawbench_path=clawbench_path,
        timeout=args.timeout,
        workspace_path=workspace_path,
        clawbench_default_model=model,
        clawbench_api_key=api_key,
        clawbench_base_url=base_url,
    )

    # --- 7. Generate epoch context (same as validator) ---
    epoch = current_block // 7200  # eval_interval_blocks default
    epoch_seed = args.seed if args.seed is not None else compute_epoch_seed(epoch, args.netuid)
    epoch_ctx = generate_epoch_context(epoch_seed)
    context_preamble = render_context_preamble(epoch_ctx)
    user_context = epoch_ctx.to_user_context()

    logger.info(f"Epoch seed: {epoch_seed}")
    logger.info(f"Context: [{epoch_ctx.user_name}, {epoch_ctx.user_role}]")

    # --- 8. Run evaluations ---
    scenarios = args.scenarios or DEFAULT_SCENARIOS
    logger.info(f"Scenarios to evaluate: {scenarios}")

    results: Dict[str, EvaluationResult] = {}
    total_cost = 0.0

    for scenario_name in scenarios:
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Running scenario: {scenario_name}")
        logger.info(f"{'=' * 60}")

        try:
            if args.num_runs > 1:
                result = await harness.evaluate_pack_consensus(
                    pack=pack,
                    scenario_name=scenario_name,
                    num_runs=args.num_runs,
                    base_seed=epoch_seed,
                    context_preamble=context_preamble,
                    user_context=user_context,
                )
            else:
                result = await harness.evaluate_pack(
                    pack=pack,
                    scenario_name=scenario_name,
                    seed=epoch_seed,
                    context_preamble=context_preamble,
                    user_context=user_context,
                )

            results[scenario_name] = result

            gate = "PASS" if result.success else "FAIL"
            logger.info(f"  Score: {result.score:.3f}")
            logger.info(f"  Gate:  {gate}")
            logger.info(f"  Tool calls: {result.tool_calls}")

            if result.cost_usd is not None:
                logger.info(f"  Cost: ${result.cost_usd:.4f}")
                total_cost += result.cost_usd

            if result.token_usage:
                tu = result.token_usage
                logger.info(
                    f"  Tokens: input={tu.get('input_tokens', 0)}, "
                    f"output={tu.get('output_tokens', 0)}, "
                    f"cache_read={tu.get('cache_read_tokens', 0)}"
                )

            if result.model_usage:
                for m in result.model_usage:
                    logger.info(
                        f"  Model: {m.get('model', '?')} "
                        f"${m.get('cost_usd', 0):.4f} "
                        f"({m.get('count', 0)} calls)"
                    )

            if result.error:
                logger.error(f"  Error: {result.error}")

            if result.rubric and args.verbose:
                logger.info("  Rubric:")
                checks = result.rubric.get("checks", result.rubric)
                if isinstance(checks, list):
                    for check in checks:
                        status = "PASS" if check.get("passed") else "FAIL"
                        logger.info(
                            f"    [{status}] {check.get('id', '?')} "
                            f"({check.get('points', 0)}/{check.get('max_points', 0)} pts) "
                            f"- {check.get('description', '')}"
                        )
                elif isinstance(checks, dict):
                    for check_id, check_data in checks.items():
                        if isinstance(check_data, dict):
                            status = "PASS" if check_data.get("passed") else "FAIL"
                            logger.info(
                                f"    [{status}] {check_id} "
                                f"({check_data.get('points', 0)}/{check_data.get('max_points', 0)} pts)"
                            )

        except Exception as e:
            logger.error(f"  EXCEPTION: {e}", exc_info=True)
            results[scenario_name] = EvaluationResult(
                scenario_name=scenario_name,
                score=0.0,
                success=False,
                tool_calls=0,
                response="",
                rubric={},
                error=str(e),
            )

    # --- 9. Summary ---
    print("\n")
    print("=" * 70)
    print(f"EVALUATION SUMMARY — Miner UID {miner_uid} ({miner_hotkey[:16]}...)")
    print(f"Pack hash: {commitment.pack_hash[:16]}...")
    print(f"Pack URL:  {commitment.pack_url}")
    print("=" * 70)
    print(f"{'Scenario':<25} {'Score':>8} {'Gate':>6} {'Cost':>10} {'Calls':>6}")
    print("-" * 70)

    total_score = 0.0
    num_passed = 0

    for scenario_name, result in results.items():
        gate = "PASS" if result.success else "FAIL"
        cost_str = f"${result.cost_usd:.4f}" if result.cost_usd is not None else "N/A"
        print(
            f"{scenario_name:<25} {result.score:>8.3f} {gate:>6} "
            f"{cost_str:>10} {result.tool_calls:>6}"
        )
        total_score += result.score
        if result.success:
            num_passed += 1

    print("-" * 70)
    avg_score = total_score / len(results) if results else 0.0
    cost_str = f"${total_cost:.4f}" if total_cost > 0 else "N/A"
    print(
        f"{'AVERAGE':<25} {avg_score:>8.3f} "
        f"{num_passed}/{len(results):>3} {cost_str:>10}"
    )
    print("=" * 70)

    all_passed = num_passed == len(results)
    print(f"\nQualification: {'FULLY QUALIFIED' if all_passed else 'NOT QUALIFIED'}")
    print(f"  Passed: {num_passed}/{len(results)} scenarios")

    # --- 10. Optional JSON output ---
    if args.output:
        output_data = {
            "miner_uid": miner_uid,
            "miner_hotkey": miner_hotkey,
            "pack_hash": commitment.pack_hash,
            "pack_url": commitment.pack_url,
            "commitment_block": commitment.block_number,
            "eval_block": current_block,
            "epoch_seed": epoch_seed,
            "scenarios": {},
        }
        for name, r in results.items():
            output_data["scenarios"][name] = {
                "score": r.score,
                "success": r.success,
                "tool_calls": r.tool_calls,
                "cost_usd": r.cost_usd,
                "token_usage": r.token_usage,
                "model_usage": r.model_usage,
                "rubric": r.rubric,
                "error": r.error,
                "response": r.response,
            }
        output_data["summary"] = {
            "avg_score": avg_score,
            "passed": num_passed,
            "total": len(results),
            "fully_qualified": all_passed,
            "total_cost_usd": total_cost,
        }
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        print(f"\nFull results written to: {args.output}")

    return 0 if all_passed else 1


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a single miner by UID using on-chain commitment data. "
                    "Read-only: no weight setting, no EMA persistence, no on-chain writes.",
    )

    parser.add_argument(
        "--miner-uid", type=int, required=True,
        help="Miner UID to evaluate",
    )

    # Network config
    parser.add_argument("--network", type=str, default=None, help="Bittensor network (default: from env or finney)")
    parser.add_argument("--netuid", type=int, default=None, help="Subnet UID (default: from env or 11)")
    parser.add_argument("--wallet-name", type=str, default=None, help="Wallet name (not required for read-only)")
    parser.add_argument("--wallet-hotkey", type=str, default=None, help="Wallet hotkey (not required for read-only)")

    # Evaluation config
    parser.add_argument(
        "--scenarios", nargs="+", default=None,
        help=f"Scenarios to evaluate (default: all). Available: {', '.join(DEFAULT_SCENARIOS)}",
    )
    parser.add_argument("--num-runs", type=int, default=1, help="Runs per scenario for consensus (default: 1)")
    parser.add_argument("--seed", type=int, default=None, help="Override epoch seed (default: auto from chain block)")
    parser.add_argument("--timeout", type=int, default=120, help="Timeout per scenario in seconds (default: 120)")

    # LLM config
    parser.add_argument("--model", type=str, default=None, help="LLM model (e.g. zhipu/glm-5)")
    parser.add_argument("--api-key", type=str, default=None, help="API key for the LLM provider")
    parser.add_argument("--base-url", type=str, default=None, help="Base URL for the LLM API")

    # Paths
    parser.add_argument(
        "--clawbench-path", type=str,
        default=os.getenv("CLAWBENCH_PATH", str(PROJECT_ROOT / "clawbench")),
        help="Path to clawbench directory",
    )
    parser.add_argument("--workspace-path", type=str, default=None, help="Workspace path for OpenClaw")

    # Output
    parser.add_argument("--output", "-o", type=str, default=None, help="Write full results to JSON file")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show detailed rubric results")
    parser.add_argument("--force", action="store_true", help="Continue even if schema validation fails")
    parser.add_argument(
        "--log-level", type=str, default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )

    args = parser.parse_args()

    # Apply env defaults for network config
    if args.network is None:
        args.network = os.getenv("NETWORK", "finney")
    if args.netuid is None:
        args.netuid = int(os.getenv("NETUID", "11"))

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )

    # Load .env.validator if available
    env_path = PROJECT_ROOT / ".env.validator"
    if env_path.exists():
        try:
            from dotenv import load_dotenv
            load_dotenv(env_path)
        except ImportError:
            pass

    exit_code = asyncio.run(run_evaluation(args))
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
