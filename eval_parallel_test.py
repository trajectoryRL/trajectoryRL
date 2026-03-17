#!/usr/bin/env python3
"""
Parallel evaluation test — mimics the validator's _evaluate_miner() flow
with parallel scenario evaluation.

Replicates the exact same environment and code paths as the production
validator, but evaluates a single pack (from a JSON file) instead of
fetching from on-chain commitments.

Usage (inside the all-in-one Docker container):
    python eval_parallel_test.py /pack.json
    python eval_parallel_test.py /pack.json --sequential  # compare with serial
"""

import asyncio
import hashlib
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("eval_parallel_test")

from trajectoryrl.utils.config import ValidatorConfig
from trajectoryrl.utils.clawbench import ClawBenchHarness, EvaluationResult
from trajectoryrl.utils.parallel_clawbench import ParallelClawBenchHarness
from trajectoryrl.utils.epoch_context import generate_epoch_context, render_context_preamble
from trajectoryrl.utils.llm_judge import TrajectoryJudge


def compute_epoch_seed(epoch: int, netuid: int = 11) -> int:
    raw = f"trajectoryrl-{netuid}-epoch-{epoch}".encode()
    return int(hashlib.sha256(raw).hexdigest()[:8], 16)


def process_scenario_result(
    trajectory_judge: TrajectoryJudge,
    scenarios: Dict[str, dict],
    miner_uid: int,
    scenario_name: str,
    result: EvaluationResult,
    scenario_costs: Dict[str, float],
    scenario_qualified: Dict[str, bool],
    scenario_token_usage: Dict[str, Dict[str, int]],
    scenario_model_usage: Dict[str, List[Dict[str, Any]]],
    scenario_judge_details: Dict[str, Dict[str, Any]],
) -> None:
    """Process a single scenario result — identical to validator._process_scenario_result."""
    if result.error:
        logger.warning(
            f"Miner {miner_uid}: {scenario_name} episode error: {result.error}"
        )
        scenario_qualified[scenario_name] = False
        return

    if result.cost_usd is not None:
        scenario_costs[scenario_name] = result.cost_usd
    if result.token_usage:
        scenario_token_usage[scenario_name] = result.token_usage
    if result.model_usage:
        scenario_model_usage[scenario_name] = result.model_usage

    # Phase 2: LLM trajectory judge
    scenario_config = scenarios.get(scenario_name, {})
    trajectory = result.trajectory or []
    judge_result = trajectory_judge.evaluate(
        scenario_config=scenario_config,
        trajectory=trajectory,
        agent_response=result.response,
    )

    qualified = judge_result.qualification_gate
    scenario_qualified[scenario_name] = qualified

    _criteria = judge_result.criteria_results
    _n = len(_criteria)
    _passed = sum(1 for cr in _criteria if cr.verdict == "PASS")
    _grounded = sum(1 for cr in _criteria if cr.grounded)
    scenario_judge_details[scenario_name] = {
        "overall_score": round(judge_result.overall_score, 4),
        "safety_passed": judge_result.safety_passed,
        "correctness_passed": judge_result.correctness_passed,
        "qualification_gate": qualified,
        "verdict": f"{_passed}/{_n}",
        "grounded": f"{_grounded}/{_n}",
        "error": judge_result.error,
    }

    cost_str = f", cost=${result.cost_usd:.4f}" if result.cost_usd is not None else ""
    gate_str = "PASS" if qualified else "FAIL"
    logger.info(
        f"Miner {miner_uid}: {scenario_name} -> "
        f"judge={judge_result.overall_score:.3f}{cost_str}, "
        f"gate={gate_str}, tool_calls={result.tool_calls}"
    )

    if judge_result.error:
        logger.warning(f"Miner {miner_uid}: {scenario_name} judge error: {judge_result.error}")

    for cr in judge_result.criteria_results:
        if cr.verdict != "PASS":
            logger.info(f"  FAIL {cr.id}: {cr.justification}")


async def run_parallel(
    pack: dict,
    config: ValidatorConfig,
    scenarios: Dict[str, dict],
    eval_scenarios: List[str],
    epoch_seed: int,
    context_preamble: str,
    trajectory_judge: "TrajectoryJudge",
    user_context: Optional[Dict] = None,
) -> Dict[str, Any]:
    """Run all scenarios in parallel — episode + judge per scenario.

    Each scenario runs its episode then immediately runs the LLM judge,
    all in parallel. No scenario waits for others.
    """
    logger.info("=" * 60)
    logger.info("PARALLEL EVALUATION (episode + judge per slot)")
    logger.info("=" * 60)

    num_slots = len(eval_scenarios)
    logger.info(f"Starting {num_slots} parallel service slots...")

    harness = ParallelClawBenchHarness(
        num_slots=num_slots,
        clawbench_path=config.clawbench_path,
        scenario_names=eval_scenarios,
        timeout=config.timeout_per_scenario,
        openclaw_bin=config.openclaw_bin,
        clawbench_default_model=config.clawbench_default_model,
        clawbench_api_key=config.clawbench_api_key,
        clawbench_base_url=config.clawbench_base_url,
    )

    start_time = time.time()
    await harness.start()
    startup_time = time.time() - start_time
    logger.info(f"All {num_slots} slots started in {startup_time:.1f}s")

    # Shared result dicts (populated by each parallel task)
    scenario_costs: Dict[str, float] = {}
    scenario_qualified: Dict[str, bool] = {}
    scenario_token_usage: Dict[str, Dict[str, int]] = {}
    scenario_model_usage: Dict[str, List[Dict[str, Any]]] = {}
    scenario_judge_details: Dict[str, Dict[str, Any]] = {}

    try:
        eval_start = time.time()

        async def _eval_and_judge(scenario_name: str) -> None:
            result = await harness.evaluate_scenario(
                pack=pack,
                scenario_name=scenario_name,
                seed=epoch_seed,
                context_preamble=context_preamble,
                user_context=user_context,
            )
            # Run blocking judge call in a thread so it doesn't block
            # the event loop (other scenarios' episodes keep running)
            await asyncio.to_thread(
                process_scenario_result,
                trajectory_judge=trajectory_judge,
                scenarios=scenarios,
                miner_uid=0,
                scenario_name=scenario_name,
                result=result,
                scenario_costs=scenario_costs,
                scenario_qualified=scenario_qualified,
                scenario_token_usage=scenario_token_usage,
                scenario_model_usage=scenario_model_usage,
                scenario_judge_details=scenario_judge_details,
            )

        await asyncio.gather(*[_eval_and_judge(s) for s in eval_scenarios])

        eval_time = time.time() - eval_start
        logger.info(f"Parallel evaluation completed in {eval_time:.1f}s")

        return {
            "scenario_costs": scenario_costs,
            "scenario_qualified": scenario_qualified,
            "scenario_judge_details": scenario_judge_details,
            "eval_time": eval_time,
            "startup_time": startup_time,
        }
    finally:
        await harness.stop()


async def run_sequential(
    pack: dict,
    config: ValidatorConfig,
    scenarios: Dict[str, dict],
    eval_scenarios: List[str],
    epoch_seed: int,
    context_preamble: str,
    user_context: Optional[Dict] = None,
) -> Dict[str, Any]:
    """Run scenarios sequentially — exactly like the current validator."""
    logger.info("=" * 60)
    logger.info("SEQUENTIAL EVALUATION")
    logger.info("=" * 60)

    harness = ClawBenchHarness(
        clawbench_path=config.clawbench_path,
        timeout=config.timeout_per_scenario,
        clawbench_default_model=config.clawbench_default_model,
        clawbench_api_key=config.clawbench_api_key,
        clawbench_base_url=config.clawbench_base_url,
    )

    results: Dict[str, EvaluationResult] = {}
    eval_start = time.time()

    for idx, scenario_name in enumerate(eval_scenarios, 1):
        logger.info(f"Scenario [{idx}/{len(eval_scenarios)}] {scenario_name}...")
        result = await harness.evaluate_pack(
            pack=pack,
            scenario_name=scenario_name,
            seed=epoch_seed,
            context_preamble=context_preamble,
            user_context=user_context,
        )
        results[scenario_name] = result

    eval_time = time.time() - eval_start
    logger.info(f"Sequential evaluation completed in {eval_time:.1f}s")

    return {
        "results": results,
        "eval_time": eval_time,
        "startup_time": 0.0,
    }


async def main():
    import argparse
    parser = argparse.ArgumentParser(description="Parallel evaluation test")
    parser.add_argument("pack_path", nargs="?", default="/pack.json",
                        help="Path to pack JSON file")
    parser.add_argument("--sequential", action="store_true",
                        help="Run sequentially instead of parallel (for comparison)")
    args = parser.parse_args()

    # Load pack
    with open(args.pack_path) as f:
        pack = json.load(f)
    logger.info(f"Loaded pack from {args.pack_path}")

    # --- Exactly what TrajectoryValidator.__init__ does ---
    config = ValidatorConfig.from_env()
    # Force parallel mode
    config.parallel_scenarios = True

    logger.info(f"Model:     {config.clawbench_default_model}")
    logger.info(f"Base URL:  {config.clawbench_base_url}")
    logger.info(f"API key:   {config.clawbench_api_key[:8]}..." if config.clawbench_api_key else "API key:   (not set)")
    logger.info(f"Clawbench: {config.clawbench_path}")
    logger.info(f"Scenarios: {config.scenarios}")

    # Load scenario configs (same as validator._load_scenarios)
    import yaml
    scenarios: Dict[str, dict] = {}
    for scenario_name in config.scenarios:
        scenario_path = config.scenarios_path / f"{scenario_name}.yaml"
        if scenario_path.exists():
            with open(scenario_path) as f:
                scenarios[scenario_name] = yaml.safe_load(f)
    logger.info(f"Loaded {len(scenarios)} scenarios")

    # Initialize LLM judge (same as validator)
    judge_model = config.judge_model or config.clawbench_default_model
    judge_api_key = config.judge_api_key or config.clawbench_api_key
    judge_base_url = config.judge_base_url or config.clawbench_base_url
    trajectory_judge = TrajectoryJudge(
        model=judge_model,
        api_key=judge_api_key,
        base_url=judge_base_url,
    )

    # Epoch context (same as validator._run_evaluation_cycle)
    current_block = 0
    epoch = current_block // config.eval_interval_blocks
    epoch_seed = compute_epoch_seed(epoch, config.netuid)
    epoch_ctx = generate_epoch_context(epoch_seed)
    context_preamble = render_context_preamble(epoch_ctx)
    user_context = epoch_ctx.to_user_context()

    logger.info(f"Epoch context (seed={epoch_seed}):")
    logger.info(f"  Date:     {epoch_ctx.weekday}, {epoch_ctx.date_str}")
    logger.info(f"  Identity: {epoch_ctx.user_name} / {epoch_ctx.user_role} @ {epoch_ctx.company}")

    eval_scenarios = sorted(config.scenarios)

    # Run evaluation
    if args.sequential:
        eval_data = await run_sequential(
            pack, config, scenarios, eval_scenarios,
            epoch_seed, context_preamble, user_context,
        )
        # Sequential: judge runs after all episodes (existing behavior)
        scenario_costs: Dict[str, float] = {}
        scenario_qualified: Dict[str, bool] = {}
        scenario_token_usage: Dict[str, Dict[str, int]] = {}
        scenario_model_usage: Dict[str, List[Dict[str, Any]]] = {}
        scenario_judge_details: Dict[str, Dict[str, Any]] = {}
        for scenario_name, result in eval_data["results"].items():
            process_scenario_result(
                trajectory_judge=trajectory_judge,
                scenarios=scenarios,
                miner_uid=0,
                scenario_name=scenario_name,
                result=result,
                scenario_costs=scenario_costs,
                scenario_qualified=scenario_qualified,
                scenario_token_usage=scenario_token_usage,
                scenario_model_usage=scenario_model_usage,
                scenario_judge_details=scenario_judge_details,
            )
    else:
        eval_data = await run_parallel(
            pack, config, scenarios, eval_scenarios,
            epoch_seed, context_preamble, trajectory_judge,
            user_context,
        )
        # Parallel: judge already ran inside each parallel task
        scenario_costs = eval_data["scenario_costs"]
        scenario_qualified = eval_data["scenario_qualified"]
        scenario_judge_details = eval_data["scenario_judge_details"]

    # --- Summary ---
    mode = "SEQUENTIAL" if args.sequential else "PARALLEL"
    print(f"\n{'='*60}")
    print(f"  {mode} EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"  {'Scenario':<25} {'Gate':>4}  {'Cost':>8}  {'Judge':>6}")
    print(f"  {'-'*25} {'-'*4}  {'-'*8}  {'-'*6}")
    for s in eval_scenarios:
        gate = "PASS" if scenario_qualified.get(s) else "FAIL"
        cost = f"${scenario_costs[s]:.4f}" if s in scenario_costs else "n/a"
        jd = scenario_judge_details.get(s, {})
        judge_score = f"{jd.get('overall_score', 0):.3f}" if jd else "n/a"
        print(f"  {s:<25} {gate:>4}  {cost:>8}  {judge_score:>6}")
    print(f"  {'-'*25} {'-'*4}  {'-'*8}  {'-'*6}")

    total_cost = sum(scenario_costs.values())
    passed = sum(1 for v in scenario_qualified.values() if v)
    total = len(scenario_qualified)
    print(f"  Qualification: {passed}/{total} scenarios passed")
    print(f"  Total cost:    ${total_cost:.4f}")
    print(f"  Eval time:     {eval_data['eval_time']:.1f}s")
    if eval_data.get("startup_time"):
        print(f"  Startup time:  {eval_data['startup_time']:.1f}s")
    print(f"  Total time:    {eval_data['eval_time'] + eval_data.get('startup_time', 0):.1f}s")
    print(f"{'='*60}")


if __name__ == "__main__":
    asyncio.run(main())
