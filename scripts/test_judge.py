#!/usr/bin/env python3
"""Test the TrajectoryJudge against real miner data.

Usage:
    # Dry-run: print the prompt the judge would see (no LLM call)
    python scripts/test_judge.py --miner-json ../miner\ 233.json --dry-run

    # Live: actually call the LLM and show results
    python scripts/test_judge.py --miner-json ../miner\ 233.json

    # With a custom agent response file
    python scripts/test_judge.py --miner-json ../miner\ 233.json --response-file response.txt

    # Override scenario
    python scripts/test_judge.py --miner-json ../miner\ 233.json --scenario team_standup

Requires .env.validator (or env vars) for LLM credentials.
"""

import argparse
import json
import logging
import os
import sys
import yaml
from pathlib import Path
from dotenv import load_dotenv

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from trajectoryrl.utils.llm_judge import TrajectoryJudge, MAX_TOOL_RESPONSE_CHARS
from trajectoryrl.utils.judge_prompts import TRAJECTORY_JUDGE_SYSTEM

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)
logger = logging.getLogger(__name__)


def load_env():
    """Load .env.validator from project root."""
    env_path = ROOT / ".env.validator"
    if env_path.exists():
        load_dotenv(env_path)
        logger.info("Loaded %s", env_path)
    else:
        logger.warning("No .env.validator found at %s", env_path)


def load_scenario_config(scenario_name: str) -> dict:
    """Load scenario YAML from scenarios/."""
    scenario_path = ROOT / "scenarios" / f"{scenario_name}.yaml"
    if not scenario_path.exists():
        raise FileNotFoundError(f"Scenario not found: {scenario_path}")
    with open(scenario_path) as f:
        return yaml.safe_load(f)


def extract_trajectory(miner_data: dict) -> list:
    """Extract tool calls from miner JSON into judge-compatible format."""
    tool_calls = miner_data.get("tool_calls", [])
    trajectory = []
    for tc in tool_calls:
        trajectory.append({
            "tool": tc.get("tool", "unknown"),
            "args": tc.get("args", {}),
            "response": tc.get("response", ""),
        })
    return trajectory


def print_prompt_stats(judge: TrajectoryJudge, scenario_config: dict,
                       trajectory: list, agent_response: str):
    """Print the formatted prompt and stats without calling the LLM."""
    criteria = (
        scenario_config.get("scoring", {}).get("criteria")
        or scenario_config.get("scoring", {}).get("checks", [])
    )
    prompt = judge._build_user_prompt(scenario_config, trajectory, agent_response, criteria)

    print("\n" + "=" * 80)
    print("SYSTEM PROMPT")
    print("=" * 80)
    print(TRAJECTORY_JUDGE_SYSTEM)

    print("\n" + "=" * 80)
    print("USER PROMPT")
    print("=" * 80)
    print(prompt)

    print("\n" + "=" * 80)
    print("STATS")
    print("=" * 80)
    print(f"  System prompt chars : {len(TRAJECTORY_JUDGE_SYSTEM):,}")
    print(f"  User prompt chars   : {len(prompt):,}")
    print(f"  Total chars         : {len(TRAJECTORY_JUDGE_SYSTEM) + len(prompt):,}")
    print(f"  Approx tokens       : ~{(len(TRAJECTORY_JUDGE_SYSTEM) + len(prompt)) // 4:,}")
    print(f"  Tool calls          : {len(trajectory)}")
    print(f"  MAX_TOOL_RESPONSE   : {MAX_TOOL_RESPONSE_CHARS}")
    print(f"  Criteria count      : {len(criteria)}")
    for c in criteria:
        print(f"    - {c['id']} ({c.get('category', '?')}, w={c.get('weight', 1)})")

    return prompt


def print_results(result):
    """Pretty-print judge results."""
    print("\n" + "=" * 80)
    print("JUDGE RESULT")
    print("=" * 80)
    print(f"  qualification_gate  : {result.qualification_gate}")
    print(f"  safety_passed       : {result.safety_passed}")
    print(f"  correctness_passed  : {result.correctness_passed}")
    print(f"  overall_score       : {result.overall_score:.3f}")
    if result.error:
        print(f"  error               : {result.error}")

    print("\n  Per-criterion results:")
    print(f"  {'ID':<30} {'Verdict':<8} {'Grounded':<10} Justification")
    print("  " + "-" * 78)
    for cr in result.criteria_results:
        verdict_mark = "PASS" if cr.verdict == "PASS" else "FAIL"
        grounded_mark = "yes" if cr.grounded else "NO"
        print(f"  {cr.id:<30} {verdict_mark:<8} {grounded_mark:<10} {cr.justification[:60]}")


def main():
    parser = argparse.ArgumentParser(description="Test TrajectoryJudge with real miner data")
    parser.add_argument("--miner-json", required=True, help="Path to miner JSON file")
    parser.add_argument("--scenario", help="Override scenario name (default: from miner JSON)")
    parser.add_argument("--response-file", help="Path to agent response text file")
    parser.add_argument("--response", help="Inline agent response text")
    parser.add_argument("--dry-run", action="store_true", help="Print prompt only, no LLM call")
    parser.add_argument("--model", help="Override LLM model")
    args = parser.parse_args()

    load_env()

    # Load miner data
    with open(args.miner_json) as f:
        miner_data = json.load(f)

    scenario_name = args.scenario or miner_data.get("scenario", "unknown")
    logger.info("Scenario: %s", scenario_name)
    logger.info("Miner: uid=%s, pack_hash=%s",
                miner_data.get("miner", {}).get("uid"),
                miner_data.get("miner", {}).get("pack_hash"))

    # Load scenario config
    scenario_config = load_scenario_config(scenario_name)

    # Extract trajectory
    trajectory = extract_trajectory(miner_data)
    logger.info("Trajectory: %d tool calls", len(trajectory))

    # Get agent response
    if args.response_file:
        with open(args.response_file) as f:
            agent_response = f.read()
    elif args.response:
        agent_response = args.response
    else:
        agent_response = "(Agent response not available — evaluate trajectory only)"
        logger.warning("No agent response provided. Use --response-file or --response.")

    # Create judge
    judge = TrajectoryJudge(
        model=args.model or os.environ.get("LLM_MODEL") or os.environ.get("CLAWBENCH_DEFAULT_MODEL", ""),
    )

    # Print prompt stats (always)
    print_prompt_stats(judge, scenario_config, trajectory, agent_response)

    if args.dry_run:
        print("\n[DRY RUN — no LLM call made]")
        return

    # Run the judge
    print("\n" + "=" * 80)
    print("CALLING LLM JUDGE...")
    print("=" * 80)

    import asyncio
    result = asyncio.run(judge.evaluate(scenario_config, trajectory, agent_response))
    print_results(result)


if __name__ == "__main__":
    main()
