#!/usr/bin/env python3
"""Evaluate a local pack file as a validator would (Season 1 sandbox).

exec python -u scripts/eval_pack.py --pack pack.json "$@"

Reads a local Season 1 pack JSON (or a bare SKILL.md), validates schema,
runs the trajrl-bench 3-container evaluation (sandbox + testee + judge),
and prints the split-half delta scoring summary.

No chain connection needed — pure local evaluation. Same harness and
scoring as ``scripts/eval_miners.py`` but skips the on-chain commitment
fetch in favour of a local pack file.

Prerequisites:
    1. Docker daemon running
    2. LLM API key configured (env var or --api-key)

Usage:
    # Evaluate a pack JSON file:
    python scripts/eval_pack.py --pack pack.json

    # Or just a SKILL.md (auto-wraps into a minimal S1 pack):
    python scripts/eval_pack.py --skill-md SKILL.md

    # Override episode count:
    python scripts/eval_pack.py --pack pack.json --num-episodes 2

    # Save eval artifacts (transcripts, evaluation.json, world.json):
    python scripts/eval_pack.py --pack pack.json -o ./eval_output

    # Save summary as JSON:
    python scripts/eval_pack.py --pack pack.json --json results.json

Environment variables (also read from .env.validator):
    LLM_MODEL                 LLM model           (default: glm-5.1)
    LLM_API_KEY               API key for LLM
    LLM_BASE_URL              LLM base URL        (default: https://open.bigmodel.cn/api/paas/v4)
    SANDBOX_IMAGE             Sandbox image       (default: ghcr.io/trajectoryrl/trajrl-bench:latest)
    HARNESS_IMAGE             Hermes image        (default: ghcr.io/trajectoryrl/hermes-agent:latest)
    SANDBOX_NUM_EPISODES      Episodes per eval   (default: 4)
    SANDBOX_TIMEOUT_PER_EPISODE  Per-episode timeout in seconds  (default: 180)
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
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from trajectoryrl.utils.sandbox_harness import (
    TrajectorySandboxHarness,
    SandboxEvaluationResult,
)
from trajectoryrl.utils.config import ValidatorConfig
from trajectoryrl.base.miner import TrajectoryMiner

logger = logging.getLogger("eval_pack")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_pack(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def pack_from_skill_md(path: str) -> dict:
    """Wrap a bare SKILL.md file into a minimal S1 pack."""
    content = Path(path).read_text()
    return TrajectoryMiner.build_s1_pack(content)


def compute_pack_hash(pack: dict) -> str:
    canonical = json.dumps(pack, sort_keys=True)
    return hashlib.sha256(canonical.encode()).hexdigest()


def compute_local_salt(pack_hash: str) -> str:
    """Deterministic per-pack salt for local evaluation runs."""
    data = f"eval_pack_cli:{pack_hash}".encode()
    return hashlib.sha256(data).hexdigest()[:16]


def print_summary(
    source: str,
    pack_hash: str,
    result: SandboxEvaluationResult,
    sandbox_version: str,
    scoring_version: int,
) -> bool:
    """Print S1 evaluation summary; returns ``qualified`` boolean."""
    W = 70
    print("\n")
    print("=" * W)
    print(f"PACK EVALUATION SUMMARY")
    print(f"  source:           {source}")
    print(f"  pack hash:        {pack_hash[:16]}...")
    print(f"  sandbox version:  {sandbox_version}")
    print(f"  scoring version:  {scoring_version}")
    print(f"  scenario:         {result.scenario_name}")
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
    return qualified


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def run(args) -> int:
    # --- 1. Load pack ----------------------------------------------------
    if args.pack:
        logger.info("Loading pack: %s", args.pack)
        pack = load_pack(args.pack)
        source = args.pack
    else:
        logger.info("Building pack from: %s", args.skill_md)
        pack = pack_from_skill_md(args.skill_md)
        source = args.skill_md

    pack_hash = compute_pack_hash(pack)
    logger.info("Pack hash: %s...", pack_hash[:16])
    logger.info("Files:     %s", list(pack.get("files", {}).keys()))

    # --- 2. S1 schema validation ----------------------------------------
    issues = TrajectoryMiner.validate_s1(pack)
    if issues:
        for issue in issues:
            logger.error("  schema: %s", issue)
        if not args.force:
            return 1
        logger.warning("--force: continuing despite schema errors")
    else:
        logger.info("S1 validation passed")

    skill_md = TrajectorySandboxHarness.extract_skill_md(pack)
    if not skill_md or not skill_md.strip():
        logger.error("Pack missing or empty SKILL.md")
        return 1

    extra_files = [
        f for f in pack.get("files", {}) if f.lower() != "skill.md"
    ]
    if extra_files:
        logger.warning("S1 pack contains unexpected files: %s", extra_files)

    logger.info("SKILL.md: %d chars", len(skill_md))

    # --- 3. Build harness via ValidatorConfig ---------------------------
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

    tmp_dir = Path(tempfile.mkdtemp(prefix="eval_pack_"))
    config = ValidatorConfig(
        netuid=args.netuid,
        network="local",
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

    # --- 4. Pull latest sandbox + harness images ------------------------
    if not args.no_pull:
        logger.info("Pulling latest images...")
        await harness.pull_latest()
    else:
        logger.info("Skipping image pull (--no-pull)")
    logger.info(
        "Sandbox version: %s, scoring_version: %d",
        harness.sandbox_version, harness.scoring_version,
    )
    if harness.sandbox_scenarios:
        logger.info("Available scenarios: %s", harness.sandbox_scenarios)

    # --- 5. Compute epoch seed + salt -----------------------------------
    epoch_seed = (
        args.seed if args.seed is not None
        else int(hashlib.sha256(pack_hash.encode()).hexdigest()[:8], 16)
    )
    validator_salt = compute_local_salt(pack_hash)
    logger.info("Epoch seed:    %d", epoch_seed)
    logger.info("Validator salt: %s", validator_salt)

    # --- 6. Run S1 sandbox evaluation -----------------------------------
    logger.info("Starting S1 sandbox evaluation...")
    try:
        result = await harness.evaluate_miner(
            skill_md=skill_md,
            epoch_seed=epoch_seed,
            pack_hash=pack_hash,
            validator_salt=validator_salt,
        )
    except Exception as e:
        logger.error("S1 evaluation failed: %s", e, exc_info=True)
        return 1

    if result.error:
        logger.error("S1 evaluation error: %s", result.error)
        return 1

    for ep in result.session_result.episodes:
        logger.info(
            "  Episode %d: quality=%.3f, duration=%.1fs%s",
            ep.episode_index, ep.quality, ep.duration_s,
            " (TIMEOUT)" if ep.timed_out else "",
        )
        if ep.error:
            logger.warning("  Episode %d error: %s", ep.episode_index, ep.error)

    logger.info(
        "S1 result: final_score=%.3f, mean_quality=%.3f, delta=%.3f, "
        "episodes=%s, scenario=%s",
        result.score, result.mean_quality, result.delta,
        result.episode_qualities, result.scenario_name,
    )

    # --- 7. Summary -----------------------------------------------------
    qualified = print_summary(
        source=source,
        pack_hash=pack_hash,
        result=result,
        sandbox_version=harness.sandbox_version,
        scoring_version=harness.scoring_version,
    )

    # --- 8. Eval artifacts ----------------------------------------------
    if args.output:
        out_dir = Path(args.output)
        out_dir.mkdir(parents=True, exist_ok=True)
        result.write_artifacts(out_dir)
        print(f"\nEval artifacts written to: {out_dir}")

    # --- 9. JSON summary ------------------------------------------------
    if args.json_output:
        sr = result.session_result
        output = {
            "source": source,
            "pack_hash": pack_hash,
            "epoch_seed": epoch_seed,
            "validator_salt": validator_salt,
            "sandbox_version": harness.sandbox_version,
            "scoring_version": harness.scoring_version,
            "scenario": result.scenario_name,
            "model": model,
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
        }
        Path(args.json_output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.json_output, "w") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        print(f"\nJSON results written to: {args.json_output}")

    return 0 if qualified else 1


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a local pack against the Season 1 trajrl-bench sandbox. "
            "No chain connection required."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  python scripts/eval_pack.py --pack pack.json\n"
            "  python scripts/eval_pack.py --skill-md SKILL.md --num-episodes 2\n"
            "  python scripts/eval_pack.py --pack pack.json -o ./eval_output\n"
        ),
    )

    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--pack", help="Season 1 pack JSON file")
    src.add_argument(
        "--skill-md", dest="skill_md",
        help="Bare SKILL.md file (auto-wrapped into a minimal S1 pack)",
    )

    parser.add_argument(
        "--netuid", type=int, default=None,
        help="Subnet UID for default-salt computation (default: from env or 11)",
    )
    parser.add_argument(
        "--num-episodes", type=int, default=None,
        help="Number of episodes per evaluation (default: 4)",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Override epoch seed (default: derived from pack hash)",
    )
    parser.add_argument(
        "--no-pull", action="store_true",
        help="Skip docker pull of sandbox + harness images",
    )

    # LLM config
    parser.add_argument("--model", type=str, default=None,
                        help="LLM model (e.g. glm-5.1)")
    parser.add_argument("--api-key", type=str, default=None,
                        help="API key for the LLM provider")
    parser.add_argument("--base-url", type=str, default=None,
                        help="Base URL for the LLM API")

    # Output
    parser.add_argument(
        "--output", "-o", type=str, default=None,
        help="Write eval artifacts (transcripts, evaluation.json) to directory",
    )
    parser.add_argument(
        "--json", dest="json_output", type=str, default=None,
        help="Write summary results to JSON file",
    )
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show detailed episode logs")
    parser.add_argument("--force", action="store_true",
                        help="Continue even if schema validation fails")
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

    if args.netuid is None:
        args.netuid = int(os.getenv("NETUID", "11"))
    if args.num_episodes is None:
        args.num_episodes = int(os.getenv("SANDBOX_NUM_EPISODES", "4"))

    log_level = getattr(logging, args.log_level)
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
    )

    try:
        exit_code = asyncio.run(run(args))
    except Exception as e:
        print(f"FATAL: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        exit_code = 1
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
