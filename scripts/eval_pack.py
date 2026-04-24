#!/usr/bin/env python3
"""Evaluate a local pack file as a validator would (Season 1 sandbox).

Reads a local Season 1 pack JSON (or a bare SKILL.md), validates schema,
runs the trajrl-bench 3-container evaluation (sandbox + testee + judge),
and prints the split-half delta scoring summary.

No chain connection needed — pure local evaluation. Same harness and
scoring as ``scripts/eval_miners.py`` but skips the on-chain commitment
fetch in favour of a local pack file.

Configuration is taken from ``.env.validator`` via ``ValidatorConfig.from_env``.
The CLI exposes only the few flags that are unique to local pack
debugging: which pack/SKILL.md to evaluate, how to render output, and
an optional epoch seed override for reproducibility.

Prerequisites:
    1. Docker daemon running
    2. LLM API key configured in ``.env.validator`` (LLM_API_KEY)

Usage:
    python scripts/eval_pack.py --pack pack.json
    python scripts/eval_pack.py --skill-md SKILL.md
    python scripts/eval_pack.py --pack pack.json -o ./eval_output
    python scripts/eval_pack.py --pack pack.json --json results.json

To override env values temporarily:
    LLM_MODEL=openai/gpt-4o python scripts/eval_pack.py --pack pack.json
    SANDBOX_NUM_EPISODES=2 python scripts/eval_pack.py --pack pack.json
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

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from trajectoryrl.utils.config import SPEC_NUMBER, ValidatorConfig
from trajectoryrl.utils.sandbox_harness import (
    SandboxEvaluationResult,
    TrajectorySandboxHarness,
)
from trajectoryrl.base.miner import TrajectoryMiner

logger = logging.getLogger("eval_pack")


def _setup_logging(level_name: str) -> None:
    """Configure stderr logging for the CLI and project loggers."""
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
        "eval_pack",
        "trajectoryrl",
        "trajectoryrl.utils.sandbox_harness",
    ):
        lg = logging.getLogger(name)
        lg.setLevel(level)
        lg.propagate = True


def _load_pack(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def _pack_from_skill_md(path: str) -> dict:
    """Wrap a bare SKILL.md file into a minimal S1 pack."""
    content = Path(path).read_text()
    return TrajectoryMiner.build_s1_pack(content)


def _compute_pack_hash(pack: dict) -> str:
    canonical = json.dumps(pack, sort_keys=True)
    return hashlib.sha256(canonical.encode()).hexdigest()


def _compute_local_salt(pack_hash: str) -> str:
    """Deterministic per-pack salt for local evaluation runs."""
    data = f"eval_pack_cli:{pack_hash}".encode()
    return hashlib.sha256(data).hexdigest()[:16]


def _print_summary(
    source: str,
    pack_hash: str,
    result: SandboxEvaluationResult,
    sandbox_version: str,
    spec_number: int,
) -> bool:
    """Render a human-readable summary; return qualified flag."""
    W = 70
    sr = result.session_result
    print("\n" + "=" * W)
    print("PACK EVALUATION SUMMARY")
    print(f"Source:          {source}")
    print(f"Pack hash:       {pack_hash[:16]}...")
    print(f"Sandbox version: {sandbox_version}")
    print(f"Spec number:     {spec_number}")
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
    _setup_logging(args.log_level)

    # Redirect cache + logs to a temp dir BEFORE from_env runs, so
    # ValidatorConfig.__post_init__ doesn't try to mkdir under
    # /var/lib/trajectoryrl (which the CLI lacks permission for).
    tmp_dir = Path(tempfile.mkdtemp(prefix="eval_pack_"))
    os.environ.setdefault("PACK_CACHE_DIR", str(tmp_dir / "packs"))
    os.environ.setdefault("LOG_DIR", str(tmp_dir / "logs"))

    config = ValidatorConfig.from_env(dotenv_path=PROJECT_ROOT / ".env.validator")

    if not config.llm_api_key:
        logger.error("No LLM API key configured. Set LLM_API_KEY in .env.validator")
        return 1

    if args.pack:
        logger.info("Loading pack: %s", args.pack)
        pack = _load_pack(args.pack)
        source = args.pack
    else:
        logger.info("Building pack from: %s", args.skill_md)
        pack = _pack_from_skill_md(args.skill_md)
        source = args.skill_md

    pack_hash = _compute_pack_hash(pack)
    logger.info("Pack hash: %s...", pack_hash[:16])
    logger.info("Files:     %s", list(pack.get("files", {}).keys()))

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

    harness = TrajectorySandboxHarness(config)
    logger.info("Model:           %s", config.llm_model)
    logger.info("Base URL:        %s", config.llm_base_url)
    logger.info("Sandbox image:   %s", config.sandbox_image)
    logger.info("Harness image:   %s", config.harness_image)
    logger.info("Episodes:        %d", config.sandbox_num_episodes)
    logger.info("Timeout/episode: %ds", config.sandbox_timeout_per_episode)

    if args.no_pull:
        logger.info("Skipping image pull (--no-pull)")
    else:
        logger.info("Pulling latest sandbox + harness images...")
        await harness.pull_latest()
    logger.info(
        "Sandbox %s, spec_number %d, scenarios=%s",
        harness.sandbox_version, SPEC_NUMBER,
        harness.sandbox_scenarios or "?",
    )

    epoch_seed = (
        args.seed if args.seed is not None
        else int(hashlib.sha256(pack_hash.encode()).hexdigest()[:8], 16)
    )
    validator_salt = _compute_local_salt(pack_hash)
    logger.info("Epoch seed:    %d", epoch_seed)
    logger.info("Validator salt: %s", validator_salt)

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

    qualified = _print_summary(
        source=source,
        pack_hash=pack_hash,
        result=result,
        sandbox_version=harness.sandbox_version,
        spec_number=SPEC_NUMBER,
    )

    if args.output:
        out_dir = Path(args.output)
        out_dir.mkdir(parents=True, exist_ok=True)
        result.write_artifacts(out_dir)
        logger.info("Artifacts written to %s", out_dir)
        print(f"\nEval artifacts written to: {out_dir}")

    if args.json_output:
        sr = result.session_result
        payload = {
            "source": source,
            "pack_hash": pack_hash,
            "epoch_seed": epoch_seed,
            "validator_salt": validator_salt,
            "sandbox_version": harness.sandbox_version,
            "spec_number": SPEC_NUMBER,
            "scenario": result.scenario_name,
            "model": config.llm_model,
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
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print(f"\nJSON results written to: {args.json_output}")

    return 0 if qualified else 1


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a local pack against the Season 1 trajrl-bench "
            "sandbox. No chain connection required. All configuration "
            "(LLM, episodes, images, ...) is taken from .env.validator; "
            "override via env vars when running."
        ),
    )

    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--pack", help="Season 1 pack JSON file")
    src.add_argument(
        "--skill-md", dest="skill_md",
        help="Bare SKILL.md file (auto-wrapped into a minimal S1 pack)",
    )

    parser.add_argument(
        "--seed", type=int, default=None,
        help="Override epoch seed (default: derived from pack hash)",
    )
    parser.add_argument(
        "--no-pull", action="store_true",
        help="Skip docker pull of sandbox + harness images",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Continue even if S1 schema validation fails",
    )
    parser.add_argument(
        "--output", "-o", type=str, default=None,
        help="Write eval artifacts (transcripts, evaluation.json) to this directory",
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
