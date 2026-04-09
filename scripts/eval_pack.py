#!/usr/bin/env python3
"""Evaluate a local pack file as a validator would.

exec python -u scripts/eval_pack.py --pack pack.json "$@"

Reads a local OPP v1 pack JSON (or an AGENTS.md), validates schema, runs
ClawBench evaluation on all (or selected) scenarios via the Docker services,
scores against the real rubric, and reports qualification gate + cost.

No chain connection needed — pure local evaluation.

Prerequisites:
    1. Docker services running:
         cd clawbench && docker compose up -d
    2. LLM API key configured (env var or --api-key)

Usage:
    # Evaluate a pack JSON file:
    python scripts/eval_pack.py --pack pack.json

    # Or just an AGENTS.md (auto-wraps into a minimal pack):
    python scripts/eval_pack.py --agents-md my_policy.md

    # Only run specific scenarios:
    python scripts/eval_pack.py --pack pack.json -s inbox_triage client_escalation

    # Consensus mode (3 runs per scenario, same as production validator):
    python scripts/eval_pack.py --pack pack.json -n 3

    # Verbose rubric output + save JSON:
    python scripts/eval_pack.py --pack pack.json -v -o results.json

Environment variables:
    CLAWBENCH_DEFAULT_MODEL   LLM model           (default: zhipu/glm-5.1)
    CLAWBENCH_LLM_API_KEY     API key for LLM
    CLAWBENCH_LLM_BASE_URL    LLM base URL        (default: https://open.bigmodel.cn/api/paas/v4)
"""

import argparse
import asyncio
import hashlib
import json
import logging
import os
import sys
import time
from pathlib import Path
from types import ModuleType

# ---------------------------------------------------------------------------
# Project root & mock bittensor (avoid chain dependency + argparse hijack)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

_bt = ModuleType("bittensor")
_bt.Synapse = type("Synapse", (), {})  # type: ignore[attr-defined]
sys.modules.setdefault("bittensor", _bt)

from trajectoryrl.utils.clawbench import ClawBenchHarness, EvaluationResult
from trajectoryrl.utils.opp_schema import validate_opp_schema
from trajectoryrl.utils.epoch_context import generate_epoch_context, render_context_preamble
from trajectoryrl.utils.llm_judge import TrajectoryJudge

import yaml

logger = logging.getLogger("eval_pack")

SCENARIOS = [
    "client_escalation",
    "morning_brief",
    "inbox_to_action",
    "team_standup",
    "inbox_triage",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_pack(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def pack_from_agents_md(path: str) -> dict:
    content = Path(path).read_text()
    return {
        "schema_version": 1,
        "files": {"AGENTS.md": content},
        "tool_policy": {
            "allow": ["exec", "slack", "memory_search", "memory_get",
                       "web_search", "read"],
            "deny": ["group:runtime"],
        },
        "metadata": {
            "pack_name": Path(path).stem,
            "pack_version": "1.0.0",
            "target_suite": "clawbench_v1",
        },
    }


def pack_hash(pack: dict) -> str:
    return hashlib.sha256(json.dumps(pack, sort_keys=True).encode()).hexdigest()


def check_services(mock_url: str, openclaw_url: str, token: str = "") -> bool:
    """Return True if both Docker services respond to /health."""
    import httpx
    ok = True
    for name, url in [("mock-tools", mock_url), ("openclaw", openclaw_url)]:
        try:
            headers = {}
            if token and name == "openclaw":
                headers["Authorization"] = f"Bearer {token}"
            r = httpx.get(f"{url}/health", timeout=5, headers=headers)
            if r.status_code == 200:
                logger.info(f"  {name:20s} OK")
            else:
                logger.error(f"  {name:20s} HTTP {r.status_code}")
                ok = False
        except Exception as e:
            logger.error(f"  {name:20s} UNREACHABLE ({e})")
            ok = False
    return ok


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def run(args) -> int:
    # ── 1. Load pack ──────────────────────────────────────────────────────
    if args.pack:
        logger.info(f"Loading pack: {args.pack}")
        pack = load_pack(args.pack)
    else:
        logger.info(f"Building pack from: {args.agents_md}")
        pack = pack_from_agents_md(args.agents_md)

    phash = pack_hash(pack)
    logger.info(f"Pack hash: {phash[:16]}...")
    logger.info(f"Files:     {list(pack.get('files', {}).keys())}")

    # ── 2. Schema validation ──────────────────────────────────────────────
    lint = validate_opp_schema(pack)
    if not lint.passed:
        for issue in lint.issues:
            logger.error(f"  schema: {issue}")
        if not args.force:
            return 1
        logger.warning("--force: continuing despite schema errors")
    else:
        logger.info("Schema:    PASSED")

    # ── 3. Check Docker services ──────────────────────────────────────────
    mock_url = os.getenv("MOCK_TOOLS_URL", "http://localhost:3001")
    openclaw_url = os.getenv("OPENCLAW_URL", "http://localhost:18789")

    openclaw_token = os.getenv("OPENCLAW_GATEWAY_TOKEN", "sandbox-token-12345")

    logger.info("Services:")
    if not check_services(mock_url, openclaw_url, token=openclaw_token):
        logger.error(
            "\nDocker services not running.  Start them:\n"
            "  cd clawbench && docker compose up -d\n"
            "Then wait ~30s and retry."
        )
        return 1

    # ── 4. ClawBench harness ──────────────────────────────────────────────
    clawbench_path = Path(args.clawbench_path)
    model = args.model or os.getenv("CLAWBENCH_DEFAULT_MODEL", "zhipu/glm-5.1")
    api_key = args.api_key or os.getenv("CLAWBENCH_LLM_API_KEY", "")
    base_url = args.base_url or os.getenv(
        "CLAWBENCH_LLM_BASE_URL", "https://open.bigmodel.cn/api/paas/v4",
    )
    if not api_key:
        logger.error("No LLM API key.  Set CLAWBENCH_LLM_API_KEY or --api-key")
        return 1

    workspace = Path(args.workspace) if args.workspace else None
    harness = ClawBenchHarness(
        clawbench_path=clawbench_path,
        timeout=args.timeout,
        workspace_path=workspace,
        clawbench_default_model=model,
        clawbench_api_key=api_key,
        clawbench_base_url=base_url,
    )
    logger.info(f"Model:     {model}")
    logger.info(f"Timeout:   {args.timeout}s/scenario")

    # ── 4b. TrajectoryJudge (Phase 2, same as validator) ─────────────────
    judge = TrajectoryJudge(
        model=model,
        api_key=api_key,
        base_url=base_url,
    )
    scenarios_path = clawbench_path / "scenarios"
    logger.info("Judge:     TrajectoryJudge enabled")

    # ── 5. Epoch context ──────────────────────────────────────────────────
    seed = args.seed if args.seed is not None else int(time.time()) % 100000
    ctx = generate_epoch_context(seed)
    preamble = render_context_preamble(ctx)
    user_ctx = ctx.to_user_context()
    logger.info(f"Seed:      {seed}")
    logger.info(f"Persona:   {ctx.user_name}, {ctx.user_role} @ {ctx.company}")

    # ── 6. Evaluate ───────────────────────────────────────────────────────
    scenarios = args.scenarios or SCENARIOS
    logger.info(f"Scenarios: {scenarios}  (runs={args.num_runs})\n")

    results: dict[str, EvaluationResult] = {}
    total_cost = 0.0

    for i, sname in enumerate(scenarios, 1):
        logger.info(f"[{i}/{len(scenarios)}] {sname}")

        try:
            if args.num_runs > 1:
                r = await harness.evaluate_pack_consensus(
                    pack=pack, scenario_name=sname,
                    num_runs=args.num_runs, base_seed=seed,
                    context_preamble=preamble, user_context=user_ctx,
                )
            else:
                r = await harness.evaluate_pack(
                    pack=pack, scenario_name=sname, seed=seed,
                    context_preamble=preamble, user_context=user_ctx,
                )
        except Exception as e:
            logger.error(f"  EXCEPTION: {e}", exc_info=True)
            r = EvaluationResult(
                scenario_name=sname, score=0.0, success=False,
                tool_calls=0, response="", rubric={}, error=str(e),
            )

        results[sname] = r

        # Phase 2: LLM trajectory judge (same as validator)
        scenario_yaml = scenarios_path / f"{sname}.yaml"
        judge_qualified = False
        judge_score = 0.0
        if scenario_yaml.exists() and not r.error:
            with open(scenario_yaml) as f:
                scenario_config = yaml.safe_load(f)
            trajectory = r.trajectory or []
            judge_result = await judge.evaluate(
                scenario_config=scenario_config,
                trajectory=trajectory,
                agent_response=r.response,
            )
            judge_qualified = judge_result.qualification_gate
            judge_score = judge_result.overall_score
            logger.info(
                f"  judge={judge_score:.3f}  "
                f"safety={'PASS' if judge_result.safety_passed else 'FAIL'}  "
                f"correctness={'PASS' if judge_result.correctness_passed else 'FAIL'}"
            )
            for cr in judge_result.criteria_results:
                status = "PASS" if cr.verdict == "PASS" else "FAIL"
                logger.info(f"    [{status}] {cr.id}: {cr.justification[:120]}")
            if judge_result.error:
                logger.error(f"  judge error: {judge_result.error}")
            # Override gate with judge result
            r.success = judge_qualified
            r.score = judge_score

        gate = "PASS" if r.success else "FAIL"
        logger.info(f"  score={r.score:.3f}  gate={gate}  calls={r.tool_calls}")

        if r.cost_usd is not None:
            logger.info(f"  cost=${r.cost_usd:.4f}")
            total_cost += r.cost_usd

        if r.token_usage:
            t = r.token_usage
            logger.info(
                f"  tokens: in={t.get('input_tokens',0)} "
                f"out={t.get('output_tokens',0)} "
                f"cache={t.get('cache_read_tokens',0)}"
            )
        if r.model_usage:
            for m in r.model_usage:
                logger.info(
                    f"  model: {m.get('model','?')} "
                    f"${m.get('cost_usd',0):.4f} ({m.get('count',0)} calls)"
                )

        if r.error:
            logger.error(f"  error: {r.error}")

        logger.info("")

    # ── 7. Summary table ──────────────────────────────────────────────────
    W = 78
    print("\n" + "=" * W)
    print(f"PACK EVALUATION SUMMARY")
    print(f"  source: {args.pack or args.agents_md}")
    print(f"  hash:   {phash[:16]}...")
    print(f"  model:  {model}")
    print("=" * W)
    hdr = f"{'scenario':<25} {'score':>7} {'gate':>6} {'cost':>10} {'calls':>6}"
    print(hdr)
    print("-" * W)

    n_pass = 0
    s_total = 0.0
    for sname, r in results.items():
        g = "PASS" if r.success else "FAIL"
        c = f"${r.cost_usd:.4f}" if r.cost_usd is not None else "-"
        print(f"{sname:<25} {r.score:>7.3f} {g:>6} {c:>10} {r.tool_calls:>6}")
        s_total += r.score
        if r.success:
            n_pass += 1

    print("-" * W)
    avg = s_total / len(results) if results else 0.0
    c = f"${total_cost:.4f}" if total_cost > 0 else "-"
    print(f"{'TOTAL':<25} {avg:>7.3f} {n_pass}/{len(results):>3} {c:>10}")
    print("=" * W)

    qualified = n_pass == len(results)
    if qualified:
        print(f"\n  QUALIFIED  {n_pass}/{len(results)} scenarios passed")
    else:
        failed = [s for s, r in results.items() if not r.success]
        print(f"\n  NOT QUALIFIED  {n_pass}/{len(results)} passed")
        print(f"  failed: {', '.join(failed)}")

    if qualified and total_cost > 0:
        print(f"\n  avg cost/scenario: ${total_cost / len(results):.4f}")
        print(f"  total cost:        ${total_cost:.4f}")

    # ── 8. JSON output ────────────────────────────────────────────────────
    if args.output:
        out = {
            "pack_hash": phash,
            "source": args.pack or args.agents_md,
            "model": model,
            "seed": seed,
            "persona": {"name": ctx.user_name, "role": ctx.user_role,
                        "company": ctx.company},
            "num_runs": args.num_runs,
            "scenarios": {
                name: {
                    "score": r.score, "success": r.success,
                    "tool_calls": r.tool_calls,
                    "cost_usd": r.cost_usd,
                    "token_usage": r.token_usage,
                    "model_usage": r.model_usage,
                    "rubric": r.rubric,
                    "error": r.error,
                    "response": r.response[:2000],
                }
                for name, r in results.items()
            },
            "summary": {
                "avg_score": round(avg, 4),
                "passed": n_pass,
                "total": len(results),
                "qualified": qualified,
                "total_cost": round(total_cost, 4) if total_cost > 0 else None,
            },
        }
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print(f"\n  results saved to {args.output}")

    return 0 if qualified else 1


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description="Evaluate a local pack against ClawBench (no chain needed).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  python scripts/eval_pack.py --pack pack.json\n"
            "  python scripts/eval_pack.py --agents-md policy.md -s inbox_triage -v\n"
            "  python scripts/eval_pack.py --pack pack.json -n 3 -o results.json\n"
        ),
    )
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--pack", help="OPP v1 pack JSON file")
    g.add_argument("--agents-md", help="AGENTS.md file (auto-wrapped into pack)")

    p.add_argument("-s", "--scenarios", nargs="+", default=None,
                   help=f"Scenarios (default: all).  {', '.join(SCENARIOS)}")
    p.add_argument("-n", "--num-runs", type=int, default=1,
                   help="Runs per scenario for consensus (default: 1)")
    p.add_argument("--seed", type=int, default=None,
                   help="Epoch seed (default: from current time)")
    p.add_argument("--timeout", type=int, default=600,
                   help="Timeout per scenario (default: 600s)")
    p.add_argument("--model", help="LLM model override")
    p.add_argument("--api-key", help="LLM API key override")
    p.add_argument("--base-url", help="LLM base URL override")
    p.add_argument("--clawbench-path", default=str(PROJECT_ROOT / "clawbench"),
                   help="Path to clawbench dir")
    p.add_argument("--workspace", default=None, help="Workspace path override")
    p.add_argument("-o", "--output", help="Save results to JSON file")
    p.add_argument("-v", "--verbose", action="store_true",
                   help="Show per-check rubric details")
    p.add_argument("--force", action="store_true",
                   help="Continue despite schema errors")
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    args = p.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)-8s | %(message)s",
    )

    env_path = PROJECT_ROOT / ".env.validator"
    if env_path.exists():
        try:
            from dotenv import load_dotenv
            load_dotenv(env_path)
        except ImportError:
            pass

    sys.exit(asyncio.run(run(args)))


if __name__ == "__main__":
    main()
