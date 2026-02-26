"""ClawBench harness integration for evaluating policy packs."""

import asyncio
import json
import logging
import math
import os
import subprocess
import tempfile
import yaml
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Result from running a scenario evaluation.

    Attributes:
        scenario_name: Name of the scenario
        score: Normalized score [0, 1]
        success: Whether the scenario passed
        tool_calls: Number of tool calls made
        response: Agent's final response
        rubric: Detailed scoring rubric results
        error: Error message if evaluation failed
    """
    scenario_name: str
    score: float
    success: bool
    tool_calls: int
    response: str
    rubric: Dict[str, Any]
    error: Optional[str] = None


class ClawBenchHarness:
    """Integrates with ClawBench for policy pack evaluation."""

    def __init__(
        self,
        clawbench_path: Path,
        timeout: int = 120,
        workspace_path: Optional[Path] = None,
    ):
        """Initialize harness.

        Args:
            clawbench_path: Path to clawbench directory
            timeout: Timeout in seconds for each scenario
            workspace_path: Shared workspace directory that OpenClaw reads from.
                If None, uses WORKSPACE_PATH env var or clawbench_path/workspace.
        """
        self.clawbench_path = clawbench_path
        self.timeout = timeout
        self.scripts_path = clawbench_path / "scripts"
        self.scenarios_path = clawbench_path / "scenarios"
        self.fixtures_path = clawbench_path / "fixtures"

        # Workspace shared with OpenClaw — pack files are written here so that
        # the OpenClaw gateway can read the updated AGENTS.md on each evaluation.
        if workspace_path is not None:
            self.workspace_path = workspace_path
        else:
            self.workspace_path = Path(
                os.environ.get("WORKSPACE_PATH", str(clawbench_path / "workspace"))
            )

        # Validate paths
        if not self.scripts_path.exists():
            raise ValueError(f"ClawBench scripts not found: {self.scripts_path}")
        if not self.scenarios_path.exists():
            raise ValueError(f"ClawBench scenarios not found: {self.scenarios_path}")

    async def evaluate_pack(
        self,
        pack: dict,
        scenario_name: str,
        seed: int = 0,
        context_preamble: str = "",
        user_context: Optional[Dict[str, str]] = None,
    ) -> EvaluationResult:
        """Evaluate a policy pack on a scenario.

        Args:
            pack: Policy pack dictionary (OPP v1 format)
            scenario_name: Name of scenario to run (e.g., "client_escalation")
            seed: Random seed for reproducibility
            context_preamble: Epoch context markdown prepended to AGENTS.md
            user_context: Dict of template overrides for {{PLACEHOLDER}}
                substitution in workspace files (USER.md, etc.).  Passed to
                run_episode.py via --user-context.

        Returns:
            EvaluationResult with score and details
        """
        logger.info(
            f"Evaluating pack on scenario={scenario_name}, seed={seed}, "
            f"pack_hash={self._compute_hash(pack)[:8]}"
        )

        try:
            # Use an isolated workspace per evaluation to prevent concurrent
            # evaluations from overwriting each other's pack files.
            with tempfile.TemporaryDirectory(
                prefix="trajectoryrl_eval_",
                dir=self.workspace_path.parent,
            ) as tmpdir:
                workspace = Path(tmpdir)
                # Write pack files so OpenClaw can read them.
                self._apply_pack_to_workspace(
                    pack, workspace, context_preamble
                )

                # Run scenario — user_context is passed to run_episode.py which
                # handles template substitution in workspace files and pushes
                # context to the mock server.
                result = await self._run_scenario(
                    scenario_name=scenario_name,
                    workspace=workspace,
                    seed=seed,
                    user_context=user_context,
                )

                return result

        except Exception as e:
            logger.error(f"Evaluation failed: {e}", exc_info=True)
            return EvaluationResult(
                scenario_name=scenario_name,
                score=0.0,
                success=False,
                tool_calls=0,
                response="",
                rubric={},
                error=str(e)
            )

    async def evaluate_pack_consensus(
        self,
        pack: dict,
        scenario_name: str,
        num_runs: int = 3,
        base_seed: int = 0,
        context_preamble: str = "",
        user_context: Optional[Dict[str, str]] = None,
    ) -> EvaluationResult:
        """Evaluate a pack multiple times and majority-vote per rubric check.

        LLM outputs vary between runs even with the same input. Running the
        scenario ``num_runs`` times and taking the majority vote on each binary
        rubric check produces a score that is far more stable across independent
        validators, enabling Yuma consensus.

        The ``base_seed`` (typically the epoch seed) is mixed into per-run
        seeds so that each epoch evaluates under slightly different conditions.
        This is part of the epoch-seeded variation mechanism that prevents
        stale solutions from holding the throne indefinitely.

        Args:
            pack: Policy pack dictionary (OPP v1 format)
            scenario_name: Name of scenario to run
            num_runs: Number of independent runs (must be odd for clean majority)
            base_seed: Epoch seed mixed into per-run seeds for variation
            context_preamble: Epoch context markdown prepended to AGENTS.md
            user_context: Dict of template overrides for {{PLACEHOLDER}}
                substitution (passed through to run_episode.py)

        Returns:
            EvaluationResult with majority-voted rubric and derived score
        """
        quorum = math.ceil(num_runs / 2)

        logger.info(
            f"Consensus evaluation: {scenario_name} x{num_runs} "
            f"(quorum={quorum}, base_seed={base_seed})"
        )

        # Run scenario num_runs times.
        # Mix base_seed (epoch) into per-run seed so each epoch evaluates
        # under slightly different random conditions.
        runs: List[EvaluationResult] = []
        for i in range(num_runs):
            run_seed = base_seed * 1000 + i
            result = await self.evaluate_pack(
                pack=pack,
                scenario_name=scenario_name,
                seed=run_seed,
                context_preamble=context_preamble,
                user_context=user_context,
            )
            runs.append(result)
            logger.debug(
                f"  Run {i}: score={result.score:.3f}, "
                f"success={result.success}, checks={len(result.rubric)}"
            )

        # If all runs failed (errors), return failure
        valid_runs = [r for r in runs if r.error is None]
        if not valid_runs:
            return EvaluationResult(
                scenario_name=scenario_name,
                score=0.0,
                success=False,
                tool_calls=0,
                response="",
                rubric={},
                error="All runs failed",
            )

        # Majority-vote each rubric check
        voted_rubric = self._majority_vote_rubric(
            [r.rubric for r in valid_runs], quorum
        )

        # Derive score from voted rubric
        voted_score = self._score_from_rubric(voted_rubric)

        # Median tool calls for stability
        tool_calls = sorted(r.tool_calls for r in valid_runs)[len(valid_runs) // 2]

        # Use response from the run closest to voted score
        closest = min(valid_runs, key=lambda r: abs(r.score - voted_score))

        logger.info(
            f"Consensus result: {scenario_name} → score={voted_score:.3f} "
            f"(individual scores: {[round(r.score, 3) for r in runs]})"
        )

        return EvaluationResult(
            scenario_name=scenario_name,
            score=voted_score,
            success=voted_score > 0.0,
            tool_calls=tool_calls,
            response=closest.response,
            rubric=voted_rubric,
        )

    @staticmethod
    def _majority_vote_rubric(
        rubrics: List[Dict[str, Any]], quorum: int
    ) -> Dict[str, Any]:
        """Majority-vote across multiple rubric dicts.

        Each rubric is the full score dict returned by ``score_episode()``,
        which contains a ``"checks"`` list of per-check dicts with ``"id"``,
        ``"passed"``, ``"points"``, ``"max_points"``, etc.

        A check passes in the voted rubric if it passed in >= ``quorum`` runs.

        Args:
            rubrics: List of rubric dicts from multiple runs
            quorum: Minimum number of passes for majority

        Returns:
            Voted rubric dict keyed by check id
        """
        if not rubrics:
            return {}

        # Collect all check results across runs, keyed by check id
        all_checks: Dict[str, List[dict]] = {}
        for rubric in rubrics:
            checks_list = rubric.get("checks")
            if isinstance(checks_list, list):
                items = [(c["id"], c) for c in checks_list if "id" in c]
            else:
                # Fallback: rubric is already {check_id: check_data}
                items = [(k, v) for k, v in rubric.items() if isinstance(v, dict)]
            for check_id, check_data in items:
                if check_id not in all_checks:
                    all_checks[check_id] = []
                all_checks[check_id].append(check_data)

        voted = {}
        for check_id, check_results in all_checks.items():
            pass_count = sum(
                1 for c in check_results
                if (isinstance(c, dict) and c.get("passed", False))
                or (isinstance(c, bool) and c)
            )
            # Use the first occurrence as the template
            template = check_results[0]
            if isinstance(template, dict):
                voted[check_id] = {
                    **template,
                    "passed": pass_count >= quorum,
                    "_votes": f"{pass_count}/{len(check_results)}",
                }
            else:
                voted[check_id] = pass_count >= quorum

        return voted

    @staticmethod
    def _score_from_rubric(rubric: Dict[str, Any]) -> float:
        """Compute a normalized score from a voted rubric.

        Scoring: sum of points for passed checks / total points.
        Falls back to passed_count / total_count if no points field.

        Args:
            rubric: Voted rubric dict

        Returns:
            Score in [0, 1]
        """
        if not rubric:
            return 0.0

        total_points = 0
        earned_points = 0

        for check_name, check_data in rubric.items():
            if isinstance(check_data, dict):
                # max_points is the full weight regardless of pass/fail;
                # "points" is earned (0 when failed), so not suitable as weight.
                points = check_data.get("max_points", check_data.get("points", 1))
                passed = check_data.get("passed", False)
            else:
                points = 1
                passed = bool(check_data)

            total_points += points
            if passed:
                earned_points += points

        if total_points == 0:
            return 0.0

        return earned_points / total_points

    def _apply_pack_to_workspace(
        self, pack: dict, workspace: Path, context_preamble: str = ""
    ) -> None:
        """Write pack files to workspace directory.

        If ``context_preamble`` is provided, it is prepended to AGENTS.md
        so the agent operates under the epoch-specific persona and date.

        Args:
            pack: Policy pack dictionary
            workspace: Workspace directory path
            context_preamble: Epoch context markdown prepended to AGENTS.md
        """
        # Create workspace directory
        workspace.mkdir(parents=True, exist_ok=True)
        workspace_abs = workspace.resolve()

        # Write files from pack.
        # SECURITY: validate every miner-supplied filename before writing.
        # Without this check, a filename like "../../etc/cron.d/exploit"
        # resolves outside the temp workspace → arbitrary file write on host.
        files = pack.get("files", {})
        for filename, content in files.items():
            if filename == "AGENTS.md" and context_preamble:
                content = context_preamble + content
            file_path = (workspace_abs / filename).resolve()
            try:
                file_path.relative_to(workspace_abs)
            except ValueError:
                logger.warning(
                    f"Skipping file with path traversal attempt: {filename!r}"
                )
                continue
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(content)
            logger.debug(f"Wrote {filename} ({len(content)} chars)")

        logger.info(f"Applied pack to workspace: {workspace}")

    async def _run_scenario(
        self,
        scenario_name: str,
        workspace: Path,
        seed: int,
        user_context: Optional[Dict[str, str]] = None,
    ) -> EvaluationResult:
        """Run a ClawBench scenario.

        Args:
            scenario_name: Scenario name
            workspace: Workspace directory with pack files
            seed: Random seed
            user_context: Template overrides passed to run_episode.py

        Returns:
            EvaluationResult
        """
        # Load scenario config
        scenario_path = self.scenarios_path / f"{scenario_name}.yaml"
        if not scenario_path.exists():
            raise ValueError(f"Scenario not found: {scenario_path}")

        with open(scenario_path) as f:
            scenario = yaml.safe_load(f)

        # Run episode via run_episode.py
        run_script = self.scripts_path / "run_episode.py"
        if not run_script.exists():
            raise ValueError(f"run_episode.py not found: {run_script}")

        cmd = [
            "python",
            str(run_script),
            "--scenario", scenario_name,
            "--workspace", str(workspace),
            "--seed", str(seed),  # Epoch-mixed seed for variation
            "--json",  # Output JSON for parsing
            "--wait",  # Wait for services to be ready
        ]

        # Pass user context as JSON for template substitution in workspace
        # files and on the mock server
        if user_context:
            cmd.extend(["--user-context", json.dumps(user_context)])

        logger.debug(f"Running command: {' '.join(cmd)}")

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(self.clawbench_path)
            )

            stdout, stderr = await asyncio.wait_for(
                proc.communicate(),
                timeout=self.timeout
            )

            # Parse JSON output
            output = stdout.decode()
            result_data = self._parse_episode_output(output)

            # Extract scoring results
            score = result_data.get("score", 0.0)
            success = result_data.get("success", False)
            tool_calls = result_data.get("tool_calls", 0)
            response = result_data.get("response", "")
            rubric = result_data.get("rubric", {})

            return EvaluationResult(
                scenario_name=scenario_name,
                score=score,
                success=success,
                tool_calls=tool_calls,
                response=response,
                rubric=rubric
            )

        except asyncio.TimeoutError:
            # Kill the child process so it doesn't keep running as an orphan.
            # Without this, timed-out evaluations leak CPU and RAM; over a
            # full epoch with many miners these accumulate significantly.
            try:
                proc.kill()
                await proc.communicate()
            except Exception:
                pass
            logger.error(f"Scenario timeout: {scenario_name}")
            return EvaluationResult(
                scenario_name=scenario_name,
                score=0.0,
                success=False,
                tool_calls=0,
                response="",
                rubric={},
                error=f"Timeout after {self.timeout}s"
            )

    def _parse_episode_output(self, output: str) -> Dict[str, Any]:
        """Parse run_episode.py JSON output.

        Args:
            output: Raw stdout from run_episode.py

        Returns:
            Parsed result dictionary
        """
        # run_episode.py outputs JSON when --json flag is used
        # Format: {"score": 0.9, "success": true, "tool_calls": 12, ...}
        # Scan lines in reverse; require "score" key to distinguish the
        # result object from stray JSON log lines.
        lines = output.strip().split("\n")
        for line in reversed(lines):
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                data = json.loads(line)
                if isinstance(data, dict) and "score" in data:
                    return data
            except json.JSONDecodeError:
                continue

        logger.error("Failed to parse episode output: no result JSON found")
        logger.debug(f"Output was: {output}")
        return {
            "score": 0.0,
            "success": False,
            "tool_calls": 0,
            "response": "",
            "rubric": {},
            "error": "Parse error: no result JSON found",
        }

    def _compute_hash(self, pack: dict) -> str:
        """Compute SHA256 hash of pack.

        Args:
            pack: Policy pack dict

        Returns:
            Hex digest
        """
        import hashlib
        content = json.dumps(pack, sort_keys=True).encode()
        return hashlib.sha256(content).hexdigest()
