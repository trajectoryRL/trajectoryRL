"""trajrl-bench sandbox harness for evaluating SKILL.md packs.

Single-container architecture (2026-05):
  - Sandbox+agent container (``ghcr.io/trajectoryrl/sandbox-agent``)
    holds the puzzle filesystem and runs Hermes as the ``agent`` user.
  - The validator drives episodes via ``docker exec -u agent`` per
    episode — no testee container, no SSH boundary.
  - One verifier container per episode runs ``tests/test.sh`` against
    the agent's output and writes a binary 0/1 reward.

The validator does NOT import trajrl-bench as a Python dependency.
It pulls the sandbox image and runs CLI commands via ``docker run``:
  - ``cli scenarios``         — list scenarios shipped by this image
  - ``cli scenario-info``     — full payload (image_repo, instruction.md,
                                tests/, agent_output_path, timeouts)

Updating scenarios = rebuild sandbox-agent + scenario images, publish
to GHCR. Validators pull on next eval. No validator code change.

Usage:
    harness = TrajectorySandboxHarness(config)
    await harness.pull_latest()
    result = await harness.evaluate_miner(skill_md, epoch_seed, ...)
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import io
import json
import logging
import secrets
import tarfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import docker
from docker.models.containers import Container
from docker.types import LogConfig

from ..utils.config import ValidatorConfig

logger = logging.getLogger(__name__)


# Hermes runtime user inside the sandbox-agent image. Must match the
# ``useradd ... -u 10000 agent`` line in
# ``trajrl-bench/docker/Dockerfile.sandbox-agent``.
_AGENT_UID = 10000


# Scenarios run per session, in order. Every validator runs the full set
# so scores are comparable across validators / windows / SPEC_NUMBER
# bumps. Adding or removing a scenario must come with a SPEC_NUMBER bump
# so cached scores invalidate. Sorted alphabetically for stability.
SANDBOX_SCENARIOS: tuple[str, ...] = (
    "break-filter-js-from-html",
    "cancel-async-tasks",
    "log-summary-date-ranges",
)


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class _EpisodeResult:
    episode_index: int
    # Scenario this episode evaluated. With multi-scenario sessions
    # episode_index alone doesn't identify the cell; pair with this.
    scenario: str = ""
    # Continuous correctness in [0.0, 1.0]: passed_tests / total_tests from
    # the verifier's ctrf.json. Falls back to ``float(reward)`` (binary
    # 0 or 1) when ctrf isn't parseable.
    quality: float = 0.0
    # Cost axis: actual_cost_usd from the agent's Hermes session export.
    # ``None`` when the session JSONL is missing / billing not pinned.
    # Reported alongside quality on artifacts and submit payloads but
    # does NOT factor into the score (separate axis — see Ning's
    # 2026-05-04 design call).
    cost_usd: float | None = None
    transcript: str = ""
    turns_log: str = ""
    turns_export_err: str = ""
    timed_out: bool = False
    error: str | None = None
    duration_s: float = 0.0
    judge_result: dict = field(default_factory=dict)
    ep_data: dict = field(default_factory=dict)


@dataclass
class _SessionResult:
    episodes: list[_EpisodeResult] = field(default_factory=list)
    mean_quality: float = 0.0  # mean of per-scenario correctness, in [0,1]
    final_score: float = 0.0   # sum of per-scenario correctness, in [0, N]
    pack_hash: str = ""
    validator_salt: str = ""
    skill_md: str = ""

    def compute_scores(self) -> None:
        """Aggregate per-scenario correctness into the session score.

        Each scenario contributes ``passed_i / total_i`` ∈ [0, 1] (the
        episode's ``quality``). Final score is the equal-weighted sum
        across scenarios — range [0, N] for N scenarios — so a perfect
        all-pass session lands at ``len(SANDBOX_SCENARIOS)``. Mean
        quality is exposed as the [0, 1] convenience aggregate.

        Rationale (Ning, 2026-05-04): equal weight per scenario, no
        learning bonus, no split-half delta. With one episode per
        scenario the within-scenario delta concept doesn't apply, and
        cross-scenario diversity is the noise reduction.
        """
        scores = [ep.quality for ep in self.episodes]
        if not scores:
            self.mean_quality = 0.0
            self.final_score = 0.0
            return
        self.mean_quality = sum(scores) / len(scores)
        self.final_score = sum(scores)


class SandboxEvaluationResult:
    """Result from a sandbox evaluation. Provides the standard result
    interface the validator pipeline consumes."""

    def __init__(self, session_result: _SessionResult,
                 scenario_name: str = ""):
        # ``scenario_name`` is retained on the result for backward
        # compat with consumers that expect a single-scenario tag, but
        # multi-scenario sessions populate ``scenarios`` / per-cell
        # results below.
        self.scenario_name = scenario_name
        self.score = session_result.final_score
        self.success = session_result.final_score > 0.0
        self.tool_calls = 0
        self.response = ""
        self.rubric = {}
        self.error: Optional[str] = None

        self.cost_usd: Optional[float] = None
        self.token_usage: Optional[Dict[str, int]] = None
        self.model_usage: Optional[List[Dict[str, Any]]] = None
        self.trajectory: Optional[List[Dict[str, Any]]] = None
        self.input_message: Optional[str] = None
        self.raw_llm_response: Optional[Dict[str, Any]] = None
        self.all_requests: Optional[List[Dict[str, Any]]] = None
        self.session_key: Optional[str] = None
        self.session_file: Optional[str] = None

        self.session_result = session_result
        self.mean_quality = session_result.mean_quality
        # Per-scenario correctness map: {scenario_name: passed/total}.
        # Empty dict if the session ran no episodes.
        self.scenario_qualities: Dict[str, float] = {
            ep.scenario: ep.quality for ep in session_result.episodes
        }
        # Ordered list of scenarios run (matches ``SANDBOX_SCENARIOS``
        # by construction; provides a stable iteration order for
        # downstream consumers).
        self.scenarios: List[str] = [ep.scenario for ep in session_result.episodes]
        # Cost axis (separate from score). Per-scenario map + aggregates.
        self.scenario_costs_usd: Dict[str, Optional[float]] = {
            ep.scenario: ep.cost_usd for ep in session_result.episodes
        }
        billed = [c for c in self.scenario_costs_usd.values() if c is not None]
        self.total_cost_usd: Optional[float] = sum(billed) if billed else None
        self.mean_cost_usd: Optional[float] = (
            sum(billed) / len(billed) if billed else None
        )

    def write_artifacts(self, out_dir: "Path | str") -> None:
        """Write per-episode transcripts + verifier outputs + session
        metadata to disk. Layout matches what ``trajrl logs --show``
        knows how to render."""
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        sr = self.session_result

        (out / "SKILL.md").write_text(sr.skill_md or "")
        (out / "metadata.json").write_text(json.dumps({
            "scenarios": self.scenarios,
            "pack_hash": sr.pack_hash,
            "validator_salt": sr.validator_salt,
            "final_score": sr.final_score,
            "mean_quality": sr.mean_quality,
            "scenario_qualities": self.scenario_qualities,
            "scenario_costs_usd": self.scenario_costs_usd,
            "total_cost_usd": self.total_cost_usd,
            "mean_cost_usd": self.mean_cost_usd,
        }, indent=2, default=str))

        for ep in sr.episodes:
            # Per-cell directory keyed by scenario name (one cell per
            # scenario in the new 1-ep-per-scenario design). Falls back
            # to ``episode_<n>`` for older results without scenario tags.
            cell_name = (
                f"scenario_{ep.scenario}" if ep.scenario
                else f"episode_{ep.episode_index}"
            )
            ep_dir = out / "episodes" / cell_name
            ep_dir.mkdir(parents=True, exist_ok=True)
            (ep_dir / "testee_transcript.txt").write_text(ep.transcript or "")
            if ep.turns_log:
                (ep_dir / "turns.jsonl").write_text(ep.turns_log)
            if ep.turns_export_err:
                (ep_dir / "turns_export.err").write_text(ep.turns_export_err)
            (ep_dir / "evaluation.json").write_text(
                json.dumps(ep.judge_result, indent=2, default=str))
            # Standalone ctrf.json so consumers don't need to dig into
            # evaluation.json["ctrf"]. The full payload is also still
            # nested inside evaluation.json for backward compat.
            ctrf = ep.judge_result.get("ctrf") if ep.judge_result else None
            if ctrf is not None:
                (ep_dir / "ctrf.json").write_text(
                    json.dumps(ctrf, indent=2, default=str))
            verifier_stdout = (
                ep.judge_result.get("verifier_stdout") if ep.judge_result else ""
            )
            if verifier_stdout:
                (ep_dir / "verifier_stdout.txt").write_text(verifier_stdout)
            (ep_dir / "episode.json").write_text(
                json.dumps(ep.ep_data, indent=2, default=str))
            if ep.error:
                (ep_dir / "error.txt").write_text(ep.error)


# ---------------------------------------------------------------------------
# Docker helpers
# ---------------------------------------------------------------------------

def _read_container_text(container, path: str) -> str:
    """Read a single text file from a container via get_archive.
    Returns '' on any failure."""
    try:
        archive, _ = container.get_archive(path)
    except docker.errors.NotFound:
        return ""
    except Exception:
        return ""
    try:
        buf = io.BytesIO()
        for chunk in archive:
            buf.write(chunk)
        buf.seek(0)
        with tarfile.open(fileobj=buf, mode="r") as tar:
            members = tar.getmembers()
            if not members:
                return ""
            f = tar.extractfile(members[0])
            if f is None:
                return ""
            return f.read().decode("utf-8", errors="replace")
    except Exception:
        return ""


def _docker_run_json(client: docker.DockerClient, image: str,
                     command: list[str], environment: dict | None = None,
                     timeout: int = 120, **kwargs) -> dict:
    """Run a command in a container and parse JSON stdout."""
    try:
        output = client.containers.run(
            image, command=command, entrypoint="",
            environment=environment or {},
            remove=True, stdout=True, stderr=True,
            mem_limit="2g", cpu_quota=100000,
            **kwargs,
        )
        return json.loads(output.decode())
    except docker.errors.ContainerError as e:
        stderr = e.stderr.decode() if e.stderr else ""
        raise RuntimeError(f"Container command failed: {stderr}") from e
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Failed to parse container output as JSON: {e}") from e


def _put_file(
    container: Container, path: str, content: str | bytes,
    mode: int = 0o644, uid: int = 0, gid: int = 0,
) -> None:
    """Write a file into a container via tar archive."""
    import posixpath
    dir_name = posixpath.dirname(path)
    file_name = posixpath.basename(path)
    data = content.encode() if isinstance(content, str) else content
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        info = tarfile.TarInfo(name=file_name)
        info.size = len(data)
        info.mode = mode
        info.uid = uid
        info.gid = gid
        tar.addfile(info, io.BytesIO(data))
    buf.seek(0)
    container.put_archive(dir_name, buf)


def _shell_quote(s: str) -> str:
    """Single-quote a string for safe inclusion in a bash -c invocation."""
    return "'" + s.replace("'", "'\\''") + "'"


def _parse_ctrf_correctness(ctrf: dict | None) -> tuple[int, int]:
    """Return (passed, total) from a pytest-json-ctrf payload.

    Falls back to (0, 0) when the payload is missing or malformed —
    callers must treat (0, 0) as "no signal" (not "100% pass").
    """
    if not isinstance(ctrf, dict):
        return 0, 0
    summary = (ctrf.get("results") or {}).get("summary") or {}
    try:
        total = int(summary.get("tests", 0))
        passed = int(summary.get("passed", 0))
    except (TypeError, ValueError):
        return 0, 0
    if total <= 0:
        return 0, 0
    return passed, total


def _parse_session_cost(turns_log: str) -> float | None:
    """Pull ``actual_cost_usd`` out of the latest Hermes session in
    a turns.jsonl blob. Returns None when the blob is empty / unparseable
    / Hermes didn't bill (e.g. unbilled provider, network error).

    The validator wipes Hermes' SQLite store between episodes so the
    JSONL should carry exactly one session per file — but we take the
    last non-empty line defensively in case that ever drifts.
    """
    if not turns_log:
        return None
    last: dict | None = None
    for line in turns_log.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            last = json.loads(line)
        except json.JSONDecodeError:
            continue
    if not isinstance(last, dict):
        return None
    cost = last.get("actual_cost_usd")
    if cost is None:
        cost = last.get("estimated_cost_usd")
    try:
        cost_f = float(cost)
    except (TypeError, ValueError):
        return None
    if cost_f < 0:
        return None
    return cost_f


def _strip_provider_prefix(model: str) -> str:
    """Remove provider routing prefixes from model identifiers."""
    for prefix in ("openrouter/", "chutes/"):
        if model.startswith(prefix):
            return model[len(prefix):]
    return model


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------

class TrajectorySandboxHarness:
    """Single-container sandbox harness.

    Per session:
      1. Pull sandbox-agent image + the configured scenario image.
      2. Run sandbox-agent container; wait for it to come up.
      3. Inject /app from the scenario image; write SKILL.md.
      4. For each episode:
         - Write INSTRUCTION.md
         - ``docker exec -u agent`` runs ``hermes chat`` against
           /workspace + /app directly. Wraps chat + ``hermes sessions
           export`` in one bash invocation; wipes the SQLite session DB
           between episodes so each episode's turns.jsonl carries only
           that episode's session.
         - Extract agent_output_path from the sandbox.
         - Run a one-shot verifier container against the output,
           parse reward.txt, set quality = float(reward).
      5. Compute split-half delta over the resulting qualities.
    """

    def __init__(self, config: ValidatorConfig):
        self.config = config
        self._docker_client: docker.DockerClient | None = None

        self._sandbox_image = config.sandbox_image

        self._testee_model = _strip_provider_prefix(config.llm_model)
        self._testee_api_key = config.llm_api_key
        self._testee_api_url = config.llm_base_url

        # Backwards-compat aliases (some external readers still expect
        # _llm_*). All three point at the agent's LLM.
        self._llm_model = self._testee_model
        self._llm_api_key = self._testee_api_key
        self._llm_api_url = self._testee_api_url

        self.sandbox_version: str = "unknown"
        self.sandbox_scenarios: list[str] = []
        self._scenario_info: dict | None = None

        self.bench_image_hash: str = "unknown"
        self.scenario_image_hash: str = "unknown"
        self.scenario_image_hashes: Dict[str, str] = {}

    @property
    def client(self) -> docker.DockerClient:
        if self._docker_client is None:
            self._docker_client = docker.from_env()
        return self._docker_client

    async def pull_latest(self) -> None:
        """Pull latest sandbox-agent + scenario images. Get new scenarios."""
        loop = asyncio.get_event_loop()
        try:
            await loop.run_in_executor(None, self._pull_sync)
        except Exception as e:
            logger.warning("Failed to pull latest images: %s (using cached)", e)

    def _cleanup_orphans(self) -> None:
        """Remove sandbox/verifier containers (and their networks) left
        behind by prior eval cycles that didn't reach their finally
        block — typically because the validator was SIGKILL'd mid-cycle.
        """
        for role in ("verifier", "sandbox"):
            try:
                orphans = self.client.containers.list(
                    all=True,
                    filters={"label": f"trajectoryrl.role={role}"},
                )
            except Exception as e:
                logger.warning("Orphan scan failed for role=%s: %s", role, e)
                continue
            for c in orphans:
                try:
                    c.remove(force=True, v=True)
                    logger.info("Removed orphan %s container %s", role, c.name)
                except Exception as e:
                    logger.warning(
                        "Failed to remove orphan %s container %s: %s",
                        role, c.name, e,
                    )
        try:
            for net in self.client.networks.list(
                filters={"label": "trajectoryrl.role=eval_net"},
            ):
                try:
                    net.remove()
                    logger.info("Removed orphan eval network %s", net.name)
                except Exception as e:
                    logger.warning(
                        "Failed to remove orphan network %s: %s", net.name, e,
                    )
        except Exception as e:
            logger.warning("Orphan network scan failed: %s", e)

    def _pull_sync(self) -> None:
        self._cleanup_orphans()

        # Sandbox-agent image first.
        try:
            old_id: str | None = None
            try:
                old_id = self.client.images.get(self._sandbox_image).id
            except docker.errors.ImageNotFound:
                pass
            logger.info("Pulling sandbox-agent image: %s", self._sandbox_image)
            self.client.images.pull(self._sandbox_image)
            try:
                new_id = self.client.images.get(self._sandbox_image).id
            except docker.errors.ImageNotFound:
                new_id = None
            if old_id and new_id and old_id != new_id:
                try:
                    self.client.images.remove(old_id)
                    logger.info(
                        "Removed superseded image %s (%s)",
                        self._sandbox_image, old_id[:19],
                    )
                except docker.errors.APIError as e:
                    logger.debug(
                        "Skip rmi superseded %s (%s): %s",
                        self._sandbox_image, old_id[:19], e,
                    )
        except Exception as e:
            logger.warning(
                "Failed to pull %s: %s (using cached)",
                self._sandbox_image, e,
            )

        try:
            img = self.client.images.get(self._sandbox_image)
            digests = img.attrs.get("RepoDigests", [])
            self.bench_image_hash = (
                digests[0].split("@", 1)[-1] if digests else img.id
            )
        except Exception as e:
            logger.warning("Failed to get digest for sandbox image: %s", e)

        # Query sandbox version + scenarios. CLI is at /usr/local/bin/hermes
        # default entrypoint may interfere — use entrypoint="".
        try:
            info = _docker_run_json(
                self.client, self._sandbox_image,
                command=["python", "-m", "trajrl_bench.cli", "scenarios"],
            )
            self.sandbox_version = info.get("version", "unknown")
            self.sandbox_scenarios = info.get("scenarios", [])
            logger.info("Sandbox version: %s, scenarios: %s",
                        self.sandbox_version, self.sandbox_scenarios)
        except Exception as e:
            logger.warning("Failed to query sandbox version: %s", e)

        # Pull the per-scenario environment images for every scenario
        # this validator runs. ``scenario_image_hashes`` keys each one
        # for audit / submit-payload metadata.
        self.scenario_image_hashes = {}
        for scenario_name in SANDBOX_SCENARIOS:
            try:
                self._scenario_info = None  # force re-fetch per scenario
                self._load_scenario_info(scenario_name)
            except Exception as e:
                logger.warning(
                    "Failed to fetch scenario-info for %s: %s",
                    scenario_name, e,
                )
                continue
            scenario_image = self._scenario_image_ref()
            if not scenario_image:
                continue
            try:
                logger.info("Pulling scenario image: %s", scenario_image)
                self.client.images.pull(scenario_image)
                img = self.client.images.get(scenario_image)
                digests = img.attrs.get("RepoDigests", [])
                digest = digests[0].split("@", 1)[-1] if digests else img.id
                self.scenario_image_hashes[scenario_name] = digest
            except Exception as e:
                logger.warning(
                    "Failed to pull scenario image %s: %s",
                    scenario_image, e,
                )
        # Stable single-string hash for back-compat with the
        # ``scenario_image_hash`` heartbeat / submit field — concatenate
        # per-scenario hashes in canonical order.
        if self.scenario_image_hashes:
            ordered = [
                self.scenario_image_hashes.get(name, "missing")
                for name in SANDBOX_SCENARIOS
            ]
            self.scenario_image_hash = ":".join(ordered)

    # ------------------------------------------------------------------
    # Scenario plumbing
    # ------------------------------------------------------------------

    def _load_scenario_info(self, scenario: str) -> dict:
        """Fetch + cache the ``cli scenario-info`` payload for a scenario."""
        if self._scenario_info and self._scenario_info.get("name") == scenario:
            return self._scenario_info
        info = _docker_run_json(
            self.client, self._sandbox_image,
            command=["python", "-m", "trajrl_bench.cli", "scenario-info",
                     "--scenario", scenario],
        )
        self._scenario_info = info
        return info

    def _sandbox_tag(self) -> str:
        """Return the channel tag suffix on the sandbox image."""
        ref = self._sandbox_image
        if "@" in ref:
            ref = ref.split("@", 1)[0]
        if ":" in ref.split("/")[-1]:
            return ref.rsplit(":", 1)[1]
        return "latest"

    def _scenario_image_ref(self) -> str:
        """Return ``image_repo:<sandbox-tag>`` for the configured scenario."""
        info = self._scenario_info
        if not info:
            return ""
        repo = info.get("image_repo", "")
        if not repo:
            return ""
        return f"{repo}:{self._sandbox_tag()}"

    # ------------------------------------------------------------------
    # Evaluation entry points
    # ------------------------------------------------------------------

    async def evaluate_miner(
        self,
        skill_md: str,
        epoch_seed: int,
        pack_hash: str = "",
        validator_salt: str = "",
    ) -> SandboxEvaluationResult:
        salt = validator_salt or hashlib.sha256(
            f"{self.config.wallet_hotkey}:{self.config.netuid}".encode()
        ).hexdigest()[:16]

        logger.info(
            "eval starting: pack_hash=%s, seed=%d, scenarios=%s",
            pack_hash[:12] if pack_hash else "?",
            epoch_seed, list(SANDBOX_SCENARIOS),
        )

        loop = asyncio.get_event_loop()
        try:
            session_result = await loop.run_in_executor(
                None, lambda: self._run_eval_sync(
                    skill_md, epoch_seed, salt, pack_hash,
                ),
            )
        except Exception as e:
            logger.error("eval failed: %s", e, exc_info=True)
            session_result = _SessionResult()
            result = SandboxEvaluationResult(session_result)
            result.error = str(e)
            return result

        result = SandboxEvaluationResult(session_result)
        logger.info(
            "eval complete: final_score=%.3f (mean_q=%.3f, scenarios=%s)",
            result.score, result.mean_quality, result.scenario_qualities,
        )
        return result

    def _run_eval_sync(
        self, skill_md: str, epoch_seed: int, salt: str, pack_hash: str,
    ) -> _SessionResult:
        """One sandbox container, one episode per scenario.

        Each scenario contributes a correctness ratio (passed/total) in
        [0, 1]; final_score is the equal-weighted sum across scenarios.
        ``/app`` is re-injected from each scenario's environment image
        between cells. ``/workspace/learned`` persists across all
        scenarios so SKILL.md can build cross-scenario state.
        """
        # Pre-load metadata for every scenario so we fail early if the
        # sandbox image is missing one. Each ``_load_scenario_info``
        # call also primes the per-scenario image ref + tests payload.
        scenario_specs: list[dict] = []
        for name in SANDBOX_SCENARIOS:
            self._scenario_info = None  # bust cache; loop reads each
            info = self._load_scenario_info(name)
            scenario_image = self._scenario_image_ref()
            if not scenario_image:
                raise RuntimeError(
                    f"scenario {name!r} has no image_repo in scenario-info "
                    f"— bench too old?"
                )
            tests_files_b64 = info.get("tests_files_b64", {})
            if "test.sh" not in tests_files_b64:
                raise RuntimeError(f"scenario {name!r} ships no test.sh")
            scenario_specs.append({
                "name": name,
                "image": scenario_image,
                "instruction_md": info["instruction_md"],
                "agent_output_path": info["agent_output_path"],
                "verifier_timeout_s": int(info.get("verifier_timeout_s", 300)),
                "tests_files_b64": tests_files_b64,
            })

        session_id = secrets.token_hex(6)
        session_result = _SessionResult(pack_hash=pack_hash, validator_salt=salt)
        session_result.skill_md = skill_md

        logger.info(
            "[%s] session start (sandbox=%s, scenarios=%s)",
            session_id, self._sandbox_image,
            [s["name"] for s in scenario_specs],
        )

        sandbox = None
        try:
            sandbox = self.client.containers.run(
                self._sandbox_image,
                name=f"sandbox_{session_id}",
                detach=True,
                # Default bridge → internet egress for both the agent
                # (LLM API) and the verifier (apt + uv installs).
                network="bridge",
                environment={
                    "LLM_API_KEY":  self._testee_api_key,
                    "LLM_BASE_URL": self._testee_api_url,
                    "LLM_MODEL":    self._testee_model,
                },
                mem_limit="4g", cpu_quota=200000,
                labels={"trajectoryrl.role": "sandbox",
                        "trajectoryrl.session": session_id},
                log_config=LogConfig(type=LogConfig.types.JSON,
                                     config={"max-size": "50m"}),
            )

            for attempt in range(30):
                try:
                    sandbox.reload()
                    if sandbox.status == "running":
                        code, _ = sandbox.exec_run(
                            ["sh", "-c", "test -d /app && test -d /workspace"]
                        )
                        if code == 0:
                            break
                except Exception:
                    pass
                time.sleep(0.5 + min(attempt * 0.2, 1.5))
            else:
                raise RuntimeError("Sandbox container failed to start")

            _put_file(sandbox, "/workspace/SKILL.md", skill_md)
            sandbox.exec_run(["chown", "root:agent", "/workspace/SKILL.md"])
            sandbox.exec_run(["chmod", "440", "/workspace/SKILL.md"])
            sandbox.exec_run(["mkdir", "-p", "/workspace/learned"])
            sandbox.exec_run([
                "chown", "-R", "agent:agent", "/workspace/learned",
            ])

            for cell_index, spec in enumerate(scenario_specs):
                # Re-inject /app from the scenario's environment image.
                # Wipes whatever the prior scenario left; learned/
                # persists.
                sandbox.exec_run(
                    ["sh", "-c", "rm -rf /app && mkdir -p /app"]
                )
                self._setup_app_dir(sandbox, spec["image"])

                episode = self._run_episode(
                    session_id=session_id,
                    episode_index=cell_index,
                    scenario=spec["name"],
                    instruction_md=spec["instruction_md"],
                    sandbox=sandbox,
                    scenario_image=spec["image"],
                    agent_output_path=spec["agent_output_path"],
                    tests_files_b64=spec["tests_files_b64"],
                    verifier_timeout_s=spec["verifier_timeout_s"],
                )
                session_result.episodes.append(episode)

        finally:
            if sandbox:
                try:
                    sandbox.stop(timeout=5)
                    sandbox.remove(force=True, v=True)
                except Exception:
                    pass

        session_result.compute_scores()
        return session_result

    def _run_episode(
        self,
        session_id: str,
        episode_index: int,
        scenario: str,
        instruction_md: str,
        sandbox: Container,
        scenario_image: str,
        agent_output_path: str,
        tests_files_b64: dict[str, str],
        verifier_timeout_s: int,
    ) -> _EpisodeResult:
        """Run one cell: docker exec hermes → extract output → verifier."""
        ep_data = {
            "scenario": scenario,
            "instruction_md": instruction_md,
            "agent_output_path": agent_output_path,
        }
        episode = _EpisodeResult(
            episode_index=episode_index,
            scenario=scenario,
            ep_data=ep_data,
        )
        t0 = time.time()
        logger.info("[%s] %s starting", session_id, scenario)

        try:
            _put_file(sandbox, "/workspace/INSTRUCTION.md", instruction_md)
            sandbox.exec_run(["chown", "root:agent", "/workspace/INSTRUCTION.md"])
            sandbox.exec_run(["chmod", "440", "/workspace/INSTRUCTION.md"])

            # Reset per-episode Hermes state:
            #   1. Wipe the SQLite session DB so this episode's chat is the
            #      only session in the store. Without this,
            #      ``hermes sessions export`` ships the cumulative DB every
            #      episode and downstream readers can't tell which line is
            #      "this episode".
            #   2. Drop turns.jsonl + turns_export.err so a stale prior
            #      export doesn't masquerade as this run.
            sandbox.exec_run([
                "sh", "-c",
                "rm -f /opt/data/state.db /opt/data/state.db-* "
                "/workspace/turns.jsonl /workspace/turns_export.err && "
                "rm -rf /opt/data/sessions && "
                "mkdir -p /opt/data/sessions && "
                "chown -R agent:agent /opt/data",
            ])

            harness_prompt = (
                "Read /workspace/SKILL.md for your approach. "
                "Read /workspace/INSTRUCTION.md for this episode's task. "
                "The task's working directory is /app/. "
                "/workspace/learned/ is your scratch space across episodes; "
                "everything in /app and /workspace/learned is writable. "
                "Solve the task. Do not modify SKILL.md."
            )

            # Wrap chat + sessions export in one bash invocation so the
            # JSONL trace lands atomic per episode.
            agent_cmd = (
                "set +e; "
                f"hermes chat -q {_shell_quote(harness_prompt)} "
                f"-m {_shell_quote(self._testee_model)} "
                "-t terminal,file,code_execution,memory "
                "--quiet --yolo --max-turns 15; "
                "chat_rc=$?; "
                "hermes sessions export /workspace/turns.jsonl "
                "2>/workspace/turns_export.err; "
                "export_rc=$?; "
                "echo \"export_rc=$export_rc\" >> /workspace/turns_export.err; "
                "exit $chat_rc"
            )

            timeout = self.config.sandbox_timeout_per_episode
            exec_id = self.client.api.exec_create(
                sandbox.id,
                cmd=["bash", "-c", agent_cmd],
                user="agent",
                workdir="/workspace",
                stdout=True, stderr=False,
                environment={
                    "OPENROUTER_API_KEY": self._testee_api_key,
                    "OPENAI_API_KEY":     self._testee_api_key,
                    "ANTHROPIC_API_KEY":  self._testee_api_key,
                    "HERMES_BUNDLED_SKILLS": "/nonexistent",
                    "HOME": "/home/agent",
                },
            )["Id"]

            stream = self.client.api.exec_start(exec_id, stream=True, demux=False)
            transcript_chunks: list[bytes] = []
            stream_deadline = time.time() + timeout
            try:
                for chunk in stream:
                    if chunk:
                        transcript_chunks.append(chunk)
                    if time.time() > stream_deadline:
                        episode.timed_out = True
                        break
            except Exception as e:
                logger.warning(
                    "[%s] %s exec stream broke: %s",
                    session_id, scenario, e,
                )
            episode.transcript = b"".join(transcript_chunks).decode(
                "utf-8", errors="replace",
            )

            inspect = self.client.api.exec_inspect(exec_id)
            chat_exit = inspect.get("ExitCode")
            logger.info(
                "[%s] %s hermes chat finished (exit=%s, timed_out=%s, %d chars)",
                session_id, scenario, chat_exit, episode.timed_out,
                len(episode.transcript),
            )

            episode.turns_log = _read_container_text(
                sandbox, "/workspace/turns.jsonl",
            )
            episode.turns_export_err = _read_container_text(
                sandbox, "/workspace/turns_export.err",
            )

            agent_output_bytes = self._extract_file(sandbox, agent_output_path)
            if agent_output_bytes is None:
                logger.warning(
                    "[%s] %s agent produced no output at %s",
                    session_id, scenario, agent_output_path,
                )

            verifier_result = self._run_verifier_container(
                image=scenario_image,
                tests_files_b64=tests_files_b64,
                agent_output_path=agent_output_path,
                agent_output_bytes=agent_output_bytes,
                session_id=session_id,
                episode_index=episode_index,
                timeout=verifier_timeout_s,
            )
            reward = int(verifier_result.get("reward", 0))
            ctrf = verifier_result.get("ctrf")

            # Continuous correctness from ctrf.json (passed/total).
            # Falls back to the binary reward when the ctrf payload is
            # absent or malformed — we'd rather record *something*
            # than zero out a passing pack.
            passed, total = _parse_ctrf_correctness(ctrf)
            if total > 0:
                episode.quality = passed / total
            else:
                episode.quality = float(reward)

            # Cost axis: pulled from the agent's Hermes session export
            # (turns.jsonl). Reported alongside quality, NOT folded
            # into the score — separate axis on the leaderboard.
            episode.cost_usd = _parse_session_cost(episode.turns_log)

            episode.judge_result = {
                "reward": reward,
                "passed": passed,
                "total": total,
                "correctness": episode.quality,
                "cost_usd": episode.cost_usd,
                "verifier_stdout": verifier_result.get("stdout", ""),
                "ctrf": ctrf,
            }
            logger.info(
                "[%s] %s quality=%.3f (reward=%d, %d/%d tests passed, "
                "cost=%s)",
                session_id, scenario, episode.quality, reward,
                passed, total,
                f"${episode.cost_usd:.4f}" if episode.cost_usd is not None
                else "n/a",
            )

        except Exception as e:
            episode.error = str(e)
            logger.error(
                "[%s] %s failed: %s",
                session_id, scenario, e, exc_info=True,
            )

        episode.duration_s = time.time() - t0
        return episode

    # ------------------------------------------------------------------
    # Container helpers
    # ------------------------------------------------------------------

    def _setup_app_dir(self, sandbox: Container, scenario_image: str) -> None:
        """Extract /app from the scenario image and inject into the sandbox."""
        temp = self.client.containers.create(scenario_image)
        try:
            bits, _ = temp.get_archive("/app")
            buf = io.BytesIO()
            for chunk in bits:
                buf.write(chunk)
            buf.seek(0)
        finally:
            try:
                temp.remove(force=True)
            except docker.errors.APIError:
                pass

        sandbox.put_archive("/", buf)
        sandbox.exec_run(["chown", "-R", "agent:agent", "/app"])
        sandbox.exec_run(["chmod", "-R", "0770", "/app"])
        logger.info(
            "Injected /app from %s into sandbox %s",
            scenario_image, sandbox.name,
        )

    def _extract_file(self, container: Container, path: str) -> bytes | None:
        """Return the raw bytes at ``path`` inside ``container`` or None."""
        try:
            archive, _ = container.get_archive(path)
        except docker.errors.NotFound:
            return None
        except Exception:
            return None
        try:
            buf = io.BytesIO()
            for chunk in archive:
                buf.write(chunk)
            buf.seek(0)
            with tarfile.open(fileobj=buf, mode="r") as tar:
                members = tar.getmembers()
                if not members:
                    return None
                f = tar.extractfile(members[0])
                if f is None:
                    return None
                return f.read()
        except Exception:
            return None

    def _run_verifier_container(
        self,
        image: str,
        tests_files_b64: dict[str, str],
        agent_output_path: str,
        agent_output_bytes: bytes | None,
        session_id: str,
        episode_index: int,
        timeout: int,
    ) -> dict:
        """Run the scenario's tests/test.sh against the agent's output."""
        name = f"verifier_{session_id}_ep{episode_index}"

        container = self.client.containers.create(
            image=image,
            name=name,
            command=["bash", "-c", "mkdir -p /logs/verifier && bash /test.sh"],
            network_mode="bridge",
            labels={
                "trajectoryrl.role": "verifier",
                "trajectoryrl.session": session_id,
                "trajectoryrl.episode": str(episode_index),
            },
            log_config=LogConfig(type=LogConfig.types.JSON,
                                 config={"max-size": "10m"}),
        )

        try:
            test_sh_b64 = tests_files_b64.get("test.sh")
            if not test_sh_b64:
                return {"reward": 0, "stdout": "no test.sh in tests_files_b64",
                        "ctrf": None}
            test_sh_bytes = base64.b64decode(test_sh_b64)

            buf = io.BytesIO()
            with tarfile.open(fileobj=buf, mode="w") as tar:
                info = tarfile.TarInfo(name="test.sh")
                info.size = len(test_sh_bytes)
                info.mode = 0o755
                info.mtime = int(time.time())
                tar.addfile(info, io.BytesIO(test_sh_bytes))
            buf.seek(0)
            container.put_archive("/", buf)

            buf2 = io.BytesIO()
            with tarfile.open(fileobj=buf2, mode="w") as tar:
                for fname, body_b64 in tests_files_b64.items():
                    if fname == "test.sh":
                        continue
                    body = base64.b64decode(body_b64)
                    info = tarfile.TarInfo(name=f"tests/{fname}")
                    info.size = len(body)
                    info.mode = 0o644
                    info.mtime = int(time.time())
                    tar.addfile(info, io.BytesIO(body))
            buf2.seek(0)
            container.put_archive("/", buf2)

            if agent_output_bytes is not None:
                out_path = Path(agent_output_path)
                buf3 = io.BytesIO()
                with tarfile.open(fileobj=buf3, mode="w") as tar:
                    info = tarfile.TarInfo(name=out_path.name)
                    info.size = len(agent_output_bytes)
                    info.mode = 0o644
                    info.mtime = int(time.time())
                    tar.addfile(info, io.BytesIO(agent_output_bytes))
                buf3.seek(0)
                container.put_archive(str(out_path.parent), buf3)

            container.start()
            logger.info(
                "Started verifier %s (id=%s)", name, container.short_id,
            )

            try:
                result = container.wait(timeout=timeout)
                exit_code = result.get("StatusCode", -1)
            except Exception:
                logger.warning(
                    "Verifier %s timed out after %ds, killing", name, timeout,
                )
                try:
                    container.kill()
                except docker.errors.APIError:
                    pass
                exit_code = -1

            try:
                verifier_log = container.logs(
                    stdout=True, stderr=True,
                ).decode(errors="replace")
            except docker.errors.APIError:
                verifier_log = ""

            raw = self._extract_file(container, "/logs/verifier/reward.txt")
            if raw is None:
                logger.warning(
                    "Verifier %s did not write reward.txt (exit_code=%d)",
                    name, exit_code,
                )
                return {"reward": 0, "stdout": verifier_log, "ctrf": None}
            try:
                reward = max(0, min(1, int(raw.decode().strip())))
            except (ValueError, TypeError) as e:
                logger.warning("Verifier %s reward.txt unparseable: %s", name, e)
                reward = 0

            ctrf = None
            ctrf_raw = self._extract_file(container, "/logs/verifier/ctrf.json")
            if ctrf_raw:
                try:
                    ctrf = json.loads(ctrf_raw.decode(errors="replace"))
                except Exception:
                    pass

            logger.info(
                "Verifier %s: reward=%d (exit_code=%d)",
                name, reward, exit_code,
            )
            return {"reward": reward, "stdout": verifier_log, "ctrf": ctrf}

        except Exception as e:
            logger.error("verifier %s failed: %s", name, e)
            return {"reward": 0, "stdout": "", "ctrf": None}
        finally:
            try:
                container.remove(force=True)
            except docker.errors.APIError:
                pass

    # ------------------------------------------------------------------
    # Static utility used by the validator pipeline to extract SKILL.md
    # from a downloaded pack.
    # ------------------------------------------------------------------

    @staticmethod
    def extract_skill_md(pack: dict) -> str | None:
        """Pull SKILL.md (or skill.md) out of a pack dict's ``files`` map."""
        files = pack.get("files") or {}
        for key in ("SKILL.md", "skill.md"):
            if key in files:
                return files[key]
        return None
