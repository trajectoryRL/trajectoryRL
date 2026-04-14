"""Season 1 trajectory-sandbox harness for evaluating SKILL.md packs.

Architecture: the validator does NOT import trajectory-sandbox as a Python
dependency. Instead, it runs scenario logic inside the sandbox Docker image
via `docker run`. This means:

  - Updating scenarios = rebuild sandbox image only (CI does this)
  - Watchtower or `docker pull` before eval gets the latest scenarios
  - Validator image is stable — just orchestration + bittensor

The flow:
  1. docker pull sandbox image (get latest scenarios)
  2. docker run sandbox generate (fixtures + instruction + world JSON)
  3. For each episode: start sandbox + harness containers, run agent, capture state
  4. docker run sandbox score (transcript + state → quality via LLM judge)
  5. Compute split-half delta from 4 quality scores

Usage:
    harness = TrajectorySandboxHarness(config)
    result = await harness.evaluate_miner(skill_md, epoch_seed)
"""

from __future__ import annotations

import asyncio
import hashlib
import io
import json
import logging
import secrets
import tarfile
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import docker
from docker.models.containers import Container
from docker.models.networks import Network
from docker.types import LogConfig

from ..utils.config import ValidatorConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result types (no trajectory-sandbox import needed)
# ---------------------------------------------------------------------------

@dataclass
class _EpisodeResult:
    episode_index: int
    quality: float = 0.0
    tool_calls: int = 0
    transcript: str = ""
    mock_state: dict = field(default_factory=dict)
    timed_out: bool = False
    error: str | None = None
    duration_s: float = 0.0
    judge_result: dict = field(default_factory=dict)


@dataclass
class _SessionResult:
    episodes: list[_EpisodeResult] = field(default_factory=list)
    early_mean: float = 0.0
    late_mean: float = 0.0
    delta: float = 0.0
    mean_quality: float = 0.0
    learning_bonus: float = 0.0
    final_score: float = 0.0
    pack_hash: str = ""
    validator_salt: str = ""
    scenario: str = ""

    def compute_scores(self, alpha: float = 0.5, early_floor: float = 0.3,
                       delta_threshold: float = 0.4) -> None:
        scores = [ep.quality for ep in self.episodes]
        if len(scores) < 4:
            self.mean_quality = sum(scores) / len(scores) if scores else 0.0
            self.final_score = self.mean_quality
            return
        self.early_mean = (scores[0] + scores[1]) / 2
        self.late_mean = (scores[2] + scores[3]) / 2
        self.delta = self.late_mean - self.early_mean
        if self.early_mean < early_floor and self.delta > delta_threshold:
            self.delta = 0.0
        self.mean_quality = sum(scores) / len(scores)
        self.learning_bonus = alpha * max(0.0, self.delta)
        self.final_score = self.mean_quality * (1 + self.learning_bonus)


class SandboxEvaluationResult:
    """Result from a Season 1 sandbox evaluation.

    Duck-type compatible with clawbench.EvaluationResult for the fields
    the validator pipeline reads.
    """

    def __init__(self, session_result: _SessionResult,
                 scenario_name: str = "incident_response"):
        self.scenario_name = scenario_name
        self.score = session_result.final_score
        self.success = session_result.final_score > 0.0
        self.tool_calls = sum(ep.tool_calls for ep in session_result.episodes)
        self.response = ""
        self.rubric = {}
        self.error: Optional[str] = None

        # Cost: not tracked in S1 (quality-based, not cost-based)
        self.cost_usd: Optional[float] = None
        self.token_usage: Optional[Dict[str, int]] = None
        self.model_usage: Optional[List[Dict[str, Any]]] = None
        self.trajectory: Optional[List[Dict[str, Any]]] = None
        self.input_message: Optional[str] = None
        self.raw_llm_response: Optional[Dict[str, Any]] = None
        self.all_requests: Optional[List[Dict[str, Any]]] = None
        self.session_key: Optional[str] = None
        self.session_file: Optional[str] = None

        # S1-specific
        self.session_result = session_result
        self.early_mean = session_result.early_mean
        self.late_mean = session_result.late_mean
        self.delta = session_result.delta
        self.mean_quality = session_result.mean_quality
        self.learning_bonus = session_result.learning_bonus
        self.episode_qualities = [ep.quality for ep in session_result.episodes]


# ---------------------------------------------------------------------------
# Docker helpers
# ---------------------------------------------------------------------------

def _docker_run_json(client: docker.DockerClient, image: str,
                     command: list[str], environment: dict | None = None,
                     timeout: int = 120) -> dict:
    """Run a command in a container and parse JSON stdout."""
    try:
        output = client.containers.run(
            image, command=command,
            environment=environment or {},
            remove=True, stdout=True, stderr=True,
            mem_limit="2g", cpu_quota=100000,
        )
        return json.loads(output.decode())
    except docker.errors.ContainerError as e:
        stderr = e.stderr.decode() if e.stderr else ""
        raise RuntimeError(f"Container command failed: {stderr}") from e
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Failed to parse container output as JSON: {e}") from e


def _put_file(container: Container, path: str, content: str | bytes) -> None:
    """Write a file into a running container via tar archive."""
    import posixpath
    dir_name = posixpath.dirname(path)
    file_name = posixpath.basename(path)
    data = content.encode() if isinstance(content, str) else content
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        info = tarfile.TarInfo(name=file_name)
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))
    buf.seek(0)
    container.put_archive(dir_name, buf)


def _generate_keypair() -> tuple[str, str]:
    """Generate ephemeral Ed25519 SSH keypair. Returns (private_key, public_key)."""
    import subprocess
    import tempfile
    import os
    with tempfile.TemporaryDirectory() as tmpdir:
        key_path = os.path.join(tmpdir, "key")
        subprocess.run(
            ["ssh-keygen", "-t", "ed25519", "-f", key_path, "-N", "",
             "-C", "eval-session", "-q"],
            check=True,
        )
        private_key = open(key_path).read()
        public_key = open(f"{key_path}.pub").read().strip()
    return private_key, public_key


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------

class TrajectorySandboxHarness:
    """Season 1 harness — all scenario logic runs inside the sandbox image.

    The validator does NOT pip-install trajectory-sandbox. Instead:
    1. `docker pull` the sandbox image (gets latest scenarios)
    2. `docker run ... generate` to produce fixtures
    3. Start sandbox + harness containers, run agent episodes
    4. `docker run ... score` to judge each episode
    5. Compute split-half delta locally (simple math)
    """

    def __init__(self, config: ValidatorConfig):
        self.config = config
        self.client = docker.from_env()

        self._sandbox_image = config.sandbox_image
        self._harness_image = config.harness_image
        self._llm_api_key = config.judge_api_key or config.clawbench_api_key
        self._llm_api_url = config.judge_base_url or config.clawbench_base_url
        self._llm_model = config.judge_model or config.clawbench_default_model

        # Sandbox version — queried at pull time, included in consensus payloads
        self.sandbox_version: str = "unknown"
        self.sandbox_scenarios: list[str] = []

        logger.info(
            "TrajectorySandboxHarness initialized "
            "(sandbox=%s, harness=%s, episodes=%d, timeout=%ds)",
            self._sandbox_image, self._harness_image,
            config.sandbox_num_episodes, config.sandbox_timeout_per_episode,
        )

    async def pull_latest(self) -> None:
        """Pull latest images and query sandbox version. Gets new scenarios."""
        loop = asyncio.get_event_loop()
        try:
            await loop.run_in_executor(None, self._pull_sync)
        except Exception as e:
            logger.warning("Failed to pull latest images: %s (using cached)", e)

    def _pull_sync(self) -> None:
        for image in [self._sandbox_image, self._harness_image]:
            try:
                logger.info("Pulling latest image: %s", image)
                self.client.images.pull(image)
            except Exception as e:
                logger.warning("Failed to pull %s: %s (using cached)", image, e)

        # Query sandbox version and available scenarios
        try:
            info = _docker_run_json(
                self.client, self._sandbox_image,
                command=["python", "-m", "trajectory_sandbox.cli", "scenarios"],
            )
            self.sandbox_version = info.get("version", "unknown")
            self.sandbox_scenarios = info.get("scenarios", [])
            logger.info("Sandbox version: %s, scenarios: %s",
                        self.sandbox_version, self.sandbox_scenarios)
        except Exception as e:
            logger.warning("Failed to query sandbox version: %s", e)

    async def evaluate_miner(
        self,
        skill_md: str,
        epoch_seed: int,
        pack_hash: str = "",
        validator_salt: str = "",
    ) -> SandboxEvaluationResult:
        """Run a full Season 1 evaluation.

        All scenario logic (fixture generation, LLM judge scoring) runs
        inside the sandbox Docker image. The validator only orchestrates
        containers and computes the final split-half delta.
        """
        num_episodes = self.config.sandbox_num_episodes
        salt = validator_salt or self._default_salt()

        logger.info(
            "S1 evaluation starting: pack_hash=%s, seed=%d, episodes=%d",
            pack_hash[:12] if pack_hash else "?", epoch_seed, num_episodes,
        )

        loop = asyncio.get_event_loop()
        try:
            session_result = await loop.run_in_executor(
                None, lambda: self._run_eval_sync(
                    skill_md, epoch_seed, salt, num_episodes, pack_hash,
                ),
            )
        except Exception as e:
            logger.error("S1 evaluation failed: %s", e, exc_info=True)
            session_result = _SessionResult()
            result = SandboxEvaluationResult(session_result)
            result.error = str(e)
            return result

        result = SandboxEvaluationResult(session_result, scenario_name=session_result.scenario)
        logger.info(
            "S1 evaluation complete: final_score=%.3f "
            "(mean_q=%.3f, delta=%.3f, episodes=%s)",
            result.score, result.mean_quality, result.delta,
            result.episode_qualities,
        )
        return result

    def _run_eval_sync(
        self, skill_md: str, epoch_seed: int, salt: str,
        num_episodes: int, pack_hash: str,
    ) -> _SessionResult:
        """Synchronous eval: generate → run episodes → score → delta."""

        session_id = secrets.token_hex(6)
        session_result = _SessionResult(pack_hash=pack_hash, validator_salt=salt)

        # -----------------------------------------------------------
        # Step 1: Generate fixtures via docker run
        # -----------------------------------------------------------
        logger.info("[%s] Generating fixtures via sandbox image...", session_id)
        gen_data = _docker_run_json(
            self.client, self._sandbox_image,
            command=["python", "-m", "trajectory_sandbox.cli", "generate",
                     "--seed", str(epoch_seed), "--salt", salt,
                     "--episodes", str(num_episodes)],
        )

        scenario = gen_data["scenario"]
        world_data = gen_data["world"]
        episodes_data = gen_data["episodes"]
        session_result.scenario = scenario

        logger.info("[%s] Scenario: %s, World: %s", session_id, scenario,
                    world_data["company"])

        # -----------------------------------------------------------
        # Step 2: Run episodes (sandbox + harness containers)
        # -----------------------------------------------------------
        private_key, public_key = _generate_keypair()
        network = None
        sandbox = None

        try:
            # Create isolated network
            network_name = f"eval_{session_id}"
            network = self.client.networks.create(
                network_name, driver="bridge", internal=True,
                labels={"trajectoryrl.role": "eval_net"},
            )

            # Start sandbox container
            sandbox = self.client.containers.run(
                self._sandbox_image,
                name=f"sandbox_{session_id}",
                detach=True, network=network.name,
                environment={
                    "SSH_PUBLIC_KEY": public_key,
                    "SSH_USER": "agent",
                },
                mem_limit="2g", cpu_quota=100000,
                labels={"trajectoryrl.role": "sandbox",
                        "trajectoryrl.session": session_id},
                log_config=LogConfig(type=LogConfig.types.JSON,
                                     config={"max-size": "50m"}),
            )

            # Wait for sandbox healthy
            for _ in range(60):
                time.sleep(1)
                try:
                    code, _ = sandbox.exec_run("echo ok")
                    if code == 0:
                        break
                except Exception:
                    pass
            else:
                raise RuntimeError("Sandbox failed to start")

            # Wait for mock services
            for _ in range(30):
                code, out = sandbox.exec_run(
                    ["sh", "-c", "curl -s http://localhost:8090/health"])
                if code == 0 and out:
                    try:
                        health = json.loads(out.decode())
                        if health.get("status") == "ok":
                            break
                    except Exception:
                        pass
                time.sleep(1)

            sandbox.reload()
            sandbox_ip = sandbox.attrs["NetworkSettings"]["Networks"][network.name]["IPAddress"]

            # Load SKILL.md
            _put_file(sandbox, "/workspace/SKILL.md", skill_md)
            sandbox.exec_run(["mkdir", "-p", "/workspace/learned"])

            # Run each episode
            for i, ep_data in enumerate(episodes_data):
                episode = self._run_episode(
                    session_id=session_id,
                    episode_index=i,
                    ep_data=ep_data,
                    world_data=world_data,
                    scenario=scenario,
                    sandbox=sandbox,
                    sandbox_ip=sandbox_ip,
                    network=network,
                    private_key=private_key,
                )
                session_result.episodes.append(episode)

        finally:
            if sandbox:
                try:
                    sandbox.stop(timeout=5)
                    sandbox.remove(force=True)
                except Exception:
                    pass
            if network:
                try:
                    network.remove()
                except Exception:
                    pass

        # -----------------------------------------------------------
        # Step 3: Compute split-half delta (simple math, no imports)
        # -----------------------------------------------------------
        session_result.compute_scores()
        return session_result

    def _run_episode(
        self, session_id: str, episode_index: int, ep_data: dict,
        world_data: dict, scenario: str,
        sandbox: Container, sandbox_ip: str,
        network: Network, private_key: str,
    ) -> _EpisodeResult:
        """Run a single episode: load fixtures, start harness, capture, score."""

        episode = _EpisodeResult(episode_index=episode_index)
        t0 = time.time()
        logger.info("[%s] Episode %d starting", session_id, episode_index)

        try:
            # Load fixtures into sandbox
            sandbox.exec_run(["sh", "-c", "curl -s -X POST http://localhost:8090/reset"])
            fixtures_json = json.dumps(ep_data["fixtures"])
            _put_file(sandbox, "/tmp/fixtures.json", fixtures_json)
            sandbox.exec_run(["sh", "-c",
                "curl -s -X POST http://localhost:8090/load_fixtures "
                "-H 'Content-Type: application/json' -d @/tmp/fixtures.json"])
            _put_file(sandbox, "/workspace/INSTRUCTION.md", ep_data["instruction_md"])

            # Start harness container
            harness_name = f"harness_{session_id}_ep{episode_index}"
            harness_prompt = (
                "Read /workspace/SKILL.md for your approach. "
                "Read /workspace/INSTRUCTION.md for the task. "
                "Check /workspace/learned/ for notes from prior episodes. "
                "Services are at http://localhost:8090 - start with /health. "
                "Do not modify SKILL.md."
            )

            harness = self.client.containers.run(
                self._harness_image,
                command=["chat", "-q", harness_prompt,
                         "--quiet", "--yolo", "--max-turns", "30"],
                name=harness_name, detach=True, network=network.name,
                environment={
                    "OPENROUTER_API_KEY": self._llm_api_key,
                    "LLM_API_KEY": self._llm_api_key,
                },
                mem_limit="4g", cpu_quota=200000,
                cap_add=["NET_ADMIN"],
                labels={"trajectoryrl.role": "harness",
                        "trajectoryrl.session": session_id},
                log_config=LogConfig(type=LogConfig.types.JSON,
                                     config={"max-size": "50m"}),
            )

            try:
                # Wait for harness to complete
                timeout = self.config.sandbox_timeout_per_episode
                try:
                    result = harness.wait(timeout=timeout)
                    exit_code = result.get("StatusCode", -1)
                except Exception:
                    logger.warning("[%s] Episode %d timed out after %ds",
                                   session_id, episode_index, timeout)
                    try:
                        harness.kill()
                    except Exception:
                        pass
                    episode.timed_out = True

                # Capture transcript
                try:
                    episode.transcript = harness.logs(
                        stdout=True, stderr=False
                    ).decode(errors="replace")
                except Exception:
                    pass

            finally:
                try:
                    harness.stop(timeout=3)
                    harness.remove(force=True)
                except Exception:
                    pass

            # Capture mock state
            code, state_raw = sandbox.exec_run(
                ["sh", "-c", "curl -s http://localhost:8090/state"])
            if code == 0 and state_raw:
                try:
                    episode.mock_state = json.loads(state_raw.decode())
                except Exception:
                    pass

            # -----------------------------------------------------------
            # Score via docker run (LLM judge runs inside sandbox image)
            # -----------------------------------------------------------
            logger.info("[%s] Episode %d scoring via sandbox image...",
                        session_id, episode_index)

            # Write scoring inputs to temp files, mount into scorer container
            import tempfile
            import os
            with tempfile.TemporaryDirectory() as tmpdir:
                # Write inputs
                world_path = os.path.join(tmpdir, "world.json")
                episode_path = os.path.join(tmpdir, "episode.json")
                transcript_path = os.path.join(tmpdir, "transcript.txt")
                state_path = os.path.join(tmpdir, "state.json")

                with open(world_path, "w") as f:
                    json.dump(world_data, f)
                with open(episode_path, "w") as f:
                    json.dump(ep_data, f)
                with open(transcript_path, "w") as f:
                    f.write(episode.transcript)
                with open(state_path, "w") as f:
                    json.dump(episode.mock_state, f)

                # Run scorer in sandbox image
                score_output = self.client.containers.run(
                    self._sandbox_image,
                    command=[
                        "python", "-m", "trajectory_sandbox.cli", "score",
                        "--world", "/data/world.json",
                        "--episode", "/data/episode.json",
                        "--transcript", "/data/transcript.txt",
                        "--state", "/data/state.json",
                        "--scenario", scenario,
                    ],
                    environment={
                        "LLM_API_KEY": self._llm_api_key,
                        "LLM_BASE_URL": self._llm_api_url,
                        "LLM_MODEL": self._llm_model,
                    },
                    volumes={tmpdir: {"bind": "/data", "mode": "ro"}},
                    remove=True, stdout=True, stderr=True,
                    mem_limit="2g",
                )

                score_data = json.loads(score_output.decode())
                episode.quality = score_data.get("quality", 0.0)
                episode.judge_result = score_data

                if score_data.get("error"):
                    logger.warning("[%s] Episode %d judge error: %s",
                                   session_id, episode_index, score_data["error"])

            logger.info("[%s] Episode %d quality=%.3f",
                        session_id, episode_index, episode.quality)

        except Exception as e:
            episode.error = str(e)
            logger.error("[%s] Episode %d failed: %s",
                         session_id, episode_index, e, exc_info=True)

        episode.duration_s = time.time() - t0
        return episode

    def _default_salt(self) -> str:
        data = f"{self.config.wallet_hotkey}:{self.config.netuid}"
        return hashlib.sha256(data.encode()).hexdigest()[:16]

    @staticmethod
    def extract_skill_md(pack: dict) -> str | None:
        """Extract SKILL.md content from a miner's pack."""
        files = pack.get("files", {})
        return files.get("SKILL.md") or files.get("skill.md")
