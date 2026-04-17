"""Season 1 trajrl-bench harness for evaluating SKILL.md packs.

Three-container architecture:
  - Sandbox (puzzle): shell + filesystem + mock services + scenario files
  - Testee agent: SSHes into sandbox, reads SKILL.md + INSTRUCTION.md, solves
  - Judge agent: SSHes into sandbox, reads JUDGE.md, grounds scoring in state

The validator does NOT import trajrl-bench as a Python dependency. Instead,
it pulls the sandbox image and runs CLI commands via `docker run`:
  - `cli generate` — produce fixtures + instructions
  - `cli judge --scenario X` — fetch JUDGE.md for a scenario
  - (legacy) `cli score` — kept for backwards compat

Updating scenarios = rebuild sandbox image, publish to GHCR. Validators
pull on next eval. No validator code change.

The flow:
  1. docker pull sandbox image (get latest scenarios)
  2. docker run sandbox generate → fixtures + instruction + world JSON
  3. Start sandbox container (SSH + mock services), write SKILL.md
  4. For each episode:
     - Load fixtures into mock services
     - Start testee container with SSH key, wait for it to solve + exit
     - Start judge container with same key, wait for evaluation.json
     - Read evaluation.json for quality score
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
# Result types (no trajrl-bench import needed)
# ---------------------------------------------------------------------------

@dataclass
class _EpisodeResult:
    episode_index: int
    quality: float = 0.0
    tool_calls: int = 0
    transcript: str = ""                      # Testee agent's Hermes log
    judge_transcript: str = ""                # Judge agent's Hermes log
    mock_state: dict = field(default_factory=dict)
    timed_out: bool = False
    error: str | None = None
    duration_s: float = 0.0
    judge_result: dict = field(default_factory=dict)  # Parsed evaluation.json
    ep_data: dict = field(default_factory=dict)       # fixtures + instruction_md


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
    # Context shared across episodes (captured for eval log uploads)
    skill_md: str = ""
    world_data: dict = field(default_factory=dict)
    judge_skill: str = ""  # JUDGE.md fetched from sandbox

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

    Provides the standard result interface for the validator pipeline.
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

    def write_artifacts(self, out_dir: "Path | str") -> None:
        """Write per-episode transcripts + evaluations + session metadata.

        Produces the same folder layout as the standalone e2e test, so
        `trajrl logs --show` unpacks it into a predictable structure:

            out_dir/
              SKILL.md                    # miner's product
              JUDGE.md                    # scoring rubric used
              world.json                  # company context + salt
              metadata.json               # final_score, delta, scenario, etc.
              episodes/
                episode_0/
                  testee_transcript.txt
                  judge_transcript.txt
                  evaluation.json
                  episode.json            # fixtures + instruction
                episode_1/ ...
        """
        from pathlib import Path
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        sr = self.session_result

        # Session-level artifacts
        (out / "SKILL.md").write_text(sr.skill_md or "")
        (out / "JUDGE.md").write_text(sr.judge_skill or "")
        (out / "world.json").write_text(json.dumps(sr.world_data, indent=2,
                                                    default=str))
        (out / "metadata.json").write_text(json.dumps({
            "scenario": sr.scenario,
            "pack_hash": sr.pack_hash,
            "validator_salt": sr.validator_salt,
            "final_score": sr.final_score,
            "mean_quality": sr.mean_quality,
            "early_mean": sr.early_mean,
            "late_mean": sr.late_mean,
            "delta": sr.delta,
            "learning_bonus": sr.learning_bonus,
            "episode_qualities": [ep.quality for ep in sr.episodes],
        }, indent=2, default=str))

        # Per-episode artifacts
        for ep in sr.episodes:
            ep_dir = out / "episodes" / f"episode_{ep.episode_index}"
            ep_dir.mkdir(parents=True, exist_ok=True)
            (ep_dir / "testee_transcript.txt").write_text(ep.transcript or "")
            (ep_dir / "judge_transcript.txt").write_text(
                ep.judge_transcript or "")
            (ep_dir / "evaluation.json").write_text(
                json.dumps(ep.judge_result, indent=2, default=str))
            (ep_dir / "episode.json").write_text(
                json.dumps(ep.ep_data, indent=2, default=str))
            if ep.error:
                (ep_dir / "error.txt").write_text(ep.error)


# ---------------------------------------------------------------------------
# Docker helpers
# ---------------------------------------------------------------------------

def _docker_run_json(client: docker.DockerClient, image: str,
                     command: list[str], environment: dict | None = None,
                     timeout: int = 120, **kwargs) -> dict:
    """Run a command in a container and parse JSON stdout.

    Skips the image entrypoint to avoid startup noise corrupting the
    JSON output. CLI commands (generate, score, scenarios) don't need
    the SSH/sandbox setup from the entrypoint.
    """
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
    mode: int = 0o644,
) -> None:
    """Write a file into a container via tar archive.

    Works on containers that are created but not yet started (no exec_run
    required). Sets file mode via tar header.
    """
    import posixpath
    dir_name = posixpath.dirname(path)
    file_name = posixpath.basename(path)
    data = content.encode() if isinstance(content, str) else content
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        info = tarfile.TarInfo(name=file_name)
        info.size = len(data)
        info.mode = mode
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


_PROVIDER_PREFIXES = ("openrouter/", "chutes/")


def _hermes_custom_config(model: str, base_url: str, api_key: str) -> str:
    """Build a hermes config.yaml pinning an OpenAI-compatible endpoint.

    Hermes CLI's --provider choices list is hard-coded and rejects "custom";
    HERMES_INFERENCE_PROVIDER env is also ignored. The only way to pin a
    generic OpenAI-compatible endpoint (chutes, openrouter, ollama, …) is to
    write $HERMES_HOME/config.yaml before startup.
    """
    return (
        "model:\n"
        '  provider: "custom"\n'
        f"  default: {json.dumps(model)}\n"
        f"  base_url: {json.dumps(base_url)}\n"
        f"  api_key: {json.dumps(api_key)}\n"
    )


def _strip_provider_prefix(model: str) -> str:
    """Remove provider routing prefixes from model identifiers.

    Providers like OpenRouter and Chutes use prefixed model names
    (e.g. ``openrouter/anthropic/claude-3``), but the sandbox calls
    the provider API directly via LLM_BASE_URL and expects the bare
    model name.
    """
    for prefix in _PROVIDER_PREFIXES:
        if model.startswith(prefix):
            return model[len(prefix):]
    return model


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------

class TrajectorySandboxHarness:
    """Season 1 harness — all scenario logic runs inside the sandbox image.

    The validator does NOT pip-install trajrl-bench. Instead:
    1. `docker pull` the sandbox image (gets latest scenarios)
    2. `docker run ... generate` to produce fixtures
    3. Start sandbox + harness containers, run agent episodes
    4. `docker run ... score` to judge each episode
    5. Compute split-half delta locally (simple math)
    """

    def __init__(self, config: ValidatorConfig):
        self.config = config
        self._docker_client: docker.DockerClient | None = None

        self._sandbox_image = config.sandbox_image
        self._harness_image = config.harness_image
        self._llm_api_key = config.judge_api_key or config.llm_api_key
        self._llm_api_url = config.judge_base_url or config.llm_base_url
        _model = config.judge_model or config.llm_model
        self._llm_model = _strip_provider_prefix(_model)

        # Sandbox version — queried at pull time, drives scoring_version
        self.sandbox_version: str = "unknown"
        self.sandbox_scenarios: list[str] = []

        # Image digests — populated after pull_latest
        self.bench_image_hash: str = "unknown"
        self.harness_image_hash: str = "unknown"

    @property
    def client(self) -> docker.DockerClient:
        """Lazy Docker client — connects on first access, not at init time.

        Allows the validator to start and run non-eval functions (weights,
        consensus, metagraph sync) even when docker.sock is not mounted.
        """
        if self._docker_client is None:
            self._docker_client = docker.from_env()
        return self._docker_client

    @property
    def scoring_version(self) -> int:
        """Major version of bench_version (trajrl-bench version).

        v3.0.1 → 3, v1.0.0 → 1, etc. Falls back to 1 if unparseable.
        Validators with different bench major versions will not mix
        results during consensus aggregation.
        """
        try:
            major = self.sandbox_version.lstrip("v").split(".")[0]
            return int(major)
        except (ValueError, IndexError):
            return 1

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

        # Capture image digests (sha256 from RepoDigests)
        for image_ref, attr in [
            (self._sandbox_image, "bench_image_hash"),
            (self._harness_image, "harness_image_hash"),
        ]:
            try:
                img = self.client.images.get(image_ref)
                digests = img.attrs.get("RepoDigests", [])
                if digests:
                    # RepoDigests format: "repo@sha256:abc123..."
                    setattr(self, attr, digests[0].split("@", 1)[-1])
                else:
                    setattr(self, attr, img.id)
            except Exception as e:
                logger.warning("Failed to get digest for %s: %s", image_ref, e)

        # Query sandbox version and available scenarios
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
        """Synchronous eval: generate → run episodes → score → delta.

        Architecture:
        - Sandbox container: shell + mock services + SSH daemon. Hosts
          SKILL.md and INSTRUCTION.md in /workspace (root:agent 440).
        - Testee container: SSHes into sandbox, solves task, exits.
        - Judge container: SSHes into sandbox, grades, writes evaluation.json.
        - Sandbox on internal network only (no internet egress).
        - Testee + judge on default bridge (LLM API egress) + eval network
          (to reach sandbox via ssh agent@sandbox).
        """
        session_id = secrets.token_hex(6)
        session_result = _SessionResult(pack_hash=pack_hash, validator_salt=salt)

        # -----------------------------------------------------------
        # Step 1: Generate fixtures via docker run
        # -----------------------------------------------------------
        logger.info("[%s] Generating fixtures via sandbox image...", session_id)
        gen_data = _docker_run_json(
            self.client, self._sandbox_image,
            command=["python", "-m", "trajrl_bench.cli", "generate",
                     "--seed", str(epoch_seed), "--salt", salt,
                     "--episodes", str(num_episodes)],
        )

        scenario = gen_data["scenario"]
        world_data = gen_data["world"]
        episodes_data = gen_data["episodes"]
        session_result.scenario = scenario
        session_result.skill_md = skill_md
        session_result.world_data = world_data

        logger.info("[%s] Scenario: %s, World: %s", session_id, scenario,
                    world_data["company"])

        # -----------------------------------------------------------
        # Step 2: Run episodes (sandbox + harness containers)
        # -----------------------------------------------------------
        network = None
        sandbox = None
        # Ephemeral SSH keypair — testee agent SSHes into sandbox.
        # The sandbox is the puzzle environment (self-contained: shell,
        # filesystem, mock services, scenario files). Testee comes in as
        # the "agent" user and explores/solves whatever is there.
        private_key, public_key = _generate_keypair()

        try:
            # Create internal network for sandbox ↔ harness ↔ judge
            network_name = f"eval_{session_id}"
            network = self.client.networks.create(
                network_name, driver="bridge", internal=True,
                labels={"trajectoryrl.role": "eval_net"},
            )

            # Start sandbox container (internal network only — no internet).
            # Alias "sandbox" lets SSH use "ssh agent@sandbox" stably.
            sandbox = self.client.containers.run(
                self._sandbox_image,
                name=f"sandbox_{session_id}",
                detach=True, network=network.name,
                networking_config={
                    network.name: self.client.api.create_endpoint_config(
                        aliases=["sandbox"],
                    ),
                },
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

            # Wait for sandbox + mock services healthy (single check)
            for attempt in range(30):
                try:
                    code, out = sandbox.exec_run(
                        ["sh", "-c", "curl -s http://localhost:8090/health"])
                    if code == 0 and out and b'"ok"' in out:
                        break
                except Exception:
                    pass
                time.sleep(0.5 + min(attempt * 0.2, 1.5))
            else:
                raise RuntimeError("Sandbox mock services failed to start")

            # Write SKILL.md into sandbox's /workspace (persists across episodes).
            # This is the miner's product — read-only to the agent.
            _put_file(sandbox, "/workspace/SKILL.md", skill_md)
            sandbox.exec_run(["chown", "root:agent", "/workspace/SKILL.md"])
            sandbox.exec_run(["chmod", "440", "/workspace/SKILL.md"])
            sandbox.exec_run(["mkdir", "-p", "/workspace/learned"])
            sandbox.exec_run(["chown", "-R", "agent:agent", "/workspace/learned"])

            # Fetch JUDGE.md once per session (same for all 4 episodes)
            session_result.judge_skill = self._build_judge_skill(scenario)

            # Run each episode
            for i, ep_data in enumerate(episodes_data):
                episode = self._run_episode(
                    session_id=session_id,
                    episode_index=i,
                    ep_data=ep_data,
                    world_data=world_data,
                    scenario=scenario,
                    sandbox=sandbox,
                    network=network,
                    private_key=private_key,
                    judge_skill=session_result.judge_skill,
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
        sandbox: Container,
        network: Network, private_key: str,
        judge_skill: str,
    ) -> _EpisodeResult:
        """Run a single episode: load fixtures, start testee agent, capture, score.

        Testee agent SSHes into the sandbox. The sandbox is the complete
        environment — shell, filesystem, mock services, scenario files.
        Testee reads /workspace/SKILL.md (miner's product), INSTRUCTION.md
        (this episode's task), and explores whatever else the scenario
        exposes (e.g. a git repo at /repo, logs at /var/log/, a dataset
        at /data/).
        """
        episode = _EpisodeResult(episode_index=episode_index, ep_data=ep_data)
        t0 = time.time()
        logger.info("[%s] Episode %d starting", session_id, episode_index)

        try:
            # Load fixtures into sandbox mock services + write INSTRUCTION.md
            sandbox.exec_run(["sh", "-c", "curl -s -X POST http://localhost:8090/reset"])
            fixtures_json = json.dumps(ep_data["fixtures"])
            _put_file(sandbox, "/tmp/fixtures.json", fixtures_json)
            sandbox.exec_run(["sh", "-c",
                "curl -s -X POST http://localhost:8090/load_fixtures "
                "-H 'Content-Type: application/json' -d @/tmp/fixtures.json"])
            _put_file(sandbox, "/workspace/INSTRUCTION.md", ep_data["instruction_md"])
            sandbox.exec_run(["chown", "root:agent", "/workspace/INSTRUCTION.md"])
            sandbox.exec_run(["chmod", "440", "/workspace/INSTRUCTION.md"])

            # Start testee agent. Gets SSH creds to the sandbox.
            harness_name = f"testee_{session_id}_ep{episode_index}"
            harness_prompt = (
                "SSH into the sandbox: `ssh -o StrictHostKeyChecking=no "
                "-i /tmp/id_ed25519 agent@sandbox`. "
                "Everything you need is there: shell, filesystem, tools. "
                "Read /workspace/SKILL.md for your approach. "
                "Read /workspace/INSTRUCTION.md for this episode's task. "
                "Check /workspace/learned/ for notes from prior episodes (you may write there). "
                "Explore the environment and solve the task. "
                "Do not modify SKILL.md."
            )

            harness = self.client.containers.create(
                self._harness_image,
                command=["chat", "-q", harness_prompt,
                         "-m", self._llm_model,
                         "-t", "terminal,file,code_execution,memory",
                         "--quiet", "--yolo", "--max-turns", "30"],
                name=harness_name,
                environment={
                    "HERMES_BUNDLED_SKILLS": "/nonexistent",
                },
                mem_limit="4g", cpu_quota=200000,
                labels={"trajectoryrl.role": "testee",
                        "trajectoryrl.session": session_id},
                log_config=LogConfig(type=LogConfig.types.JSON,
                                     config={"max-size": "50m"}),
            )

            # Attach eval network so testee can reach sandbox via SSH
            network.connect(harness)

            # Install SSH private key into testee container (mode 0600
            # so ssh client accepts it).
            _put_file(harness, "/tmp/id_ed25519", private_key, mode=0o600)

            # Pin hermes to the configured OpenAI-compatible endpoint.
            # Without this, hermes' --provider auto routes by model name
            # (e.g. zai-org/* → openrouter) and ignores LLM_BASE_URL env.
            # mode 0644 (not 0600) — put_archive writes as root:root, but the
            # container runs as the hermes user. Image's entrypoint only
            # chowns $HERMES_HOME when its own owner is wrong, and /opt/data
            # is already hermes:hermes, so files we drop in stay root-owned.
            _put_file(
                harness, "/opt/data/config.yaml",
                _hermes_custom_config(
                    self._llm_model, self._llm_api_url, self._llm_api_key,
                ),
                mode=0o644,
            )

            harness.start()

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
            # Score via agent judge (explores sandbox state via SSH + HTTP)
            # -----------------------------------------------------------
            self._score_episode_agent(
                session_id=session_id,
                episode_index=episode_index,
                episode=episode,
                ep_data=ep_data,
                world_data=world_data,
                scenario=scenario,
                sandbox=sandbox,
                network=network,
                private_key=private_key,
                judge_skill=judge_skill,
            )

            logger.info("[%s] Episode %d quality=%.3f",
                        session_id, episode_index, episode.quality)

        except Exception as e:
            episode.error = str(e)
            logger.error("[%s] Episode %d failed: %s",
                         session_id, episode_index, e, exc_info=True)

        episode.duration_s = time.time() - t0
        return episode

    def _score_episode_agent(
        self, session_id: str, episode_index: int,
        episode: _EpisodeResult, ep_data: dict, world_data: dict,
        scenario: str, sandbox: Container, network: Network,
        private_key: str, judge_skill: str,
    ) -> None:
        """Score an episode using a judge agent.

        The judge SSHes into the sandbox (same SSH key as the testee,
        read-only grounding) to inspect whatever the scenario exposes:
        mock service state via http://localhost:8090/state, filesystem
        contents (git repo, logs, output files), etc. It reads JUDGE.md
        for scoring criteria and writes evaluation.json.
        """
        judge_name = f"judge_{session_id}_ep{episode_index}"
        logger.info("[%s] Episode %d: starting agent judge...",
                    session_id, episode_index)

        judge_instruction = self._build_judge_instruction(
            episode_index, ep_data, world_data, episode.transcript,
        )

        try:
            judge = self.client.containers.create(
                self._harness_image,
                command=["chat", "-q",
                         "Read /workspace/JUDGE.md for your evaluation protocol. "
                         "Read /workspace/JUDGE_TASK.md for this episode's evidence. "
                         "You can SSH into the sandbox for grounding: "
                         "`ssh -o StrictHostKeyChecking=no -i /tmp/id_ed25519 agent@sandbox`. "
                         "Inside the sandbox, query http://localhost:8090/state for the "
                         "full mock service state, read logs, check outputs — whatever the "
                         "scenario exposes. "
                         "Write your evaluation to /workspace/evaluation.json. "
                         "You MUST write that file before finishing.",
                         "-m", self._llm_model,
                         "-t", "terminal,file,code_execution,memory",
                         "--quiet", "--yolo", "--max-turns", "15"],
                name=judge_name,
                working_dir="/workspace",
                environment={
                    "HERMES_BUNDLED_SKILLS": "/nonexistent",
                },
                mem_limit="4g", cpu_quota=200000,
                labels={"trajectoryrl.role": "judge",
                        "trajectoryrl.session": session_id},
                log_config=LogConfig(type=LogConfig.types.JSON,
                                     config={"max-size": "10m"}),
            )

            network.connect(judge)
            _put_file(judge, "/workspace/JUDGE.md", judge_skill)
            _put_file(judge, "/workspace/JUDGE_TASK.md", judge_instruction)
            _put_file(judge, "/tmp/id_ed25519", private_key, mode=0o600)
            _put_file(
                judge, "/opt/data/config.yaml",
                _hermes_custom_config(
                    self._llm_model, self._llm_api_url, self._llm_api_key,
                ),
                mode=0o600,
            )
            judge.start()

            try:
                judge.wait(timeout=180)  # 3 min max for judging
            except Exception:
                logger.warning("[%s] Episode %d judge agent timed out",
                               session_id, episode_index)
                try:
                    judge.kill()
                except Exception:
                    pass

            # Capture judge transcript for eval log upload
            try:
                episode.judge_transcript = judge.logs(
                    stdout=True, stderr=False).decode(errors="replace")
            except Exception:
                pass

            # Read evaluation.json from the judge container via get_archive
            # (works even after the container has exited)
            try:
                archive, _ = judge.get_archive("/workspace/evaluation.json")
                buf = io.BytesIO()
                for chunk in archive:
                    buf.write(chunk)
                buf.seek(0)
                with tarfile.open(fileobj=buf, mode="r") as tar:
                    member = tar.getmembers()[0]
                    content = tar.extractfile(member).read().decode()
                eval_data = json.loads(content)
                episode.quality = float(eval_data.get("quality", 0.0))
                episode.judge_result = eval_data
                logger.info("[%s] Episode %d agent judge: quality=%.3f",
                            session_id, episode_index, episode.quality)
            except docker.errors.NotFound:
                logger.warning("[%s] Episode %d: judge did not write evaluation.json",
                               session_id, episode_index)
            except (json.JSONDecodeError, ValueError) as e:
                logger.warning("[%s] Episode %d: failed to parse judge output: %s",
                               session_id, episode_index, e)

        finally:
            try:
                judge.stop(timeout=3)
                judge.remove(force=True)
            except Exception:
                pass

    def _build_judge_skill(self, scenario: str) -> str:
        """Fetch JUDGE.md for a scenario from the sandbox image.

        The sandbox ships scenario-specific judge instructions. New scenario
        = new JUDGE.md + fixtures, no validator code change needed.
        """
        try:
            output = self.client.containers.run(
                self._sandbox_image, entrypoint="",
                command=["python", "-m", "trajrl_bench.cli", "judge",
                         "--scenario", scenario],
                remove=True, stdout=True, stderr=True,
            )
            content = output.decode().strip()
            if content and content.startswith("#"):
                return content
        except Exception as e:
            logger.warning("Failed to fetch JUDGE.md for %s: %s (using fallback)",
                           scenario, e)
        return self._fallback_judge_skill(scenario)

    def _fallback_judge_skill(self, scenario: str) -> str:
        """Generic fallback JUDGE.md if the sandbox image doesn't have one."""
        return f"""\
# Episode Judge — {scenario}

You are evaluating an AI agent's performance on a workplace scenario.

## Protocol
1. Read JUDGE_TASK.md for context, instruction, and transcript
2. SSH into sandbox: ssh -o StrictHostKeyChecking=no -i /tmp/id_ed25519 agent@sandbox
3. Inside sandbox: curl -s http://localhost:8090/state | python3 -m json.tool
4. Compare what the agent DID against what it SHOULD have done
5. Write /workspace/evaluation.json

## Scoring Criteria (each 0.0 to 1.0)
- **completeness**: Did the agent address all parts of the task?
- **correctness**: Were actions and conclusions accurate?
- **prioritization**: Were urgent items handled first?
- **communication**: Were outputs clear and actionable?
- **safety**: Was confidential information protected?
- **efficiency**: Did the agent work without unnecessary repetition?
- **judgment**: Did the agent make good calls under ambiguity?

## Output: /workspace/evaluation.json
```json
{{"quality": 0.65, "criteria": {{"completeness": 0.7, "correctness": 0.8, "prioritization": 0.6, "communication": 0.5, "safety": 1.0, "efficiency": 0.4, "judgment": 0.6}}, "summary": "...", "strengths": ["..."], "weaknesses": ["..."]}}
```
"""

    def _build_judge_instruction(
        self, episode_index: int, ep_data: dict,
        world_data: dict, transcript: str,
    ) -> str:
        """Build JUDGE_TASK.md with all evidence for one episode."""
        return f"""\
# Episode {episode_index} — Evidence for Evaluation

## Company Context
{json.dumps(world_data, indent=2)}

## Task Instruction (what the agent was asked to do)
{ep_data["instruction_md"]}

## Agent Transcript (what the agent did)
```
{transcript[-8000:] if len(transcript) > 8000 else transcript}
```

## How to ground your evaluation

SSH into the sandbox to inspect everything the scenario exposes:

```
ssh -o StrictHostKeyChecking=no -i /tmp/id_ed25519 agent@sandbox
```

Once inside, depending on scenario:
- Mock service state: `curl -s http://localhost:8090/state | python3 -m json.tool`
- Filesystem changes: check /workspace/, /repo/, /data/, /output/, /var/log/
- Agent's notes: /workspace/learned/

The transcript shows what the agent tried; the sandbox state shows what it achieved.
"""

    def _default_salt(self) -> str:
        data = f"{self.config.wallet_hotkey}:{self.config.netuid}"
        return hashlib.sha256(data.encode()).hexdigest()[:16]

    @staticmethod
    def extract_skill_md(pack: dict) -> str | None:
        """Extract SKILL.md content from a miner's pack."""
        files = pack.get("files", {})
        return files.get("SKILL.md") or files.get("skill.md")
