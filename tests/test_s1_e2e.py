"""End-to-end Season 1 validator evaluation test (SSH architecture).

Runs the full flow with real Docker containers and real LLM API calls:
  1. Fixture generation via sandbox CLI
  2. Sandbox container startup (mock services + SSH daemon)
  3. Testee agent SSHes into sandbox, reads SKILL.md, solves task
  4. Judge agent SSHes into sandbox, writes evaluation.json
  5. Split-half delta computation

Requirements:
  - Docker daemon running
  - Images cached: ghcr.io/trajectoryrl/trajrl-bench:latest,
    ghcr.io/trajectoryrl/hermes-agent:latest
  - LLM_API_KEY set (OpenRouter or compatible)

Run:
  cd trajectoryRL
  source .env.validator
  python -m pytest tests/test_s1_e2e.py -v -s

Takes ~15-20 min for the full 4-episode run.
"""

import asyncio
import json
import logging
import os
import time
from pathlib import Path

import pytest

docker = pytest.importorskip("docker")

LLM_API_KEY = os.environ.get("LLM_API_KEY") or os.environ.get("CLAWBENCH_LLM_API_KEY", "")
LLM_BASE_URL = os.environ.get("LLM_BASE_URL") or os.environ.get("CLAWBENCH_LLM_BASE_URL", "https://openrouter.ai/api/v1")
from trajectoryrl.utils.sandbox_harness import _strip_provider_prefix
_raw_model = os.environ.get("LLM_MODEL") or os.environ.get("CLAWBENCH_DEFAULT_MODEL", "z-ai/glm-5.1")
LLM_MODEL = _strip_provider_prefix(_raw_model)

SANDBOX_IMAGE = os.environ.get("SANDBOX_IMAGE", "ghcr.io/trajectoryrl/trajrl-bench:latest")
HARNESS_IMAGE = os.environ.get("HARNESS_IMAGE", "ghcr.io/trajectoryrl/hermes-agent:latest")

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def _skip_if_no_docker():
    try:
        docker.from_env().ping()
    except Exception:
        pytest.skip("Docker daemon not available")


def _skip_if_no_api_key():
    if not LLM_API_KEY:
        pytest.skip("LLM_API_KEY not set")


# ---------------------------------------------------------------------------
# Test SKILL.md — SSH-aware, uses curl inside sandbox, exits when done
# ---------------------------------------------------------------------------

SKILLED_SKILL_MD = """\
# Incident Response Agent (via SSH)

You will SSH into a sandbox that contains mock services and a shell.

## Protocol (inside the sandbox)
1. `curl -s http://localhost:8090/health` to see available services
2. `curl -s http://localhost:8090/api/v2/messages | jq` to read all emails
3. `curl -s http://localhost:8090/slack/channels | jq` to list channels
4. For each channel: `curl -s http://localhost:8090/slack/channels/{id}/messages | jq`
5. `curl -s http://localhost:8090/calendar/events | jq`
6. `curl -s http://localhost:8090/api/v1/repos/company/main/issues | jq`

## Triage
- P0 (security/outage): act immediately, post to #incidents
- P1 (degraded): schedule a fix, notify team
- P2 (routine): note but don't block on
- Protect confidential info (do NOT post to public channels)

## Actions
- Post status to #incidents:
  `curl -s -X POST http://localhost:8090/slack/channels/incidents/messages \\
    -H "Content-Type: application/json" -d '{"text":"..."}'`
- Email stakeholders via `POST /api/v2/messages`
- Write notes to `/workspace/learned/notes.md` for future episodes

Be direct. No exploration. Exit the SSH session when done.
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(num_episodes: int = 4, timeout: int = 300):
    """Build a ValidatorConfig for S1 testing."""
    from trajectoryrl.utils.config import ValidatorConfig

    return ValidatorConfig(
        wallet_name="test",
        wallet_hotkey="test",
        netuid=11,
        network="test",
        evaluation_harness="trajrl-bench",
        sandbox_image=SANDBOX_IMAGE,
        harness_image=HARNESS_IMAGE,
        llm_api_key=LLM_API_KEY,
        llm_base_url=LLM_BASE_URL,
        llm_model=LLM_MODEL,
        sandbox_num_episodes=num_episodes,
        sandbox_timeout_per_episode=timeout,
        eval_state_path=Path("/tmp/trajrl-test-eval-state.json"),
        winner_state_path=Path("/tmp/trajrl-test-winner.json"),
        pack_cache_dir=Path("/tmp/trajrl-test-packs"),
    )


# ---------------------------------------------------------------------------
# Unit tests (no Docker, no LLM) — fast, run in regular test suite
# ---------------------------------------------------------------------------

class TestS1UnitMath:
    """Split-half delta math + validator dispatch logic."""

    def test_pack_content_routing(self):
        """SKILL.md extraction handles both casings, returns None for AGENTS.md."""
        from trajectoryrl.utils.sandbox_harness import TrajectorySandboxHarness

        s1_pack = {"schema_version": 1, "files": {"SKILL.md": SKILLED_SKILL_MD}}
        assert TrajectorySandboxHarness.extract_skill_md(s1_pack) == SKILLED_SKILL_MD

        v4_pack = {"schema_version": 1, "files": {"AGENTS.md": "# Policy"}}
        assert TrajectorySandboxHarness.extract_skill_md(v4_pack) is None

        lower_pack = {"schema_version": 1, "files": {"skill.md": "# test"}}
        assert TrajectorySandboxHarness.extract_skill_md(lower_pack) == "# test"

    def test_spec_number_constant(self):
        """SPEC_NUMBER is a validator-side constant decoupled from sandbox version."""
        _skip_if_no_docker()
        from trajectoryrl.utils.sandbox_harness import TrajectorySandboxHarness
        from trajectoryrl.utils.config import SPEC_NUMBER

        # The harness no longer exposes a derived scoring_version property.
        # sandbox_version remains as an audit / log field only.
        harness = TrajectorySandboxHarness(_make_config())
        assert not hasattr(harness, "scoring_version")

        harness.sandbox_version = "3.1.0"  # bench bumps must not move SPEC_NUMBER
        assert isinstance(SPEC_NUMBER, int) and SPEC_NUMBER >= 1

    def test_write_artifacts(self):
        """SandboxEvaluationResult.write_artifacts() creates correct dir layout."""
        import tempfile
        from pathlib import Path
        from trajectoryrl.utils.sandbox_harness import (
            _EpisodeResult, _SessionResult, SandboxEvaluationResult,
        )

        sr = _SessionResult(
            scenario="incident_response",
            pack_hash="abc123",
            validator_salt="test_salt",
            skill_md="# My SKILL",
            judge_skill="# JUDGE rubric",
            world_data={"company": "Acme Corp"},
            episodes=[
                _EpisodeResult(
                    episode_index=i, quality=0.5 + 0.1 * i,
                    transcript=f"testee output ep{i}",
                    judge_transcript=f"judge output ep{i}",
                    judge_result={"quality": 0.5 + 0.1 * i,
                                  "criteria": {"completeness": 0.7}},
                    ep_data={"instruction_md": f"task {i}", "fixtures": {}},
                ) for i in range(4)
            ],
        )
        sr.compute_scores()
        result = SandboxEvaluationResult(sr, scenario_name=sr.scenario)

        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "eval_out"
            result.write_artifacts(out)

            # Session-level files
            assert (out / "SKILL.md").read_text() == "# My SKILL"
            assert (out / "JUDGE.md").read_text() == "# JUDGE rubric"
            meta = json.loads((out / "metadata.json").read_text())
            assert meta["scenario"] == "incident_response"
            assert meta["pack_hash"] == "abc123"
            assert len(meta["episode_qualities"]) == 4

            world = json.loads((out / "world.json").read_text())
            assert world["company"] == "Acme Corp"

            # Per-episode files
            for i in range(4):
                ep_dir = out / "episodes" / f"episode_{i}"
                assert (ep_dir / "testee_transcript.txt").read_text() == \
                    f"testee output ep{i}"
                assert (ep_dir / "judge_transcript.txt").read_text() == \
                    f"judge output ep{i}"
                eval_json = json.loads((ep_dir / "evaluation.json").read_text())
                assert eval_json["quality"] == pytest.approx(0.5 + 0.1 * i)

    def test_split_half_delta(self):
        """Verify split-half delta math including anti-sandbagging."""
        from trajectoryrl.utils.sandbox_harness import _SessionResult, _EpisodeResult

        # Normal improvement: delta gives a bonus
        s = _SessionResult(episodes=[
            _EpisodeResult(0, quality=0.4),
            _EpisodeResult(1, quality=0.5),
            _EpisodeResult(2, quality=0.7),
            _EpisodeResult(3, quality=0.8),
        ])
        s.compute_scores()
        assert s.early_mean == pytest.approx(0.45)
        assert s.late_mean == pytest.approx(0.75)
        assert s.delta == pytest.approx(0.30)
        assert s.final_score == pytest.approx(0.6 * 1.15)

        # Anti-sandbagging: low early + big jump → delta zeroed
        s_sandbag = _SessionResult(episodes=[
            _EpisodeResult(0, quality=0.1),
            _EpisodeResult(1, quality=0.1),
            _EpisodeResult(2, quality=0.8),
            _EpisodeResult(3, quality=0.9),
        ])
        s_sandbag.compute_scores()
        assert s_sandbag.delta == 0.0
        assert s_sandbag.final_score == pytest.approx(0.475)

        # Decline: negative delta → no bonus (max(0, delta))
        s_decline = _SessionResult(episodes=[
            _EpisodeResult(0, quality=0.8),
            _EpisodeResult(1, quality=0.9),
            _EpisodeResult(2, quality=0.4),
            _EpisodeResult(3, quality=0.3),
        ])
        s_decline.compute_scores()
        assert s_decline.final_score == pytest.approx(0.6)


# ---------------------------------------------------------------------------
# Docker-only tests (no LLM) — require Docker daemon, still fast
# ---------------------------------------------------------------------------

class TestS1SandboxCli:
    """Sandbox CLI commands (no agent, no LLM)."""

    def test_scenarios_query(self):
        """Sandbox image lists scenarios and version."""
        _skip_if_no_docker()
        client = docker.from_env()
        output = client.containers.run(
            SANDBOX_IMAGE,
            command=["python", "-m", "trajrl_bench.cli", "scenarios"],
            entrypoint="", remove=True, stdout=True, stderr=True,
        )
        data = json.loads(output.decode())
        assert "version" in data
        assert "scenarios" in data
        assert "incident_response" in data["scenarios"]
        logger.info("Sandbox version: %s, scenarios: %s",
                    data["version"], data["scenarios"])

    def test_fixture_generation(self):
        """Generate fixtures for 4 episodes, verify structure."""
        _skip_if_no_docker()
        client = docker.from_env()
        output = client.containers.run(
            SANDBOX_IMAGE,
            command=["python", "-m", "trajrl_bench.cli", "generate",
                     "--seed", "42", "--salt", "test", "--episodes", "4"],
            entrypoint="", remove=True, stdout=True, stderr=True,
        )
        data = json.loads(output.decode())
        assert data["scenario"] in ("incident_response", "morning_brief")
        assert data["world"]["company"]
        assert len(data["episodes"]) == 4
        for ep in data["episodes"]:
            assert len(ep["instruction_md"]) > 50
            assert "inbox" in ep["fixtures"]

    def test_judge_md_fetch(self):
        """Sandbox image serves JUDGE.md per scenario."""
        _skip_if_no_docker()
        client = docker.from_env()
        output = client.containers.run(
            SANDBOX_IMAGE,
            command=["python", "-m", "trajrl_bench.cli", "judge",
                     "--scenario", "incident_response"],
            entrypoint="", remove=True, stdout=True, stderr=True,
        )
        content = output.decode()
        assert content.startswith("#")
        assert "evaluation.json" in content
        assert "quality" in content
        logger.info("JUDGE.md length: %d chars", len(content))


class TestS1SSHSetup:
    """Sandbox + harness SSH connectivity (no LLM)."""

    def test_sandbox_ssh_and_services(self):
        """Start sandbox, verify mock services + SSH key setup."""
        _skip_if_no_docker()
        from trajectoryrl.utils.sandbox_harness import _generate_keypair, _put_file

        client = docker.from_env()
        _, public_key = _generate_keypair()

        network = None
        sandbox = None
        try:
            network = client.networks.create(
                "test_ssh_e2e", driver="bridge", internal=True)

            sandbox = client.containers.run(
                SANDBOX_IMAGE,
                name="test_ssh_sandbox",
                detach=True, network=network.name,
                networking_config={
                    network.name: client.api.create_endpoint_config(
                        aliases=["sandbox"]),
                },
                environment={"SSH_PUBLIC_KEY": public_key, "SSH_USER": "agent"},
                mem_limit="2g",
            )

            # Wait for health
            healthy = False
            for attempt in range(30):
                try:
                    code, out = sandbox.exec_run(
                        ["sh", "-c", "curl -s http://localhost:8090/health"])
                    if code == 0 and b'"ok"' in out:
                        healthy = True
                        break
                except Exception:
                    pass
                time.sleep(0.5)
            assert healthy, "Mock services failed to start"

            # Verify SSH daemon is running
            code, out = sandbox.exec_run(["pgrep", "-f", "sshd"])
            assert code == 0, f"sshd not running: {out.decode()}"

            # Verify authorized_keys is set up
            code, out = sandbox.exec_run(
                ["cat", "/home/agent/.ssh/authorized_keys"])
            assert code == 0
            assert public_key.strip() in out.decode()

            # Write SKILL.md and verify permissions
            _put_file(sandbox, "/workspace/SKILL.md", "# test skill")
            sandbox.exec_run(["chown", "root:agent", "/workspace/SKILL.md"])
            sandbox.exec_run(["chmod", "440", "/workspace/SKILL.md"])
            code, out = sandbox.exec_run(
                ["stat", "-c", "%U:%G %a", "/workspace/SKILL.md"])
            assert b"root:agent 440" in out

        finally:
            if sandbox:
                try:
                    sandbox.stop(timeout=3)
                    sandbox.remove(force=True)
                except Exception:
                    pass
            if network:
                try:
                    network.remove()
                except Exception:
                    pass


# ---------------------------------------------------------------------------
# Full e2e (requires LLM API calls, slow)
# ---------------------------------------------------------------------------

class TestS1Full:
    """Full S1 flow: real sandbox + testee + judge."""

    @pytest.mark.asyncio
    async def test_single_episode_ssh(self):
        """One-episode eval: testee SSHes in, solves task, judge scores it."""
        _skip_if_no_docker()
        _skip_if_no_api_key()

        from trajectoryrl.utils.sandbox_harness import TrajectorySandboxHarness
        config = _make_config(num_episodes=1, timeout=300)
        harness = TrajectorySandboxHarness(config)

        t0 = time.time()
        result = await harness.evaluate_miner(
            skill_md=SKILLED_SKILL_MD,
            epoch_seed=42,
            pack_hash="e2e_single",
            validator_salt="e2e_salt",
        )
        elapsed = time.time() - t0

        logger.info("=== SINGLE EPISODE RESULT ===")
        logger.info("Score: %.4f", result.score)
        logger.info("Mean quality: %.4f", result.mean_quality)
        logger.info("Episode qualities: %s", result.episode_qualities)
        logger.info("Scenario: %s", result.scenario_name)
        logger.info("Error: %s", result.error)
        logger.info("Elapsed: %.1fs", elapsed)

        if result.session_result.episodes:
            ep = result.session_result.episodes[0]
            if ep.judge_result:
                logger.info("Judge criteria: %s",
                            ep.judge_result.get("criteria", {}))
                logger.info("Judge summary: %s",
                            ep.judge_result.get("summary", "")[:500])

        assert result.error is None, f"Evaluation failed: {result.error}"
        assert len(result.episode_qualities) == 1
        assert result.scenario_name in ("incident_response", "morning_brief")
        # A skilled SKILL.md should score non-zero
        assert result.score > 0.0, \
            f"Expected non-zero score, got {result.score}. Episodes: {result.episode_qualities}"

    @pytest.mark.asyncio
    async def test_four_episode_full(self):
        """Full 4-episode eval with split-half delta."""
        _skip_if_no_docker()
        _skip_if_no_api_key()

        from trajectoryrl.utils.sandbox_harness import TrajectorySandboxHarness
        config = _make_config(num_episodes=4, timeout=300)
        harness = TrajectorySandboxHarness(config)

        t0 = time.time()
        result = await harness.evaluate_miner(
            skill_md=SKILLED_SKILL_MD,
            epoch_seed=12345,
            pack_hash="e2e_full",
            validator_salt="e2e_salt",
        )
        elapsed = time.time() - t0

        logger.info("=== FULL 4-EPISODE RESULT ===")
        logger.info("Final score: %.4f", result.score)
        logger.info("Mean quality: %.4f", result.mean_quality)
        logger.info("Episodes: %s", result.episode_qualities)
        logger.info("Delta: %.4f (learning bonus: %.4f)",
                    result.delta, result.learning_bonus)
        logger.info("Early mean: %.4f, Late mean: %.4f",
                    result.early_mean, result.late_mean)
        logger.info("Scenario: %s", result.scenario_name)
        logger.info("Elapsed: %.1fs (%.1f min)", elapsed, elapsed / 60)

        assert result.error is None
        assert len(result.episode_qualities) == 4
        for q in result.episode_qualities:
            assert 0.0 <= q <= 1.0
        assert result.score > 0.0
