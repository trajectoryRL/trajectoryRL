"""Tests for the Season 1 trajrl-bench harness adapter.

Tests the result mapping, config wiring, and SKILL.md extraction
without requiring Docker or real LLM API calls.
"""

import threading
import time

import pytest

from trajectoryrl.utils.config import ValidatorConfig
from trajectoryrl.utils.sandbox_harness import (
    SandboxEvaluationResult,
    TrajectorySandboxHarness,
    _drain_exec_stream_with_deadline,
    _EpisodeResult,
    _parse_ctrf_correctness,
    _parse_session_cost,
    _SessionResult,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_session_result(
    cells: list[tuple[str, float]] | list[float],
) -> _SessionResult:
    """Build a session result. Accepts either ``[(scenario, q), ...]``
    or a flat list of qualities (auto-tagged with ``scenario_<i>``)."""
    eps: list[_EpisodeResult] = []
    for i, item in enumerate(cells):
        if isinstance(item, tuple):
            scenario, q = item
        else:
            scenario, q = f"scenario_{i}", item
        eps.append(_EpisodeResult(episode_index=i, scenario=scenario, quality=q))
    result = _SessionResult(episodes=eps)
    result.compute_scores()
    return result


# ---------------------------------------------------------------------------
# Tests: SandboxEvaluationResult mapping
# ---------------------------------------------------------------------------

class TestSandboxEvaluationResult:
    def test_maps_final_score_sums_per_scenario(self):
        # Three scenarios, equal weight per scenario → final = sum.
        sr = _make_session_result([
            ("cancel-async-tasks",        0.833),
            ("log-summary-date-ranges",   1.0),
            ("break-filter-js-from-html", 0.5),
        ])
        result = SandboxEvaluationResult(sr)
        assert abs(result.score - (0.833 + 1.0 + 0.5)) < 1e-6
        assert result.success is True

    def test_zero_score_not_qualified(self):
        sr = _make_session_result([0.0, 0.0, 0.0])
        result = SandboxEvaluationResult(sr)
        assert result.score == 0.0
        assert result.success is False

    def test_scenario_qualities_exposed(self):
        sr = _make_session_result([
            ("a", 0.5), ("b", 0.6), ("c", 0.7),
        ])
        result = SandboxEvaluationResult(sr)
        assert result.scenario_qualities == {"a": 0.5, "b": 0.6, "c": 0.7}
        assert result.scenarios == ["a", "b", "c"]

    def test_mean_quality_in_unit_range(self):
        sr = _make_session_result([
            ("a", 0.4), ("b", 0.4), ("c", 1.0),
        ])
        result = SandboxEvaluationResult(sr)
        assert abs(result.mean_quality - 0.6) < 1e-9
        # final_score is a *sum* (range [0, N]); mean_quality is the
        # convenience [0, 1] aggregate.
        assert abs(result.score - 1.8) < 1e-9

    def test_error_field_default_none(self):
        sr = _make_session_result([0.5, 0.5, 0.5])
        result = SandboxEvaluationResult(sr)
        assert result.error is None


# ---------------------------------------------------------------------------
# Tests: per-scenario session scoring
# ---------------------------------------------------------------------------

class TestSessionResultScoring:
    def test_sum_across_scenarios(self):
        sr = _make_session_result([("a", 0.5), ("b", 0.833), ("c", 1.0)])
        # final_score = sum of correctness ratios (no learning bonus,
        # equal weight per scenario).
        assert abs(sr.final_score - 2.333) < 1e-3
        assert abs(sr.mean_quality - 0.7777) < 1e-3

    def test_no_episodes(self):
        sr = _SessionResult()
        sr.compute_scores()
        assert sr.final_score == 0.0
        assert sr.mean_quality == 0.0

    def test_perfect_session(self):
        sr = _make_session_result([("a", 1.0), ("b", 1.0), ("c", 1.0)])
        assert sr.final_score == 3.0
        assert sr.mean_quality == 1.0

    def test_partial_credit_visible(self):
        # The whole point of moving off binary scoring: a pack that
        # passes 5 of 6 tests on one scenario gets credit instead of 0.
        sr = _make_session_result([("scenario_a", 5 / 6)])
        assert abs(sr.final_score - 5 / 6) < 1e-9
        assert abs(sr.mean_quality - 5 / 6) < 1e-9


# ---------------------------------------------------------------------------
# Tests: Config wiring
# ---------------------------------------------------------------------------

class TestConfigWiring:
    def test_config_fields_exist(self):
        assert hasattr(ValidatorConfig, "image_channel")
        assert hasattr(ValidatorConfig, "sandbox_image")
        assert hasattr(ValidatorConfig, "sandbox_timeout_per_episode")

    def test_dropped_legacy_fields(self):
        # ``sandbox_scenario`` and ``sandbox_num_episodes`` were removed
        # 2026-05-04 — the harness now hardcodes the scenario set.
        assert not hasattr(ValidatorConfig, "sandbox_scenario")
        assert not hasattr(ValidatorConfig, "sandbox_num_episodes")

    def test_default_timeout(self):
        assert ValidatorConfig.sandbox_timeout_per_episode == 600


# ---------------------------------------------------------------------------
# Tests: image channel resolution
# ---------------------------------------------------------------------------

def _config(tmp_path, **overrides) -> ValidatorConfig:
    """Build a ValidatorConfig pointed at a tmp dir so the
    ``__post_init__`` mkdir doesn't try to create
    ``/var/lib/trajectoryrl`` on the test runner."""
    return ValidatorConfig(
        pack_cache_dir=tmp_path / "packs",
        log_dir=tmp_path / "logs",
        eval_state_path=tmp_path / "eval_state.json",
        winner_state_path=tmp_path / "winner_state.json",
        pack_first_seen_path=tmp_path / "pack_first_seen.json",
        active_set_dir=tmp_path / "active_sets",
        **overrides,
    )


class TestImageChannel:
    def test_default_channel_derives_latest(self, tmp_path):
        cfg = _config(tmp_path)
        assert cfg.image_channel == "latest"
        assert cfg.sandbox_image == "ghcr.io/trajectoryrl/sandbox-agent:latest"

    def test_staging_channel_derives_staging_tag(self, tmp_path):
        cfg = _config(tmp_path, image_channel="staging")
        assert cfg.sandbox_image == "ghcr.io/trajectoryrl/sandbox-agent:staging"

    def test_arbitrary_channel_derives_matching_tag(self, tmp_path):
        cfg = _config(tmp_path, image_channel="v1.2.0-rc.1")
        assert cfg.sandbox_image == "ghcr.io/trajectoryrl/sandbox-agent:v1.2.0-rc.1"

    def test_explicit_sandbox_image_overrides_channel(self, tmp_path):
        cfg = _config(
            tmp_path,
            image_channel="staging",
            sandbox_image="ghcr.io/custom/sandbox-agent:debug",
        )
        assert cfg.sandbox_image == "ghcr.io/custom/sandbox-agent:debug"


# ---------------------------------------------------------------------------
# Tests: ctrf-derived continuous correctness
# ---------------------------------------------------------------------------

class TestParseCtrfCorrectness:
    def test_partial_pass(self):
        ctrf = {"results": {"summary": {"tests": 6, "passed": 5, "failed": 1}}}
        assert _parse_ctrf_correctness(ctrf) == (5, 6)

    def test_all_pass(self):
        ctrf = {"results": {"summary": {"tests": 4, "passed": 4, "failed": 0}}}
        assert _parse_ctrf_correctness(ctrf) == (4, 4)

    def test_all_fail(self):
        ctrf = {"results": {"summary": {"tests": 3, "passed": 0, "failed": 3}}}
        assert _parse_ctrf_correctness(ctrf) == (0, 3)

    def test_missing_payload(self):
        # None / non-dict / missing summary all return (0, 0) — caller
        # uses that as the "fall back to binary reward" signal.
        assert _parse_ctrf_correctness(None) == (0, 0)
        assert _parse_ctrf_correctness({}) == (0, 0)
        assert _parse_ctrf_correctness({"results": {}}) == (0, 0)
        assert _parse_ctrf_correctness({"results": {"summary": {"tests": 0}}}) == (0, 0)

    def test_malformed_counts(self):
        # Hand-rolled non-int counts shouldn't crash, just yield no signal.
        ctrf = {"results": {"summary": {"tests": "six", "passed": 5}}}
        assert _parse_ctrf_correctness(ctrf) == (0, 0)


# ---------------------------------------------------------------------------
# Tests: cost extraction from turns.jsonl
# ---------------------------------------------------------------------------

class TestParseSessionCost:
    def test_picks_actual_over_estimated(self):
        import json as _json
        line = _json.dumps({
            "id": "abc", "actual_cost_usd": 0.0184, "estimated_cost_usd": 0.02,
        })
        assert _parse_session_cost(line + "\n") == 0.0184

    def test_falls_back_to_estimated(self):
        import json as _json
        line = _json.dumps({"id": "abc", "actual_cost_usd": None,
                            "estimated_cost_usd": 0.012})
        assert _parse_session_cost(line) == 0.012

    def test_takes_last_session_when_multiple(self):
        # Defensive: validator wipes state.db between episodes so this
        # shouldn't happen, but if cumulative export ever leaks through
        # we still pick the most recent.
        import json as _json
        first = _json.dumps({"id": "old", "actual_cost_usd": 0.10})
        last = _json.dumps({"id": "new", "actual_cost_usd": 0.02})
        assert _parse_session_cost(first + "\n" + last + "\n") == 0.02

    def test_empty_blob(self):
        assert _parse_session_cost("") is None
        assert _parse_session_cost("\n\n") is None

    def test_unparseable_line(self):
        assert _parse_session_cost("not json\n") is None

    def test_negative_cost_ignored(self):
        import json as _json
        line = _json.dumps({"id": "x", "actual_cost_usd": -0.05})
        assert _parse_session_cost(line) is None

    def test_missing_cost_field(self):
        import json as _json
        line = _json.dumps({"id": "x", "messages": []})
        assert _parse_session_cost(line) is None


# ---------------------------------------------------------------------------
# Tests: SandboxEvaluationResult cost aggregation
# ---------------------------------------------------------------------------

class TestWriteArtifacts:
    def test_ctrf_and_verifier_stdout_split_out(self, tmp_path):
        sr = _SessionResult(episodes=[
            _EpisodeResult(
                episode_index=0,
                quality=0.833,
                cost_usd=0.005,
                judge_result={
                    "reward": 0,
                    "passed": 5,
                    "total": 6,
                    "correctness": 0.833,
                    "cost_usd": 0.005,
                    "verifier_stdout": "pytest output here\n",
                    "ctrf": {"results": {"summary": {"tests": 6, "passed": 5}}},
                },
            ),
        ])
        sr.compute_scores()
        result = SandboxEvaluationResult(sr, scenario_name="cancel-async-tasks")
        out = tmp_path / "artifacts"
        result.write_artifacts(out)

        ep0 = out / "episodes" / "episode_0"
        # ctrf is its own file
        assert (ep0 / "ctrf.json").exists()
        import json as _json
        ctrf = _json.loads((ep0 / "ctrf.json").read_text())
        assert ctrf["results"]["summary"]["passed"] == 5
        # verifier_stdout is also broken out
        assert (ep0 / "verifier_stdout.txt").read_text() == "pytest output here\n"
        # evaluation.json still carries the full blob for back-compat
        eval_blob = _json.loads((ep0 / "evaluation.json").read_text())
        assert eval_blob["correctness"] == 0.833
        assert eval_blob["ctrf"]["results"]["summary"]["passed"] == 5

    def test_skips_ctrf_when_absent(self, tmp_path):
        # Episode where the verifier never wrote ctrf (e.g. timeout).
        sr = _SessionResult(episodes=[
            _EpisodeResult(
                episode_index=0,
                quality=0.0,
                judge_result={"reward": 0, "passed": 0, "total": 0,
                              "verifier_stdout": "", "ctrf": None},
            ),
        ])
        sr.compute_scores()
        result = SandboxEvaluationResult(sr)
        out = tmp_path / "artifacts"
        result.write_artifacts(out)

        ep0 = out / "episodes" / "episode_0"
        assert not (ep0 / "ctrf.json").exists()
        assert not (ep0 / "verifier_stdout.txt").exists()


class TestEvalResultCostAggregation:
    def test_sums_billed_scenarios(self):
        sr = _SessionResult(episodes=[
            _EpisodeResult(episode_index=0, scenario="a", quality=0.5, cost_usd=0.01),
            _EpisodeResult(episode_index=1, scenario="b", quality=0.5, cost_usd=0.02),
        ])
        sr.compute_scores()
        result = SandboxEvaluationResult(sr)
        assert result.total_cost_usd == pytest.approx(0.03)
        assert result.mean_cost_usd == pytest.approx(0.015)
        assert result.scenario_costs_usd == {"a": 0.01, "b": 0.02}

    def test_partial_billing(self):
        # One scenario unbilled (e.g. Hermes export failed) — aggregate
        # over the billed ones, don't fabricate zeros.
        sr = _SessionResult(episodes=[
            _EpisodeResult(episode_index=0, scenario="a", quality=0.5, cost_usd=0.04),
            _EpisodeResult(episode_index=1, scenario="b", quality=0.5, cost_usd=None),
        ])
        sr.compute_scores()
        result = SandboxEvaluationResult(sr)
        assert result.total_cost_usd == pytest.approx(0.04)
        assert result.mean_cost_usd == pytest.approx(0.04)

    def test_no_billing(self):
        sr = _SessionResult(episodes=[
            _EpisodeResult(episode_index=0, scenario="a", quality=0.5),
            _EpisodeResult(episode_index=1, scenario="b", quality=0.5),
        ])
        sr.compute_scores()
        result = SandboxEvaluationResult(sr)
        assert result.total_cost_usd is None
        assert result.mean_cost_usd is None


# ---------------------------------------------------------------------------
# Tests: extract_skill_md
# ---------------------------------------------------------------------------

class TestExtractSkillMd:
    def test_extracts_from_files(self):
        pack = {"files": {"SKILL.md": "# My Skill\nDo stuff."}}
        assert TrajectorySandboxHarness.extract_skill_md(pack) == "# My Skill\nDo stuff."

    def test_lowercase_fallback(self):
        pack = {"files": {"skill.md": "# lowercase"}}
        assert TrajectorySandboxHarness.extract_skill_md(pack) == "# lowercase"

    def test_returns_none_for_v4_pack(self):
        pack = {"files": {"AGENTS.md": "# Agent config"}}
        assert TrajectorySandboxHarness.extract_skill_md(pack) is None

    def test_returns_none_for_empty_files(self):
        pack = {"files": {}}
        assert TrajectorySandboxHarness.extract_skill_md(pack) is None

    def test_returns_none_for_no_files_key(self):
        pack = {"agents": []}
        assert TrajectorySandboxHarness.extract_skill_md(pack) is None


# ---------------------------------------------------------------------------
# Tests: shell_verifier scenario plumbing
# ---------------------------------------------------------------------------

def _make_harness(sandbox_image: str = "", tmp_path=None) -> TrajectorySandboxHarness:
    """Build a harness without going near docker.from_env().

    The helpers we test only touch ``self.config`` + cached attribute
    state; we never call ``self.client``.
    """
    import tempfile
    from pathlib import Path

    if tmp_path is None:
        tmp_path = Path(tempfile.mkdtemp())
    cfg = ValidatorConfig(
        pack_cache_dir=tmp_path / "packs",
        log_dir=tmp_path / "logs",
        eval_state_path=tmp_path / "eval_state.json",
        winner_state_path=tmp_path / "winner_state.json",
        pack_first_seen_path=tmp_path / "pack_first_seen.json",
        active_set_dir=tmp_path / "active_sets",
    )
    if sandbox_image:
        cfg.sandbox_image = sandbox_image
    return TrajectorySandboxHarness(cfg)


class TestSandboxTag:
    def test_named_tag(self):
        h = _make_harness(sandbox_image="ghcr.io/trajectoryrl/sandbox-agent:v4.0.0")
        assert h._sandbox_tag() == "v4.0.0"

    def test_latest_tag(self):
        h = _make_harness(sandbox_image="ghcr.io/trajectoryrl/sandbox-agent:latest")
        assert h._sandbox_tag() == "latest"

    def test_no_tag_falls_back_to_latest(self):
        h = _make_harness(sandbox_image="ghcr.io/trajectoryrl/sandbox-agent")
        assert h._sandbox_tag() == "latest"

    def test_digest_pin_strips_to_latest(self):
        h = _make_harness(
            sandbox_image="ghcr.io/trajectoryrl/sandbox-agent@sha256:abc123",
        )
        assert h._sandbox_tag() == "latest"


class TestScenarioImageRef:
    def test_returns_empty_when_no_info_loaded(self):
        h = _make_harness()
        assert h._scenario_image_ref() == ""

    def test_combines_repo_with_sandbox_tag(self):
        h = _make_harness(sandbox_image="ghcr.io/trajectoryrl/sandbox-agent:v4.0.0")
        h._scenario_info = {
            "name": "cancel-async-tasks",
            "image_repo": "ghcr.io/trajectoryrl/scenario-cancel-async-tasks",
        }
        assert h._scenario_image_ref() == (
            "ghcr.io/trajectoryrl/scenario-cancel-async-tasks:v4.0.0"
        )

    def test_returns_empty_when_image_repo_missing(self):
        h = _make_harness()
        h._scenario_info = {"name": "cancel-async-tasks"}
        assert h._scenario_image_ref() == ""


# ---------------------------------------------------------------------------
# Regression tests for the exec-stream timeout helper.
#
# These pin down the contract that ``_drain_exec_stream_with_deadline``
# returns within ``timeout + slack`` even when the stream iterator blocks
# silently — the wild-caught case being a Hermes process parked on a
# stuck LLM API call that emits no stdout. The naive
# ``for chunk in stream: ... if deadline: break`` pattern parks in
# ``next()`` and never re-checks the deadline.
# ---------------------------------------------------------------------------

class TestDrainExecStreamWithDeadline:

    def test_returns_all_chunks_when_iterator_completes(self):
        def stream():
            yield b"hello "
            yield b"world"

        chunks, timed_out = _drain_exec_stream_with_deadline(
            stream(), timeout=5.0,
        )

        assert b"".join(chunks) == b"hello world"
        assert timed_out is False

    @pytest.mark.timeout(3)
    def test_fires_deadline_when_iterator_blocks_silently(self):
        """An iterator that emits nothing and never returns — Hermes
        blocked on a stalled API call — must still trip the deadline
        within ``timeout + small slack``. The pre-fix inline code
        deadlocks here because the deadline check sat inside the
        ``for chunk in stream`` body and never re-runs without a
        chunk to wake it."""
        never_ends = threading.Event()

        def stream():
            never_ends.wait()  # blocks forever
            yield b"won't get here"  # pragma: no cover

        try:
            start = time.monotonic()
            chunks, timed_out = _drain_exec_stream_with_deadline(
                stream(), timeout=0.5,
            )
            elapsed = time.monotonic() - start

            assert timed_out is True
            assert chunks == []
            # Slack budget for thread wake-up jitter on CI. Anything
            # past this strongly suggests an in-loop deadline check
            # reintroduced the deadlock.
            assert elapsed < 1.5, (
                f"deadline took {elapsed:.2f}s; deadlock likely reintroduced"
            )
        finally:
            never_ends.set()

    def test_zero_timeout_returns_immediately_as_timed_out(self):
        """A 0-timeout call (or one past deadline at entry) must
        return ``timed_out=True`` even if the iterator is empty —
        the pre-fix in-loop check never executes the body on an
        empty iterator and so leaves ``timed_out`` at its False
        default. Pins the deadline-check-runs-first contract."""
        chunks, timed_out = _drain_exec_stream_with_deadline(
            iter([]), timeout=0,
        )
        assert chunks == []
        assert timed_out is True

    @pytest.mark.timeout(3)
    def test_collects_chunks_before_deadline_when_iterator_then_blocks(self):
        emitted_one = threading.Event()
        never_ends = threading.Event()

        def stream():
            yield b"partial-output"
            emitted_one.set()
            never_ends.wait()
            yield b"never"  # pragma: no cover

        try:
            chunks, timed_out = _drain_exec_stream_with_deadline(
                stream(), timeout=0.5,
            )

            assert emitted_one.is_set(), (
                "first chunk should have been produced before deadline"
            )
            assert b"".join(chunks) == b"partial-output"
            assert timed_out is True
        finally:
            never_ends.set()

    def test_iterator_exception_is_caught_and_reported(self):
        def stream():
            yield b"head"
            raise ConnectionResetError("docker socket closed")

        chunks, timed_out = _drain_exec_stream_with_deadline(
            stream(), timeout=5.0,
        )

        # Pre-exception chunks are still surfaced; an iterator that
        # ends early via exception is not the same thing as a timeout.
        assert b"".join(chunks) == b"head"
        assert timed_out is False

    @pytest.mark.timeout(3)
    def test_invokes_on_deadline_callback_exactly_once_on_timeout(self):
        """The callback exists so the caller can release the blocked
        exec socket (pkill the in-container process). It must fire
        exactly once at deadline — never zero times, never twice.
        Pre-fix the deadline never fires on a silent iterator so the
        callback is never invoked: this test fails by callback
        invocation count = 0."""
        called: list[float] = []
        never_ends = threading.Event()

        def stream():
            never_ends.wait()
            yield b""  # pragma: no cover

        try:
            _drain_exec_stream_with_deadline(
                stream(),
                timeout=0.3,
                on_deadline=lambda: called.append(time.monotonic()),
            )

            assert len(called) == 1, (
                f"on_deadline must fire exactly once; fired {len(called)}"
            )
        finally:
            never_ends.set()

    def test_does_not_invoke_callback_when_iterator_completes_in_time(self):
        called: list[bool] = []

        def stream():
            yield b"done"

        _drain_exec_stream_with_deadline(
            stream(),
            timeout=5.0,
            on_deadline=lambda: called.append(True),
        )

        assert called == [], "on_deadline must not fire on clean completion"

    @pytest.mark.timeout(3)
    def test_callback_failure_does_not_leak_out(self):
        """The on_deadline callback runs in-process; if it raises
        (e.g. pkill against a torn-down container), the helper must
        swallow the exception so the caller still gets the partial
        transcript and proceeds to teardown. Pre-fix this never
        gets exercised because the callback isn't called; once the
        fix lets the deadline fire, a raising callback would tear
        down the helper without this safety net."""
        never_ends = threading.Event()

        def stream():
            never_ends.wait()
            yield b""  # pragma: no cover

        def boom() -> None:
            raise RuntimeError("docker SDK lost the container")

        try:
            chunks, timed_out = _drain_exec_stream_with_deadline(
                stream(), timeout=0.3, on_deadline=boom,
            )

            assert timed_out is True
            assert chunks == []
        finally:
            never_ends.set()

    @pytest.mark.timeout(3)
    def test_deadline_survives_wall_clock_jumping_backwards(self, monkeypatch):
        """Wall-clock time can jump backwards under NTP correction or
        manual clock adjustment. A deadline computed from ``time.time()``
        becomes unreachable when the clock jumps back, leaving the
        helper waiting on a deadline that never arrives.

        Simulate it by monkeypatching ``time.time`` in the harness
        module to return monotonically *decreasing* values. A correct
        helper anchors its deadline on ``time.monotonic()`` (which
        cannot regress) so the deadline still fires; a helper that
        anchors on ``time.time()`` will park past the configured
        timeout and pytest-timeout will fail this test at 3 s."""
        never_ends = threading.Event()

        def stream():
            never_ends.wait()
            yield b""  # pragma: no cover

        from trajectoryrl.utils import sandbox_harness

        fake_now = [1_000_000.0]

        def receding_clock() -> float:
            # Each call rewinds the wall clock by 0.5 s — much faster
            # than any plausible NTP correction, but the direction is
            # what matters for the test.
            fake_now[0] -= 0.5
            return fake_now[0]

        monkeypatch.setattr(sandbox_harness.time, "time", receding_clock)

        try:
            start = time.monotonic()
            chunks, timed_out = _drain_exec_stream_with_deadline(
                stream(), timeout=0.3,
            )
            elapsed = time.monotonic() - start

            assert timed_out is True
            assert chunks == []
            assert elapsed < 1.5, (
                f"deadline took {elapsed:.2f}s under a receding wall clock; "
                "helper likely still anchored on time.time() instead of "
                "time.monotonic()"
            )
        finally:
            never_ends.set()


# ---------------------------------------------------------------------------
# _pick_episode_timeout — per-scenario agent timeout from task.toml.
# ---------------------------------------------------------------------------

from trajectoryrl.utils.sandbox_harness import _pick_episode_timeout


class TestPickEpisodeTimeout:
    """The scenario-info payload now carries ``agent_timeout_s`` per
    scenario (trajrl-bench PR #42). The harness should prefer that
    value when present and fall back to the global config default
    otherwise — so a path-tracing scenario that pins 900 s gets 900 s
    even when the validator's global is 600 s, and an older
    sandbox-agent image still works.
    """

    def test_uses_scenario_pinned_timeout_when_present(self):
        spec = {
            "name": "path-tracing",
            "agent_timeout_s": 900,
        }
        assert _pick_episode_timeout(spec, config_default=600) == 900.0

    def test_falls_back_to_config_default_when_missing(self):
        spec = {"name": "old-bench-no-agent-field"}
        assert _pick_episode_timeout(spec, config_default=600) == 600.0

    def test_falls_back_to_config_default_when_none(self):
        spec = {"name": "x", "agent_timeout_s": None}
        assert _pick_episode_timeout(spec, config_default=600) == 600.0

    def test_falls_back_to_config_default_when_zero_or_negative(self):
        # A 0 or negative agent_timeout_s would defeat the purpose
        # of having a deadline. Treat it as "not set" and fall back.
        for bad in (0, -1, -900):
            spec = {"name": "x", "agent_timeout_s": bad}
            assert _pick_episode_timeout(spec, config_default=600) == 600.0, (
                f"agent_timeout_s={bad} should have fallen back"
            )

    def test_falls_back_to_config_default_when_nan(self):
        # NaN would propagate into ``time.monotonic() + timeout`` and
        # produce a NaN deadline that no clock reading can clear,
        # giving us back the deadlock we just fixed via a different
        # door. Reject explicitly so a future refactor of the
        # validation predicate cannot accidentally let it through.
        spec = {"name": "x", "agent_timeout_s": float("nan")}
        assert _pick_episode_timeout(spec, config_default=600) == 600.0

    def test_returns_float_for_deadline_arithmetic(self):
        # The caller passes the result directly into
        # ``time.monotonic() + timeout``; a non-float (e.g. an int
        # from JSON) is fine numerically but a returned float is the
        # documented contract.
        spec = {"name": "x", "agent_timeout_s": 123}
        result = _pick_episode_timeout(spec, config_default=600)
        assert isinstance(result, float)
        assert result == 123.0
