"""Guards for the per-spec scenario registry (dynamic-spec selection).

The validator resolves which scenario set to eval from the server-provided
``epoch.spec_number`` (GET /epoch/current, driven by the web spec_schedule),
looking it up in ``SPEC_SCENARIOS``. These tests pin that registry to a
checked-in snapshot so it can't silently drift from the web's
``SCORING_BY_SPEC`` (trajectoryrl.web/src/lib/scenario-spec.ts) — a divergence
would make validators eval a different set than the web assumes.
"""
from trajectoryrl.utils import config
from trajectoryrl.utils.sandbox_harness import (
    SPEC_SCENARIOS,
    SANDBOX_SCENARIOS,
    LATEST_SPEC,
    scenarios_for_spec,
    _all_known_scenarios,
)

# MUST match the web's SCORING_BY_SPEC[*].scenarios exactly.
EXPECTED_SPEC_SCENARIOS = {
    17: (
        "configure-git-webserver",
        "db-wal-recovery",
        "git-leak-recovery",
        "kv-store-grpc",
        "largest-eigenval",
        "nginx-request-logging",
        "path-tracing",
        "race-condition-fix",
        "regex-chess",
        "swe-bench-astropy-2",
        "write-compressor",
    ),
    18: (
        "configure-git-webserver",
        "db-wal-recovery",
        "git-leak-recovery",
        "git-multibranch",
        "largest-eigenval",
        "nginx-request-logging",
        "path-tracing",
        "race-condition-fix",
        "regex-chess",
        "schemelike-metacircular-eval",
        "swe-bench-astropy-2",
        "write-compressor",
    ),
}


def test_registry_matches_snapshot():
    """Sync guard vs. the web SCORING_BY_SPEC snapshot."""
    assert SPEC_SCENARIOS == EXPECTED_SPEC_SCENARIOS


def test_spec_number_is_max_known():
    """config.SPEC_NUMBER is the build's max-known spec (fallback when the
    server omits epoch.spec_number). It cannot import the registry, so this
    test enforces the invariant instead."""
    assert config.SPEC_NUMBER == LATEST_SPEC == max(SPEC_SCENARIOS)


def test_scenarios_for_spec_resolves_known():
    assert scenarios_for_spec(17) == EXPECTED_SPEC_SCENARIOS[17]
    assert scenarios_for_spec(18) == EXPECTED_SPEC_SCENARIOS[18]


def test_scenarios_for_spec_unknown_returns_none():
    # Server ahead of this build → eval loop abstains (no submission).
    assert scenarios_for_spec(999) is None


def test_sandbox_scenarios_alias_is_latest():
    assert SANDBOX_SCENARIOS == SPEC_SCENARIOS[LATEST_SPEC]


def test_each_spec_set_is_sorted_and_unique():
    for spec, names in SPEC_SCENARIOS.items():
        assert list(names) == sorted(names), f"spec {spec} not alphabetically sorted"
        assert len(set(names)) == len(names), f"spec {spec} has duplicate scenarios"


def test_all_known_scenarios_is_sorted_union():
    union = set()
    for names in SPEC_SCENARIOS.values():
        union |= set(names)
    assert list(_all_known_scenarios()) == sorted(union)
