"""Own-container resolution must survive a Watchtower recreate (stale hostname)."""
import types
import pytest
from trajectoryrl.utils.sandbox_harness import TrajectorySandboxHarness


class _Ctr:
    def __init__(self, cid, name, hostname):
        self.id, self.name = cid, name
        self.attrs = {"Config": {"Hostname": hostname}}


class _Client:
    """get() only knows real ids; list() returns the live set."""
    def __init__(self, live):
        self._live = live
        self.containers = types.SimpleNamespace(get=self._get, list=lambda: list(self._live))

    def _get(self, ident):
        for c in self._live:
            if c.id.startswith(ident) or c.name == ident:
                return c
        raise RuntimeError(f"404 no such container: {ident}")


def _harness(client, hostname, proc_id, monkeypatch, tmp_path):
    h = TrajectorySandboxHarness.__new__(TrajectorySandboxHarness)
    h._docker_client = client
    h._self_container_checked = False
    h._self_container = None
    monkeypatch.setattr("os.path.exists", lambda p: True if p == "/.dockerenv" else False)
    monkeypatch.setattr("socket.gethostname", lambda: hostname)
    monkeypatch.setattr(TrajectorySandboxHarness, "_own_container_id_from_proc", staticmethod(lambda: proc_id))
    return h


REAL = "a7b993ebfb1c" + "0" * 52
STALE = "438b39003e95"


def test_resolves_by_hostname_when_fresh(monkeypatch, tmp_path):
    c = _Ctr(REAL, "trajectoryrl_validator", REAL[:12])
    h = _harness(_Client([c]), REAL[:12], None, monkeypatch, tmp_path)
    assert h._own_container() is c


def test_resolves_by_proc_id_when_hostname_is_stale(monkeypatch, tmp_path):
    c = _Ctr(REAL, "trajectoryrl_validator", STALE)
    h = _harness(_Client([c]), STALE, REAL, monkeypatch, tmp_path)
    assert h._own_container() is c


def test_resolves_by_hostname_scan_when_proc_unavailable(monkeypatch, tmp_path):
    """The uid-74 case: Watchtower recreate, no usable /proc id."""
    c = _Ctr(REAL, "trajectoryrl_validator", STALE)
    h = _harness(_Client([c]), STALE, None, monkeypatch, tmp_path)
    assert h._own_container() is c


def test_returns_none_when_truly_absent(monkeypatch, tmp_path):
    h = _harness(_Client([]), STALE, None, monkeypatch, tmp_path)
    assert h._own_container() is None


def test_result_is_cached(monkeypatch, tmp_path):
    c = _Ctr(REAL, "trajectoryrl_validator", STALE)
    h = _harness(_Client([c]), STALE, None, monkeypatch, tmp_path)
    assert h._own_container() is c
    h._docker_client = _Client([])   # a second call must not re-resolve
    assert h._own_container() is c


def test_in_docker_without_own_container_is_fatal(monkeypatch):
    """The gateway fallback is host-process-only; inside docker it must raise.

    Regression guard for SN11 uid 74: the sidecar came up pointed at the
    episode network's gateway, every model call failed, and the session scored
    baseline credit at $0 while looking healthy.
    """
    import trajectoryrl.utils.sandbox_harness as sh

    h = TrajectorySandboxHarness.__new__(TrajectorySandboxHarness)
    h._self_container_checked = True
    h._self_container = None                      # resolution already failed
    h._meter = types.SimpleNamespace(port=8790)
    h._meter_lock = __import__("threading").Lock()
    monkeypatch.setattr(sh.os.path, "exists", lambda p: p == "/.dockerenv")

    net = types.SimpleNamespace(
        name="pnet_x", attrs={"IPAM": {"Config": [{"Gateway": "172.21.0.1"}]}},
    )
    with pytest.raises(RuntimeError, match="no route to the meter"):
        h._start_policy_sidecar("sess", "scen", object(), "tok", {}, net)


def test_all_infra_errors_count_as_provider_failure():
    """A session where every episode failed for OUR reason must be discarded.

    Those episodes never ran hermes, so chat_exit is None and the hermes-side
    test reads "unknown". Without this branch the session would be POSTed as a
    zero score and blame the miner for a broken validator.
    """
    from trajectoryrl.utils.sandbox_harness import (
        _looks_like_provider_failure, INFRA_ERROR_PREFIX,
    )

    def ep(error=None, chat_exit=None, cost=None, timed_out=False):
        return types.SimpleNamespace(
            error=error, chat_exit=chat_exit, cost_usd=cost, timed_out=timed_out,
        )

    infra = [ep(error=INFRA_ERROR_PREFIX + "no route to the meter") for _ in range(26)]
    assert _looks_like_provider_failure(infra) is True

    # A miner-side sidecar crash is NOT ours: it must still score the episode.
    miner = [ep(error="policy did not answer /v1/models") for _ in range(26)]
    assert _looks_like_provider_failure(miner) is False

    # One billed episode proves the provider answered: never an infra discard.
    mixed = infra[:-1] + [ep(cost=0.42, chat_exit=0)]
    assert _looks_like_provider_failure(mixed) is False
