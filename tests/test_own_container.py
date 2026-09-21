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
