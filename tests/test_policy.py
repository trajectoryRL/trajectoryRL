"""Season 2 routing policies: pack extraction, the metering proxy, and the sidecar runtime helpers."""
from __future__ import annotations

import asyncio
import json
import secrets
import socket
import sys
import threading
import time
from unittest.mock import MagicMock

import pytest
from aiohttp import ClientSession, web

# Mock bittensor so importing trajectoryrl.* doesn't pull in the SDK.
sys.modules.setdefault("bittensor", MagicMock())

from trajectoryrl.policy import (  # noqa: E402
    MODEL_ALLOWLIST, POLICY_FILES_MAX_BYTES, RUNTIME_FILE, extract_policy_files,
)
from trajectoryrl.policy.meter import PolicyMeter, cost_of  # noqa: E402


def _free_port() -> int:
    s = socket.socket(); s.bind(("127.0.0.1", 0)); p = s.getsockname()[1]; s.close(); return p


# ---------------------------------------------------------------- pack extraction

def test_extract_policy_files_skill_only_is_default_policy():
    assert extract_policy_files({"schema_version": 1, "files": {"SKILL.md": "# hi"}}) == {}


def test_extract_policy_files_keeps_everything_but_skill():
    files = {"SKILL.md": "# hi", "policy.py": "print(1)", "prompts/a.txt": "x", "n": 3}
    out = extract_policy_files({"schema_version": 1, "files": files})
    assert out == {"policy.py": "print(1)", "prompts/a.txt": "x"}


def test_extract_policy_files_rejects_traversal_and_size():
    with pytest.raises(ValueError):
        extract_policy_files({"files": {"SKILL.md": "x", "../evil.py": "y"}})
    with pytest.raises(ValueError):
        extract_policy_files({"files": {"SKILL.md": "x", "big.txt": "a" * (POLICY_FILES_MAX_BYTES + 1)}})


def test_runtime_file_ships():
    src = RUNTIME_FILE.read_text()
    assert "def serve(" in src and "class AdvisersPolicy" in src


# ---------------------------------------------------------------- cost table

def test_cost_of_uses_cached_price():
    usd, pt, cached, ct = cost_of("kimi-k3", {"prompt_tokens": 1000, "completion_tokens": 10,
                                              "prompt_tokens_details": {"cached_tokens": 900}})
    assert pt == 1000 and cached == 900 and ct == 10
    assert usd == pytest.approx(100 * 1.95e-6 + 900 * 0.195e-6 + 10 * 9.75e-6)


# ---------------------------------------------------------------- meter against a fake upstream

class _FakeUpstream:
    """Minimal OpenAI-compatible upstream: echoes usage, optionally streams."""

    def __init__(self):
        self.port = _free_port(); self.calls = 0; self.thread = None
        self.rids: list[str] = []; self.was_busy = False

    async def chat(self, req):
        self.calls += 1
        body = await req.json()
        marks = json.dumps(body.get("messages"))
        if "slow" in marks:
            await asyncio.sleep(0.5)          # lets concurrent calls overlap (reservation tests)
        usage = {"prompt_tokens": 100, "completion_tokens": 50, "prompt_tokens_details": {"cached_tokens": 40}}
        msg = {"role": "assistant", "content": "hi"}
        if "plain-openai" in marks:           # a non-Engy upstream: only the standard response id
            return web.json_response({"id": "gen-plain", "choices": [{"message": msg, "finish_reason": "stop"}], "usage": usage})
        # Engy's response identity: the full id on the header (non-stream) and in x_engy, a 24-char prefix in `id`.
        rid = secrets.token_hex(32); self.rids.append(rid)
        cid = f"chatcmpl-{rid[:24]}"; hdr = {"X-Engy-Request-Id": rid}
        if "busy-once" in marks and not self.was_busy:
            self.was_busy = True
            return web.json_response({"error": {"message": "capacity"}}, status=429, headers=hdr)
        if body.get("stream"):
            resp = web.StreamResponse(headers={"Content-Type": "text/event-stream"}); await resp.prepare(req)
            first = {"id": cid, "choices": [{"delta": {"content": "hi"}, "finish_reason": None}]}
            done = {"id": cid, "choices": [{"delta": {}, "finish_reason": "stop"}], "usage": usage}
            if "cut-stream" not in marks:     # a stream that never reaches its finish chunk carries only `id`
                done["x_engy"] = {"request_id": rid}
            await resp.write(b"data: " + json.dumps(first).encode() + b"\n\n")
            await resp.write(b"data: " + json.dumps(done).encode() + b"\n\n")
            await resp.write(b"data: [DONE]\n\n"); await resp.write_eof(); return resp
        return web.json_response({"id": cid, "choices": [{"message": msg, "finish_reason": "stop"}], "usage": usage,
                                  "x_engy": {"request_id": rid}}, headers=hdr)

    def start(self):
        ready = threading.Event()

        def run():
            loop = asyncio.new_event_loop(); asyncio.set_event_loop(loop)
            app = web.Application(); app.router.add_post("/v1/chat/completions", self.chat)
            runner = web.AppRunner(app)
            loop.run_until_complete(runner.setup())
            loop.run_until_complete(web.TCPSite(runner, "127.0.0.1", self.port).start())
            ready.set(); loop.run_forever()

        self.thread = threading.Thread(target=run, daemon=True); self.thread.start(); ready.wait(5)


@pytest.fixture(scope="module")
def meter():
    up = _FakeUpstream(); up.start()
    m = PolicyMeter(f"http://127.0.0.1:{up.port}/v1", "operator-key", port=_free_port())
    m.start()
    return m, up


async def _post(meter, token, body):
    async with ClientSession() as cs:
        async with cs.post(f"http://127.0.0.1:{meter.port}/v1/chat/completions", json=body,
                           headers={"Authorization": f"Bearer {token}"}) as r:
            return r.status, await r.read(), dict(r.headers)


@pytest.mark.asyncio
async def test_meter_rejects_unknown_token(meter):
    m, _ = meter
    status, body, _ = await _post(m, "nope", {"model": "kimi-k3", "messages": []})
    assert status == 401


@pytest.mark.asyncio
async def test_meter_allowlist_and_cost_and_cap(meter):
    m, up = meter
    tok = m.mint("t/scenario", cap_usd=0.001)
    status, body, _ = await _post(m, tok, {"model": "gpt-5.4", "messages": []})
    assert status == 400 and b"allowlist" in body
    # non-stream call: cost recorded from usage
    status, body, hdr = await _post(m, tok, {"model": "kimi-k3", "messages": [{"role": "user", "content": "x"}]})
    assert status == 200 and json.loads(body)["choices"][0]["message"]["content"] == "hi"
    u = m.usage(tok)
    expected = 60 * 1.95e-6 + 40 * 0.195e-6 + 50 * 9.75e-6
    assert u.spent_usd == pytest.approx(expected) and u.calls == 1 and u.by_model == {"kimi-k3": pytest.approx(expected)}
    assert float(hdr["x-trajrl-budget-remaining-usd"]) == pytest.approx(0.001 - expected, abs=1e-6)
    # stream call: usage parsed off the SSE tail
    status, body, hdr = await _post(m, tok, {"model": "glm-5.3-flash", "messages": [], "stream": True})
    assert status == 200 and b"data: [DONE]" in body
    # the client sees [DONE] a moment before the meter thread finishes its accounting: poll briefly
    for _ in range(100):
        u = m.usage(tok)
        if u.calls == 2:
            break
        await asyncio.sleep(0.01)
    assert u.calls == 2 and "glm-5.3-flash" in u.by_model and u.tokens["completion"] == 100
    assert u.rows[-1]["content_sha"] and u.rows[-1]["content_len"] == 2   # provenance fingerprint recorded
    # cap: force it and expect 402 with no upstream call
    u.spent_usd = 1.0; calls_before = up.calls
    status, body, hdr = await _post(m, tok, {"model": "kimi-k3", "messages": []})
    assert status == 402 and up.calls == calls_before and m.usage(tok).refused_cap == 1
    # close: token retired, summary complete
    summary = m.close(tok).summary()
    assert summary["calls"] == 2 and summary["refused_cap"] == 1 and summary["refused_model"] == 1
    status, _, _ = await _post(m, tok, {"model": "kimi-k3", "messages": []})
    assert status == 401


async def _rows(m, tok, n):
    """The meter books a streamed call a moment after the client sees [DONE]: wait for row ``n``."""
    for _ in range(200):
        rows = m.usage(tok).rows
        if len(rows) >= n:
            return rows
        await asyncio.sleep(0.01)
    raise AssertionError(f"meter recorded {len(m.usage(tok).rows)} rows, wanted {n}")


@pytest.mark.asyncio
async def test_row_records_engy_request_id_for_a_plain_call(meter):
    m, up = meter
    tok = m.mint("t/rid-plain", cap_usd=1.0)
    status, _, _ = await _post(m, tok, {"model": "kimi-k3", "messages": [{"role": "user", "content": "x"}]})
    assert status == 200
    row = (await _rows(m, tok, 1))[-1]
    assert row["rid"] == up.rids[-1] and row["retry_rids"] == []
    m.close(tok)


@pytest.mark.asyncio
async def test_row_records_engy_request_id_from_the_stream_finish_chunk(meter):
    m, up = meter
    tok = m.mint("t/rid-stream", cap_usd=1.0)
    status, _, _ = await _post(m, tok, {"model": "kimi-k3", "messages": [], "stream": True})
    assert status == 200
    row = (await _rows(m, tok, 1))[-1]
    assert row["rid"] == up.rids[-1]       # the full id, not the truncated chatcmpl- prefix
    m.close(tok)


@pytest.mark.asyncio
async def test_row_falls_back_to_the_chunk_id_when_the_stream_has_no_finish_handle(meter):
    m, up = meter
    tok = m.mint("t/rid-cut", cap_usd=1.0)
    await _post(m, tok, {"model": "kimi-k3", "messages": [{"role": "user", "content": "cut-stream"}], "stream": True})
    row = (await _rows(m, tok, 1))[-1]
    assert row["rid"] == f"chatcmpl-{up.rids[-1][:24]}"
    m.close(tok)


@pytest.mark.asyncio
async def test_row_keeps_the_ids_of_attempts_the_meter_retried(meter):
    m, up = meter
    tok = m.mint("t/rid-retry", cap_usd=1.0)
    status, _, _ = await _post(m, tok, {"model": "kimi-k3", "messages": [{"role": "user", "content": "busy-once"}]})
    assert status == 200
    row = (await _rows(m, tok, 1))[-1]
    assert row["retries"] == 1
    assert row["retry_rids"] == [up.rids[-2]] and row["rid"] == up.rids[-1]
    m.close(tok)


@pytest.mark.asyncio
async def test_row_falls_back_to_the_response_id_for_a_non_engy_upstream(meter):
    m, _ = meter
    tok = m.mint("t/rid-other", cap_usd=1.0)
    await _post(m, tok, {"model": "kimi-k3", "messages": [{"role": "user", "content": "plain-openai"}]})
    row = (await _rows(m, tok, 1))[-1]
    assert row["rid"] == "gen-plain"
    m.close(tok)


# ---------------------------------------------------------------- runtime helpers (imported as a module)

@pytest.fixture(scope="module")
def rt():
    import importlib.util
    spec = importlib.util.spec_from_file_location("trajrl_policy", RUNTIME_FILE)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


def test_session_key_uses_first_five_messages(rt):
    base = [{"role": "system", "content": "S"}, {"role": "user", "content": "U"}, {"role": "assistant", "content": "A"},
            {"role": "tool", "content": "skill"}, {"role": "tool", "content": "instruction-1"}]
    other = base[:4] + [{"role": "tool", "content": "instruction-2"}]
    assert rt.session_key(base) == rt.session_key(base + [{"role": "assistant", "content": "later"}])
    assert rt.session_key(base) != rt.session_key(other)
    assert rt.session_key(base[:2]).startswith("pre")


def test_transcript_signals(rt):
    sess = {"t0": time.time() - 1000, "last_finish": "length"}
    msgs = [{"role": "tool", "content": "Traceback (most recent call last): boom"}] * 3
    fired = rt.transcript_signals(msgs, sess)
    assert {"stall", "len", "toolerr"} <= set(fired)
    assert "noprog" not in rt.transcript_signals([{"role": "assistant", "content": "cat > /app/run.py"}] * 8, {"t0": time.time()})


def test_rendered_sse_roundtrip(rt):
    msg = {"content": "hello", "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "terminal", "arguments": "{}"}}]}
    out = rt.Rendered(msg, "auto", "tool_calls", {"prompt_tokens": 1}, True, "req1").sse().decode()
    events = [json.loads(l[6:]) for l in out.splitlines() if l.startswith("data: ") and l != "data: [DONE]"]
    assert events[0]["choices"][0]["delta"]["content"] == "hello"
    assert events[1]["choices"][0]["delta"]["tool_calls"][0]["function"]["name"] == "terminal"
    assert events[-1]["choices"][0]["finish_reason"] == "tool_calls" and out.endswith("data: [DONE]\n\n")


def test_policy_from_config(rt):
    assert rt.policy_from_config({"kind": "pin", "model": "glm-5.3-flash"}).name == "pin:glm-5.3-flash"
    p = rt.policy_from_config({"kind": "escalate", "cheap": "a", "strong": "b", "signals": ["stall"]})
    assert p.signals == {"stall"}
    p = rt.policy_from_config({"kind": "advisers", "writer": "kimi-k3", "advisers": ["x", "y"]})
    assert p.writer == "kimi-k3" and p.advisers == ["x", "y"]
    with pytest.raises(ValueError):
        rt.policy_from_config({"kind": "nope"})


def test_allowlist_is_the_prod_catalog():
    assert set(MODEL_ALLOWLIST) == {"deepseek-v4-flash-0731", "deepseek-v4.1-flash", "glm-5.2", "glm-5.3",
                                    "glm-5.3-flash", "kimi-k3", "qwen3.6-35b-a3b", "qwen3.8-27b"}


def test_meter_falls_back_to_ephemeral_port_when_busy():
    busy = socket.socket(); busy.bind(("0.0.0.0", 0)); busy.listen(1); port = busy.getsockname()[1]
    try:
        m = PolicyMeter("http://127.0.0.1:1/v1", "k", port=port)
        m.start()
        assert m.port != port and m.port > 0
    finally:
        busy.close()


def test_ensure_meter_is_started_once_under_parallel_workers():
    import threading
    from trajectoryrl.utils import sandbox_harness as SH
    from trajectoryrl.utils.config import ValidatorConfig
    import tempfile, pathlib
    tmp = pathlib.Path(tempfile.mkdtemp())
    cfg = ValidatorConfig(llm_api_key="k", pack_cache_dir=tmp / "p", log_dir=tmp / "l",
                          eval_state_path=tmp / "e.json", winner_state_path=tmp / "w.json",
                          pack_first_seen_path=tmp / "f.json", active_set_dir=tmp / "a")
    h = SH.TrajectorySandboxHarness(cfg)
    seen = []
    def go():
        seen.append(h._ensure_meter())
    ts = [threading.Thread(target=go) for _ in range(6)]
    [t.start() for t in ts]; [t.join() for t in ts]
    assert len({id(m) for m in seen}) == 1 and seen[0].port > 0


# ---------------------------------------------------------------- review fixes (PR #323)

def test_close_drops_episode_and_keeps_memory_bounded():
    m = PolicyMeter("http://127.0.0.1:1/v1", "k", port=_free_port())
    toks = [m.mint(f"e{i}", cap_usd=1.0) for i in range(50)]
    for t in toks:
        assert m.close(t) is not None
    assert m.episodes == {} and m.close(toks[0]) is None


def test_reserve_for_refuses_clamps_and_fits():
    from trajectoryrl.policy.meter import reserve_for, estimate_prompt_tokens, MIN_MAX_TOKENS
    body = {"messages": [{"role": "user", "content": "x" * 3000}]}   # ~1000 prompt tokens
    est = estimate_prompt_tokens(body); assert 1000 <= est <= 1400   # ~1012 chars/3 x 1.25 safety + 16
    # plenty of room: requested max_tokens honoured, reservation = prompt + completion worst case
    r, mt, clamped = reserve_for("kimi-k3", {**body, "max_tokens": 1000}, room_usd=1.0)
    assert mt == 1000 and not clamped and r == pytest.approx(est * 1.95e-6 + 1000 * 9.75e-6, rel=1e-6)
    # tight room: max_tokens clamped to what fits
    r, mt, clamped = reserve_for("kimi-k3", {**body, "max_tokens": 100000}, room_usd=0.01)
    assert clamped and MIN_MAX_TOKENS <= mt < 100000 and r <= 0.01
    # no room even for the prompt or for MIN_MAX_TOKENS of completion: refused
    assert reserve_for("kimi-k3", body, room_usd=0.001)[1] is None
    assert reserve_for("kimi-k3", body, room_usd=est * 1.95e-6 + 10 * 9.75e-6)[1] is None



@pytest.mark.asyncio
async def test_meter_enforces_cap_before_forwarding_and_reports_budget(meter):
    m, up = meter
    tok = m.mint("t/cap", cap_usd=0.0005)          # less than one 8192-token default completion on kimi
    calls_before = up.calls
    status, body, hdr = await _post(m, tok, {"model": "kimi-k3", "messages": [{"role": "user", "content": "x"}]})
    assert status == 402 and up.calls == calls_before and b"cannot fit" in body
    # a call that fits after clamping goes through with max_tokens reduced
    status, body, hdr = await _post(m, tok, {"model": "deepseek-v4.1-flash", "messages": [{"role": "user", "content": "x"}],
                                             "max_tokens": 1_000_000})
    assert status == 200
    u = m.usage(tok)
    assert u.clamped == 1 and u.rows[-1]["clamped"] and u.rows[-1]["max_tokens"] < 1_000_000
    assert u.reserved_usd == pytest.approx(0.0)      # reservation released after the call
    # authoritative budget endpoint
    async with ClientSession() as cs:
        async with cs.get(f"http://127.0.0.1:{m.port}/v1/budget", headers={"Authorization": f"Bearer {tok}"}) as r:
            assert r.status == 200
            j = await r.json()
            assert j["remaining_usd"] == pytest.approx(0.0005 - u.spent_usd, abs=1e-6) and j["reserved_usd"] == 0
        async with cs.get(f"http://127.0.0.1:{m.port}/v1/budget", headers={"Authorization": "Bearer nope"}) as r:
            assert r.status == 401


@pytest.mark.asyncio
async def test_concurrent_calls_share_one_reservation_pool(meter):
    m, up = meter
    # room for roughly one small call at a time: 3 concurrent flash calls with max_tokens 1000 need
    # 3 x (~16 prompt tokens x 0.04e-6 + 1000 x 0.08e-6) ~ 3 x 8.1e-5; give room for two
    tok = m.mint("t/conc", cap_usd=1.7e-4)
    body = {"model": "deepseek-v4.1-flash", "messages": [{"role": "user", "content": "slow"}], "max_tokens": 1000}
    results = await asyncio.gather(*[_post(m, tok, dict(body)) for _ in range(3)])
    statuses = sorted(r[0] for r in results)
    u = m.usage(tok)
    # three full reservations (3 x ~8.1e-5) do not fit in 1.7e-4: the meter must have clamped or refused
    # at least one of the overlapping calls, and the reservation pool is empty again afterwards
    assert statuses.count(200) >= 2 and (u.clamped >= 1 or statuses.count(402) >= 1)
    assert u.spent_usd <= 1.7e-4 + 1e-9 and u.reserved_usd == pytest.approx(0.0)
    sent = [r["max_tokens"] for r in u.rows if r.get("max_tokens")]
    assert max(sent) == 1000 and (min(sent) < 1000 or statuses.count(402) >= 1)   # third call clamped or refused


def test_read_policy_dir_rejects_binary_and_skips_pyc(tmp_path):
    from trajectoryrl.base.miner import TrajectoryMiner
    d = tmp_path / "pol"; d.mkdir()
    (d / "policy.json").write_text('{"kind": "pin", "model": "glm-5.3-flash"}')
    (d / "__pycache__").mkdir(); (d / "__pycache__" / "x.cpython-313.pyc").write_bytes(b"\x00\x01")
    (d / "stale.pyc").write_bytes(b"\x00\x01")
    assert TrajectoryMiner.read_policy_dir(str(d)) == {"policy.json": '{"kind": "pin", "model": "glm-5.3-flash"}'}
    (d / "logo.png").write_bytes(b"\x89PNG\x00\xff\xfe")
    with pytest.raises(ValueError, match="not UTF-8"):
        TrajectoryMiner.read_policy_dir(str(d))


def test_validate_s1_applies_policy_file_rules():
    from trajectoryrl.base.miner import TrajectoryMiner
    ok = {"schema_version": 1, "files": {"SKILL.md": "# s", "policy.json": "{}"}}
    assert TrajectoryMiner.validate_s1(ok) == []
    bad = {"schema_version": 1, "files": {"SKILL.md": "# s", "../evil.py": "x"}}
    assert any("policy files" in i for i in TrajectoryMiner.validate_s1(bad))


def test_prompt_estimate_counts_tools_and_non_ascii():
    from trajectoryrl.policy.meter import estimate_prompt_tokens
    base = {"model": "kimi-k3", "messages": [{"role": "user", "content": "x" * 300}], "stream": True, "max_tokens": 5}
    e0 = estimate_prompt_tokens(base)
    with_tools = {**base, "tools": [{"type": "function", "function": {"name": "t", "parameters": {"schema": "y" * 3000}}}]}
    assert estimate_prompt_tokens(with_tools) > e0 + 1000          # tools are billed input
    cjk = {**base, "messages": [{"role": "user", "content": "\u4e2d" * 300}]}
    assert estimate_prompt_tokens(cjk) >= 300 * 1.25                # one token per CJK character, with safety factor
    assert estimate_prompt_tokens({**base, "max_tokens": 100000}) == e0   # generation knobs are not billed input
    dense = {**base, "messages": [{"role": "user", "content": "!@#$%^&*()" * 30}]}
    assert estimate_prompt_tokens(dense) >= 300 * 1.25                # symbol-dense ASCII counts one token per character


@pytest.mark.asyncio
async def test_overshoot_is_booked_and_closes_the_cap(meter):
    m, up = meter
    # fake upstream bills 100 prompt tokens whatever we send; a 1-char prompt reserves far less -> overshoot recorded
    # reservation for this call: ~50 prompt tokens + 50 completion on kimi ~ 5.9e-4 (cap 8e-4 leaves >= 64 tokens of room); the fake bills 100 prompt
    # tokens (60 uncached + 40 cached) + 50 completion ~ 6.1e-4 -> overshoot ~4e-5, and spent ends above the cap
    tok = m.mint("t/over", cap_usd=0.0008)
    status, body, _ = await _post(m, tok, {"model": "kimi-k3", "messages": [{"role": "user", "content": "x"}], "max_tokens": 50})
    assert status == 200
    for _ in range(100):
        u = m.usage(tok)
        if u.calls == 1:
            break
        await asyncio.sleep(0.01)
    assert u.overshoot_usd > 0 and u.summary()["overshoot_usd"] > 0
    # spent is now above the cap: the next call is refused before forwarding
    calls_before = up.calls
    status, body, _ = await _post(m, tok, {"model": "kimi-k3", "messages": [{"role": "user", "content": "x"}], "max_tokens": 50})
    assert status == 402 and up.calls == calls_before


def test_drain_aborts_on_should_abort():
    from trajectoryrl.utils.sandbox_harness import _drain_exec_stream_with_deadline
    import itertools
    def silent():
        while True:
            time.sleep(0.2); yield b""
    killed = []
    t0 = time.time()
    chunks, timed_out = _drain_exec_stream_with_deadline(silent(), timeout=30, on_deadline=lambda: killed.append(1),
                                                        poll_interval_s=0.05, should_abort=lambda: time.time() - t0 > 0.3)
    assert timed_out and killed == [1] and time.time() - t0 < 5


def test_runtime_passthrough_auth_headers(rt, monkeypatch):
    monkeypatch.setattr(rt, "UPSTREAM_AUTH", "passthrough")
    srv = rt.Server(rt.PinPolicy("glm-5.3-flash"))
    assert {r.resource.canonical for r in srv.app.router.routes()} >= {"/v1/chat/completions", "/v1/models", "/health"}
    assert srv.hdr_for("Bearer sk-customer")["Authorization"] == "Bearer sk-customer"
    assert srv.hdr_for(None)["Authorization"] == f"Bearer {rt.EPISODE_TOKEN}"
    monkeypatch.setattr(rt, "UPSTREAM_AUTH", "token")
    assert srv.hdr_for("Bearer sk-customer")["Authorization"] == f"Bearer {rt.EPISODE_TOKEN}"
def test_scan_policy_files_flags_dispatch_and_blobs():
    from trajectoryrl.policy import scan_policy_files
    names = ("git-leak-recovery", "postgres-csv-clean", "puzzle-solver")
    clean = {"policy.py": "req['model'] = 'glm-5.3-flash' if 'git' in txt else 'kimi-k3'"}
    r = scan_policy_files(clean, names)
    assert r["scenario_hits"] == 0 and r["b64_blobs"] == 0
    bad = {"policy.py": "TABLE = {'git-leak-recovery': 'kimi-k3', 'puzzle-solver': 'glm-5.3'}\nPAYLOAD = '" + ("QUJD" * 60) + "'\nMSG = 'x' * 1"}
    r = scan_policy_files(bad, names)
    assert r["scenario_hits"] == 2 and set(r["scenario_names"]) == {"git-leak-recovery", "puzzle-solver"} and r["b64_blobs"] == 1 and r["long_literals"] >= 1
