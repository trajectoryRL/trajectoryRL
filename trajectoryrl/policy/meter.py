"""PolicyMeter: the validator-owned upstream every miner policy must go through.

OpenAI-compatible passthrough to the locked inference endpoint with the operator's key. The sidecar authenticates
with a per-episode token minted by the harness. Per token the meter enforces the model allowlist and the safety cap,
records every call (model, tokens, cached tokens, cost at the frozen price table, latency, status), and returns
``x-trajrl-budget-remaining-usd`` on every response. In-stream concurrency error frames from the gateway are retried
here so a policy never sees them.

Runs in a daemon thread with its own event loop so the synchronous harness (``_run_eval_sync`` in an executor
thread) can mint tokens and read usage without touching the validator's main loop.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import secrets
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from aiohttp import ClientSession, ClientTimeout, web

from . import METER_PORT, MODEL_PRICES

logger = logging.getLogger(__name__)


@dataclass
class EpisodeUsage:
    name: str
    cap_usd: float
    spent_usd: float = 0.0
    calls: int = 0
    errors: int = 0
    refused_cap: int = 0
    refused_model: int = 0
    by_model: dict[str, float] = field(default_factory=dict)
    tokens: dict[str, int] = field(default_factory=lambda: {"prompt": 0, "cached": 0, "completion": 0})
    reserved_usd: float = 0.0   # worst-case cost of calls in flight (released when each returns)
    clamped: int = 0            # calls whose max_tokens the meter reduced to fit the cap
    overshoot_usd: float = 0.0  # actual cost above reservation (prompt estimate too low); should stay 0
    last_call_ts: float = 0.0   # wall clock of the last completed call (policy-stall watchdog)
    rows: list[dict] = field(default_factory=list)   # one per call, for artifacts / provenance
    t0: float = field(default_factory=time.time)

    def summary(self) -> dict[str, Any]:
        return {
            "cost_usd": round(self.spent_usd, 6), "calls": self.calls, "errors": self.errors,
            "refused_cap": self.refused_cap, "refused_model": self.refused_model, "clamped": self.clamped,
            "overshoot_usd": round(self.overshoot_usd, 6),
            "by_model": {m: round(v, 6) for m, v in sorted(self.by_model.items())},
            "tokens": dict(self.tokens), "cap_usd": self.cap_usd,
        }


def norm_args(args: Any) -> str:
    """Tool-call arguments as canonical JSON when they parse, else the raw string stripped."""
    if isinstance(args, (dict, list)):
        return json.dumps(args, sort_keys=True, separators=(",", ":"))
    txt = (args or "") if isinstance(args, str) else str(args)
    try:
        return json.dumps(json.loads(txt), sort_keys=True, separators=(",", ":"))
    except Exception:  # noqa: BLE001
        return txt.strip()


def content_fingerprint(content: str, tool_calls: list) -> dict:
    """Hashes used by the provenance check: sha of the assistant text and sha of (name, canonical args) per tool call."""
    text = (content or "").strip()
    calls = [(tc.get("name") or (tc.get("function") or {}).get("name") or "",
              norm_args(tc.get("arguments") if "arguments" in tc else (tc.get("function") or {}).get("arguments")))
             for tc in (tool_calls or []) if isinstance(tc, dict)]
    return {
        "content_sha": hashlib.sha256(text.encode()).hexdigest()[:16] if text else None,
        "content_len": len(text),
        "tool_sha": hashlib.sha256(json.dumps(calls, sort_keys=True).encode()).hexdigest()[:16] if calls else None,
        "tool_n": len(calls),
    }


# Worst-case prompt-token estimate from the request body: no cache hits assumed.
# Every billed field counts (messages, tools/functions, response_format, system
# fields...), so the whole body is measured minus the generation knobs. ASCII
# is taken at ~3 characters per token (conservative for code and JSON); every
# non-ASCII character counts as a full token (CJK is ~1 token/char); a 1.25x
# safety factor covers tokenizer differences. After the call the real
# usage.prompt_tokens is reconciled against the estimate and any shortfall is
# booked as overshoot (observable per episode) and closes the cap for the
# rest of the episode.
CHARS_PER_TOKEN = 3.0
NON_ASCII_TOKENS_PER_CHAR = 1.0
PROMPT_SAFETY = 1.25
_UNBILLED_KEYS = frozenset({"stream", "stream_options", "max_tokens", "max_completion_tokens", "temperature",
                            "top_p", "n", "seed", "user", "model", "logprobs", "top_logprobs", "presence_penalty",
                            "frequency_penalty", "stop", "parallel_tool_calls", "tool_choice", "reasoning_effort"})
# When a request carries no max_tokens, this is the completion length reserved
# and clamped to; a policy that needs more must say so and have the budget.
DEFAULT_MAX_TOKENS = 8192
MIN_MAX_TOKENS = 64


def estimate_prompt_tokens(body: dict) -> int:
    try:
        billed = {k: v for k, v in body.items() if k not in _UNBILLED_KEYS}
        text = json.dumps(billed, ensure_ascii=False, separators=(",", ":"))
    except Exception:  # noqa: BLE001
        return 16
    # Letters, digits and spaces tokenize at ~3+ chars/token; punctuation and
    # symbols (and anything non-ASCII) are close to one token per character,
    # which is also what an adversary would send to make the estimate low.
    non_ascii = 0; dense = 0; plain = 0
    for ch in text:
        o = ord(ch)
        if o > 127:
            non_ascii += 1
        elif ch.isalnum() or ch in " \n\t":
            plain += 1
        else:
            dense += 1
    return int((plain / CHARS_PER_TOKEN + (dense + non_ascii) * NON_ASCII_TOKENS_PER_CHAR) * PROMPT_SAFETY) + 16


def reserve_for(model: str, body: dict, room_usd: float) -> tuple[float, int | None, bool]:
    """Decide what a call may cost before it is made.

    Returns (reserved_usd, max_tokens_to_send or None if the call must be
    refused, clamped). The reservation is the worst case: every prompt token
    at the uncached price plus max_tokens at the completion price. If the
    requested (or default) completion length does not fit in ``room_usd``,
    max_tokens is reduced to what fits; below MIN_MAX_TOKENS the call is refused."""
    p, c, _ = MODEL_PRICES[model]
    prompt_usd = estimate_prompt_tokens(body) * p
    if prompt_usd >= room_usd:
        return 0.0, None, False
    want = body.get("max_tokens") or body.get("max_completion_tokens") or DEFAULT_MAX_TOKENS
    try:
        want = int(want)
    except (TypeError, ValueError):
        want = DEFAULT_MAX_TOKENS
    fits = int((room_usd - prompt_usd) / c) if c > 0 else want
    if fits < MIN_MAX_TOKENS:
        return 0.0, None, False
    clamped = fits < want
    mt = min(want, fits)
    return prompt_usd + mt * c, mt, clamped


def cost_of(model: str, usage: dict | None) -> tuple[float, int, int, int]:
    p, c, cr = MODEL_PRICES[model]
    u = usage or {}
    pt = int(u.get("prompt_tokens") or 0)
    ct = int(u.get("completion_tokens") or 0)
    cached = int(((u.get("prompt_tokens_details") or {}).get("cached_tokens")) or 0)
    return (pt - cached) * p + cached * cr + ct * c, pt, cached, ct


class PolicyMeter:
    def __init__(self, upstream_base_url: str, api_key: str, port: int = METER_PORT,
                 max_rows_per_episode: int = 2000):
        self.up = upstream_base_url.rstrip("/")
        self.key = api_key
        self.port = port
        self.max_rows = max_rows_per_episode
        self.episodes: dict[str, EpisodeUsage] = {}
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._ready = threading.Event()
        self._error: Optional[BaseException] = None

    # ------------------------------------------------------------------ lifecycle
    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="policy-meter", daemon=True)
        self._thread.start()
        if not self._ready.wait(15):
            raise RuntimeError("policy meter did not start")
        if self._error:
            raise RuntimeError(f"policy meter failed to start: {self._error!r}")
        logger.info("policy meter listening on :%d, upstream %s", self.port, self.up)

    def _run(self) -> None:
        loop = asyncio.new_event_loop(); self._loop = loop
        asyncio.set_event_loop(loop)
        app = web.Application(client_max_size=64 * 1024 * 1024)
        app.router.add_post("/v1/chat/completions", self._chat)
        app.router.add_get("/v1/models", self._models)
        app.router.add_get("/health", self._health)
        app.router.add_get("/v1/budget", self._budget)
        runner = web.AppRunner(app, access_log=None)

        async def _serve():
            await runner.setup()
            # Preferred port first; if another process holds it (a second
            # validator on the same host, a stale lab meter) fall back to an
            # ephemeral port. The sidecar learns the real port from the
            # harness, so any port works.
            try:
                site = web.TCPSite(runner, "0.0.0.0", self.port)
                await site.start()
            except OSError as e:
                logger.warning("policy meter port %d busy (%s); using an ephemeral port", self.port, e)
                site = web.TCPSite(runner, "0.0.0.0", 0)
                await site.start()
            self.port = site._server.sockets[0].getsockname()[1]  # noqa: SLF001

        try:
            loop.run_until_complete(_serve())
        except BaseException as e:  # noqa: BLE001
            self._error = e; self._ready.set(); return
        self._ready.set()
        loop.run_forever()

    # ------------------------------------------------------------------ harness API (thread-safe)
    def mint(self, name: str, cap_usd: float) -> str:
        token = f"ep-{secrets.token_urlsafe(24)}"
        with self._lock:
            self.episodes[token] = EpisodeUsage(name=name, cap_usd=cap_usd)
        return token

    def usage(self, token: str) -> Optional[EpisodeUsage]:
        with self._lock:
            return self.episodes.get(token)

    def close(self, token: str) -> Optional[EpisodeUsage]:
        """Retire a token and return its usage. The entry is dropped: a late
        call with the token gets 401 (unknown token), and nothing is kept
        in the meter, so memory stays bounded by the in-flight episodes."""
        with self._lock:
            return self.episodes.pop(token, None)

    # ------------------------------------------------------------------ HTTP handlers (meter loop)
    def _ep(self, request: web.Request) -> Optional[EpisodeUsage]:
        token = (request.headers.get("Authorization") or "").replace("Bearer ", "").strip()
        with self._lock:
            return self.episodes.get(token)

    async def _models(self, request):
        return web.json_response({"object": "list", "data": [{"id": m, "object": "model", "owned_by": "engy"} for m in MODEL_PRICES]})

    async def _health(self, request):
        return web.json_response({"ok": True, "episodes": len(self.episodes)})

    async def _budget(self, request: web.Request):
        """Authoritative remaining budget for the caller's episode. A streamed
        response's budget header is computed before the call (headers cannot
        change after the stream starts), so the runtime asks here after each
        streamed call."""
        ep = self._ep(request)
        if ep is None:
            return web.json_response({"error": {"message": "unknown or closed episode token", "type": "auth"}}, status=401)
        with self._lock:
            return web.json_response({"cap_usd": ep.cap_usd, "spent_usd": round(ep.spent_usd, 6),
                                      "reserved_usd": round(ep.reserved_usd, 6),
                                      "remaining_usd": round(max(0.0, ep.cap_usd - ep.spent_usd - ep.reserved_usd), 6)})

    async def _chat(self, request: web.Request):
        ep = self._ep(request)
        if ep is None:
            return web.json_response({"error": {"message": "unknown or closed episode token", "type": "auth"}}, status=401)
        body = await request.json()
        model = body.get("model"); stream = bool(body.get("stream"))
        hdr_rem = {"x-trajrl-budget-remaining-usd": f"{max(0.0, ep.cap_usd - ep.spent_usd - ep.reserved_usd):.6f}"}
        if model not in MODEL_PRICES:
            ep.refused_model += 1
            return web.json_response({"error": {"message": f"model {model!r} is not in the allowlist {sorted(MODEL_PRICES)}",
                                                "type": "allowlist"}}, status=400, headers=hdr_rem)
        # Reserve the worst case for this call (prompt at list price + max_tokens
        # at the completion price) against what is left after every other call
        # in flight has reserved its own. The cap is a bound, not a suggestion:
        # nothing is forwarded that could push the episode past it.
        with self._lock:
            room = ep.cap_usd - ep.spent_usd - ep.reserved_usd
            reserved, mt, clamped = reserve_for(model, body, room)
            if mt is None:
                ep.refused_cap += 1
                refuse = True
            else:
                refuse = False
                ep.reserved_usd += reserved
                if clamped:
                    ep.clamped += 1
                body["max_tokens"] = mt
                body.pop("max_completion_tokens", None)
        if refuse:
            self._row(ep, dict(ts=round(time.time(), 3), model=model, status=402, usd=0.0, err="cap"))
            return web.json_response({"error": {"message": f"episode safety cap ${ep.cap_usd:.2f}: this call cannot fit "
                                                           f"(remaining ${max(0.0, room):.4f})", "type": "cap"}},
                                     status=402, headers={"x-trajrl-budget-remaining-usd": f"{max(0.0, room):.6f}"})
        if stream:
            body.setdefault("stream_options", {})["include_usage"] = True
        hdr = {"Authorization": f"Bearer {self.key}", "Content-Type": "application/json"}
        t0 = time.time(); usage = None; finish = None; status = 0; err = None; retries = 0; ttfb = None; fp = {}
        async with ClientSession(timeout=ClientTimeout(total=900)) as cs:
            while True:
                async with cs.post(f"{self.up}/chat/completions", json=body, headers=hdr) as up:
                    status = up.status
                    if status in (429, 503) and retries < 6:
                        await up.read(); retries += 1; await asyncio.sleep(1.5 * retries); continue
                    if status != 200:
                        data = await up.read(); err = data[:300].decode("utf-8", "replace")
                        resp = web.Response(body=data, status=status, headers=hdr_rem,
                                            content_type=(up.headers.get("Content-Type") or "application/json").split(";")[0])
                    elif not stream:
                        data = await up.read(); ttfb = round(time.time() - t0, 3)
                        try:
                            d = json.loads(data); usage = d.get("usage")
                            ch0 = (d.get("choices") or [{}])[0]
                            finish = ch0.get("finish_reason")
                            msg = ch0.get("message") or {}
                            fp = content_fingerprint(msg.get("content") if isinstance(msg.get("content"), str) else "",
                                                     msg.get("tool_calls") or [])
                        except Exception:  # noqa: BLE001
                            pass
                        resp = web.Response(body=data, status=200, content_type="application/json")
                    else:
                        first = await up.content.readany(); ttfb = round(time.time() - t0, 3)
                        if (b'"error"' in first[:400] or b"concurrency" in first[:400]) and retries < 8:
                            err = first[:300].decode("utf-8", "replace"); retries += 1
                            await asyncio.sleep(2.0 * retries); continue
                        resp = web.StreamResponse(status=200, headers={
                            "Content-Type": up.headers.get("Content-Type", "text/event-stream"),
                            "Cache-Control": "no-cache", **hdr_rem})
                        await resp.prepare(request)
                        buf = b""

                        async def _gen():
                            yield first
                            async for c in up.content.iter_any():
                                yield c

                        text_parts: list[str] = []; tcs: dict[int, dict] = {}
                        async for chunk in _gen():
                            await resp.write(chunk); buf += chunk
                            while b"\n\n" in buf:
                                ev, buf = buf.split(b"\n\n", 1)
                                for line in ev.split(b"\n"):
                                    if line.startswith(b"data: ") and line != b"data: [DONE]":
                                        try:
                                            j = json.loads(line[6:])
                                            if j.get("usage"):
                                                usage = j["usage"]
                                            for ch in j.get("choices") or []:
                                                if ch.get("finish_reason"):
                                                    finish = ch["finish_reason"]
                                                delta = ch.get("delta") or {}
                                                if isinstance(delta.get("content"), str):
                                                    text_parts.append(delta["content"])
                                                for tc in delta.get("tool_calls") or []:
                                                    slot = tcs.setdefault(int(tc.get("index", 0)), {"name": "", "arguments": ""})
                                                    fn = tc.get("function") or {}
                                                    if fn.get("name"):
                                                        slot["name"] = fn["name"]
                                                    if fn.get("arguments"):
                                                        slot["arguments"] += fn["arguments"]
                                        except Exception:  # noqa: BLE001
                                            pass
                        await resp.write_eof()
                        fp = content_fingerprint("".join(text_parts), [tcs[i] for i in sorted(tcs)])
                    break
        usd, pt, cached, ct = cost_of(model, usage) if status == 200 else (0.0, 0, 0, 0)
        if status == 200 and usage is None and err is None:
            err = "no-usage"
        with self._lock:
            ep.reserved_usd = max(0.0, ep.reserved_usd - reserved)
            if usd > reserved:
                # the estimate was low (tokenizer denser than assumed): the excess is
                # booked and visible; spent may now sit above cap, which refuses
                # every further call for the episode.
                ep.overshoot_usd += usd - reserved
            ep.spent_usd += usd; ep.calls += 1; ep.last_call_ts = time.time()
            ep.by_model[model] = ep.by_model.get(model, 0.0) + usd
            ep.tokens["prompt"] += pt; ep.tokens["cached"] += cached; ep.tokens["completion"] += ct
            if status != 200:
                ep.errors += 1
        if not stream:
            resp.headers["x-trajrl-budget-remaining-usd"] = f"{max(0.0, ep.cap_usd - ep.spent_usd):.6f}"
        self._row(ep, dict(ts=round(t0, 3), model=model, status=status, stream=stream, prompt=pt, cached=cached,
                           completion=ct, finish=finish, err=err, retries=retries, ttfb=ttfb, max_tokens=mt, clamped=clamped,
                           s=round(time.time() - t0, 3), usd=round(usd, 8), **fp))
        return resp

    def _row(self, ep: EpisodeUsage, row: dict) -> None:
        with self._lock:
            if len(ep.rows) < self.max_rows:
                ep.rows.append(row)
