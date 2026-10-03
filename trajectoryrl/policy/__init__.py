"""Routing-policy evaluation (SN11 Season 2).

A miner's pack carries a fusion policy (``policy.py`` or ``policy.json``) that the validator runs in a sidecar
container next to the scenario container. Hermes talks to the sidecar as if it were the LLM; the sidecar can only
reach the validator's metering proxy (``PolicyMeter``), which forwards to Engy with the operator's key, enforces the
model allowlist and a per-episode safety cap, and records every call. The runtime shipped into the sidecar lives in
``runtime/trajrl_policy.py``.

Constants here are part of the scoring spec: change them only with a SPEC_NUMBER bump.
"""
from __future__ import annotations

from pathlib import Path

RUNTIME_DIR = Path(__file__).parent / "runtime"
RUNTIME_FILE = RUNTIME_DIR / "trajrl_policy.py"

# Port the policy serves on inside the sidecar; Hermes is pointed at http://policy:<port>/v1.
POLICY_PORT = 8800
# Port the meter listens on inside the validator process.
METER_PORT = 8790
# Network alias the sidecar sees for the meter when the validator runs in a container.
METER_ALIAS = "meter"
POLICY_ALIAS = "policy"

# Sidecar resource limits (separate from the scenario container's 4 GB / 2 CPU).
SIDECAR_MEM_LIMIT = "1g"
SIDECAR_CPU_QUOTA = 100000  # 1 CPU
SIDECAR_HEALTH_TIMEOUT_S = 60.0
# Policy-stall watchdog: kill the chat if no metered model call has completed
# this long after chat start, or if none completed in the last POLICY_IDLE_S
# (the agent's long tool commands must stay under this).
POLICY_FIRST_CALL_S = 120.0
POLICY_IDLE_S = 600.0

# Safety cap per episode in USD at the frozen price table. Not a scoring term (Ning 2026-09-18: cost is
# reported, not scored, at launch); it bounds validator spend against runaway policies. 20 scenarios x $0.60 =
# $12 worst case per eval.
EPISODE_CAP_USD = 0.6

# Model allowlist and frozen prices ($ per token: prompt, completion, cache-read), the Engy catalog as of
# 2026-09-18. Every validator must compute the same cost from the same table, so it ships in code.
MODEL_PRICES: dict[str, tuple[float, float, float]] = {
    "deepseek-v4-flash-0731": (0.045e-6, 0.09e-6, 0.009e-6),
    "deepseek-v4.1-flash": (0.04e-6, 0.08e-6, 0.008e-6),
    "glm-5.2": (0.68e-6, 1.5e-6, 0.18e-6),
    "glm-5.3": (0.98e-6, 3.08e-6, 0.18e-6),
    "glm-5.3-flash": (0.135e-6, 0.45e-6, 0.027e-6),
    "kimi-k3": (1.95e-6, 9.75e-6, 0.195e-6),
    "qwen3.6-35b-a3b": (0.045e-6, 0.3e-6, 0.015e-6),
    "qwen3.8-27b": (0.045e-6, 0.32e-6, 0.015e-6),
}
MODEL_ALLOWLIST: tuple[str, ...] = tuple(sorted(MODEL_PRICES))

# Pack files that make up the policy. Everything in ``files`` except SKILL.md is copied into the sidecar's
# /policy directory; these two are the entry points the runtime looks for.
POLICY_ENTRY_PY = "policy.py"
POLICY_ENTRY_JSON = "policy.json"
# Maximum total bytes of policy files (the web caps the whole pack at 32 KB today; this is the validator's own bound).
POLICY_FILES_MAX_BYTES = 512 * 1024


def extract_policy_files(pack: dict) -> dict[str, str]:
    """Return the pack files that belong to the policy (everything except SKILL.md), or {} for a SKILL.md-only
    pack, which is evaluated as the default pin policy on the locked testee model."""
    files = pack.get("files") if isinstance(pack, dict) else None
    if not isinstance(files, dict):
        return {}
    out: dict[str, str] = {}
    for name, content in files.items():
        if name == "SKILL.md" or not isinstance(content, str):
            continue
        if not isinstance(name, str) or name.startswith("/") or ".." in name.split("/"):
            raise ValueError(f"bad policy file name {name!r}")
        out[name] = content
    total = sum(len(c.encode("utf-8")) for c in out.values())
    if total > POLICY_FILES_MAX_BYTES:
        raise ValueError(f"policy files too large: {total} bytes (max {POLICY_FILES_MAX_BYTES})")
    return out


# ---------------------------------------------------------------------------
# Shadow static scan of policy files (Season 2). Records signals of
# scenario-identity dispatch and hidden payloads; reported with the score,
# NOT scored. The web's LLM audit covers SKILL.md at launch; this is the
# validator-side signal for policy code until that audit reads code too.
# ---------------------------------------------------------------------------
import base64 as _b64
import re as _re

_B64_BLOB = _re.compile(r"[A-Za-z0-9+/]{200,}={0,2}")
_STRING_LIT = _re.compile(r"(['\"])(?:(?!\1).){40,}\1")


def scan_policy_files(files: dict[str, str], scenario_names: "tuple[str, ...] | list[str]") -> dict:
    """Cheap, deterministic signals over the policy files.

    - scenario_hits: how many active scenario names appear verbatim (scenario-identity dispatch)
    - scenario_names: which ones (first 10)
    - long_literals: string literals of 40+ chars (canned outputs, templates)
    - b64_blobs: base64-looking runs of 200+ chars (hidden payloads)
    - bytes: total policy bytes
    """
    text = "\n".join(files.get(k, "") for k in sorted(files))
    low = text.lower()
    hits = [n for n in scenario_names if n.lower() in low]
    blobs = 0
    for m in _B64_BLOB.finditer(text):
        try:
            _b64.b64decode(m.group(0) + "=" * (-len(m.group(0)) % 4), validate=False)
            blobs += 1
        except Exception:  # noqa: BLE001
            pass
    return {
        "scenario_hits": len(hits),
        "scenario_names": hits[:10],
        "long_literals": len(_STRING_LIT.findall(text)),
        "b64_blobs": blobs,
        "bytes": len(text.encode("utf-8")),
    }
