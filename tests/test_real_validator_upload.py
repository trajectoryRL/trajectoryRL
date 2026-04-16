#!/usr/bin/env python3
"""Real-validator e2e test: run full S1 eval + upload to dashboard.

Uses the real validator wallet (signs the upload) but fabricates a
"miner" side (synthetic keypair + hardcoded SKILL.md). This lets us
verify the full production flow without needing a real miner on-chain:

  1. Validator spawns sandbox + testee + judge containers
  2. 4 episodes run, split-half delta computed
  3. Artifacts (SKILL.md, JUDGE.md, world, per-episode transcripts +
     evaluation.json + fixtures) written to eval_dir
  4. eval_dir tarred and uploaded to https://trajrl.com/api/validators/logs/upload
  5. Script queries GET /api/eval-logs to verify the record exists
  6. Downloads the tar.gz, extracts, verifies structure matches what
     we wrote

Requirements:
  - Docker daemon running
  - ghcr.io/trajectoryrl/trajrl-bench + hermes-agent images available
  - Real validator wallet at ~/.bittensor/wallets/{WALLET_NAME}/
  - .env.validator sourced (CLAWBENCH_LLM_API_KEY etc.)
  - Network access to trajrl.com

Usage:
  cd trajectoryRL
  source .env.validator  # or: set -a; source .env.validator; set +a
  python tests/test_real_validator_upload.py

Takes ~15 minutes (4 full episodes).
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import os
import secrets
import sys
import tarfile
import tempfile
import time
from datetime import datetime
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("real_validator_test")


# ---------------------------------------------------------------------------
# Hardcoded "miner" input — a skilled SKILL.md for incident_response
# ---------------------------------------------------------------------------

SKILL_MD = """\
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
- Write notes to /workspace/learned/notes.md for future episodes

Be direct. No exploration. Exit the SSH session when done.
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

from trajectoryrl.utils.sandbox_harness import _strip_provider_prefix


def _make_config():
    """Build a minimal ValidatorConfig just for running the sandbox harness."""
    from trajectoryrl.utils.config import ValidatorConfig

    # clawbench_path must exist (validator config validates it)
    clawbench_path = Path("/tmp/trajrl-real-test-clawbench")
    clawbench_path.mkdir(parents=True, exist_ok=True)
    (clawbench_path / "scenarios").mkdir(parents=True, exist_ok=True)

    api_key = os.environ["CLAWBENCH_LLM_API_KEY"]
    base_url = os.environ.get("CLAWBENCH_LLM_BASE_URL",
                              "https://openrouter.ai/api/v1")
    model = _strip_provider_prefix(os.environ.get(
        "CLAWBENCH_DEFAULT_MODEL", "z-ai/glm-5.1"))

    return ValidatorConfig(
        wallet_name=os.environ.get("WALLET_NAME", "sn11-owner"),
        wallet_hotkey=os.environ.get("WALLET_HOTKEY", "default"),
        netuid=int(os.environ.get("NETUID", 11)),
        network=os.environ.get("NETWORK", "finney"),
        evaluation_harness="trajrl-bench",
        sandbox_image=os.environ.get(
            "SANDBOX_IMAGE",
            "ghcr.io/trajectoryrl/trajrl-bench:latest",
        ),
        harness_image=os.environ.get(
            "HARNESS_IMAGE",
            "ghcr.io/trajectoryrl/hermes-agent:latest",
        ),
        clawbench_api_key=api_key,
        clawbench_base_url=base_url,
        clawbench_default_model=model,
        sandbox_num_episodes=4,
        sandbox_timeout_per_episode=300,
        clawbench_path=clawbench_path,
        ema_state_path=Path("/tmp/trajrl-real-test-ema.json"),
        winner_state_path=Path("/tmp/trajrl-real-test-winner.json"),
        pack_cache_dir=Path("/tmp/trajrl-real-test-packs"),
        log_dir=Path("/tmp/trajrl-real-test-logs"),
    )


def _gen_fake_miner_keypair():
    """Generate a temporary keypair for the 'miner' side.

    We don't want to upload under a real miner's hotkey; use a fresh
    ss58 address so the backend stores this test cleanly (can filter
    by prefix later).
    """
    from bittensor_wallet import Keypair

    kp = Keypair.create_from_mnemonic(Keypair.generate_mnemonic())
    return kp.ss58_address


def _tar_eval_dir(eval_dir: Path) -> bytes:
    """Tar + gzip everything under eval_dir, mirroring _collect_eval_logs."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for child in sorted(eval_dir.iterdir()):
            tar.add(str(child), arcname=child.name)
    return buf.getvalue()


def _list_tar_contents(tar_bytes: bytes) -> list[str]:
    buf = io.BytesIO(tar_bytes)
    with tarfile.open(fileobj=buf, mode="r:gz") as tar:
        return sorted(tar.getnames())


# ---------------------------------------------------------------------------
# Main flow
# ---------------------------------------------------------------------------

async def main():
    print("=" * 72)
    print("  REAL VALIDATOR E2E — S1 eval + dashboard upload")
    print("=" * 72)

    # -- Validator wallet (real, signs the upload) --
    import bittensor as bt

    config = _make_config()
    wallet = bt.Wallet(name=config.wallet_name, hotkey=config.wallet_hotkey)
    try:
        validator_hotkey = wallet.hotkey.ss58_address
    except Exception as e:
        print(f"FAIL: could not load validator hotkey: {e}")
        sys.exit(1)

    print(f"  Validator: {validator_hotkey}")
    print(f"  Model:     {config.clawbench_default_model}")
    print(f"  Sandbox:   {config.sandbox_image}")
    print(f"  Hermes:    {config.harness_image}")

    # -- Fake miner (synthetic keypair, not on-chain) --
    miner_hotkey = _gen_fake_miner_keypair()
    miner_uid = 999  # clearly synthetic
    pack_hash = "test_" + secrets.token_hex(8)
    print(f"  Miner:     {miner_hotkey} (synthetic)")
    print(f"  Pack hash: {pack_hash}")

    # -- Eval id + directory (mirror validator's _prepare_eval_log_capture) --
    eval_id = f"manualtest_{datetime.now():%Y%m%d_%H%M%S}"
    eval_dir = config.log_dir / "evals" / eval_id / miner_hotkey[:16]
    eval_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Eval ID:   {eval_id}")
    print(f"  Eval dir:  {eval_dir}")

    # -- Run the real sandbox harness --
    from trajectoryrl.utils.sandbox_harness import TrajectorySandboxHarness
    harness = TrajectorySandboxHarness(config)

    print("\n1. Running sandbox eval (4 episodes)...")
    t0 = time.time()
    result = await harness.evaluate_miner(
        skill_md=SKILL_MD,
        epoch_seed=42,
        pack_hash=pack_hash,
        validator_salt="manual-test-salt",
    )
    eval_elapsed = time.time() - t0

    if result.error:
        print(f"FAIL: eval error: {result.error}")
        sys.exit(1)

    print(f"   DONE in {eval_elapsed:.0f}s")
    print(f"   Final score:   {result.score:.3f}")
    print(f"   Mean quality:  {result.mean_quality:.3f}")
    print(f"   Episodes:      {result.episode_qualities}")
    print(f"   Delta:         {result.delta:.3f}")

    # -- Write artifacts into eval_dir --
    print(f"\n2. Writing artifacts to {eval_dir}...")
    result.write_artifacts(eval_dir)

    # Show what was written
    print("   Structure:")
    for p in sorted(eval_dir.rglob("*")):
        if p.is_file():
            rel = p.relative_to(eval_dir)
            print(f"     {rel}  ({p.stat().st_size}B)")

    # -- Tar it up --
    print("\n3. Creating tar.gz archive...")
    archive = _tar_eval_dir(eval_dir)
    print(f"   Size: {len(archive):,} bytes")
    print(f"   Contents ({len(_list_tar_contents(archive))} entries):")
    for name in _list_tar_contents(archive):
        print(f"     {name}")

    # -- Upload --
    print(f"\n4. Uploading to dashboard API...")
    from trajectoryrl.utils.status_reporter import upload_eval_logs

    ok = await upload_eval_logs(
        wallet,
        eval_id=eval_id,
        miner_hotkey=miner_hotkey,
        miner_uid=miner_uid,
        block_height=0,  # manual test, no chain
        pack_hash=pack_hash,
        log_archive=archive,
    )
    print(f"   Upload result: {'OK' if ok else 'FAIL'}")
    if not ok:
        print("   (check network + API endpoint)")
        sys.exit(1)

    # -- Query back via public API to confirm --
    print(f"\n5. Querying /api/eval-logs to verify...")
    import httpx

    await asyncio.sleep(3)  # let the backend index
    api_base = os.environ.get("TRAJECTORYRL_API_BASE_URL",
                              "https://trajrl.com")
    query_url = f"{api_base}/api/eval-logs"

    async with httpx.AsyncClient() as client:
        resp = await client.get(query_url, params={
            "validator": validator_hotkey,
            "eval_id": eval_id,
            "limit": 5,
        })
        resp.raise_for_status()
        data = resp.json()

    logs = data.get("logs", [])
    print(f"   Found {len(logs)} log entries for eval_id={eval_id}")
    if not logs:
        print("   WARN: no logs found (may need more time to index)")
        print(f"   Raw response: {json.dumps(data, indent=2)[:500]}")
    else:
        hit = logs[0]
        print(f"   evalId:     {hit.get('evalId')}")
        print(f"   miner:      {hit.get('minerHotkey', '')[:16]}...")
        print(f"   packHash:   {hit.get('packHash', '')[:16]}...")
        print(f"   gcsUrl:     {hit.get('gcsUrl', '')[:80]}...")
        print(f"   createdAt:  {hit.get('createdAt')}")

        # -- Download and verify structure --
        gcs_url = hit.get("gcsUrl")
        if gcs_url:
            print(f"\n6. Downloading tar.gz from GCS...")
            async with httpx.AsyncClient() as client:
                r = await client.get(gcs_url, timeout=30)
                r.raise_for_status()
                downloaded = r.content
            print(f"   Downloaded: {len(downloaded):,} bytes")

            with tempfile.TemporaryDirectory() as tmp:
                buf = io.BytesIO(downloaded)
                with tarfile.open(fileobj=buf, mode="r:gz") as tar:
                    tar.extractall(tmp)
                tmp_path = Path(tmp)

                print(f"   Extracted structure:")
                for p in sorted(tmp_path.rglob("*")):
                    if p.is_file():
                        rel = p.relative_to(tmp_path)
                        print(f"     {rel}  ({p.stat().st_size}B)")

                # Verify the key S1 artifacts are present
                required = [
                    "SKILL.md", "JUDGE.md", "world.json", "metadata.json",
                    "episodes/episode_0/testee_transcript.txt",
                    "episodes/episode_0/judge_transcript.txt",
                    "episodes/episode_0/evaluation.json",
                    "episodes/episode_3/evaluation.json",
                ]
                missing = [r for r in required if not (tmp_path / r).exists()]
                if missing:
                    print(f"\n   FAIL: missing required files: {missing}")
                    sys.exit(1)
                else:
                    print(f"\n   OK: all {len(required)} required artifacts present in download")

    # -- Summary --
    print("\n" + "=" * 72)
    print(f"  PASS")
    print(f"    quality={result.score:.3f}")
    print(f"    eval+upload took {time.time() - t0:.0f}s total")
    print(f"    Local eval_dir: {eval_dir}")
    print(f"    Retrieve via:  trajrl logs --eval-id {eval_id} --show")
    print("=" * 72)


if __name__ == "__main__":
    asyncio.run(main())
