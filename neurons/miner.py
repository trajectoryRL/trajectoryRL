#!/usr/bin/env python3
"""TrajectoryRL Miner — CLI + run modes.

CLI commands (one-shot):
    python neurons/miner.py build --agents-md ./AGENTS.md -o pack.json
    python neurons/miner.py validate pack.json
    python neurons/miner.py submit https://example.com/pack.json
    python neurons/miner.py status

Run modes (long-running daemon):
    python neurons/miner.py run --mode demo      # submit sample pack periodically
    python neurons/miner.py run --mode default    # LLM generate → build → upload → submit

Config is loaded from .env.miner (or environment variables):
    WALLET_NAME, WALLET_HOTKEY, NETUID, NETWORK, CHECK_INTERVAL, LOG_LEVEL
    ANTHROPIC_API_KEY, GENERATOR_MODEL, S3_BUCKET, S3_ENDPOINT_URL, S3_REGION, PACK_URL
"""

import argparse
import asyncio
import json
import logging
import sys
from typing import Optional

from trajectoryrl.utils.config import MinerConfig

logger = logging.getLogger(__name__)

DEMO_PACK_URL = "https://trajrl.com/samples/pack.json"


def _fetch_pack(url: str) -> dict:
    import urllib.request

    logger.info("Downloading pack from %s", url)
    req = urllib.request.Request(url, headers={"User-Agent": "TrajectoryRL-Miner/1.0"})
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read())


def _get_onchain_hash(miner) -> Optional[str]:
    from trajectoryrl.utils.commitments import parse_commitment

    try:
        raw = miner.get_current_commitment()
        if raw:
            parsed = parse_commitment(raw)
            if parsed:
                return parsed[0]
    except Exception:
        logger.debug("Failed to read on-chain commitment", exc_info=True)
    return None


def _make_miner(config: MinerConfig):
    from trajectoryrl.base.miner import TrajectoryMiner

    return TrajectoryMiner(
        wallet_name=config.wallet_name,
        wallet_hotkey=config.wallet_hotkey,
        netuid=config.netuid,
        network=config.network,
    )


# ===================================================================
# run --mode demo
# ===================================================================


async def _run_demo(config: MinerConfig):
    """Periodically fetch the sample pack and submit on-chain."""
    from trajectoryrl.base.miner import TrajectoryMiner

    miner = _make_miner(config)
    interval = config.check_interval

    logger.info("=== Demo mode ===")
    logger.info("  pack_url: %s", DEMO_PACK_URL)
    logger.info("  interval: %ds", interval)

    last_hash = _get_onchain_hash(miner)
    if last_hash:
        logger.info("  on-chain: %s...", last_hash[:16])

    while True:
        try:
            pack = _fetch_pack(DEMO_PACK_URL)

            result = TrajectoryMiner.validate(pack)
            if not result.passed:
                logger.error("Pack validation failed: %s", result.issues)
                await asyncio.sleep(interval)
                continue

            pack_hash = TrajectoryMiner.compute_pack_hash(pack)
            if pack_hash == last_hash:
                logger.info("Pack unchanged (%s...), sleeping %ds",
                            pack_hash[:16], interval)
                await asyncio.sleep(interval)
                continue

            logger.info("Submitting pack %s...", pack_hash[:16])
            if miner.submit(pack, pack_url=DEMO_PACK_URL):
                last_hash = pack_hash
                logger.info("Submitted successfully")
            else:
                logger.error("Submission failed, retrying next cycle")

            await asyncio.sleep(interval)

        except (KeyboardInterrupt, asyncio.CancelledError):
            logger.info("Demo mode stopped")
            break
        except Exception:
            logger.exception("Error in demo loop, retrying in %ds", interval)
            await asyncio.sleep(interval)


# ===================================================================
# run --mode default
# ===================================================================


async def _run_default(config: MinerConfig):
    """Production mining loop: generate AGENTS.md → build pack → upload → submit.

    Each cycle:
        1. Generate (or improve) AGENTS.md via LLM
        2. Build OPP v1 pack
        3. Validate locally
        4. Skip if hash unchanged
        5. Upload to S3 (or use pre-set PACK_URL)
        6. Submit on-chain commitment
    """
    from trajectoryrl.base.miner import TrajectoryMiner
    from trajectoryrl.utils.pack_generator import generate_agents_md

    # --- Config validation (fail fast) ---
    if not config.anthropic_api_key:
        logger.error("ANTHROPIC_API_KEY is required for default mode")
        sys.exit(1)

    # Init storage (reads S3_BUCKET from env; raises ValueError if missing)
    storage = None
    if not config.pack_url:
        from trajectoryrl.utils.oss_storage import OSSStorage
        try:
            storage = OSSStorage()
        except ValueError as e:
            logger.error("S3 not configured and PACK_URL not set: %s", e)
            sys.exit(1)

    miner = _make_miner(config)
    interval = config.check_interval

    logger.info("=== Default mode ===")
    logger.info("  model:    %s", config.generator_model)
    if storage:
        logger.info("  upload:   %s (bucket)", storage.bucket)
    else:
        logger.info("  pack_url: %s (you must upload pack yourself)", config.pack_url)
    logger.info("  interval: %ds", interval)

    last_hash = _get_onchain_hash(miner)
    if last_hash:
        logger.info("  on-chain: %s...", last_hash[:16])

    previous_agents_md: Optional[str] = None

    try:
        while True:
            try:
                # Step 1: Generate AGENTS.md
                logger.info("Generating AGENTS.md...")
                agents_md = await asyncio.to_thread(
                    generate_agents_md,
                    api_key=config.anthropic_api_key,
                    model=config.generator_model,
                    previous_agents_md=previous_agents_md,
                )

                # Step 2: Build pack
                pack = TrajectoryMiner.build_pack(agents_md=agents_md)

                # Step 3: Validate
                result = TrajectoryMiner.validate(pack)
                if not result.passed:
                    logger.error("Pack validation failed: %s", result.issues)
                    await asyncio.sleep(interval)
                    continue

                # Step 4: Check if hash changed
                pack_hash = TrajectoryMiner.compute_pack_hash(pack)
                if pack_hash == last_hash:
                    logger.info("Pack unchanged (%s...), sleeping %ds",
                                pack_hash[:16], interval)
                    previous_agents_md = agents_md
                    await asyncio.sleep(interval)
                    continue

                # Step 5: Upload (or save locally for manual upload)
                if storage:
                    pack_url = await asyncio.to_thread(
                        storage.upload_pack, pack,
                    )
                else:
                    pack_url = config.pack_url
                    local_path = "pack.json"
                    TrajectoryMiner.save_pack(pack, local_path)
                    logger.info(
                        "Pack saved to %s — upload to %s before "
                        "validators fetch it",
                        local_path,
                        pack_url,
                    )

                # Step 6: Submit on-chain
                logger.info("Submitting pack %s...", pack_hash[:16])
                if miner.submit(pack, pack_url=pack_url):
                    last_hash = pack_hash
                    logger.info("Submitted successfully")
                else:
                    logger.error("Submission failed, retrying next cycle")

                # Store for next cycle's improvement prompt
                previous_agents_md = agents_md

                await asyncio.sleep(interval)

            except (KeyboardInterrupt, asyncio.CancelledError):
                logger.info("Default mode stopped")
                break
            except Exception:
                logger.exception("Error in default loop, retrying in %ds", interval)
                await asyncio.sleep(interval)
    finally:
        miner.close()


# ===================================================================
# CLI commands
# ===================================================================


def cmd_build(args):
    from trajectoryrl.base.miner import TrajectoryMiner

    pack = TrajectoryMiner.build_pack(
        agents_md=args.agents_md,
        pack_name=args.pack_name,
        pack_version=args.pack_version,
        soul_md=args.soul_md,
    )

    pack_hash = TrajectoryMiner.save_pack(pack, args.output)
    size = len(json.dumps(pack, sort_keys=True))

    print(f"Pack built: {args.output}")
    print(f"  Hash:  {pack_hash}")
    print(f"  Size:  {size} bytes (limit: 32768)")
    print(f"  Files: {list(pack['files'].keys())}")

    result = TrajectoryMiner.validate(pack)
    if result.passed:
        print("  Schema: PASSED")
    else:
        print("  Schema: FAILED")
        for issue in result.issues:
            print(f"    - {issue}")
        return 1
    return 0


def cmd_validate(args):
    from trajectoryrl.base.miner import TrajectoryMiner

    pack = TrajectoryMiner.load_pack(args.pack_path)
    result = TrajectoryMiner.validate(pack)
    pack_hash = TrajectoryMiner.compute_pack_hash(pack)
    size = len(json.dumps(pack, sort_keys=True))

    print(f"Pack: {args.pack_path}")
    print(f"  Hash:    {pack_hash}")
    print(f"  Size:    {size} bytes (limit: 32768)")
    print(f"  Name:    {pack.get('metadata', {}).get('pack_name', '?')}")
    print(f"  Version: {pack.get('metadata', {}).get('pack_version', '?')}")
    print(f"  Files:   {list(pack.get('files', {}).keys())}")

    if result.passed:
        print("  Schema:  PASSED")
        return 0
    else:
        print("  Schema:  FAILED")
        for issue in result.issues:
            print(f"    - {issue}")
        return 1


def cmd_submit(args):
    config = MinerConfig.from_env()
    miner = _make_miner(config)
    pack = _fetch_pack(args.pack_url)
    success = miner.submit(pack=pack, pack_url=args.pack_url)

    if success:
        from trajectoryrl.base.miner import TrajectoryMiner
        pack_hash = TrajectoryMiner.compute_pack_hash(pack)
        print(f"Submitted successfully!")
        print(f"  Pack hash: {pack_hash}")
        print(f"  Pack URL:  {args.pack_url}")
        return 0
    else:
        print("Submission failed. Check logs for details.")
        return 1


def cmd_status(args):
    from trajectoryrl.utils.commitments import parse_commitment

    config = MinerConfig.from_env()
    miner = _make_miner(config)
    raw = miner.get_current_commitment()
    if raw is None:
        print("No commitment found on-chain.")
        return 1

    print(f"Raw commitment: {raw}")
    parsed = parse_commitment(raw)
    if parsed:
        pack_hash, pack_url = parsed
        print(f"  Pack hash: {pack_hash}")
        print(f"  Pack URL:  {pack_url}")
    else:
        print("  (could not parse commitment)")
    return 0


def cmd_run(args):
    config = MinerConfig.from_env()
    if args.interval is not None:
        config.check_interval = args.interval
    if args.mode == "demo":
        return asyncio.run(_run_demo(config))
    else:
        return asyncio.run(_run_default(config))


# ===================================================================
# Entry point
# ===================================================================


def main():
    parser = argparse.ArgumentParser(
        description="TrajectoryRL Miner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--log-level", default=None,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    sub = parser.add_subparsers(dest="command")

    p_build = sub.add_parser("build", help="Build pack.json from AGENTS.md")
    p_build.add_argument("--agents-md", required=True)
    p_build.add_argument("--soul-md", default=None)
    p_build.add_argument("--pack-name", default="my-pack")
    p_build.add_argument("--pack-version", default="1.0.0")
    p_build.add_argument("--output", "-o", default="pack.json")

    p_validate = sub.add_parser("validate", help="Validate pack.json locally")
    p_validate.add_argument("pack_path")

    p_submit = sub.add_parser("submit", help="Submit pack on-chain")
    p_submit.add_argument("pack_url", help="Public URL where pack.json is hosted")

    p_run = sub.add_parser("run", help="Run miner daemon")
    p_run.add_argument(
        "--mode", required=True, choices=["demo", "default"],
        help="demo: submit sample pack; default: LLM generate + upload + submit",
    )
    p_run.add_argument(
        "--interval", type=int, default=None,
        help="Seconds between submission cycles (overrides CHECK_INTERVAL)",
    )

    sub.add_parser("status", help="Check on-chain commitment")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 0

    config = MinerConfig.from_env()
    log_level = args.log_level or config.log_level
    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )

    dispatch = {
        "build": cmd_build,
        "validate": cmd_validate,
        "submit": cmd_submit,
        "status": cmd_status,
        "run": cmd_run,
    }
    return dispatch[args.command](args)


if __name__ == "__main__":
    sys.exit(main() or 0)
