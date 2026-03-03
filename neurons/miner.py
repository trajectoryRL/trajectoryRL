#!/usr/bin/env python3
"""TrajectoryRL Miner — CLI entry point and submission daemon.

Subcommands:
    build     Build pack.json from AGENTS.md file
    validate  Validate a pack.json locally
    submit    Upload pack + submit on-chain commitment
    status    Check current on-chain commitment

Daemon mode:
    When invoked with no subcommand and PACK_URL is set in the environment,
    runs a long-lived loop that detects pack changes and submits on-chain
    commitments automatically.

Examples:
    # Build a pack from your AGENTS.md
    python neurons/miner.py build --agents-md ./AGENTS.md --output pack.json

    # Validate before submitting
    python neurons/miner.py validate pack.json

    # Submit (pack already uploaded to a public HTTP endpoint)
    python neurons/miner.py submit pack.json \\
        --pack-url https://trajrl.com/samples/pack.json \\
        --wallet.name miner --wallet.hotkey default

    # Check current commitment
    python neurons/miner.py status --wallet.name miner --wallet.hotkey default

    # Run as daemon (Docker / long-running)
    PACK_URL=https://trajrl.com/samples/pack.json \\
    PACK_PATH=/app/packs/pack.json \\
        python neurons/miner.py
"""

import argparse
import asyncio
import json
import logging
import os
import sys
from typing import Optional

logger = logging.getLogger(__name__)


# ===================================================================
# Daemon helpers
# ===================================================================


def _load_or_build_pack(config) -> Optional[dict]:
    """Load pack from file or build from AGENTS.md.

    Args:
        config: MinerConfig instance.

    Returns:
        Pack dict, or None on error.
    """
    from trajectoryrl.base.miner import TrajectoryMiner

    try:
        if config.pack_path:
            return TrajectoryMiner.load_pack(config.pack_path)
        if config.agents_md_path:
            return TrajectoryMiner.build_pack(agents_md=config.agents_md_path)
    except FileNotFoundError:
        logger.error("File not found: %s", config.pack_path or config.agents_md_path)
    except json.JSONDecodeError as e:
        logger.error("Invalid JSON in pack file: %s", e)
    except Exception as e:
        logger.error("Failed to load/build pack: %s", e)
    return None


def _verify_onchain(miner, expected_hash: str) -> bool:
    """Check that the on-chain commitment matches the expected pack hash.

    Args:
        miner: TrajectoryMiner instance.
        expected_hash: Expected pack hash hex string.

    Returns:
        True if on-chain hash matches expected, False otherwise.
    """
    from trajectoryrl.utils.commitments import parse_commitment

    raw = miner.get_current_commitment()
    if raw is None:
        logger.warning("No on-chain commitment found")
        return False

    parsed = parse_commitment(raw)
    if parsed is None:
        logger.warning("Could not parse on-chain commitment: %s", raw)
        return False

    onchain_hash, _ = parsed
    if onchain_hash != expected_hash:
        logger.warning(
            "On-chain hash mismatch: expected %s..., got %s...",
            expected_hash[:16],
            onchain_hash[:16],
        )
        return False

    logger.debug("On-chain commitment matches expected hash")
    return True


async def _run_daemon():
    """Main daemon loop: detect changes, submit."""
    from trajectoryrl.base.miner import TrajectoryMiner
    from trajectoryrl.utils.commitments import parse_commitment
    from trajectoryrl.utils.config import MinerConfig

    # Set up logging early so config validation errors are visible
    log_level = os.environ.get("LOG_LEVEL", "INFO")
    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )

    config = MinerConfig.from_env()

    logger.info("Starting miner daemon")
    logger.info("  pack_url:        %s", config.pack_url)
    logger.info("  pack_path:       %s", config.pack_path or "(none)")
    logger.info("  agents_md_path:  %s", config.agents_md_path or "(none)")
    logger.info("  check_interval:  %ds", config.check_interval)

    miner = TrajectoryMiner(
        wallet_name=config.wallet_name,
        wallet_hotkey=config.wallet_hotkey,
        netuid=config.netuid,
        network=config.network,
    )

    # Read existing on-chain commitment to avoid redundant submission on restart
    last_submitted_hash: Optional[str] = None
    try:
        raw = miner.get_current_commitment()
        if raw:
            parsed = parse_commitment(raw)
            if parsed:
                last_submitted_hash = parsed[0]
                logger.info("Existing on-chain hash: %s...", last_submitted_hash[:16])
    except Exception:
        logger.debug("Could not read existing commitment, will submit on first cycle")

    while True:
        try:
            # 1. Load or build pack
            pack = _load_or_build_pack(config)
            if pack is None:
                logger.error("Could not load pack, retrying in %ds", config.check_interval)
                await asyncio.sleep(config.check_interval)
                continue

            # 2. Validate schema
            result = TrajectoryMiner.validate(pack)
            if not result.passed:
                logger.error("Pack validation failed: %s", result.issues)
                await asyncio.sleep(config.check_interval)
                continue

            # 3. Compute hash — skip if unchanged
            pack_hash = TrajectoryMiner.compute_pack_hash(pack)
            if pack_hash == last_submitted_hash:
                logger.info("Pack unchanged (hash: %s...), verifying on-chain", pack_hash[:16])
                _verify_onchain(miner, pack_hash)
                await asyncio.sleep(config.check_interval)
                continue

            logger.info("Pack changed: %s... → %s...",
                        (last_submitted_hash or "none")[:16], pack_hash[:16])

            # 4. Submit on-chain commitment
            success = miner.submit_commitment(pack_hash, config.pack_url)
            if success:
                last_submitted_hash = pack_hash
                logger.info(
                    "Submission complete: hash=%s... url=%s",
                    pack_hash[:16], config.pack_url,
                )
            else:
                logger.error("On-chain submission failed, will retry next cycle")

            # 5. Sleep
            await asyncio.sleep(config.check_interval)

        except (KeyboardInterrupt, asyncio.CancelledError):
            logger.info("Daemon stopped by user")
            break
        except Exception:
            logger.exception("Unexpected error in daemon loop, retrying in %ds", config.check_interval)
            await asyncio.sleep(config.check_interval)


# ===================================================================
# CLI subcommands
# ===================================================================


def cmd_build(args):
    """Build pack.json from AGENTS.md."""
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

    # Validate
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
    """Validate a pack.json locally."""
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
    """Submit pack to the network."""
    from trajectoryrl.base.miner import TrajectoryMiner

    miner = TrajectoryMiner(
        wallet_name=args.wallet_name,
        wallet_hotkey=args.wallet_hotkey,
        netuid=args.netuid,
        network=args.network,
    )

    pack = TrajectoryMiner.load_pack(args.pack_path)

    success = miner.submit(
        pack=pack,
        pack_url=args.pack_url,
    )

    if success:
        pack_hash = TrajectoryMiner.compute_pack_hash(pack)
        print(f"Submitted successfully!")
        print(f"  Pack hash: {pack_hash}")
        print(f"  Pack URL:  {args.pack_url}")
        return 0
    else:
        print("Submission failed. Check logs for details.")
        return 1


def cmd_status(args):
    """Check current on-chain commitment."""
    from trajectoryrl.base.miner import TrajectoryMiner
    from trajectoryrl.utils.commitments import parse_commitment

    miner = TrajectoryMiner(
        wallet_name=args.wallet_name,
        wallet_hotkey=args.wallet_hotkey,
        netuid=args.netuid,
        network=args.network,
    )

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


# ===================================================================
# Entry point
# ===================================================================


def main():
    parser = argparse.ArgumentParser(
        description="TrajectoryRL Miner — build and submit policy packs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    sub = parser.add_subparsers(dest="command", help="Subcommand")

    # --- build ---
    p_build = sub.add_parser("build", help="Build pack.json from AGENTS.md")
    p_build.add_argument(
        "--agents-md", required=True,
        help="Path to AGENTS.md file",
    )
    p_build.add_argument(
        "--soul-md", default=None,
        help="Path to optional SOUL.md file",
    )
    p_build.add_argument(
        "--pack-name", default="my-pack",
        help="Pack name for metadata (default: my-pack)",
    )
    p_build.add_argument(
        "--pack-version", default="1.0.0",
        help="Semver version (default: 1.0.0)",
    )
    p_build.add_argument(
        "--output", "-o", default="pack.json",
        help="Output path (default: pack.json)",
    )

    # --- validate ---
    p_validate = sub.add_parser("validate", help="Validate pack.json locally")
    p_validate.add_argument("pack_path", help="Path to pack.json")

    # --- submit ---
    p_submit = sub.add_parser("submit", help="Submit pack on-chain")
    p_submit.add_argument("pack_path", help="Path to pack.json")
    p_submit.add_argument(
        "--pack-url", required=True,
        help="Public HTTP(S) URL where pack.json is hosted",
    )
    _add_wallet_args(p_submit)

    # --- status ---
    p_status = sub.add_parser("status", help="Check on-chain commitment")
    _add_wallet_args(p_status)

    args = parser.parse_args()

    # Daemon mode: PACK_URL set + no subcommand → _run_daemon owns logging
    if args.command is None:
        if os.environ.get("PACK_URL"):
            return asyncio.run(_run_daemon())
        logging.basicConfig(
            level=getattr(logging, args.log_level),
            format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        )
        parser.print_help()
        print("\n[hint] Set PACK_URL and PACK_PATH to run as a daemon.")
        return 0

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )

    handlers = {
        "build": cmd_build,
        "validate": cmd_validate,
        "submit": cmd_submit,
        "status": cmd_status,
    }
    return handlers[args.command](args)


def _add_wallet_args(parser):
    """Add common Bittensor wallet/network args."""
    parser.add_argument("--wallet.name", dest="wallet_name", default="miner")
    parser.add_argument("--wallet.hotkey", dest="wallet_hotkey", default="default")
    parser.add_argument("--netuid", type=int, default=11)
    parser.add_argument("--network", default="finney")


if __name__ == "__main__":
    sys.exit(main() or 0)
