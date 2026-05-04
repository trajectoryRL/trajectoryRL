#!/usr/bin/env python3
"""TrajectoryRL Miner — CLI toolbox for pack building and submission.

Commands:
    python neurons/miner.py build       SKILL.md [-o pack.json]
    python neurons/miner.py validate    pack.json
    python neurons/miner.py upload      pack.json [--bucket ...] [--endpoint-url ...]
    python neurons/miner.py web-submit  pack.json [--commit-onchain] [--api-base-url ...]
    python neurons/miner.py submit      <pack_url>
    python neurons/miner.py status

Config is loaded from .env.miner (or environment variables):
    WALLET_NAME, WALLET_HOTKEY, NETUID, NETWORK
    S3_BUCKET, S3_ENDPOINT_URL, S3_REGION, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY
    TRAJECTORYRL_API_BASE_URL (defaults to https://trajrl.com)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logger = logging.getLogger(__name__)


def _fetch_pack(url: str) -> dict:
    import urllib.request

    logger.info("Downloading pack from %s", url)
    req = urllib.request.Request(url, headers={"User-Agent": "TrajectoryRL-Miner/1.0"})
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read())


def _make_miner(config):
    from trajectoryrl.base.miner import TrajectoryMiner

    return TrajectoryMiner(
        wallet_name=config.wallet_name,
        wallet_hotkey=config.wallet_hotkey,
        netuid=config.netuid,
        network=config.network,
    )


# ===================================================================
# CLI commands
# ===================================================================


def cmd_build(args):
    from trajectoryrl.base.miner import TrajectoryMiner

    skill_path = args.skill_md
    try:
        skill_content = open(skill_path).read()
    except FileNotFoundError:
        print(f"Error: file not found: {skill_path}")
        return 1

    if not skill_content.strip():
        print(f"Error: {skill_path} is empty")
        return 1

    pack = TrajectoryMiner.build_s1_pack(skill_content)
    pack_hash = TrajectoryMiner.save_pack(pack, args.output)
    size = len(json.dumps(pack, sort_keys=True))

    print(f"Pack built: {args.output}")
    print(f"  Hash:  {pack_hash}")
    print(f"  Size:  {size} bytes (limit: 32768)")

    if size > 32768:
        print("  Valid: FAILED (exceeds 32 KB size limit)")
        return 1

    print("  Valid: PASSED")
    return 0


def cmd_validate(args):
    from trajectoryrl.base.miner import TrajectoryMiner

    try:
        pack = TrajectoryMiner.load_pack(args.pack_path)
    except FileNotFoundError:
        print(f"Error: file not found: {args.pack_path}")
        return 1
    except json.JSONDecodeError as e:
        print(f"Error: invalid JSON: {e}")
        return 1

    pack_hash = TrajectoryMiner.compute_pack_hash(pack)
    size = len(json.dumps(pack, sort_keys=True))

    print(f"Pack: {args.pack_path}")
    print(f"  Hash:    {pack_hash}")
    print(f"  Size:    {size} bytes (limit: 32768)")
    print(f"  Files:   {list(pack.get('files', {}).keys())}")

    issues = TrajectoryMiner.validate_s1(pack)
    if not issues:
        print("  Schema:  PASSED")
        return 0
    else:
        print("  Schema:  FAILED")
        for issue in issues:
            print(f"    - {issue}")
        return 1


def cmd_upload(args):
    from trajectoryrl.utils.oss_storage import OSSStorage

    try:
        with open(args.pack_path) as f:
            pack = json.load(f)
    except FileNotFoundError:
        print(f"Error: file not found: {args.pack_path}")
        return 1
    except json.JSONDecodeError as e:
        print(f"Error: invalid JSON: {e}")
        return 1

    kwargs = {}
    if args.bucket:
        kwargs["bucket"] = args.bucket
    if args.endpoint_url:
        kwargs["endpoint_url"] = args.endpoint_url
    if args.region:
        kwargs["region"] = args.region

    try:
        storage = OSSStorage(**kwargs) if kwargs else OSSStorage()
    except ValueError as e:
        print(f"Error: S3 not configured: {e}")
        print("Set S3_BUCKET, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY in environment")
        return 1

    try:
        url = storage.upload_pack(pack)
    except Exception as e:
        print(f"Error: upload failed: {e}")
        return 1

    print(f"Uploaded: {url}")
    print(f"Use this URL with: python neurons/miner.py submit {url}")
    return 0


def cmd_web_submit(args):
    from trajectoryrl.base.miner import TrajectoryMiner, DEFAULT_MINER_SUBMIT_URL
    from trajectoryrl.utils.config import MinerConfig

    try:
        pack = TrajectoryMiner.load_pack(args.pack_path)
    except FileNotFoundError:
        print(f"Error: file not found: {args.pack_path}")
        return 1
    except json.JSONDecodeError as e:
        print(f"Error: invalid JSON: {e}")
        return 1

    issues = TrajectoryMiner.validate_s1(pack)
    if issues:
        print("Error: pack validation failed:")
        for issue in issues:
            print(f"  - {issue}")
        return 1

    pack_hash = TrajectoryMiner.compute_pack_hash(pack)
    submit_url = (
        f"{args.api_base_url.rstrip('/')}/api/v2/miners/submit"
        if args.api_base_url else DEFAULT_MINER_SUBMIT_URL
    )

    config = MinerConfig.from_env()
    miner = _make_miner(config)
    print(f"Pack hash: {pack_hash}")
    print(f"Endpoint: {submit_url}")
    print("Submitting pack content to web service...")

    response = miner.submit_pack_via_api(pack, submit_url=submit_url)
    if response is None:
        miner.close()
        print("Submission failed. Check logs for details.")
        return 1

    pack_url = response.get("pack_url")
    print(f"Submitted! pack_url = {pack_url}")
    print(f"  submission_id:    {response.get('submission_id')}")
    print(f"  cooldown_seconds: {response.get('cooldown_seconds')}")
    print(f"  next_upload_at:   {response.get('next_upload_allowed_at')}")
    print(f"  pre_eval_status:  {response.get('pre_eval_status')}")

    if not args.commit_onchain:
        miner.close()
        print()
        print(f"Next step: python neurons/miner.py submit {pack_url}")
        return 0

    print()
    print("Submitting on-chain commitment...")
    success = miner.submit_commitment(pack_hash, pack_url)
    miner.close()
    if success:
        print("On-chain commitment submitted.")
        return 0
    print("On-chain commitment failed. Check logs.")
    return 1


def cmd_submit(args):
    from trajectoryrl.base.miner import TrajectoryMiner
    from trajectoryrl.utils.config import MinerConfig

    config = MinerConfig.from_env()
    miner = _make_miner(config)

    try:
        pack = _fetch_pack(args.pack_url)
    except Exception as e:
        print(f"Error: cannot fetch pack from {args.pack_url}: {e}")
        return 1

    issues = TrajectoryMiner.validate_s1(pack)
    if issues:
        print("Error: pack validation failed:")
        for issue in issues:
            print(f"  - {issue}")
        return 1

    pack_hash = TrajectoryMiner.compute_pack_hash(pack)
    print(f"Pack hash: {pack_hash}")
    print(f"Pack URL:  {args.pack_url}")
    print(f"Submitting on-chain...")

    success = miner.submit_commitment(pack_hash, args.pack_url)
    miner.close()

    if success:
        print("Submitted successfully!")
        return 0
    else:
        print("Submission failed. Check logs for details.")
        return 1


def cmd_status(args):
    from trajectoryrl.utils.commitments import parse_commitment
    from trajectoryrl.utils.config import MinerConfig

    config = MinerConfig.from_env()
    miner = _make_miner(config)
    raw = miner.get_current_commitment()
    miner.close()

    if raw is None:
        print("No commitment found on-chain.")
        return 1

    print(f"On-chain commitment:")
    parsed = parse_commitment(raw)
    if parsed:
        pack_hash, pack_url = parsed
        print(f"  Pack hash: {pack_hash}")
        print(f"  Pack URL:  {pack_url}")
    else:
        print(f"  Raw: {raw}")
        print("  (could not parse commitment)")
    return 0


# ===================================================================
# Entry point
# ===================================================================


def main():
    from pathlib import Path

    # Load .env.miner if present (wallet, S3 config, etc.)
    try:
        from dotenv import load_dotenv
        env_path = Path(__file__).resolve().parent.parent / ".env.miner"
        if env_path.exists():
            load_dotenv(env_path)
    except ImportError:
        pass

    parser = argparse.ArgumentParser(
        description="TrajectoryRL Miner CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  %(prog)s build SKILL.md -o pack.json
  %(prog)s validate pack.json
  %(prog)s upload pack.json                                # self-host on S3 / R2 / etc.
  %(prog)s web-submit pack.json --commit-onchain           # managed GCS via trajrl.com
  %(prog)s submit https://your-bucket.s3.amazonaws.com/pack.json
  %(prog)s status
""",
    )
    parser.add_argument(
        "--log-level", default=None,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    sub = parser.add_subparsers(dest="command")

    # build
    p_build = sub.add_parser("build", help="Build pack.json from SKILL.md")
    p_build.add_argument("skill_md", help="Path to SKILL.md file")
    p_build.add_argument("--output", "-o", default="pack.json", help="Output path")

    # validate
    p_validate = sub.add_parser("validate", help="Validate pack.json locally")
    p_validate.add_argument("pack_path", help="Path to pack.json")

    # upload
    p_upload = sub.add_parser("upload", help="Upload pack.json to S3-compatible storage")
    p_upload.add_argument("pack_path", help="Path to pack.json")
    p_upload.add_argument("--bucket", default=None, help="Override S3_BUCKET")
    p_upload.add_argument("--endpoint-url", default=None, help="Override S3_ENDPOINT_URL")
    p_upload.add_argument("--region", default=None, help="Override S3_REGION")

    # web-submit
    p_web = sub.add_parser(
        "web-submit",
        help="Submit pack via /api/v2/miners/submit (managed GCS hosting)",
    )
    p_web.add_argument("pack_path", help="Path to pack.json")
    p_web.add_argument(
        "--commit-onchain", action="store_true",
        help="Also run the on-chain commitment step after a successful submit",
    )
    p_web.add_argument(
        "--api-base-url", default=None,
        help="Override TRAJECTORYRL_API_BASE_URL for this run (default: https://trajrl.com)",
    )

    # submit
    p_submit = sub.add_parser("submit", help="Submit pack on-chain")
    p_submit.add_argument("pack_url", help="Public URL where pack.json is hosted")

    # status
    sub.add_parser("status", help="Check on-chain commitment")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 0

    log_level = args.log_level or "WARNING"
    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )

    dispatch = {
        "build": cmd_build,
        "validate": cmd_validate,
        "upload": cmd_upload,
        "web-submit": cmd_web_submit,
        "submit": cmd_submit,
        "status": cmd_status,
    }
    return dispatch[args.command](args)


if __name__ == "__main__":
    sys.exit(main() or 0)
