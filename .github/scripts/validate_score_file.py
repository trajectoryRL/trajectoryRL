#!/usr/bin/env python3
"""CI validation script for validator score file PRs.

Runs inside the GitHub Actions pipeline on the validator-scores repo.
Validates schema, filename consistency, sr25519 signature, and
metagraph registration before auto-merge.

Reference: INCENTIVE_MECHANISM.md § Validator Consensus
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

NETUID = int(os.environ.get("NETUID", "11"))
SUBTENSOR_NETWORK = os.environ.get("SUBTENSOR_NETWORK", "finney")

REQUIRED_TOP_FIELDS = {"validator_hotkey", "epoch", "block_height", "scores", "signature"}
REQUIRED_SCORE_FIELDS = {"final_score", "per_scenario"}

FILE_PATTERN = re.compile(r"^epoch-(\d+)/([A-Za-z0-9]+)\.json$")


def validate_filename(filepath: str) -> tuple[list[str], re.Match | None]:
    """Ensure filepath matches ``epoch-{N}/{hotkey}.json``."""
    m = FILE_PATTERN.match(filepath)
    if not m:
        return [f"Path must match epoch-{{N}}/{{hotkey}}.json, got: {filepath}"], None
    return [], m


def validate_schema(data: dict) -> list[str]:
    """Check required fields and types."""
    errors: list[str] = []

    missing = REQUIRED_TOP_FIELDS - set(data.keys())
    if missing:
        errors.append(f"Missing required fields: {missing}")
        return errors

    if not isinstance(data["validator_hotkey"], str):
        errors.append("validator_hotkey must be a string")
    if not isinstance(data["epoch"], int):
        errors.append("epoch must be an integer")
    if not isinstance(data["block_height"], int):
        errors.append("block_height must be an integer")
    if not isinstance(data["signature"], str):
        errors.append("signature must be a string")
    if not isinstance(data["scores"], dict):
        errors.append("scores must be a dict")
        return errors

    for uid_str, score_data in data["scores"].items():
        if not isinstance(score_data, dict):
            errors.append(f"scores[{uid_str}] must be a dict")
            continue
        for field in REQUIRED_SCORE_FIELDS:
            if field not in score_data:
                errors.append(f"scores[{uid_str}] missing '{field}'")

    return errors


def validate_consistency(data: dict, match: re.Match) -> list[str]:
    """Cross-check filename epoch/hotkey against JSON content."""
    errors: list[str] = []
    file_epoch = int(match.group(1))
    file_hotkey = match.group(2)

    if data.get("epoch") != file_epoch:
        errors.append(
            f"Epoch mismatch: filename says {file_epoch}, content says {data.get('epoch')}"
        )
    if data.get("validator_hotkey") != file_hotkey:
        errors.append(
            f"Hotkey mismatch: filename says {file_hotkey}, "
            f"content says {data.get('validator_hotkey')}"
        )
    return errors


def validate_signature(data: dict) -> list[str]:
    """Verify sr25519 payload signature."""
    try:
        import bittensor as bt
    except ImportError:
        return ["bittensor package not installed, cannot verify signature"]

    errors: list[str] = []
    try:
        hotkey = data["validator_hotkey"]
        signature = data["signature"]
        kp = bt.Keypair(ss58_address=hotkey)

        signable = {k: v for k, v in data.items() if k != "signature"}
        canonical = json.dumps(signable, sort_keys=True, separators=(",", ":"))
        if not kp.verify(canonical.encode("utf-8"), bytes.fromhex(signature)):
            errors.append("Invalid sr25519 signature")
    except Exception as e:
        errors.append(f"Signature verification error: {e}")
    return errors


def validate_metagraph(hotkey: str) -> list[str]:
    """Check hotkey is a registered validator with non-zero stake."""
    try:
        import bittensor as bt
    except ImportError:
        return ["bittensor package not installed, cannot query metagraph"]

    errors: list[str] = []
    try:
        subtensor = bt.Subtensor(network=SUBTENSOR_NETWORK)
        metagraph = subtensor.metagraph(netuid=NETUID)

        if hotkey not in metagraph.hotkeys:
            errors.append(f"Hotkey {hotkey} is not registered in netuid {NETUID}")
            return errors

        uid = metagraph.hotkeys.index(hotkey)
        stake = float(metagraph.S[uid])
        if stake <= 0:
            errors.append(f"Validator {hotkey} has zero stake")
    except Exception as e:
        errors.append(f"Metagraph query failed: {e}")
    return errors


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: validate_score_file.py <score_json_path> [<repo_relative_path>]")
        sys.exit(1)

    json_path = sys.argv[1]
    repo_path = sys.argv[2] if len(sys.argv) > 2 else json_path

    all_errors: list[str] = []

    # 1. Filename pattern
    fn_errors, match = validate_filename(repo_path)
    all_errors.extend(fn_errors)

    # 2. Parse JSON
    try:
        data = json.loads(Path(json_path).read_text())
    except Exception as e:
        print(f"::error::Cannot parse JSON: {e}")
        sys.exit(1)

    # 3. Schema
    all_errors.extend(validate_schema(data))

    # 4. Filename <-> content consistency
    if match:
        all_errors.extend(validate_consistency(data, match))

    # 5. sr25519 signature
    all_errors.extend(validate_signature(data))

    # 6. Metagraph registration + stake
    all_errors.extend(validate_metagraph(data.get("validator_hotkey", "")))

    if all_errors:
        print("::error::Score file validation failed")
        for e in all_errors:
            print(f"  - {e}")
        sys.exit(1)

    print(f"PASS: {repo_path}")


if __name__ == "__main__":
    main()
