"""Analyze validator consensus from on-chain data and simulate winner election.

Connects to Bittensor subtensor, reads the current validator consensus
commitments (each validator keeps only one on-chain record, overwritten
every cycle), downloads payloads from IPFS (kubo API + public gateways)
with JSON integrity validation per source, runs the same filter pipeline
and consensus computation used in production, then applies Winner
Protection to determine the elected winner.

New-season consensus: highest stake-weighted score wins.  There is no
qualification gate — miners are either scored or disqualified (stake-
weighted majority).  Winner Protection uses score_delta (challenger must
beat winner_score × (1 + δ) to dethrone).

Usage:
    python tools/analyze_consensus.py
    python tools/analyze_consensus.py --network finney --netuid 11
    python tools/analyze_consensus.py --prev-winner 5Ew5PrAd... --prev-winner-score 0.75
"""

import argparse
import asyncio
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import aiohttp
import bittensor as bt

from trajectoryrl.utils.consensus import (
    ConsensusPointer, ConsensusPayload, SCORING_VERSION,
)
from trajectoryrl.utils.consensus_filter import run_filter_pipeline
from trajectoryrl.utils.commitments import (
    fetch_validator_consensus_commitments, decode_dual_address,
)
from trajectoryrl.scoring import compute_consensus_scores
from trajectoryrl.utils.winner_state import (
    WinnerState, select_winner_with_protection,
)

NETUID = 11
NETWORK = "finney"
IPFS_API_URL = "http://ipfs.metahash73.com:5001/api/v0"
IPFS_GATEWAYS = [
    "https://ipfs.io",
    "https://dweb.link",
    "https://cloudflare-ipfs.com",
    "https://gateway.pinata.cloud",
]
CONSENSUS_PROTOCOL_VERSION = 1
CLAWBENCH_VERSION = "0.1.0"
DOWNLOAD_TIMEOUT = 60


def sep(char="=", width=80):
    print(char * width)


async def _fetch_raw(session: aiohttp.ClientSession, url: str, method: str = "GET", **kwargs) -> Optional[bytes]:
    """Fetch raw bytes from a URL, return None on any failure."""
    try:
        req = session.get if method == "GET" else session.post
        async with req(url, timeout=aiohttp.ClientTimeout(total=DOWNLOAD_TIMEOUT), **kwargs) as resp:
            if resp.status != 200:
                return None
            return await resp.read()
    except Exception:
        return None


async def download_ipfs_payload(cid: str) -> Optional[ConsensusPayload]:
    """Download an IPFS CID, trying every source until valid JSON is obtained.

    Unlike ConsensusStore.download_payload, this validates JSON completeness
    per source and moves on to the next if the data is truncated.
    """
    sources: List[Tuple[str, str, dict]] = [
        ("kubo API", f"{IPFS_API_URL}/cat", {"method": "POST", "params": {"arg": cid}}),
    ]
    for gw in IPFS_GATEWAYS:
        sources.append((gw, f"{gw}/ipfs/{cid}", {"method": "GET"}))

    async with aiohttp.ClientSession() as session:
        for name, url, opts in sources:
            method = opts.pop("method", "GET")
            data = await _fetch_raw(session, url, method=method, **opts)
            if data is None:
                continue
            try:
                payload = ConsensusPayload.deserialize(data)
                return payload
            except Exception:
                print(f"      {name}: got {len(data)} bytes but JSON invalid, trying next...")
                continue

    return None


async def download_gcs_payload(gcs_url: str) -> Optional[ConsensusPayload]:
    """Download a payload directly from a GCS URL."""
    async with aiohttp.ClientSession() as session:
        data = await _fetch_raw(session, gcs_url)
        if data is None:
            return None
        try:
            return ConsensusPayload.deserialize(data)
        except Exception:
            print(f"      GCS: got {len(data)} bytes but JSON invalid")
            return None


async def download_payload_dual(content_address: str) -> Optional[ConsensusPayload]:
    """Download payload from a dual-address string (IPFS first, GCS fallback)."""
    ipfs_cid, gcs_url = decode_dual_address(content_address)

    if ipfs_cid:
        payload = await download_ipfs_payload(ipfs_cid)
        if payload is not None:
            return payload
        if gcs_url:
            print("      IPFS failed, falling back to GCS...")

    if gcs_url:
        payload = await download_gcs_payload(gcs_url)
        if payload is not None:
            return payload

    return None


async def run(args):
    # ---- 1. Connect to chain ------------------------------------------------
    print(f"Connecting to subtensor (network={args.network})...")
    subtensor = bt.Subtensor(network=args.network)
    print(f"Loading metagraph for netuid={args.netuid}...")
    metagraph = subtensor.metagraph(args.netuid)
    print(f"  Metagraph loaded: {metagraph.n} neurons")

    # ---- 2. Read consensus commitments from chain ---------------------------
    chain_commitments = fetch_validator_consensus_commitments(
        subtensor, args.netuid, metagraph,
    )
    if not chain_commitments:
        print("No validator consensus commitments found on chain!")
        return

    window_counts: Dict[int, int] = defaultdict(int)
    hk_to_uid: Dict[str, int] = {}
    for uid in range(len(metagraph.hotkeys)):
        hk_to_uid[metagraph.hotkeys[uid]] = uid

    for vc in chain_commitments:
        window_counts[vc.window_number] += 1

    print("\n  Window distribution:")
    for w in sorted(window_counts.keys()):
        print(f"    Window {w}: {window_counts[w]} validators")

    target_window = max(window_counts, key=window_counts.get)
    print(f"\n  Target window: {target_window}")

    # ---- 3. Download payloads (with per-source JSON validation) -------------
    scoring_ver = args.scoring_version if args.scoring_version is not None else SCORING_VERSION
    max_retries = 3
    submissions = []
    skipped_sv = 0
    for vc in chain_commitments:
        if vc.window_number != target_window:
            continue
        if vc.scoring_version != scoring_ver:
            skipped_sv += 1
            continue
        uid = hk_to_uid.get(vc.validator_hotkey, -1)
        ipfs_cid, gcs_url = decode_dual_address(vc.content_address)
        addr_summary = f"ipfs={ipfs_cid or '(none)'}, gcs={(gcs_url or '(none)')[:40]}"
        print(f"  UID {uid} ({vc.validator_hotkey[:12]}...) {addr_summary}")

        pointer = ConsensusPointer(
            protocol_version=vc.protocol_version,
            window_number=vc.window_number,
            content_address=vc.content_address,
            validator_hotkey=vc.validator_hotkey,
        )
        payload = None
        for attempt in range(1, max_retries + 1):
            payload = await download_payload_dual(vc.content_address)
            if payload is not None:
                break
            if attempt < max_retries:
                wait = attempt * 3
                print(f"    Attempt {attempt}/{max_retries} failed, retry in {wait}s...")
                await asyncio.sleep(wait)

        if payload is not None:
            submissions.append((pointer, payload))
            print(f"    OK — {len(payload.scores)} miners evaluated")
        else:
            print(f"    FAILED after {max_retries} attempts")

    total_target = sum(1 for vc in chain_commitments if vc.window_number == target_window)
    if skipped_sv:
        print(f"\n  Skipped {skipped_sv} commitments with mismatched scoring_version (expected {scoring_ver})")
    print(f"\n  Downloaded {len(submissions)}/{total_target - skipped_sv} payloads")

    if not submissions:
        print("\nFailed to download any payloads!")
        return

    # ---- 4. Build validator stakes ------------------------------------------
    validator_stakes: Dict[str, float] = {}
    for uid in range(len(metagraph.hotkeys)):
        hotkey = metagraph.hotkeys[uid]
        stake = float(metagraph.stake[uid])
        if stake > 0:
            validator_stakes[hotkey] = stake

    # ---- 5. Run filter pipeline ---------------------------------------------
    validated, stats = run_filter_pipeline(
        submissions=submissions,
        expected_window=target_window,
        validator_stakes=validator_stakes,
        min_stake=0.0,
        local_version=CLAWBENCH_VERSION,
        expected_protocol=CONSENSUS_PROTOCOL_VERSION,
        expected_scoring_version=scoring_ver,
    )
    print(f"\n  Filter pipeline: {stats.summary()}")

    if not validated:
        print("\nAll submissions filtered out!")
        return

    # ---- 6. Compute consensus scores -----------------------------------------
    consensus_scores, consensus_disqualified = compute_consensus_scores(validated)

    # Filter disqualified miners from scores before winner selection
    eligible_scores = {
        hk: s for hk, s in consensus_scores.items()
        if hk not in consensus_disqualified
    }

    # ---- 7. Winner selection ------------------------------------------------
    prev_state = WinnerState(
        winner_hotkey=args.prev_winner,
        winner_score=args.prev_winner_score,
    )
    winner_hk, updated_state = select_winner_with_protection(
        consensus_scores=eligible_scores,
        state=prev_state,
        score_delta=args.score_delta,
    )

    # ---- 8. Print results ---------------------------------------------------
    n_disqualified = len(consensus_disqualified)

    sep()
    print(f"CONSENSUS RESULTS — Window {target_window}")
    print(f"  {len(consensus_scores)} miners, {len(eligible_scores)} eligible, "
          f"{n_disqualified} disqualified, "
          f"{len(validated)} validators passed filter")
    print(f"  Winner protection score delta: {args.score_delta}")
    sep("-")

    print(f"\n  {'Rank':>4}  {'UID':>5}  {'Hotkey':<20}  {'Score':>10}  {'Status':>6}")
    print(f"  {'----':>4}  {'---':>5}  {'------':<20}  {'-----':>10}  {'------':>6}")

    for rank, (hk, score) in enumerate(
        sorted(consensus_scores.items(), key=lambda x: x[1], reverse=True), 1
    ):
        uid = hk_to_uid.get(hk, -1)
        status = "DISQ" if hk in consensus_disqualified else "OK"
        marker = " <- WINNER" if hk == winner_hk else ""
        print(f"  {rank:>4}  {uid:>5}  {hk[:16]}...  {score:>10.4f}  {status:>6}{marker}")

    sep("=")
    if winner_hk:
        winner_uid = hk_to_uid.get(winner_hk, -1)
        print(f"  WINNER: UID {winner_uid}  {winner_hk}")
        print(f"  SCORE:  {consensus_scores.get(winner_hk, 0):.4f}")

        print(f"\n  Per-validator breakdown:")
        for sub_ptr, sub_payload in submissions:
            val_uid = hk_to_uid.get(sub_ptr.validator_hotkey, -1)
            val_stake = validator_stakes.get(sub_ptr.validator_hotkey, 0)
            if winner_hk in sub_payload.scores:
                disq = sub_payload.disqualified.get(winner_hk)
                s = sub_payload.scores[winner_hk]
                disq_str = f", disqualified={disq}" if disq else ""
                print(f"    UID {val_uid:>3} (stake={val_stake:>10.4f}): "
                      f"score={s:.4f}{disq_str}")
            elif winner_hk in sub_payload.disqualified:
                reason = sub_payload.disqualified[winner_hk]
                print(f"    UID {val_uid:>3} (stake={val_stake:>10.4f}): "
                      f"disqualified ({reason})")
            else:
                print(f"    UID {val_uid:>3} (stake={val_stake:>10.4f}): (not evaluated)")
    else:
        print("  NO WINNER — all miners disqualified or no scores")
    sep("=")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze on-chain validator consensus and simulate winner election",
    )
    parser.add_argument("--network", default=NETWORK)
    parser.add_argument("--netuid", type=int, default=NETUID)
    parser.add_argument("--prev-winner", type=str, default=None)
    parser.add_argument("--prev-winner-score", type=float, default=None)
    parser.add_argument("--score-delta", type=float, default=0.10)
    parser.add_argument(
        "--scoring-version", type=int, default=None,
        help=f"Override scoring version filter (default: {SCORING_VERSION})",
    )
    args = parser.parse_args()

    asyncio.run(run(args))


if __name__ == "__main__":
    main()
