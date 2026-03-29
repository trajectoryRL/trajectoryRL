#!/usr/bin/env python3
"""Scan on-chain validators on SN11 and check min_stake eligibility.

Usage:
    python scripts/scan_validators.py
    python scripts/scan_validators.py --network test --netuid 11
    python scripts/scan_validators.py --min-stake 5000
"""

import argparse
import sys

import bittensor as bt


def main():
    parser = argparse.ArgumentParser(description="Scan validators on-chain for min_stake eligibility")
    parser.add_argument("--network", default="finney", help="Subtensor network (default: finney)")
    parser.add_argument("--netuid", type=int, default=11, help="Subnet UID (default: 11)")
    parser.add_argument("--min-stake", type=float, default=10000.0, help="Minimum stake threshold (default: 10000.0)")
    args = parser.parse_args()

    print(f"Connecting to {args.network}, netuid={args.netuid} ...")
    subtensor = bt.Subtensor(network=args.network)
    metagraph = subtensor.metagraph(args.netuid)

    n = getattr(metagraph, "n", 0)
    block = getattr(metagraph, "block", "?")
    print(f"Metagraph synced: {n} neurons at block {block}\n")

    validators = []
    for uid in range(n):
        hotkey = metagraph.hotkeys[uid]
        stake = float(metagraph.stake[uid])
        if stake > 0:
            validators.append({
                "uid": uid,
                "hotkey": hotkey,
                "stake": stake,
                "eligible": stake >= args.min_stake,
            })

    validators.sort(key=lambda v: v["stake"], reverse=True)

    eligible = [v for v in validators if v["eligible"]]
    ineligible = [v for v in validators if not v["eligible"]]

    print(f"Min stake threshold: {args.min_stake:,.2f} τ")
    print(f"Total neurons with stake > 0: {len(validators)}")
    print(f"Eligible (>= {args.min_stake:,.2f} τ): {len(eligible)}")
    print(f"Ineligible: {len(ineligible)}")
    print()

    if eligible:
        print(f"{'='*80}")
        print(f"  ELIGIBLE VALIDATORS ({len(eligible)})")
        print(f"{'='*80}")
        print(f"  {'UID':>5}  {'Stake (τ)':>14}  {'Hotkey'}")
        print(f"  {'---':>5}  {'--------':>14}  {'------'}")
        for v in eligible:
            print(f"  {v['uid']:>5}  {v['stake']:>14,.2f}  {v['hotkey']}")
        total_eligible_stake = sum(v["stake"] for v in eligible)
        print(f"\n  Total eligible stake: {total_eligible_stake:,.2f} τ")

    if ineligible:
        print(f"\n{'='*80}")
        print(f"  INELIGIBLE VALIDATORS ({len(ineligible)})")
        print(f"{'='*80}")
        print(f"  {'UID':>5}  {'Stake (τ)':>14}  {'Hotkey'}")
        print(f"  {'---':>5}  {'--------':>14}  {'------'}")
        for v in ineligible:
            print(f"  {v['uid']:>5}  {v['stake']:>14,.2f}  {v['hotkey']}")

    print()


if __name__ == "__main__":
    main()
