#!/usr/bin/env python3
"""Report how many *distinct* documents the challenger queue actually contains.

Downloads every publicly mirrored pack from the platform API, groups them into
near-duplicate clusters with the repo's own NCD measure, and prints what the
queue would look like if one slot were allowed per cluster.

The grouping ignores identity entirely: it does not matter whether one
submitter used a single coldkey with many UIDs, or many coldkeys with one UID
each. Only submission time and pack content decide.

Usage:
    python3 tools/pack_duplication_report.py
    python3 tools/pack_duplication_report.py --threshold 0.85
    python3 tools/pack_duplication_report.py --api https://trajrl.com --json report.json

Note: the platform publishes a pack's mirror URL 24 h after submission, so the
newest submissions are necessarily absent from the report.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from trajectoryrl.utils.ncd import (  # noqa: E402
    SIMILARITY_THRESHOLD,
    cluster_packs,
    queue_duplicates,
)

DEFAULT_API = "https://trajrl.com"


def fetch(url: str, tries: int = 3):
    for attempt in range(tries):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "trajrl-tools"})
            with urllib.request.urlopen(request, timeout=40) as response:
                return json.loads(response.read())
        except urllib.error.HTTPError as exc:
            if exc.code in (403, 404):
                return None
        except Exception:  # noqa: BLE001
            pass
        time.sleep(1.5 * (attempt + 1))
    return None


def collect(api: str, workers: int) -> list[dict]:
    listing = fetch(f"{api}/api/submissions") or {}
    submissions = listing.get("submissions", [])
    print(f"submissions listed by the platform: {len(submissions)}", flush=True)

    def one(submission: dict):
        meta = fetch(f"{api}/api/miners/{submission['minerHotkey']}/packs/{submission['packHash']}") or {}
        url = meta.get("gcsPackUrl")
        if not url:
            return None                      # still inside its 24 h window
        pack = fetch(url)
        if not pack or not pack.get("files"):
            return None
        return {
            "id": submission["id"],
            "uid": meta.get("minerUid"),
            "coldkey": meta.get("minerColdkey"),
            "status": submission.get("evalStatus"),
            "submitted_at": submission["submittedAt"],
            "pack": pack,
        }

    with ThreadPoolExecutor(workers) as pool:
        rows = [row for row in pool.map(one, submissions) if row]
    print(f"packs publicly available: {len(rows)}", flush=True)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api", default=DEFAULT_API)
    parser.add_argument("--threshold", type=float, default=SIMILARITY_THRESHOLD)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--json", dest="json_out", help="write the full report here")
    args = parser.parse_args()

    rows = collect(args.api, args.workers)
    if not rows:
        print("nothing to report")
        return 1

    packs = {row["id"]: row["pack"] for row in rows}
    by_id = {row["id"]: row for row in rows}
    clusters = cluster_packs(packs, threshold=args.threshold)

    print(f"\n{len(rows)} packs -> {len(clusters)} distinct documents at threshold {args.threshold}\n")
    print(f"  {'packs':>5s}  {'coldkeys':>8s}  {'UIDs':>4s}  policy files")
    for cluster in clusters:
        coldkeys = {by_id[k]["coldkey"] for k in cluster}
        policies = collections.Counter()
        for key in cluster:
            names = [n for n in packs[key].get("files", {}) if n != "SKILL.md"]
            policies[",".join(sorted(names)) or "SKILL.md only"] += 1
        summary = ", ".join(f"{count}x {name}" for name, count in policies.most_common(3))
        print(f"  {len(cluster):5d}  {len(coldkeys):8d}  {len(cluster):4d}  {summary}")

    entries = [(row["id"], row["pack"], row["submitted_at"]) for row in rows]
    duplicates = queue_duplicates(entries, threshold=args.threshold)
    kept = len(rows) - len(duplicates)
    print(
        f"\nqueue admission with one slot per document: {kept} kept, "
        f"{len(duplicates)} refused as near-copies of an earlier submission "
        f"({len(duplicates) / len(rows):.0%} of the queue)"
    )

    spread = [c for c in clusters if len({by_id[k]["coldkey"] for k in c}) > 1]
    if spread:
        print("\nclusters spanning more than one coldkey (identity counting would miss these):")
        for cluster in spread:
            coldkeys = {by_id[k]["coldkey"] for k in cluster}
            print(f"  {len(cluster)} packs across {len(coldkeys)} coldkeys")

    if args.json_out:
        report = {
            "threshold": args.threshold,
            "packs": len(rows),
            "documents": len(clusters),
            "clusters": [
                {
                    "size": len(c),
                    "coldkeys": sorted({by_id[k]["coldkey"] for k in c if by_id[k]["coldkey"]}),
                    "uids": sorted(by_id[k]["uid"] for k in c if by_id[k]["uid"] is not None),
                    "submission_ids": sorted(c),
                }
                for c in clusters
            ],
            "would_refuse": duplicates,
        }
        with open(args.json_out, "w") as handle:
            json.dump(report, handle, indent=1, default=str)
        print(f"\nfull report written to {args.json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
