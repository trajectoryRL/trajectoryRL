"""Fetch and analyze evaluation reports for a TrajectoryRL validator.

Score-based analysis: shows score distribution, disqualification status,
weight allocation, and per-miner drill-down.

Usage:
    python analyze_validator.py                     # interactive: list validators, pick one
    python analyze_validator.py <hotkey>             # analyze a specific validator
    python analyze_validator.py <hotkey> --deep      # include per-miner drill-down
    python analyze_validator.py --list               # just list validators
    python analyze_validator.py <hotkey> --dump      # dump raw JSON to file
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "trajrl"))

from trajrl.api import TrajRLClient
from trajrl.display import (
    console,
    trunc,
    relative_time,
    score_fmt,
)

from rich.panel import Panel
from rich.table import Table
from rich import box


def list_validators(client: TrajRLClient) -> list[dict]:
    data = client.validators()
    validators = data.get("validators", [])
    if not validators:
        console.print("[yellow]No validators found.[/]")
        return []

    table = Table(title="Available Validators", box=box.ROUNDED)
    table.add_column("#", justify="right", style="dim")
    table.add_column("Hotkey", style="cyan")
    table.add_column("UID", justify="right")
    table.add_column("LLM Model")
    table.add_column("Version")
    table.add_column("Last Eval")
    table.add_column("Last Seen")
    for i, v in enumerate(validators, 1):
        table.add_row(
            str(i),
            v.get("hotkey", "—"),
            str(v.get("uid", "—")),
            v.get("llmModel") or "—",
            v.get("version") or "—",
            relative_time(v.get("lastEvalAt")),
            relative_time(v.get("lastSeen")),
        )
    console.print(table)
    return validators


def analyze_scores(client: TrajRLClient, validator_hotkey: str) -> dict:
    """Fetch scores and compute aggregated statistics."""
    data = client.scores_by_validator(validator_hotkey)
    entries = data.get("entries", [])
    if not entries:
        console.print(f"[yellow]No score entries for validator {trunc(validator_hotkey)}.[/]")
        return data

    disqualified = [e for e in entries if e.get("disqualified")]
    eligible = [e for e in entries if not e.get("disqualified")]
    scores = [e["score"] for e in entries if e.get("score") is not None]
    weights = [e["weight"] for e in entries if e.get("weight") is not None]

    summary_lines = [
        f"  Miners evaluated: {len(entries)}",
        f"  Eligible: [green]{len(eligible)}[/]  |  Disqualified: [red]{len(disqualified)}[/]",
    ]

    if scores:
        summary_lines.append(
            f"  Score — min: {min(scores):.2f}  avg: {statistics.mean(scores):.2f}  "
            f"max: {max(scores):.2f}  median: {statistics.median(scores):.2f}"
        )
    if weights:
        nonzero_w = [w for w in weights if w > 0]
        summary_lines.append(
            f"  Weights — nonzero: {len(nonzero_w)}/{len(weights)}  "
            f"top: {max(weights):.4f}" if weights else "  Weights — none"
        )

    console.print(Panel(
        "\n".join(summary_lines),
        title=f"Score Summary — {trunc(validator_hotkey)}",
        border_style="cyan",
    ))

    _print_disqualification_breakdown(disqualified)
    _print_weight_distribution(client, validator_hotkey)
    _print_scenario_heatmap(entries)
    _print_leaderboard(entries)

    return data


def _print_disqualification_breakdown(disqualified: list[dict]) -> None:
    if not disqualified:
        return
    reasons: dict[str, int] = {}
    for e in disqualified:
        reason = e.get("disqualifiedReason") or e.get("rejectionStage") or "unknown"
        reasons[reason] = reasons.get(reason, 0) + 1

    table = Table(title="Disqualification Breakdown", box=box.SIMPLE_HEAVY)
    table.add_column("Reason", style="red")
    table.add_column("Count", justify="right")
    for reason, cnt in sorted(reasons.items(), key=lambda x: -x[1]):
        table.add_row(reason, str(cnt))
    console.print(table)


_RE_MINER_WEIGHT = re.compile(
    r"Miner\s+(\d+)\s+\(([^)]+)\):\s+"
    r"score=([\d.]+),\s+status=(\w+)(.*)"
)
_RE_OWNER_WEIGHT = re.compile(
    r"Owner\s+UID\s+(\d+):\s+weight=([\d.]+)\s+\(burn\)"
)
_RE_BURN = re.compile(r"Burn fraction:\s+([\d.]+%)")
_RE_SET_OK = re.compile(r"(Weights set successfully|On-chain set_weights committed successfully)")
_RE_FALLBACK = re.compile(r"(Fallback weights set|setting fallback weight)", re.I)


def _parse_weight_results(log_text: str) -> dict:
    """Parse WEIGHT RESULTS section(s) from a validator cycle log.

    When multiple sections exist the last one wins.
    """
    miners: list[dict] = []
    owner: dict | None = None
    burn_fraction: str | None = None
    success = False
    fallback = False

    in_section = False
    for line in log_text.splitlines():
        if "WEIGHT RESULTS" in line or "ON-CHAIN WEIGHT SUBMISSION" in line or "CONSENSUS RESULTS" in line:
            in_section = True
            miners.clear()
            owner = None
            burn_fraction = None
            continue

        if in_section:
            m = _RE_BURN.search(line)
            if m:
                burn_fraction = m.group(1)
                continue
            m = _RE_OWNER_WEIGHT.search(line)
            if m:
                owner = {"uid": int(m.group(1)), "weight": float(m.group(2))}
                continue
            m = _RE_MINER_WEIGHT.search(line)
            if m:
                miners.append({
                    "uid": int(m.group(1)),
                    "hotkey": m.group(2),
                    "score": float(m.group(3)),
                    "status": m.group(4),
                    "marker": m.group(5).strip(),
                })
                continue
            if "=====" in line:
                continue
            if line.strip() and not line.strip().startswith(("Miner", "Owner")):
                in_section = False

        if _RE_SET_OK.search(line):
            success = True
        if _RE_FALLBACK.search(line):
            fallback = True

    return {
        "miners": miners,
        "owner": owner,
        "burn_fraction": burn_fraction,
        "success": success,
        "fallback": fallback,
    }


def _print_weight_distribution(client: TrajRLClient, validator_hotkey: str) -> None:
    """Fetch the latest cycle log and display set-weights data from it."""
    console.print("[dim]Fetching cycle log...[/]")
    try:
        result = client.cycle_log(validator_hotkey)
    except ValueError as exc:
        console.print(f"[yellow]{exc}[/]")
        return

    log_entry = result["log_entry"]
    console.print(
        f"[dim]Cycle log {log_entry.get('evalId', '?')} "
        f"({relative_time(log_entry.get('createdAt'))})[/]"
    )

    wr = _parse_weight_results(result["text"])
    miners = wr["miners"]

    if not miners and not wr["owner"]:
        if wr["fallback"]:
            console.print(Panel(
                "  [yellow]Fallback weights were set (no qualified miners).[/]",
                title="Set-Weights (from cycle log)",
                border_style="yellow",
            ))
        else:
            console.print("[yellow]No ON-CHAIN WEIGHT SUBMISSION found in cycle log.[/]")
        return

    all_weights = [m.get("weight", 0) for m in miners]
    if wr["owner"]:
        all_weights.append(wr["owner"]["weight"])
    total_weight = sum(all_weights) or 1.0

    summary_lines = [
        f"  Eval ID: [bold]{log_entry.get('evalId', '?')}[/]  "
        f"({relative_time(log_entry.get('createdAt'))})",
    ]
    if wr["burn_fraction"]:
        summary_lines.append(f"  Burn fraction: {wr['burn_fraction']}")
    if wr["owner"]:
        summary_lines.append(
            f"  Owner UID {wr['owner']['uid']}: "
            f"weight={wr['owner']['weight']:.4f} (burn)"
        )

    summary_lines.append(
        f"  Miners: {len(miners)} total"
    )

    winner = next((m for m in miners if "WINNER" in m.get("marker", "")), None)
    if winner:
        summary_lines.append(
            f"  [bold yellow]Winner:[/] UID {winner['uid']} "
            f"({winner['hotkey']})  "
            f"score={winner['score']:.4f}  status={winner.get('status', '?')}"
        )

    ok_miners = [m for m in miners if m.get("status") == "OK"]
    disq_miners = [m for m in miners if m.get("status") == "DISQ"]
    summary_lines.append(
        f"  Status: [green]{len(ok_miners)} OK[/]  |  [red]{len(disq_miners)} DISQ[/]"
    )

    if wr["success"]:
        summary_lines.append("  Status: [green]Weights set successfully[/]")
    elif wr["fallback"]:
        summary_lines.append("  Status: [yellow]Fallback weights used[/]")

    console.print(Panel(
        "\n".join(summary_lines),
        title="Set-Weights (from cycle log)",
        border_style="yellow",
    ))

    if not miners:
        return

    sorted_miners = sorted(miners, key=lambda m: m["score"], reverse=True)

    table = Table(title="Miner Scores (from cycle log)", box=box.ROUNDED)
    table.add_column("#", justify="right", style="dim")
    table.add_column("UID", justify="right")
    table.add_column("Hotkey", style="cyan")
    table.add_column("Score", justify="right", style="bold yellow")
    table.add_column("Status", justify="center")
    table.add_column("Note")

    for i, m in enumerate(sorted_miners, 1):
        status_style = "green" if m.get("status") == "OK" else "red"
        marker = m.get("marker", "")
        note_style = "bold yellow" if "WINNER" in marker else "dim"

        table.add_row(
            str(i),
            str(m["uid"]),
            m["hotkey"],
            f"{m['score']:.4f}",
            f"[{status_style}]{m.get('status', '?')}[/]",
            f"[{note_style}]{marker}[/]" if marker else "",
        )

    if wr["owner"]:
        o = wr["owner"]
        table.add_row(
            "—", str(o["uid"]), "[dim]owner (burn)[/]",
            "—", "—", "[red]BURN[/]",
        )
    console.print(table)

    disq_list = [m for m in miners if m.get("status") == "DISQ"]
    if disq_list:
        table2 = Table(
            title=f"Disqualified Miners ({len(disq_list)})",
            box=box.SIMPLE,
        )
        table2.add_column("UID", justify="right")
        table2.add_column("Hotkey", style="cyan")
        table2.add_column("Score", justify="right")
        for m in sorted(disq_list, key=lambda x: x.get("score", 0), reverse=True):
            table2.add_row(str(m["uid"]), m["hotkey"], f"{m['score']:.4f}")
        console.print(table2)


def _print_scenario_heatmap(entries: list[dict]) -> None:
    """Aggregate scenario-level scores across all miners."""
    scenario_stats: dict[str, dict] = {}
    for e in entries:
        sc = e.get("scenarioScores") or {}
        for name, info in sc.items():
            if not isinstance(info, dict):
                continue
            if name not in scenario_stats:
                scenario_stats[name] = {"total": 0, "scores": []}
            scenario_stats[name]["total"] += 1
            if info.get("score") is not None:
                scenario_stats[name]["scores"].append(info["score"])

    if not scenario_stats:
        return

    table = Table(title="Scenario Analysis", box=box.ROUNDED)
    table.add_column("Scenario")
    table.add_column("Avg Score", justify="right")
    table.add_column("Min", justify="right")
    table.add_column("Max", justify="right")
    table.add_column("Miners", justify="right")
    for name, s in sorted(scenario_stats.items()):
        avg_score = statistics.mean(s["scores"]) if s["scores"] else None
        min_score = min(s["scores"]) if s["scores"] else None
        max_score = max(s["scores"]) if s["scores"] else None
        table.add_row(
            name,
            score_fmt(avg_score),
            score_fmt(min_score),
            score_fmt(max_score),
            str(s["total"]),
        )
    console.print(table)


def _print_leaderboard(entries: list[dict]) -> None:
    ranked = sorted(
        [e for e in entries if e.get("score") is not None],
        key=lambda e: e["score"],
        reverse=True,
    )[:15]
    if not ranked:
        return

    table = Table(title="Top 15 Miners by Score", box=box.ROUNDED)
    table.add_column("#", justify="right", style="dim")
    table.add_column("Miner", style="cyan")
    table.add_column("UID", justify="right")
    table.add_column("Score", justify="right", style="bold")
    table.add_column("Weight", justify="right")
    table.add_column("Status")
    for i, e in enumerate(ranked, 1):
        is_disq = e.get("disqualified") or e.get("rejected")
        status = "[red]DISQ[/]" if is_disq else "[green]OK[/]"
        table.add_row(
            str(i),
            trunc(e.get("minerHotkey")),
            str(e.get("uid") if e.get("uid") is not None else "—"),
            score_fmt(e.get("score")),
            score_fmt(e.get("weight")),
            status,
        )
    console.print(table)


def deep_miner_analysis(client: TrajRLClient, entries: list[dict], top_n: int = 5) -> None:
    """Drill into the top N miners for detailed per-validator/per-pack data."""
    ranked = sorted(
        [e for e in entries if e.get("minerHotkey")],
        key=lambda e: e.get("score") or 0,
        reverse=True,
    )[:top_n]

    for e in ranked:
        hotkey = e["minerHotkey"]
        console.rule(f"[bold cyan]Miner {trunc(hotkey)}[/]")
        try:
            miner_data = client.miner(hotkey)
        except Exception as exc:
            console.print(f"  [red]Failed to fetch miner data: {exc}[/]")
            continue

        _print_miner_overview(miner_data)
        _print_miner_validators(miner_data)
        _print_miner_submissions(miner_data)


def _print_miner_overview(data: dict) -> None:
    lines = [
        f"  Rank: {data.get('rank', '—')}  |  Score: {score_fmt(data.get('score'))}",
        f"  Active: {'yes' if data.get('isActive') else 'no'}",
        f"  Pack: {trunc(data.get('packHash'), 10)}  |  Winner: {'yes' if data.get('isWinner') else 'no'}",
    ]
    ban = data.get("banRecord")
    if ban and ban.get("failedPackCount", 0) > 0:
        lines.append(f"  [red]Failed packs: {ban['failedPackCount']}[/]")
    console.print(Panel("\n".join(lines), border_style="dim"))


def _print_miner_validators(data: dict) -> None:
    validators = data.get("validators", [])
    if not validators:
        return
    table = Table(title="Per-Validator Results", box=box.SIMPLE)
    table.add_column("Validator", style="cyan")
    table.add_column("Score", justify="right")
    table.add_column("Block", justify="right")
    table.add_column("Reported")
    for v in validators:
        table.add_row(
            trunc(v.get("hotkey")),
            score_fmt(v.get("score")),
            str(v.get("blockHeight") or "—"),
            relative_time(v.get("createdAt")),
        )
    console.print(table)


def _print_miner_submissions(data: dict) -> None:
    subs = data.get("recentSubmissions", [])
    if not subs:
        return
    table = Table(title="Recent Submissions", box=box.SIMPLE)
    table.add_column("Pack Hash", style="cyan")
    table.add_column("Status")
    table.add_column("Reason", max_width=60)
    table.add_column("Submitted")
    for s in subs:
        status = s.get("evalStatus", "—")
        style = "green" if status == "passed" else "red"
        table.add_row(
            trunc(s.get("packHash"), 10),
            f"[{style}]{status}[/]",
            (s.get("evalReason") or "—")[:60],
            relative_time(s.get("submittedAt")),
        )
    console.print(table)


def fetch_eval_logs(client: TrajRLClient, validator_hotkey: str, limit: int = 20) -> None:
    """Show recent eval logs for the validator."""
    data = client.eval_logs(validator=validator_hotkey, limit=limit)
    logs = data.get("logs", [])
    if not logs:
        console.print("[dim]No eval logs found.[/]")
        return

    table = Table(title=f"Eval Logs — {trunc(validator_hotkey)}", box=box.ROUNDED)
    table.add_column("Eval ID")
    table.add_column("Type")
    table.add_column("Miner", style="cyan")
    table.add_column("Pack Hash")
    table.add_column("Created")
    for log in logs:
        table.add_row(
            log.get("evalId", "—"),
            log.get("logType", "—"),
            trunc(log.get("minerHotkey")),
            trunc(log.get("packHash"), 10),
            relative_time(log.get("createdAt")),
        )
    console.print(table)


def dump_raw(client: TrajRLClient, validator_hotkey: str) -> None:
    """Dump full raw JSON responses to a file for offline inspection."""
    out = {}
    console.print("[dim]Fetching validators...[/]")
    out["validators"] = client.validators()
    console.print("[dim]Fetching scores...[/]")
    out["scores"] = client.scores_by_validator(validator_hotkey)
    console.print("[dim]Fetching eval logs...[/]")
    out["eval_logs"] = client.eval_logs(validator=validator_hotkey, limit=100)

    outpath = Path(__file__).parent / f"dump_{trunc(validator_hotkey, 8).replace('…', '_')}.json"
    outpath.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    console.print(f"[green]Dumped to {outpath}[/]")


def interactive_pick(validators: list[dict]) -> str | None:
    """Prompt user to pick a validator by index."""
    try:
        choice = input("\nEnter validator # (or full hotkey): ").strip()
    except (EOFError, KeyboardInterrupt):
        return None
    if not choice:
        return None
    if choice.isdigit():
        idx = int(choice) - 1
        if 0 <= idx < len(validators):
            return validators[idx]["hotkey"]
        console.print("[red]Invalid index.[/]")
        return None
    return choice


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze a TrajectoryRL validator report.")
    parser.add_argument("hotkey", nargs="?", help="Validator SS58 hotkey (interactive if omitted)")
    parser.add_argument("--list", action="store_true", help="List validators and exit")
    parser.add_argument("--deep", action="store_true", help="Drill into top miners")
    parser.add_argument("--deep-n", type=int, default=5, help="Number of miners for --deep (default 5)")
    parser.add_argument("--logs", action="store_true", help="Show recent eval logs")
    parser.add_argument("--dump", action="store_true", help="Dump raw JSON to file")
    parser.add_argument("--base-url", default="https://trajrl.com", help="API base URL")
    args = parser.parse_args()

    client = TrajRLClient(base_url=args.base_url)

    if args.list:
        list_validators(client)
        return

    hotkey = args.hotkey
    if not hotkey:
        validators = list_validators(client)
        if not validators:
            return
        hotkey = interactive_pick(validators)
        if not hotkey:
            return

    console.rule(f"[bold]Validator Report: {trunc(hotkey)}[/]")

    if args.dump:
        dump_raw(client, hotkey)
        return

    scores_data = analyze_scores(client, hotkey)

    if args.logs:
        fetch_eval_logs(client, hotkey)

    if args.deep:
        entries = scores_data.get("entries", [])
        deep_miner_analysis(client, entries, top_n=args.deep_n)

    console.print()


if __name__ == "__main__":
    main()
