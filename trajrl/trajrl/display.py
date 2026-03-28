"""Rich formatters for TTY output."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


# -- helpers ---------------------------------------------------------------

def trunc(hotkey: str | None, n: int = 6) -> str:
    """Truncate a hotkey: 5Cd6ht…sn11"""
    if not hotkey:
        return "—"
    if len(hotkey) <= n + 4:
        return hotkey
    return f"{hotkey[:n]}…{hotkey[-4:]}"


def relative_time(ts: str | None) -> str:
    """ISO/Postgres timestamp → '2h ago'."""
    if not ts:
        return "—"
    try:
        # handle both ISO 8601 and postgres timestamp formats
        clean = ts.replace("+00", "+00:00").replace(" ", "T")
        if not clean.endswith("Z") and "+" not in clean[10:]:
            clean += "+00:00"
        dt = datetime.fromisoformat(clean.replace("Z", "+00:00"))
        delta = datetime.now(timezone.utc) - dt
        secs = int(delta.total_seconds())
        if secs < 0:
            return "just now"
        if secs < 60:
            return f"{secs}s ago"
        if secs < 3600:
            return f"{secs // 60}m ago"
        if secs < 86400:
            return f"{secs // 3600}h ago"
        return f"{secs // 86400}d ago"
    except (ValueError, TypeError):
        return str(ts)


def qual(v: bool | None) -> str:
    if v is None:
        return "—"
    return "[green]\u2713[/]" if v else "[red]\u2717[/]"


def cost(v: float | None) -> str:
    if v is None:
        return "—"
    return f"${v:.4f}"


def score_fmt(v: float | None) -> str:
    if v is None:
        return "—"
    return f"{v:.2f}"


def size_fmt(b: int | None) -> str:
    if not b:
        return "—"
    if b < 1024:
        return f"{b}B"
    if b < 1024 * 1024:
        return f"{b // 1024}KB"
    return f"{b / (1024 * 1024):.1f}MB"


# -- command displays ------------------------------------------------------

def display_status(validators_data: dict, submissions_data: dict) -> None:
    valis = validators_data.get("validators", [])
    subs = submissions_data.get("submissions", [])

    now = datetime.now(timezone.utc)
    active = 0
    latest_eval: str | None = None
    models: dict[str, int] = {}
    for v in valis:
        # count active (seen < 1h)
        if v.get("lastSeen"):
            try:
                clean = v["lastSeen"].replace("+00", "+00:00").replace(" ", "T")
                if not clean.endswith("Z") and "+" not in clean[10:]:
                    clean += "+00:00"
                dt = datetime.fromisoformat(clean.replace("Z", "+00:00"))
                if (now - dt).total_seconds() < 3600:
                    active += 1
            except (ValueError, TypeError):
                pass
        # latest eval
        le = v.get("lastEvalAt")
        if le and (latest_eval is None or le > latest_eval):
            latest_eval = le
        # models
        m = v.get("llmModel")
        if m:
            models[m] = models.get(m, 0) + 1

    passed = sum(1 for s in subs if s.get("evalStatus") == "passed")
    failed = sum(1 for s in subs if s.get("evalStatus") == "failed")

    model_str = ", ".join(f"{m} ({c})" for m, c in sorted(models.items(), key=lambda x: -x[1]))

    lines = [
        f"  Validators: {len(valis)} total, {active} active (seen <1h)",
        f"  LLM Models: {model_str or '—'}",
        f"  Latest Eval: {relative_time(latest_eval)}",
        f"  Submissions: {passed} passed, {failed} failed (last batch)",
    ]
    console.print(Panel("\n".join(lines), title="Network Status", border_style="cyan"))


def display_validators(data: dict) -> None:
    valis = data.get("validators", [])
    table = Table(title=f"Validators ({len(valis)})")
    table.add_column("Hotkey", style="cyan")
    table.add_column("UID", justify="right")
    table.add_column("Version")
    table.add_column("LLM Model")
    table.add_column("Last Eval")
    table.add_column("Last Seen")
    for v in valis:
        table.add_row(
            trunc(v.get("hotkey")),
            str(v.get("uid", "—")),
            v.get("version") or "—",
            v.get("llmModel") or "—",
            relative_time(v.get("lastEvalAt")),
            relative_time(v.get("lastSeen")),
        )
    console.print(table)


def display_scores(data: dict) -> None:
    entries = data.get("entries", [])
    vali = data.get("validator", "")
    table = Table(title=f"Scores from {trunc(vali)} ({len(entries)} miners)")
    table.add_column("Miner", style="cyan")
    table.add_column("UID", justify="right")
    table.add_column("Qual", justify="center")
    table.add_column("Cost", justify="right")
    table.add_column("Score", justify="right")
    table.add_column("Weight", justify="right")
    table.add_column("Scenarios")
    table.add_column("Rejected")
    for e in entries:
        sc = e.get("scenarioScores") or {}
        passed = sum(1 for s in sc.values() if isinstance(s, dict) and s.get("qualified"))
        total = len(sc)
        sc_str = f"{passed}/{total}" if total else "—"

        rej = ""
        if e.get("rejected"):
            stage = e.get("rejectionStage") or ""
            rej = f"[red]{stage}[/]"

        table.add_row(
            trunc(e.get("minerHotkey")),
            str(e.get("uid") if e.get("uid") is not None else "—"),
            qual(e.get("qualified")),
            cost(e.get("costUsd")),
            score_fmt(e.get("score")),
            score_fmt(e.get("weight")),
            sc_str,
            rej or "—",
        )
    console.print(table)


def display_miner(data: dict) -> None:
    hk = data.get("hotkey", "")
    uid = data.get("uid")
    rank = data.get("rank")

    header_parts = [
        f"Rank: {rank or '—'}",
        f"Qualified: {'yes' if data.get('qualified') else 'no'}",
        f"Cost: {cost(data.get('totalCostUsd'))}",
        f"Score: {score_fmt(data.get('score'))}",
    ]
    detail_parts = [
        f"Confidence: {data.get('confidence', '—')}",
        f"Coverage: {data.get('coverage', '—')}",
        f"Active: {'yes' if data.get('isActive') else 'no'}",
        f"Banned: {'yes' if data.get('isBanned') else 'no'}",
    ]
    pack_parts = [
        f"Pack: {trunc(data.get('packHash'), 10)}",
        f"Winner: {'yes' if data.get('isWinner') else 'no'}",
        f"Bootstrap: {'yes' if data.get('isBootstrap') else 'no'}",
    ]

    lines = [
        "  " + " | ".join(header_parts),
        "  " + " | ".join(detail_parts),
        "  " + " | ".join(pack_parts),
    ]

    # ban info
    ban = data.get("banRecord")
    if ban and ban.get("failedPackCount", 0) > 0:
        lines.append(f"  [red]Ban Record: {ban['failedPackCount']} failed packs[/]")
        for fp in ban.get("failedPacks", [])[:3]:
            reason = (fp.get("reason") or "")[:80]
            lines.append(f"    - {trunc(fp.get('pack_hash'), 10)}: {reason}…")

    console.print(Panel(
        "\n".join(lines),
        title=f"Miner {trunc(hk)} (UID {uid or '—'})",
        border_style="cyan",
    ))

    # scenario summary
    scenarios = data.get("scenarioSummary", [])
    if scenarios:
        st = Table(title="Scenario Summary")
        st.add_column("Name")
        st.add_column("Avg Cost", justify="right")
        st.add_column("Avg Score", justify="right")
        st.add_column("Qualified")
        st.add_column("Validators", justify="right")
        for s in scenarios:
            st.add_row(
                s.get("name", "—"),
                cost(s.get("avgCost")),
                score_fmt(s.get("avgScore")),
                f"{s.get('qualCount', 0)}/{s.get('validatorCount', 0)}",
                str(s.get("validatorCount", "—")),
            )
        console.print(st)

    # per-validator
    validators = data.get("validators", [])
    if validators:
        vt = Table(title="Validator Reports")
        vt.add_column("Validator", style="cyan")
        vt.add_column("Qual", justify="center")
        vt.add_column("Cost", justify="right")
        vt.add_column("Score", justify="right")
        vt.add_column("Block", justify="right")
        vt.add_column("Reported")
        vt.add_column("Rejected")
        for v in validators:
            rej = ""
            if v.get("rejected"):
                rej = f"[red]{v.get('rejectionStage', '')}[/]"
            vt.add_row(
                trunc(v.get("hotkey")),
                qual(v.get("qualified")),
                cost(v.get("costUsd")),
                score_fmt(v.get("score")),
                str(v.get("blockHeight") or "—"),
                relative_time(v.get("createdAt")),
                rej or "—",
            )
        console.print(vt)

    # recent submissions
    subs = data.get("recentSubmissions", [])
    if subs:
        st2 = Table(title="Recent Submissions")
        st2.add_column("Pack Hash", style="cyan")
        st2.add_column("Status")
        st2.add_column("Reason")
        st2.add_column("Submitted")
        for s in subs:
            status = s.get("evalStatus", "—")
            style = "green" if status == "passed" else "red"
            reason = (s.get("evalReason") or "—")[:60]
            st2.add_row(
                trunc(s.get("packHash"), 10),
                f"[{style}]{status}[/]",
                reason,
                relative_time(s.get("submittedAt")),
            )
        console.print(st2)


def display_pack(data: dict) -> None:
    ph = data.get("packHash", "")
    summary = data.get("summary", {})

    lines = [
        f"  Status: {data.get('evalStatus', '—')}",
        f"  Miner: {trunc(data.get('minerHotkey'))} (UID {data.get('minerUid', '—')})",
        f"  Qualified: {qual(summary.get('qualified'))}  ({summary.get('qualifiedCount', 0)}/{summary.get('validatorCount', 0)} validators)",
        f"  Best Cost: {cost(summary.get('bestCost'))}  |  Avg Cost: {cost(summary.get('avgCost'))}",
    ]
    if data.get("evalReason"):
        lines.append(f"  [red]Reason: {data['evalReason']}[/]")

    console.print(Panel("\n".join(lines), title=f"Pack {trunc(ph, 10)}", border_style="cyan"))

    validators = data.get("validators", [])
    if validators:
        for v in validators:
            vt = Table(title=f"Validator {trunc(v.get('hotkey'))}")
            vt.add_column("Scenario")
            vt.add_column("Cost", justify="right")
            vt.add_column("Score", justify="right")
            vt.add_column("Qualified", justify="center")
            for sc in v.get("scenarios", []):
                vt.add_row(
                    sc.get("name", "—"),
                    cost(sc.get("cost")),
                    score_fmt(sc.get("score")),
                    qual(sc.get("qualified")),
                )
            console.print(vt)


def display_submissions(data: dict, failed_only: bool = False) -> None:
    subs = data.get("submissions", [])
    if failed_only:
        subs = [s for s in subs if s.get("evalStatus") == "failed"]

    label = "Failed Submissions" if failed_only else f"Submissions ({len(subs)})"
    table = Table(title=label)
    table.add_column("Miner", style="cyan")
    table.add_column("Pack Hash", style="cyan")
    table.add_column("Status")
    table.add_column("Reason", max_width=50)
    table.add_column("Submitted")
    for s in subs:
        status = s.get("evalStatus", "—")
        style = "green" if status == "passed" else "red"
        table.add_row(
            trunc(s.get("minerHotkey")),
            trunc(s.get("packHash"), 10),
            f"[{style}]{status}[/]",
            (s.get("evalReason") or "—")[:50],
            relative_time(s.get("submittedAt")),
        )
    console.print(table)


def display_eval_history(data: dict, validator: str | None = None) -> None:
    """Show a deduplicated list of eval cycle IDs."""
    logs = data.get("logs", [])
    seen: dict[str, dict] = {}
    for log in logs:
        eid = log.get("evalId", "")
        if eid and eid not in seen:
            seen[eid] = log

    title = f"Eval IDs ({len(seen)})"
    if validator:
        title += f" — {trunc(validator)}"
    table = Table(title=title)
    table.add_column("Eval ID", style="bold")
    table.add_column("Validator", style="cyan")
    table.add_column("Block", justify="right")
    table.add_column("Logs", justify="right")
    table.add_column("Created")

    log_counts: dict[str, int] = {}
    for log in logs:
        eid = log.get("evalId", "")
        log_counts[eid] = log_counts.get(eid, 0) + 1

    for eid, entry in seen.items():
        table.add_row(
            eid,
            trunc(entry.get("validatorHotkey")),
            str(entry.get("blockHeight") or "—"),
            str(log_counts.get(eid, 0)),
            relative_time(entry.get("createdAt")),
        )
    console.print(table)


def display_cycle_log(data: dict) -> None:
    """Show cycle log metadata and raw text content."""
    log_entry = data.get("log_entry", {})
    text = data.get("text", "")

    lines = [
        f"  Eval ID: [bold]{log_entry.get('evalId', '—')}[/]",
        f"  Validator: [cyan]{trunc(log_entry.get('validatorHotkey'))}[/]",
        f"  Block: {log_entry.get('blockHeight', '—')}",
        f"  Size: {size_fmt(log_entry.get('sizeBytes'))}",
        f"  Created: {relative_time(log_entry.get('createdAt'))}",
    ]
    console.print(Panel("\n".join(lines), title="Cycle Log", border_style="cyan"))
    console.print(text)


def display_logs(data: dict) -> None:
    logs = data.get("logs", [])
    table = Table(title=f"Eval Logs ({len(logs)})")
    table.add_column("Eval ID")
    table.add_column("Type")
    table.add_column("Validator", style="cyan")
    table.add_column("Miner", style="cyan")
    table.add_column("Pack Hash")
    table.add_column("Size", justify="right")
    table.add_column("GCS URL", max_width=50)
    table.add_column("Created")
    for log in logs:
        table.add_row(
            log.get("evalId", "—"),
            log.get("logType", "—"),
            trunc(log.get("validatorHotkey")),
            trunc(log.get("minerHotkey")),
            trunc(log.get("packHash"), 10),
            size_fmt(log.get("sizeBytes")),
            (log.get("gcsUrl") or "—")[-50:],
            relative_time(log.get("createdAt")),
        )
    console.print(table)
