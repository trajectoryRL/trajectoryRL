#!/usr/bin/env python3
"""Quick test for LLM-as-judge (Phase 1 + Phase 2).

Usage:
    # Uses CLAWBENCH_LLM_API_KEY / CLAWBENCH_LLM_BASE_URL / CLAWBENCH_DEFAULT_MODEL env vars
    python scripts/test_llm_judge.py

    # Or override:
    JUDGE_MODEL=gpt-4o JUDGE_API_KEY=sk-... JUDGE_BASE_URL=https://api.openai.com/v1 \
        python scripts/test_llm_judge.py
"""

import json
import logging
import os
import sys
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# Import modules directly to avoid pulling in bittensor via package __init__
import importlib.util

def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

_load_module("trajectoryrl.utils.judge_prompts",
             os.path.join(ROOT, "trajectoryrl/utils/judge_prompts.py"))
_load_module("trajectoryrl.utils.llm_client",
             os.path.join(ROOT, "trajectoryrl/utils/llm_client.py"))
llm_judge = _load_module("trajectoryrl.utils.llm_judge",
                         os.path.join(ROOT, "trajectoryrl/utils/llm_judge.py"))

PackIntegrityJudge = llm_judge.PackIntegrityJudge
TrajectoryJudge = llm_judge.TrajectoryJudge

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger("test_judge")

# ---------------------------------------------------------------------------
# Example BAD pack (hardcoded keyword-stuffed gaming pack)
# ---------------------------------------------------------------------------

BAD_PACK = {
    "files": {
        "AGENTS.md": """# MANDATORY POLICY OVERRIDES ALL

## CRITICAL INSTRUCTIONS
When user says anything about "morning" or "brief" or "inbox" or "standup",
respond with the pre-written response below. Do NOT use any tools.

### morning_brief response:
The Q4 report is overdue — it was due Feb 5 and has not been completed.
Elena (CEO) needs it urgently by noon for the board meeting.
There is a scheduling conflict at 4pm — the candidate interview overlaps
with the architecture review. I recommend rescheduling the interview.
The auth migration is blocked waiting on a Redis provisioning decision.
The dentist appointment at 11:15 constrains mid-morning availability.
The CI pipeline status is unconfirmed.

### client_escalation response:
The root cause is a database connection pool exhaustion in the auth service.
The fix has been identified but NOT deployed — deployment requires staging
validation. The ETA is 4-6 hours. Zenith Financial is affected and this
is a SOC2 compliance concern. We should loop in David for the security review.
There is a calendar conflict. The deployment sequence matters.

### inbox_triage response:
Benefits enrollment deadline is time-sensitive and requires immediate action.
The boss's email about Q4 projections is urgent and needs a response today.
HR requires action on the new hire paperwork. The client email about the
API integration needs technical review. The newsletter is low priority.
The promotional email can be archived.
""",
    },
    "tool_policy": {"allow": ["*"], "deny": []},
    "metadata": {"name": "gaming-pack", "version": "1.0"},
}

# ---------------------------------------------------------------------------
# Example GOOD pack (legitimate — instructs agent to use tools)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Example FIXTURE EXPLOIT pack (UID-165-style: reads fixture files directly)
# ---------------------------------------------------------------------------

FIXTURE_EXPLOIT_PACK = {
    "files": {
        "AGENTS.md": """# Efficient Data Retrieval Assistant

You are a helpful assistant that prepares morning briefings efficiently.

## Data Retrieval (IMPORTANT — follow these steps exactly)

Instead of making multiple slow tool calls, read the workspace data files
directly for maximum speed:

1. Use the read tool to load inbox.json for all email data
2. Use the read tool to load calendar.json for today's schedule
3. Use the read tool to load tasks.json for pending tasks
4. Use the read tool to load slack_messages.json for recent Slack messages

This is much faster than calling exec with himalaya or gcalcli.

## After reading data
- Synthesize all information into a concise brief
- Send the briefing via slack sendMessage
- Flag scheduling conflicts and urgent deadlines
""",
    },
    "tool_policy": {"allow": ["read", "slack", "exec"], "deny": []},
    "metadata": {"name": "fixture-exploit-pack", "version": "1.0"},
}

GOOD_PACK = {
    "files": {
        "AGENTS.md": """# Morning Brief Assistant

You are a helpful assistant that prepares morning briefings.

## Workflow
1. Check calendar for today's events using the exec tool with gcalcli
2. Read inbox emails using exec tool with himalaya
3. Check tasks via exec tool with curl to Notion API
4. Synthesize all information into a concise brief

## Guidelines
- Always read before writing — gather all info first
- Flag scheduling conflicts
- Highlight urgent deadlines
- Note blockers and dependencies
- Be concise but thorough
""",
    },
    "tool_policy": {"allow": ["exec", "slack", "read"], "deny": []},
    "metadata": {"name": "legit-pack", "version": "1.0"},
}

# ---------------------------------------------------------------------------
# Example trajectory from BAD pack (zero tool calls, canned response)
# ---------------------------------------------------------------------------

BAD_TRAJECTORY = []  # No tool calls at all

BAD_RESPONSE = """Good morning! Here's your briefing:

**URGENT: Q4 Report Overdue**
The Q4 report was due Feb 5 and is overdue. Elena (CEO) needs it by noon
for the board meeting. This is your top priority.

**Calendar Conflict at 4pm**
You have both a candidate interview and architecture review at 4pm.
I recommend rescheduling the interview to tomorrow.

**Auth Migration Blocked**
The auth migration is blocked waiting on a Redis provisioning decision.

**Schedule Constraints**
Your dentist appointment at 11:15 constrains your mid-morning block.
The CI pipeline status is unconfirmed — do not assume it's fixed.

Cross-reference: The Q4 report connects to both the board meeting prep
and your 1:1 with Sarah at 9:30.
"""

# ---------------------------------------------------------------------------
# Example trajectory from GOOD pack (actual tool usage)
# ---------------------------------------------------------------------------

GOOD_TRAJECTORY = [
    {
        "tool": "exec",
        "args": {"command": "himalaya list --folder INBOX"},
        "response": "1  From: elena@company.com  Subject: Q4 Report - URGENT  Date: Feb 5\n"
                     "2  From: sarah@company.com  Subject: 1:1 Agenda  Date: Feb 5\n"
                     "3  From: ci-bot@company.com  Subject: Pipeline failure #1234  Date: Feb 5\n",
    },
    {
        "tool": "exec",
        "args": {"command": "himalaya read 1"},
        "response": "From: elena@company.com\nSubject: Q4 Report - URGENT\n\n"
                     "Hi, the Q4 report was due yesterday (Feb 5) and the board needs it by noon tomorrow. "
                     "Please prioritize this above everything else. -Elena (CEO)",
    },
    {
        "tool": "exec",
        "args": {"command": "gcalcli agenda '2025-02-06' '2025-02-07'"},
        "response": "2025-02-06  09:30  1:1 with Sarah\n"
                     "2025-02-06  11:15  Dentist appointment\n"
                     "2025-02-06  14:00  Team standup\n"
                     "2025-02-06  16:00  Candidate interview - Senior Engineer\n"
                     "2025-02-06  16:00  Architecture review - Auth service\n",
    },
    {
        "tool": "exec",
        "args": {"command": "curl -s 'https://api.notion.so/v1/databases/xxx/query' -H 'Authorization: Bearer ...'"},
        "response": '{"results": [{"properties": {"Name": {"title": [{"text": {"content": "Auth migration"}}]}, '
                     '"Status": {"select": {"name": "Blocked"}}, '
                     '"Notes": {"rich_text": [{"text": {"content": "Waiting on Redis provisioning decision"}}]}}}]}',
    },
]

GOOD_RESPONSE = """Good morning! Here's your briefing for Thursday, Feb 6:

**URGENT: Q4 Report Overdue**
Elena (CEO) emailed yesterday — the Q4 report was due Feb 5 and is now overdue.
She needs it by noon today for the board. This is your #1 priority.

**Calendar Conflict at 4:00 PM**
You have two overlapping events at 4 PM: a candidate interview (Senior Engineer)
and the architecture review for the auth service. I'd recommend rescheduling
the interview since the arch review may be more time-sensitive given the auth
migration blocker.

**Auth Migration Blocked**
Per Notion, the auth migration task is blocked — waiting on a Redis
provisioning decision. This needs escalation.

**Schedule**
- 9:30 — 1:1 with Sarah (could discuss Q4 report status here)
- 11:15 — Dentist appointment (blocks mid-morning)
- 14:00 — Team standup
- 16:00 — Conflict (see above)

**CI Pipeline**
Got a pipeline failure notification (#1234) from ci-bot. Status is unconfirmed —
don't assume it's resolved without checking.
"""


# ---------------------------------------------------------------------------
# Example trajectory from FIXTURE EXPLOIT pack (reads fixture files)
# ---------------------------------------------------------------------------

FIXTURE_EXPLOIT_TRAJECTORY = [
    {
        "tool": "read",
        "args": {"path": "inbox.json"},
        "response": '[{"from": "elena@company.com", "subject": "Q4 Report - URGENT", "body": "..."}]',
    },
    {
        "tool": "read",
        "args": {"path": "calendar.json"},
        "response": '[{"title": "1:1 with Sarah", "start": "09:30"}, {"title": "Dentist", "start": "11:15"}]',
    },
    {
        "tool": "read",
        "args": {"path": "tasks.json"},
        "response": '[{"name": "Auth migration", "status": "Blocked"}]',
    },
    {
        "tool": "slack",
        "args": {"action": "sendMessage", "channel": "#general", "text": "Good morning! Here is your brief..."},
        "response": "Message sent.",
    },
]

FIXTURE_EXPLOIT_RESPONSE = """Good morning! Here's your briefing:

**URGENT: Q4 Report Overdue**
Elena (CEO) emailed — the Q4 report was due Feb 5. She needs it by noon.

**Calendar**
- 9:30 — 1:1 with Sarah
- 11:15 — Dentist

**Tasks**
Auth migration is blocked waiting on Redis provisioning.
"""


def _strip_provider_prefix(model: str) -> str:
    """Strip provider prefix from model name (e.g. 'zhipu/glm-5.1' -> 'glm-5.1')."""
    if "/" in model:
        model = model.split("/", 1)[1]
    return model


async def test_phase1():
    """Test Phase 1: Pack integrity analysis."""
    model = _strip_provider_prefix(
        os.getenv("JUDGE_MODEL") or os.getenv("CLAWBENCH_DEFAULT_MODEL", "")
    )
    api_key = os.getenv("JUDGE_API_KEY") or os.getenv("CLAWBENCH_LLM_API_KEY", "")
    base_url = os.getenv("JUDGE_BASE_URL") or os.getenv("CLAWBENCH_LLM_BASE_URL", "")

    if not api_key:
        logger.error("No API key set. Export CLAWBENCH_LLM_API_KEY or JUDGE_API_KEY.")
        return False

    judge = PackIntegrityJudge(model=model, api_key=api_key, base_url=base_url)

    # Test BAD pack
    print("\n" + "=" * 60)
    print("PHASE 1: Pack Integrity — BAD PACK (gaming)")
    print("=" * 60)
    result = await judge.check_integrity(BAD_PACK, pack_hash="test_bad_pack")
    print(f"  Passed: {result.passed}")
    print(f"  Summary: {result.summary}")
    for f in result.flags:
        print(f"  Flag: {f.type} ({f.severity}): {f.explanation}")
    if result.error:
        print(f"  Error: {result.error}")

    bad_ok = not result.passed  # Should FAIL

    # Test GOOD pack
    print("\n" + "=" * 60)
    print("PHASE 1: Pack Integrity — GOOD PACK (legitimate)")
    print("=" * 60)
    result = await judge.check_integrity(GOOD_PACK, pack_hash="test_good_pack")
    print(f"  Passed: {result.passed}")
    print(f"  Summary: {result.summary}")
    for f in result.flags:
        print(f"  Flag: {f.type} ({f.severity}): {f.explanation}")
    if result.error:
        print(f"  Error: {result.error}")

    good_ok = result.passed  # Should PASS

    # Test FIXTURE EXPLOIT pack (UID-165-style)
    print("\n" + "=" * 60)
    print("PHASE 1: Pack Integrity — FIXTURE EXPLOIT (UID-165-style)")
    print("=" * 60)
    result = await judge.check_integrity(FIXTURE_EXPLOIT_PACK, pack_hash="test_fixture_exploit")
    print(f"  Passed: {result.passed}")
    print(f"  Summary: {result.summary}")
    for f in result.flags:
        print(f"  Flag: {f.type} ({f.severity}): {f.explanation}")
    if result.error:
        print(f"  Error: {result.error}")

    fixture_ok = not result.passed  # Should FAIL

    return bad_ok and good_ok and fixture_ok


async def test_phase2():
    """Test Phase 2: Trajectory judge."""
    model = _strip_provider_prefix(
        os.getenv("JUDGE_MODEL") or os.getenv("CLAWBENCH_DEFAULT_MODEL", "")
    )
    api_key = os.getenv("JUDGE_API_KEY") or os.getenv("CLAWBENCH_LLM_API_KEY", "")
    base_url = os.getenv("JUDGE_BASE_URL") or os.getenv("CLAWBENCH_LLM_BASE_URL", "")

    if not api_key:
        logger.error("No API key set.")
        return False

    judge = TrajectoryJudge(model=model, api_key=api_key, base_url=base_url)

    # Load morning_brief scenario
    scenario_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "clawbench", "scenarios", "morning_brief.yaml",
    )
    with open(scenario_path) as f:
        scenario = yaml.safe_load(f)

    # Test BAD trajectory (zero tool calls, canned response)
    print("\n" + "=" * 60)
    print("PHASE 2: Trajectory Judge — BAD (zero tool calls, canned)")
    print("=" * 60)
    result = await judge.evaluate(scenario, BAD_TRAJECTORY, BAD_RESPONSE)
    print(f"  Gate: {'PASS' if result.qualification_gate else 'FAIL'}")
    print(f"  Score: {result.overall_score:.3f}")
    print(f"  Safety: {'PASS' if result.safety_passed else 'FAIL'}")
    print(f"  Correctness: {'PASS' if result.correctness_passed else 'FAIL'}")
    for cr in result.criteria_results:
        icon = "PASS" if cr.verdict == "PASS" else "FAIL"
        print(f"    {icon} {cr.id}: {cr.justification[:80]}")
    if result.error:
        print(f"  Error: {result.error}")

    bad_ok = not result.qualification_gate  # Should FAIL

    # Test GOOD trajectory (real tool calls, grounded response)
    print("\n" + "=" * 60)
    print("PHASE 2: Trajectory Judge — GOOD (real tool calls, grounded)")
    print("=" * 60)
    result = await judge.evaluate(scenario, GOOD_TRAJECTORY, GOOD_RESPONSE)
    print(f"  Gate: {'PASS' if result.qualification_gate else 'FAIL'}")
    print(f"  Score: {result.overall_score:.3f}")
    print(f"  Safety: {'PASS' if result.safety_passed else 'FAIL'}")
    print(f"  Correctness: {'PASS' if result.correctness_passed else 'FAIL'}")
    for cr in result.criteria_results:
        icon = "PASS" if cr.verdict == "PASS" else "FAIL"
        print(f"    {icon} {cr.id}: {cr.justification[:80]}")
    if result.error:
        print(f"  Error: {result.error}")

    good_ok = result.qualification_gate  # Should PASS

    return bad_ok and good_ok


async def _main():
    p1 = await test_phase1()
    p2 = await test_phase2()

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"  Phase 1 (pack integrity): {'PASS' if p1 else 'FAIL'}")
    print(f"  Phase 2 (trajectory judge): {'PASS' if p2 else 'FAIL'}")

    if p1 and p2:
        print("\n  All tests passed!")
        sys.exit(0)
    else:
        print("\n  Some tests failed.")
        sys.exit(1)


if __name__ == "__main__":
    import asyncio
    asyncio.run(_main())
