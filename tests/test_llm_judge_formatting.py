"""Tests for LLM judge trajectory formatting helpers.

Covers every branch of _clean_response, _format_args_md,
_format_response_md, and _format_trajectory — the Markdown
formatting pipeline that converts raw tool responses into
compact, readable text for the LLM judge.

No LLM calls. No network. Pure unit tests.
"""

import json
import sys
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# Mock bittensor before importing trajectoryrl
# ---------------------------------------------------------------------------
_mock_bt = MagicMock()
class _MockSynapse:
    pass
_mock_bt.Synapse = _MockSynapse
sys.modules.setdefault("bittensor", _mock_bt)

from trajectoryrl.utils.llm_judge import TrajectoryJudge, MAX_TOOL_RESPONSE_CHARS


# ---------------------------------------------------------------------------
# Reusable fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def judge():
    return TrajectoryJudge()


# --- Realistic tool response shapes (from mock_tools/server.py) ---

EXEC_HIMALAYA_RESPONSE = {
    "status": "completed",
    "exitCode": 0,
    "durationMs": 42,
    "aggregated": '[{"id": 1, "from": "alice@co.com", "subject": "Meeting"}]',
}

EXEC_GCALCLI_RESPONSE = {
    "status": "completed",
    "exitCode": 0,
    "durationMs": 42,
    "aggregated": json.dumps({
        "items": [
            {"id": "evt_1", "title": "Standup", "start": "2026-02-06T09:00:00-08:00",
             "end": "2026-02-06T09:30:00-08:00", "location": "Zoom",
             "notes": "Daily standup"},
            {"id": "evt_2", "title": "Sprint ends", "start": "2026-02-07T17:00:00-08:00",
             "end": "2026-02-07T17:00:00-08:00", "location": "",
             "notes": ""},
        ]
    }),
}

EXEC_NOTION_RESPONSE = {
    "status": "completed",
    "exitCode": 0,
    "durationMs": 42,
    "aggregated": json.dumps({
        "results": [
            {"id": "TC-880", "title": "Auth migration", "status": "in_progress",
             "priority": "critical", "assignee": "marcus", "notes": "Blocked on Redis"},
            {"id": "TC-931", "title": "Write postmortem", "status": "open",
             "priority": "medium", "assignee": "marcus", "notes": "Due today"},
        ]
    }),
}

EXEC_PLAIN_TEXT_RESPONSE = {
    "status": "completed",
    "exitCode": 0,
    "durationMs": 42,
    "aggregated": "commit abc123\nAuthor: alice\nDate: 2026-02-05\n\n    Fix bug #42",
}

SLACK_RESPONSE = {
    "ok": True,
    "messages": [
        {"id": "sm_001", "channel": "#engineering", "author": "alice",
         "text": "PR is up", "timestamp": "2026-02-05T17:45:00-08:00"},
        {"id": "sm_002", "channel": "#incidents", "author": "sentry-bot",
         "text": "Error rate 12%", "timestamp": "2026-02-05T22:20:00-08:00"},
        {"id": "sm_003", "channel": "#engineering", "author": "bob",
         "text": "LGTM", "timestamp": "2026-02-05T18:00:00-08:00"},
    ],
}

WEB_SEARCH_RESPONSE = {
    "provider": "google",
    "tookMs": 150,
    "cached": False,
    "results": [
        {"title": "Result 1", "url": "https://example.com", "snippet": "Some text"},
    ],
}

MEMORY_SEARCH_RESPONSE = {
    "provider": "local",
    "tookMs": 5,
    "results": [
        {"key": "project_goals", "value": "Ship v2 by March"},
    ],
}

READ_RESPONSE = {
    "path": "/workspace/README.md",
    "content": "# Project\nThis is a project.",
}


# =========================================================================
# _clean_response
# =========================================================================

class TestCleanResponse:
    """Every branch of _clean_response."""

    def test_non_dict_passthrough(self, judge):
        """Non-dict responses (string, list, int, None) pass through unchanged."""
        assert judge._clean_response("exec", {}, "plain text") == "plain text"
        assert judge._clean_response("exec", {}, [1, 2, 3]) == [1, 2, 3]
        assert judge._clean_response("exec", {}, 42) == 42
        assert judge._clean_response("exec", {}, None) is None

    def test_exec_parses_json_aggregated(self, judge):
        """exec: aggregated JSON string → parsed Python object."""
        result = judge._clean_response("exec", {}, EXEC_HIMALAYA_RESPONSE)
        assert isinstance(result, list)
        assert result[0]["from"] == "alice@co.com"

    def test_exec_parses_nested_dict(self, judge):
        """exec: aggregated with nested items/results → parsed dict."""
        result = judge._clean_response("exec", {}, EXEC_GCALCLI_RESPONSE)
        assert isinstance(result, dict)
        assert "items" in result
        assert len(result["items"]) == 2

    def test_exec_invalid_json_returns_string(self, judge):
        """exec: aggregated is non-JSON text → returned as string."""
        result = judge._clean_response("exec", {}, EXEC_PLAIN_TEXT_RESPONSE)
        assert isinstance(result, str)
        assert "commit abc123" in result

    def test_exec_empty_aggregated(self, judge):
        """exec: aggregated is empty string → empty string."""
        resp = {"status": "completed", "exitCode": 0, "durationMs": 1, "aggregated": ""}
        result = judge._clean_response("exec", {}, resp)
        assert result == ""

    def test_exec_aggregated_not_string(self, judge):
        """exec: aggregated is already a dict (unlikely but handled)."""
        resp = {"status": "completed", "aggregated": {"already": "parsed"}}
        result = judge._clean_response("exec", {}, resp)
        assert result == {"already": "parsed"}

    def test_exec_no_aggregated_key(self, judge):
        """exec: response dict without aggregated → empty string."""
        resp = {"status": "completed", "exitCode": 1, "durationMs": 5}
        result = judge._clean_response("exec", {}, resp)
        assert result == ""

    def test_exec_strips_wrapper_keys(self, judge):
        """exec: wrapper keys (status, exitCode, durationMs) are NOT in output."""
        result = judge._clean_response("exec", {}, EXEC_GCALCLI_RESPONSE)
        assert "status" not in result
        assert "exitCode" not in result
        assert "durationMs" not in result

    def test_slack_extracts_messages(self, judge):
        """slack: {ok, messages} → just the messages list."""
        result = judge._clean_response("slack", {}, SLACK_RESPONSE)
        assert isinstance(result, list)
        assert len(result) == 3
        assert result[0]["author"] == "alice"

    def test_slack_no_messages_key(self, judge):
        """slack: response without messages key → returned as-is."""
        resp = {"ok": True, "error": "channel_not_found"}
        result = judge._clean_response("slack", {}, resp)
        assert result == resp

    def test_web_search_extracts_results(self, judge):
        """web_search: strips provider/tookMs/cached, keeps results."""
        result = judge._clean_response("web_search", {}, WEB_SEARCH_RESPONSE)
        assert isinstance(result, list)
        assert result[0]["title"] == "Result 1"

    def test_memory_search_extracts_results(self, judge):
        """memory_search: strips provider/tookMs, keeps results."""
        result = judge._clean_response("memory_search", {}, MEMORY_SEARCH_RESPONSE)
        assert isinstance(result, list)
        assert result[0]["key"] == "project_goals"

    def test_web_search_no_results(self, judge):
        """web_search without results key → returned as-is."""
        resp = {"provider": "google", "error": "timeout"}
        result = judge._clean_response("web_search", {}, resp)
        assert result == resp

    def test_read_extracts_content(self, judge):
        """read: {path, content} → just the content string."""
        result = judge._clean_response("read", {}, READ_RESPONSE)
        assert result == "# Project\nThis is a project."

    def test_read_no_content(self, judge):
        """read: response without content key → returned as-is."""
        resp = {"error": "file not found"}
        result = judge._clean_response("read", {}, resp)
        assert result == resp

    def test_unknown_tool_passthrough(self, judge):
        """Unknown tool → dict returned unchanged."""
        resp = {"custom": "data", "other": 123}
        result = judge._clean_response("unknown_tool", {}, resp)
        assert result == resp


# =========================================================================
# _format_args_md
# =========================================================================

class TestFormatArgsMd:
    """Every branch of _format_args_md."""

    def test_exec_shows_command(self, judge):
        assert judge._format_args_md("exec", {"command": "himalaya list", "timeout": 30000}) == "himalaya list"

    def test_exec_no_command_key(self, judge):
        result = judge._format_args_md("exec", {"timeout": 5000})
        assert "timeout" in result  # falls back to str(args)

    def test_slack_shows_action(self, judge):
        assert judge._format_args_md("slack", {"action": "readMessages"}) == "readMessages"

    def test_read_shows_path(self, judge):
        assert judge._format_args_md("read", {"path": "/workspace/notes.md"}) == "/workspace/notes.md"

    def test_memory_search_shows_query(self, judge):
        assert judge._format_args_md("memory_search", {"query": "project goals"}) == "project goals"

    def test_web_search_shows_query(self, judge):
        assert judge._format_args_md("web_search", {"query": "python docs"}) == "python docs"

    def test_web_fetch_shows_url(self, judge):
        assert judge._format_args_md("web_fetch", {"url": "https://example.com"}) == "https://example.com"

    def test_memory_get_shows_key(self, judge):
        assert judge._format_args_md("memory_get", {"key": "user_prefs"}) == "user_prefs"

    def test_unknown_tool_json(self, judge):
        result = judge._format_args_md("custom_tool", {"foo": "bar", "n": 1})
        parsed = json.loads(result)
        assert parsed == {"foo": "bar", "n": 1}

    def test_empty_args(self, judge):
        result = judge._format_args_md("exec", {})
        assert result == str({})  # "command" key missing → str(args)


# =========================================================================
# _format_response_md
# =========================================================================

class TestFormatResponseMd:
    """Every branch of _format_response_md."""

    # --- String inputs ---

    def test_string_nonempty(self, judge):
        assert judge._format_response_md("exec", {}, "hello world") == "hello world"

    def test_string_empty(self, judge):
        assert judge._format_response_md("exec", {}, "") == "(empty)"

    # --- Non-dict/non-list inputs ---

    def test_integer(self, judge):
        assert judge._format_response_md("exec", {}, 42) == "42"

    def test_none(self, judge):
        assert judge._format_response_md("exec", {}, None) == "None"

    def test_boolean(self, judge):
        assert judge._format_response_md("exec", {}, True) == "True"

    # --- exec: calendar items ---

    def test_calendar_items(self, judge):
        data = {
            "items": [
                {"title": "Standup", "start": "9:00", "location": "Zoom", "notes": "Daily"},
                {"title": "Retro", "start": "15:00", "location": "", "notes": ""},
            ]
        }
        result = judge._format_response_md("exec", {}, data)
        assert "- **Standup** | 9:00 | Zoom |" in result
        assert '"Daily"' in result
        # No location or notes for Retro → not included
        assert "- **Retro** | 15:00" in result
        assert result.count("- **") == 2

    def test_calendar_empty_items(self, judge):
        data = {"items": []}
        assert judge._format_response_md("exec", {}, data) == "(empty)"

    def test_calendar_no_optional_fields(self, judge):
        """Calendar item with only title and start (no location, no notes)."""
        data = {"items": [{"title": "Meeting", "start": "10:00"}]}
        result = judge._format_response_md("exec", {}, data)
        assert "**Meeting** | 10:00" in result
        # Should NOT have trailing pipes for empty fields
        assert '""' not in result

    # --- exec: notion/sprint tasks ---

    def test_notion_tasks(self, judge):
        data = {
            "results": [
                {"id": "TC-100", "title": "Fix bug", "status": "open",
                 "priority": "high", "assignee": "alice", "notes": "Urgent"},
                {"id": "TC-200", "title": "Add test", "status": "done",
                 "priority": "low", "assignee": "bob", "notes": ""},
            ]
        }
        result = judge._format_response_md("exec", {}, data)
        assert "**TC-100** Fix bug" in result
        assert "open" in result
        assert "high" in result
        assert "alice" in result
        assert '"Urgent"' in result
        assert "**TC-200** Add test" in result
        # TC-200 has no notes → no trailing quote
        lines = result.split("\n")
        tc200_line = [l for l in lines if "TC-200" in l][0]
        assert '""' not in tc200_line

    def test_notion_empty_results(self, judge):
        data = {"results": []}
        assert judge._format_response_md("exec", {}, data) == "(empty)"

    def test_notion_minimal_task(self, judge):
        """Task with only id — all other fields missing."""
        data = {"results": [{"id": "TC-999"}]}
        result = judge._format_response_md("exec", {}, data)
        assert "**TC-999**" in result

    # --- slack: grouped by channel ---

    def test_slack_grouped_by_channel(self, judge):
        data = [
            {"channel": "#eng", "author": "alice", "text": "Hello",
             "timestamp": "2026-02-05T17:45:00-08:00"},
            {"channel": "#incidents", "author": "bot", "text": "Alert",
             "timestamp": "2026-02-05T22:20:00-08:00"},
            {"channel": "#eng", "author": "bob", "text": "Hi",
             "timestamp": "2026-02-05T18:00:00-08:00"},
        ]
        result = judge._format_response_md("slack", {}, data)
        # Channels are sorted alphabetically
        eng_pos = result.index("#eng")
        inc_pos = result.index("#incidents")
        assert eng_pos < inc_pos
        # Messages grouped under their channel
        assert "alice" in result
        assert "bob" in result
        assert "bot" in result

    def test_slack_timestamp_shortening(self, judge):
        """ISO timestamps are shortened to human-readable format."""
        data = [
            {"channel": "#test", "author": "user", "text": "msg",
             "timestamp": "2026-02-05T17:45:00-08:00"},
        ]
        result = judge._format_response_md("slack", {}, data)
        # Should be shortened (no full ISO string)
        assert "2026-02-05T17:45:00-08:00" not in result
        # Should contain shortened form
        assert "feb" in result.lower()

    def test_slack_bad_timestamp(self, judge):
        """Invalid timestamp falls back to original string."""
        data = [
            {"channel": "#test", "author": "user", "text": "msg",
             "timestamp": "not-a-date"},
        ]
        result = judge._format_response_md("slack", {}, data)
        assert "not-a-date" in result

    def test_slack_no_timestamp(self, judge):
        """Missing timestamp → empty string."""
        data = [
            {"channel": "#test", "author": "user", "text": "msg"},
        ]
        result = judge._format_response_md("slack", {}, data)
        assert "user" in result
        assert "msg" in result

    def test_slack_empty_list(self, judge):
        """Empty message list → falls through to compact JSON."""
        result = judge._format_response_md("slack", {}, [])
        assert result == "[]"

    def test_slack_single_channel(self, judge):
        data = [
            {"channel": "#general", "author": "a", "text": "1", "timestamp": ""},
            {"channel": "#general", "author": "b", "text": "2", "timestamp": ""},
        ]
        result = judge._format_response_md("slack", {}, data)
        assert result.count("**#general:**") == 1  # one header
        assert result.count("- ") == 2  # two messages

    # --- generic list of dicts (web_search, memory_search results) ---

    def test_generic_list_of_dicts(self, judge):
        """Cleaned web_search/memory_search results → key=value bullet list."""
        data = [
            {"title": "Doc 1", "url": "https://a.com", "provider": "google", "tookMs": 5},
            {"title": "Doc 2", "url": "https://b.com", "cached": True},
        ]
        # Not slack (no "channel" key) → hits the generic list-of-dicts branch
        result = judge._format_response_md("web_search", {}, data)
        assert "title=Doc 1" in result
        assert "url=https://a.com" in result
        # provider/tookMs/cached should be stripped
        assert "provider=" not in result
        assert "tookMs=" not in result
        assert "cached=" not in result

    # --- fallback: compact JSON ---

    def test_fallback_dict(self, judge):
        """Dict that doesn't match any pattern → compact JSON."""
        data = {"custom_key": "custom_value", "number": 42}
        result = judge._format_response_md("exec", {}, data)
        parsed = json.loads(result)
        assert parsed == data

    def test_fallback_nested(self, judge):
        """Nested dict → compact JSON."""
        data = {"outer": {"inner": [1, 2, 3]}}
        result = judge._format_response_md("exec", {}, data)
        parsed = json.loads(result)
        assert parsed == data

    # --- exec items/results: not triggered for non-exec tools ---

    def test_non_exec_with_items_key(self, judge):
        """A non-exec tool with 'items' key doesn't trigger calendar formatter."""
        data = {"items": [{"title": "X", "start": "10:00"}]}
        result = judge._format_response_md("slack", {}, data)
        # Should fall through to compact JSON, not calendar format
        parsed = json.loads(result)
        assert parsed == data


# =========================================================================
# _format_trajectory (integration)
# =========================================================================

class TestFormatTrajectory:
    """End-to-end tests for the full formatting pipeline."""

    def test_empty_trajectory(self, judge):
        result = judge._format_trajectory([])
        assert "No tool calls" in result

    def test_single_call(self, judge):
        trajectory = [
            {"tool": "exec", "args": {"command": "ls"}, "response": EXEC_PLAIN_TEXT_RESPONSE},
        ]
        result = judge._format_trajectory(trajectory)
        assert "### Call 1: exec" in result
        assert "**ls**" in result
        assert "commit abc123" in result

    def test_pipeline_strips_then_formats(self, judge):
        """Full pipeline: clean → format for each tool type."""
        trajectory = [
            {"tool": "exec", "args": {"command": "gcalcli agenda"},
             "response": EXEC_GCALCLI_RESPONSE},
            {"tool": "exec", "args": {"command": "curl notion"},
             "response": EXEC_NOTION_RESPONSE},
            {"tool": "slack", "args": {"action": "readMessages"},
             "response": SLACK_RESPONSE},
        ]
        result = judge._format_trajectory(trajectory)
        # Calendar formatted as bullets
        assert "**Standup**" in result
        # Notion tasks formatted as bullets
        assert "**TC-880**" in result
        assert "**TC-931**" in result
        # Slack grouped by channel
        assert "#engineering" in result or "#incidents" in result
        # Wrapper keys stripped
        assert "exitCode" not in result
        assert "durationMs" not in result
        assert '"ok": true' not in result.lower()

    def test_truncation_preserves_head_and_tail(self, judge):
        """Responses exceeding MAX_TOOL_RESPONSE_CHARS keep head + tail."""
        # Create a response that will exceed the limit
        big_messages = []
        for i in range(100):
            big_messages.append({
                "channel": "#test",
                "author": f"user_{i}",
                "text": f"Message number {i} with some padding text to make it longer " * 3,
                "timestamp": f"2026-02-05T{10+i%12}:00:00-08:00",
            })
        trajectory = [
            {"tool": "slack", "args": {"action": "readMessages"},
             "response": {"ok": True, "messages": big_messages}},
        ]
        result = judge._format_trajectory(trajectory)
        assert "[truncated" in result
        # Head content present
        assert "user_0" in result
        # Tail content present (last messages)
        assert "user_99" in result

    def test_non_dict_args_handled(self, judge):
        """Tool call with non-dict args doesn't crash."""
        trajectory = [
            {"tool": "exec", "args": "raw string args", "response": "some output"},
        ]
        result = judge._format_trajectory(trajectory)
        assert "### Call 1: exec" in result

    def test_missing_keys_handled(self, judge):
        """Tool call with missing tool/args/response keys doesn't crash."""
        trajectory = [
            {},  # completely empty
            {"tool": "exec"},  # missing args and response
        ]
        result = judge._format_trajectory(trajectory)
        assert "Call 1:" in result
        assert "Call 2:" in result

    def test_overall_trajectory_truncation(self, judge):
        """Trajectory exceeding MAX_TRAJECTORY_CHARS is truncated."""
        # Create many tool calls to exceed 60K chars
        trajectory = []
        for i in range(200):
            trajectory.append({
                "tool": "exec",
                "args": {"command": f"echo {'x' * 200}"},
                "response": {"status": "completed", "exitCode": 0,
                             "durationMs": 1, "aggregated": "x" * 500},
            })
        result = judge._format_trajectory(trajectory)
        assert "[trajectory truncated]" in result

    def test_call_numbering(self, judge):
        """Calls are numbered sequentially."""
        trajectory = [
            {"tool": "exec", "args": {"command": "cmd1"}, "response": {"aggregated": "out1", "status": "completed"}},
            {"tool": "slack", "args": {"action": "read"}, "response": {"ok": True, "messages": []}},
            {"tool": "read", "args": {"path": "/file"}, "response": {"content": "data"}},
        ]
        result = judge._format_trajectory(trajectory)
        assert "### Call 1: exec" in result
        assert "### Call 2: slack" in result
        assert "### Call 3: read" in result
