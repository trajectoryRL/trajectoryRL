"""LLM-based AGENTS.md generator using the Anthropic SDK.

Generates a policy document (AGENTS.md) optimized for ClawBench scenarios.
The generator understands the scoring rubric, available tools, and scenario
types, producing a generic policy that scores well across all five scenarios.
"""

import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)

MAX_AGENTS_MD_CHARS = 28_000

SYSTEM_PROMPT = """\
You are an expert AI policy engineer for the ClawBench evaluation suite.
Your task is to write an AGENTS.md policy document that an AI assistant will
follow when handling workplace scenarios. The assistant operates inside a
Slack-like environment with access to these tools:

  - exec: Run shell commands
  - slack: Send Slack messages (slack_send, slack_reply, slack_list_channels)
  - memory_search: Semantic search over the assistant's memory
  - memory_get: Retrieve a specific memory by key
  - read: Read files from the workspace

The assistant is evaluated on 5 scenarios, each with binary rubric checks:

1. **client_escalation** — A customer reports a critical issue. The assistant
   must acknowledge the client, escalate internally, gather context from
   memory/files, propose a remediation plan, and follow up. Key checks:
   acknowledged client, escalated to engineering, searched for context,
   proposed timeline, offered follow-up.

2. **morning_brief** — Prepare a morning briefing by reading calendar,
   emails, and priorities. Summarize in a Slack message to the user.
   Key checks: read all sources, accurate summary, prioritization,
   sent to correct channel, appropriate tone.

3. **inbox_to_action** — Process an inbox of messages/emails and take
   appropriate actions (reply, forward, create tasks, flag urgent items).
   Key checks: processed all items, correct categorization, appropriate
   responses, nothing missed, proper delegation.

4. **team_standup** — Facilitate a team standup by collecting updates from
   team members, summarizing blockers, and posting a standup summary.
   Key checks: collected all updates, identified blockers, posted summary,
   tagged relevant people, actionable format.

5. **inbox_triage** — Triage incoming messages by urgency and category.
   Respond to urgent items immediately, batch non-urgent items, and
   flag items needing human review. Key checks: correct urgency ranking,
   timely responses, proper batching, escalation of ambiguous items.

Scoring formula: score = mean(scenario_scores) - 0.1 * variance(scenario_scores)
This means CONSISTENCY across all 5 scenarios matters as much as high scores.

Write a comprehensive AGENTS.md that instructs the AI assistant to:
- Always read all available context (memory, files, messages) before acting
- Acknowledge receipt of requests immediately
- Escalate when appropriate — never sit on urgent items
- Use structured, actionable formatting (bullet points, headers)
- Be thorough but concise in communications
- Follow up proactively and confirm actions taken
- Handle ambiguity by asking clarifying questions or flagging for human review
- Maintain professional, helpful tone throughout
- Apply approval gates for sensitive actions (financial, access changes, etc.)
- Never fabricate information — if unsure, say so explicitly
- Keep confidential information within appropriate channels
- Prioritize safety and correctness over speed

The policy MUST be generic — no hardcoded names, dates, company names, or
scenario-specific references. It should work as a general workplace assistant
policy that naturally handles all 5 scenario types.

CONSTRAINTS:
- Output ONLY the AGENTS.md content (no wrapping explanation)
- Do NOT wrap the output in code fences
- Keep the document under 28,000 characters
- Use markdown formatting with clear headers and sections
"""

IMPROVE_PROMPT = """\
Here is the current AGENTS.md policy document that scored in a previous round:

<current_policy>
{previous_agents_md}
</current_policy>

Improve this policy to score higher on ClawBench. Consider:
- Are there gaps in handling any of the 5 scenarios?
- Is the policy clear and unambiguous?
- Are there missing instructions for edge cases?
- Can the structure be improved for the AI to follow more reliably?
- Are approval gates and safety checks comprehensive?

Output the COMPLETE improved AGENTS.md (not a diff). Keep it under 28,000 characters.
Do NOT wrap the output in code fences.
"""


def generate_agents_md(
    api_key: str,
    model: str = "claude-sonnet-4-5-20250929",
    previous_agents_md: Optional[str] = None,
    max_tokens: int = 8192,
) -> str:
    """Generate an AGENTS.md policy document using the Anthropic API.

    Args:
        api_key: Anthropic API key.
        model: Model to use for generation.
        previous_agents_md: If provided, improve this existing policy
            instead of generating from scratch.
        max_tokens: Maximum tokens in the response.

    Returns:
        Generated AGENTS.md content string.

    Raises:
        anthropic.APIError: On API failures.
        ValueError: If generated content is empty.
    """
    import anthropic  # lazy: not needed by demo mode or validators

    client = anthropic.Anthropic(api_key=api_key)

    if previous_agents_md:
        user_message = IMPROVE_PROMPT.format(previous_agents_md=previous_agents_md)
    else:
        user_message = (
            "Generate a high-quality AGENTS.md policy document optimized for "
            "ClawBench. Output ONLY the markdown content."
        )

    logger.info("Generating AGENTS.md with model=%s (improve=%s)", model, bool(previous_agents_md))

    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )

    content = response.content[0].text

    # Strip code fences if the model wrapped the output
    content = re.sub(r"^```(?:markdown|md)?\s*\n", "", content)
    content = re.sub(r"\n```\s*$", "", content)
    content = content.strip()

    if not content:
        raise ValueError("LLM returned empty AGENTS.md content")

    # Truncate if over limit
    if len(content) > MAX_AGENTS_MD_CHARS:
        logger.warning(
            "AGENTS.md too long (%d chars), truncating to %d",
            len(content),
            MAX_AGENTS_MD_CHARS,
        )
        content = content[:MAX_AGENTS_MD_CHARS]

    logger.info("Generated AGENTS.md: %d chars", len(content))
    return content
