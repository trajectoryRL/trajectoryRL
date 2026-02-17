"""Epoch context generation for evaluation variation.

Each epoch generates a unique evaluation context (persona, date, company, etc.)
from the deterministic epoch seed.  This context is prepended to the miner's
AGENTS.md before evaluation, and converted to a user_context dict that
ClawBench uses for {{PLACEHOLDER}} template substitution in USER.md.

Variation space: ~35 million unique contexts
(365 dates × 20 names × 10 roles × 10 companies × 8 departments × 6 timezones)
"""

import random
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Dict

# ---------------------------------------------------------------
# Pools — epoch seed selects one element from each pool
# ---------------------------------------------------------------

NAMES = [
    "Jordan Rivera", "Alex Chen", "Sam Patel", "Morgan Kim",
    "Casey Williams", "Riley Johnson", "Dakota Lee", "Quinn Martinez",
    "Avery Thompson", "Jamie Garcia", "Taylor Nguyen", "Drew Campbell",
    "Reese Foster", "Skyler Brooks", "Sage Mitchell", "Parker Adams",
    "Hayden Scott", "Cameron Blake", "Logan Reeves", "Blake Torres",
]

ROLES = [
    "Product Manager", "Engineering Lead", "Senior Developer",
    "Design Lead", "Marketing Manager", "Operations Director",
    "Team Lead", "Project Manager", "Technical Architect",
    "Data Science Lead",
]

COMPANIES = [
    "Meridian Technologies", "Vertex Labs", "Cascade Systems",
    "Pinnacle Software", "Atlas Digital", "Quantum Dynamics",
    "Summit Innovations", "Harbor Analytics", "Crestline Solutions",
    "Brightpath Engineering",
]

DEPARTMENTS = [
    "Engineering", "Product", "Marketing", "Operations",
    "Design", "Sales", "Customer Success", "Research",
]

TIMEZONES = [
    ("America/New_York", "ET"),
    ("America/Chicago", "CT"),
    ("America/Denver", "MT"),
    ("America/Los_Angeles", "PT"),
    ("Europe/London", "GMT"),
    ("Asia/Tokyo", "JST"),
]


@dataclass
class EpochContext:
    """Epoch-specific evaluation context generated from the epoch seed."""

    date_str: str       # "February 17, 2026"
    weekday: str        # "Tuesday"
    user_name: str      # "Jordan Rivera"
    user_role: str      # "Product Manager"
    company: str        # "Meridian Technologies"
    department: str     # "Engineering"
    timezone: str       # "America/New_York"
    timezone_abbr: str  # "ET"

    def to_user_context(self) -> Dict[str, str]:
        """Convert to a user_context dict for ClawBench {{PLACEHOLDER}} substitution.

        Returns:
            Dict with keys matching template variables in USER.md fixtures:
            USER_NAME, USER_FIRST_NAME, USER_ROLE, COMPANY.
        """
        return {
            "USER_NAME": self.user_name,
            "USER_FIRST_NAME": self.user_name.split()[0],
            "USER_ROLE": self.user_role,
            "COMPANY": self.company,
        }


def generate_epoch_context(epoch_seed: int) -> EpochContext:
    """Generate a deterministic evaluation context from the epoch seed.

    All validators compute the same context for the same epoch seed,
    ensuring consistent evaluation conditions.

    Args:
        epoch_seed: Deterministic seed from compute_epoch_seed()

    Returns:
        EpochContext with persona, date, and environment details
    """
    rng = random.Random(epoch_seed)

    # Pick a date in 2026 (day 0-364)
    base = date(2026, 1, 1)
    day_offset = rng.randint(0, 364)
    d = base + timedelta(days=day_offset)

    tz_name, tz_abbr = rng.choice(TIMEZONES)

    return EpochContext(
        date_str=d.strftime("%B %d, %Y"),
        weekday=d.strftime("%A"),
        user_name=rng.choice(NAMES),
        user_role=rng.choice(ROLES),
        company=rng.choice(COMPANIES),
        department=rng.choice(DEPARTMENTS),
        timezone=tz_name,
        timezone_abbr=tz_abbr,
    )


def render_context_preamble(ctx: EpochContext) -> str:
    """Render epoch context as a markdown preamble for AGENTS.md.

    This block is prepended to the miner's AGENTS.md so the agent
    operates under a specific persona and date each epoch.  Policies
    that hardcode identity details will conflict with this preamble
    and score poorly.

    Args:
        ctx: EpochContext to render

    Returns:
        Markdown string to prepend to AGENTS.md
    """
    return (
        f"<!-- Epoch Evaluation Context — generated per epoch, do not hardcode -->\n"
        f"> **Date**: {ctx.weekday}, {ctx.date_str}  \n"
        f"> **Your Name**: {ctx.user_name}  \n"
        f"> **Role**: {ctx.user_role} at {ctx.company}  \n"
        f"> **Department**: {ctx.department}  \n"
        f"> **Timezone**: {ctx.timezone} ({ctx.timezone_abbr})\n"
        f"\n"
        f"---\n"
        f"\n"
    )
