"""Tests for queue-admission dedup helpers (trajectoryrl.utils.ncd).

``queue_duplicates`` decides, for a set of packs waiting in the challenger
queue, which ones are near-copies of an earlier entry. It keys on submission
time and content only, so the outcome does not change with the number of
hotkeys or coldkeys behind the submissions.
"""

import sys
from unittest.mock import MagicMock

# Mock bittensor before importing trajectoryrl modules (some imports trigger it)
if "bittensor" not in sys.modules:
    _mock_bt = MagicMock()

    class _MockSynapse:
        pass

    _mock_bt.Synapse = _MockSynapse
    sys.modules["bittensor"] = _mock_bt

from trajectoryrl.utils.ncd import (  # noqa: E402
    SIMILARITY_THRESHOLD,
    cluster_packs,
    pack_similarity,
    queue_duplicates,
)


def _pack(content: str) -> dict:
    return {"files": {"SKILL.md": content}}


# Two documents that give the same advice in different words and structure.
DOC_A = """
# Ship the file

Your work is judged by running the artifact at the path the task names. Write a
minimal valid version there before you investigate anything, then improve it in
place. Run the task's own command from a neutral directory, twice, and copy the
deliverable alone into an empty directory to confirm it still loads. Assume no
network: check that a library is present before building on it. Wrap anything
that could hang in a timeout, and send large output to a file rather than into
the conversation.
"""

DOC_B = """
# A method for timed engineering work

Treat the request as a handover. Begin by writing down every named output, the
exact command that will check it, and any stated limit. Only then look at what
the environment already provides, and extend it rather than replacing it. Prove
a result with the consumer that will actually read it: a parser, a compiler, a
probe against the real port. Measure a threshold instead of judging it by eye,
and keep raising effort until you clear the number with margin.
"""

# A per-wallet edit of DOC_A: the hash changes, the document does not.
DOC_A_EDITED = DOC_A.replace("Ship the file", "Ship the file first").replace(
    "in\nplace", "in place"
) + "\n<!-- 8f2a11c4 -->\n"


def test_near_copy_of_earlier_entry_is_flagged():
    """A lightly edited resubmission loses to the entry that arrived first."""
    assert pack_similarity(_pack(DOC_A), _pack(DOC_A_EDITED)) >= SIMILARITY_THRESHOLD

    dupes = queue_duplicates(
        [
            ("first", _pack(DOC_A), 100.0),
            ("second", _pack(DOC_A_EDITED), 200.0),
        ]
    )

    assert dupes == {"second": "first"}


def test_distinct_documents_both_keep_their_slot():
    dupes = queue_duplicates(
        [
            ("a", _pack(DOC_A), 100.0),
            ("b", _pack(DOC_B), 200.0),
        ]
    )

    assert dupes == {}


def test_earliest_submission_wins_regardless_of_input_order():
    """Ordering is decided by submitted_at, not by the order passed in."""
    entries = [
        ("late", _pack(DOC_A_EDITED), 900.0),
        ("early", _pack(DOC_A), 100.0),
    ]

    assert queue_duplicates(entries) == {"late": "early"}
    assert queue_duplicates(list(reversed(entries))) == {"late": "early"}


def test_identity_does_not_buy_extra_slots():
    """One document across many submissions keeps exactly one slot.

    This is the property coldkey- or hotkey-counting cannot provide: the caller
    may spread the same document over any number of identities, and the result
    is unchanged.
    """
    entries = [
        (f"sub{i}", _pack(DOC_A_EDITED.replace("8f2a11c4", f"{i:08x}")), float(i))
        for i in range(8)
    ]
    entries.append(("other", _pack(DOC_B), 99.0))

    dupes = queue_duplicates(entries)

    kept = {k for k, _, _ in entries} - set(dupes)
    assert kept == {"sub0", "other"}
    assert all(v == "sub0" for k, v in dupes.items())


def test_empty_and_single_entry_queues():
    assert queue_duplicates([]) == {}
    assert queue_duplicates([("only", _pack(DOC_A), 1.0)]) == {}


def test_malformed_packs_are_skipped_not_raised():
    dupes = queue_duplicates(
        [
            ("broken", {"files": {}}, 1.0),
            ("also_broken", None, 2.0),
            ("good", _pack(DOC_A), 3.0),
        ]
    )

    assert "good" not in dupes


def test_threshold_is_respected():
    entries = [("a", _pack(DOC_A), 1.0), ("b", _pack(DOC_B), 2.0)]

    assert queue_duplicates(entries, threshold=0.0) == {"b": "a"}
    assert queue_duplicates(entries, threshold=1.01) == {}


def test_cluster_packs_groups_near_copies():
    packs = {
        "a1": _pack(DOC_A),
        "a2": _pack(DOC_A_EDITED),
        "b1": _pack(DOC_B),
    }

    clusters = cluster_packs(packs)

    assert len(clusters) == 2
    assert sorted(clusters[0]) == ["a1", "a2"]
    assert clusters[1] == ["b1"]


def test_cluster_packs_is_deterministic():
    packs = {f"k{i}": _pack(DOC_A_EDITED.replace("8f2a11c4", f"{i:08x}")) for i in range(5)}
    packs["distinct"] = _pack(DOC_B)

    assert cluster_packs(packs) == cluster_packs(dict(reversed(list(packs.items()))))
