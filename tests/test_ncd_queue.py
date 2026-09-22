"""Tests for queue-admission dedup helpers (trajectoryrl.utils.ncd).

``classify_queue`` / ``queue_duplicates`` decide, for packs waiting in the
challenger queue, which are near-copies of an earlier entry. They key on
submission time and content only, compare every sizeable file (not only
SKILL.md) with lzma, and flag a review band below the refusal threshold.
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
    MIN_FILE_BYTES,
    QUEUE_REFUSE_THRESHOLD,
    REFUSE,
    REVIEW,
    classify_queue,
    cluster_packs,
    queue_duplicates,
    queue_similarity,
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
    assert queue_similarity(_pack(DOC_A), _pack(DOC_A_EDITED)) >= QUEUE_REFUSE_THRESHOLD

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


# A realistic policy file, long enough to be fingerprinted.
POLICY = """
from trajrl_policy import Policy, serve

CHAIN = ["kimi-k3", "glm-5.3", "glm-5.3-flash"]


class Fusion(Policy):
    async def handle(self, req, ctx):
        remaining = ctx.remaining_usd if ctx.remaining_usd is not None else 1.0
        for model in CHAIN:
            if self.reservation(req, model) + 0.05 <= remaining:
                req["model"] = model
                break
        return await ctx.upstream(req)

    def reservation(self, req, model):
        return len(str(req.get("messages"))) / 3.0 * 2e-6 + 8192 * 1e-5


serve(Fusion())
"""


def _pack_with(skill: str, policy: str | None = None, policy_name: str = "policy.py") -> dict:
    files = {"SKILL.md": skill}
    if policy is not None:
        files[policy_name] = policy
    return {"files": files}


def test_copied_policy_under_reworded_prose_is_refused():
    """The evasion a SKILL.md-only fingerprint misses: keep the policy, reword the prose."""
    original = _pack_with(DOC_A, POLICY)
    reworded = _pack_with(DOC_B, POLICY)

    verdicts = classify_queue([("orig", original, 1.0), ("copy", reworded, 2.0)])

    assert verdicts["copy"][0] == REFUSE
    assert verdicts["copy"][1] == "orig"


def test_short_builtin_configs_do_not_count_as_copies():
    """Two packs sharing a tiny built-in policy.json are not copies of each other."""
    pin = '{"kind": "pin", "model": "kimi-k3"}'
    assert len(pin) < MIN_FILE_BYTES

    verdicts = classify_queue(
        [
            ("a", _pack_with(DOC_A, pin, "policy.json"), 1.0),
            ("b", _pack_with(DOC_B, pin, "policy.json"), 2.0),
        ]
    )

    assert verdicts == {}


def test_review_band_flags_without_refusing():
    """A score between the two thresholds keeps its slot but is flagged."""
    entries = [("a", _pack(DOC_A), 1.0), ("b", _pack(DOC_A_EDITED), 2.0)]
    similarity = queue_similarity(_pack(DOC_A), _pack(DOC_A_EDITED))

    verdicts = classify_queue(entries, refuse=similarity + 0.01, review=similarity - 0.01)

    assert verdicts["b"][0] == REVIEW
    assert "b" not in queue_duplicates(entries, threshold=similarity + 0.01)


def test_flagged_entry_still_keeps_its_slot_for_later_comparisons():
    """A reviewed entry is admitted, so a later copy of it is matched against it."""
    a, b = _pack(DOC_A), _pack(DOC_B)
    b_copy = _pack(DOC_B.replace("handover", "hand-over") + "\n<!-- 01 -->\n")

    verdicts = classify_queue(
        [("a", a, 1.0), ("b", b, 2.0), ("b_copy", b_copy, 3.0)],
        refuse=QUEUE_REFUSE_THRESHOLD,
        review=0.0,
    )

    assert verdicts["b"][0] == REVIEW
    assert verdicts["b_copy"] == (REFUSE, "b", verdicts["b_copy"][2])


def test_identical_packs_score_near_one():
    """lzma keeps identical input near 1.0 (zlib tops out around 0.97)."""
    pack = _pack_with(DOC_A + DOC_B, POLICY)
    assert queue_similarity(pack, pack) > 0.98
