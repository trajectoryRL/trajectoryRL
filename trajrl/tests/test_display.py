"""Tests for display formatting."""

from trajrl.display import qual


def test_qual_true_contains_checkmark():
    result = qual(True)
    assert "\u2713" in result, f"Expected ✓ in qual(True), got {result!r}"


def test_qual_false_contains_cross():
    result = qual(False)
    assert "\u2717" in result, f"Expected ✗ in qual(False), got {result!r}"


def test_qual_none_returns_dash():
    assert qual(None) == "—"


def test_qual_no_escaped_unicode():
    """Ensure unicode chars are actual symbols, not escaped strings."""
    for v in [True, False]:
        result = qual(v)
        assert "\\u" not in result, f"qual({v}) contains escaped unicode: {result!r}"
