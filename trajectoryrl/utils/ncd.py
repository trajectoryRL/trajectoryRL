"""Normalized Compression Distance (NCD) similarity check for policy packs.

Used to detect copy-paste attacks. Compares AGENTS.md content between a
challenger pack and the current winner using zlib compression as a proxy
for information-theoretic similarity.

Reference: INCENTIVE_MECHANISM.md § Pack Similarity Detection (NCD)
"""

import re
import zlib

SIMILARITY_THRESHOLD = 0.80


def normalize_policy(text: str) -> str:
    """Strip formatting noise before comparison.

    - Lowercase
    - Strip markdown heading markers (# symbols)
    - Collapse all whitespace to a single space
    """
    text = text.lower()
    text = re.sub(r"#+ *", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def pack_similarity(pack_a: dict, pack_b: dict) -> float:
    """Compute NCD-based similarity between two packs.

    Compares the AGENTS.md content after normalization.

    Returns:
        Similarity score in [0, 1]. 1.0 = identical, 0.0 = unrelated.
    """
    a = normalize_policy(pack_a["files"]["AGENTS.md"])
    b = normalize_policy(pack_b["files"]["AGENTS.md"])

    a_bytes = a.encode("utf-8")
    b_bytes = b.encode("utf-8")
    ca = len(zlib.compress(a_bytes, 9))
    cb = len(zlib.compress(b_bytes, 9))
    cab = len(zlib.compress(a_bytes + b_bytes, 9))

    ncd = (cab - min(ca, cb)) / max(ca, cb)
    return 1.0 - ncd


def is_too_similar(
    pack_challenger: dict,
    pack_winner: dict | None,
    threshold: float = SIMILARITY_THRESHOLD,
) -> bool:
    """Check if challenger pack is too similar to winner pack.

    Returns True if similarity >= threshold (pack should be rejected).
    Returns False if no current winner exists.
    """
    if pack_winner is None:
        return False
    return pack_similarity(pack_challenger, pack_winner) >= threshold
