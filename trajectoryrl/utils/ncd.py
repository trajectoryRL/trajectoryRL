"""Normalized Compression Distance (NCD) similarity check for policy packs.

Used to detect copy-paste attacks. Compares SKILL.md content between
packs using zlib compression as a proxy for information-theoretic similarity.

Deduplication is pairwise: all active miners are compared against each
other, with on-chain block_number determining priority (lower = original).

``queue_duplicates`` applies the same measure at queue admission, keyed on
submission order rather than identity: it answers "is this pack a near-copy of
something already waiting in the queue?". Because it compares content, it is
unaffected by how many hotkeys or coldkeys a submitter controls — registering
more identities does not buy more queue slots for the same document.

Reference: INCENTIVE_MECHANISM.md § Pack Similarity Detection (NCD)
"""

import logging
import re
import zlib
from typing import Dict, Hashable, List, Sequence, Set, Tuple

logger = logging.getLogger(__name__)

SIMILARITY_THRESHOLD = 0.80

def _extract_policy(pack: dict) -> str:
    """Extract the SKILL.md policy file content from a pack."""
    return pack.get("files", {}).get("SKILL.md", "")


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

    Compares the primary policy file after normalization.

    Returns:
        Similarity score in [0, 1]. 1.0 = identical, 0.0 = unrelated.
    """
    a = normalize_policy(_extract_policy(pack_a))
    b = normalize_policy(_extract_policy(pack_b))

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


def deduplicate_packs(
    pack_info: Dict[str, Tuple[dict, int, str]],
    threshold: float = SIMILARITY_THRESHOLD,
) -> Dict[str, str]:
    """Pairwise NCD dedup with pack_hash fast-path.

    Identifies copy-cat miners by comparing all packs pairwise.
    Uses pack_hash grouping for exact copies (O(N)), then NCD for
    paraphrased copies among unique packs. Priority is determined by
    on-chain block_number (lower = first mover = original).

    Args:
        pack_info: {hotkey: (pack, block_number, pack_hash)}
        threshold: NCD similarity threshold

    Returns:
        Dict mapping excluded hotkey -> original hotkey it copied from.
    """
    if len(pack_info) < 2:
        return {}

    excluded: Dict[str, str] = {}

    # --- Layer 1: group by pack_hash (exact copies, O(N)) ---
    hash_groups: Dict[str, list] = {}
    for hotkey, (pack, block_number, pack_hash) in pack_info.items():
        hash_groups.setdefault(pack_hash, []).append(
            (hotkey, block_number, pack)
        )

    unique_reps: Dict[str, Tuple[str, dict, int]] = {}
    for pack_hash, members in hash_groups.items():
        members.sort(key=lambda x: x[1])
        first_hotkey, first_block, first_pack = members[0]
        unique_reps[pack_hash] = (first_hotkey, first_pack, first_block)

        for hotkey, block_number, _ in members[1:]:
            excluded[hotkey] = first_hotkey
            logger.info(
                f"NCD dedup: {hotkey[:8]} excluded (exact copy of "
                f"{first_hotkey[:8]}, pack_hash={pack_hash[:12]})"
            )

    if len(unique_reps) < 2:
        return excluded

    # --- Layer 2: NCD pairwise among unique representatives ---
    precomputed: Dict[str, Tuple[bytes, int]] = {}
    for pack_hash, (hotkey, pack, block_number) in unique_reps.items():
        try:
            text = normalize_policy(
                _extract_policy(pack)
            ).encode("utf-8")
            precomputed[pack_hash] = (text, len(zlib.compress(text, 9)))
        except (KeyError, TypeError):
            continue

    sorted_hashes = sorted(
        precomputed.keys(),
        key=lambda h: unique_reps[h][2],
    )

    flagged_hashes: Set[str] = set()
    for i, hash_i in enumerate(sorted_hashes):
        if hash_i in flagged_hashes:
            continue
        text_i, ci = precomputed[hash_i]
        original_hotkey = unique_reps[hash_i][0]

        for hash_j in sorted_hashes[i + 1:]:
            if hash_j in flagged_hashes:
                continue
            text_j, cj = precomputed[hash_j]
            cab = len(zlib.compress(text_i + text_j, 9))
            max_c = max(ci, cj)
            if max_c == 0:
                # Both texts compress to zero → both empty → identical
                similarity = 1.0
            else:
                ncd = (cab - min(ci, cj)) / max_c
                similarity = 1.0 - ncd

            if similarity >= threshold:
                copier_hotkey = unique_reps[hash_j][0]
                excluded[copier_hotkey] = original_hotkey
                flagged_hashes.add(hash_j)
                logger.info(
                    f"NCD dedup: {copier_hotkey[:8]} excluded "
                    f"(similarity={similarity:.3f} with "
                    f"{original_hotkey[:8]})"
                )
                # Re-attribute Layer 1 members of hash_j: they were
                # mapped to copier_hotkey (their group's first mover),
                # but that rep is now itself excluded. Update them to
                # point to the true original from hash_i's group.
                for member_hk, _, _ in hash_groups[hash_j][1:]:
                    excluded[member_hk] = original_hotkey

    return excluded


def _fingerprint(pack: dict) -> Tuple[bytes, int]:
    """Normalized policy bytes and their compressed length, computed once."""
    text = normalize_policy(_extract_policy(pack)).encode("utf-8")
    return text, len(zlib.compress(text, 9))


def _similarity(a: Tuple[bytes, int], b: Tuple[bytes, int]) -> float:
    """NCD similarity between two precomputed fingerprints."""
    (text_a, ca), (text_b, cb) = a, b
    max_c = max(ca, cb)
    if max_c == 0:
        return 1.0
    cab = len(zlib.compress(text_a + text_b, 9))
    return 1.0 - (cab - min(ca, cb)) / max_c


def queue_duplicates(
    entries: Sequence[Tuple[Hashable, dict, float]],
    threshold: float = SIMILARITY_THRESHOLD,
) -> Dict[Hashable, Hashable]:
    """Find packs that duplicate an earlier entry in the same queue.

    Identity-agnostic: only submission time and content decide. The earliest
    entry of a near-duplicate group keeps its slot; later ones are returned as
    duplicates so the caller can refuse them *before* they reach
    ``pending_eval`` (which leaves the recycle receipt reusable).

    Args:
        entries: ``(key, pack, submitted_at)`` triples. ``key`` identifies the
            submission (e.g. its id); ``submitted_at`` is any sortable
            timestamp. Ties break on ``repr(key)`` so the result is
            deterministic.
        threshold: NCD similarity at or above which a pack counts as a copy.

    Returns:
        ``{duplicate_key: kept_key}`` for every entry that duplicates an
        earlier one. Entries that keep their slot are absent.
    """
    ordered = sorted(entries, key=lambda e: (e[2], repr(e[0])))
    kept: List[Tuple[Hashable, Tuple[bytes, int]]] = []
    duplicates: Dict[Hashable, Hashable] = {}

    for key, pack, _ in ordered:
        try:
            fp = _fingerprint(pack)
        except (AttributeError, KeyError, TypeError):
            continue
        for kept_key, kept_fp in kept:
            similarity = _similarity(fp, kept_fp)
            if similarity >= threshold:
                duplicates[key] = kept_key
                logger.info(
                    "queue dedup: %s duplicates %s (similarity=%.3f)",
                    key, kept_key, similarity,
                )
                break
        else:
            kept.append((key, fp))

    return duplicates


def cluster_packs(
    packs: Dict[Hashable, dict],
    threshold: float = SIMILARITY_THRESHOLD,
) -> List[List[Hashable]]:
    """Group packs into near-duplicate clusters.

    Reporting helper: shows how many *distinct* documents a set of packs
    contains. Keys are visited in sorted order so clustering is deterministic.

    Returns:
        A list of clusters, each a list of keys, largest cluster first. The
        first key of a cluster is its representative.
    """
    fps: Dict[Hashable, Tuple[bytes, int]] = {}
    for key in sorted(packs, key=repr):
        try:
            fps[key] = _fingerprint(packs[key])
        except (AttributeError, KeyError, TypeError):
            continue

    clusters: List[List[Hashable]] = []
    reps: List[Tuple[Hashable, Tuple[bytes, int]]] = []
    for key, fp in fps.items():
        for index, (_, rep_fp) in enumerate(reps):
            if _similarity(fp, rep_fp) >= threshold:
                clusters[index].append(key)
                break
        else:
            reps.append((key, fp))
            clusters.append([key])

    return sorted(clusters, key=len, reverse=True)
