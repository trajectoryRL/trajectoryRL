"""Normalized Compression Distance (NCD) similarity check for policy packs.

Used to detect copy-paste attacks. Compares SKILL.md content between
packs using zlib compression as a proxy for information-theoretic similarity.

Deduplication is pairwise: all active miners are compared against each
other, with on-chain block_number determining priority (lower = original).

``classify_queue`` / ``queue_duplicates`` apply a stricter variant at queue
admission, keyed on submission order rather than identity: "is this pack a
near-copy of something already waiting in the queue?". Because it compares
content, it is unaffected by how many hotkeys or coldkeys a submitter controls.

The queue measure differs from ``pack_similarity`` in three ways, each chosen
from measurements on real packs:

- **lzma instead of zlib.** zlib never reaches 1.0 on identical input (a real
  21 KB SKILL.md scores ~0.968 against itself), which compresses the usable
  band; lzma scores identical input ~0.994 and separates copies from
  independent work more widely. bz2 is unsuitable: it scores byte-identical
  files below 0.72.
- **Every file, not only SKILL.md.** In Season 2 the policy files decide
  scoring behaviour, so a copied ``policy.py`` under reworded prose must count
  as a copy. Each file present in both packs is compared separately and the
  maximum is taken. Files under ``MIN_FILE_BYTES`` are skipped, because short
  built-in configurations (``{"kind": "pin", ...}``) legitimately coincide.
- **A review band.** Scores in ``[QUEUE_REVIEW_THRESHOLD,
  QUEUE_REFUSE_THRESHOLD)`` are flagged for review rather than refused, so a
  genuine derivative of a public baseline is not refused after its fee is paid.

Reference: INCENTIVE_MECHANISM.md § Pack Similarity Detection (NCD)
"""

import logging
import lzma
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


# Queue-admission thresholds, calibrated with lzma on labelled real packs:
#   identical pack .......................... 0.994
#   known copies (per-wallet edits) ......... 0.931 - 0.970
#   independent rewrites of the same ideas .. 0.582 - 0.609
#   unrelated packs ......................... 0.265 - 0.369
# The refusal line sits below every observed copy; the review band catches
# derivative work between the two groups.
QUEUE_REFUSE_THRESHOLD = 0.90
QUEUE_REVIEW_THRESHOLD = 0.75
MIN_FILE_BYTES = 256

REFUSE = "refuse"
REVIEW = "review"

_Fingerprint = Dict[str, Tuple[bytes, int]]


def _lzma_len(data: bytes) -> int:
    return len(lzma.compress(data, preset=9 | lzma.PRESET_EXTREME))


def _file_fingerprints(pack: dict) -> _Fingerprint:
    """Normalized bytes and compressed length for every sizeable text file."""
    files = (pack or {}).get("files") or {}
    out: _Fingerprint = {}
    for name, body in files.items():
        if not isinstance(body, str):
            continue
        data = normalize_policy(body).encode("utf-8")
        if len(data) < MIN_FILE_BYTES:
            continue
        out[name] = (data, _lzma_len(data))
    return out


def _ncd_similarity(a: Tuple[bytes, int], b: Tuple[bytes, int]) -> float:
    (text_a, ca), (text_b, cb) = a, b
    max_c = max(ca, cb)
    if max_c == 0:
        return 1.0
    return 1.0 - (_lzma_len(text_a + text_b) - min(ca, cb)) / max_c


def _fingerprint_similarity(a: _Fingerprint, b: _Fingerprint) -> float:
    """Highest per-file similarity across files the two packs share."""
    shared = set(a) & set(b)
    return max((_ncd_similarity(a[name], b[name]) for name in shared), default=0.0)


def queue_similarity(pack_a: dict, pack_b: dict) -> float:
    """Queue-admission similarity between two packs, in [0, 1]."""
    return _fingerprint_similarity(_file_fingerprints(pack_a), _file_fingerprints(pack_b))


def classify_queue(
    entries: Sequence[Tuple[Hashable, dict, float]],
    refuse: float = QUEUE_REFUSE_THRESHOLD,
    review: float = QUEUE_REVIEW_THRESHOLD,
) -> Dict[Hashable, Tuple[str, Hashable, float]]:
    """Classify queued packs against the entries that arrived before them.

    Identity-agnostic: only submission time and content decide. Each entry is
    compared with every earlier entry that kept its slot. The earliest entry of
    a near-duplicate group always keeps its slot.

    Args:
        entries: ``(key, pack, submitted_at)`` triples. Ties on
            ``submitted_at`` break on ``repr(key)`` so results are
            deterministic.
        refuse: at or above this similarity the entry is refused. The caller
            should refuse it *before* ``pending_eval`` so the recycle receipt
            stays reusable.
        review: at or above this (and below ``refuse``) the entry keeps its
            slot but is flagged for review.

    Returns:
        ``{key: (verdict, matched_key, similarity)}`` for refused and flagged
        entries, where ``verdict`` is ``REFUSE`` or ``REVIEW``. Entries that
        are admitted without a flag are absent.
    """
    ordered = sorted(entries, key=lambda e: (e[2], repr(e[0])))
    kept: List[Tuple[Hashable, _Fingerprint]] = []
    verdicts: Dict[Hashable, Tuple[str, Hashable, float]] = {}

    for key, pack, _ in ordered:
        try:
            fp = _file_fingerprints(pack)
        except (AttributeError, TypeError):
            continue
        if not fp:
            kept.append((key, fp))
            continue
        best_key, best = None, 0.0
        for kept_key, kept_fp in kept:
            similarity = _fingerprint_similarity(fp, kept_fp)
            if similarity > best:
                best_key, best = kept_key, similarity
        if best_key is not None and best >= refuse:
            verdicts[key] = (REFUSE, best_key, best)
            logger.info("queue dedup: refuse %s (copy of %s, similarity=%.3f)", key, best_key, best)
            continue
        if best_key is not None and best >= review:
            verdicts[key] = (REVIEW, best_key, best)
            logger.info("queue dedup: review %s (close to %s, similarity=%.3f)", key, best_key, best)
        kept.append((key, fp))

    return verdicts


def queue_duplicates(
    entries: Sequence[Tuple[Hashable, dict, float]],
    threshold: float = QUEUE_REFUSE_THRESHOLD,
) -> Dict[Hashable, Hashable]:
    """Refusals only: ``{refused_key: key_it_copies}``. See ``classify_queue``."""
    verdicts = classify_queue(entries, refuse=threshold, review=threshold)
    return {k: v[1] for k, v in verdicts.items() if v[0] == REFUSE}


def cluster_packs(
    packs: Dict[Hashable, dict],
    threshold: float = QUEUE_REFUSE_THRESHOLD,
) -> List[List[Hashable]]:
    """Group packs into near-duplicate clusters using the queue measure.

    Reporting helper: shows how many *distinct* documents a set of packs
    contains. Keys are visited in sorted order so clustering is deterministic.

    Returns:
        Clusters as lists of keys, largest first; the first key of each is its
        representative.
    """
    fps: Dict[Hashable, _Fingerprint] = {}
    for key in sorted(packs, key=repr):
        try:
            fps[key] = _file_fingerprints(packs[key])
        except (AttributeError, TypeError):
            continue

    clusters: List[List[Hashable]] = []
    reps: List[_Fingerprint] = []
    for key, fp in fps.items():
        for index, rep_fp in enumerate(reps):
            if fp and _fingerprint_similarity(fp, rep_fp) >= threshold:
                clusters[index].append(key)
                break
        else:
            reps.append(fp)
            clusters.append([key])

    return sorted(clusters, key=len, reverse=True)
