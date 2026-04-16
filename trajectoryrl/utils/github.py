"""HTTP-based pack fetching and verification utilities.

Validators fetch policy packs from miner-provided HTTP(S) URLs
and verify the content hash
matches the on-chain commitment.

Historical note: This module previously implemented GitHubVerifier
which used git clone/fetch to retrieve packs from GitHub repos.
The current PackFetcher uses plain HTTP GET, removing the dependency
on git and GitHub-specific APIs.
"""

import hashlib
import json
import logging
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

_HTTP_TIMEOUT = 30  # seconds for HTTP requests
_MAX_RESPONSE_BYTES = 64 * 1024  # 64KB safety limit for pack downloads


@dataclass
class PackVerificationResult:
    """Result of HTTP-based pack verification.

    Attributes:
        valid: Whether verification passed
        pack_content: Parsed pack dict (if valid)
        raw_text: Raw text content before JSON parsing (if valid)
        error: Error message (if invalid)
    """
    valid: bool
    pack_content: Optional[dict] = None
    raw_text: Optional[str] = None
    error: Optional[str] = None


class PackFetcher:
    """Fetches and verifies policy packs from HTTP(S) URLs.

    Miners host their pack.json on any publicly accessible HTTP endpoint.
    Validators download the pack and verify its SHA256 hash matches
    the on-chain commitment.
    """

    def __init__(
        self,
        cache_dir: Optional[Path] = None,
    ):
        """Initialize fetcher.

        Args:
            cache_dir: Directory for caching downloaded packs.
                Packs are cached by their content hash to avoid
                redundant downloads.
        """
        import tempfile
        self.cache_dir = cache_dir or Path(tempfile.gettempdir()) / "trajectoryrl_pack_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"PackFetcher initialized with cache: {self.cache_dir}")

    def cleanup_cache(self, max_size_mb: int = 100) -> None:
        """Evict least-recently-used cached packs to stay under the size limit.

        Entries are sorted by modification time (oldest first) and removed
        until total cache size is under ``max_size_mb``.
        """
        if not self.cache_dir.exists():
            return

        entries = []
        for entry in self.cache_dir.iterdir():
            if not entry.is_file():
                if entry.is_dir():
                    size = sum(f.stat().st_size for f in entry.rglob("*") if f.is_file())
                else:
                    continue
            else:
                size = entry.stat().st_size
            mtime = entry.stat().st_mtime
            entries.append((entry, size, mtime))

        total_bytes = sum(e[1] for e in entries)
        max_bytes = max_size_mb * 1024 * 1024

        if total_bytes <= max_bytes:
            logger.info(
                f"Cache size {total_bytes / 1024 / 1024:.1f} MB "
                f"<= {max_size_mb} MB, no eviction needed"
            )
            return

        entries.sort(key=lambda e: e[2])

        evicted = 0
        for path, size, _ in entries:
            if total_bytes <= max_bytes:
                break
            logger.info(f"Evicting cached pack: {path.name} ({size / 1024:.1f} KB)")
            if path.is_dir():
                shutil.rmtree(path, ignore_errors=True)
            else:
                path.unlink(missing_ok=True)
            total_bytes -= size
            evicted += 1

        if evicted:
            logger.info(
                f"Cache cleanup: evicted {evicted} entries, "
                f"remaining {total_bytes / 1024 / 1024:.1f} MB"
            )

    async def verify_submission(
        self,
        pack_url: str,
        pack_hash: str,
    ) -> PackVerificationResult:
        """Fetch a pack from an HTTP URL and verify its content hash.

        Args:
            pack_url: Public HTTP(S) URL to the pack.json file
            pack_hash: Expected SHA256 hash of the canonical pack JSON

        Returns:
            PackVerificationResult with validation outcome
        """
        logger.info(f"Verifying submission: {pack_url} (expected hash={pack_hash[:12]}…)")

        # Step 1: Check cache
        cached = self._load_from_cache(pack_hash)
        if cached is not None:
            logger.info(f"Cache hit for hash={pack_hash[:12]}…")
            return PackVerificationResult(valid=True, pack_content=cached)

        # Step 2: Fetch raw content via HTTP
        raw_content = await self._fetch_pack(pack_url)
        if raw_content is None:
            return PackVerificationResult(
                valid=False,
                error=f"Failed to fetch pack from {pack_url}"
            )

        # Step 3: Try JSON parse — determines hash verification strategy
        try:
            pack_content = json.loads(raw_content)
        except (json.JSONDecodeError, ValueError):
            pack_content = None

        if pack_content is not None:
            # JSON pack (v4.0): hash the canonical re-serialization
            canonical = json.dumps(pack_content, sort_keys=True).encode()
            computed_hash = hashlib.sha256(canonical).hexdigest()
        else:
            # Plain text (S1): hash raw bytes as-is
            computed_hash = hashlib.sha256(raw_content.encode()).hexdigest()

        if computed_hash != pack_hash:
            return PackVerificationResult(
                valid=False,
                error=f"Hash mismatch: expected {pack_hash[:8]}, got {computed_hash[:8]}"
            )

        logger.info(f"Verification passed for {pack_url} (hash={pack_hash[:12]}…)")

        if pack_content is not None:
            self._save_to_cache(pack_hash, pack_content)

        return PackVerificationResult(
            valid=True,
            pack_content=pack_content,
            raw_text=raw_content,
        )

    async def _fetch_pack(self, pack_url: str) -> Optional[str]:
        """Download pack content from an HTTP URL.

        Args:
            pack_url: Public HTTP(S) URL

        Returns:
            Raw response text, or None on failure
        """
        try:
            async with httpx.AsyncClient(follow_redirects=True) as client:
                resp = await client.get(
                    pack_url,
                    timeout=_HTTP_TIMEOUT,
                    headers={"Accept": "*/*"},
                )

                if resp.status_code != 200:
                    logger.warning(
                        f"HTTP {resp.status_code} from {pack_url}"
                    )
                    return None

                content_length = len(resp.content)
                if content_length > _MAX_RESPONSE_BYTES:
                    logger.warning(
                        f"Response too large: {content_length} bytes "
                        f"(max {_MAX_RESPONSE_BYTES}) from {pack_url}"
                    )
                    return None

                return resp.text

        except httpx.TimeoutException:
            logger.warning(f"Timeout fetching {pack_url}")
            return None
        except httpx.RequestError as e:
            logger.warning(f"HTTP request error for {pack_url}: {e}")
            return None
        except Exception as e:
            logger.warning(f"Unexpected error fetching {pack_url}: {e}")
            return None

    def _load_from_cache(self, pack_hash: str) -> Optional[dict]:
        """Load a previously cached pack by its content hash."""
        cache_path = self.cache_dir / f"{pack_hash}.json"
        if not cache_path.exists():
            return None
        try:
            with open(cache_path) as f:
                pack = json.load(f)
            cache_path.touch()
            return pack
        except Exception as e:
            logger.info(f"Cache read failed for {pack_hash[:12]}…: {e}")
            return None

    def _save_to_cache(self, pack_hash: str, pack: dict) -> None:
        """Cache a verified pack by its content hash."""
        cache_path = self.cache_dir / f"{pack_hash}.json"
        try:
            with open(cache_path, "w") as f:
                json.dump(pack, f, sort_keys=True)
        except Exception as e:
            logger.info(f"Cache write failed for {pack_hash[:12]}…: {e}")


# Backward-compatible aliases
GitHubVerifier = PackFetcher
GitVerificationResult = PackVerificationResult
