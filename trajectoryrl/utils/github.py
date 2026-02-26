"""GitHub repository verification utilities."""

import asyncio
import json
import logging
import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Tuple

import httpx

_SUBPROCESS_TIMEOUT = 60  # seconds for network git ops (clone/fetch)
_GIT_LOCAL_TIMEOUT = 10   # seconds for local-only git ops (cat-file/show)


async def _run_subprocess(
    cmd: list,
    timeout: float = _GIT_LOCAL_TIMEOUT,
) -> subprocess.CompletedProcess:
    """Run a command asynchronously without blocking the event loop.

    All git/subprocess calls in this module previously used blocking
    ``subprocess.run()``, which stalls the entire asyncio event loop
    during network I/O (clone, fetch) — sometimes for 30-60 seconds.
    """
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.communicate()
        raise subprocess.TimeoutExpired(cmd, timeout)

    return subprocess.CompletedProcess(
        cmd, proc.returncode,
        stdout=stdout.decode() if stdout else "",
        stderr=stderr.decode() if stderr else "",
    )

logger = logging.getLogger(__name__)


@dataclass
class GitVerificationResult:
    """Result of Git repository verification.

    Attributes:
        valid: Whether verification passed
        commit_timestamp: Server-side push timestamp (not forgeable git date)
        pack_content: Parsed pack dict (if valid)
        error: Error message (if invalid)
    """
    valid: bool
    commit_timestamp: Optional[float] = None
    pack_content: Optional[dict] = None
    error: Optional[str] = None


class GitHubVerifier:
    """Verifies policy pack submissions from GitHub repositories.

    Uses server-side GitHub push timestamps (not forgeable git committer
    dates) to establish chronological ordering for first-mover advantage.
    """

    GITHUB_API = "https://api.github.com"
    GITHUB_GRAPHQL = "https://api.github.com/graphql"

    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        github_token: Optional[str] = None,
    ):
        """Initialize verifier.

        Args:
            cache_dir: Directory for caching cloned repos (default: temp dir)
            github_token: Optional GitHub API token (falls back to GITHUB_TOKEN
                env var). Not required — Events API + Compare API are public
                for public repos. Token enables higher rate limits and GraphQL
                fallback for commits older than 90 days.
        """
        self.cache_dir = cache_dir or Path(tempfile.gettempdir()) / "trajectoryrl_git_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.github_token = github_token or os.getenv("GITHUB_TOKEN")
        logger.info(f"GitHubVerifier initialized with cache: {self.cache_dir}")

    def cleanup_cache(self, max_size_mb: int = 100) -> None:
        """Evict least-recently-used cloned repos to stay under the size limit.

        Repos are sorted by modification time (oldest first) and removed
        until total cache size is under ``max_size_mb``.
        """
        if not self.cache_dir.exists():
            return

        # Collect (path, size_bytes, mtime) for each cached repo dir
        entries = []
        for entry in self.cache_dir.iterdir():
            if not entry.is_dir():
                continue
            size = sum(f.stat().st_size for f in entry.rglob("*") if f.is_file())
            mtime = entry.stat().st_mtime
            entries.append((entry, size, mtime))

        total_bytes = sum(e[1] for e in entries)
        max_bytes = max_size_mb * 1024 * 1024

        if total_bytes <= max_bytes:
            logger.debug(
                f"Cache size {total_bytes / 1024 / 1024:.1f} MB "
                f"<= {max_size_mb} MB, no eviction needed"
            )
            return

        # Sort by mtime ascending (oldest first = evict first)
        entries.sort(key=lambda e: e[2])

        evicted = 0
        for path, size, _ in entries:
            if total_bytes <= max_bytes:
                break
            logger.info(f"Evicting cached repo: {path.name} ({size / 1024 / 1024:.1f} MB)")
            shutil.rmtree(path, ignore_errors=True)
            total_bytes -= size
            evicted += 1

        if evicted:
            logger.info(
                f"Cache cleanup: evicted {evicted} repos, "
                f"remaining {total_bytes / 1024 / 1024:.1f} MB"
            )

    async def verify_submission(
        self,
        repo_url: str,
        git_commit_hash: str,
        pack_hash: str,
        on_chain_submission_time: float
    ) -> GitVerificationResult:
        """Verify a GitHub-based pack submission.

        Args:
            repo_url: Public GitHub repository URL
            git_commit_hash: Git commit SHA (40-char hex)
            pack_hash: Expected SHA256 hash of pack content
            on_chain_submission_time: Unix timestamp of on-chain submission

        Returns:
            GitVerificationResult with validation outcome
        """
        logger.info(f"Verifying submission: {repo_url}@{git_commit_hash[:8]}")

        # Step 1: Clone or update repo
        repo_path = await self._clone_or_update_repo(repo_url)
        if repo_path is None:
            return GitVerificationResult(
                valid=False,
                error="Failed to clone repository"
            )

        # Step 2: Verify commit exists
        commit_exists = await self._verify_commit_exists(repo_path, git_commit_hash)
        if not commit_exists:
            return GitVerificationResult(
                valid=False,
                error=f"Commit {git_commit_hash[:8]} not found in repository"
            )

        # Step 3: Get SERVER-SIDE push timestamp from GitHub API.
        # Git committer dates are trivially forged via `git commit --date`,
        # so we use the GitHub-recorded push time instead.
        push_timestamp = await self._get_server_push_timestamp(
            repo_url, git_commit_hash
        )
        if push_timestamp is None:
            return GitVerificationResult(
                valid=False,
                error=(
                    "Cannot verify server-side push timestamp. "
                    "Ensure the repo is public. Set GITHUB_TOKEN for "
                    "GraphQL fallback if the push is older than 90 days."
                )
            )

        # Step 4: Verify push timestamp is before on-chain submission
        if push_timestamp > on_chain_submission_time:
            return GitVerificationResult(
                valid=False,
                error=(
                    f"Push timestamp ({push_timestamp:.0f}) is after "
                    f"on-chain submission ({on_chain_submission_time:.0f})"
                )
            )

        # Detect possible backdating: large divergence between git committer
        # date and server-side push date suggests `git commit --date` abuse.
        git_committer_ts = await self._get_commit_timestamp(
            repo_path, git_commit_hash
        )
        if git_committer_ts is not None:
            divergence = push_timestamp - git_committer_ts
            if divergence > 300:  # pushed >5 min after claimed commit date
                logger.warning(
                    f"Timestamp divergence: push={push_timestamp:.0f}, "
                    f"git_committer={git_committer_ts:.0f} "
                    f"(diff={divergence:.0f}s) — possible backdating"
                )

        commit_timestamp = push_timestamp
        logger.info(
            f"Push timestamp verified: {push_timestamp:.0f} < "
            f"{on_chain_submission_time:.0f}"
        )

        # Step 5: Extract pack from commit
        pack_content = await self._extract_pack_from_commit(repo_path, git_commit_hash)
        if pack_content is None:
            return GitVerificationResult(
                valid=False,
                error="Failed to extract pack from commit"
            )

        # Step 6: Verify pack hash
        import hashlib
        computed_hash = hashlib.sha256(
            json.dumps(pack_content, sort_keys=True).encode()
        ).hexdigest()

        if computed_hash != pack_hash:
            return GitVerificationResult(
                valid=False,
                error=f"Pack hash mismatch: expected {pack_hash[:8]}, got {computed_hash[:8]}"
            )

        logger.info(f"✓ Verification passed for {git_commit_hash[:8]}")

        return GitVerificationResult(
            valid=True,
            commit_timestamp=commit_timestamp,
            pack_content=pack_content
        )

    async def _clone_or_update_repo(self, repo_url: str, _retry: bool = True) -> Optional[Path]:
        """Clone repository or update if already cached.

        Args:
            repo_url: GitHub repository URL

        Returns:
            Path to cloned repo, or None if failed
        """
        # Create safe directory name from repo URL using owner__repo to
        # avoid collisions between different owners with the same repo name.
        parts = repo_url.rstrip("/").split("/")
        owner, repo_name = parts[-2], parts[-1].replace(".git", "")
        repo_path = self.cache_dir / f"{owner}__{repo_name}"

        try:
            if repo_path.exists():
                # Update existing repo
                logger.debug(f"Updating cached repo: {repo_path}")
                result = await _run_subprocess(
                    ["git", "-C", str(repo_path), "fetch", "--all"],
                    timeout=_SUBPROCESS_TIMEOUT,
                )
                if result.returncode != 0:
                    logger.warning(f"Failed to update repo: {result.stderr}")
                    if not _retry:
                        return None
                    # Try to clone fresh (shutil.rmtree is fast — no network I/O)
                    shutil.rmtree(repo_path, ignore_errors=True)
                    return await self._clone_or_update_repo(repo_url, _retry=False)
            else:
                # Clone fresh
                logger.debug(f"Cloning repo to: {repo_path}")
                result = await _run_subprocess(
                    ["git", "clone", "--quiet", repo_url, str(repo_path)],
                    timeout=_SUBPROCESS_TIMEOUT,
                )
                if result.returncode != 0:
                    logger.error(f"Failed to clone repo: {result.stderr}")
                    return None

            return repo_path

        except subprocess.TimeoutExpired:
            logger.error("Git operation timed out")
            return None
        except Exception as e:
            logger.error(f"Error cloning/updating repo: {e}")
            return None

    async def _verify_commit_exists(self, repo_path: Path, commit_hash: str) -> bool:
        """Verify that commit exists in repository.

        Args:
            repo_path: Path to local git repository
            commit_hash: Git commit SHA

        Returns:
            True if commit exists, False otherwise
        """
        try:
            result = await _run_subprocess(
                ["git", "-C", str(repo_path), "cat-file", "-e", commit_hash],
                timeout=_GIT_LOCAL_TIMEOUT,
            )
            return result.returncode == 0
        except Exception as e:
            logger.error(f"Error verifying commit: {e}")
            return False

    async def _get_commit_timestamp(self, repo_path: Path, commit_hash: str) -> Optional[float]:
        """Get git committer timestamp (LOCAL — potentially forged).

        Only used for divergence detection, NOT for ordering decisions.
        See _get_server_push_timestamp() for the trusted timestamp.

        Args:
            repo_path: Path to local git repository
            commit_hash: Git commit SHA

        Returns:
            Unix timestamp, or None if failed
        """
        try:
            result = await _run_subprocess(
                ["git", "-C", str(repo_path), "show", "-s", "--format=%ct", commit_hash],
                timeout=_GIT_LOCAL_TIMEOUT,
            )
            if result.returncode != 0:
                return None

            return float(result.stdout.strip())

        except Exception as e:
            logger.error(f"Error getting commit timestamp: {e}")
            return None

    # ------------------------------------------------------------------
    # Server-side push timestamp verification (anti-forgery)
    # ------------------------------------------------------------------

    async def _get_server_push_timestamp(
        self, repo_url: str, commit_hash: str
    ) -> Optional[float]:
        """Get server-side push timestamp from GitHub API.

        Git committer dates are trivially forged with `git commit --date`
        or GIT_COMMITTER_DATE. This method queries GitHub's servers for the
        timestamp when they actually received the push, which cannot be
        manipulated by the committer.

        Tries in order:
          1. GitHub REST Events API (no auth needed for public repos)
          2. GitHub GraphQL Commit.pushedDate (requires GITHUB_TOKEN)

        Returns:
            Unix timestamp of when GitHub received the push, or None
        """
        owner, repo = self._parse_github_url(repo_url)

        # Method 1: Events API (REST, works without token for public repos)
        ts = await self._get_push_timestamp_events_api(owner, repo, commit_hash)
        if ts is not None:
            return ts

        # Method 2: GraphQL pushedDate (requires token)
        if self.github_token:
            ts = await self._get_push_timestamp_graphql(owner, repo, commit_hash)
            if ts is not None:
                return ts

        token_status = "no GITHUB_TOKEN set" if not self.github_token else "GraphQL also failed"
        logger.error(
            f"Cannot determine server-side push timestamp for {commit_hash[:8]}. "
            f"Events API returned nothing and {token_status}."
        )
        return None

    async def _get_push_timestamp_events_api(
        self, owner: str, repo: str, commit_hash: str
    ) -> Optional[float]:
        """Query GitHub REST Events API for the PushEvent containing this commit.

        The Events API returns server-side ``created_at`` timestamps that
        cannot be forged. Limited to the last 90 days / 300 events.

        PushEvent payloads contain ``head`` (tip SHA) and ``before`` (previous
        HEAD). We first try a direct ``head`` match (fast path for single-commit
        pushes — the common case for miners). If that fails we use the Compare
        API to enumerate all commits in the ``before...head`` range.

        Returns:
            Unix timestamp, or None if the commit isn't found in recent events
        """
        url = f"{self.GITHUB_API}/repos/{owner}/{repo}/events"
        headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        if self.github_token:
            headers["Authorization"] = f"Bearer {self.github_token}"

        try:
            async with httpx.AsyncClient() as client:
                # Paginate (up to 10 pages x 100 events = 1000 events max)
                for page in range(1, 11):
                    resp = await client.get(
                        url,
                        headers=headers,
                        params={"per_page": 100, "page": page},
                        timeout=10,
                    )
                    if resp.status_code == 403:
                        logger.warning("Events API rate-limited (403)")
                        break
                    if resp.status_code != 200:
                        logger.warning(f"Events API returned {resp.status_code}")
                        break

                    events = resp.json()
                    if not events:
                        break

                    for event in events:
                        if event.get("type") != "PushEvent":
                            continue
                        payload = event.get("payload", {})
                        head = payload.get("head")
                        before = payload.get("before")

                        # Fast path: commit is the tip of the push
                        if head == commit_hash:
                            return self._parse_event_timestamp(event)

                        # Slow path: commit may be in the before...head range.
                        # Use the Compare API (public, no auth needed).
                        if head and before:
                            found = await self._commit_in_push_range(
                                client, headers, owner, repo,
                                before, head, commit_hash,
                            )
                            if found:
                                return self._parse_event_timestamp(event)

        except httpx.RequestError as e:
            logger.warning(f"Events API request error: {e}")
        except Exception as e:
            logger.warning(f"Events API unexpected error: {e}")

        return None

    @staticmethod
    def _parse_event_timestamp(event: dict) -> float:
        """Extract unix timestamp from a GitHub event's created_at field."""
        created_at = event["created_at"]
        dt = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
        ts = dt.timestamp()
        logger.info(f"Push timestamp via Events API: {created_at} (unix={ts:.0f})")
        return ts

    async def _commit_in_push_range(
        self,
        client: httpx.AsyncClient,
        headers: dict,
        owner: str,
        repo: str,
        before: str,
        head: str,
        commit_hash: str,
    ) -> bool:
        """Check whether commit_hash is in the before...head range via Compare API.

        The Compare API is public (no auth required for public repos).

        Returns:
            True if commit_hash is among the commits between before and head
        """
        compare_url = (
            f"{self.GITHUB_API}/repos/{owner}/{repo}/compare/{before}...{head}"
        )
        try:
            resp = await client.get(compare_url, headers=headers, timeout=10)
            if resp.status_code != 200:
                return False
            commits = resp.json().get("commits", [])
            return any(c.get("sha") == commit_hash for c in commits)
        except Exception as e:
            logger.debug(f"Compare API error for {before[:8]}...{head[:8]}: {e}")
            return False

    async def _get_push_timestamp_graphql(
        self, owner: str, repo: str, commit_hash: str
    ) -> Optional[float]:
        """Query GitHub GraphQL API for Commit.pushedDate.

        The `pushedDate` field is set server-side by GitHub when it receives
        the push. Unlike `committedDate`, it cannot be forged by the committer.

        Requires GITHUB_TOKEN with repo read access.

        Returns:
            Unix timestamp, or None if unavailable
        """
        query = """
        query ($owner: String!, $repo: String!, $oid: GitObjectID!) {
          repository(owner: $owner, name: $repo) {
            object(oid: $oid) {
              ... on Commit {
                pushedDate
                committedDate
              }
            }
          }
        }
        """
        variables = {"owner": owner, "repo": repo, "oid": commit_hash}

        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    self.GITHUB_GRAPHQL,
                    headers={
                        "Authorization": f"Bearer {self.github_token}",
                        "Content-Type": "application/json",
                    },
                    json={"query": query, "variables": variables},
                    timeout=10,
                )
                if resp.status_code != 200:
                    logger.warning(f"GraphQL API returned {resp.status_code}")
                    return None

                data = resp.json()
                errors = data.get("errors")
                if errors:
                    logger.warning(f"GraphQL errors: {errors}")
                    return None

                obj = (
                    data.get("data", {})
                    .get("repository", {})
                    .get("object")
                )
                if not obj:
                    logger.warning("Commit not found via GraphQL")
                    return None

                pushed_date = obj.get("pushedDate")
                if not pushed_date:
                    logger.warning("pushedDate is null")
                    return None

                dt = datetime.fromisoformat(pushed_date.replace("Z", "+00:00"))
                ts = dt.timestamp()

                # Log divergence for forensics
                committed_date = obj.get("committedDate")
                if committed_date:
                    committed_dt = datetime.fromisoformat(
                        committed_date.replace("Z", "+00:00")
                    )
                    divergence = abs(ts - committed_dt.timestamp())
                    if divergence > 300:
                        logger.warning(
                            f"GraphQL timestamp divergence: "
                            f"pushed={pushed_date}, committed={committed_date} "
                            f"(diff={divergence:.0f}s)"
                        )

                logger.info(
                    f"Push timestamp via GraphQL: {pushed_date} (unix={ts:.0f})"
                )
                return ts

        except httpx.RequestError as e:
            logger.warning(f"GraphQL request error: {e}")
        except Exception as e:
            logger.warning(f"GraphQL unexpected error: {e}")

        return None

    @staticmethod
    def _parse_github_url(repo_url: str) -> Tuple[str, str]:
        """Extract (owner, repo) from a GitHub URL.

        Handles:
          https://github.com/owner/repo
          https://github.com/owner/repo.git
          https://github.com/owner/repo/

        Returns:
            (owner, repo_name) tuple
        """
        stripped = repo_url.rstrip("/")
        if stripped.endswith(".git"):
            stripped = stripped[:-4]
        parts = stripped.split("/")
        return parts[-2], parts[-1]

    async def _extract_pack_from_commit(
        self,
        repo_path: Path,
        commit_hash: str
    ) -> Optional[dict]:
        """Extract policy pack from git commit.

        Looks for pack.json or constructs pack from AGENTS.md, SOUL.md, etc.

        Args:
            repo_path: Path to local git repository
            commit_hash: Git commit SHA

        Returns:
            Parsed pack dict, or None if failed
        """
        try:
            # First try to find pack.json
            result = await _run_subprocess(
                ["git", "-C", str(repo_path), "show", f"{commit_hash}:pack.json"],
                timeout=_GIT_LOCAL_TIMEOUT,
            )

            if result.returncode == 0:
                # Found pack.json
                pack = json.loads(result.stdout)
                logger.debug("Extracted pack from pack.json")
                return pack

            # Otherwise, construct pack from individual files
            logger.debug("pack.json not found, constructing from files")
            pack = {
                "schema_version": 1,
                "files": {},
                "tool_policy": {},
                "metadata": {}
            }

            # Extract AGENTS.md
            result = await _run_subprocess(
                ["git", "-C", str(repo_path), "show", f"{commit_hash}:AGENTS.md"],
                timeout=_GIT_LOCAL_TIMEOUT,
            )
            if result.returncode == 0:
                pack["files"]["AGENTS.md"] = result.stdout

            # Extract SOUL.md (optional)
            result = await _run_subprocess(
                ["git", "-C", str(repo_path), "show", f"{commit_hash}:SOUL.md"],
                timeout=_GIT_LOCAL_TIMEOUT,
            )
            if result.returncode == 0:
                pack["files"]["SOUL.md"] = result.stdout

            # Extract tool_policy.json (optional)
            result = await _run_subprocess(
                ["git", "-C", str(repo_path), "show", f"{commit_hash}:tool_policy.json"],
                timeout=_GIT_LOCAL_TIMEOUT,
            )
            if result.returncode == 0:
                pack["tool_policy"] = json.loads(result.stdout)

            if not pack["files"]:
                logger.warning("No pack files found in commit")
                return None

            return pack

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse pack JSON: {e}")
            return None
        except Exception as e:
            logger.error(f"Error extracting pack from commit: {e}")
            return None
