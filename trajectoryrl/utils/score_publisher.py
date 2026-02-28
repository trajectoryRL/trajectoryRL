"""Shared score bucket: publish and aggregate validator scores.

Validators publish per-UID scores to a shared GitHub repository
(``trajectoryRL/validator-scores``) via signed JSON files, then pull
all published scores and compute a stake-weighted consensus.

Reference: INCENTIVE_MECHANISM.md § Validator Consensus
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import bittensor as bt

logger = logging.getLogger(__name__)

UPSTREAM_REPO = "trajectoryRL/validator-scores"


@dataclass
class ValidatorScoreFile:
    """A single validator's published score file."""

    validator_hotkey: str
    epoch: int
    block_height: int
    scores: Dict[str, dict]  # uid_str -> {final_score, per_scenario}
    signature: str  # sr25519 signature hex


@dataclass
class ConsensusResult:
    """Result of stake-weighted consensus computation."""

    consensus_scores: Dict[int, float]  # uid -> consensus_score
    num_validators: int
    total_stake: float


class ScorePublisher:
    """Manages score publishing and retrieval from the shared validator-scores repo."""

    def __init__(
        self,
        wallet_name: str,
        wallet_hotkey: str,
        fork_repo_url: str,
        local_path: Path,
        github_token: str,
        git_email: str,
        git_name: str,
    ):
        self.wallet = bt.Wallet(name=wallet_name, hotkey=wallet_hotkey)
        self.fork_repo_url = fork_repo_url
        self.local_path = Path(local_path)
        self.github_token = github_token
        self.git_email = git_email
        self.git_name = git_name
        self._initialized = False

    def _fork_nwo(self) -> str:
        """Extract owner/repo from fork URL (``name-with-owner`` format)."""
        url = self.fork_repo_url.rstrip("/")
        if url.endswith(".git"):
            url = url[:-4]
        parts = url.split("/")
        return f"{parts[-2]}/{parts[-1]}" if len(parts) >= 2 else ""

    def _auth_url(self, url: str) -> str:
        """Embed github_token into an HTTPS GitHub URL."""
        if self.github_token and "github.com" in url:
            return url.replace(
                "https://github.com",
                f"https://x-access-token:{self.github_token}@github.com",
            )
        return url

    async def _ensure_repo(self):
        """Clone fork and set up origin + upstream remotes with auth."""
        if self._initialized:
            return

        auth_origin = self._auth_url(self.fork_repo_url)
        auth_upstream = self._auth_url(
            f"https://github.com/{UPSTREAM_REPO}.git"
        )
        git = ["git", "-C", str(self.local_path)]

        if not self.local_path.exists():
            await _run_git(["git", "clone", auth_origin, str(self.local_path)])
        else:
            await _run_git(git + ["remote", "set-url", "origin", auth_origin], check=False)

        result = await _run_git(git + ["remote", "get-url", "upstream"], check=False)
        if result.returncode != 0:
            await _run_git(git + ["remote", "add", "upstream", auth_upstream])
        else:
            await _run_git(git + ["remote", "set-url", "upstream", auth_upstream], check=False)

        await _run_git(git + ["config", "--local", "credential.helper", ""], check=False)
        self._initialized = True

    def sign_payload(self, payload: dict) -> str:
        """Sign the canonical JSON payload with sr25519.

        The payload is serialized with sorted keys and no extra whitespace.
        The ``signature`` field (if present) is excluded before signing.
        """
        signable = {k: v for k, v in payload.items() if k != "signature"}
        canonical = json.dumps(signable, sort_keys=True, separators=(",", ":"))
        sig = self.wallet.hotkey.sign(canonical.encode("utf-8"))
        return sig.hex() if isinstance(sig, bytes) else str(sig)

    def verify_signature(self, payload: dict, signature: str, hotkey_ss58: str) -> bool:
        """Verify sr25519 signature over payload.

        Constructs a temporary ``bt.Wallet``-compatible keypair from the
        given *hotkey_ss58* address for sr25519 verification.
        """
        try:
            kp = bt.Keypair(ss58_address=hotkey_ss58)
            signable = {k: v for k, v in payload.items() if k != "signature"}
            canonical = json.dumps(signable, sort_keys=True, separators=(",", ":"))
            return kp.verify(canonical.encode("utf-8"), bytes.fromhex(signature))
        except Exception as e:
            logger.warning(f"Signature verification failed: {e}")
            return False

    async def publish_scores(
        self,
        epoch: int,
        block_height: int,
        scores: Dict[str, dict],
    ) -> bool:
        """Publish scores to the shared repo via PR.

        Creates ``epoch-{N}/{hotkey}.json``, commits to fork, opens PR.
        """
        await self._ensure_repo()

        hotkey = self.wallet.hotkey.ss58_address
        payload: Dict[str, Any] = {
            "validator_hotkey": hotkey,
            "epoch": epoch,
            "block_height": block_height,
            "scores": scores,
        }
        payload["signature"] = self.sign_payload(payload)

        logger.info(f"Publishing scores to {self.fork_repo_url}")

        git = ["git", "-C", str(self.local_path)]
        fork_owner = self._fork_nwo().split("/")[0]
        branch = f"scores/epoch-{epoch}-{hotkey[:8]}"

        # 1. Sync local main with upstream
        await _run_git(git + ["checkout", "main"], check=False)
        await _run_git(git + ["pull", "upstream", "main"], check=False)

        # 2. Create feature branch from main
        await _run_git(git + ["checkout", "-b", branch], check=False)

        # 3. Write score file
        epoch_dir = self.local_path / f"epoch-{epoch}"
        epoch_dir.mkdir(parents=True, exist_ok=True)
        score_file = epoch_dir / f"{hotkey}.json"
        score_file.write_text(json.dumps(payload, indent=2, sort_keys=True))

        # 4. Commit
        await _run_git(git + ["add", str(score_file)])
        await _run_git(
            git + [
                "-c", f"user.email={self.git_email}",
                "-c", f"user.name={self.git_name}",
                "commit", "-m", f"scores: epoch {epoch} validator {hotkey[:8]}",
            ],
            check=False,
        )

        # 5. Push branch to fork
        logger.info(f"Pushing branch {branch}")
        result = await _run_git(
            git + ["push", "origin", branch, "--force"],
            check=False,
        )
        if result.returncode != 0:
            logger.error(f"Failed to push scores: {result.stderr}")
            return False

        # 6. Create PR from fork branch to upstream main
        pr_result = await _run_git(
            [
                "gh", "pr", "create",
                "--repo", UPSTREAM_REPO,
                "--head", f"{fork_owner}:{branch}",
                "--base", "main",
                "--title", branch,
                "--body", f"Validator {hotkey} epoch {epoch} scores",
            ],
            check=False,
        )
        if pr_result.returncode != 0:
            logger.warning(f"PR creation returned: {pr_result.stderr}")

        return True

    async def pull_all_scores(self, epoch: int) -> List[ValidatorScoreFile]:
        """Pull all published scores for a given epoch from upstream."""
        import shutil
        await self._ensure_repo()

        # Sync fork with upstream, then pull locally
        await _run_git(
            ["gh", "repo", "sync", self._fork_nwo(), "--branch", "main"],
            check=False,
        )
        await _run_git(
            ["git", "-C", str(self.local_path), "checkout", "main"],
            check=False,
        )
        await _run_git(
            ["git", "-C", str(self.local_path), "pull", "origin", "main"],
            check=False,
        )

        epoch_dir = self.local_path / f"epoch-{epoch}"
        if epoch_dir.exists():
            shutil.rmtree(epoch_dir)

        await _run_git(
            ["git", "-C", str(self.local_path), "checkout", "origin/main", "--",
             f"epoch-{epoch}/"],
            check=False,
        )

        if not epoch_dir.exists():
            return []

        results = []
        for score_path in epoch_dir.glob("*.json"):
            try:
                data = json.loads(score_path.read_text())
                hotkey = data["validator_hotkey"]
                signature = data["signature"]

                # Verify sr25519 signature to reject forged score files
                if not self.verify_signature(data, signature, hotkey):
                    logger.warning(
                        f"Score file {score_path.name}: invalid signature, skipping"
                    )
                    continue

                sf = ValidatorScoreFile(
                    validator_hotkey=hotkey,
                    epoch=data["epoch"],
                    block_height=data["block_height"],
                    scores=data["scores"],
                    signature=signature,
                )
                if sf.epoch != epoch:
                    logger.warning(
                        f"Score file {score_path.name}: epoch mismatch "
                        f"(file says {sf.epoch}, expected {epoch}), skipping"
                    )
                    continue
                results.append(sf)
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Invalid score file {score_path}: {e}")

        return results

    @staticmethod
    def compute_consensus(
        score_files: List[ValidatorScoreFile],
        metagraph,
    ) -> ConsensusResult:
        """Compute stake-weighted consensus scores across validators.

        ``consensus_score[uid] = sum(stake_i * score_i[uid]) / sum(stake_i)``

        Validators with zero stake are excluded.
        """
        if not score_files:
            return ConsensusResult(
                consensus_scores={}, num_validators=0, total_stake=0.0
            )

        # Build hotkey -> stake mapping
        hotkey_stake: Dict[str, float] = {}
        for uid in range(len(metagraph.hotkeys)):
            hk = metagraph.hotkeys[uid]
            stake = float(metagraph.S[uid]) if hasattr(metagraph, "S") else 0.0
            hotkey_stake[hk] = stake

        # Deduplicate: keep the latest score file per validator hotkey.
        # A validator that restarts mid-epoch can push two files for the same
        # epoch, and double-counting their stake inflates the denominator.
        deduped: Dict[str, "ValidatorScoreFile"] = {}
        for sf in score_files:
            prev = deduped.get(sf.validator_hotkey)
            if prev is None or sf.block_height > prev.block_height:
                deduped[sf.validator_hotkey] = sf
        score_files = list(deduped.values())

        # Aggregate: weighted sum per UID
        weighted_sums: Dict[int, float] = {}
        stake_sums: Dict[int, float] = {}
        total_stake = 0.0
        valid_count = 0

        for sf in score_files:
            stake = hotkey_stake.get(sf.validator_hotkey, 0.0)
            if stake <= 0:
                continue

            valid_count += 1
            total_stake += stake

            for uid_str, score_data in sf.scores.items():
                uid = int(uid_str)
                score = score_data.get("final_score", 0.0)
                weighted_sums[uid] = weighted_sums.get(uid, 0.0) + stake * score
                stake_sums[uid] = stake_sums.get(uid, 0.0) + stake

        consensus_scores: Dict[int, float] = {}
        for uid in weighted_sums:
            if stake_sums[uid] > 0:
                consensus_scores[uid] = weighted_sums[uid] / stake_sums[uid]

        return ConsensusResult(
            consensus_scores=consensus_scores,
            num_validators=valid_count,
            total_stake=total_stake,
        )



_GIT_TIMEOUT = 120  # seconds; covers slow push/fetch over bad connections


async def _run_git(
    cmd: list,
    check: bool = True,
    cwd: Optional[str] = None,
    timeout: float = _GIT_TIMEOUT,
) -> subprocess.CompletedProcess:
    """Run a git/gh command asynchronously with a hard timeout.

    Without a timeout, ``proc.communicate()`` can block the entire asyncio
    event loop forever on network issues (unreachable GitHub, stalled push).
    """
    cmd_str = " ".join(cmd)
    logger.debug(f"Running: {cmd_str}")

    env = {
        **os.environ,
        "GIT_TERMINAL_PROMPT": "0",
        "GH_PROMPT_DISABLED": "1",
        "GH_NO_UPDATE_NOTIFIER": "1",
    }
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=env,
        cwd=cwd,
    )
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.communicate()
        msg = f"Git command timed out after {timeout}s: {cmd_str}"
        logger.error(msg)
        result = subprocess.CompletedProcess(
            cmd, returncode=-1, stdout="", stderr=msg
        )
        if check:
            raise subprocess.CalledProcessError(-1, cmd, "", msg)
        return result

    result = subprocess.CompletedProcess(
        cmd, proc.returncode,
        stdout=stdout.decode() if stdout else "",
        stderr=stderr.decode() if stderr else "",
    )
    if result.returncode != 0:
        logger.debug(f"Command failed (rc={result.returncode}): {cmd_str}\n{result.stderr}")
    if check and result.returncode != 0:
        raise subprocess.CalledProcessError(
            result.returncode, cmd, result.stdout, result.stderr
        )
    return result
