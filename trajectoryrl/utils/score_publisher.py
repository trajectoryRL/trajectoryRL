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
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

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
        wallet,
        fork_repo_url: str,
        local_path: Path,
        github_token: Optional[str] = None,
    ):
        self.wallet = wallet
        self.fork_repo_url = fork_repo_url
        self.local_path = Path(local_path)
        self.github_token = github_token
        self._initialized = False

    async def _ensure_repo(self):
        """Clone fork and set up upstream remote if not done."""
        if self._initialized:
            return

        if not self.local_path.exists():
            await _run_git(
                ["git", "clone", self.fork_repo_url, str(self.local_path)]
            )
        # Add upstream if missing
        result = await _run_git(
            ["git", "-C", str(self.local_path), "remote", "get-url", "upstream"],
            check=False,
        )
        if result.returncode != 0:
            await _run_git(
                [
                    "git", "-C", str(self.local_path),
                    "remote", "add", "upstream",
                    f"https://github.com/{UPSTREAM_REPO}.git",
                ]
            )
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

    @staticmethod
    def verify_signature(payload: dict, signature: str, hotkey_ss58: str) -> bool:
        """Verify sr25519 signature over payload.

        Uses bittensor_wallet.Keypair for verification.
        """
        try:
            from bittensor_wallet import Keypair

            kp = Keypair(ss58_address=hotkey_ss58)
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

        # Pull latest upstream
        await _run_git(
            ["git", "-C", str(self.local_path), "fetch", "upstream", "main"],
            check=False,
        )
        await _run_git(
            ["git", "-C", str(self.local_path), "checkout", "main"],
            check=False,
        )
        await _run_git(
            ["git", "-C", str(self.local_path), "reset", "--hard", "upstream/main"],
            check=False,
        )

        # Write score file
        epoch_dir = self.local_path / f"epoch-{epoch}"
        epoch_dir.mkdir(parents=True, exist_ok=True)
        score_file = epoch_dir / f"{hotkey}.json"
        score_file.write_text(json.dumps(payload, indent=2, sort_keys=True))

        # Commit and push
        branch = f"scores/epoch-{epoch}-{hotkey}"
        await _run_git(
            ["git", "-C", str(self.local_path), "checkout", "-B", branch]
        )
        await _run_git(
            ["git", "-C", str(self.local_path), "add", str(score_file)]
        )
        await _run_git(
            [
                "git", "-C", str(self.local_path),
                # Set identity inline so commit succeeds even without a global
                # git config (common on freshly provisioned validator machines).
                "-c", "user.email=validator@trajectoryrl.local",
                "-c", "user.name=TrajectoryRL Validator",
                "commit", "-m", f"scores: epoch {epoch} validator {hotkey[:8]}",
            ],
            check=False,
        )
        result = await _run_git(
            ["git", "-C", str(self.local_path), "push", "origin", branch, "--force"],
            check=False,
        )
        if result.returncode != 0:
            logger.error(f"Failed to push scores: {result.stderr}")
            return False

        # Open PR via gh CLI
        pr_result = await _run_git(
            [
                "gh", "pr", "create",
                "--repo", UPSTREAM_REPO,
                "--head", f"{self._fork_owner()}:{branch}",
                "--title", f"scores: epoch {epoch} {hotkey}",
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

        # Fetch latest
        await _run_git(
            ["git", "-C", str(self.local_path), "fetch", "upstream", "main"],
            check=False,
        )

        # Remove any stale local epoch directory before checking out from
        # upstream. Without this, a failed/missing upstream checkout leaves
        # old files in place and the validator reads stale scores silently.
        epoch_dir = self.local_path / f"epoch-{epoch}"
        if epoch_dir.exists():
            shutil.rmtree(epoch_dir)

        await _run_git(
            ["git", "-C", str(self.local_path), "checkout", "upstream/main", "--",
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
                if not ScorePublisher.verify_signature(data, signature, hotkey):
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

    def _fork_owner(self) -> str:
        """Extract owner from fork URL."""
        # https://github.com/owner/repo.git -> owner
        url = self.fork_repo_url.rstrip("/")
        if url.endswith(".git"):
            url = url[:-4]
        parts = url.split("/")
        return parts[-2] if len(parts) >= 2 else ""


_GIT_TIMEOUT = 120  # seconds; covers slow push/fetch over bad connections


async def _run_git(
    cmd: list,
    check: bool = True,
    timeout: float = _GIT_TIMEOUT,
) -> subprocess.CompletedProcess:
    """Run a git/gh command asynchronously with a hard timeout.

    Without a timeout, ``proc.communicate()`` can block the entire asyncio
    event loop forever on network issues (unreachable GitHub, stalled push).
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
        msg = f"Git command timed out after {timeout}s: {' '.join(cmd)}"
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
    if check and result.returncode != 0:
        raise subprocess.CalledProcessError(
            result.returncode, cmd, result.stdout, result.stderr
        )
    return result
