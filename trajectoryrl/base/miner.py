"""TrajectoryRL Miner — Pack building and on-chain submission.

Miners don't run a server. The workflow is:
    1. Write AGENTS.md (policy document)
    2. Build a pack.json (OPP v1 format)
    3. Push to a public GitHub repo
    4. Submit on-chain commitment via set_commitment

The on-chain commitment is block-timestamped, establishing first-mover
precedence. Validators read commitments and fetch packs from GitHub.

Reference: INCENTIVE_MECHANISM.md § Submission Protocol
"""

import hashlib
import json
import logging
import subprocess
from pathlib import Path
from typing import Dict, Optional

import bittensor as bt

from ..utils.opp_schema import validate_opp_schema
from ..utils.commitments import parse_commitment

logger = logging.getLogger(__name__)


class TrajectoryMiner:
    """TrajectoryRL miner for building and submitting policy packs.

    Example (submit existing pack)::

        miner = TrajectoryMiner(wallet_name="miner", wallet_hotkey="default")
        pack = miner.build_pack(agents_md="path/to/AGENTS.md")
        miner.submit(pack, repo="myuser/my-pack", git_commit="abc123...")

    Example (validate locally)::

        miner = TrajectoryMiner(wallet_name="miner", wallet_hotkey="default")
        pack = miner.load_pack("pack.json")
        result = miner.validate(pack)
        print(result)
    """

    def __init__(
        self,
        wallet_name: str = "miner",
        wallet_hotkey: str = "default",
        netuid: int = 11,
        network: str = "finney",
    ):
        self.wallet_name = wallet_name
        self.wallet_hotkey = wallet_hotkey
        self.netuid = netuid
        self.network = network

        # Lazy-init Bittensor (only needed for on-chain operations)
        self._wallet: Optional[bt.Wallet] = None
        self._subtensor: Optional[bt.Subtensor] = None

    @property
    def wallet(self) -> bt.Wallet:
        if self._wallet is None:
            self._wallet = bt.Wallet(
                name=self.wallet_name, hotkey=self.wallet_hotkey
            )
        return self._wallet

    @property
    def subtensor(self) -> bt.Subtensor:
        if self._subtensor is None:
            self._subtensor = bt.Subtensor(network=self.network)
        return self._subtensor

    # ------------------------------------------------------------------
    # Pack building
    # ------------------------------------------------------------------

    @staticmethod
    def build_pack(
        agents_md: str,
        pack_name: str = "my-pack",
        pack_version: str = "1.0.0",
        soul_md: Optional[str] = None,
        extra_files: Optional[Dict[str, str]] = None,
        tool_allow: Optional[list] = None,
        tool_deny: Optional[list] = None,
        stop_rules: Optional[list] = None,
    ) -> dict:
        """Build an OPP v1 pack from an AGENTS.md string or file path.

        Args:
            agents_md: AGENTS.md content string, or path to a file.
            pack_name: Pack name for metadata.
            pack_version: Semver string (e.g., "1.0.0").
            soul_md: Optional SOUL.md content or file path.
            extra_files: Optional dict of filename -> content to include.
            tool_allow: Tools to allow (default: exec, slack, memory_search,
                memory_get, read).
            tool_deny: Tools to deny (default: admin_*, shell).
            stop_rules: Optional stop rules list.

        Returns:
            OPP v1 pack dict, ready for JSON serialization.
        """
        # Read from file if path exists
        agents_content = _read_or_use(agents_md)
        soul_content = _read_or_use(soul_md) if soul_md else None

        files = {"AGENTS.md": agents_content}
        if soul_content:
            files["SOUL.md"] = soul_content
        if extra_files:
            files.update(extra_files)

        if tool_allow is None:
            tool_allow = ["exec", "slack", "memory_search", "memory_get", "read"]
        if tool_deny is None:
            tool_deny = ["admin_*", "shell"]

        pack = {
            "schema_version": 1,
            "files": files,
            "tool_policy": {
                "allow": tool_allow,
                "deny": tool_deny,
            },
            "metadata": {
                "pack_name": pack_name,
                "pack_version": pack_version,
                "target_suite": "clawbench_v1",
            },
        }

        if stop_rules:
            pack["stop_rules"] = stop_rules

        return pack

    @staticmethod
    def load_pack(path: str) -> dict:
        """Load a pack.json from disk.

        Args:
            path: Path to pack.json file.

        Returns:
            Parsed pack dict.

        Raises:
            FileNotFoundError: If path doesn't exist.
            json.JSONDecodeError: If file is not valid JSON.
        """
        with open(path) as f:
            return json.load(f)

    @staticmethod
    def save_pack(pack: dict, path: str) -> str:
        """Save pack to disk and return its SHA256 hash.

        Args:
            pack: OPP v1 pack dict.
            path: Output file path.

        Returns:
            SHA256 hex digest of the canonical JSON.
        """
        canonical = json.dumps(pack, sort_keys=True)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            f.write(canonical)
        return hashlib.sha256(canonical.encode()).hexdigest()

    @staticmethod
    def compute_pack_hash(pack: dict) -> str:
        """Compute content-addressed SHA256 hash of a pack.

        Uses ``json.dumps(pack, sort_keys=True)`` for deterministic
        serialization, matching the validator's hash computation.

        Args:
            pack: OPP v1 pack dict.

        Returns:
            64-char lowercase hex SHA256 digest.
        """
        canonical = json.dumps(pack, sort_keys=True)
        return hashlib.sha256(canonical.encode()).hexdigest()

    @staticmethod
    def validate(pack: dict) -> "ValidationResult":
        """Validate pack against OPP v1 schema locally.

        Args:
            pack: Pack dict to validate.

        Returns:
            ValidationResult with passed/issues.
        """
        return validate_opp_schema(pack)

    # ------------------------------------------------------------------
    # On-chain commitment
    # ------------------------------------------------------------------

    @staticmethod
    def format_commitment(
        pack_hash: str,
        git_commit_hash: str,
        repo: str,
    ) -> str:
        """Format a commitment string for on-chain submission.

        Args:
            pack_hash: SHA256 hex of pack JSON (64 chars).
            git_commit_hash: Git commit hash (40 chars).
            repo: GitHub repo as "owner/repo" or full URL.

        Returns:
            Pipe-delimited commitment string (≤128 bytes).

        Raises:
            ValueError: If inputs are invalid.
        """
        if len(pack_hash) != 64:
            raise ValueError(f"pack_hash must be 64 hex chars, got {len(pack_hash)}")
        if len(git_commit_hash) != 40:
            raise ValueError(
                f"git_commit_hash must be 40 hex chars, got {len(git_commit_hash)}"
            )

        # Use shorthand if full URL provided
        if repo.startswith("https://github.com/"):
            repo = repo[len("https://github.com/"):]

        # Strip trailing slashes and .git
        repo = repo.rstrip("/")
        if repo.endswith(".git"):
            repo = repo[:-4].rstrip("/")

        commitment = f"{pack_hash}|{git_commit_hash}|{repo}"
        if len(commitment.encode()) > 128:
            raise ValueError(
                f"Commitment too long: {len(commitment.encode())} bytes (max 128)"
            )

        # Verify round-trip
        parsed = parse_commitment(commitment)
        if parsed is None:
            raise ValueError(f"Commitment failed round-trip validation: {commitment}")

        return commitment

    def submit_commitment(
        self,
        pack_hash: str,
        git_commit_hash: str,
        repo: str,
    ) -> bool:
        """Submit on-chain commitment via set_commitment.

        Args:
            pack_hash: SHA256 hex of pack JSON.
            git_commit_hash: Git commit hash of the pack in the repo.
            repo: GitHub repo ("owner/repo" or full URL).

        Returns:
            True if commitment was submitted successfully.
        """
        commitment = self.format_commitment(pack_hash, git_commit_hash, repo)
        logger.info(f"Submitting commitment: {commitment}")

        try:
            self.subtensor.set_commitment(
                wallet=self.wallet,
                netuid=self.netuid,
                data=commitment,
            )
            logger.info("Commitment submitted successfully!")
            return True
        except bt.NotRegisteredError:
            logger.error(
                "Miner hotkey is not registered on subnet %d. "
                "Run: btcli subnet register --netuid %d --wallet.name %s --wallet.hotkey %s",
                self.netuid, self.netuid, self.wallet_name, self.wallet_hotkey,
            )
            return False
        except bt.ChainConnectionError as e:
            logger.error(
                "Cannot connect to %s network. Check network connectivity "
                "and that the chain endpoint is reachable: %s", self.network, e,
            )
            return False
        except bt.KeyFileError as e:
            logger.error(
                "Wallet key file error: %s. "
                "Check wallet exists with: btcli wallet list", e,
            )
            return False
        except bt.ChainTransactionError as e:
            logger.error("Chain transaction failed: %s", e)
            return False
        except Exception as e:
            logger.error("Failed to submit commitment (%s): %s", type(e).__name__, e)
            return False

    def get_current_commitment(self) -> Optional[str]:
        """Read this miner's current on-chain commitment.

        Returns:
            Raw commitment string, or None if not set.
        """
        try:
            hotkey = self.wallet.hotkey.ss58_address
            commitments = self.subtensor.get_all_commitments(netuid=self.netuid)
            return commitments.get(hotkey)
        except bt.ChainConnectionError as e:
            logger.error(
                "Cannot connect to %s network. Check network connectivity "
                "and that the chain endpoint is reachable: %s", self.network, e,
            )
            return None
        except bt.KeyFileError as e:
            logger.error(
                "Wallet key file error: %s. "
                "Check wallet exists with: btcli wallet list", e,
            )
            return None
        except Exception as e:
            logger.error("Failed to read commitment (%s): %s", type(e).__name__, e)
            return None

    # ------------------------------------------------------------------
    # Git helpers (for push workflow)
    # ------------------------------------------------------------------

    @staticmethod
    def git_push_pack(
        pack: dict,
        repo_path: str,
        commit_message: str = "Update policy pack",
    ) -> Optional[str]:
        """Write pack.json to a local git repo, commit, and push.

        Args:
            pack: OPP v1 pack dict.
            repo_path: Path to local git repository.
            commit_message: Git commit message.

        Returns:
            Git commit hash if successful, None on failure.
        """
        repo = Path(repo_path)
        if not (repo / ".git").exists():
            logger.error(f"Not a git repository: {repo_path}")
            return None

        # Write pack.json
        pack_path = repo / "pack.json"
        canonical = json.dumps(pack, sort_keys=True, indent=2)
        pack_path.write_text(canonical)

        # Also write AGENTS.md as standalone for readability
        agents_md = pack.get("files", {}).get("AGENTS.md", "")
        if agents_md:
            (repo / "AGENTS.md").write_text(agents_md)

        try:
            # Stage, commit, push
            subprocess.run(
                ["git", "add", "pack.json", "AGENTS.md"],
                cwd=repo_path, check=True, capture_output=True,
            )
            subprocess.run(
                [
                    "git",
                    "-c", "user.email=miner@trajectoryrl.local",
                    "-c", "user.name=TrajectoryRL Miner",
                    "commit", "-m", commit_message,
                ],
                cwd=repo_path, check=True, capture_output=True,
            )
            subprocess.run(
                ["git", "push"],
                cwd=repo_path, check=True, capture_output=True,
            )

            # Get commit hash
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=repo_path, check=True, capture_output=True, text=True,
            )
            commit_hash = result.stdout.strip()
            logger.info(f"Pushed pack to {repo_path} (commit: {commit_hash[:12]}...)")
            return commit_hash

        except subprocess.CalledProcessError as e:
            logger.error(f"Git operation failed: {e.stderr.decode() if e.stderr else e}")
            return None

    # ------------------------------------------------------------------
    # Full submit workflow
    # ------------------------------------------------------------------

    def submit(
        self,
        pack: dict,
        repo: str,
        git_commit: Optional[str] = None,
        repo_path: Optional[str] = None,
    ) -> bool:
        """Full submission workflow: validate, optionally push, commit on-chain.

        Args:
            pack: OPP v1 pack dict.
            repo: GitHub repo ("owner/repo" or full URL).
            git_commit: Git commit hash. If None, pushes pack via repo_path.
            repo_path: Local git repo path (used if git_commit is None).

        Returns:
            True if everything succeeded.
        """
        # 1. Validate locally
        result = self.validate(pack)
        if not result.passed:
            logger.error(f"Pack validation failed: {result.issues}")
            return False

        pack_hash = self.compute_pack_hash(pack)
        logger.info(f"Pack hash: {pack_hash}")
        logger.info(f"Pack size: {len(json.dumps(pack))} bytes")

        # 2. Push to GitHub if no commit hash provided
        if git_commit is None:
            if repo_path is None:
                logger.error("Must provide either git_commit or repo_path")
                return False
            git_commit = self.git_push_pack(pack, repo_path)
            if git_commit is None:
                return False

        # 3. Submit on-chain
        return self.submit_commitment(pack_hash, git_commit, repo)


def _read_or_use(value: str) -> str:
    """If value is a file path that exists, read it. Otherwise use as-is."""
    if len(value) > 4096:
        return value
    try:
        p = Path(value)
        if p.exists() and p.is_file():
            return p.read_text()
    except OSError:
        pass
    return value
