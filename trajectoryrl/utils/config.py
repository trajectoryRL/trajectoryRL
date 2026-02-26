"""Validator configuration."""

import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class ValidatorConfig:
    """Configuration for TrajectoryRL validator.

    Attributes:
        # Bittensor config
        wallet_name: Wallet name
        wallet_hotkey: Hotkey name
        netuid: Subnet UID (11 for TrajectoryRL)
        network: Bittensor network (finney, test, local)

        # ClawBench config
        clawbench_path: Path to clawbench directory
        scenarios: List of scenario names to evaluate
        scenarios_path: Path to scenarios directory

        # Evaluation config
        seeds_per_task: Number of seeds to run per task (for variance)
        epoch_interval: Seconds between evaluation epochs
        timeout_per_scenario: Max seconds per scenario evaluation

        # Scoring config
        rho_reliability: Weight for variance penalty (0-1)
        delta_threshold: First-mover advantage threshold (0-1)

        # Pack caching
        pack_cache_dir: Directory for caching downloaded packs
        pack_cache_max_size: Max cache size in MB

        # Logging
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_dir: Directory for log files
    """

    # Bittensor config
    wallet_name: str = "validator"
    wallet_hotkey: str = "default"
    netuid: int = 11
    network: str = "finney"

    # ClawBench config
    clawbench_path: Path = field(
        default_factory=lambda: Path(__file__).parent.parent.parent.parent / "clawbench"
    )
    clawbench_commit: str = "e50824df75e10989c0adaf398b6897b5284701d5"
    scenarios: List[str] = field(
        default_factory=lambda: [
            "client_escalation",
            "morning_brief",
            "inbox_to_action",
            "team_standup",
            "inbox_triage",
        ]
    )
    scenarios_path: Optional[Path] = None

    # Evaluation config
    seeds_per_task: int = 1  # Runs per scenario (spec: "once per scenario")
    epoch_interval: int = 86400  # 24 hours (86400 seconds)
    timeout_per_scenario: int = 120  # 2 minutes max per scenario

    # Scoring config
    rho_reliability: float = 0.1  # 10% weight on variance
    delta_threshold: float = 0.05  # 5% first-mover advantage threshold

    # Consensus config (mitigates LLM non-determinism across validators)
    consensus_epsilon: float = 0.02  # Scores within ε are tied; tie → first-mover wins

    # Bootstrap config (graduated rewards until enough miners join)
    bootstrap_threshold: int = 10  # When active miners < this, use top-3 curve (70/20/10)

    # NCD similarity threshold (reject packs >= this similarity to current winner)
    similarity_threshold: float = 0.80

    # Inactivity tracking
    inactivity_window: int = 2  # Epochs before losing first-mover protection

    # GitHub verification
    github_token: Optional[str] = None  # GITHUB_TOKEN env var; needed for push timestamp

    # Validator score publishing (shared score bucket)
    validator_scores_fork_url: Optional[str] = None  # Validator's fork of validator-scores repo
    validator_scores_local_path: Path = field(
        default_factory=lambda: Path("/tmp/trajectoryrl_validator_scores")
    )

    # Weight cadence (set weights every tempo, not just every epoch)
    weight_interval_blocks: int = 360  # 1 tempo ≈ 72 min at 12s/block

    # Pack caching
    pack_cache_dir: Path = field(
        default_factory=lambda: Path("/tmp/trajectoryrl_packs")
    )
    pack_cache_max_size: int = 100  # MB

    # Logging
    log_level: str = "INFO"
    log_dir: Path = field(
        default_factory=lambda: Path("./logs")
    )

    def __post_init__(self):
        """Set derived paths and create directories."""
        # Set scenarios_path if not provided
        if self.scenarios_path is None:
            self.scenarios_path = self.clawbench_path / "scenarios"

        # Create directories
        self.pack_cache_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Validate clawbench path
        if not self.clawbench_path.exists():
            raise ValueError(
                f"clawbench_path does not exist: {self.clawbench_path}\n"
                f"Run: git submodule update --init --recursive"
            )
        if not self.scenarios_path.exists():
            raise ValueError(
                f"scenarios_path does not exist: {self.scenarios_path}"
            )

        # Verify clawbench version
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=self.clawbench_path,
                capture_output=True,
                text=True,
                check=True
            )
            actual_commit = result.stdout.strip()

            if actual_commit != self.clawbench_commit:
                raise ValueError(
                    f"ClawBench version mismatch!\n"
                    f"Expected: {self.clawbench_commit}\n"
                    f"Actual:   {actual_commit}\n"
                    f"Run: cd {self.clawbench_path} && git checkout {self.clawbench_commit}"
                )
        except subprocess.CalledProcessError as e:
            raise ValueError(
                f"Failed to verify clawbench version: {e}\n"
                f"Ensure {self.clawbench_path} is a valid git repository"
            )

    @classmethod
    def from_env(cls) -> "ValidatorConfig":
        """Load configuration from environment variables.

        Returns:
            ValidatorConfig instance
        """
        return cls(
            wallet_name=os.getenv("WALLET_NAME", "validator"),
            wallet_hotkey=os.getenv("WALLET_HOTKEY", "default"),
            netuid=int(os.getenv("NETUID", "11")),
            network=os.getenv("NETWORK", "finney"),
            clawbench_path=Path(
                os.getenv(
                    "CLAWBENCH_PATH",
                    str(Path(__file__).parent.parent.parent.parent / "clawbench")
                )
            ),
            epoch_interval=int(os.getenv("EPOCH_INTERVAL", "86400")),
            github_token=os.getenv("GITHUB_TOKEN"),
            log_level=os.getenv("LOG_LEVEL", "INFO"),
            similarity_threshold=float(os.getenv("SIMILARITY_THRESHOLD", "0.80")),
            inactivity_window=int(os.getenv("INACTIVITY_WINDOW", "2")),
            validator_scores_fork_url=os.getenv("VALIDATOR_SCORES_FORK_URL"),
            weight_interval_blocks=int(os.getenv("WEIGHT_INTERVAL_BLOCKS", "360")),
        )
