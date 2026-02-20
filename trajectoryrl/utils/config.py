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
        tasks_per_epoch: Number of tasks to sample per epoch
        seeds_per_task: Number of seeds to run per task (for variance)
        epoch_interval: Seconds between evaluation epochs
        timeout_per_scenario: Max seconds per scenario evaluation

        # Scoring config
        lambda_cost: Weight for cost penalty (0-1)
        mu_safety: Weight for safety penalty (0-1)
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
    tasks_per_epoch: int = 4  # Select 4 scenarios from pool per epoch
    seeds_per_task: int = 3  # Runs per scenario for majority-vote consensus
    epoch_interval: int = 14400  # 4 hours (14400 seconds)
    timeout_per_scenario: int = 120  # 2 minutes max per scenario

    # Scoring config
    lambda_cost: float = 0.3  # Reserved: cost/safety are scored via rubric checks, not separate penalties
    mu_safety: float = 0.4  # Reserved: cost/safety are scored via rubric checks, not separate penalties
    rho_reliability: float = 0.1  # 10% weight on variance
    delta_threshold: float = 0.05  # 5% first-mover advantage threshold
    scenarios_per_epoch: int = 4  # How many scenarios to evaluate per epoch (from pool)

    # Consensus config (mitigates LLM non-determinism across validators)
    score_quantization: float = 0.05  # Round scores to nearest 0.05
    consensus_epsilon: float = 0.02  # Scores within ε are tied; tie → first-mover wins

    # Minimum score to be eligible for rewards (INCENTIVE_MECHANISM.md § Pack Requirements)
    min_score_threshold: float = 0.30  # Miners below this get weight=0

    # Bootstrap config (graduated rewards until enough miners join)
    bootstrap_threshold: int = 10  # When active miners < this, use top-3 curve (70/20/10)

    # GitHub verification
    github_token: Optional[str] = None  # GITHUB_TOKEN env var; needed for push timestamp

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
            epoch_interval=int(os.getenv("EPOCH_INTERVAL", "14400")),
            github_token=os.getenv("GITHUB_TOKEN"),
            log_level=os.getenv("LOG_LEVEL", "INFO"),
        )
