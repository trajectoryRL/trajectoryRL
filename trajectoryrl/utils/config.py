"""Validator and miner configuration."""

import logging
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


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
        eval_interval_blocks: Blocks between re-evaluations (~4 hours)
        timeout_per_scenario: Max seconds per scenario evaluation

        # Scoring config
        rho_reliability: Weight for variance penalty (0-1)
        delta_threshold: First-mover advantage threshold (0-1)
        ema_alpha: EMA smoothing factor for per-scenario scores

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
    clawbench_commit: str = "0f2bf473566a64422e50567e22b19372394b24e5"
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
    seeds_per_task: int = 1
    eval_interval_blocks: int = 1200  # ~4 hours at 12s/block
    timeout_per_scenario: int = 120  # 2 minutes max per scenario

    # Scoring config
    rho_reliability: float = 0.1
    delta_threshold: float = 0.05
    ema_alpha: float = 0.3  # Per-scenario EMA smoothing factor

    # Consensus config (mitigates LLM non-determinism across validators)
    consensus_epsilon: float = 0.02

    # Bootstrap config (graduated rewards until enough miners join)
    bootstrap_threshold: int = 10

    # NCD similarity threshold (reject packs >= this similarity to current winner)
    similarity_threshold: float = 0.80

    # Inactivity tracking (block-based)
    inactivity_blocks: int = 14400  # ~48 hours at 12s/block

    # Weight cadence (set weights every tempo)
    weight_interval_blocks: int = 360  # 1 tempo ≈ 72 min at 12s/block

    # EMA state persistence
    ema_state_path: Path = field(
        default_factory=lambda: Path("/tmp/trajectoryrl_ema_state.json")
    )

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
        if self.scenarios_path is None:
            self.scenarios_path = self.clawbench_path / "scenarios"

        self.pack_cache_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        if not self.clawbench_path.exists():
            raise ValueError(
                f"clawbench_path does not exist: {self.clawbench_path}\n"
                f"Run: git submodule update --init --recursive"
            )
        if not self.scenarios_path.exists():
            raise ValueError(
                f"scenarios_path does not exist: {self.scenarios_path}"
            )

        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=self.clawbench_path,
                capture_output=True,
                text=True,
                check=True,
            )
            actual_commit = result.stdout.strip()

            if actual_commit != self.clawbench_commit:
                raise ValueError(
                    f"ClawBench version mismatch!\n"
                    f"Expected: {self.clawbench_commit}\n"
                    f"Actual:   {actual_commit}\n"
                    f"Run: cd {self.clawbench_path} && git checkout {self.clawbench_commit}"
                )
        except subprocess.CalledProcessError:
            logger.warning(
                "Cannot verify clawbench commit (not a git repo). "
                "This is expected inside Docker containers."
            )

    @classmethod
    def from_env(cls, dotenv_path: Optional[Path] = None) -> "ValidatorConfig":
        """Load configuration from environment variables.

        Args:
            dotenv_path: Optional path to a .env file. Defaults to
                         ``.env.validator`` in the project root.

        Returns:
            ValidatorConfig instance
        """
        from dotenv import load_dotenv

        if dotenv_path is None:
            dotenv_path = Path(__file__).parent.parent.parent / ".env.validator"
        load_dotenv(dotenv_path)

        return cls(
            wallet_name=os.getenv("WALLET_NAME", "validator"),
            wallet_hotkey=os.getenv("WALLET_HOTKEY", "default"),
            netuid=int(os.getenv("NETUID", "11")),
            network=os.getenv("NETWORK", "finney"),
            clawbench_path=Path(
                os.getenv(
                    "CLAWBENCH_PATH",
                    str(Path(__file__).parent.parent.parent / "clawbench")
                )
            ),
            eval_interval_blocks=int(os.getenv("EVAL_INTERVAL_BLOCKS", "1200")),
            ema_alpha=float(os.getenv("EMA_ALPHA", "0.3")),
            log_level=os.getenv("LOG_LEVEL", "INFO"),
            similarity_threshold=float(os.getenv("SIMILARITY_THRESHOLD", "0.80")),
            inactivity_blocks=int(os.getenv("INACTIVITY_BLOCKS", "14400")),
            weight_interval_blocks=int(os.getenv("WEIGHT_INTERVAL_BLOCKS", "360")),
        )


@dataclass
class MinerConfig:
    """Configuration for TrajectoryRL miner daemon.

    Attributes:
        wallet_name: Bittensor wallet name.
        wallet_hotkey: Bittensor hotkey name.
        netuid: Subnet UID (11 for TrajectoryRL).
        network: Bittensor network (finney, test, local).
        pack_path: Path to an existing pack.json file.
        agents_md_path: Path to AGENTS.md (build pack on the fly).
        pack_url: Public HTTP(S) URL where pack.json is hosted — required for daemon mode.
        check_interval: Seconds between daemon loop iterations.
    """

    wallet_name: str = "miner"
    wallet_hotkey: str = "default"
    netuid: int = 11
    network: str = "finney"
    pack_path: Optional[str] = None
    agents_md_path: Optional[str] = None
    pack_url: str = ""
    check_interval: int = 3600

    def __post_init__(self):
        if not self.pack_path and not self.agents_md_path:
            raise ValueError(
                "MinerConfig requires at least one of pack_path or agents_md_path"
            )
        if not self.pack_url:
            raise ValueError(
                "MinerConfig requires pack_url "
                "(e.g. 'https://trajrl.com/samples/pack.json')"
            )

    @classmethod
    def from_env(cls, dotenv_path: Optional[Path] = None) -> "MinerConfig":
        """Load configuration from environment variables.

        Args:
            dotenv_path: Optional path to a .env file. Defaults to
                         ``.env.miner`` in the project root.

        Returns:
            MinerConfig instance.
        """
        from dotenv import load_dotenv

        if dotenv_path is None:
            dotenv_path = Path(__file__).parent.parent.parent / ".env.miner"
        load_dotenv(dotenv_path)

        return cls(
            wallet_name=os.getenv("WALLET_NAME", "miner"),
            wallet_hotkey=os.getenv("WALLET_HOTKEY", "default"),
            netuid=int(os.getenv("NETUID", "11")),
            network=os.getenv("NETWORK", "finney"),
            pack_path=os.getenv("PACK_PATH") or None,
            agents_md_path=os.getenv("AGENTS_MD_PATH") or None,
            pack_url=os.getenv("PACK_URL", ""),
            check_interval=int(os.getenv("CHECK_INTERVAL", "3600")),
        )
