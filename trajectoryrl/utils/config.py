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
    clawbench_commit: str = "25d678066ed884a888703e00561d0838f178d5b4"
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

    # Cost-based scoring config
    cost_delta: float = 0.10  # Challenger must be 10% cheaper to dethrone
    cost_ema_alpha: float = 0.3  # EMA smoothing for per-scenario cost
    required_categories: List[str] = field(
        default_factory=lambda: ["safety", "correctness"]
    )

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

    # ClawBench LLM configuration (passed to init container & OpenClaw gateway)
    clawbench_default_model: str = "zhipu/glm-5"
    clawbench_api_key: str = ""
    clawbench_base_url: str = "https://open.bigmodel.cn/api/paas/v4"

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
            clawbench_default_model=os.getenv("CLAWBENCH_DEFAULT_MODEL", "zhipu/glm-5"),
            clawbench_api_key=os.getenv("CLAWBENCH_LLM_API_KEY", ""),
            clawbench_base_url=os.getenv(
                "CLAWBENCH_LLM_BASE_URL",
                "https://open.bigmodel.cn/api/paas/v4",
            ),
        )


@dataclass
class MinerConfig:
    """Configuration for TrajectoryRL miner.

    Attributes:
        wallet_name: Wallet name
        wallet_hotkey: Hotkey name
        netuid: Subnet UID (11 for TrajectoryRL)
        network: Bittensor network (finney, test, local)
        check_interval: Seconds between submission cycles in run mode
        log_level: Logging level
        llm_api_key: API key for the OpenAI-compatible LLM endpoint
        llm_base_url: Base URL for the OpenAI-compatible LLM endpoint
        llm_model: Model name for AGENTS.md generation (e.g. glm-5)
        pack_url: Pre-set pack URL (skips S3 upload if set)
    """

    wallet_name: str = "miner"
    wallet_hotkey: str = "default"
    netuid: int = 11
    network: str = "finney"

    check_interval: int = 3600

    log_level: str = "INFO"

    # LLM pack generation (default mode) — OpenAI-compatible endpoint
    llm_api_key: str = ""
    llm_base_url: str = "https://open.bigmodel.cn/api/paas/v4"
    llm_model: str = "glm-5"

    # Pre-built pack URL (skips S3 upload if set)
    pack_url: str = ""

    @classmethod
    def from_env(cls, dotenv_path: Optional[Path] = None) -> "MinerConfig":
        """Load configuration from environment variables.

        Args:
            dotenv_path: Optional path to a .env file. Defaults to
                         ``.env.miner`` in the project root.
        """
        from dotenv import load_dotenv

        if dotenv_path is None:
            dotenv_path = Path(__file__).parent.parent.parent / ".env.miner"
        if dotenv_path.exists():
            load_dotenv(dotenv_path)

        return cls(
            wallet_name=os.getenv("WALLET_NAME", "miner"),
            wallet_hotkey=os.getenv("WALLET_HOTKEY", "default"),
            netuid=int(os.getenv("NETUID", "11")),
            network=os.getenv("NETWORK", "finney"),
            check_interval=int(os.getenv("CHECK_INTERVAL", "3600")),
            log_level=os.getenv("LOG_LEVEL", "INFO"),
            llm_api_key=os.getenv("LLM_API_KEY", ""),
            llm_base_url=os.getenv(
                "LLM_BASE_URL",
                "https://open.bigmodel.cn/api/paas/v4",
            ),
            llm_model=os.getenv("LLM_MODEL", "glm-5"),
            pack_url=os.getenv("PACK_URL", ""),
        )
