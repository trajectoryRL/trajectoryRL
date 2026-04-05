"""Validator and miner configuration."""

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

DEFAULT_LLM_BASE_URL = "https://open.bigmodel.cn/api/paas/v4"
DEFAULT_LLM_MODEL = "glm-5"
DEFAULT_CLAWBENCH_MODEL = f"zhipu/{DEFAULT_LLM_MODEL}"


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
        eval_interval_blocks: Blocks between re-evaluations (~24 hours)
        timeout_per_scenario: Max seconds per scenario evaluation

        # Scoring config
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
    scenarios: List[str] = field(
        default_factory=lambda: [
            "client_escalation",
            "morning_brief",
            "inbox_to_action",
            "team_standup",
            "inbox_triage",
            "hiring_debrief",
            "post_incident_review",
        ]
    )
    scenarios_path: Optional[Path] = None

    # Evaluation config
    seeds_per_task: int = 1
    eval_interval_blocks: int = 7200  # ~24 hours at 12s/block (window length)
    eval_utc_hour: int = 0           # UTC hour to trigger daily eval cycle (0 = midnight)
    timeout_per_scenario: int = 300  # 5 minutes max per scenario

    # Evaluation window config (block-aligned consensus protocol)
    global_anchor_block: int = 0     # anchor block for window alignment
    window_publish_pct: float = 0.80  # T_publish: 80% of window for evaluation
    window_aggregate_pct: float = 0.90  # T_aggregate: 90% (10% propagation interval)

    # Scoring config
    delta_threshold: float = 0.05

    # Cost-based scoring config
    cost_delta: float = 0.10  # Winner Protection: challenger must be 10% cheaper
    required_categories: List[str] = field(
        default_factory=lambda: ["safety", "correctness"]
    )

    # Consensus config (mitigates LLM non-determinism across validators)
    consensus_epsilon: float = 0.02
    consensus_protocol_version: int = 1
    qualification_stake_threshold: float = 0.5  # stake-weighted majority for qualification

    # Consensus CAS: IPFS primary, trajrl.com API fallback
    ipfs_api_url: str = "http://ipfs.metahash73.com:5001/api/v0"
    ipfs_gateway_urls: List[str] = field(
        default_factory=lambda: ["https://ipfs.io", "https://dweb.link"]
    )
    consensus_api_url: str = "https://trajrl.com"
    min_validator_stake: float = 10000.0  # minimum stake weight (α) for consensus participation

    # Bootstrap config (graduated rewards until enough miners join)
    bootstrap_threshold: int = 10

    # NCD similarity threshold (reject packs >= this similarity to current winner)
    similarity_threshold: float = 0.80

    # Coldkey blacklist: miners under these coldkeys are skipped entirely
    coldkey_blacklist: List[str] = field(default_factory=list)

    # Inactivity tracking (block-based)
    inactivity_blocks: int = 14400  # ~48 hours at 12s/block

    # Weight cadence (set weights every tempo)
    weight_interval_blocks: int = 360  # 1 tempo ≈ 72 min at 12s/block

    # Startup aggregation: run consensus aggregation before entering main loop
    aggregate_when_start: bool = True

    # Disable winner protection to force all validators to converge on the
    # same lowest-cost winner (use once to clear divergent cached state).
    disable_winner_protection: bool = False

    # ClawBench LLM configuration (passed to init container & OpenClaw gateway)
    clawbench_default_model: str = DEFAULT_CLAWBENCH_MODEL
    clawbench_api_key: str = ""
    clawbench_base_url: str = DEFAULT_LLM_BASE_URL

    # LLM Judge configuration (Phase 1 integrity + Phase 2 trajectory)
    # Defaults to same LLM as ClawBench episodes if left empty.
    judge_model: str = ""
    judge_api_key: str = ""
    judge_base_url: str = ""

    # EMA state persistence
    ema_state_path: Path = field(
        default_factory=lambda: Path("/var/lib/trajectoryrl/ema_state.json")
    )

    # Winner Protection state persistence
    winner_state_path: Path = field(
        default_factory=lambda: Path("/var/lib/trajectoryrl/winner_state.json")
    )

    # Pack caching
    pack_cache_dir: Path = field(
        default_factory=lambda: Path("/var/lib/trajectoryrl/packs")
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
            # --- Bittensor ---
            wallet_name=os.getenv("WALLET_NAME", "validator"),
            wallet_hotkey=os.getenv("WALLET_HOTKEY", "default"),
            netuid=int(os.getenv("NETUID", "11")),
            network=os.getenv("NETWORK", "finney"),
            # --- Paths ---
            clawbench_path=Path(
                os.getenv(
                    "CLAWBENCH_PATH",
                    str(Path(__file__).parent.parent.parent / "clawbench")
                )
            ),
            ema_state_path=Path(os.getenv("EMA_STATE_PATH", "/var/lib/trajectoryrl/ema_state.json")),
            winner_state_path=Path(os.getenv("WINNER_STATE_PATH", "/var/lib/trajectoryrl/winner_state.json")),
            # --- LLM ---
            clawbench_default_model=os.getenv("CLAWBENCH_DEFAULT_MODEL", DEFAULT_CLAWBENCH_MODEL),
            clawbench_api_key=os.getenv("CLAWBENCH_LLM_API_KEY", ""),
            clawbench_base_url=os.getenv("CLAWBENCH_LLM_BASE_URL", DEFAULT_LLM_BASE_URL),
            # --- LLM Judge (optional, falls back to clawbench LLM if empty) ---
            judge_model=os.getenv("JUDGE_MODEL", ""),
            judge_api_key=os.getenv("JUDGE_API_KEY", ""),
            judge_base_url=os.getenv("JUDGE_BASE_URL", ""),
            # --- Operational ---
            log_level=os.getenv("LOG_LEVEL", "INFO"),
            # --- Consensus CAS ---
            ipfs_api_url=os.getenv("IPFS_API_URL", "http://ipfs.metahash73.com:5001/api/v0"),
            ipfs_gateway_urls=[
                gw.strip() for gw in
                os.getenv("IPFS_GATEWAYS", "https://ipfs.io,https://dweb.link").split(",")
                if gw.strip()
            ],
            consensus_api_url=os.getenv("CONSENSUS_API_URL", "https://trajrl.com"),
            # --- Startup aggregation ---
            aggregate_when_start=os.getenv("AGGREGATE_WHEN_START", "1") == "1",
            disable_winner_protection=os.getenv("DISABLE_WINNER_PROTECTION", "1") == "1",
            # --- IM parameters are hardcoded (dataclass defaults) ---
            # Do NOT load from env: cost_delta,
            # rho_reliability, consensus_epsilon, bootstrap_threshold,
            # similarity_threshold, max_commitment_age_blocks,
            # inactivity_blocks, eval_interval_blocks, weight_interval_blocks.
            # All validators must use identical IM values for consensus.
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
    llm_base_url: str = DEFAULT_LLM_BASE_URL
    llm_model: str = DEFAULT_LLM_MODEL

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
            llm_api_key=os.getenv("CLAWBENCH_LLM_API_KEY", ""),
            llm_base_url=os.getenv("CLAWBENCH_LLM_BASE_URL", DEFAULT_LLM_BASE_URL),
            llm_model=os.getenv("CLAWBENCH_DEFAULT_MODEL", DEFAULT_LLM_MODEL),
            pack_url=os.getenv("PACK_URL", ""),
        )
