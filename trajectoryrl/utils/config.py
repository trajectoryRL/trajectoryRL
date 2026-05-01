"""Validator and miner configuration."""

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

DEFAULT_LLM_BASE_URL = "https://open.bigmodel.cn/api/paas/v4"
DEFAULT_LLM_MODEL = "glm-5.1"

# Image channel drives the tag of sandbox/harness images pulled by the
# validator at runtime. Compose files set this per-channel (latest, staging,
# etc.). Full overrides via SANDBOX_IMAGE / HARNESS_IMAGE take precedence.
DEFAULT_IMAGE_CHANNEL = "latest"
SANDBOX_IMAGE_REPO = "ghcr.io/trajectoryrl/trajrl-bench"
HARNESS_IMAGE_REPO = "ghcr.io/trajectoryrl/hermes-agent"

# SPEC_NUMBER identifies a "scoring specification": the combination of
# scenario set, scoring methodology, and judge prompt that determines whether
# two evaluations produce comparable scores.
#
# Bump SPEC_NUMBER whenever a change makes new scores incomparable with old
# ones (adding/removing scenarios, changing weights, modifying judge prompts,
# changing the aggregation rule). Bench-image patch releases that preserve
# scoring semantics do NOT bump it.
#
# This constant ships with the validator binary; it is decoupled from the
# trajrl-bench image version (now used purely for audit / log purposes).
#
# Aggregation derives its target spec number from on-chain stake distribution
# rather than reading this constant directly — see
# ``consensus_filter.select_target_spec_number`` for details. SPEC_NUMBER is
# consulted only as the fallback target when no on-chain spec_number group
# reaches >50% stake, and as the value written into outgoing commitments /
# payloads.
SPEC_NUMBER = 7

# Backwards-compatible alias for legacy callers / persisted state. Will be
# removed after one validator release cycle.
SCORING_VERSION = SPEC_NUMBER


@dataclass
class ValidatorConfig:
    """Configuration for TrajectoryRL validator.

    Attributes:
        # Bittensor config
        wallet_name: Wallet name
        wallet_hotkey: Hotkey name
        netuid: Subnet UID (11 for TrajectoryRL)
        network: Bittensor network (finney, test, local)

        # Evaluation config
        eval_interval_blocks: Blocks between re-evaluations (~24 hours)

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

    # Evaluation config
    eval_interval_blocks: int = 7200  # ~24 hours at 12s/block (window length)
    eval_utc_hour: int = 0           # UTC hour to trigger daily eval cycle (0 = midnight)

    # Evaluation window config (block-aligned consensus protocol)
    global_anchor_block: int = 0     # anchor block for window alignment
    window_publish_pct: float = 0.80  # T_publish: 80% of window for evaluation
    window_aggregate_pct: float = 0.90  # T_aggregate: 90% (10% propagation interval)

    # Scoring config
    delta_threshold: float = 0.05

    # Score-based winner protection config
    score_delta: float = 0.10  # Winner Protection: challenger must score 10% higher
    required_categories: List[str] = field(
        default_factory=lambda: ["safety", "correctness"]
    )

    # Consensus config (mitigates LLM non-determinism across validators)
    consensus_epsilon: float = 0.02
    consensus_protocol_version: int = 2

    # Consensus CAS: IPFS primary, trajrl.com API fallback
    ipfs_api_url: str = "http://ipfs.metahash73.com:5001/api/v0"
    ipfs_gateway_urls: List[str] = field(
        default_factory=lambda: ["https://ipfs.io", "https://dweb.link"]
    )
    consensus_api_url: str = "https://trajrl.com"
    min_validator_stake: float = 10000.0  # minimum stake weight (α) for consensus participation
    quorum_threshold: float = 0.5  # aggregate only when submitted stake share > threshold

    # Fraction of zero scores above which a submission is treated as a
    # free-rider / near-zero-signal payload and dropped from consensus.
    # 1.0 = only all-zero payloads dropped (legacy behaviour).
    zero_signal_threshold: float = 0.95

    # Bootstrap config (graduated rewards until enough miners join)
    bootstrap_threshold: int = 10

    # Coldkey blacklist: miners under these coldkeys are skipped entirely
    coldkey_blacklist: List[str] = field(default_factory=list)

    # Inactivity tracking (block-based)
    inactivity_blocks: int = 14400  # ~48 hours at 12s/block

    # Weight cadence (set weights every tempo)
    weight_interval_blocks: int = 360  # 1 tempo ≈ 72 min at 12s/block

    # Startup aggregation: run consensus aggregation before entering main loop.
    # Default off: startup aggregation reads chain commitments and trusts them,
    # which is unsafe when chain commitments may be stale/poisoned (e.g. from a
    # prior ws-timeout incident). The main loop's aggregation phase is the
    # authoritative path; the startup variant is now opt-in only.
    aggregate_when_start: bool = False

    # Startup full cycle: run eval → propagation → aggregation before main loop.
    # When enabled, ``aggregate_when_start`` is ignored (full cycle includes it).
    full_cycle_on_startup: bool = False

    # One-shot rescue: when set to a window number N, on startup the validator
    # rolls back persisted state so it re-evaluates and re-submits window N
    # (consensus_window=N-1, target_window=N, target_submit_done=False, and
    # last_eval_window clamped to N-1 if it was at/past N). Per-miner caches
    # (eval_pack_hash, scenario_scores) are preserved so already-evaluated
    # miners with unchanged pack_hash skip re-eval. Also disables
    # ``aggregate_when_start`` and makes ``_check_own_commitment_on_chain(N)``
    # return False so the stale on-chain pointer is overwritten.
    # Operators clear this on the next deploy after rescue completes.
    rescue_resubmit_window: Optional[int] = None

    # Disable winner protection to force all validators to converge on the
    # same lowest-cost winner (use once to clear divergent cached state).
    disable_winner_protection: bool = False

    # LLM configuration (used by sandbox harness and LLM judges).
    llm_model: str = DEFAULT_LLM_MODEL
    llm_api_key: str = ""
    llm_base_url: str = DEFAULT_LLM_BASE_URL

    # LLM Judge configuration (Phase 1 integrity + Phase 2 trajectory)
    # Defaults to the primary LLM config if left empty.
    judge_model: str = ""
    judge_api_key: str = ""
    judge_base_url: str = ""

    # trajrl-bench sandbox evaluation
    # image_channel is the canonical knob: it selects the tag (e.g. "latest",
    # "staging", "v1.2.0-rc.1") used to construct sandbox_image / harness_image
    # when those are left empty. Direct programmatic assignment to
    # sandbox_image / harness_image (e.g. tests) still wins.
    image_channel: str = DEFAULT_IMAGE_CHANNEL
    sandbox_image: str = ""
    harness_image: str = ""
    sandbox_timeout_per_episode: int = 600  # 10 min per episode (Qwen3-class testees)
    sandbox_num_episodes: int = 4
    # Scenario the sandbox CLI should generate fixtures for. The CLI
    # supports incident_response, morning_brief, and codebase_fix (see
    # trajrl_bench.fixture_factory.SCENARIOS). Default is codebase_fix
    # — the newest scenario and the one the mistakes-and-memory
    # learning rubric is designed around; operators can still pin an
    # older scenario via ``SANDBOX_SCENARIO``.
    sandbox_scenario: str = "codebase_fix"

    # Evaluation state persistence
    eval_state_path: Path = field(
        default_factory=lambda: Path("/var/lib/trajectoryrl/eval_state.json")
    )

    # Winner Protection state persistence
    winner_state_path: Path = field(
        default_factory=lambda: Path("/var/lib/trajectoryrl/winner_state.json")
    )

    # Pack ownership lock (pack_first_seen) persistence. Separate from
    # eval_state_path so the ownership table can be inspected / reset
    # independently of per-hotkey eval caches. On first start, legacy
    # entries embedded in eval_state.json are migrated automatically
    # (see BaseValidatorNeuron._load_eval_state).
    pack_first_seen_path: Path = field(
        default_factory=lambda: Path("/var/lib/trajectoryrl/pack_first_seen.json")
    )

    # Per-window active-set snapshot directory. One file per window
    # (``active_set_window_{N}.json``) freezes the deterministic
    # commitment subset for that window so that mid-window restarts
    # rehydrate the same eval set instead of re-querying the chain.
    active_set_dir: Path = field(
        default_factory=lambda: Path("/var/lib/trajectoryrl/active_sets")
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
        """Create required directories and derive image refs from channel."""
        self.pack_cache_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        if not self.sandbox_image:
            self.sandbox_image = f"{SANDBOX_IMAGE_REPO}:{self.image_channel}"
        if not self.harness_image:
            self.harness_image = f"{HARNESS_IMAGE_REPO}:{self.image_channel}"

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
            eval_state_path=Path(os.getenv("EVAL_STATE_PATH", "/var/lib/trajectoryrl/eval_state.json")),
            winner_state_path=Path(os.getenv("WINNER_STATE_PATH", "/var/lib/trajectoryrl/winner_state.json")),
            pack_first_seen_path=Path(os.getenv("PACK_FIRST_SEEN_PATH", "/var/lib/trajectoryrl/pack_first_seen.json")),
            active_set_dir=Path(os.getenv("ACTIVE_SET_DIR", "/var/lib/trajectoryrl/active_sets")),
            pack_cache_dir=Path(os.getenv("PACK_CACHE_DIR", "/var/lib/trajectoryrl/packs")),
            log_dir=Path(os.getenv("LOG_DIR", "./logs")),
            # --- LLM ---
            llm_model=os.getenv("LLM_MODEL", DEFAULT_LLM_MODEL),
            llm_api_key=os.getenv("LLM_API_KEY", ""),
            llm_base_url=os.getenv("LLM_BASE_URL", DEFAULT_LLM_BASE_URL),
            # --- LLM Judge (optional, falls back to primary LLM if empty) ---
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
            quorum_threshold=float(os.getenv("QUORUM_THRESHOLD", "0.5")),
            # --- trajrl-bench ---
            # IMAGE_CHANNEL is the only env-driven knob: it selects the tag
            # for both sandbox and harness images. Programmatic callers can
            # still pass sandbox_image / harness_image directly to bypass it.
            image_channel=os.getenv("IMAGE_CHANNEL", DEFAULT_IMAGE_CHANNEL),
            sandbox_timeout_per_episode=int(os.getenv("SANDBOX_TIMEOUT_PER_EPISODE", "600")),
            sandbox_num_episodes=int(os.getenv("SANDBOX_NUM_EPISODES", "4")),
            sandbox_scenario=os.getenv("SANDBOX_SCENARIO", "codebase_fix"),
            # --- Startup aggregation ---
            aggregate_when_start=os.getenv("AGGREGATE_WHEN_START", "0") == "1",
            full_cycle_on_startup=os.getenv("FULL_CYCLE_ON_STARTUP", "0") == "1",
            disable_winner_protection=os.getenv("DISABLE_WINNER_PROTECTION", "1") == "1",
            # --- One-shot rescue (operators set RESCUE_RESUBMIT_WINDOW=N to
            # force re-eval+resubmit of window N on next restart). ---
            rescue_resubmit_window=(
                int(os.environ["RESCUE_RESUBMIT_WINDOW"])
                if os.environ.get("RESCUE_RESUBMIT_WINDOW", "").strip()
                else None
            ),
            # --- IM parameters are hardcoded (dataclass defaults) ---
            # Do NOT load from env: score_delta,
            # rho_reliability, consensus_epsilon, bootstrap_threshold,
            # max_commitment_age_blocks,
            # inactivity_blocks, eval_interval_blocks, weight_interval_blocks.
            # All validators must use identical IM values for consensus.
        )


@dataclass
class MinerConfig:
    """Configuration for TrajectoryRL miner CLI.

    Attributes:
        wallet_name: Wallet name
        wallet_hotkey: Hotkey name
        netuid: Subnet UID (11 for TrajectoryRL)
        network: Bittensor network (finney, test, local)
        log_level: Logging level
    """

    wallet_name: str = "miner"
    wallet_hotkey: str = "default"
    netuid: int = 11
    network: str = "finney"

    log_level: str = "INFO"

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
            log_level=os.getenv("LOG_LEVEL", "INFO"),
        )
