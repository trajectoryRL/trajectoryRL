"""Block-based evaluation window computation.

All validators operate on synchronized evaluation windows derived from chain
block height.  Any validator can independently compute the current window
number and phase -- no central coordination needed.

Window phases (80/10/10 split):
  - evaluation  (block 0 .. T_publish):   run ClawBench, compute local cost EMA
  - submission  (T_publish):               upload payload to CAS, register pointer
  - propagation (T_publish .. T_aggregate): wait for submissions to propagate
  - aggregation (T_aggregate .. end):       read submissions, filter, consensus
"""

from dataclasses import dataclass
from enum import Enum


class WindowPhase(str, Enum):
    EVALUATION = "evaluation"
    SUBMISSION = "submission"
    PROPAGATION = "propagation"
    AGGREGATION = "aggregation"


@dataclass
class WindowConfig:
    """Configuration for evaluation window timing.

    All validators must use identical values for deterministic synchronization.
    """
    window_length: int = 7200       # blocks per window (~24h at 12s/block)
    global_anchor: int = 0          # anchor block for window alignment
    publish_pct: float = 0.80       # T_publish as fraction of window
    aggregate_pct: float = 0.90     # T_aggregate as fraction of window

    @property
    def publish_block(self) -> int:
        return int(self.window_length * self.publish_pct)

    @property
    def aggregate_block(self) -> int:
        return int(self.window_length * self.aggregate_pct)


@dataclass(frozen=True)
class EvaluationWindow:
    """Snapshot of the current evaluation window state."""
    window_number: int
    window_start: int
    block_offset: int
    phase: WindowPhase
    blocks_into_phase: int
    blocks_remaining_in_phase: int

    @property
    def publish_deadline_block(self) -> int:
        """Absolute block number of T_publish for this window."""
        return self.window_start + self._publish_block

    @property
    def aggregate_start_block(self) -> int:
        """Absolute block number of T_aggregate for this window."""
        return self.window_start + self._aggregate_block

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)


def compute_window(current_block: int, config: WindowConfig) -> EvaluationWindow:
    """Compute evaluation window state from current block height.

    Pure function -- deterministic for any given (current_block, config) pair.
    Every validator calling this with the same inputs gets the same result.
    """
    if current_block < config.global_anchor:
        effective_block = config.global_anchor
    else:
        effective_block = current_block

    blocks_since_anchor = effective_block - config.global_anchor
    window_number = blocks_since_anchor // config.window_length
    window_start = config.global_anchor + window_number * config.window_length
    block_offset = effective_block - window_start

    publish_block = config.publish_block
    aggregate_block = config.aggregate_block

    if block_offset < publish_block:
        phase = WindowPhase.EVALUATION
        blocks_into = block_offset
        blocks_remaining = publish_block - block_offset
    elif block_offset < aggregate_block:
        phase = WindowPhase.PROPAGATION
        blocks_into = block_offset - publish_block
        blocks_remaining = aggregate_block - block_offset
    else:
        phase = WindowPhase.AGGREGATION
        blocks_into = block_offset - aggregate_block
        blocks_remaining = config.window_length - block_offset

    return EvaluationWindow(
        window_number=window_number,
        window_start=window_start,
        block_offset=block_offset,
        phase=phase,
        blocks_into_phase=blocks_into,
        blocks_remaining_in_phase=blocks_remaining,
    )


def is_new_window(current_block: int, last_window_number: int,
                  config: WindowConfig) -> bool:
    """Check if current_block is in a different window than last_window_number."""
    window = compute_window(current_block, config)
    return window.window_number > last_window_number


def should_submit(current_block: int, config: WindowConfig) -> bool:
    """Check if the validator should submit evaluation results now.

    Returns True when the block offset is at or past T_publish but
    before T_aggregate (i.e., in the propagation phase -- the moment
    to submit is at the start of this phase).
    """
    window = compute_window(current_block, config)
    return window.phase == WindowPhase.PROPAGATION


def should_aggregate(current_block: int, config: WindowConfig) -> bool:
    """Check if the validator should run consensus aggregation now."""
    window = compute_window(current_block, config)
    return window.phase == WindowPhase.AGGREGATION


def can_evaluate(current_block: int, config: WindowConfig) -> bool:
    """Check if we are in the evaluation phase (safe to run ClawBench)."""
    window = compute_window(current_block, config)
    return window.phase == WindowPhase.EVALUATION


def window_progress_pct(current_block: int, config: WindowConfig) -> float:
    """Return how far through the current window we are (0.0 to 1.0)."""
    window = compute_window(current_block, config)
    return window.block_offset / config.window_length
