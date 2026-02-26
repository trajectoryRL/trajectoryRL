"""Scoring functions for policy pack evaluation."""

import logging
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

from ..utils.clawbench import EvaluationResult

logger = logging.getLogger(__name__)


@dataclass
class AggregatedScore:
    """Aggregated score across multiple evaluations.

    Attributes:
        mean_score: Weighted average score across scenarios
        variance: Weighted variance in scores (for reliability penalty)
        success_rate: Fraction of successful evaluations
        total_evaluations: Number of evaluations run
        scenario_scores: Dict of scenario_name -> score
        scenario_weights: Dict of scenario_name -> weight (used in mean)
    """
    mean_score: float
    variance: float
    success_rate: float
    total_evaluations: int
    scenario_scores: Dict[str, float]
    scenario_weights: Dict[str, float] = field(default_factory=dict)


class TrajectoryScorer:
    """Scores policy packs based on ClawBench scenario results."""

    def __init__(
        self,
        rho_reliability: float = 0.1,
        consensus_epsilon: float = 0.02,
        bootstrap_threshold: int = 10,
    ):
        """Initialize scorer.

        Args:
            rho_reliability: Weight for variance penalty
            consensus_epsilon: Scores within ε of each other are treated as
                tied. Ties go to the first-mover.
            bootstrap_threshold: When active miners < this, use graduated
                reward curve instead of winner-take-all.
        """
        self.rho_reliability = rho_reliability
        self.consensus_epsilon = consensus_epsilon
        self.bootstrap_threshold = bootstrap_threshold

        logger.info(
            f"Scorer initialized: ρ={rho_reliability}, "
            f"ε={consensus_epsilon}, bootstrap_threshold={bootstrap_threshold}"
        )

    def aggregate_scores(
        self,
        results: List[EvaluationResult],
        scenario_weights: Optional[Dict[str, float]] = None,
    ) -> AggregatedScore:
        """Aggregate multiple evaluation results with optional scenario weights.

        When ``scenario_weights`` is provided, the mean score and variance are
        computed as a weighted average/variance across scenarios.  Weights are
        read from the ``weight`` field in each scenario YAML (default 1.0).

        Args:
            results: List of EvaluationResult from multiple scenarios/seeds
            scenario_weights: Optional dict of scenario_name -> weight.
                If None, all scenarios are weighted equally.

        Returns:
            AggregatedScore with weighted mean, variance, success rate
        """
        if not results:
            return AggregatedScore(
                mean_score=0.0,
                variance=0.0,
                success_rate=0.0,
                total_evaluations=0,
                scenario_scores={},
                scenario_weights={},
            )

        successes = [r.success for r in results]
        success_rate = float(np.mean(successes))

        # Group by scenario
        scenario_groups: Dict[str, List[float]] = {}
        for result in results:
            if result.scenario_name not in scenario_groups:
                scenario_groups[result.scenario_name] = []
            scenario_groups[result.scenario_name].append(result.score)

        # Average within scenarios
        scenario_scores = {
            name: float(np.mean(scores))
            for name, scores in scenario_groups.items()
        }

        # Resolve weights (default 1.0 for missing entries)
        weights = {}
        for name in scenario_scores:
            weights[name] = (
                scenario_weights.get(name, 1.0)
                if scenario_weights
                else 1.0
            )

        # Weighted mean across scenarios
        total_weight = sum(weights.values())
        if total_weight > 0:
            mean_score = sum(
                scenario_scores[n] * weights[n] for n in scenario_scores
            ) / total_weight
        else:
            mean_score = 0.0

        # Weighted variance
        if len(scenario_scores) > 1 and total_weight > 0:
            variance = sum(
                weights[n] * (scenario_scores[n] - mean_score) ** 2
                for n in scenario_scores
            ) / total_weight
        else:
            variance = 0.0

        logger.info(
            f"Aggregated {len(results)} results: "
            f"mean={mean_score:.3f}, var={variance:.3f}, success={success_rate:.1%}"
        )
        if scenario_weights:
            logger.debug(f"Scenario weights: {weights}")

        return AggregatedScore(
            mean_score=mean_score,
            variance=variance,
            success_rate=success_rate,
            total_evaluations=len(results),
            scenario_scores=scenario_scores,
            scenario_weights=weights,
        )

    def compute_final_score(
        self,
        aggregated: AggregatedScore
    ) -> float:
        """Compute final score with reliability penalty.

        Args:
            aggregated: AggregatedScore from multiple evaluations

        Returns:
            Final score in [0, 1]
        """
        # Apply variance penalty
        reliability_penalty = self.rho_reliability * aggregated.variance
        final = aggregated.mean_score - reliability_penalty

        # Clamp to [0, 1]
        final = max(0.0, min(1.0, final))

        logger.debug(
            f"Final score: {final:.3f} = "
            f"{aggregated.mean_score:.3f} - {reliability_penalty:.3f}"
        )

        return final

    def select_winner(
        self,
        scores: Dict[int, float],
        first_mover_data: Dict[str, Tuple[float, float]],
        delta: float = 0.05,
        num_active_miners: Optional[int] = None,
        uid_to_hotkey: Optional[Dict[int, str]] = None,
    ) -> Dict[int, float]:
        """Select winner using winner-take-all with first-mover advantage.

        When the subnet has fewer active miners than ``bootstrap_threshold``,
        rewards are distributed using a graduated curve (top-3 get 70/20/10)
        to encourage early adoption.  Once the miner count reaches the
        threshold, pure winner-take-all resumes.

        Consensus-safe: miners whose scores differ by less than
        ``consensus_epsilon`` are treated as tied. Ties are broken in favour
        of the earliest submitter (by on-chain block number), which every
        validator can resolve identically.

        Args:
            scores: Dict of miner_uid -> score [0, 1]
            first_mover_data: Dict of hotkey -> (score, block_number).
                block_number is the on-chain block at which the miner first
                submitted; lower block number = earlier submission.
            delta: First-mover threshold (new score must beat best + delta)
            num_active_miners: Total active miners in metagraph.
                If None, defaults to len(scores).
            uid_to_hotkey: Dict of miner_uid -> hotkey. Required to bridge
                UID-keyed scores with hotkey-keyed first_mover_data.

        Returns:
            Dict of miner_uid -> weight (sums to 1.0)
        """
        if not scores:
            return {}

        _uid_to_hotkey = uid_to_hotkey or {}

        n_miners = num_active_miners if num_active_miners is not None else len(scores)

        # --- Bootstrap phase: graduated rewards ---
        if n_miners < self.bootstrap_threshold:
            return self._bootstrap_weights(scores, first_mover_data, _uid_to_hotkey)

        # --- Steady-state: winner-take-all ---

        # Find current best score
        best_uid = max(scores.keys(), key=lambda uid: scores[uid])
        best_score = scores[best_uid]

        # Consensus epsilon: if multiple miners are within ε of the best,
        # treat them as tied and pick the earliest submitter.
        eps = self.consensus_epsilon
        tied_uids = [
            uid for uid, s in scores.items()
            if abs(s - best_score) <= eps
        ]

        resolved_by_epsilon = False
        if len(tied_uids) > 1 and first_mover_data:
            # Break tie by earliest block number (look up by hotkey)
            tied_with_ts = []
            for uid in tied_uids:
                hk = _uid_to_hotkey.get(uid)
                if hk and hk in first_mover_data:
                    tied_with_ts.append((uid, first_mover_data[hk][1]))
            if tied_with_ts:
                tied_with_ts.sort(key=lambda x: x[1])  # lowest block first
                best_uid = tied_with_ts[0][0]
                best_score = scores[best_uid]
                resolved_by_epsilon = True
                logger.info(
                    f"Consensus tie-break: {len(tied_uids)} miners within "
                    f"ε={eps} of {best_score:.3f} → earliest is Miner {best_uid}"
                )

        # Apply first-mover protection (constant δ threshold).
        # Only an EARLIER submitter can block a later challenger.
        # Skip if winner was already resolved by epsilon tie-break (which
        # already used block numbers, so re-applying delta would undo it).
        if first_mover_data and not resolved_by_epsilon:
            # Look up the current best's block number
            best_hotkey = _uid_to_hotkey.get(best_uid)
            best_block = (
                first_mover_data[best_hotkey][1]
                if best_hotkey and best_hotkey in first_mover_data
                else float("inf")
            )

            for hotkey, (_, ts) in sorted(
                first_mover_data.items(),
                key=lambda x: x[1][1]  # Sort by block number (ascending)
            ):
                # Only earlier submitters can block the current best
                if ts >= best_block:
                    continue

                # Reverse-lookup: find the UID currently using this hotkey
                uid = next(
                    (u for u, hk in _uid_to_hotkey.items() if hk == hotkey),
                    None,
                )
                if uid is not None and uid in scores:
                    # Use current score (not historical best) for protection
                    # threshold so stale packs lose protection naturally.
                    current = scores[uid]
                    required_score = current + delta
                    if best_score <= required_score:
                        logger.info(
                            f"First-mover protection: Miner {uid} (score={current:.3f}) "
                            f"blocks Miner {best_uid} (score={best_score:.3f}, "
                            f"required={required_score:.3f})"
                        )
                        best_uid = uid
                        best_score = current
                        break

        # Winner takes all
        weights = {uid: 0.0 for uid in scores.keys()}
        weights[best_uid] = 1.0

        logger.info(
            f"Winner-take-all: Miner {best_uid} wins with score {best_score:.3f} "
            f"(delta={delta}, ε={eps})"
        )

        return weights

    def _bootstrap_weights(
        self,
        scores: Dict[int, float],
        first_mover_data: Dict[str, Tuple[float, float]],
        uid_to_hotkey: Dict[int, str],
    ) -> Dict[int, float]:
        """Graduated reward curve for the bootstrap phase.

        Top-3 miners receive 70% / 20% / 10% of rewards. Ties are broken
        by earliest block number (same rule as steady-state).

        Args:
            scores: Dict of miner_uid -> score
            first_mover_data: Dict of hotkey -> (score, block_number)
            uid_to_hotkey: Dict of miner_uid -> hotkey

        Returns:
            Dict of miner_uid -> weight (sums to 1.0)
        """
        BOOTSTRAP_SHARES = [0.70, 0.20, 0.10]

        # Sort by score desc, breaking ties by earliest block number
        def sort_key(uid: int) -> Tuple[float, float]:
            hk = uid_to_hotkey.get(uid)
            ts = first_mover_data[hk][1] if hk and hk in first_mover_data else float("inf")
            return (-scores[uid], ts)

        ranked = sorted(scores.keys(), key=sort_key)

        weights = {uid: 0.0 for uid in scores.keys()}
        for i, uid in enumerate(ranked):
            if i < len(BOOTSTRAP_SHARES):
                weights[uid] = BOOTSTRAP_SHARES[i]

        # Normalize so weights always sum to 1.0 (handles <3 miners)
        total = sum(weights.values())
        if total > 0 and total != 1.0:
            weights = {uid: w / total for uid, w in weights.items()}

        logger.info(
            f"Bootstrap phase ({len(scores)} miners < {self.bootstrap_threshold}): "
            f"graduated rewards {dict((uid, w) for uid, w in weights.items() if w > 0)}"
        )

        return weights
