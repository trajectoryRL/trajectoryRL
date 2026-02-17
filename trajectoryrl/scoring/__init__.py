"""Scoring functions for policy pack evaluation."""

import logging
import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass

from ..utils.clawbench import EvaluationResult

logger = logging.getLogger(__name__)


@dataclass
class AggregatedScore:
    """Aggregated score across multiple evaluations.

    Attributes:
        mean_score: Average score across scenarios/seeds
        variance: Variance in scores (for reliability penalty)
        success_rate: Fraction of successful evaluations
        total_evaluations: Number of evaluations run
        scenario_scores: Dict of scenario_name -> score
    """
    mean_score: float
    variance: float
    success_rate: float
    total_evaluations: int
    scenario_scores: Dict[str, float]

    @property
    def final_score(self) -> float:
        """Compute final score with reliability penalty.

        Returns:
            Final score in [0, 1]
        """
        return max(0.0, min(1.0, self.mean_score))


class TrajectoryScorer:
    """Scores policy packs based on ClawBench scenario results."""

    def __init__(
        self,
        lambda_cost: float = 0.3,
        mu_safety: float = 0.4,
        rho_reliability: float = 0.1,
        score_quantization: float = 0.05,
        consensus_epsilon: float = 0.02,
    ):
        """Initialize scorer.

        Args:
            lambda_cost: Weight for cost penalty (unused in ClawBench)
            mu_safety: Weight for safety penalty (already in ClawBench scoring)
            rho_reliability: Weight for variance penalty
            score_quantization: Round final scores to this grid (e.g., 0.05 →
                scores snap to 0.00, 0.05, 0.10, …). Reduces validator
                disagreement caused by LLM non-determinism.
            consensus_epsilon: Scores within ε of each other are treated as
                tied. Ties go to the first-mover.
        """
        self.lambda_cost = lambda_cost
        self.mu_safety = mu_safety
        self.rho_reliability = rho_reliability
        self.score_quantization = score_quantization
        self.consensus_epsilon = consensus_epsilon

        logger.info(
            f"Scorer initialized: λ={lambda_cost}, μ={mu_safety}, "
            f"ρ={rho_reliability}, q={score_quantization}, ε={consensus_epsilon}"
        )

    def aggregate_scores(
        self,
        results: List[EvaluationResult]
    ) -> AggregatedScore:
        """Aggregate multiple evaluation results.

        Args:
            results: List of EvaluationResult from multiple scenarios/seeds

        Returns:
            AggregatedScore with mean, variance, success rate
        """
        if not results:
            return AggregatedScore(
                mean_score=0.0,
                variance=0.0,
                success_rate=0.0,
                total_evaluations=0,
                scenario_scores={}
            )

        # Extract scores
        scores = [r.score for r in results]
        successes = [r.success for r in results]

        # Compute statistics
        mean_score = float(np.mean(scores))
        variance = float(np.var(scores)) if len(scores) > 1 else 0.0
        success_rate = float(np.mean(successes))

        # Group by scenario
        scenario_scores = {}
        for result in results:
            if result.scenario_name not in scenario_scores:
                scenario_scores[result.scenario_name] = []
            scenario_scores[result.scenario_name].append(result.score)

        # Average within scenarios
        scenario_scores = {
            name: float(np.mean(scores))
            for name, scores in scenario_scores.items()
        }

        logger.info(
            f"Aggregated {len(results)} results: "
            f"mean={mean_score:.3f}, var={variance:.3f}, success={success_rate:.1%}"
        )

        return AggregatedScore(
            mean_score=mean_score,
            variance=variance,
            success_rate=success_rate,
            total_evaluations=len(results),
            scenario_scores=scenario_scores
        )

    def compute_final_score(
        self,
        aggregated: AggregatedScore
    ) -> float:
        """Compute final score with reliability penalty and quantization.

        Quantization snaps scores to a fixed grid so that independent
        validators are far more likely to arrive at the same number despite
        LLM non-determinism in individual runs.

        Args:
            aggregated: AggregatedScore from multiple evaluations

        Returns:
            Quantized final score in [0, 1]
        """
        # Apply variance penalty
        reliability_penalty = self.rho_reliability * aggregated.variance
        final = aggregated.mean_score - reliability_penalty

        # Clamp to [0, 1]
        final = max(0.0, min(1.0, final))

        # Quantize
        final = self.quantize_score(final)

        logger.debug(
            f"Final score: {final:.3f} = "
            f"quantize({aggregated.mean_score:.3f} - {reliability_penalty:.3f})"
        )

        return final

    def quantize_score(self, score: float) -> float:
        """Round score to the nearest quantization step.

        Example with q=0.05: 0.87 → 0.85, 0.88 → 0.90, 0.925 → 0.90

        Args:
            score: Raw score in [0, 1]

        Returns:
            Quantized score in [0, 1]
        """
        if self.score_quantization <= 0:
            return score
        q = self.score_quantization
        return round(round(score / q) * q, 10)  # round(, 10) avoids float drift

    def select_winner(
        self,
        scores: Dict[int, float],
        first_mover_data: Dict[int, Tuple[float, float]],
        delta: float = 0.05,
    ) -> Dict[int, float]:
        """Select winner using winner-take-all with first-mover advantage.

        Consensus-safe: miners whose quantized scores differ by less than
        ``consensus_epsilon`` are treated as tied. Ties are broken in favour
        of the earliest submitter (by push timestamp), which every validator
        can resolve identically.

        Anti-stagnation is handled by epoch-seeded scenario variation (not
        by decaying δ).  See ``TrajectoryValidator._select_epoch_scenarios``.

        Args:
            scores: Dict of miner_uid -> quantized score [0, 1]
            first_mover_data: Dict of miner_uid -> (score, timestamp)
            delta: First-mover threshold (new score must beat best + delta)

        Returns:
            Dict of miner_uid -> weight (winner=1.0, others=0.0)
        """
        if not scores:
            return {}

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
            # Break tie by earliest push timestamp
            tied_with_ts = [
                (uid, first_mover_data[uid][1])
                for uid in tied_uids
                if uid in first_mover_data
            ]
            if tied_with_ts:
                tied_with_ts.sort(key=lambda x: x[1])  # earliest first
                best_uid = tied_with_ts[0][0]
                best_score = scores[best_uid]
                resolved_by_epsilon = True
                logger.info(
                    f"Consensus tie-break: {len(tied_uids)} miners within "
                    f"ε={eps} of {best_score:.3f} → earliest is Miner {best_uid}"
                )

        # Apply first-mover protection (constant δ threshold).
        # Skip if winner was already resolved by epsilon tie-break (which
        # already used timestamps, so re-applying delta would undo it).
        if first_mover_data and not resolved_by_epsilon:
            for uid, (first_score, _) in sorted(
                first_mover_data.items(),
                key=lambda x: x[1][1]  # Sort by timestamp
            ):
                if uid in scores:
                    required_score = first_score + delta
                    if best_score <= required_score and uid != best_uid:
                        logger.info(
                            f"First-mover protection: Miner {uid} (score={first_score:.3f}) "
                            f"blocks Miner {best_uid} (score={best_score:.3f}, "
                            f"required={required_score:.3f})"
                        )
                        best_uid = uid
                        best_score = scores[uid]
                        break

        # Winner takes all
        weights = {uid: 0.0 for uid in scores.keys()}
        weights[best_uid] = 1.0

        logger.info(
            f"Winner-take-all: Miner {best_uid} wins with score {best_score:.3f} "
            f"(delta={delta}, ε={eps})"
        )

        return weights

    def normalize_scores_to_weights(
        self,
        scores: Dict[int, float],
        temperature: float = 0.1
    ) -> Dict[int, float]:
        """Normalize miner scores to weights using softmax.

        DEPRECATED: Use select_winner() for winner-take-all mechanism.

        Args:
            scores: Dict of miner_uid -> score [0, 1]
            temperature: Softmax temperature (lower = more concentrated)

        Returns:
            Dict of miner_uid -> weight (sums to 1.0)
        """
        logger.warning(
            "normalize_scores_to_weights() is DEPRECATED. "
            "Use select_winner() for winner-take-all mechanism."
        )

        if not scores:
            return {}

        uids = list(scores.keys())
        raw_scores = np.array([scores[uid] for uid in uids])

        # Softmax normalization
        exp_scores = np.exp(raw_scores / temperature)
        weights = exp_scores / exp_scores.sum()

        result = {uid: float(w) for uid, w in zip(uids, weights)}

        logger.info(
            f"Normalized {len(scores)} scores to weights "
            f"(temp={temperature}, spread={weights.std():.3f})"
        )

        return result
