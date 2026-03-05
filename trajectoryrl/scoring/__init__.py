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

        eps = self.consensus_epsilon

        # First-mover iterative selection: start with the earliest miner
        # as champion, then iterate through subsequent miners in block
        # order.  A later miner must beat the current champion's score
        # by more than δ to take the crown.  This naturally handles
        # transitive protection (A protects against B which protects
        # against C) without needing recursive checks.
        #
        # Miners without first_mover_data are appended at the end with
        # infinite block number so they can still participate but never
        # receive first-mover protection.

        # Build (uid, score, block_number) tuples, sorted by block asc
        uid_entries = []
        for uid, score in scores.items():
            hk = _uid_to_hotkey.get(uid)
            block = (
                first_mover_data[hk][1]
                if hk and hk in first_mover_data
                else float("inf")
            )
            uid_entries.append((uid, score, block))

        uid_entries.sort(key=lambda e: e[2])  # earliest first

        # The earliest miner is the initial champion
        best_uid, best_score, _ = uid_entries[0]

        for uid, score, _ in uid_entries[1:]:
            # Consensus epsilon: scores within ε are treated as tied,
            # and the earlier miner (current champion) keeps the crown.
            if score > best_score + eps:
                # Challenger must also beat δ threshold
                if score > best_score + delta:
                    logger.info(
                        f"First-mover overtake: Miner {uid} (score={score:.3f}) "
                        f"beats champion Miner {best_uid} (score={best_score:.3f}, "
                        f"required>{best_score + delta:.3f})"
                    )
                    best_uid = uid
                    best_score = score

        # Winner takes all
        weights = {uid: 0.0 for uid in scores.keys()}
        weights[best_uid] = 1.0

        logger.info(
            f"Winner-take-all: Miner {best_uid} wins with score {best_score:.3f} "
            f"(delta={delta}, ε={eps})"
        )

        return weights

    def select_winner_by_cost(
        self,
        costs: Dict[int, float],
        qualified: Dict[int, bool],
        first_mover_data: Dict[str, Tuple[float, float]],
        cost_delta: float = 0.10,
        num_active_miners: Optional[int] = None,
        uid_to_hotkey: Optional[Dict[int, str]] = None,
    ) -> Dict[int, float]:
        """Select winner by lowest cost, with qualification gate.

        Only qualified miners (all safety + correctness checks passed)
        compete. Among qualified miners, the one with lowest cost wins.
        A challenger must be at least cost_delta (10%) cheaper than the
        current champion to dethrone them (multiplicative first-mover
        protection).

        Args:
            costs: Dict of miner_uid -> cost_usd
            qualified: Dict of miner_uid -> bool (gate pass)
            first_mover_data: Dict of hotkey -> (best_cost, block_number)
            cost_delta: Multiplicative threshold (challenger < champion * (1 - delta))
            num_active_miners: Total active miners for bootstrap check
            uid_to_hotkey: Dict of miner_uid -> hotkey

        Returns:
            Dict of miner_uid -> weight (sums to 1.0)
        """
        if not costs:
            return {}

        _uid_to_hotkey = uid_to_hotkey or {}

        # Filter to qualified miners only
        qualified_uids = {uid for uid, q in qualified.items() if q}
        qualified_costs = {uid: c for uid, c in costs.items() if uid in qualified_uids}

        if not qualified_costs:
            logger.warning("No qualified miners — all disqualified by gate")
            return {uid: 0.0 for uid in costs}

        n_miners = num_active_miners if num_active_miners is not None else len(qualified_costs)

        # Bootstrap phase: graduated rewards
        if n_miners < self.bootstrap_threshold:
            return self._bootstrap_weights_by_cost(
                qualified_costs, first_mover_data, _uid_to_hotkey, costs,
            )

        # Steady-state: winner-take-all by lowest cost
        # Build (uid, cost, block_number) tuples, sorted by block asc
        uid_entries = []
        for uid, cost in qualified_costs.items():
            hk = _uid_to_hotkey.get(uid)
            block = (
                first_mover_data[hk][1]
                if hk and hk in first_mover_data
                else float("inf")
            )
            uid_entries.append((uid, cost, block))

        uid_entries.sort(key=lambda e: e[2])  # earliest first

        # Earliest qualified miner is the initial champion
        best_uid, best_cost, _ = uid_entries[0]

        for uid, cost, _ in uid_entries[1:]:
            # Challenger must be significantly cheaper (multiplicative delta)
            if cost < best_cost * (1 - cost_delta):
                logger.info(
                    f"Cost overtake: Miner {uid} (${cost:.4f}) beats "
                    f"champion Miner {best_uid} (${best_cost:.4f}, "
                    f"required<${best_cost * (1 - cost_delta):.4f})"
                )
                best_uid = uid
                best_cost = cost

        # Winner takes all; disqualified miners get 0
        weights = {uid: 0.0 for uid in costs}
        weights[best_uid] = 1.0

        logger.info(
            f"Cost winner-take-all: Miner {best_uid} wins with "
            f"cost=${best_cost:.4f} (delta={cost_delta})"
        )

        return weights

    def _bootstrap_weights_by_cost(
        self,
        qualified_costs: Dict[int, float],
        first_mover_data: Dict[str, Tuple[float, float]],
        uid_to_hotkey: Dict[int, str],
        all_costs: Dict[int, float],
    ) -> Dict[int, float]:
        """Graduated reward curve for bootstrap phase, ranked by cost.

        Top-3 lowest-cost qualified miners receive 70% / 20% / 10%.
        Ties broken by earliest block number.

        Args:
            qualified_costs: Qualified miner_uid -> cost
            first_mover_data: hotkey -> (best_cost, block_number)
            uid_to_hotkey: miner_uid -> hotkey
            all_costs: All miner costs (for zero-weight entries)

        Returns:
            Dict of miner_uid -> weight (sums to 1.0)
        """
        BOOTSTRAP_SHARES = [0.70, 0.20, 0.10]

        # Sort by cost asc (lowest first), break ties by earliest block
        def sort_key(uid: int) -> Tuple[float, float]:
            hk = uid_to_hotkey.get(uid)
            ts = first_mover_data[hk][1] if hk and hk in first_mover_data else float("inf")
            return (qualified_costs[uid], ts)

        ranked = sorted(qualified_costs.keys(), key=sort_key)

        weights = {uid: 0.0 for uid in all_costs}
        for i, uid in enumerate(ranked):
            if i < len(BOOTSTRAP_SHARES):
                weights[uid] = BOOTSTRAP_SHARES[i]

        total = sum(weights.values())
        if total > 0 and total != 1.0:
            weights = {uid: w / total for uid, w in weights.items()}

        logger.info(
            f"Bootstrap cost phase ({len(qualified_costs)} qualified miners): "
            f"graduated rewards {dict((uid, w) for uid, w in weights.items() if w > 0)}"
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
