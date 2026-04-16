"""Scoring functions for policy pack evaluation."""

import logging
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

from ..utils.consensus_filter import ValidatedSubmission

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
    """Scores policy packs based on trajrl-bench evaluation results."""

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
        results: list,
        scenario_weights: Optional[Dict[str, float]] = None,
    ) -> AggregatedScore:
        """Aggregate multiple evaluation results with optional scenario weights.

        When ``scenario_weights`` is provided, the mean score and variance are
        computed as a weighted average/variance across scenarios.  Weights are
        read from the ``weight`` field in each scenario YAML (default 1.0).

        Args:
            results: List of result objects with .scenario_name, .score, .success
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
        champion_hotkey: Optional[str] = None,
    ) -> Dict[int, float]:
        """Select winner using winner-take-all with first-mover advantage.

        When the subnet has fewer active miners than ``bootstrap_threshold``,
        rewards are distributed using a graduated curve (top-3 get 70/20/10)
        to encourage early adoption.  Once the miner count reaches the
        threshold, pure winner-take-all resumes.

        The δ threshold only protects the **current champion** (the miner
        who won in the previous cycle).  A challenger must score more than
        δ above the champion to dethrone them.  If no previous champion is
        known, the earliest miner (by on-chain block) is used.

        Args:
            scores: Dict of miner_uid -> score [0, 1]
            first_mover_data: Dict of hotkey -> (score, block_number).
                block_number is the on-chain block at which the miner
                submitted; lower block number = earlier submission.
            delta: First-mover threshold (new score must beat best + delta)
            num_active_miners: Total active miners in metagraph.
                If None, defaults to len(scores).
            uid_to_hotkey: Dict of miner_uid -> hotkey. Required to bridge
                UID-keyed scores with hotkey-keyed first_mover_data.
            champion_hotkey: Hotkey of the previous cycle's winner.
                If None, the earliest miner (by block) is used.

        Returns:
            Dict of miner_uid -> weight (sums to 1.0)
        """
        if not scores:
            return {}

        _uid_to_hotkey = uid_to_hotkey or {}
        _hotkey_to_uid = {hk: uid for uid, hk in _uid_to_hotkey.items()}

        n_miners = num_active_miners if num_active_miners is not None else len(scores)

        # --- Bootstrap phase: graduated rewards ---
        if n_miners < self.bootstrap_threshold:
            return self._bootstrap_weights(scores, first_mover_data, _uid_to_hotkey)

        # --- Steady-state: champion vs all challengers ---

        eps = self.consensus_epsilon

        # Resolve champion UID
        champion_uid = None
        if champion_hotkey and champion_hotkey in _hotkey_to_uid:
            cuid = _hotkey_to_uid[champion_hotkey]
            if cuid in scores:
                champion_uid = cuid

        # Fallback: earliest miner by block number
        if champion_uid is None:
            earliest_block = float("inf")
            for uid in scores:
                hk = _uid_to_hotkey.get(uid)
                block = (
                    first_mover_data[hk][1]
                    if hk and hk in first_mover_data
                    else float("inf")
                )
                if block < earliest_block:
                    earliest_block = block
                    champion_uid = uid
            if champion_uid is None:
                champion_uid = max(scores, key=scores.get)

        champion_score = scores[champion_uid]

        # Find the best challenger (highest score, ties broken by earliest block)
        def _sort_key(uid: int) -> Tuple[float, float]:
            hk = _uid_to_hotkey.get(uid)
            block = (
                first_mover_data[hk][1]
                if hk and hk in first_mover_data
                else float("inf")
            )
            return (-scores[uid], block)

        best_challenger_uid = min(scores, key=_sort_key)
        best_challenger_score = scores[best_challenger_uid]

        # Decide winner
        if best_challenger_uid == champion_uid:
            winner_uid = champion_uid
        elif best_challenger_score > champion_score + eps and \
                best_challenger_score > champion_score + delta:
            logger.info(
                f"First-mover overtake: Miner {best_challenger_uid} "
                f"(score={best_challenger_score:.3f}) beats champion "
                f"Miner {champion_uid} (score={champion_score:.3f}, "
                f"required>{champion_score + delta:.3f})"
            )
            winner_uid = best_challenger_uid
        else:
            logger.info(
                f"Champion Miner {champion_uid} (score={champion_score:.3f}) "
                f"retains: best challenger Miner {best_challenger_uid} "
                f"(score={best_challenger_score:.3f}) does not clear δ "
                f"(required>{champion_score + delta:.3f})"
            )
            winner_uid = champion_uid

        # Winner takes all
        weights = {uid: 0.0 for uid in scores.keys()}
        weights[winner_uid] = 1.0

        logger.info(
            f"Winner-take-all: Miner {winner_uid} wins with score "
            f"{scores[winner_uid]:.3f} (delta={delta}, ε={eps})"
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
        champion_hotkey: Optional[str] = None,
    ) -> Dict[int, float]:
        """Select winner by lowest cost, with qualification gate.

        Only qualified miners (all safety + correctness checks passed)
        compete. Among qualified miners, the one with lowest cost wins.

        The δ threshold only protects the **current champion** (the miner
        who won in the previous cycle).  A challenger must be at least
        cost_delta (10%) cheaper than the champion to dethrone them.
        If no previous champion is known, the earliest qualified miner
        (by on-chain block) is used as the initial champion.

        Args:
            costs: Dict of miner_uid -> cost_usd
            qualified: Dict of miner_uid -> bool (gate pass)
            first_mover_data: Dict of hotkey -> (best_cost, block_number)
            cost_delta: Multiplicative threshold (challenger < champion * (1 - delta))
            num_active_miners: Total active miners for bootstrap check
            uid_to_hotkey: Dict of miner_uid -> hotkey
            champion_hotkey: Hotkey of the previous cycle's winner.
                If None, the earliest qualified miner (by block) is used.

        Returns:
            Dict of miner_uid -> weight (sums to 1.0)
        """
        if not costs:
            return {}

        _uid_to_hotkey = uid_to_hotkey or {}
        _hotkey_to_uid = {hk: uid for uid, hk in _uid_to_hotkey.items()}

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

        # Steady-state: champion vs all challengers
        #
        # 1. Identify the current champion (previous winner, or earliest
        #    qualified miner by block if no previous winner).
        # 2. Find the lowest-cost qualified miner (best challenger).
        # 3. If best challenger IS the champion → champion retains.
        # 4. If best challenger is different → must beat
        #    champion_cost × (1 - δ) to dethrone.
        # 5. Ties (same cost) broken by earliest block number.

        # Resolve champion UID
        champion_uid = None
        if champion_hotkey and champion_hotkey in _hotkey_to_uid:
            cuid = _hotkey_to_uid[champion_hotkey]
            if cuid in qualified_costs:
                champion_uid = cuid

        # Fallback: earliest qualified miner by block number
        if champion_uid is None:
            earliest_block = float("inf")
            for uid in qualified_costs:
                hk = _uid_to_hotkey.get(uid)
                block = (
                    first_mover_data[hk][1]
                    if hk and hk in first_mover_data
                    else float("inf")
                )
                if block < earliest_block:
                    earliest_block = block
                    champion_uid = uid
            # If still None (no first_mover_data at all), pick lowest cost
            if champion_uid is None:
                champion_uid = min(qualified_costs, key=qualified_costs.get)

        champion_cost = qualified_costs[champion_uid]

        # Find the best challenger (lowest cost, ties broken by earliest block)
        def _sort_key(uid: int) -> Tuple[float, float]:
            hk = _uid_to_hotkey.get(uid)
            block = (
                first_mover_data[hk][1]
                if hk and hk in first_mover_data
                else float("inf")
            )
            return (qualified_costs[uid], block)

        best_challenger_uid = min(qualified_costs, key=_sort_key)
        best_challenger_cost = qualified_costs[best_challenger_uid]

        # Decide on-chain weight winner
        if best_challenger_uid == champion_uid:
            winner_uid = champion_uid
        elif best_challenger_cost < champion_cost * (1 - cost_delta):
            logger.info(
                f"[weight-computation] δ overtake: Miner {best_challenger_uid} "
                f"(${best_challenger_cost:.4f}) dethrones incumbent "
                f"Miner {champion_uid} (${champion_cost:.4f}, "
                f"required<${champion_cost * (1 - cost_delta):.4f})"
            )
            winner_uid = best_challenger_uid
        else:
            logger.info(
                f"[weight-computation] Incumbent Miner {champion_uid} "
                f"(${champion_cost:.4f}) retains: best challenger "
                f"Miner {best_challenger_uid} (${best_challenger_cost:.4f}) "
                f"does not clear δ (required<${champion_cost * (1 - cost_delta):.4f})"
            )
            winner_uid = champion_uid

        weights = {uid: 0.0 for uid in costs}
        weights[winner_uid] = 1.0

        logger.info(
            f"[weight-computation] On-chain weight winner: Miner {winner_uid} "
            f"(cost=${qualified_costs[winner_uid]:.4f}, delta={cost_delta})"
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


def compute_consensus_scores(
    validated_submissions: List[ValidatedSubmission],
    disqualify_stake_threshold: float = 0.5,
) -> Tuple[Dict[str, float], Dict[str, str]]:
    """Compute stake-weighted consensus scores and disqualification across validators.

    Scores: for each miner, the consensus score is the stake-weighted average
    of quality scores reported by validators that did NOT disqualify the miner:

        consensus_score[miner] = Σ(stake_i × score_i) / Σ(stake_i)
                                  (where i ∈ {validators that did not disqualify miner})

    Disqualification uses stake-weighted majority: a miner is consensus-
    disqualified only when the fraction of reporting stake that flagged it
    exceeds ``disqualify_stake_threshold`` (default 0.5).

    Args:
        validated_submissions: Submissions that passed the filter pipeline,
            each with an attached validator_stake.
        disqualify_stake_threshold: Fraction of reporting stake that must
            vote to disqualify a miner for it to be consensus-disqualified.

    Returns:
        Tuple of (consensus_scores, consensus_disqualified):
          - consensus_scores: Dict[miner_hotkey -> stake-weighted score (0.0–1.0)]
          - consensus_disqualified: Dict[miner_hotkey -> reason] for miners
            where disqualification stake exceeds the threshold
    """
    if not validated_submissions:
        return {}, {}

    miner_weighted_score: Dict[str, float] = {}
    miner_total_stake: Dict[str, float] = {}

    # Track disqualification votes by stake
    miner_disq_stake: Dict[str, float] = {}
    miner_reporting_stake: Dict[str, float] = {}
    miner_disq_reasons: Dict[str, str] = {}

    for sub in validated_submissions:
        stake = sub.validator_stake
        if stake <= 0:
            continue

        disqualified = sub.payload.disqualified

        all_miners = set(sub.payload.scores.keys()) | set(disqualified.keys())
        for miner_hk in all_miners:
            if miner_hk not in miner_reporting_stake:
                miner_reporting_stake[miner_hk] = 0.0
                miner_disq_stake[miner_hk] = 0.0
            miner_reporting_stake[miner_hk] += stake

            if miner_hk in disqualified:
                miner_disq_stake[miner_hk] += stake
                miner_disq_reasons[miner_hk] = disqualified[miner_hk]

        for miner_hk, score in sub.payload.scores.items():
            if miner_hk in disqualified:
                continue

            if miner_hk not in miner_total_stake:
                miner_total_stake[miner_hk] = 0.0
                miner_weighted_score[miner_hk] = 0.0

            miner_total_stake[miner_hk] += stake
            miner_weighted_score[miner_hk] += stake * score

    # Compute consensus disqualification via stake-weighted majority
    consensus_disqualified: Dict[str, str] = {}
    for miner_hk in miner_reporting_stake:
        reporting = miner_reporting_stake[miner_hk]
        if reporting > 0:
            disq_ratio = miner_disq_stake[miner_hk] / reporting
            if disq_ratio > disqualify_stake_threshold:
                consensus_disqualified[miner_hk] = miner_disq_reasons.get(
                    miner_hk, "consensus_disqualified"
                )

    consensus_scores: Dict[str, float] = {}
    for miner_hk in miner_total_stake:
        total_stake = miner_total_stake[miner_hk]
        if total_stake > 0:
            consensus_scores[miner_hk] = miner_weighted_score[miner_hk] / total_stake
        else:
            consensus_scores[miner_hk] = 0.0

    logger.info(
        "Consensus computed: %d miners (%d disqualified by stake majority), "
        "%d validators",
        len(consensus_scores), len(consensus_disqualified),
        len(validated_submissions),
    )

    return consensus_scores, consensus_disqualified
