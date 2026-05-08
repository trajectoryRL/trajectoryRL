"""Scoring placeholder.

In v6.0 of the Incentive Mechanism, score aggregation and winner
selection are performed server-side (see
``docs/INCENTIVE_MECHANISM.md`` §Server-Coordinated Stake-Weighted
Aggregation). The validator runs only the per-challenger sandbox eval
and posts a signed score; it does not maintain any local consensus or
aggregation logic.

The v5.x ``TrajectoryScorer`` / ``compute_consensus_scores`` /
``AggregatedScore`` symbols have been removed along with the
consensus_filter, consensus_store, and consensus modules.
"""
