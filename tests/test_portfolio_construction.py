import numpy as np

from src.optimize import (
    naive_signal_weights,
    rank_based_weights,
    risk_aware_optimized_weights,
)
from src.simulate_data import simulate_market_data


def test_portfolio_weights_are_fully_invested():
    _, scores, covariance = simulate_market_data(seed=42)

    portfolios = [
        naive_signal_weights(scores),
        rank_based_weights(scores, top_n=5),
        risk_aware_optimized_weights(scores, covariance, max_weight=0.25),
    ]

    for weights in portfolios:
        assert np.isclose(weights.sum(), 1.0)
        assert (weights >= -1e-10).all()


def test_risk_aware_optimizer_respects_max_weight():
    _, scores, covariance = simulate_market_data(seed=42)
    max_weight = 0.25

    weights = risk_aware_optimized_weights(scores, covariance, max_weight=max_weight)

    assert weights.max() <= max_weight + 1e-8
