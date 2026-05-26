import numpy as np
import pandas as pd
from scipy.optimize import minimize


def naive_signal_weights(scores: pd.Series) -> pd.Series:
    """Convert positive alpha scores into proportional long-only weights."""
    positive_scores = scores.clip(lower=0)

    if positive_scores.sum() == 0:
        return pd.Series(1 / len(scores), index=scores.index, name="naive_signal")

    weights = positive_scores / positive_scores.sum()
    weights.name = "naive_signal"
    return weights


def rank_based_weights(scores: pd.Series, top_n: int = 5) -> pd.Series:
    """Equally weight the top-ranked assets by alpha score."""
    if top_n <= 0:
        raise ValueError("top_n must be positive")
    if top_n > len(scores):
        raise ValueError("top_n cannot exceed the number of assets")

    weights = pd.Series(0.0, index=scores.index, name="rank_based")
    selected = scores.sort_values(ascending=False).head(top_n).index
    weights.loc[selected] = 1 / top_n
    return weights


def risk_aware_optimized_weights(
    scores: pd.Series,
    covariance: pd.DataFrame,
    risk_aversion: float = 20.0,
    max_weight: float = 0.25,
    periods_per_year: int = 252,
) -> pd.Series:
    """Solve a long-only signal-risk portfolio optimization problem.

    Objective:
        maximize s'w - lambda * w'(annualized Σ)w

    Implemented as minimization:
        minimize -(s'w - lambda * w'(annualized Σ)w)

    Constraints:
        sum(w) = 1
        0 <= w_i <= max_weight

    Notes:
        The score vector is dimensionless, so this is best interpreted as a
        research allocation score rather than a direct expected-return model.
        The risk penalty makes the optimizer trade off signal exposure
        against estimated covariance risk and concentration.
    """
    if risk_aversion < 0:
        raise ValueError("risk_aversion must be non-negative")
    if not 0 < max_weight <= 1:
        raise ValueError("max_weight must be between 0 and 1")

    assets = list(scores.index)
    n_assets = len(assets)

    if max_weight * n_assets < 1:
        raise ValueError("max_weight is too low to create a fully invested portfolio")

    s = scores.loc[assets].to_numpy(dtype=float)
    sigma = covariance.loc[assets, assets].to_numpy(dtype=float) * periods_per_year
    n = len(assets)

    def objective(w):
        signal_exposure = s @ w
        portfolio_variance = w @ sigma @ w
        return -(signal_exposure - risk_aversion * portfolio_variance)

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bounds = [(0.0, max_weight) for _ in range(n)]
    x0 = np.full(n, 1 / n)

    result = minimize(
        objective,
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 1000, "ftol": 1e-12},
    )

    if not result.success:
        raise RuntimeError(f"Optimization failed: {result.message}")

    weights = pd.Series(result.x, index=assets, name="risk_aware")
    weights[weights.abs() < 1e-10] = 0.0
    return weights


def portfolio_construction_metrics(
    weights: pd.Series,
    scores: pd.Series,
    covariance: pd.DataFrame,
    periods_per_year: int = 252,
) -> dict:
    """Calculate ex-ante portfolio construction diagnostics."""
    aligned_assets = weights.index
    w = weights.to_numpy(dtype=float)
    s = scores.loc[aligned_assets].to_numpy(dtype=float)
    sigma = covariance.loc[aligned_assets, aligned_assets].to_numpy(dtype=float)

    signal_exposure = float(s @ w)
    annualized_variance = float(w @ (sigma * periods_per_year) @ w)
    annualized_volatility = float(np.sqrt(max(annualized_variance, 0)))
    concentration = float(np.sum(w**2))
    effective_positions = float(1 / concentration) if concentration > 0 else np.nan

    return {
        "signal_exposure": signal_exposure,
        "ex_ante_ann_volatility": annualized_volatility,
        "concentration_hhi": concentration,
        "effective_positions": effective_positions,
        "max_weight": float(weights.max()),
    }


# Backwards-compatible alias for the first project version.
portfolio_metrics = portfolio_construction_metrics
