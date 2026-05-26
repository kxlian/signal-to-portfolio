import numpy as np
import pandas as pd


def simulate_market_data(
    n_assets: int = 12,
    n_days: int = 504,
    seed: int = 42,
    signal_strength: float = 0.00015,
):
    """Simulate daily asset returns, alpha scores, and a covariance matrix.

    The data is synthetic by design. This keeps the project focused on the
    portfolio-construction step rather than on claiming a live market edge.

    The generated alpha scores are weakly linked to simulated expected
    returns so the example has a coherent research setting: higher-scoring
    assets should, on average, have slightly higher simulated returns, while
    risk and covariance still matter for allocation.
    """
    rng = np.random.default_rng(seed)
    asset_names = [f"Asset_{i + 1:02d}" for i in range(n_assets)]

    raw_scores = rng.normal(0, 1, size=n_assets)
    scores = (raw_scores - raw_scores.mean()) / raw_scores.std(ddof=0)
    scores_series = pd.Series(scores, index=asset_names, name="alpha_score")

    # Create a random positive semi-definite covariance matrix using a
    # three-factor structure plus idiosyncratic variance.
    factor_loadings = rng.normal(0, 0.25, size=(n_assets, 3))
    factor_cov = np.diag([0.00020, 0.00012, 0.00008])
    idiosyncratic_var = np.diag(rng.uniform(0.00005, 0.00018, size=n_assets))
    covariance_matrix = factor_loadings @ factor_cov @ factor_loadings.T + idiosyncratic_var

    expected_daily_returns = 0.00010 + signal_strength * scores + rng.normal(
        0, 0.00005, size=n_assets
    )

    returns = rng.multivariate_normal(
        mean=expected_daily_returns,
        cov=covariance_matrix,
        size=n_days,
    )

    returns_df = pd.DataFrame(returns, columns=asset_names)
    covariance_df = returns_df.cov()

    return returns_df, scores_series, covariance_df
