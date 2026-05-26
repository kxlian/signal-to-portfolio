import numpy as np
import pandas as pd


def portfolio_return_series(returns: pd.DataFrame, weights: pd.Series) -> pd.Series:
    """Convert asset returns and static weights into portfolio returns."""
    aligned_assets = list(weights.index)
    return_series = returns.loc[:, aligned_assets] @ weights.loc[aligned_assets]
    return_series.name = weights.name or "portfolio"
    return return_series


def cumulative_returns(return_series: pd.Series) -> pd.Series:
    """Calculate cumulative returns from a periodic return series."""
    cumulative = (1 + return_series).cumprod() - 1
    cumulative.name = return_series.name
    return cumulative


def max_drawdown(return_series: pd.Series) -> float:
    """Calculate maximum drawdown from a periodic return series."""
    wealth = (1 + return_series).cumprod()
    running_peak = wealth.cummax()
    drawdown = wealth / running_peak - 1
    return float(drawdown.min())


def realized_performance_metrics(
    returns: pd.DataFrame,
    weights: pd.Series,
    periods_per_year: int = 252,
) -> dict:
    """Calculate realized performance diagnostics on simulated data."""
    portfolio_returns = portfolio_return_series(returns, weights)
    n_periods = len(portfolio_returns)

    total_return = float((1 + portfolio_returns).prod() - 1)
    ann_return = float((1 + total_return) ** (periods_per_year / n_periods) - 1)
    ann_volatility = float(portfolio_returns.std(ddof=1) * np.sqrt(periods_per_year))
    sharpe_ratio = float(ann_return / ann_volatility) if ann_volatility > 0 else np.nan

    return {
        "realized_ann_return": ann_return,
        "realized_ann_volatility": ann_volatility,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": max_drawdown(portfolio_returns),
        "total_return": total_return,
    }
