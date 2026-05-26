from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def _prepare_output_path(output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path


def plot_weights(weight_table: pd.DataFrame, output_path: str | Path) -> None:
    """Plot portfolio weights for each construction approach."""
    output_path = _prepare_output_path(output_path)
    ax = weight_table.plot(kind="bar", figsize=(12, 6))
    ax.set_title("Portfolio Weights by Construction Method")
    ax.set_ylabel("Portfolio weight")
    ax.set_xlabel("Asset")
    ax.legend(title="Method")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_cumulative_returns(cumulative_return_table: pd.DataFrame, output_path: str | Path) -> None:
    """Plot cumulative simulated returns for each portfolio."""
    output_path = _prepare_output_path(output_path)
    ax = cumulative_return_table.plot(figsize=(12, 6))
    ax.set_title("Cumulative Simulated Portfolio Returns")
    ax.set_ylabel("Cumulative return")
    ax.set_xlabel("Trading day")
    ax.legend(title="Method")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_signal_risk_tradeoff(metrics_table: pd.DataFrame, output_path: str | Path) -> None:
    """Plot signal exposure against ex-ante annualized volatility."""
    output_path = _prepare_output_path(output_path)
    ax = metrics_table.plot(
        kind="scatter",
        x="ex_ante_ann_volatility",
        y="signal_exposure",
        figsize=(8, 6),
    )

    for method, row in metrics_table.iterrows():
        ax.annotate(
            method,
            (row["ex_ante_ann_volatility"], row["signal_exposure"]),
            xytext=(5, 5),
            textcoords="offset points",
        )

    ax.set_title("Signal Exposure vs. Ex-Ante Risk")
    ax.set_xlabel("Ex-ante annualized volatility")
    ax.set_ylabel("Signal exposure")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


# Backwards-compatible alias for the first project version.
plot_metrics = plot_signal_risk_tradeoff
