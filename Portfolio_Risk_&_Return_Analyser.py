import argparse
import sys
import os
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

TRADING_DAYS = 252

def fetch_price_data(tickers: List[str], period: str = "3y", interval: str = "1d") -> pd.DataFrame:
    """
    Try fetching prices from yfinance. If it fails (no internet/API issues),
    fall back to data/sample_prices.csv bundled with the project.
    """
    try:
        import yfinance as yf
        data = yf.download(tickers=tickers, period=period, interval=interval, auto_adjust=True, progress=False)["Close"]
        if isinstance(data, pd.Series):  # single ticker edge-case
            data = data.to_frame(name=tickers[0])
        # Drop any columns that are entirely NaN (unavailable tickers)
        data = data.dropna(how="all", axis=1)
        if data.empty:
            raise RuntimeError("Downloaded data is empty.")
        return data
    except Exception as e:
        print(f"[WARN] Live download failed: {e}")
        sample_path = os.path.join(os.path.dirname(__file__), "data", "sample_prices.csv")
        print(f"[INFO] Falling back to sample data at: {sample_path}")
        df = pd.read_csv(sample_path, parse_dates=["Date"], index_col="Date")
        # Filter to requested tickers if present; otherwise keep what's in sample
        keep = [t for t in tickers if t in df.columns]
        return df[keep] if keep else df

def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.pct_change().dropna(how="all")

def annualized_metrics(returns: pd.DataFrame) -> Tuple[pd.Series, pd.DataFrame]:
    mean_daily = returns.mean()
    cov_daily = returns.cov()
    mean_annual = mean_daily * TRADING_DAYS
    cov_annual = cov_daily * TRADING_DAYS
    return mean_annual, cov_annual

def max_drawdown(price_series: pd.Series) -> float:
    running_max = price_series.cummax()
    drawdown = price_series / running_max - 1.0
    return drawdown.min()

def portfolio_performance(weights: np.ndarray, mean_annual: pd.Series, cov_annual: pd.DataFrame, risk_free: float = 0.0) -> Tuple[float, float, float]:
    ret = float(np.dot(weights, mean_annual.values))
    vol = float(np.sqrt(np.dot(weights.T, np.dot(cov_annual.values, weights))))
    sharpe = (ret - risk_free) / vol if vol > 0 else np.nan
    return ret, vol, sharpe

def simulate_portfolios(n: int, mean_annual: pd.Series, cov_annual: pd.DataFrame, risk_free: float = 0.0) -> pd.DataFrame:
    tickers = mean_annual.index.tolist()
    results = []
    for _ in range(n):
        w = np.random.random(len(tickers))
        w /= w.sum()
        ret, vol, sharpe = portfolio_performance(w, mean_annual, cov_annual, risk_free)
        row = {"return": ret, "volatility": vol, "sharpe": sharpe}
        row.update({f"w_{t}": w[i] for i, t in enumerate(tickers)})
        results.append(row)
    return pd.DataFrame(results)

def plot_risk_return(df_portfolios: pd.DataFrame, outpath: str):
    plt.figure()
    plt.scatter(df_portfolios["volatility"], df_portfolios["return"], s=10, alpha=0.6)
    # Highlight max Sharpe and min volatility portfolios
    best_sharpe = df_portfolios.loc[df_portfolios["sharpe"].idxmax()]
    min_vol = df_portfolios.loc[df_portfolios["volatility"].idxmin()]
    plt.scatter([best_sharpe["volatility"]], [best_sharpe["return"]], marker="*", s=200)
    plt.scatter([min_vol["volatility"]], [min_vol["return"]], marker="X", s=120)
    plt.xlabel("Volatility (Ïƒ, annualized)")
    plt.ylabel("Return (annualized)")
    plt.title("Riskâ€“Return Scatter & Efficient Frontier (simulated)")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()

def plot_correlation_heatmap(returns: pd.DataFrame, outpath: str):
    corr = returns.corr()
    fig, ax = plt.subplots()
    cax = ax.imshow(corr.values, interpolation="nearest")
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticklabels(corr.columns)
    fig.colorbar(cax)
    ax.set_title("Correlation Heatmap (daily returns)")
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)

def summarize_portfolio(prices: pd.DataFrame, returns: pd.DataFrame, mean_annual: pd.Series, cov_annual: pd.DataFrame, risk_free: float) -> pd.DataFrame:
    # Equal weights baseline
    n = prices.shape[1]
    w = np.array([1.0 / n] * n)
    ret, vol, sharpe = portfolio_performance(w, mean_annual, cov_annual, risk_free)
    # Max drawdown per asset
    mdds = {t: max_drawdown(prices[t].dropna()) for t in prices.columns}
    summary = pd.DataFrame({
        "Annualized Return": mean_annual,
        "Annualized Volatility": np.sqrt(np.diag(cov_annual.values)),
        "Max Drawdown": pd.Series(mdds)
    })
    summary.loc["Equal-Weight Portfolio"] = [ret, vol, np.nan]  # MDD not directly comparable
    return summary

def parse_args():
    p = argparse.ArgumentParser(description="MSCI-Style Portfolio Risk & Return Analyzer")
    p.add_argument("--tickers", type=str, default="TCS.NS,INFY.NS,RELIANCE.NS,HDFCBANK.NS,ICICIBANK.NS", help="Comma-separated list of tickers")
    p.add_argument("--period", type=str, default="3y", help="yfinance period, e.g., 1y, 2y, 5y, max")
    p.add_argument("--interval", type=str, default="1d", help="yfinance interval, e.g., 1d, 1wk")
    p.add_argument("--risk_free", type=float, default=0.06, help="Annual risk-free rate (e.g., 0.06 for 6%)")
    p.add_argument("--sims", type=int, default=5000, help="Number of random portfolios to simulate")
    p.add_argument("--outdir", type=str, default="outputs", help="Directory to save charts")
    return p.parse_args()

def main():
    args = parse_args()
    tickers = [t.strip() for t in args.tickers.split(",") if t.strip()]
    prices = fetch_price_data(tickers, args.period, args.interval).dropna(how="all")
    returns = compute_returns(prices)
    mean_annual, cov_annual = annualized_metrics(returns)

    os.makedirs(args.outdir, exist_ok=True)
    # Simulate
    dfp = simulate_portfolios(args.sims, mean_annual, cov_annual, args.risk_free)
    # Plots
    plot_risk_return(dfp, os.path.join(args.outdir, "risk_return.png"))
    plot_correlation_heatmap(returns, os.path.join(args.outdir, "correlation_heatmap.png"))
    # Summary table
    summary = summarize_portfolio(prices, returns, mean_annual, cov_annual, args.risk_free)
    summary_path = os.path.join(args.outdir, "summary.csv")
    summary.to_csv(summary_path)

    print("=== Portfolio Risk & Return Analyzer ===")
    print(f"Tickers: {', '.join(prices.columns)}")
    print(f"Data points: {len(prices)}")
    print(f"Outputs saved to: {args.outdir}")
    print(f"- Risk-Return Scatter: {os.path.join(args.outdir, 'risk_return.png')}")
    print(f"- Risk-Return Scatter: " + os.path.join(args.outdir, "risk_return.png"))
    print(f"- Correlation Heatmap: " + os.path.join(args.outdir, "correlation_heatmap.png"))
    print(f"- Summary CSV: " + os.path.join(args.outdir, "summary.csv"))
    print(f"- Correlation Heatmap: {os.path.join(args.outdir, 'correlation_heatmap.png')}")
    print(f"- Summary CSV: {summary_path}")
    # Best Sharpe summary
    best_row = dfp.loc[dfp['sharpe'].idxmax()]
    print(f"Best Sharpe (simulated): Sharpe={best_row['sharpe']:.2f}, Ret={best_row['return']:.2%}, Vol={best_row['volatility']:.2%}")

if __name__ == "__main__":
    main()

