# Portfolio-Risk-and-Return-Analyzer
finance-domain project that computes "annualized return", "volatility", "Sharpe ratio", "max drawdown", and simulates an "efficient frontier" for a basket of stocks. Designed to align with focus on "indexes, factors, and risk analytics".

---

## ✨ Features
- Pulls historical prices with `yfinance` (auto-adjusted close).
- Falls back to a bundled offline dataset (`data/sample_prices.csv`) so it runs even without internet.
- Computes: annualized return & volatility, Sharpe ratio (adjusted for a user-supplied risk-free rate), and max drawdown.
- Simulates thousands of random portfolios to approximate the **efficient frontier**.
- Generates two publication-ready charts:
  - Risk–Return scatter with best Sharpe & minimum-volatility portfolios highlighted.
  - Correlation heatmap of daily returns.
- Saves a "summary.csv" you can open in Excel.
- Clean CLI, simple code structure, and well-commented functions.

---

## 🧰 Tech Stack
"Python, Pandas, NumPy, Matplotlib, yfinance"

---

