# AI Hedge Fund — Nifty 50

An AI-powered quantitative trading system for Nifty 50 stocks, built from scratch. Combines mathematical signal agents, an LLM orchestrator, and a risk management engine to generate trade decisions with full reasoning.

> Built as a 2nd semester BTech CS student. Companion to the SSRN research paper: *"Why Machine Learning Trading Strategies Fail: An Empirical Analysis of Nifty 50"*

---

## Backtest Results (2020–2024)

| Metric | Value |
|---|---|
| Total Return | +123.89% |
| Benchmark (Nifty 50) | ~55% |
| Sharpe Ratio | tracked per run |
| Max Drawdown | tracked per run |

---

## System Architecture

```
Data Pipeline → 5 Signal Agents → LLM Orchestrator → Risk Engine → Decision
                      ↑
              HMM Regime Detection
              (adjusts agent weights dynamically)
```

### Signal Agents

| Agent | Method | What it detects |
|---|---|---|
| Kalman Momentum | Kalman Filter + ADX | Price trend direction and strength |
| Advanced Mean Reversion | ADF test + Hurst Exponent + Cointegration | Overbought/oversold conditions |
| ML Classifier | Random Forest | Next-day price direction probability |
| GBM Monte Carlo | Geometric Brownian Motion (10,000 sims) | Probability distribution of future prices |
| HMM Regime | Hidden Markov Model | Market state (bull / high-vol / bear) |

### LLM Orchestrator
- Uses **Llama 3.3 70B** via Groq API (free tier)
- Receives all agent scores + market context
- Outputs: trade recommendation, thesis, confidence level

### Risk Engine
- Kelly Criterion position sizing
- Regime-aware limits (tighter in bear market)
- Hurst-adjusted stop losses
- Monte Carlo CVaR check
- Portfolio drawdown circuit breaker

---

## Live Dashboard

Streamlit dashboard with two modes:

**Run Full Analysis** — runs all 5 agents, LLM synthesis, risk decision  
**Run Forecasting** — 30-day GBM price forecast with confidence bands, bull/bear/base scenarios, support/resistance levels

---

## Project Structure

```
ai-hedge-fund-nifty50/
├── pipeline/
│   ├── fetch_data.py          # Download Nifty 50 OHLCV from Yahoo Finance
│   ├── clean_data.py          # Clean and validate raw data
│   └── store_data.py          # Store in SQLite database
├── agents/
│   ├── regime_agent.py        # HMM market regime detection
│   ├── kalman_momentum_agent.py       # Kalman Filter + ADX momentum
│   ├── advanced_mean_reversion_agent.py # ADF + Hurst + Cointegration
│   ├── ml_agent.py            # Random Forest classifier
│   └── gbm_monte_carlo_agent.py      # GBM price simulation
├── risk/
│   └── risk_engine.py         # Kelly + VaR + drawdown risk management
├── orchestrator_v2.py         # Main decision engine
├── backtester_v2.py           # Historical backtesting
├── forecaster.py              # 30-day GBM price forecasting
└── dashboard.py               # Streamlit live dashboard
```

---

## Setup

```bash
# Clone the repo
git clone https://github.com/yakshmaan/ai-hedge-fund-nifty50.git
cd ai-hedge-fund-nifty50

# Install dependencies
pip install yfinance pandas numpy scikit-learn streamlit plotly requests

# Set your Groq API key (free at console.groq.com)
export GROQ_API_KEY=your_key_here

# Run the data pipeline (one time)
python pipeline/fetch_data.py
python pipeline/clean_data.py
python pipeline/store_data.py

# Run the orchestrator on a stock
python orchestrator_v2.py RELIANCE
python orchestrator_v2.py RELIANCE TCS HDFCBANK

# Run backtest
python backtester_v2.py
python backtester_v2.py --symbol RELIANCE

# Launch dashboard
streamlit run dashboard.py
```

---

## Mathematical Concepts Used

- **Geometric Brownian Motion**: `dS = μS dt + σS dW` — models stochastic price evolution
- **Kalman Filter**: recursive Bayesian estimator tracking price level and velocity
- **Hidden Markov Model**: detects latent market regimes from observable returns
- **Kelly Criterion**: `f* = (pb - q) / b` — optimal position sizing
- **Augmented Dickey-Fuller Test**: tests stationarity of price series
- **Hurst Exponent**: measures mean-reversion vs trending behavior (H < 0.5 = mean-reverting)
- **Engle-Granger Cointegration**: pairs trading signal from stationary spread
- **Value at Risk / CVaR**: tail risk measurement at 95% confidence

---

## Research Paper

**"Why Machine Learning Trading Strategies Fail: An Empirical Analysis of Nifty 50"**  
Published on SSRN — backtests Moving Average Crossover and Random Forest strategies against buy-and-hold on Nifty 50 data, demonstrating systematic underperformance and the conditions under which ML strategies fail.

---

## Disclaimer

This system is for educational and research purposes only. It does not constitute financial advice. Paper trading only — do not use with real capital without extensive further validation.

---

*Built by Yaksh — BTech CS, Semester 2*
