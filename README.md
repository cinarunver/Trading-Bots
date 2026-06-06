# QuantiFine

A quantitative algorithmic trading system for Binance Futures, built around a multi-layer signal architecture: market regime detection, technical entry triggers, and volatility-based position sizing.

Developed iteratively across six versions — from a basic signal scanner to a fully interactive live dashboard with a Ladder Stop mechanism.

---

## Strategy Architecture

The system is organized around three layers that must all agree before a trade is taken:

**1. Regime Filter**
Uses the **Hurst Exponent** to classify the market before any signal is evaluated.
- `H > 0.55` → Trending market → Trend-following logic activates
- `H < 0.45` → Mean-reverting market → Range/scalp logic activates
- `0.45 < H < 0.55` → No-trade zone (regime ambiguous)

**2. Signal Generation**
Entry triggers that only fire after the regime check passes:
- **Donchian Channel Ladder Stop** — 10 concurrent channels (periods 10–100). An entry is confirmed when price breaks 2+ channels simultaneously; the stop is placed at the first broken channel.
- **ADX** — Trend strength confirmation threshold
- **RSI / CCI / Z-Score** — Additional filters to avoid overextended entries

**3. Risk Engine**
Dynamic sizing instead of fixed-lot entries:
- **ATR-based stop distance** — Stop loss scaled to current volatility
- **Half-Kelly Criterion** — Position size = `(edge / odds) / 2`, capped by max portfolio risk per trade
- Default: 2% of account balance risked per trade

---

## Version History & Results

### v3.1 — Sniper Bot

Early multi-asset scanner. 15-bar Donchian breakout with ADX + RSI filters. Paper trading support. No Hurst filter yet — entries based purely on Donchian breakout + ADX > 25.

**Backtest Results** — $50 starting, 15m timeframe, 1,000 bars

| Asset | Final Balance | Return |
|-------|--------------|--------|
| BTC/USD | $51.82 | +3.6% |
| ETH/USD | $57.01 | +14.0% |
| SOL/USD | $46.02 | -8.0% |
| XRP/USD | $55.23 | +10.5% |

**Monte Carlo — BTC/USD** (2,000 scenarios × 1,000 bars, $50 starting)

| Percentile | Final Balance |
|-----------|--------------|
| P95 | $52 |
| P90 | $51 |
| P75 | $50 |
| P50 | $49 |
| P25 | $47 |
| P10 | $45 |
| P5 | $45 |

> 16,563 simulated trades across all scenarios — Win: **40.3%** / Loss: 59.7%

**Monte Carlo — SOL/USD** (v3.2, 2,000 scenarios × 1,000 bars, Profit Rate: 41.1%)

| Percentile | Final Balance |
|-----------|--------------|
| P95 | $68.5 |
| P90 | $62.9 |
| P75 | $54.6 |
| P50 | $47.2 |
| P25 | $43.1 |
| P10 | $38.6 |
| P5 | $37.1 |

> 31,095 simulated trades — Win: **26.8%** / Loss: 73.2% — wider range due to SOL volatility

<table>
<tr>
<td><img src="v3.1/BTC_backtest_result.png"/></td>
<td><img src="v3.1/ETH_backtest_result.png"/></td>
</tr>
<tr>
<td><img src="v3.1/SOL_backtest_result.png"/></td>
<td><img src="v3.1/XRP_backtest_result.png"/></td>
</tr>
</table>

<table>
<tr>
<td><img src="v3.1/BTC_monte_carlo_result.png"/></td>
<td><img src="monte_carlo_2000.png"/></td>
</tr>
</table>

---

### v4 — Backtest Engine

Added a full backtesting engine and per-asset analysis charts. Monte Carlo simulation introduced alongside the backtest to measure forward-looking probability distributions. Starting balance raised to $1,000 for more realistic sizing.

**Monte Carlo Results — ETH/USDT** ($1,000 starting, 16 trades, Win: **43.8%**)

| Percentile | Final Balance |
|-----------|--------------|
| P95 | $1,323 |
| P75 | $1,166 |
| P50 | $1,079 |
| P25 | $1,006 |
| P5 | $929 |

<table>
<tr>
<td><img src="v4/Backtest_BTC_USDT_20260202_1634.png"/></td>
<td><img src="v4/Backtest_ETH_USDT_20260202_1636.png"/></td>
<td><img src="v4/Backtest_SOL_USDT_20260202_1637.png"/></td>
</tr>
</table>

---

### v5 — Hurst + Donchian Dual-Regime Strategy

Introduced the Hurst Exponent as a regime gate (H > 0.55 for trend, H < 0.45 for mean-reversion). Added real trading support via Binance API. ATR-based stop distances replace fixed stops. Monte Carlo upgraded to project 2,000 future trades with compounding.

**Simulation Results — BTC/USDT** (v5 Logic)

| Metric | Value |
|--------|-------|
| Trades | 36 |
| Win Rate | 36.1% |
| Net PnL | +$107.36 |

**Monte Carlo — 2,000 Future Trade Projection**

| Metric | Value |
|--------|-------|
| Average Balance | $8,763 |
| Median Balance | $6,197 |
| Profit Chance | **100%** |
| Ruin Risk | **0%** |

> Projection compounds returns across 2,000 sequential trades drawn from the backtest return distribution.

<table>
<tr>
<td><img src="v5/Sim_v5_BTC_USDT.png"/></td>
<td><img src="v5/MonteCarlo_v5_Result.png"/></td>
</tr>
</table>

<table>
<tr>
<td><img src="v5/Sim_v5_AVAX_USDT.png"/></td>
<td><img src="v5/Sim_v5_DOGE_USDT.png"/></td>
<td><img src="v5/Sim_v5_SOL_USDT.png"/></td>
</tr>
</table>

---

### v6_oyun_alanı — Experimental Playground

Strategy prototyping sandbox. Refined the Ladder Stop mechanism design (10 Donchian channels with decrementing stop indices), added Z-Score for mean-reversion entries, and exported the final strategy logic as a TradingView Pine Script for visual validation.

<img src="v6_oyun_alanı/algo.png" width="400"/>

---

### v6.1 — Production (Ladder Stop Dashboard)

Production version. Live Dash dashboard refreshing every 15 seconds. 10-level Donchian Ladder Stop with dynamic trailing. Full indicator panel (Hurst, ATR, RSI, ADX, CCI). Half-Kelly position sizing.

**Backtest Results** — $1,000 starting, 1h timeframe, ~5,000 bars (Jan–Feb 2026)

| Asset | Net PnL | Trades | Win Rate | Profit Factor | Avg Win |
|-------|---------|--------|----------|---------------|---------|
| BTC/USDT | **+$53.61** | 5 | 80.0% | 5.25 | $17.18 |
| ETH/USDT | -$26.08 | 7 | 28.6% | 0.46 | $10.06 |
| SOL/USDT | **+$25.66** | 11 | 54.5% | 1.47 | $15.62 |
| XRP/USDT | **+$30.39** | 6 | 50.0% | 1.75 | $25.35 |
| DOGE/USDT | -$74.78 | 11 | 45.5% | 0.24 | $4.52 |

> Period tested coincides with a broad crypto bear leg (BTC $95K → $63K). BTC performed best due to cleaner trend structure (fewer false breakouts, higher Hurst readings).

<table>
<tr>
<td><img src="Backtest_v6.1_Ladder_BTC_USDT.png"/></td>
<td><img src="Backtest_v6.1_Ladder_ETH_USDT.png"/></td>
</tr>
<tr>
<td><img src="Backtest_v6.1_Ladder_SOL_USDT.png"/></td>
<td><img src="Backtest_v6.1_Ladder_XRP_USDT.png"/></td>
</tr>
</table>

<img src="Backtest_v6.1_Ladder_DOGE_USDT.png" width="49%"/>

---

## Getting Started

### Requirements

```bash
pip install ccxt pandas pandas-ta numpy dash plotly matplotlib colorama rich
```

### Running the Live Dashboard (v6.1)

```bash
cd v6.1
python v6.1.py
```

Then open `http://127.0.0.1:8050` in your browser. The dashboard refreshes every 15 seconds and displays the current Ladder Stop level, Hurst value, and all indicators.

> No API key required for paper trading / dashboard mode — data is fetched from Binance public endpoints.

### Running the Live Trading Bot (v5)

```bash
cd v5
python v5_real.py
```

You will be prompted to enter your Binance API key and secret at startup, or you can set them directly in the script:

```python
API_KEY = ""      # paste your key here
API_SECRET = ""   # paste your secret here
```

> **Warning:** Always test with paper trading mode before connecting real funds. Set `PAPER_TRADING: True` in CONFIG.

### Running the Backtest (v6.1)

```bash
cd v6.1
python backtest_v6.1.py
```

### Running the Monte Carlo Simulation (v5)

```bash
cd v5
python montecarlo_v5.py
```

---

## Configuration

All strategy parameters are in the `CONFIG` dict at the top of each script. Key parameters:

```python
CONFIG = {
    "SYMBOL_LIST":       ["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT", "DOGE/USDT"],
    "TIMEFRAME":         "1h",
    "HURST_TREND":       0.52,      # Minimum Hurst for trend mode
    "ATR_MULTIPLIER":    2.0,       # Stop loss distance (× ATR)
    "RISK_PER_TRADE":    0.02,      # 2% of account per trade
    "PERIODS":           [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],  # Donchian periods
}
```

---

## Project Structure

```
QuantiFine/
├── v3.1/                   # Sniper Bot (early version)
│   ├── Sniper_Bot.py
│   ├── MonteCarlo.py
│   └── *.png               # Backtest / Monte Carlo charts
├── v4/                     # Backtest + analysis charts
│   ├── v4.py
│   ├── backtest_v4.py
│   └── *.png
├── v5/                     # Full strategy — live + paper trading
│   ├── v5.py               # Paper trading scanner
│   ├── v5_real.py          # Live trading bot
│   ├── backtest_v5.py
│   ├── montecarlo_v5.py    # 10,000-run MC simulation
│   └── *.png
├── v6_oyun_alanı/          # Experimental / prototyping
│   ├── v6.py
│   ├── backtest_v6.py
│   ├── montecarlo_v6.py
│   └── quantifine_v6_strategy.pine   # TradingView Pine Script
└── v6.1/                   # Production — live Dash dashboard
    ├── v6.1.py             # Interactive dashboard
    ├── backtest_v6.1.py    # Full backtest engine
    └── *.png
```

---

## Indicators Used

| Indicator | Role |
|-----------|------|
| Hurst Exponent | Market regime classifier |
| Donchian Channels (×10) | Ladder Stop entry/exit |
| ADX | Trend strength filter |
| ATR | Stop distance + position sizing |
| RSI | Overbought/oversold filter |
| CCI | Momentum confirmation |
| Z-Score | Mean-reversion signal |
| Half-Kelly | Position size calculation |

---

## Disclaimer

This project is for educational and research purposes only. Algorithmic trading involves significant financial risk. Past backtest performance does not guarantee future results. Use at your own risk.
