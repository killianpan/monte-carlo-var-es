# Monte Carlo VaR/ES & Stress Tests
 
Python project simulating future stock price paths with a Geometric
Brownian Motion (GBM) Monte Carlo model, and using them to compute
Value-at-Risk (VaR), Expected Shortfall (ES) and deterministic/statistical
stress scenarios — benchmarked against a historical-returns approach.
 
Historical prices are downloaded live via `yfinance` (Yahoo Finance), so
no dataset is bundled with this repository.
 
## Table of contents
 
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Project structure](#project-structure)
- [Methodology](#methodology)
- [Authors](#authors)
## Requirements
 
- Python 3.10+
- Internet connection (data is downloaded live via `yfinance`)
## Installation
 
```bash
git clone https://github.com/killianpan/monte-carlo-var-es.git
cd monte-carlo-var-es
pip install -r requirements.txt
```
 
## Usage
 
### Object-oriented version (`main.py`)
 
```bash
python3 main.py
```
 
Downloads historical data for a given ticker (`AAPL` by default, edit the
`ticker`/`start`/`end` variables in `main.py`), estimates the drift (mu)
and volatility (sigma) from historical log-returns, simulates 1,000 GBM
paths over one year, and prints/plots:
- The GBM Monte Carlo paths alongside the recent price history
- VaR and ES at a given confidence level (95% by default), both via
  Monte Carlo and via the historical-returns method
- A deterministic stress test (a fixed price shock) and a statistical
  k-sigma stress test
- PnL and loss distributions for both approaches, with the VaR level marked
### Notebook version (`Monte_Carlo.ipynb`)
 
Open the notebook and run the cells sequentially — it walks step by step
through the same building blocks (data download, log-returns, GBM path
simulation, historical + Monte Carlo overlay, and a short discussion on
using random vs. historically-estimated mu/sigma) in a more exploratory,
function-based format.
 
## Project structure
 
| File | Role |
|---|---|
| `stock_data.py` | `StockData` class: downloads historical prices via `yfinance`, computes log-returns, estimates mu/sigma, exposes the last price (`S0`) and basic price/return plots. |
| `gbm_simulator.py` | `GBMSimulator` class: simulates GBM price paths from a given `S0`, `mu`, `sigma`, extracts terminal prices at a chosen horizon, and plots the simulated paths (alone or overlaid on recent history). |
| `risk_engine.py` | `RiskEngine` class: computes VaR/ES from Monte Carlo terminal prices or from historical log-returns, runs deterministic and k-sigma stress tests, and plots PnL/loss distributions with the VaR level. |
| `main.py` | Ties the three classes together into a full run: data → simulation → risk metrics → plots. |
| `Monte_Carlo.ipynb` | Earlier, function-based notebook covering the same steps in a more exploratory format, following the course's exercise structure. |
 
## Methodology
 
- **Log-returns**: computed as `ln(1 + pct_change)` from daily close prices.
- **Drift and volatility**: `mu` and `sigma` are the historical mean and
  standard deviation of daily log-returns (rather than arbitrary values),
  as recommended by the course to obtain economically meaningful simulations.
- **GBM simulation**: `S_{t+1} = S_t · exp((mu - 0.5σ²)·dt + σ·√dt·Z)`,
  with `Z ~ N(0,1)`, generating full price paths (not just terminal
  values) with the same daily frequency as the historical data.
- **VaR / Expected Shortfall**: computed two ways —
  - *Monte Carlo*: from the P&L implied by simulated terminal prices at a
    chosen horizon.
  - *Historical*: from the P&L implied by realized historical log-returns
    (with a rolling sum for multi-day horizons).
  
  VaR is the loss at the chosen quantile (95% by default); ES is the
  average loss beyond that quantile.
- **Stress tests**:
  - *Deterministic*: an immediate price shock of a given magnitude (e.g. -20%).
  - *Statistical (k-sigma)*: a loss scenario built from `mu_h - k·σ_h`
    over a given horizon, capturing a k-standard-deviation adverse move.
## Authors
 
Killian Pan — Python project (Financial Markets course)
