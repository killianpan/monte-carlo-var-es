from stock_data import StockData
from gbm_simulator import GBMSimulator
from risk_engine import RiskEngine

def main():
    ticker = "AAPL"
    start = "2020-01-01"
    end = "2024-12-31"

    notional = 10_000
    alpha = 0.95

    # 1) DATA
    data = StockData(ticker, start, end)
    data.download()
    data.plot_price()

    log_returns = data.compute_log_returns()
    data.plot_log_returns()

    mu, sigma = data.estimate_mu_sigma()
    print(f"mu={mu:.6f} (annualisé ~ {mu*252:.4f})")
    print(f"sigma={sigma:.6f} (annualisé ~ {sigma*(252**0.5):.4f})")

    # 2) SIMULATION GBM (Monte Carlo)
    sim = GBMSimulator(ticker=ticker, S0=data.S0, mu=mu, sigma=sigma)
    sim.simulate_paths(dt=1, n_steps=252, n_paths=1000, seed=10)       # n_steps et n_paths changeable
    # sim.plot_paths()
    sim.plot_history_with_mc(hist_close=data.last_close_series(), max_paths_to_plot=15)

    # 3) RISK
    horizon_steps = 252     # horizon_steps modifiable
    ST = sim.terminal_prices(horizon_steps)

    res_mc = RiskEngine.var_es_mc(ST, S0=data.S0, notional=notional, alpha=alpha)
    print(f"VaR/ES Monte Carlo ({horizon_steps}j):", res_mc)

    res_hist = RiskEngine.var_es_historical_log_returns(log_returns, notional=notional, alpha=alpha, horizon_days=1)    # horizon_days changeable
    print("VaR/ES Historique (1j):", res_hist)

    print("Stress -20%:", RiskEngine.stress_price_shock(S0=data.S0, shock=-0.20, notional=notional))
    print("Stress 3-sigma (10j):", RiskEngine.stress_k_sigma(mu=mu, sigma=sigma, k=3, horizon_days=10, notional=notional))

    # Plots risques MC
    pnl = RiskEngine.pnl_from_terminal_prices(ST, S0=data.S0, notional=notional)
    losses = RiskEngine.losses_from_pnl(pnl)

    pnl_hist = RiskEngine.pnl_from_log_returns(log_returns, notional=notional, horizon_days=1)
    losses_hist = RiskEngine.losses_from_pnl(pnl_hist)

    RiskEngine.plot_pnl_distribution(pnl, title="Distribution des PnL (MC)")
    RiskEngine.plot_losses_with_var(losses, alpha=alpha, title="Distribution des pertes (MC) + VaR")

    RiskEngine.plot_pnl_distribution(pnl_hist, title="Distribution des PnL (Historique)")
    RiskEngine.plot_losses_with_var(losses_hist, alpha=alpha, title="Distribution des pertes (Historique) + VaR")

if __name__ == "__main__":
    main()

