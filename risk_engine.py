import numpy as np
import matplotlib.pyplot as plt


class RiskEngine:
    """
    Calcule VaR / Expected Shortfall + stress tests
    Entrées typiques:
    - log_returns (historique)
    - S0 + terminal prices ST (Monte Carlo)
    """

    # ---------- Core transforms ----------
    @staticmethod
    def pnl_from_terminal_prices(ST: np.ndarray, S0: float, notional: float = 1.0) -> np.ndarray:
        ST = np.asarray(ST, dtype=float)
        S0 = float(S0)
        notional = float(notional)
        return notional * (ST / S0 - 1.0)

    @staticmethod
    def losses_from_pnl(pnl: np.ndarray) -> np.ndarray:
        pnl = np.asarray(pnl, dtype=float)
        return -pnl  # pertes positives

    # ---------- VaR / ES ----------
    @staticmethod
    def var_es_from_losses(losses: np.ndarray, alpha: float = 0.95) -> dict:
        losses = np.asarray(losses, dtype=float)
        var = float(np.quantile(losses, alpha))
        tail = losses[losses >= var]
        es = float(tail.mean()) if tail.size else var
        return {"alpha": alpha, "VaR": var, "ES": es}

    # Monte Carlo
    @staticmethod
    def var_es_mc(ST: np.ndarray, S0: float, notional: float = 1.0, alpha: float = 0.95) -> dict:
        pnl = RiskEngine.pnl_from_terminal_prices(ST, S0=S0, notional=notional)
        losses = RiskEngine.losses_from_pnl(pnl)
        out = RiskEngine.var_es_from_losses(losses, alpha=alpha)
        out["method"] = "MC"
        return out

    # Historique
    @staticmethod
    def var_es_historical_log_returns(
        log_returns,
        notional: float = 1.0,
        alpha: float = 0.95,
        horizon_days: int = 1,
    ) -> dict:
        """
        Convertit log-returns -> rendement simple -> pnl -> losses -> VaR/ES
        Pour horizon_days > 1 : somme glissante des log-returns.
        """

        if horizon_days > 1:
            lr = log_returns.rolling(horizon_days).sum().dropna().to_numpy()
        else:
            lr = log_returns.to_numpy()

        r = np.exp(lr) - 1.0
        pnl = float(notional) * r
        losses = -pnl

        out = RiskEngine.var_es_from_losses(losses, alpha=alpha)
        out["method"] = "Historical"
        return out

    # ---------- Stress tests ----------
    # Déterministe
    @staticmethod
    def stress_price_shock(S0: float, shock: float = -0.2, notional: float = 1.0) -> dict:
        """
        shock=-0.2 => baisse immédiate de 20%
        """
        S0 = float(S0)
        ST = S0 * (1.0 + float(shock))
        pnl = float(notional) * (ST / S0 - 1.0)     # Cacul PnL d'un scénario fixe
        return {"shock": shock, "PnL": float(pnl), "Loss": float(-pnl)}

    # Statistique
    @staticmethod
    def stress_k_sigma(mu: float, sigma: float, k: float = 3.0, horizon_days: int = 10, notional: float = 1.0) -> dict:
        """
        Scénario défavorable: LR = mu_h - k*sigma_h
        """
        mu_h = float(mu) * horizon_days
        sig_h = float(sigma) * np.sqrt(horizon_days)
        lr = mu_h - float(k) * sig_h
        loss = float(notional) * (1.0 - np.exp(lr))
        return {"k": k, "horizon_days": horizon_days, "Loss": float(loss)}

    # ---------- Risk plots ----------
    @staticmethod
    def plot_pnl_distribution(pnl: np.ndarray, bins: int = 60, title: str = "Distribution des PnL") -> None:
        pnl = np.asarray(pnl, dtype=float)
        plt.figure(figsize=(11, 5))
        plt.hist(pnl, bins=bins, alpha=0.85)
        plt.title(title)
        plt.xlabel("PnL")
        plt.ylabel("Fréquence")
        plt.grid(True, alpha=0.3)
        plt.show()

    @staticmethod
    def plot_losses_with_var(losses: np.ndarray, alpha: float = 0.95, bins: int = 60, title: str = "Pertes + VaR") -> None:
        losses = np.asarray(losses, dtype=float)
        var_ = float(np.quantile(losses, alpha))

        plt.figure(figsize=(11, 5))
        plt.hist(losses, bins=bins, alpha=0.85)
        plt.axvline(var_, linewidth=2, label=f"VaR {int(alpha*100)}% = {var_:.2f}")
        plt.title(title)
        plt.xlabel("Loss")
        plt.ylabel("Fréquence")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.show()

    @staticmethod
    def pnl_from_log_returns(log_returns, notional: float = 1.0, horizon_days: int = 1) -> np.ndarray:
        if horizon_days > 1:
            lr = log_returns.rolling(horizon_days).sum().dropna().to_numpy()
        else:
            lr = log_returns.to_numpy()

        r = np.exp(lr) - 1.0
        return float(notional) * r



