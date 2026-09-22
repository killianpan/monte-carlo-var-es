import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class GBMSimulator:
    """
    Simule des trajectoires futures GBM:
    S_{t+1} = S_t * exp((mu - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z)
    """

    def __init__(self, ticker: str, S0: float, mu: float, sigma: float):
        self.ticker = ticker
        self.S0 = float(S0)
        self.mu = float(mu)
        self.sigma = float(sigma)

    def simulate_paths(self, dt: float, n_steps: int, n_paths: int, seed: int = 0) -> pd.DataFrame:
        np.random.seed(seed)
        Z = np.random.normal(0,1, size=(n_steps, n_paths))

        parameters = (self.mu - 0.5 * self.sigma**2) * dt + self.sigma * np.sqrt(dt) * Z

        paths = np.zeros((n_steps + 1, n_paths), dtype=float) # S0 + n_steps
        paths[0, :] = self.S0
        paths[1:, :] = self.S0 * np.exp(np.cumsum(parameters, axis=0))

        self.df_paths = pd.DataFrame(
            paths,
            index=np.arange(0, n_steps + 1),
            columns=[f"path_{i+1}" for i in range(n_paths)],
        )
        return self.df_paths

    # Récupère les prix au temps final (utile pour caculer VaR/ES)
    def terminal_prices(self, horizon_steps: int | None = None) -> np.ndarray:
        if not hasattr(self, "df_paths"):
            raise ValueError("Lance simulate_paths() avant.")
        if horizon_steps is None:
            horizon_steps = len(self.df_paths) - 1  # le dernier index
        if horizon_steps >= len(self.df_paths):
            raise ValueError("horizon_steps trop grand.")
        return self.df_paths.iloc[horizon_steps].to_numpy(dtype=float)

    # utile pour visualiser uniquement les trajectoires MC
    def plot_paths(self, max_paths_to_plot: int = 30) -> None:
        """
        Plot uniquement MC (limite d'affichage pour lisibilité).
        """
        if not hasattr(self, "df_paths"):
            raise ValueError("Lance simulate_paths() avant.")

        cols = self.df_paths.columns[:max_paths_to_plot]
        plt.figure(figsize=(12, 6))
        plt.plot(self.df_paths.index, self.df_paths[cols].values, alpha=0.7)
        plt.title(f"{self.ticker} - Monte Carlo (GBM)")
        plt.xlabel("Step")
        plt.ylabel("Prix")
        plt.grid(True, alpha=0.3)
        plt.show()

    # plot historique réel récent + trajecroires simulées MC
    def plot_history_with_mc(
        self,
        hist_close,
        max_paths_to_plot: int = 15
    ) -> None:
        """
        hist_close: pd.Series de prix close (index = dates)
        Affiche historique + MC avec un index date business.
        """
        if not hasattr(self, "df_paths"):
            raise ValueError("Lance simulate_paths() avant.")

        hist = hist_close.astype(float).dropna()
        dates_hist = hist.index
        last_date = dates_hist[-1]

        # len(df_paths) points
        mc_dates = pd.bdate_range(last_date, periods=len(self.df_paths))    # génère len(self.df_paths) jours ouvrés à partir de last_date (1er jour)


        plt.figure(figsize=(12, 6))
        plt.plot(dates_hist, hist.values, linewidth=2, label=f"{self.ticker} Historique Close")

        cols = self.df_paths.columns[:max_paths_to_plot]
        for col in cols:
            plt.plot(mc_dates, self.df_paths[col].values, alpha=0.7)    # len(x) = len(y)

        plt.title(f"{self.ticker} - Historique + Monte Carlo (GBM)")
        plt.xlabel("Date")
        plt.ylabel("Prix")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.show()