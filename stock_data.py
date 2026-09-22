import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

class StockData:
    """
    - Télécharge les données (avec retry/backoff + cache)
    - Calcule log-returns
    - Estime mu/sigma
    """

    def __init__(self, ticker: str, start_date: str, end_date: str):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date

    def download(self) -> pd.DataFrame:
        self.df = yf.download(self.ticker, start=self.start_date, end=self.end_date,
                              progress=False, auto_adjust=False, threads=False)
        if self.df.empty or "Close" not in self.df.columns:
            raise ValueError("Données vides ou colonne Close absente.")
        return self.df


    def compute_log_returns(self) -> pd.Series:
        if not hasattr(self, "df"):
            raise ValueError("Télécharge d'abord les données avec download().")

        prices = self.df["Close"].astype(float)
        self.log_returns = np.log(1 + prices.pct_change()).dropna()
        return self.log_returns

    def estimate_mu_sigma(self) -> tuple[float, float]:
        if not hasattr(self, "log_returns"):
            self.compute_log_returns()

        self.mu = self.log_returns.mean().item()
        self.sigma = self.log_returns.std().item()
        return self.mu, self.sigma

    @property
    # dernier prix observé = prix initial de la simulation MC + stress tests
    def S0(self) -> float:
        if not hasattr(self, "df"):
            raise ValueError("Télécharge d'abord les données.")
        return self.df["Close"].iloc[-1].item()

    def last_close_series(self, n: int = 252) -> pd.Series:
        """
        Pour plot historique récent (1 an par défaut).
        """
        if not hasattr(self, "df"):
            raise ValueError("Télécharge d'abord les données.")
        return self.df["Close"].astype(float).dropna().iloc[-n:]

    def plot_price(self) -> None:
        plt.figure(figsize=(11, 5))
        plt.plot(self.df.index, self.df["Close"].astype(float))
        plt.title(f"{self.ticker} Price")
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.grid(True, alpha=0.3)
        plt.show()

    def plot_log_returns(self) -> None:
        plt.figure(figsize=(11, 5))
        plt.plot(self.log_returns.index, self.log_returns.values)
        plt.title(f"{self.ticker} Log Returns")
        plt.xlabel("Date")
        plt.ylabel("Log return")
        plt.grid(True, alpha=0.3)
        plt.show()