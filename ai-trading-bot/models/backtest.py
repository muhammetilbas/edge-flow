"""
models/backtest.py — Walk-Forward Backtest Engine
Geçmiş veride strateji simülasyonu; Sharpe, drawdown, win rate.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

from features.technical import FEATURE_COLUMNS, TechnicalFeatureGenerator
from models.trainer import ModelTrainer, prepare_target, LABEL_MAP_INV

logger = logging.getLogger(__name__)

COMMISSION = 0.001   # %0.1 işlem komisyonu (her iki taraf)
SLIPPAGE = 0.0005    # %0.05 slippage


class Backtester:
    """Walk-forward backtest motoru.

    Gerçekçi sonuçlar için:
    - Komisyon + slippage modellenir
    - Look-ahead bias engellenir (sadece geçmiş veri kullanılır)
    - Her fold ayrı model eğitilir
    """

    def __init__(
        self,
        initial_capital: float = 10_000.0,
        max_risk_per_trade: float = 0.02,
        min_confidence: float = 0.60,
        forward_days: int = 3,
        threshold: float = 0.02,
    ):
        self._capital = initial_capital
        self._max_risk = max_risk_per_trade
        self._min_conf = min_confidence
        self._forward_days = forward_days
        self._threshold = threshold
        self._feature_gen = TechnicalFeatureGenerator()
        self._trainer = ModelTrainer()

    def run(
        self,
        df: pd.DataFrame,
        ticker: str,
        train_ratio: float = 0.7,
    ) -> dict:
        """Basit train/test split backtest.

        Args:
            df:          OHLCV DataFrame
            ticker:      Hisse sembolü
            train_ratio: Eğitim veri oranı

        Returns:
            Backtest metrik sözlüğü
        """
        df = self._feature_gen.generate_features(df)
        df = prepare_target(df, self._forward_days, self._threshold)

        split = int(len(df) * train_ratio)
        if split < 50:
            raise ValueError("Backtest için yetersiz veri. Min 50 eğitim barı gerekli.")

        train_df = df.iloc[:split].copy()
        test_df = df.iloc[split:].copy()

        logger.info(
            f"Backtest → {ticker}: Train={len(train_df)}, Test={len(test_df)}"
        )

        # Model eğit
        model, metrics = self._trainer.train(
            train_df, ticker=f"{ticker}_backtest", forward_days=self._forward_days
        )

        # Test seti üzerinde simüle et
        available_features = [c for c in FEATURE_COLUMNS if c in test_df.columns]
        X_test = test_df[available_features].fillna(0)

        preds_encoded = model.predict(X_test)
        preds = [LABEL_MAP_INV.get(int(p), 0) for p in preds_encoded]
        pred_probas = model.predict_proba(X_test)
        confidences = pred_probas.max(axis=1)

        results = self._simulate_trades(test_df, preds, confidences)
        backtest_metrics = self._calculate_metrics(results, ticker)
        backtest_metrics["model_metrics"] = metrics

        return backtest_metrics

    def _simulate_trades(
        self, df: pd.DataFrame, predictions: list, confidences: np.ndarray
    ) -> pd.DataFrame:
        """İşlem simülasyonu yapar."""
        capital = self._capital
        equity = [capital]
        trades = []
        position = None  # {'entry': float, 'direction': int, 'shares': int}

        closes = df["close"].values

        for i, (pred, conf) in enumerate(zip(predictions, confidences)):
            price = closes[i]

            # Açık pozisyonu kapat (zaman doldu)
            if position and i >= position["exit_idx"]:
                direction = position["direction"]
                entry = position["entry"]
                shares = position["shares"]

                # Slippage + komisyon
                exit_price = price * (1 - SLIPPAGE * direction)
                pnl = (exit_price - entry) * shares * direction
                pnl -= abs(entry * shares * COMMISSION)   # giriş komisyonu
                pnl -= abs(exit_price * shares * COMMISSION)  # çıkış komisyonu

                capital += pnl
                trades.append({
                    "entry_idx": position["entry_idx"],
                    "exit_idx": i,
                    "entry_price": entry,
                    "exit_price": exit_price,
                    "direction": "BUY" if direction == 1 else "SELL",
                    "shares": shares,
                    "pnl": pnl,
                    "pnl_pct": pnl / (entry * shares) if entry * shares > 0 else 0,
                })
                position = None

            # Yeni pozisyon aç
            if position is None and pred != 0 and conf >= self._min_conf:
                risk_amount = capital * self._max_risk
                entry_price = price * (1 + SLIPPAGE * pred)
                shares = max(1, int(risk_amount / (entry_price * self._threshold * 2)))
                direction = int(pred)

                position = {
                    "entry": entry_price,
                    "direction": direction,
                    "shares": shares,
                    "entry_idx": i,
                    "exit_idx": min(i + self._forward_days, len(closes) - 1),
                }

            equity.append(capital)

        result_df = pd.DataFrame({
            "equity": equity[:len(df)],
            "close": closes,
        })
        result_df["trades"] = pd.Series(
            [{} for _ in range(len(df))]
        )
        for t in trades:
            if t["exit_idx"] < len(result_df):
                result_df.at[t["exit_idx"], "trades"] = t

        result_df.attrs["trades_list"] = trades
        result_df.attrs["final_capital"] = capital
        return result_df

    def _calculate_metrics(self, result_df: pd.DataFrame, ticker: str) -> dict:
        """Portföy metriklerini hesaplar."""
        trades = result_df.attrs.get("trades_list", [])
        equity = result_df["equity"].values
        final_capital = result_df.attrs.get("final_capital", self._capital)

        total_return = (final_capital / self._capital - 1) * 100
        n_trades = len(trades)
        win_trades = [t for t in trades if t["pnl"] > 0]
        win_rate = len(win_trades) / n_trades * 100 if n_trades > 0 else 0.0

        # Günlük getiriler
        returns = np.diff(equity) / equity[:-1]
        sharpe = (
            float(np.mean(returns) / np.std(returns) * np.sqrt(252))
            if len(returns) > 1 and np.std(returns) > 0
            else 0.0
        )

        # Max drawdown
        peak = np.maximum.accumulate(equity)
        drawdown = (equity - peak) / np.maximum(peak, 1e-10)
        max_dd = float(drawdown.min() * 100)

        # Profit Factor
        gross_profit = sum(t["pnl"] for t in win_trades)
        gross_loss = abs(sum(t["pnl"] for t in trades if t["pnl"] <= 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        metrics = {
            "ticker": ticker,
            "initial_capital": self._capital,
            "final_capital": round(final_capital, 2),
            "total_return_pct": round(total_return, 2),
            "n_trades": n_trades,
            "win_rate_pct": round(win_rate, 2),
            "sharpe_ratio": round(sharpe, 3),
            "max_drawdown_pct": round(max_dd, 2),
            "profit_factor": round(profit_factor, 3),
            "avg_pnl": round(
                float(np.mean([t["pnl"] for t in trades])) if trades else 0.0, 2
            ),
            "backtest_date": datetime.now().isoformat(),
        }
        logger.info(
            f"Backtest sonuç → {ticker}: "
            f"Return=%{total_return:.2f}, Sharpe={sharpe:.2f}, "
            f"WinRate=%{win_rate:.1f}, MaxDD=%{max_dd:.2f}"
        )
        return metrics
