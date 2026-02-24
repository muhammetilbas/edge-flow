"""
models/predictor.py — Canlı Tahmin Modülü
Eğitilmiş model ile BUY/SELL/HOLD sinyali üretir.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

from features.technical import FEATURE_COLUMNS
from models.trainer import LABEL_MAP_INV, ModelTrainer

logger = logging.getLogger(__name__)


class SignalPredictor:
    """Eğitilmiş ML modeli ile canlı sinyal üretir.

    Modeli bellekte cache'de tutar; yeni model eğitildiğinde otomatik yükler.
    """

    def __init__(self):
        self._models: dict[str, tuple] = {}  # ticker → (model, metrics, label_map)
        self._trainer = ModelTrainer()

    def load_model(self, ticker: str, model_type: str = "xgb") -> bool:
        """Kaydedilmiş en son modeli yükler."""
        path = self._trainer.get_latest_model_path(ticker, model_type)
        if not path:
            logger.warning(f"{ticker} için kayıtlı model bulunamadı.")
            return False
        model, metrics, label_map = ModelTrainer.load_model(path)
        self._models[ticker] = (model, metrics, label_map)
        logger.info(f"{ticker} modeli yüklendi. Test accuracy: {metrics.get('test_accuracy', 'N/A')}")
        return True

    def generate_signal(
        self,
        ticker: str,
        df: pd.DataFrame,
        sentiment: dict,
        vix: float = 0.0,
        spy_correlation: float = 0.0,
    ) -> dict:
        """Tek ticker için sinyal üretir.

        Model'in BUY/SELL/HOLD olasılıklarını + teknik göstergeleri
        birleştirerek aksiyon odaklı sinyal verir.
        """
        if ticker not in self._models:
            loaded = self.load_model(ticker)
            if not loaded:
                return self._no_model_signal(ticker)

        model, metrics, label_map = self._models[ticker]

        # Feature vektörü hazırla
        df = df.copy()
        df["sentiment_score"] = float(sentiment.get("score", 0.0))
        df["vix"] = vix
        df["spy_correlation"] = spy_correlation

        available_features = [c for c in FEATURE_COLUMNS if c in df.columns]
        if not available_features:
            logger.error(f"{ticker}: Hiçbir feature mevcut değil!")
            return self._no_model_signal(ticker)

        latest = df[available_features].iloc[-1:].copy()

        # NaN kontrolü
        nan_cols = latest.columns[latest.isna().any()].tolist()
        if nan_cols:
            latest = latest.fillna(0)

        try:
            pred_proba = model.predict_proba(latest)[0]
        except Exception as e:
            logger.error(f"{ticker} tahmin hatası: {e}")
            return self._no_model_signal(ticker)

        # Olasılıkları al (model classes: [0=SELL, 1=HOLD, 2=BUY])
        prob_sell = float(pred_proba[0]) if len(pred_proba) > 0 else 0.0
        prob_hold = float(pred_proba[1]) if len(pred_proba) > 1 else 0.0
        prob_buy  = float(pred_proba[2]) if len(pred_proba) > 2 else 0.0

        # ── Teknik gösterge boost ────────────────────────
        last_row = df.iloc[-1]
        rsi = float(last_row.get("rsi_14", 50))
        ema_trend = float(last_row.get("ema_trend", 0))
        macd_hist = float(last_row.get("macd_hist", 0))
        bb_pos = float(last_row.get("bb_position", 0.5))

        # RSI aşırı alım/satım boost
        if rsi < 35:
            prob_buy += 0.12
        elif rsi < 45:
            prob_buy += 0.06
        elif rsi > 65:
            prob_sell += 0.12
        elif rsi > 55:
            prob_sell += 0.06

        # EMA trend boost
        if ema_trend > 0.01:
            prob_buy += 0.08
        elif ema_trend < -0.01:
            prob_sell += 0.08

        # MACD momentum
        if macd_hist > 0:
            prob_buy += 0.04
        elif macd_hist < 0:
            prob_sell += 0.04

        # Bollinger Band position
        if bb_pos < 0.2:     # Alt banda yakın → BUY fırsatı
            prob_buy += 0.05
        elif bb_pos > 0.8:   # Üst banda yakın → SELL fırsatı
            prob_sell += 0.05

        # ── Sinyal kararı ────────────────────────────────
        buy_score = prob_buy
        sell_score = prob_sell
        min_action_threshold = 0.12

        if buy_score > sell_score and buy_score > min_action_threshold:
            direction = "BUY"
            confidence = min(buy_score / (buy_score + sell_score + 0.001), 0.99)
        elif sell_score > buy_score and sell_score > min_action_threshold:
            direction = "SELL"
            confidence = min(sell_score / (buy_score + sell_score + 0.001), 0.99)
        else:
            direction = "HOLD"
            confidence = prob_hold

        signal = {
            "ticker": ticker,
            "direction": direction,
            "confidence": round(confidence, 4),
            "prob_buy": round(prob_buy, 4),
            "prob_hold": round(prob_hold, 4),
            "prob_sell": round(prob_sell, 4),
            "timestamp": datetime.now().isoformat(),
            "model_available": True,
            "features_used": available_features,
        }
        logger.info(
            f"Sinyal → {ticker}: {direction} "
            f"(conf={confidence:.2f}, buy={buy_score:.2f}, sell={sell_score:.2f}, rsi={rsi:.0f})"
        )
        return signal

    def generate_bulk_signals(
        self,
        tickers_data: dict[str, dict],
    ) -> list[dict]:
        """Birden fazla ticker için sinyal üretir."""
        signals = []
        for ticker, data in tickers_data.items():
            signal = self.generate_signal(
                ticker=ticker,
                df=data["df"],
                sentiment=data.get("sentiment", {}),
                vix=data.get("vix", 0.0),
                spy_correlation=data.get("spy_correlation", 0.0),
            )
            signals.append(signal)

        signals.sort(key=lambda s: (s["direction"] == "HOLD", -s["confidence"]))
        return signals

    @staticmethod
    def _no_model_signal(ticker: str) -> dict:
        """Model yokken HOLD sinyali döndürür."""
        return {
            "ticker": ticker,
            "direction": "HOLD",
            "confidence": 0.0,
            "prob_buy": 0.0,
            "prob_hold": 1.0,
            "prob_sell": 0.0,
            "timestamp": datetime.now().isoformat(),
            "model_available": False,
            "reason": "Eğitilmiş model bulunamadı.",
        }
