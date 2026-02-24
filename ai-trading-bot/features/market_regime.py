"""
features/market_regime.py — Piyasa Rejimi Sınıflandırıcı
BULL / BEAR / SIDEWAYS tespiti.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

try:
    import pandas_ta as ta
    PANDAS_TA_AVAILABLE = True
except ImportError:
    PANDAS_TA_AVAILABLE = False

logger = logging.getLogger(__name__)

# Rejim sabitleri
BULL = "BULL"
BEAR = "BEAR"
SIDEWAYS = "SIDEWAYS"


class MarketRegimeClassifier:
    """EMA ve momentum bazlı piyasa rejimi sınıflandırıcı.

    Hem genel piyasa (SPY) hem de bireysel hisseler için kullanılabilir.
    """

    def classify(self, df: pd.DataFrame) -> str:
        """Fiyat verisi üzerinden rejimi belirler.

        Algoritma:
        - EMA(50) ve EMA(200) hesapla
        - Close > EMA50 > EMA200  → BULL
        - Close < EMA50 < EMA200  → BEAR
        - Diğer durumlar           → SIDEWAYS

        Args:
            df: 'close' sütunlu OHLCV DataFrame

        Returns:
            'BULL', 'BEAR' veya 'SIDEWAYS'
        """
        df = df.copy()
        df.columns = [c.lower() for c in df.columns]

        if len(df) < 201:
            logger.warning(
                f"Rejim tespiti için yetersiz veri ({len(df)} bar). "
                "Min 201 bar gerekli. SIDEWAYS döndürülüyor."
            )
            return SIDEWAYS

        # EMA hesapla
        if PANDAS_TA_AVAILABLE:
            df["ema_50"] = ta.ema(df["close"], length=50)
            df["ema_200"] = ta.ema(df["close"], length=200)
        else:
            df["ema_50"] = df["close"].ewm(span=50, adjust=False).mean()
            df["ema_200"] = df["close"].ewm(span=200, adjust=False).mean()

        current = df.iloc[-1]
        close = current["close"]
        ema50 = current["ema_50"]
        ema200 = current["ema_200"]

        if close > ema50 > ema200:
            regime = BULL
        elif close < ema50 < ema200:
            regime = BEAR
        else:
            regime = SIDEWAYS

        logger.info(
            f"Piyasa Rejimi: {regime} "
            f"(Close={close:.2f}, EMA50={ema50:.2f}, EMA200={ema200:.2f})"
        )
        return regime

    def classify_with_confidence(self, df: pd.DataFrame) -> dict:
        """Rejim tespiti + güven skoru döndürür.

        Returns:
            {
                'regime': 'BULL/BEAR/SIDEWAYS',
                'confidence': float (0.0-1.0),
                'ema_50': float,
                'ema_200': float,
                'close': float,
                'trend_strength': float  # EMA ayrışma yüzdesi
            }
        """
        df = df.copy()
        df.columns = [c.lower() for c in df.columns]

        if len(df) < 201:
            return {
                "regime": SIDEWAYS,
                "confidence": 0.0,
                "ema_50": 0.0,
                "ema_200": 0.0,
                "close": float(df["close"].iloc[-1]) if len(df) > 0 else 0.0,
                "trend_strength": 0.0,
            }

        if PANDAS_TA_AVAILABLE:
            df["ema_50"] = ta.ema(df["close"], length=50)
            df["ema_200"] = ta.ema(df["close"], length=200)
        else:
            df["ema_50"] = df["close"].ewm(span=50, adjust=False).mean()
            df["ema_200"] = df["close"].ewm(span=200, adjust=False).mean()

        current = df.iloc[-1]
        close = float(current["close"])
        ema50 = float(current["ema_50"])
        ema200 = float(current["ema_200"])

        # Trend gücü: EMA50 ile EMA200 arasındaki yüzde fark
        trend_strength = abs(ema50 - ema200) / ema200 * 100 if ema200 != 0 else 0.0

        if close > ema50 > ema200:
            regime = BULL
            confidence = min(1.0, trend_strength / 5.0)  # %5 farkta tam güven
        elif close < ema50 < ema200:
            regime = BEAR
            confidence = min(1.0, trend_strength / 5.0)
        else:
            regime = SIDEWAYS
            confidence = max(0.1, 1.0 - trend_strength / 3.0)

        return {
            "regime": regime,
            "confidence": round(confidence, 3),
            "ema_50": round(ema50, 2),
            "ema_200": round(ema200, 2),
            "close": round(close, 2),
            "trend_strength": round(trend_strength, 3),
        }

    def get_regime_for_signal_filter(
        self, regime: str, signal_direction: str
    ) -> bool:
        """Rejime göre sinyal filtreleme (counter-trend işlem engeli).

        Args:
            regime:           Mevcut piyasa rejimi
            signal_direction: 'BUY', 'SELL' veya 'HOLD'

        Returns:
            True = sinyale izin ver, False = filtrele
        """
        # BEAR piyasada BUY sinyallerine daha temkinli yaklaş
        if regime == BEAR and signal_direction == "BUY":
            logger.warning("BEAR piyasada BUY sinyali! Ek doğrulama gerekebilir.")
            return False  # daha katı modda filtrele

        # BULL piyasada SELL (short) sinyallerine dikkat
        if regime == BULL and signal_direction == "SELL":
            logger.warning("BULL piyasada SELL sinyali! Counter-trend.")
            return True  # izin ver ama uyar

        return True
