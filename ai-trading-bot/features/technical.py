"""
features/technical.py — Teknik İndikatörler
RSI, EMA, ATR, MACD, Bollinger Bands, Volume features.
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
    logging.warning("pandas_ta kurulu değil. Bazı indikatörler manuel hesaplanacak.")

logger = logging.getLogger(__name__)

# ML modelinde kullanılacak özellik listesi
FEATURE_COLUMNS = [
    "rsi_14",
    "rsi_7",
    "ema_trend",
    "atr_pct",
    "bb_position",
    "volume_spike",
    "gap_flag",
    "momentum_3d",
    "momentum_5d",
    "macd_hist",
    "sentiment_score",
    "vix",
    "spy_correlation",
]


class TechnicalFeatureGenerator:
    """Fiyat verisinden teknik özellikler üretir.

    pandas_ta kütüphanesini kullanır;
    yoksa temel indikatörler manuel hesaplanır.
    """

    def __init__(self):
        if not PANDAS_TA_AVAILABLE:
            logger.warning("pandas_ta mevcut değil; manuel hesaplama moduna geçiliyor.")

    def generate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Tüm teknik özellikleri hesaplar ve DataFrame'e ekler.

        Args:
            df: 'open', 'high', 'low', 'close', 'volume' sütunlu DataFrame

        Returns:
            Tüm özelliklerle zenginleştirilmiş DataFrame (NaN'lar temizlenmiş)
        """
        df = df.copy()
        df.columns = [c.lower() for c in df.columns]

        # Zorunlu sütun kontrolü
        required = {"open", "high", "low", "close", "volume"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"DataFrame'de eksik sütunlar: {missing}")

        if PANDAS_TA_AVAILABLE:
            df = self._generate_with_pandas_ta(df)
        else:
            df = self._generate_manual(df)

        return df.dropna()

    def _generate_with_pandas_ta(self, df: pd.DataFrame) -> pd.DataFrame:
        """pandas_ta ile tam indikatör seti."""
        # ── Momentum ──────────────────────────────────────────
        df["rsi_14"] = ta.rsi(df["close"], length=14)
        df["rsi_7"] = ta.rsi(df["close"], length=7)

        # ── Trend ─────────────────────────────────────────────
        df["ema_9"] = ta.ema(df["close"], length=9)
        df["ema_21"] = ta.ema(df["close"], length=21)
        df["ema_50"] = ta.ema(df["close"], length=50)
        df["ema_cross_9_21"] = (df["ema_9"] > df["ema_21"]).astype(int)
        df["ema_cross_21_50"] = (df["ema_21"] > df["ema_50"]).astype(int)
        df["ema_trend"] = df["ema_cross_9_21"] + df["ema_cross_21_50"]  # 0, 1, 2

        # ── Volatility ────────────────────────────────────────
        df["atr"] = ta.atr(df["high"], df["low"], df["close"], length=14)
        df["atr_pct"] = df["atr"] / df["close"]

        bb = ta.bbands(df["close"], length=20, std=2.0)
        if bb is not None:
            bbl_col = [c for c in bb.columns if c.startswith("BBL")]
            bbu_col = [c for c in bb.columns if c.startswith("BBU")]
            if bbl_col and bbu_col:
                bbl = bb[bbl_col[0]]
                bbu = bb[bbu_col[0]]
                band_width = bbu - bbl
                df["bb_position"] = (df["close"] - bbl) / band_width.replace(0, np.nan)
            else:
                df["bb_position"] = 0.5
        else:
            df["bb_position"] = 0.5

        # ── Volume ────────────────────────────────────────────
        df["volume_ma20"] = df["volume"].rolling(20).mean()
        df["volume_spike"] = df["volume"] / df["volume_ma20"].replace(0, np.nan)

        # ── Price Action ─────────────────────────────────────
        prev_close = df["close"].shift(1)
        df["gap_flag"] = (df["open"] - prev_close) / prev_close.replace(0, np.nan)
        df["momentum_3d"] = df["close"].pct_change(3)
        df["momentum_5d"] = df["close"].pct_change(5)

        # ── MACD ──────────────────────────────────────────────
        macd = ta.macd(df["close"])
        if macd is not None:
            hist_col = [c for c in macd.columns if "h" in c.lower() or "hist" in c.lower()]
            if hist_col:
                df["macd_hist"] = macd[hist_col[0]]
            else:
                df["macd_hist"] = 0.0
        else:
            df["macd_hist"] = 0.0

        # ── Placeholder kolonlar (dışarıdan doldurulacak) ────
        if "sentiment_score" not in df.columns:
            df["sentiment_score"] = 0.0
        if "vix" not in df.columns:
            df["vix"] = 0.0
        if "spy_correlation" not in df.columns:
            df["spy_correlation"] = 0.0

        return df

    def _generate_manual(self, df: pd.DataFrame) -> pd.DataFrame:
        """pandas_ta olmadan temel indikatörler (fallback)."""
        # RSI (manuel)
        def calc_rsi(series: pd.Series, period: int) -> pd.Series:
            delta = series.diff()
            gain = delta.clip(lower=0).rolling(period).mean()
            loss = -delta.clip(upper=0).rolling(period).mean()
            rs = gain / loss.replace(0, np.nan)
            return 100 - (100 / (1 + rs))

        df["rsi_14"] = calc_rsi(df["close"], 14)
        df["rsi_7"] = calc_rsi(df["close"], 7)

        # EMA (manuel)
        df["ema_9"] = df["close"].ewm(span=9, adjust=False).mean()
        df["ema_21"] = df["close"].ewm(span=21, adjust=False).mean()
        df["ema_50"] = df["close"].ewm(span=50, adjust=False).mean()
        df["ema_cross_9_21"] = (df["ema_9"] > df["ema_21"]).astype(int)
        df["ema_cross_21_50"] = (df["ema_21"] > df["ema_50"]).astype(int)
        df["ema_trend"] = df["ema_cross_9_21"] + df["ema_cross_21_50"]

        # ATR (manuel)
        tr = pd.concat([
            df["high"] - df["low"],
            (df["high"] - df["close"].shift(1)).abs(),
            (df["low"] - df["close"].shift(1)).abs(),
        ], axis=1).max(axis=1)
        df["atr"] = tr.rolling(14).mean()
        df["atr_pct"] = df["atr"] / df["close"].replace(0, np.nan)

        # Bollinger Bands (manuel)
        ma20 = df["close"].rolling(20).mean()
        std20 = df["close"].rolling(20).std()
        bbl = ma20 - 2 * std20
        bbu = ma20 + 2 * std20
        band_width = bbu - bbl
        df["bb_position"] = (df["close"] - bbl) / band_width.replace(0, np.nan)

        # Volume
        df["volume_ma20"] = df["volume"].rolling(20).mean()
        df["volume_spike"] = df["volume"] / df["volume_ma20"].replace(0, np.nan)

        # Price action
        prev_close = df["close"].shift(1)
        df["gap_flag"] = (df["open"] - prev_close) / prev_close.replace(0, np.nan)
        df["momentum_3d"] = df["close"].pct_change(3)
        df["momentum_5d"] = df["close"].pct_change(5)

        # MACD (manuel)
        ema12 = df["close"].ewm(span=12, adjust=False).mean()
        ema26 = df["close"].ewm(span=26, adjust=False).mean()
        macd_line = ema12 - ema26
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        df["macd_hist"] = macd_line - signal_line

        # Placeholders
        if "sentiment_score" not in df.columns:
            df["sentiment_score"] = 0.0
        if "vix" not in df.columns:
            df["vix"] = 0.0
        if "spy_correlation" not in df.columns:
            df["spy_correlation"] = 0.0

        return df

    def enrich_with_macro(
        self, df: pd.DataFrame, vix_value: float, spy_correlation: float
    ) -> pd.DataFrame:
        """Makro göstergeleri feature DataFrame'ine ekler."""
        df = df.copy()
        df["vix"] = vix_value
        df["spy_correlation"] = spy_correlation
        return df

    def enrich_with_sentiment(
        self, df: pd.DataFrame, sentiment_score: float
    ) -> pd.DataFrame:
        """Sentiment skorunu feature DataFrame'ine ekler."""
        df = df.copy()
        df["sentiment_score"] = sentiment_score
        return df

    def get_latest_features(self, df: pd.DataFrame) -> pd.Series:
        """Son satırın feature değerlerini döndürür."""
        available = [c for c in FEATURE_COLUMNS if c in df.columns]
        return df[available].iloc[-1]
