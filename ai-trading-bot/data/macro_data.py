"""
data/macro_data.py — Makro Veri Modülü
VIX, SPY, sektör ETF'leri ve makro göstergeler.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

# Takip edilecek makro semboller
MACRO_SYMBOLS = {
    "vix": "^VIX",           # Volatilite İndeksi (Fear Index)
    "spy": "SPY",             # S&P 500 ETF
    "qqq": "QQQ",             # Nasdaq-100 ETF
    "dia": "DIA",             # Dow Jones ETF
    "gld": "GLD",             # Altın ETF
    "tlt": "TLT",             # Uzun Vadeli ABD Tahvili
    "xlk": "XLK",             # Teknoloji Sektör ETF
    "xlf": "XLF",             # Finans Sektör ETF
    "xle": "XLE",             # Enerji Sektör ETF
    "xlv": "XLV",             # Sağlık Sektör ETF
    "hyg": "HYG",             # High-Yield Tahvil ETF (Risk iştahı)
    "uup": "UUP",             # USD Endeksi ETF
}


class MacroDataCollector:
    """VIX, SPY ve diğer makro göstergeleri toplar."""

    def __init__(self):
        logger.info("MacroDataCollector başlatıldı.")

    def get_vix(self, period: str = "60d") -> pd.Series:
        """VIX (Fear & Greed) endeksini çeker.

        Args:
            period: Dönem ('30d', '60d', '1y' vb.)

        Returns:
            VIX kapanış değerleri (pd.Series)
        """
        try:
            data = yf.download("^VIX", period=period, auto_adjust=True, progress=False)
            if data.empty:
                raise ValueError("VIX verisi boş döndü.")
            series = data["Close"].squeeze()
            logger.info(f"VIX: {len(series)} gün verisi alındı. Son: {float(series.iloc[-1]):.2f}")
            return series
        except Exception as e:
            logger.error(f"VIX veri hatası: {e}")
            raise

    def get_spy(self, period: str = "60d") -> pd.DataFrame:
        """SPY (S&P 500) OHLCV verisi çeker.

        Returns:
            OHLCV DataFrame
        """
        try:
            data = yf.download("SPY", period=period, auto_adjust=True, progress=False)
            data.columns = [c.lower() for c in data.columns]
            data = data.reset_index()
            data.rename(columns={"date": "timestamp"}, inplace=True)
            logger.info(f"SPY: {len(data)} gün verisi alındı.")
            return data
        except Exception as e:
            logger.error(f"SPY veri hatası: {e}")
            raise

    def get_symbol(self, symbol: str, period: str = "60d") -> pd.DataFrame:
        """Genel yfinance sembol verisi çeker."""
        try:
            data = yf.download(symbol, period=period, auto_adjust=True, progress=False)
            if data.empty:
                raise ValueError(f"{symbol} verisi boş döndü.")
            data.columns = [c.lower() for c in data.columns]
            data = data.reset_index()
            data.rename(columns={"date": "timestamp", "Date": "timestamp"}, inplace=True)
            return data
        except Exception as e:
            logger.error(f"{symbol} veri hatası: {e}")
            raise

    def get_macro_snapshot(self) -> dict:
        """Tüm önemli makro göstergelerin anlık değerini döndürür.

        Returns:
            {
                'vix': float,
                'vix_regime': 'LOW/MEDIUM/HIGH/VERY_HIGH',
                'spy_price': float,
                'spy_change_1d': float,  # % günlük değişim
                'spy_change_5d': float,  # % 5 günlük değişim
                'market_fear': str,      # 'GREEDY/NEUTRAL/FEARFUL/PANIC'
                'timestamp': str
            }
        """
        try:
            vix_series = self.get_vix(period="10d")
            spy_data = self.get_spy(period="10d")

            vix_current = float(vix_series.iloc[-1])
            spy_close = spy_data["close"].values

            spy_price = float(spy_close[-1])
            spy_change_1d = float((spy_close[-1] / spy_close[-2] - 1) * 100) if len(spy_close) >= 2 else 0.0
            spy_change_5d = float((spy_close[-1] / spy_close[-5] - 1) * 100) if len(spy_close) >= 5 else 0.0

            # VIX rejimi
            if vix_current < 15:
                vix_regime = "LOW"
                market_fear = "GREEDY"
            elif vix_current < 20:
                vix_regime = "MEDIUM"
                market_fear = "NEUTRAL"
            elif vix_current < 30:
                vix_regime = "HIGH"
                market_fear = "FEARFUL"
            else:
                vix_regime = "VERY_HIGH"
                market_fear = "PANIC"

            snapshot = {
                "vix": round(vix_current, 2),
                "vix_regime": vix_regime,
                "spy_price": round(spy_price, 2),
                "spy_change_1d": round(spy_change_1d, 3),
                "spy_change_5d": round(spy_change_5d, 3),
                "market_fear": market_fear,
                "timestamp": datetime.now().isoformat(),
            }
            logger.info(f"Makro snapshot: VIX={vix_current:.2f}, SPY={spy_price:.2f}, Fear={market_fear}")
            return snapshot

        except Exception as e:
            logger.error(f"Makro snapshot hatası: {e}")
            return {
                "vix": 0.0,
                "vix_regime": "UNKNOWN",
                "spy_price": 0.0,
                "spy_change_1d": 0.0,
                "spy_change_5d": 0.0,
                "market_fear": "UNKNOWN",
                "timestamp": datetime.now().isoformat(),
            }

    def get_sector_performance(self) -> dict[str, float]:
        """Sektör ETF'lerinin günlük performansını döndürür (%).

        Returns:
            {'XLK': 1.2, 'XLF': -0.5, ...}
        """
        sector_etfs = {"xlk": "XLK", "xlf": "XLF", "xle": "XLE", "xlv": "XLV"}
        perf = {}
        for name, symbol in sector_etfs.items():
            try:
                data = self.get_symbol(symbol, period="5d")
                closes = data["close"].values
                if len(closes) >= 2:
                    change = (closes[-1] / closes[-2] - 1) * 100
                    perf[symbol] = round(float(change), 3)
            except Exception as e:
                logger.warning(f"Sektör ETF hatası ({symbol}): {e}")
        return perf

    def get_spy_correlation(self, stock_df: pd.DataFrame, period: int = 30) -> float:
        """Bir hissenin SPY ile korelasyonunu hesaplar.

        Args:
            stock_df: Hisse OHLCV DataFrame'i ('close' sütunlu)
            period:   Korelasyon penceresi (gün)

        Returns:
            Pearson korelasyon katsayısı (-1 ile 1 arası)
        """
        try:
            spy_data = self.get_spy(period=f"{period + 5}d")
            stock_returns = stock_df["close"].pct_change().dropna().tail(period)
            spy_returns = spy_data["close"].pct_change().dropna().tail(period)

            min_len = min(len(stock_returns), len(spy_returns))
            if min_len < 5:
                return 0.0

            corr = float(
                stock_returns.tail(min_len).reset_index(drop=True)
                .corr(spy_returns.tail(min_len).reset_index(drop=True))
            )
            return round(corr, 4)
        except Exception as e:
            logger.warning(f"SPY korelasyon hesaplama hatası: {e}")
            return 0.0
