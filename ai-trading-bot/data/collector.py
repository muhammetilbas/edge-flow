"""
data/collector.py — Veri Toplama Modülü
Alpaca (ABD Hisse) + CCXT (Kripto) + yfinance veri çekme.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Optional

import ccxt
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


# ── Alpaca Veri Sınıfı ──────────────────────────────────────────────────────

class StockDataCollector:
    """Alpaca API üzerinden ABD hisse senedi OHLCV verisi çeker."""

    def __init__(self, api_key: str, secret_key: str, paper: bool = True):
        try:
            from alpaca.data.historical import StockHistoricalDataClient
            from alpaca.data.requests import StockBarsRequest
            from alpaca.data.timeframe import TimeFrame

            self._TimeFrame = TimeFrame
            self._StockBarsRequest = StockBarsRequest
            self._client = StockHistoricalDataClient(api_key, secret_key)
            logger.info("Alpaca StockHistoricalDataClient başlatıldı.")
        except ImportError:
            logger.error("alpaca-py kurulu değil: pip install alpaca-py")
            raise

    def get_ohlcv(self, ticker: str, days: int = 60) -> pd.DataFrame:
        """Belirli hisse için günlük OHLCV verisi çeker.

        Args:
            ticker: Hisse sembolü (örn. 'AAPL')
            days:   Kaç günlük veri çekileceği

        Returns:
            OHLCV sütunlarına sahip DataFrame
        """
        request = self._StockBarsRequest(
            symbol_or_symbols=ticker,
            timeframe=self._TimeFrame.Day,
            start=datetime.now() - timedelta(days=days),
        )
        try:
            df = self._client.get_stock_bars(request).df
            df = df.reset_index()
            # Multi-index olduğunda düzelt
            if "symbol" in df.columns:
                df = df[df["symbol"] == ticker].copy()
            df.columns = [c.lower() for c in df.columns]
            # Standart sütun adları
            rename = {"t": "timestamp", "o": "open", "h": "high",
                      "l": "low", "c": "close", "v": "volume"}
            df.rename(columns=rename, inplace=True)
            logger.info(f"{ticker}: {len(df)} günlük bar alındı.")
            return df
        except Exception as e:
            logger.error(f"{ticker} için Alpaca veri hatası: {e}")
            raise

    def get_latest_price(self, ticker: str) -> float:
        """Hissenin son fiyatını döndürür."""
        df = self.get_ohlcv(ticker, days=5)
        return float(df["close"].iloc[-1])


# ── Kripto Veri Sınıfı ──────────────────────────────────────────────────────

class CryptoDataCollector:
    """CCXT/Binance üzerinden kripto OHLCV verisi çeker."""

    def __init__(self, api_key: str = "", secret_key: str = "", testnet: bool = True):
        params: dict = {}
        if api_key:
            params["apiKey"] = api_key
            params["secret"] = secret_key
        if testnet:
            params["options"] = {"defaultType": "future"}

        self._exchange = ccxt.binance(params)
        if testnet:
            self._exchange.set_sandbox_mode(True)
        logger.info("CCXT Binance exchange başlatıldı.")

    def get_ohlcv(
        self,
        symbol: str = "BTC/USDT",
        timeframe: str = "1d",
        limit: int = 100,
    ) -> pd.DataFrame:
        """Kripto çifti için OHLCV verisi çeker.

        Args:
            symbol:    Kripto çifti (örn. 'BTC/USDT')
            timeframe: Zaman dilimi ('1d', '4h', '1h' vb.)
            limit:     Kaç bar alınacağı

        Returns:
            OHLCV sütunlarına sahip DataFrame
        """
        try:
            raw = self._exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
            df = pd.DataFrame(
                raw, columns=["timestamp", "open", "high", "low", "close", "volume"]
            )
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            logger.info(f"{symbol}: {len(df)} bar alındı ({timeframe}).")
            return df
        except Exception as e:
            logger.error(f"{symbol} için CCXT veri hatası: {e}")
            raise

    def get_latest_price(self, symbol: str = "BTC/USDT") -> float:
        """Kripto çiftinin son fiyatını döndürür."""
        ticker = self._exchange.fetch_ticker(symbol)
        return float(ticker["last"])


# ── Yfinance Veri Sınıfı ────────────────────────────────────────────────────

class YfinanceCollector:
    """yfinance üzerinden hisse/ETF verisi çeker (yedek kaynak)."""

    def get_ohlcv(self, ticker: str, period: str = "60d") -> pd.DataFrame:
        """yfinance ile OHLCV verisi çeker.

        Args:
            ticker: Yahoo Finance sembolü
            period: Dönem ('60d', '1y', '2y' vb.)

        Returns:
            OHLCV DataFrame
        """
        try:
            t = yf.Ticker(ticker)
            data = t.history(period=period, auto_adjust=True)
            if data.empty:
                raise ValueError(f"{ticker} için veri bulunamadı.")
            # Flatten MultiIndex columns if present
            if hasattr(data.columns, 'levels'):
                data.columns = [c[0] if isinstance(c, tuple) else c for c in data.columns]
            data.columns = [c.lower() for c in data.columns]
            data = data.reset_index()
            data.rename(columns={"Date": "timestamp", "date": "timestamp"}, inplace=True)
            # Sadece gerekli sütunları tut
            needed = ["timestamp", "open", "high", "low", "close", "volume"]
            available = [c for c in needed if c in data.columns]
            data = data[available].copy()
            logger.info(f"yfinance {ticker}: {len(data)} bar alındı.")
            return data
        except Exception as e:
            logger.error(f"yfinance {ticker} hatası: {e}")
            raise


# ── Ana DataManager Sınıfı ──────────────────────────────────────────────────

class DataManager:
    """Tüm veri kaynaklarını yöneten ana sınıf.

    Hem Alpaca hem CCXT hem de yfinance üzerinden veri çekebilir;
    cache mekanizması sayesinde aynı veriyi tekrar tekrar çekmez.
    """

    def __init__(self, config: dict):
        self._config = config
        self._cache: dict[str, tuple[datetime, pd.DataFrame]] = {}
        self._cache_ttl_minutes = 15

        # Alpaca
        alpaca_cfg = config.get("alpaca", {})
        self._stock_collector = StockDataCollector(
            api_key=alpaca_cfg.get("api_key", ""),
            secret_key=alpaca_cfg.get("secret_key", ""),
            paper=alpaca_cfg.get("paper", True),
        )

        # CCXT
        binance_cfg = config.get("binance", {})
        self._crypto_collector = CryptoDataCollector(
            api_key=binance_cfg.get("api_key", ""),
            secret_key=binance_cfg.get("secret_key", ""),
            testnet=binance_cfg.get("testnet", True),
        )

        self._yf_collector = YfinanceCollector()

    def _is_cache_valid(self, key: str) -> bool:
        if key not in self._cache:
            return False
        cached_at, _ = self._cache[key]
        return (datetime.now() - cached_at).seconds < self._cache_ttl_minutes * 60

    def get_stock_data(self, ticker: str, days: int = 60, use_cache: bool = True) -> pd.DataFrame:
        """Hisse verisi döndürür; önce cache'e bakar."""
        cache_key = f"stock_{ticker}_{days}"
        if use_cache and self._is_cache_valid(cache_key):
            logger.debug(f"Cache hit: {cache_key}")
            return self._cache[cache_key][1]
        try:
            df = self._stock_collector.get_ohlcv(ticker, days=days)
        except Exception:
            logger.warning(f"Alpaca başarısız, yfinance'a geçiliyor: {ticker}")
            period = f"{days}d" if days <= 365 else f"{max(1, days // 365)}y"
            df = self._yf_collector.get_ohlcv(ticker, period=period)
        self._cache[cache_key] = (datetime.now(), df)
        return df

    def get_crypto_data(
        self, symbol: str = "BTC/USDT", timeframe: str = "1d", limit: int = 100,
        use_cache: bool = True
    ) -> pd.DataFrame:
        """Kripto verisi döndürür; önce cache'e bakar."""
        cache_key = f"crypto_{symbol}_{timeframe}_{limit}"
        if use_cache and self._is_cache_valid(cache_key):
            logger.debug(f"Cache hit: {cache_key}")
            return self._cache[cache_key][1]
        df = self._crypto_collector.get_ohlcv(symbol, timeframe=timeframe, limit=limit)
        self._cache[cache_key] = (datetime.now(), df)
        return df

    def get_latest_stock_price(self, ticker: str) -> float:
        """Hissenin güncel fiyatını döndürür."""
        return self._stock_collector.get_latest_price(ticker)

    def get_latest_crypto_price(self, symbol: str = "BTC/USDT") -> float:
        """Kripto güncel fiyatını döndürür."""
        return self._crypto_collector.get_latest_price(symbol)

    def clear_cache(self) -> None:
        """Tüm cache'i temizler."""
        self._cache.clear()
        logger.info("Veri cache'i temizlendi.")
