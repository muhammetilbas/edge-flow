"""
features/sentiment_features.py — Sentiment Skoru → ML Feature
Haber skoru ve web sentiment'ını ML özelliklerine dönüştürür.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class SentimentFeatureBuilder:
    """Sentiment analiz sonuçlarını ML modelinin anlayabileceği
    sayısal özelliklere dönüştürür.

    Özellikler:
    - sentiment_score:       -1.0 ila 1.0 arası ham skor
    - sentiment_normalized:  0.0 ila 1.0 arası normalize skor
    - sentiment_bullish:     Binary (score > 0.2)
    - sentiment_bearish:     Binary (score < -0.2)
    - risk_keyword_count:    Risk kelimesi sayısı
    - has_earnings_news:     Earnings haberi var mı
    - has_fda_news:          FDA haberi var mı (pharma için)
    - news_volume:           Toplam haber sayısı
    - sentiment_momentum_3d: Son 3 günün ortalama sentiment trendi
    """

    EARNINGS_KEYWORDS = {
        "earnings", "eps", "revenue", "profit", "loss", "guidance",
        "beat", "miss", "quarterly", "annual", "forecast"
    }
    FDA_KEYWORDS = {"fda", "approval", "clinical", "trial", "drug"}
    HIGH_RISK_KEYWORDS = {
        "lawsuit", "sec", "fraud", "investigation", "bankruptcy",
        "default", "delisted", "recall", "scandal", "hack", "breach"
    }

    def __init__(self):
        self._history: dict[str, list[dict]] = {}  # ticker → sentiment geçmişi

    def build_features(
        self,
        ticker: str,
        sentiment_result: dict,
        news_texts: Optional[list[str]] = None,
    ) -> dict:
        """Sentiment sonuçlarından ML özellikleri üretir.

        Args:
            ticker:           Hisse sembolü
            sentiment_result: SentimentAnalyzer.analyze_sentiment() çıktısı
            news_texts:       Ham haber metinleri (keyword analizi için)

        Returns:
            Sayısal özellikler sözlüğü
        """
        score = float(sentiment_result.get("score", 0.0))
        risk_keywords = sentiment_result.get("risk_keywords", [])
        news_count = int(sentiment_result.get("news_count", 0))
        all_text = " ".join(news_texts or []).lower()

        # Geçmişe kaydet
        self._record_history(ticker, score)

        features = {
            "sentiment_score": round(score, 4),
            "sentiment_normalized": round((score + 1.0) / 2.0, 4),  # 0-1 arasy
            "sentiment_bullish": int(score > 0.2),
            "sentiment_bearish": int(score < -0.2),
            "risk_keyword_count": len(risk_keywords),
            "has_high_risk": int(
                any(kw in all_text for kw in self.HIGH_RISK_KEYWORDS)
            ),
            "has_earnings_news": int(
                any(kw in all_text for kw in self.EARNINGS_KEYWORDS)
            ),
            "has_fda_news": int(
                any(kw in all_text for kw in self.FDA_KEYWORDS)
            ),
            "news_volume": news_count,
            "news_volume_spike": self._calc_volume_spike(ticker, news_count),
            "sentiment_momentum_3d": self._calc_momentum(ticker, days=3),
            "sentiment_volatility": self._calc_volatility(ticker),
        }
        logger.debug(f"Sentiment features → {ticker}: {features}")
        return features

    def _record_history(self, ticker: str, score: float) -> None:
        """Ticker için sentiment geçmişini günceller (max 30 gün)."""
        if ticker not in self._history:
            self._history[ticker] = []
        self._history[ticker].append({
            "score": score,
            "timestamp": datetime.now().isoformat(),
        })
        # Max 30 kayıt tut
        self._history[ticker] = self._history[ticker][-30:]

    def _calc_momentum(self, ticker: str, days: int = 3) -> float:
        """Son N günün sentiment trend değişimini hesaplar."""
        history = self._history.get(ticker, [])
        if len(history) < days + 1:
            return 0.0
        recent_avg = np.mean([h["score"] for h in history[-days:]])
        older_avg = np.mean([h["score"] for h in history[-days * 2:-days]])
        return round(float(recent_avg - older_avg), 4)

    def _calc_volatility(self, ticker: str) -> float:
        """Sentiment skorunun standart sapmasını hesaplar."""
        history = self._history.get(ticker, [])
        if len(history) < 3:
            return 0.0
        scores = [h["score"] for h in history[-10:]]
        return round(float(np.std(scores)), 4)

    def _calc_volume_spike(self, ticker: str, current_count: int) -> float:
        """Haber hacminin ortalamasına oranını hesaplar."""
        history = self._history.get(ticker, [])
        if len(history) < 5:
            return 1.0
        # Geçmiş ortalama (yaklaşık)
        avg_count = 5.0  # varsayılan baseline
        return round(current_count / max(avg_count, 1), 2)

    def get_simple_sentiment_score(self, sentiment_result: dict) -> float:
        """Basit sentiment skoru döndürür (ML modeline enjekte için)."""
        return float(sentiment_result.get("score", 0.0))
