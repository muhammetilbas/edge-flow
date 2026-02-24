"""
ai/explainer.py — OpenAI (ChatGPT) ile Trade Açıklaması Üretimi
"""

from __future__ import annotations

import logging
import time

from openai import OpenAI, RateLimitError
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)


class TradeExplainer:
    """Bir trade sinyalini teknik verilerle birlikte ChatGPT'ye açıklatır.

    Üretilen açıklama Telegram mesajlarında ve dashboard'da gösterilir.
    """

    SYSTEM_PROMPT = (
        "Sen deneyimli bir teknik analist ve quant trader'sın. "
        "Trade kararlarını nesnel, teknik ve kısa bir dille açıklarsın. "
        "Kesin tahmin yapmazsın; gözlemsel ve analitik kalırsın."
    )

    def __init__(self, api_key: str, model: str = "gpt-4o"):
        self._client = OpenAI(api_key=api_key)
        self._model = model
        logger.info(f"TradeExplainer başlatıldı. Model: {model}")

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def explain_trade_signal(
        self,
        ticker: str,
        features: dict,
        prediction: str,
        confidence: float,
        market_regime: str = "UNKNOWN",
    ) -> str:
        """Bir trade sinyalini teknik verilerle açıklar.

        Args:
            ticker:        Hisse/kripto sembolü
            features:      Teknik indikatörler sözlüğü
            prediction:    'BUY', 'SELL' veya 'HOLD'
            confidence:    Model güven skoru (0.0-1.0)
            market_regime: 'BULL', 'BEAR', 'SIDEWAYS'

        Returns:
            3-4 cümlelik teknik açıklama metni
        """
        prompt = f"""
Aşağıdaki trade kararını analiz et ve kısa bir teknik açıklama yaz.

═══════════════════════════════
Hisse/Kripto: {ticker}
Model Kararı: {prediction} (Güven: %{confidence * 100:.0f})
Piyasa Rejimi: {market_regime}
═══════════════════════════════

Teknik Göstergeler:
• RSI(14): {features.get('rsi_14', 'N/A')}
• RSI(7):  {features.get('rsi_7', 'N/A')}
• EMA Trend (0-2): {features.get('ema_trend', 'N/A')}
• ATR%: {features.get('atr_pct', 'N/A')}
• Bollinger Band Pozisyonu: {features.get('bb_position', 'N/A')}
• Hacim Spike: {features.get('volume_spike', 'N/A')}x
• MACD Histogram: {features.get('macd_hist', 'N/A')}
• 3 Günlük Momentum: {features.get('momentum_3d', 'N/A')}
• 5 Günlük Momentum: {features.get('momentum_5d', 'N/A')}
• Sentiment Skoru: {features.get('sentiment_score', 'N/A')}
• VIX: {features.get('vix', 'N/A')}

Kurallar:
- 3-4 cümle yaz, daha fazla yazma
- Türkçe yaz
- Gözlemsel kal, kesin tahmin yapma
- Önemli teknik sinyalleri vurgula
- "muhtemelen", "görünüyor", "işaret ediyor" gibi ifadeleri kullan
"""
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                max_tokens=300,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
            )
            explanation = response.choices[0].message.content.strip()
            logger.info(f"Trade açıklaması üretildi: {ticker} ({prediction})")
            return explanation
        except RateLimitError:
            logger.warning("OpenAI rate limit! 60s bekleniyor...")
            time.sleep(60)
            raise
        except Exception as e:
            logger.error(f"Trade açıklama hatası ({ticker}): {e}")
            return (
                f"{ticker} için {prediction} sinyali üretildi. "
                f"Güven skoru: %{confidence * 100:.0f}. "
                f"Teknik açıklama geçici olarak kullanılamıyor."
            )

    def explain_portfolio_summary(
        self, positions: list[dict], daily_pnl: float, total_pnl: float
    ) -> str:
        """Günlük portföy özetini açıklar."""
        if not positions:
            return "Portföyde aktif pozisyon bulunmuyor."

        positions_text = "\n".join([
            f"• {p['ticker']}: {p['direction']} | P&L: %{p.get('pnl_pct', 0):.2f}"
            for p in positions[:5]
        ])

        prompt = f"""
Portföy durumunu kısaca değerlendir (2-3 cümle, Türkçe):

Günlük P&L: %{daily_pnl:.2f}
Toplam P&L: %{total_pnl:.2f}
Açık Pozisyonlar:
{positions_text}
"""
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                max_tokens=200,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Portföy açıklama hatası: {e}")
            return f"Günlük P&L: %{daily_pnl:.2f} | Toplam P&L: %{total_pnl:.2f}"
