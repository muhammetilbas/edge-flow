"""
ai/market_brief.py — Günlük Piyasa Özeti Üretimi
OpenAI (ChatGPT) API ile sabah/akşam briefingi oluşturur.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime

from openai import OpenAI, RateLimitError
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)


class MarketBriefGenerator:
    """Günlük piyasa brifinglerini ChatGPT ile üretir.

    Sabah açılışında ve akşam kapanışında Telegram'a gönderilir.
    """

    def __init__(self, api_key: str, model: str = "gpt-4o"):
        self._client = OpenAI(api_key=api_key)
        self._model = model
        logger.info(f"MarketBriefGenerator başlatıldı. Model: {model}")

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def generate_daily_brief(
        self,
        portfolio: dict,
        signals: list[dict],
        macro: dict,
        brief_type: str = "morning",
    ) -> str:
        """Günlük piyasa brifingini üretir.

        Args:
            portfolio:  Portföy bilgileri (daily_pnl, total_pnl, open_positions)
            signals:    Günün sinyalleri [{'ticker':..., 'direction':...}]
            macro:      Makro veriler (vix, spy_change, market_fear)
            brief_type: 'morning' veya 'closing'

        Returns:
            Telegram formatında emoji'li kısa özet metni
        """
        buy_count = sum(1 for s in signals if s.get("direction") == "BUY")
        sell_count = sum(1 for s in signals if s.get("direction") == "SELL")

        session_label = "Sabah Açılış" if brief_type == "morning" else "Kapanış"

        prompt = f"""
{session_label} piyasa özeti oluştur (Telegram için emoji kullan).

PIYASA DURUMU:
• VIX: {macro.get('vix', 0):.2f} ({macro.get('vix_regime', 'N/A')})
• SPY Günlük: %{macro.get('spy_change_1d', 0):.2f}
• Piyasa Hissi: {macro.get('market_fear', 'N/A')}

PORTFÖY:
• Günlük P&L: %{portfolio.get('daily_pnl', 0):.2f}
• Toplam P&L: %{portfolio.get('total_pnl', 0):.2f}
• Açık Pozisyon: {portfolio.get('open_positions', 0)}

SİNYALLER: Toplam {len(signals)} sinyal
• 🟢 AL: {buy_count}
• 🔴 SAT: {sell_count}

Kural: Maksimum 280 karakter yaz. Önemli riski vurgula. Türkçe yaz.
"""
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                max_tokens=200,
                messages=[{"role": "user", "content": prompt}],
            )
            brief = response.choices[0].message.content.strip()
            logger.info(f"{brief_type} brief üretildi ({len(brief)} karakter).")
            return brief
        except RateLimitError:
            logger.warning("OpenAI rate limit! 60s bekleniyor...")
            time.sleep(60)
            raise
        except Exception as e:
            logger.error(f"Brief üretim hatası: {e}")
            return self._fallback_brief(macro, portfolio, signals, brief_type)

    def generate_signal_explanation(
        self, ticker: str, signal: dict, position: dict
    ) -> str:
        """Tek bir sinyal için kısa açıklama üretir."""
        prompt = f"""
{ticker} için {signal.get('direction')} sinyalini Türkçe 2 cümle ile açıkla.
Güven: %{signal.get('confidence', 0) * 100:.0f}
Giriş: ${position.get('entry', 0):.2f}
Stop: ${position.get('stop_loss', 0):.2f}
Hedef: ${position.get('take_profit', 0):.2f}
Kısa ve teknik kal.
"""
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                max_tokens=150,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Sinyal açıklama hatası ({ticker}): {e}")
            return f"{ticker}: {signal.get('direction')} sinyali."

    @staticmethod
    def _fallback_brief(
        macro: dict, portfolio: dict, signals: list, brief_type: str
    ) -> str:
        """API hatası durumunda fallback metin üretir."""
        pnl = portfolio.get("daily_pnl", 0)
        vix = macro.get("vix", 0)
        fear = macro.get("market_fear", "N/A")
        buy_c = sum(1 for s in signals if s.get("direction") == "BUY")
        sell_c = sum(1 for s in signals if s.get("direction") == "SELL")

        emoji = "📈" if pnl > 0 else "📉"
        session = "🌅 Sabah" if brief_type == "morning" else "🌆 Kapanış"
        return (
            f"{session} Özeti {datetime.now().strftime('%d.%m %H:%M')}\n"
            f"{emoji} P&L: %{pnl:.2f} | VIX: {vix:.2f} ({fear})\n"
            f"🟢 AL: {buy_c} | 🔴 SAT: {sell_c}"
        )
