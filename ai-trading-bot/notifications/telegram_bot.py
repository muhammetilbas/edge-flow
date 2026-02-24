"""
notifications/telegram_bot.py — Telegram Sinyal Botu
Sinyal, günlük brief ve alarm mesajları gönderir.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class TelegramNotifier:
    """Telegram Bot API üzerinden mesaj gönderir.

    python-telegram-bot v21+ kullanılır.
    """

    def __init__(self, token: str, chat_id: str):
        if not token or token == "YOUR_TELEGRAM_BOT_TOKEN":
            logger.warning("Telegram token ayarlanmamış. Bildirimler devre dışı.")
            self._enabled = False
            return

        try:
            from telegram import Bot
            from telegram.constants import ParseMode
            self._bot = Bot(token=token)
            self._chat_id = chat_id
            self._ParseMode = ParseMode
            self._enabled = True
            logger.info("TelegramNotifier başlatıldı.")
        except ImportError:
            logger.error("python-telegram-bot kurulu değil: pip install python-telegram-bot")
            self._enabled = False

    # ── Sinyal Mesajları ────────────────────────────────────

    async def send_signal(
        self,
        signal: dict,
        position: dict,
        explanation: str,
    ) -> bool:
        """Trade sinyalini Telegram'a gönderir.

        Args:
            signal:      generate_signal() çıktısı
            position:    calculate_position() çıktısı
            explanation: AI açıklaması

        Returns:
            True = gönderildi, False = hata
        """
        if not self._enabled:
            return False

        from notifications.formatters import format_signal_message
        message = format_signal_message(signal, position, explanation)
        return await self._send(message)

    async def send_daily_brief(self, brief: str, pnl: float, brief_type: str = "morning") -> bool:
        """Günlük piyasa özetini gönderir."""
        if not self._enabled:
            return False

        from notifications.formatters import format_daily_brief_message
        message = format_daily_brief_message(brief, pnl, brief_type)
        return await self._send(message)

    async def send_portfolio_summary(self, portfolio_summary: dict) -> bool:
        """Portföy özeti gönderir."""
        if not self._enabled:
            return False

        from notifications.formatters import format_portfolio_summary
        message = format_portfolio_summary(portfolio_summary)
        return await self._send(message)

    async def send_error_alert(self, error: str, context: str = "") -> bool:
        """Hata uyarısı gönderir."""
        if not self._enabled:
            return False

        from notifications.formatters import format_error_alert
        message = format_error_alert(error, context)
        return await self._send(message)

    async def send_system_halted(self, reason: str, daily_pnl: float) -> bool:
        """Sistem durdurulma bildirimi gönderir."""
        if not self._enabled:
            return False

        from notifications.formatters import format_system_halted
        message = format_system_halted(reason, daily_pnl)
        return await self._send(message)

    async def send_text(self, text: str) -> bool:
        """Ham metin mesajı gönderir."""
        if not self._enabled:
            return False
        return await self._send(text)

    # ── Senkron Yardımcılar ─────────────────────────────────

    def send_signal_sync(self, signal: dict, position: dict, explanation: str) -> bool:
        """Sync wrapper: asyncio event loop dışında kullanım için."""
        return asyncio.run(self.send_signal(signal, position, explanation))

    def send_text_sync(self, text: str) -> bool:
        """Sync wrapper: ham metin gönderir."""
        return asyncio.run(self.send_text(text))

    # ── İç Metod ────────────────────────────────────────────

    async def _send(self, text: str) -> bool:
        """Telegram Bot API ile mesaj gönderir."""
        try:
            await self._bot.send_message(
                chat_id=self._chat_id,
                text=text,
                parse_mode=self._ParseMode.MARKDOWN,
            )
            logger.debug(f"Telegram mesajı gönderildi ({len(text)} karakter).")
            return True
        except Exception as e:
            logger.error(f"Telegram gönderim hatası: {e}")
            return False

    @property
    def is_enabled(self) -> bool:
        return self._enabled
