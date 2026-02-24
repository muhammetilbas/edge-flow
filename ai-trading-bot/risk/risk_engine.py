"""
risk/risk_engine.py — Risk Yönetim Motoru
ATR bazlı stop loss, position sizing, 1:2 R/R, günlük limit.
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Risk sabit değerleri
DEFAULT_MAX_RISK_PER_TRADE = 0.02   # Portföyün %2'si
DEFAULT_ATR_MULTIPLIER = 2.0        # Stop loss = ATR × 2
DEFAULT_RISK_REWARD = 2.0           # 1:2 risk/reward
MIN_CONFIDENCE = 0.60               # Minimum sinyal güveni


class RiskEngine:
    """Pozisyon boyutlandırma ve risk hesaplama motoru.

    Her trade için:
    - ATR bazlı dinamik stop loss
    - Portföy değerine göre lot hesaplama
    - 1:2 risk/reward take profit
    - Günlük kayıp limiti kontrolü
    """

    def __init__(
        self,
        portfolio_value: float,
        max_risk_per_trade: float = DEFAULT_MAX_RISK_PER_TRADE,
        atr_multiplier: float = DEFAULT_ATR_MULTIPLIER,
        risk_reward_ratio: float = DEFAULT_RISK_REWARD,
        max_open_positions: int = 5,
        daily_loss_limit: float = 0.05,
    ):
        self.portfolio_value = portfolio_value
        self.max_risk_per_trade = max_risk_per_trade
        self.atr_multiplier = atr_multiplier
        self.risk_reward_ratio = risk_reward_ratio
        self.max_open_positions = max_open_positions
        self.daily_loss_limit = daily_loss_limit
        self._daily_pnl = 0.0
        self._open_positions_count = 0

    def calculate_position(
        self,
        signal: dict,
        current_price: float,
        atr: float,
        min_confidence: float = MIN_CONFIDENCE,
    ) -> dict:
        """Sinyal için tam risk hesaplama ve pozisyon önerisi.

        Args:
            signal:          generate_signal() çıktısı
            current_price:   İşlem anı fiyatı
            atr:             Average True Range değeri
            min_confidence:  Minimum güven eşiği

        Returns:
            {
                'action': 'BUY'/'SELL'/'SKIP',
                'shares': int,
                'entry': float,
                'stop_loss': float,
                'take_profit': float,
                'risk_reward': str,
                'risk_amount': float,
                'reason': str  (SKIP durumunda)
            }
        """
        direction = signal.get("direction", "HOLD")
        confidence = float(signal.get("confidence", 0.0))

        # ── Filtreler ─────────────────────────────────────────
        if direction == "HOLD":
            return self._skip("HOLD sinyali")

        if confidence < min_confidence:
            return self._skip(f"Düşük güven ({confidence:.2f} < {min_confidence})")

        if self._open_positions_count >= self.max_open_positions:
            return self._skip(f"Max pozisyon limitine ulaşıldı ({self.max_open_positions})")

        if self._daily_pnl <= -self.daily_loss_limit:
            return self._skip(
                f"Günlük kayıp limitine ulaşıldı (%{self._daily_pnl * 100:.2f})"
            )

        if atr <= 0:
            return self._skip("ATR sıfır veya negatif")

        # ── Pozisyon Hesaplama ─────────────────────────────────
        stop_distance = atr * self.atr_multiplier

        if direction == "BUY":
            stop_loss_price = current_price - stop_distance
            take_profit_price = current_price + (stop_distance * self.risk_reward_ratio)
        else:  # SELL / Short
            stop_loss_price = current_price + stop_distance
            take_profit_price = current_price - (stop_distance * self.risk_reward_ratio)

        # Risk miktarı ve lot hesablama
        risk_amount = self.portfolio_value * self.max_risk_per_trade
        shares = int(risk_amount / stop_distance)
        shares = max(1, shares)  # En az 1 lot

        # Notional value kontrolü (portföyün %20'sini geçme)
        max_notional = self.portfolio_value * 0.20
        if shares * current_price > max_notional:
            shares = max(1, int(max_notional / current_price))
            logger.warning(
                f"Lot max notional'a göre düşürüldü: {shares} adet"
            )

        result = {
            "action": direction,
            "shares": shares,
            "entry": round(current_price, 4),
            "stop_loss": round(stop_loss_price, 4),
            "take_profit": round(take_profit_price, 4),
            "risk_reward": f"1:{self.risk_reward_ratio:.0f}",
            "risk_amount": round(risk_amount, 2),
            "stop_distance": round(stop_distance, 4),
            "notional_value": round(shares * current_price, 2),
            "confidence": confidence,
        }
        logger.info(
            f"Pozisyon hesaplandı: {direction} {shares} adet @ ${current_price:.2f} "
            f"| SL=${stop_loss_price:.2f} | TP=${take_profit_price:.2f} "
            f"| Risk=${risk_amount:.2f}"
        )
        return result

    def update_portfolio_value(self, new_value: float) -> None:
        """Portföy değerini günceller (her işlem sonrası çağrılmalı)."""
        old = self.portfolio_value
        self.portfolio_value = new_value
        self._daily_pnl = (new_value / old - 1) if old > 0 else 0.0
        logger.debug(f"Portföy güncellendi: ${new_value:.2f} (Günlük: %{self._daily_pnl * 100:.2f})")

    def register_open_position(self) -> None:
        """Açık pozisyon sayısını artırır."""
        self._open_positions_count += 1

    def register_closed_position(self) -> None:
        """Açık pozisyon sayısını düşürür."""
        self._open_positions_count = max(0, self._open_positions_count - 1)

    def reset_daily_stats(self) -> None:
        """Günlük P&L sıfırlar (her sabah çağrılmalı)."""
        self._daily_pnl = 0.0
        logger.info("Günlük risk istatistikleri sıfırlandı.")

    def is_market_halted(self) -> bool:
        """Günlük kayıp limitinde piyasa durdu mu kontrol eder."""
        return self._daily_pnl <= -self.daily_loss_limit

    @staticmethod
    def _skip(reason: str) -> dict:
        """SKIP sonucu döndürür."""
        logger.info(f"Pozisyon atlandı: {reason}")
        return {"action": "SKIP", "reason": reason, "shares": 0}
