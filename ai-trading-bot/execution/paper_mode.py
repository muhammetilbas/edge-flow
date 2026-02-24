"""
execution/paper_mode.py — Paper Trading Simülasyon Modu
Gerçek para kullanmadan işlem simulasyonu.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)


class PaperTrader:
    """Gerçek para olmadan tam işlem simülasyonu.

    Alpaca veya CCXT gerçek API'yi çağırmak yerine bu sınıfı kullan.
    Tüm işlemler bellekte simüle edilir ve loga yazılır.
    """

    def __init__(self, initial_capital: float = 10_000.0):
        self._capital = initial_capital
        self._cash = initial_capital
        self._positions: dict[str, dict] = {}
        self._orders: list[dict] = []
        self._order_counter = 1
        logger.info(f"PaperTrader başlatıldı. Sermaye: ${initial_capital:,.2f}")

    def execute_order(
        self,
        ticker: str,
        side: str,
        qty: int,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        asset_type: str = "stock",
    ) -> dict:
        """Paper order simüle eder.

        Args:
            ticker:      Sembol
            side:        'BUY' veya 'SELL'
            qty:         Lot sayısı
            entry_price: Giriş fiyatı
            stop_loss:   Stop loss fiyatı
            take_profit: Take profit fiyatı
            asset_type:  'stock' veya 'crypto'

        Returns:
            Simüle edilmiş order sonucu
        """
        if qty <= 0:
            return {"status": "SKIPPED", "reason": "Sıfır lot"}

        notional = qty * entry_price
        commission = notional * 0.001  # %0.1 komisyon sim.

        if side.upper() == "BUY":
            total_cost = notional + commission
            if total_cost > self._cash:
                logger.warning(
                    f"Yetersiz nakit! Gerekli: ${total_cost:.2f}, "
                    f"Mevcut: ${self._cash:.2f}"
                )
                return {"status": "REJECTED", "reason": "Yetersiz nakit"}
            self._cash -= total_cost
        else:
            # Short için marjin kontrolü (basit)
            self._cash -= commission

        order_id = f"PAPER_{self._order_counter:05d}"
        self._order_counter += 1

        order = {
            "order_id": order_id,
            "ticker": ticker,
            "side": side.upper(),
            "qty": qty,
            "entry_price": round(entry_price, 4),
            "stop_loss": round(stop_loss, 4),
            "take_profit": round(take_profit, 4),
            "commission": round(commission, 4),
            "status": "FILLED",
            "asset_type": asset_type,
            "timestamp": datetime.now().isoformat(),
        }
        self._orders.append(order)
        self._positions[ticker] = order

        logger.info(
            f"[PAPER] {side} {qty}x{ticker} @ ${entry_price:.4f} "
            f"| SL=${stop_loss:.2f} | TP=${take_profit:.2f} "
            f"| Komisyon=${commission:.2f} | Bakiye=${self._cash:.2f}"
        )
        return {"status": "FILLED", "order_id": order_id}

    def close_position(
        self, ticker: str, exit_price: float, reason: str = "MANUAL"
    ) -> Optional[dict]:
        """Pozisyonu simüle ederek kapatır."""
        if ticker not in self._positions:
            logger.warning(f"[PAPER] {ticker}: Kapatılacak pozisyon yok.")
            return None

        pos = self._positions.pop(ticker)
        qty = pos["qty"]
        entry = pos["entry_price"]
        side_mult = 1 if pos["side"] == "BUY" else -1
        commission = qty * exit_price * 0.001

        pnl = (exit_price - entry) * qty * side_mult - commission
        self._cash += qty * exit_price - commission

        result = {
            "ticker": ticker,
            "side": pos["side"],
            "qty": qty,
            "entry_price": entry,
            "exit_price": round(exit_price, 4),
            "pnl": round(pnl, 4),
            "pnl_pct": round(pnl / (entry * qty) * 100, 3) if entry * qty > 0 else 0,
            "commission": round(commission, 4),
            "reason": reason,
            "closed_at": datetime.now().isoformat(),
        }
        logger.info(
            f"[PAPER] Kapatıldı: {ticker} {exit_price:.4f} "
            f"| P&L=${pnl:.2f} (%{result['pnl_pct']:.2f}) | {reason}"
        )
        return result

    def update_prices(self, prices: dict[str, float]) -> list[dict]:
        """Fiyatları güncelleyip SL/TP tetiklenip tetiklenmediğini kontrol eder."""
        closed = []
        for ticker, pos in list(self._positions.items()):
            price = prices.get(ticker)
            if price is None:
                continue

            side = pos["side"]
            if side == "BUY":
                if price >= pos["take_profit"]:
                    result = self.close_position(ticker, price, "TP")
                    if result:
                        closed.append(result)
                elif price <= pos["stop_loss"]:
                    result = self.close_position(ticker, price, "SL")
                    if result:
                        closed.append(result)
            else:
                if price <= pos["take_profit"]:
                    result = self.close_position(ticker, price, "TP")
                    if result:
                        closed.append(result)
                elif price >= pos["stop_loss"]:
                    result = self.close_position(ticker, price, "SL")
                    if result:
                        closed.append(result)
        return closed

    def get_account(self) -> dict:
        """Paper hesap durumunu döndürür."""
        positions_value = sum(
            p["qty"] * p["entry_price"] for p in self._positions.values()
        )
        return {
            "cash": round(self._cash, 2),
            "positions_value": round(positions_value, 2),
            "total_equity": round(self._cash + positions_value, 2),
            "initial_capital": self._capital,
            "total_pnl_pct": round(
                (self._cash + positions_value) / self._capital * 100 - 100, 3
            ),
            "open_positions": len(self._positions),
            "total_orders": len(self._orders),
        }

    @property
    def order_history(self) -> list[dict]:
        return self._orders

    @property
    def open_positions_list(self) -> list[dict]:
        return list(self._positions.values())
