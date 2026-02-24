"""
risk/portfolio.py — Portföy Takip ve Yönetim
Açık/kapanan pozisyonlar, P&L hesaplama, drawdown takibi.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class Position:
    """Tek bir ticaret pozisyonunu temsil eder."""

    def __init__(
        self,
        ticker: str,
        direction: str,
        shares: int,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        asset_type: str = "stock",  # 'stock' veya 'crypto'
    ):
        self.ticker = ticker
        self.direction = direction  # 'BUY' veya 'SELL'
        self.shares = shares
        self.entry_price = entry_price
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.asset_type = asset_type
        self.opened_at = datetime.now().isoformat()
        self.closed_at: Optional[str] = None
        self.exit_price: Optional[float] = None
        self.pnl: Optional[float] = None
        self.status = "OPEN"  # 'OPEN', 'CLOSED_TP', 'CLOSED_SL', 'CLOSED_MANUAL'

    @property
    def cost_basis(self) -> float:
        return self.entry_price * self.shares

    def unrealized_pnl(self, current_price: float) -> float:
        """Gerçekleşmemiş P&L hesaplar."""
        multiplier = 1 if self.direction == "BUY" else -1
        return (current_price - self.entry_price) * self.shares * multiplier

    def unrealized_pnl_pct(self, current_price: float) -> float:
        """Yüzde olarak gerçekleşmemiş P&L."""
        if self.cost_basis == 0:
            return 0.0
        return self.unrealized_pnl(current_price) / self.cost_basis * 100

    def close(self, exit_price: float, reason: str = "MANUAL") -> float:
        """Pozisyonu kapatır, gerçekleşen P&L döndürür."""
        self.exit_price = exit_price
        self.closed_at = datetime.now().isoformat()
        multiplier = 1 if self.direction == "BUY" else -1
        self.pnl = (exit_price - self.entry_price) * self.shares * multiplier
        self.status = f"CLOSED_{reason}"
        return self.pnl

    def should_stop_loss(self, current_price: float) -> bool:
        """Stop loss tetiklenip tetiklenmediğini kontrol eder."""
        if self.direction == "BUY":
            return current_price <= self.stop_loss
        return current_price >= self.stop_loss

    def should_take_profit(self, current_price: float) -> bool:
        """Take profit tetiklenip tetiklenmediğini kontrol eder."""
        if self.direction == "BUY":
            return current_price >= self.take_profit
        return current_price <= self.take_profit

    def to_dict(self) -> dict:
        return {
            "ticker": self.ticker,
            "direction": self.direction,
            "shares": self.shares,
            "entry_price": self.entry_price,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "asset_type": self.asset_type,
            "opened_at": self.opened_at,
            "closed_at": self.closed_at,
            "exit_price": self.exit_price,
            "pnl": self.pnl,
            "status": self.status,
            "cost_basis": self.cost_basis,
        }


class Portfolio:
    """Tüm pozisyonları ve portföy performansını yönetir."""

    def __init__(self, initial_capital: float = 10_000.0):
        self._initial_capital = initial_capital
        self._cash = initial_capital
        self._positions: dict[str, Position] = {}  # ticker → Position
        self._closed_trades: list[dict] = []
        self._equity_history: list[float] = [initial_capital]
        self._peak_equity = initial_capital
        self._daily_start_equity = initial_capital

    # ── Pozisyon Yönetimi ────────────────────────────────────

    def open_position(self, position: Position) -> bool:
        """Yeni pozisyon açar."""
        if position.ticker in self._positions:
            logger.warning(f"{position.ticker} zaten açık pozisyon var!")
            return False

        cost = position.cost_basis
        if cost > self._cash:
            logger.warning(
                f"Yetersiz nakit! Gerekli: ${cost:.2f}, Mevcut: ${self._cash:.2f}"
            )
            return False

        self._cash -= cost
        self._positions[position.ticker] = position
        logger.info(
            f"Pozisyon açıldı: {position.ticker} {position.direction} "
            f"{position.shares} adet @ ${position.entry_price:.2f}"
        )
        return True

    def close_position(
        self, ticker: str, exit_price: float, reason: str = "MANUAL"
    ) -> Optional[float]:
        """Pozisyonu kapatır, P&L döndürür."""
        if ticker not in self._positions:
            logger.warning(f"{ticker}: Kapatılacak pozisyon bulunamadı.")
            return None

        pos = self._positions.pop(ticker)
        pnl = pos.close(exit_price, reason)
        self._cash += pos.exit_price * pos.shares + pnl  # nakit geri gelir
        trade_record = pos.to_dict()
        self._closed_trades.append(trade_record)
        self._update_equity_history(self.total_equity({ticker: exit_price}))

        logger.info(
            f"Pozisyon kapatıldı: {ticker} @ ${exit_price:.2f} | "
            f"P&L: ${pnl:.2f} ({reason})"
        )
        return pnl

    def check_stops(self, prices: dict[str, float]) -> list[str]:
        """Tüm açık pozisyonlar için SL/TP kontrolü yapar.

        Args:
            prices: {ticker: current_price}

        Returns:
            Kapatılan ticker listesi
        """
        closed = []
        for ticker, pos in list(self._positions.items()):
            price = prices.get(ticker)
            if price is None:
                continue

            if pos.should_take_profit(price):
                self.close_position(ticker, price, reason="TP")
                closed.append(ticker)
            elif pos.should_stop_loss(price):
                self.close_position(ticker, price, reason="SL")
                closed.append(ticker)
        return closed

    # ── Performans Metrikleri ────────────────────────────────

    def total_equity(self, current_prices: Optional[dict] = None) -> float:
        """Toplam portföy değerini hesaplar (nakit + açık pozisyonlar)."""
        positions_value = sum(
            pos.entry_price * pos.shares +
            (pos.unrealized_pnl(current_prices.get(pos.ticker, pos.entry_price))
             if current_prices else 0)
            for pos in self._positions.values()
        )
        return self._cash + positions_value

    def daily_pnl(self) -> float:
        """Günlük P&L yüzdesi."""
        current = self.total_equity()
        if self._daily_start_equity == 0:
            return 0.0
        return (current / self._daily_start_equity - 1) * 100

    def total_pnl(self) -> float:
        """Başlangıçtan bu yana toplam P&L yüzdesi."""
        current = self.total_equity()
        return (current / self._initial_capital - 1) * 100

    def max_drawdown(self) -> float:
        """Maksimum drawdown yüzdesi."""
        equity = np.array(self._equity_history)
        if len(equity) < 2:
            return 0.0
        peak = np.maximum.accumulate(equity)
        dd = (equity - peak) / np.maximum(peak, 1e-10)
        return float(dd.min() * 100)

    def win_rate(self) -> float:
        """Kazanılan trade yüzdesi."""
        if not self._closed_trades:
            return 0.0
        wins = sum(1 for t in self._closed_trades if (t.get("pnl") or 0) > 0)
        return wins / len(self._closed_trades) * 100

    def sharpe_ratio(self) -> float:
        """Yıllıklandırılmış Sharpe Ratio."""
        equity = np.array(self._equity_history)
        if len(equity) < 3:
            return 0.0
        returns = np.diff(equity) / equity[:-1]
        if np.std(returns) == 0:
            return 0.0
        return float(np.mean(returns) / np.std(returns) * np.sqrt(252))

    def get_summary(self, current_prices: Optional[dict] = None) -> dict:
        """Kapsamlı portföy özet raporu."""
        return {
            "initial_capital": self._initial_capital,
            "current_equity": round(self.total_equity(current_prices), 2),
            "cash": round(self._cash, 2),
            "open_positions": len(self._positions),
            "closed_trades": len(self._closed_trades),
            "daily_pnl": round(self.daily_pnl(), 3),
            "total_pnl": round(self.total_pnl(), 3),
            "max_drawdown": round(self.max_drawdown(), 3),
            "win_rate": round(self.win_rate(), 2),
            "sharpe_ratio": round(self.sharpe_ratio(), 3),
        }

    def _update_equity_history(self, equity: float) -> None:
        self._equity_history.append(equity)
        self._peak_equity = max(self._peak_equity, equity)

    def reset_daily(self) -> None:
        """Günlük başlangıç değerini günceller."""
        self._daily_start_equity = self.total_equity()

    @property
    def open_positions(self) -> list[dict]:
        return [p.to_dict() for p in self._positions.values()]

    @property
    def closed_trades_list(self) -> list[dict]:
        return self._closed_trades
