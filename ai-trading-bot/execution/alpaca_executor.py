"""
execution/alpaca_executor.py — Alpaca ABD Hisse Emir Yürütme
Market order, bracket order, pozisyon takibi.
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)


class AlpacaExecutor:
    """Alpaca Trading API üzerinden hisse emri yürütür.

    paper=True → Alpaca Paper Trading (test)
    paper=False → Gerçek para (DİKKAT!)
    """

    def __init__(self, api_key: str, secret_key: str, paper: bool = True):
        try:
            from alpaca.trading.client import TradingClient
            from alpaca.trading.requests import (
                MarketOrderRequest,
                LimitOrderRequest,
                TakeProfitRequest,
                StopLossRequest,
                BracketOrderRequest,
            )
            from alpaca.trading.enums import OrderSide, TimeInForce, OrderType

            self._TradingClient = TradingClient
            self._MarketOrderRequest = MarketOrderRequest
            self._TakeProfitRequest = TakeProfitRequest
            self._StopLossRequest = StopLossRequest
            self._BracketOrderRequest = BracketOrderRequest
            self._OrderSide = OrderSide
            self._TimeInForce = TimeInForce

            self._client = TradingClient(api_key, secret_key, paper=paper)
            self._paper = paper
            mode = "PAPER" if paper else "LIVE"
            logger.info(f"AlpacaExecutor başlatıldı [{mode} MODE].")

        except ImportError:
            logger.error("alpaca-py kurulu değil: pip install alpaca-py")
            raise

    def execute_order(
        self,
        ticker: str,
        side: str,
        qty: int,
        stop_loss: float,
        take_profit: float,
    ) -> dict:
        """Market order + stoplu bracket order gönderir.

        Args:
            ticker:      Hisse sembolü
            side:        'BUY' veya 'SELL'
            qty:         Lot sayısı
            stop_loss:   Stop loss fiyatı
            take_profit: Take profit fiyatı

        Returns:
            Order yanıt sözlüğü
        """
        if qty <= 0:
            logger.warning(f"{ticker}: Geçersiz lot sayısı ({qty}). İşlem iptal.")
            return {"status": "SKIPPED", "reason": "Sıfır lot"}

        order_side = (
            self._OrderSide.BUY if side.upper() == "BUY" else self._OrderSide.SELL
        )

        try:
            # Bracket order: market giriş + otomatik TP + SL
            order_request = self._MarketOrderRequest(
                symbol=ticker,
                qty=qty,
                side=order_side,
                time_in_force=self._TimeInForce.DAY,
                take_profit=self._TakeProfitRequest(limit_price=round(take_profit, 2)),
                stop_loss=self._StopLossRequest(stop_price=round(stop_loss, 2)),
            )

            response = self._client.submit_order(order_request)
            logger.info(
                f"Alpaca order gönderildi: {ticker} {side} {qty} adet "
                f"| TP=${take_profit:.2f} | SL=${stop_loss:.2f} "
                f"| ID={response.id}"
            )
            return {
                "status": "SUBMITTED",
                "order_id": str(response.id),
                "ticker": ticker,
                "side": side,
                "qty": qty,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
            }

        except Exception as e:
            logger.error(f"Alpaca order hatası ({ticker}): {e}")
            return {"status": "ERROR", "error": str(e)}

    def get_account(self) -> dict:
        """Hesap bilgilerini döndürür."""
        try:
            account = self._client.get_account()
            return {
                "cash": float(account.cash),
                "portfolio_value": float(account.portfolio_value),
                "buying_power": float(account.buying_power),
                "equity": float(account.equity),
                "day_trade_count": getattr(account, "daytrade_count", 0),
            }
        except Exception as e:
            logger.error(f"Alpaca hesap bilgisi hatası: {e}")
            return {}

    def get_positions(self) -> list[dict]:
        """Açık pozisyonları döndürür."""
        try:
            positions = self._client.get_all_positions()
            return [
                {
                    "ticker": p.symbol,
                    "qty": float(p.qty),
                    "avg_entry_price": float(p.avg_entry_price),
                    "current_price": float(p.current_price),
                    "unrealized_pl": float(p.unrealized_pl),
                    "unrealized_plpc": float(p.unrealized_plpc),
                    "side": p.side,
                }
                for p in positions
            ]
        except Exception as e:
            logger.error(f"Alpaca pozisyon hatası: {e}")
            return []

    def cancel_all_orders(self) -> None:
        """Bekleyen tüm emirleri iptal eder."""
        try:
            self._client.cancel_orders()
            logger.info("Tüm bekleyen emirler iptal edildi.")
        except Exception as e:
            logger.error(f"Emir iptal hatası: {e}")
