"""
execution/crypto_executor.py — CCXT/Binance Kripto Emir Yürütme
"""

from __future__ import annotations

import logging
from typing import Optional

import ccxt

logger = logging.getLogger(__name__)


class CryptoExecutor:
    """CCXT üzerinden kripto emir yürütür.

    testnet=True → Binance Testnet (test)
    testnet=False → Gerçek para (DİKKAT!)
    """

    def __init__(self, api_key: str, secret_key: str, testnet: bool = True):
        params: dict = {}
        if api_key and api_key != "YOUR_BINANCE_API_KEY":
            params["apiKey"] = api_key
            params["secret"] = secret_key

        self._exchange = ccxt.binance(params)
        if testnet:
            self._exchange.set_sandbox_mode(True)

        mode = "TESTNET" if testnet else "LIVE"
        logger.info(f"CryptoExecutor başlatıldı [{mode} MODE].")

    def execute_order(
        self,
        symbol: str,
        side: str,
        amount: float,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
    ) -> dict:
        """Market order + opsiyonel stop loss gönderir.

        Args:
            symbol:      Kripto çifti (örn. 'BTC/USDT')
            side:        'BUY' veya 'SELL'
            amount:      İşlem miktarı (base currency)
            stop_loss:   Stop loss fiyatı (opsiyonel)
            take_profit: Take profit fiyatı (opsiyonel)

        Returns:
            Order yanıt sözlüğü
        """
        if amount <= 0:
            return {"status": "SKIPPED", "reason": "Sıfır miktar"}

        side_ccxt = side.lower()
        results = {}

        try:
            # Ana market order
            main_order = self._exchange.create_order(
                symbol=symbol,
                type="market",
                side=side_ccxt,
                amount=amount,
            )
            results["main_order"] = {
                "id": main_order.get("id"),
                "status": main_order.get("status"),
            }
            logger.info(
                f"Kripto order: {symbol} {side} {amount} "
                f"| ID={main_order.get('id')}"
            )

            # Stop loss order
            if stop_loss:
                sl_side = "sell" if side_ccxt == "buy" else "buy"
                sl_order = self._exchange.create_order(
                    symbol=symbol,
                    type="stop_market",
                    side=sl_side,
                    amount=amount,
                    params={"stopPrice": round(stop_loss, 4)},
                )
                results["stop_loss_order"] = {"id": sl_order.get("id")}
                logger.info(f"Stop loss order: ${stop_loss:.4f}")

            # Take profit order
            if take_profit:
                tp_side = "sell" if side_ccxt == "buy" else "buy"
                try:
                    tp_order = self._exchange.create_order(
                        symbol=symbol,
                        type="take_profit_market",
                        side=tp_side,
                        amount=amount,
                        params={"stopPrice": round(take_profit, 4)},
                    )
                    results["take_profit_order"] = {"id": tp_order.get("id")}
                    logger.info(f"Take profit order: ${take_profit:.4f}")
                except Exception as e:
                    logger.warning(f"TP order gönderilemedi: {e}")

            results["status"] = "SUBMITTED"
            results["symbol"] = symbol
            results["side"] = side
            results["amount"] = amount
            return results

        except ccxt.InsufficientFunds as e:
            logger.error(f"Yetersiz bakiye ({symbol}): {e}")
            return {"status": "ERROR", "error": "Yetersiz bakiye"}
        except ccxt.NetworkError as e:
            logger.error(f"Ağ hatası ({symbol}): {e}")
            return {"status": "ERROR", "error": str(e)}
        except Exception as e:
            logger.error(f"Kripto order hatası ({symbol}): {e}")
            return {"status": "ERROR", "error": str(e)}

    def get_balance(self, currency: str = "USDT") -> float:
        """Belirtilen para biriminin bakiyesini döndürür."""
        try:
            balance = self._exchange.fetch_balance()
            return float(balance.get("free", {}).get(currency, 0.0))
        except Exception as e:
            logger.error(f"Bakiye sorgu hatası: {e}")
            return 0.0

    def get_positions(self) -> list[dict]:
        """Açık future pozisyonlarını döndürür."""
        try:
            positions = self._exchange.fetch_positions()
            return [
                {
                    "symbol": p["symbol"],
                    "side": p["side"],
                    "contracts": p.get("contracts", 0),
                    "entry_price": p.get("entryPrice", 0),
                    "unrealized_pnl": p.get("unrealizedPnl", 0),
                }
                for p in positions
                if p.get("contracts", 0) != 0
            ]
        except Exception as e:
            logger.error(f"Kripto pozisyon sorgu hatası: {e}")
            return []

    def cancel_all_orders(self, symbol: Optional[str] = None) -> None:
        """Bekleyen emirleri iptal eder."""
        try:
            self._exchange.cancel_all_orders(symbol)
            logger.info(f"Kripto emirler iptal edildi ({symbol or 'tümü'}).")
        except Exception as e:
            logger.error(f"Kripto emir iptal hatası: {e}")
