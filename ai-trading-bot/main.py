"""
main.py — AI Trading Bot Ana Giriş Noktası
Tüm modülleri birleştirir ve botu çalıştırır.
"""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path
from typing import Optional

import yaml
from dotenv import load_dotenv

# Proje kökünü Python path'e ekle
sys.path.insert(0, str(Path(__file__).parent))

# .env yükle
load_dotenv()

# ── Modül İmportları ─────────────────────────────────────────
from data.collector import DataManager
from data.news_collector import NewsCollector
from data.web_scraper import WebScraper
from data.macro_data import MacroDataCollector
from ai.sentiment import SentimentAnalyzer
from ai.explainer import TradeExplainer
from ai.market_brief import MarketBriefGenerator
from features.technical import TechnicalFeatureGenerator
from features.market_regime import MarketRegimeClassifier
from models.predictor import SignalPredictor
from models.trainer import ModelTrainer
from risk.risk_engine import RiskEngine
from risk.portfolio import Portfolio, Position
from execution.paper_mode import PaperTrader
from notifications.telegram_bot import TelegramNotifier
from monitoring.logger import TradeLogger
from scheduler import run_scheduler, load_config, load_tickers

# ── Logging ──────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("storage/bot.log", mode="a", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)


# ── Ana Bot Sınıfı ───────────────────────────────────────────

class AITradingBot:
    """Tüm katmanları birleştiren ana bot sınıfı.

    Kullanım:
        bot = AITradingBot(config)
        await bot.start()
    """

    def __init__(self, config: dict, tickers: dict):
        self._cfg = config
        self._tickers = tickers

        # Trading parametreleri
        tc = config.get("trading", {})
        self._paper_mode = tc.get("paper_mode", True)
        self._min_confidence = tc.get("min_confidence", 0.60)
        self._portfolio_value = tc.get("portfolio_value", 10000.0)

        # Aktif hisse listesi
        self._stock_watchlist = [
            t["symbol"] for t in tickers.get("stocks", [])
            if t.get("active", False) and not t.get("reference_only", False)
        ]
        self._crypto_watchlist = [
            t["symbol"] for t in tickers.get("crypto", [])
            if t.get("active", False)
        ]

        logger.info(
            f"Bot başlatılıyor | Mode: {'PAPER' if self._paper_mode else '⚠️ LIVE'}\n"
            f"  Hisse: {self._stock_watchlist}\n"
            f"  Kripto: {self._crypto_watchlist}"
        )

        # ── Veri Katmanı ──────────────────────────────────────
        self._data_manager = DataManager(config)
        self._news_collector = NewsCollector(
            newsapi_key=config.get("news_api", {}).get("api_key", "")
        )
        reddit_cfg = config.get("reddit", {})
        self._web_scraper = WebScraper(
            reddit_client_id=reddit_cfg.get("client_id", ""),
            reddit_client_secret=reddit_cfg.get("client_secret", ""),
        )
        self._macro_collector = MacroDataCollector()

        # ── AI Katmanı (OpenAI / ChatGPT) ─────────────────────
        openai_cfg = config.get("openai", {})
        openai_key = openai_cfg.get("api_key", "")
        openai_model = openai_cfg.get("model", "gpt-4o")
        self._sentiment_analyzer = SentimentAnalyzer(openai_key, model=openai_model)
        self._trade_explainer = TradeExplainer(openai_key, model=openai_model)
        self._market_brief = MarketBriefGenerator(openai_key, model=openai_model)

        # ── Feature & Model Katmanı ───────────────────────────
        self._feature_gen = TechnicalFeatureGenerator()
        self._regime_classifier = MarketRegimeClassifier()
        self._predictor = SignalPredictor()

        # ── Risk & Execution ──────────────────────────────────
        self._risk_engine = RiskEngine(
            portfolio_value=self._portfolio_value,
            max_risk_per_trade=tc.get("max_risk_per_trade", 0.02),
            max_open_positions=tc.get("max_open_positions", 5),
            daily_loss_limit=tc.get("daily_loss_limit", 0.05),
        )
        self._portfolio = Portfolio(initial_capital=self._portfolio_value)
        self._paper_trader = PaperTrader(initial_capital=self._portfolio_value)

        # ── Bildirim & Loglama ────────────────────────────────
        tg_cfg = config.get("telegram", {})
        self._telegram = TelegramNotifier(
            token=tg_cfg.get("token", ""),
            chat_id=str(tg_cfg.get("chat_id", "")),
        )
        self._logger = TradeLogger(
            db_path=config.get("database", {}).get("path", "data/trades.db")
        )

        # Veri cache
        self._market_data: dict = {}
        self._news_cache: dict = {}
        self._macro_snapshot: dict = {}

    # ── Ana İşlemler ─────────────────────────────────────────

    async def update_market_data(self) -> None:
        """Tüm watchlist için veri günceller (15 dakikada bir)."""
        logger.info("📡 Piyasa verileri güncelleniyor...")

        # Makro veri
        try:
            self._macro_snapshot = self._macro_collector.get_macro_snapshot()
        except Exception as e:
            logger.error(f"Makro veri hatası: {e}")
            self._macro_snapshot = {"vix": 0.0, "spy_change_1d": 0.0, "market_fear": "UNKNOWN"}

        # Hisse verileri
        for ticker in self._stock_watchlist:
            try:
                df = self._data_manager.get_stock_data(ticker, days=90, use_cache=False)
                df_feat = self._feature_gen.generate_features(df)
                df_feat = self._feature_gen.enrich_with_macro(
                    df_feat,
                    vix_value=self._macro_snapshot.get("vix", 0.0),
                    spy_correlation=self._macro_collector.get_spy_correlation(df),
                )
                self._market_data[ticker] = df_feat

                # Haber ve sentiment
                news = self._news_collector.get_news(ticker)
                self._news_cache[ticker] = news

            except Exception as e:
                logger.error(f"Veri güncelleme hatası ({ticker}): {e}")

        logger.info(f"✅ {len(self._market_data)} hisse verisi güncellendi.")

    async def generate_and_execute_signals(self) -> None:
        """Sinyal üret, risk hesapla, işlem yap (saatlik)."""
        if self._risk_engine.is_market_halted():
            logger.warning("🛑 Günlük kayıp limiti! Yeni işlem açılmıyor.")
            await self._telegram.send_system_halted(
                "Günlük kayıp limiti aşıldı",
                self._risk_engine._daily_pnl,
            )
            return

        signals_generated = []

        for ticker in self._stock_watchlist:
            df = self._market_data.get(ticker)
            if df is None or df.empty:
                logger.warning(f"{ticker}: Veri yok, atlanıyor.")
                continue

            try:
                # Sentiment analizi
                news = self._news_cache.get(ticker, [])
                sentiment_result = self._sentiment_analyzer.analyze_sentiment(news, ticker)
                df = self._feature_gen.enrich_with_sentiment(
                    df, sentiment_result.get("score", 0.0)
                )

                # Piyasa rejimi
                spy_df = self._data_manager.get_stock_data("SPY", days=300)
                regime = self._regime_classifier.classify(spy_df)

                # Sinyal üret
                signal = self._predictor.generate_signal(
                    ticker=ticker,
                    df=df,
                    sentiment=sentiment_result,
                    vix=self._macro_snapshot.get("vix", 0.0),
                    spy_correlation=float(df.get("spy_correlation", [0.0]).iloc[-1] if hasattr(df.get("spy_correlation"), "iloc") else 0.0),
                )
                signal["market_regime"] = regime
                self._logger.log_signal(signal)
                signals_generated.append(signal)

                # Sadece güçlü sinyalleri işle
                if signal["direction"] == "HOLD" or signal["confidence"] < self._min_confidence:
                    continue

                # Risk hesapla
                current_price = float(df["close"].iloc[-1])
                atr_value = float(df["atr"].iloc[-1]) if "atr" in df.columns else current_price * 0.02
                position = self._risk_engine.calculate_position(signal, current_price, atr_value)

                if position.get("action") == "SKIP":
                    logger.info(f"{ticker} atlandı: {position.get('reason')}")
                    continue

                # Trade açıklaması (Claude)
                explanation = self._trade_explainer.explain_trade_signal(
                    ticker=ticker,
                    features=df.iloc[-1].to_dict(),
                    prediction=signal["direction"],
                    confidence=signal["confidence"],
                    market_regime=regime,
                )

                # Telegram bildirimi
                await self._telegram.send_signal(signal, position, explanation)

                # İşlem yürüt
                if self._paper_mode:
                    order_result = self._paper_trader.execute_order(
                        ticker=ticker,
                        side=signal["direction"],
                        qty=position["shares"],
                        entry_price=current_price,
                        stop_loss=position["stop_loss"],
                        take_profit=position["take_profit"],
                    )
                    logger.info(f"[PAPER] {ticker} işlemi: {order_result['status']}")

                self._logger.log_event(
                    "TRADE_OPEN",
                    f"{ticker} {signal['direction']} {position['shares']} adet",
                )

            except Exception as e:
                logger.error(f"Sinyal işleme hatası ({ticker}): {e}")
                await self.send_error(f"{ticker} sinyal hatası", str(e))

        logger.info(f"✅ {len(signals_generated)} sinyal üretildi.")

    async def send_morning_brief(self) -> None:
        """Sabah piyasa özetini üretir ve gönderir."""
        portfolio_summary = self._portfolio.get_summary()
        if self._paper_mode:
            paper_account = self._paper_trader.get_account()
            portfolio_summary["daily_pnl"] = paper_account.get("total_pnl_pct", 0.0)
            portfolio_summary["total_pnl"] = paper_account.get("total_pnl_pct", 0.0)
            portfolio_summary["open_positions"] = paper_account.get("open_positions", 0)

        brief = self._market_brief.generate_daily_brief(
            portfolio=portfolio_summary,
            signals=[],
            macro=self._macro_snapshot,
            brief_type="morning",
        )
        await self._telegram.send_daily_brief(
            brief, portfolio_summary.get("daily_pnl", 0.0), "morning"
        )

    async def send_closing_summary(self) -> None:
        """Kapanış özetini üretir ve gönderir."""
        portfolio_summary = self._portfolio.get_summary()
        brief = self._market_brief.generate_daily_brief(
            portfolio=portfolio_summary,
            signals=[],
            macro=self._macro_snapshot,
            brief_type="closing",
        )
        await self._telegram.send_daily_brief(
            brief, portfolio_summary.get("daily_pnl", 0.0), "closing"
        )

    async def daily_reset(self) -> None:
        """Günlük istatistikleri sıfırlar."""
        self._risk_engine.reset_daily_stats()
        self._portfolio.reset_daily()
        self._data_manager.clear_cache()
        self._logger.log_event("DAILY_RESET", "Günlük istatistikler sıfırlandı.")
        logger.info("🔄 Günlük reset tamamlandı.")

    async def send_error(self, context: str, error: str) -> None:
        """Hata bildirimi gönderir."""
        await self._telegram.send_error_alert(error, context)

    # ── Başlatma ─────────────────────────────────────────────

    async def start(self) -> None:
        """Botu başlatır."""
        logger.info(
            "\n"
            "╔══════════════════════════════════════╗\n"
            "║      🤖 AI Trading Bot v1.0          ║\n"
            "║   ABD Hisse + Kripto | Python 3.11+  ║\n"
            "╚══════════════════════════════════════╝"
        )
        mode = "📄 PAPER MODE" if self._paper_mode else "⚠️  LIVE MODE (Gerçek Para!)"
        logger.info(f"Trading modu: {mode}")

        await self._telegram.send_text(
            f"🤖 *AI Trading Bot Başlatıldı*\n"
            f"Mode: {mode}\n"
            f"Hisse: {', '.join(self._stock_watchlist)}\n"
            f"Kripto: {', '.join(self._crypto_watchlist)}"
        )
        await run_scheduler(self, self._cfg)


# ── CLI Giriş Noktası ────────────────────────────────────────

async def main():
    """Ana fonksiyon."""
    config = load_config("config/config.yaml")
    tickers = load_tickers("config/tickers.yaml")
    bot = AITradingBot(config, tickers)
    await bot.start()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Bot manuel olarak durduruldu.")
    except Exception as e:
        logger.critical(f"Bot kritik hata ile çöktü: {e}", exc_info=True)
        sys.exit(1)
