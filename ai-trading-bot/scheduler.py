"""
scheduler.py — APScheduler ile Görev Zamanlayıcı
15 dakikada veri, saatlik sinyal, sabah/akşam brief.
"""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

import yaml
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

# Logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("storage/bot.log", mode="a", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)


def load_config(path: str = "config/config.yaml") -> dict:
    """YAML konfigürasyon dosyasını yükler."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_tickers(path: str = "config/tickers.yaml") -> dict:
    """Ticker listesini yükler."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ── Görev Fonksiyonları ──────────────────────────────────────

async def task_update_data(bot) -> None:
    """Her 15 dakikada çalışır: veri + haber güncelle."""
    logger.info("⏱ Görev: Veri güncelleme başladı...")
    try:
        await bot.update_market_data()
        logger.info("✅ Veri güncelleme tamamlandı.")
    except Exception as e:
        logger.error(f"❌ Veri güncelleme hatası: {e}")
        await bot.send_error("Veri güncelleme hatası", str(e))


async def task_generate_signals(bot) -> None:
    """Her saat çalışır: sinyal üret + koşullu işlem yap."""
    logger.info("⏱ Görev: Sinyal üretimi başladı...")
    try:
        await bot.generate_and_execute_signals()
        logger.info("✅ Sinyal üretimi tamamlandı.")
    except Exception as e:
        logger.error(f"❌ Sinyal üretimi hatası: {e}")
        await bot.send_error("Sinyal üretim hatası", str(e))


async def task_morning_brief(bot) -> None:
    """Sabah 09:30 ET'de çalışır: piyasa açılış özeti."""
    logger.info("🌅 Görev: Sabah brifing...")
    try:
        await bot.send_morning_brief()
    except Exception as e:
        logger.error(f"❌ Sabah brifing hatası: {e}")


async def task_closing_summary(bot) -> None:
    """16:00 ET'de çalışır: kapanış özeti."""
    logger.info("🌆 Görev: Kapanış özeti...")
    try:
        await bot.send_closing_summary()
    except Exception as e:
        logger.error(f"❌ Kapanış özeti hatası: {e}")


async def task_daily_reset(bot) -> None:
    """Gece yarısı: günlük istatistikleri sıfırla."""
    logger.info("🔄 Günlük reset yapılıyor...")
    try:
        await bot.daily_reset()
    except Exception as e:
        logger.error(f"❌ Günlük reset hatası: {e}")


# ── Scheduler Başlatma ───────────────────────────────────────

def create_scheduler(bot, config: dict) -> AsyncIOScheduler:
    """APScheduler'ı yapılandırır ve döndürür."""
    sched_cfg = config.get("scheduler", {})
    timezone = sched_cfg.get("timezone", "America/New_York")

    scheduler = AsyncIOScheduler(timezone=timezone)

    # 15 dakikada bir veri güncelle
    scheduler.add_job(
        func=lambda: asyncio.ensure_future(task_update_data(bot)),
        trigger=IntervalTrigger(minutes=sched_cfg.get("data_update_interval_minutes", 15)),
        id="update_data",
        name="Veri Güncelleme",
        replace_existing=True,
    )

    # Saatlik sinyal üretimi
    scheduler.add_job(
        func=lambda: asyncio.ensure_future(task_generate_signals(bot)),
        trigger=IntervalTrigger(hours=sched_cfg.get("signal_interval_hours", 1)),
        id="generate_signals",
        name="Sinyal Üretimi",
        replace_existing=True,
    )

    # Sabah brifing (09:30 ET)
    scheduler.add_job(
        func=lambda: asyncio.ensure_future(task_morning_brief(bot)),
        trigger=CronTrigger(
            hour=sched_cfg.get("morning_brief_hour", 9),
            minute=sched_cfg.get("morning_brief_minute", 30),
            timezone=timezone,
        ),
        id="morning_brief",
        name="Sabah Özeti",
        replace_existing=True,
    )

    # Kapanış özeti (16:00 ET)
    scheduler.add_job(
        func=lambda: asyncio.ensure_future(task_closing_summary(bot)),
        trigger=CronTrigger(
            hour=sched_cfg.get("closing_summary_hour", 16),
            minute=sched_cfg.get("closing_summary_minute", 0),
            timezone=timezone,
        ),
        id="closing_summary",
        name="Kapanış Özeti",
        replace_existing=True,
    )

    # Gece yarısı reset (00:05 ET)
    scheduler.add_job(
        func=lambda: asyncio.ensure_future(task_daily_reset(bot)),
        trigger=CronTrigger(hour=0, minute=5, timezone=timezone),
        id="daily_reset",
        name="Günlük Reset",
        replace_existing=True,
    )

    logger.info(
        f"Scheduler hazır: {len(scheduler.get_jobs())} görev tanımlandı "
        f"(timezone={timezone})"
    )
    return scheduler


async def run_scheduler(bot, config: dict) -> None:
    """Scheduler'ı başlatır ve çalıştırır."""
    scheduler = create_scheduler(bot, config)
    scheduler.start()
    logger.info("🚀 Scheduler başlatıldı. Çalışıyor...")

    # İlk çalıştırma: hemen veri güncelle ve sinyal üret
    logger.info("📡 İlk veri güncelleme tetikleniyor...")
    await task_update_data(bot)

    try:
        while True:
            await asyncio.sleep(60)
    except (KeyboardInterrupt, SystemExit):
        logger.info("🛑 Scheduler durduruluyor...")
        scheduler.shutdown()


if __name__ == "__main__":
    # Standalone test
    async def _standalone_test():
        config = load_config()
        logger.info(f"Config yüklendi. Paper mode: {config['trading']['paper_mode']}")
        logger.info("Scheduler test için ana bot gerekli. main.py kullanın.")
    asyncio.run(_standalone_test())
