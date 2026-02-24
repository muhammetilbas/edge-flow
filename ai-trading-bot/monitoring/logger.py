"""
monitoring/logger.py — SQLite Trade Loglama
Tüm trade, sinyal ve sistem olaylarını SQLite'a kaydeder.
"""

from __future__ import annotations

import logging
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = "storage/trades.db"


class TradeLogger:
    """SQLite üzerinde trade ve sinyal geçmişi yönetir."""

    CREATE_TRADES_SQL = """
    CREATE TABLE IF NOT EXISTS trades (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker      TEXT    NOT NULL,
        direction   TEXT    NOT NULL,
        asset_type  TEXT    DEFAULT 'stock',
        shares      REAL    NOT NULL,
        entry_price REAL    NOT NULL,
        exit_price  REAL,
        stop_loss   REAL,
        take_profit REAL,
        pnl         REAL,
        pnl_pct     REAL,
        confidence  REAL,
        status      TEXT    DEFAULT 'OPEN',
        reason      TEXT,
        opened_at   TEXT    NOT NULL,
        closed_at   TEXT,
        order_id    TEXT,
        paper_mode  INTEGER DEFAULT 1
    )
    """

    CREATE_SIGNALS_SQL = """
    CREATE TABLE IF NOT EXISTS signals (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker      TEXT    NOT NULL,
        direction   TEXT    NOT NULL,
        confidence  REAL,
        prob_buy    REAL,
        prob_sell   REAL,
        prob_hold   REAL,
        sentiment   REAL,
        vix         REAL,
        market_regime TEXT,
        executed    INTEGER DEFAULT 0,
        timestamp   TEXT    NOT NULL
    )
    """

    CREATE_EVENTS_SQL = """
    CREATE TABLE IF NOT EXISTS system_events (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        event_type  TEXT NOT NULL,
        message     TEXT,
        level       TEXT DEFAULT 'INFO',
        timestamp   TEXT NOT NULL
    )
    """

    def __init__(self, db_path: str = DEFAULT_DB_PATH):
        self._db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
        logger.info(f"TradeLogger başlatıldı: {db_path}")

    def _init_db(self) -> None:
        """Tabloları oluşturur."""
        with self._conn() as conn:
            conn.execute(self.CREATE_TRADES_SQL)
            conn.execute(self.CREATE_SIGNALS_SQL)
            conn.execute(self.CREATE_EVENTS_SQL)
            conn.commit()

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    # ── Trade Loglama ────────────────────────────────────────

    def log_trade_open(self, trade: dict) -> int:
        """Açılan trade'i kaydeder, ID döndürür."""
        sql = """
        INSERT INTO trades
            (ticker, direction, asset_type, shares, entry_price,
             stop_loss, take_profit, confidence, status, opened_at,
             order_id, paper_mode)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'OPEN', ?, ?, ?)
        """
        with self._conn() as conn:
            cursor = conn.execute(sql, (
                trade.get("ticker"),
                trade.get("direction"),
                trade.get("asset_type", "stock"),
                trade.get("shares", 0),
                trade.get("entry_price", 0),
                trade.get("stop_loss"),
                trade.get("take_profit"),
                trade.get("confidence"),
                trade.get("opened_at", datetime.now().isoformat()),
                trade.get("order_id"),
                int(trade.get("paper_mode", True)),
            ))
            conn.commit()
            return cursor.lastrowid

    def log_trade_close(self, trade_id: int, exit_price: float, pnl: float, reason: str) -> None:
        """Kapatılan trade'i günceller."""
        pnl_pct = None  # Dışarıdan hesaplanabilir
        sql = """
        UPDATE trades
        SET exit_price=?, pnl=?, status=?, reason=?, closed_at=?
        WHERE id=?
        """
        with self._conn() as conn:
            conn.execute(sql, (
                exit_price,
                pnl,
                f"CLOSED_{reason}",
                reason,
                datetime.now().isoformat(),
                trade_id,
            ))
            conn.commit()

    # ── Sinyal Loglama ───────────────────────────────────────

    def log_signal(self, signal: dict, executed: bool = False) -> None:
        """Üretilen sinyali kaydeder."""
        sql = """
        INSERT INTO signals
            (ticker, direction, confidence, prob_buy, prob_sell, prob_hold,
             sentiment, vix, market_regime, executed, timestamp)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        with self._conn() as conn:
            conn.execute(sql, (
                signal.get("ticker"),
                signal.get("direction"),
                signal.get("confidence"),
                signal.get("prob_buy"),
                signal.get("prob_sell"),
                signal.get("prob_hold"),
                signal.get("sentiment_score"),
                signal.get("vix"),
                signal.get("market_regime"),
                int(executed),
                signal.get("timestamp", datetime.now().isoformat()),
            ))
            conn.commit()

    # ── Sistem Olayları ──────────────────────────────────────

    def log_event(self, event_type: str, message: str, level: str = "INFO") -> None:
        """Sistem olayını kaydeder."""
        sql = "INSERT INTO system_events (event_type, message, level, timestamp) VALUES (?, ?, ?, ?)"
        with self._conn() as conn:
            conn.execute(sql, (event_type, message, level, datetime.now().isoformat()))
            conn.commit()

    # ── Sorgular ─────────────────────────────────────────────

    def get_trades(self, limit: int = 100, status: str = None) -> list[dict]:
        """Trade geçmişini döndürür."""
        sql = "SELECT * FROM trades"
        params = []
        if status:
            sql += " WHERE status LIKE ?"
            params.append(f"%{status}%")
        sql += " ORDER BY opened_at DESC LIMIT ?"
        params.append(limit)
        with self._conn() as conn:
            rows = conn.execute(sql, params).fetchall()
            return [dict(r) for r in rows]

    def get_performance_summary(self) -> dict:
        """Genel performans özeti döndürür."""
        sql = """
        SELECT
            COUNT(*) as total_trades,
            SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as win_trades,
            SUM(pnl) as total_pnl,
            AVG(pnl) as avg_pnl,
            MAX(pnl) as best_trade,
            MIN(pnl) as worst_trade
        FROM trades
        WHERE status LIKE 'CLOSED%'
        """
        with self._conn() as conn:
            row = conn.execute(sql).fetchone()
            if row and row["total_trades"]:
                data = dict(row)
                data["win_rate"] = (
                    data["win_trades"] / data["total_trades"] * 100
                    if data["total_trades"] > 0 else 0
                )
                return data
            return {}
