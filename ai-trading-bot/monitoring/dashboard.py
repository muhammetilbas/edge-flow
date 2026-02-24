"""
monitoring/dashboard.py — Streamlit İzleme Paneli + Otomasyon
Sinyal üret, parametreleri ayarla, otomatik Alpaca trade aç.
"""

from __future__ import annotations

import sys
import sqlite3
import time
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import yaml
import yfinance as yf
import os
from dotenv import load_dotenv

load_dotenv()

# Proje kökünü path'e ekle
sys.path.insert(0, str(Path(__file__).parent.parent))

from features.technical import TechnicalFeatureGenerator
from models.predictor import SignalPredictor
from risk.risk_engine import RiskEngine
from monitoring.logger import TradeLogger
from notifications.telegram_bot import TelegramNotifier
from notifications.formatters import format_order_executed, format_signal_summary

# ── Sayfa Konfigürasyonu ────────────────────────────────────
st.set_page_config(
    page_title="🤖 AI Trading Bot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

DB_PATH = "storage/trades.db"
CONFIG_PATH = "config/config.yaml"
TICKERS_PATH = "config/tickers.yaml"

# ── Stiller ─────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header { font-size: 2.2rem; font-weight: 700; color: #00d26a; margin-bottom: 0.2rem; }
    .sub-header { font-size: 1rem; color: #888; margin-bottom: 1.5rem; }
    div[data-testid="metric-container"] {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid #2d2d44;
        border-radius: 12px;
        padding: 0.7rem 1rem;
    }
    .signal-buy { background: #00d26a22; border-left: 4px solid #00d26a; padding: 0.8rem; border-radius: 8px; margin: 0.5rem 0; }
    .signal-sell { background: #ff4b5c22; border-left: 4px solid #ff4b5c; padding: 0.8rem; border-radius: 8px; margin: 0.5rem 0; }
    .signal-hold { background: #88888822; border-left: 4px solid #888; padding: 0.8rem; border-radius: 8px; margin: 0.5rem 0; }
    .exec-ok { background: #00d26a22; border: 1px solid #00d26a; padding: 0.5rem; border-radius: 8px; margin: 0.3rem 0; }
    .exec-fail { background: #ff4b5c22; border: 1px solid #ff4b5c; padding: 0.5rem; border-radius: 8px; margin: 0.3rem 0; }
    .stButton > button { border-radius: 8px; font-weight: 600; }
</style>
""", unsafe_allow_html=True)


# ── Yardımcı Fonksiyonlar ───────────────────────────────────

def load_config():
    if Path(CONFIG_PATH).exists():
        with open(CONFIG_PATH) as f:
            return yaml.safe_load(f)
    return {}

def load_tickers():
    if Path(TICKERS_PATH).exists():
        with open(TICKERS_PATH) as f:
            data = yaml.safe_load(f)
        stocks = [t["symbol"] for t in data.get("stocks", [])
                  if t.get("active", True) and not t.get("reference_only")]
        return sorted(stocks)
    return ["AAPL", "MSFT", "NVDA", "TSLA"]

@st.cache_data(ttl=30)
def load_trades(db_path: str = DB_PATH) -> pd.DataFrame:
    if not Path(db_path).exists():
        return pd.DataFrame()
    try:
        conn = sqlite3.connect(db_path)
        df = pd.read_sql("SELECT * FROM trades ORDER BY opened_at DESC LIMIT 500", conn)
        conn.close()
        return df
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=30)
def load_signals(db_path: str = DB_PATH) -> pd.DataFrame:
    if not Path(db_path).exists():
        return pd.DataFrame()
    try:
        conn = sqlite3.connect(db_path)
        df = pd.read_sql("SELECT * FROM signals ORDER BY timestamp DESC LIMIT 200", conn)
        conn.close()
        return df
    except Exception:
        return pd.DataFrame()

def compute_metrics(trades_df: pd.DataFrame) -> dict:
    defaults = {"total_pnl": 0.0, "daily_pnl": 0.0, "total_trades": 0,
                "win_rate": 0.0, "sharpe": 0.0, "max_dd": 0.0}
    if trades_df.empty or "status" not in trades_df.columns:
        return defaults
    closed = trades_df[trades_df["status"].str.startswith("CLOSED", na=False)]
    if closed.empty:
        return defaults
    total_pnl = float(closed["pnl"].sum()) if "pnl" in closed else 0.0
    wins = closed[closed["pnl"] > 0] if "pnl" in closed else pd.DataFrame()
    win_rate = len(wins) / len(closed) * 100 if len(closed) > 0 else 0.0
    pnls = closed["pnl"].dropna().values
    equity = 10000 + pnls.cumsum()
    returns = pd.Series(equity).pct_change().dropna()
    sharpe = float(returns.mean() / returns.std() * (252**0.5)) if len(returns) > 1 and returns.std() > 0 else 0.0
    peak = pd.Series(equity).cummax()
    drawdown = (pd.Series(equity) - peak) / peak
    max_dd = float(drawdown.min() * 100) if len(drawdown) > 0 else 0.0
    return {
        "total_pnl": round(total_pnl, 2),
        "daily_pnl": round(float(closed.tail(10)["pnl"].sum()), 2),
        "total_trades": len(closed),
        "win_rate": round(win_rate, 1),
        "sharpe": round(sharpe, 2),
        "max_dd": round(max_dd, 2),
    }


def _send_telegram(message: str):
    """Telegram bildirim gönderir (requests ile direkt API çağrısı)."""
    try:
        import requests
        token = os.getenv("TELEGRAM_TOKEN", "")
        chat_id = os.getenv("TELEGRAM_CHAT_ID", "")
        if not token or token == "YOUR_TELEGRAM_BOT_TOKEN" or not chat_id:
            return
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        payload = {
            "chat_id": chat_id,
            "text": message,
            "parse_mode": "Markdown",
        }
        requests.post(url, json=payload, timeout=10)
    except Exception:
        pass


def fetch_stock_data(ticker: str, days: int = 90) -> pd.DataFrame:
    """Alpaca veya yfinance'tan veri çeker."""
    config = load_config()
    alpaca_cfg = config.get("alpaca", {})
    api_key = alpaca_cfg.get("api_key", "")

    df = None
    if api_key and "YOUR_" not in api_key:
        try:
            from data.collector import StockDataCollector
            collector = StockDataCollector(api_key, alpaca_cfg.get("secret_key", ""), paper=True)
            df = collector.get_ohlcv(ticker, days=days)
        except Exception:
            df = None

    if df is None or df.empty:
        t = yf.Ticker(ticker)
        period = f"{days}d" if days <= 365 else f"{max(1, days // 365)}y"
        data = t.history(period=period, auto_adjust=True)
        if hasattr(data.columns, 'levels'):
            data.columns = [c[0] if isinstance(c, tuple) else c for c in data.columns]
        data.columns = [c.lower() for c in data.columns]
        data = data.reset_index()
        data.rename(columns={"Date": "timestamp", "date": "timestamp"}, inplace=True)
        needed = ["timestamp", "open", "high", "low", "close", "volume"]
        df = data[[c for c in needed if c in data.columns]].copy()

    # Sütun kontrolü
    required = {"open", "high", "low", "close", "volume"}
    if not required.issubset(set(df.columns)):
        raise ValueError(f"{ticker}: OHLCV sütunları eksik")
    return df


def execute_alpaca_order(ticker: str, side: str, qty: int, stop_loss: float, take_profit: float) -> dict:
    """Alpaca Paper API üzerinden emir gönderir."""
    config = load_config()
    alpaca_cfg = config.get("alpaca", {})
    api_key = alpaca_cfg.get("api_key", "")
    secret_key = alpaca_cfg.get("secret_key", "")

    if not api_key or "YOUR_" in api_key:
        return {"status": "error", "message": "Alpaca API key ayarlanmamış"}

    try:
        from alpaca.trading.client import TradingClient
        from alpaca.trading.requests import MarketOrderRequest, OrderSide, TimeInForce

        client = TradingClient(api_key, secret_key, paper=True)

        order_side = OrderSide.BUY if side == "BUY" else OrderSide.SELL

        order_data = MarketOrderRequest(
            symbol=ticker,
            qty=qty,
            side=order_side,
            time_in_force=TimeInForce.DAY,
        )

        order = client.submit_order(order_data)

        return {
            "status": "success",
            "order_id": str(order.id),
            "symbol": ticker,
            "side": side,
            "qty": qty,
            "message": f"✅ {side} {qty}x {ticker} emri gönderildi!",
        }
    except Exception as e:
        return {"status": "error", "message": f"❌ Emir hatası: {str(e)}"}


def generate_signal_for_ticker(ticker: str, min_confidence: float) -> dict:
    """Tek bir hisse için sinyal üretir."""
    try:
        df = fetch_stock_data(ticker, days=90)
    except Exception as e:
        return {"ticker": ticker, "direction": "HOLD", "confidence": 0, "reason": f"Veri hatası: {e}"}

    try:
        fg = TechnicalFeatureGenerator()
        df_feat = fg.generate_features(df)
    except Exception as e:
        return {"ticker": ticker, "direction": "HOLD", "confidence": 0, "reason": f"Feature hatası: {e}"}

    if df_feat.empty or len(df_feat) < 10:
        return {"ticker": ticker, "direction": "HOLD", "confidence": 0, "reason": "Yetersiz veri"}

    predictor = SignalPredictor()
    loaded = predictor.load_model(ticker)

    if not loaded:
        return {"ticker": ticker, "direction": "HOLD", "confidence": 0, "reason": "Model eğitilmemiş"}

    signal = predictor.generate_signal(
        ticker=ticker, df=df_feat, sentiment={"score": 0.0}, vix=0.0, spy_correlation=0.0,
    )

    current_price = float(df_feat["close"].iloc[-1])
    atr_value = float(df_feat["atr"].iloc[-1]) if "atr" in df_feat.columns else current_price * 0.02

    signal["current_price"] = current_price
    signal["atr"] = atr_value

    # Teknik göstergeler
    last = df_feat.iloc[-1]
    signal["technicals"] = {
        "RSI(14)": round(float(last.get("rsi_14", 0)), 1),
        "RSI(7)": round(float(last.get("rsi_7", 0)), 1),
        "ATR%": round(float(last.get("atr_pct", 0)) * 100, 2),
        "EMA Trend": round(float(last.get("ema_trend", 0)), 2),
        "MACD Hist": round(float(last.get("macd_hist", 0)), 4),
        "BB Pos": round(float(last.get("bb_position", 0)), 2),
        "Vol Spike": round(float(last.get("volume_spike", 0)), 2),
    }

    try:
        logger = TradeLogger(DB_PATH)
        logger.log_signal(signal, executed=False)
    except Exception:
        pass

    return signal


# ── SIDEBAR (Trading Parametreleri) ─────────────────────────

def render_sidebar():
    with st.sidebar:
        st.header("⚙️ Trading Ayarları")

        st.markdown("### 💰 Portföy")
        portfolio_value = st.number_input(
            "Portföy Değeri ($)", min_value=100.0, max_value=1_000_000.0,
            value=10_000.0, step=1000.0, key="portfolio_value"
        )

        st.markdown("### 📊 Risk Yönetimi")
        risk_pct = st.slider(
            "Trade Başına Risk (%)", min_value=0.5, max_value=10.0,
            value=2.0, step=0.5, key="risk_pct"
        )

        max_position_pct = st.slider(
            "Max Pozisyon Büyüklüğü (%)", min_value=5.0, max_value=50.0,
            value=20.0, step=5.0, key="max_pos_pct"
        )

        min_confidence = st.slider(
            "Min Güven Skoru (%)", min_value=30, max_value=90,
            value=55, step=5, key="min_confidence"
        )

        st.markdown("### 🤖 Otomasyon")
        auto_execute = st.toggle(
            "Otomatik Alım/Satım", value=False, key="auto_execute",
            help="Açıkken AL/SAT sinyali gelince otomatik Alpaca emri gönderir"
        )

        if auto_execute:
            st.warning("⚠️ Sinyaller otomatik olarak Alpaca Paper Trading'de işlenecek!")

        st.divider()

        # Eğitilmiş modeller
        st.markdown("### 🧠 Eğitilmiş Modeller")
        model_dir = Path("models/saved")
        if model_dir.exists():
            models = sorted(model_dir.glob("*.pkl"))
            if models:
                for m in models:
                    st.write(f"✅ **{m.stem.split('_')[0]}**")
            else:
                st.write("❌ Model yok — önce eğitin")
        else:
            st.write("❌ Model dizini yok")

        st.divider()
        st.caption("AI Trading Bot v1.0 | Paper Mode")
        st.caption(f"Güncelleme: {datetime.now().strftime('%H:%M:%S')}")

    return {
        "portfolio_value": portfolio_value,
        "risk_pct": risk_pct / 100,
        "max_position_pct": max_position_pct / 100,
        "min_confidence": min_confidence / 100,
        "auto_execute": auto_execute,
    }


# ── ANA DASHBOARD ───────────────────────────────────────────

def render_dashboard():
    st.markdown('<p class="main-header">🤖 AI Trading Bot Dashboard</p>', unsafe_allow_html=True)
    st.markdown(f'<p class="sub-header">Alpaca Paper Trading | {datetime.now().strftime("%d.%m.%Y %H:%M")}</p>', unsafe_allow_html=True)

    params = render_sidebar()

    trades_df = load_trades()
    signals_df = load_signals()
    metrics = compute_metrics(trades_df)

    # ── Metrik Kartları ──────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("💰 Toplam P&L", f"${metrics['total_pnl']:,.2f}",
                delta=f"${metrics['daily_pnl']:,.2f}")
    col2.metric("🎯 Win Rate", f"%{metrics['win_rate']:.1f}")
    col3.metric("📐 Sharpe", f"{metrics['sharpe']:.2f}")
    col4.metric("📉 Max DD", f"%{metrics['max_dd']:.2f}")

    st.divider()

    # ── TAB YAPISI ───────────────────────────────────────────
    tab1, tab2, tab3 = st.tabs(["🚀 Sinyal Üret & İşle", "📋 İşlem Geçmişi", "📈 Performans"])

    # ═══════════════════════════════════════════════════════
    # TAB 1: Sinyal Üret + Otomatik İşlem
    # ═══════════════════════════════════════════════════════
    with tab1:
        st.subheader("🚀 Sinyal Üretimi ve Otomasyon")

        all_tickers = load_tickers()

        # "Hepsini Seç" flag'ini widget'tan ÖNCE ayarla
        if "select_all" not in st.session_state:
            st.session_state["select_all"] = False

        c1, c2, c3 = st.columns([4, 1, 1])
        with c2:
            st.write("")
            st.write("")
            select_all = st.button("📋 Hepsini Seç")
            if select_all:
                st.session_state["select_all"] = not st.session_state["select_all"]
                st.rerun()
        with c1:
            default_tickers = all_tickers if st.session_state["select_all"] else ["AAPL", "NVDA", "TSLA", "MSFT"]
            selected = st.multiselect(
                "Hisse seç",
                all_tickers,
                default=default_tickers,
            )
        with c3:
            st.write("")
            st.write("")
            run_signals = st.button("⚡ SİNYAL ÜRET", type="primary")

        # Parametre özeti
        auto_label = "🟢 OTOMATİK AÇIK" if params["auto_execute"] else "🔴 OTOMATİK KAPALI"
        st.caption(
            f"Portföy: **${params['portfolio_value']:,.0f}** | "
            f"Risk: **%{params['risk_pct']*100:.1f}** | "
            f"Max Pozisyon: **%{params['max_position_pct']*100:.0f}** | "
            f"Min Güven: **%{params['min_confidence']*100:.0f}** | "
            f"{auto_label}"
        )

        if run_signals and selected:
            st.divider()
            progress = st.progress(0, text="Başlatılıyor...")

            risk_engine = RiskEngine(
                portfolio_value=params["portfolio_value"],
                max_risk_per_trade=params["risk_pct"],
                max_open_positions=10,
            )

            buy_signals = []
            sell_signals = []
            hold_signals = []
            executed_orders = []

            for i, ticker in enumerate(selected):
                progress.progress(
                    (i + 1) / len(selected),
                    text=f"📡 {ticker} analiz ediliyor... ({i+1}/{len(selected)})"
                )

                signal = generate_signal_for_ticker(ticker, params["min_confidence"])
                direction = signal.get("direction", "HOLD")
                confidence = signal.get("confidence", 0)
                price = signal.get("current_price", 0)
                atr = signal.get("atr", price * 0.02)
                reason = signal.get("reason", "")

                # Pozisyon hesapla
                max_shares_by_pct = int((params["portfolio_value"] * params["max_position_pct"]) / price) if price > 0 else 0
                position = risk_engine.calculate_position(signal, price, atr) if direction != "HOLD" else {}
                shares = min(position.get("shares", 0), max_shares_by_pct) if position else 0

                signal["calc_shares"] = shares
                signal["position"] = position

                if direction == "BUY" and confidence >= params["min_confidence"]:
                    buy_signals.append(signal)
                elif direction == "SELL" and confidence >= params["min_confidence"]:
                    sell_signals.append(signal)
                else:
                    hold_signals.append(signal)

                # Otomatik alım/satım
                if params["auto_execute"] and direction in ("BUY", "SELL") and confidence >= params["min_confidence"] and shares > 0:
                    order_result = execute_alpaca_order(
                        ticker=ticker,
                        side=direction,
                        qty=shares,
                        stop_loss=position.get("stop_loss", 0),
                        take_profit=position.get("take_profit", 0),
                    )
                    executed_orders.append({"ticker": ticker, "price": price, "confidence": confidence, **order_result})

                    # Telegram bildirimi — her emir için
                    if order_result["status"] == "success":
                        _send_telegram(format_order_executed(
                            ticker=ticker, side=direction, qty=shares,
                            price=price, order_id=order_result.get("order_id", "N/A"),
                            confidence=confidence,
                            stop_loss=position.get("stop_loss", 0),
                            take_profit=position.get("take_profit", 0),
                        ))

            progress.empty()

            # ── Sonuçları Göster ─────────────────────────────
            if executed_orders:
                st.markdown("### ⚡ Otomatik Emirler")
                for order in executed_orders:
                    if order["status"] == "success":
                        st.markdown(f'<div class="exec-ok">✅ <strong>{order["ticker"]}</strong>: {order["message"]}</div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="exec-fail">❌ <strong>{order["ticker"]}</strong>: {order["message"]}</div>', unsafe_allow_html=True)
                st.divider()

            # Telegram sinyal özeti
            ok_orders = sum(1 for o in executed_orders if o.get("status") == "success")
            _send_telegram(format_signal_summary(
                buy_signals=buy_signals,
                sell_signals=sell_signals,
                hold_count=len(hold_signals),
                executed_count=ok_orders,
            ))

            # AL sinyalleri
            if buy_signals:
                st.markdown(f"### 🟢 AL Sinyalleri ({len(buy_signals)})")
                for sig in sorted(buy_signals, key=lambda x: x.get("confidence", 0), reverse=True):
                    _render_signal_card(sig, params)

            # SAT sinyalleri
            if sell_signals:
                st.markdown(f"### 🔴 SAT Sinyalleri ({len(sell_signals)})")
                for sig in sorted(sell_signals, key=lambda x: x.get("confidence", 0), reverse=True):
                    _render_signal_card(sig, params)

            # HOLD sinyalleri
            if hold_signals:
                with st.expander(f"⚪ BEKLE Sinyalleri ({len(hold_signals)})"):
                    for sig in hold_signals:
                        ticker = sig.get("ticker", "?")
                        conf = sig.get("confidence", 0)
                        price = sig.get("current_price", 0)
                        reason = sig.get("reason", "")
                        label = f"**{ticker}** — ${price:.2f}"
                        if reason:
                            label += f"  |  _{reason}_"
                        else:
                            label += f"  |  Güven: %{conf*100:.0f}"
                        st.write(label)

            st.divider()
            summary = f"✅ **{len(selected)}** hisse analiz edildi: "
            summary += f"🟢 {len(buy_signals)} AL, 🔴 {len(sell_signals)} SAT, ⚪ {len(hold_signals)} BEKLE"
            if executed_orders:
                ok = sum(1 for o in executed_orders if o["status"] == "success")
                summary += f" | ⚡ {ok} emir gönderildi"
            st.success(summary)

        elif run_signals and not selected:
            st.warning("⚠️ En az bir hisse seçin!")

    # ═══════════════════════════════════════════════════════
    # TAB 2: İşlem Geçmişi
    # ═══════════════════════════════════════════════════════
    with tab2:
        left_col, right_col = st.columns([2, 1])
        with left_col:
            st.subheader("📋 Son İşlemler")
            if not trades_df.empty:
                display_cols = [c for c in
                    ["ticker", "direction", "shares", "entry_price", "exit_price", "pnl", "status", "opened_at"]
                    if c in trades_df.columns]
                st.dataframe(trades_df[display_cols].head(30), width='stretch', hide_index=True)
            else:
                st.info("Henüz trade yok. '🚀 Sinyal Üret' sekmesinden otomasyonu aç.")
        with right_col:
            st.subheader("📡 Son Sinyaller")
            if not signals_df.empty:
                display_sig = [c for c in ["ticker", "direction", "confidence", "timestamp"]
                    if c in signals_df.columns]
                st.dataframe(signals_df[display_sig].head(20), width='stretch', hide_index=True)
            else:
                st.info("Sinyal yok.")

    # ═══════════════════════════════════════════════════════
    # TAB 3: Performans
    # ═══════════════════════════════════════════════════════
    with tab3:
        if not trades_df.empty and "pnl" in trades_df.columns:
            closed_df = trades_df[trades_df["status"].str.startswith("CLOSED", na=False)].copy()
            if not closed_df.empty:
                closed_df = closed_df.sort_values("closed_at")
                closed_df["equity"] = 10000 + closed_df["pnl"].fillna(0).cumsum()
                fig = px.area(closed_df, x="closed_at", y="equity",
                    title="📈 Equity Curve", template="plotly_dark",
                    color_discrete_sequence=["#00d26a"])
                fig.update_layout(paper_bgcolor="#0e1117", plot_bgcolor="#0e1117", showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Equity curve — trade'ler kapandıkça oluşur.")
        else:
            st.info("Performans verisi trade'ler oluştukça görünecek.")


def _render_signal_card(sig: dict, params: dict):
    """Tek bir sinyal kartı render eder."""
    direction = sig.get("direction", "HOLD")
    ticker = sig.get("ticker", "?")
    confidence = sig.get("confidence", 0)
    price = sig.get("current_price", 0)
    position = sig.get("position", {})
    technicals = sig.get("technicals", {})
    shares = sig.get("calc_shares", 0)

    emoji = "🟢" if direction == "BUY" else "🔴"
    css = "signal-buy" if direction == "BUY" else "signal-sell"
    dir_tr = "AL" if direction == "BUY" else "SAT"
    cost = shares * price

    st.markdown(f"""
    <div class="{css}">
        <strong>{emoji} {ticker}</strong> — <strong>{dir_tr}</strong> |
        Güven: <strong>%{confidence * 100:.0f}</strong> |
        Fiyat: <strong>${price:.2f}</strong> |
        Adet: <strong>{shares}</strong> |
        Maliyet: <strong>${cost:,.2f}</strong>
    </div>
    """, unsafe_allow_html=True)

    if position and position.get("action") != "SKIP":
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("📦 Adet", f"{shares}")
        c2.metric("🛑 Stop Loss", f"${position.get('stop_loss', 0):.2f}")
        c3.metric("🎯 Take Profit", f"${position.get('take_profit', 0):.2f}")
        c4.metric("💵 Maliyet", f"${cost:,.2f}")

    if technicals:
        with st.expander(f"📊 {ticker} Teknik Göstergeler"):
            cols = st.columns(4)
            for j, (key, val) in enumerate(technicals.items()):
                cols[j % 4].metric(key, val)


# ── Çalıştır ─────────────────────────────────────────────────
render_dashboard()
