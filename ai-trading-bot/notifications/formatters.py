"""
notifications/formatters.py — Telegram Mesaj Formatlama
Sinyal ve özet mesajlarını Telegram Markdown formatına çevirir.
"""

from __future__ import annotations

from datetime import datetime


def format_signal_message(
    signal: dict, position: dict, explanation: str
) -> str:
    """Trade sinyali için Telegram Markdown mesajı oluşturur."""
    direction = signal.get("direction", "HOLD")
    ticker = signal.get("ticker", "N/A")
    confidence = signal.get("confidence", 0.0)
    prob_buy = signal.get("prob_buy", 0.0)
    prob_sell = signal.get("prob_sell", 0.0)
    timestamp = signal.get("timestamp", datetime.now().isoformat())[:16]

    direction_emoji = {"BUY": "🟢", "SELL": "🔴", "HOLD": "⚪️"}.get(direction, "⚪️")
    direction_tr = {"BUY": "AL", "SELL": "SAT", "HOLD": "BEKLE"}.get(direction, direction)

    entry = position.get("entry", 0)
    stop_loss = position.get("stop_loss", 0)
    take_profit = position.get("take_profit", 0)
    shares = position.get("shares", 0)
    risk_reward = position.get("risk_reward", "1:2")
    risk_amount = position.get("risk_amount", 0)

    msg = (
        f"{direction_emoji} *{ticker}* — {direction_tr}\n\n"
        f"📊 *Analiz*\n"
        f"• Güven: %{confidence * 100:.0f}\n"
        f"• Yükseliş İhtimali: %{prob_buy * 100:.0f}\n"
        f"• Düşüş İhtimali: %{prob_sell * 100:.0f}\n\n"
        f"💰 *Pozisyon*\n"
        f"• Adet: {shares}\n"
        f"• Giriş: ${entry:.2f}\n"
        f"• Stop Loss: ${stop_loss:.2f}\n"
        f"• Take Profit: ${take_profit:.2f}\n"
        f"• Risk/Ödül: {risk_reward}\n"
        f"• Risk: ${risk_amount:.2f}\n\n"
        f"🧠 *AI Açıklaması*\n"
        f"{explanation}\n\n"
        f"⏰ {timestamp}"
    )
    return msg


def format_daily_brief_message(brief: str, pnl: float, brief_type: str = "morning") -> str:
    """Günlük özet mesajı formatlar."""
    emoji = "📈" if pnl > 0 else "📉"
    session = "🌅 Sabah Özeti" if brief_type == "morning" else "🌆 Kapanış Özeti"
    date_str = datetime.now().strftime("%d.%m.%Y")

    return (
        f"{emoji} *{session} — {date_str}*\n\n"
        f"{brief}\n\n"
        f"💼 Günlük P&L: *%{pnl:.2f}*"
    )


def format_portfolio_summary(portfolio_summary: dict) -> str:
    """Portföy özet mesajı formatlar."""
    total_pnl = portfolio_summary.get("total_pnl", 0)
    daily_pnl = portfolio_summary.get("daily_pnl", 0)
    equity = portfolio_summary.get("current_equity", 0)
    open_pos = portfolio_summary.get("open_positions", 0)
    win_rate = portfolio_summary.get("win_rate", 0)
    sharpe = portfolio_summary.get("sharpe_ratio", 0)
    max_dd = portfolio_summary.get("max_drawdown", 0)

    pnl_emoji = "📈" if total_pnl > 0 else "📉"

    return (
        f"📊 *Portföy Durumu*\n\n"
        f"{pnl_emoji} Toplam P&L: *%{total_pnl:.2f}*\n"
        f"📅 Günlük P&L: %{daily_pnl:.2f}\n"
        f"💵 Portföy: ${equity:,.2f}\n"
        f"📂 Açık Pozisyon: {open_pos}\n"
        f"🎯 Win Rate: %{win_rate:.1f}\n"
        f"📐 Sharpe: {sharpe:.2f}\n"
        f"📉 Max Drawdown: %{abs(max_dd):.2f}"
    )


def format_error_alert(error: str, context: str = "") -> str:
    """Hata uyarısı mesajı formatlar."""
    timestamp = datetime.now().strftime("%d.%m %H:%M")
    return (
        f"🚨 *HATA UYARISI*\n\n"
        f"⏰ {timestamp}\n"
        f"📍 Konum: {context or 'Bilinmiyor'}\n"
        f"❗ Hata: `{error[:300]}`"
    )


def format_system_halted(reason: str, daily_pnl: float) -> str:
    """Sistem durdurulma bildirimi formatlar."""
    return (
        f"🛑 *SİSTEM DURDURULDU*\n\n"
        f"⚠️ Sebep: {reason}\n"
        f"📉 Günlük Kayıp: %{abs(daily_pnl):.2f}\n"
        f"⏰ {datetime.now().strftime('%d.%m.%Y %H:%M')}\n\n"
        f"_Manuel müdahale gerekebilir._"
    )


def format_order_executed(ticker: str, side: str, qty: int, price: float,
                          order_id: str, confidence: float,
                          stop_loss: float = 0, take_profit: float = 0) -> str:
    """Emir çalıştırma bildirimi formatlar."""
    emoji = "🟢" if side == "BUY" else "🔴"
    side_tr = "ALIM" if side == "BUY" else "SATIM"
    cost = qty * price
    timestamp = datetime.now().strftime("%d.%m.%Y %H:%M")

    msg = (
        f"{emoji} *{side_tr} EMRİ ÇALIŞTIRILDI*\n\n"
        f"📌 *{ticker}*\n"
        f"• Yön: {side_tr}\n"
        f"• Adet: {qty}\n"
        f"• Fiyat: ${price:.2f}\n"
        f"• Maliyet: ${cost:,.2f}\n"
        f"• Güven: %{confidence * 100:.0f}\n"
    )

    if stop_loss > 0:
        msg += f"• 🛑 Stop Loss: ${stop_loss:.2f}\n"
    if take_profit > 0:
        msg += f"• 🎯 Take Profit: ${take_profit:.2f}\n"

    msg += f"\n🆔 Emir: `{order_id[:12]}...`\n"
    msg += f"⏰ {timestamp}"
    return msg


def format_signal_summary(buy_signals: list, sell_signals: list,
                          hold_count: int, executed_count: int) -> str:
    """Toplu sinyal özeti formatlar."""
    timestamp = datetime.now().strftime("%d.%m.%Y %H:%M")
    total = len(buy_signals) + len(sell_signals) + hold_count

    msg = f"📊 *SİNYAL ÖZETİ — {timestamp}*\n\n"
    msg += f"🔍 Analiz: *{total}* hisse\n"
    msg += f"🟢 AL: *{len(buy_signals)}* | 🔴 SAT: *{len(sell_signals)}* | ⚪ BEKLE: *{hold_count}*\n"

    if executed_count > 0:
        msg += f"⚡ Otomatik emir: *{executed_count}*\n"

    if buy_signals:
        msg += "\n*🟢 AL Sinyalleri:*\n"
        for sig in buy_signals[:8]:
            t = sig.get("ticker", "?")
            c = sig.get("confidence", 0)
            p = sig.get("current_price", 0)
            msg += f"  • {t}: ${p:.2f} (güven %{c*100:.0f})\n"
        if len(buy_signals) > 8:
            msg += f"  _...ve {len(buy_signals) - 8} tane daha_\n"

    if sell_signals:
        msg += "\n*🔴 SAT Sinyalleri:*\n"
        for sig in sell_signals[:8]:
            t = sig.get("ticker", "?")
            c = sig.get("confidence", 0)
            p = sig.get("current_price", 0)
            msg += f"  • {t}: ${p:.2f} (güven %{c*100:.0f})\n"
        if len(sell_signals) > 8:
            msg += f"  _...ve {len(sell_signals) - 8} tane daha_\n"

    return msg
