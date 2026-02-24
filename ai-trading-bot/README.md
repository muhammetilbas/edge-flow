# 🤖 AI Trading Bot

Yapay zeka destekli otomatik borsa analiz ve paper trading botu.

## Özellikler
- 📊 115+ hisse analizi (S&P 500 majörleri)
- 🧠 XGBoost ML modelleri ile sinyal üretimi
- 🤖 Alpaca Paper Trading ile otomatik alım/satım
- 📱 Telegram bildirimleri
- 📈 Streamlit dashboard

## Kurulum

```bash
pip install -r requirements.txt
```

## Çalıştırma

```bash
streamlit run monitoring/dashboard.py
```

## Ayarlar
`.env` dosyasında API anahtarlarınızı girin:
- `OPENAI_API_KEY`
- `ALPACA_API_KEY` / `ALPACA_SECRET_KEY`
- `TELEGRAM_TOKEN` / `TELEGRAM_CHAT_ID`
