# 🤖 AI Trading Bot

> Kişisel portföy yönetimi için AI destekli, otomatik işlem sistemi.  
> **ABD Hisse (Alpaca Paper) + Kripto (Binance Testnet) | Python 3.11+**

---

## 📐 Mimari

```
Veri Kaynakları → AI/Claude → Feature Engineering → XGBoost/LightGBM
      ↓                                                      ↓
  NewsAPI/Reddit                                        Risk Engine
      ↓                                                      ↓
  Macro (VIX/SPY)                              Alpaca (Paper) / CCXT / PaperTrader
                                                             ↓
                                                 Telegram + Streamlit Dashboard
```

---

## 🚀 Kurulum

### 1. Ortam Hazırlığı

```bash
# Python 3.11+ gerekli
python --version

# Sanal ortam oluştur
python -m venv .venv
source .venv/bin/activate      # macOS/Linux
# .venv\Scripts\activate       # Windows

# Bağımlılıkları kur
pip install -r requirements.txt
```

### 2. API Keys

```bash
# Örnek .env dosyasını kopyala
cp .env.example .env

# .env dosyasını düzenle ve kendi API key'lerini gir
# ASLA .env'i Git'e push etme!
```

Gerekli API hesapları:
| Servis | Kayıt Linki | Maliyet |
|--------|------------|---------|
| **Alpaca** | [alpaca.markets](https://alpaca.markets) | Ücretsiz (paper) |
| **Binance Testnet** | [testnet.binance.vision](https://testnet.binance.vision) | Ücretsiz |
| **Claude AI** | [console.anthropic.com](https://console.anthropic.com) | API başına ücret |
| **NewsAPI** | [newsapi.org](https://newsapi.org) | Ücretsiz (100 req/gün) |
| **Telegram Bot** | [@BotFather](https://t.me/BotFather) | Ücretsiz |

### 3. Konfigürasyon

```bash
# config/config.yaml içinde ayarları kontrol et
# Key'ler .env'den okunur (python-dotenv)

# Takip edilecek sembolleri düzenle:
# config/tickers.yaml
```

---

## ▶️ Çalıştırma

### Model Eğitimi (önce yapılmalı)

```python
# Python shell veya Jupyter
from data.collector import DataManager
from features.technical import TechnicalFeatureGenerator
from models.trainer import ModelTrainer

# Konfigürasyonu yükle
import yaml
config = yaml.safe_load(open('config/config.yaml'))

# Veri çek ve model eğit
dm = DataManager(config)
fg = TechnicalFeatureGenerator()
trainer = ModelTrainer()

df = dm.get_stock_data('AAPL', days=365)
df_feat = fg.generate_features(df)
model, metrics = trainer.train(df_feat, ticker='AAPL')
print(metrics)
```

### Bot Başlatma

```bash
# Paper trading modunda başlat (güvenli)
python main.py

# Sadece dashboard
streamlit run monitoring/dashboard.py
```

### Backtest

```python
from models.backtest import Backtester

bt = Backtester(initial_capital=10000)
results = bt.run(df_feat, ticker='AAPL')
print(f"Sharpe: {results['sharpe_ratio']:.2f}")
print(f"Win Rate: %{results['win_rate_pct']:.1f}")
print(f"Max DD: %{results['max_drawdown_pct']:.2f}")
```

---

## 📁 Proje Yapısı

```
ai-trading-bot/
├── config/             # Konfigürasyon dosyaları
├── data/               # Veri toplama modülleri
│   ├── collector.py      → Alpaca + CCXT + yfinance
│   ├── news_collector.py → NewsAPI + RSS
│   ├── web_scraper.py    → Reddit + Finviz
│   └── macro_data.py     → VIX, SPY, sektör ETF
├── ai/                 # Claude AI modülleri
│   ├── sentiment.py      → Haber sentiment analizi
│   ├── explainer.py      → Trade açıklaması
│   └── market_brief.py   → Günlük piyasa özeti
├── features/           # Feature engineering
│   ├── technical.py      → RSI, EMA, ATR, MACD, BB
│   ├── sentiment_features.py
│   └── market_regime.py  → BULL/BEAR/SIDEWAYS
├── models/             # ML modelleri
│   ├── trainer.py        → XGBoost + LightGBM
│   ├── predictor.py      → Canlı tahmin
│   ├── backtest.py       → Walk-forward backtest
│   └── saved/            → Eğitilmiş modeller (.pkl)
├── risk/               # Risk yönetimi
│   ├── risk_engine.py    → Position sizing, SL/TP
│   └── portfolio.py      → Portföy takip
├── execution/          # İşlem yürütme
│   ├── alpaca_executor.py
│   ├── crypto_executor.py
│   └── paper_mode.py     → Simülasyon modu
├── notifications/      # Bildirimler
│   ├── telegram_bot.py
│   └── formatters.py
├── monitoring/         # İzleme
│   ├── dashboard.py      → Streamlit
│   └── logger.py         → SQLite
├── scheduler.py        # APScheduler görevleri
├── main.py             # Ana giriş noktası
├── requirements.txt
└── .env.example
```

---

## ⚙️ Özellikler

| Özellik | Durum |
|---------|-------|
| Alpaca Paper Trading | ✅ |
| Binance Testnet | ✅ |
| Claude AI Sentiment | ✅ |
| XGBoost / LightGBM | ✅ |
| Walk-forward Backtest | ✅ |
| ATR bazlı Stop Loss | ✅ |
| Telegram Bildirimleri | ✅ |
| Streamlit Dashboard | ✅ |
| SQLite Loglama | ✅ |
| APScheduler | ✅ |
| Paper Mode | ✅ |
| News RSS Fallback | ✅ |
| Reddit Scraping | ✅ |
| Pandas-TA Fallback | ✅ |

---

## ⚠️ Kritik Uyarılar

> **Backtest ≠ Gerçek Sonuç**  
> Look-ahead bias, slippage ve komisyon varlığında sonuçlar değişebilir.

> **Önce Paper Trading!**  
> En az 3 ay paper mode çalıştır. Sharpe > 1.0 + Win Rate > %50 olmadan gerçek paraya geçme.

> **Risk Kuralları**  
> - Max pozisyon: portföyün %2'si  
> - Max açık pozisyon: 5  
> - Günlük %5 kayıpda sistem durur

---

*Versiyon: 1.0 | Python 3.11+ | 2026*
