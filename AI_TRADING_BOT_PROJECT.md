# 🤖 AI Trading Bot — Proje Dokümantasyonu

> **Amaç:** Kendi portföyünü yönetmek için anlık çalışan, AI destekli, otomatik işlem yapan bir sistem.  
> **Piyasalar:** ABD Hisse (Alpaca) + Kripto (CCXT/Binance)  
> **Dil:** Python 3.11+

---

## 📐 Sistem Mimarisi (Genel Bakış)

```
┌─────────────────────────────────────────────────────────────┐
│                      VERİ KAYNAKLARI                        │
│  Alpaca API │ CCXT (Binance) │ NewsAPI │ yfinance │ Web     │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│                   VERI & HABER KATMANI                       │
│  DataCollector │ NewsCollector │ WebScraper │ Macro Data     │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│                    AI / LLM KATMANI                          │
│  Claude API → Sentiment Analizi & Haber Skorlama             │
│  Claude API → Trade Açıklaması Üretimi                       │
│  Claude API → Piyasa Özeti & Günlük Brief                    │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│                  FEATURE ENGINEERING                         │
│  Teknik İndikatörler │ Sentiment Score │ Macro Features      │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│                    MODEL KATMANI                             │
│  XGBoost (Direction) │ LightGBM (Confidence) │ Risk Filter  │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│                    RİSK ENGİNE                               │
│  Position Sizing │ Stop Loss │ Take Profit │ Max Drawdown    │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│                  EXECUTION KATMANI                           │
│  Alpaca (Hisse) │ CCXT Binance (Kripto) │ Paper Mode         │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│                 BİLDİRİM & İZLEME                            │
│  Telegram Bot │ Streamlit Dashboard │ SQLite Logs            │
└─────────────────────────────────────────────────────────────┘
```

---

## 📁 Proje Klasör Yapısı

```
ai-trading-bot/
│
├── config/
│   ├── config.yaml            # Tüm ayarlar (API key'ler, parametreler)
│   └── tickers.yaml           # Takip edilecek semboller
│
├── data/
│   ├── collector.py           # Alpaca + CCXT + yfinance veri çekme
│   ├── news_collector.py      # NewsAPI + RSS haber toplama
│   ├── web_scraper.py         # Reddit (r/stocks, r/wallstreetbets) + Twitter/X
│   └── macro_data.py          # VIX, SPY, sektör ETF'leri
│
├── ai/
│   ├── sentiment.py           # Claude API ile haber sentiment analizi
│   ├── explainer.py           # Trade açıklaması üretimi
│   └── market_brief.py        # Günlük piyasa özeti (Telegram'a gönderilir)
│
├── features/
│   ├── technical.py           # RSI, EMA, ATR, MACD, Bollinger vb.
│   ├── sentiment_features.py  # Haber skoru → feature
│   └── market_regime.py       # Bull/Bear/Sideways classifier
│
├── models/
│   ├── trainer.py             # Model eğitim pipeline
│   ├── predictor.py           # Canlı tahmin
│   ├── backtest.py            # Walk-forward backtest
│   └── saved/                 # Eğitilmiş model dosyaları (.pkl)
│
├── risk/
│   ├── risk_engine.py         # Pozisyon sizing, stop loss, take profit
│   └── portfolio.py           # Portföy takibi ve limit kontrolü
│
├── execution/
│   ├── alpaca_executor.py     # ABD hisse işlem yürütme
│   ├── crypto_executor.py     # Kripto işlem yürütme (CCXT)
│   └── paper_mode.py          # Gerçek para olmadan simülasyon
│
├── notifications/
│   ├── telegram_bot.py        # Telegram sinyal botu
│   └── formatters.py          # Mesaj formatlama
│
├── monitoring/
│   ├── dashboard.py           # Streamlit dashboard
│   └── logger.py              # SQLite trade log
│
├── scheduler.py               # APScheduler ile anlık görev yönetimi
├── main.py                    # Ana çalıştırma dosyası
├── requirements.txt
└── README.md
```

---

## 1️⃣ Veri Katmanı

### Alpaca (ABD Hisse)
```python
# data/collector.py
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

client = StockHistoricalDataClient(API_KEY, SECRET_KEY)

def get_ohlcv(ticker: str, days: int = 60):
    request = StockBarsRequest(
        symbol_or_symbols=ticker,
        timeframe=TimeFrame.Day,
        start=datetime.now() - timedelta(days=days)
    )
    return client.get_stock_bars(request).df
```

### CCXT (Kripto - Binance)
```python
import ccxt

exchange = ccxt.binance({'apiKey': '...', 'secret': '...'})

def get_crypto_ohlcv(symbol: str = 'BTC/USDT', limit: int = 100):
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe='1d', limit=limit)
    return pd.DataFrame(ohlcv, columns=['timestamp','open','high','low','close','volume'])
```

### Macro Data (VIX, SPY)
```python
import yfinance as yf

def get_macro():
    vix = yf.download('^VIX', period='60d')['Close']
    spy = yf.download('SPY', period='60d')['Close']
    return vix, spy
```

---

## 2️⃣ Haber & Web Takip

### NewsAPI (Finansal Haberler)
```python
# data/news_collector.py
from newsapi import NewsApiClient

newsapi = NewsApiClient(api_key=NEWS_API_KEY)

def get_news(ticker: str, company_name: str) -> list[str]:
    articles = newsapi.get_everything(
        q=f'{ticker} OR {company_name}',
        language='en',
        sort_by='publishedAt',
        page_size=10
    )
    return [a['title'] + '. ' + (a['description'] or '') 
            for a in articles['articles']]
```

### Web Scraping (Reddit WSB, Finviz)
```python
# data/web_scraper.py
import praw
import requests
from bs4 import BeautifulSoup

# Reddit sentiment
reddit = praw.Reddit(client_id='...', client_secret='...', user_agent='bot')

def get_reddit_mentions(ticker: str) -> list[str]:
    subreddit = reddit.subreddit('wallstreetbets+stocks+investing')
    posts = []
    for post in subreddit.search(ticker, limit=20, sort='new'):
        posts.append(post.title)
    return posts

# Finviz haberler (ücretsiz)
def get_finviz_news(ticker: str) -> list[str]:
    url = f'https://finviz.com/quote.ashx?t={ticker}'
    headers = {'User-Agent': 'Mozilla/5.0'}
    soup = BeautifulSoup(requests.get(url, headers=headers).text, 'html.parser')
    news_table = soup.find(id='news-table')
    headlines = [row.find('a').text for row in news_table.findAll('tr')]
    return headlines[:15]
```

---

## 3️⃣ AI / Claude API Entegrasyonu

### Sentiment Analizi
```python
# ai/sentiment.py
import anthropic
import json

client = anthropic.Anthropic(api_key=CLAUDE_API_KEY)

def analyze_sentiment(news_list: list[str], ticker: str) -> dict:
    news_text = '\n'.join([f'- {n}' for n in news_list[:10]])
    
    response = client.messages.create(
        model='claude-opus-4-6',
        max_tokens=300,
        messages=[{
            'role': 'user',
            'content': f"""
Sen bir finansal analist asistanısın. Aşağıdaki haberleri analiz et.

Hisse: {ticker}
Haberler:
{news_text}

Sadece JSON döndür, başka hiçbir şey yazma:
{{
  "sentiment": "positive/negative/neutral",
  "score": -1.0 ile 1.0 arası float,
  "key_themes": ["tema1", "tema2"],
  "risk_keywords": ["risk1", "risk2"],
  "summary": "2 cümle özet"
}}
"""
        }]
    )
    return json.loads(response.content[0].text)
```

### Trade Açıklaması
```python
# ai/explainer.py
def explain_trade_signal(ticker: str, features: dict, prediction: str, confidence: float) -> str:
    response = client.messages.create(
        model='claude-opus-4-6',
        max_tokens=400,
        messages=[{
            'role': 'user',
            'content': f"""
Hisse: {ticker}
Model Kararı: {prediction} (Güven: %{confidence*100:.0f})
Teknik Veriler:
- RSI(14): {features.get('rsi_14')}
- EMA Trend: {features.get('ema_trend')}
- ATR: {features.get('atr')}
- Hacim Spike: {features.get('volume_spike')}
- Sentiment Skoru: {features.get('sentiment_score')}
- Piyasa Rejimi: {features.get('market_regime')}

Bu trade kararını kısa ve teknik bir dille açıkla (3-4 cümle).
Kesin tahmin yapma, gözlemsel ve analitik kal.
"""
        }]
    )
    return response.content[0].text
```

### Günlük Piyasa Brief'i
```python
# ai/market_brief.py
def generate_daily_brief(portfolio: dict, signals: list, macro: dict) -> str:
    response = client.messages.create(
        model='claude-opus-4-6',
        max_tokens=600,
        messages=[{
            'role': 'user',
            'content': f"""
Günlük piyasa özeti oluştur.
VIX: {macro['vix']:.2f}
SPY Günlük Değişim: {macro['spy_change']:.2f}%
Portföy P&L: {portfolio['daily_pnl']:.2f}%
Sinyal Sayısı: {len(signals)} (AL: {sum(1 for s in signals if s['direction']=='BUY')}, SAT: {sum(1 for s in signals if s['direction']=='SELL')})

Telegram için emoji kullanarak kısa bir günlük özet yaz (max 300 karakter).
"""
        }]
    )
    return response.content[0].text
```

---

## 4️⃣ Feature Engineering

```python
# features/technical.py
import pandas_ta as ta

def generate_features(df: pd.DataFrame) -> pd.DataFrame:
    # Momentum
    df['rsi_14'] = ta.rsi(df['close'], length=14)
    df['rsi_7']  = ta.rsi(df['close'], length=7)
    
    # Trend
    df['ema_9']  = ta.ema(df['close'], length=9)
    df['ema_21'] = ta.ema(df['close'], length=21)
    df['ema_50'] = ta.ema(df['close'], length=50)
    df['ema_cross_9_21']  = (df['ema_9'] > df['ema_21']).astype(int)
    df['ema_cross_21_50'] = (df['ema_21'] > df['ema_50']).astype(int)
    df['ema_trend'] = df['ema_cross_9_21'] + df['ema_cross_21_50']  # 0,1,2
    
    # Volatility
    df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14)
    df['atr_pct'] = df['atr'] / df['close']
    bb = ta.bbands(df['close'], length=20)
    df['bb_position'] = (df['close'] - bb['BBL_20_2.0']) / (bb['BBU_20_2.0'] - bb['BBL_20_2.0'])
    
    # Volume
    df['volume_ma20']    = df['volume'].rolling(20).mean()
    df['volume_spike']   = df['volume'] / df['volume_ma20']
    
    # Price Action
    df['gap_flag']       = ((df['open'] - df['close'].shift(1)) / df['close'].shift(1))
    df['momentum_3d']    = df['close'].pct_change(3)
    df['momentum_5d']    = df['close'].pct_change(5)
    
    # MACD
    macd = ta.macd(df['close'])
    df['macd_hist'] = macd['MACDh_12_26_9']
    
    return df.dropna()
```

### Market Regime Classifier
```python
# features/market_regime.py
def classify_market_regime(spy_data: pd.DataFrame) -> str:
    spy_data['ema_50']  = ta.ema(spy_data['close'], length=50)
    spy_data['ema_200'] = ta.ema(spy_data['close'], length=200)
    
    current = spy_data.iloc[-1]
    if current['close'] > current['ema_50'] > current['ema_200']:
        return 'BULL'
    elif current['close'] < current['ema_50'] < current['ema_200']:
        return 'BEAR'
    else:
        return 'SIDEWAYS'
```

---

## 5️⃣ Model Katmanı

```python
# models/trainer.py
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit

def prepare_target(df: pd.DataFrame, forward_days: int = 3, threshold: float = 0.02):
    future_return = df['close'].shift(-forward_days) / df['close'] - 1
    # 1 = +2%'den fazla yükseliş, 0 = nötr, -1 = düşüş
    df['target'] = 0
    df.loc[future_return >  threshold, 'target'] = 1
    df.loc[future_return < -threshold, 'target'] = -1
    return df.dropna()

FEATURES = [
    'rsi_14', 'rsi_7', 'ema_trend', 'atr_pct', 'bb_position',
    'volume_spike', 'gap_flag', 'momentum_3d', 'momentum_5d',
    'macd_hist', 'sentiment_score', 'vix', 'spy_correlation'
]

def train_model(df: pd.DataFrame):
    X = df[FEATURES]
    y = df['target']
    
    # Walk-forward CV (zaman serisi için kritik)
    tscv = TimeSeriesSplit(n_splits=5)
    
    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    model.fit(X, y)
    return model
```

### Tahmin (Canlı)
```python
# models/predictor.py
def generate_signal(ticker: str, model, df: pd.DataFrame, sentiment: dict) -> dict:
    latest = df[FEATURES].iloc[-1].copy()
    latest['sentiment_score'] = sentiment['score']
    
    pred_proba = model.predict_proba([latest])[0]
    pred_class = model.predict([latest])[0]
    
    direction_map = {-1: 'SELL', 0: 'HOLD', 1: 'BUY'}
    
    return {
        'ticker':     ticker,
        'direction':  direction_map[pred_class],
        'confidence': float(max(pred_proba)),
        'prob_up':    float(pred_proba[2]),
        'prob_down':  float(pred_proba[0]),
        'timestamp':  datetime.now().isoformat()
    }
```

---

## 6️⃣ Risk Engine

```python
# risk/risk_engine.py
class RiskEngine:
    def __init__(self, portfolio_value: float, max_risk_per_trade: float = 0.02):
        self.portfolio_value = portfolio_value
        self.max_risk_per_trade = max_risk_per_trade  # %2 per trade
    
    def calculate_position(self, signal: dict, current_price: float, atr: float) -> dict:
        if signal['confidence'] < 0.60:  # Düşük güven → işlem yok
            return {'action': 'SKIP', 'reason': 'Low confidence'}
        
        # ATR bazlı stop loss
        stop_loss_distance = atr * 2
        stop_loss_price    = current_price - stop_loss_distance  # BUY için
        
        # Position sizing: Kaybetmeyi göze aldığım miktar / stop mesafesi
        risk_amount  = self.portfolio_value * self.max_risk_per_trade
        shares       = int(risk_amount / stop_loss_distance)
        
        take_profit  = current_price + (stop_loss_distance * 2)  # 1:2 R/R
        
        return {
            'action':         signal['direction'],
            'shares':         shares,
            'entry':          current_price,
            'stop_loss':      round(stop_loss_price, 2),
            'take_profit':    round(take_profit, 2),
            'risk_reward':    '1:2',
            'risk_amount':    round(risk_amount, 2)
        }
```

---

## 7️⃣ Execution Katmanı

### Alpaca (ABD Hisse)
```python
# execution/alpaca_executor.py
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

trading_client = TradingClient(API_KEY, SECRET_KEY, paper=True)  # paper=False → gerçek para

def execute_order(ticker: str, side: str, qty: int, stop_loss: float, take_profit: float):
    # Market order
    order = MarketOrderRequest(
        symbol=ticker,
        qty=qty,
        side=OrderSide.BUY if side == 'BUY' else OrderSide.SELL,
        time_in_force=TimeInForce.DAY
    )
    response = trading_client.submit_order(order)
    
    # Bracket order (stop + take profit otomatik)
    # Alpaca bracket order için ayrı implementasyon gerekli
    return response
```

### CCXT (Kripto)
```python
# execution/crypto_executor.py
def execute_crypto_order(symbol: str, side: str, amount: float, stop_loss: float):
    order = exchange.create_order(
        symbol=symbol,
        type='market',
        side=side.lower(),
        amount=amount
    )
    # Stop loss
    exchange.create_order(
        symbol=symbol,
        type='stop_market',
        side='sell' if side == 'buy' else 'buy',
        amount=amount,
        params={'stopPrice': stop_loss}
    )
    return order
```

---

## 8️⃣ Telegram Bot

```python
# notifications/telegram_bot.py
from telegram import Bot

bot = Bot(token=TELEGRAM_TOKEN)
CHAT_ID = 'senin_chat_id'

async def send_signal(signal: dict, position: dict, explanation: str):
    direction_emoji = {'BUY': '🟢', 'SELL': '🔴', 'HOLD': '⚪️'}
    
    message = f"""
{direction_emoji[signal['direction']]} *{signal['ticker']}* — {signal['direction']}

📊 *Analiz*
• Güven: %{signal['confidence']*100:.0f}
• Yükseliş Olasılığı: %{signal['prob_up']*100:.0f}
• Düşüş Olasılığı: %{signal['prob_down']*100:.0f}

💰 *Pozisyon*
• Lot: {position['shares']} adet
• Giriş: ${position['entry']}
• Stop Loss: ${position['stop_loss']}
• Take Profit: ${position['take_profit']}
• Risk/Reward: {position['risk_reward']}

🧠 *AI Açıklaması*
{explanation}

⏰ {signal['timestamp'][:16]}
"""
    await bot.send_message(chat_id=CHAT_ID, text=message, parse_mode='Markdown')

async def send_daily_brief(brief: str, pnl: float):
    emoji = '📈' if pnl > 0 else '📉'
    await bot.send_message(
        chat_id=CHAT_ID,
        text=f"{emoji} *Günlük Özet*\n\n{brief}\n\nP&L: %{pnl:.2f}",
        parse_mode='Markdown'
    )
```

---

## 9️⃣ Scheduler (Anlık Çalışma)

```python
# scheduler.py
from apscheduler.schedulers.asyncio import AsyncIOScheduler
import asyncio

scheduler = AsyncIOScheduler()

# Her 15 dakikada veri güncelle
@scheduler.scheduled_job('interval', minutes=15)
async def update_data():
    for ticker in WATCHLIST:
        data = collector.get_ohlcv(ticker)
        news = news_collector.get_news(ticker)
        # cache'e yaz

# Her saat sinyal üret
@scheduler.scheduled_job('interval', hours=1)
async def generate_signals():
    for ticker in WATCHLIST:
        signal  = predictor.generate_signal(ticker, ...)
        if signal['direction'] != 'HOLD' and signal['confidence'] > 0.60:
            position    = risk_engine.calculate_position(signal, ...)
            explanation = explainer.explain_trade_signal(...)
            await telegram.send_signal(signal, position, explanation)
            if not PAPER_MODE:
                executor.execute_order(...)

# Piyasa açılışında brief gönder (09:30 ET)
@scheduler.scheduled_job('cron', hour=9, minute=30, timezone='America/New_York')
async def morning_brief():
    brief = market_brief.generate_daily_brief(...)
    await telegram.send_daily_brief(brief, portfolio.daily_pnl())

# Piyasa kapanışında özet (16:00 ET)
@scheduler.scheduled_job('cron', hour=16, minute=0, timezone='America/New_York')
async def closing_summary():
    await telegram.send_daily_brief(...)

if __name__ == '__main__':
    scheduler.start()
    asyncio.get_event_loop().run_forever()
```

---

## 🔟 Monitoring Dashboard

```python
# monitoring/dashboard.py
import streamlit as st
import sqlite3

st.set_page_config(page_title='AI Trading Bot', layout='wide')

def load_trades():
    conn = sqlite3.connect('trades.db')
    return pd.read_sql('SELECT * FROM trades ORDER BY timestamp DESC', conn)

col1, col2, col3, col4 = st.columns(4)
col1.metric('Toplam P&L', f'%{total_pnl:.2f}', delta=f'%{daily_pnl:.2f} bugün')
col2.metric('Sharpe Ratio', f'{sharpe:.2f}')
col3.metric('Max Drawdown', f'%{max_dd:.2f}')
col4.metric('Win Rate', f'%{win_rate:.0f}')

st.plotly_chart(equity_curve_chart)
st.dataframe(load_trades())
```

---

## 📦 Gereksinimler

```txt
# requirements.txt

# Broker & Veri
alpaca-py==0.29.0
ccxt==4.3.0
yfinance==0.2.40
newsapi-python==0.2.7
praw==7.7.1
beautifulsoup4==4.12.3
requests==2.31.0

# Feature Engineering
pandas==2.2.0
pandas-ta==0.3.14b
numpy==1.26.4
scikit-learn==1.4.0

# Model
xgboost==2.0.3
lightgbm==4.3.0

# AI / LLM
anthropic==0.29.0
openai==1.30.0         # isteğe bağlı

# Bildirim
python-telegram-bot==21.3

# Scheduler
APScheduler==3.10.4

# Dashboard
streamlit==1.35.0
plotly==5.22.0

# Veritabanı
sqlalchemy==2.0.30
```

---

## ⚙️ Config Dosyası

```yaml
# config/config.yaml

alpaca:
  api_key: "YOUR_ALPACA_KEY"
  secret_key: "YOUR_ALPACA_SECRET"
  paper: true                   # ← Önce true, sonra false

binance:
  api_key: "YOUR_BINANCE_KEY"
  secret_key: "YOUR_BINANCE_SECRET"
  testnet: true

claude:
  api_key: "YOUR_CLAUDE_KEY"
  model: "claude-opus-4-6"

news_api:
  api_key: "YOUR_NEWSAPI_KEY"

telegram:
  token: "YOUR_TELEGRAM_BOT_TOKEN"
  chat_id: "YOUR_CHAT_ID"

trading:
  paper_mode: true
  max_risk_per_trade: 0.02      # %2
  max_open_positions: 5
  min_confidence: 0.60
  forward_days: 3               # Swing trade: 3 gün
  target_threshold: 0.02        # %2 hareket hedefi

watchlist:
  stocks:
    - AAPL
    - NVDA
    - TSLA
    - MSFT
    - META
    - GOOGL
  crypto:
    - BTC/USDT
    - ETH/USDT
    - SOL/USDT
```

---

## 📅 Geliştirme Yol Haritası

| Ay | Hedef | Durum |
|----|-------|-------|
| **Ay 1** | Veri pipeline + Haber toplama + Claude sentiment | 🔲 |
| **Ay 1** | Feature engineering + backtest altyapısı | 🔲 |
| **Ay 2** | XGBoost model eğitimi + walk-forward validation | 🔲 |
| **Ay 2** | Risk engine + Telegram bot | 🔲 |
| **Ay 3** | Paper trading (Alpaca) + canlı sinyal | 🔲 |
| **Ay 3** | Streamlit dashboard + performans takibi | 🔲 |
| **Ay 4** | Kripto entegrasyonu (CCXT + Binance) | 🔲 |
| **Ay 4+** | Gerçek para + model iyileştirme | 🔲 |

---

## ⚠️ Kritik Uyarılar

**Backtest ≠ Gerçek Sonuç**
- Look-ahead bias'a dikkat et. Hiçbir feature geleceği kullanmamalı.
- Transaction cost'u modellemeden backtest anlamsız (slippage + commission).
- Walk-forward validation şart: 2018-2021 train → 2022 test → 2023 forward.

**Risk Yönetimi Her Şeyden Önce**
- Max pozisyon başına portföyün %2'si.
- Aynı anda max 5 açık pozisyon.
- Günlük %5 kayıpda tüm sistemi durdur.

**Önce Paper, Sonra Gerçek Para**
- En az 3 ay paper trading yap.
- Sharpe > 1.0 ve win rate > %50 olmadan canlıya geçme.

---

*Son güncelleme: 2026-02-24 | Versiyon: 1.0*
