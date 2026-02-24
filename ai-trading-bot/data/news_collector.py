"""
data/news_collector.py — Haber Toplama Modülü
NewsAPI + RSS feed entegrasyonu ile finansal haberler.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Optional
import time

import feedparser
import requests

logger = logging.getLogger(__name__)

# Ücretsiz finansal RSS kaynakları
RSS_FEEDS = {
    "reuters_markets": "https://feeds.reuters.com/reuters/businessNews",
    "marketwatch": "https://feeds.content.dowjones.io/public/rss/mw_topstories",
    "seeking_alpha": "https://seekingalpha.com/feed.xml",
    "yahoo_finance": "https://finance.yahoo.com/news/rssindex",
    "cnbc": "https://www.cnbc.com/id/100003114/device/rss/rss.html",
    "ft": "https://www.ft.com/?format=rss",
}

COMPANY_MAP = {
    "AAPL": "Apple",
    "NVDA": "NVIDIA",
    "TSLA": "Tesla",
    "MSFT": "Microsoft",
    "META": "Meta Facebook",
    "GOOGL": "Google Alphabet",
    "JPM": "JPMorgan",
    "BAC": "Bank of America",
}


class NewsAPICollector:
    """NewsAPI üzerinden finansal haberler toplar."""

    def __init__(self, api_key: str):
        try:
            from newsapi import NewsApiClient
            self._client = NewsApiClient(api_key=api_key)
            self._api_key = api_key
            logger.info("NewsAPI istemcisi başlatıldı.")
        except ImportError:
            logger.error("newsapi-python kurulu değil: pip install newsapi-python")
            raise

    def get_news(
        self,
        ticker: str,
        company_name: Optional[str] = None,
        days_back: int = 3,
        page_size: int = 10,
    ) -> list[str]:
        """Ticker için son haberleri çeker.

        Args:
            ticker:       Hisse/kripto sembolü
            company_name: Şirket adı (ek arama için)
            days_back:    Kaç günlük haber alınacağı
            page_size:    Maksimum haber sayısı

        Returns:
            Haber başlık + açıklama metinlerinin listesi
        """
        company = company_name or COMPANY_MAP.get(ticker, ticker)
        query = f"{ticker} OR {company}"
        from_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")

        try:
            articles = self._client.get_everything(
                q=query,
                language="en",
                sort_by="publishedAt",
                from_param=from_date,
                page_size=page_size,
            )
            results = []
            for a in articles.get("articles", []):
                title = a.get("title", "") or ""
                desc = a.get("description", "") or ""
                if title:
                    results.append(f"{title}. {desc}".strip())
            logger.info(f"NewsAPI → {ticker}: {len(results)} haber bulundu.")
            return results
        except Exception as e:
            logger.error(f"NewsAPI hatası ({ticker}): {e}")
            return []


class RSSCollector:
    """RSS feed'lerinden haber başlıkları toplar (ücretsiz)."""

    def __init__(self, timeout: int = 10):
        self._timeout = timeout

    def get_feed_articles(self, feed_url: str, max_items: int = 20) -> list[str]:
        """Belirtilen RSS feed'inden makale başlıklarını çeker."""
        try:
            feed = feedparser.parse(feed_url)
            items = []
            for entry in feed.entries[:max_items]:
                title = getattr(entry, "title", "")
                summary = getattr(entry, "summary", "")
                if title:
                    items.append(f"{title}. {summary[:200]}".strip())
            logger.info(f"RSS ({feed_url[:50]}...): {len(items)} makale.")
            return items
        except Exception as e:
            logger.warning(f"RSS feed hatası ({feed_url}): {e}")
            return []

    def get_ticker_relevant_articles(
        self, ticker: str, company_name: Optional[str] = None, max_total: int = 15
    ) -> list[str]:
        """Tüm RSS kaynaklarından ticker ile ilgili haberleri filtreler."""
        company = company_name or COMPANY_MAP.get(ticker, ticker)
        keywords = {ticker.lower(), company.lower().split()[0]}
        relevant = []

        for feed_name, url in RSS_FEEDS.items():
            articles = self.get_feed_articles(url, max_items=30)
            for article in articles:
                if any(kw in article.lower() for kw in keywords):
                    relevant.append(article)
                if len(relevant) >= max_total:
                    break
            if len(relevant) >= max_total:
                break
            time.sleep(0.5)  # Rate limit

        logger.info(f"RSS → {ticker}: {len(relevant)} ilgili haber.")
        return relevant[:max_total]


class NewsCollector:
    """Ana haber toplama sınıfı: NewsAPI + RSS birleştirir."""

    def __init__(self, newsapi_key: Optional[str] = None):
        self._newsapi: Optional[NewsAPICollector] = None
        self._rss = RSSCollector()

        if newsapi_key and newsapi_key != "YOUR_NEWSAPI_KEY":
            try:
                self._newsapi = NewsAPICollector(newsapi_key)
            except Exception:
                logger.warning("NewsAPI başlatılamadı, sadece RSS kullanılacak.")

    def get_news(
        self,
        ticker: str,
        company_name: Optional[str] = None,
        max_articles: int = 15,
    ) -> list[str]:
        """Verilen ticker için haber listesi döndürür.

        NewsAPI varsa onu kullanır, yoksa RSS'e döner.

        Args:
            ticker:       Hisse sembolü
            company_name: Şirket adı
            max_articles: Maksimum döndürülecek haber sayısı

        Returns:
            Haber metinlerinin listesi
        """
        articles = []

        if self._newsapi:
            articles = self._newsapi.get_news(
                ticker, company_name=company_name, page_size=max_articles
            )

        # NewsAPI sonuçları yetersizse RSS ekle
        if len(articles) < 5:
            rss_articles = self._rss.get_ticker_relevant_articles(
                ticker, company_name=company_name, max_total=max_articles - len(articles)
            )
            articles.extend(rss_articles)

        return articles[:max_articles]

    def get_bulk_news(
        self, tickers: list[str], max_per_ticker: int = 10
    ) -> dict[str, list[str]]:
        """Birden fazla ticker için toplu haber çeker."""
        result = {}
        for ticker in tickers:
            result[ticker] = self.get_news(ticker, max_articles=max_per_ticker)
            time.sleep(1)  # Rate limit
        return result
