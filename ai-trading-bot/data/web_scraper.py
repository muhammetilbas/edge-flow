"""
data/web_scraper.py — Web Kazıma Modülü
Reddit (WSB, stocks, investing) + Finviz haber çekme.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

FINVIZ_BASE_URL = "https://finviz.com/quote.ashx"
REDDIT_SEARCH_URL = "https://www.reddit.com/search.json"

DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}


# ── Reddit Scraper ───────────────────────────────────────────────────────────

class RedditScraper:
    """Reddit'ten hisse/kripto ile ilgili postları çeker.

    PRAW kullanılıyorsa daha güvenilir; yoksa JSON API ile fallback.
    """

    def __init__(
        self,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        user_agent: str = "AITradingBot/1.0",
    ):
        self._praw_reddit = None
        if client_id and client_id != "YOUR_REDDIT_CLIENT_ID":
            try:
                import praw
                self._praw_reddit = praw.Reddit(
                    client_id=client_id,
                    client_secret=client_secret,
                    user_agent=user_agent,
                )
                logger.info("PRAW Reddit istemcisi başlatıldı.")
            except ImportError:
                logger.warning("praw kurulu değil. JSON API kullanılacak.")
            except Exception as e:
                logger.warning(f"PRAW başlatılamadı: {e}. JSON API kullanılacak.")

    def get_reddit_mentions(
        self,
        ticker: str,
        subreddits: str = "wallstreetbets+stocks+investing",
        limit: int = 25,
    ) -> list[str]:
        """Subredditlerden ticker ile ilgili post başlıklarını çeker.

        Args:
            ticker:     Aranacak hisse/kripto sembolü
            subreddits: '+' ile birleştirilmiş subreddit listesi
            limit:      Maksimum post sayısı

        Returns:
            Post başlıkları listesi
        """
        if self._praw_reddit:
            return self._get_via_praw(ticker, subreddits, limit)
        return self._get_via_json_api(ticker, subreddits, limit)

    def _get_via_praw(self, ticker: str, subreddits: str, limit: int) -> list[str]:
        """PRAW ile Reddit'ten veri çeker."""
        try:
            sub = self._praw_reddit.subreddit(subreddits)
            posts = []
            for post in sub.search(ticker, limit=limit, sort="new"):
                text = post.title
                if post.selftext:
                    text += f" {post.selftext[:200]}"
                posts.append(text.strip())
            logger.info(f"PRAW → {ticker}: {len(posts)} post bulundu.")
            return posts
        except Exception as e:
            logger.error(f"PRAW hatası: {e}")
            return self._get_via_json_api(ticker, subreddits, limit)

    def _get_via_json_api(self, ticker: str, subreddits: str, limit: int) -> list[str]:
        """Reddit JSON API ile veri çeker (PRAW yokken fallback)."""
        try:
            url = f"https://www.reddit.com/r/{subreddits}/search.json"
            params = {"q": ticker, "sort": "new", "limit": limit, "restrict_sr": "1"}
            resp = requests.get(url, headers=DEFAULT_HEADERS, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            posts = []
            for child in data.get("data", {}).get("children", []):
                post_data = child.get("data", {})
                title = post_data.get("title", "")
                if title:
                    posts.append(title)
            logger.info(f"Reddit JSON API → {ticker}: {len(posts)} post.")
            return posts
        except Exception as e:
            logger.error(f"Reddit JSON API hatası ({ticker}): {e}")
            return []

    def get_wsb_hot(self, limit: int = 20) -> list[str]:
        """WallStreetBets subredditindeki hot postları çeker."""
        if self._praw_reddit:
            try:
                sub = self._praw_reddit.subreddit("wallstreetbets")
                return [post.title for post in sub.hot(limit=limit)]
            except Exception as e:
                logger.error(f"WSB hot hatası: {e}")
        return []


# ── Finviz Scraper ───────────────────────────────────────────────────────────

class FinvizScraper:
    """Finviz'den haber başlıkları ve temel hisse verileri çeker."""

    def __init__(self, request_delay: float = 1.5):
        self._delay = request_delay

    def get_finviz_news(self, ticker: str, max_headlines: int = 15) -> list[str]:
        """Finviz'den ticker ile ilgili son haber başlıklarını çeker.

        Args:
            ticker:        Hisse sembolü
            max_headlines: Maksimum başlık sayısı

        Returns:
            Haber başlıkları listesi
        """
        url = f"{FINVIZ_BASE_URL}?t={ticker}"
        try:
            resp = requests.get(url, headers=DEFAULT_HEADERS, timeout=10)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")

            news_table = soup.find(id="news-table")
            if not news_table:
                logger.warning(f"Finviz haber tablosu bulunamadı: {ticker}")
                return []

            headlines = []
            for row in news_table.find_all("tr"):
                link = row.find("a")
                if link and link.text.strip():
                    headlines.append(link.text.strip())
                    if len(headlines) >= max_headlines:
                        break

            logger.info(f"Finviz → {ticker}: {len(headlines)} başlık.")
            time.sleep(self._delay)
            return headlines
        except requests.HTTPError as e:
            logger.error(f"Finviz HTTP hatası ({ticker}): {e}")
            return []
        except Exception as e:
            logger.error(f"Finviz scraping hatası ({ticker}): {e}")
            return []

    def get_stock_info(self, ticker: str) -> dict:
        """Finviz'den temel hisse bilgilerini çeker (P/E, Sector, vb.)."""
        url = f"{FINVIZ_BASE_URL}?t={ticker}"
        try:
            resp = requests.get(url, headers=DEFAULT_HEADERS, timeout=10)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")

            info = {}
            # Finviz'de fundamental veriler table hücrelerinde
            cells = soup.find_all("td", {"class": "snapshot-td2"})
            labels = soup.find_all("td", {"class": "snapshot-td2-cp"})

            for label, cell in zip(labels, cells):
                key = label.text.strip()
                val = cell.text.strip()
                if key:
                    info[key] = val

            time.sleep(self._delay)
            return info
        except Exception as e:
            logger.error(f"Finviz info hatası ({ticker}): {e}")
            return {}


# ── Ana WebScraper Sınıfı ───────────────────────────────────────────────────

class WebScraper:
    """Reddit ve Finviz'i birleştiren ana web kazıyıcı."""

    def __init__(
        self,
        reddit_client_id: Optional[str] = None,
        reddit_client_secret: Optional[str] = None,
        reddit_user_agent: str = "AITradingBot/1.0",
    ):
        self._reddit = RedditScraper(
            client_id=reddit_client_id,
            client_secret=reddit_client_secret,
            user_agent=reddit_user_agent,
        )
        self._finviz = FinvizScraper()

    def get_all_web_sentiment(
        self, ticker: str, max_reddit: int = 20, max_finviz: int = 15
    ) -> dict[str, list[str]]:
        """Tüm web kaynaklarından ticker ile ilgili içerikleri toplar.

        Returns:
            {'reddit': [...], 'finviz': [...], 'all': [...]}
        """
        reddit_posts = self._reddit.get_reddit_mentions(ticker, limit=max_reddit)
        finviz_news = self._finviz.get_finviz_news(ticker, max_headlines=max_finviz)

        all_texts = reddit_posts + finviz_news
        return {
            "reddit": reddit_posts,
            "finviz": finviz_news,
            "all": all_texts,
        }

    def get_reddit_sentiment(self, ticker: str, limit: int = 25) -> list[str]:
        """Sadece Reddit verisi döndürür."""
        return self._reddit.get_reddit_mentions(ticker, limit=limit)

    def get_finviz_news(self, ticker: str, max_headlines: int = 15) -> list[str]:
        """Sadece Finviz haberleri döndürür."""
        return self._finviz.get_finviz_news(ticker, max_headlines=max_headlines)
