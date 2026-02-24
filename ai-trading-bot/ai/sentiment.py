"""
ai/sentiment.py — OpenAI (ChatGPT) ile Haber Sentiment Analizi
"""

from __future__ import annotations

import json
import logging
import time
from typing import Optional

from openai import OpenAI, RateLimitError
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)


class SentimentAnalyzer:
    """OpenAI ChatGPT API üzerinden haber listelerinin sentiment analizini yapar.

    Her çağrı için JSON formatında sentiment skoru, anahtar temalar ve
    risk kelimeleri döndürür.
    """

    SYSTEM_PROMPT = (
        "Sen uzman bir finansal analiz asistanısın. "
        "Haber başlıklarını analiz edip yatırımcılar için anlamlı "
        "sentiment skoru üretirsin. Yanıtlarını daima geçerli JSON olarak ver."
    )

    def __init__(self, api_key: str, model: str = "gpt-4o"):
        self._client = OpenAI(api_key=api_key)
        self._model = model
        logger.info(f"SentimentAnalyzer başlatıldı. Model: {model}")

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def analyze_sentiment(
        self, news_list: list[str], ticker: str, max_news: int = 10
    ) -> dict:
        """Haber listesi için sentiment analizi yapar.

        Args:
            news_list: Analiz edilecek haber metinleri
            ticker:    İlgili hisse/kripto sembolü
            max_news:  Maksimum kullanılacak haber sayısı

        Returns:
            {
                'sentiment': 'positive/negative/neutral',
                'score': float  (-1.0 to 1.0),
                'key_themes': list[str],
                'risk_keywords': list[str],
                'summary': str,
                'ticker': str
            }
        """
        if not news_list:
            return self._empty_result(ticker, reason="Haber bulunamadı")

        news_text = "\n".join([f"- {n}" for n in news_list[:max_news]])

        prompt = f"""
Sen bir finansal analist asistanısın. Aşağıdaki haberleri analiz et.

Hisse/Kripto: {ticker}
Haberler:
{news_text}

Sadece aşağıdaki JSON formatında yanıt ver, başka hiçbir şey yazma:
{{
  "sentiment": "positive veya negative veya neutral",
  "score": -1.0 ile 1.0 arasında float (olumlu haberler için pozitif),
  "key_themes": ["tema1", "tema2", "tema3"],
  "risk_keywords": ["risk1", "risk2"],
  "summary": "Haberlerin 2 cümlelik özeti"
}}
"""
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                max_tokens=400,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
            )
            raw_text = response.choices[0].message.content.strip()

            # JSON temizleme (markdown code block varsa sil)
            if "```" in raw_text:
                raw_text = raw_text.split("```")[1]
                if raw_text.startswith("json"):
                    raw_text = raw_text[4:]

            result = json.loads(raw_text)
            result["ticker"] = ticker
            result["news_count"] = len(news_list[:max_news])

            logger.info(
                f"Sentiment → {ticker}: {result['sentiment']} "
                f"(score={result['score']:.2f})"
            )
            return result

        except json.JSONDecodeError as e:
            logger.error(f"Sentiment JSON parse hatası ({ticker}): {e}")
            return self._empty_result(ticker, reason="JSON parse hatası")
        except RateLimitError:
            logger.warning("OpenAI rate limit! 60s bekleniyor...")
            time.sleep(60)
            raise
        except Exception as e:
            logger.error(f"Sentiment analiz hatası ({ticker}): {e}")
            return self._empty_result(ticker, reason=str(e))

    def analyze_bulk(
        self, news_dict: dict[str, list[str]], delay_seconds: float = 1.5
    ) -> dict[str, dict]:
        """Birden fazla ticker için toplu sentiment analizi.

        Args:
            news_dict:     {ticker: [news_list]}
            delay_seconds: API çağrıları arası bekleme süresi

        Returns:
            {ticker: sentiment_result}
        """
        results = {}
        for ticker, news_list in news_dict.items():
            results[ticker] = self.analyze_sentiment(news_list, ticker)
            time.sleep(delay_seconds)
        return results

    @staticmethod
    def _empty_result(ticker: str, reason: str = "") -> dict:
        """Hata durumunda nötr varsayılan sonuç döndürür."""
        return {
            "sentiment": "neutral",
            "score": 0.0,
            "key_themes": [],
            "risk_keywords": [],
            "summary": reason or "Analiz yapılamadı.",
            "ticker": ticker,
            "news_count": 0,
        }
