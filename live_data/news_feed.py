"""Live financial news fetchers for NSE tickers (yfinance + Yahoo RSS fallback)."""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any
from xml.etree import ElementTree

import requests

logger = logging.getLogger(__name__)

_NEWS_HTTP_TIMEOUT = 8
_YAHOO_RSS_TEMPLATE = (
    'https://feeds.finance.yahoo.com/rss/2.0/headline?s={symbol}&region=IN&lang=en-IN'
)
_BROWSER_HEADERS: dict[str, str] = {
    'User-Agent': (
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
        '(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    ),
    'Accept': 'application/rss+xml, application/xml, text/xml, */*',
    'Accept-Language': 'en-US,en;q=0.9',
}

_POSITIVE_KEYWORDS: tuple[str, ...] = (
    'surge', 'soar', 'gain', 'rise', 'rally', 'beat', 'upgrade', 'growth',
    'profit', 'record', 'bullish', 'outperform', 'strong', 'boost', 'jump',
)
_NEGATIVE_KEYWORDS: tuple[str, ...] = (
    'fall', 'drop', 'decline', 'slump', 'miss', 'downgrade', 'loss', 'crash',
    'bearish', 'underperform', 'weak', 'cut', 'plunge', 'sink', 'warning',
)


@dataclass(frozen=True)
class NewsArticle:
    """Normalized financial news item for dashboard rendering."""

    headline: str
    timestamp: datetime
    publisher: str
    summary: str
    url: str
    symbol: str
    source_engine: str
    sentiment: str = 'Neutral'

    def to_dict(self) -> dict[str, Any]:
        return {
            'headline': self.headline,
            'timestamp_iso': self.timestamp.isoformat(),
            'publisher': self.publisher,
            'summary': self.summary,
            'url': self.url,
            'symbol': self.symbol,
            'source_engine': self.source_engine,
            'sentiment': self.sentiment,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> NewsArticle:
        ts_raw = data.get('timestamp_iso', '')
        try:
            timestamp = datetime.fromisoformat(str(ts_raw))
            if timestamp.tzinfo is None:
                timestamp = timestamp.replace(tzinfo=timezone.utc)
        except (TypeError, ValueError):
            timestamp = datetime.now(tz=timezone.utc)
        return cls(
            headline=str(data.get('headline', '') or 'Untitled'),
            timestamp=timestamp,
            publisher=str(data.get('publisher', '') or 'Unknown'),
            summary=str(data.get('summary', '') or ''),
            url=str(data.get('url', '') or ''),
            symbol=str(data.get('symbol', '') or ''),
            source_engine=str(data.get('source_engine', '') or 'unknown'),
            sentiment=str(data.get('sentiment', 'Neutral') or 'Neutral'),
        )


def normalize_news_symbol(symbol: str) -> str:
    """Ensure ticker is Yahoo-compatible (e.g. RELIANCE.NS)."""
    sym = str(symbol or '').strip().upper()
    if not sym:
        return sym
    if '.' not in sym:
        sym = f'{sym}.NS'
    return sym


def _strip_html(text: str) -> str:
    cleaned = re.sub(r'<[^>]+>', ' ', str(text or ''))
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned


def _infer_sentiment(headline: str, summary: str) -> str:
    blob = f'{headline} {summary}'.lower()
    pos = sum(1 for kw in _POSITIVE_KEYWORDS if kw in blob)
    neg = sum(1 for kw in _NEGATIVE_KEYWORDS if kw in blob)
    if pos > neg and pos > 0:
        return 'Positive'
    if neg > pos and neg > 0:
        return 'Negative'
    return 'Neutral'


def _parse_unix_timestamp(value: Any) -> datetime:
    try:
        if value is None:
            raise ValueError('missing timestamp')
        if isinstance(value, (int, float)):
            return datetime.fromtimestamp(float(value), tz=timezone.utc)
        if isinstance(value, str) and value.isdigit():
            return datetime.fromtimestamp(float(value), tz=timezone.utc)
    except (TypeError, ValueError, OSError):
        pass
    return datetime.now(tz=timezone.utc)


def _fetch_yfinance_news(symbol: str, max_articles: int) -> list[NewsArticle]:
    try:
        import yfinance as yf
    except ImportError as exc:
        logger.debug('yfinance unavailable for news: %s', exc)
        return []

    articles: list[NewsArticle] = []
    try:
        raw_items = yf.Ticker(symbol).news or []
    except Exception as exc:
        logger.warning('yfinance news fetch failed for %s: %s', symbol, exc)
        return []

    for item in raw_items[: max(max_articles * 2, max_articles)]:
        if not isinstance(item, dict):
            continue
        headline = str(
            item.get('title')
            or item.get('headline')
            or item.get('content', {}).get('title', '')
            or ''
        ).strip()
        if not headline:
            continue

        link = str(
            item.get('link')
            or item.get('url')
            or item.get('content', {}).get('clickThroughUrl', {}).get('url', '')
            or item.get('content', {}).get('canonicalUrl', {}).get('url', '')
            or ''
        ).strip()

        publisher = str(
            item.get('publisher')
            or item.get('publisherName')
            or item.get('content', {}).get('provider', {}).get('displayName', '')
            or 'Yahoo Finance'
        ).strip()

        summary = _strip_html(
            str(
                item.get('summary')
                or item.get('description')
                or item.get('content', {}).get('summary', '')
                or item.get('content', {}).get('description', '')
                or ''
            )
        )

        ts = _parse_unix_timestamp(
            item.get('providerPublishTime')
            or item.get('pubDate')
            or item.get('content', {}).get('pubDate')
        )

        articles.append(
            NewsArticle(
                headline=headline,
                timestamp=ts,
                publisher=publisher or 'Yahoo Finance',
                summary=summary or headline,
                url=link,
                symbol=symbol,
                source_engine='yfinance',
                sentiment=_infer_sentiment(headline, summary),
            )
        )
    return articles


def _fetch_yahoo_rss_news(symbol: str, max_articles: int) -> list[NewsArticle]:
    articles: list[NewsArticle] = []
    rss_symbol = symbol.replace('.NS', '') if symbol.endswith('.NS') else symbol
    url = _YAHOO_RSS_TEMPLATE.format(symbol=requests.utils.quote(rss_symbol))

    try:
        response = requests.get(
            url,
            headers=_BROWSER_HEADERS,
            timeout=_NEWS_HTTP_TIMEOUT,
        )
        response.raise_for_status()
        root = ElementTree.fromstring(response.content)
    except Exception as exc:
        logger.warning('Yahoo RSS news fetch failed for %s: %s', symbol, exc)
        return []

    channel = root.find('channel')
    if channel is None:
        return []

    for item in channel.findall('item')[:max_articles]:
        title = (item.findtext('title') or '').strip()
        if not title:
            continue
        link = (item.findtext('link') or '').strip()
        pub_date = (item.findtext('pubDate') or '').strip()
        description = _strip_html(item.findtext('description') or '')
        publisher = (item.findtext('source') or 'Yahoo Finance').strip()

        try:
            timestamp = parsedate_to_datetime(pub_date)
            if timestamp.tzinfo is None:
                timestamp = timestamp.replace(tzinfo=timezone.utc)
        except (TypeError, ValueError):
            timestamp = datetime.now(tz=timezone.utc)

        articles.append(
            NewsArticle(
                headline=title,
                timestamp=timestamp,
                publisher=publisher or 'Yahoo Finance',
                summary=description or title,
                url=link,
                symbol=symbol,
                source_engine='yahoo_rss',
                sentiment=_infer_sentiment(title, description),
            )
        )
    return articles


def _dedupe_and_sort(articles: list[NewsArticle], max_articles: int) -> list[NewsArticle]:
    seen_urls: set[str] = set()
    seen_headlines: set[str] = set()
    unique: list[NewsArticle] = []

    for article in sorted(articles, key=lambda a: a.timestamp, reverse=True):
        url_key = article.url.lower().strip()
        headline_key = article.headline.lower().strip()
        if url_key and url_key in seen_urls:
            continue
        if headline_key in seen_headlines:
            continue
        if url_key:
            seen_urls.add(url_key)
        seen_headlines.add(headline_key)
        unique.append(article)
        if len(unique) >= max_articles:
            break
    return unique


def fetch_stock_news(symbol: str, max_articles: int = 8) -> list[NewsArticle]:
    """
    Fetch latest financial news for an NSE ticker.

    Primary: yfinance ``Ticker.news`` · Fallback: Yahoo Finance RSS headlines.
    Never raises — returns an empty list on total failure.
    """
    sym = normalize_news_symbol(symbol)
    if not sym:
        return []

    limit = max(1, min(int(max_articles), 20))
    combined: list[NewsArticle] = []

    try:
        combined.extend(_fetch_yfinance_news(sym, limit))
    except Exception as exc:
        logger.warning('yfinance news pipeline error for %s: %s', sym, exc)

    if len(combined) < limit:
        try:
            combined.extend(_fetch_yahoo_rss_news(sym, limit))
        except Exception as exc:
            logger.warning('Yahoo RSS news pipeline error for %s: %s', sym, exc)

    return _dedupe_and_sort(combined, limit)
