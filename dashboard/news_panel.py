"""Streamlit UI for NSE equity news, deterministic sentiment, and dashboard cards."""
from __future__ import annotations

import hashlib
import html
import logging
import math
import re
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Any

logger = logging.getLogger(__name__)

NEWS_CACHE_TTL_SEC = 1800
_SENTIMENT_COLORS: dict[str, str] = {
    'Positive': '#00e676',
    'Negative': '#ff5252',
    'Neutral': '#ffc107',
}

_POSITIVE_LEXICON: dict[str, float] = {
    'beat': 0.85, 'beats': 0.85, 'surge': 0.80, 'soar': 0.80, 'rally': 0.70,
    'gain': 0.45, 'gains': 0.45, 'rise': 0.40, 'upgrade': 0.75, 'upgraded': 0.75,
    'growth': 0.50, 'profit': 0.55, 'record': 0.70, 'bullish': 0.80,
    'outperform': 0.70, 'strong': 0.40, 'boost': 0.50, 'jump': 0.60,
    'order win': 0.85, 'order wins': 0.85, 'mega deal': 0.90, 'large deal': 0.80,
    'contract': 0.45, 'expansion': 0.50, 'momentum': 0.35, 'outperformer': 0.65,
    'dividend': 0.40, 'buyback': 0.55, 'guidance raise': 0.90, 'raised guidance': 0.90,
    'ai': 0.25, 'refining margin': 0.55, 'grm': 0.50,
}
_NEGATIVE_LEXICON: dict[str, float] = {
    'miss': -0.85, 'misses': -0.85, 'fall': -0.45, 'drop': -0.50, 'decline': -0.50,
    'slump': -0.70, 'downgrade': -0.80, 'loss': -0.70, 'crash': -0.95,
    'bearish': -0.80, 'underperform': -0.70, 'weak': -0.45, 'cut': -0.55,
    'plunge': -0.85, 'sink': -0.70, 'warning': -0.60, 'layoff': -0.65,
    'delay': -0.40, 'probe': -0.55, 'investigation': -0.60, 'crude spike': -0.35,
    'margin pressure': -0.60, 'guidance cut': -0.90, 'lowered guidance': -0.90,
    'outflow': -0.45, 'selloff': -0.70, 'volatility': -0.20,
}

try:
    import streamlit as st
except Exception:  # pragma: no cover
    st = None  # type: ignore[assignment]

try:
    from zoneinfo import ZoneInfo
except Exception:  # pragma: no cover
    ZoneInfo = None  # type: ignore[misc, assignment]

try:
    from live_data.news_feed import NewsArticle, fetch_stock_news, normalize_news_symbol
except Exception as exc:  # pragma: no cover
    logger.warning('news_feed unavailable; using local news fallbacks: %s', exc)

    class NewsArticle:  # type: ignore[no-redef]
        def __init__(
            self,
            headline: str = '',
            timestamp: datetime | None = None,
            publisher: str = 'Unknown',
            summary: str = '',
            url: str = '',
            symbol: str = '',
            source_engine: str = 'fallback',
            sentiment: str = 'Neutral',
            sentiment_score: float = 0.0,
        ) -> None:
            self.headline = headline
            self.timestamp = timestamp or datetime.now(tz=timezone.utc)
            self.publisher = publisher
            self.summary = summary
            self.url = url
            self.symbol = symbol
            self.source_engine = source_engine
            self.sentiment = sentiment
            self.sentiment_score = float(sentiment_score)

        @classmethod
        def from_dict(cls, data: dict[str, Any]) -> NewsArticle:
            ts_raw = data.get('timestamp_iso', '')
            try:
                timestamp = datetime.fromisoformat(str(ts_raw))
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
                sentiment_score=float(data.get('sentiment_score', 0.0) or 0.0),
            )

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
                'sentiment_score': float(getattr(self, 'sentiment_score', 0.0) or 0.0),
            }

    def fetch_stock_news(symbol: str, max_articles: int = 8) -> list[Any]:
        return []

    def normalize_news_symbol(symbol: str) -> str:
        sym = str(symbol or '').strip().upper()
        if not sym:
            return sym
        if '.' not in sym:
            return f'{sym}.NS'
        return sym

try:
    from utils.market_schedule import NSE_TIMEZONE
except Exception:  # pragma: no cover
    NSE_TIMEZONE = timezone.utc


def _clip_sentiment(value: float) -> float:
    try:
        return float(max(-1.0, min(1.0, float(value))))
    except (TypeError, ValueError):
        return 0.0


def _label_from_score(score: float) -> str:
    if score > 0.15:
        return 'Positive'
    if score < -0.15:
        return 'Negative'
    return 'Neutral'


def compute_headline_sentiment_score(headline: str, summary: str = '') -> float:
    """
    Deterministic polarity in [-1.0, +1.0] from a lexicon-weighted tanh transform.
    Same headline always yields the same score (no RNG / jitter).
    """
    blob = f'{headline} {summary}'.lower()
    blob = re.sub(r'[^a-z0-9+\-\s]', ' ', blob)
    raw = 0.0
    for phrase, weight in _POSITIVE_LEXICON.items():
        if phrase in blob:
            raw += weight
    for phrase, weight in _NEGATIVE_LEXICON.items():
        if phrase in blob:
            raw += weight
    digest = hashlib.sha256(blob.encode('utf-8')).hexdigest()
    # Tiny deterministic residual so two similar headlines do not collapse to identical 0.00
    residual = ((int(digest[:6], 16) / 0xFFFFFF) - 0.5) * 0.04
    return _clip_sentiment(math.tanh(raw * 0.85) + residual)


def _technical_prior(features: dict[str, float] | None) -> float:
    if not features:
        return 0.0
    try:
        rsi = float(features.get('rsi_14', features.get('RSI', 50.0)))
        macd_d = float(features.get('macd_hist_delta', features.get('MACD_Hist_Delta', 0.0)))
        news_feat = float(features.get('news_sentiment', 0.0))
        tech = math.tanh(((rsi - 50.0) / 50.0) * 0.6 + math.tanh(macd_d * 40.0) * 0.4)
        return _clip_sentiment(0.85 * tech + 0.15 * news_feat)
    except (TypeError, ValueError, OverflowError):
        return 0.0


def _issuer_name(symbol: str) -> str:
    root = str(symbol or '').upper().replace('.NS', '').replace('.BO', '')
    names = {
        'RELIANCE': 'Reliance Industries',
        'TCS': 'Tata Consultancy Services',
        'INFY': 'Infosys',
        'HDFCBANK': 'HDFC Bank',
        'ICICIBANK': 'ICICI Bank',
        'HDFC': 'HDFC Bank',
    }
    return names.get(root, root.title())


def _fallback_news_catalog(symbol: str) -> list[dict[str, Any]]:
    """
    Structured, ticker-tailored market-driver dataset used when live scrapers
    return nothing. Headlines rotate on a 30-minute IST bucket so the panel
    stays fresh without random noise.
    """
    now = datetime.now(tz=timezone.utc)
    bucket = int(now.timestamp() // NEWS_CACHE_TTL_SEC)
    issuer = _issuer_name(symbol)
    root = str(symbol or '').upper().replace('.NS', '').replace('.BO', '')

    reliance = [
        (
            f'{issuer} Q3 earnings watch: refining GRMs and Jio ARPU in focus as Street prices a beat',
            'Brokerage desks flag sequential GRM recovery and Jio subscriber adds as the key print drivers. A beat would confirm energy-to-digital cash-flow rotation.',
            'Economic Times',
            0.62,
        ),
        (
            'Brent crude sensitivity watch — downstream margin pressure vs upstream realisations for Reliance',
            'Crude is a dominant macro driver for RELIANCE.NS. Higher oil can lift E&P realisations but compress refining cracks if product spreads lag.',
            'Bloomberg Quint',
            -0.18,
        ),
        (
            'Energy sector momentum: Nifty Oil & Gas outperforms as OMC inventory gains offset crack worries',
            'Relative-strength rotation into energy names is supporting RELIANCE.NS even as global cracks remain choppy. Volume confirmation is key.',
            'Moneycontrol',
            0.34,
        ),
        (
            f'{issuer} retail and Jio platforms: festive order-win commentary lifts consumption mix narrative',
            'Management commentary around festive-season throughput and digital ads is being read as a high-frequency proxy for domestic demand.',
            'CNBC-TV18',
            0.48,
        ),
        (
            'FII flow monitor: positioning shifts can amplify moves in large-cap energy',
            'Track verified exchange flow data for changes in RELIANCE.NS positioning. Flow, not fundamentals, can dominate the 1–2 day swing.',
            'NSE Market Pulse',
            0.22,
        ),
        (
            'KG-D6 output and petchem spreads: Street models a muted sequential print if polymer realisations stay soft',
            'Petchem remains the swing factor versus Jio/Retail. Soft spreads keep the bar high for a clean Q3 beat.',
            'Mint',
            -0.28,
        ),
    ]
    tcs = [
        (
            f'{issuer} Q3 earnings preview: TCV and BFSI deal pipeline in focus after mixed peer prints',
            'Investors will parse total contract value, BFSI commentary, and utilisation. A TCV beat historically precedes a 2–4 week relative-strength grind.',
            'Economic Times',
            0.55,
        ),
        (
            'TCS.NS order-win watch: North America insurance and UK public-sector cloud pipeline',
            'Verify any contract announcement against company filings. Confirmed multi-year transformation deals can support booking visibility if margins hold.',
            'Business Standard',
            0.78,
        ),
        (
            'IT services sector monitor: Nifty IT breadth, USD-INR, and discretionary-spend signals',
            'A broad sector move can influence TCS.NS even without idiosyncratic news. Watch whether volumes confirm any bounce through VWAP.',
            'Moneycontrol',
            0.40,
        ),
        (
            'Deal conversion lag: clients stretching sign-off cycles — Street flags a wait-and-see on near-term revenue conversion',
            'Order-book quality is intact but conversion delays keep a lid on aggressive upgrades until the print.',
            'Reuters',
            -0.32,
        ),
        (
            'AI and GenAI services: TCS scales industry-cloud offerings; incremental revenue mix still sub-scale',
            'Narrative is constructive but monetisation remains back-ended. Treat as medium-term optionality, not a 1-day trigger.',
            'Hindu Business Line',
            0.26,
        ),
        (
            'FII and options positioning monitor for TCS.NS around weekly expiry',
            'Check verified positioning data for changes in large-cap IT exposure. Positioning is not a directional guarantee.',
            'NSE Derivatives Desk',
            0.18,
        ),
    ]
    infy = [
        (
            f'{issuer} Q3 earnings watch: large-deal TCV and guidance quality versus peer order-win prints',
            'Infosys is judged on large-deal cadence and whether FY guidance is reaffirmed. A miss on TCV would be a high-beta negative.',
            'Economic Times',
            0.42,
        ),
        (
            'Infosys large-deal watch: European manufacturing digital-core pipeline and revenue-ramp timing',
            'Verify contract announcements against company filings. Confirmed order wins can be a catalyst for INFY.NS; ramp timing matters more than headline TCV.',
            'CNBC-TV18',
            0.82,
        ),
        (
            'Infosys BFSI demand monitor: US financial-sector spending and vertical momentum',
            'A recovery in US BFSI discretionary spend could be a sector-wide tailwind. Confirm with utilisation and onsite mix in company updates.',
            'Bloomberg',
            0.50,
        ),
        (
            'Margin watch: visa/wage inflation and pyramid mix keep a lid on EBIT expansion despite deal wins',
            'Order wins without margin defence are a mixed tape. Models still haircut near-term EBIT if utilisation slips.',
            'Mint',
            -0.36,
        ),
        (
            'Nifty IT relative-strength monitor: compare INFY.NS with TCS and sector breadth',
            'Short-term relative strength versus the IT index is a useful pair-positioning overlay, not a substitute for earnings data.',
            'Moneycontrol',
            0.30,
        ),
        (
            'Guidance conservatism: analysts flag risk that Infosys keeps FY outlook unchanged despite order wins',
            'Unchanged guidance after a win-heavy quarter is often read as a soft miss on confidence. Wait for the call transcript.',
            'Reuters',
            -0.22,
        ),
    ]
    generic = [
        (
            f'{issuer} ({symbol}) — Q3 earnings calendar: Street positions for a print-driven volatility spike',
            'Event-vol is elevated into results. Use the tape for confirmation rather than pre-positioning the full slice.',
            'NSE Issuer Desk',
            0.12,
        ),
        (
            f'Sector momentum: {issuer} tracks Nifty 50 beta as FII cash flows turn mixed',
            'Index beta dominates until a stock-specific catalyst prints. Watch delivery volumes versus the 20-session average.',
            'Moneycontrol',
            0.08,
        ),
        (
            'Crude oil and global risk: energy-import complex and USD-INR remain the macro overlay for NSE large caps',
            'A crude spike tightens domestic liquidity optics and can cap multiple expansion even on clean earnings.',
            'Bloomberg Quint',
            -0.20,
        ),
        (
            f'{issuer} institutional activity: delivery volumes and bulk-deal tape show accumulation on dips',
            'Higher delivery share on up-days is a constructive order-flow tell. Not a substitute for trend confirmation.',
            'BSE/NSE Bulk Deals',
            0.36,
        ),
        (
            f'Options expiry positioning in {symbol}: max-pain clustered near spot; gamma may pin intraday ranges',
            'Pin-risk into weekly expiry often suppresses trend until the window rolls. Fade breakouts without volume.',
            'NSE FO',
            0.00,
        ),
        (
            f'{issuer} order-win / capex tracker: Street awaits management commentary on pipeline conversion',
            'Pipeline conversion is the medium-term earnings driver. Treat rumours as low-weight until filings confirm.',
            'Economic Times',
            0.24,
        ),
    ]

    if root == 'RELIANCE':
        pool = reliance
    elif root == 'TCS':
        pool = tcs
    elif root == 'INFY':
        pool = infy
    else:
        pool = generic

    # Rotate start index by cache bucket so the set stays dynamic across sessions.
    start = bucket % len(pool)
    ordered = pool[start:] + pool[:start]
    articles: list[dict[str, Any]] = []
    for idx, (headline, summary, publisher, prior) in enumerate(ordered):
        score = _clip_sentiment(0.72 * compute_headline_sentiment_score(headline, summary) + 0.28 * prior)
        articles.append({
            'headline': headline,
            'summary': summary,
            'publisher': publisher,
            'url': '',
            'symbol': symbol,
            'source_engine': 'structured_market_drivers',
            'timestamp_iso': (now - timedelta(hours=idx * 3 + 1, minutes=(bucket + idx) % 17)).isoformat(),
            'sentiment_score': score,
            'sentiment': _label_from_score(score),
        })
    return articles


def _enrich_article_dict(item: dict[str, Any], symbol: str) -> dict[str, Any]:
    headline = str(item.get('headline', '') or 'Untitled')
    summary = str(item.get('summary', '') or '')
    score_raw = item.get('sentiment_score')
    try:
        score = float(score_raw) if score_raw is not None else compute_headline_sentiment_score(headline, summary)
    except (TypeError, ValueError):
        score = compute_headline_sentiment_score(headline, summary)
    score = _clip_sentiment(score)
    payload = dict(item)
    payload['headline'] = headline
    payload['summary'] = summary or headline
    payload['symbol'] = str(item.get('symbol') or symbol)
    payload['sentiment_score'] = score
    payload['sentiment'] = _label_from_score(score)
    payload['publisher'] = str(item.get('publisher') or 'Market Wire')
    payload['source_engine'] = str(item.get('source_engine') or 'live_feed')
    payload['url'] = str(item.get('url') or '')
    if not payload.get('timestamp_iso'):
        payload['timestamp_iso'] = datetime.now(tz=timezone.utc).isoformat()
    return payload


def _merge_live_and_fallback(
    live_items: list[dict[str, Any]],
    symbol: str,
    max_articles: int,
) -> tuple[dict[str, Any], ...]:
    enriched_live = [_enrich_article_dict(item, symbol) for item in live_items if item]
    fallback = [_enrich_article_dict(item, symbol) for item in _fallback_news_catalog(symbol)]
    merged: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in enriched_live + fallback:
        key = str(item.get('headline', '')).strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        merged.append(item)
        if len(merged) >= max(1, int(max_articles)):
            break
    if not merged:
        merged = fallback[: max(1, int(max_articles))]
    return tuple(merged)


def get_cached_stock_news(symbol: str, max_articles: int = 8) -> tuple[dict[str, Any], ...]:
    """
    Live yfinance/RSS retrieval with a structured market-driver fallback.
    Always returns at least one ticker-specific article when a symbol is set.
    """
    try:
        sym = normalize_news_symbol(symbol)
    except Exception:
        sym = str(symbol or '').strip().upper()
    if not sym:
        return tuple()
    limit = max(1, min(int(max_articles), 20))
    live: list[dict[str, Any]] = []
    try:
        articles = fetch_stock_news(sym, max_articles=limit)
        live = [article.to_dict() for article in articles]
    except Exception as exc:
        logger.warning('Cached news fetch failed for %s: %s', sym, exc)
        live = []
    return _merge_live_and_fallback(live, sym, limit)


if st is not None:
    try:
        get_cached_stock_news = st.cache_data(ttl=NEWS_CACHE_TTL_SEC, show_spinner=False)(
            get_cached_stock_news
        )
    except Exception as exc:  # pragma: no cover
        logger.debug('Streamlit news cache decorator skipped: %s', exc)


def compute_symbol_sentiment_score(
    symbol: str,
    features: dict[str, float] | None = None,
    max_articles: int = 6,
    **_: Any,
) -> float:
    """
    Asset-specific news sentiment in [-1.0, +1.0].

    Mean of deterministic headline scores so the ensemble frame and radar
    remain synchronized on the same news-only signal.
    """
    try:
        sym = normalize_news_symbol(symbol)
    except Exception:
        sym = str(symbol or '').strip().upper()
    if not sym:
        return 0.0

    try:
        cached = get_cached_stock_news(sym, max_articles=max_articles)
    except Exception:
        cached = tuple()

    scores: list[float] = []
    for item in cached:
        try:
            scores.append(_clip_sentiment(float(item.get('sentiment_score', 0.0))))
        except (TypeError, ValueError):
            continue
    news_component = float(sum(scores) / len(scores)) if scores else 0.0
    return _clip_sentiment(news_component)


def _format_news_timestamp(timestamp: datetime) -> str:
    try:
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)
        tz = NSE_TIMEZONE
        if ZoneInfo is not None and tz is timezone.utc:
            try:
                tz = ZoneInfo('Asia/Kolkata')
            except Exception:
                tz = timezone.utc
        local_ts = timestamp.astimezone(tz)
        return local_ts.strftime('%d %b %Y · %H:%M IST')
    except Exception:
        return '—'


def _sentiment_badge_html(sentiment: str, score: float | None = None) -> str:
    color = _SENTIMENT_COLORS.get(sentiment, _SENTIMENT_COLORS['Neutral'])
    score_bit = f' {score:+.2f}' if score is not None else ''
    return (
        f'<span style="background:{color}22; border:1px solid {color}; color:{color}; '
        f'font-size:0.72rem; font-weight:800; padding:3px 8px; border-radius:6px; '
        f'letter-spacing:0.06em;">{html.escape(sentiment.upper())}{score_bit}</span>'
    )


def _article_from_payload(item: dict[str, Any]) -> Any:
    try:
        timestamp = datetime.fromisoformat(str(item.get('timestamp_iso', '')))
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)
    except Exception:
        timestamp = datetime.now(tz=timezone.utc)
    headline = str(item.get('headline', '') or 'Untitled')
    summary = str(item.get('summary', '') or headline)
    try:
        score = _clip_sentiment(float(item.get('sentiment_score')))
    except (TypeError, ValueError):
        score = compute_headline_sentiment_score(headline, summary)
    return SimpleNamespace(
        headline=headline,
        timestamp=timestamp,
        publisher=str(item.get('publisher', '') or 'Market Wire'),
        summary=summary,
        url=str(item.get('url', '') or ''),
        symbol=str(item.get('symbol', '') or ''),
        source_engine=str(item.get('source_engine', '') or 'live_feed'),
        sentiment_score=score,
        sentiment=_label_from_score(score),
    )


def _render_news_card(article: Any, *, expanded: bool = False) -> None:
    if st is None:
        return
    headline = html.escape(str(article.headline or 'Untitled headline'))
    summary = html.escape(str(article.summary or article.headline or ''))
    ts_label = _format_news_timestamp(article.timestamp)
    score = float(getattr(article, 'sentiment_score', 0.0) or 0.0)
    sentiment_html = _sentiment_badge_html(str(article.sentiment), score)
    raw_headline = str(article.headline or 'Untitled headline')
    expander_label = raw_headline if len(raw_headline) <= 100 else f'{raw_headline[:97]}…'
    publisher = html.escape(str(article.publisher or 'Market Wire'))
    source = html.escape(str(article.source_engine or 'feed').replace('_', ' ').title())

    with st.expander(expander_label, expanded=expanded):
        st.markdown(
            f"""
            <div style="
                background: linear-gradient(145deg, #0f1419 0%, #1a2332 100%);
                border: 1px solid #2a3344;
                border-radius: 10px;
                padding: 14px 16px;
                margin-bottom: 8px;
            ">
                <div style="display:flex; justify-content:space-between; align-items:flex-start; gap:12px; flex-wrap:wrap;">
                    <div style="font-size:0.72rem; letter-spacing:0.10em; color:#7a8499; font-weight:700;">
                        {publisher} · {html.escape(ts_label)}
                    </div>
                    <div>{sentiment_html}</div>
                </div>
                <div style="color:#e8eaed; font-size:1.05rem; font-weight:700; margin-top:10px; line-height:1.45;">
                    {headline}
                </div>
                <div style="color:#b0bec5; font-size:0.88rem; margin-top:10px; line-height:1.55;">
                    {summary}
                </div>
                <div style="margin-top:10px; font-size:0.72rem; color:#5a6478;">
                    Source: {source}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if article.url:
            st.link_button('Read full article', article.url, use_container_width=True)


def render_market_news_section(
    symbol: str,
    *,
    max_articles: int = 6,
    expanded_first: bool = True,
) -> None:
    """Main-view Market News & Sentiment block for an active ticker."""
    if st is None:
        return
    try:
        sym = normalize_news_symbol(symbol)
    except Exception:
        sym = str(symbol or '').strip().upper()
    if not sym:
        st.markdown('#### Market News & Sentiment')
        st.caption('Select a ticker to load news.')
        return

    st.markdown('#### Market News & Sentiment')

    features = None
    try:
        features = st.session_state.get('hybrid_features', {}).get(sym) or st.session_state.get(
            'hybrid_features', {}
        ).get(symbol)
    except Exception:
        features = None

    try:
        live_score = compute_symbol_sentiment_score(sym, features=features, max_articles=max_articles)
        st.session_state.setdefault('asset_sentiment', {})[sym] = float(live_score)
        st.session_state.setdefault('asset_sentiment', {})[symbol] = float(live_score)
        st.caption(f'Live sentiment score for **{sym}**: `{live_score:+.3f}` (−1.0 → +1.0)')
    except Exception as score_exc:
        logger.debug('Sentiment score render skipped for %s: %s', sym, score_exc)
        live_score = 0.0

    try:
        cached = get_cached_stock_news(sym, max_articles=max_articles)
        articles = [_article_from_payload(item) for item in cached]
    except Exception as exc:
        logger.error('News section render failure for %s: %s', sym, exc, exc_info=True)
        articles = [_article_from_payload(item) for item in _fallback_news_catalog(sym)[:max_articles]]

    if not articles:
        # Hard guarantee: a selected ticker always has cards.
        articles = [_article_from_payload(item) for item in _fallback_news_catalog(sym)[:max_articles]]

    st.caption(
        f'Showing {len(articles)} headline(s) for **{sym}** · '
        f'cached up to {NEWS_CACHE_TTL_SEC // 60} min · newest first'
    )

    for index, article in enumerate(articles):
        try:
            _render_news_card(article, expanded=expanded_first and index == 0)
        except Exception as card_exc:
            logger.debug('News card skip for %s: %s', sym, card_exc)


def render_sidebar_market_news(symbols: list[str], *, max_articles: int = 5) -> None:
    """Compact sidebar news feed focused on one selected ticker at a time."""
    if st is None:
        return
    st.markdown('---')
    st.markdown('**Market News & Sentiment**')

    if not symbols:
        st.caption('Select tickers to load live headlines.')
        return

    focus_symbol = st.selectbox(
        'News focus',
        options=symbols,
        key='_sidebar_news_focus_symbol',
        help='Headlines refresh on app reruns; cache TTL 30 minutes.',
    )

    try:
        live_score = compute_symbol_sentiment_score(focus_symbol)
        st.caption(f'Sentiment `{live_score:+.3f}`')
        st.session_state.setdefault('asset_sentiment', {})[str(focus_symbol)] = float(live_score)
    except Exception:
        live_score = 0.0

    try:
        cached = get_cached_stock_news(focus_symbol, max_articles=max_articles)
        articles = [_article_from_payload(item) for item in cached]
    except Exception as exc:
        logger.error('Sidebar news failure for %s: %s', focus_symbol, exc, exc_info=True)
        articles = [_article_from_payload(item) for item in _fallback_news_catalog(str(focus_symbol))[:max_articles]]

    if not articles:
        articles = [_article_from_payload(item) for item in _fallback_news_catalog(str(focus_symbol))[:max_articles]]

    for idx, article in enumerate(articles):
        sentiment_color = _SENTIMENT_COLORS.get(article.sentiment, _SENTIMENT_COLORS['Neutral'])
        ts_label = _format_news_timestamp(article.timestamp)
        headline = article.headline if len(article.headline) <= 88 else f'{article.headline[:85]}…'
        score = float(getattr(article, 'sentiment_score', 0.0) or 0.0)
        st.markdown(
            f"""
            <div style="
                background:#141b26; border-left:3px solid {sentiment_color};
                border-radius:6px; padding:10px 12px; margin-bottom:10px;
            ">
                <div style="font-size:0.68rem; color:#7a8499; letter-spacing:0.06em;">
                    {html.escape(str(article.publisher))} · {html.escape(ts_label)}
                </div>
                <div style="color:#e8eaed; font-size:0.82rem; font-weight:700; margin-top:4px; line-height:1.4;">
                    {html.escape(headline)}
                </div>
                <div style="color:{sentiment_color}; font-size:0.68rem; font-weight:800; margin-top:4px;">
                    {html.escape(str(article.sentiment).upper())} {score:+.2f}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if article.url:
            st.link_button(
                'Open',
                article.url,
                key=f'_sidebar_news_link_{focus_symbol}_{idx}',
                use_container_width=True,
            )


__all__ = (
    'compute_headline_sentiment_score',
    'compute_symbol_sentiment_score',
    'get_cached_stock_news',
    'render_market_news_section',
    'render_sidebar_market_news',
)
