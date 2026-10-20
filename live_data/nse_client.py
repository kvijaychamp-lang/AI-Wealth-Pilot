"""Live NSE market data client and helpers."""
import json
import random
import time
from pathlib import Path
from typing import Any

import requests
try:
    import yfinance as yf
except Exception:
    yf = None

from utils.logger import get_api_logger, get_live_data_logger, log_api_call, log_live_data_event

logger = get_api_logger()

LIVE_QUOTE_ENDPOINT = 'https://www.nseindia.com/api/quote-equity?symbol={symbol}'

# Rotating User-Agents to defeat scraping blocks and 403 errors
ROTATING_USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.131 Safari/537.36',
]

def _get_random_user_agent() -> str:
    """Return a random User-Agent from rotating pool."""
    return random.choice(ROTATING_USER_AGENTS)


def fetch_live_quote(symbol: str) -> dict[str, Any]:
    """Fetch live quote details for a given NSE symbol with cache-bypass.

    Attempts the NSE endpoint first with rotating User-Agents and dynamic
    cache-busting parameters. If the request fails (for example a 403 block),
    best-effort fallback to Yahoo Finance via `yfinance` is attempted.
    """
    # Rotating User-Agent defeats 403 scraping blocks
    user_agent = _get_random_user_agent()
    headers = {
        'User-Agent': user_agent,
        'Accept': 'application/json',
        'Cache-Control': 'no-cache, no-store, must-revalidate',
        'Pragma': 'no-cache',
        'Expires': '0',
    }
    
    base_url = LIVE_QUOTE_ENDPOINT.format(symbol=symbol.replace('.NS', ''))
    # Cache-busting timestamp parameter to defeat downstream caching
    cache_buster = str(int(time.time() * 1000))  # Millisecond timestamp
    url = f"{base_url}&_t={cache_buster}"

    try:
        log_api_call(url, 'GET', symbol=symbol)
        response = requests.get(url, headers=headers, timeout=2)
        response.raise_for_status()
        payload = response.json()

        log_api_call(url, 'GET', status_code=response.status_code, symbol=symbol)
        log_live_data_event('QUOTE_FETCHED', symbol, status='SUCCESS')
        logger.debug('Fetched live data for %s', symbol)
        return payload

    except requests.RequestException as e:
        status_code = getattr(e.response, 'status_code', None)
        log_api_call(url, 'GET', status_code=status_code, symbol=symbol, error=str(e))
        log_live_data_event('QUOTE_FETCH_FAILED', symbol, error=str(e))
        logger.error('Failed to fetch live data for %s (HTTP %s): %s', symbol, status_code, e)

        # Fallback: try Yahoo Finance via yfinance if available
        if yf is not None:
            try:
                ticker = yf.Ticker(symbol)
                
                # Try fast_info first for high-frequency live price (updates multiple times per minute)
                price = ticker.fast_info.get('lastPrice')
                if price is not None and price > 0:
                    import pandas as pd
                    # Use current local time as timestamp for the live tick
                    timestamp = pd.Timestamp.now().isoformat()
                    fallback_payload = {'priceInfo': {'lastPrice': float(price), 'lastUpdatedTime': timestamp}}
                    log_live_data_event('QUOTE_FETCHED_FALLBACK', symbol, status='SUCCESS', source='yahoo_fast_info')
                    logger.info('Used Yahoo Finance fast_info fallback for %s: ₹%.2f', symbol, price)
                    return fallback_payload
                
                # Fallback to history if fast_info fails: use intraday 5m over 5d via yf.download
                df = yf.download(tickers=symbol, period='5d', interval='5m', progress=False, auto_adjust=False)
                if not df.empty:
                    df = df.dropna()
                    df = df.tail(100).reset_index()
                    if 'Datetime' not in df.columns:
                        if 'Date' in df.columns:
                            df.rename(columns={'Date': 'Datetime'}, inplace=True)
                        else:
                            df['Datetime'] = df.iloc[:, 0]
                    last_row = df.iloc[-1]
                    price = float(last_row['Close'])
                    timestamp = pd.to_datetime(last_row['Datetime']).isoformat()
                    fallback_payload = {'priceInfo': {'lastPrice': price, 'lastUpdatedTime': timestamp}}
                    log_live_data_event('QUOTE_FETCHED_FALLBACK', symbol, status='SUCCESS', source='yahoo_history')
                    logger.info('Used Yahoo Finance history fallback for %s: ₹%.2f', symbol, price)
                    return fallback_payload
            except Exception as yf_err:
                logger.warning('Yahoo fallback failed for %s: %s', symbol, yf_err)

        # If all fallbacks fail, propagate original exception
        raise


def save_quote(symbol: str, data: dict[str, Any]) -> Path:
    """Save live quote JSON locally for analysis or debugging."""
    output_dir = Path('data') / 'live'
    output_dir.mkdir(parents=True, exist_ok=True)
    file_path = output_dir / f'{symbol.replace(".", "_")}.json'
    file_path.write_text(json.dumps(data, indent=2))
    return file_path
