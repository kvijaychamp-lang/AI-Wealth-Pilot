"""Live market data streaming for Indian NSE stocks."""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Iterable
import numpy as np
import pandas as pd

import requests
from requests.exceptions import RequestException

from live_data.nse_client import LIVE_QUOTE_ENDPOINT

logger = logging.getLogger(__name__)

DEFAULT_FETCH_INTERVAL = 10
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_BACKOFF = 3
DEFAULT_RECONNECT_DELAY = 15


@dataclass
class LiveQuote:
    """A structured live quote stored in memory."""
    symbol: str
    timestamp: str
    price: float | None
    raw: dict[str, Any]


class LiveQuoteStore:
    """In-memory storage for the latest live quotes."""

    def __init__(self) -> None:
        self._quotes: dict[str, LiveQuote] = {}
        self._lock = threading.Lock()

    def update(self, quote: LiveQuote) -> None:
        """Store or update the latest live quote for a symbol."""
        with self._lock:
            self._quotes[quote.symbol] = quote
            logger.debug('Updated quote store for %s', quote.symbol)

    def get(self, symbol: str) -> LiveQuote | None:
        """Retrieve the latest quote for a single symbol."""
        with self._lock:
            return self._quotes.get(symbol)

    def all_quotes(self) -> dict[str, LiveQuote]:
        """Return a snapshot of all currently stored quotes."""
        with self._lock:
            return dict(self._quotes)


class LiveQuoteAPIClient:
    """API client for fetching live NSE quotes with session reuse."""

    def __init__(self, timeout: int = 10) -> None:
        self.session = requests.Session()
        self.timeout = timeout
        self.headers = {
            'User-Agent': 'Mozilla/5.0',
            'Accept': 'application/json',
        }
        self.session.headers.update(self.headers)

    def fetch(self, symbol: str) -> dict[str, Any]:
        """Fetch live quote JSON for a given symbol."""
        from live_data.nse_client import fetch_live_quote
        return fetch_live_quote(symbol)

    def reset_session(self) -> None:
        """Reset the HTTP session after connection failures."""
        self.session.close()
        self.session = requests.Session()
        self.session.headers.update(self.headers)
        logger.warning('Live quote API session reset')


class LiveDataStreamer:
    """Continuously stream live prices for a set of NSE symbols."""

    def __init__(
        self,
        symbols: Iterable[str],
        fetch_interval: int = DEFAULT_FETCH_INTERVAL,
        max_retries: int = DEFAULT_MAX_RETRIES,
        retry_backoff: int = DEFAULT_RETRY_BACKOFF,
        reconnect_delay: int = DEFAULT_RECONNECT_DELAY,
        api_client: LiveQuoteAPIClient | None = None,
    ) -> None:
        self.symbols = list(symbols)
        self.fetch_interval = fetch_interval
        self.max_retries = max_retries
        self.retry_backoff = retry_backoff
        self.reconnect_delay = reconnect_delay
        self.api_client = api_client or LiveQuoteAPIClient()
        self.store = LiveQuoteStore()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._callbacks: list[Callable[[LiveQuote], None]] = []

    def start(self) -> None:
        """Start the live streaming worker thread."""
        if self._thread and self._thread.is_alive():
            logger.warning('LiveDataStreamer already running')
            return

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        logger.info('Started LiveDataStreamer for symbols: %s', self.symbols)

    def stop(self) -> None:
        """Signal the streamer to stop gracefully."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=self.fetch_interval + 5)
            logger.info('Stopped LiveDataStreamer')

    def subscribe(self, callback: Callable[[LiveQuote], None]) -> None:
        """Register a callback to receive updated quotes."""
        self._callbacks.append(callback)

    def _run(self) -> None:
        """Main loop that fetches quotes for all configured symbols."""
        while not self._stop_event.is_set():
            all_failed = True
            for symbol in self.symbols:
                if self._stop_event.is_set():
                    break
                try:
                    quote = self._fetch_with_retry(symbol)
                    self.store.update(quote)
                    self._notify_subscribers(quote)
                    all_failed = False
                except Exception as exc:
                    logger.warning('Unable to update symbol %s: %s', symbol, exc)
                    # If fetch failed, try to generate a simulated tick immediately
                    last = self.store.get(symbol)
                    if last and last.price is not None:
                        base = float(last.price)
                        # small random walk noise scaled to price magnitude
                        noise = np.random.normal(loc=0.0, scale=0.0008)
                        simulated_price = max(0.0, base * (1.0 + noise))
                        now_ts = pd.Timestamp.now().isoformat()
                        sim_quote = LiveQuote(symbol=symbol, timestamp=now_ts, price=simulated_price, raw={'simulated': True, 'reason': 'fetch_failed'})
                        self.store.update(sim_quote)
                        self._notify_subscribers(sim_quote)

            # Sleep briefly before next fetch cycle (respects fetch_interval)
            time.sleep(max(self.fetch_interval, 0.1))

    def _fetch_with_retry(self, symbol: str) -> LiveQuote:
        """Fetch a single symbol quote with retry and backoff."""
        last_exception: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                payload = self.api_client.fetch(symbol)
                quote = self._parse_quote(symbol, payload)
                logger.info('Fetched live quote for %s on attempt %d', symbol, attempt)
                return quote
            except RequestException as error:
                logger.warning('Network error fetching %s: %s', symbol, error)
                last_exception = error
            except ValueError as error:
                logger.warning('Data error for %s: %s', symbol, error)
                last_exception = error
            except Exception as error:
                logger.exception('Unexpected error fetching %s', symbol)
                last_exception = error

            if attempt < self.max_retries:
                wait_seconds = self.retry_backoff * attempt
                logger.info('Retrying %s in %d seconds', symbol, wait_seconds)
                time.sleep(wait_seconds)
                self.api_client.reset_session()

        raise RuntimeError(f'Failed to fetch live quote for {symbol}') from last_exception

    def _parse_quote(self, symbol: str, payload: dict[str, Any]) -> LiveQuote:
        """Parse raw JSON payload into a LiveQuote object."""
        if 'priceInfo' in payload and payload['priceInfo']:
            price_data = payload['priceInfo']
            timestamp = price_data.get('lastUpdatedTime', '')
            price = self._extract_price(price_data)
            return LiveQuote(symbol=symbol, timestamp=timestamp, price=price, raw=payload)

        raise ValueError('Unexpected live quote payload format')

    @staticmethod
    def _extract_price(price_data: dict[str, Any]) -> float | None:
        """Extract the most reliable live price field from the payload."""
        for key in ('lastPrice', 'closePrice', 'lastTradedPrice'):
            if key in price_data and price_data[key] is not None:
                try:
                    return float(price_data[key])
                except (TypeError, ValueError):
                    continue
        return None

    def _notify_subscribers(self, quote: LiveQuote) -> None:
        """Notify registered callbacks whenever a quote is updated."""
        for callback in self._callbacks:
            try:
                callback(quote)
            except Exception:
                logger.exception('Subscriber callback failed for %s', quote.symbol)

    def get_latest_quotes(self) -> dict[str, LiveQuote]:
        """Return the latest in-memory quote snapshot."""
        return self.store.all_quotes()


def run_example() -> None:
    """Example usage of the LiveDataStreamer."""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
    symbols = ['RELIANCE.NS', 'TCS.NS', 'INFY.NS']
    streamer = LiveDataStreamer(symbols=symbols, fetch_interval=15)

    def print_quote(quote: LiveQuote) -> None:
        print(f'{quote.symbol} @ {quote.timestamp}: {quote.price}')

    streamer.subscribe(print_quote)
    streamer.start()

    try:
        time.sleep(60)
    finally:
        streamer.stop()


if __name__ == '__main__':
    run_example()
