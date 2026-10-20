"""Live quote streaming utilities for near real-time prediction."""
import logging
import time
from typing import Generator

from live_data.nse_client import fetch_live_quote
from config.nse_timing import is_nse_market_open

logger = logging.getLogger(__name__)


def live_quote_stream(symbols: list[str], interval_seconds: int = 60) -> Generator[dict[str, dict], None, None]:
    """Yield live quotes for the provided symbols while the NSE market is open."""
    while True:
        if not is_nse_market_open():
            logger.info('NSE market is closed. Waiting to resume live stream.')
            time.sleep(interval_seconds)
            continue

        snapshot = {}
        for symbol in symbols:
            try:
                snapshot[symbol] = fetch_live_quote(symbol)
            except Exception as exc:
                logger.warning('Failed to fetch live quote for %s: %s', symbol, exc)
        yield snapshot
        time.sleep(interval_seconds)
