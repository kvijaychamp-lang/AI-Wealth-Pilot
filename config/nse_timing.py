"""NSE market timing and session helpers."""
from datetime import datetime, time


NSE_OPEN_TIME = time(hour=9, minute=15)
NSE_CLOSE_TIME = time(hour=15, minute=30)
WEEKEND_DAYS = {5, 6}


def is_nse_market_open(timestamp: datetime | None = None) -> bool:
    """Return whether the NSE is open for trading at the supplied timestamp."""
    if timestamp is None:
        timestamp = datetime.now()

    if timestamp.weekday() in WEEKEND_DAYS:
        return False

    current_time = timestamp.time()
    return NSE_OPEN_TIME <= current_time <= NSE_CLOSE_TIME


def time_until_market_open(timestamp: datetime | None = None) -> int:
    """Return seconds until the next NSE market open."""
    if timestamp is None:
        timestamp = datetime.now()

    if is_nse_market_open(timestamp):
        return 0

    next_open = datetime(
        year=timestamp.year,
        month=timestamp.month,
        day=timestamp.day,
        hour=NSE_OPEN_TIME.hour,
        minute=NSE_OPEN_TIME.minute,
    )

    if timestamp.time() > NSE_CLOSE_TIME or timestamp.weekday() in WEEKEND_DAYS:
        next_open = next_open.replace(day=timestamp.day + 1)

    return int((next_open - timestamp).total_seconds())
