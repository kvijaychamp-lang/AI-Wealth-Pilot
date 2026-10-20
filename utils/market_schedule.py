"""Market timing utilities for NSE using pandas_market_calendars."""
from __future__ import annotations

import logging
import time
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo
from typing import Callable, Iterable

import pandas as pd
import pandas_market_calendars as mcal

logger = logging.getLogger(__name__)
NSE_TIMEZONE = ZoneInfo('Asia/Kolkata')
NSE_CALENDAR = mcal.get_calendar('NSE')


def _localize_timestamp(timestamp: pd.Timestamp) -> datetime:
    """Convert a pandas timestamp to a timezone-aware datetime in NSE local time."""
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize('UTC')
    return timestamp.tz_convert(NSE_TIMEZONE).to_pydatetime()


def is_weekend(target_date: date | None = None) -> bool:
    """Return True when the given date is a Saturday or Sunday."""
    if target_date is None:
        target_date = datetime.now(tz=NSE_TIMEZONE).date()
    result = target_date.weekday() >= 5
    logger.debug('Checked weekend for %s: %s', target_date, result)
    return result


def get_market_schedule(target_date: date) -> dict[str, datetime]:
    """Return NSE market open and close times for the requested date."""
    schedule = NSE_CALENDAR.schedule(
        start_date=target_date.isoformat(), end_date=target_date.isoformat()
    )

    if schedule.empty:
        logger.warning('No market schedule for %s', target_date)
        raise ValueError(f'NSE is closed on {target_date}')

    row = schedule.iloc[0]
    market_open = _localize_timestamp(row['market_open'])
    market_close = _localize_timestamp(row['market_close'])
    logger.debug('Market schedule for %s open=%s close=%s', target_date, market_open, market_close)
    return {'open': market_open, 'close': market_close}


def is_market_holiday(target_date: date | None = None) -> bool:
    """Detect if the given date is a holiday on the NSE calendar."""
    if target_date is None:
        target_date = datetime.now(tz=NSE_TIMEZONE).date()

    try:
        get_market_schedule(target_date)
        return False
    except ValueError:
        logger.info('Detected market holiday or closure on %s', target_date)
        return True


def is_market_open(current: datetime | None = None) -> bool:
    """Return True when the NSE market is currently open."""
    if current is None:
        current = datetime.now(tz=NSE_TIMEZONE)
    else:
        current = current.astimezone(NSE_TIMEZONE)

    if is_weekend(current.date()):
        logger.info('Market is closed because today is weekend: %s', current.date())
        return False

    try:
        schedule = get_market_schedule(current.date())
    except ValueError:
        return False

    open_time = schedule['open']
    close_time = schedule['close']
    is_open = open_time <= current < close_time
    logger.debug('Market open check at %s: %s', current, is_open)
    return is_open


def is_market_active(current: datetime | None = None) -> bool:
    """Return True when today is a trading day and current time is within active session.

    This is a convenience wrapper used by orchestration code to decide whether
    live prediction loops and UI streaming should proceed. It is timezone-aware
    and uses the NSE calendar to exclude holidays and weekends.
    """
    if current is None:
        current = datetime.now(tz=NSE_TIMEZONE)
    else:
        current = current.astimezone(NSE_TIMEZONE)

    # Exclude weekends/holidays quickly
    if is_weekend(current.date()):
        logger.debug('Market not active: weekend %s', current.date())
        return False
    if is_market_holiday(current.date()):
        logger.debug('Market not active: holiday %s', current.date())
        return False

    try:
        schedule = get_market_schedule(current.date())
    except ValueError:
        logger.debug('Market not active: no schedule for %s', current.date())
        return False

    open_time = schedule['open']
    close_time = schedule['close']

    # Only allow active period between official open and close (inclusive start, exclusive end)
    active = open_time <= current < close_time
    logger.debug('Market active check at %s: %s (open=%s close=%s)', current, active, open_time, close_time)
    return active


def _next_valid_market_date(start_date: date) -> date:
    """Return the next date when NSE is open starting from the given date."""
    end_date = start_date + timedelta(days=60)
    valid_days = NSE_CALENDAR.valid_days(start_date=start_date.isoformat(), end_date=end_date.isoformat())
    if valid_days.empty:
        logger.error('No valid trading days found after %s', start_date)
        raise ValueError('Unable to find next NSE trading day')
    next_date = valid_days[0].date()
    logger.debug('Next valid market date after %s is %s', start_date, next_date)
    return next_date


def get_next_market_open(current: datetime | None = None) -> datetime:
    """Return the next NSE market opening datetime after now."""
    if current is None:
        current = datetime.now(tz=NSE_TIMEZONE)
    else:
        current = current.astimezone(NSE_TIMEZONE)

    if not is_market_holiday(current.date()):
        schedule = get_market_schedule(current.date())
        if current < schedule['open']:
            return schedule['open']
        if current < schedule['close']:
            return current

    next_date = _next_valid_market_date(current.date() + timedelta(days=1))
    schedule = get_market_schedule(next_date)
    return schedule['open']


def get_next_market_close(current: datetime | None = None) -> datetime:
    """Return the next NSE market closing datetime after now."""
    if current is None:
        current = datetime.now(tz=NSE_TIMEZONE)
    else:
        current = current.astimezone(NSE_TIMEZONE)

    try:
        schedule = get_market_schedule(current.date())
    except ValueError:
        next_open = get_next_market_open(current)
        return get_market_schedule(next_open.date())['close']

    if current < schedule['close']:
        return schedule['close']

    next_date = _next_valid_market_date(current.date() + timedelta(days=1))
    return get_market_schedule(next_date)['close']


def should_run_market_hours(current: datetime | None = None) -> bool:
    """Return True when tasks should run during NSE market hours."""
    result = is_market_open(current)
    logger.debug('Market hours run check: %s', result)
    return result


def wait_until_market_open(check_interval_seconds: int = 60) -> datetime:
    """Block until the next NSE market open time."""
    while True:
        next_open = get_next_market_open()
        now = datetime.now(tz=NSE_TIMEZONE)
        wait_seconds = (next_open - now).total_seconds()
        if wait_seconds <= 0:
            logger.info('Market is now open at %s', next_open)
            return next_open

        logger.info('Waiting %.0f seconds until market open at %s', wait_seconds, next_open)
        time.sleep(min(check_interval_seconds, max(1, wait_seconds)))


def run_during_market_hours(task: Callable[[], None], check_interval_seconds: int = 60) -> None:
    """Run a task only when NSE market hours are active."""
    while True:
        if should_run_market_hours():
            task()
        else:
            wait_until_market_open(check_interval_seconds)
        time.sleep(check_interval_seconds)
