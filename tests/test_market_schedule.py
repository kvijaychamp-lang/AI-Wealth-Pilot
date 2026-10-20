import datetime
from zoneinfo import ZoneInfo

import pytest

import utils.market_schedule as ms

ASIA = ZoneInfo('Asia/Kolkata')


def make_dt(date_str: str, time_str: str = '10:00:00') -> datetime.datetime:
    dt = datetime.datetime.fromisoformat(f"{date_str}T{time_str}")
    return dt.replace(tzinfo=ASIA)


def test_active_during_open(monkeypatch):
    # Simulate a weekday with market open window
    test_dt = make_dt('2026-06-09', '10:00:00')  # Tuesday

    open_dt = make_dt('2026-06-09', '09:15:00')
    close_dt = make_dt('2026-06-09', '15:30:00')

    monkeypatch.setattr(ms, 'is_weekend', lambda d: False)
    monkeypatch.setattr(ms, 'is_market_holiday', lambda d=None: False)
    monkeypatch.setattr(ms, 'get_market_schedule', lambda d: {'open': open_dt, 'close': close_dt})

    assert ms.is_market_active(test_dt) is True


def test_inactive_weekend():
    # Saturday should be inactive
    sat = datetime.datetime(2026, 6, 6, 12, 0, tzinfo=ASIA)  # Saturday
    assert ms.is_market_active(sat) is False


def test_inactive_holiday(monkeypatch):
    test_dt = make_dt('2026-06-09', '10:00:00')
    # Force holiday detection
    monkeypatch.setattr(ms, 'is_weekend', lambda d: False)
    monkeypatch.setattr(ms, 'is_market_holiday', lambda d=None: True)
    assert ms.is_market_active(test_dt) is False


def test_inactive_before_open(monkeypatch):
    test_dt = make_dt('2026-06-09', '08:00:00')
    open_dt = make_dt('2026-06-09', '09:15:00')
    close_dt = make_dt('2026-06-09', '15:30:00')

    monkeypatch.setattr(ms, 'is_weekend', lambda d: False)
    monkeypatch.setattr(ms, 'is_market_holiday', lambda d=None: False)
    monkeypatch.setattr(ms, 'get_market_schedule', lambda d: {'open': open_dt, 'close': close_dt})

    assert ms.is_market_active(test_dt) is False


def test_inactive_after_close(monkeypatch):
    test_dt = make_dt('2026-06-09', '16:00:00')
    open_dt = make_dt('2026-06-09', '09:15:00')
    close_dt = make_dt('2026-06-09', '15:30:00')

    monkeypatch.setattr(ms, 'is_weekend', lambda d: False)
    monkeypatch.setattr(ms, 'is_market_holiday', lambda d=None: False)
    monkeypatch.setattr(ms, 'get_market_schedule', lambda d: {'open': open_dt, 'close': close_dt})

    assert ms.is_market_active(test_dt) is False
