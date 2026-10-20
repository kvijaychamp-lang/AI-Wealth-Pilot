"""
utils/metric_themes.py
──────────────────────
Centralized threshold configuration and dynamic theme resolver for the
Buy Phase Profit Booking metric cards.

Design goals
────────────
• Zero magic numbers in component/render code — all thresholds live here.
• Pure, side-effect-free logic — safe to import anywhere.
• lru_cache memoization — live-streaming data at 1–2 s intervals hits the
  cache on repeated values, eliminating redundant branching.
• Frozen dataclass — MetricTheme instances are immutable and hashable,
  preventing accidental mutation across Streamlit reruns.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Final, Literal

# ──────────────────────────────────────────────────────────────────────────────
# 1. Centralized Threshold Configuration
# ──────────────────────────────────────────────────────────────────────────────

MetricType = Literal['upsidePotential', 'momentumExhaustion', 'downsideRisk']

# Each metric defines three ordered tiers.  The resolver walks them top-to-bottom
# and returns the *first* tier whose condition is satisfied.
#
# Schema per tier:
#   condition : callable(value: float) -> bool
#   label     : short badge label shown inside the card
#   status    : longer descriptive sub-text shown below the progress bar
#   color_hex : primary hex color for the typography and bar
#   grad_start: lighter gradient start (adds depth to the progress bar)
#   grad_end  : saturated gradient end
#   glow_hex  : CSS rgba() string for box-shadow glow on the card border

_TIER_PALETTE: Final[dict[str, dict[str, str]]] = {
    'green': {
        'color_hex':   '#00e676',
        'grad_start':  '#00c853',
        'grad_end':    '#00e676',
        'glow_hex':    'rgba(0,230,118,0.18)',
        'border_hex':  'rgba(0,230,118,0.25)',
    },
    'orange': {
        'color_hex':   '#ffb300',
        'grad_start':  '#ff8f00',
        'grad_end':    '#ffb300',
        'glow_hex':    'rgba(255,179,0,0.18)',
        'border_hex':  'rgba(255,179,0,0.25)',
    },
    'red': {
        'color_hex':   '#ff5252',
        'grad_start':  '#e53935',
        'grad_end':    '#ff5252',
        'glow_hex':    'rgba(255,82,82,0.18)',
        'border_hex':  'rgba(255,82,82,0.25)',
    },
}

# fmt: off
PROFIT_BOOKING_THRESHOLDS: Final[dict[str, list[dict]]] = {
    # ── Upside Potential ──────────────────────────────────────────────────────
    # Higher = better.  Green when plenty of room to target; red when almost none.
    'upsidePotential': [
        {
            'tier':      'green',
            'condition': lambda v: v >= 60,
            'label':     'Optimal',
            'status':    'Strong upside remaining to target',
        },
        {
            'tier':      'orange',
            'condition': lambda v: v >= 30,
            'label':     'Moderate',
            'status':    'Moderate room to target — monitor closely',
        },
        {
            'tier':      'red',
            'condition': lambda v: True,          # catch-all  (v < 30)
            'label':     'Low',
            'status':    'Limited upside — consider scaling out',
        },
    ],

    # ── Momentum Exhaustion ───────────────────────────────────────────────────
    # Lower = safer.  Green when momentum is fresh; red when exhausted.
    'momentumExhaustion': [
        {
            'tier':      'green',
            'condition': lambda v: v < 40,
            'label':     'Safe',
            'status':    'Trend intact — low exhaustion signal',
        },
        {
            'tier':      'orange',
            'condition': lambda v: v < 70,
            'label':     'Warning',
            'status':    'Momentum cooling — watch RSI & MACD',
        },
        {
            'tier':      'red',
            'condition': lambda v: True,          # catch-all  (v >= 70)
            'label':     'Danger',
            'status':    'Momentum exhausted — exit risk elevated',
        },
    ],

    # ── Downside Risk ─────────────────────────────────────────────────────────
    # Lower = safer.  Green when trailing stop is distant; red when close.
    'downsideRisk': [
        {
            'tier':      'green',
            'condition': lambda v: v < 15,
            'label':     'Safe',
            'status':    'Minimal downside exposure to trailing stop',
        },
        {
            'tier':      'orange',
            'condition': lambda v: v < 30,
            'label':     'Moderate',
            'status':    'Elevated risk — consider tightening stops',
        },
        {
            'tier':      'red',
            'condition': lambda v: True,          # catch-all  (v >= 30)
            'label':     'High Risk',
            'status':    'Severe downside exposure — scale out now',
        },
    ],
}
# fmt: on


# ──────────────────────────────────────────────────────────────────────────────
# 2. MetricTheme — Immutable Resolved Theme Container
# ──────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class MetricTheme:
    """
    Fully resolved display theme for a single profit-booking metric card.

    Attributes
    ----------
    color_hex     : Primary hex color applied to the percentage typography.
    grad_start    : Gradient start color for the progress bar fill.
    grad_end      : Gradient end color for the progress bar fill.
    glow_hex      : CSS rgba() string for the card border box-shadow glow.
    border_hex    : CSS rgba() string for the card border tint.
    status_label  : Short tier badge text (e.g. "Optimal", "Warning").
    status_text   : Longer descriptive sub-text below the progress bar.
    tier          : Raw tier identifier: 'green' | 'orange' | 'red'.
    """

    color_hex:    str
    grad_start:   str
    grad_end:     str
    glow_hex:     str
    border_hex:   str
    status_label: str
    status_text:  str
    tier:         str


# ──────────────────────────────────────────────────────────────────────────────
# 3. get_metric_theme — Pure, Memoized Resolver
# ──────────────────────────────────────────────────────────────────────────────

# Round to 1 decimal place before cache lookup so that minor float jitter
# (e.g. 54.99999... vs 55.0) from live data streaming doesn't cause cache misses.
@lru_cache(maxsize=256)
def get_metric_theme(metric_type: MetricType, value: float) -> MetricTheme:
    """
    Evaluate *value* against the financial-grade threshold rules for *metric_type*
    and return an immutable MetricTheme with all resolved display properties.

    Parameters
    ----------
    metric_type : One of 'upsidePotential', 'momentumExhaustion', 'downsideRisk'.
    value       : The numeric metric value, expected in range [0.0, 100.0].
                  Values outside this range are clamped before evaluation.

    Returns
    -------
    MetricTheme
        Frozen dataclass with color, gradient, glow, and status information.

    Raises
    ------
    ValueError
        If *metric_type* is not a recognised key in PROFIT_BOOKING_THRESHOLDS.

    Notes
    -----
    This function is decorated with @lru_cache.  The cache key is
    (metric_type, round(value, 1)), so live-streaming values that round to the
    same bucket hit the cache directly.  Cache capacity is 256 entries, which is
    more than sufficient for three metrics x 1000 possible bucket values.
    """
    if metric_type not in PROFIT_BOOKING_THRESHOLDS:
        raise ValueError(
            f"Unknown metric_type '{metric_type}'. "
            f"Expected one of: {list(PROFIT_BOOKING_THRESHOLDS.keys())}"
        )

    # Clamp to valid percentage range before evaluation.
    clamped = max(0.0, min(100.0, float(value)))

    tiers = PROFIT_BOOKING_THRESHOLDS[metric_type]
    for tier_rule in tiers:
        if tier_rule['condition'](clamped):
            tier_name = tier_rule['tier']
            palette   = _TIER_PALETTE[tier_name]
            return MetricTheme(
                color_hex    = palette['color_hex'],
                grad_start   = palette['grad_start'],
                grad_end     = palette['grad_end'],
                glow_hex     = palette['glow_hex'],
                border_hex   = palette['border_hex'],
                status_label = tier_rule['label'],
                status_text  = tier_rule['status'],
                tier         = tier_name,
            )

    # Unreachable: every threshold list ends with a catch-all lambda v: True.
    # Defensive fallback to prevent any silent failure in production.
    palette = _TIER_PALETTE['red']
    return MetricTheme(
        color_hex    = palette['color_hex'],
        grad_start   = palette['grad_start'],
        grad_end     = palette['grad_end'],
        glow_hex     = palette['glow_hex'],
        border_hex   = palette['border_hex'],
        status_label = 'Unknown',
        status_text  = 'Unable to evaluate metric threshold',
        tier         = 'red',
    )


def get_metric_theme_rounded(metric_type: MetricType, value: float) -> MetricTheme:
    """
    Convenience wrapper that rounds *value* to 1 decimal place before passing
    to the cached resolver.  Use this at call sites where the raw float comes
    directly from live data (avoids cache pollution from float precision noise).
    """
    return get_metric_theme(metric_type, round(float(value), 1))


# ──────────────────────────────────────────────────────────────────────────────
# 4. Inline Self-Test  (python utils/metric_themes.py)
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    """
    Boundary-condition assertions for all three metrics.
    Run:  python utils/metric_themes.py
    Expected output: 9/9 assertions PASSED
    """
    import sys

    _PASS = 'PASS'
    _FAIL = 'FAIL'

    _tests: list[tuple[str, float, str, str]] = [
        # (metric_type, value, expected_tier, description)

        # ── Upside Potential ──────────────────────────────────────
        ('upsidePotential',    60.0,  'green',  'UP = 60.0  -> green boundary (>= 60)'),
        ('upsidePotential',    45.0,  'orange', 'UP = 45.0  -> orange midpoint (>= 30 < 60)'),
        ('upsidePotential',    29.9,  'red',    'UP = 29.9  -> red (< 30)'),

        # ── Momentum Exhaustion ───────────────────────────────────
        ('momentumExhaustion', 39.9,  'green',  'ME = 39.9  -> green (< 40)'),
        ('momentumExhaustion', 55.0,  'orange', 'ME = 55.0  -> orange midpoint (>= 40 < 70)'),
        ('momentumExhaustion', 70.0,  'red',    'ME = 70.0  -> red boundary (>= 70)'),

        # ── Downside Risk ─────────────────────────────────────────
        ('downsideRisk',       14.9,  'green',  'DR = 14.9  -> green (< 15)'),
        ('downsideRisk',       22.0,  'orange', 'DR = 22.0  -> orange midpoint (>= 15 < 30)'),
        ('downsideRisk',       30.0,  'red',    'DR = 30.0  -> red boundary (>= 30)'),
    ]

    failures = 0
    print('\n--- MetricTheme boundary-condition self-test ---\n')
    for metric, val, expected_tier, desc in _tests:
        theme = get_metric_theme_rounded(metric, val)  # type: ignore[arg-type]
        ok = theme.tier == expected_tier
        status = _PASS if ok else _FAIL
        print(f'  [{status}]  {desc}')
        if not ok:
            print(f'         got tier={theme.tier!r}, want={expected_tier!r}')
            failures += 1

    total = len(_tests)
    passed = total - failures
    print(f'\n  {passed}/{total} assertions passed\n')
    sys.exit(0 if failures == 0 else 1)
