"""Predictive Transition Probability Engine — Proximity to Trigger metrics."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

WEALTH_ALLOC_BUY_THRESHOLD = 0.55
WEALTH_ALLOC_WAIT_FLOOR = 0.50

# Component weights — buy-side (pre-trigger accumulation)
_BUY_WEIGHTS: dict[str, float] = {
    'ensemble_proximity': 0.35,
    'rsi_recovery': 0.25,
    'ma_convergence': 0.25,
    'volume_spike': 0.15,
}

# Component weights — exit-side (profit-booking / reversal)
_EXIT_WEIGHTS: dict[str, float] = {
    'ensemble_decay': 0.30,
    'rsi_exhaustion': 0.25,
    'momentum_divergence': 0.20,
    'volatility_expansion': 0.15,
    'upper_band_stretch': 0.10,
}

# Human-readable labels for each component key (used by UI component bars)
_BUY_COMPONENT_LABELS: dict[str, str] = {
    'ensemble_proximity': 'Ensemble P(UP) Proximity',
    'rsi_recovery': 'RSI Recovery Slope',
    'ma_convergence': 'MA Stack Convergence',
    'volume_spike': 'Volume Breakout Spike',
}

_EXIT_COMPONENT_LABELS: dict[str, str] = {
    'ensemble_decay': 'Ensemble P(UP) Decay',
    'rsi_exhaustion': 'RSI Overbought Exhaustion',
    'momentum_divergence': 'MACD Momentum Divergence',
    'volatility_expansion': 'ATR Volatility Expansion',
    'upper_band_stretch': 'BB Upper Band Stretch',
}

# Urgency tier thresholds
_URGENCY_HIGH_PCT = 65.0
_URGENCY_MODERATE_PCT = 35.0


@dataclass(frozen=True)
class TransitionProbabilityResult:
    """Dual-sided proximity metric for a single ticker tick."""

    scenario: str
    primary_metric_pct: float
    primary_label: str
    badge_label: str
    badge_color: str
    bar_color_start: str
    bar_color_end: str
    component_scores: dict[str, float] = field(default_factory=dict)
    narrative: str = ''
    fallback: bool = False
    urgency_tier: str = 'LOW'  # 'HIGH' | 'MODERATE' | 'LOW'

    def component_labels(self) -> dict[str, str]:
        """Human-readable label for each component key."""
        if self.scenario == 'PROXIMITY_TO_EXIT':
            return dict(_EXIT_COMPONENT_LABELS)
        return dict(_BUY_COMPONENT_LABELS)

    def to_dict(self) -> dict[str, Any]:
        return {
            'scenario': self.scenario,
            'primary_metric_pct': float(self.primary_metric_pct),
            'primary_label': self.primary_label,
            'badge_label': self.badge_label,
            'badge_color': self.badge_color,
            'bar_color_start': self.bar_color_start,
            'bar_color_end': self.bar_color_end,
            'component_scores': dict(self.component_scores),
            'narrative': self.narrative,
            'fallback': self.fallback,
            'urgency_tier': self.urgency_tier,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TransitionProbabilityResult:
        return cls(
            scenario=str(data.get('scenario', 'UNKNOWN')),
            primary_metric_pct=float(data.get('primary_metric_pct', 0.0) or 0.0),
            primary_label=str(data.get('primary_label', '')),
            badge_label=str(data.get('badge_label', '')),
            badge_color=str(data.get('badge_color', '#7a8499')),
            bar_color_start=str(data.get('bar_color_start', '#7a8499')),
            bar_color_end=str(data.get('bar_color_end', '#7a8499')),
            component_scores={
                str(k): float(v) for k, v in (data.get('component_scores') or {}).items()
            },
            narrative=str(data.get('narrative', '')),
            fallback=bool(data.get('fallback', False)),
            urgency_tier=str(data.get('urgency_tier', 'LOW')),
        )


def _clamp_pct(value: float) -> float:
    if np.isnan(value) or np.isinf(value):
        return 0.0
    return float(max(0.0, min(100.0, value)))


def _safe_float(value: Any, default: float = float('nan')) -> float:
    try:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _extract_indicator_context(df_display: pd.DataFrame | None) -> dict[str, float]:
    """Pull latest and short-lookback indicator values from OHLCV display frame."""
    ctx: dict[str, float] = {
        'rsi': float('nan'),
        'rsi_prev': float('nan'),
        'sma_10': float('nan'),
        'sma_20': float('nan'),
        'ema_12': float('nan'),
        'close': float('nan'),
        'volume': float('nan'),
        'volume_ma': float('nan'),
        'bb_upper': float('nan'),
        'bb_lower': float('nan'),
        'macd': float('nan'),
        'macd_prev': float('nan'),
        'atr': float('nan'),
        'atr_median': float('nan'),
        'bb_squeeze': float('nan'),
    }
    if df_display is None or df_display.empty:
        return ctx

    frame = df_display.copy()
    latest = frame.iloc[-1]
    ctx['rsi'] = _safe_float(latest.get('RSI'))
    ctx['sma_10'] = _safe_float(latest.get('SMA_10'))
    ctx['sma_20'] = _safe_float(latest.get('SMA_20'))
    ctx['ema_12'] = _safe_float(latest.get('EMA_12'))
    ctx['close'] = _safe_float(latest.get('Close'))
    ctx['volume'] = _safe_float(latest.get('Volume'))
    ctx['volume_ma'] = _safe_float(latest.get('Volume_MA'))
    ctx['bb_upper'] = _safe_float(latest.get('BB_Upper'))
    ctx['bb_lower'] = _safe_float(latest.get('BB_Lower'))
    ctx['macd'] = _safe_float(latest.get('MACD'))
    ctx['bb_squeeze'] = _safe_float(latest.get('BB_Squeeze_Width'))

    if len(frame) >= 4 and 'RSI' in frame.columns:
        ctx['rsi_prev'] = _safe_float(frame['RSI'].iloc[-4])
    if len(frame) >= 4 and 'MACD' in frame.columns:
        ctx['macd_prev'] = _safe_float(frame['MACD'].iloc[-4])

    if 'ATR_14' in frame.columns:
        atr_series = frame['ATR_14'].dropna()
        if not atr_series.empty:
            ctx['atr'] = _safe_float(atr_series.iloc[-1])
            ctx['atr_median'] = _safe_float(atr_series.tail(20).median())

    return ctx


def _score_ensemble_buy_proximity(ensemble_prob: float) -> float | None:
    """How close ensemble P(UP) is to crossing the official BUY threshold (55%)."""
    if np.isnan(ensemble_prob):
        return None
    prob = _safe_float(ensemble_prob, 0.0)
    # Map 35% → 0, 55% → 100 (aligned with wealth allocation gate)
    return _clamp_pct((prob - 0.35) / (WEALTH_ALLOC_BUY_THRESHOLD - 0.35) * 100.0)


def _score_rsi_recovery(rsi: float, rsi_prev: float) -> float | None:
    """RSI climbing out of bearish/neutral zone toward bullish momentum."""
    if np.isnan(rsi):
        return None
    slope = 0.0 if np.isnan(rsi_prev) else rsi - rsi_prev
    # Recovery zone: RSI 28–58 with positive slope scores highest
    level_score = _clamp_pct((rsi - 22.0) / 36.0 * 100.0)
    slope_bonus = _clamp_pct(max(0.0, slope) * 8.0)
    if rsi > 68.0:
        level_score *= 0.55
    return _clamp_pct(level_score * 0.75 + slope_bonus * 0.25)


def _score_ma_convergence(
    close: float,
    sma_10: float,
    sma_20: float,
    ema_12: float,
) -> float | None:
    """Moving-average stack aligning bullish (price > EMA, SMA10 converging SMA20)."""
    if np.isnan(sma_10) or np.isnan(sma_20) or sma_20 <= 0:
        return None
    spread_pct = (sma_10 - sma_20) / sma_20 * 100.0
    # Narrowing bearish spread (-2% to 0) or positive golden-cross proximity
    # Base 0 at -5% spread, Base 100 at +2% spread (eliminates hardcoded 50% bug)
    convergence = _clamp_pct((spread_pct + 5.0) / 7.0 * 100.0)
    price_above_ema = 0.0
    if not np.isnan(close) and not np.isnan(ema_12) and ema_12 > 0:
        price_above_ema = _clamp_pct((close / ema_12 - 0.995) / 0.025 * 100.0)
    return _clamp_pct(convergence * 0.6 + price_above_ema * 0.4)


def _score_volume_spike(volume: float, volume_ma: float) -> float | None:
    """Relative volume expansion confirming breakout interest."""
    if np.isnan(volume) or np.isnan(volume_ma) or volume_ma <= 0:
        return None
    ratio = volume / volume_ma
    return _clamp_pct((ratio - 0.75) / 0.75 * 100.0)


def _compute_buy_transition_probability(
    ensemble_prob: float,
    indicators: dict[str, float],
) -> tuple[float, dict[str, float]]:
    raw_components = {
        'ensemble_proximity': _score_ensemble_buy_proximity(ensemble_prob),
        'rsi_recovery': _score_rsi_recovery(indicators['rsi'], indicators['rsi_prev']),
        'ma_convergence': _score_ma_convergence(
            indicators['close'],
            indicators['sma_10'],
            indicators['sma_20'],
            indicators['ema_12'],
        ),
        'volume_spike': _score_volume_spike(indicators['volume'], indicators['volume_ma']),
    }
    
    components = {}
    total_valid_weight = 0.0
    weighted_sum = 0.0
    
    for k, v in raw_components.items():
        if v is not None:
            components[k] = v
            total_valid_weight += _BUY_WEIGHTS[k]
            weighted_sum += v * _BUY_WEIGHTS[k]
            
    if total_valid_weight > 0.0:
        final_score = weighted_sum / total_valid_weight
    else:
        # Fallback to ensemble proximity mapping if all technicals are NaN
        prob = _safe_float(ensemble_prob, 0.0)
        final_score = _clamp_pct((prob - 0.35) / (WEALTH_ALLOC_BUY_THRESHOLD - 0.35) * 100.0)
        components['ensemble_proximity'] = final_score
        
    return _clamp_pct(final_score), components


def _score_ensemble_decay(ensemble_prob: float) -> float | None:
    """Probability of slipping below BUY / into WAIT as P(UP) fades."""
    if np.isnan(ensemble_prob):
        return None
    prob = _safe_float(ensemble_prob, WEALTH_ALLOC_BUY_THRESHOLD)
    if prob >= WEALTH_ALLOC_BUY_THRESHOLD:
        # Still BUY but closer to 55% = lower exit pressure; 70%+ extended = higher
        extension = _clamp_pct((prob - WEALTH_ALLOC_BUY_THRESHOLD) / 0.20 * 100.0)
        return _clamp_pct(extension * 0.55)
    # Already fading — high exit probability
    return _clamp_pct((WEALTH_ALLOC_BUY_THRESHOLD - prob) / 0.10 * 100.0)


def _score_rsi_exhaustion(rsi: float, rsi_prev: float) -> float | None:
    if np.isnan(rsi):
        return None
    overbought = _clamp_pct((rsi - 62.0) / 28.0 * 100.0)
    slope = 0.0 if np.isnan(rsi_prev) else rsi - rsi_prev
    rollover = _clamp_pct(max(0.0, -slope) * 10.0)
    return _clamp_pct(overbought * 0.7 + rollover * 0.3)


def _score_momentum_divergence(
    close: float,
    macd: float,
    macd_prev: float,
    df_display: pd.DataFrame | None,
) -> float | None:
    """MACD histogram fade + simple price/MACD divergence."""
    if np.isnan(macd) or np.isnan(macd_prev):
        return None
    macd_fade = _clamp_pct(max(0.0, macd_prev - macd) * 120.0)
    divergence = 0.0
    if df_display is not None and len(df_display) >= 6 and 'Close' in df_display.columns:
        closes = df_display['Close'].astype(float)
        price_higher = closes.iloc[-1] > closes.iloc[-5]
        macd_lower = macd < macd_prev
        if price_higher and macd_lower:
            divergence = 75.0
    return _clamp_pct(macd_fade * 0.55 + divergence * 0.45)


def _score_volatility_expansion(
    atr: float,
    atr_median: float,
    bb_squeeze: float,
) -> float | None:
    if np.isnan(atr) or np.isnan(atr_median) or atr_median <= 0:
        return None
    atr_ratio = atr / atr_median
    atr_score = _clamp_pct((atr_ratio - 0.85) / 0.55 * 100.0)
    squeeze_score = _clamp_pct(_safe_float(bb_squeeze, 0.04) / 0.08 * 100.0)
    return _clamp_pct(atr_score * 0.65 + squeeze_score * 0.35)


def _score_upper_band_stretch(close: float, bb_upper: float, bb_lower: float) -> float | None:
    if np.isnan(close) or np.isnan(bb_upper) or np.isnan(bb_lower):
        return None
    band_width = bb_upper - bb_lower
    if band_width <= 0:
        return None
    position = (close - bb_lower) / band_width
    return _clamp_pct((position - 0.72) / 0.28 * 100.0)


def _compute_reversal_exit_probability(
    ensemble_prob: float,
    indicators: dict[str, float],
    df_display: pd.DataFrame | None,
) -> tuple[float, dict[str, float]]:
    raw_components = {
        'ensemble_decay': _score_ensemble_decay(ensemble_prob),
        'rsi_exhaustion': _score_rsi_exhaustion(indicators['rsi'], indicators['rsi_prev']),
        'momentum_divergence': _score_momentum_divergence(
            indicators['close'],
            indicators['macd'],
            indicators['macd_prev'],
            df_display,
        ),
        'volatility_expansion': _score_volatility_expansion(
            indicators['atr'],
            indicators['atr_median'],
            indicators['bb_squeeze'],
        ),
        'upper_band_stretch': _score_upper_band_stretch(
            indicators['close'],
            indicators['bb_upper'],
            indicators['bb_lower'],
        ),
    }
    
    components = {}
    total_valid_weight = 0.0
    weighted_sum = 0.0
    
    for k, v in raw_components.items():
        if v is not None:
            components[k] = v
            total_valid_weight += _EXIT_WEIGHTS[k]
            weighted_sum += v * _EXIT_WEIGHTS[k]
            
    if total_valid_weight > 0.0:
        final_score = weighted_sum / total_valid_weight
    else:
        # Fallback to ensemble decay mapping if all technicals are NaN
        prob = _safe_float(ensemble_prob, WEALTH_ALLOC_BUY_THRESHOLD)
        if prob >= WEALTH_ALLOC_BUY_THRESHOLD:
            extension = _clamp_pct((prob - WEALTH_ALLOC_BUY_THRESHOLD) / 0.20 * 100.0)
            final_score = _clamp_pct(extension * 0.55)
        else:
            final_score = _clamp_pct((WEALTH_ALLOC_BUY_THRESHOLD - prob) / 0.10 * 100.0)
        components['ensemble_decay'] = final_score
        
    return _clamp_pct(final_score), components


def _urgency_tier(pct: float) -> str:
    """Map a probability percentage to a 3-tier urgency label."""
    if pct >= _URGENCY_HIGH_PCT:
        return 'HIGH'
    if pct >= _URGENCY_MODERATE_PCT:
        return 'MODERATE'
    return 'LOW'


def _fallback_result(allocation_status: str) -> TransitionProbabilityResult:
    status = str(allocation_status or 'AVOID').upper()
    if status == 'BUY':
        return TransitionProbabilityResult(
            scenario='PROXIMITY_TO_EXIT',
            primary_metric_pct=0.0,
            primary_label='Reversal / Exit Probability',
            badge_label='DATA SYNC',
            badge_color='#78909c',
            bar_color_start='#ffb74d',
            bar_color_end='#ef5350',
            component_scores={},
            narrative='Live indicator sync in progress — exit proximity temporarily unavailable.',
            fallback=True,
            urgency_tier='LOW',
        )
    return TransitionProbabilityResult(
        scenario='PROXIMITY_TO_BUY',
        primary_metric_pct=0.0,
        primary_label='Buy Transition Probability',
        badge_label='DATA SYNC',
        badge_color='#78909c',
        bar_color_start='#26a69a',
        bar_color_end='#00e676',
        component_scores={},
        narrative='Live indicator sync in progress — buy proximity temporarily unavailable.',
        fallback=True,
        urgency_tier='LOW',
    )


def compute_transition_probabilities(
    allocation_status: str,
    *,
    ensemble_prob: float,
    df_display: pd.DataFrame | None = None,
    prediction_rsi: float | None = None,
    prediction_bb_squeeze: float | None = None,
    prediction_atr: float | None = None,
) -> TransitionProbabilityResult:
    """
    Dual-sided Proximity to Trigger engine.

    WAIT / AVOID → Buy Transition Probability (%)
    BUY          → Reversal / Exit Probability (%)
    """
    status = str(allocation_status or 'AVOID').upper()
    indicators = _extract_indicator_context(df_display)

    if prediction_rsi is not None and not np.isnan(_safe_float(prediction_rsi)):
        indicators['rsi'] = _safe_float(prediction_rsi, indicators['rsi'])
    if prediction_bb_squeeze is not None:
        indicators['bb_squeeze'] = _safe_float(prediction_bb_squeeze, indicators['bb_squeeze'])
    if prediction_atr is not None and not np.isnan(_safe_float(prediction_atr)):
        indicators['atr'] = _safe_float(prediction_atr, indicators['atr'])

    try:
        if status == 'BUY':
            pct, components = _compute_reversal_exit_probability(
                ensemble_prob,
                indicators,
                df_display,
            )
            if pct >= 65.0:
                narrative = 'Elevated profit-booking risk — momentum exhaustion signals building. Consider scaling out or setting tighter stops.'
            elif pct >= 40.0:
                narrative = 'Moderate reversal pressure — monitor for WAIT/exit confirmation. RSI and MACD divergence are key triggers.'
            else:
                narrative = 'Trend intact — exit proximity remains low vs. trigger thresholds. Momentum supports continued holding.'
            
            logger.debug(
                '[Transition Engine] %s Reversal/Exit: %.1f%% | Components: %s',
                status, pct, ' + '.join([f"{k}={v:.1f}%" for k, v in components.items()])
            )
            
            return TransitionProbabilityResult(
                scenario='PROXIMITY_TO_EXIT',
                primary_metric_pct=pct,
                primary_label='Reversal / Exit Probability',
                badge_label='PROFIT BOOKING RISK' if pct >= 55 else 'WATCH EXIT ZONE',
                badge_color='#ff6f00' if pct >= 55 else '#ffb300',
                bar_color_start='#ffb74d',
                bar_color_end='#ef5350',
                component_scores=components,
                narrative=narrative,
                fallback=False,
                urgency_tier=_urgency_tier(pct),
            )

        pct, components = _compute_buy_transition_probability(ensemble_prob, indicators)
        if pct >= 60.0:
            narrative = 'Bullish pre-trigger alignment — stock approaching official BUY gate. Proactive accumulation window is open.'
        elif pct >= 35.0:
            narrative = 'Early accumulation zone — momentum recovering but not yet at BUY threshold. Watch for RSI and MA cross confirmation.'
        else:
            narrative = 'Bearish/neutral posture persists — low proximity to BUY breakout. Patience recommended before committing capital.'
        
        logger.debug(
            '[Transition Engine] %s Buy Transition: %.1f%% | Components: %s',
            status, pct, ' + '.join([f"{k}={v:.1f}%" for k, v in components.items()])
        )
        
        return TransitionProbabilityResult(
            scenario='PROXIMITY_TO_BUY',
            primary_metric_pct=pct,
            primary_label='Buy Transition Probability',
            badge_label='PRE-BUY ACCUMULATION' if pct >= 50 else 'BULLISH BUILDUP',
            badge_color='#00c853' if pct >= 50 else '#26a69a',
            bar_color_start='#26a69a',
            bar_color_end='#00e676',
            component_scores=components,
            narrative=narrative,
            fallback=False,
            urgency_tier=_urgency_tier(pct),
        )
    except Exception:
        return _fallback_result(status)
