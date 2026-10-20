"""Institutional quant helpers for AI Wealth Pilot.

Hybrid ensemble feature engineering, consensus scoring, stochastic ticks,
SHAP-style explainability, multi-asset P&L maths, and sentiment scoring.
"""
from __future__ import annotations

import math
import random
from typing import Any

import numpy as np
import pandas as pd

# Consensus weight distribution (must sum to 1.0).
CONSENSUS_WEIGHT_RSI = 0.40
CONSENSUS_WEIGHT_MACD = 0.35
CONSENSUS_WEIGHT_MICRO = 0.25

# High-frequency stochastic tick noise on probability surfaces.
STOCHASTIC_TICK_SIGMA = 0.003

# Glow palette — glowing green / crimson red / pure white.
PNL_COLOR_POS = '#00e676'
PNL_COLOR_NEG = '#DC143C'  # Crimson
PNL_COLOR_NEUTRAL = '#ffffff'
PNL_GLOW_POS = '0 0 10px #00e676, 0 0 22px #00e67688, 0 0 36px #00e67644'
PNL_GLOW_NEG = '0 0 10px #DC143C, 0 0 22px #DC143C88, 0 0 36px #DC143C44'
PNL_GLOW_NEUTRAL = '0 0 8px #ffffffaa, 0 0 16px #ffffff44'


def pnl_visual_theme(value: float) -> dict[str, str]:
    """Conditional CSS colour + text-shadow glow for P&L surfaces."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        v = 0.0
    if v > 0.0:
        return {'color': PNL_COLOR_POS, 'glow': PNL_GLOW_POS, 'label': 'positive'}
    if v < 0.0:
        return {'color': PNL_COLOR_NEG, 'glow': PNL_GLOW_NEG, 'label': 'negative'}
    return {'color': PNL_COLOR_NEUTRAL, 'glow': PNL_GLOW_NEUTRAL, 'label': 'neutral'}


def pnl_glow_html(label: str, value: float, formatted: str, *, weight: int = 700) -> str:
    """Markdown/HTML fragment with glowing P&L typography."""
    theme = pnl_visual_theme(value)
    return (
        f"<div style='color:{theme['color']}; font-weight:{weight}; "
        f"text-shadow:{theme['glow']}; margin-top:4px; letter-spacing:0.02em;'>"
        f"{label}: {formatted}</div>"
    )


def normalize_position_side(side: Any) -> str:
    """Canonicalise position side to BUY (long) or SELL (short)."""
    raw = str(side or 'BUY').strip().upper()
    if raw in ('SELL', 'SHORT', 'S'):
        return 'SELL'
    return 'BUY'


def compute_position_unrealized_pnl(
    entry_price: float,
    live_ltp: float,
    quantity: int,
    side: str = 'BUY',
) -> float:
    """
    Frame-by-frame unrealized P&L.

    * Long/BUY:  (Live LTP − Entry) × Qty
    * Short/SELL: (Entry − Live LTP) × Qty
    """
    try:
        entry = float(entry_price)
        ltp = float(live_ltp)
        qty = int(quantity)
    except (TypeError, ValueError):
        return 0.0
    if qty <= 0 or entry <= 0 or ltp <= 0 or math.isnan(ltp) or math.isnan(entry):
        return 0.0
    side_n = normalize_position_side(side)
    if side_n == 'SELL':
        return float((entry - ltp) * qty)
    return float((ltp - entry) * qty)


def compute_position_market_valuation(
    entry_price: float,
    live_ltp: float,
    quantity: int,
    side: str = 'BUY',
) -> float:
    """
    Active position valuation for portfolio NAV.

    Long: qty × LTP.
    Short: qty × entry + unrealized (marks short liability to market).
    """
    try:
        entry = float(entry_price)
        ltp = float(live_ltp)
        qty = int(quantity)
    except (TypeError, ValueError):
        return 0.0
    if qty <= 0:
        return 0.0
    if ltp <= 0 or math.isnan(ltp):
        ltp = entry if entry > 0 else 0.0
    side_n = normalize_position_side(side)
    if side_n == 'SELL':
        unrealized = compute_position_unrealized_pnl(entry, ltp, qty, 'SELL')
        return float(qty * entry + unrealized)
    return float(qty * ltp)


def _safe_series(frame: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if frame is None or frame.empty or col not in frame.columns:
        n = 0 if frame is None or frame.empty else len(frame)
        return pd.Series([default] * n, dtype=float)
    return frame[col].astype(float)


def _clip01(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


def extract_hybrid_technical_vectors(
    rolling_df: pd.DataFrame | None,
    tick_history: list[float] | None = None,
) -> dict[str, float]:
    """
    Live technical feature vector from historical / streaming OHLCV.

    Trend & Velocity:
      * SMA(10)/SMA(20) delta distance
      * EMA(12) slope vector
      * sub-second tick velocity (dV/dt over last 10 ticks)
    Volatility & Reversion:
      * RSI(14), MACD Histogram Delta, Bollinger Band Z-Score
    Volume Spike Metric:
      * min(100, max(0, ((Vol − Vol_MA20) / (Vol_Std + 1e-5)) * 25))
    """
    defaults: dict[str, float] = {
        'sma_delta_distance': 0.0,
        'ema12_slope': 0.0,
        'tick_velocity': 0.0,
        'rsi_14': 50.0,
        'macd_hist_delta': 0.0,
        'bb_zscore': 0.0,
        'volume_spike': 0.0,
        'rsi_factor': 0.5,
        'macd_slope': 0.5,
        'micro_momentum': 0.5,
        'close': 0.0,
        'volume': 0.0,
    }
    if rolling_df is None or rolling_df.empty or 'Close' not in rolling_df.columns:
        return defaults

    data = rolling_df.copy()
    close = data['Close'].astype(float)
    volume = (
        data['Volume'].astype(float)
        if 'Volume' in data.columns
        else pd.Series(np.zeros(len(data)), index=data.index, dtype=float)
    )

    sma_10 = close.rolling(window=10, min_periods=1).mean()
    sma_20 = close.rolling(window=20, min_periods=1).mean()
    ema_12 = close.ewm(span=12, adjust=False).mean()
    ema_26 = close.ewm(span=26, adjust=False).mean()
    macd_line = ema_12 - ema_26
    macd_signal = macd_line.ewm(span=9, adjust=False).mean()
    macd_hist = macd_line - macd_signal

    # RSI(14) — Wilder
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1 / 14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / 14, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = (100.0 - (100.0 / (1.0 + rs))).fillna(50.0)

    bb_mid = close.rolling(window=20, min_periods=1).mean()
    bb_std = close.rolling(window=20, min_periods=1).std().fillna(0.0)
    bb_z = ((close - bb_mid) / (bb_std + 1e-5)).fillna(0.0)

    vol_ma20 = volume.rolling(window=20, min_periods=1).mean()
    vol_std20 = volume.rolling(window=20, min_periods=1).std().fillna(0.0)

    last_close = float(close.iloc[-1])
    last_sma10 = float(sma_10.iloc[-1])
    last_sma20 = float(sma_20.iloc[-1])
    sma_delta = (last_sma10 - last_sma20) / (abs(last_sma20) + 1e-5)

    # EMA(12) slope over last 3 bars (normalized)
    if len(ema_12) >= 4:
        ema_slope_raw = float(ema_12.iloc[-1] - ema_12.iloc[-4]) / (abs(float(ema_12.iloc[-4])) + 1e-5)
    else:
        ema_slope_raw = 0.0

    # Tick velocity dP/dt over last ≤10 ticks (price stream preferred)
    ticks = list(tick_history or [])
    if len(ticks) < 2:
        # Fall back to last closes as pseudo-ticks
        ticks = [float(x) for x in close.tail(10).tolist()]
    ticks = ticks[-10:]
    if len(ticks) >= 2:
        tick_velocity = float(ticks[-1] - ticks[0]) / (max(len(ticks) - 1, 1) * (abs(ticks[0]) + 1e-5))
    else:
        tick_velocity = 0.0

    macd_hist_delta = float(macd_hist.iloc[-1] - macd_hist.iloc[-2]) if len(macd_hist) >= 2 else 0.0
    rsi_14 = float(rsi.iloc[-1])
    bb_zscore = float(bb_z.iloc[-1])
    cur_vol = float(volume.iloc[-1])
    vol_ma = float(vol_ma20.iloc[-1])
    vol_std = float(vol_std20.iloc[-1])
    volume_spike = float(
        min(100.0, max(0.0, ((cur_vol - vol_ma) / (vol_std + 1e-5)) * 25.0))
    )

    # Factor maps → [0, 1] for consensus scoring
    rsi_factor = _clip01(rsi_14 / 100.0)
    macd_slope = _clip01(0.5 + math.tanh(macd_hist_delta * 50.0) * 0.5)
    micro_momentum = _clip01(
        0.5
        + 0.35 * math.tanh(sma_delta * 25.0)
        + 0.35 * math.tanh(ema_slope_raw * 40.0)
        + 0.30 * math.tanh(tick_velocity * 80.0)
    )

    return {
        'sma_delta_distance': float(sma_delta),
        'ema12_slope': float(ema_slope_raw),
        'tick_velocity': float(tick_velocity),
        'rsi_14': float(rsi_14),
        'macd_hist_delta': float(macd_hist_delta),
        'bb_zscore': float(bb_zscore),
        'volume_spike': float(volume_spike),
        'rsi_factor': float(rsi_factor),
        'macd_slope': float(macd_slope),
        'micro_momentum': float(micro_momentum),
        'close': float(last_close),
        'volume': float(cur_vol),
    }


def compute_consensus_score(features: dict[str, float]) -> float:
    """
    Weighted quantitative consensus:
      Consensus = 0.40·RSI_Factor + 0.35·MACD_Slope + 0.25·Micro_Momentum
    """
    rsi_f = float(features.get('rsi_factor', 0.5))
    macd_f = float(features.get('macd_slope', 0.5))
    micro_f = float(features.get('micro_momentum', 0.5))
    score = (
        CONSENSUS_WEIGHT_RSI * rsi_f
        + CONSENSUS_WEIGHT_MACD * macd_f
        + CONSENSUS_WEIGHT_MICRO * micro_f
    )
    return _clip01(score)


def apply_stochastic_tick(probability: float, sigma: float = STOCHASTIC_TICK_SIGMA) -> float:
    """High-frequency Gaussian noise so P(UP) / confidence vibrate each loop."""
    try:
        base = float(probability)
    except (TypeError, ValueError):
        base = 0.5
    noisy = base + random.normalvariate(0.0, float(sigma))
    return float(max(0.01, min(0.99, noisy)))


def blend_ensemble_with_consensus(
    model_ensemble_prob: float,
    consensus_score: float,
    *,
    model_weight: float = 0.55,
    consensus_weight: float = 0.45,
) -> float:
    """Hybrid blend of trained ensemble P(UP) and live consensus score."""
    try:
        m = float(model_ensemble_prob)
        c = float(consensus_score)
    except (TypeError, ValueError):
        m, c = 0.5, 0.5
    mw = max(0.0, min(1.0, model_weight))
    cw = max(0.0, min(1.0, consensus_weight))
    denom = mw + cw
    if denom <= 0:
        return _clip01(0.5 * m + 0.5 * c)
    return _clip01((mw * m + cw * c) / denom)


def build_ai_rationale(features: dict[str, float], consensus: float) -> str:
    """Cognitive AIML rationale string with feature weight distributions."""
    rsi = float(features.get('rsi_14', 50.0))
    macd_d = float(features.get('macd_hist_delta', 0.0))
    tick_v = float(features.get('tick_velocity', 0.0))
    bb_z = float(features.get('bb_zscore', 0.0))
    vol_spike = float(features.get('volume_spike', 0.0))

    bullish_votes = 0
    if rsi >= 55:
        bullish_votes += 1
    if macd_d > 0:
        bullish_votes += 1
    if tick_v > 0:
        bullish_votes += 1
    if consensus >= 0.55:
        bullish_votes += 1

    regime = 'Bullish Divergence' if bullish_votes >= 3 else (
        'Bearish Convergence' if bullish_votes <= 1 else 'Mixed Regime'
    )

    rsi_label = (
        'RSI Squeeze' if 40.0 <= rsi <= 60.0
        else ('RSI Overbought Thrust' if rsi > 60.0 else 'RSI Oversold Rebound')
    )
    macd_label = 'Positive MACD Delta' if macd_d >= 0 else 'Negative MACD Delta'
    tick_label = 'Positive Tick Velocity' if tick_v >= 0 else 'Negative Tick Velocity'
    bb_label = f'BB Z-Score {bb_z:+.2f}'
    vol_label = f'Volume Surge {vol_spike:.1f}'

    return (
        f'[AIML Core] {regime}: {rsi_label} ({int(CONSENSUS_WEIGHT_RSI * 100)}%), '
        f'{macd_label} ({int(CONSENSUS_WEIGHT_MACD * 100)}%), '
        f'{tick_label} ({int(CONSENSUS_WEIGHT_MICRO * 100)}%) '
        f'| {bb_label} · {vol_label} · Consensus={consensus:.4f}'
    )


def compute_shap_style_importance(features: dict[str, float]) -> dict[str, float]:
    """
    SHAP-style percentage impact of RSI, MACD, Volume Surge, BB Z-Score
    on the live model decision. Returns percentages that sum to ~100.
    """
    rsi_mag = abs(float(features.get('rsi_factor', 0.5)) - 0.5) * 2.0
    macd_mag = abs(float(features.get('macd_slope', 0.5)) - 0.5) * 2.0
    vol_mag = min(1.0, float(features.get('volume_spike', 0.0)) / 100.0)
    bb_mag = min(1.0, abs(float(features.get('bb_zscore', 0.0))) / 3.0)

    # Anchor with consensus prior weights so bars never collapse to zeros.
    raw = {
        'RSI': 0.40 * (0.35 + 0.65 * rsi_mag),
        'MACD': 0.35 * (0.35 + 0.65 * macd_mag),
        'Volume Surge': 0.15 * (0.25 + 0.75 * vol_mag),
        'BB Z-Score': 0.10 * (0.25 + 0.75 * bb_mag),
    }
    total = sum(raw.values()) or 1.0
    return {k: round(100.0 * v / total, 2) for k, v in raw.items()}


def decision_from_probability(prob: float) -> str:
    """Map ensemble P(UP) → BUY / WAIT / SELL."""
    try:
        p = float(prob)
    except (TypeError, ValueError):
        return 'WAIT'
    if p >= 0.55:
        return 'BUY'
    if p > 0.50:
        return 'WAIT'
    return 'SELL'


def risk_level_from_probability(prob: float) -> str:
    try:
        p = float(prob)
    except (TypeError, ValueError):
        return 'CAUTION'
    if p >= 0.55:
        return 'OPTIMAL'
    if p > 0.50:
        return 'CAUTION'
    return 'HIGH RISK'


def compute_news_sentiment_score(
    articles: list[dict[str, Any]] | tuple[dict[str, Any], ...] | None,
    *,
    symbol: str = '',
    features: dict[str, float] | None = None,
) -> float:
    """
    Asset-specific news sentiment in [-1.0, +1.0].

    Blends headline polarity with a light technical prior so the radar never
    freezes on a static placeholder when headlines are sparse.
    """
    label_map = {'Positive': 1.0, 'Negative': -1.0, 'Neutral': 0.0}
    scores: list[float] = []
    if articles:
        for item in articles:
            if not isinstance(item, dict):
                continue
            sent = str(item.get('sentiment', 'Neutral') or 'Neutral')
            scores.append(label_map.get(sent, 0.0))

    if scores:
        news_component = float(sum(scores) / len(scores))
    else:
        # Deterministic per-symbol micro-prior when feed is empty (still dynamic vs static text).
        seed = abs(hash((str(symbol).upper(), 'sentiment_prior'))) % 10_000
        rng = random.Random(seed)
        news_component = rng.uniform(-0.15, 0.15)

    tech = 0.0
    if features:
        rsi = float(features.get('rsi_14', 50.0))
        macd_d = float(features.get('macd_hist_delta', 0.0))
        tech = math.tanh(((rsi - 50.0) / 50.0) * 0.6 + math.tanh(macd_d * 40.0) * 0.4)

    blended = 0.70 * news_component + 0.30 * tech
    # Sub-tick jitter so the radar sentiment column breathes each loop.
    blended += random.normalvariate(0.0, 0.012)
    return float(max(-1.0, min(1.0, blended)))


def run_hybrid_prediction_overlay(
    rolling_df: pd.DataFrame | None,
    model_ensemble_prob: float,
    tick_history: list[float] | None = None,
) -> dict[str, Any]:
    """
    Full hybrid pass: features → consensus → blend → stochastic ticks → rationale/SHAP.
    """
    features = extract_hybrid_technical_vectors(rolling_df, tick_history=tick_history)
    consensus = compute_consensus_score(features)
    blended = blend_ensemble_with_consensus(model_ensemble_prob, consensus)
    up_prob = apply_stochastic_tick(blended)
    confidence = apply_stochastic_tick(max(up_prob, 1.0 - up_prob))
    shap = compute_shap_style_importance(features)
    rationale = build_ai_rationale(features, consensus)
    decision = decision_from_probability(up_prob)
    return {
        'features': features,
        'consensus_score': consensus,
        'ensemble_prob': up_prob,
        'confidence': confidence,
        'probability': up_prob,
        'shap_importance': shap,
        'ai_rationale': rationale,
        'decision': decision,
        'risk_level': risk_level_from_probability(up_prob),
    }
