from types import SimpleNamespace

import pandas as pd

from dashboard import news_panel
from dashboard.app import (
    PredictionResult,
    build_buy_profit_dashboard_data,
    derive_actionable_card_metrics,
    is_exact_buy_status,
)


def test_build_buy_profit_dashboard_data_returns_metrics_for_strict_buy_status() -> None:
    allocation = {'status': 'BUY'}
    transition = SimpleNamespace(
        primary_metric_pct=35.0,
        component_scores={
            'rsi_exhaustion': 80.0,
            'momentum_divergence': 20.0,
            'volatility_expansion': 30.0,
            'upper_band_stretch': 10.0,
        },
    )

    data = build_buy_profit_dashboard_data(allocation, transition)

    assert data is not None
    assert data['upsidePotential'] == 65.0
    assert data['momentumExhaustion'] == 56.0
    assert data['downsideRisk'] == 22.0


def test_build_buy_profit_dashboard_data_is_case_sensitive() -> None:
    allocation = {'status': 'buy'}
    transition = SimpleNamespace(
        primary_metric_pct=35.0,
        component_scores={
            'rsi_exhaustion': 80.0,
            'momentum_divergence': 20.0,
            'volatility_expansion': 30.0,
            'upper_band_stretch': 10.0,
        },
    )

    assert build_buy_profit_dashboard_data(allocation, transition) is None


def test_is_exact_buy_status_requires_exact_buy_value() -> None:
    assert is_exact_buy_status('BUY') is True
    assert is_exact_buy_status('buy') is False
    assert is_exact_buy_status('Buy') is False
    assert is_exact_buy_status('WAIT') is False
    assert is_exact_buy_status(None) is False


def _card_test_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            'Close': [100.0, 101.0, 110.0],
            'RSI': [55.0, 60.0, 72.0],
            'EMA_12': [99.0, 100.0, 108.0],
            'SMA_20': [100.0, 100.5, 107.0],
            'MACD': [0.5, 0.7, 1.4],
            'MACD_Signal': [0.6, 0.65, 1.2],
            'MACD_Hist': [-0.1, 0.05, 0.2],
            'MACD_Hist_Delta': [0.15, 0.15, 0.15],
            'BB_Upper': [105.0, 106.0, 108.0],
            'BB_ZScore': [0.2, 0.5, 1.5],
            'Volume_Surge': [10.0, 20.0, 50.0],
        }
    )


def _card_test_prediction(*, model_probabilities_available: bool = True) -> PredictionResult:
    return PredictionResult(
        direction='UP',
        confidence=0.7,
        prob_dict={'UP': 0.7, 'DOWN': 0.3, 'HOLD': 0.3},
        is_high_confidence=True,
        winning_class='UP',
        feature_columns=(),
        xgb_prob=0.7,
        lgbm_prob=0.65,
        ensemble_prob=0.68,
        model_probabilities_available=model_probabilities_available,
    )


def test_actionable_card_triggers_explicit_exit_zone_from_raw_indicators() -> None:
    result = derive_actionable_card_metrics(_card_test_frame(), _card_test_prediction(), 'WAIT')

    assert result['trend_exhaustion_watch'] is True
    assert result['transition'].badge_label == 'TREND EXHAUSTION WATCH — EXIT ZONE'
    assert result['exit_pct'] > 50.0
    assert 0.0 <= result['buy_transition_pct'] <= 100.0


def test_unavailable_component_models_use_neutral_conviction_priors() -> None:
    result = derive_actionable_card_metrics(
        _card_test_frame(),
        _card_test_prediction(model_probabilities_available=False),
        'WAIT',
    )

    assert result['conviction'] == 0.5
    assert 'XGBoost/LightGBM outputs are unavailable' in result['guidance']
    assert 'xgb_p_up' not in result['transition'].component_scores


def test_news_sentiment_is_same_for_feature_frame_and_radar(monkeypatch) -> None:
    monkeypatch.setattr(
        news_panel,
        'get_cached_stock_news',
        lambda symbol, max_articles=6: ({'sentiment_score': 0.4},),
    )

    feature_score = news_panel.compute_symbol_sentiment_score(
        'INFY.NS', features={'rsi_14': 5.0}
    )
    radar_score = news_panel.compute_symbol_sentiment_score(
        'INFY.NS', features={'rsi_14': 95.0}
    )

    assert feature_score == radar_score == 0.4


def test_news_payload_adapter_handles_frozen_upstream_article_type() -> None:
    article = news_panel._article_from_payload(
        {
            'headline': 'Order win drives growth',
            'timestamp_iso': '2026-01-01T00:00:00+00:00',
        }
    )

    assert article.sentiment == 'Positive'
    assert article.sentiment_score > 0.0
