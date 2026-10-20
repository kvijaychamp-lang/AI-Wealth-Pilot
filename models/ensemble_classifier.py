"""Ensemble training and inference for stock direction prediction."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, precision_score,
                             recall_score)
from sklearn.model_selection import TimeSeriesSplit, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier
except ImportError as error:
    raise ImportError(
        'XGBoost is required to run ensemble_classifier.py. Install it with: pip install xgboost'
    ) from error

try:
    from lightgbm import LGBMClassifier
except ImportError as error:
    raise ImportError(
        'LightGBM is required to run ensemble_classifier.py. Install it with: pip install lightgbm'
    ) from error

logger = logging.getLogger(__name__)

MODEL_OUTPUT_PATH = Path('models') / 'saved' / 'ensemble_stock_direction.joblib'


def create_target_column(
    df: pd.DataFrame,
    price_col: str = 'Close',
    target_col: str = 'Target',
    shift_period: int = 1,
) -> pd.DataFrame:
    """Create a binary next-day direction target for time-series data.

    The target is built using Close[t+1] > Close[t]. The final row is dropped
    after shifting because its next-day price is unavailable, which avoids
    leaking future values into the feature matrix.
    """
    if price_col not in df.columns:
        logger.error('Price column %s not found in dataset', price_col)
        raise ValueError(f'Price column {price_col} not found in dataset')

    data = df.copy()
    data[target_col] = np.where(
        data[price_col].shift(-shift_period) > data[price_col],
        1,
        0,
    )
    # Remove the last row with no valid next-day target.
    data = data.iloc[:-shift_period].reset_index(drop=True)
    logger.info('Created binary target column %s using price column %s', target_col, price_col)
    return data


def prepare_features(
    df: pd.DataFrame,
    target_col: str = 'Target',
    date_col: str = 'Date',
) -> tuple[pd.DataFrame, pd.Series]:
    """Prepare features and labels, dropping columns that cause lookahead bias."""
    labeled_df = create_target_column(df, target_col=target_col)

    if date_col in labeled_df.columns:
        labeled_df = labeled_df.drop(columns=[date_col])
        logger.debug('Dropped date column from feature set: %s', date_col)

    if target_col not in labeled_df.columns:
        raise ValueError(f'Missing target column: {target_col}')

    X = labeled_df.drop(columns=[target_col])
    y = labeled_df[target_col]

    if X.empty:
        raise ValueError('No features remain after dropping target and date columns')

    return X, y


def build_ensemble_pipeline(random_state: int = 42) -> Pipeline:
    """Build a scikit-learn pipeline with scaling and a soft voting ensemble."""
    xgb = XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        use_label_encoder=False,
        max_depth=4,
        learning_rate=0.03,
        n_estimators=300,
        reg_alpha=0.5,
        reg_lambda=1.5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=random_state,
        n_jobs=-1,
        verbosity=0,
    )
    lgbm = LGBMClassifier(
        objective='binary',
        boosting_type='gbdt',
        max_depth=5,
        learning_rate=0.03,
        n_estimators=300,
        reg_alpha=0.5,
        reg_lambda=1.5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=random_state,
        n_jobs=-1,
    )

    voting = VotingClassifier(
        estimators=[('xgb', xgb), ('lgbm', lgbm)],
        voting='soft',
        n_jobs=-1,
    )

    pipeline = Pipeline(
        [
            # StandardScaler is fit inside each training fold only, preventing
            # scaling from leaking validation information into the training step.
            ('scaler', StandardScaler()),
            ('classifier', voting),
        ]
    )
    return pipeline


def train_ensemble_model(
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    random_state: int = 42,
) -> tuple[Pipeline, dict[str, np.ndarray]]:
    """Train the ensemble using TimeSeriesSplit cross-validation."""
    if X.shape[0] != len(y):
        raise ValueError('Feature matrix and target vector must have the same number of rows')

    pipeline = build_ensemble_pipeline(random_state=random_state)
    cv = TimeSeriesSplit(n_splits=n_splits)

    logger.info('Starting time-series cross-validation with %d splits', n_splits)
    cv_results = cross_validate(
        pipeline,
        X,
        y,
        cv=cv,
        scoring=['accuracy', 'precision', 'recall'],
        return_train_score=False,
        n_jobs=-1,
    )

    logger.info(
        'CV results: accuracy=%.4f +/- %.4f, precision=%.4f +/- %.4f, recall=%.4f +/- %.4f',
        np.mean(cv_results['test_accuracy']),
        np.std(cv_results['test_accuracy']),
        np.mean(cv_results['test_precision']),
        np.std(cv_results['test_precision']),
        np.mean(cv_results['test_recall']),
        np.std(cv_results['test_recall']),
    )

    # Fit on the full dataset after validating with time-series CV.
    pipeline.fit(X, y)
    return pipeline, cv_results


def save_model(model: BaseEstimator, path: Path = MODEL_OUTPUT_PATH) -> Path:
    """Serialize the trained ensemble to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    logger.info('Saved ensemble model to %s', path)
    return path


def load_model(path: Path = MODEL_OUTPUT_PATH) -> BaseEstimator:
    """Load a trained model from disk."""
    if not path.exists():
        raise FileNotFoundError(f'Model file not found: {path}')
    return joblib.load(path)


def get_ensemble_prediction(
    model: BaseEstimator,
    latest_features_row: pd.DataFrame | np.ndarray,
    user_threshold: float = 0.55,
) -> dict[str, Any]:
    """Return an UP/HOLD signal and confidence percent from the ensemble."""
    if isinstance(latest_features_row, pd.Series):
        latest_features_row = latest_features_row.to_frame().T

    if isinstance(latest_features_row, np.ndarray) and latest_features_row.ndim == 1:
        latest_features_row = latest_features_row.reshape(1, -1)

    proba = model.predict_proba(latest_features_row)
    up_probability = float(proba[0][1])
    confidence_pct = round(up_probability * 100, 2)

    if up_probability >= user_threshold:
        direction = 'UP'
    else:
        direction = 'HOLD/NO SIGNAL'

    return {
        'Direction': direction,
        'Confidence': f'{confidence_pct}%',
        'Probability': up_probability,
    }


def evaluate_final_model(
    model: BaseEstimator,
    X: pd.DataFrame,
    y: pd.Series,
) -> dict[str, Any]:
    """Evaluate the fully trained model on a dataset."""
    if X.shape[0] != len(y):
        raise ValueError('Feature matrix and target vector lengths do not match')

    predictions = model.predict(X)
    accuracy = accuracy_score(y, predictions)
    precision = precision_score(y, predictions, zero_division=0)
    recall = recall_score(y, predictions, zero_division=0)
    matrix = confusion_matrix(y, predictions)
    logger.info('Final evaluation: accuracy=%.4f precision=%.4f recall=%.4f', accuracy, precision, recall)

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'confusion_matrix': matrix,
    }


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s',
    )

    sample_path = Path('data') / 'processed' / 'RELIANCE_NS.csv'
    if not sample_path.exists():
        raise FileNotFoundError('Sample dataset not found: %s' % sample_path)

    raw_data = pd.read_csv(sample_path)
    X, y = prepare_features(raw_data)

    model, cv_results = train_ensemble_model(X, y, n_splits=5)
    save_model(model)
    metrics = evaluate_final_model(model, X, y)

    logger.info('Model trained and saved successfully. Full dataset metrics: %s', metrics)
    latest_row = X.tail(1)
    signal = get_ensemble_prediction(model, latest_row, user_threshold=0.55)
    logger.info('Latest row signal: %s', signal)
