"""Random Forest stock direction prediction model module."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, precision_score,
                             recall_score)

logger = logging.getLogger(__name__)


def load_csv(path: Path) -> pd.DataFrame:
    """Load dataset from a CSV file into a pandas DataFrame."""
    if not path.exists():
        logger.error('Dataset not found: %s', path)
        raise FileNotFoundError(f'Dataset not found: {path}')

    df = pd.read_csv(path)
    logger.info('Loaded dataset from %s with %d rows', path, len(df))
    return df


def prepare_train_test_data(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Ensure training and test sets are ready for model training."""
    if X_train.empty or X_test.empty:
        logger.error('Training or test datasets are empty')
        raise ValueError('Training and test datasets must not be empty')

    if y_train.empty or y_test.empty:
        logger.error('Target vectors are empty')
        raise ValueError('Target vectors must not be empty')

    logger.debug('Prepared train/test datasets')
    return X_train, X_test, y_train, y_test


def train_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_estimators: int = 100,
    random_state: int = 42,
    max_depth: int | None = None,
) -> RandomForestClassifier:
    """Train a Random Forest classifier on stock prediction features."""
    logger.info('Training Random Forest model with %d trees', n_estimators)
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        max_depth=max_depth,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    logger.info('Random Forest training complete')
    return model


def predict_direction(model: RandomForestClassifier, X: pd.DataFrame) -> np.ndarray:
    """Predict the stock direction labels for the feature set."""
    predictions = model.predict(X)
    logger.debug('Made %d predictions', len(predictions))
    return predictions


def format_direction_labels(predictions: np.ndarray) -> np.ndarray:
    """Convert numeric 0/1 labels into UP/DOWN strings."""
    return np.where(predictions == 1, 'UP', 'DOWN')


def evaluate_model(
    model: RandomForestClassifier,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, Any]:
    """Evaluate the trained model and return performance metrics."""
    predictions = predict_direction(model, X_test)
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, zero_division=0)
    recall = recall_score(y_test, predictions, zero_division=0)
    matrix = confusion_matrix(y_test, predictions)

    logger.info('Model evaluation completed: accuracy=%.4f precision=%.4f recall=%.4f',
                accuracy, precision, recall)

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'confusion_matrix': matrix,
        'predictions': predictions,
        'formatted_predictions': format_direction_labels(predictions),
    }


def save_model(model: RandomForestClassifier, model_path: Path) -> Path:
    """Save the trained Random Forest model to disk using joblib."""
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    logger.info('Saved model to %s', model_path)
    return model_path


def plot_feature_importance(
    model: RandomForestClassifier,
    feature_names: list[str],
    output_path: Path,
    top_n: int = 20,
) -> Path:
    """Render and save a feature importance chart for the trained model."""
    if not hasattr(model, 'feature_importances_'):
        logger.error('Model is not fitted or does not support feature importances')
        raise ValueError('Model must be fitted to plot feature importances')

    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]
    top_features = [feature_names[i] for i in indices]
    top_importances = importances[indices]

    plt.figure(figsize=(10, 6))
    plt.title('Top Feature Importances')
    plt.barh(range(len(top_features)), top_importances[::-1], align='center')
    plt.yticks(range(len(top_features)), top_features[::-1])
    plt.xlabel('Importance')
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()

    logger.info('Saved feature importance chart to %s', output_path)
    return output_path


def save_metrics(metrics: dict[str, Any], output_path: Path) -> Path:
    """Save evaluation metrics to a text report for auditing."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w', encoding='utf-8') as file:
        file.write('Random Forest Stock Direction Model Evaluation\n')
        file.write('==========================================\n')
        file.write(f"Accuracy: {metrics['accuracy']:.4f}\n")
        file.write(f"Precision: {metrics['precision']:.4f}\n")
        file.write(f"Recall: {metrics['recall']:.4f}\n")
        file.write('\nConfusion Matrix:\n')
        file.write(np.array2string(metrics['confusion_matrix']))
        file.write('\n')
    logger.info('Saved evaluation metrics to %s', output_path)
    return output_path


def run_training_pipeline(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    model_output_path: Path,
    importance_chart_path: Path,
    metrics_output_path: Path,
    feature_names: list[str],
    n_estimators: int = 100,
    random_state: int = 42,
    max_depth: int | None = None,
) -> dict[str, Any]:
    """Execute the full Random Forest training and evaluation pipeline."""
    X_train, X_test, y_train, y_test = prepare_train_test_data(
        X_train, X_test, y_train, y_test
    )

    model = train_random_forest(
        X_train, y_train,
        n_estimators=n_estimators,
        random_state=random_state,
        max_depth=max_depth,
    )

    metrics = evaluate_model(model, X_test, y_test)
    save_model(model, model_output_path)
    plot_feature_importance(model, feature_names, importance_chart_path)
    save_metrics(metrics, metrics_output_path)

    return metrics


def format_predictions_to_labels(predictions: np.ndarray) -> list[str]:
    """Format numeric predictions into human-readable UP/DOWN labels."""
    return format_direction_labels(predictions).tolist()


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s',
    )

    train_dir = Path('data') / 'ml'
    model_dir = Path('models') / 'saved'
    output_dir = Path('logs')
    try:
        X_train = pd.read_csv(train_dir / 'X_train.csv')
        X_test = pd.read_csv(train_dir / 'X_test.csv')
        y_train = pd.read_csv(train_dir / 'y_train.csv').iloc[:, 0]
        y_test = pd.read_csv(train_dir / 'y_test.csv').iloc[:, 0]

        metrics = run_training_pipeline(
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            model_output_path=model_dir / 'random_forest_stock_direction.joblib',
            importance_chart_path=output_dir / 'feature_importance.png',
            metrics_output_path=output_dir / 'evaluation_metrics.txt',
            feature_names=list(X_train.columns),
        )

        print('Training complete. Metrics:')
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print('Confusion matrix:')
        print(metrics['confusion_matrix'])
    except Exception as exc:
        logger.exception('Failed to run Random Forest training pipeline: %s', exc)
