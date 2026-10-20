"""Evaluation helpers for prediction quality assessment."""
import numpy as np


def mean_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute the mean absolute percentage error (MAPE)."""
    return float(np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), 1e-8))) * 100)


def root_mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute the root mean squared error (RMSE)."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
