"""Centralized logging system for the stock market AI project."""
from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

# Constants
LOG_DIR = Path('logs')
LOG_FORMAT = '%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s'
DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
MAX_BYTES = 5_242_880  # 5MB
BACKUP_COUNT = 5

# Ensure log directory exists
LOG_DIR.mkdir(parents=True, exist_ok=True)


class StockLogger:
    """Centralized logger for the stock market AI project."""

    _loggers = {}

    @classmethod
    def get_logger(cls, name: str, level: Optional[str] = None) -> logging.Logger:
        """Get or create a logger with the specified name."""
        if name in cls._loggers:
            return cls._loggers[name]

        logger = logging.getLogger(name)
        logger.setLevel(getattr(logging, level.upper(), logging.INFO) if level else logging.INFO)

        # Prevent duplicate handlers
        if not logger.handlers:
            # File handler with rotation
            file_handler = RotatingFileHandler(
                LOG_DIR / f'{name.replace(".", "_")}.log',
                maxBytes=MAX_BYTES,
                backupCount=BACKUP_COUNT,
                encoding='utf-8'
            )
            file_handler.setFormatter(logging.Formatter(LOG_FORMAT, DATE_FORMAT))
            logger.addHandler(file_handler)

            # Console handler
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(logging.Formatter(LOG_FORMAT, DATE_FORMAT))
            logger.addHandler(console_handler)

        cls._loggers[name] = logger
        return logger

    @classmethod
    def configure_root_logger(cls, level: str = 'INFO') -> None:
        """Configure the root logger for the entire application."""
        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, level.upper(), logging.INFO))

        # Remove any existing handlers
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

        # Add rotating file handler for main application log
        main_handler = RotatingFileHandler(
            LOG_DIR / 'application.log',
            maxBytes=MAX_BYTES,
            backupCount=BACKUP_COUNT,
            encoding='utf-8'
        )
        main_handler.setFormatter(logging.Formatter(LOG_FORMAT, DATE_FORMAT))
        root_logger.addHandler(main_handler)

        # Add console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(logging.Formatter(LOG_FORMAT, DATE_FORMAT))
        root_logger.addHandler(console_handler)

        # Log the configuration
        root_logger.info(f"Root logger configured with level {level}")


# Convenience functions for specific logging needs
def get_prediction_logger() -> logging.Logger:
    """Get logger for prediction-related events."""
    return StockLogger.get_logger('predictions')


def get_training_logger() -> logging.Logger:
    """Get logger for model training events."""
    return StockLogger.get_logger('training')


def get_live_data_logger() -> logging.Logger:
    """Get logger for live data streaming events."""
    return StockLogger.get_logger('live_data')


def get_api_logger() -> logging.Logger:
    """Get logger for API-related events."""
    return StockLogger.get_logger('api')


def get_simulation_logger() -> logging.Logger:
    """Get logger for paper trading simulation events."""
    return StockLogger.get_logger('simulation')


def get_dashboard_logger() -> logging.Logger:
    """Get logger for dashboard events."""
    return StockLogger.get_logger('dashboard')


# Legacy function for backward compatibility
def configure_logger(log_level: str = 'INFO') -> None:
    """Legacy function - use StockLogger.configure_root_logger() instead."""
    StockLogger.configure_root_logger(log_level)


# Logging utility functions
def log_prediction(symbol: str, direction: str, confidence: float, **kwargs) -> None:
    """Log prediction events with structured data."""
    logger = get_prediction_logger()
    extra_info = ' | '.join(f"{k}={v}" for k, v in kwargs.items())
    message = f"PREDICTION | {symbol} | {direction} | {confidence:.3f}"
    if extra_info:
        message += f" | {extra_info}"
    logger.info(message)


def log_trade(symbol: str, trade_type: str, quantity: int, price: float, **kwargs) -> None:
    """Log trading events."""
    logger = get_simulation_logger()
    extra_info = ' | '.join(f"{k}={v}" for k, v in kwargs.items())
    message = f"TRADE | {symbol} | {trade_type} | {quantity} shares @ ₹{price:.2f}"
    if extra_info:
        message += f" | {extra_info}"
    logger.info(message)


def log_api_call(endpoint: str, method: str = 'GET', status_code: Optional[int] = None, **kwargs) -> None:
    """Log API calls."""
    logger = get_api_logger()
    status_info = f" | Status: {status_code}" if status_code else ""
    extra_info = ' | '.join(f"{k}={v}" for k, v in kwargs.items())
    message = f"API | {method} {endpoint}{status_info}"
    if extra_info:
        message += f" | {extra_info}"

    if status_code and status_code >= 400:
        logger.error(message)
    else:
        logger.info(message)


def log_model_training(model_name: str, status: str, **kwargs) -> None:
    """Log model training events."""
    logger = get_training_logger()
    extra_info = ' | '.join(f"{k}={v}" for k, v in kwargs.items())
    message = f"TRAINING | {model_name} | {status}"
    if extra_info:
        message += f" | {extra_info}"
    logger.info(message)


def log_live_data_event(event_type: str, symbol: str, **kwargs) -> None:
    """Log live data streaming events."""
    logger = get_live_data_logger()
    extra_info = ' | '.join(f"{k}={v}" for k, v in kwargs.items())
    message = f"LIVE_DATA | {event_type} | {symbol}"
    if extra_info:
        message += f" | {extra_info}"
    logger.info(message)


def log_error(component: str, error_message: str, **kwargs) -> None:
    """Log errors with component information."""
    logger = StockLogger.get_logger(component)
    extra_info = ' | '.join(f"{k}={v}" for k, v in kwargs.items())
    message = f"ERROR | {error_message}"
    if extra_info:
        message += f" | {extra_info}"
    logger.error(message)


def log_performance_metric(metric_name: str, value: float, **kwargs) -> None:
    """Log performance metrics."""
    logger = get_simulation_logger()
    extra_info = ' | '.join(f"{k}={v}" for k, v in kwargs.items())
    message = f"PERFORMANCE | {metric_name} | {value}"
    if extra_info:
        message += f" | {extra_info}"
    logger.info(message)
