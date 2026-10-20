"""Shared constants used across the project."""
from pathlib import Path

DEFAULT_STOCK_SYMBOLS = [
    'RELIANCE.NS', 'TCS.NS', 'INFY.NS', 'HDFC.NS', 'ICICIBANK.NS'
]

MODEL_FILENAME = 'stock_predictor.joblib'
DEFAULT_DATA_WINDOW = 90
DATE_FORMAT = '%Y-%m-%d'
DATA_DIR = Path('data')
