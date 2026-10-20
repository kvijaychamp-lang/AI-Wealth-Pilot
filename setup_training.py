#!/usr/bin/env python
"""Setup script to train model and prepare data files for the dashboard using mock data."""
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.absolute()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.chdir(PROJECT_ROOT)

from config.settings import AppConfig
from utils.logger import StockLogger
from utils.constants import DEFAULT_STOCK_SYMBOLS

logger = StockLogger.get_logger('setup_training')


def create_mock_data(symbol: str, days: int = 1825) -> pd.DataFrame:
    """Create realistic mock stock data for demonstration."""
    np.random.seed(hash(symbol) % 2**32)
    
    dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
    
    # Create realistic OHLCV data
    close_prices = 100 + np.cumsum(np.random.randn(days) * 1.5)
    
    return pd.DataFrame({
        'Date': dates,
        'Open': close_prices + np.random.randn(days) * 0.5,
        'High': close_prices + abs(np.random.randn(days) * 1),
        'Low': close_prices - abs(np.random.randn(days) * 1),
        'Close': close_prices,
        'Volume': np.random.randint(1000000, 10000000, days)
    })


def prepare_ml_dataset():
    """Prepare ML dataset with features and target variable."""
    logger.info("Preparing ML dataset with mock data...")
    
    all_features = []
    all_targets = []
    feature_names = None
    
    for symbol in DEFAULT_STOCK_SYMBOLS:
        logger.info(f"Creating mock data and features for {symbol}...")
        
        # Create mock data
        df = create_mock_data(symbol)
        
        # Build simple features
        df['SMA_10'] = df['Close'].rolling(window=10).mean()
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['EMA_12'] = df['Close'].ewm(span=12).mean()
        df['RSI'] = compute_rsi(df['Close'])
        df['MACD'] = df['Close'].ewm(span=12).mean() - df['Close'].ewm(span=26).mean()
        df['Volume_MA'] = df['Volume'].rolling(window=10).mean()
        df['Price_Change'] = df['Close'].pct_change()
        
        # Create target variable (1 if next day price goes up, 0 otherwise)
        df['target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
        
        # Drop NaN values
        df = df.dropna()
        
        if df.empty:
            logger.warning(f"No valid features for {symbol}, skipping")
            continue
        
        # Extract features
        feature_cols = ['SMA_10', 'SMA_20', 'EMA_12', 'RSI', 'MACD', 'Volume_MA', 'Price_Change']
        if not feature_names:
            feature_names = feature_cols
        
        X = df[feature_cols].values
        y = df['target'].values
        
        all_features.append(X)
        all_targets.append(y)
        
        logger.info(f"Created {len(X)} samples from {symbol}")
    
    # Combine all data
    X = np.vstack(all_features) if all_features else np.array([])
    y = np.concatenate(all_targets) if all_targets else np.array([])
    
    if len(X) == 0:
        logger.error("No features could be extracted!")
        return None, None, None
    
    logger.info(f"Total dataset: {len(X)} samples, {len(feature_names)} features")
    
    return X, y, feature_names


def compute_rsi(prices, period=14):
    """Compute Relative Strength Index."""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def train_model(X, y, config):
    """Train RandomForest classifier."""
    logger.info("Training RandomForest model...")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    logger.info(f"Train set: {len(X_train)}, Test set: {len(X_test)}")
    
    # Train model
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        min_samples_split=10,
        min_samples_leaf=4,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    
    logger.info(f"Train R² score: {train_score:.4f}")
    logger.info(f"Test R² score: {test_score:.4f}")
    
    return model


def save_model_and_features(model, feature_names, config):
    """Save trained model and feature names."""
    logger.info("Saving model and features...")
    
    # Create save directory
    model_dir = config.models.model_dir / 'saved'
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Save model
    model_path = config.models.random_forest_path
    joblib.dump(model, model_path)
    logger.info(f"Model saved to {model_path}")
    
    # Save feature names to CSV
    features_df = pd.DataFrame({'feature': feature_names})
    features_path = config.data.processed_data_dir / 'feature_names.csv'
    features_path.parent.mkdir(parents=True, exist_ok=True)
    features_df.to_csv(features_path, index=False)
    logger.info(f"Feature names saved to {features_path}")
    
    return model_path, features_path


def main():
    """Main setup function."""
    try:
        logger.info("=" * 80)
        logger.info("STARTING MODEL TRAINING AND DATA PREPARATION")
        logger.info("=" * 80)
        
        config = AppConfig.load()
        
        # Prepare ML dataset
        X, y, feature_names = prepare_ml_dataset()
        
        if X is None or len(X) == 0:
            logger.error("Could not prepare ML dataset!")
            return False
        
        # Train model
        model = train_model(X, y, config)
        
        # Save model and features
        model_path, features_path = save_model_and_features(model, feature_names, config)
        
        logger.info("=" * 80)
        logger.info("✅ MODEL TRAINING COMPLETED SUCCESSFULLY!")
        logger.info(f"Model saved: {model_path}")
        logger.info(f"Features saved: {features_path}")
        logger.info("=" * 80)
        
        return True
        
    except Exception as e:
        logger.error(f"Setup failed: {e}", exc_info=True)
        return False


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)

