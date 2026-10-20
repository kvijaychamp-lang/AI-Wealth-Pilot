"""Prepare stock prediction datasets for machine learning training."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)
ML_DATA_DIR = Path('data') / 'ml'


def ensure_directory(path: Path) -> None:
    """Create a directory if it does not exist."""
    path.mkdir(parents=True, exist_ok=True)
    logger.debug('Ensured directory exists: %s', path)


def validate_dataset(df: pd.DataFrame, required_columns: Iterable[str]) -> None:
    """Validate that the dataset contains all required columns."""
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        logger.error('Missing required columns: %s', missing)
        raise ValueError(f'Missing required columns: {missing}')
    logger.debug('Dataset validation passed')


def create_target_column(
    df: pd.DataFrame,
    price_col: str = 'Close',
    target_col: str = 'Target',
    shift_period: int = 1,
) -> pd.DataFrame:
    """Create a binary target for next-period price direction.

    1 means the next candle/day close price is higher than the current close.
    0 means the next candle/day close price is equal or lower.
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
    data = data.iloc[:-shift_period].reset_index(drop=True)
    logger.info('Created binary target column %s using price column %s', target_col, price_col)
    return data


def remove_data_leakage(
    df: pd.DataFrame,
    leakage_columns: Iterable[str] | None = None,
    target_column: str = 'Target',
    date_column: str = 'Date',
) -> pd.DataFrame:
    """Remove columns that may leak future information into training features."""
    data = df.copy()
    if leakage_columns:
        for col in leakage_columns:
            if col in data.columns:
                data = data.drop(columns=col)
                logger.info('Removed leakage column: %s', col)

    potential_leakage = [col for col in data.columns if 'future' in col.lower()]
    if potential_leakage:
        data = data.drop(columns=potential_leakage)
        logger.warning('Dropped inferred leakage columns: %s', potential_leakage)

    if date_column in data.columns:
        data = data.drop(columns=[date_column])
        logger.debug('Dropped date column from feature set: %s', date_column)

    if target_column in data.columns:
        data = data.drop(columns=[target_column])
        logger.debug('Dropped target column from feature set: %s', target_column)

    return data


def split_features_and_target(
    df: pd.DataFrame,
    target_column: str = 'Target',
    drop_columns: Iterable[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Split the dataset into feature matrix X and target vector y."""
    if target_column not in df.columns:
        logger.error('Target column %s missing from dataset', target_column)
        raise ValueError(f'Target column {target_column} missing from dataset')

    features = df.drop(columns=[target_column])
    if drop_columns:
        for col in drop_columns:
            if col in features.columns:
                features = features.drop(columns=[col])
                logger.debug('Dropped non-feature column: %s', col)

    logger.info('Split dataset into features and target')
    return features, df[target_column]


def normalize_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    columns_to_normalize: Iterable[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    """Normalize numeric features to avoid scale differences in machine learning."""
    scaler = StandardScaler()
    train_copy = X_train.copy()
    test_copy = X_test.copy()

    if columns_to_normalize is None:
        columns_to_normalize = train_copy.select_dtypes(include=[np.number]).columns

    norm_columns = [col for col in columns_to_normalize if col in train_copy.columns]
    if not norm_columns:
        logger.warning('No numeric columns found for normalization')
        return train_copy, test_copy, scaler

    logger.info('Normalizing columns: %s', list(norm_columns))
    train_copy[norm_columns] = scaler.fit_transform(train_copy[norm_columns])
    test_copy[norm_columns] = scaler.transform(test_copy[norm_columns])
    return train_copy, test_copy, scaler


def create_train_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
    shuffle: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Create a train-test split suitable for time series data when shuffle=False."""
    if not 0 < test_size < 1:
        logger.error('Invalid test_size: %s', test_size)
        raise ValueError('test_size must be between 0 and 1')

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        shuffle=shuffle,
    )
    logger.info('Created train-test split: %d train rows, %d test rows', len(X_train), len(X_test))
    return X_train, X_test, y_train, y_test


def save_ml_datasets(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    output_dir: Path = ML_DATA_DIR,
) -> dict[str, Path]:
    """Save the processed machine learning datasets to disk."""
    ensure_directory(output_dir)
    paths: dict[str, Path] = {}

    paths['X_train'] = output_dir / 'X_train.csv'
    paths['X_test'] = output_dir / 'X_test.csv'
    paths['y_train'] = output_dir / 'y_train.csv'
    paths['y_test'] = output_dir / 'y_test.csv'

    X_train.to_csv(paths['X_train'], index=False)
    X_test.to_csv(paths['X_test'], index=False)
    y_train.to_csv(paths['y_train'], index=False, header=True)
    y_test.to_csv(paths['y_test'], index=False, header=True)

    logger.info('Saved ML datasets to %s', output_dir)
    return paths


def prepare_ml_dataset(
    raw_df: pd.DataFrame,
    price_col: str = 'Close',
    target_col: str = 'Target',
    leakage_columns: Iterable[str] | None = None,
    drop_columns: Iterable[str] | None = None,
    test_size: float = 0.2,
    normalize: bool = True,
    output_dir: Path = ML_DATA_DIR,
) -> dict[str, Path]:
    """Full pipeline: create target, remove leakage, split, normalize, and save datasets."""
    validate_dataset(raw_df, [price_col])

    labeled_df = create_target_column(raw_df, price_col=price_col, target_col=target_col)
    features_df = remove_data_leakage(labeled_df, leakage_columns=leakage_columns, target_column=target_col)
    X, y = split_features_and_target(pd.concat([features_df, labeled_df[[target_col]]], axis=1), target_column=target_col, drop_columns=drop_columns)

    X_train, X_test, y_train, y_test = create_train_test_split(X, y, test_size=test_size, shuffle=False)

    if normalize:
        X_train, X_test, _scaler = normalize_features(X_train, X_test)

    return save_ml_datasets(X_train, X_test, y_train, y_test, output_dir=output_dir)


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s',
    )
    example_path = Path('data/processed/RELIANCE_NS.csv')
    if example_path.exists():
        raw_data = pd.read_csv(example_path)
        saved_paths = prepare_ml_dataset(raw_data)
        print('Saved ML datasets:')
        for label, path in saved_paths.items():
            print(label, path)
    else:
        print('Example file not found:', example_path)
