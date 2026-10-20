"""
Secure application configuration using environment variables.

SECURITY BEST PRACTICES IMPLEMENTED:
1. Type-safe environment variable loading
2. Validation of required variables at startup
3. Secure handling of sensitive API keys (never logged)
4. Environment-specific validation rules
5. Path validation and normalization
6. Fail-fast approach for production deployments
7. Comprehensive error messages without exposing sensitive data
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from utils.env_loader import (
    load_env,
    safe_get_env_var,
    validate_environment,
    EnvironmentValidationError
)


@dataclass
class APIConfig:
    """API-related configuration with secure key handling."""
    nse_api_key: str = field(default="", repr=False)  # Never show in repr
    data_api_key: str = field(default="", repr=False)  # Never show in repr
    openai_api_key: str = field(default="", repr=False)  # Never show in repr
    database_url: str = field(default="", repr=False)  # Never show in repr
    max_api_retries: int = 3
    api_timeout: int = 30


@dataclass
class ModelConfig:
    """Model-related configuration."""
    model_dir: Path
    default_model_version: str
    random_forest_model: str
    lstm_model: str

    @property
    def random_forest_path(self) -> Path:
        """Get full path to random forest model."""
        return self.model_dir / self.random_forest_model

    @property
    def lstm_model_path(self) -> Path:
        """Get full path to LSTM model."""
        return self.model_dir / self.lstm_model


@dataclass
class StockSymbols:
    """Stock symbol configuration."""
    default_symbols: List[str]
    nifty_50_symbols: List[str]
    banking_symbols: List[str]
    it_symbols: List[str]


@dataclass
class DataConfig:
    """Data directory configuration."""
    raw_data_dir: Path
    processed_data_dir: Path
    cache_dir: Path


@dataclass
class LoggingConfig:
    """Logging configuration."""
    log_level: str
    max_log_size_mb: int
    log_backup_count: int


@dataclass
class StreamlitConfig:
    """Streamlit dashboard configuration."""
    port: int
    host: str
    debug: bool


@dataclass
class TrainingConfig:
    """Model training configuration."""
    train_test_split: float
    random_seed: int
    batch_size: int
    epochs: int


@dataclass
class LiveDataConfig:
    """Live data streaming configuration."""
    update_interval: int
    max_api_retries: int
    api_timeout: int


@dataclass
class PredictionConfig:
    """Prediction engine configuration."""
    confidence_threshold: float
    max_daily_predictions: int
    prediction_cache_ttl: int


@dataclass
class AppConfig:
    """
    Main application configuration with secure environment variable handling.

    SECURITY: This class centralizes all configuration loading and validation.
    Sensitive values are marked with repr=False to prevent accidental logging.
    """
    # Environment
    app_env: str

    # Sub-configurations
    api: APIConfig
    models: ModelConfig
    stocks: StockSymbols
    data: DataConfig
    logging: LoggingConfig
    streamlit: StreamlitConfig
    training: TrainingConfig
    live_data: LiveDataConfig
    predictions: PredictionConfig

    @classmethod
    def load(cls) -> 'AppConfig':
        """
        Load configuration from environment variables with validation.

        SECURITY: This method validates all environment variables before
        creating the configuration object, ensuring the application fails
        fast if critical configuration is missing.

        Returns:
            AppConfig: Validated application configuration

        Raises:
            EnvironmentValidationError: If environment validation fails
        """
        env_vars = load_env()

        # Extract and validate app environment first
        app_env = safe_get_env_var('APP_ENV', env_vars, 'development', str)

        # Validate environment variables
        validate_environment(env_vars, app_env)

        # Load sub-configurations
        api_config = cls._load_api_config(env_vars)
        model_config = cls._load_model_config(env_vars)
        stock_config = cls._load_stock_config(env_vars)
        data_config = cls._load_data_config(env_vars)
        logging_config = cls._load_logging_config(env_vars)
        streamlit_config = cls._load_streamlit_config(env_vars)
        training_config = cls._load_training_config(env_vars)
        live_data_config = cls._load_live_data_config(env_vars)
        prediction_config = cls._load_prediction_config(env_vars)

        return cls(
            app_env=app_env,
            api=api_config,
            models=model_config,
            stocks=stock_config,
            data=data_config,
            logging=logging_config,
            streamlit=streamlit_config,
            training=training_config,
            live_data=live_data_config,
            predictions=prediction_config,
        )

    @classmethod
    def _load_api_config(cls, env_vars: dict) -> APIConfig:
        """Load API configuration with secure key handling."""
        return APIConfig(
            nse_api_key=safe_get_env_var('NSE_API_KEY', env_vars, '', str),
            data_api_key=safe_get_env_var('DATA_API_KEY', env_vars, '', str),
            openai_api_key=safe_get_env_var('OPENAI_API_KEY', env_vars, '', str),
            database_url=safe_get_env_var('DATABASE_URL', env_vars, '', str),
            max_api_retries=safe_get_env_var('MAX_API_RETRIES', env_vars, 3, int),
            api_timeout=safe_get_env_var('API_TIMEOUT', env_vars, 30, int),
        )

    @classmethod
    def _load_model_config(cls, env_vars: dict) -> ModelConfig:
        """Load model configuration with path validation."""
        model_dir = Path(safe_get_env_var('MODEL_DIR', env_vars, 'models', str))

        # Ensure model directory exists
        model_dir.mkdir(parents=True, exist_ok=True)

        return ModelConfig(
            model_dir=model_dir,
            default_model_version=safe_get_env_var('DEFAULT_MODEL_VERSION', env_vars, 'v1.0', str),
            random_forest_model=safe_get_env_var('RANDOM_FOREST_MODEL', env_vars, 'random_forest_classifier.pkl', str),
            lstm_model=safe_get_env_var('LSTM_MODEL', env_vars, 'lstm_predictor.pkl', str),
        )

    @classmethod
    def _load_stock_config(cls, env_vars: dict) -> StockSymbols:
        """Load stock symbol configuration."""
        return StockSymbols(
            default_symbols=safe_get_env_var('DEFAULT_STOCK_SYMBOLS', env_vars,
                                           'RELIANCE.NS,TCS.NS,INFY.NS,HDFCBANK.NS,ICICIBANK.NS', list),
            nifty_50_symbols=safe_get_env_var('NIFTY_50_SYMBOLS', env_vars, [], list),
            banking_symbols=safe_get_env_var('BANKING_SYMBOLS', env_vars, [], list),
            it_symbols=safe_get_env_var('IT_SYMBOLS', env_vars, [], list),
        )

    @classmethod
    def _load_data_config(cls, env_vars: dict) -> DataConfig:
        """Load data directory configuration."""
        raw_data_dir = Path(safe_get_env_var('RAW_DATA_DIR', env_vars, 'data/raw', str))
        processed_data_dir = Path(safe_get_env_var('PROCESSED_DATA_DIR', env_vars, 'data/ml/processed', str))
        cache_dir = Path(safe_get_env_var('CACHE_DIR', env_vars, '.cache', str))

        # Ensure directories exist
        for directory in [raw_data_dir, processed_data_dir, cache_dir]:
            directory.mkdir(parents=True, exist_ok=True)

        return DataConfig(
            raw_data_dir=raw_data_dir,
            processed_data_dir=processed_data_dir,
            cache_dir=cache_dir,
        )

    @classmethod
    def _load_logging_config(cls, env_vars: dict) -> LoggingConfig:
        """Load logging configuration."""
        return LoggingConfig(
            log_level=safe_get_env_var('LOG_LEVEL', env_vars, 'INFO', str),
            max_log_size_mb=safe_get_env_var('MAX_LOG_SIZE_MB', env_vars, 5, int),
            log_backup_count=safe_get_env_var('LOG_BACKUP_COUNT', env_vars, 5, int),
        )

    @classmethod
    def _load_streamlit_config(cls, env_vars: dict) -> StreamlitConfig:
        """Load Streamlit configuration."""
        return StreamlitConfig(
            port=safe_get_env_var('STREAMLIT_PORT', env_vars, 8501, int),
            host=safe_get_env_var('STREAMLIT_HOST', env_vars, 'localhost', str),
            debug=safe_get_env_var('STREAMLIT_DEBUG', env_vars, True, bool),
        )

    @classmethod
    def _load_training_config(cls, env_vars: dict) -> TrainingConfig:
        """Load training configuration."""
        return TrainingConfig(
            train_test_split=safe_get_env_var('TRAIN_TEST_SPLIT', env_vars, 0.8, float),
            random_seed=safe_get_env_var('RANDOM_SEED', env_vars, 42, int),
            batch_size=safe_get_env_var('BATCH_SIZE', env_vars, 32, int),
            epochs=safe_get_env_var('EPOCHS', env_vars, 100, int),
        )

    @classmethod
    def _load_live_data_config(cls, env_vars: dict) -> LiveDataConfig:
        """Load live data configuration."""
        return LiveDataConfig(
            update_interval=safe_get_env_var('LIVE_UPDATE_INTERVAL', env_vars, 60, int),
            max_api_retries=safe_get_env_var('MAX_API_RETRIES', env_vars, 3, int),
            api_timeout=safe_get_env_var('API_TIMEOUT', env_vars, 30, int),
        )

    @classmethod
    def _load_prediction_config(cls, env_vars: dict) -> PredictionConfig:
        """Load prediction configuration."""
        return PredictionConfig(
            confidence_threshold=safe_get_env_var('PREDICTION_THRESHOLD', env_vars, 0.6, float),
            max_daily_predictions=safe_get_env_var('MAX_DAILY_PREDICTIONS', env_vars, 100, int),
            prediction_cache_ttl=safe_get_env_var('PREDICTION_CACHE_TTL', env_vars, 300, int),
        )

    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.app_env == 'production'

    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.app_env == 'development'

    def get_log_level(self) -> str:
        """Get the appropriate log level for the current environment."""
        if self.is_production():
            return 'WARNING' if self.logging.log_level == 'INFO' else self.logging.log_level
        return self.logging.log_level
