# Indian Stock Market Prediction System

A clean, beginner-friendly Python project structure for an AI-powered Indian stock market prediction system. This production-style scaffold supports:

- Historical stock analysis
- Live NSE stock prediction
- Machine learning training
- Streamlit dashboard visualization
- NSE market timing logic

## Project structure

- `config/` - application settings and market timing helpers
- `data/` - historical data loading and data repository
- `indicators/` - technical indicator calculations and signal logic
- `live_data/` - live NSE quote ingestion and streaming helpers
- `dashboard/` - Streamlit dashboard components and app
- `utils/` - logging, environment loading, and shared constants
- `training/` - training pipelines and model orchestration
- `predictions/` - prediction logic and evaluation helpers
- `models/` - model definitions, training helpers, and persistence
- `logs/` - runtime log files

## Requirements

- Python 3.12
- Windows-compatible development environment
- VS Code

## Setup

1. Create a Python 3.12 virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Configuration

This project uses a secure, environment-based configuration system with comprehensive validation.

### Environment Variables

Create a `.env` file in the project root with the following variables:

```bash
# =============================================================================
# STOCK MARKET AI CONFIGURATION - SECURE ENVIRONMENT VARIABLES
# =============================================================================

# APPLICATION ENVIRONMENT
APP_ENV=development  # development, staging, production

# =============================================================================
# API KEYS - STORE SECURELY (Never commit to version control!)
# =============================================================================
NSE_API_KEY=your_nse_api_key_here
DATA_API_KEY=your_data_api_key_here
OPENAI_API_KEY=your_openai_key_here
DATABASE_URL=your_database_connection_string

# =============================================================================
# MODEL CONFIGURATION
# =============================================================================
MODEL_DIR=models
DEFAULT_MODEL_VERSION=v1.0
RANDOM_FOREST_MODEL=random_forest_classifier.pkl
LSTM_MODEL=lstm_predictor.pkl

# =============================================================================
# STOCK SYMBOLS CONFIGURATION
# =============================================================================
DEFAULT_STOCK_SYMBOLS=RELIANCE.NS,TCS.NS,INFY.NS,HDFCBANK.NS,ICICIBANK.NS
# Nifty 50 symbols (comma-separated)
NIFTY_50_SYMBOLS=ADANIPORTS.NS,ADANIENT.NS,...(full list in .env)
BANKING_SYMBOLS=HDFCBANK.NS,ICICIBANK.NS,KOTAKBANK.NS
IT_SYMBOLS=TCS.NS,INFY.NS,WIPRO.NS,HCLTECH.NS

# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================
LOG_LEVEL=INFO  # DEBUG, INFO, WARNING, ERROR, CRITICAL
MAX_LOG_SIZE_MB=5
LOG_BACKUP_COUNT=5

# =============================================================================
# STREAMLIT DASHBOARD CONFIGURATION
# =============================================================================
STREAMLIT_PORT=8501
STREAMLIT_HOST=localhost
STREAMLIT_DEBUG=true

# =============================================================================
# TRAINING CONFIGURATION
# =============================================================================
TRAIN_TEST_SPLIT=0.8
RANDOM_SEED=42
BATCH_SIZE=32
EPOCHS=100

# =============================================================================
# LIVE DATA CONFIGURATION
# =============================================================================
LIVE_UPDATE_INTERVAL=60
MAX_API_RETRIES=3
API_TIMEOUT=30

# =============================================================================
# PREDICTION CONFIGURATION
# =============================================================================
PREDICTION_THRESHOLD=0.6
MAX_DAILY_PREDICTIONS=100
PREDICTION_CACHE_TTL=300
```

### Security Best Practices

1. **Never commit `.env` files** - Add `.env` to `.gitignore`
2. **Use strong, unique API keys** - Rotate keys regularly
3. **Environment-specific keys** - Different keys for dev/staging/prod
4. **Store sensitive keys securely** - Use environment vaults when possible
5. **Validate at startup** - Application fails fast if critical config missing
6. **Never log sensitive values** - API keys are marked `repr=False`

### Configuration Validation

Run the configuration validator to ensure your setup is correct:

```bash
python validate_config.py
```

This will:
- ✅ Load and validate all environment variables
- ✅ Check API key configuration status (without revealing keys)
- ✅ Verify directory structure exists
- ✅ Display configuration summary
- ✅ Warn about production requirements

### Configuration Classes

The configuration is organized into typed dataclasses:

- `APIConfig` - API keys and connection settings
- `ModelConfig` - Model paths and versions
- `StockSymbols` - Stock symbol lists
- `DataConfig` - Data directory paths
- `LoggingConfig` - Logging settings
- `StreamlitConfig` - Dashboard settings
- `TrainingConfig` - ML training parameters
- `LiveDataConfig` - Live data streaming settings
- `PredictionConfig` - Prediction engine settings

### Usage in Code

```python
from config.settings import AppConfig

# Load validated configuration
config = AppConfig.load()

# Access configuration sections
api_key = config.api.nse_api_key  # Secure access
model_path = config.models.random_forest_path  # Full path
symbols = config.stocks.default_symbols  # List of symbols
log_level = config.get_log_level()  # Environment-appropriate level
```

## Run
- Predict using saved model: `python main.py --mode predict --symbol INFY.NS`
- Launch dashboard: `streamlit run dashboard/app.py`

## Notes

This project is designed to scale from a small stock universe to Nifty 50-level workflows.
Refer to the comments inside each module for more details on architecture and extension points.
