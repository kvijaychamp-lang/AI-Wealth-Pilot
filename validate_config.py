#!/usr/bin/env python3
"""
Configuration validation script.

This script validates that the secure configuration system is working correctly.
Run this after setting up your .env file to ensure all configurations are valid.
"""
import os
import sys
from pathlib import Path

# CRITICAL: Add project root to Python path for proper module resolution
PROJECT_ROOT = Path(__file__).parent.absolute()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Change working directory to project root
os.chdir(PROJECT_ROOT)

from config.settings import AppConfig
from utils.logger import StockLogger


def main():
    """Validate configuration and display loaded settings."""
    logger = StockLogger.get_logger('config_validation')

    try:
        logger.info("Loading application configuration...")

        # Load configuration
        config = AppConfig.load()

        logger.info("✅ Configuration loaded successfully!")
        logger.info(f"Environment: {config.app_env}")
        logger.info(f"Production mode: {config.is_production()}")

        # Display configuration sections (without sensitive data)
        logger.info("📊 Configuration Summary:")
        logger.info(f"  Models: {len(list(config.models.model_dir.glob('*.pkl')))} model files found")
        logger.info(f"  Default stocks: {len(config.stocks.default_symbols)} symbols configured")
        logger.info(f"  Nifty 50 stocks: {len(config.stocks.nifty_50_symbols)} symbols configured")
        logger.info(f"  Log level: {config.get_log_level()}")
        logger.info(f"  Streamlit port: {config.streamlit.port}")
        logger.info(f"  Training split: {config.training.train_test_split}")
        logger.info(f"  Prediction threshold: {config.predictions.confidence_threshold}")

        # Check API keys (without revealing them)
        api_keys_status = []
        if config.api.nse_api_key:
            api_keys_status.append("NSE API: ✅ Configured")
        else:
            api_keys_status.append("NSE API: ⚠️  Not configured")

        if config.api.data_api_key:
            api_keys_status.append("Data API: ✅ Configured")
        else:
            api_keys_status.append("Data API: ⚠️  Not configured")

        if config.api.openai_api_key:
            api_keys_status.append("OpenAI API: ✅ Configured")
        else:
            api_keys_status.append("OpenAI API: ⚠️  Not configured")

        logger.info("🔐 API Keys Status:")
        for status in api_keys_status:
            logger.info(f"  {status}")

        # Check directories
        logger.info("📁 Directory Status:")
        dirs_to_check = [
            ("Models", config.models.model_dir),
            ("Raw Data", config.data.raw_data_dir),
            ("Processed Data", config.data.processed_data_dir),
            ("Cache", config.data.cache_dir),
        ]

        for name, path in dirs_to_check:
            if path.exists():
                logger.info(f"  {name}: ✅ {path}")
            else:
                logger.warning(f"  {name}: ⚠️  {path} does not exist")

        logger.info("🎉 Configuration validation completed successfully!")

        # Production warnings
        if config.is_production():
            logger.warning("⚠️  PRODUCTION MODE: Ensure all API keys are properly configured!")
            if not config.api.nse_api_key:
                logger.error("❌ CRITICAL: NSE API key is required in production!")

    except Exception as e:
        logger.error(f"❌ Configuration validation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()