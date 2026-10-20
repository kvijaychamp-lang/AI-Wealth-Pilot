"""
AI Stock Market Prediction System - Main Entry Point

WORKFLOW OVERVIEW:
1. CONFIGURATION: Load and validate environment settings
2. DATA LOADING: Fetch historical stock data from Yahoo Finance
3. PREPROCESSING: Clean and prepare data for analysis
4. FEATURE ENGINEERING: Calculate technical indicators and features
5. MODEL TRAINING: Train ML models on historical data
6. PREDICTION: Generate real-time stock predictions
7. DASHBOARD: Launch Streamlit web interface for visualization
8. LIVE TRADING: Connect to live market data (optional)

ARCHITECTURE:
- Modular design with clear separation of concerns
- Centralized logging and configuration management
- Error handling with graceful degradation
- Production-ready with proper resource management
"""
import argparse
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

# CRITICAL: Add project root to Python path for proper module resolution
PROJECT_ROOT = Path(__file__).parent.absolute()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Change working directory to project root
os.chdir(PROJECT_ROOT)

from config.settings import AppConfig
from utils.logger import StockLogger


class StockPredictionSystem:
    """
    Main application class that orchestrates all system components.

    This class provides a unified interface to all modules:
    - Data management (loading, preprocessing, storage)
    - Feature engineering (technical indicators, signals)
    - Model training and prediction
    - Live data streaming
    - Dashboard visualization
    - Market timing and scheduling
    """

    def __init__(self, config: AppConfig):
        """
        Initialize the stock prediction system with configuration.

        Args:
            config: Application configuration with all settings
        """
        self.config = config
        self.logger = StockLogger.get_logger('system')

        # Initialize core components
        self._data_manager = None
        self._feature_engineer = None
        self._model_manager = None
        self._prediction_engine = None
        self._live_data_streamer = None
        self._dashboard = None

        self.logger.info("Stock Prediction System initialized")
        self.logger.info(f"Environment: {config.app_env}")
        self.logger.info(f"Symbols configured: {len(config.stocks.default_symbols)}")

    def initialize_components(self) -> None:
        """Initialize all system components with proper error handling."""
        try:
            self.logger.info("Initializing system components...")

            # Data Management Component
            from data.loader import HistoricalDataLoader
            from data.repository import StockDataRepository
            self._data_loader = HistoricalDataLoader
            self._data_repository = StockDataRepository
            self.logger.info("[OK] Data management components initialized")

            # Feature Engineering Component
            from indicators.feature_builder import build_feature_set
            self._feature_builder = build_feature_set
            self.logger.info("[OK] Feature engineering components initialized")

            # Model Management Component
            from models.model_factory import ModelFactory
            self._model_factory = ModelFactory
            self.logger.info("[OK] Model management components initialized")

            # Prediction Engine Component
            from predictions.predictor import StockPredictor
            self._stock_predictor = StockPredictor
            self.logger.info("[OK] Prediction engine components initialized")

            # Live Data Component (optional)
            try:
                from live_data.live_stream import LiveDataStreamer
                self._live_streamer = LiveDataStreamer
                self.logger.info("[OK] Live data components initialized")
            except ImportError as e:
                self.logger.warning(f"Live data components not available: {e}")

            # Dashboard Component
            try:
                from dashboard.app import main as dashboard_main
                self._dashboard_main = dashboard_main
                self.logger.info("[OK] Dashboard components initialized")
            except ImportError as e:
                self.logger.warning(f"Dashboard components not available: {e}")

            self.logger.info("All components initialized successfully")

        except Exception as e:
            self.logger.error(f"Failed to initialize components: {e}")
            raise

    def run_training_pipeline(self, symbols: Optional[list[str]] = None) -> bool:
        """
        Execute the complete training pipeline.

        WORKFLOW:
        1. Load historical data for specified symbols
        2. Preprocess and clean data
        3. Calculate technical indicators
        4. Train machine learning model
        5. Save trained model

        Args:
            symbols: List of stock symbols to train on (uses config default if None)

        Returns:
            True if training completed successfully
        """
        if symbols is None:
            symbols = self.config.stocks.default_symbols

        self.logger.info(f"Starting training pipeline for {len(symbols)} symbols")
        self.logger.info(f"Symbols: {', '.join(symbols)}")

        try:
            # Step 1: Train model using the built training pipeline
            self.logger.info("Step 1: Starting model training...")
            from training.trainer import TrainingPipeline as ModelTrainer

            trainer = ModelTrainer(self.config)
            trainer.run()

            self.logger.info("✅ Training pipeline completed successfully")
            return True

        except Exception as e:
            self.logger.error(f"Training pipeline failed: {e}")
            return False

    def run_prediction(self, symbol: str) -> Optional[dict]:
        """
        Generate prediction for a specific stock symbol.

        WORKFLOW:
        1. Load latest market data for symbol
        2. Calculate current technical indicators
        3. Load trained model
        4. Generate prediction with confidence score

        Args:
            symbol: Stock symbol to predict

        Returns:
            Prediction result dictionary or None if failed
        """
        self.logger.info(f"Generating prediction for {symbol}")

        try:
            # Load predictor
            predictor = self._stock_predictor(self.config)

            # Generate prediction
            prediction = predictor.predict(symbol=symbol)

            result = {
                'symbol': symbol,
                'prediction': prediction,
                'timestamp': datetime.now().isoformat(),
                'status': 'success'
            }

            self.logger.info(f"Prediction generated: {symbol} -> {prediction}")
            return result

        except Exception as e:
            self.logger.error(f"Prediction failed for {symbol}: {e}")
            return {
                'symbol': symbol,
                'error': str(e),
                'timestamp': datetime.now().isoformat(),
                'status': 'failed'
            }

    def run_realtime_predictions(self, symbols: Optional[list[str]] = None) -> dict:
        """
        Generate real-time predictions for multiple symbols.

        Uses the real-time prediction engine for live market analysis.

        Args:
            symbols: List of symbols to predict (uses config default if None)

        Returns:
            Dictionary of prediction results
        """
        if symbols is None:
            symbols = self.config.stocks.default_symbols

        self.logger.info(f"Running real-time predictions for {len(symbols)} symbols")

        from zoneinfo import ZoneInfo
        from utils.market_schedule import is_market_active
        from predictions.realtime_engine import load_model, load_symbol_history, predict_stock_direction
        from indicators.feature_builder import build_feature_set

        now = datetime.now(tz=ZoneInfo('Asia/Kolkata'))
        if not is_market_active(now):
            # Market is closed — return paused state with last available close per symbol
            results = {}
            for symbol in symbols:
                try:
                    raw_df = load_symbol_history(symbol)
                    last_close = float(raw_df['Close'].iloc[-1]) if 'Close' in raw_df.columns else None
                    results[symbol] = {
                        'status': 'paused',
                        'data_paused': True,
                        'message': 'Data Paused: Market Closed',
                        'last_close': last_close,
                        'timestamp': now.isoformat(),
                    }
                except Exception as e:
                    self.logger.error(f'Failed to fetch last close for {symbol}: {e}')
                    results[symbol] = {
                        'status': 'failed',
                        'error': str(e),
                        'timestamp': now.isoformat(),
                    }

            self.logger.info('Market closed; realtime predictions paused for all symbols')
            return results

        # Market is active — run live predictions
        results = {}
        feature_cols = ['SMA_10', 'SMA_20', 'EMA_12', 'RSI', 'MACD', 'Volume_MA', 'Price_Change']

        model = load_model(self.config.models.random_forest_path)

        for symbol in symbols:
            try:
                raw_df = load_symbol_history(symbol)
                indicator_df = build_feature_set(raw_df)

                available_features = [c for c in feature_cols if c in indicator_df.columns]
                if not available_features:
                    raise RuntimeError(f'No model features available for {symbol}')

                latest_features = indicator_df[available_features].iloc[-1]

                direction, confidence, probabilities = predict_stock_direction(model, latest_features)

                results[symbol] = {
                    'direction': direction,
                    'confidence': confidence,
                    'probabilities': probabilities,
                    'timestamp': datetime.now().isoformat(),
                    'status': 'success',
                    'display_latest_close': float(raw_df['Close'].iloc[-1]) if 'Close' in raw_df.columns else None,
                }

            except Exception as e:
                self.logger.error(f"Real-time prediction failed for {symbol}: {e}")
                results[symbol] = {
                    'error': str(e),
                    'timestamp': datetime.now().isoformat(),
                    'status': 'failed'
                }

        successful = sum(1 for r in results.values() if r.get('status') == 'success')
        self.logger.info(f"Real-time predictions completed: {successful}/{len(symbols)} successful")
        return results

    def start_dashboard(self) -> None:
        """Launch the Streamlit dashboard for visualization."""
        self.logger.info("Starting Streamlit dashboard...")

        try:
            # Import and run dashboard
            from dashboard.app import main as dashboard_main
            self.logger.info("✅ Dashboard launched successfully")
            dashboard_main()

        except ImportError:
            self.logger.error("Dashboard module not available")
            results = {}

            # Use timezone-aware check to determine whether to run live predictions
            from zoneinfo import ZoneInfo
            from utils.market_schedule import is_market_active
            from predictions.realtime_engine import load_symbol_history

            now = datetime.now(tz=ZoneInfo('Asia/Kolkata'))
            market_active = is_market_active(now)

            if not market_active:
                # Market closed: do not append live rows or produce new timestamp bars.
                # Instead, freeze UI updates and provide the last available close per symbol.
                for symbol in symbols:
                    try:
                        raw_df = load_symbol_history(symbol)
                        last_close = float(raw_df['Close'].iloc[-1]) if 'Close' in raw_df.columns else None
                        results[symbol] = {
                            'status': 'paused',
                            'data_paused': True,
                            'message': 'Data Paused: Market Closed',
                            'last_close': last_close,
                            'timestamp': now.isoformat(),
                        }
                    except Exception as e:
                        self.logger.error(f'Failed to fetch last close for {symbol}: {e}')
                        results[symbol] = {
                            'status': 'failed',
                            'error': str(e),
                            'timestamp': now.isoformat(),
                        }

                self.logger.info('Market closed; realtime predictions paused for all symbols')
                return results

            # Market is active; proceed with live predictions
            for symbol in symbols:
                try:
                    # Use real-time engine for advanced predictions
                    from predictions.realtime_engine import (
                        load_model,
                        build_realtime_features,
                        predict_stock_direction,
                    )

                    # Load model and features
                    model = load_model(self.config.models.random_forest_path)
                    features = build_realtime_features(symbol, self.config.stocks.default_symbols)

                    # Generate prediction
                    direction, confidence, probabilities = predict_stock_direction(model, features)

                    results[symbol] = {
                        'direction': direction,
                        'confidence': confidence,
                        'probabilities': probabilities,
                        'timestamp': datetime.now().isoformat(),
                        'status': 'success'
                    }

                except Exception as e:
                    self.logger.error(f"Real-time prediction failed for {symbol}: {e}")
                    results[symbol] = {
                        'error': str(e),
                        'timestamp': datetime.now().isoformat(),
                        'status': 'failed'
                    }
        if symbols is None:
            symbols = self.config.stocks.default_symbols

        self.logger.info(f"Starting live data stream for {len(symbols)} symbols")

        try:
            from live_data.live_stream import LiveDataStreamer
            streamer = LiveDataStreamer(
                symbols,
                fetch_interval=self.config.live_data.update_interval,
                max_retries=self.config.live_data.max_api_retries,
            )
            streamer.start()
            self.logger.info("✅ Live data streaming started")

        except ImportError:
            self.logger.warning("Live data streaming not available")
        except Exception as e:
            self.logger.error(f"Live data streaming failed: {e}")
            raise

    def validate_system_health(self) -> dict:
        """
        Perform system health check.

        Returns:
            Dictionary with health status of all components
        """
        health_status = {
            'configuration': True,
            'data_loading': False,
            'model_loading': False,
            'prediction_engine': False,
            'dashboard': False,
            'live_data': False
        }

        # Check data loading
        try:
            from data.loader import HistoricalDataLoader
            loader = HistoricalDataLoader(symbols=['RELIANCE.NS'], start_date='2024-01-01', end_date='2024-01-02')
            data = loader.fetch()
            health_status['data_loading'] = len(data) > 0
        except Exception:
            pass

        # Check model loading
        try:
            if self.config.models.random_forest_path.exists():
                from models.model_factory import ModelFactory
                model = ModelFactory.load(self.config.models.random_forest_path)
                health_status['model_loading'] = True
        except Exception:
            pass

        # Check prediction engine
        try:
            from predictions.predictor import StockPredictor
            predictor = StockPredictor(self.config)
            health_status['prediction_engine'] = True
        except Exception:
            pass

        # Check dashboard
        try:
            import streamlit
            health_status['dashboard'] = True
        except Exception:
            pass

        # Check live data
        try:
            from live_data.live_stream import LiveDataStreamer
            health_status['live_data'] = True
        except Exception:
            pass

        return health_status


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments with comprehensive options."""
    parser = argparse.ArgumentParser(
        description='AI-powered Indian Stock Market Prediction System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --mode train                    # Train models
  python main.py --mode predict --symbol RELIANCE.NS  # Predict single stock
  python main.py --mode dashboard                # Launch dashboard
  python main.py --mode simulate --days 30       # Run simulation
  python main.py --mode health                   # System health check
  python main.py --mode realtime                 # Real-time predictions
        """
    )

    parser.add_argument(
        '--mode',
        choices=['train', 'predict', 'dashboard', 'simulate', 'health', 'realtime', 'live'],
        default='dashboard',
        help='Operation mode'
    )

    parser.add_argument(
        '--symbol',
        default='RELIANCE.NS',
        help='Stock symbol for prediction (default: RELIANCE.NS)'
    )

    parser.add_argument(
        '--symbols',
        nargs='+',
        help='List of stock symbols (overrides config defaults)'
    )

    parser.add_argument(
        '--days',
        type=int,
        default=30,
        help='Number of days for simulation (default: 30)'
    )

    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )

    return parser.parse_args()


def main() -> None:
    """Main application entry point."""
    args = parse_arguments()

    try:
        # Load configuration
        config = AppConfig.load()

        # Configure logging
        log_level = 'DEBUG' if args.verbose else config.get_log_level()
        StockLogger.configure_root_logger(log_level)

        # Initialize system
        system = StockPredictionSystem(config)
        system.initialize_components()

        # Execute requested mode
        if args.mode == 'train':
            success = system.run_training_pipeline(symbols=args.symbols)
            sys.exit(0 if success else 1)

        elif args.mode == 'predict':
            result = system.run_prediction(args.symbol)
            if result and result['status'] == 'success':
                print(f"Prediction for {result['symbol']}: {result['prediction']}")
                sys.exit(0)
            else:
                print(f"Prediction failed: {result.get('error', 'Unknown error')}")
                sys.exit(1)

        elif args.mode == 'dashboard':
            system.start_dashboard()

        elif args.mode == 'simulate':
            system.run_simulation(days=args.days, symbols=args.symbols)

        elif args.mode == 'health':
            health = system.validate_system_health()
            print("System Health Check:")
            for component, status in health.items():
                status_icon = "[OK]" if status else "[FAIL]"
                print(f"  {status_icon} {component}")
            sys.exit(0 if all(health.values()) else 1)

        elif args.mode == 'realtime':
            results = system.run_realtime_predictions(symbols=args.symbols)
            for symbol, result in results.items():
                if result['status'] == 'success':
                    print(f"{symbol}: {result['direction']} ({result['confidence']:.2f})")
                else:
                    print(f"{symbol}: ERROR - {result['error']}")
            sys.exit(0)

        elif args.mode == 'live':
            system.run_live_data_stream(symbols=args.symbols)

    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()


if __name__ == '__main__':
    args = parse_args()
    main(args)


if __name__ == '__main__':
    args = parse_args()
    main(args)
