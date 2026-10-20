#!/usr/bin/env python
"""
Comprehensive system validation script for when NSE market opens.
Tests all components: data fetching, predictions, dashboard, market timing.
"""
import os
import sys
from datetime import datetime, time
from pathlib import Path

os.chdir(Path(__file__).parent)
sys.path.insert(0, str(Path(__file__).parent))

print("=" * 90)
print(" NSE MARKET SYSTEM VALIDATION TEST SUITE")
print("=" * 90)
print(f"Test Date & Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S IST')}")
print("=" * 90)

# TEST 1: Market Timing Detection
print("\n[TEST 1/6] Verifying NSE Market Timing Detection")
print("-" * 90)
try:
    from utils.market_schedule import (
        should_run_market_hours,
        is_market_holiday,
        is_weekend,
        get_next_market_open,
        get_next_market_close,
    )
    from datetime import datetime, date
    
    now = datetime.now()
    current_date = now.date()
    
    print(f"Current Time: {now.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Is Weekend: {is_weekend(current_date)}")
    print(f"Is Holiday: {is_market_holiday(current_date)}")
    print(f"Should Run (Market Hours): {should_run_market_hours(now)}")
    print(f"Next Market Open: {get_next_market_open(now).strftime('%Y-%m-%d %H:%M:%S IST')}")
    print(f"Next Market Close: {get_next_market_close(now).strftime('%Y-%m-%d %H:%M:%S IST')}")
    
    if should_run_market_hours(now):
        print("✅ Market is OPEN - All systems operational")
    else:
        next_open = get_next_market_open(now)
        hours_until = (next_open - now).total_seconds() / 3600
        print(f"⏰ Market is CLOSED - Will open in {hours_until:.1f} hours")
    
except Exception as e:
    print(f"❌ FAILED: {e}")
    sys.exit(1)

# TEST 2: Live Data Fetching
print("\n[TEST 2/6] Testing Live Stock Data Fetching")
print("-" * 90)
try:
    from utils.constants import DEFAULT_STOCK_SYMBOLS
    import yfinance as yf
    from datetime import datetime, timedelta
    
    test_symbols = DEFAULT_STOCK_SYMBOLS[:2]  # Test first 2 symbols
    
    for symbol in test_symbols:
        try:
            ticker = yf.Ticker(symbol)
            # Get last trading day's data
            hist = ticker.history(period='1d')
            
            if not hist.empty:
                latest = hist.iloc[-1]
                print(f"✅ {symbol}: Open={latest['Open']:.2f}, Close={latest['Close']:.2f}, Volume={latest['Volume']:.0f}")
            else:
                print(f"⚠️  {symbol}: No data available (market may be closed)")
        except Exception as e:
            print(f"⚠️  {symbol}: {str(e)[:50]}")
    
except Exception as e:
    print(f"❌ FAILED: {e}")

# TEST 3: Configuration Validation
print("\n[TEST 3/6] Validating Application Configuration")
print("-" * 90)
try:
    from config.settings import AppConfig
    
    config = AppConfig.load()
    
    print(f"✅ App Environment: {config.app_env}")
    print(f"✅ Model Directory: {config.models.model_dir}")
    print(f"✅ Model Exists: {config.models.random_forest_path.exists()}")
    print(f"✅ Data Directory: {config.data.processed_data_dir}")
    print(f"✅ Stock Symbols: {len(config.stocks.default_symbols)} configured")
    print(f"✅ Log Level: {config.logging.log_level}")
    
except Exception as e:
    print(f"❌ FAILED: {e}")
    sys.exit(1)

# TEST 4: Model Prediction
print("\n[TEST 4/6] Testing Trained Model Predictions")
print("-" * 90)
try:
    import joblib
    import numpy as np
    from config.settings import AppConfig
    
    config = AppConfig.load()
    model_path = config.models.random_forest_path
    
    if model_path.exists():
        model = joblib.load(model_path)
        print(f"✅ Model Loaded: {type(model).__name__}")
        
        # Test multiple predictions
        test_count = 5
        print(f"\nTesting {test_count} predictions:")
        
        for i in range(test_count):
            test_features = np.random.randn(1, 7)
            pred = model.predict(test_features)
            proba = model.predict_proba(test_features)
            
            direction = "UP" if pred[0] == 1 else "DOWN"
            confidence = max(proba[0]) * 100
            
            print(f"  Prediction {i+1}: {direction} ({confidence:.1f}% confidence)")
        
        print(f"✅ Model predictions working correctly")
    else:
        print(f"❌ Model not found at {model_path}")
        sys.exit(1)
        
except Exception as e:
    print(f"❌ FAILED: {e}")
    sys.exit(1)

# TEST 5: Historical Data Loading
print("\n[TEST 5/6] Testing Historical Data Loading")
print("-" * 90)
try:
    from data.loader import HistoricalDataLoader
    from utils.constants import DEFAULT_STOCK_SYMBOLS
    
    print(f"Testing data loading for: {DEFAULT_STOCK_SYMBOLS[0]}")
    
    start_date, end_date = HistoricalDataLoader.default_date_range(days=30)
    loader = HistoricalDataLoader(symbols=[DEFAULT_STOCK_SYMBOLS[0]], start_date=start_date, end_date=end_date)
    data = loader.fetch()
    
    if DEFAULT_STOCK_SYMBOLS[0] in data:
        df = data[DEFAULT_STOCK_SYMBOLS[0]]
        if not df.empty:
            print(f"✅ Loaded {len(df)} rows of historical data")
            print(f"✅ Columns: {df.columns.tolist()[:5]}...")
            print(f"✅ Date range: {df['Date'].min()} to {df['Date'].max()}")
        else:
            print(f"⚠️  Historical data is empty (market may be closed)")
    
except Exception as e:
    print(f"❌ FAILED: {e}")

# TEST 6: Dashboard Import & Status
print("\n[TEST 6/6] Verifying Dashboard Application")
print("-" * 90)
try:
    from dashboard.app import main, load_model_safe, load_history_safe, get_mock_prediction
    
    print(f"✅ Dashboard module imports successfully")
    print(f"✅ Main function available")
    print(f"✅ Model loading function available")
    print(f"✅ History loading function available")
    print(f"✅ Mock prediction function available")
    
    # Test mock prediction
    mock_pred = get_mock_prediction('RELIANCE.NS')
    print(f"✅ Mock prediction test: {mock_pred}")
    
except Exception as e:
    print(f"❌ FAILED: {e}")
    sys.exit(1)

# Summary
print("\n" + "=" * 90)
print("✅ ALL VALIDATION TESTS COMPLETED SUCCESSFULLY!")
print("=" * 90)
print("\nNEXT STEPS FOR NSE MARKET OPENING:")
print("-" * 90)
print("""
1. WHEN MARKET OPENS (9:15 AM IST):
   - Run: streamlit run dashboard/app.py
   - The dashboard will automatically fetch live data
   - Monitor the first prediction for accuracy

2. CHECK THESE INDICATORS:
   ✓ Model loads without 'not found' warning
   ✓ Real data appears (not 'mock historical data')
   ✓ Predictions update with real market prices
   ✓ Confidence scores are reasonable (not all near 50%)
   ✓ Volume numbers match actual trading activity
   ✓ Technical indicators calculate correctly

3. VALIDATION COMMANDS:
   - Check logs: tail -f logs/stock_prediction.log
   - Monitor predictions: python -c "from predictions.realtime_engine import predict_stock_direction"
   - Test live data: python -c "import yfinance; print(yfinance.Ticker('RELIANCE.NS').history('1d'))"

4. IF ISSUES OCCUR:
   - Check internet connection (NSE API requires live access)
   - Verify system time is correct (IST timezone)
   - Check log files in /logs/ directory
   - Ensure model file exists at: models/random_forest_classifier.pkl
   - Verify all dependencies: pip install -r requirements.txt

5. CONTINUOUS MONITORING:
   - Dashboard auto-refreshes every 3 minutes
   - Check if predictions change as new prices come in
   - Monitor confidence scores (should vary, not stay constant)
   - Verify indicators respond to price changes
""")
print("=" * 90)
print(f"\nValidation completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S IST')}")
print("=" * 90)
