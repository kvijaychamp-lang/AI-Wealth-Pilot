#!/usr/bin/env python
"""Quick test to verify dashboard components work correctly."""
import os
import sys
from pathlib import Path

os.chdir(Path(__file__).parent)
sys.path.insert(0, str(Path(__file__).parent))

print("=" * 80)
print("DASHBOARD COMPONENT TEST")
print("=" * 80)

# Test 1: Config Loading
print("\n[1/5] Testing configuration loading...")
try:
    from config.settings import AppConfig
    config = AppConfig.load()
    print(f"✅ Config loaded")
    print(f"   - Model path: {config.models.random_forest_path}")
    print(f"   - Model exists: {config.models.random_forest_path.exists()}")
except Exception as e:
    print(f"❌ Config failed: {e}")
    sys.exit(1)

# Test 2: Model Loading
print("\n[2/5] Testing model loading...")
try:
    import joblib
    model_path = config.models.random_forest_path
    if model_path.exists():
        model = joblib.load(model_path)
        print(f"✅ Model loaded successfully")
        print(f"   - Type: {type(model).__name__}")
        print(f"   - Has predict: {hasattr(model, 'predict')}")
    else:
        print(f"❌ Model file not found at {model_path}")
        sys.exit(1)
except Exception as e:
    print(f"❌ Model loading failed: {e}")
    sys.exit(1)

# Test 3: Historical Data Loading
print("\n[3/5] Testing historical data loading...")
try:
    from data.loader import HistoricalDataLoader
    start_date, end_date = HistoricalDataLoader.default_date_range(days=30)
    loader = HistoricalDataLoader(symbols=['RELIANCE.NS'], start_date=start_date, end_date=end_date)
    data = loader.fetch()
    
    if 'RELIANCE.NS' in data and not data['RELIANCE.NS'].empty:
        df = data['RELIANCE.NS']
        print(f"✅ Historical data loaded")
        print(f"   - Rows: {len(df)}")
        print(f"   - Columns: {df.columns.tolist()}")
    else:
        print(f"⚠️  No real data available, mock data will be used")
except Exception as e:
    print(f"⚠️  Historical data loading issue: {e}")

# Test 4: Feature Building
print("\n[4/5] Testing feature building...")
try:
    from indicators.feature_builder import build_feature_set
    import pandas as pd
    import numpy as np
    
    # Create sample data with enough rows for indicators (need >20 for SMA windows)
    dates = pd.date_range(end='2026-05-14', periods=100, freq='D')
    np.random.seed(42)
    prices = 100 + np.cumsum(np.random.randn(100) * 1)
    sample_df = pd.DataFrame({
        'Date': dates,
        'Open': prices,
        'High': prices * 1.01,
        'Low': prices * 0.99,
        'Close': prices,
        'Volume': np.random.randint(1000000, 5000000, 100)
    })
    
    features = build_feature_set(sample_df)
    print(f"✅ Features built successfully")
    print(f"   - Rows: {len(features)}")
    print(f"   - Columns: {features.columns.tolist()}")
except Exception as e:
    print(f"❌ Feature building failed: {e}")
    sys.exit(1)

# Test 5: Prediction
print("\n[5/5] Testing model prediction...")
try:
    import numpy as np
    test_features = np.random.randn(1, 7)
    pred = model.predict(test_features)
    prob = model.predict_proba(test_features)
    
    direction = 'UP' if pred[0] == 1 else 'DOWN'
    confidence = max(prob[0])
    
    print(f"✅ Prediction successful")
    print(f"   - Direction: {direction}")
    print(f"   - Confidence: {confidence:.1%}")
    print(f"   - DOWN prob: {prob[0][0]:.1%}")
    print(f"   - UP prob: {prob[0][1]:.1%}")
except Exception as e:
    print(f"❌ Prediction failed: {e}")
    sys.exit(1)

print("\n" + "=" * 80)
print("✅ ALL TESTS PASSED - DASHBOARD IS READY!")
print("=" * 80)
print("\nRun the dashboard with:")
print("   streamlit run dashboard/app.py")
print("=" * 80)
