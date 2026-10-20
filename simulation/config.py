"""Configuration for paper trading simulation."""
from pathlib import Path

from simulation.simulator import SimulationConfig

# Default simulation configuration
DEFAULT_SIMULATION_CONFIG = SimulationConfig(
    symbols=['RELIANCE.NS', 'TCS.NS', 'INFY.NS', 'HDFC.NS', 'ICICIBANK.NS'],
    initial_cash=100000.0,
    max_position_size=0.1,  # 10% of portfolio per position
    trading_fee=0.001,     # 0.1% trading fee
    min_trade_size=1,
    max_trades_per_day=5,
    prediction_threshold=0.6  # 60% confidence minimum
)

# Storage paths
SIMULATION_DATA_DIR = Path('data') / 'simulation'
TRADE_HISTORY_PATH = SIMULATION_DATA_DIR / 'trades.json'
PERFORMANCE_REPORTS_PATH = SIMULATION_DATA_DIR / 'reports'

# Create directories if they don't exist
SIMULATION_DATA_DIR.mkdir(parents=True, exist_ok=True)
PERFORMANCE_REPORTS_PATH.mkdir(parents=True, exist_ok=True)