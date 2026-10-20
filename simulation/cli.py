"""Command-line interface for paper trading simulation."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Add the parent directory to the path so we can import simulation modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from simulation.config import DEFAULT_SIMULATION_CONFIG, PERFORMANCE_REPORTS_PATH
from simulation.simulator import PaperTradingSimulator
from utils.logger import StockLogger, get_simulation_logger, log_performance_metric

logger = get_simulation_logger()

# Import prediction engine (this would need to be adjusted based on actual implementation)
try:
    from predictions.realtime_engine import RealTimePredictionEngine
except ImportError:
    # Mock prediction engine for demonstration
    class MockPredictionEngine:
        def build_realtime_features(self, symbol, features):
            import random
            return {
                'Close': 100.0 + random.uniform(-10, 10),
                'symbol': symbol
            }

        def predict_stock_direction(self, model, features):
            import random
            direction = "UP" if random.random() > 0.5 else "DOWN"
            confidence = random.uniform(0.5, 0.9)
            probabilities = {"UP": confidence if direction == "UP" else 1-confidence,
                           "DOWN": 1-confidence if direction == "UP" else confidence}
            return direction, confidence, probabilities

        model = None  # Mock model

    RealTimePredictionEngine = MockPredictionEngine


def setup_logging(level: str = 'INFO'):
    """Setup logging configuration."""
    StockLogger.configure_root_logger(level)


def run_simulation(args):
    """Run the paper trading simulation."""
    setup_logging(args.log_level)

    # Initialize prediction engine
    prediction_engine = RealTimePredictionEngine()

    # Create simulator
    simulator = PaperTradingSimulator(DEFAULT_SIMULATION_CONFIG)

    # Run simulation
    print(f"Starting simulation for {args.days} days...")
    report = simulator.run_simulation(prediction_engine, duration_days=args.days)

    # Save report
    report_path = PERFORMANCE_REPORTS_PATH / f"simulation_report_{args.days}days.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)

    print(f"Simulation completed. Report saved to: {report_path}")

    # Print summary
    portfolio_status = simulator.get_portfolio_status()
    print("\nFinal Portfolio Status:")
    print(f"  Cash: ₹{portfolio_status['cash']:.2f}")
    print(f"  Total Value: ₹{portfolio_status['total_value']:.2f}")
    print(f"  Total P&L: ₹{portfolio_status['total_pnl']:.2f}")

    metrics = report['portfolio_metrics']
    print("\nPerformance Metrics:")
    print(f"  Total Return: {metrics['total_return_pct']:.2f}%")
    print(f"  Win Rate: {metrics['win_rate']:.2f}%")
    print(f"  Profit Factor: {metrics['profit_factor']:.2f}")
    print(f"  Total Trades: {metrics['total_trades']}")

    pred_metrics = report['prediction_metrics']
    print("\nPrediction Metrics:")
    print(f"  Accuracy: {pred_metrics['accuracy']:.2f}%")
    print(f"  Total Predictions: {pred_metrics['total_predictions']}")


def show_status(args):
    """Show current portfolio status."""
    setup_logging(args.log_level)

    simulator = PaperTradingSimulator(DEFAULT_SIMULATION_CONFIG)
    status = simulator.get_portfolio_status()

    print("Current Portfolio Status:")
    print(f"Cash: ₹{status['cash']:.2f}")
    print(f"Total Value: ₹{status['total_value']:.2f}")
    print(f"Total P&L: ₹{status['total_pnl']:.2f}")

    if status['positions']:
        print("\nPositions:")
        for symbol, pos in status['positions'].items():
            print(f"  {symbol}: {pos['quantity']} shares @ ₹{pos['average_price']:.2f} "
                  f"(Current: ₹{pos['current_price']:.2f}, P&L: ₹{pos['unrealized_pnl']:.2f})")
    else:
        print("\nNo open positions.")


def reset_simulation(args):
    """Reset the simulation to initial state."""
    setup_logging(args.log_level)

    simulator = PaperTradingSimulator(DEFAULT_SIMULATION_CONFIG)
    simulator.reset_simulation()
    print("Simulation reset to initial state.")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description='Paper Trading Simulation for AI Stock Predictions')
    parser.add_argument('--log-level', default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Set logging level')

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Run simulation command
    run_parser = subparsers.add_parser('run', help='Run paper trading simulation')
    run_parser.add_argument('--days', type=int, default=30, help='Number of days to simulate')
    run_parser.set_defaults(func=run_simulation)

    # Status command
    status_parser = subparsers.add_parser('status', help='Show current portfolio status')
    status_parser.set_defaults(func=show_status)

    # Reset command
    reset_parser = subparsers.add_parser('reset', help='Reset simulation to initial state')
    reset_parser.set_defaults(func=reset_simulation)

    args = parser.parse_args()

    if args.command:
        args.func(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()