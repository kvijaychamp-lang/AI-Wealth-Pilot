"""Main paper trading simulation engine."""
from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

from simulation.evaluator import PerformanceEvaluator
from simulation.portfolio import Portfolio
from simulation.repository import TradeRepository
from simulation.trade import Trade, TradeType
from utils.logger import get_simulation_logger, log_trade, log_performance_metric

logger = get_simulation_logger()


@dataclass
class SimulationConfig:
    """Configuration for the trading simulation."""
    symbols: List[str]
    initial_cash: float = 100000.0
    max_position_size: float = 0.1  # Max 10% of portfolio per position
    trading_fee: float = 0.001  # 0.1% trading fee
    min_trade_size: int = 1
    max_trades_per_day: int = 5
    prediction_threshold: float = 0.6  # Minimum confidence for trade


class PaperTradingSimulator:
    """Main engine for paper trading simulation using AI predictions."""

    def __init__(self, config: SimulationConfig, storage_path: Optional[Path] = None):
        """Initialize the simulator."""
        self.config = config
        self.portfolio = Portfolio(initial_cash=config.initial_cash)
        self.repository = TradeRepository(storage_path or Path('data') / 'simulation' / 'trades.json')
        self.evaluator = PerformanceEvaluator()

        # Load existing trades if any
        existing_trades = self.repository.load_all_trades()
        for trade in existing_trades:
            self.portfolio.execute_trade(trade)

        logger.info(f"Initialized simulator with {len(existing_trades)} existing trades")

    def run_simulation(self, prediction_engine, duration_days: int = 30) -> Dict:
        """Run the trading simulation for specified duration."""
        logger.info(f"Starting simulation for {duration_days} days")

        start_date = datetime.now()
        end_date = start_date + timedelta(days=duration_days)

        current_date = start_date
        daily_trades = 0
        last_trade_date = None

        while current_date <= end_date:
            # Reset daily trade counter if it's a new day
            if last_trade_date != current_date.date():
                daily_trades = 0
                last_trade_date = current_date.date()

            # Skip weekends and holidays (simplified)
            if current_date.weekday() >= 5:  # Saturday = 5, Sunday = 6
                current_date += timedelta(days=1)
                continue

            # Limit trades per day
            if daily_trades >= self.config.max_trades_per_day:
                current_date += timedelta(hours=1)
                continue

            # Process each symbol
            for symbol in self.config.symbols:
                try:
                    trade = self._generate_trade(symbol, prediction_engine, current_date)
                    if trade:
                        success = self.portfolio.execute_trade(trade)
                        if success:
                            self.repository.save_trade(trade)
                            daily_trades += 1
                            logger.info(f"Executed trade: {trade.trade_id}")

                            # Small delay between trades
                            time.sleep(0.1)

                except Exception as e:
                    logger.error(f"Error processing {symbol}: {e}")

            # Move to next hour (simulate intraday trading)
            current_date += timedelta(hours=1)

        # Generate final report
        report = self.evaluator.generate_performance_report(
            self.portfolio,
            self.portfolio.get_trade_history(),
            self.config.initial_cash
        )

        logger.info(f"Simulation completed. Final portfolio value: {self.portfolio.get_portfolio_value():.2f}")
        return report

    def _generate_trade(self, symbol: str, prediction_engine, current_time: datetime) -> Optional[Trade]:
        """Generate a trade based on AI prediction."""
        try:
            # Get prediction from the engine
            features = prediction_engine.build_realtime_features(symbol, [])
            direction, confidence, probabilities = prediction_engine.predict_stock_direction(
                prediction_engine.model, features
            )

            # Check if prediction meets threshold
            if confidence < self.config.prediction_threshold:
                return None

            # Determine trade type based on prediction
            trade_type = TradeType.BUY if direction == "UP" else TradeType.SELL

            # Check if we have position for selling
            if trade_type == TradeType.SELL:
                position = self.portfolio.get_position(symbol)
                if not position or position.quantity < self.config.min_trade_size:
                    return None

            # Calculate position size
            portfolio_value = self.portfolio.get_portfolio_value()
            max_position_value = portfolio_value * self.config.max_position_size
            current_price = features.get('Close', 100.0)

            if trade_type == TradeType.BUY:
                # For buys, calculate quantity based on available cash
                available_cash = self.portfolio.cash * 0.9  # Leave 10% buffer
                max_quantity = int(available_cash / (current_price * (1 + self.config.trading_fee)))
                quantity = min(max_quantity, int(max_position_value / current_price))
            else:
                # For sells, sell portion of position
                position = self.portfolio.get_position(symbol)
                quantity = min(position.quantity, int(max_position_value / current_price))

            if quantity < self.config.min_trade_size:
                return None

            # Calculate fees
            fees = quantity * current_price * self.config.trading_fee

            # Create trade
            trade = Trade(
                symbol=symbol,
                trade_type=trade_type,
                quantity=quantity,
                price=current_price,
                timestamp=current_time,
                fees=fees,
                prediction_confidence=confidence,
                prediction_direction=direction
            )

            return trade

        except Exception as e:
            logger.error(f"Error generating trade for {symbol}: {e}")
            return None

    def get_portfolio_status(self) -> Dict:
        """Get current portfolio status."""
        return {
            'cash': self.portfolio.cash,
            'positions': {
                symbol: {
                    'quantity': position.quantity,
                    'average_price': position.average_price,
                    'current_price': position.current_price,
                    'market_value': position.market_value,
                    'unrealized_pnl': position.unrealized_pnl
                }
                for symbol, position in self.portfolio.positions.items()
            },
            'total_value': self.portfolio.get_portfolio_value(),
            'realized_pnl': self.portfolio.get_realized_pnl(),
            'unrealized_pnl': self.portfolio.get_unrealized_pnl(),
            'total_pnl': self.portfolio.get_total_pnl()
        }

    def get_trade_history(self, symbol: Optional[str] = None) -> List[Trade]:
        """Get trade history."""
        return self.portfolio.get_trade_history(symbol)

    def reset_simulation(self) -> None:
        """Reset the simulation to initial state."""
        self.portfolio = Portfolio(initial_cash=self.config.initial_cash)
        self.repository.clear_trades()
        logger.info("Simulation reset to initial state")

    def load_historical_trades(self) -> None:
        """Load historical trades from repository."""
        trades = self.repository.load_all_trades()
        for trade in trades:
            self.portfolio.execute_trade(trade)
        logger.info(f"Loaded {len(trades)} historical trades")