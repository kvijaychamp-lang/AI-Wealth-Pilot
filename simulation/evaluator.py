"""Performance evaluation for paper trading simulation."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

from simulation.trade import Trade, TradeStatus, TradeType

logger = logging.getLogger(__name__)


@dataclass
class PredictionMetrics:
    """Metrics for prediction accuracy evaluation."""
    total_predictions: int = 0
    correct_predictions: int = 0
    accuracy: float = 0.0
    profitable_trades: int = 0
    total_trades: int = 0
    win_rate: float = 0.0

    def update_accuracy(self, prediction_direction: str, actual_outcome: str) -> None:
        """Update prediction accuracy metrics."""
        self.total_predictions += 1
        if prediction_direction.upper() == actual_outcome.upper():
            self.correct_predictions += 1
        self.accuracy = self.correct_predictions / self.total_predictions if self.total_predictions > 0 else 0.0

    def update_win_rate(self, trade: Trade, pnl: float) -> None:
        """Update win rate metrics."""
        self.total_trades += 1
        if pnl > 0:
            self.profitable_trades += 1
        self.win_rate = self.profitable_trades / self.total_trades if self.total_trades > 0 else 0.0


@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics for the trading simulation."""
    total_return: float = 0.0
    annualized_return: float = 0.0
    volatility: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0
    prediction_accuracy: float = 0.0


class PerformanceEvaluator:
    """Evaluates trading performance and prediction accuracy."""

    def __init__(self):
        """Initialize the evaluator."""
        self.prediction_metrics = PredictionMetrics()

    def evaluate_portfolio_performance(self, portfolio, trades: List[Trade], initial_value: float) -> PerformanceMetrics:
        """Evaluate comprehensive portfolio performance."""
        if not trades:
            return PerformanceMetrics()

        # Calculate basic metrics
        total_return = portfolio.get_total_pnl()
        total_value = portfolio.get_portfolio_value()

        # Calculate trade-based metrics
        winning_trades = []
        losing_trades = []

        for trade in trades:
            if trade.status == TradeStatus.EXECUTED and trade.is_sell:
                # Calculate P&L for this trade (simplified)
                pnl = (trade.price * trade.quantity) - trade.fees
                if pnl > 0:
                    winning_trades.append(pnl)
                else:
                    losing_trades.append(pnl)

        total_trades = len([t for t in trades if t.status == TradeStatus.EXECUTED])
        winning_count = len(winning_trades)
        losing_count = len(losing_trades)

        # Calculate averages
        avg_win = sum(winning_trades) / len(winning_trades) if winning_trades else 0.0
        avg_loss = abs(sum(losing_trades) / len(losing_trades)) if losing_trades else 0.0

        # Calculate profit factor
        total_wins = sum(winning_trades)
        total_losses = abs(sum(losing_trades))
        profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')

        # Calculate win rate
        win_rate = winning_count / total_trades if total_trades > 0 else 0.0

        # Calculate prediction accuracy
        prediction_accuracy = self.prediction_metrics.accuracy

        return PerformanceMetrics(
            total_return=total_return,
            total_trades=total_trades,
            winning_trades=winning_count,
            losing_trades=losing_count,
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            profit_factor=profit_factor,
            prediction_accuracy=prediction_accuracy,
        )

    def evaluate_prediction_accuracy(self, trades: List[Trade]) -> PredictionMetrics:
        """Evaluate prediction accuracy from trade history."""
        # Reset metrics
        self.prediction_metrics = PredictionMetrics()

        # Group trades by symbol and analyze prediction vs actual outcome
        symbol_trades = {}
        for trade in trades:
            if trade.status == TradeStatus.EXECUTED:
                if trade.symbol not in symbol_trades:
                    symbol_trades[trade.symbol] = []
                symbol_trades[trade.symbol].append(trade)

        # For each symbol, analyze prediction accuracy
        for symbol, symbol_trades_list in symbol_trades.items():
            # Sort trades by timestamp
            sorted_trades = sorted(symbol_trades_list, key=lambda x: x.timestamp)

            for i, trade in enumerate(sorted_trades):
                if trade.prediction_direction and i < len(sorted_trades) - 1:
                    # Determine actual outcome based on next trade or price movement
                    next_trade = sorted_trades[i + 1]
                    actual_direction = "UP" if next_trade.price > trade.price else "DOWN"

                    self.prediction_metrics.update_accuracy(trade.prediction_direction, actual_direction)

                    # Calculate P&L for win rate
                    pnl = (next_trade.price - trade.price) * trade.quantity if trade.is_buy else (trade.price - next_trade.price) * trade.quantity
                    self.prediction_metrics.update_win_rate(trade, pnl)

        return self.prediction_metrics

    def calculate_drawdown(self, portfolio_values: List[float]) -> float:
        """Calculate maximum drawdown from portfolio value history."""
        if not portfolio_values:
            return 0.0

        peak = portfolio_values[0]
        max_drawdown = 0.0

        for value in portfolio_values:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak
            max_drawdown = max(max_drawdown, drawdown)

        return max_drawdown

    def calculate_sharpe_ratio(self, returns: List[float], risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio from return series."""
        if not returns:
            return 0.0

        avg_return = sum(returns) / len(returns)
        if len(returns) < 2:
            return 0.0

        volatility = (sum((r - avg_return) ** 2 for r in returns) / len(returns)) ** 0.5

        if volatility == 0:
            return float('inf') if avg_return > 0 else 0.0

        return (avg_return - risk_free_rate) / volatility

    def generate_performance_report(self, portfolio, trades: List[Trade], initial_value: float) -> Dict:
        """Generate a comprehensive performance report."""
        metrics = self.evaluate_portfolio_performance(portfolio, trades, initial_value)
        prediction_metrics = self.evaluate_prediction_accuracy(trades)

        return {
            'portfolio_metrics': {
                'total_return': metrics.total_return,
                'total_return_pct': (metrics.total_return / initial_value) * 100 if initial_value > 0 else 0,
                'win_rate': metrics.win_rate * 100,
                'profit_factor': metrics.profit_factor,
                'total_trades': metrics.total_trades,
                'avg_win': metrics.avg_win,
                'avg_loss': metrics.avg_loss,
            },
            'prediction_metrics': {
                'accuracy': prediction_metrics.accuracy * 100,
                'total_predictions': prediction_metrics.total_predictions,
                'correct_predictions': prediction_metrics.correct_predictions,
            },
            'summary': {
                'status': 'PROFITABLE' if metrics.total_return > 0 else 'LOSING',
                'recommendation': 'CONTINUE' if metrics.win_rate > 0.5 and prediction_metrics.accuracy > 0.5 else 'REVIEW_STRATEGY'
            }
        }