"""Portfolio management for paper trading simulation."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from simulation.trade import Position, PortfolioSnapshot, Trade, TradeStatus, TradeType
from utils.logger import get_simulation_logger, log_trade

logger = get_simulation_logger()


@dataclass
class Portfolio:
    """Manages virtual portfolio for paper trading simulation."""
    initial_cash: float = 100000.0
    cash: float = field(init=False)
    positions: Dict[str, Position] = field(default_factory=dict)
    trade_history: List[Trade] = field(default_factory=list)

    def __post_init__(self):
        """Initialize cash with initial amount."""
        self.cash = self.initial_cash

    def execute_trade(self, trade: Trade) -> bool:
        """Execute a trade and update portfolio accordingly."""
        if trade.status != TradeStatus.PENDING:
            logger.warning(f"Trade {trade.trade_id} is not pending, cannot execute")
            return False

        if trade.is_buy:
            return self._execute_buy(trade)
        else:
            return self._execute_sell(trade)

    def _execute_buy(self, trade: Trade) -> bool:
        """Execute a buy trade."""
        total_cost = trade.total_cost

        if self.cash < total_cost:
            logger.warning(f"Insufficient cash for trade {trade.trade_id}. Required: {total_cost}, Available: {self.cash}")
            return False

        # Update cash
        self.cash -= total_cost

        # Update or create position
        if trade.symbol in self.positions:
            existing_position = self.positions[trade.symbol]
            total_quantity = existing_position.quantity + trade.quantity
            total_cost_basis = (existing_position.quantity * existing_position.average_price) + (trade.quantity * trade.price)
            new_average_price = total_cost_basis / total_quantity

            self.positions[trade.symbol] = Position(
                symbol=trade.symbol,
                quantity=total_quantity,
                average_price=new_average_price
            )
        else:
            self.positions[trade.symbol] = Position(
                symbol=trade.symbol,
                quantity=trade.quantity,
                average_price=trade.price
            )

        trade.status = TradeStatus.EXECUTED
        self.trade_history.append(trade)
        logger.info(f"Executed BUY trade: {trade.trade_id} - {trade.quantity} shares of {trade.symbol} at {trade.price}")
        return True

    def _execute_sell(self, trade: Trade) -> bool:
        """Execute a sell trade."""
        if trade.symbol not in self.positions:
            logger.warning(f"No position in {trade.symbol} to sell")
            return False

        position = self.positions[trade.symbol]

        if position.quantity < trade.quantity:
            logger.warning(f"Insufficient shares for trade {trade.trade_id}. Required: {trade.quantity}, Available: {position.quantity}")
            return False

        # Calculate proceeds
        proceeds = (trade.quantity * trade.price) - trade.fees
        self.cash += proceeds

        # Update position
        remaining_quantity = position.quantity - trade.quantity
        if remaining_quantity == 0:
            del self.positions[trade.symbol]
        else:
            # Average price remains the same for remaining shares
            self.positions[trade.symbol] = Position(
                symbol=trade.symbol,
                quantity=remaining_quantity,
                average_price=position.average_price
            )

        trade.status = TradeStatus.EXECUTED
        self.trade_history.append(trade)
        logger.info(f"Executed SELL trade: {trade.trade_id} - {trade.quantity} shares of {trade.symbol} at {trade.price}")
        return True

    def update_prices(self, price_updates: Dict[str, float]):
        """Update current prices for positions."""
        for symbol, price in price_updates.items():
            if symbol in self.positions:
                self.positions[symbol].current_price = price

    def get_portfolio_value(self) -> float:
        """Calculate total portfolio value (cash + positions)."""
        positions_value = sum(position.market_value for position in self.positions.values())
        return self.cash + positions_value

    def get_realized_pnl(self) -> float:
        """Calculate realized profit/loss from closed trades."""
        realized_pnl = 0.0

        # Group trades by symbol for easier calculation
        symbol_trades = {}
        for trade in self.trade_history:
            if trade.status == TradeStatus.EXECUTED:
                if trade.symbol not in symbol_trades:
                    symbol_trades[trade.symbol] = []
                symbol_trades[trade.symbol].append(trade)

        # Calculate P&L for each symbol
        for symbol, trades in symbol_trades.items():
            buy_cost = 0.0
            sell_proceeds = 0.0

            for trade in sorted(trades, key=lambda x: x.timestamp):
                if trade.is_buy:
                    buy_cost += trade.total_cost
                else:
                    sell_proceeds += (trade.quantity * trade.price) - trade.fees

            # For simplicity, assume all buys are matched with sells
            realized_pnl += sell_proceeds - buy_cost

        return realized_pnl

    def get_unrealized_pnl(self) -> float:
        """Calculate unrealized profit/loss from open positions."""
        return sum(position.unrealized_pnl for position in self.positions.values())

    def get_total_pnl(self) -> float:
        """Calculate total profit/loss (realized + unrealized)."""
        return self.get_realized_pnl() + self.get_unrealized_pnl()

    def get_portfolio_snapshot(self) -> PortfolioSnapshot:
        """Get current portfolio snapshot."""
        from datetime import datetime
        return PortfolioSnapshot(
            timestamp=datetime.now(),
            cash=self.cash,
            positions=self.positions.copy(),
            total_value=self.get_portfolio_value()
        )

    def get_trade_history(self, symbol: Optional[str] = None) -> List[Trade]:
        """Get trade history, optionally filtered by symbol."""
        if symbol is None:
            return self.trade_history.copy()
        return [trade for trade in self.trade_history if trade.symbol == symbol]

    def get_position(self, symbol: str) -> Optional[Position]:
        """Get position for a specific symbol."""
        return self.positions.get(symbol)