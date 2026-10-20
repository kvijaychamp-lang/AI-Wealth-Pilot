"""Trade models and data structures for paper trading simulation."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional


class TradeType(Enum):
    """Types of trades in the simulation."""
    BUY = "BUY"
    SELL = "SELL"


class TradeStatus(Enum):
    """Status of trades in the simulation."""
    PENDING = "PENDING"
    EXECUTED = "EXECUTED"
    CANCELLED = "CANCELLED"


@dataclass
class Trade:
    """Represents a single trade in the paper trading simulation."""
    symbol: str
    trade_type: TradeType
    quantity: int
    price: float
    timestamp: datetime
    status: TradeStatus = TradeStatus.PENDING
    trade_id: Optional[str] = None
    fees: float = 0.0
    prediction_confidence: Optional[float] = None
    prediction_direction: Optional[str] = None

    def __post_init__(self):
        """Generate trade ID if not provided."""
        if self.trade_id is None:
            self.trade_id = f"{self.symbol}_{self.timestamp.strftime('%Y%m%d_%H%M%S')}_{self.trade_type.value}"

    @property
    def total_cost(self) -> float:
        """Calculate total cost including fees."""
        return (self.quantity * self.price) + self.fees

    @property
    def is_buy(self) -> bool:
        """Check if this is a buy trade."""
        return self.trade_type == TradeType.BUY

    @property
    def is_sell(self) -> bool:
        """Check if this is a sell trade."""
        return self.trade_type == TradeType.SELL


@dataclass
class Position:
    """Represents a position in a stock."""
    symbol: str
    quantity: int
    average_price: float
    current_price: Optional[float] = None

    @property
    def market_value(self) -> float:
        """Calculate current market value of the position."""
        if self.current_price is None:
            return self.quantity * self.average_price
        return self.quantity * self.current_price

    @property
    def unrealized_pnl(self) -> float:
        """Calculate unrealized profit/loss."""
        if self.current_price is None:
            return 0.0
        return (self.current_price - self.average_price) * self.quantity


@dataclass
class PortfolioSnapshot:
    """Snapshot of portfolio state at a point in time."""
    timestamp: datetime
    cash: float
    positions: dict[str, Position]
    total_value: float

    @property
    def total_positions_value(self) -> float:
        """Calculate total value of all positions."""
        return sum(position.market_value for position in self.positions.values())