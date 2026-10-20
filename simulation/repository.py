"""Repository for storing and retrieving trade history."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List, Optional

from simulation.trade import Trade

logger = logging.getLogger(__name__)


class TradeRepository:
    """Repository for persisting and retrieving trade data."""

    def __init__(self, storage_path: Path):
        """Initialize repository with storage path."""
        self.storage_path = storage_path
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)

    def save_trade(self, trade: Trade) -> None:
        """Save a single trade to storage."""
        try:
            # Load existing trades
            trades = self._load_all_trades()

            # Add new trade
            trade_dict = self._trade_to_dict(trade)
            trades.append(trade_dict)

            # Save back to file
            self._save_all_trades(trades)
            logger.debug(f"Saved trade {trade.trade_id} to storage")

        except Exception as e:
            logger.error(f"Failed to save trade {trade.trade_id}: {e}")
            raise

    def save_trades(self, trades: List[Trade]) -> None:
        """Save multiple trades to storage."""
        try:
            # Load existing trades
            existing_trades = self._load_all_trades()

            # Add new trades
            for trade in trades:
                trade_dict = self._trade_to_dict(trade)
                existing_trades.append(trade_dict)

            # Save back to file
            self._save_all_trades(existing_trades)
            logger.debug(f"Saved {len(trades)} trades to storage")

        except Exception as e:
            logger.error(f"Failed to save trades: {e}")
            raise

    def load_trades(self, symbol: Optional[str] = None) -> List[Trade]:
        """Load trades from storage, optionally filtered by symbol."""
        try:
            trades_data = self._load_all_trades()
            trades = [self._dict_to_trade(trade_dict) for trade_dict in trades_data]

            if symbol:
                trades = [trade for trade in trades if trade.symbol == symbol]

            logger.debug(f"Loaded {len(trades)} trades from storage")
            return trades

        except Exception as e:
            logger.error(f"Failed to load trades: {e}")
            return []

    def load_all_trades(self) -> List[Trade]:
        """Load all trades from storage."""
        return self.load_trades()

    def clear_trades(self) -> None:
        """Clear all trades from storage."""
        try:
            self._save_all_trades([])
            logger.info("Cleared all trades from storage")
        except Exception as e:
            logger.error(f"Failed to clear trades: {e}")
            raise

    def _load_all_trades(self) -> List[dict]:
        """Load all trades from JSON file."""
        if not self.storage_path.exists():
            return []

        try:
            with open(self.storage_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            logger.warning(f"Could not load trades from {self.storage_path}, starting fresh")
            return []

    def _save_all_trades(self, trades: List[dict]) -> None:
        """Save all trades to JSON file."""
        with open(self.storage_path, 'w', encoding='utf-8') as f:
            json.dump(trades, f, indent=2, default=str)

    def _trade_to_dict(self, trade: Trade) -> dict:
        """Convert Trade object to dictionary for JSON storage."""
        return {
            'trade_id': trade.trade_id,
            'symbol': trade.symbol,
            'trade_type': trade.trade_type.value,
            'quantity': trade.quantity,
            'price': trade.price,
            'timestamp': trade.timestamp.isoformat(),
            'status': trade.status.value,
            'fees': trade.fees,
            'prediction_confidence': trade.prediction_confidence,
            'prediction_direction': trade.prediction_direction,
        }

    def _dict_to_trade(self, trade_dict: dict) -> Trade:
        """Convert dictionary to Trade object."""
        from datetime import datetime
        from simulation.trade import TradeType, TradeStatus

        return Trade(
            trade_id=trade_dict['trade_id'],
            symbol=trade_dict['symbol'],
            trade_type=TradeType(trade_dict['trade_type']),
            quantity=trade_dict['quantity'],
            price=trade_dict['price'],
            timestamp=datetime.fromisoformat(trade_dict['timestamp']),
            status=TradeStatus(trade_dict['status']),
            fees=trade_dict.get('fees', 0.0),
            prediction_confidence=trade_dict.get('prediction_confidence'),
            prediction_direction=trade_dict.get('prediction_direction'),
        )