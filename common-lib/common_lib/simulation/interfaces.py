"""
Simulation Interfaces Module

This module provides interfaces for simulation components used across services,
helping to break circular dependencies between services.
"""
from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, Any, List, Optional, Union
from datetime import datetime


class MarketRegimeType(str, Enum):
    """Market regime types for simulation and risk management."""
    TRENDING_BULLISH = "trending_bullish"
    TRENDING_BEARISH = "trending_bearish"
    RANGING_NARROW = "ranging_narrow"
    RANGING_WIDE = "ranging_wide"
    VOLATILE = "volatile"
    CHOPPY = "choppy"
    BREAKOUT = "breakout"
    REVERSAL = "reversal"
    CRISIS = "crisis"
    NORMAL = "normal"


class NewsImpactLevel(str, Enum):
    """Impact levels for news events."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class SentimentLevel(str, Enum):
    """Sentiment levels for market sentiment."""
    VERY_BEARISH = "very_bearish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"
    BULLISH = "bullish"
    VERY_BULLISH = "very_bullish"


class IMarketRegimeSimulator(ABC):
    """Interface for market regime simulation components."""

    @abstractmethod
    def get_current_regime(self, symbol: str) -> MarketRegimeType:
        """
        Get the current market regime for a symbol.

        Args:
            symbol: The trading symbol to check

        Returns:
            The current market regime type
        """
        pass

    @abstractmethod
    def get_all_regimes(self) -> Dict[str, MarketRegimeType]:
        """
        Get the current market regimes for all symbols.

        Returns:
            Dictionary mapping symbols to their current market regime
        """
        pass

    @abstractmethod
    def get_regime_probabilities(self, symbol: str) -> Dict[MarketRegimeType, float]:
        """
        Get the probability distribution across different regime types.

        Args:
            symbol: The trading symbol to check

        Returns:
            Dictionary mapping regime types to their probabilities
        """
        pass

    @abstractmethod
    def get_regime_history(self, symbol: str, lookback_periods: int = 10) -> List[Dict[str, Any]]:
        """
        Get the history of regime changes.

        Args:
            symbol: The trading symbol to check
            lookback_periods: Number of historical periods to return

        Returns:
            List of historical regime data
        """
        pass


class IBrokerSimulator(ABC):
    """Interface for broker simulation components."""

    @abstractmethod
    def get_current_price(self, symbol: str) -> Dict[str, float]:
        """
        Get the current price for a symbol.

        Args:
            symbol: The trading symbol

        Returns:
            Dictionary with bid and ask prices
        """
        pass

    @abstractmethod
    def get_account_summary(self) -> Dict[str, Any]:
        """
        Get the current account summary.

        Returns:
            Dictionary with account information
        """
        pass

    @abstractmethod
    async def submit_order(self, symbol: str, order_type: str, direction: str,
                          size: float, price: Optional[float] = None) -> Dict[str, Any]:
        """
        Submit a new order.

        Args:
            symbol: The trading symbol
            order_type: Type of order (market, limit, etc.)
            direction: Order direction (buy or sell)
            size: Order size
            price: Order price (for limit orders)

        Returns:
            Order result information
        """
        pass

    @abstractmethod
    def get_positions(self) -> Dict[str, Any]:
        """
        Get all current positions.

        Returns:
            Dictionary of current positions
        """
        pass


class IRiskManager(ABC):
    """Interface for risk management components."""

    @abstractmethod
    async def check_order(self, symbol: str, direction: str, size: float,
                         current_price: Dict[str, float]) -> Dict[str, Any]:
        """
        Check if an order meets risk criteria.

        Args:
            symbol: The trading symbol
            direction: Order direction (buy or sell)
            size: Order size
            current_price: Current market price

        Returns:
            Risk check result with approval status
        """
        pass

    @abstractmethod
    def add_position(self, symbol: str, size: float, price: float,
                    direction: str, leverage: float = 1.0) -> None:
        """
        Add a new position for risk tracking.

        Args:
            symbol: The trading symbol
            size: Position size
            price: Entry price
            direction: Position direction (long or short)
            leverage: Position leverage
        """
        pass

    @abstractmethod
    def check_risk_limits(self) -> List[Dict[str, Any]]:
        """
        Check if any risk limits are breached.

        Returns:
            List of breached risk limits
        """
        pass

    @abstractmethod
    def get_portfolio_metrics(self) -> Dict[str, Any]:
        """
        Get current portfolio risk metrics.

        Returns:
            Dictionary of risk metrics
        """
        pass


class NewsEvent:
    """Data class for news events."""

    def __init__(
        self,
        event_id: str,
        title: str,
        timestamp: datetime,
        impact_level: NewsImpactLevel,
        affected_currencies: List[str],
        expected_value: Optional[float] = None,
        actual_value: Optional[float] = None,
        previous_value: Optional[float] = None,
        description: Optional[str] = None
    ):
    """
      init  .
    
    Args:
        event_id: Description of event_id
        title: Description of title
        timestamp: Description of timestamp
        impact_level: Description of impact_level
        affected_currencies: Description of affected_currencies
        expected_value: Description of expected_value
        actual_value: Description of actual_value
        previous_value: Description of previous_value
        description: Description of description
    
    """

        self.event_id = event_id
        self.title = title
        self.timestamp = timestamp
        self.impact_level = impact_level
        self.affected_currencies = affected_currencies
        self.expected_value = expected_value
        self.actual_value = actual_value
        self.previous_value = previous_value
        self.description = description

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'event_id': self.event_id,
            'title': self.title,
            'timestamp': self.timestamp.isoformat(),
            'impact_level': self.impact_level.value,
            'affected_currencies': self.affected_currencies,
            'expected_value': self.expected_value,
            'actual_value': self.actual_value,
            'previous_value': self.previous_value,
            'description': self.description
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'NewsEvent':
        """Create from dictionary representation."""
        return cls(
            event_id=data['event_id'],
            title=data['title'],
            timestamp=datetime.fromisoformat(data['timestamp']) if isinstance(data['timestamp'], str) else data['timestamp'],
            impact_level=NewsImpactLevel(data['impact_level']) if isinstance(data['impact_level'], str) else data['impact_level'],
            affected_currencies=data['affected_currencies'],
            expected_value=data.get('expected_value'),
            actual_value=data.get('actual_value'),
            previous_value=data.get('previous_value'),
            description=data.get('description')
        )


class INewsSentimentSimulator(ABC):
    """Interface for news and sentiment simulation components."""

    @abstractmethod
    def get_current_news(self, currency: Optional[str] = None) -> List[NewsEvent]:
        """
        Get current news events.

        Args:
            currency: Optional filter for specific currency

        Returns:
            List of current news events
        """
        pass

    @abstractmethod
    def get_upcoming_news(self, hours_ahead: int = 24,
                         currency: Optional[str] = None) -> List[NewsEvent]:
        """
        Get upcoming news events.

        Args:
            hours_ahead: Hours to look ahead
            currency: Optional filter for specific currency

        Returns:
            List of upcoming news events
        """
        pass

    @abstractmethod
    def get_current_sentiment(self, currency: str) -> SentimentLevel:
        """
        Get current market sentiment for a currency.

        Args:
            currency: Currency to check

        Returns:
            Current sentiment level
        """
        pass

    @abstractmethod
    def get_sentiment_history(self, currency: str,
                             lookback_periods: int = 10) -> List[Dict[str, Any]]:
        """
        Get sentiment history for a currency.

        Args:
            currency: Currency to check
            lookback_periods: Number of historical periods to return

        Returns:
            List of historical sentiment data
        """
        pass

    @abstractmethod
    def get_sentiment_impact(self, symbol: str) -> Dict[str, float]:
        """
        Get estimated sentiment impact on a symbol.

        Args:
            symbol: Trading symbol to check

        Returns:
            Dictionary with impact metrics
        """
        pass
