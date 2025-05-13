"""
Simulation Adapters Module

This module provides adapter implementations for simulation interfaces,
helping to break circular dependencies between services.
"""
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
import logging

from common_lib.simulation.interfaces import (
    IMarketRegimeSimulator,
    IBrokerSimulator,
    INewsSentimentSimulator,
    MarketRegimeType,
    NewsEvent,
    NewsImpactLevel,
    SentimentLevel
)
from core_foundations.utils.logger import get_logger

logger = get_logger(__name__)


class MarketRegimeSimulatorAdapter(IMarketRegimeSimulator):
    """
    Adapter for market regime simulator that implements the common interface.

    This adapter can either wrap an actual simulator instance or provide
    standalone functionality to avoid circular dependencies.
    """

    def __init__(self, simulator_instance=None):
        """
        Initialize the adapter.

        Args:
            simulator_instance: Optional actual simulator instance to wrap
        """
        self.simulator = simulator_instance
        self.default_regime = MarketRegimeType.RANGING_NARROW
        self._regime_cache = {}

    def get_current_regime(self, symbol: str) -> MarketRegimeType:
        """
        Get the current market regime for a symbol.

        Args:
            symbol: The trading symbol to check

        Returns:
            The current market regime type
        """
        if self.simulator:
            try:
                # Try to use the wrapped simulator if available
                return self.simulator.get_current_regime(symbol)
            except Exception as e:
                logger.warning(f"Error getting regime from simulator: {str(e)}")

        # Fallback to cached or default regime
        return self._regime_cache.get(symbol, self.default_regime)

    def get_all_regimes(self) -> Dict[str, MarketRegimeType]:
        """
        Get the current market regimes for all symbols.

        Returns:
            Dictionary mapping symbols to their current market regime
        """
        if self.simulator:
            try:
                return self.simulator.get_all_regimes()
            except Exception as e:
                logger.warning(f"Error getting all regimes from simulator: {str(e)}")

        return self._regime_cache

    def get_regime_probabilities(self, symbol: str) -> Dict[MarketRegimeType, float]:
        """
        Get the probability distribution across different regime types.

        Args:
            symbol: The trading symbol to check

        Returns:
            Dictionary mapping regime types to their probabilities
        """
        if self.simulator:
            try:
                return self.simulator.get_regime_probabilities(symbol)
            except Exception as e:
                logger.warning(f"Error getting regime probabilities from simulator: {str(e)}")

        # Return default probabilities with highest for current regime
        current_regime = self.get_current_regime(symbol)
        probabilities = {regime: 0.05 for regime in MarketRegimeType}
        probabilities[current_regime] = 0.65

        return probabilities

    def get_regime_history(self, symbol: str, lookback_periods: int = 10) -> List[Dict[str, Any]]:
        """
        Get the history of regime changes.

        Args:
            symbol: The trading symbol to check
            lookback_periods: Number of historical periods to return

        Returns:
            List of historical regime data
        """
        if self.simulator:
            try:
                return self.simulator.get_regime_history(symbol, lookback_periods)
            except Exception as e:
                logger.warning(f"Error getting regime history from simulator: {str(e)}")

        # Return empty history if no simulator available
        return []

    def set_regime(self, symbol: str, regime: MarketRegimeType) -> None:
        """
        Set the current regime for a symbol (for testing/simulation).

        Args:
            symbol: The trading symbol
            regime: The market regime to set
        """
        self._regime_cache[symbol] = regime
        logger.info(f"Set {symbol} regime to {regime}")


class BrokerSimulatorAdapter(IBrokerSimulator):
    """
    Adapter for broker simulator that implements the common interface.

    This adapter can either wrap an actual simulator instance or provide
    standalone functionality to avoid circular dependencies.
    """

    def __init__(self, simulator_instance=None):
        """
        Initialize the adapter.

        Args:
            simulator_instance: Optional actual simulator instance to wrap
        """
        self.simulator = simulator_instance
        self._price_cache = {}
        self._positions = {}
        self._account = {
            "balance": 100000.0,
            "equity": 100000.0,
            "margin_used": 0.0,
            "margin_level": 100.0,
            "free_margin": 100000.0
        }

    def get_current_price(self, symbol: str) -> Dict[str, float]:
        """
        Get the current price for a symbol.

        Args:
            symbol: The trading symbol

        Returns:
            Dictionary with bid and ask prices
        """
        if self.simulator:
            try:
                return self.simulator.get_current_price(symbol)
            except Exception as e:
                logger.warning(f"Error getting price from simulator: {str(e)}")

        # Return cached or default price
        if symbol not in self._price_cache:
            # Generate default prices for common forex pairs
            if symbol == "EUR/USD":
                self._price_cache[symbol] = {"bid": 1.1000, "ask": 1.1002}
            elif symbol == "GBP/USD":
                self._price_cache[symbol] = {"bid": 1.3000, "ask": 1.3003}
            elif symbol == "USD/JPY":
                self._price_cache[symbol] = {"bid": 110.00, "ask": 110.03}
            else:
                self._price_cache[symbol] = {"bid": 1.0000, "ask": 1.0001}

        return self._price_cache[symbol]

    def get_account_summary(self) -> Dict[str, Any]:
        """
        Get the current account summary.

        Returns:
            Dictionary with account information
        """
        if self.simulator:
            try:
                return self.simulator.get_account_summary()
            except Exception as e:
                logger.warning(f"Error getting account summary from simulator: {str(e)}")

        return self._account

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
        if self.simulator:
            try:
                return await self.simulator.submit_order(
                    symbol=symbol,
                    order_type=order_type,
                    direction=direction,
                    size=size,
                    price=price
                )
            except Exception as e:
                logger.warning(f"Error submitting order to simulator: {str(e)}")

        # Simple simulation logic if no simulator available
        order_id = f"order_{len(self._positions) + 1}"
        current_price = self.get_current_price(symbol)
        execution_price = current_price["ask"] if direction == "buy" else current_price["bid"]

        # Add position
        position_id = f"pos_{len(self._positions) + 1}"
        self._positions[position_id] = {
            "symbol": symbol,
            "direction": direction,
            "size": size,
            "entry_price": execution_price,
            "current_price": execution_price,
            "pnl": 0.0,
            "open_time": datetime.now().isoformat()
        }

        return {
            "success": True,
            "order_id": order_id,
            "position_id": position_id,
            "execution_price": execution_price,
            "timestamp": datetime.now().isoformat()
        }

    def get_positions(self) -> Dict[str, Any]:
        """
        Get all current positions.

        Returns:
            Dictionary of current positions
        """
        if self.simulator:
            try:
                return self.simulator.get_positions()
            except Exception as e:
                logger.warning(f"Error getting positions from simulator: {str(e)}")

        return self._positions

    def set_price(self, symbol: str, bid: float, ask: float) -> None:
        """
        Set the current price for a symbol (for testing/simulation).

        Args:
            symbol: The trading symbol
            bid: Bid price
            ask: Ask price
        """
        self._price_cache[symbol] = {"bid": bid, "ask": ask}

        # Update positions with new prices
        for pos_id, position in self._positions.items():
            if position["symbol"] == symbol:
                position["current_price"] = bid if position["direction"] == "sell" else ask

                # Update PnL
                price_diff = position["current_price"] - position["entry_price"]
                if position["direction"] == "sell":
                    price_diff = -price_diff
                position["pnl"] = price_diff * position["size"]

        logger.debug(f"Set {symbol} price to bid: {bid}, ask: {ask}")


class NewsSentimentSimulatorAdapter(INewsSentimentSimulator):
    """
    Adapter for news and sentiment simulator that implements the common interface.

    This adapter can either wrap an actual simulator instance or provide
    standalone functionality to avoid circular dependencies.
    """

    def __init__(self, simulator_instance=None):
        """
        Initialize the adapter.

        Args:
            simulator_instance: Optional actual simulator instance to wrap
        """
        self.simulator = simulator_instance
        self._sentiment_cache = {
            "EUR": SentimentLevel.NEUTRAL,
            "USD": SentimentLevel.BULLISH,
            "GBP": SentimentLevel.BEARISH,
            "JPY": SentimentLevel.NEUTRAL,
            "AUD": SentimentLevel.BULLISH
        }

        # Generate some default news events
        self._news_cache = [
            NewsEvent(
                event_id="news_1",
                title="ECB Interest Rate Decision",
                timestamp=datetime.now() + datetime.timedelta(hours=2),
                impact_level=NewsImpactLevel.HIGH,
                affected_currencies=["EUR"],
                expected_value=0.0,
                previous_value=0.0,
                description="European Central Bank interest rate decision"
            ),
            NewsEvent(
                event_id="news_2",
                title="US Non-Farm Payrolls",
                timestamp=datetime.now() + datetime.timedelta(hours=24),
                impact_level=NewsImpactLevel.HIGH,
                affected_currencies=["USD"],
                expected_value=650000,
                previous_value=559000,
                description="US employment change excluding agriculture"
            ),
            NewsEvent(
                event_id="news_3",
                title="UK GDP",
                timestamp=datetime.now() + datetime.timedelta(hours=48),
                impact_level=NewsImpactLevel.MEDIUM,
                affected_currencies=["GBP"],
                expected_value=1.5,
                previous_value=1.1,
                description="UK Gross Domestic Product YoY"
            )
        ]

    def get_current_news(self, currency: Optional[str] = None) -> List[NewsEvent]:
        """
        Get current news events.

        Args:
            currency: Optional filter for specific currency

        Returns:
            List of current news events
        """
        if self.simulator:
            try:
                return self.simulator.get_current_news(currency)
            except Exception as e:
                logger.warning(f"Error getting current news from simulator: {str(e)}")

        # Fallback implementation
        now = datetime.now()
        if currency:
            return [news for news in self._news_cache
                   if currency in news.affected_currencies
                   and news.timestamp <= now]

        return [news for news in self._news_cache if news.timestamp <= now]

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
        if self.simulator:
            try:
                return self.simulator.get_upcoming_news(hours_ahead, currency)
            except Exception as e:
                logger.warning(f"Error getting upcoming news from simulator: {str(e)}")

        # Fallback implementation
        now = datetime.now()
        max_time = now + datetime.timedelta(hours=hours_ahead)

        if currency:
            return [news for news in self._news_cache
                   if currency in news.affected_currencies
                   and now < news.timestamp <= max_time]

        return [news for news in self._news_cache if now < news.timestamp <= max_time]

    def get_current_sentiment(self, currency: str) -> SentimentLevel:
        """
        Get current market sentiment for a currency.

        Args:
            currency: Currency to check

        Returns:
            Current sentiment level
        """
        if self.simulator:
            try:
                return self.simulator.get_current_sentiment(currency)
            except Exception as e:
                logger.warning(f"Error getting sentiment from simulator: {str(e)}")

        # Fallback implementation
        return self._sentiment_cache.get(currency, SentimentLevel.NEUTRAL)

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
        if self.simulator:
            try:
                return self.simulator.get_sentiment_history(currency, lookback_periods)
            except Exception as e:
                logger.warning(f"Error getting sentiment history from simulator: {str(e)}")

        # Fallback implementation - return empty history
        return []

    def get_sentiment_impact(self, symbol: str) -> Dict[str, float]:
        """
        Get estimated sentiment impact on a symbol.

        Args:
            symbol: Trading symbol to check

        Returns:
            Dictionary with impact metrics
        """
        if self.simulator:
            try:
                return self.simulator.get_sentiment_impact(symbol)
            except Exception as e:
                logger.warning(f"Error getting sentiment impact from simulator: {str(e)}")

        # Fallback implementation
        # Parse the symbol to get the currencies
        if '/' in symbol:
            base, quote = symbol.split('/')
            base_sentiment = self.get_current_sentiment(base)
            quote_sentiment = self.get_current_sentiment(quote)

            # Convert sentiment to numeric value
            sentiment_values = {
                SentimentLevel.VERY_BEARISH: -2.0,
                SentimentLevel.BEARISH: -1.0,
                SentimentLevel.NEUTRAL: 0.0,
                SentimentLevel.BULLISH: 1.0,
                SentimentLevel.VERY_BULLISH: 2.0
            }

            base_value = sentiment_values.get(base_sentiment, 0.0)
            quote_value = sentiment_values.get(quote_sentiment, 0.0)

            # Calculate net impact (positive means bullish for the pair)
            net_impact = base_value - quote_value

            return {
                'net_impact': net_impact,
                'base_sentiment': base_sentiment.value,
                'quote_sentiment': quote_sentiment.value,
                'upcoming_news_count': len(self.get_upcoming_news(24, base)) + len(self.get_upcoming_news(24, quote)),
                'confidence': 0.8
            }

        return {
            'net_impact': 0.0,
            'confidence': 0.5
        }

    def set_sentiment(self, currency: str, sentiment: SentimentLevel) -> None:
        """
        Set the current sentiment for a currency (for testing/simulation).

        Args:
            currency: The currency
            sentiment: The sentiment level to set
        """
        self._sentiment_cache[currency] = sentiment
        logger.debug(f"Set {currency} sentiment to {sentiment}")

    def add_news_event(self, news_event: NewsEvent) -> None:
        """
        Add a news event to the cache (for testing/simulation).

        Args:
            news_event: The news event to add
        """
        self._news_cache.append(news_event)
        logger.debug(f"Added news event: {news_event.title}")
