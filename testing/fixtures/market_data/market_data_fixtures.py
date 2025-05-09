"""
Market Data Test Fixtures

This module provides test fixtures for market data components, including:
1. OHLCV data fixtures for different timeframes and currency pairs
2. Tick data fixtures
3. Order book fixtures
4. Market event fixtures
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import pytest

# Add project root to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))


@pytest.fixture
def sample_ohlcv_data():
    """
    Generate sample OHLCV data for testing.
    
    Returns:
        DataFrame with OHLCV data
    """
    # Create date range
    dates = pd.date_range(start='2023-01-01', periods=100, freq='1H')
    
    # Generate random OHLCV data
    np.random.seed(42)  # For reproducibility
    
    # Generate random walk for close prices
    close = np.random.normal(0, 1, size=100).cumsum() + 100
    
    # Generate other OHLCV columns based on close
    high = close + np.random.uniform(0, 1, size=100)
    low = close - np.random.uniform(0, 1, size=100)
    open_price = close - np.random.uniform(-0.5, 0.5, size=100)
    volume = np.random.uniform(1000, 5000, size=100)
    
    # Create DataFrame
    df = pd.DataFrame({
        'timestamp': dates,
        'open': open_price,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    })
    
    return df


@pytest.fixture
def sample_tick_data():
    """
    Generate sample tick data for testing.
    
    Returns:
        DataFrame with tick data
    """
    # Create date range with millisecond precision
    base_time = datetime(2023, 1, 1)
    timestamps = [base_time + timedelta(milliseconds=i*100) for i in range(1000)]
    
    # Generate random tick data
    np.random.seed(42)  # For reproducibility
    
    # Generate random walk for prices
    price = np.random.normal(0, 0.001, size=1000).cumsum() + 1.2000
    
    # Generate bid/ask prices and volumes
    bid = price - np.random.uniform(0, 0.0002, size=1000)
    ask = price + np.random.uniform(0, 0.0002, size=1000)
    bid_volume = np.random.uniform(10000, 50000, size=1000)
    ask_volume = np.random.uniform(10000, 50000, size=1000)
    
    # Create DataFrame
    df = pd.DataFrame({
        'timestamp': timestamps,
        'bid': bid,
        'ask': ask,
        'bid_volume': bid_volume,
        'ask_volume': ask_volume
    })
    
    return df


@pytest.fixture
def sample_order_book():
    """
    Generate sample order book data for testing.
    
    Returns:
        Dictionary with order book data
    """
    # Generate random order book
    np.random.seed(42)  # For reproducibility
    
    # Base price
    base_price = 1.2000
    
    # Generate bid levels (below base price)
    bid_levels = []
    for i in range(10):
        price = base_price - (i + 1) * 0.0001
        volume = np.random.uniform(100000, 500000)
        bid_levels.append({
            'price': price,
            'volume': volume
        })
    
    # Generate ask levels (above base price)
    ask_levels = []
    for i in range(10):
        price = base_price + (i + 1) * 0.0001
        volume = np.random.uniform(100000, 500000)
        ask_levels.append({
            'price': price,
            'volume': volume
        })
    
    # Create order book
    order_book = {
        'timestamp': datetime(2023, 1, 1),
        'symbol': 'EUR/USD',
        'bids': bid_levels,
        'asks': ask_levels
    }
    
    return order_book


@pytest.fixture
def sample_market_events():
    """
    Generate sample market events for testing.
    
    Returns:
        List of market events
    """
    # Generate random market events
    np.random.seed(42)  # For reproducibility
    
    # Event types
    event_types = ['price_spike', 'liquidity_change', 'spread_widening', 'high_volatility']
    
    # Generate events
    events = []
    base_time = datetime(2023, 1, 1)
    
    for i in range(20):
        event_time = base_time + timedelta(minutes=i*30)
        event_type = np.random.choice(event_types)
        
        # Generate event-specific data
        if event_type == 'price_spike':
            data = {
                'magnitude': np.random.uniform(0.001, 0.005),
                'direction': np.random.choice(['up', 'down'])
            }
        elif event_type == 'liquidity_change':
            data = {
                'change_percent': np.random.uniform(-30, 30),
                'side': np.random.choice(['bid', 'ask', 'both'])
            }
        elif event_type == 'spread_widening':
            data = {
                'factor': np.random.uniform(1.5, 5.0),
                'duration_seconds': np.random.uniform(10, 60)
            }
        elif event_type == 'high_volatility':
            data = {
                'volatility_increase': np.random.uniform(1.5, 3.0),
                'duration_minutes': np.random.uniform(5, 30)
            }
        
        events.append({
            'timestamp': event_time,
            'symbol': 'EUR/USD',
            'type': event_type,
            'data': data
        })
    
    return events


@pytest.fixture
def currency_pairs():
    """
    Provide a list of currency pairs for testing.
    
    Returns:
        List of currency pairs
    """
    return [
        'EUR/USD',
        'GBP/USD',
        'USD/JPY',
        'AUD/USD',
        'USD/CAD',
        'NZD/USD',
        'USD/CHF',
        'EUR/GBP'
    ]


@pytest.fixture
def timeframes():
    """
    Provide a list of timeframes for testing.
    
    Returns:
        List of timeframes
    """
    return [
        '1m',
        '5m',
        '15m',
        '30m',
        '1h',
        '4h',
        '1d',
        '1w'
    ]


@pytest.fixture
def market_data_factory():
    """
    Factory fixture for generating market data.
    
    Returns:
        Function to generate market data
    """
    def _factory(
        symbol: str,
        timeframe: str,
        start_date: datetime,
        end_date: Optional[datetime] = None,
        include_gaps: bool = False
    ) -> pd.DataFrame:
        """
        Generate market data for a specific symbol and timeframe.
        
        Args:
            symbol: Currency pair symbol
            timeframe: Timeframe string (e.g., '1h', '1d')
            start_date: Start date
            end_date: End date (default: start_date + 100 periods)
            include_gaps: Whether to include gaps in the data
            
        Returns:
            DataFrame with OHLCV data
        """
        # Parse timeframe
        if timeframe.endswith('m'):
            freq = f"{timeframe[:-1]}min"
        elif timeframe.endswith('h'):
            freq = f"{timeframe[:-1]}H"
        elif timeframe.endswith('d'):
            freq = f"{timeframe[:-1]}D"
        elif timeframe.endswith('w'):
            freq = f"{timeframe[:-1]}W"
        else:
            raise ValueError(f"Invalid timeframe: {timeframe}")
        
        # Set end date if not provided
        if end_date is None:
            # Generate 100 periods by default
            if timeframe.endswith('m'):
                end_date = start_date + timedelta(minutes=int(timeframe[:-1]) * 100)
            elif timeframe.endswith('h'):
                end_date = start_date + timedelta(hours=int(timeframe[:-1]) * 100)
            elif timeframe.endswith('d'):
                end_date = start_date + timedelta(days=int(timeframe[:-1]) * 100)
            elif timeframe.endswith('w'):
                end_date = start_date + timedelta(weeks=int(timeframe[:-1]) * 100)
        
        # Create date range
        dates = pd.date_range(start=start_date, end=end_date, freq=freq)
        
        # Remove some dates if including gaps
        if include_gaps:
            # Remove ~5% of dates randomly
            np.random.seed(42)  # For reproducibility
            drop_indices = np.random.choice(
                range(len(dates)),
                size=int(len(dates) * 0.05),
                replace=False
            )
            dates = dates.delete(drop_indices)
        
        # Generate random OHLCV data
        np.random.seed(hash(symbol) % 2**32)  # Seed based on symbol for variety
        
        # Base price depends on the symbol
        if symbol == 'EUR/USD':
            base_price = 1.2000
        elif symbol == 'GBP/USD':
            base_price = 1.4000
        elif symbol == 'USD/JPY':
            base_price = 110.00
        elif symbol == 'AUD/USD':
            base_price = 0.7500
        elif symbol == 'USD/CAD':
            base_price = 1.2500
        elif symbol == 'NZD/USD':
            base_price = 0.7000
        elif symbol == 'USD/CHF':
            base_price = 0.9000
        elif symbol == 'EUR/GBP':
            base_price = 0.8500
        else:
            base_price = 1.0000
        
        # Generate random walk for close prices
        volatility = 0.001  # Base volatility
        
        # Adjust volatility based on timeframe
        if timeframe.endswith('m'):
            volatility *= 0.5
        elif timeframe.endswith('h'):
            volatility *= 1.0
        elif timeframe.endswith('d'):
            volatility *= 2.0
        elif timeframe.endswith('w'):
            volatility *= 4.0
        
        # Generate close prices
        close = np.random.normal(0, volatility, size=len(dates)).cumsum() + base_price
        
        # Generate other OHLCV columns based on close
        high = close + np.random.uniform(0, volatility, size=len(dates))
        low = close - np.random.uniform(0, volatility, size=len(dates))
        open_price = close - np.random.uniform(-volatility/2, volatility/2, size=len(dates))
        
        # Volume depends on the symbol and timeframe
        base_volume = 1000
        if timeframe.endswith('m'):
            base_volume *= 1
        elif timeframe.endswith('h'):
            base_volume *= 10
        elif timeframe.endswith('d'):
            base_volume *= 100
        elif timeframe.endswith('w'):
            base_volume *= 1000
        
        volume = np.random.uniform(base_volume, base_volume * 5, size=len(dates))
        
        # Create DataFrame
        df = pd.DataFrame({
            'timestamp': dates,
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        })
        
        return df
    
    return _factory


@pytest.fixture
def market_data_service_mock():
    """
    Mock for the market data service.
    
    Returns:
        Mock object for the market data service
    """
    from unittest.mock import MagicMock
    
    # Create mock
    mock = MagicMock()
    
    # Set up mock methods
    async def get_ohlcv(symbol, timeframe, start_date, end_date=None, limit=None):
        """Mock implementation of get_ohlcv."""
        # Use the market_data_factory to generate data
        factory = market_data_factory()
        data = factory(symbol, timeframe, start_date, end_date)
        
        # Apply limit if provided
        if limit is not None and limit < len(data):
            data = data.tail(limit)
        
        return data
    
    async def get_tick_data(symbol, start_date, end_date=None, limit=None):
        """Mock implementation of get_tick_data."""
        # Generate tick data
        ticks = sample_tick_data()
        
        # Filter by date range
        ticks = ticks[(ticks['timestamp'] >= start_date)]
        if end_date is not None:
            ticks = ticks[(ticks['timestamp'] <= end_date)]
        
        # Apply limit if provided
        if limit is not None and limit < len(ticks):
            ticks = ticks.tail(limit)
        
        return ticks
    
    async def get_order_book(symbol, depth=10):
        """Mock implementation of get_order_book."""
        # Generate order book
        order_book = sample_order_book()
        
        # Adjust depth
        if depth < 10:
            order_book['bids'] = order_book['bids'][:depth]
            order_book['asks'] = order_book['asks'][:depth]
        
        return order_book
    
    # Assign mock methods
    mock.get_ohlcv = get_ohlcv
    mock.get_tick_data = get_tick_data
    mock.get_order_book = get_order_book
    
    return mock