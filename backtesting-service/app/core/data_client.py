"""
Data client for fetching market data.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class DataClient:
    """Client for fetching market data from various sources."""
    
    async def get_market_data(self, symbol, timeframe, start_date, end_date):
        """
        Fetch market data for a specific symbol and timeframe.
        
        Args:
            symbol (str): The trading symbol (e.g., 'EURUSD')
            timeframe (str): The timeframe (e.g., '1h', '4h', '1d')
            start_date (datetime): The start date for the data
            end_date (datetime): The end date for the data
            
        Returns:
            pd.DataFrame: DataFrame with OHLCV data
        """
        logger.info(f"Fetching market data for {symbol} {timeframe} from {start_date} to {end_date}")
        
        # In a real implementation, this would fetch data from a database or API
        # For now, we'll generate synthetic data
        
        # Calculate the number of periods based on the timeframe
        if timeframe == '1h':
            periods = int((end_date - start_date).total_seconds() / 3600) + 1
        elif timeframe == '4h':
            periods = int((end_date - start_date).total_seconds() / (3600 * 4)) + 1
        elif timeframe == '1d':
            periods = int((end_date - start_date).total_seconds() / (3600 * 24)) + 1
        else:
            periods = 100  # Default
        
        # Generate synthetic data
        np.random.seed(42)  # For reproducibility
        
        # Create date range
        date_range = pd.date_range(start=start_date, periods=periods, freq=timeframe)
        
        # Generate price data
        base_price = 1.0 if symbol.startswith('EUR') else (0.8 if symbol.startswith('GBP') else 110.0)
        volatility = 0.002  # Daily volatility
        
        # Generate random walk
        returns = np.random.normal(0, volatility, periods)
        price_path = base_price * (1 + np.cumsum(returns))
        
        # Create OHLCV data
        data = pd.DataFrame({
            'open': price_path,
            'high': price_path * (1 + np.random.uniform(0, 0.003, periods)),
            'low': price_path * (1 - np.random.uniform(0, 0.003, periods)),
            'close': price_path * (1 + np.random.normal(0, 0.001, periods)),
            'volume': np.random.randint(1000, 10000, periods)
        }, index=date_range)
        
        # Ensure high is always the highest price
        data['high'] = data[['open', 'high', 'close']].max(axis=1)
        
        # Ensure low is always the lowest price
        data['low'] = data[['open', 'low', 'close']].min(axis=1)
        
        return data
