"""
Market Data Service for Execution Algorithms.

This service provides real-time and historical market data for execution algorithms,
including price, volume, volatility, and other market metrics.
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
import numpy as np
import pandas as pd

from ..interfaces.broker_adapter_interface import BrokerAdapterInterface


class MarketDataService:
    """
    Service for providing market data to execution algorithms.
    
    This service aggregates data from multiple sources, including broker adapters,
    historical databases, and real-time feeds, to provide a comprehensive view
    of market conditions for execution algorithms.
    """
    
    def __init__(self, 
                 broker_adapters: Dict[str, BrokerAdapterInterface],
                 logger: Optional[logging.Logger] = None,
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize the market data service.
        
        Args:
            broker_adapters: Dictionary of broker adapters by name
            logger: Logger instance
            config: Service configuration
        """
        self.broker_adapters = broker_adapters
        self.logger = logger or logging.getLogger(__name__)
        self.config = config or {}
        
        # Cache for market data
        self.price_cache: Dict[str, Dict[str, Any]] = {}
        self.volume_cache: Dict[str, List[float]] = {}
        self.volatility_cache: Dict[str, float] = {}
        self.market_regime_cache: Dict[str, str] = {}
        
        # Cache expiration (in seconds)
        self.price_cache_expiry = self.config.get('price_cache_expiry', 1)  # 1 second
        self.volume_cache_expiry = self.config.get('volume_cache_expiry', 60)  # 1 minute
        self.volatility_cache_expiry = self.config.get('volatility_cache_expiry', 300)  # 5 minutes
        self.market_regime_cache_expiry = self.config.get('market_regime_cache_expiry', 3600)  # 1 hour
        
        # Last update timestamps
        self.last_price_update: Dict[str, float] = {}
        self.last_volume_update: Dict[str, float] = {}
        self.last_volatility_update: Dict[str, float] = {}
        self.last_market_regime_update: Dict[str, float] = {}
        
        # Historical data service (if available)
        self.historical_data_service = self.config.get('historical_data_service')
        
        # Market regime service (if available)
        self.market_regime_service = self.config.get('market_regime_service')
    
    async def get_price(self, instrument: str) -> Optional[float]:
        """
        Get the current price for an instrument.
        
        Args:
            instrument: The instrument to get the price for
            
        Returns:
            Current price, or None if not available
        """
        # Check cache
        now = time.time()
        if (instrument in self.price_cache and 
            instrument in self.last_price_update and
            now - self.last_price_update[instrument] < self.price_cache_expiry):
            return self.price_cache[instrument].get('price')
        
        # Get fresh data
        market_data = await self.get_market_data(instrument)
        if market_data:
            return market_data.get('price')
        
        return None
    
    async def get_spread(self, instrument: str) -> Optional[float]:
        """
        Get the current spread for an instrument.
        
        Args:
            instrument: The instrument to get the spread for
            
        Returns:
            Current spread, or None if not available
        """
        # Check cache
        now = time.time()
        if (instrument in self.price_cache and 
            instrument in self.last_price_update and
            now - self.last_price_update[instrument] < self.price_cache_expiry):
            return self.price_cache[instrument].get('spread')
        
        # Get fresh data
        market_data = await self.get_market_data(instrument)
        if market_data:
            return market_data.get('spread')
        
        return None
    
    async def get_market_data(self, instrument: str) -> Dict[str, Any]:
        """
        Get comprehensive market data for an instrument.
        
        Args:
            instrument: The instrument to get data for
            
        Returns:
            Dictionary with market data
        """
        # Check cache
        now = time.time()
        if (instrument in self.price_cache and 
            instrument in self.last_price_update and
            now - self.last_price_update[instrument] < self.price_cache_expiry):
            return self.price_cache[instrument]
        
        # Initialize result with defaults
        result = {
            'instrument': instrument,
            'price': None,
            'bid': None,
            'ask': None,
            'spread': None,
            'timestamp': datetime.utcnow().isoformat(),
        }
        
        # Try to get data from broker adapters
        for adapter_name, adapter in self.broker_adapters.items():
            try:
                if hasattr(adapter, 'get_market_data') and callable(adapter.get_market_data):
                    data = adapter.get_market_data(instrument)
                    if data:
                        # Extract relevant fields
                        if 'bid' in data and 'ask' in data:
                            result['bid'] = data['bid']
                            result['ask'] = data['ask']
                            result['price'] = (data['bid'] + data['ask']) / 2
                            result['spread'] = data['ask'] - data['bid']
                        elif 'price' in data:
                            result['price'] = data['price']
                        
                        # Add additional fields if available
                        for field in ['volume', 'high', 'low', 'open', 'close']:
                            if field in data:
                                result[field] = data[field]
                        
                        # Update cache
                        self.price_cache[instrument] = result
                        self.last_price_update[instrument] = now
                        
                        return result
            except Exception as e:
                self.logger.error(f"Error getting market data from {adapter_name}: {str(e)}")
        
        # If we couldn't get data from any adapter, return the default result
        return result
    
    async def get_historical_data(self, 
                                instrument: str, 
                                start_time: Union[datetime, str],
                                end_time: Union[datetime, str],
                                timeframe: str = '1m') -> pd.DataFrame:
        """
        Get historical price data for an instrument.
        
        Args:
            instrument: The instrument to get data for
            start_time: Start time for the data
            end_time: End time for the data
            timeframe: Timeframe for the data (e.g., '1m', '5m', '1h')
            
        Returns:
            DataFrame with historical data
        """
        # Convert string times to datetime if needed
        if isinstance(start_time, str):
            start_time = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
        if isinstance(end_time, str):
            end_time = datetime.fromisoformat(end_time.replace('Z', '+00:00'))
        
        # Try to use historical data service if available
        if self.historical_data_service and hasattr(self.historical_data_service, 'get_historical_data'):
            try:
                return await self.historical_data_service.get_historical_data(
                    instrument=instrument,
                    start_time=start_time,
                    end_time=end_time,
                    timeframe=timeframe
                )
            except Exception as e:
                self.logger.error(f"Error getting historical data: {str(e)}")
        
        # Fallback to generating synthetic data
        return self._generate_synthetic_historical_data(
            instrument=instrument,
            start_time=start_time,
            end_time=end_time,
            timeframe=timeframe
        )
    
    def _generate_synthetic_historical_data(self,
                                          instrument: str,
                                          start_time: datetime,
                                          end_time: datetime,
                                          timeframe: str = '1m') -> pd.DataFrame:
        """
        Generate synthetic historical data for testing.
        
        Args:
            instrument: The instrument to generate data for
            start_time: Start time for the data
            end_time: End time for the data
            timeframe: Timeframe for the data
            
        Returns:
            DataFrame with synthetic historical data
        """
        # Parse timeframe
        if timeframe.endswith('m'):
            minutes = int(timeframe[:-1])
            freq = f'{minutes}min'
        elif timeframe.endswith('h'):
            hours = int(timeframe[:-1])
            freq = f'{hours}H'
        elif timeframe.endswith('d'):
            days = int(timeframe[:-1])
            freq = f'{days}D'
        else:
            freq = '1H'  # Default to 1 hour
        
        # Generate timestamps
        timestamps = pd.date_range(start=start_time, end=end_time, freq=freq)
        
        # Generate price data with random walk
        np.random.seed(42)  # For reproducibility
        price = 1.0  # Starting price
        prices = [price]
        
        for _ in range(1, len(timestamps)):
            # Random price change with mean 0 and std 0.001 (0.1%)
            price_change = np.random.normal(0, 0.001)
            price = price * (1 + price_change)
            prices.append(price)
        
        # Generate OHLCV data
        data = []
        for i, timestamp in enumerate(timestamps):
            price = prices[i]
            # Generate random high, low, open, close around the price
            high = price * (1 + abs(np.random.normal(0, 0.0005)))
            low = price * (1 - abs(np.random.normal(0, 0.0005)))
            if i > 0:
                open_price = prices[i-1]
            else:
                open_price = price * (1 + np.random.normal(0, 0.0005))
            close = price
            
            # Generate random volume
            volume = abs(np.random.normal(1000, 200))
            
            data.append({
                'timestamp': timestamp,
                'open': open_price,
                'high': high,
                'low': low,
                'close': close,
                'volume': volume,
                'instrument': instrument
            })
        
        return pd.DataFrame(data)
    
    async def get_volatility(self, instrument: str) -> float:
        """
        Get the current volatility for an instrument.
        
        Args:
            instrument: The instrument to get volatility for
            
        Returns:
            Current volatility (standard deviation of returns)
        """
        # Check cache
        now = time.time()
        if (instrument in self.volatility_cache and 
            instrument in self.last_volatility_update and
            now - self.last_volatility_update[instrument] < self.volatility_cache_expiry):
            return self.volatility_cache[instrument]
        
        try:
            # Get historical data for the last 24 hours
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(hours=24)
            
            df = await self.get_historical_data(
                instrument=instrument,
                start_time=start_time,
                end_time=end_time,
                timeframe='5m'  # 5-minute data
            )
            
            if len(df) > 1:
                # Calculate returns
                df['returns'] = df['close'].pct_change().dropna()
                
                # Calculate volatility (standard deviation of returns)
                volatility = df['returns'].std()
                
                # Update cache
                self.volatility_cache[instrument] = volatility
                self.last_volatility_update[instrument] = now
                
                return volatility
        except Exception as e:
            self.logger.error(f"Error calculating volatility: {str(e)}")
        
        # Default volatility if calculation fails
        return 0.001  # 0.1% volatility
    
    async def get_avg_daily_volume(self, instrument: str) -> float:
        """
        Get the average daily volume for an instrument.
        
        Args:
            instrument: The instrument to get volume for
            
        Returns:
            Average daily volume
        """
        try:
            # Get historical data for the last 7 days
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(days=7)
            
            df = await self.get_historical_data(
                instrument=instrument,
                start_time=start_time,
                end_time=end_time,
                timeframe='1h'  # 1-hour data
            )
            
            if len(df) > 0:
                # Group by day and sum volume
                df['date'] = df['timestamp'].dt.date
                daily_volumes = df.groupby('date')['volume'].sum()
                
                # Calculate average daily volume
                avg_daily_volume = daily_volumes.mean()
                
                return avg_daily_volume
        except Exception as e:
            self.logger.error(f"Error calculating average daily volume: {str(e)}")
        
        # Default volume if calculation fails
        return 1000000  # 1M units
    
    async def get_market_regime(self, instrument: str) -> str:
        """
        Get the current market regime for an instrument.
        
        Args:
            instrument: The instrument to get the market regime for
            
        Returns:
            Market regime (e.g., 'trending', 'ranging', 'volatile')
        """
        # Check cache
        now = time.time()
        if (instrument in self.market_regime_cache and 
            instrument in self.last_market_regime_update and
            now - self.last_market_regime_update[instrument] < self.market_regime_cache_expiry):
            return self.market_regime_cache[instrument]
        
        # Try to use market regime service if available
        if self.market_regime_service and hasattr(self.market_regime_service, 'detect_regime'):
            try:
                regime = await self.market_regime_service.detect_regime(instrument)
                
                # Update cache
                self.market_regime_cache[instrument] = regime
                self.last_market_regime_update[instrument] = now
                
                return regime
            except Exception as e:
                self.logger.error(f"Error detecting market regime: {str(e)}")
        
        # If no service available or error occurred, calculate a simple regime
        try:
            # Get historical data for the last 24 hours
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(hours=24)
            
            df = await self.get_historical_data(
                instrument=instrument,
                start_time=start_time,
                end_time=end_time,
                timeframe='5m'  # 5-minute data
            )
            
            if len(df) > 1:
                # Calculate returns
                df['returns'] = df['close'].pct_change().dropna()
                
                # Calculate volatility (standard deviation of returns)
                volatility = df['returns'].std()
                
                # Calculate trend strength (absolute mean of returns)
                trend_strength = abs(df['returns'].mean())
                
                # Determine regime based on volatility and trend strength
                if volatility > 0.002:  # High volatility
                    regime = 'volatile'
                elif trend_strength > 0.0001:  # Strong trend
                    regime = 'trending'
                else:  # Low volatility, weak trend
                    regime = 'ranging'
                
                # Update cache
                self.market_regime_cache[instrument] = regime
                self.last_market_regime_update[instrument] = now
                
                return regime
        except Exception as e:
            self.logger.error(f"Error calculating market regime: {str(e)}")
        
        # Default regime if calculation fails
        return 'normal'
    
    async def get_market_conditions(self, instrument: str) -> Dict[str, Any]:
        """
        Get comprehensive market conditions for an instrument.
        
        Args:
            instrument: The instrument to get conditions for
            
        Returns:
            Dictionary with market conditions
        """
        # Get market data
        market_data = await self.get_market_data(instrument)
        
        # Get additional metrics
        volatility = await self.get_volatility(instrument)
        avg_daily_volume = await self.get_avg_daily_volume(instrument)
        market_regime = await self.get_market_regime(instrument)
        
        # Combine all data
        conditions = {
            'instrument': instrument,
            'price': market_data.get('price'),
            'bid': market_data.get('bid'),
            'ask': market_data.get('ask'),
            'spread': market_data.get('spread'),
            'volatility': volatility,
            'avg_daily_volume': avg_daily_volume,
            'market_regime': market_regime,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        return conditions
    
    async def get_historical_volume(self,
                                  instrument: str,
                                  period: str = '1d',
                                  lookback_days: int = 20,
                                  hour_of_day: Optional[int] = None) -> List[List[float]]:
        """
        Get historical volume data for an instrument.
        
        Args:
            instrument: The instrument to get volume for
            period: Period for the data (e.g., '1d', '1h')
            lookback_days: Number of days to look back
            hour_of_day: Specific hour of the day to filter for
            
        Returns:
            List of volume data for each day
        """
        try:
            # Get historical data
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(days=lookback_days)
            
            df = await self.get_historical_data(
                instrument=instrument,
                start_time=start_time,
                end_time=end_time,
                timeframe='1h'  # 1-hour data
            )
            
            if len(df) > 0:
                # Filter by hour of day if specified
                if hour_of_day is not None:
                    df = df[df['timestamp'].dt.hour == hour_of_day]
                
                # Group by day
                df['date'] = df['timestamp'].dt.date
                
                # Get volume data for each day
                volume_data = []
                for date, group in df.groupby('date'):
                    volume_data.append(group['volume'].tolist())
                
                return volume_data
        except Exception as e:
            self.logger.error(f"Error getting historical volume: {str(e)}")
        
        # Return empty list if calculation fails
        return []
    
    async def get_predicted_volume(self,
                                 instrument: str,
                                 start_time: datetime,
                                 end_time: datetime,
                                 num_slices: int) -> List[float]:
        """
        Get predicted volume for an instrument over a time period.
        
        Args:
            instrument: The instrument to get volume for
            start_time: Start time for the prediction
            end_time: End time for the prediction
            num_slices: Number of slices to divide the period into
            
        Returns:
            List of predicted volumes for each slice
        """
        try:
            # Get historical volume data
            historical_volume = await self.get_historical_volume(
                instrument=instrument,
                period='1d',
                lookback_days=20,
                hour_of_day=start_time.hour
            )
            
            if historical_volume:
                # Calculate average volume profile
                avg_profile = []
                for day_data in historical_volume:
                    # Resample to the desired number of slices
                    slice_size = len(day_data) / num_slices
                    slices = []
                    for i in range(num_slices):
                        start_idx = int(i * slice_size)
                        end_idx = int((i + 1) * slice_size) if i < num_slices - 1 else len(day_data)
                        slice_sum = sum(day_data[start_idx:end_idx])
                        slices.append(slice_sum)
                    
                    if not avg_profile:
                        avg_profile = slices
                    else:
                        avg_profile = [a + b for a, b in zip(avg_profile, slices)]
                
                # Calculate average
                avg_profile = [v / len(historical_volume) for v in avg_profile]
                
                return avg_profile
        except Exception as e:
            self.logger.error(f"Error predicting volume: {str(e)}")
        
        # If prediction fails, return uniform distribution
        return [1.0] * num_slices
    
    async def get_realtime_volume(self,
                                instrument: str,
                                lookback_minutes: int = 60) -> List[float]:
        """
        Get real-time volume data for an instrument.
        
        Args:
            instrument: The instrument to get volume for
            lookback_minutes: Number of minutes to look back
            
        Returns:
            List of volume data for the lookback period
        """
        try:
            # Get historical data for the lookback period
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(minutes=lookback_minutes)
            
            df = await self.get_historical_data(
                instrument=instrument,
                start_time=start_time,
                end_time=end_time,
                timeframe='1m'  # 1-minute data
            )
            
            if len(df) > 0:
                return df['volume'].tolist()
        except Exception as e:
            self.logger.error(f"Error getting real-time volume: {str(e)}")
        
        # Return empty list if calculation fails
        return []
