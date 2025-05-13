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
from ..error import with_exception_handling, async_with_exception_handling, MarketDataError, DataFetchError, ServiceUnavailableError


from utils.utils import (
    with_broker_api_resilience,
    with_market_data_resilience,
    with_order_execution_resilience,
    with_risk_management_resilience,
    with_database_resilience
)

class MarketDataService:
    """
    Service for providing market data to execution algorithms.

    This service aggregates data from multiple sources, including broker adapters,
    historical databases, and real-time feeds, to provide a comprehensive view
    of market conditions for execution algorithms.
    """

    def __init__(self, broker_adapters: Dict[str, BrokerAdapterInterface],
        logger: Optional[logging.Logger]=None, config: Optional[Dict[str,
        Any]]=None):
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
        self.price_cache: Dict[str, Dict[str, Any]] = {}
        self.volume_cache: Dict[str, List[float]] = {}
        self.volatility_cache: Dict[str, float] = {}
        self.market_regime_cache: Dict[str, str] = {}
        self.price_cache_expiry = self.config_manager.get('price_cache_expiry', 1)
        self.volume_cache_expiry = self.config_manager.get('volume_cache_expiry', 60)
        self.volatility_cache_expiry = self.config.get(
            'volatility_cache_expiry', 300)
        self.market_regime_cache_expiry = self.config.get(
            'market_regime_cache_expiry', 3600)
        self.last_price_update: Dict[str, float] = {}
        self.last_volume_update: Dict[str, float] = {}
        self.last_volatility_update: Dict[str, float] = {}
        self.last_market_regime_update: Dict[str, float] = {}
        self.historical_data_service = self.config.get(
            'historical_data_service')
        self.market_regime_service = self.config_manager.get('market_regime_service')

    @with_market_data_resilience('get_price')
    @async_with_exception_handling
    async def get_price(self, instrument: str) ->Optional[float]:
        """
        Get the current price for an instrument.

        Args:
            instrument: The instrument to get the price for

        Returns:
            Current price, or None if not available

        Raises:
            MarketDataError: If there's an error getting the price
        """
        if not instrument:
            raise MarketDataError(message='Instrument cannot be empty',
                symbol=instrument)
        now = time.time()
        if (instrument in self.price_cache and instrument in self.
            last_price_update and now - self.last_price_update[instrument] <
            self.price_cache_expiry):
            return self.price_cache[instrument].get('price')
        try:
            market_data = await self.get_market_data(instrument)
            if market_data:
                return market_data.get('price')
            return None
        except Exception as e:
            raise MarketDataError(message=
                f'Failed to get price for {instrument}: {str(e)}', symbol=
                instrument, details={'error': str(e)})

    @with_broker_api_resilience('get_spread')
    @async_with_exception_handling
    async def get_spread(self, instrument: str) ->Optional[float]:
        """
        Get the current spread for an instrument.

        Args:
            instrument: The instrument to get the spread for

        Returns:
            Current spread, or None if not available

        Raises:
            MarketDataError: If there's an error getting the spread
        """
        if not instrument:
            raise MarketDataError(message='Instrument cannot be empty',
                symbol=instrument)
        now = time.time()
        if (instrument in self.price_cache and instrument in self.
            last_price_update and now - self.last_price_update[instrument] <
            self.price_cache_expiry):
            return self.price_cache[instrument].get('spread')
        try:
            market_data = await self.get_market_data(instrument)
            if market_data:
                return market_data.get('spread')
            return None
        except Exception as e:
            raise MarketDataError(message=
                f'Failed to get spread for {instrument}: {str(e)}', symbol=
                instrument, details={'error': str(e)})

    @with_market_data_resilience('get_market_data')
    @async_with_exception_handling
    async def get_market_data(self, instrument: str) ->Dict[str, Any]:
        """
        Get comprehensive market data for an instrument.

        Args:
            instrument: The instrument to get data for

        Returns:
            Dictionary with market data

        Raises:
            MarketDataError: If there's an error getting market data
        """
        if not instrument:
            raise MarketDataError(message='Instrument cannot be empty',
                symbol=instrument)
        now = time.time()
        if (instrument in self.price_cache and instrument in self.
            last_price_update and now - self.last_price_update[instrument] <
            self.price_cache_expiry):
            return self.price_cache[instrument]
        result = {'instrument': instrument, 'price': None, 'bid': None,
            'ask': None, 'spread': None, 'timestamp': datetime.utcnow().
            isoformat()}
        adapter_errors = []
        for adapter_name, adapter in self.broker_adapters.items():
            try:
                if hasattr(adapter, 'get_market_data') and callable(adapter
                    .get_market_data):
                    data = adapter.get_market_data(instrument)
                    if data:
                        if 'bid' in data and 'ask' in data:
                            result['bid'] = data['bid']
                            result['ask'] = data['ask']
                            result['price'] = (data['bid'] + data['ask']) / 2
                            result['spread'] = data['ask'] - data['bid']
                        elif 'price' in data:
                            result['price'] = data['price']
                        for field in ['volume', 'high', 'low', 'open', 'close'
                            ]:
                            if field in data:
                                result[field] = data[field]
                        self.price_cache[instrument] = result
                        self.last_price_update[instrument] = now
                        return result
            except Exception as e:
                error_msg = (
                    f'Error getting market data from {adapter_name}: {str(e)}')
                self.logger.error(error_msg)
                adapter_errors.append({'adapter': adapter_name, 'error':
                    str(e)})
        if adapter_errors and len(adapter_errors) == len(self.broker_adapters):
            raise MarketDataError(message=
                f'Failed to get market data for {instrument} from all adapters'
                , symbol=instrument, details={'adapter_errors': adapter_errors}
                )
        return result

    @with_broker_api_resilience('get_historical_data')
    @async_with_exception_handling
    async def get_historical_data(self, instrument: str, start_time: Union[
        datetime, str], end_time: Union[datetime, str], timeframe: str='1m'
        ) ->pd.DataFrame:
        """
        Get historical price data for an instrument.

        Args:
            instrument: The instrument to get data for
            start_time: Start time for the data
            end_time: End time for the data
            timeframe: Timeframe for the data (e.g., '1m', '5m', '1h')

        Returns:
            DataFrame with historical data

        Raises:
            MarketDataError: If there's an error getting historical data
        """
        if not instrument:
            raise MarketDataError(message='Instrument cannot be empty',
                symbol=instrument)
        valid_timeframes = ['1m', '5m', '15m', '30m', '1h', '4h', '1d']
        if timeframe not in valid_timeframes:
            raise MarketDataError(message=
                f"Invalid timeframe: {timeframe}. Valid timeframes are: {', '.join(valid_timeframes)}"
                , symbol=instrument, details={'timeframe': timeframe,
                'valid_timeframes': valid_timeframes})
        try:
            if isinstance(start_time, str):
                start_time = datetime.fromisoformat(start_time.replace('Z',
                    '+00:00'))
            if isinstance(end_time, str):
                end_time = datetime.fromisoformat(end_time.replace('Z',
                    '+00:00'))
            if start_time >= end_time:
                raise MarketDataError(message=
                    'Start time must be before end time', symbol=instrument,
                    details={'start_time': start_time.isoformat(),
                    'end_time': end_time.isoformat()})
            if self.historical_data_service and hasattr(self.
                historical_data_service, 'get_historical_data'):
                try:
                    return (await self.historical_data_service.
                        get_historical_data(instrument=instrument,
                        start_time=start_time, end_time=end_time, timeframe
                        =timeframe))
                except Exception as e:
                    self.logger.error(
                        f'Error getting historical data from service: {str(e)}'
                        )
            return self._generate_synthetic_historical_data(instrument=
                instrument, start_time=start_time, end_time=end_time,
                timeframe=timeframe)
        except Exception as e:
            if isinstance(e, MarketDataError):
                raise
            raise MarketDataError(message=
                f'Failed to get historical data for {instrument}: {str(e)}',
                symbol=instrument, details={'error': str(e), 'timeframe':
                timeframe, 'start_time': start_time.isoformat() if
                isinstance(start_time, datetime) else start_time,
                'end_time': end_time.isoformat() if isinstance(end_time,
                datetime) else end_time})

    @with_exception_handling
    def _generate_synthetic_historical_data(self, instrument: str,
        start_time: datetime, end_time: datetime, timeframe: str='1m'
        ) ->pd.DataFrame:
        """
        Generate synthetic historical data for testing.

        Args:
            instrument: The instrument to generate data for
            start_time: Start time for the data
            end_time: End time for the data
            timeframe: Timeframe for the data

        Returns:
            DataFrame with synthetic historical data

        Raises:
            MarketDataError: If there's an error generating synthetic data
        """
        try:
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
                freq = '1H'
            timestamps = pd.date_range(start=start_time, end=end_time, freq
                =freq)
            np.random.seed(42)
            price = 1.0
            prices = [price]
            for _ in range(1, len(timestamps)):
                price_change = np.random.normal(0, 0.001)
                price = price * (1 + price_change)
                prices.append(price)
            data = []
            for i, timestamp in enumerate(timestamps):
                price = prices[i]
                high = price * (1 + abs(np.random.normal(0, 0.0005)))
                low = price * (1 - abs(np.random.normal(0, 0.0005)))
                if i > 0:
                    open_price = prices[i - 1]
                else:
                    open_price = price * (1 + np.random.normal(0, 0.0005))
                close = price
                volume = abs(np.random.normal(1000, 200))
                data.append({'timestamp': timestamp, 'open': open_price,
                    'high': high, 'low': low, 'close': close, 'volume':
                    volume, 'instrument': instrument})
            return pd.DataFrame(data)
        except Exception as e:
            raise MarketDataError(message=
                f'Failed to generate synthetic historical data for {instrument}: {str(e)}'
                , symbol=instrument, details={'error': str(e), 'timeframe':
                timeframe, 'start_time': start_time.isoformat(), 'end_time':
                end_time.isoformat()})

    @with_broker_api_resilience('get_volatility')
    @async_with_exception_handling
    async def get_volatility(self, instrument: str) ->float:
        """
        Get the current volatility for an instrument.

        Args:
            instrument: The instrument to get volatility for

        Returns:
            Current volatility (standard deviation of returns)

        Raises:
            MarketDataError: If there's an error calculating volatility
        """
        if not instrument:
            raise MarketDataError(message='Instrument cannot be empty',
                symbol=instrument)
        now = time.time()
        if (instrument in self.volatility_cache and instrument in self.
            last_volatility_update and now - self.last_volatility_update[
            instrument] < self.volatility_cache_expiry):
            return self.volatility_cache[instrument]
        try:
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(hours=24)
            df = await self.get_historical_data(instrument=instrument,
                start_time=start_time, end_time=end_time, timeframe='5m')
            if len(df) > 1:
                df['returns'] = df['close'].pct_change().dropna()
                volatility = df['returns'].std()
                self.volatility_cache[instrument] = volatility
                self.last_volatility_update[instrument] = now
                return volatility
            else:
                self.logger.warning(
                    f'Not enough data to calculate volatility for {instrument}'
                    )
                return 0.001
        except Exception as e:
            if isinstance(e, MarketDataError):
                raise
            raise MarketDataError(message=
                f'Failed to calculate volatility for {instrument}: {str(e)}',
                symbol=instrument, details={'error': str(e)})

    @with_broker_api_resilience('get_avg_daily_volume')
    @async_with_exception_handling
    async def get_avg_daily_volume(self, instrument: str) ->float:
        """
        Get the average daily volume for an instrument.

        Args:
            instrument: The instrument to get volume for

        Returns:
            Average daily volume

        Raises:
            MarketDataError: If there's an error calculating average daily volume
        """
        if not instrument:
            raise MarketDataError(message='Instrument cannot be empty',
                symbol=instrument)
        try:
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(days=7)
            df = await self.get_historical_data(instrument=instrument,
                start_time=start_time, end_time=end_time, timeframe='1h')
            if len(df) > 0:
                df['date'] = df['timestamp'].dt.date
                daily_volumes = df.groupby('date')['volume'].sum()
                avg_daily_volume = daily_volumes.mean()
                return avg_daily_volume
            else:
                self.logger.warning(
                    f'Not enough data to calculate average daily volume for {instrument}'
                    )
                return 1000000
        except Exception as e:
            if isinstance(e, MarketDataError):
                raise
            raise MarketDataError(message=
                f'Failed to calculate average daily volume for {instrument}: {str(e)}'
                , symbol=instrument, details={'error': str(e)})

    @with_market_data_resilience('get_market_regime')
    @async_with_exception_handling
    async def get_market_regime(self, instrument: str) ->str:
        """
        Get the current market regime for an instrument.

        Args:
            instrument: The instrument to get the market regime for

        Returns:
            Market regime (e.g., 'trending', 'ranging', 'volatile')

        Raises:
            MarketDataError: If there's an error detecting market regime
        """
        if not instrument:
            raise MarketDataError(message='Instrument cannot be empty',
                symbol=instrument)
        now = time.time()
        if (instrument in self.market_regime_cache and instrument in self.
            last_market_regime_update and now - self.
            last_market_regime_update[instrument] < self.
            market_regime_cache_expiry):
            return self.market_regime_cache[instrument]
        if self.market_regime_service and hasattr(self.
            market_regime_service, 'detect_regime'):
            try:
                regime = await self.market_regime_service.detect_regime(
                    instrument)
                self.market_regime_cache[instrument] = regime
                self.last_market_regime_update[instrument] = now
                return regime
            except Exception as e:
                self.logger.error(
                    f'Error detecting market regime from service: {str(e)}')
        try:
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(hours=24)
            df = await self.get_historical_data(instrument=instrument,
                start_time=start_time, end_time=end_time, timeframe='5m')
            if len(df) > 1:
                df['returns'] = df['close'].pct_change().dropna()
                volatility = df['returns'].std()
                trend_strength = abs(df['returns'].mean())
                if volatility > 0.002:
                    regime = 'volatile'
                elif trend_strength > 0.0001:
                    regime = 'trending'
                else:
                    regime = 'ranging'
                self.market_regime_cache[instrument] = regime
                self.last_market_regime_update[instrument] = now
                return regime
            else:
                self.logger.warning(
                    f'Not enough data to calculate market regime for {instrument}'
                    )
                return 'normal'
        except Exception as e:
            if isinstance(e, MarketDataError):
                raise
            raise MarketDataError(message=
                f'Failed to detect market regime for {instrument}: {str(e)}',
                symbol=instrument, details={'error': str(e)})

    @with_market_data_resilience('get_market_conditions')
    @async_with_exception_handling
    async def get_market_conditions(self, instrument: str) ->Dict[str, Any]:
        """
        Get comprehensive market conditions for an instrument.

        Args:
            instrument: The instrument to get conditions for

        Returns:
            Dictionary with market conditions

        Raises:
            MarketDataError: If there's an error getting market conditions
        """
        if not instrument:
            raise MarketDataError(message='Instrument cannot be empty',
                symbol=instrument)
        try:
            market_data = await self.get_market_data(instrument)
            volatility = await self.get_volatility(instrument)
            avg_daily_volume = await self.get_avg_daily_volume(instrument)
            market_regime = await self.get_market_regime(instrument)
            conditions = {'instrument': instrument, 'price': market_data.
                get('price'), 'bid': market_data.get('bid'), 'ask':
                market_data.get('ask'), 'spread': market_data.get('spread'),
                'volatility': volatility, 'avg_daily_volume':
                avg_daily_volume, 'market_regime': market_regime,
                'timestamp': datetime.utcnow().isoformat()}
            return conditions
        except Exception as e:
            if isinstance(e, MarketDataError):
                raise
            raise MarketDataError(message=
                f'Failed to get market conditions for {instrument}: {str(e)}',
                symbol=instrument, details={'error': str(e)})

    @with_broker_api_resilience('get_historical_volume')
    @async_with_exception_handling
    async def get_historical_volume(self, instrument: str, period: str='1d',
        lookback_days: int=20, hour_of_day: Optional[int]=None) ->List[List
        [float]]:
        """
        Get historical volume data for an instrument.

        Args:
            instrument: The instrument to get volume for
            period: Period for the data (e.g., '1d', '1h')
            lookback_days: Number of days to look back
            hour_of_day: Specific hour of the day to filter for

        Returns:
            List of volume data for each day

        Raises:
            MarketDataError: If there's an error getting historical volume
        """
        if not instrument:
            raise MarketDataError(message='Instrument cannot be empty',
                symbol=instrument)
        if lookback_days <= 0:
            raise MarketDataError(message=
                'Lookback days must be greater than 0', symbol=instrument,
                details={'lookback_days': lookback_days})
        if hour_of_day is not None and (hour_of_day < 0 or hour_of_day > 23):
            raise MarketDataError(message=
                'Hour of day must be between 0 and 23', symbol=instrument,
                details={'hour_of_day': hour_of_day})
        try:
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(days=lookback_days)
            df = await self.get_historical_data(instrument=instrument,
                start_time=start_time, end_time=end_time, timeframe='1h')
            if len(df) > 0:
                if hour_of_day is not None:
                    df = df[df['timestamp'].dt.hour == hour_of_day]
                df['date'] = df['timestamp'].dt.date
                volume_data = []
                for date, group in df.groupby('date'):
                    volume_data.append(group['volume'].tolist())
                return volume_data
            else:
                self.logger.warning(
                    f'Not enough data to get historical volume for {instrument}'
                    )
                return []
        except Exception as e:
            if isinstance(e, MarketDataError):
                raise
            raise MarketDataError(message=
                f'Failed to get historical volume for {instrument}: {str(e)}',
                symbol=instrument, details={'error': str(e), 'period':
                period, 'lookback_days': lookback_days, 'hour_of_day':
                hour_of_day})

    @with_broker_api_resilience('get_predicted_volume')
    @async_with_exception_handling
    async def get_predicted_volume(self, instrument: str, start_time:
        datetime, end_time: datetime, num_slices: int) ->List[float]:
        """
        Get predicted volume for an instrument over a time period.

        Args:
            instrument: The instrument to get volume for
            start_time: Start time for the prediction
            end_time: End time for the prediction
            num_slices: Number of slices to divide the period into

        Returns:
            List of predicted volumes for each slice

        Raises:
            MarketDataError: If there's an error predicting volume
        """
        if not instrument:
            raise MarketDataError(message='Instrument cannot be empty',
                symbol=instrument)
        if start_time >= end_time:
            raise MarketDataError(message=
                'Start time must be before end time', symbol=instrument,
                details={'start_time': start_time.isoformat(), 'end_time':
                end_time.isoformat()})
        if num_slices <= 0:
            raise MarketDataError(message=
                'Number of slices must be greater than 0', symbol=
                instrument, details={'num_slices': num_slices})
        try:
            historical_volume = await self.get_historical_volume(instrument
                =instrument, period='1d', lookback_days=20, hour_of_day=
                start_time.hour)
            if historical_volume:
                avg_profile = []
                for day_data in historical_volume:
                    slice_size = len(day_data) / num_slices
                    slices = []
                    for i in range(num_slices):
                        start_idx = int(i * slice_size)
                        end_idx = int((i + 1) * slice_size
                            ) if i < num_slices - 1 else len(day_data)
                        slice_sum = sum(day_data[start_idx:end_idx])
                        slices.append(slice_sum)
                    if not avg_profile:
                        avg_profile = slices
                    else:
                        avg_profile = [(a + b) for a, b in zip(avg_profile,
                            slices)]
                avg_profile = [(v / len(historical_volume)) for v in
                    avg_profile]
                return avg_profile
            else:
                self.logger.warning(
                    f'Not enough historical volume data for {instrument} to make prediction'
                    )
                return [1.0] * num_slices
        except Exception as e:
            if isinstance(e, MarketDataError):
                raise
            raise MarketDataError(message=
                f'Failed to predict volume for {instrument}: {str(e)}',
                symbol=instrument, details={'error': str(e), 'start_time':
                start_time.isoformat(), 'end_time': end_time.isoformat(),
                'num_slices': num_slices})

    @with_broker_api_resilience('get_realtime_volume')
    @async_with_exception_handling
    async def get_realtime_volume(self, instrument: str, lookback_minutes:
        int=60) ->List[float]:
        """
        Get real-time volume data for an instrument.

        Args:
            instrument: The instrument to get volume for
            lookback_minutes: Number of minutes to look back

        Returns:
            List of volume data for the lookback period

        Raises:
            MarketDataError: If there's an error getting real-time volume
        """
        if not instrument:
            raise MarketDataError(message='Instrument cannot be empty',
                symbol=instrument)
        if lookback_minutes <= 0:
            raise MarketDataError(message=
                'Lookback minutes must be greater than 0', symbol=
                instrument, details={'lookback_minutes': lookback_minutes})
        try:
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(minutes=lookback_minutes)
            df = await self.get_historical_data(instrument=instrument,
                start_time=start_time, end_time=end_time, timeframe='1m')
            if len(df) > 0:
                return df['volume'].tolist()
            else:
                self.logger.warning(
                    f'Not enough data to get real-time volume for {instrument}'
                    )
                return []
        except Exception as e:
            if isinstance(e, MarketDataError):
                raise
            raise MarketDataError(message=
                f'Failed to get real-time volume for {instrument}: {str(e)}',
                symbol=instrument, details={'error': str(e),
                'lookback_minutes': lookback_minutes})
