"""
Data providers for the backtesting framework.
"""
import logging
import pandas as pd
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, Any, List, AsyncGenerator, Tuple, Optional
from core_foundations.models.trading import MarketData
from analysis_engine.clients.service_client_factory import ServiceClientFactory
logger = logging.getLogger(__name__)


from analysis_engine.core.exceptions_bridge import (
    with_exception_handling,
    async_with_exception_handling,
    ForexTradingPlatformError,
    ServiceError,
    DataError,
    ValidationError
)

class BaseDataProvider(ABC):
    """Abstract base class for data providers."""

    @abstractmethod
    async def stream_data(self, start_date: datetime, end_date: datetime,
        symbols: List[str], timeframe: str='1H') ->AsyncGenerator[Tuple[
        datetime, Dict[str, MarketData]], None]:
        """
        Asynchronously streams market data for the given symbols and date range.

        Yields:
            A tuple containing the timestamp and a dictionary mapping symbol
            to its MarketData for that timestamp.
        """
        pass
        if False:
            yield datetime.now(), {}


class MarketDataServiceProvider(BaseDataProvider):
    """
    Provides market data from the Market Data Service using the adapter pattern.
    """

    def __init__(self, config: Dict[str, Any]=None):
    """
      init  .
    
    Args:
        config: Description of config
        Any]: Description of Any]
    
    """

        self.config = config or {}
        factory = ServiceClientFactory()
        self.market_data_provider = factory.create_market_data_provider()
        logger.info('MarketDataServiceProvider initialized.')

    @async_with_exception_handling
    async def stream_data(self, start_date: datetime, end_date: datetime,
        symbols: List[str], timeframe: str='1H') ->AsyncGenerator[Tuple[
        datetime, Dict[str, MarketData]], None]:
        """
        Streams historical market data from the Market Data Service.

        Args:
            start_date: Start date for data retrieval
            end_date: End date for data retrieval
            symbols: List of symbols to retrieve data for
            timeframe: Timeframe for the data

        Yields:
            A tuple containing the timestamp and a dictionary mapping symbol
            to its MarketData for that timestamp.
        """
        logger.info(
            f'Streaming market data from {start_date} to {end_date} for {symbols}'
            )
        try:
            for symbol in symbols:
                df = await self.market_data_provider.get_historical_data(symbol
                    =symbol, timeframe=timeframe, start_time=start_date,
                    end_time=end_date)
                for timestamp, row in df.iterrows():
                    market_data = MarketData(symbol=symbol, timestamp=
                        timestamp, open=row.get('open', 0.0), high=row.get(
                        'high', 0.0), low=row.get('low', 0.0), close=row.
                        get('close', 0.0), volume=row.get('volume', 0.0))
                    yield timestamp, {symbol: market_data}
        except Exception as e:
            logger.error(f'Error streaming market data: {str(e)}', exc_info
                =True)
        logger.info('Finished streaming market data.')


class GeneratedDataProvider(BaseDataProvider):
    """
    Generates synthetic market data using simulators.
    """

    def __init__(self, config: Dict[str, Any]=None):
    """
      init  .
    
    Args:
        config: Description of config
        Any]: Description of Any]
    
    """

        self.config = config or {}
        logger.info('GeneratedDataProvider initialized.')

    @async_with_exception_handling
    async def stream_data(self, start_date: datetime, end_date: datetime,
        symbols: List[str], timeframe: str='1H') ->AsyncGenerator[Tuple[
        datetime, Dict[str, MarketData]], None]:
        """
        Streams generated market data.

        This implementation generates synthetic data without external dependencies.
        """
        logger.info(
            f'Generating synthetic data from {start_date} to {end_date} for {symbols}'
            )
        try:
            import pandas as pd
            from datetime import timedelta
            if timeframe.endswith('m'):
                freq = f'{timeframe[:-1]}min'
            elif timeframe.endswith('h'):
                freq = f'{timeframe[:-1]}H'
            elif timeframe.endswith('d'):
                freq = f'{timeframe[:-1]}D'
            else:
                freq = '1H'
            timestamps = pd.date_range(start=start_date, end=end_date, freq
                =freq)
            for timestamp in timestamps:
                market_data_batch = {}
                for symbol in symbols:
                    import numpy as np
                    import random
                    seed = int(timestamp.timestamp()) + hash(symbol) % 10000
                    np.random.seed(seed)
                    random.seed(seed)
                    base_price = 1.0
                    if symbol.startswith('EUR'):
                        base_price = 1.1
                    elif symbol.startswith('GBP'):
                        base_price = 1.3
                    elif symbol.startswith('JPY'):
                        base_price = 110.0
                    price_volatility = 0.002
                    close_price = base_price * (1 + np.random.normal(0,
                        price_volatility))
                    high_price = close_price * (1 + random.uniform(0, 
                        price_volatility * 2))
                    low_price = close_price * (1 - random.uniform(0, 
                        price_volatility * 2))
                    open_price = low_price + random.uniform(0, high_price -
                        low_price)
                    volume = np.random.poisson(1000)
                    market_data_batch[symbol] = MarketData(symbol=symbol,
                        timestamp=timestamp, open=open_price, high=
                        high_price, low=low_price, close=close_price,
                        volume=volume)
                yield timestamp, market_data_batch
        except Exception as e:
            logger.error(f'Error during generated data streaming: {e}',
                exc_info=True)
        logger.info('Finished generating data stream.')


class HistoricalDatabaseProvider(BaseDataProvider):
    """
    Provides historical market data fetched from a database.
    (Requires a database connection and schema)
    """

    def __init__(self, db_connection_params: Dict[str, Any], config: Dict[
        str, Any]):
    """
      init  .
    
    Args:
        db_connection_params: Description of db_connection_params
        Any]: Description of Any]
        config: Description of config
        Any]: Description of Any]
    
    """

        self.db_params = db_connection_params
        self.config = config
        self.db_client = None
        logger.info(
            'HistoricalDatabaseProvider initialized (Placeholder DB Client).')

    @async_with_exception_handling
    async def stream_data(self, start_date: datetime, end_date: datetime,
        symbols: List[str], timeframe: str='1H') ->AsyncGenerator[Tuple[
        datetime, Dict[str, MarketData]], None]:
        """Streams historical market data from the database."""
        logger.info(
            f'Streaming historical data from DB: {start_date} to {end_date} for {symbols}'
            )
        try:
            timestamps = pd.date_range(start_date, end_date, freq=timeframe)
            current_batch_ts = None
            current_batch_data = {}
            for ts in timestamps:
                batch_for_ts = {}
                for symbol in symbols:
                    simulated_close = 1.1 + ts.hour * 0.001
                    batch_for_ts[symbol] = MarketData(symbol=symbol,
                        timestamp=ts, open=simulated_close - 0.0005, high=
                        simulated_close + 0.001, low=simulated_close - 
                        0.001, close=simulated_close, volume=1000)
                if batch_for_ts:
                    yield ts, batch_for_ts
        except Exception as e:
            logger.error(f'Error streaming historical data: {e}', exc_info=True
                )
        logger.info('Finished streaming historical data.')
        return
