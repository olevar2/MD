"""
Data providers for the backtesting framework.
"""
import logging
import pandas as pd
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, Any, List, AsyncGenerator, Tuple, Optional

from core_foundations.models.trading import MarketData
# Use adapter pattern for service dependencies
from analysis_engine.clients.service_client_factory import ServiceClientFactory

logger = logging.getLogger(__name__)

class BaseDataProvider(ABC):
    """Abstract base class for data providers."""

    @abstractmethod
    async def stream_data(
        self,
        start_date: datetime,
        end_date: datetime,
        symbols: List[str],
        timeframe: str = '1H' # Example: Default timeframe
    ) -> AsyncGenerator[Tuple[datetime, Dict[str, MarketData]], None]:
        """
        Asynchronously streams market data for the given symbols and date range.

        Yields:
            A tuple containing the timestamp and a dictionary mapping symbol
            to its MarketData for that timestamp.
        """
        pass
        # Ensure StopAsyncIteration is raised when done
        if False: # Dummy condition to make this a generator
            yield datetime.now(), {}


class MarketDataServiceProvider(BaseDataProvider):
    """
    Provides market data from the Market Data Service using the adapter pattern.
    """
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        # Create the market data provider using the adapter factory
        factory = ServiceClientFactory()
        self.market_data_provider = factory.create_market_data_provider()
        logger.info("MarketDataServiceProvider initialized.")

    async def stream_data(
        self,
        start_date: datetime,
        end_date: datetime,
        symbols: List[str],
        timeframe: str = '1H'
    ) -> AsyncGenerator[Tuple[datetime, Dict[str, MarketData]], None]:
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
        logger.info(f"Streaming market data from {start_date} to {end_date} for {symbols}")

        try:
            # Process each symbol
            for symbol in symbols:
                # Get historical data for the symbol
                df = await self.market_data_provider.get_historical_data(
                    symbol=symbol,
                    timeframe=timeframe,
                    start_time=start_date,
                    end_time=end_date
                )

                # Process each timestamp
                for timestamp, row in df.iterrows():
                    # Create MarketData object
                    market_data = MarketData(
                        symbol=symbol,
                        timestamp=timestamp,
                        open=row.get('open', 0.0),
                        high=row.get('high', 0.0),
                        low=row.get('low', 0.0),
                        close=row.get('close', 0.0),
                        volume=row.get('volume', 0.0)
                    )

                    # Yield the data
                    yield timestamp, {symbol: market_data}

        except Exception as e:
            logger.error(f"Error streaming market data: {str(e)}", exc_info=True)

        logger.info("Finished streaming market data.")


class GeneratedDataProvider(BaseDataProvider):
    """
    Generates synthetic market data using simulators.
    """
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        logger.info("GeneratedDataProvider initialized.")

    async def stream_data(
        self,
        start_date: datetime,
        end_date: datetime,
        symbols: List[str],
        timeframe: str = '1H'
    ) -> AsyncGenerator[Tuple[datetime, Dict[str, MarketData]], None]:
        """
        Streams generated market data.

        This implementation generates synthetic data without external dependencies.
        """
        logger.info(f"Generating synthetic data from {start_date} to {end_date} for {symbols}")

        try:
            # Generate timestamps based on timeframe
            import pandas as pd
            from datetime import timedelta

            # Parse timeframe to determine frequency
            if timeframe.endswith('m'):
                freq = f"{timeframe[:-1]}min"
            elif timeframe.endswith('h'):
                freq = f"{timeframe[:-1]}H"
            elif timeframe.endswith('d'):
                freq = f"{timeframe[:-1]}D"
            else:
                freq = "1H"  # Default to 1 hour

            # Generate timestamps
            timestamps = pd.date_range(start=start_date, end=end_date, freq=freq)

            # Generate synthetic data for each timestamp
            for timestamp in timestamps:
                market_data_batch = {}

                for symbol in symbols:
                    # Generate synthetic price data
                    import numpy as np
                    import random

                    # Use timestamp and symbol to seed the random generator for consistency
                    seed = int(timestamp.timestamp()) + hash(symbol) % 10000
                    np.random.seed(seed)
                    random.seed(seed)

                    # Base price depends on the symbol
                    base_price = 1.0
                    if symbol.startswith("EUR"):
                        base_price = 1.1
                    elif symbol.startswith("GBP"):
                        base_price = 1.3
                    elif symbol.startswith("JPY"):
                        base_price = 110.0

                    # Add some randomness
                    price_volatility = 0.002  # 0.2% volatility
                    close_price = base_price * (1 + np.random.normal(0, price_volatility))

                    # Generate OHLC based on close price
                    high_price = close_price * (1 + random.uniform(0, price_volatility * 2))
                    low_price = close_price * (1 - random.uniform(0, price_volatility * 2))
                    open_price = low_price + random.uniform(0, high_price - low_price)

                    # Generate volume
                    volume = np.random.poisson(1000)

                    # Create MarketData object
                    market_data_batch[symbol] = MarketData(
                        symbol=symbol,
                        timestamp=timestamp,
                        open=open_price,
                        high=high_price,
                        low=low_price,
                        close=close_price,
                        volume=volume
                    )

                # Yield the batch
                yield timestamp, market_data_batch

        except Exception as e:
            logger.error(f"Error during generated data streaming: {e}", exc_info=True)

        logger.info("Finished generating data stream.")


class HistoricalDatabaseProvider(BaseDataProvider):
    """
    Provides historical market data fetched from a database.
    (Requires a database connection and schema)
    """
    def __init__(self, db_connection_params: Dict[str, Any], config: Dict[str, Any]):
        self.db_params = db_connection_params
        self.config = config
        # TODO: Initialize database connection pool or client
        self.db_client = None # Placeholder
        logger.info("HistoricalDatabaseProvider initialized (Placeholder DB Client).")

    async def stream_data(
        self,
        start_date: datetime,
        end_date: datetime,
        symbols: List[str],
        timeframe: str = '1H'
    ) -> AsyncGenerator[Tuple[datetime, Dict[str, MarketData]], None]:
        """Streams historical market data from the database."""
        logger.info(f"Streaming historical data from DB: {start_date} to {end_date} for {symbols}")
        # Placeholder implementation
        # 1. Connect to DB (if not already connected)
        # 2. Construct query based on dates, symbols, timeframe
        # 3. Fetch data in chunks or use a cursor
        # 4. Process rows and yield MarketData objects

        # Example structure (replace with actual DB interaction)
        try:
            # Assume query fetches data ordered by timestamp
            # Example: SELECT timestamp, symbol, open, high, low, close, volume FROM market_data ...
            # cursor = await self.db_client.fetch_cursor(...) # Fictional DB client method

            # Simulate fetching data in chunks
            timestamps = pd.date_range(start_date, end_date, freq=timeframe) # Example timestamps
            current_batch_ts = None
            current_batch_data = {}

            for ts in timestamps:
                # Simulate fetching data for this timestamp for all symbols
                # In reality, you'd process rows from the DB query result
                batch_for_ts = {}
                for symbol in symbols:
                    # Simulate finding data for this symbol/ts
                    # Replace with actual data retrieval and MarketData creation
                    simulated_close = 1.1000 + (ts.hour * 0.001) # Dummy data
                    batch_for_ts[symbol] = MarketData(
                        symbol=symbol,
                        timestamp=ts,
                        open=simulated_close - 0.0005,
                        high=simulated_close + 0.0010,
                        low=simulated_close - 0.0010,
                        close=simulated_close,
                        volume=1000
                    )

                # Yield data grouped by timestamp
                if batch_for_ts:
                    yield ts, batch_for_ts

            # Simulate end of stream

        except Exception as e:
            logger.error(f"Error streaming historical data: {e}", exc_info=True)
            # Handle DB errors, connection issues etc.

        logger.info("Finished streaming historical data.")
        # Ensure generator stops
        return

# TODO: Add other providers like CSVDataProvider, LiveReplayProvider etc. as needed
