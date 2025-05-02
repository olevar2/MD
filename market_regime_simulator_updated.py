import numpy as np
import pandas as pd
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union, AsyncGenerator
from datetime import datetime, timedelta
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

# Keep the rest of the imports and class definitions as they are...

# Then update the generate_data_stream method to use a thread executor:

async def generate_data_stream(
    self,
    start_date: datetime,
    end_date: datetime,
    symbols: List[str],
    timeframe: str = '1H'
) -> AsyncGenerator[Tuple[datetime, Dict[str, Dict]], None]:
    """
    Asynchronously streams generated market data for the given symbols and date range.

    This acts as a wrapper around the simulate method, adapting it to the
    streaming interface required by data providers.

    Args:
        start_date: The starting timestamp for data generation.
        end_date: The ending timestamp for data generation.
        symbols: List of symbols to generate data for (Note: current generator is single-symbol).
        timeframe: Pandas frequency string for the timeframe (e.g., '1H', '15T').

    Yields:
        A tuple containing the timestamp and a dictionary mapping symbol
        to its OHLCV data dict for that timestamp.
    """
    logger.info(f"Starting data stream generation from {start_date} to {end_date} for {symbols} ({timeframe})")

    try:
        timeframe_delta = pd.to_timedelta(timeframe)
        timeframe_minutes = int(timeframe_delta.total_seconds() / 60)
    except ValueError:
        logger.error(f"Invalid timeframe string: {timeframe}. Using default 60 minutes.")
        timeframe_minutes = 60
        timeframe_delta = timedelta(minutes=60)

    # Calculate duration in terms of number of candles
    duration = int((end_date - start_date) / timeframe_delta)
    if duration <= 0:
        logger.warning("End date is not after start date. No data will be generated.")
        return

    # --- Call the existing simulate method in a thread executor ---
    # Note: The current simulate/generator seems single-symbol.
    # We'll generate data for the first symbol and yield it for all requested symbols.
    # A multi-symbol generator would be needed for true multi-symbol simulation.
    target_symbol = symbols[0] if symbols else "SYNTHETIC_EURUSD"
    logger.warning(f"MarketRegimeSimulator currently generates data for one symbol ({target_symbol}) and replicates it for all requested symbols: {symbols}")

    # Run the simulation in a thread executor to avoid blocking the event loop
    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor() as executor:
        df_generated, regime_changes = await loop.run_in_executor(
            executor,
            lambda: self.simulate(
                duration=duration,
                timeframe_minutes=timeframe_minutes,
                start_time=start_date,
                # Pass other relevant config from self.config if needed
            )
        )

    if df_generated.empty:
        logger.warning("Simulation generated an empty DataFrame.")
        return

    # --- Stream the generated DataFrame ---
    for timestamp, row in df_generated.iterrows():
        candle_data = row.to_dict()  # Contains open, high, low, close, volume, return, spread_bps, regime
        market_data_batch = {}
        for symbol in symbols:
            # Create a copy for each symbol
            # In a real multi-symbol generator, data would differ per symbol
            market_data_batch[symbol] = candle_data.copy()

        yield timestamp, market_data_batch
        # Add a small delay to avoid blocking the event loop completely
        await asyncio.sleep(0)

    logger.info(f"Finished data stream generation for {symbols}")
