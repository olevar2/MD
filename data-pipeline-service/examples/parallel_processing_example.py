"""
Example usage of the parallel processing framework.

This module demonstrates how to use the parallel processing framework
for various data pipeline tasks, including multi-instrument processing,
multi-timeframe processing, and batch feature engineering.
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any

import pandas as pd
import numpy as np

from data_pipeline_service.parallel import (
    BatchFeatureProcessor,
    FeatureSpec,
    MultiInstrumentProcessor,
    MultiTimeframeProcessor,
    ParallelizationMethod,
    TaskPriority,
    TimeframeHierarchy,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


# Example functions for processing

def process_instrument(instrument: str) -> Dict[str, Any]:
    """
    Example function to process a single instrument.
    
    Args:
        instrument: Instrument symbol
        
    Returns:
        Processing result
    """
    logger.info(f"Processing instrument: {instrument}")
    
    # Simulate processing time
    time.sleep(0.5)
    
    # Generate some random data
    return {
        "instrument": instrument,
        "price": round(np.random.uniform(1.0, 2.0), 4),
        "volume": int(np.random.uniform(1000, 10000)),
        "timestamp": datetime.now().isoformat()
    }


def process_timeframe(timeframe: str) -> pd.DataFrame:
    """
    Example function to process a single timeframe.
    
    Args:
        timeframe: Timeframe string
        
    Returns:
        DataFrame with processed data
    """
    logger.info(f"Processing timeframe: {timeframe}")
    
    # Simulate processing time based on timeframe size
    # Larger timeframes take longer to process
    minutes = TimeframeHierarchy.get_minutes(timeframe)
    time.sleep(max(0.1, minutes / 1000))
    
    # Generate some random data
    periods = 100
    index = pd.date_range(end=datetime.now(), periods=periods, freq=timeframe)
    
    return pd.DataFrame({
        'open': np.random.normal(100, 5, periods),
        'high': np.random.normal(102, 5, periods),
        'low': np.random.normal(98, 5, periods),
        'close': np.random.normal(101, 5, periods),
        'volume': np.random.randint(1000, 10000, periods)
    }, index=index)


def calculate_feature(data: pd.DataFrame, feature: FeatureSpec) -> pd.Series:
    """
    Example function to calculate a feature.
    
    Args:
        data: Input DataFrame
        feature: Feature specification
        
    Returns:
        Calculated feature as a Series
    """
    logger.info(f"Calculating feature: {feature}")
    
    # Simulate processing time
    time.sleep(0.2)
    
    # Calculate feature based on name
    if feature.name == 'sma':
        window = feature.params.get('window', 20)
        return data['close'].rolling(window=window).mean()
    elif feature.name == 'rsi':
        window = feature.params.get('window', 14)
        delta = data['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=window).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    elif feature.name == 'bollinger_bands':
        window = feature.params.get('window', 20)
        std_dev = feature.params.get('std_dev', 2)
        sma = data['close'].rolling(window=window).mean()
        std = data['close'].rolling(window=window).std()
        return sma + std_dev * std
    else:
        # Default to a simple moving average
        return data['close'].rolling(window=20).mean()


async def example_multi_instrument_processing():
    """Example of multi-instrument processing."""
    logger.info("Running multi-instrument processing example")
    
    # Create processor
    processor = MultiInstrumentProcessor()
    
    # Define instruments
    instruments = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD', 'USDCHF', 'NZDUSD']
    
    # Process instruments in parallel
    results = await processor.process_instruments(
        instruments=instruments,
        process_func=process_instrument,
        priority=TaskPriority.MEDIUM,
        parallelization_method=ParallelizationMethod.THREAD,
        batch_size=3  # Process in batches of 3
    )
    
    # Print results
    logger.info(f"Processed {len(results)} instruments")
    for instrument, result in results.items():
        logger.info(f"  {instrument}: price={result['price']}, volume={result['volume']}")


async def example_multi_timeframe_processing():
    """Example of multi-timeframe processing."""
    logger.info("Running multi-timeframe processing example")
    
    # Create processor
    processor = MultiTimeframeProcessor()
    
    # Define timeframes
    timeframes = ['1m', '5m', '15m', '30m', '1h', '4h', '1d']
    
    # Process timeframes in parallel
    results = await processor.process_timeframes(
        timeframes=timeframes,
        process_func=process_timeframe,
        priority=TaskPriority.MEDIUM,
        parallelization_method=ParallelizationMethod.THREAD,
        respect_hierarchy=True  # Process larger timeframes first
    )
    
    # Print results
    logger.info(f"Processed {len(results)} timeframes")
    for timeframe, df in results.items():
        logger.info(f"  {timeframe}: shape={df.shape}, last_close={df['close'].iloc[-1]:.2f}")


async def example_batch_feature_processing():
    """Example of batch feature processing."""
    logger.info("Running batch feature processing example")
    
    # Create processor
    processor = BatchFeatureProcessor()
    
    # Create sample data
    data = pd.DataFrame({
        'open': np.random.normal(100, 5, 100),
        'high': np.random.normal(102, 5, 100),
        'low': np.random.normal(98, 5, 100),
        'close': np.random.normal(101, 5, 100),
        'volume': np.random.randint(1000, 10000, 100)
    })
    
    # Define features
    features = [
        FeatureSpec(
            name='sma',
            params={'window': 20},
            priority=TaskPriority.MEDIUM
        ),
        FeatureSpec(
            name='rsi',
            params={'window': 14},
            priority=TaskPriority.HIGH
        ),
        FeatureSpec(
            name='bollinger_bands',
            params={'window': 20, 'std_dev': 2},
            priority=TaskPriority.LOW
        )
    ]
    
    # Calculate features in parallel
    results = await processor.calculate_features(
        data=data,
        features=features,
        calculate_func=calculate_feature
    )
    
    # Print results
    logger.info(f"Calculated {len(results)} features")
    for feature_name, series in results.items():
        logger.info(f"  {feature_name}: last_value={series.iloc[-1]:.2f}")


async def main():
    """Run all examples."""
    await example_multi_instrument_processing()
    print()
    
    await example_multi_timeframe_processing()
    print()
    
    await example_batch_feature_processing()


if __name__ == "__main__":
    asyncio.run(main())
