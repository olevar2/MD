"""
Example of using the Manipulation Detection client

This module demonstrates how to use the standardized Manipulation Detection client
to interact with the Analysis Engine Service API.
"""
import asyncio
from typing import Dict, List, Any
from datetime import datetime
from analysis_engine.clients.standardized import get_client_factory
from analysis_engine.monitoring.structured_logging import get_structured_logger
logger = get_structured_logger(__name__)
EXAMPLE_OHLCV = [{'timestamp': '2025-04-01T00:00:00', 'open': 1.0765,
    'high': 1.078, 'low': 1.076, 'close': 1.0775, 'volume': 1000}, {
    'timestamp': '2025-04-01T01:00:00', 'open': 1.0775, 'high': 1.079,
    'low': 1.077, 'close': 1.0785, 'volume': 1200}, {'timestamp':
    '2025-04-01T02:00:00', 'open': 1.0785, 'high': 1.0795, 'low': 1.078,
    'close': 1.079, 'volume': 1100}, {'timestamp': '2025-04-01T03:00:00',
    'open': 1.079, 'high': 1.08, 'low': 1.0785, 'close': 1.0795, 'volume': 
    1300}, {'timestamp': '2025-04-01T04:00:00', 'open': 1.0795, 'high': 
    1.081, 'low': 1.079, 'close': 1.0805, 'volume': 1400}, {'timestamp':
    '2025-04-01T05:00:00', 'open': 1.0805, 'high': 1.082, 'low': 1.08,
    'close': 1.0815, 'volume': 1500}, {'timestamp': '2025-04-01T06:00:00',
    'open': 1.0815, 'high': 1.083, 'low': 1.081, 'close': 1.0825, 'volume':
    1600}, {'timestamp': '2025-04-01T07:00:00', 'open': 1.0825, 'high': 
    1.084, 'low': 1.082, 'close': 1.0835, 'volume': 1700}, {'timestamp':
    '2025-04-01T08:00:00', 'open': 1.0835, 'high': 1.085, 'low': 1.083,
    'close': 1.0845, 'volume': 1800}, {'timestamp': '2025-04-01T09:00:00',
    'open': 1.0845, 'high': 1.086, 'low': 1.084, 'close': 1.0855, 'volume':
    1900}, {'timestamp': '2025-04-01T10:00:00', 'open': 1.0855, 'high': 
    1.087, 'low': 1.085, 'close': 1.0865, 'volume': 2000}, {'timestamp':
    '2025-04-01T11:00:00', 'open': 1.0865, 'high': 1.088, 'low': 1.086,
    'close': 1.0875, 'volume': 2100}, {'timestamp': '2025-04-01T12:00:00',
    'open': 1.0875, 'high': 1.089, 'low': 1.087, 'close': 1.0885, 'volume':
    2200}, {'timestamp': '2025-04-01T13:00:00', 'open': 1.0885, 'high': 
    1.09, 'low': 1.088, 'close': 1.0895, 'volume': 2300}, {'timestamp':
    '2025-04-01T14:00:00', 'open': 1.0895, 'high': 1.091, 'low': 1.089,
    'close': 1.0905, 'volume': 2400}, {'timestamp': '2025-04-01T15:00:00',
    'open': 1.0905, 'high': 1.092, 'low': 1.09, 'close': 1.0915, 'volume': 
    2500}, {'timestamp': '2025-04-01T16:00:00', 'open': 1.0915, 'high': 
    1.093, 'low': 1.091, 'close': 1.0925, 'volume': 2600}, {'timestamp':
    '2025-04-01T17:00:00', 'open': 1.0925, 'high': 1.094, 'low': 1.092,
    'close': 1.0935, 'volume': 2700}, {'timestamp': '2025-04-01T18:00:00',
    'open': 1.0935, 'high': 1.095, 'low': 1.093, 'close': 1.0945, 'volume':
    2800}, {'timestamp': '2025-04-01T19:00:00', 'open': 1.0945, 'high': 
    1.096, 'low': 1.094, 'close': 1.0955, 'volume': 2900}, {'timestamp':
    '2025-04-01T20:00:00', 'open': 1.0955, 'high': 1.097, 'low': 1.095,
    'close': 1.0965, 'volume': 3000}, {'timestamp': '2025-04-01T21:00:00',
    'open': 1.0965, 'high': 1.098, 'low': 1.096, 'close': 1.0975, 'volume':
    3100}, {'timestamp': '2025-04-01T22:00:00', 'open': 1.0975, 'high': 
    1.099, 'low': 1.097, 'close': 1.0985, 'volume': 3200}, {'timestamp':
    '2025-04-01T23:00:00', 'open': 1.0985, 'high': 1.1, 'low': 1.098,
    'close': 1.0995, 'volume': 3300}]
EXAMPLE_METADATA = {'symbol': 'EUR/USD', 'timeframe': '1h'}
for i in range(100):
    last_data = EXAMPLE_OHLCV[-1]
    new_data = {'timestamp': f'2025-04-02T{i % 24:02d}:00:00', 'open':
        last_data['close'], 'high': last_data['close'] + 0.0015, 'low': 
        last_data['close'] - 0.001, 'close': last_data['close'] + 0.0005,
        'volume': last_data['volume'] + 100}
    EXAMPLE_OHLCV.append(new_data)


from analysis_engine.core.exceptions_bridge import (
    with_exception_handling,
    async_with_exception_handling,
    ForexTradingPlatformError,
    ServiceError,
    DataError,
    ValidationError
)

@async_with_exception_handling
async def detect_manipulation_patterns_example():
    """
    Example of detecting manipulation patterns using the standardized client.
    """
    factory = get_client_factory()
    client = factory.get_manipulation_detection_client()
    sensitivity = 1.2
    include_protection = True
    try:
        result = await client.detect_manipulation_patterns(ohlcv=
            EXAMPLE_OHLCV, metadata=EXAMPLE_METADATA, sensitivity=
            sensitivity, include_protection=include_protection)
        logger.info(f'Detected manipulation patterns: {result}')
        return result
    except Exception as e:
        logger.error(f'Error detecting manipulation patterns: {str(e)}')
        raise


@async_with_exception_handling
async def detect_stop_hunting_example():
    """
    Example of detecting stop hunting patterns using the standardized client.
    """
    factory = get_client_factory()
    client = factory.get_manipulation_detection_client()
    lookback = 30
    recovery_threshold = 0.5
    try:
        result = await client.detect_stop_hunting(ohlcv=EXAMPLE_OHLCV,
            metadata=EXAMPLE_METADATA, lookback=lookback,
            recovery_threshold=recovery_threshold)
        logger.info(f'Detected stop hunting patterns: {result}')
        return result
    except Exception as e:
        logger.error(f'Error detecting stop hunting patterns: {str(e)}')
        raise


@async_with_exception_handling
async def detect_fake_breakouts_example():
    """
    Example of detecting fake breakout patterns using the standardized client.
    """
    factory = get_client_factory()
    client = factory.get_manipulation_detection_client()
    threshold = 0.7
    try:
        result = await client.detect_fake_breakouts(ohlcv=EXAMPLE_OHLCV,
            metadata=EXAMPLE_METADATA, threshold=threshold)
        logger.info(f'Detected fake breakout patterns: {result}')
        return result
    except Exception as e:
        logger.error(f'Error detecting fake breakout patterns: {str(e)}')
        raise


@async_with_exception_handling
async def detect_volume_anomalies_example():
    """
    Example of detecting volume anomalies using the standardized client.
    """
    factory = get_client_factory()
    client = factory.get_manipulation_detection_client()
    z_threshold = 2.0
    try:
        result = await client.detect_volume_anomalies(ohlcv=EXAMPLE_OHLCV,
            metadata=EXAMPLE_METADATA, z_threshold=z_threshold)
        logger.info(f'Detected volume anomalies: {result}')
        return result
    except Exception as e:
        logger.error(f'Error detecting volume anomalies: {str(e)}')
        raise


async def main():
    """
    Run the examples.
    """
    logger.info('Running Manipulation Detection client examples')
    await detect_manipulation_patterns_example()
    await detect_stop_hunting_example()
    await detect_fake_breakouts_example()
    await detect_volume_anomalies_example()
    logger.info('Completed Manipulation Detection client examples')


if __name__ == '__main__':
    asyncio.run(main())
