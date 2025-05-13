"""
Example of using the Market Regime client

This module demonstrates how to use the standardized Market Regime client
to interact with the Analysis Engine Service API.
"""
import asyncio
import pandas as pd
from typing import Dict, List, Any
from datetime import datetime, timedelta
from analysis_engine.clients.standardized import get_client_factory
from analysis_engine.monitoring.structured_logging import get_structured_logger
logger = get_structured_logger(__name__)


from analysis_engine.core.exceptions_bridge import (
    with_exception_handling,
    async_with_exception_handling,
    ForexTradingPlatformError,
    ServiceError,
    DataError,
    ValidationError
)

@async_with_exception_handling
async def detect_market_regime_example():
    """
    Example of detecting market regime using the standardized client.
    """
    factory = get_client_factory()
    client = factory.get_market_regime_client()
    symbol = 'EURUSD'
    timeframe = '1h'
    ohlc_data = [{'timestamp': '2025-04-01T00:00:00', 'open': 1.0765,
        'high': 1.078, 'low': 1.076, 'close': 1.0775, 'volume': 1000}, {
        'timestamp': '2025-04-01T01:00:00', 'open': 1.0775, 'high': 1.079,
        'low': 1.077, 'close': 1.0785, 'volume': 1200}, {'timestamp':
        '2025-04-01T02:00:00', 'open': 1.0785, 'high': 1.0795, 'low': 1.078,
        'close': 1.079, 'volume': 1100}, {'timestamp':
        '2025-04-01T03:00:00', 'open': 1.079, 'high': 1.08, 'low': 1.0785,
        'close': 1.0795, 'volume': 1300}, {'timestamp':
        '2025-04-01T04:00:00', 'open': 1.0795, 'high': 1.0805, 'low': 1.079,
        'close': 1.08, 'volume': 1200}, {'timestamp': '2025-04-01T05:00:00',
        'open': 1.08, 'high': 1.081, 'low': 1.0795, 'close': 1.0805,
        'volume': 1100}, {'timestamp': '2025-04-01T06:00:00', 'open': 
        1.0805, 'high': 1.0815, 'low': 1.08, 'close': 1.081, 'volume': 1000
        }, {'timestamp': '2025-04-01T07:00:00', 'open': 1.081, 'high': 
        1.082, 'low': 1.0805, 'close': 1.0815, 'volume': 1200}, {
        'timestamp': '2025-04-01T08:00:00', 'open': 1.0815, 'high': 1.0825,
        'low': 1.081, 'close': 1.082, 'volume': 1300}, {'timestamp':
        '2025-04-01T09:00:00', 'open': 1.082, 'high': 1.083, 'low': 1.0815,
        'close': 1.0825, 'volume': 1400}, {'timestamp':
        '2025-04-01T10:00:00', 'open': 1.0825, 'high': 1.0835, 'low': 1.082,
        'close': 1.083, 'volume': 1500}, {'timestamp':
        '2025-04-01T11:00:00', 'open': 1.083, 'high': 1.084, 'low': 1.0825,
        'close': 1.0835, 'volume': 1600}, {'timestamp':
        '2025-04-01T12:00:00', 'open': 1.0835, 'high': 1.0845, 'low': 1.083,
        'close': 1.084, 'volume': 1700}, {'timestamp':
        '2025-04-01T13:00:00', 'open': 1.084, 'high': 1.085, 'low': 1.0835,
        'close': 1.0845, 'volume': 1800}, {'timestamp':
        '2025-04-01T14:00:00', 'open': 1.0845, 'high': 1.0855, 'low': 1.084,
        'close': 1.085, 'volume': 1900}, {'timestamp':
        '2025-04-01T15:00:00', 'open': 1.085, 'high': 1.086, 'low': 1.0845,
        'close': 1.0855, 'volume': 2000}, {'timestamp':
        '2025-04-01T16:00:00', 'open': 1.0855, 'high': 1.0865, 'low': 1.085,
        'close': 1.086, 'volume': 2100}, {'timestamp':
        '2025-04-01T17:00:00', 'open': 1.086, 'high': 1.087, 'low': 1.0855,
        'close': 1.0865, 'volume': 2200}, {'timestamp':
        '2025-04-01T18:00:00', 'open': 1.0865, 'high': 1.0875, 'low': 1.086,
        'close': 1.087, 'volume': 2300}, {'timestamp':
        '2025-04-01T19:00:00', 'open': 1.087, 'high': 1.088, 'low': 1.0865,
        'close': 1.0875, 'volume': 2400}]
    try:
        result = await client.detect_market_regime(symbol=symbol, timeframe
            =timeframe, ohlc_data=ohlc_data)
        logger.info(f'Detected market regime: {result}')
        return result
    except Exception as e:
        logger.error(f'Error detecting market regime: {str(e)}')
        raise


@async_with_exception_handling
async def get_regime_history_example():
    """
    Example of getting regime history using the standardized client.
    """
    factory = get_client_factory()
    client = factory.get_market_regime_client()
    symbol = 'EURUSD'
    timeframe = '1h'
    limit = 5
    try:
        result = await client.get_regime_history(symbol=symbol, timeframe=
            timeframe, limit=limit)
        logger.info(f'Got regime history: {result}')
        return result
    except Exception as e:
        logger.error(f'Error getting regime history: {str(e)}')
        raise


@async_with_exception_handling
async def analyze_tool_regime_performance_example():
    """
    Example of analyzing tool regime performance using the standardized client.
    """
    factory = get_client_factory()
    client = factory.get_market_regime_client()
    tool_id = 'macd_crossover_v1'
    timeframe = '1h'
    instrument = 'EUR_USD'
    from_date = datetime.now() - timedelta(days=30)
    to_date = datetime.now()
    try:
        result = await client.analyze_tool_regime_performance(tool_id=
            tool_id, timeframe=timeframe, instrument=instrument, from_date=
            from_date, to_date=to_date)
        logger.info(f'Analyzed tool regime performance: {result}')
        return result
    except Exception as e:
        logger.error(f'Error analyzing tool regime performance: {str(e)}')
        raise


async def main():
    """
    Run the examples.
    """
    logger.info('Running Market Regime client examples')
    await detect_market_regime_example()
    await get_regime_history_example()
    await analyze_tool_regime_performance_example()
    logger.info('Completed Market Regime client examples')


if __name__ == '__main__':
    asyncio.run(main())
