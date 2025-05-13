"""
Example script for using the Analysis Engine Adapter.

This script demonstrates how to use the Analysis Engine Adapter to access
technical indicators and market analysis functionality.
"""
import asyncio
import pandas as pd
from datetime import datetime, timedelta
from adapters.adapter_factory_1 import adapter_factory
from common_lib.interfaces.analysis_engine import IAnalysisProvider, IIndicatorProvider, IPatternRecognizer


from core.exceptions_bridge_1 import (
    with_exception_handling,
    async_with_exception_handling,
    ForexTradingPlatformError,
    ServiceError,
    DataError,
    ValidationError
)

@async_with_exception_handling
async def main():
    """Run the example."""
    analysis_provider = adapter_factory.get_analysis_provider()
    analysis_provider_interface = adapter_factory.get_adapter(IAnalysisProvider
        )
    indicator_provider = adapter_factory.get_adapter(IIndicatorProvider)
    pattern_recognizer = adapter_factory.get_adapter(IPatternRecognizer)
    print('Example 1: Detect market regime')
    try:
        regime = await analysis_provider.detect_market_regime(symbol=
            'EURUSD', timeframe='1h', lookback_bars=100)
        print(f'Market regime: {regime}')
    except Exception as e:
        print(f'Error detecting market regime: {str(e)}')
    print('\nExample 2: Calculate indicator')
    try:
        data = pd.DataFrame({'open': [1.0, 1.1, 1.2, 1.3, 1.4], 'high': [
            1.1, 1.2, 1.3, 1.4, 1.5], 'low': [0.9, 1.0, 1.1, 1.2, 1.3],
            'close': [1.0, 1.1, 1.2, 1.3, 1.4], 'volume': [100, 200, 300, 
            400, 500]})
        result = await indicator_provider.calculate_indicator(indicator_name
            ='sma', data=data, parameters={'period': 3})
        print(f'SMA result:\n{result}')
    except Exception as e:
        print(f'Error calculating indicator: {str(e)}')
    print('\nExample 3: Get technical indicators')
    try:
        indicators = await analysis_provider.get_technical_indicators(symbol
            ='EURUSD', timeframe='1h', indicators=[{'name': 'sma', 'params':
            {'period': 20}}, {'name': 'rsi', 'params': {'period': 14}}],
            start_time=datetime.utcnow() - timedelta(days=7), end_time=
            datetime.utcnow())
        print(f'Technical indicators: {indicators}')
    except Exception as e:
        print(f'Error getting technical indicators: {str(e)}')
    print('\nExample 4: Recognize patterns')
    try:
        data = pd.DataFrame({'open': [1.0, 1.1, 1.2, 1.3, 1.4, 1.3, 1.2, 
            1.1, 1.0, 1.1, 1.2, 1.3, 1.4, 1.3, 1.2], 'high': [1.1, 1.2, 1.3,
            1.4, 1.5, 1.4, 1.3, 1.2, 1.1, 1.2, 1.3, 1.4, 1.5, 1.4, 1.3],
            'low': [0.9, 1.0, 1.1, 1.2, 1.3, 1.2, 1.1, 1.0, 0.9, 1.0, 1.1, 
            1.2, 1.3, 1.2, 1.1], 'close': [1.0, 1.1, 1.2, 1.3, 1.4, 1.3, 
            1.2, 1.1, 1.0, 1.1, 1.2, 1.3, 1.4, 1.3, 1.2], 'volume': [100, 
            200, 300, 400, 500, 400, 300, 200, 100, 200, 300, 400, 500, 400,
            300]})
        patterns = await pattern_recognizer.recognize_patterns(data=data,
            pattern_types=['double_top', 'double_bottom'])
        print(f'Recognized patterns: {patterns}')
    except Exception as e:
        print(f'Error recognizing patterns: {str(e)}')
    print('\nExample 5: Get pattern types')
    try:
        pattern_types = await pattern_recognizer.get_pattern_types()
        print(f'Pattern types: {pattern_types}')
    except Exception as e:
        print(f'Error getting pattern types: {str(e)}')


if __name__ == '__main__':
    adapter_factory.initialize()
    asyncio.run(main())
