"""
API Testing Client

This module provides a command-line testing client for the analysis-engine-service APIs,
allowing developers to verify functionality and gather example responses.
"""
import argparse
import json
import logging
import pandas as pd
import requests
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
logging.basicConfig(level=logging.INFO, format=
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('api_test_client')


from analysis_engine.core.exceptions_bridge import (
    with_exception_handling,
    async_with_exception_handling,
    ForexTradingPlatformError,
    ServiceError,
    DataError,
    ValidationError
)

class ApiTestClient:
    """
    Test client for the analysis-engine-service APIs.
    Allows testing of API endpoints from the command line.
    """

    def __init__(self, base_url: str='http://localhost:8000/api/v1'):
        """
        Initialize the API test client
        
        Args:
            base_url: Base URL for the analysis-engine-service API
        """
        self.base_url = base_url
        self.logger = logger

    @with_exception_handling
    def test_market_regime_detection(self, symbol: str, timeframe: str,
        days: int=30) ->None:
        """
        Test the market regime detection API endpoint
        
        Args:
            symbol: Trading symbol to analyze
            timeframe: Chart timeframe
            days: Number of days of data to generate
        """
        ohlc_data = self._generate_sample_data(days)
        payload = {'symbol': symbol, 'timeframe': timeframe, 'ohlc_data':
            ohlc_data}
        try:
            endpoint = f'{self.base_url}/market-regime/detect/'
            self.logger.info(f'Testing endpoint: {endpoint}')
            response = requests.post(endpoint, json=payload)
            if response.status_code == 200:
                result = response.json()
                self.logger.info(
                    f"Detected market regime: {result['regime']} (confidence: {result['confidence']})"
                    )
                return result
            else:
                self.logger.error(
                    f'Error: {response.status_code} - {response.text}')
                return None
        except Exception as e:
            self.logger.error(
                f'Error testing market regime detection: {str(e)}')
            return None

    @with_exception_handling
    def test_adaptive_parameters(self, strategy_id: str, symbol: str,
        timeframe: str, days: int=30, adaptation_strategy: str='moderate'
        ) ->None:
        """
        Test the adaptive parameters generation API endpoint
        
        Args:
            strategy_id: ID of the strategy
            symbol: Trading symbol
            timeframe: Chart timeframe
            days: Number of days of data to generate
            adaptation_strategy: Strategy for parameter adaptation
        """
        ohlc_data = self._generate_sample_data(days)
        available_tools = ['macd', 'rsi', 'bollinger_bands',
            'fibonacci_retracement']
        payload = {'strategy_id': strategy_id, 'symbol': symbol,
            'timeframe': timeframe, 'ohlc_data': ohlc_data,
            'available_tools': available_tools, 'adaptation_strategy':
            adaptation_strategy}
        try:
            endpoint = f'{self.base_url}/adaptive-layer/generate-parameters/'
            self.logger.info(f'Testing endpoint: {endpoint}')
            response = requests.post(endpoint, json=payload)
            if response.status_code == 200:
                result = response.json()
                self.logger.info(
                    f'Generated adaptive parameters for strategy: {strategy_id}'
                    )
                self.logger.info(
                    f"Parameters: {json.dumps(result.get('parameters', {}), indent=2)}"
                    )
                return result
            else:
                self.logger.error(
                    f'Error: {response.status_code} - {response.text}')
                return None
        except Exception as e:
            self.logger.error(
                f'Error testing adaptive parameters generation: {str(e)}')
            return None

    @with_exception_handling
    def test_strategy_recommendations(self, strategy_id: str, symbol: str,
        timeframe: str, days: int=30) ->None:
        """
        Test the strategy recommendations API endpoint
        
        Args:
            strategy_id: ID of the strategy
            symbol: Trading symbol
            timeframe: Chart timeframe
            days: Number of days of data to generate
        """
        ohlc_data = self._generate_sample_data(days)
        current_tools = ['macd', 'rsi']
        all_available_tools = ['macd', 'rsi', 'bollinger_bands',
            'fibonacci_retracement', 'stochastic', 'adx', 'atr', 'ema',
            'sma', 'ichimoku']
        payload = {'strategy_id': strategy_id, 'symbol': symbol,
            'timeframe': timeframe, 'ohlc_data': ohlc_data, 'current_tools':
            current_tools, 'all_available_tools': all_available_tools}
        try:
            endpoint = (
                f'{self.base_url}/adaptive-layer/strategy-recommendations/')
            self.logger.info(f'Testing endpoint: {endpoint}')
            response = requests.post(endpoint, json=payload)
            if response.status_code == 200:
                result = response.json()
                self.logger.info(
                    f'Generated recommendations for strategy: {strategy_id}')
                if 'tool_recommendations' in result:
                    self.logger.info('Top recommended tools:')
                    for i, tool in enumerate(result['tool_recommendations']
                        [:5], 1):
                        self.logger.info(
                            f"{i}. {tool['tool_id']} - Win Rate: {tool.get('win_rate')}%, Profit Factor: {tool.get('profit_factor')}"
                            )
                if 'tools_to_add' in result:
                    self.logger.info(
                        f"Tools to add: {', '.join(t['tool_id'] for t in result['tools_to_add'])}"
                        )
                if 'tools_to_remove' in result:
                    self.logger.info(
                        f"Tools to remove: {', '.join(t['tool_id'] for t in result['tools_to_remove'])}"
                        )
                return result
            else:
                self.logger.error(
                    f'Error: {response.status_code} - {response.text}')
                return None
        except Exception as e:
            self.logger.error(
                f'Error testing strategy recommendations: {str(e)}')
            return None

    @with_exception_handling
    def test_tool_regime_performance(self, tool_id: str, timeframe:
        Optional[str]=None) ->None:
        """
        Test the tool regime performance analysis API endpoint
        
        Args:
            tool_id: ID of the trading tool
            timeframe: Optional filter by timeframe
        """
        payload = {'tool_id': tool_id}
        if timeframe:
            payload['timeframe'] = timeframe
        try:
            endpoint = f'{self.base_url}/market-regime/regime-analysis/'
            self.logger.info(f'Testing endpoint: {endpoint}')
            response = requests.post(endpoint, json=payload)
            if response.status_code == 200:
                result = response.json()
                self.logger.info(f'Analyzed performance for tool: {tool_id}')
                if 'regimes' in result:
                    self.logger.info('Performance by regime:')
                    for regime, metrics in result['regimes'].items():
                        self.logger.info(f'Regime: {regime}')
                        self.logger.info(
                            f"  Win Rate: {metrics.get('win_rate')}%")
                        self.logger.info(
                            f"  Profit Factor: {metrics.get('profit_factor')}")
                        self.logger.info(
                            f"  Sample Size: {metrics.get('sample_size')}")
                return result
            else:
                self.logger.error(
                    f'Error: {response.status_code} - {response.text}')
                return None
        except Exception as e:
            self.logger.error(
                f'Error testing tool regime performance: {str(e)}')
            return None

    def _generate_sample_data(self, days: int=30) ->List[Dict[str, Any]]:
        """
        Generate sample OHLC data for testing
        
        Args:
            days: Number of days of data to generate
            
        Returns:
            List of OHLC data points as dictionaries
        """
        data = []
        base_price = 1.1
        volatility = 0.002
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        current_date = start_date
        while current_date <= end_date:
            if current_date.weekday() >= 5:
                current_date += timedelta(days=1)
                continue
            import random
            daily_change = random.normalvariate(0, volatility)
            for hour in range(0, 24, 4):
                timestamp = current_date.replace(hour=hour, minute=0, second=0)
                open_price = base_price + daily_change
                high_price = open_price + abs(random.normalvariate(0, 
                    volatility / 2))
                low_price = open_price - abs(random.normalvariate(0, 
                    volatility / 2))
                close_price = (high_price + low_price) / 2
                volume = random.randint(800, 1200)
                data_point = {'timestamp': timestamp.isoformat(), 'open':
                    round(open_price, 5), 'high': round(high_price, 5),
                    'low': round(low_price, 5), 'close': round(close_price,
                    5), 'volume': volume}
                data.append(data_point)
                base_price = close_price
            current_date += timedelta(days=1)
        return data


def main():
    """Main function to run the API test client."""
    parser = argparse.ArgumentParser(description=
        'Test the analysis-engine-service APIs')
    parser.add_argument('--base-url', type=str, default=
        'http://localhost:8000/api/v1', help=
        'Base URL for the analysis-engine-service API')
    parser.add_argument('--endpoint', type=str, choices=['regime', 'params',
        'recommendations', 'performance'], default='regime', help=
        'API endpoint to test')
    parser.add_argument('--symbol', type=str, default='EURUSD', help=
        'Trading symbol')
    parser.add_argument('--timeframe', type=str, default='4h', help=
        'Chart timeframe')
    parser.add_argument('--days', type=int, default=30, help=
        'Number of days of data to generate')
    parser.add_argument('--strategy-id', type=str, default=
        'test_strategy_1', help='Strategy ID')
    parser.add_argument('--tool-id', type=str, default='macd', help='Tool ID')
    args = parser.parse_args()
    client = ApiTestClient(args.base_url)
    if args.endpoint == 'regime':
        client.test_market_regime_detection(args.symbol, args.timeframe,
            args.days)
    elif args.endpoint == 'params':
        client.test_adaptive_parameters(args.strategy_id, args.symbol, args
            .timeframe, args.days)
    elif args.endpoint == 'recommendations':
        client.test_strategy_recommendations(args.strategy_id, args.symbol,
            args.timeframe, args.days)
    elif args.endpoint == 'performance':
        client.test_tool_regime_performance(args.tool_id, args.timeframe)


if __name__ == '__main__':
    main()
