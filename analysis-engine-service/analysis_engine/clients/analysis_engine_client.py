"""
Analysis Engine API Client

This module provides client utilities for interacting with the analysis-engine-service APIs,
making it easy for other services to utilize market regime detection and adaptive layer functionality.
"""
import logging
import requests
import pandas as pd
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from analysis_engine.core.exceptions_bridge import with_exception_handling, async_with_exception_handling, ForexTradingPlatformError, ServiceError, DataError, ValidationError


from analysis_engine.resilience.utils import (
    with_resilience,
    with_analysis_resilience,
    with_database_resilience
)

class AnalysisEngineClient:
    """
    Client for interacting with the analysis-engine-service APIs.
    Provides easy access to market regime detection and adaptive layer functionality.
    """

    def __init__(self, base_url: str=
        'http://analysis-engine-service:8000/api/v1'):
        """
        Initialize the analysis engine client
        
        Args:
            base_url: Base URL for the analysis-engine-service API
        """
        self.base_url = base_url
        self.logger = logging.getLogger(__name__)

    @with_exception_handling
    def detect_market_regime(self, symbol: str, timeframe: str, ohlc_data:
        Union[pd.DataFrame, List[Dict]]) ->Dict[str, Any]:
        """
        Detect the current market regime based on price data
        
        Args:
            symbol: Trading symbol
            timeframe: Chart timeframe
            ohlc_data: Price data as DataFrame or list of dictionaries
            
        Returns:
            Dictionary with detected regime and supporting metrics
        """
        try:
            if isinstance(ohlc_data, pd.DataFrame):
                ohlc_data = ohlc_data.to_dict(orient='records')
            payload = {'symbol': symbol, 'timeframe': timeframe,
                'ohlc_data': ohlc_data}
            endpoint = f'{self.base_url}/market-regime/detect/'
            response = requests.post(endpoint, json=payload)
            if response.status_code == 200:
                return response.json()
            else:
                self.logger.error(
                    f'Failed to detect market regime: {response.text}')
                return {'regime': 'unknown', 'confidence': 0.0, 'error':
                    response.text}
        except Exception as e:
            self.logger.error(f'Error detecting market regime: {str(e)}')
            return {'regime': 'unknown', 'confidence': 0.0, 'error': str(e)}

    @with_resilience('get_regime_history')
    @with_exception_handling
    def get_regime_history(self, symbol: str, timeframe: str, limit: int=10
        ) ->List[Dict[str, Any]]:
        """
        Get historical regime data for a specific symbol and timeframe
        
        Args:
            symbol: Trading symbol
            timeframe: Chart timeframe
            limit: Maximum number of historical entries to return
            
        Returns:
            List of historical regime entries
        """
        try:
            payload = {'symbol': symbol, 'timeframe': timeframe, 'limit': limit
                }
            endpoint = f'{self.base_url}/market-regime/history/'
            response = requests.post(endpoint, json=payload)
            if response.status_code == 200:
                return response.json()
            else:
                self.logger.error(
                    f'Failed to get regime history: {response.text}')
                return []
        except Exception as e:
            self.logger.error(f'Error getting regime history: {str(e)}')
            return []

    @with_exception_handling
    def generate_adaptive_parameters(self, strategy_id: str, symbol: str,
        timeframe: str, ohlc_data: Union[pd.DataFrame, List[Dict]],
        available_tools: List[str], adaptation_strategy: str='moderate'
        ) ->Dict[str, Any]:
        """
        Generate adaptive parameters based on market conditions and tool effectiveness
        
        Args:
            strategy_id: ID of the strategy
            symbol: Trading symbol
            timeframe: Chart timeframe
            ohlc_data: Price data as DataFrame or list of dictionaries
            available_tools: List of tool IDs available for the strategy
            adaptation_strategy: Strategy for parameter adaptation (conservative, moderate, aggressive, experimental)
            
        Returns:
            Dictionary with adaptive parameters
        """
        try:
            if isinstance(ohlc_data, pd.DataFrame):
                ohlc_data = ohlc_data.to_dict(orient='records')
            payload = {'strategy_id': strategy_id, 'symbol': symbol,
                'timeframe': timeframe, 'ohlc_data': ohlc_data,
                'available_tools': available_tools, 'adaptation_strategy':
                adaptation_strategy}
            endpoint = f'{self.base_url}/adaptive-layer/generate-parameters/'
            response = requests.post(endpoint, json=payload)
            if response.status_code == 200:
                return response.json()
            else:
                self.logger.error(
                    f'Failed to generate adaptive parameters: {response.text}')
                return {'error': response.text}
        except Exception as e:
            self.logger.error(f'Error generating adaptive parameters: {str(e)}'
                )
            return {'error': str(e)}

    @with_resilience('update_strategy_parameters')
    @with_exception_handling
    def update_strategy_parameters(self, strategy_id: str, symbol: str,
        timeframe: str, ohlc_data: Union[pd.DataFrame, List[Dict]],
        available_tools: List[str], adaptation_strategy: str='moderate',
        strategy_execution_api_url: Optional[str]=None) ->Dict[str, Any]:
        """
        Update strategy parameters and apply them to the strategy execution engine
        
        Args:
            strategy_id: ID of the strategy
            symbol: Trading symbol
            timeframe: Chart timeframe
            ohlc_data: Price data as DataFrame or list of dictionaries
            available_tools: List of tool IDs available for the strategy
            adaptation_strategy: Strategy for parameter adaptation
            strategy_execution_api_url: URL for the strategy execution engine API
            
        Returns:
            Dictionary with update status information
        """
        try:
            if isinstance(ohlc_data, pd.DataFrame):
                ohlc_data = ohlc_data.to_dict(orient='records')
            payload = {'strategy_id': strategy_id, 'symbol': symbol,
                'timeframe': timeframe, 'ohlc_data': ohlc_data,
                'available_tools': available_tools, 'adaptation_strategy':
                adaptation_strategy, 'strategy_execution_api_url':
                strategy_execution_api_url}
            endpoint = f'{self.base_url}/adaptive-layer/update-strategy/'
            response = requests.post(endpoint, json=payload)
            if response.status_code == 200:
                return response.json()
            else:
                self.logger.error(
                    f'Failed to update strategy parameters: {response.text}')
                return {'error': response.text}
        except Exception as e:
            self.logger.error(f'Error updating strategy parameters: {str(e)}')
            return {'error': str(e)}

    @with_resilience('get_strategy_recommendations')
    @with_exception_handling
    def get_strategy_recommendations(self, strategy_id: str, symbol: str,
        timeframe: str, ohlc_data: Union[pd.DataFrame, List[Dict]],
        current_tools: List[str], all_available_tools: List[str]) ->Dict[
        str, Any]:
        """
        Get recommendations for optimizing a strategy based on effectiveness data
        
        Args:
            strategy_id: ID of the strategy
            symbol: Trading symbol
            timeframe: Chart timeframe
            ohlc_data: Price data as DataFrame or list of dictionaries
            current_tools: List of tool IDs currently used by the strategy
            all_available_tools: List of all available tool IDs
            
        Returns:
            Dictionary with strategy optimization recommendations
        """
        try:
            if isinstance(ohlc_data, pd.DataFrame):
                ohlc_data = ohlc_data.to_dict(orient='records')
            payload = {'strategy_id': strategy_id, 'symbol': symbol,
                'timeframe': timeframe, 'ohlc_data': ohlc_data,
                'current_tools': current_tools, 'all_available_tools':
                all_available_tools}
            endpoint = (
                f'{self.base_url}/adaptive-layer/strategy-recommendations/')
            response = requests.post(endpoint, json=payload)
            if response.status_code == 200:
                return response.json()
            else:
                self.logger.error(
                    f'Failed to get strategy recommendations: {response.text}')
                return {'error': response.text}
        except Exception as e:
            self.logger.error(
                f'Error getting strategy recommendations: {str(e)}')
            return {'error': str(e)}

    @with_analysis_resilience('analyze_tool_regime_performance')
    @with_exception_handling
    def analyze_tool_regime_performance(self, tool_id: str, timeframe:
        Optional[str]=None, instrument: Optional[str]=None, from_date:
        Optional[datetime]=None, to_date: Optional[datetime]=None) ->Dict[
        str, Any]:
        """
        Get the performance metrics of a tool across different market regimes
        
        Args:
            tool_id: ID of the trading tool
            timeframe: Optional filter by timeframe
            instrument: Optional filter by trading instrument
            from_date: Optional start date for analysis
            to_date: Optional end date for analysis
            
        Returns:
            Dictionary with performance metrics by regime
        """
        try:
            payload = {'tool_id': tool_id}
            if timeframe:
                payload['timeframe'] = timeframe
            if instrument:
                payload['instrument'] = instrument
            if from_date:
                payload['from_date'] = from_date.isoformat()
            if to_date:
                payload['to_date'] = to_date.isoformat()
            endpoint = f'{self.base_url}/market-regime/regime-analysis/'
            response = requests.post(endpoint, json=payload)
            if response.status_code == 200:
                return response.json()
            else:
                self.logger.error(
                    f'Failed to analyze tool regime performance: {response.text}'
                    )
                return {}
        except Exception as e:
            self.logger.error(
                f'Error analyzing tool regime performance: {str(e)}')
            return {}

    @with_exception_handling
    def find_optimal_market_conditions(self, tool_id: str, min_sample_size:
        int=10, timeframe: Optional[str]=None, instrument: Optional[str]=None
        ) ->Dict[str, Any]:
        """
        Find the optimal market conditions for a specific tool
        
        Args:
            tool_id: ID of the trading tool
            min_sample_size: Minimum number of signals required for reliable analysis
            timeframe: Optional filter by timeframe
            instrument: Optional filter by trading instrument
            
        Returns:
            Dictionary with optimal market conditions
        """
        try:
            payload = {'tool_id': tool_id, 'min_sample_size': min_sample_size}
            if timeframe:
                payload['timeframe'] = timeframe
            if instrument:
                payload['instrument'] = instrument
            endpoint = f'{self.base_url}/market-regime/optimal-conditions/'
            response = requests.post(endpoint, json=payload)
            if response.status_code == 200:
                return response.json()
            else:
                self.logger.error(
                    f'Failed to find optimal market conditions: {response.text}'
                    )
                return {}
        except Exception as e:
            self.logger.error(
                f'Error finding optimal market conditions: {str(e)}')
            return {}
