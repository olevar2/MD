"""
Visualization Integration Adapter

This module connects the visualization components with the existing indicator
implementations from the feature-store-service and analysis-engine-service.
"""
import logging
from typing import Dict, List, Any, Optional, Union, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import asyncio
import json
from ui_service.components.visualization_library import IndicatorVisualizationLibrary, InteractiveParameterController, SynchronizedDisplayManager
import sys
import os
from pathlib import Path
REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(REPO_ROOT))
from feature_store_service.indicators import indicator_registry
from feature_store_service.indicators.moving_averages import simple_moving_average, exponential_moving_average, weighted_moving_average, hull_moving_average
from feature_store_service.indicators.oscillators import relative_strength_index, stochastic_oscillator, macd, commodity_channel_index
from feature_store_service.indicators.volatility import bollinger_bands, average_true_range, keltner_channels
from feature_store_service.indicators.volume import on_balance_volume, volume_weighted_average_price
from analysis_engine.analysis.signal_system import SignalSystem
from analysis_engine.analysis.confluence.confluence_analyzer import ConfluenceAnalyzer
from analysis_engine.analysis.indicator_interface import IndicatorInterface
from analysis_engine.analysis.market_regime import MarketRegimeAnalyzer
logger = logging.getLogger(__name__)


from ui_service.error.exceptions_bridge import (
    with_exception_handling,
    async_with_exception_handling,
    ForexTradingPlatformError,
    ServiceError,
    DataError,
    ValidationError
)

class VisualizationAdapter:
    """
    Adapter to connect indicator implementations with the visualization components.
    This class handles the data transformation needed between the indicator outputs
    and the visualization input format.
    """

    def __init__(self):
        """Initialize the visualization adapter"""
        self.visualization_library = IndicatorVisualizationLibrary()
        self.indicator_registry = indicator_registry.IndicatorRegistry()
        self.sync_manager = SynchronizedDisplayManager()
        self.parameter_controllers = {}
        self.indicator_interface = IndicatorInterface()
        self._register_indicators()

    def _register_indicators(self):
        """Register all available indicators with both registry and visualization"""
        logger.info('Registering indicators with the adapter')
        self.indicator_registry.register_common_indicators()

    def get_indicator_metadata(self) ->List[Dict[str, Any]]:
        """
        Get metadata for all available indicators

        Returns:
            List of dictionaries with indicator metadata
        """
        indicators = self.indicator_registry.get_all_indicators()
        metadata = []
        for indicator_name in indicators:
            indicator_info = self.indicator_registry.get_indicator(
                indicator_name)
            if indicator_info:
                parameters = indicator_info.get('parameters', {})
                category = indicator_info.get('category', 'other')
                viz_type = self._map_category_to_viz_type(category)
                metadata.append({'name': indicator_name, 'display_name':
                    indicator_name.replace('_', ' ').title(), 'description':
                    indicator_info.get('description', ''), 'category':
                    category, 'visualization_type': viz_type, 'parameters':
                    parameters})
        return metadata

    def _map_category_to_viz_type(self, category: str) ->str:
        """Map indicator category to visualization type"""
        category_map = {'moving_average': 'moving_average', 'oscillator':
            'oscillator', 'volatility': 'volatility', 'volume': 'volume',
            'pattern': 'pattern', 'momentum': 'oscillator', 'trend':
            'moving_average', 'harmonic': 'harmonic', 'multi_timeframe':
            'multi_timeframe', 'correlation': 'correlation'}
        return category_map.get(category.lower(), 'default')

    @with_exception_handling
    def calculate_indicator(self, indicator_name: str, price_data: pd.
        DataFrame, params: Dict[str, Any]=None) ->pd.DataFrame:
        """
        Calculate an indicator using the registry

        Args:
            indicator_name: Name of the indicator to calculate
            price_data: OHLCV price data
            params: Parameters for the indicator

        Returns:
            DataFrame with calculated indicator values
        """
        try:
            params = params or {}
            return self.indicator_registry.calculate(indicator_name,
                price_data, **params)
        except Exception as e:
            logger.error(f'Error calculating {indicator_name}: {str(e)}')
            raise

    def prepare_visualization_data(self, indicator_name: str,
        indicator_data: (pd.DataFrame or pd.Series), price_data: pd.
        DataFrame=None, params: Dict[str, Any]=None) ->Dict[str, Any]:
        """
        Transform indicator data to the format expected by the visualization library

        Args:
            indicator_name: Name of the indicator
            indicator_data: Calculated indicator values
            price_data: Original price data (optional)
            params: Parameters used for calculation

        Returns:
            Dictionary with data for visualization
        """
        params = params or {}
        indicator_info = self.indicator_registry.get_indicator(indicator_name)
        if not indicator_info:
            logger.warning(f'No metadata found for indicator: {indicator_name}'
                )
            category = 'other'
        else:
            category = indicator_info.get('category', 'other')
        viz_type = self._map_category_to_viz_type(category)
        viz_data = {'type': viz_type, 'name': indicator_name.replace('_',
            ' ').title(), 'data': self._convert_data_for_visualization(
            indicator_data), 'parameters': params}
        if price_data is not None:
            viz_data['price_data'] = self._convert_data_for_visualization(
                price_data)
        return viz_data

    def _convert_data_for_visualization(self, data: (pd.DataFrame or pd.Series)
        ) ->Dict[str, Any]:
        """
        Convert DataFrame or Series to format expected by visualization components

        Args:
            data: DataFrame or Series with indicator values

        Returns:
            Dictionary with data in visualization format
        """
        if isinstance(data, pd.DataFrame):
            result = {}
            if isinstance(data.index, pd.DatetimeIndex):
                result['timestamps'] = data.index.strftime('%Y-%m-%d %H:%M:%S'
                    ).tolist()
            else:
                result['timestamps'] = data.index.tolist()
            for column in data.columns:
                series_data = data[column].values
                result[column] = [(None if np.isnan(x) else float(x)) for x in
                    series_data]
            return result
        elif isinstance(data, pd.Series):
            result = {}
            if isinstance(data.index, pd.DatetimeIndex):
                result['timestamps'] = data.index.strftime('%Y-%m-%d %H:%M:%S'
                    ).tolist()
            else:
                result['timestamps'] = data.index.tolist()
            series_data = data.values
            result['values'] = [(None if np.isnan(x) else float(x)) for x in
                series_data]
            return result
        return {'timestamps': [], 'values': []}

    @with_exception_handling
    def render_indicator_visualization(self, indicator_name: str,
        price_data: pd.DataFrame, container_id: str, params: Dict[str, Any]
        =None, options: Dict[str, Any]=None) ->Any:
        """
        Calculate an indicator and render its visualization

        Args:
            indicator_name: Name of the indicator to calculate
            price_data: OHLCV price data
            container_id: DOM container ID for rendering
            params: Parameters for the indicator
            options: Visualization options

        Returns:
            Visualization component
        """
        params = params or {}
        options = options or {}
        try:
            indicator_data = self.calculate_indicator(indicator_name,
                price_data, params)
            viz_data = self.prepare_visualization_data(indicator_name,
                indicator_data, price_data, params)
            return self.visualization_library.render_indicator(viz_data,
                container_id, options)
        except Exception as e:
            logger.error(f'Error rendering {indicator_name}: {str(e)}')
            return self._render_error_visualization(indicator_name,
                container_id, str(e))

    def _render_error_visualization(self, indicator_name: str, container_id:
        str, error_message: str) ->Any:
        """Render an error visualization for failed indicators"""
        error_data = {'type': 'error', 'name': f'Error: {indicator_name}',
            'error': error_message}
        return self.visualization_library.render_default_indicator(error_data,
            container_id, {'height': 200})

    def create_parameter_controller(self, indicator_name: str, chart_id:
        str, price_data: pd.DataFrame, initial_params: Dict[str, Any]=None
        ) ->InteractiveParameterController:
        """
        Create an interactive parameter controller for an indicator

        Args:
            indicator_name: Name of the indicator
            chart_id: ID of the chart to control
            price_data: Price data for recalculation
            initial_params: Initial parameters

        Returns:
            InteractiveParameterController instance
        """
        initial_params = initial_params or {}
        indicator_info = self.indicator_registry.get_indicator(indicator_name)
        if not indicator_info:
            logger.warning(
                f'No metadata found for parameter controller: {indicator_name}'
                )
            parameter_info = {}
        else:
            parameter_info = indicator_info.get('parameters', {})
        controller = InteractiveParameterController(chart_id, initial_params)
        controller_id = f'{indicator_name}_{chart_id}'
        self.parameter_controllers[controller_id] = {'controller':
            controller, 'indicator_name': indicator_name, 'price_data':
            price_data, 'chart_id': chart_id}

        def parameter_changed(new_value):
    """
    Parameter changed.
    
    Args:
        new_value: Description of new_value
    
    """

            self._update_indicator_visualization(controller_id)
        for param_name, param_description in parameter_info.items():
            if param_name in initial_params:
                controller.register_callback(param_name, parameter_changed)
                param_value = initial_params[param_name]
                if isinstance(param_value, bool):
                    controller.create_toggle(param_name, param_description)
                elif isinstance(param_value, (int, float)):
                    if isinstance(param_value, int):
                        min_val = max(1, param_value // 2)
                        max_val = param_value * 2
                        step = 1
                    else:
                        min_val = max(0.1, param_value / 2)
                        max_val = param_value * 2
                        step = (max_val - min_val) / 20
                    controller.create_slider(param_name, min_val, max_val, step
                        )
                elif isinstance(param_value, str):
                    options = [{'value': param_value, 'label': param_value}]
                    controller.create_dropdown(param_name, options)
        return controller

    def _update_indicator_visualization(self, controller_id: str):
        """Update indicator visualization after parameter change"""
        controller_info = self.parameter_controllers.get(controller_id)
        if not controller_info:
            logger.warning(f'Controller not found: {controller_id}')
            return
        controller = controller_info['controller']
        indicator_name = controller_info['indicator_name']
        price_data = controller_info['price_data']
        chart_id = controller_info['chart_id']
        return self.render_indicator_visualization(indicator_name,
            price_data, chart_id, controller.parameters)

    def create_synchronized_display(self, group_id: str, indicators: List[
        Dict[str, Any]]) ->None:
        """
        Create a synchronized display group for multiple indicators

        Args:
            group_id: ID for the synchronization group
            indicators: List of indicator configurations with chart IDs
        """
        chart_ids = [indicator.get('chart_id') for indicator in indicators if
            indicator.get('chart_id')]
        self.sync_manager.create_sync_group(group_id, chart_ids)

    @async_with_exception_handling
    async def analyze_indicator_confluence(self, indicators_data: Dict[str,
        pd.DataFrame or pd.Series], threshold: float=0.7) ->Dict[str, Any]:
        """
        Analyze confluence between multiple indicators

        Args:
            indicators_data: Dictionary mapping indicator names to their data
            threshold: Confluence threshold

        Returns:
            Dictionary with confluence analysis results
        """
        try:
            analyzer = ConfluenceAnalyzer()
            market_data = {'symbol': 'unknown', 'timeframe': 'current',
                'market_data': {'open': indicators_data.get('open', pd.
                Series()), 'high': indicators_data.get('high', pd.Series()),
                'low': indicators_data.get('low', pd.Series()), 'close':
                indicators_data.get('close', pd.Series()), 'volume':
                indicators_data.get('volume', pd.Series()), 'timestamp':
                indicators_data.get('timestamp', pd.Series())}}
            results = await analyzer.analyze(market_data)
            return {'confluence_score': results.result.get(
                'confluence_zones', []), 'signal_strength': results.
                metadata.get('zone_count', 0), 'direction': 'neutral',
                'indicators_agreement': results.result.get(
                'effective_tools', {})}
        except Exception as e:
            logger.error(f'Error analyzing indicator confluence: {str(e)}')
            return {'confluence_score': 0, 'signal_strength': 0,
                'direction': 'error', 'error': str(e)}
