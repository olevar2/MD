"""
Advanced Indicator Adapter Module.

This module provides adapter classes that connect the Feature Store Service's indicator system
with the more complex analytical components in the Analysis Engine Service.
"""
from typing import Dict, Any, List, Optional, Type, Union
import pandas as pd
import importlib
import inspect
import sys
import logging
from pathlib import Path
from core.base_indicator import BaseIndicator
from core.advanced_indicator_optimization import graceful_fallback, performance_tracking, optimizer, AdvancedIndicatorError
from core.profiling import log_and_time
logger = logging.getLogger(__name__)


from feature_store_service.error.exceptions_bridge import (
    with_exception_handling,
    async_with_exception_handling,
    ForexTradingPlatformError,
    ServiceError,
    DataError,
    ValidationError
)

class AdvancedIndicatorAdapter(BaseIndicator):
    """
    Adapter that connects Feature Store to Analysis Engine advanced indicators.
    
    This adapter class serves as a bridge between the Feature Store Service's indicator system
    and the advanced technical analysis components in the Analysis Engine Service.
    It wraps advanced indicators and presents them with the BaseIndicator interface.
    """
    category = 'advanced'

    def __init__(self, advanced_indicator_class: Any, name_prefix: str='',
        output_column_mapping: Optional[Dict[str, str]]=None, **params):
        """
        Initialize the advanced indicator adapter.
        
        Args:
            advanced_indicator_class: Class from the analysis engine to adapt
            name_prefix: Optional prefix for output column names
            output_column_mapping: Optional mapping of advanced indicator output columns 
                                  to feature store column names
            **params: Parameters to pass to the advanced indicator constructor
        """
        self.advanced_indicator_class = advanced_indicator_class
        self.advanced_indicator = advanced_indicator_class(**params)
        self.params = params
        self.name_prefix = name_prefix
        self.output_column_mapping = output_column_mapping or {}
        class_name = advanced_indicator_class.__name__
        self.name = f'{name_prefix}{class_name}' if name_prefix else class_name

    @log_and_time
    @graceful_fallback(log_error=True)
    @performance_tracking(threshold_ms=500)
    @with_exception_handling
    def calculate(self, data: pd.DataFrame) ->pd.DataFrame:
        """
        Calculate indicator values using the advanced indicator.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with indicator values added as new columns
        """

        def calculation_function(input_data):
    """
    Calculation function.
    
    Args:
        input_data: Description of input_data
    
    """

            result = input_data.copy()
            if hasattr(self.advanced_indicator, 'calculate'):
                advanced_result = self.advanced_indicator.calculate(input_data)
            elif hasattr(self.advanced_indicator, 'analyze'):
                advanced_result = self.advanced_indicator.analyze(input_data)
            else:
                raise AttributeError(
                    f'Advanced indicator {self.advanced_indicator.__class__.__name__} has no calculate or analyze method'
                    )
            if isinstance(advanced_result, pd.DataFrame):
                new_columns = [col for col in advanced_result.columns if 
                    col not in input_data.columns]
                for col in new_columns:
                    output_col = self.output_column_mapping.get(col, col)
                    if self.name_prefix and not output_col.startswith(self.
                        name_prefix):
                        output_col = f'{self.name_prefix}_{output_col}'
                    result[output_col] = advanced_result[col]
            elif isinstance(advanced_result, dict):
                for key, value in advanced_result.items():
                    output_key = self.output_column_mapping.get(key, key)
                    if self.name_prefix and not output_key.startswith(self.
                        name_prefix):
                        output_key = f'{self.name_prefix}_{output_key}'
                    result[output_key] = value
            elif hasattr(advanced_result, '__iter__') and not isinstance(
                advanced_result, str):
                col_name = self.name.lower()
                result[col_name] = advanced_result
            return result
        try:
            return optimizer.optimize_calculation(data=data, indicator_name
                =self.name, params=self.params, calculation_func=
                calculation_function)
        except Exception as e:
            logger.error(
                f'Error calculating advanced indicator {self.name}: {str(e)}')
            raise AdvancedIndicatorError(
                f'Failed to calculate {self.name}: {str(e)}') from e

    @classmethod
    @log_and_time
    @with_exception_handling
    def create_adapter(cls, advanced_indicator_path: str,
        advanced_indicator_name: str, **params) ->'AdvancedIndicatorAdapter':
        """
        Create an adapter for a specific advanced indicator class.
        
        Args:
            advanced_indicator_path: Import path to the advanced indicator module
            advanced_indicator_name: Name of the advanced indicator class
            **params: Parameters to pass to the advanced indicator constructor
            
        Returns:
            AdvancedIndicatorAdapter instance
        """
        try:
            module = importlib.import_module(advanced_indicator_path)
            advanced_indicator_class = getattr(module, advanced_indicator_name)
            return cls(advanced_indicator_class, **params)
        except ImportError:
            logger.error(f'Could not import module {advanced_indicator_path}')
            raise
        except AttributeError:
            logger.error(
                f'Could not find class {advanced_indicator_name} in module {advanced_indicator_path}'
                )
            raise


@log_and_time
@with_exception_handling
def load_advanced_indicators(advanced_ta_path: Optional[str]=None) ->Dict[
    str, Type]:
    """
    Discover and load advanced indicator classes from the analysis engine.
    
    Args:
        advanced_ta_path: Optional path to the advanced TA module
        
    Returns:
        Dictionary mapping indicator names to their class objects
    """
    if not advanced_ta_path:
        advanced_ta_path = 'analysis_engine.analysis.advanced_ta'
    try:
        advanced_ta = importlib.import_module(advanced_ta_path)
        advanced_indicators = {}
        base_class_names = ['AdvancedAnalysisBase', 'PatternRecognitionBase']
        for submodule_name in dir(advanced_ta):
            if submodule_name.startswith('_'):
                continue
            try:
                submodule = importlib.import_module(
                    f'{advanced_ta_path}.{submodule_name}')
                for name, obj in inspect.getmembers(submodule, inspect.isclass
                    ):
                    is_advanced_indicator = False
                    for base_class_name in base_class_names:
                        if hasattr(submodule, base_class_name) and issubclass(
                            obj, getattr(submodule, base_class_name)):
                            is_advanced_indicator = True
                            break
                    if is_advanced_indicator and not name.startswith('_'):
                        advanced_indicators[name] = obj
            except (ImportError, AttributeError) as e:
                logger.warning(
                    f'Could not import submodule {submodule_name}: {str(e)}')
        return advanced_indicators
    except ImportError:
        logger.error(f'Could not import advanced TA package {advanced_ta_path}'
            )
        return {}
