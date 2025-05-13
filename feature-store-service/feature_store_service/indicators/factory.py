"""
Indicator Factory Module

This module provides a factory for creating technical indicators,
supporting a wide range of standard and custom indicators for trading analysis.
"""
import logging
from typing import Dict, Any, Type, Optional
import pandas as pd
from feature_store_service.indicators.base_indicator import BaseIndicator
from feature_store_service.indicators.moving_averages import SimpleMovingAverage as SMA, ExponentialMovingAverage as EMA, WeightedMovingAverage as WMA
from feature_store_service.indicators.oscillators import RelativeStrengthIndex as RSI, MACD, Stochastic, CommodityChannelIndex as CCI, WilliamsR as Williams_R, RateOfChange as ROC
from feature_store_service.indicators.volatility import BollingerBands, AverageTrueRange as ATR, HistoricalVolatility as StandardDeviation
from feature_store_service.indicators.volume import OnBalanceVolume as OBV, VolumeWeightedAveragePrice as VWAP, ChaikinMoneyFlow as MoneyFlow
from feature_store_service.indicators.trend import AverageDirectionalIndex as ADX, Supertrend, ParabolicSAR
INDICATOR_REGISTRY = {'sma': SMA, 'ema': EMA, 'wma': WMA, 'vwap': VWAP,
    'rsi': RSI, 'macd': MACD, 'stochastic': Stochastic, 'bollinger_bands':
    BollingerBands, 'atr': ATR, 'std_dev': StandardDeviation, 'obv': OBV,
    'vwap': VWAP, 'mfi': MoneyFlow, 'adx': ADX, 'supertrend': Supertrend,
    'parabolic_sar': ParabolicSAR, 'cci': CCI, 'williams_r': Williams_R,
    'roc': ROC}


from feature_store_service.error.exceptions_bridge import (
    with_exception_handling,
    async_with_exception_handling,
    ForexTradingPlatformError,
    ServiceError,
    DataError,
    ValidationError
)

@with_exception_handling
def create_indicator(indicator_name: str, parameters: Dict[str, Any]
    ) ->BaseIndicator:
    """
    Factory function to create technical indicator instances
    
    Args:
        indicator_name: Name of the indicator to create (case-insensitive)
        parameters: Dictionary of parameters for the indicator
        
    Returns:
        Instance of the requested indicator
        
    Raises:
        ValueError: If the indicator name is not recognized
    """
    logger = logging.getLogger(__name__)
    indicator_key = indicator_name.lower()
    if indicator_key not in INDICATOR_REGISTRY:
        logger.error(f'Unknown indicator: {indicator_name}')
        raise ValueError(f"Indicator '{indicator_name}' is not supported")
    indicator_class = INDICATOR_REGISTRY[indicator_key]
    try:
        indicator = indicator_class(**parameters)
        logger.debug(
            f'Created {indicator_name} indicator with parameters: {parameters}'
            )
        return indicator
    except Exception as e:
        logger.error(f'Error creating {indicator_name}: {str(e)}')
        raise ValueError(
            f"Failed to create indicator '{indicator_name}': {str(e)}")


def register_custom_indicator(name: str, indicator_class: Type[BaseIndicator]):
    """
    Register a custom indicator class
    
    Args:
    """
    Args class.
    
    Attributes:
        Add attributes here
    """

        name: Name for the indicator (will be converted to lowercase)
        indicator_class: Class implementing the BaseIndicator interface
        
    Raises:
        ValueError: If an indicator with this name already exists
    """
    logger = logging.getLogger(__name__)
    indicator_key = name.lower()
    if indicator_key in INDICATOR_REGISTRY:
        logger.warning(f'Indicator {name} already exists, will be overwritten')
    INDICATOR_REGISTRY[indicator_key] = indicator_class
    logger.info(f'Registered custom indicator: {name}')


def get_available_indicators() ->Dict[str, Dict[str, Any]]:
    """
    Get information about all available indicators
    
    Returns:
        Dictionary mapping indicator names to their metadata
    """
    result = {}
    for name, indicator_class in INDICATOR_REGISTRY.items():
        metadata = getattr(indicator_class, 'metadata', {})
        if not metadata:
            metadata = {'name': name, 'description': getattr(
                indicator_class, '__doc__', 'No description available'),
                'parameters': getattr(indicator_class, 'default_params', {})}
        result[name] = metadata
    return result
