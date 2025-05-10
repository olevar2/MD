"""
Indicator Adapter Module

This module provides adapters for retrieving pre-calculated technical indicators
from the feature store and applying asset-specific adjustments if necessary.
"""

import logging
from typing import Dict, List, Any, Optional, Union, Callable
import pandas as pd
import numpy as np

from analysis_engine.multi_asset.asset_registry import AssetClass
from analysis_engine.multi_asset.asset_adapter import BaseAssetAdapter
from analysis_engine.adapters.multi_asset_adapter import MultiAssetServiceAdapter
from analysis_engine.services.feature_store_client import FeatureStoreClient

logger = logging.getLogger(__name__)


class IndicatorAdapter:
    """
    Adapter for retrieving and adapting technical indicators

    This class fetches pre-calculated indicators from the feature store service
    and applies asset-specific adjustments if necessary.
    """

    def __init__(
        self,
        multi_asset_service: Optional[MultiAssetServiceAdapter] = None,
        feature_store_client: Optional[FeatureStoreClient] = None
    ):
        """Initialize the indicator adapter"""
        self.multi_asset_service = multi_asset_service or MultiAssetServiceAdapter()
        self.feature_store_client = feature_store_client or FeatureStoreClient()
        self.logger = logging.getLogger(__name__)

    def get_indicator(
        self,
        indicator_type: str,
        symbol: str,
        data: pd.DataFrame = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        Get an indicator from feature store with asset-specific adjustments

        Args:
            indicator_type: Type of indicator ('sma', 'ema', 'rsi', etc.)
            symbol: Symbol for the asset
            data: Optional DataFrame with price data (if not provided, will be fetched)
            **kwargs: Parameters for the indicator

        Returns:
            DataFrame with the requested indicator
        """
        # Get asset information
        asset_info = self.multi_asset_service.get_asset_info(symbol)
        if not asset_info:
            self.logger.warning(f"Asset info not found for {symbol}, using default parameters")
            asset_class = AssetClass.FOREX
        else:
            asset_class = asset_info.get("asset_class")

        # Adjust parameters based on asset class
        adjusted_params = self._adjust_parameters_for_asset(asset_class, indicator_type, kwargs)

        # If data is not provided, fetch it from feature store
        if data is None:
            # Determine required data length based on parameters
            lookback = self._get_required_lookback(indicator_type, adjusted_params)
            data = self.feature_store_client.get_market_data(symbol, lookback=lookback)

        # Check if indicator already exists in the data
        indicator_cols = self._get_indicator_columns(indicator_type, adjusted_params)
        if all(col in data.columns for col in indicator_cols):
            self.logger.debug(f"Indicator {indicator_type} columns already in DataFrame")
            return data

        # Fetch indicator from feature store
        indicator_data = self.feature_store_client.get_indicator(
            indicator_type=indicator_type,
            symbol=symbol,
            params=adjusted_params
        )

        # Merge with original data if both exist
        if indicator_data is not None and len(indicator_data) > 0:
            # Use merge or join based on index type
            if isinstance(data.index, pd.DatetimeIndex) and isinstance(indicator_data.index, pd.DatetimeIndex):
                result = data.join(indicator_data, how='left')
            else:
                # Try to merge on timestamp or other common column
                merge_cols = [col for col in ['timestamp', 'date', 'time']
                             if col in data.columns and col in indicator_data.columns]
                if merge_cols:
                    result = pd.merge(data, indicator_data, on=merge_cols, how='left')
                else:
                    self.logger.warning(f"Could not determine how to merge data frames for {indicator_type}")
                    # Concatenate columns directly if indexes match
                    if len(data) == len(indicator_data):
                        for col in indicator_cols:
                            if col in indicator_data.columns and col not in data.columns:
                                data[col] = indicator_data[col].values
                    result = data

            return result
        else:
            self.logger.warning(f"Could not fetch {indicator_type} from feature store")
            return data

    def moving_average(
        self,
        data: pd.DataFrame,
        symbol: str,
        column: str = 'close',
        period: int = 20,
        ma_type: str = 'sma'
    ) -> pd.DataFrame:
        """
        Get moving average from feature store with asset-specific adjustments

        Args:
            data: Price data DataFrame
            symbol: Symbol for the asset
            column: Column to calculate MA on
            period: Period for the moving average
            ma_type: Type of moving average ('sma', 'ema', 'wma')

        Returns:
            DataFrame with moving average added
        """
        # Adjust period based on asset class
        asset_info = self.multi_asset_service.get_asset_info(symbol)
        if not asset_info:
            asset_class = AssetClass.FOREX
        else:
            asset_class = asset_info.get("asset_class")

        adjusted_period = self._adjust_ma_period_for_asset(asset_class, period)

        # Get the MA from feature store
        params = {
            "window": adjusted_period,
            "price_column": column,
            "ma_type": ma_type
        }

        return self.get_indicator(ma_type, symbol, data, **params)

    def relative_strength_index(
        self,
        data: pd.DataFrame,
        symbol: str,
        column: str = 'close',
        period: int = 14
    ) -> pd.DataFrame:
        """
        Get RSI from feature store with asset-specific adjustments

        Args:
            data: Price data DataFrame
            symbol: Symbol for the asset
            column: Column to calculate RSI on
            period: Period for RSI

        Returns:
            DataFrame with RSI added
        """
        # Get adjusted period based on asset class
        asset_info = self.multi_asset_service.get_asset_info(symbol)
        if not asset_info:
            asset_class = AssetClass.FOREX
        else:
            asset_class = asset_info.get("asset_class")

        adjusted_period = self._adjust_rsi_period_for_asset(asset_class, period)

        # Get RSI from feature store
        params = {
            "window": adjusted_period,
            "price_column": column
        }

        return self.get_indicator("rsi", symbol, data, **params)

    def bollinger_bands(
        self,
        data: pd.DataFrame,
        symbol: str,
        column: str = 'close',
        period: int = 20,
        std_dev: float = 2.0
    ) -> pd.DataFrame:
        """
        Get Bollinger Bands from feature store with asset-specific adjustments

        Args:
            data: Price data DataFrame
            symbol: Symbol for the asset
            column: Column to calculate Bollinger Bands on
            period: Period for moving average
            std_dev: Number of standard deviations

        Returns:
            DataFrame with Bollinger Bands added
        """
        # Get asset information
        asset_info = self.multi_asset_service.get_asset_info(symbol)
        if not asset_info:
            asset_class = AssetClass.FOREX
            std_dev_multiplier = 1.0
        else:
            asset_class = asset_info.get("asset_class")
            # Adjust standard deviation multiplier based on asset class
            std_dev_multiplier = self._get_std_dev_multiplier(asset_class)

        # Adjust period and standard deviation
        adjusted_period = self._adjust_ma_period_for_asset(asset_class, period)
        adjusted_std_dev = std_dev * std_dev_multiplier

        # Get Bollinger Bands from feature store
        params = {
            "window": adjusted_period,
            "price_column": column,
            "std_dev": adjusted_std_dev
        }

        return self.get_indicator("bollinger_bands", symbol, data, **params)

    def average_true_range(
        self,
        data: pd.DataFrame,
        symbol: str,
        period: int = 14
    ) -> pd.DataFrame:
        """
        Get ATR from feature store with asset-specific adjustments

        Args:
            data: Price data DataFrame with high, low, close columns
            symbol: Symbol for the asset
            period: Period for ATR

        Returns:
            DataFrame with ATR added
        """
        # Get asset information
        asset_info = self.multi_asset_service.get_asset_info(symbol)
        if not asset_info:
            asset_class = AssetClass.FOREX
        else:
            asset_class = asset_info.get("asset_class")

        # Get asset adapter for appropriate normalization
        adapter = self.multi_asset_service.get_adapter(symbol)

        # Adjust period based on asset class
        adjusted_period = self._adjust_atr_period_for_asset(asset_class, period)

        # Get ATR from feature store
        params = {
            "period": adjusted_period
        }

        result = self.get_indicator("atr", symbol, data, **params)

        # Add normalized ATR appropriate to the asset class if not already present
        atr_col = f"atr_{adjusted_period}"

        if atr_col in result.columns:
            if asset_class == AssetClass.FOREX:
                # For forex, normalize to pips if not already present
                pip_col = f"atr_pips_{adjusted_period}"
                if pip_col not in result.columns:
                    pip_factor = asset_info.get("trading_parameters", {}).get("pip_value", 0.0001)
                    if pip_factor > 0:
                        result[pip_col] = result[atr_col] / pip_factor
            elif asset_class in [AssetClass.CRYPTO, AssetClass.STOCKS]:
                # For crypto and stocks, express as percentage of price if not already present
                pct_col = f"atr_pct_{adjusted_period}"
                if pct_col not in result.columns:
                    result[pct_col] = (result[atr_col] / result["close"]) * 100

        return result

    def macd(
        self,
        data: pd.DataFrame,
        symbol: str,
        column: str = 'close',
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9
    ) -> pd.DataFrame:
        """
        Get MACD from feature store with asset-specific adjustments

        Args:
            data: Price data DataFrame
            symbol: Symbol for the asset
            column: Column to calculate MACD on
            fast_period: Fast EMA period
            slow_period: Slow EMA period
            signal_period: Signal line period

        Returns:
            DataFrame with MACD added
        """
        # Get asset information
        asset_info = self.multi_asset_service.get_asset_info(symbol)
        if not asset_info:
            asset_class = AssetClass.FOREX
        else:
            asset_class = asset_info.get("asset_class")

        # Adjust periods based on asset class
        adjusted_fast = self._adjust_macd_fast_period(asset_class, fast_period)
        adjusted_slow = self._adjust_macd_slow_period(asset_class, slow_period)
        adjusted_signal = self._adjust_macd_signal_period(asset_class, signal_period)

        # Get MACD from feature store
        params = {
            "price_column": column,
            "fast_period": adjusted_fast,
            "slow_period": adjusted_slow,
            "signal_period": adjusted_signal
        }

        return self.get_indicator("macd", symbol, data, **params)

    def _adjust_parameters_for_asset(
        self,
        asset_class: AssetClass,
        indicator_type: str,
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Adjust indicator parameters based on asset class"""
        adjusted_params = params.copy()

        # Apply adjustment based on indicator type
        if indicator_type in ['sma', 'ema', 'wma']:
            if 'window' in adjusted_params:
                adjusted_params['window'] = self._adjust_ma_period_for_asset(
                    asset_class, adjusted_params['window']
                )
        elif indicator_type == 'rsi':
            if 'window' in adjusted_params:
                adjusted_params['window'] = self._adjust_rsi_period_for_asset(
                    asset_class, adjusted_params['window']
                )
        elif indicator_type == 'atr':
            if 'period' in adjusted_params:
                adjusted_params['period'] = self._adjust_atr_period_for_asset(
                    asset_class, adjusted_params['period']
                )
        elif indicator_type == 'macd':
            if 'fast_period' in adjusted_params:
                adjusted_params['fast_period'] = self._adjust_macd_fast_period(
                    asset_class, adjusted_params['fast_period']
                )
            if 'slow_period' in adjusted_params:
                adjusted_params['slow_period'] = self._adjust_macd_slow_period(
                    asset_class, adjusted_params['slow_period']
                )
            if 'signal_period' in adjusted_params:
                adjusted_params['signal_period'] = self._adjust_macd_signal_period(
                    asset_class, adjusted_params['signal_period']
                )
        elif indicator_type == 'bollinger_bands':
            if 'window' in adjusted_params:
                adjusted_params['window'] = self._adjust_ma_period_for_asset(
                    asset_class, adjusted_params['window']
                )
            if 'std_dev' in adjusted_params:
                adjusted_params['std_dev'] = adjusted_params['std_dev'] * self._get_std_dev_multiplier(asset_class)

        return adjusted_params

    def _get_indicator_columns(self, indicator_type: str, params: Dict[str, Any]) -> List[str]:
        """Get the expected column names for an indicator based on its type and params"""
        if indicator_type == 'sma':
            return [f"sma_{params.get('window', 20)}"]
        elif indicator_type == 'ema':
            return [f"ema_{params.get('span', 20)}"]
        elif indicator_type == 'wma':
            return [f"wma_{params.get('window', 20)}"]
        elif indicator_type == 'rsi':
            return [f"rsi_{params.get('window', 14)}"]
        elif indicator_type == 'bollinger_bands':
            window = params.get('window', 20)
            return [
                f"bb_middle_{window}",
                f"bb_upper_{window}",
                f"bb_lower_{window}"
            ]
        elif indicator_type == 'atr':
            period = params.get('period', 14)
            return [f"atr_{period}"]
        elif indicator_type == 'macd':
            # Standard MACD column names
            return ["macd_line", "macd_signal", "macd_histogram"]
        else:
            # For unknown indicators, return empty list
            return []

    def _get_required_lookback(self, indicator_type: str, params: Dict[str, Any]) -> int:
        """
        Get the required lookback period for an indicator based on its type and params

        Returns:
            Number of periods needed for calculation (adding buffer for warm-up)
        """
        # Base lookback on the largest parameter plus a buffer
        buffer = 50  # Extra buffer for warm-up

        if indicator_type in ['sma', 'ema', 'wma']:
            return params.get('window', 20) + buffer
        elif indicator_type == 'rsi':
            return params.get('window', 14) * 2 + buffer
        elif indicator_type == 'bollinger_bands':
            return params.get('window', 20) + buffer
        elif indicator_type == 'atr':
            return params.get('period', 14) + buffer
        elif indicator_type == 'macd':
            slow = params.get('slow_period', 26)
            signal = params.get('signal_period', 9)
            return slow + signal + buffer
        else:
            # Default lookback for unknown indicators
            return 100

    def _adjust_ma_period_for_asset(self, asset_class: AssetClass, period: int) -> int:
        """Adjust moving average period based on asset class"""
        if asset_class == AssetClass.FOREX:
            return period  # Standard MA periods work well for forex
        elif asset_class == AssetClass.CRYPTO:
            return int(period * 1.2)  # Slightly longer for crypto due to volatility
        elif asset_class == AssetClass.STOCKS:
            return period  # Standard for stocks
        else:
            return period  # Default

    def _adjust_rsi_period_for_asset(self, asset_class: AssetClass, period: int) -> int:
        """Adjust RSI period based on asset class"""
        if asset_class == AssetClass.FOREX:
            return period  # Standard RSI period works well for forex
        elif asset_class == AssetClass.CRYPTO:
            return max(10, int(period * 0.8))  # Shorter for crypto to capture faster moves
        elif asset_class == AssetClass.STOCKS:
            return period  # Standard for stocks
        else:
            return period  # Default

    def _adjust_atr_period_for_asset(self, asset_class: AssetClass, period: int) -> int:
        """Adjust ATR period based on asset class"""
        if asset_class == AssetClass.FOREX:
            return period  # Standard period works well for forex
        elif asset_class == AssetClass.CRYPTO:
            return max(10, int(period * 0.8))  # Shorter for crypto
        elif asset_class == AssetClass.STOCKS:
            return period  # Standard for stocks
        else:
            return period  # Default

    def _adjust_macd_fast_period(self, asset_class: AssetClass, period: int) -> int:
        """Adjust MACD fast period based on asset class"""
        if asset_class == AssetClass.CRYPTO:
            return max(8, int(period * 0.9))  # Slightly faster for crypto
        else:
            return period  # Standard for other assets

    def _adjust_macd_slow_period(self, asset_class: AssetClass, period: int) -> int:
        """Adjust MACD slow period based on asset class"""
        if asset_class == AssetClass.CRYPTO:
            return int(period * 0.9)  # Slightly faster for crypto
        else:
            return period  # Standard for other assets

    def _adjust_macd_signal_period(self, asset_class: AssetClass, period: int) -> int:
        """Adjust MACD signal period based on asset class"""
        return period  # Same for all asset classes

    def _get_std_dev_multiplier(self, asset_class: AssetClass) -> float:
        """Get standard deviation multiplier for Bollinger Bands based on asset class"""
        if asset_class == AssetClass.FOREX:
            return 1.0  # Standard
        elif asset_class == AssetClass.CRYPTO:
            return 1.3  # Wider bands for higher volatility
        elif asset_class == AssetClass.STOCKS:
            return 1.0  # Standard
        else:
            return 1.0  # Default
