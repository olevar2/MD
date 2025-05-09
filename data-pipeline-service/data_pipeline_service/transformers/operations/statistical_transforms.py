"""
Statistical Transformation Operations

This module provides operations for statistical transformations of market data.
"""

import logging
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
from scipy import stats

from ..base_transformer import BaseMarketDataTransformer

logger = logging.getLogger(__name__)


class StatisticalTransformer(BaseMarketDataTransformer):
    """
    Transformer for statistical transformations of market data.
    
    This transformer applies statistical operations such as normalization,
    standardization, outlier detection, and other statistical transformations.
    """
    
    def __init__(self, parameters: Dict[str, Any] = None):
        """
        Initialize the statistical transformer.
        
        Args:
            parameters: Configuration parameters for the transformer
        """
        default_params = {
            "z_score_window": 20,
            "percentile_window": 50,
            "outlier_threshold": 3.0,  # Z-score threshold for outliers
            "handle_outliers": "clip",  # "clip", "remove", or "none"
            "rolling_stats_window": 20
        }
        
        # Merge default parameters with provided parameters
        merged_params = {**default_params, **(parameters or {})}
        super().__init__("statistical_transformer", merged_params)
    
    def transform(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Apply statistical transformations to market data.
        
        Args:
            data: Market data DataFrame
            **kwargs: Additional arguments for transformation
            
        Returns:
            Statistically transformed market data
        """
        # Create a copy to avoid modifying the original
        transformed = data.copy()
        
        # Check if we have the necessary price data
        if 'close' not in transformed.columns:
            self.logger.warning("Close price not available, skipping statistical transformations")
            return transformed
        
        # Calculate Z-scores
        self._calculate_z_scores(transformed)
        
        # Calculate percentile ranks
        self._calculate_percentile_ranks(transformed)
        
        # Handle outliers if requested
        if self.parameters["handle_outliers"] != "none":
            self._handle_outliers(transformed)
        
        # Calculate rolling statistics
        self._calculate_rolling_statistics(transformed)
        
        # Calculate return statistics
        if 'log_return' in transformed.columns:
            self._calculate_return_statistics(transformed)
        
        return transformed
    
    def get_required_columns(self) -> List[str]:
        """
        Get the list of required columns for this transformer.
        
        Returns:
            List of required column names
        """
        # Minimum required columns for statistical transformations
        return ["close"]
    
    def _calculate_z_scores(self, data: pd.DataFrame):
        """
        Calculate Z-scores for numeric columns.
        
        Args:
            data: Market data DataFrame
        """
        window = self.parameters["z_score_window"]
        
        # Calculate Z-scores for price
        if 'close' in data.columns:
            data['close_mean'] = data['close'].rolling(window=window).mean()
            data['close_std'] = data['close'].rolling(window=window).std()
            data['close_z_score'] = (data['close'] - data['close_mean']) / data['close_std']
        
        # Calculate Z-scores for returns
        if 'log_return' in data.columns:
            data['return_mean'] = data['log_return'].rolling(window=window).mean()
            data['return_std'] = data['log_return'].rolling(window=window).std()
            data['return_z_score'] = (data['log_return'] - data['return_mean']) / data['return_std']
        
        # Calculate Z-scores for volume
        if 'volume' in data.columns:
            data['volume_mean'] = data['volume'].rolling(window=window).mean()
            data['volume_std'] = data['volume'].rolling(window=window).std()
            data['volume_z_score'] = (data['volume'] - data['volume_mean']) / data['volume_std']
    
    def _calculate_percentile_ranks(self, data: pd.DataFrame):
        """
        Calculate percentile ranks for numeric columns.
        
        Args:
            data: Market data DataFrame
        """
        window = self.parameters["percentile_window"]
        
        # Calculate percentile ranks for price
        if 'close' in data.columns:
            data['close_rank'] = data['close'].rolling(window=window).apply(
                lambda x: stats.percentileofscore(x, x.iloc[-1]) / 100,
                raw=True
            )
        
        # Calculate percentile ranks for returns
        if 'log_return' in data.columns:
            data['return_rank'] = data['log_return'].rolling(window=window).apply(
                lambda x: stats.percentileofscore(x, x.iloc[-1]) / 100,
                raw=True
            )
        
        # Calculate percentile ranks for volume
        if 'volume' in data.columns:
            data['volume_rank'] = data['volume'].rolling(window=window).apply(
                lambda x: stats.percentileofscore(x, x.iloc[-1]) / 100,
                raw=True
            )
    
    def _handle_outliers(self, data: pd.DataFrame):
        """
        Handle outliers in the data.
        
        Args:
            data: Market data DataFrame
        """
        threshold = self.parameters["outlier_threshold"]
        method = self.parameters["handle_outliers"]
        
        # Handle outliers in returns
        if 'return_z_score' in data.columns:
            if method == "clip":
                # Clip outliers to threshold
                data['log_return'] = np.where(
                    data['return_z_score'] > threshold,
                    data['return_mean'] + threshold * data['return_std'],
                    np.where(
                        data['return_z_score'] < -threshold,
                        data['return_mean'] - threshold * data['return_std'],
                        data['log_return']
                    )
                )
            elif method == "remove":
                # Set outliers to NaN
                data.loc[abs(data['return_z_score']) > threshold, 'log_return'] = np.nan
        
        # Handle outliers in volume
        if 'volume_z_score' in data.columns:
            if method == "clip":
                # Clip outliers to threshold
                data['volume'] = np.where(
                    data['volume_z_score'] > threshold,
                    data['volume_mean'] + threshold * data['volume_std'],
                    np.where(
                        data['volume_z_score'] < -threshold,
                        data['volume_mean'] - threshold * data['volume_std'],
                        data['volume']
                    )
                )
            elif method == "remove":
                # Set outliers to NaN
                data.loc[abs(data['volume_z_score']) > threshold, 'volume'] = np.nan
    
    def _calculate_rolling_statistics(self, data: pd.DataFrame):
        """
        Calculate rolling statistics for numeric columns.
        
        Args:
            data: Market data DataFrame
        """
        window = self.parameters["rolling_stats_window"]
        
        # Calculate rolling statistics for price
        if 'close' in data.columns:
            # Skewness
            data['close_skew'] = data['close'].rolling(window=window).skew()
            
            # Kurtosis
            data['close_kurt'] = data['close'].rolling(window=window).kurt()
            
            # Min and max
            data['close_min'] = data['close'].rolling(window=window).min()
            data['close_max'] = data['close'].rolling(window=window).max()
            
            # Range
            data['close_range'] = data['close_max'] - data['close_min']
            data['close_range_pct'] = data['close_range'] / data['close_min'] * 100
        
        # Calculate rolling statistics for volume
        if 'volume' in data.columns:
            # Skewness
            data['volume_skew'] = data['volume'].rolling(window=window).skew()
            
            # Kurtosis
            data['volume_kurt'] = data['volume'].rolling(window=window).kurt()
            
            # Min and max
            data['volume_min'] = data['volume'].rolling(window=window).min()
            data['volume_max'] = data['volume'].rolling(window=window).max()
            
            # Range
            data['volume_range'] = data['volume_max'] - data['volume_min']
            data['volume_range_pct'] = data['volume_range'] / data['volume_min'] * 100
    
    def _calculate_return_statistics(self, data: pd.DataFrame):
        """
        Calculate return statistics.
        
        Args:
            data: Market data DataFrame
        """
        window = self.parameters["rolling_stats_window"]
        
        # Calculate return statistics
        if 'log_return' in data.columns:
            # Volatility (annualized)
            data['volatility'] = data['log_return'].rolling(window=window).std() * np.sqrt(252)
            
            # Sharpe ratio (simplified, assuming zero risk-free rate)
            data['sharpe_ratio'] = (data['log_return'].rolling(window=window).mean() / 
                                  data['log_return'].rolling(window=window).std()) * np.sqrt(252)
            
            # Downside deviation
            data['downside_deviation'] = data['log_return'].rolling(window=window).apply(
                lambda x: np.sqrt(np.mean(np.minimum(x, 0) ** 2)),
                raw=True
            )
            
            # Sortino ratio (simplified, assuming zero risk-free rate)
            data['sortino_ratio'] = (data['log_return'].rolling(window=window).mean() / 
                                   data['downside_deviation']) * np.sqrt(252)
            
            # Maximum drawdown
            data['cumulative_return'] = (1 + data['log_return']).cumprod()
            data['rolling_max'] = data['cumulative_return'].rolling(window=window, min_periods=1).max()
            data['drawdown'] = (data['cumulative_return'] / data['rolling_max']) - 1
            data['max_drawdown'] = data['drawdown'].rolling(window=window).min()