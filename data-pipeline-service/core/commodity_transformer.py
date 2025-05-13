"""
Commodity Market Data Transformer

This module provides specialized transformation for commodity market data.
"""

import logging
from typing import Dict, List, Any, Optional
import pandas as pd

from ..base_transformer import BaseMarketDataTransformer
from ..operations.normalization import DataNormalizer

logger = logging.getLogger(__name__)


class CommodityTransformer(BaseMarketDataTransformer):
    """
    Specialized transformer for commodity market data.
    
    This transformer applies commodity-specific transformations such as
    seasonality analysis, inventory impact, and other commodity-specific features.
    """
    
    def __init__(self, parameters: Dict[str, Any] = None):
        """
        Initialize the commodity transformer.
        
        Args:
            parameters: Configuration parameters for the transformer
        """
        super().__init__("commodity_transformer", parameters)
        self.normalizer = DataNormalizer()
    
    def transform(self, data: pd.DataFrame, asset_info: Dict[str, Any] = None) -> pd.DataFrame:
        """
        Transform commodity market data.
        
        Args:
            data: Market data DataFrame
            asset_info: Asset information
            
        Returns:
            Transformed commodity market data
        """
        # Start with generic normalization
        normalized = self.normalizer.transform(data)
        
        # Add commodity-specific features
        if 'timestamp' in normalized.columns:
            # Add seasonality indicators
            normalized['month'] = normalized['timestamp'].dt.month
            normalized['quarter'] = normalized['timestamp'].dt.quarter
            
            # Flag common reporting days
            normalized['is_report_day'] = self._is_report_day(
                normalized['timestamp'], 
                asset_info.get('commodity_type') if asset_info else None
            )
        
        # Add contango/backwardation indicators if futures data available
        if asset_info and 'futures_curve' in asset_info:
            # This would require additional data not in the main DataFrame
            # In a real implementation, this would fetch the futures curve data
            pass
        
        return normalized
    
    def transform_statistics(self, stats_dict: Dict[str, Any], 
                           asset_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Transform commodity statistics.
        
        Args:
            stats_dict: Statistics dictionary
            asset_info: Asset information
            
        Returns:
            Transformed statistics
        """
        # Create a copy to avoid modifying the original
        transformed = stats_dict.copy()
        
        # Add commodity-specific metrics
        commodity_type = asset_info.get('commodity_type') if asset_info else None
        
        if commodity_type == 'energy':
            # Energy-specific metrics
            if 'volatility' in transformed:
                transformed['energy_volatility_index'] = transformed['volatility'] * 1.5  # Example calculation
        
        elif commodity_type == 'metals':
            # Metals-specific metrics
            if 'correlation_with_dollar' in transformed:
                transformed['dollar_sensitivity'] = -transformed['correlation_with_dollar'] * 100
        
        elif commodity_type == 'agriculture':
            # Agriculture-specific metrics
            if 'seasonality' in transformed:
                transformed['seasonal_strength'] = transformed['seasonality'] * 100
        
        return transformed
    
    def _is_report_day(self, timestamps: pd.Series, commodity_type: Optional[str]) -> pd.Series:
        """
        Determine if each timestamp is a commodity report day.
        
        Args:
            timestamps: Series of timestamps
            commodity_type: Type of commodity
            
        Returns:
            Boolean series indicating report days
        """
        # Initialize all as False
        is_report_day = pd.Series(False, index=timestamps.index)
        
        if commodity_type == 'energy':
            # EIA Petroleum Status Report - Wednesdays
            is_report_day = is_report_day | (timestamps.dt.dayofweek == 2)
            
        elif commodity_type == 'agriculture':
            # USDA Reports - typically 8-12th of month
            is_report_day = is_report_day | ((timestamps.dt.day >= 8) & (timestamps.dt.day <= 12))
            
        elif commodity_type == 'metals':
            # Commitment of Traders - Fridays
            is_report_day = is_report_day | (timestamps.dt.dayofweek == 4)
        
        return is_report_day