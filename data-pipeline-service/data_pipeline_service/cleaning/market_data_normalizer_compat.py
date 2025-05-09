"""
Backward Compatibility Module for Market Data Normalizer

This module provides backward compatibility for code that uses the original
MarketDataNormalizer class from the cleaning package.
"""

import warnings
from typing import Dict, List, Any
import pandas as pd

from data_pipeline_service.transformers.market_data_transformer import MarketDataTransformer
from data_pipeline_service.adapters.multi_asset_adapter import MultiAssetServiceAdapter


class MarketDataNormalizer:
    """
    Backward compatibility class for the original MarketDataNormalizer.
    
    This class provides the same interface as the original MarketDataNormalizer class
    but delegates to the new MarketDataTransformer.
    """
    
    def __init__(self, multi_asset_service: MultiAssetServiceAdapter = None):
        """
        Initialize the backward compatibility class.
        
        Args:
            multi_asset_service: Service for retrieving asset information
        """
        warnings.warn(
            "MarketDataNormalizer from cleaning package is deprecated and will be removed in a future version. "
            "Use MarketDataTransformer from transformers package instead.",
            DeprecationWarning,
            stacklevel=2
        )
        self.transformer = MarketDataTransformer(multi_asset_service)
    
    def normalize(self, data: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        Backward compatibility method for normalize.
        
        Args:
            data: Market data DataFrame
            symbol: Asset symbol
            
        Returns:
            Normalized market data
        """
        return self.transformer.transform(data, symbol, operations=["normalize"])
    
    def normalize_with_features(self, data: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        Backward compatibility method for normalize_with_features.
        
        Args:
            data: Market data DataFrame
            symbol: Asset symbol
            
        Returns:
            Normalized market data with features
        """
        return self.transformer.transform(data, symbol, operations=["normalize", "feature_generation"])
    
    def normalize_statistics(self, stats_dict: Dict[str, Any], symbol: str) -> Dict[str, Any]:
        """
        Backward compatibility method for normalize_statistics.
        
        Args:
            stats_dict: Statistics dictionary
            symbol: Asset symbol
            
        Returns:
            Normalized statistics
        """
        return self.transformer.transform_statistics(stats_dict, symbol)