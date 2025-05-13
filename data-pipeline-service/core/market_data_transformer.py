"""
Market Data Transformer

This module provides the main transformer for market data across different asset classes.
"""

import logging
from typing import Dict, List, Any, Optional, Union
import pandas as pd

from common_lib.multi_asset.interfaces import AssetClassEnum as AssetClass
from adapters.multi_asset_adapter import MultiAssetServiceAdapter

from .base_transformer import BaseMarketDataTransformer
from .asset_specific.forex_transformer import ForexTransformer
from .asset_specific.crypto_transformer import CryptoTransformer
from .asset_specific.stock_transformer import StockTransformer
from .asset_specific.commodity_transformer import CommodityTransformer
from .asset_specific.index_transformer import IndexTransformer
from .operations.normalization import DataNormalizer

logger = logging.getLogger(__name__)


class MarketDataTransformer(BaseMarketDataTransformer):
    """
    Main transformer for market data across different asset classes.
    
    This class orchestrates the transformation of market data by delegating to
    specialized transformers based on asset class and operation type.
    """
    
    def __init__(self, multi_asset_service: Optional[MultiAssetServiceAdapter] = None,
               parameters: Dict[str, Any] = None):
        """
        Initialize the market data transformer.
        
        Args:
            multi_asset_service: Service for retrieving asset information
            parameters: Configuration parameters for the transformer
        """
        super().__init__("market_data_transformer", parameters)
        self.multi_asset_service = multi_asset_service or MultiAssetServiceAdapter()
        
        # Initialize specialized transformers
        self.forex_transformer = ForexTransformer()
        self.crypto_transformer = CryptoTransformer()
        self.stock_transformer = StockTransformer()
        self.commodity_transformer = CommodityTransformer()
        self.index_transformer = IndexTransformer()
        
        # Initialize operation transformers
        self.normalizer = DataNormalizer()
    
    def transform(self, data: pd.DataFrame, symbol: str, operations: List[str] = None) -> pd.DataFrame:
        """
        Transform market data based on asset class and requested operations.
        
        Args:
            data: Market data DataFrame
            symbol: Asset symbol
            operations: List of operations to perform (default: ["normalize"])
            
        Returns:
            Transformed market data
        """
        # Default operations
        if operations is None:
            operations = ["normalize"]
        
        # Get asset information
        asset_info = self.multi_asset_service.get_asset_info(symbol)
        if not asset_info:
            self.logger.warning(f"Asset info not found for {symbol}, using generic transformation")
            return self.normalizer.transform(data)
        
        # Get asset class
        asset_class = asset_info.get("asset_class")
        
        # Apply asset-specific transformation
        transformed_data = self._transform_by_asset_class(data, asset_class, asset_info)
        
        # Apply additional operations
        for operation in operations:
            if operation == "normalize":
                # Already applied in asset-specific transformation
                continue
            elif operation == "feature_generation":
                # Apply feature generation
                from .operations.feature_generation import FeatureGenerator
                feature_generator = FeatureGenerator()
                transformed_data = feature_generator.transform(transformed_data)
            elif operation == "statistical":
                # Apply statistical transformations
                from .operations.statistical_transforms import StatisticalTransformer
                statistical_transformer = StatisticalTransformer()
                transformed_data = statistical_transformer.transform(transformed_data)
            else:
                self.logger.warning(f"Unknown operation: {operation}")
        
        return transformed_data
    
    def _transform_by_asset_class(self, data: pd.DataFrame, asset_class: Any, 
                                asset_info: Dict[str, Any]) -> pd.DataFrame:
        """
        Transform data based on asset class.
        
        Args:
            data: Market data DataFrame
            asset_class: Asset class
            asset_info: Asset information
            
        Returns:
            Transformed market data
        """
        if asset_class == AssetClass.FOREX:
            return self.forex_transformer.transform(data, asset_info=asset_info)
        elif asset_class == AssetClass.CRYPTO:
            return self.crypto_transformer.transform(data, asset_info=asset_info)
        elif asset_class == AssetClass.STOCKS:
            return self.stock_transformer.transform(data, asset_info=asset_info)
        elif asset_class == AssetClass.COMMODITIES:
            return self.commodity_transformer.transform(data, asset_info=asset_info)
        elif asset_class == AssetClass.INDICES:
            return self.index_transformer.transform(data, asset_info=asset_info)
        else:
            # Generic normalization for unknown asset classes
            return self.normalizer.transform(data)
    
    def transform_statistics(self, stats_dict: Dict[str, Any], symbol: str) -> Dict[str, Any]:
        """
        Transform asset statistics for consistent reporting.
        
        Args:
            stats_dict: Statistics dictionary
            symbol: Asset symbol
            
        Returns:
            Transformed statistics
        """
        # Get asset information
        asset_info = self.multi_asset_service.get_asset_info(symbol)
        if not asset_info:
            return stats_dict
        
        # Get asset class
        asset_class = asset_info.get("asset_class")
        
        # Create a copy to avoid modifying the original
        transformed = stats_dict.copy()
        
        # Apply asset-specific transformations
        if asset_class == AssetClass.FOREX:
            transformed = self.forex_transformer.transform_statistics(transformed, asset_info)
        elif asset_class == AssetClass.CRYPTO:
            transformed = self.crypto_transformer.transform_statistics(transformed, asset_info)
        elif asset_class == AssetClass.STOCKS:
            transformed = self.stock_transformer.transform_statistics(transformed, asset_info)
        elif asset_class == AssetClass.COMMODITIES:
            transformed = self.commodity_transformer.transform_statistics(transformed, asset_info)
        elif asset_class == AssetClass.INDICES:
            transformed = self.index_transformer.transform_statistics(transformed, asset_info)
        
        return transformed