"""
Multi-Asset Data Validation Module

This module provides validation rules and normalizers for different asset classes.
"""

from typing import Dict, List, Any, Optional, Union, Tuple
import logging
import re
from datetime import datetime
import pandas as pd
import numpy as np

from analysis_engine.multi_asset.asset_registry import AssetClass, AssetRegistry
from analysis_engine.services.multi_asset_service import MultiAssetService

logger = logging.getLogger(__name__)


class MultiAssetValidator:
    """
    Validator for multi-asset data inputs
    
    This class provides validation methods specific to each asset class,
    ensuring data quality and consistency across the platform.
    """
    
    def __init__(self, multi_asset_service: Optional[MultiAssetService] = None):
        """Initialize the multi-asset validator"""
        self.multi_asset_service = multi_asset_service or MultiAssetService()
        self.logger = logging.getLogger(__name__)
        
    def validate_symbol(self, symbol: str) -> Tuple[bool, Optional[AssetClass], Optional[str]]:
        """
        Validate a trading symbol and determine its asset class
        
        Args:
            symbol: Trading symbol to validate
            
        Returns:
            Tuple of (is_valid, asset_class, error_message)
        """
        # Check if symbol exists in asset registry
        asset_info = self.multi_asset_service.get_asset_info(symbol)
        if asset_info:
            return True, asset_info.get("asset_class"), None
            
        # Symbol not found in registry, try to infer asset class from pattern
        inferred_class = self._infer_asset_class_from_pattern(symbol)
        if inferred_class:
            return True, inferred_class, "Symbol not in registry but matches pattern"
            
        return False, None, "Invalid symbol format or unknown symbol"
    
    def validate_position_data(self, position_data: Dict[str, Any]) -> Tuple[bool, Dict[str, str]]:
        """
        Validate position data for any asset class
        
        Args:
            position_data: Position data dictionary
            
        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = {}
        
        # Check required fields
        required_fields = ["symbol", "direction", "quantity", "entry_price", "account_id"]
        for field in required_fields:
            if field not in position_data or position_data[field] is None:
                errors[field] = f"Missing required field: {field}"
                
        if errors:
            return False, errors
            
        # Validate symbol and get asset class
        symbol = position_data.get("symbol")
        is_valid, asset_class, error_msg = self.validate_symbol(symbol)
        
        if not is_valid:
            errors["symbol"] = error_msg or "Invalid symbol"
            return False, errors
            
        # Add asset class to position data if not present
        if "asset_class" not in position_data and asset_class:
            position_data["asset_class"] = asset_class
            
        # Validate based on asset class
        if asset_class == AssetClass.FOREX:
            return self._validate_forex_position(position_data, errors)
        elif asset_class == AssetClass.CRYPTO:
            return self._validate_crypto_position(position_data, errors)
        elif asset_class == AssetClass.STOCKS:
            return self._validate_stock_position(position_data, errors)
        elif asset_class == AssetClass.COMMODITIES:
            return self._validate_commodity_position(position_data, errors)
        else:
            # Default validation
            return self._validate_common_fields(position_data, errors)
    
    def validate_market_data(self, 
                           data: pd.DataFrame, 
                           symbol: str, 
                           timeframe: str) -> Tuple[bool, pd.DataFrame, Dict[str, str]]:
        """
        Validate market data for a specific symbol and timeframe
        
        Args:
            data: Market data DataFrame
            symbol: Trading symbol
            timeframe: Data timeframe
            
        Returns:
            Tuple of (is_valid, validated_data, error_messages)
        """
        errors = {}
        
        # Check if DataFrame is empty
        if data.empty:
            errors["data"] = "Empty DataFrame"
            return False, data, errors
            
        # Check required columns based on asset class
        required_columns = ["open", "high", "low", "close", "volume", "timestamp"]
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            errors["columns"] = f"Missing required columns: {', '.join(missing_columns)}"
            return False, data, errors
            
        # Get asset class
        asset_info = self.multi_asset_service.get_asset_info(symbol)
        asset_class = asset_info.get("asset_class") if asset_info else None
        
        # Create a copy to avoid modifying the original
        validated_data = data.copy()
        
        # Check for and handle common data issues
        validated_data = self._handle_common_data_issues(validated_data, errors)
        
        # Asset-specific validation
        if asset_class == AssetClass.FOREX:
            validated_data = self._validate_forex_data(validated_data, symbol, errors)
        elif asset_class == AssetClass.CRYPTO:
            validated_data = self._validate_crypto_data(validated_data, symbol, errors)
        elif asset_class == AssetClass.STOCKS:
            validated_data = self._validate_stock_data(validated_data, symbol, errors)
            
        # Check if there are critical errors
        has_critical_errors = any(errors.get(k) for k in errors if k.startswith("critical_"))
        
        return not has_critical_errors, validated_data, errors
    
    def _infer_asset_class_from_pattern(self, symbol: str) -> Optional[AssetClass]:
        """Infer asset class based on symbol pattern"""
        symbol = symbol.upper()
        
        # Forex pattern: 6 letters or 7 with /, usually currency pairs
        if re.match(r'^[A-Z]{3}[/]?[A-Z]{3}$', symbol):
            return AssetClass.FOREX
            
        # Crypto pattern: ends with USDT, BTC, ETH, etc.
        if re.search(r'(BTC|ETH|USDT|USDC|BNB)$', symbol):
            return AssetClass.CRYPTO
            
        # Stock pattern: 1-5 letters, possibly with . for exchange
        if re.match(r'^[A-Z]{1,5}(\.[A-Z]{1,3})?$', symbol):
            return AssetClass.STOCKS
            
        # Common commodity symbols
        commodities = ["GOLD", "SILVER", "XAUUSD", "XAGUSD", "OIL", "WTICRD", "BRENT", "NGAS"]
        if symbol in commodities or symbol.startswith(("CL", "GC", "SI", "NG")):
            return AssetClass.COMMODITIES
            
        # Index patterns
        indices = ["SPX", "NDX", "DJI", "FTSE", "DAX", "CAC", "NIKKEI"]
        if any(idx in symbol for idx in indices) or symbol.endswith("INDEX"):
            return AssetClass.INDICES
            
        return None
    
    def _validate_common_fields(self, position_data: Dict[str, Any], errors: Dict[str, str]) -> Tuple[bool, Dict[str, str]]:
        """Validate common fields for all asset types"""
        # Check quantity is positive
        if position_data.get("quantity", 0) <= 0:
            errors["quantity"] = "Quantity must be greater than zero"
            
        # Check entry price is positive
        if position_data.get("entry_price", 0) <= 0:
            errors["entry_price"] = "Entry price must be greater than zero"
            
        # Check direction is valid
        direction = position_data.get("direction", "").lower()
        if direction not in ["long", "short"]:
            errors["direction"] = "Direction must be 'long' or 'short'"
            
        return len(errors) == 0, errors
    
    def _validate_forex_position(self, position_data: Dict[str, Any], errors: Dict[str, str]) -> Tuple[bool, Dict[str, str]]:
        """Validate forex-specific position data"""
        # Start with common validation
        is_valid, errors = self._validate_common_fields(position_data, errors)
        
        # Check for valid lot size
        quantity = position_data.get("quantity", 0)
        standard_lot = 100000
        
        if quantity % 1000 != 0:
            errors["quantity"] = "Forex quantity should be in multiples of 1000 units"
            
        # Ensure pip value is present for forex
        if "pip_value" not in position_data:
            # Try to get from asset registry
            symbol = position_data.get("symbol")
            asset_info = self.multi_asset_service.get_asset_info(symbol)
            
            if asset_info and "trading_parameters" in asset_info:
                pip_value = asset_info["trading_parameters"].get("pip_value")
                if pip_value:
                    position_data["pip_value"] = pip_value
                else:
                    errors["pip_value"] = "Pip value is required for forex positions"
            else:
                errors["pip_value"] = "Pip value is required for forex positions"
                
        return len(errors) == 0, errors
    
    def _validate_crypto_position(self, position_data: Dict[str, Any], errors: Dict[str, str]) -> Tuple[bool, Dict[str, str]]:
        """Validate crypto-specific position data"""
        # Start with common validation
        is_valid, errors = self._validate_common_fields(position_data, errors)
        
        # Ensure crypto-specific fields
        # No specific validation needed, but could add exchange-specific rules
        
        return len(errors) == 0, errors
    
    def _validate_stock_position(self, position_data: Dict[str, Any], errors: Dict[str, str]) -> Tuple[bool, Dict[str, str]]:
        """Validate stock-specific position data"""
        # Start with common validation
        is_valid, errors = self._validate_common_fields(position_data, errors)
        
        # Check quantity is a whole number for stocks
        quantity = position_data.get("quantity", 0)
        if quantity != int(quantity):
            errors["quantity"] = "Stock quantity must be a whole number"
            
        return len(errors) == 0, errors
    
    def _validate_commodity_position(self, position_data: Dict[str, Any], errors: Dict[str, str]) -> Tuple[bool, Dict[str, str]]:
        """Validate commodity-specific position data"""
        # Start with common validation
        is_valid, errors = self._validate_common_fields(position_data, errors)
        
        # Additional commodity-specific validation could be added here
        
        return len(errors) == 0, errors
    
    def _handle_common_data_issues(self, data: pd.DataFrame, errors: Dict[str, str]) -> pd.DataFrame:
        """Handle common data issues like missing values, duplicates, etc."""
        # Check for missing values
        missing_counts = data.isnull().sum()
        has_missing = missing_counts.any()
        
        if has_missing:
            for col in missing_counts.index:
                if missing_counts[col] > 0:
                    errors[f"missing_{col}"] = f"{missing_counts[col]} missing values in {col}"
            
            # Forward fill missing values for OHLC
            for col in ["open", "high", "low", "close"]:
                if col in data.columns:
                    data[col] = data[col].ffill()
                    
            # Forward fill volume, or fill with 0
            if "volume" in data.columns:
                data["volume"] = data["volume"].fillna(0)
        
        # Check for duplicate timestamps
        if "timestamp" in data.columns:
            duplicate_counts = data["timestamp"].duplicated().sum()
            if duplicate_counts > 0:
                errors["duplicate_timestamps"] = f"{duplicate_counts} duplicate timestamps found"
                # Keep the last occurrence of each timestamp
                data = data.drop_duplicates(subset=["timestamp"], keep="last")
                
        # Check for out-of-order timestamps
        if "timestamp" in data.columns:
            is_sorted = data["timestamp"].is_monotonic_increasing
            if not is_sorted:
                errors["timestamp_order"] = "Timestamps are not in ascending order"
                # Sort by timestamp
                data = data.sort_values("timestamp")
                
        # Ensure indexes are reset after any filtering
        data = data.reset_index(drop=True)
        
        return data
    
    def _validate_forex_data(self, data: pd.DataFrame, symbol: str, errors: Dict[str, str]) -> pd.DataFrame:
        """Validate forex-specific data"""
        # Check for invalid prices (e.g., zero or negative)
        for col in ["open", "high", "low", "close"]:
            if col in data.columns:
                invalid_count = (data[col] <= 0).sum()
                if invalid_count > 0:
                    errors[f"invalid_{col}"] = f"{invalid_count} invalid {col} prices"
                    # Replace with forward fill or previous valid value
                    mask = data[col] <= 0
                    data.loc[mask, col] = np.nan
                    data[col] = data[col].ffill()
        
        # Check for high-low relationship
        if "high" in data.columns and "low" in data.columns:
            invalid_hl = (data["high"] < data["low"]).sum()
            if invalid_hl > 0:
                errors["invalid_high_low"] = f"{invalid_hl} cases where high < low"
                # Swap high and low where high < low
                mask = data["high"] < data["low"]
                data.loc[mask, ["high", "low"]] = data.loc[mask, ["low", "high"]].values
                
        # Check for price gaps (specific to forex)
        price_gaps = self._check_price_gaps(data, symbol, asset_class=AssetClass.FOREX)
        if price_gaps > 0:
            errors["price_gaps"] = f"{price_gaps} significant price gaps detected"
            
        return data
    
    def _validate_crypto_data(self, data: pd.DataFrame, symbol: str, errors: Dict[str, str]) -> pd.DataFrame:
        """Validate crypto-specific data"""
        # Check for invalid prices (e.g., zero or negative)
        for col in ["open", "high", "low", "close"]:
            if col in data.columns:
                invalid_count = (data[col] <= 0).sum()
                if invalid_count > 0:
                    errors[f"invalid_{col}"] = f"{invalid_count} invalid {col} prices"
                    # Replace with forward fill or previous valid value
                    mask = data[col] <= 0
                    data.loc[mask, col] = np.nan
                    data[col] = data[col].ffill()
        
        # Check for volume spikes (common in crypto)
        if "volume" in data.columns:
            mean_vol = data["volume"].mean()
            std_vol = data["volume"].std()
            extreme_vol = (data["volume"] > mean_vol + 5 * std_vol).sum()
            
            if extreme_vol > 0:
                errors["extreme_volume"] = f"{extreme_vol} extreme volume spikes detected"
                
        return data
    
    def _validate_stock_data(self, data: pd.DataFrame, symbol: str, errors: Dict[str, str]) -> pd.DataFrame:
        """Validate stock-specific data"""
        # Check for invalid prices (e.g., zero or negative)
        for col in ["open", "high", "low", "close"]:
            if col in data.columns:
                invalid_count = (data[col] <= 0).sum()
                if invalid_count > 0:
                    errors[f"invalid_{col}"] = f"{invalid_count} invalid {col} prices"
                    # Replace with forward fill or previous valid value
                    mask = data[col] <= 0
                    data.loc[mask, col] = np.nan
                    data[col] = data[col].ffill()
        
        # Check for trading hours if timestamp includes time
        if "timestamp" in data.columns and data["timestamp"].dtype != 'datetime64[ns]':
            data["timestamp"] = pd.to_datetime(data["timestamp"])
            
        if "timestamp" in data.columns:
            hours = data["timestamp"].dt.hour
            weekend_trading = ((data["timestamp"].dt.dayofweek == 5) | 
                              (data["timestamp"].dt.dayofweek == 6)).sum()
                              
            if weekend_trading > 0:
                errors["weekend_trading"] = f"{weekend_trading} records during weekend"
                
        return data
    
    def _check_price_gaps(self, data: pd.DataFrame, symbol: str, asset_class: AssetClass) -> int:
        """Check for significant price gaps based on asset class"""
        if len(data) < 2:
            return 0
            
        # Get previous close and current open
        prev_close = data["close"].shift(1)
        curr_open = data["open"]
        
        # Calculate gap size
        gap_pct = abs(curr_open - prev_close) / prev_close * 100
        
        # Define threshold based on asset class
        if asset_class == AssetClass.FOREX:
            threshold = 0.5  # 0.5% for forex is significant
        elif asset_class == AssetClass.CRYPTO:
            threshold = 5.0  # 5% for crypto
        elif asset_class == AssetClass.STOCKS:
            threshold = 2.0  # 2% for stocks
        else:
            threshold = 1.0  # Default
            
        # Count significant gaps
        significant_gaps = (gap_pct > threshold).sum()
        
        return significant_gaps
