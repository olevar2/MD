"""
Correlation Analysis module for Market Analysis Service.

This module provides algorithms for analyzing correlations between symbols.
"""
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)

class CorrelationAnalyzer:
    """
    Class for analyzing correlations between symbols.
    """
    
    def __init__(self):
        """
        Initialize the Correlation Analyzer.
        """
        pass
        
    def analyze_correlations(
        self,
        data: Dict[str, pd.DataFrame],
        window_size: int = 20,
        method: str = "pearson",
        additional_parameters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Analyze correlations between symbols.
        
        Args:
            data: Dictionary mapping symbols to DataFrames
            window_size: Window size for rolling correlation
            method: Correlation method (pearson, spearman, kendall)
            additional_parameters: Additional parameters for analysis
            
        Returns:
            Correlation analysis results
        """
        if additional_parameters is None:
            additional_parameters = {}
            
        # Extract symbols
        symbols = list(data.keys())
        
        if len(symbols) < 2:
            logger.warning("At least two symbols are required for correlation analysis")
            return {
                "correlation_matrix": {},
                "correlation_pairs": []
            }
            
        # Extract close prices
        close_prices = {}
        
        for symbol, df in data.items():
            if "close" in df.columns:
                close_prices[symbol] = df["close"]
                
        # Create a DataFrame with close prices
        prices_df = pd.DataFrame(close_prices)
        
        # Calculate returns
        returns_df = prices_df.pct_change().dropna()
        
        # Calculate correlation matrix
        correlation_matrix = returns_df.corr(method=method)
        
        # Convert correlation matrix to dictionary
        correlation_dict = {}
        
        for symbol1 in symbols:
            correlation_dict[symbol1] = {}
            
            for symbol2 in symbols:
                if symbol1 in correlation_matrix.index and symbol2 in correlation_matrix.columns:
                    correlation_dict[symbol1][symbol2] = float(correlation_matrix.loc[symbol1, symbol2])
                else:
                    correlation_dict[symbol1][symbol2] = None
                    
        # Calculate rolling correlations for each pair
        correlation_pairs = []
        
        for i, symbol1 in enumerate(symbols):
            for j, symbol2 in enumerate(symbols):
                if i < j:  # Avoid duplicate pairs
                    if symbol1 in returns_df.columns and symbol2 in returns_df.columns:
                        # Calculate correlation
                        correlation = correlation_matrix.loc[symbol1, symbol2]
                        
                        # Calculate p-value
                        p_value = self._calculate_p_value(returns_df[symbol1], returns_df[symbol2], method)
                        
                        # Calculate rolling correlation
                        rolling_corr = returns_df[symbol1].rolling(window=window_size).corr(returns_df[symbol2])
                        
                        # Convert rolling correlation to list of data points
                        rolling_corr_data = []
                        
                        for idx, value in rolling_corr.items():
                            if not pd.isna(value):
                                timestamp = idx.isoformat() if hasattr(idx, 'isoformat') else str(idx)
                                
                                rolling_corr_data.append({
                                    "timestamp": timestamp,
                                    "correlation": float(value)
                                })
                                
                        correlation_pairs.append({
                            "symbol1": symbol1,
                            "symbol2": symbol2,
                            "correlation": float(correlation),
                            "p_value": float(p_value),
                            "rolling_correlation": rolling_corr_data
                        })
                        
        return {
            "correlation_matrix": correlation_dict,
            "correlation_pairs": correlation_pairs
        }
        
    def _calculate_p_value(
        self,
        x: pd.Series,
        y: pd.Series,
        method: str
    ) -> float:
        """
        Calculate p-value for correlation.
        
        Args:
            x: First series
            y: Second series
            method: Correlation method
            
        Returns:
            p-value
        """
        try:
            import scipy.stats as stats
            
            # Remove NaN values
            mask = ~(pd.isna(x) | pd.isna(y))
            x_clean = x[mask]
            y_clean = y[mask]
            
            if len(x_clean) < 2:
                return 1.0
                
            if method == "pearson":
                _, p_value = stats.pearsonr(x_clean, y_clean)
            elif method == "spearman":
                _, p_value = stats.spearmanr(x_clean, y_clean)
            elif method == "kendall":
                _, p_value = stats.kendalltau(x_clean, y_clean)
            else:
                # Default to pearson
                _, p_value = stats.pearsonr(x_clean, y_clean)
                
            return p_value
            
        except Exception as e:
            logger.error(f"Error calculating p-value: {e}")
            return 1.0
            
    def analyze_correlation_breakdown(
        self,
        data: Dict[str, pd.DataFrame],
        window_size: int = 20,
        method: str = "pearson",
        breakdown_threshold: float = 0.3,
        additional_parameters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Analyze correlation breakdown risk.
        
        Args:
            data: Dictionary mapping symbols to DataFrames
            window_size: Window size for rolling correlation
            method: Correlation method (pearson, spearman, kendall)
            breakdown_threshold: Threshold for correlation breakdown
            additional_parameters: Additional parameters for analysis
            
        Returns:
            Correlation breakdown analysis results
        """
        if additional_parameters is None:
            additional_parameters = {}
            
        # Extract symbols
        symbols = list(data.keys())
        
        if len(symbols) < 2:
            logger.warning("At least two symbols are required for correlation breakdown analysis")
            return {
                "breakdown_pairs": []
            }
            
        # Extract close prices
        close_prices = {}
        
        for symbol, df in data.items():
            if "close" in df.columns:
                close_prices[symbol] = df["close"]
                
        # Create a DataFrame with close prices
        prices_df = pd.DataFrame(close_prices)
        
        # Calculate returns
        returns_df = prices_df.pct_change().dropna()
        
        # Calculate correlation matrix
        correlation_matrix = returns_df.corr(method=method)
        
        # Calculate rolling correlations for each pair
        breakdown_pairs = []
        
        for i, symbol1 in enumerate(symbols):
            for j, symbol2 in enumerate(symbols):
                if i < j:  # Avoid duplicate pairs
                    if symbol1 in returns_df.columns and symbol2 in returns_df.columns:
                        # Calculate correlation
                        correlation = correlation_matrix.loc[symbol1, symbol2]
                        
                        # Calculate rolling correlation
                        rolling_corr = returns_df[symbol1].rolling(window=window_size).corr(returns_df[symbol2])
                        
                        # Check for correlation breakdown
                        if len(rolling_corr) >= 2 * window_size:
                            # Calculate average correlation in first and second half
                            first_half = rolling_corr.iloc[-2 * window_size:-window_size].mean()
                            second_half = rolling_corr.iloc[-window_size:].mean()
                            
                            # Calculate correlation change
                            correlation_change = abs(second_half - first_half)
                            
                            if correlation_change > breakdown_threshold:
                                # Calculate breakdown risk
                                breakdown_risk = correlation_change / breakdown_threshold
                                
                                # Convert rolling correlation to list of data points
                                rolling_corr_data = []
                                
                                for idx, value in rolling_corr.iloc[-2 * window_size:].items():
                                    if not pd.isna(value):
                                        timestamp = idx.isoformat() if hasattr(idx, 'isoformat') else str(idx)
                                        
                                        rolling_corr_data.append({
                                            "timestamp": timestamp,
                                            "correlation": float(value)
                                        })
                                        
                                breakdown_pairs.append({
                                    "symbol1": symbol1,
                                    "symbol2": symbol2,
                                    "current_correlation": float(correlation),
                                    "correlation_change": float(correlation_change),
                                    "breakdown_risk": float(breakdown_risk),
                                    "rolling_correlation": rolling_corr_data
                                })
                                
        # Sort by breakdown risk (descending)
        breakdown_pairs.sort(key=lambda x: x["breakdown_risk"], reverse=True)
        
        return {
            "breakdown_pairs": breakdown_pairs
        }