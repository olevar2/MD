"""
Advanced Correlation Analysis Module

This module provides enhanced correlation analysis functionality for currency pairs, 
including dynamic timeframe analysis, lead-lag relationships, and correlation breakdown detection.
"""

from typing import Dict, List, Any, Union, Optional, Tuple
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from scipy.stats import pearsonr, spearmanr
from statsmodels.tsa.stattools import grangercausalitytests, coint

from analysis_engine.analysis.base_analyzer import BaseAnalyzer
from analysis_engine.models.analysis_result import AnalysisResult
from analysis_engine.models.market_data import MarketData
from analysis_engine.caching.cache_service import cache_result # Added import

logger = logging.getLogger(__name__)

class CurrencyCorrelationEnhanced(BaseAnalyzer):
    """
    Enhanced analyzer for currency correlation patterns and relationships.
    
    This analyzer detects correlations between currency pairs across different timeframes,
    identifies lead-lag relationships, and detects correlation breakdowns that may
    signal trading opportunities.
    """
    
    def __init__(self, parameters: Dict[str, Any] = None):
        """
        Initialize the enhanced currency correlation analyzer
        
        Args:
            parameters: Configuration parameters for the analyzer
        """
        default_params = {
            "correlation_method": "pearson",  # 'pearson' or 'spearman'
            "correlation_windows": [5, 10, 20, 60, 120],  # days for correlation calculation
            "significant_correlation_threshold": 0.7,  # absolute value threshold for significant correlation
            "correlation_change_threshold": 0.3,  # threshold for significant correlation change
            "granger_maxlag": 10,  # maximum lag for Granger causality test
            "granger_significance": 0.05,  # p-value threshold for Granger causality
            "cointegration_significance": 0.05,  # p-value threshold for cointegration test
            "base_currency": "USD",  # base currency for correlation analysis
            "major_pairs": ["EUR/USD", "USD/JPY", "GBP/USD", "USD/CHF", "USD/CAD", "AUD/USD", "NZD/USD"],
            "minor_pairs": ["EUR/GBP", "EUR/JPY", "GBP/JPY", "AUD/JPY", "EUR/CHF"],
            "crosses": ["EUR/GBP", "EUR/AUD", "GBP/AUD", "EUR/CAD", "GBP/CAD"]
        }
        
        # Merge default parameters with provided parameters
        merged_params = {**default_params, **(parameters or {})}
        super().__init__("currency_correlation_enhanced", merged_params)
        
    def _calculate_returns(self, price_data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate percentage returns for price data
        
        Args:
            price_data: DataFrame with price data
            
        Returns:
            DataFrame with percentage returns
        """
        # Use close prices to calculate returns
        if 'close' in price_data.columns:
            returns = price_data['close'].pct_change().dropna()
        else:
            # If no 'close' column, use the first numeric column
            numeric_cols = price_data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) == 0:
                raise ValueError("No numeric columns found in price data")
            returns = price_data[numeric_cols[0]].pct_change().dropna()
            
        return returns
    
    def _calculate_correlation_matrix(self, returns_dict: Dict[str, pd.Series], window: int) -> pd.DataFrame:
        """
        Calculate correlation matrix for a specific window
        
        Args:
            returns_dict: Dictionary mapping pair names to return series
            window: Window size for correlation calculation
            
        Returns:
            DataFrame with correlation matrix
        """
        # Create a DataFrame with all return series
        returns_df = pd.DataFrame(returns_dict)
        
        # Calculate rolling correlation matrix
        if window >= len(returns_df):
            # If window is larger than available data, use all data
            correlation_matrix = returns_df.corr(method=self.parameters["correlation_method"])
        else:
            # Calculate rolling correlation using the specified window
            correlation_matrix = returns_df.iloc[-window:].corr(method=self.parameters["correlation_method"])
            
        return correlation_matrix
    
    def _detect_lead_lag(self, pair1_returns: pd.Series, pair2_returns: pd.Series, 
                         pair1_name: str, pair2_name: str) -> Dict[str, Any]:
        """
        Detect lead-lag relationship between two currency pairs
        
        Args:
            pair1_returns: Return series for first pair
            pair2_returns: Return series for second pair
            pair1_name: Name of first pair
            pair2_name: Name of second pair
            
        Returns:
            Dictionary with lead-lag relationship information
        """
        max_lag = min(self.parameters["granger_maxlag"], len(pair1_returns) // 5)
        
        # Create DataFrame for Granger causality test
        data = pd.DataFrame({
            pair1_name: pair1_returns,
            pair2_name: pair2_returns
        }).dropna()
        
        # If insufficient data, return empty result
        if len(data) <= max_lag:
            return {
                "pair1": pair1_name,
                "pair2": pair2_name,
                "lead_lag_relationship": "insufficient_data",
                "granger_results": None,
                "p_value": None,
                "lead_lag": None
            }
        
        # Test if pair1 Granger-causes pair2
        pair1_causes_pair2 = None
        try:
            granger_1_to_2 = grangercausalitytests(data[[pair2_name, pair1_name]], max_lag, verbose=False)
            
            # Find the minimum p-value and corresponding lag
            min_p_value_1_to_2 = 1.0
            best_lag_1_to_2 = 0
            
            for lag, result in granger_1_to_2.items():
                # Get p-value from the F-test
                p_value = result[0]['ssr_ftest'][1]
                if p_value < min_p_value_1_to_2:
                    min_p_value_1_to_2 = p_value
                    best_lag_1_to_2 = lag
                    
            pair1_causes_pair2 = {
                "p_value": min_p_value_1_to_2,
                "lag": best_lag_1_to_2,
                "significant": min_p_value_1_to_2 < self.parameters["granger_significance"]
            }
        except Exception as e:
            logger.warning(f"Error in Granger test {pair1_name} -> {pair2_name}: {e}")
            
        # Test if pair2 Granger-causes pair1
        pair2_causes_pair1 = None
        try:
            granger_2_to_1 = grangercausalitytests(data[[pair1_name, pair2_name]], max_lag, verbose=False)
            
            # Find the minimum p-value and corresponding lag
            min_p_value_2_to_1 = 1.0
            best_lag_2_to_1 = 0
            
            for lag, result in granger_2_to_1.items():
                # Get p-value from the F-test
                p_value = result[0]['ssr_ftest'][1]
                if p_value < min_p_value_2_to_1:
                    min_p_value_2_to_1 = p_value
                    best_lag_2_to_1 = lag
                    
            pair2_causes_pair1 = {
                "p_value": min_p_value_2_to_1,
                "lag": best_lag_2_to_1,
                "significant": min_p_value_2_to_1 < self.parameters["granger_significance"]
            }
        except Exception as e:
            logger.warning(f"Error in Granger test {pair2_name} -> {pair1_name}: {e}")
        
        # Determine lead-lag relationship
        lead_lag = None
        p_value = None
        
        if pair1_causes_pair2 and pair1_causes_pair2["significant"] and \
           (not pair2_causes_pair1 or not pair2_causes_pair1["significant"]):
            # pair1 leads pair2
            lead_lag = f"{pair1_name} leads {pair2_name}"
            p_value = pair1_causes_pair2["p_value"]
            
        elif pair2_causes_pair1 and pair2_causes_pair1["significant"] and \
             (not pair1_causes_pair2 or not pair1_causes_pair2["significant"]):
            # pair2 leads pair1
            lead_lag = f"{pair2_name} leads {pair1_name}"
            p_value = pair2_causes_pair1["p_value"]
            
        elif pair1_causes_pair2 and pair1_causes_pair2["significant"] and \
             pair2_causes_pair1 and pair2_causes_pair1["significant"]:
            # Bi-directional relationship
            lead_lag = f"Bi-directional between {pair1_name} and {pair2_name}"
            p_value = min(pair1_causes_pair2["p_value"], pair2_causes_pair1["p_value"])
            
        else:
            # No significant lead-lag relationship
            lead_lag = f"No clear lead-lag relationship"
            p_value = 1.0
            
        return {
            "pair1": pair1_name,
            "pair2": pair2_name,
            "lead_lag_relationship": lead_lag,
            "p_value": p_value,
            "pair1_causes_pair2": pair1_causes_pair2,
            "pair2_causes_pair1": pair2_causes_pair1
        }
    
    def _test_cointegration(self, pair1_prices: pd.Series, pair2_prices: pd.Series, 
                          pair1_name: str, pair2_name: str) -> Dict[str, Any]:
        """
        Test for cointegration between two currency pairs
        
        Args:
            pair1_prices: Price series for first pair
            pair2_prices: Price series for second pair
            pair1_name: Name of first pair
            pair2_name: Name of second pair
            
        Returns:
            Dictionary with cointegration test results
        """
        # Create DataFrame for cointegration test
        data = pd.DataFrame({
            pair1_name: pair1_prices,
            pair2_name: pair2_prices
        }).dropna()
        
        # If insufficient data, return empty result
        if len(data) < 30:  # Need reasonable sample size for cointegration test
            return {
                "pair1": pair1_name,
                "pair2": pair2_name,
                "cointegrated": False,
                "p_value": None,
                "test_statistic": None,
                "critical_values": None
            }
            
        try:
            # Perform Engle-Granger cointegration test
            coint_result = coint(data[pair1_name], data[pair2_name])
            test_statistic = coint_result[0]
            p_value = coint_result[1]
            critical_values = coint_result[2]
            
            # Check if pairs are cointegrated
            cointegrated = p_value < self.parameters["cointegration_significance"]
            
            return {
                "pair1": pair1_name,
                "pair2": pair2_name,
                "cointegrated": cointegrated,
                "p_value": p_value,
                "test_statistic": test_statistic,
                "critical_values": {
                    "1%": critical_values[0],
                    "5%": critical_values[1],
                    "10%": critical_values[2]
                }
            }
        except Exception as e:
            logger.warning(f"Error in cointegration test between {pair1_name} and {pair2_name}: {e}")
            return {
                "pair1": pair1_name,
                "pair2": pair2_name,
                "cointegrated": False,
                "error": str(e)
            }
    
    def _detect_correlation_breakdown(self, current_corr_matrix: pd.DataFrame, 
                                    historical_corr_matrix: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Detect significant changes in correlation between currency pairs
        
        Args:
            current_corr_matrix: Current correlation matrix
            historical_corr_matrix: Historical correlation matrix for comparison
            
        Returns:
            List of detected correlation breakdowns
        """
        breakdowns = []
        threshold = self.parameters["correlation_change_threshold"]
        
        # Get pairs from the correlation matrix
        pairs = current_corr_matrix.columns
        
        for i in range(len(pairs)):
            for j in range(i+1, len(pairs)):
                pair1 = pairs[i]
                pair2 = pairs[j]
                
                # Get current and historical correlation
                current_corr = current_corr_matrix.loc[pair1, pair2]
                historical_corr = historical_corr_matrix.loc[pair1, pair2]
                
                # Calculate absolute change in correlation
                corr_change = abs(current_corr - historical_corr)
                
                # Check if change exceeds threshold
                if corr_change >= threshold:
                    breakdowns.append({
                        "pair1": pair1,
                        "pair2": pair2,
                        "current_correlation": current_corr,
                        "historical_correlation": historical_corr,
                        "correlation_change": current_corr - historical_corr,
                        "absolute_change": corr_change,
                        "breakdown_type": "increasing" if current_corr > historical_corr else "decreasing"
                    })
                    
        return breakdowns
    
    def _generate_correlation_signals(self, correlation_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate trading signals based on correlation analysis
        
        Args:
            correlation_data: Dictionary with correlation analysis results
            
        Returns:
            List of generated trading signals
        """
        signals = []
        
        # Process correlation breakdowns
        for breakdown in correlation_data.get("correlation_breakdowns", []):
            # Only consider significant historical correlations
            if abs(breakdown["historical_correlation"]) >= self.parameters["significant_correlation_threshold"]:
                # Correlation weakening signal
                if breakdown["breakdown_type"] == "decreasing" and breakdown["historical_correlation"] > 0:
                    signals.append({
                        "signal_type": "correlation_breakdown",
                        "pair1": breakdown["pair1"],
                        "pair2": breakdown["pair2"],
                        "direction": "divergence",
                        "confidence": min(abs(breakdown["correlation_change"]) * 2, 1.0),
                        "rationale": f"Positive correlation breakdown between {breakdown['pair1']} and {breakdown['pair2']}",
                        "suggested_action": "Consider directional trade based on stronger fundamental pair"
                    })
                # Negative correlation weakening
                elif breakdown["breakdown_type"] == "decreasing" and breakdown["historical_correlation"] < 0:
                    signals.append({
                        "signal_type": "correlation_breakdown",
                        "pair1": breakdown["pair1"],
                        "pair2": breakdown["pair2"],
                        "direction": "convergence",
                        "confidence": min(abs(breakdown["correlation_change"]) * 2, 1.0),
                        "rationale": f"Negative correlation breakdown between {breakdown['pair1']} and {breakdown['pair2']}",
                        "suggested_action": "Consider directional trade based on stronger fundamental pair"
                    })
        
        # Process lead-lag relationships
        for relationship in correlation_data.get("lead_lag_relationships", []):
            if "leads" in relationship["lead_lag_relationship"] and "Bi-directional" not in relationship["lead_lag_relationship"]:
                # Extract leading and lagging pairs
                parts = relationship["lead_lag_relationship"].split(" leads ")
                if len(parts) == 2:
                    leading_pair = parts[0]
                    lagging_pair = parts[1]
                    
                    signals.append({
                        "signal_type": "lead_lag",
                        "leading_pair": leading_pair,
                        "lagging_pair": lagging_pair,
                        "confidence": 1 - min(relationship["p_value"] * 10, 0.9),  # Convert p-value to confidence
                        "rationale": f"{leading_pair} movements tend to precede {lagging_pair}",
                        "suggested_action": f"Monitor {leading_pair} for early signals on {lagging_pair} direction"
                    })
        
        # Process cointegration relationships
        for cointegration in correlation_data.get("cointegration_tests", []):
            if cointegration.get("cointegrated", False):
                signals.append({
                    "signal_type": "cointegration",
                    "pair1": cointegration["pair1"],
                    "pair2": cointegration["pair2"],
                    "confidence": 1 - min(cointegration["p_value"] * 10, 0.9),  # Convert p-value to confidence
                    "rationale": f"{cointegration['pair1']} and {cointegration['pair2']} are cointegrated, suggesting mean-reversion opportunities",
                    "suggested_action": "Consider mean-reversion trading strategy between pairs"
                })
                
        return signals
    
    def _create_correlation_heatmap_data(self, correlation_matrix: pd.DataFrame) -> Dict[str, Any]:
        """
        Create data structure for correlation heatmap visualization
        
        Args:
            correlation_matrix: Correlation matrix
            
        Returns:
            Dictionary with heatmap data
        """
        pairs = correlation_matrix.columns.tolist()
        heatmap_data = []
        
        for i, pair1 in enumerate(pairs):
            for j, pair2 in enumerate(pairs):
                value = correlation_matrix.iloc[i, j]
                heatmap_data.append({
                    "x": j,
                    "y": i,
                    "pair_x": pair2,
                    "pair_y": pair1,
                    "value": value
                })
                
        return {
            "pairs": pairs,
            "data": heatmap_data
        }
        
    @cache_result(ttl=1800) # Cache for 30 minutes
    def analyze(self, data: Dict[str, Dict[str, Any]]) -> AnalysisResult:
        """
        Analyze currency correlations across pairs and timeframes
        
        Args:
            data: Dictionary mapping pair names to market data dictionaries
            
        Returns:
            AnalysisResult with correlation analysis
        """
        if not data or not isinstance(data, dict):
            return AnalysisResult(
                analyzer_name=self.name,
                result_data={"error": "No data provided or invalid data format"},
                is_valid=False
            )
            
        # Extract pairs to analyze
        pairs_to_analyze = list(data.keys())
        
        if len(pairs_to_analyze) < 2:
            return AnalysisResult(
                analyzer_name=self.name,
                result_data={"error": "At least two currency pairs are required for correlation analysis"},
                is_valid=False
            )
            
        # Extract price and returns data for each pair
        price_data = {}
        returns_data = {}
        
        for pair, pair_data in data.items():
            if isinstance(pair_data, dict) and "ohlc" in pair_data:
                # Convert to DataFrame if it's a dict with ohlc key
                df = pd.DataFrame(pair_data["ohlc"])
                price_data[pair] = df
                returns_data[pair] = self._calculate_returns(df)
            elif isinstance(pair_data, pd.DataFrame):
                # Use DataFrame directly
                price_data[pair] = pair_data
                returns_data[pair] = self._calculate_returns(pair_data)
            else:
                logger.warning(f"Invalid data format for pair {pair}, skipping")
                
        if len(price_data) < 2:
            return AnalysisResult(
                analyzer_name=self.name,
                result_data={"error": "Insufficient valid data for correlation analysis"},
                is_valid=False
            )
            
        # Calculate correlation matrices for different windows
        correlation_matrices = {}
        for window in self.parameters["correlation_windows"]:
            try:
                correlation_matrices[window] = self._calculate_correlation_matrix(returns_data, window)
            except Exception as e:
                logger.warning(f"Error calculating correlation matrix for window {window}: {e}")
                
        if not correlation_matrices:
            return AnalysisResult(
                analyzer_name=self.name,
                result_data={"error": "Failed to calculate correlation matrices"},
                is_valid=False
            )
            
        # Get the current (shortest window) and historical (longest window) correlation matrices
        current_window = min(self.parameters["correlation_windows"])
        historical_window = max(self.parameters["correlation_windows"])
        
        current_matrix = correlation_matrices.get(current_window)
        historical_matrix = correlation_matrices.get(historical_window)
        
        # Detect correlation breakdowns
        correlation_breakdowns = []
        if current_matrix is not None and historical_matrix is not None:
            correlation_breakdowns = self._detect_correlation_breakdown(current_matrix, historical_matrix)
            
        # Calculate lead-lag relationships for select pairs
        lead_lag_relationships = []
        pairs_for_lead_lag = self.parameters["major_pairs"] + self.parameters["minor_pairs"]
        pairs_for_lead_lag = [p for p in pairs_for_lead_lag if p in returns_data]
        
        if len(pairs_for_lead_lag) >= 2:
            for i in range(len(pairs_for_lead_lag)):
                for j in range(i+1, len(pairs_for_lead_lag)):
                    pair1 = pairs_for_lead_lag[i]
                    pair2 = pairs_for_lead_lag[j]
                    
                    if pair1 in returns_data and pair2 in returns_data:
                        lead_lag = self._detect_lead_lag(
                            returns_data[pair1], returns_data[pair2], pair1, pair2
                        )
                        lead_lag_relationships.append(lead_lag)
                        
        # Test for cointegration between pairs
        cointegration_tests = []
        for i in range(len(pairs_for_lead_lag)):
            for j in range(i+1, len(pairs_for_lead_lag)):
                pair1 = pairs_for_lead_lag[i]
                pair2 = pairs_for_lead_lag[j]
                
                if pair1 in price_data and pair2 in price_data:
                    # Use close prices for cointegration test
                    pair1_prices = price_data[pair1]['close'] if 'close' in price_data[pair1].columns else price_data[pair1].iloc[:, 0]
                    pair2_prices = price_data[pair2]['close'] if 'close' in price_data[pair2].columns else price_data[pair2].iloc[:, 0]
                    
                    coint_test = self._test_cointegration(
                        pair1_prices, pair2_prices, pair1, pair2
                    )
                    cointegration_tests.append(coint_test)
                    
        # Compile results
        correlation_data = {
            "current_window": current_window,
            "historical_window": historical_window,
            "correlation_matrices": {str(window): matrix.to_dict() for window, matrix in correlation_matrices.items()},
            "correlation_breakdowns": correlation_breakdowns,
            "lead_lag_relationships": lead_lag_relationships,
            "cointegration_tests": cointegration_tests,
        }
        
        # Generate correlation-based trading signals
        signals = self._generate_correlation_signals(correlation_data)
        correlation_data["trading_signals"] = signals
        
        # Create heatmap visualization data
        if current_matrix is not None:
            correlation_data["current_heatmap"] = self._create_correlation_heatmap_data(current_matrix)
        
        if historical_matrix is not None:
            correlation_data["historical_heatmap"] = self._create_correlation_heatmap_data(historical_matrix)
        
        return AnalysisResult(
            analyzer_name=self.name,
            result_data=correlation_data,
            is_valid=True
        )
