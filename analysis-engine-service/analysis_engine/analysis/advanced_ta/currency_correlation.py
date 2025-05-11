"""
Currency Correlation Analyzer Module

This module provides functionality for analyzing correlations between currency pairs,
which is essential for risk management and understanding cross-market relationships.
It includes tools for calculating correlation coefficients, visualizing correlation
matrices, and detecting correlation regime changes.
"""
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timedelta

from analysis_engine.analysis.base_analyzer import BaseAnalyzer
from analysis_engine.models.market_data import MarketData
from analysis_engine.models.analysis_result import AnalysisResult
# Assuming rolling_correlation exists and handles NaNs appropriately
# from analysis_engine.utils.statistics import rolling_correlation

logger = logging.getLogger(__name__)

# Placeholder for rolling_correlation if not available
def rolling_correlation(series1: pd.Series, series2: pd.Series, window: int) -> pd.Series:
    """
    Placeholder: Calculates rolling correlation between two series.
    """
    if series1.isnull().all() or series2.isnull().all():
        return pd.Series(np.nan, index=series1.index)
    # Ensure indices align for calculation
    aligned_s1, aligned_s2 = series1.align(series2, join='inner')
    if aligned_s1.empty or len(aligned_s1) < window:
        return pd.Series(np.nan, index=series1.index) # Return NaNs matching original index

    # Calculate rolling correlation
    rolling_corr = aligned_s1.rolling(window=window).corr(aligned_s2)

    # Reindex to match the original series1 index, filling missing values with NaN
    return rolling_corr.reindex(series1.index)


class CurrencyCorrelationAnalyzer(BaseAnalyzer):
    """
    Analyzer for currency pair correlations.

    Calculates and tracks correlations between multiple currency pairs over time,
    using pre-calculated OHLCV data provided for each instrument.
    Helps identify diversification opportunities and risks from correlated positions.
    """

    DEFAULT_PARAMS = {
        "base_window": 20,          # Short-term window for correlation calculation
        "long_window": 60,          # Long-term window for correlation calculation
        "correlation_threshold": 0.7,  # Threshold for strong positive/negative correlation
        "regime_change_threshold": 0.3,  # Min absolute change between long/short corr for regime change
        "update_frequency_hours": 4,  # Hours between full recalculation attempts
        "min_data_points": 60,      # Minimum data points required (should be >= long_window)
        "price_column": "close",    # Column to use for price data
        "instrument_list": [],      # Optional: List of specific instruments to analyze (if empty, uses all provided)
        "calculate_rolling": True,  # Whether to calculate rolling correlations (can be expensive)
        "calculate_stability": True # Whether to calculate stability metrics
    }

    def __init__(self, parameters: Dict[str, Any] = None):
        """
        Initialize the Currency Correlation Analyzer.

        Args:
            parameters: Configuration parameters, overriding defaults.
        """
        resolved_params = self.DEFAULT_PARAMS.copy()
        if parameters:
            resolved_params.update(parameters)

        # Ensure min_data_points is sufficient for the longest window
        resolved_params["min_data_points"] = max(
            resolved_params["min_data_points"],
            resolved_params["long_window"]
        )

        super().__init__(name="currency_correlation", parameters=resolved_params)
        self.correlation_cache = {}
        self.last_full_calculation_time = None
        logger.info(f"Initialized CurrencyCorrelationAnalyzer with params: {self.parameters}")

    def analyze(self, data: Dict[str, MarketData]) -> AnalysisResult:
        """Perform correlation analysis on provided currency pair data.

        Args:
            data: Dictionary mapping instrument symbols to their MarketData objects.
                  Each MarketData object must contain a DataFrame with OHLCV data.

        Returns:
            AnalysisResult containing correlation matrices, rolling correlations (optional),
            regime changes, strong correlations, and stability metrics (optional).
        """
        if not data or not isinstance(data, dict) or len(data) < 2:
            msg = "Insufficient data for correlation analysis (need at least 2 instruments)"
            logger.warning(msg)
            return AnalysisResult(analyzer_name=self.name, result={"error": msg}, is_valid=False)

        # --- Data Preparation ---
        instruments = self.parameters["instrument_list"] or list(data.keys())
        price_col = self.parameters["price_column"]
        min_points = self.parameters["min_data_points"]
        price_series_dict = {}

        valid_instruments = []
        for instrument in instruments:
            if instrument not in data or not isinstance(data[instrument], MarketData) or data[instrument].data is None:
                logger.warning(f"Instrument {instrument} missing or has invalid MarketData, skipping.")
                continue

            df = data[instrument].data
            if price_col not in df.columns:
                logger.warning(f"Price column '{price_col}' not found for instrument {instrument}, skipping.")
                continue

            if len(df) < min_points:
                logger.warning(f"Insufficient data points for {instrument} (need {min_points}, got {len(df)}), skipping.")
                continue

            price_series = df[price_col].copy()
            if price_series.isna().all():
                 logger.warning(f"Price data for {instrument} is all NaN, skipping.")
                 continue

            price_series_dict[instrument] = price_series
            valid_instruments.append(instrument)

        if len(valid_instruments) < 2:
            msg = f"Insufficient valid instruments for correlation analysis after filtering (need at least 2, got {len(valid_instruments)})"
            logger.warning(msg)
            return AnalysisResult(analyzer_name=self.name, result={"error": msg}, is_valid=False)

        # --- Combine and Calculate Returns ---
        try:
            # Combine into a single DataFrame, aligning by index (timestamps)
            prices_df = pd.DataFrame(price_series_dict).sort_index()

            # Handle missing values (e.g., due to different trading hours/holidays)
            # Forward fill is common, but consider implications. Interpolation might be another option.
            prices_df = prices_df.ffill() # Forward fill first
            prices_df = prices_df.bfill() # Back fill remaining NaNs at the beginning

            # Check again if any columns are still all NaN after filling
            all_nan_cols = prices_df.columns[prices_df.isna().all()].tolist()
            if all_nan_cols:
                logger.warning(f"Instruments {all_nan_cols} are still all NaN after fill, removing from analysis.")
                prices_df = prices_df.drop(columns=all_nan_cols)
                valid_instruments = [inst for inst in valid_instruments if inst not in all_nan_cols]
                if len(valid_instruments) < 2:
                     msg = "Insufficient valid instruments after NaN handling."
                     logger.warning(msg)
                     return AnalysisResult(analyzer_name=self.name, result={"error": msg}, is_valid=False)

            # Calculate percentage returns
            returns_df = prices_df.pct_change().dropna() # dropna removes the first row with NaN returns

            if returns_df.empty or len(returns_df) < max(self.parameters["base_window"], self.parameters["long_window"]):
                 msg = f"Insufficient returns data after calculation (need at least {max(self.parameters['base_window'], self.parameters['long_window'])} rows, got {len(returns_df)})"
                 logger.warning(msg)
                 return AnalysisResult(analyzer_name=self.name, result={"error": msg}, is_valid=False)

        except Exception as e:
            logger.error(f"Error preparing price/returns DataFrame: {e}", exc_info=True)
            return AnalysisResult(analyzer_name=self.name, result={"error": f"Data preparation failed: {str(e)}"}, is_valid=False)

        # --- Perform Correlation Calculations ---
        try:
            base_window = self.parameters["base_window"]
            long_window = self.parameters["long_window"]

            # Base window correlation (using the most recent `base_window` returns)
            base_corr_matrix = self._calculate_correlation_matrix(returns_df, base_window)

            # Long window correlation (using the most recent `long_window` returns)
            long_corr_matrix = self._calculate_correlation_matrix(returns_df, long_window)

            # Calculate rolling correlations (optional)
            rolling_corrs = {}
            if self.parameters["calculate_rolling"]:
                rolling_corrs = self._calculate_rolling_correlations(returns_df, base_window)

            # Detect correlation regime changes
            regime_changes = self._detect_correlation_regime_changes(base_corr_matrix, long_corr_matrix)

            # Identify strongly correlated and inversely correlated pairs
            strong_correlations = self._identify_strong_correlations(base_corr_matrix)

            # Calculate correlation stability metrics (optional)
            correlation_stability = {}
            if self.parameters["calculate_stability"]:
                correlation_stability = self._calculate_correlation_stability(returns_df)

        except Exception as e:
            logger.error(f"Error during correlation calculations: {e}", exc_info=True)
            return AnalysisResult(analyzer_name=self.name, result={"error": f"Calculation failed: {str(e)}"}, is_valid=False)

        # --- Compile and Cache Results ---
        current_time = datetime.now()
        self.correlation_cache = {
            "base_corr_matrix": base_corr_matrix,
            "long_corr_matrix": long_corr_matrix,
            "rolling_correlations": rolling_corrs, # Store even if empty
            "last_returns_df": returns_df, # Cache returns for potential incremental update
            "last_updated": current_time
        }
        self.last_full_calculation_time = current_time

        # Prepare result data (convert DataFrames/Series to dicts for JSON compatibility)
        result_data = {
            "base_correlation_matrix": base_corr_matrix.to_dict() if base_corr_matrix is not None else None,
            "long_correlation_matrix": long_corr_matrix.to_dict() if long_corr_matrix is not None else None,
            "rolling_correlations": { # Convert rolling corr Series to dicts
                pair: series.to_dict() if series is not None else None
                for pair, series in rolling_corrs.items()
            },
            "regime_changes": regime_changes,
            "strong_correlations": strong_correlations,
            "correlation_stability": correlation_stability,
            "analyzed_instruments": valid_instruments,
            "timestamp": current_time.isoformat()
        }

        return AnalysisResult(analyzer_name=self.name, result=result_data, is_valid=True)

    def update_incremental(
        self, data: Dict[str, MarketData], previous_result: AnalysisResult
    ) -> AnalysisResult:
        """Update correlation analysis incrementally with new data.
        Currently falls back to full recalculation.

        Args:
            data: Dictionary mapping instrument symbols to their market data.
            previous_result: Results from previous analysis (not currently used effectively).

        Returns:
            Updated analysis results (via full recalculation).
        """
        # Check if a full recalculation is needed based on time
        update_freq_hours = self.parameters["update_frequency_hours"]
        if (self.last_full_calculation_time is None or
            (datetime.now() - self.last_full_calculation_time) > timedelta(hours=update_freq_hours)):
            logger.info(f"Performing full correlation recalculation (time since last > {update_freq_hours}h)")
            return self.analyze(data)

        # Basic check: If the set of instruments changes significantly, recalculate
        if set(data.keys()) != set(previous_result.result.get("analyzed_instruments", [])):
             logger.info("Instrument set changed, performing full correlation recalculation.")
             return self.analyze(data)

        # --- Attempt Incremental Update (Simplified - recalculates using cached + new data) ---
        # A true incremental update would involve updating rolling window calculations efficiently.
        # This version recalculates but avoids re-reading all historical data if possible.
        # For simplicity and robustness, falling back to full analyze() is often safer.
        logger.debug("Incremental update requested for CurrencyCorrelation, performing full recalculation for robustness.")
        return self.analyze(data)

        # --- More Complex Incremental Logic (Example - Requires careful implementation) ---
        # try:
        #     # 1. Get latest data point for each instrument
        #     # 2. Append to cached returns_df
        #     # 3. Recalculate correlations using the updated returns_df
        #     # This avoids re-reading full history but still recalculates correlations
        #     if not self.correlation_cache or "last_returns_df" not in self.correlation_cache:
        #         logger.info("Cache missing required data, falling back to full analysis.")
        #         return self.analyze(data)
        #     ...
        # except Exception as e:
        #     logger.error(f"Error during incremental correlation update: {e}", exc_info=True)
        #     logger.info("Falling back to full recalculation after incremental error.")
        #     return self.analyze(data)

    def _calculate_correlation_matrix(self, returns_df: pd.DataFrame, window: int) -> Optional[pd.DataFrame]:
        """Calculate correlation matrix for the most recent 'window' periods.

        Args:
            returns_df: DataFrame containing returns for multiple instruments.
            window: Window size for correlation calculation.

        Returns:
            Correlation matrix as DataFrame, or None if insufficient data.
        """
        if len(returns_df) < window:
            logger.warning(f"Insufficient data for correlation window {window} (need {window}, got {len(returns_df)})")
            return None
        # Use the most recent 'window' periods. Pandas .corr() handles NaNs within the window.
        return returns_df.tail(window).corr()

    def _calculate_rolling_correlations(
        self, returns_df: pd.DataFrame, window: int
    ) -> Dict[str, Optional[pd.Series]]:
        """Calculate rolling correlations for all pairs of instruments.

        Args:
            returns_df: DataFrame containing returns for multiple instruments.
            window: Window size for rolling correlation.

        Returns:
            Dictionary mapping pair names (e.g., "EURUSD_vs_GBPUSD") to rolling correlation Series.
            Returns None for a pair if calculation fails.
        """
        instruments = returns_df.columns
        result = {}
        min_periods_rolling = window // 2 # Minimum observations required in the window

        for i in range(len(instruments)):
            for j in range(i + 1, len(instruments)):
                inst1 = instruments[i]
                inst2 = instruments[j]
                pair_name = f"{inst1}_vs_{inst2}"
                try:
                    # Use the placeholder/imported rolling_correlation function
                    # Pandas rolling corr handles NaNs by default (pairwise)
                    rolling_corr = returns_df[inst1].rolling(window=window, min_periods=min_periods_rolling).corr(returns_df[inst2])
                    result[pair_name] = rolling_corr
                except Exception as e:
                    logger.error(f"Error calculating rolling correlation for {pair_name}: {e}")
                    result[pair_name] = None # Indicate failure for this pair

        return result

    def _detect_correlation_regime_changes(
        self, current_corr: Optional[pd.DataFrame], long_term_corr: Optional[pd.DataFrame]
    ) -> List[Dict[str, Any]]:
        """Detect significant changes between short-term and long-term correlation.

        Args:
            current_corr: Short-term correlation matrix (e.g., base_window).
            long_term_corr: Long-term correlation matrix (e.g., long_window).

        Returns:
            List of detected correlation regime changes.
        """
        changes = []
        if current_corr is None or long_term_corr is None:
            logger.warning("Cannot detect regime changes, missing correlation matrix.")
            return changes

        threshold = self.parameters["regime_change_threshold"]
        instruments = current_corr.columns

        # Ensure both matrices have the same instruments
        common_instruments = instruments.intersection(long_term_corr.columns)

        for i in common_instruments:
            for j in common_instruments:
                if i >= j:  # Skip diagonal and lower triangle (redundant)
                    continue

                try:
                    current_val = current_corr.loc[i, j]
                    long_term_val = long_term_corr.loc[i, j]

                    # Check if values are valid numbers
                    if pd.isna(current_val) or pd.isna(long_term_val):
                        continue

                    diff = current_val - long_term_val
                    if abs(diff) > threshold:
                        changes.append({
                            "pair": (i, j),
                            "current_correlation": round(current_val, 4),
                            "long_term_correlation": round(long_term_val, 4),
                            "change": round(diff, 4),
                            "is_significant": True
                        })
                except KeyError:
                    # Should not happen if using common_instruments, but handle defensively
                    logger.debug(f"KeyError comparing correlation for pair ({i}, {j}). Skipping.")
                    continue
                except Exception as e:
                    logger.error(f"Error comparing correlation for pair ({i}, {j}): {e}")
                    continue

        return changes

    def _identify_strong_correlations(self, corr_matrix: Optional[pd.DataFrame]) -> Dict[str, List[Tuple[str, str, float]]]:
        """Identify strongly positively and negatively correlated pairs from a matrix.

        Args:
            corr_matrix: Correlation matrix.

        Returns:
            Dictionary with lists of strongly correlated ('positive') and
            inversely correlated ('negative') pairs.
        """
        results: Dict[str, List[Tuple[str, str, float]]] = {
            "positive": [],
            "negative": []
        }
        if corr_matrix is None:
            logger.warning("Cannot identify strong correlations, missing correlation matrix.")
            return results

        threshold = self.parameters["correlation_threshold"]
        instruments = corr_matrix.columns

        for i in range(len(instruments)):
            for j in range(i + 1, len(instruments)):
                inst1 = instruments[i]
                inst2 = instruments[j]
                try:
                    correlation = corr_matrix.loc[inst1, inst2]

                    if pd.isna(correlation):
                        continue

                    if correlation > threshold:
                        results["positive"].append((inst1, inst2, round(correlation, 4)))
                    elif correlation < -threshold:
                        results["negative"].append((inst1, inst2, round(correlation, 4)))
                except KeyError:
                     logger.debug(f"KeyError identifying strong correlation for pair ({inst1}, {inst2}). Skipping.")
                     continue
                except Exception as e:
                    logger.error(f"Error identifying strong correlation for pair ({inst1}, {inst2}): {e}")
                    continue

        # Sort by absolute correlation strength (descending)
        results["positive"].sort(key=lambda x: x[2], reverse=True)
        results["negative"].sort(key=lambda x: x[2], reverse=False) # Most negative first

        return results

    def _calculate_correlation_stability(self, returns_df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate metrics for correlation stability by comparing two halves of the data.

        Args:
            returns_df: DataFrame containing returns.

        Returns:
            Dictionary of correlation stability metrics, or error message.
        """
        min_points_per_half = max(10, self.parameters["long_window"] // 2) # Need enough points in each half

        if len(returns_df) < min_points_per_half * 2:
            return {"error": f"Insufficient data for stability calculation (need at least {min_points_per_half * 2} rows)"}

        # Split data into two halves
        half_point = len(returns_df) // 2
        first_half_returns = returns_df.iloc[:half_point]
        second_half_returns = returns_df.iloc[half_point:]

        # Calculate correlation matrices for each half
        first_half_corr = first_half_returns.corr()
        second_half_corr = second_half_returns.corr()

        # Align matrices in case columns differ slightly (shouldn't happen with current logic)
        first_half_corr, second_half_corr = first_half_corr.align(second_half_corr, join='inner')

        if first_half_corr.empty:
             return {"error": "Correlation matrix for stability calculation is empty after alignment."}

        # Calculate the absolute difference between correlation matrices
        difference = (second_half_corr - first_half_corr).abs()

        # Calculate stability metrics from the upper triangle (excluding diagonal)
        upper_triangle_indices = np.triu_indices_from(difference.values, k=1)
        if not upper_triangle_indices or len(upper_triangle_indices[0]) == 0:
             return {"message": "No pairs to compare for stability (only one instrument?)."} # Only one instrument?

        flattened_diff = difference.values[upper_triangle_indices]
        # Filter out NaNs that might occur if a pair had insufficient data in one half
        valid_diffs = flattened_diff[~np.isnan(flattened_diff)]

        if len(valid_diffs) == 0:
             return {"error": "Could not calculate stability metrics (all differences were NaN)."}

        avg_change = valid_diffs.mean()
        max_change = valid_diffs.max()
        std_dev_change = valid_diffs.std()

        # Identify most stable and most volatile pairs
        indices = np.argsort(valid_diffs) # Sorts ascending (most stable first)
        row_indices, col_indices = upper_triangle_indices
        instruments = difference.columns

        # Map sorted indices back to original pair names
        num_pairs_to_show = min(3, len(valid_diffs)) # Show top/bottom 3 or fewer

        most_stable_pairs = [
            (instruments[row_indices[idx]], instruments[col_indices[idx]], round(valid_diffs[idx], 4))
            for idx in indices[:num_pairs_to_show]
        ]

        most_volatile_pairs = [
            (instruments[row_indices[idx]], instruments[col_indices[idx]], round(valid_diffs[idx], 4))
            for idx in indices[-num_pairs_to_show:][::-1] # Get last N and reverse for descending order
        ]

        return {
            "average_correlation_change": round(avg_change, 4),
            "maximum_correlation_change": round(max_change, 4),
            "stdev_correlation_change": round(std_dev_change, 4),
            "most_stable_pairs": most_stable_pairs,
            "most_volatile_pairs": most_volatile_pairs,
            "comparison_period_length": half_point
        }
