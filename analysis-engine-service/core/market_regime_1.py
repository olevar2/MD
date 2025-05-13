"""
Market Regime Analyzer Module

This module identifies current market regimes (trending, ranging, volatile, etc.)
and provides insights into market conditions that influence trading decisions.
It uses multiple statistical methods to classify market states and provide
probabilistic regime identification.
"""
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime
from enum import Enum
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from analysis_engine.analysis.base_analyzer import BaseAnalyzer
from analysis_engine.models.market_data import MarketData
from analysis_engine.models.analysis_result import AnalysisResult

logger = logging.getLogger(__name__)

class MarketRegimeType(Enum):
    """Enumeration of market regime types"""
    STRONG_UPTREND = "strong_uptrend"
    WEAK_UPTREND = "weak_uptrend"
    RANGING = "ranging"
    WEAK_DOWNTREND = "weak_downtrend"
    STRONG_DOWNTREND = "strong_downtrend"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    TRANSITION = "transition"
    UNCERTAIN = "uncertain"

class MarketRegimeAnalyzer(BaseAnalyzer):
    """
    Analyzer for identifying market regimes

    Uses a combination of volatility (ATR) and trend (ADX, MA slope) indicators
    to classify the market into different states.
    """

    DEFAULT_PARAMS = {
        "atr_period": 14,
        "adx_period": 14,
        "ma_fast_period": 20,
        "ma_slow_period": 50,
        "volatility_threshold_low": 0.5,  # Multiple of ATR average
        "volatility_threshold_high": 1.5, # Multiple of ATR average
        "adx_threshold_trend": 25,      # ADX level to indicate trending
        "ma_slope_threshold": 0.0001    # Minimum slope to consider trending
    }

    def __init__(self, parameters: Dict[str, Any] = None):
        """
        Initialize the Market Regime Analyzer

        Args:
            parameters: Configuration parameters for the analyzer, overriding defaults.
        """
        resolved_params = self.DEFAULT_PARAMS.copy()
        if parameters:
            resolved_params.update(parameters)
        super().__init__("Market Regime Analyzer", resolved_params)
        logger.info(f"Initialized MarketRegimeAnalyzer with params: {resolved_params}")

    async def analyze(self, market_data: MarketData) -> AnalysisResult:
        """
        Perform market regime analysis on the provided market data.

        Args:
            market_data: MarketData object containing OHLCV data.

        Returns:
            AnalysisResult containing the identified market regime.
        """
        df = market_data.data
        params = self.parameters

        if len(df) < max(params["atr_period"], params["adx_period"], params["ma_slow_period"]):
            logger.warning("Not enough data for Market Regime Analysis")
            return AnalysisResult(analyzer_name=self.name, result={"regime": MarketRegimeType.UNCERTAIN})

        # Calculate necessary indicators
        df_with_indicators = self._calculate_indicators(df.copy(), params)

        # Get the latest values
        latest_atr = df_with_indicators['atr'].iloc[-1]
        latest_adx = df_with_indicators['adx'].iloc[-1]
        latest_ma_fast_slope = df_with_indicators['ma_fast_slope'].iloc[-1]
        latest_ma_slow_slope = df_with_indicators['ma_slow_slope'].iloc[-1]

        # Determine Volatility Regime
        avg_atr = df_with_indicators['atr'].rolling(window=params["ma_slow_period"]).mean().iloc[-1]
        volatility_regime = MarketRegimeType.UNCERTAIN
        if not pd.isna(latest_atr) and not pd.isna(avg_atr) and avg_atr > 1e-9:
            atr_ratio = latest_atr / avg_atr
            if atr_ratio < params["volatility_threshold_low"]:
                volatility_regime = MarketRegimeType.LOW_VOLATILITY
            elif atr_ratio > params["volatility_threshold_high"]:
                volatility_regime = MarketRegimeType.HIGH_VOLATILITY
            else:
                 # Considered normal volatility, doesn't override trend/range
                 pass

        # Determine Trend Regime using ADX and MA slopes
        is_trending_adx = not pd.isna(latest_adx) and latest_adx > params["adx_threshold_trend"]
        is_uptrend_ma = not pd.isna(latest_ma_fast_slope) and latest_ma_fast_slope > params["ma_slope_threshold"] and \
                        not pd.isna(latest_ma_slow_slope) and latest_ma_slow_slope > params["ma_slope_threshold"]
        is_downtrend_ma = not pd.isna(latest_ma_fast_slope) and latest_ma_fast_slope < -params["ma_slope_threshold"] and \
                          not pd.isna(latest_ma_slow_slope) and latest_ma_slow_slope < -params["ma_slope_threshold"]

        trend_regime = MarketRegimeType.RANGING # Default to ranging if no trend detected

        if is_trending_adx:
            if is_uptrend_ma:
                # Strong if both MAs agree strongly, weak otherwise
                trend_regime = MarketRegimeType.STRONG_UPTREND if latest_ma_fast_slope > 2 * params["ma_slope_threshold"] else MarketRegimeType.WEAK_UPTREND
            elif is_downtrend_ma:
                trend_regime = MarketRegimeType.STRONG_DOWNTREND if latest_ma_fast_slope < -2 * params["ma_slope_threshold"] else MarketRegimeType.WEAK_DOWNTREND
            else:
                # ADX high but MAs conflicting/flat -> potentially transition or high vol range
                trend_regime = MarketRegimeType.TRANSITION
        else: # Not trending according to ADX
            # Check if MAs suggest a very weak trend or flat range
            if is_uptrend_ma:
                trend_regime = MarketRegimeType.WEAK_UPTREND
            elif is_downtrend_ma:
                trend_regime = MarketRegimeType.WEAK_DOWNTREND
            else:
                trend_regime = MarketRegimeType.RANGING

        # Combine Volatility and Trend/Range
        final_regime = trend_regime # Start with trend/range classification

        # Volatility can override or add context
        if volatility_regime == MarketRegimeType.HIGH_VOLATILITY:
            # If trending, it's a volatile trend. If ranging, it's a volatile range.
            # We can keep the trend/range label but note the high volatility.
            # Or, classify purely as HIGH_VOLATILITY if that's more useful.
            final_regime = MarketRegimeType.HIGH_VOLATILITY # Option: Prioritize high vol
        elif volatility_regime == MarketRegimeType.LOW_VOLATILITY:
            # Low volatility usually reinforces ranging or indicates a weak/pausing trend.
            if final_regime == MarketRegimeType.RANGING:
                 final_regime = MarketRegimeType.LOW_VOLATILITY # Option: Prioritize low vol if ranging
            # If trending, keep the trend label but note low vol contextually.

        analysis_data = {
            "regime": final_regime,
            "details": {
                "atr": latest_atr,
                "adx": latest_adx,
                "ma_fast_slope": latest_ma_fast_slope,
                "ma_slow_slope": latest_ma_slow_slope,
                "volatility_state": volatility_regime.value if volatility_regime != MarketRegimeType.UNCERTAIN else "normal",
                "trend_state": trend_regime.value
            }
        }

        return AnalysisResult(analyzer_name=self.name, result=analysis_data)

    def _calculate_indicators(self, df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        """Helper to calculate ATR, ADX, and MA slopes."""
        # ATR
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['atr'] = tr.rolling(window=params["atr_period"]).mean()

        # ADX
        adx_period = params["adx_period"]
        plus_dm = df['high'].diff()
        minus_dm = df['low'].diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm > 0] = 0
        minus_dm = np.abs(minus_dm)

        # Smoothed True Range (similar to ATR calculation)
        atr_adx = tr.ewm(alpha=1/adx_period, adjust=False).mean()

        # Smoothed Directional Movement
        plus_di = 100 * (plus_dm.ewm(alpha=1/adx_period, adjust=False).mean() / atr_adx)
        minus_di = 100 * (minus_dm.ewm(alpha=1/adx_period, adjust=False).mean() / atr_adx)

        # Directional Movement Index (DX)
        dx = 100 * (np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-9)) # Add epsilon to avoid div by zero

        # Average Directional Index (ADX)
        df['adx'] = dx.ewm(alpha=1/adx_period, adjust=False).mean()
        df['+DI'] = plus_di # Optional: Include DI lines
        df['-DI'] = minus_di # Optional: Include DI lines

        # Moving Averages and Slopes
        df['ma_fast'] = df['close'].rolling(window=params["ma_fast_period"]).mean()
        df['ma_slow'] = df['close'].rolling(window=params["ma_slow_period"]).mean()

        # Calculate slope (change over 1 period)
        df['ma_fast_slope'] = df['ma_fast'].diff()
        df['ma_slow_slope'] = df['ma_slow'].diff()

        return df

    async def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate market regime and add columns to the DataFrame.

        Args:
            df: DataFrame containing OHLCV data.

        Returns:
            DataFrame with added market regime columns.
        """
        params = self.parameters
        result_df = df.copy()

        if len(result_df) < max(params["atr_period"], params["adx_period"], params["ma_slow_period"]):
            result_df['market_regime'] = MarketRegimeType.UNCERTAIN.value
            result_df['volatility_state'] = MarketRegimeType.UNCERTAIN.value
            result_df['trend_state'] = MarketRegimeType.UNCERTAIN.value
            return result_df

        # Calculate necessary indicators
        result_df = self._calculate_indicators(result_df, params)

        # Calculate rolling average ATR for thresholding
        avg_atr = result_df['atr'].rolling(window=params["ma_slow_period"]).mean()

        # Apply regime logic row-by-row (vectorization possible but complex for combined logic)
        regimes = []
        vol_states = []
        trend_states = []
        for i in range(len(result_df)):
            if i < max(params["atr_period"], params["adx_period"], params["ma_slow_period"]):
                regimes.append(MarketRegimeType.UNCERTAIN.value)
                vol_states.append(MarketRegimeType.UNCERTAIN.value)
                trend_states.append(MarketRegimeType.UNCERTAIN.value)
                continue

            latest_atr = result_df['atr'].iloc[i]
            latest_adx = result_df['adx'].iloc[i]
            latest_ma_fast_slope = result_df['ma_fast_slope'].iloc[i]
            latest_ma_slow_slope = result_df['ma_slow_slope'].iloc[i]
            current_avg_atr = avg_atr.iloc[i]

            # Determine Volatility Regime
            volatility_regime = MarketRegimeType.UNCERTAIN
            if not pd.isna(latest_atr) and not pd.isna(current_avg_atr) and current_avg_atr > 1e-9:
                atr_ratio = latest_atr / current_avg_atr
                if atr_ratio < params["volatility_threshold_low"]:
                    volatility_regime = MarketRegimeType.LOW_VOLATILITY
                elif atr_ratio > params["volatility_threshold_high"]:
                    volatility_regime = MarketRegimeType.HIGH_VOLATILITY
                else:
                    volatility_regime = None # Normal volatility

            # Determine Trend Regime
            is_trending_adx = not pd.isna(latest_adx) and latest_adx > params["adx_threshold_trend"]
            is_uptrend_ma = not pd.isna(latest_ma_fast_slope) and latest_ma_fast_slope > params["ma_slope_threshold"] and \
                            not pd.isna(latest_ma_slow_slope) and latest_ma_slow_slope > params["ma_slope_threshold"]
            is_downtrend_ma = not pd.isna(latest_ma_fast_slope) and latest_ma_fast_slope < -params["ma_slope_threshold"] and \
                              not pd.isna(latest_ma_slow_slope) and latest_ma_slow_slope < -params["ma_slope_threshold"]

            trend_regime = MarketRegimeType.RANGING
            if is_trending_adx:
                if is_uptrend_ma:
                    trend_regime = MarketRegimeType.STRONG_UPTREND if latest_ma_fast_slope > 2 * params["ma_slope_threshold"] else MarketRegimeType.WEAK_UPTREND
                elif is_downtrend_ma:
                    trend_regime = MarketRegimeType.STRONG_DOWNTREND if latest_ma_fast_slope < -2 * params["ma_slope_threshold"] else MarketRegimeType.WEAK_DOWNTREND
                else:
                    trend_regime = MarketRegimeType.TRANSITION
            else:
                if is_uptrend_ma:
                    trend_regime = MarketRegimeType.WEAK_UPTREND
                elif is_downtrend_ma:
                    trend_regime = MarketRegimeType.WEAK_DOWNTREND
                else:
                    trend_regime = MarketRegimeType.RANGING

            # Combine
            final_regime = trend_regime
            if volatility_regime == MarketRegimeType.HIGH_VOLATILITY:
                final_regime = MarketRegimeType.HIGH_VOLATILITY
            elif volatility_regime == MarketRegimeType.LOW_VOLATILITY and final_regime == MarketRegimeType.RANGING:
                final_regime = MarketRegimeType.LOW_VOLATILITY

            regimes.append(final_regime.value)
            vol_states.append(volatility_regime.value if volatility_regime else "normal")
            trend_states.append(trend_regime.value)

        result_df['market_regime'] = regimes
        result_df['volatility_state'] = vol_states
        result_df['trend_state'] = trend_states

        return result_df

# Example Usage
if __name__ == '__main__':
    import asyncio

    async def run_example():
    """
    Run example.
    
    """

        # Create sample data (more needed for proper analysis)
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        price = (np.sin(np.linspace(0, 10, 100)) * 5 +
                np.random.randn(100) * 1.5 + 100)
        sample_df = pd.DataFrame({'close': price}, index=dates)
        sample_df['high'] = sample_df['close'] + np.random.rand(100) * 2
        sample_df['low'] = sample_df['close'] - np.random.rand(100) * 2
        sample_df['open'] = sample_df['low'] + (sample_df['high'] - sample_df['low']) * np.random.rand(100)
        sample_df['volume'] = np.random.randint(1000, 5000, 100)

        print("Sample Data Head:")
        print(sample_df.head())

        # Initialize analyzer
        analyzer = MarketRegimeAnalyzer()
        market_data_obj = MarketData(symbol="EURUSD", timeframe="D1", data=sample_df)

        # Perform analysis (gets latest regime)
        analysis_result = await analyzer.analyze(market_data_obj)
        print(f"\nLatest Market Regime Analysis Result:")
        print(analysis_result.result)

        # Calculate indicator columns for the whole DataFrame
        result_with_indicator = await analyzer.calculate(sample_df)
        print("\nDataFrame with Market Regime Columns (tail):")
        print(result_with_indicator.tail(10)[['close', 'atr', 'adx', 'ma_fast_slope', 'market_regime', 'volatility_state', 'trend_state']])

        # Show counts of different regimes identified
        print("\nRegime Counts:")
        print(result_with_indicator['market_regime'].value_counts())

    # Run the async example
    asyncio.run(run_example())
