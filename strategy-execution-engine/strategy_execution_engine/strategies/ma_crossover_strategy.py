"""
Moving Average Crossover Strategy for Forex Trading

This module implements a simple MA crossover strategy as an example.
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime, timedelta

from .base_strategy import BaseStrategy
from strategy_execution_engine.clients.feature_store_client import FeatureStoreClient

class MACrossoverStrategy(BaseStrategy):
    """
    Moving Average Crossover Strategy

    This strategy generates buy signals when the fast MA crosses above the slow MA,
    and sell signals when the fast MA crosses below the slow MA.
    """

    def __init__(self, name: str, parameters: Dict[str, Any] = None):
        """
        Initialize the strategy with parameters

        Args:
            name: Strategy name
            parameters: Dictionary with strategy parameters
        """
        default_params = {
            "fast_period": 10,
            "slow_period": 30,
            "signal_threshold": 0.0,
            "take_profit_pips": 50,
            "stop_loss_pips": 25,
            "use_feature_store": True  # Flag to control whether to use feature store
        }

        # Override defaults with provided parameters
        merged_params = {**default_params, **(parameters or {})}

        super().__init__(name, merged_params)

        # Initialize feature store client if enabled
        self.feature_store_client = None
        if self.parameters["use_feature_store"]:
            self.feature_store_client = FeatureStoreClient(
                use_cache=True,
                cache_ttl=300  # 5 minutes cache TTL
            )

        # Set strategy metadata
        self.set_metadata({
            "description": "Moving Average Crossover strategy for Forex",
            "author": "Forex Trading Platform",
            "category": "trend_following",
            "tags": ["moving_average", "crossover", "trend"]
        })

    async def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals based on moving average crossover

        Args:
            data: Market data DataFrame with OHLCV data

        Returns:
            DataFrame with signals column (1 for buy, -1 for sell, 0 for no action)
        """
        # Make a copy to avoid modifying the input DataFrame
        result = data.copy()

        # Check if required columns exist
        if 'close' not in result.columns:
            raise ValueError("Input data must contain 'close' column")

        # Extract parameters
        fast_period = self.parameters["fast_period"]
        slow_period = self.parameters["slow_period"]

        # Get moving averages - either from feature store or calculate directly
        if self.feature_store_client and self.parameters["use_feature_store"]:
            try:
                # Get symbol from data
                symbol = data.get('symbol', 'unknown')
                if isinstance(symbol, pd.Series):
                    symbol = symbol.iloc[0]

                # Get start and end dates
                start_date = data.index[0] if not data.empty else datetime.now() - timedelta(days=30)
                end_date = data.index[-1] if not data.empty else datetime.now()

                # Get indicators from feature store
                indicators = await self.feature_store_client.get_indicators(
                    symbol=symbol,
                    start_date=start_date,
                    end_date=end_date,
                    timeframe='1h',  # Assuming 1h timeframe, adjust as needed
                    indicators=[f"sma_{fast_period}", f"sma_{slow_period}"]
                )

                # If we got data from the feature store, use it
                if not indicators.empty and f"sma_{fast_period}" in indicators.columns and f"sma_{slow_period}" in indicators.columns:
                    # Merge with our result dataframe
                    # First ensure timestamps are aligned
                    indicators.set_index('timestamp', inplace=True)

                    # Add the indicators to our result dataframe
                    result[f'ma_{fast_period}'] = indicators[f"sma_{fast_period}"]
                    result[f'ma_{slow_period}'] = indicators[f"sma_{slow_period}"]
                else:
                    # Fall back to direct calculation
                    result[f'ma_{fast_period}'] = result['close'].rolling(window=fast_period).mean()
                    result[f'ma_{slow_period}'] = result['close'].rolling(window=slow_period).mean()
            except Exception as e:
                # Log error and fall back to direct calculation
                print(f"Error getting indicators from feature store: {str(e)}, falling back to direct calculation")
                result[f'ma_{fast_period}'] = result['close'].rolling(window=fast_period).mean()
                result[f'ma_{slow_period}'] = result['close'].rolling(window=slow_period).mean()
        else:
            # Calculate moving averages directly
            result[f'ma_{fast_period}'] = result['close'].rolling(window=fast_period).mean()
            result[f'ma_{slow_period}'] = result['close'].rolling(window=slow_period).mean()

        # Initialize signal column
        result['signal'] = 0

        # Calculate crossover signals
        result['ma_diff'] = result[f'ma_{fast_period}'] - result[f'ma_{slow_period}']
        result['ma_diff_prev'] = result['ma_diff'].shift(1)

        # Buy signal: Fast MA crosses above Slow MA
        buy_condition = (result['ma_diff'] > 0) & (result['ma_diff_prev'] <= 0)
        result.loc[buy_condition, 'signal'] = 1

        # Sell signal: Fast MA crosses below Slow MA
        sell_condition = (result['ma_diff'] < 0) & (result['ma_diff_prev'] >= 0)
        result.loc[sell_condition, 'signal'] = -1

        # Add take profit and stop loss levels
        result['tp_level'] = np.nan
        result['sl_level'] = np.nan

        # Set TP/SL for buy signals
        buy_indices = result.index[buy_condition]
        for idx in buy_indices:
            entry_price = result.loc[idx, 'close']
            result.loc[idx, 'tp_level'] = entry_price + (self.parameters["take_profit_pips"] * 0.0001)
            result.loc[idx, 'sl_level'] = entry_price - (self.parameters["stop_loss_pips"] * 0.0001)

        # Set TP/SL for sell signals
        sell_indices = result.index[sell_condition]
        for idx in sell_indices:
            entry_price = result.loc[idx, 'close']
            result.loc[idx, 'tp_level'] = entry_price - (self.parameters["take_profit_pips"] * 0.0001)
            result.loc[idx, 'sl_level'] = entry_price + (self.parameters["stop_loss_pips"] * 0.0001)

        return result

    def calculate_position_size(self, data: pd.DataFrame, account_balance: float, risk_per_trade: float) -> pd.Series:
        """
        Calculate position size for each signal based on risk parameters

        Args:
            data: Market data with signals
            account_balance: Current account balance
            risk_per_trade: Risk per trade as percentage of account balance

        Returns:
            Series with position sizes
        """
        position_sizes = pd.Series(index=data.index, dtype=float)
        position_sizes.fillna(0, inplace=True)

        # Only calculate position size for rows with signals
        signal_rows = data[data['signal'] != 0].index

        for idx in signal_rows:
            # Skip if stop loss level is not defined
            if pd.isna(data.loc[idx, 'sl_level']):
                continue

            # Calculate risk amount in account currency
            risk_amount = account_balance * (risk_per_trade / 100)

            # Calculate stop loss distance in pips
            entry_price = data.loc[idx, 'close']
            stop_loss_price = data.loc[idx, 'sl_level']
            stop_loss_distance = abs(entry_price - stop_loss_price)

            # Calculate position size based on risk
            # For FOREX: position_size = risk_amount / (stop_loss_distance_in_pips * pip_value)
            # Here we simplify by assuming pip_value = 1
            if stop_loss_distance > 0:
                position_sizes[idx] = risk_amount / (stop_loss_distance * 10000)  # Convert to standard lots
            else:
                position_sizes[idx] = 0

        return position_sizes

    def get_required_indicators(self) -> List[Dict[str, Any]]:
        """
        Get list of technical indicators required by this strategy

        Returns:
            List of indicator configurations
        """
        return [
            {
                "name": "sma",
                "params": {"period": self.parameters["fast_period"]},
                "series": "close"
            },
            {
                "name": "sma",
                "params": {"period": self.parameters["slow_period"]},
                "series": "close"
            }
        ]

    def validate_parameters(self) -> Tuple[bool, Optional[str]]:
        """
        Validate strategy parameters

        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check required parameters
        required_params = ["fast_period", "slow_period", "take_profit_pips", "stop_loss_pips"]
        for param in required_params:
            if param not in self.parameters:
                return False, f"Missing required parameter: {param}"

        # Validate parameter values
        if self.parameters["fast_period"] >= self.parameters["slow_period"]:
            return False, "Fast period must be smaller than slow period"

        if self.parameters["fast_period"] <= 1:
            return False, "Fast period must be greater than 1"

        if self.parameters["slow_period"] <= 1:
            return False, "Slow period must be greater than 1"

        if self.parameters["take_profit_pips"] <= 0:
            return False, "Take profit must be greater than 0"

        if self.parameters["stop_loss_pips"] <= 0:
            return False, "Stop loss must be greater than 0"

        return True, None

    async def cleanup(self) -> None:
        """Clean up resources used by the strategy."""
        if self.feature_store_client:
            await self.feature_store_client.close()
            print("Closed feature store client connection")
