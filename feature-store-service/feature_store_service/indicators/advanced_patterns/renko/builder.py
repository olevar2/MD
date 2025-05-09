"""
Renko Chart Builder Module.

This module provides functionality to convert OHLC data to Renko bricks.
"""

from typing import List, Dict, Any, Optional, Tuple, Union
import pandas as pd
import numpy as np
from datetime import datetime

from feature_store_service.indicators.base_indicator import BaseIndicator
from feature_store_service.indicators.advanced_patterns.renko.models import (
    RenkoBrick,
    RenkoDirection
)


class RenkoChartBuilder(BaseIndicator):
    """
    Builds Renko charts from OHLC data.
    
    Renko charts filter out market noise by focusing solely on price movements
    of a specified size.
    """
    
    category = "chart_type"
    
    def __init__(
        self,
        brick_size: Optional[float] = None,
        brick_method: str = "atr",
        atr_period: int = 14,
        price_field: str = "close",
        **kwargs
    ):
        """
        Initialize the Renko chart builder.
        
        Args:
            brick_size: Size of each brick (None = auto-calculate using brick_method)
            brick_method: Method to calculate brick size ('atr', 'fixed', 'percentage')
            atr_period: Period for ATR calculation when brick_method='atr'
            price_field: Price field to use ('close', 'open', 'high', 'low', 'hl2', 'hlc3', 'ohlc4')
            **kwargs: Additional parameters
        """
        self.brick_size = brick_size
        self.brick_method = brick_method
        self.atr_period = atr_period
        self.price_field = price_field
    
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Renko chart data from OHLC data.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with Renko chart data
        """
        # Validate input data
        required_cols = ['open', 'high', 'low', 'close']
        for col in required_cols:
            if col not in data.columns:
                raise ValueError(f"Data must contain '{col}' column")
        
        # Make a copy to avoid modifying the input data
        result = data.copy()
        
        # Build Renko bricks
        bricks = self.build_renko_bricks(result)
        
        # Convert bricks to DataFrame columns
        result['renko_brick_direction'] = 0
        result['renko_brick_open'] = np.nan
        result['renko_brick_close'] = np.nan
        result['renko_brick_size'] = np.nan
        result['renko_brick_index'] = np.nan
        
        # Map bricks to original DataFrame
        for brick in bricks:
            if brick.close_time is not None:
                # Find the index in the original DataFrame
                idx = result.index[result.index <= brick.close_time][-1]
                
                # Set brick values
                result.loc[idx, 'renko_brick_direction'] = 1 if brick.direction == RenkoDirection.UP else -1
                result.loc[idx, 'renko_brick_open'] = brick.open_price
                result.loc[idx, 'renko_brick_close'] = brick.close_price
                result.loc[idx, 'renko_brick_size'] = brick.size
                result.loc[idx, 'renko_brick_index'] = brick.index
        
        # Forward fill brick index to make it easier to identify which brick each row belongs to
        result['renko_brick_index'] = result['renko_brick_index'].fillna(method='ffill')
        
        # Add brick count column
        result['renko_brick_count'] = result['renko_brick_index'].notna().cumsum()
        
        # Add trend column (consecutive bricks in same direction)
        result['renko_trend'] = (result['renko_brick_direction'].fillna(0) != 0).astype(int)
        result['renko_trend'] = result['renko_trend'].replace(0, np.nan)
        
        # Calculate trend direction and length
        direction_changes = result['renko_brick_direction'].fillna(0).diff().ne(0) & result['renko_brick_direction'].fillna(0).ne(0)
        result['renko_trend_start'] = direction_changes.astype(int)
        result['renko_trend_length'] = result.groupby((direction_changes.cumsum()))['renko_trend'].cumsum()
        
        return result
    
    def build_renko_bricks(self, data: pd.DataFrame) -> List[RenkoBrick]:
        """
        Build Renko bricks from OHLC data.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            List of RenkoBrick objects
        """
        # Determine brick size
        brick_size = self._calculate_brick_size(data)
        
        # Get price series based on price_field
        if self.price_field == 'close':
            prices = data['close']
        elif self.price_field == 'open':
            prices = data['open']
        elif self.price_field == 'high':
            prices = data['high']
        elif self.price_field == 'low':
            prices = data['low']
        elif self.price_field == 'hl2':
            prices = (data['high'] + data['low']) / 2
        elif self.price_field == 'hlc3':
            prices = (data['high'] + data['low'] + data['close']) / 3
        elif self.price_field == 'ohlc4':
            prices = (data['open'] + data['high'] + data['low'] + data['close']) / 4
        else:
            raise ValueError(f"Invalid price_field: {self.price_field}")
        
        # Initialize bricks list
        bricks = []
        
        # Initialize first brick
        current_price = prices.iloc[0]
        current_direction = None
        brick_open = current_price
        brick_close = current_price
        
        # Process each price
        for i, (timestamp, price) in enumerate(zip(data.index, prices)):
            if current_direction is None:
                # First brick
                if price >= current_price + brick_size:
                    # Up brick
                    current_direction = RenkoDirection.UP
                    brick_close = brick_open + brick_size
                    bricks.append(RenkoBrick(
                        direction=current_direction,
                        open_price=brick_open,
                        close_price=brick_close,
                        open_time=timestamp,
                        close_time=timestamp,
                        index=len(bricks)
                    ))
                    brick_open = brick_close
                    current_price = brick_close
                elif price <= current_price - brick_size:
                    # Down brick
                    current_direction = RenkoDirection.DOWN
                    brick_close = brick_open - brick_size
                    bricks.append(RenkoBrick(
                        direction=current_direction,
                        open_price=brick_open,
                        close_price=brick_close,
                        open_time=timestamp,
                        close_time=timestamp,
                        index=len(bricks)
                    ))
                    brick_open = brick_close
                    current_price = brick_close
            else:
                # Subsequent bricks
                if current_direction == RenkoDirection.UP:
                    # Check for up brick
                    while price >= current_price + brick_size:
                        brick_close = brick_open + brick_size
                        bricks.append(RenkoBrick(
                            direction=current_direction,
                            open_price=brick_open,
                            close_price=brick_close,
                            open_time=timestamp,
                            close_time=timestamp,
                            index=len(bricks)
                        ))
                        brick_open = brick_close
                        current_price = brick_close
                    
                    # Check for reversal (2 bricks)
                    if price <= current_price - 2 * brick_size:
                        # Reversal to down
                        current_direction = RenkoDirection.DOWN
                        brick_close = brick_open - brick_size
                        bricks.append(RenkoBrick(
                            direction=current_direction,
                            open_price=brick_open,
                            close_price=brick_close,
                            open_time=timestamp,
                            close_time=timestamp,
                            index=len(bricks)
                        ))
                        brick_open = brick_close
                        current_price = brick_close
                        
                        # Add second down brick
                        brick_close = brick_open - brick_size
                        bricks.append(RenkoBrick(
                            direction=current_direction,
                            open_price=brick_open,
                            close_price=brick_close,
                            open_time=timestamp,
                            close_time=timestamp,
                            index=len(bricks)
                        ))
                        brick_open = brick_close
                        current_price = brick_close
                else:  # current_direction == RenkoDirection.DOWN
                    # Check for down brick
                    while price <= current_price - brick_size:
                        brick_close = brick_open - brick_size
                        bricks.append(RenkoBrick(
                            direction=current_direction,
                            open_price=brick_open,
                            close_price=brick_close,
                            open_time=timestamp,
                            close_time=timestamp,
                            index=len(bricks)
                        ))
                        brick_open = brick_close
                        current_price = brick_close
                    
                    # Check for reversal (2 bricks)
                    if price >= current_price + 2 * brick_size:
                        # Reversal to up
                        current_direction = RenkoDirection.UP
                        brick_close = brick_open + brick_size
                        bricks.append(RenkoBrick(
                            direction=current_direction,
                            open_price=brick_open,
                            close_price=brick_close,
                            open_time=timestamp,
                            close_time=timestamp,
                            index=len(bricks)
                        ))
                        brick_open = brick_close
                        current_price = brick_close
                        
                        # Add second up brick
                        brick_close = brick_open + brick_size
                        bricks.append(RenkoBrick(
                            direction=current_direction,
                            open_price=brick_open,
                            close_price=brick_close,
                            open_time=timestamp,
                            close_time=timestamp,
                            index=len(bricks)
                        ))
                        brick_open = brick_close
                        current_price = brick_close
        
        return bricks
    
    def _calculate_brick_size(self, data: pd.DataFrame) -> float:
        """
        Calculate the brick size based on the specified method.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            Brick size
        """
        if self.brick_size is not None:
            return self.brick_size
        
        if self.brick_method == 'atr':
            # Calculate ATR
            high_low = data['high'] - data['low']
            high_close_prev = abs(data['high'] - data['close'].shift(1))
            low_close_prev = abs(data['low'] - data['close'].shift(1))
            
            true_range = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
            atr = true_range.rolling(window=self.atr_period).mean().iloc[-1]
            
            return atr
        elif self.brick_method == 'percentage':
            # Default to 1% of current price
            percentage = 0.01
            return data['close'].iloc[-1] * percentage
        else:  # 'fixed' or any other value
            # Default to 1% of average price
            return data['close'].mean() * 0.01