"""
Order Flow Indicators Module

This module provides implementations for Order Flow analysis indicators:
- OrderFlowIndicator: Base class for order flow analysis
- CumulativeDeltaIndicator: Implementation of Cumulative Delta analysis
- VolumeProfilerRealTime: Implementation of Real-time Volume Profile analysis

These indicators analyze order flow data to provide insights into buying and selling
pressure, volume imbalances, and market dynamics beyond traditional price action.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
from dataclasses import dataclass, field
from collections import defaultdict
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from analysis_engine.analysis.advanced_ta.base import AdvancedAnalysisBase
from analysis_engine.utils.validation import validate_dataframe
logger = logging.getLogger(__name__)


from analysis_engine.resilience.utils import (
    with_resilience,
    with_analysis_resilience,
    with_database_resilience
)

@dataclass
class OrderFlowLevel:
    """Represents a price level with buying and selling volume."""
    price: float
    buy_volume: int = 0
    sell_volume: int = 0
    total_volume: int = 0
    delta: int = 0

    def update(self, buy_vol: int=0, sell_vol: int=0):
        """Update the order flow level with new volume data."""
        self.buy_volume += buy_vol
        self.sell_volume += sell_vol
        self.total_volume = self.buy_volume + self.sell_volume
        self.delta = self.buy_volume - self.sell_volume


@dataclass
class OrderFlowBar:
    """Represents an order flow bar with price levels and statistics."""
    time: datetime
    open: float
    high: float
    low: float
    close: float
    total_volume: int = 0
    buy_volume: int = 0
    sell_volume: int = 0
    delta: int = 0
    levels: Dict[float, OrderFlowLevel] = field(default_factory=dict)
    point_of_control: float = 0.0

    @with_resilience('update_from_levels')
    def update_from_levels(self):
        """Update bar statistics from level data."""
        self.total_volume = 0
        self.buy_volume = 0
        self.sell_volume = 0
        max_volume = 0
        poc = 0.0
        for price, level in self.levels.items():
            self.total_volume += level.total_volume
            self.buy_volume += level.buy_volume
            self.sell_volume += level.sell_volume
            if level.total_volume > max_volume:
                max_volume = level.total_volume
                poc = price
        self.delta = self.buy_volume - self.sell_volume
        self.point_of_control = poc


class OrderFlowIndicator(AdvancedAnalysisBase):
    """
    Base class for Order Flow analysis indicators.
    
    Order Flow analysis focuses on the interaction between buyers and sellers at each price level,
    providing insights into supply and demand dynamics beyond traditional price charts.
    """

    def __init__(self, name: str='OrderFlowIndicator', tick_size: float=
        0.0001, price_precision: int=5):
        """
        Initialize the OrderFlowIndicator.
        
        Args:
            name: Name of the indicator
            tick_size: Minimum price movement for the instrument
            price_precision: Number of decimal places for price rounding
        """
        super().__init__(name=name)
        self.tick_size = tick_size
        self.price_precision = price_precision

    def calculate(self, data: pd.DataFrame) ->pd.DataFrame:
        """
        Base method to calculate order flow metrics.
        
        Args:
            data: DataFrame with OHLCV data and volume direction information
            
        Returns:
            DataFrame with order flow metrics
        """
        return data.copy()

    def _round_price(self, price: float) ->float:
        """
        Round the price to the specified precision.
        
        Args:
            price: Price value to round
            
        Returns:
            Rounded price
        """
        return round(price, self.price_precision)

    def _infer_trade_direction(self, data: pd.DataFrame) ->pd.DataFrame:
        """
        Infer trade direction (buy/sell) from price movements.
        
        When tick data with actual buy/sell information is not available,
        this method uses price action to estimate trade direction.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with inferred buy/sell volumes
        """
        result = data.copy()
        if 'buy_volume' not in result.columns:
            result['buy_volume'] = 0
        if 'sell_volume' not in result.columns:
            result['sell_volume'] = 0
        result['price_change'] = result['close'].diff()
        buy_mask = result['price_change'] > 0
        sell_mask = result['price_change'] < 0
        unchanged_mask = result['price_change'] == 0
        result.loc[buy_mask, 'buy_volume'] = result.loc[buy_mask, 'volume']
        result.loc[sell_mask, 'sell_volume'] = result.loc[sell_mask, 'volume']
        result.loc[unchanged_mask, 'buy_volume'] = result.loc[
            unchanged_mask, 'volume'] / 2
        result.loc[unchanged_mask, 'sell_volume'] = result.loc[
            unchanged_mask, 'volume'] / 2
        return result

    def _create_order_flow_bars(self, data: pd.DataFrame) ->List[OrderFlowBar]:
        """
        Create order flow bars from DataFrame.
        
        Args:
            data: DataFrame with OHLCV and buy/sell volume
            
        Returns:
            List of OrderFlowBar objects
        """
        if ('buy_volume' not in data.columns or 'sell_volume' not in data.
            columns):
            data = self._infer_trade_direction(data)
        bars = []
        for idx, row in data.iterrows():
            time = row.name if isinstance(data.index, pd.DatetimeIndex
                ) else idx
            bar = OrderFlowBar(time=time, open=row['open'], high=row['high'
                ], low=row['low'], close=row['close'], total_volume=row[
                'volume'], buy_volume=row['buy_volume'], sell_volume=row[
                'sell_volume'], delta=row['buy_volume'] - row['sell_volume'])
            bars.append(bar)
        return bars


class CumulativeDeltaIndicator(OrderFlowIndicator):
    """
    Implementation of Cumulative Delta (CD) analysis.
    
    Cumulative Delta tracks the net buying/selling pressure over time by
    calculating the cumulative sum of the difference between buying and
    selling volume, providing insights into order flow trends and potential
    divergences with price.
    """

    def __init__(self, tick_size: float=0.0001, price_precision: int=5,
        reset_session: bool=False):
        """
        Initialize the CumulativeDeltaIndicator.
        
        Args:
            tick_size: Minimum price movement for the instrument
            price_precision: Number of decimal places for price rounding
            reset_session: Whether to reset cumulative delta at the start of each session
        """
        super().__init__(name='CumulativeDeltaIndicator', tick_size=
            tick_size, price_precision=price_precision)
        self.reset_session = reset_session

    def calculate(self, data: pd.DataFrame) ->pd.DataFrame:
        """
        Calculate Cumulative Delta metrics.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with Cumulative Delta metrics
        """
        validate_dataframe(data, required_columns=['open', 'high', 'low',
            'close', 'volume'])
        if ('buy_volume' not in data.columns or 'sell_volume' not in data.
            columns):
            data = self._infer_trade_direction(data)
        result = data.copy()
        result['delta'] = result['buy_volume'] - result['sell_volume']
        if self.reset_session and isinstance(result.index, pd.DatetimeIndex):
            result['session_date'] = result.index.date
            result['cumulative_delta'] = result.groupby('session_date')['delta'
                ].cumsum()
            result.drop('session_date', axis=1, inplace=True)
        else:
            result['cumulative_delta'] = result['delta'].cumsum()
        result['cd_high'] = result.groupby(result.index.date)[
            'cumulative_delta'].cummax() if isinstance(result.index, pd.
            DatetimeIndex) else result['cumulative_delta'].cummax()
        result['cd_low'] = result.groupby(result.index.date)['cumulative_delta'
            ].cummin() if isinstance(result.index, pd.DatetimeIndex
            ) else result['cumulative_delta'].cummin()
        result['price_direction'] = np.sign(result['close'].diff())
        result['cd_direction'] = np.sign(result['cumulative_delta'].diff())
        result['delta_divergence'] = result['price_direction'] * result[
            'cd_direction']
        return result

    def visualize(self, data: pd.DataFrame, window: Optional[int]=None
        ) ->plt.Figure:
        """
        Create a visualization of Cumulative Delta and price.
        
        Args:
            data: DataFrame with CD metrics
            window: Optional window size to visualize (most recent N bars)
            
        Returns:
            Matplotlib Figure with CD visualization
        """
        if window is not None and window > 0:
            plot_data = data.iloc[-window:]
        else:
            plot_data = data
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True,
            gridspec_kw={'height_ratios': [2, 1]})
        ax1.plot(plot_data.index, plot_data['close'], label='Close Price')
        ax1.set_ylabel('Price')
        ax1.set_title('Price and Cumulative Delta')
        ax1.legend(loc='upper left')
        ax1.grid(True)
        ax2.plot(plot_data.index, plot_data['cumulative_delta'], label=
            'Cumulative Delta', color='blue')
        ax2.fill_between(plot_data.index, 0, plot_data['cumulative_delta'],
            where=plot_data['cumulative_delta'] >= 0, color='green', alpha=0.3)
        ax2.fill_between(plot_data.index, 0, plot_data['cumulative_delta'],
            where=plot_data['cumulative_delta'] < 0, color='red', alpha=0.3)
        ax2.set_ylabel('Cumulative Delta')
        ax2.legend(loc='upper left')
        ax2.grid(True)
        divergence_points = plot_data[plot_data['delta_divergence'] < 0]
        if not divergence_points.empty:
            ax1.scatter(divergence_points.index, divergence_points['close'],
                color='purple', marker='^', label='Divergence')
            ax2.scatter(divergence_points.index, divergence_points[
                'cumulative_delta'], color='purple', marker='^')
        if isinstance(plot_data.index, pd.DatetimeIndex):
            fig.autofmt_xdate()
        plt.tight_layout()
        return fig


class VolumeProfilerRealTime(OrderFlowIndicator):
    """
    Implementation of Real-time Volume Profile analysis.
    
    Volume Profile provides a horizontal histogram of volume traded at each price level,
    showing where most trading activity occurred and identifying key reference levels.
    This real-time implementation updates continuously with new data.
    """

    def __init__(self, tick_size: float=0.0001, price_precision: int=5,
        value_area_volume: float=0.7, num_profile_bars: int=30,
        dynamic_profile: bool=True):
        """
        Initialize the VolumeProfilerRealTime.
        
        Args:
            tick_size: Minimum price movement for the instrument
            price_precision: Number of decimal places for price rounding
            value_area_volume: Percentage of volume to include in the value area (default: 0.7 for 70%)
            num_profile_bars: Number of bars to include in each profile
            dynamic_profile: Whether to use a rolling window (True) or fixed periods (False)
        """
        super().__init__(name='VolumeProfilerRealTime', tick_size=tick_size,
            price_precision=price_precision)
        self.value_area_volume = value_area_volume
        self.num_profile_bars = num_profile_bars
        self.dynamic_profile = dynamic_profile
        self.profiles = {}

    def calculate(self, data: pd.DataFrame) ->pd.DataFrame:
        """
        Calculate Volume Profile metrics.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with Volume Profile metrics
        """
        validate_dataframe(data, required_columns=['open', 'high', 'low',
            'close', 'volume'])
        if ('buy_volume' not in data.columns or 'sell_volume' not in data.
            columns):
            data = self._infer_trade_direction(data)
        result = data.copy()
        price_levels = self._create_price_levels(result)
        result['delta'] = result['buy_volume'] - result['sell_volume']
        result['delta_percent'] = result['delta'] / result['volume'] * 100
        if self.dynamic_profile:
            self._calculate_rolling_profiles(result)
        else:
            self._calculate_fixed_profiles(result)
        result['vp_point_of_control'] = np.nan
        result['vp_value_area_high'] = np.nan
        result['vp_value_area_low'] = np.nan
        result['vp_delta_at_poc'] = np.nan
        for profile_id, profile in self.profiles.items():
            if isinstance(profile_id, tuple):
                start_date, end_date = profile_id
                mask = (result.index >= start_date) & (result.index <= end_date
                    )
            else:
                end_idx = profile_id
                start_idx = max(0, end_idx - self.num_profile_bars + 1)
                mask = result.index.isin(result.index[start_idx:end_idx + 1])
            result.loc[mask, 'vp_point_of_control'] = profile[
                'point_of_control']
            result.loc[mask, 'vp_value_area_high'] = profile['value_area_high']
            result.loc[mask, 'vp_value_area_low'] = profile['value_area_low']
            result.loc[mask, 'vp_delta_at_poc'] = profile['delta_at_poc']
        return result

    def _create_price_levels(self, data: pd.DataFrame) ->Dict[float,
        OrderFlowLevel]:
        """
        Create price levels with volume information.
        
        Args:
            data: DataFrame with OHLCV and buy/sell volume
            
        Returns:
            Dictionary mapping price levels to OrderFlowLevel objects
        """
        price_levels = {}
        min_price = data['low'].min()
        max_price = data['high'].max()
        current_price = self._round_price(min_price)
        while current_price <= max_price:
            price_levels[current_price] = OrderFlowLevel(price=current_price)
            current_price = self._round_price(current_price + self.tick_size)
        for idx, row in data.iterrows():
            bar_min = self._round_price(row['low'])
            bar_max = self._round_price(row['high'])
            num_levels = max(1, int((bar_max - bar_min) / self.tick_size) + 1)
            buy_per_level = row['buy_volume'] / num_levels
            sell_per_level = row['sell_volume'] / num_levels
            current_price = bar_min
            while current_price <= bar_max:
                if current_price in price_levels:
                    price_levels[current_price].update(buy_vol=int(
                        buy_per_level), sell_vol=int(sell_per_level))
                current_price = self._round_price(current_price + self.
                    tick_size)
        return price_levels

    def _calculate_rolling_profiles(self, data: pd.DataFrame):
        """
        Calculate rolling volume profiles.
        
        This creates a volume profile for each bar using a rolling window of the previous N bars.
        
        Args:
            data: DataFrame with OHLCV and buy/sell volume
        """
        self.profiles = {}
        for i in range(self.num_profile_bars - 1, len(data)):
            window_data = data.iloc[max(0, i - self.num_profile_bars + 1):i + 1
                ]
            price_levels = self._create_price_levels(window_data)
            poc_price = max(price_levels, key=lambda p: price_levels[p].
                total_volume, default=0)
            poc_level = price_levels.get(poc_price, OrderFlowLevel(price=0))
            value_area = self._calculate_value_area(price_levels)
            self.profiles[i] = {'price_levels': price_levels,
                'point_of_control': poc_price, 'value_area_high':
                value_area[1], 'value_area_low': value_area[0],
                'delta_at_poc': poc_level.delta if poc_level else 0,
                'start_time': window_data.index[0] if len(window_data) > 0 else
                None, 'end_time': window_data.index[-1] if len(window_data) >
                0 else None}

    def _calculate_fixed_profiles(self, data: pd.DataFrame):
        """
        Calculate fixed-period volume profiles.
        
        This creates volume profiles for specific periods like days or weeks.
        
        Args:
            data: DataFrame with OHLCV and buy/sell volume
        """
        self.profiles = {}
        if isinstance(data.index, pd.DatetimeIndex):
            grouped = data.groupby(data.index.date)
            for date, group_data in grouped:
                start_time = group_data.index.min()
                end_time = group_data.index.max()
                price_levels = self._create_price_levels(group_data)
                poc_price = max(price_levels, key=lambda p: price_levels[p]
                    .total_volume, default=0)
                poc_level = price_levels.get(poc_price, OrderFlowLevel(price=0)
                    )
                value_area = self._calculate_value_area(price_levels)
                self.profiles[start_time, end_time] = {'price_levels':
                    price_levels, 'point_of_control': poc_price,
                    'value_area_high': value_area[1], 'value_area_low':
                    value_area[0], 'delta_at_poc': poc_level.delta if
                    poc_level else 0, 'date': date, 'start_time':
                    start_time, 'end_time': end_time}

    def _calculate_value_area(self, price_levels: Dict[float, OrderFlowLevel]
        ) ->Tuple[float, float]:
        """
        Calculate the value area (zone containing X% of volume).
        
        Args:
            price_levels: Dictionary of price levels with volume information
            
        Returns:
            Tuple of (value_area_low, value_area_high)
        """
        if not price_levels:
            return 0.0, 0.0
        poc_price = max(price_levels.keys(), key=lambda p: price_levels[p].
            total_volume)
        sorted_prices = sorted(price_levels.keys())
        total_volume = sum(level.total_volume for level in price_levels.
            values())
        target_volume = total_volume * self.value_area_volume
        current_volume = price_levels[poc_price].total_volume
        poc_idx = sorted_prices.index(poc_price)
        va_high_idx = poc_idx
        va_low_idx = poc_idx
        while current_volume < target_volume:
            high_vol = 0
            if va_high_idx < len(sorted_prices) - 1:
                high_price = sorted_prices[va_high_idx + 1]
                high_vol = price_levels[high_price].total_volume
            low_vol = 0
            if va_low_idx > 0:
                low_price = sorted_prices[va_low_idx - 1]
                low_vol = price_levels[low_price].total_volume
            if high_vol >= low_vol and va_high_idx < len(sorted_prices) - 1:
                va_high_idx += 1
                current_volume += high_vol
            elif va_low_idx > 0:
                va_low_idx -= 1
                current_volume += low_vol
            else:
                break
        return sorted_prices[va_low_idx], sorted_prices[va_high_idx]

    def visualize_profile(self, profile_id: Union[int, Tuple[datetime,
        datetime]], show_delta: bool=True) ->plt.Figure:
        """
        Create a visualization of a specific volume profile.
        
        Args:
            profile_id: Identifier for the profile to visualize
            show_delta: Whether to color bars based on delta (green/red)
            
        Returns:
            Matplotlib Figure with volume profile visualization
        """
        if profile_id not in self.profiles:
            raise ValueError(f'No profile with ID {profile_id}')
        profile = self.profiles[profile_id]
        price_levels = profile['price_levels']
        fig, ax = plt.subplots(figsize=(10, 12))
        sorted_prices = sorted(price_levels.keys())
        volumes = [price_levels[p].total_volume for p in sorted_prices]
        deltas = [price_levels[p].delta for p in sorted_prices]
        bars = ax.barh(sorted_prices, volumes, align='center', height=self.
            tick_size * 0.8)
        if show_delta:
            for i, bar in enumerate(bars):
                if deltas[i] > 0:
                    bar.set_color('green')
                elif deltas[i] < 0:
                    bar.set_color('red')
                else:
                    bar.set_color('gray')
        va_low = profile['value_area_low']
        va_high = profile['value_area_high']
        va_mask = [(p >= va_low and p <= va_high) for p in sorted_prices]
        ax.axhspan(va_low - self.tick_size / 2, va_high + self.tick_size / 
            2, color='blue', alpha=0.1)
        poc = profile['point_of_control']
        ax.axhline(y=poc, color='blue', linestyle='--', alpha=0.7)
        if isinstance(profile_id, tuple):
            title = (
                f'Volume Profile: {profile_id[0].date()} to {profile_id[1].date()}'
                )
        else:
            title = f'Volume Profile: Bar {profile_id}'
        ax.set_title(
            f"""{title}
POC: {poc:.{self.price_precision}f}, VAH: {va_high:.{self.price_precision}f}, VAL: {va_low:.{self.price_precision}f}"""
            )
        ax.set_xlabel('Volume')
        ax.set_ylabel('Price')
        ax.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        return fig
