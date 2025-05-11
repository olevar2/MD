"""
Time-Price Indicators Module

This module provides implementations for Time-Price-Opportunity (TPO) and Market Profile indicators:
- TimeProfileIndicator: Base class for time-based profiling
- MarketProfileIndicator: Implementation of Market Profile analysis
- TPOIndicator: Implementation of Time-Price-Opportunity analysis

These indicators analyze price distribution across time periods, creating a
visual representation of where price spent the most time, helping to identify
value areas, points of control, and price acceptance/rejection zones.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from collections import defaultdict

from analysis_engine.analysis.advanced_ta.base import AdvancedAnalysisBase
from analysis_engine.utils.validation import validate_dataframe

logger = logging.getLogger(__name__)


@dataclass
class PriceLevel:
    """Represents a single price level in a market profile."""
    price: float
    count: int = 0
    time_periods: List[str] = field(default_factory=list)
    letters: List[str] = field(default_factory=list)


@dataclass
class ValueArea:
    """Represents a value area in a market profile."""
    high: float
    low: float
    volume_percent: float
    point_of_control: float


class TimeProfileIndicator(AdvancedAnalysisBase):
    """
    Base class for time-based price profiling indicators.

    Time-based price profiling analyzes price distribution across different time periods,
    creating a visual and statistical representation of price acceptance and rejection.
    """

    def __init__(self, name: str = "TimeProfileIndicator", period: str = "day",
                 price_precision: int = 5, value_area_volume: float = 0.7):
        """
        Initialize the TimeProfileIndicator.

        Args:
            name: Name of the indicator
            period: Time period for analysis ('day', 'week', 'month', etc.)
            price_precision: Number of decimal places for price rounding
            value_area_volume: Percentage of volume to include in the value area (default: 0.7 for 70%)
        """
        super().__init__(name=name)
        self.period = period
        self.price_precision = price_precision
        self.value_area_volume = value_area_volume

        # Create a letter mapping for time periods (for TPO)
        self.letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Base method to calculate the time profile.

        Args:
            data: DataFrame with OHLCV data and datetime index

        Returns:
            DataFrame with time profile analysis results
        """
        # This is a base implementation that will be overridden by subclasses
        return data.copy()

    def _round_price(self, price: float) -> float:
        """
        Round the price to the specified precision.

        Args:
            price: Price value to round

        Returns:
            Rounded price
        """
        return round(price, self.price_precision)

    def _create_price_levels(self, data: pd.DataFrame) -> Dict[float, PriceLevel]:
        """
        Create price levels from data.

        Args:
            data: DataFrame with OHLCV data

        Returns:
            Dictionary mapping price levels to PriceLevel objects
        """
        # This is a placeholder method to be implemented by subclasses
        return {}

    def _calculate_value_area(self, price_levels: Dict[float, PriceLevel]) -> ValueArea:
        """
        Calculate the value area from price levels.

        Args:
            price_levels: Dictionary mapping price levels to PriceLevel objects

        Returns:
            ValueArea object with high, low, and point of control
        """
        if not price_levels:
            return ValueArea(0.0, 0.0, 0.0, 0.0)

        # Find the point of control (price level with most hits)
        poc_price = max(price_levels.keys(), key=lambda p: price_levels[p].count)
        poc_count = price_levels[poc_price].count

        # Sort price levels by price
        sorted_prices = sorted(price_levels.keys())

        # Calculate total count
        total_count = sum(price_levels[p].count for p in sorted_prices)

        # Target count for value area (e.g., 70% of total)
        target_count = total_count * self.value_area_volume

        # Start from POC and expand outward until we reach the target count
        current_count = poc_count
        va_high_idx = sorted_prices.index(poc_price)
        va_low_idx = va_high_idx

        while current_count < target_count and (va_high_idx < len(sorted_prices) - 1 or va_low_idx > 0):
            # Check which direction to expand
            high_count = 0
            if va_high_idx < len(sorted_prices) - 1:
                high_count = price_levels[sorted_prices[va_high_idx + 1]].count

            low_count = 0
            if va_low_idx > 0:
                low_count = price_levels[sorted_prices[va_low_idx - 1]].count

            # Expand in direction with higher count
            if high_count >= low_count and va_high_idx < len(sorted_prices) - 1:
                va_high_idx += 1
                current_count += high_count
            elif va_low_idx > 0:
                va_low_idx -= 1
                current_count += low_count
            else:
                break

        va_high = sorted_prices[va_high_idx]
        va_low = sorted_prices[va_low_idx]

        return ValueArea(
            high=va_high,
            low=va_low,
            volume_percent=current_count / total_count if total_count else 0,
            point_of_control=poc_price
        )


class MarketProfileIndicator(TimeProfileIndicator):
    """
    Implementation of Market Profile analysis.

    Market Profile is a form of data visualization that organizes price and time information
    to show price distribution by time period, revealing where price spent the most time.
    """

    def __init__(self, period: str = "day", price_precision: int = 5,
                 value_area_volume: float = 0.7, tick_size: float = 0.0001,
                 session_open: str = "00:00", session_close: str = "23:59"):
        """
        Initialize the MarketProfileIndicator.

        Args:
            period: Time period for analysis ('day', 'week', 'month', etc.)
            price_precision: Number of decimal places for price rounding
            value_area_volume: Percentage of volume to include in the value area (default: 0.7 for 70%)
            tick_size: The minimum price movement for the instrument
            session_open: Session open time (HH:MM)
            session_close: Session close time (HH:MM)
        """
        super().__init__(name="MarketProfileIndicator", period=period,
                         price_precision=price_precision,
                         value_area_volume=value_area_volume)
        self.tick_size = tick_size
        self.session_open = session_open
        self.session_close = session_close

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Market Profile analysis.

        Args:
            data: DataFrame with OHLCV data and datetime index

        Returns:
            DataFrame with Market Profile analysis results
        """
        validate_dataframe(data, required_columns=['open', 'high', 'low', 'close', 'volume'])

        # Ensure data has a datetime index
        if not isinstance(data.index, pd.DatetimeIndex):
            if 'datetime' in data.columns:
                data = data.set_index('datetime')
            else:
                raise ValueError("Data must have a datetime index or a 'datetime' column")

        result = data.copy()

        # Group data by the specified period
        if self.period == 'day':
            grouped = data.groupby(data.index.date)
        elif self.period == 'week':
            grouped = data.groupby(data.index.isocalendar().week)
        elif self.period == 'month':
            grouped = data.groupby([data.index.year, data.index.month])
        else:
            raise ValueError(f"Unsupported period: {self.period}")

        # Process each period
        profiles = {}
        for period_key, period_data in grouped:
            # Convert period_key to string for consistent key format
            if isinstance(period_key, tuple):
                period_str = f"{period_key[0]}-{period_key[1]}"
            else:
                period_str = str(period_key)

            # Create price levels for this period
            price_levels = self._create_price_levels_for_period(period_data)

            # Calculate value area for this period
            value_area = self._calculate_value_area(price_levels)

            # Store profile for this period
            profiles[period_str] = {
                'price_levels': price_levels,
                'value_area': value_area,
                'period_start': period_data.index.min(),
                'period_end': period_data.index.max(),
                'total_volume': period_data['volume'].sum()
            }

        # Add Market Profile information to result DataFrame
        result['mp_value_area_high'] = np.nan
        result['mp_value_area_low'] = np.nan
        result['mp_point_of_control'] = np.nan

        for period_str, profile in profiles.items():
            va = profile['value_area']
            period_mask = result.index.isin(period_data.index)

            result.loc[period_mask, 'mp_value_area_high'] = va.high
            result.loc[period_mask, 'mp_value_area_low'] = va.low
            result.loc[period_mask, 'mp_point_of_control'] = va.point_of_control

        # Store profiles in instance for later use (visualization, etc.)
        self.profiles = profiles

        return result

    def _create_price_levels_for_period(self, data: pd.DataFrame) -> Dict[float, PriceLevel]:
        """
        Create price levels for a specific period.

        Args:
            data: DataFrame with OHLCV data for a specific period

        Returns:
            Dictionary mapping price levels to PriceLevel objects
        """
        price_levels = {}

        # Create price buckets based on tick size
        for _, row in data.iterrows():
            # Create a range of prices from low to high with tick_size intervals
            current_price = self._round_price(row['low'])
            high_price = self._round_price(row['high'])

            while current_price <= high_price:
                # Create or update price level
                if current_price not in price_levels:
                    price_levels[current_price] = PriceLevel(price=current_price)

                # Increment count and record time period
                price_levels[current_price].count += 1
                time_str = row.name.strftime('%H:%M')
                if time_str not in price_levels[current_price].time_periods:
                    price_levels[current_price].time_periods.append(time_str)

                # Move to next price level
                current_price = self._round_price(current_price + self.tick_size)

        return price_levels

    def visualize_profile(self, period_key: str) -> plt.Figure:
        """
        Create a visualization of the Market Profile for a specific period.

        Args:
            period_key: Key for the period to visualize

        Returns:
            Matplotlib Figure with Market Profile visualization
        """
        if not hasattr(self, 'profiles') or period_key not in self.profiles:
            raise ValueError(f"No profile data for period: {period_key}")

        profile = self.profiles[period_key]
        price_levels = profile['price_levels']
        value_area = profile['value_area']

        # Create figure
        fig, ax = plt.subplots(figsize=(10, 12))

        # Sort prices for visualization
        sorted_prices = sorted(price_levels.keys())

        # Create horizontal bars for each price level
        y_pos = np.arange(len(sorted_prices))
        counts = [price_levels[p].count for p in sorted_prices]

        # Plot horizontal bars
        bars = ax.barh(y_pos, counts, align='center')

        # Highlight the value area
        va_prices = [p for p in sorted_prices if value_area.low <= p <= value_area.high]
        va_indices = [sorted_prices.index(p) for p in va_prices]
        for i, bar in enumerate(bars):
            if i in va_indices:
                bar.set_color('blue')
            # Highlight the point of control
            if sorted_prices[i] == value_area.point_of_control:
                bar.set_color('red')

        # Set y-axis labels as price levels
        ax.set_yticks(y_pos)
        ax.set_yticklabels([f"{p:.{self.price_precision}f}" for p in sorted_prices])

        # Set labels and title
        ax.set_xlabel('Count')
        ax.set_ylabel('Price')
        ax.set_title(f'Market Profile for {period_key}\nVAH: {value_area.high:.{self.price_precision}f}, '
                     f'POC: {value_area.point_of_control:.{self.price_precision}f}, '
                     f'VAL: {value_area.low:.{self.price_precision}f}')

        # Add grid
        ax.grid(True, linestyle='--', alpha=0.7)

        # Add legend
        ax.legend(['Regular', 'Value Area', 'Point of Control'])

        # Adjust layout
        plt.tight_layout()

        return fig


class TPOIndicator(TimeProfileIndicator):
    """
    Implementation of Time-Price-Opportunity (TPO) analysis.

    TPO is a charting technique that organizes price and time data to show
    where price spent the most time, using letters to represent time periods.
    """

    def __init__(self, period: str = "day", price_precision: int = 5,
                 value_area_volume: float = 0.7, tick_size: float = 0.0001,
                 time_interval: int = 30):
        """
        Initialize the TPOIndicator.

        Args:
            period: Time period for analysis ('day', 'week', 'month', etc.)
            price_precision: Number of decimal places for price rounding
            value_area_volume: Percentage of volume to include in the value area (default: 0.7 for 70%)
            tick_size: The minimum price movement for the instrument
            time_interval: Time interval in minutes for each TPO letter/period
        """
        super().__init__(name="TPOIndicator", period=period,
                         price_precision=price_precision,
                         value_area_volume=value_area_volume)
        self.tick_size = tick_size
        self.time_interval = time_interval

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate TPO analysis.

        Args:
            data: DataFrame with OHLCV data and datetime index

        Returns:
            DataFrame with TPO analysis results
        """
        validate_dataframe(data, required_columns=['open', 'high', 'low', 'close'])

        # Ensure data has a datetime index
        if not isinstance(data.index, pd.DatetimeIndex):
            if 'datetime' in data.columns:
                data = data.set_index('datetime')
            else:
                raise ValueError("Data must have a datetime index or a 'datetime' column")

        result = data.copy()

        # Group data by the specified period
        if self.period == 'day':
            grouped = data.groupby(data.index.date)
        elif self.period == 'week':
            grouped = data.groupby(data.index.isocalendar().week)
        elif self.period == 'month':
            grouped = data.groupby([data.index.year, data.index.month])
        else:
            raise ValueError(f"Unsupported period: {self.period}")

        # Process each period
        profiles = {}
        for period_key, period_data in grouped:
            # Convert period_key to string for consistent key format
            if isinstance(period_key, tuple):
                period_str = f"{period_key[0]}-{period_key[1]}"
            else:
                period_str = str(period_key)

            # Create TPO profile for this period
            profile = self._create_tpo_profile(period_data)

            # Calculate value area for this period
            value_area = self._calculate_value_area(profile['price_levels'])

            # Store profile for this period
            profiles[period_str] = {
                'price_levels': profile['price_levels'],
                'value_area': value_area,
                'period_start': period_data.index.min(),
                'period_end': period_data.index.max(),
                'tpo_map': profile['tpo_map']
            }

        # Add TPO information to result DataFrame
        result['tpo_value_area_high'] = np.nan
        result['tpo_value_area_low'] = np.nan
        result['tpo_point_of_control'] = np.nan

        for period_str, profile in profiles.items():
            va = profile['value_area']
            # Create mask for this period's data
            if isinstance(period_key, tuple):  # For year-month
                year, month = period_key
                period_mask = (result.index.year == year) & (result.index.month == month)
            else:  # For date or week
                if self.period == 'day':
                    period_mask = result.index.date == period_key
                elif self.period == 'week':
                    period_mask = result.index.isocalendar().week == period_key

            result.loc[period_mask, 'tpo_value_area_high'] = va.high
            result.loc[period_mask, 'tpo_value_area_low'] = va.low
            result.loc[period_mask, 'tpo_point_of_control'] = va.point_of_control

        # Store profiles in instance for later use (visualization, etc.)
        self.profiles = profiles

        return result

    def _create_tpo_profile(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Create TPO profile for a specific period.

        Args:
            data: DataFrame with OHLCV data for a specific period

        Returns:
            Dictionary with price levels and TPO map
        """
        price_levels = {}
        tpo_map = {}

        # Sort data by datetime
        data = data.sort_index()

        # Find min/max price for the period
        min_price = data['low'].min()
        max_price = data['high'].max()

        # Create price range with tick size
        price_range = []
        current_price = self._round_price(min_price)
        while current_price <= max_price:
            price_range.append(current_price)
            price_levels[current_price] = PriceLevel(price=current_price)
            current_price = self._round_price(current_price + self.tick_size)

        # Group data into time intervals and assign letters
        start_time = data.index[0].replace(hour=0, minute=0, second=0, microsecond=0)
        end_time = data.index[-1].replace(hour=23, minute=59, second=59, microsecond=999999)

        current_time = start_time
        letter_idx = 0

        while current_time <= end_time and letter_idx < len(self.letters):
            letter = self.letters[letter_idx]
            interval_end = current_time + timedelta(minutes=self.time_interval)

            # Get data for this interval
            interval_data = data[(data.index >= current_time) & (data.index < interval_end)]

            if not interval_data.empty:
                # Record prices visited during this interval
                for _, row in interval_data.iterrows():
                    current_price = self._round_price(row['low'])
                    high_price = self._round_price(row['high'])

                    while current_price <= high_price:
                        if current_price in price_levels:
                            # Add letter to this price level
                            if letter not in price_levels[current_price].letters:
                                price_levels[current_price].letters.append(letter)
                                price_levels[current_price].count += 1

                            # Update TPO map
                            if letter not in tpo_map:
                                tpo_map[letter] = []
                            if current_price not in tpo_map[letter]:
                                tpo_map[letter].append(current_price)

                        # Move to next price level
                        current_price = self._round_price(current_price + self.tick_size)

            # Move to next time interval
            current_time = interval_end
            letter_idx += 1

        return {
            'price_levels': price_levels,
            'tpo_map': tpo_map
        }

    def visualize_tpo(self, period_key: str) -> plt.Figure:
        """
        Create a visualization of the TPO profile for a specific period.

        Args:
            period_key: Key for the period to visualize

        Returns:
            Matplotlib Figure with TPO visualization
        """
        if not hasattr(self, 'profiles') or period_key not in self.profiles:
            raise ValueError(f"No profile data for period: {period_key}")

        profile = self.profiles[period_key]
        price_levels = profile['price_levels']
        value_area = profile['value_area']
        tpo_map = profile['tpo_map']

        # Create figure
        fig, ax = plt.subplots(figsize=(12, 15))

        # Sort prices for visualization
        sorted_prices = sorted(price_levels.keys())

        # Create horizontal plot with letters
        y_pos = np.arange(len(sorted_prices))

        # For each letter/time period, plot its price points
        x_offset = 0
        for letter, prices in sorted(tpo_map.items()):
            for price in prices:
                price_idx = sorted_prices.index(price)
                text = ax.text(x_offset, price_idx, letter, ha='center', va='center',
                              fontsize=8, fontweight='bold')

                # Highlight value area
                if value_area.low <= price <= value_area.high:
                    text.set_backgroundcolor('lightblue')

                # Highlight point of control
                if price == value_area.point_of_control:
                    text.set_backgroundcolor('salmon')

            x_offset += 1

        # Set y-axis labels as price levels
        ax.set_yticks(y_pos)
        ax.set_yticklabels([f"{p:.{self.price_precision}f}" for p in sorted_prices])

        # Set x-axis to show time periods
        ax.set_xticks(np.arange(len(tpo_map)))
        ax.set_xticklabels(sorted(tpo_map.keys()))

        # Set labels and title
        ax.set_ylabel('Price')
        ax.set_xlabel('Time Period')
        ax.set_title(f'TPO Profile for {period_key}\nVAH: {value_area.high:.{self.price_precision}f}, '
                     f'POC: {value_area.point_of_control:.{self.price_precision}f}, '
                     f'VAL: {value_area.low:.{self.price_precision}f}')

        # Add grid
        ax.grid(True, linestyle='--', alpha=0.7)

        # Adjust layout
        plt.tight_layout()

        return fig
