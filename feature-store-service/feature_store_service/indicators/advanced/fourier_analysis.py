"""
Fourier Analysis Indicator.

This module implements spectral analysis using Fourier transforms to
identify dominant cycles in price data, with period estimation,
cycle strength measurement, and forward prediction capabilities.
"""

from typing import Dict, Any, List, Optional, Tuple, Union
import pandas as pd
import numpy as np
from enum import Enum
import scipy.fftpack
from scipy import signal
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

from feature_store_service.indicators.base_indicator import BaseIndicator


class FourierAnalysisIndicator(BaseIndicator):
    """
    Fourier Analysis Indicator.
    
    This indicator implements spectral analysis using Fourier transforms to
    identify dominant cycles in price data, with period estimation,
    cycle strength measurement, and forward prediction capabilities.
    """
    
    category = "cyclical"
    
    def __init__(
        self, 
        window: int = 252,
        min_periods: int = None,
        max_cycles: int = 5,
        min_cycle_length: int = 5,
        max_cycle_length: Optional[int] = None,
        detrend_method: str = "linear",
        column: str = "close",
        **kwargs
    ):
        """
        Initialize Fourier Analysis indicator.
        
        Args:
            window: Rolling window for Fourier analysis
            min_periods: Minimum periods for calculation (default: window//2)
            max_cycles: Maximum number of dominant cycles to track
            min_cycle_length: Minimum length of cycles to consider (in bars)
            max_cycle_length: Maximum length of cycles to consider (in bars)
            detrend_method: Method for detrending ('linear', 'polynomial', or None)
            column: Data column to use for calculations (default: 'close')
            **kwargs: Additional parameters
        """
        self.window = window
        self.min_periods = min_periods if min_periods is not None else window // 2
        self.max_cycles = max_cycles
        self.min_cycle_length = min_cycle_length
        self.max_cycle_length = max_cycle_length
        self.detrend_method = detrend_method
        self.column = column
        self.name = "fourier_analysis"
        
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Fourier analysis on the given data.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with Fourier analysis results
        """
        if self.column not in data.columns:
            raise ValueError(f"Data must contain '{self.column}' column")
            
        # Make a copy to avoid modifying the input data
        result = data.copy()
        
        # Apply rolling window FFT analysis
        self._apply_rolling_fft(result)
        
        # Calculate combined cycle components
        self._calculate_combined_cycle(result)
        
        return result
    
    def _detrend(self, series: np.ndarray) -> np.ndarray:
        """
        Remove trend from time series data.
        
        Args:
            series: Numpy array with time series data
            
        Returns:
            Detrended time series data
        """
        if self.detrend_method is None:
            return series
            
        x = np.arange(len(series))
        
        if self.detrend_method == "linear":
            # Linear detrending
            mask = ~np.isnan(series)
            if np.sum(mask) <= 1:  # Not enough data points
                return np.zeros_like(series)
                
            # Perform linear regression only on non-nan values
            coeffs = np.polyfit(x[mask], series[mask], 1)
            trend = np.polyval(coeffs, x)
            return series - trend
            
        elif self.detrend_method == "polynomial":
            # Polynomial detrending (degree 2)
            mask = ~np.isnan(series)
            if np.sum(mask) <= 2:  # Not enough data points
                return np.zeros_like(series)
                
            # Perform polynomial fit only on non-nan values
            coeffs = np.polyfit(x[mask], series[mask], 2)
            trend = np.polyval(coeffs, x)
            return series - trend
            
        else:
            # Use signal detrend (removes mean or linear trend)
            mask = ~np.isnan(series)
            if np.sum(mask) <= 1:  # Not enough data points
                return np.zeros_like(series)
                
            # Create output array with NaNs preserved
            output = np.full_like(series, np.nan)
            # Detrend only non-nan values
            output[mask] = signal.detrend(series[mask])
            return output
    
    def _apply_fft(self, series: np.ndarray) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
        """
        Apply Fast Fourier Transform to time series data.
        
        Args:
            series: Time series data
            
        Returns:
            Tuple containing:
                - Frequencies array
                - Power spectrum array
                - List of dominant cycle information
        """
        n = len(series)
        if n < self.min_periods:
            # Not enough data
            empty_freq = np.array([])
            empty_power = np.array([])
            return empty_freq, empty_power, []
            
        # Apply Hamming window to reduce spectral leakage
        hamming_window = np.hamming(n)
        windowed_series = series * hamming_window
        
        # Compute FFT
        fft_output = scipy.fftpack.fft(windowed_series)
        
        # Get amplitudes (power)
        power = np.abs(fft_output) ** 2
        
        # Get frequencies (in cycles per sample)
        freqs = scipy.fftpack.fftfreq(n)
        
        # Only keep first half (positive frequencies)
        positive_freq_idx = freqs > 0
        freqs = freqs[positive_freq_idx]
        power = power[positive_freq_idx]
        
        # Convert frequencies to periods (in bars)
        periods = 1.0 / freqs
        
        # Filter out periods outside our desired range
        valid_periods_mask = periods >= self.min_cycle_length
        if self.max_cycle_length is not None:
            valid_periods_mask &= periods <= self.max_cycle_length
        
        periods = periods[valid_periods_mask]
        power_filtered = power[valid_periods_mask]
        
        if len(periods) == 0:
            # No valid periods in range
            return np.array([]), np.array([]), []
        
        # Sort by power
        sort_idx = np.argsort(power_filtered)[::-1]  # Descending
        periods_sorted = periods[sort_idx]
        power_sorted = power_filtered[sort_idx]
        
        # Get dominant cycles
        max_cycles = min(self.max_cycles, len(periods_sorted))
        dominant_cycles = []
        
        # Calculate total power for normalization
        total_power = np.sum(power)
        
        for i in range(max_cycles):
            period = periods_sorted[i]
            power_val = power_sorted[i]
            
            # Calculate relative power (as percentage of total)
            relative_power = (power_val / total_power) * 100 if total_power > 0 else 0
            
            # Calculate frequency (cycles per bar)
            frequency = 1.0 / period
            
            # Add to dominant cycles list
            dominant_cycles.append({
                'period': period,
                'frequency': frequency,
                'power': power_val,
                'relative_power': relative_power,
                'rank': i + 1
            })
        
        return freqs, power, dominant_cycles
        
    def _apply_rolling_fft(self, data: pd.DataFrame) -> None:
        """
        Apply rolling window FFT analysis.
        
        Args:
            data: DataFrame to analyze and update with results
        """
        # Pre-allocate arrays for faster computation
        n = len(data)
        
        # Create matrices for dominant cycle periods and powers
        for i in range(1, self.max_cycles + 1):
            data[f'cycle_{i}_period'] = np.nan
            data[f'cycle_{i}_power'] = np.nan
        
        # Create arrays for combined signals
        data['combined_cycle_signal'] = np.nan
        
        # Apply rolling window FFT
        for i in range(self.window, n + 1):
            window_data = data.iloc[i - self.window:i][self.column].values
            
            # Skip windows with too many NaN values
            if np.isnan(window_data).sum() > (self.window - self.min_periods):
                continue
                
            # Fill NaN values (if any) using interpolation
            if np.isnan(window_data).any():
                # Get indices of non-NaN values
                valid_indices = np.where(~np.isnan(window_data))[0]
                
                # Get corresponding values
                valid_values = window_data[valid_indices]
                
                # Interpolate at all indices
                all_indices = np.arange(len(window_data))
                
                # Use linear interpolation
                window_data = np.interp(
                    all_indices,
                    valid_indices,
                    valid_values
                )
            
            # Detrend the data
            detrended_data = self._detrend(window_data)
            
            # Apply FFT and get dominant cycles
            _, _, dominant_cycles = self._apply_fft(detrended_data)
            
            # Store dominant cycle information
            for j, cycle in enumerate(dominant_cycles):
                if j < self.max_cycles:  # Only store up to max_cycles
                    cycle_idx = j + 1
                    data.iloc[i-1, data.columns.get_loc(f'cycle_{cycle_idx}_period')] = cycle['period']
                    data.iloc[i-1, data.columns.get_loc(f'cycle_{cycle_idx}_power')] = cycle['relative_power']
        
        # Forward fill to have continuous values (with limit to avoid filling too much)
        fill_limit = min(self.window // 2, 20)
        for i in range(1, self.max_cycles + 1):
            data[f'cycle_{i}_period'] = data[f'cycle_{i}_period'].fillna(method='ffill', limit=fill_limit)
            data[f'cycle_{i}_power'] = data[f'cycle_{i}_power'].fillna(method='ffill', limit=fill_limit)
    
    def _calculate_combined_cycle(self, data: pd.DataFrame) -> None:
        """
        Calculate combined cycle component based on dominant cycles.
        
        Args:
            data: DataFrame to update with combined cycle
        """
        # Get price series
        price = data[self.column].values
        n = len(price)
        
        # Create time array (normalize to 0-1 range)
        t = np.arange(n) / n
        
        # Initialize combined cycle array
        combined_cycle = np.zeros(n)
        
        # For each point, reconstruct signal based on recent dominant cycles
        for i in range(self.window, n):
            # Get the dominant cycles at this point
            cycles = []
            for j in range(1, self.max_cycles + 1):
                period = data.iloc[i].get(f'cycle_{j}_period')
                power = data.iloc[i].get(f'cycle_{j}_power')
                
                if not (np.isnan(period) or np.isnan(power)):
                    cycles.append((period, power))
            
            # Skip if no valid cycles
            if not cycles:
                continue
            
            # Calculate phase and amplitude for reconstruction
            signal = np.zeros(self.window)
            
            for period, power in cycles:
                if period <= 1:  # Skip invalid periods
                    continue
                    
                # Convert relative power to amplitude (square root of power)
                amplitude = np.sqrt(power / 100)
                
                # Generate cycle component (sine wave)
                freq = 1.0 / period
                phase_array = 2.0 * np.pi * freq * np.arange(self.window)
                signal += amplitude * np.sin(phase_array)
            
            # Normalize signal to percentage of price
            if len(signal) > 0:
                price_window = price[i - self.window:i]
                if len(price_window) > 0 and not np.all(np.isnan(price_window)):
                    price_mean = np.nanmean(price_window)
                    if price_mean != 0:
                        signal = (signal / price_mean) * 100
            
            # Store last value as current cycle signal
            data.iloc[i, data.columns.get_loc('combined_cycle_signal')] = signal[-1]
        
        # Calculate cycle momentum (rate of change of cycle signal)
        data['cycle_momentum'] = data['combined_cycle_signal'].diff(5)
        
        # Calculate cycle direction (sign of cycle signal)
        data['cycle_direction'] = np.sign(data['combined_cycle_signal'])
        
        # Calculate cycle turning points
        data['cycle_turning_point'] = np.where(
            data['cycle_direction'].diff() != 0,
            data['cycle_direction'],
            0
        )
    
    def get_cycle_forecast(self, data: pd.DataFrame, forecast_periods: int = 20) -> pd.DataFrame:
        """
        Generate cycle forecast based on identified dominant cycles.
        
        Args:
            data: DataFrame with calculated cycles
            forecast_periods: Number of periods to forecast
            
        Returns:
            DataFrame with forecasted cycles
        """
        # Create forecast dataframe
        forecast = pd.DataFrame(index=range(forecast_periods))
        
        # Get the last data point
        last_idx = len(data) - 1
        
        # Extract dominant cycles from the last data point
        cycles = []
        for i in range(1, self.max_cycles + 1):
            period_col = f'cycle_{i}_period'
            power_col = f'cycle_{i}_power'
            
            if period_col in data.columns and power_col in data.columns:
                period = data.iloc[last_idx].get(period_col)
                power = data.iloc[last_idx].get(power_col)
                
                if not (np.isnan(period) or np.isnan(power)):
                    cycles.append((period, power))
        
        # If no valid cycles, return empty forecast
        if not cycles:
            forecast['cycle_forecast'] = np.nan
            return forecast
        
        # Get the last window of actual data
        last_window = min(self.window, len(data))
        actual_data = data.iloc[-last_window:][self.column].values
        
        # Calculate the average price for normalization
        price_mean = np.nanmean(actual_data)
        
        # Determine the initial phase for each cycle to match recent data
        initial_phases = []
        
        for period, power in cycles:
            if period <= 1:  # Skip invalid periods
                continue
                
            # Find phase that best matches recent behavior
            best_phase = 0
            best_error = float('inf')
            
            for phase in np.linspace(0, 2*np.pi, 20):
                # Generate cycle with this phase
                freq = 1.0 / period
                amplitude = np.sqrt(power / 100)
                t = np.arange(last_window)
                cycle = amplitude * np.sin(2.0 * np.pi * freq * t + phase)
                
                # Calculate error (only on the last few points)
                last_n = min(int(period * 2), last_window)
                if last_n > 0:
                    cycle_signal = data.iloc[-last_n:]['combined_cycle_signal'].values
                    if not np.all(np.isnan(cycle_signal)):
                        error = np.nanmean((cycle[-last_n:] - cycle_signal) ** 2)
                        
                        if error < best_error:
                            best_error = error
                            best_phase = phase
                            
            initial_phases.append((period, power, best_phase))
        
        # Generate forecast
        forecast_signal = np.zeros(forecast_periods)
        time_points = np.arange(forecast_periods)
        
        for period, power, phase in initial_phases:
            # Generate cycle component (sine wave)
            freq = 1.0 / period
            amplitude = np.sqrt(power / 100)
            phase_array = 2.0 * np.pi * freq * time_points + phase
            forecast_signal += amplitude * np.sin(phase_array)
        
        # Store forecast
        forecast['cycle_forecast'] = forecast_signal
        
        # Calculate forecast in price units
        if price_mean != 0:
            forecast['price_forecast'] = price_mean * (1 + forecast_signal / 100)
        
        # Calculate cycle direction
        forecast['cycle_direction'] = np.sign(forecast_signal)
        
        # Calculate cycle turning points
        forecast['cycle_turning_point'] = np.where(
            forecast['cycle_direction'].diff() != 0,
            forecast['cycle_direction'],
            0
        )
        
        return forecast
    
    def get_cycle_visualization_data(self, result: pd.DataFrame) -> Dict[str, Any]:
        """
        Get data for visualizing cycle analysis.
        
        Args:
            result: DataFrame with calculated cycles
            
        Returns:
            Dictionary with visualization data
        """
        # Extract dominant cycle information
        cycles_data = []
        
        # Get the last row (most recent cycles)
        last_row = result.iloc[-1]
        
        for i in range(1, self.max_cycles + 1):
            period_col = f'cycle_{i}_period'
            power_col = f'cycle_{i}_power'
            
            if period_col in result.columns and power_col in result.columns:
                period = last_row.get(period_col)
                power = last_row.get(power_col)
                
                if not (np.isnan(period) or np.isnan(power)):
                    cycles_data.append({
                        'rank': i,
                        'period': float(period),
                        'power': float(power)
                    })
        
        # Prepare data for periodogram visualization
        periodogram = {}
        
        # Create sample periodogram data from the last window
        if len(result) >= self.window:
            window_data = result.iloc[-self.window:][self.column].values
            
            # Apply FFT if we have enough data
            if np.isnan(window_data).sum() <= (self.window - self.min_periods):
                # Fill NaN values if needed
                if np.isnan(window_data).any():
                    valid_indices = np.where(~np.isnan(window_data))[0]
                    valid_values = window_data[valid_indices]
                    all_indices = np.arange(len(window_data))
                    window_data = np.interp(all_indices, valid_indices, valid_values)
                
                # Detrend the data
                detrended_data = self._detrend(window_data)
                
                # Apply FFT
                freqs, power, _ = self._apply_fft(detrended_data)
                
                # Convert frequencies to periods
                periods = 1.0 / freqs if len(freqs) > 0 else []
                
                # Sort by period
                if len(periods) > 0:
                    sort_idx = np.argsort(periods)
                    periods = periods[sort_idx]
                    power = power[sort_idx]
                    
                    # Normalize power
                    if len(power) > 0 and np.max(power) > 0:
                        power = power / np.max(power) * 100
                    
                    periodogram = {
                        'periods': periods.tolist(),
                        'power': power.tolist()
                    }
        
        # Return visualization data
        return {
            'dominant_cycles': cycles_data,
            'periodogram': periodogram,
            'cycle_signal': result['combined_cycle_signal'].dropna().tolist(),
            'cycle_momentum': result['cycle_momentum'].dropna().tolist() if 'cycle_momentum' in result.columns else [],
            'cycle_turning_points': result.loc[result['cycle_turning_point'] != 0].index.tolist() if 'cycle_turning_point' in result.columns else []
        }
""""""
