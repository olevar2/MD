"""
Visualization Utilities for Gann Tools.

This module provides visualization utilities for Gann tools.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle, Polygon
from typing import Dict, Any, List, Optional, Tuple, Union
import math

def plot_gann_angles(data: pd.DataFrame, result: pd.DataFrame, angle_types: List[str]=['1x1', '1x2', '2x1'], title: str='Gann Angles', figsize: Tuple[int, int]=(12, 8), save_path: Optional[str]=None) -> plt.Figure:
    """
    Plot Gann angles on a price chart.
    
    Args:
        data: Original OHLCV data
        result: DataFrame with Gann angle calculations
        angle_types: List of angle types to plot
        title: Plot title
        figsize: Figure size
        save_path: Path to save the figure (if None, figure is not saved)
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(data.index, data['close'], color='black', alpha=0.5, label='Close Price')
    pivot_idx = result.loc[result['gann_angle_pivot_idx']].index[0]
    pivot_price = result.loc[pivot_idx, 'gann_angle_pivot_price']
    ax.scatter([pivot_idx], [pivot_price], color='red', s=100, zorder=5, label='Pivot Point')
    colors = {'1x8': '#1f77b4', '1x4': '#ff7f0e', '1x3': '#2ca02c', '1x2': '#d62728', '1x1': '#9467bd', '2x1': '#8c564b', '3x1': '#e377c2', '4x1': '#7f7f7f', '8x1': '#bcbd22'}
    for angle_type in angle_types:
        up_col = f'gann_angle_up_{angle_type}'
        down_col = f'gann_angle_down_{angle_type}'
        if up_col in result.columns:
            color = colors.get(angle_type, 'blue')
            ax.plot(result.index, result[up_col], color=color, linestyle='-', label=f'{angle_type} Up ({_angle_to_degrees(angle_type)}°)')
        if down_col in result.columns:
            color = colors.get(angle_type, 'blue')
            ax.plot(result.index, result[down_col], color=color, linestyle='--', label=f'{angle_type} Down ({_angle_to_degrees(angle_type)}°)')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.set_title(title)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=45)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig

def plot_gann_fan(data: pd.DataFrame, result: pd.DataFrame, fan_angles: List[str]=['1x1', '1x2', '2x1', '1x4', '4x1'], title: str='Gann Fan', figsize: Tuple[int, int]=(12, 8), save_path: Optional[str]=None) -> plt.Figure:
    """
    Plot Gann fan on a price chart.
    
    Args:
        data: Original OHLCV data
        result: DataFrame with Gann fan calculations
        fan_angles: List of fan angles to plot
        title: Plot title
        figsize: Figure size
        save_path: Path to save the figure (if None, figure is not saved)
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(data.index, data['close'], color='black', alpha=0.5, label='Close Price')
    pivot_idx = result.loc[result['gann_fan_pivot_idx']].index[0]
    pivot_price = result.loc[pivot_idx, 'gann_fan_pivot_price']
    ax.scatter([pivot_idx], [pivot_price], color='red', s=100, zorder=5, label='Pivot Point')
    colors = {'1x8': '#1f77b4', '1x4': '#ff7f0e', '1x3': '#2ca02c', '1x2': '#d62728', '1x1': '#9467bd', '2x1': '#8c564b', '3x1': '#e377c2', '4x1': '#7f7f7f', '8x1': '#bcbd22'}
    for angle_type in fan_angles:
        col = f'gann_fan_{angle_type}'
        if col in result.columns:
            color = colors.get(angle_type, 'blue')
            ax.plot(result.index, result[col], color=color, linestyle='-', label=f'{angle_type} ({_angle_to_degrees(angle_type)}°)')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.set_title(title)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=45)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig

def plot_gann_square(data: pd.DataFrame, result: pd.DataFrame, angles: List[int]=[45, 90, 135, 180], levels: List[int]=[1, 2], title: str='Gann Square', figsize: Tuple[int, int]=(12, 8), save_path: Optional[str]=None) -> plt.Figure:
    """
    Plot Gann square levels on a price chart.
    
    Args:
        data: Original OHLCV data
        result: DataFrame with Gann square calculations
        angles: List of angles to plot
        levels: List of levels to plot
        title: Plot title
        figsize: Figure size
        save_path: Path to save the figure (if None, figure is not saved)
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(data.index, data['close'], color='black', alpha=0.5, label='Close Price')
    colors = {45: '#1f77b4', 90: '#ff7f0e', 135: '#2ca02c', 180: '#d62728', 225: '#9467bd', 270: '#8c564b', 315: '#e377c2', 360: '#7f7f7f'}
    for level in levels:
        for angle in angles:
            sup_col = f'gann_sq_sup_{angle}_{level}'
            res_col = f'gann_sq_res_{angle}_{level}'
            if sup_col in result.columns:
                color = colors.get(angle, 'blue')
                ax.axhline(y=result[sup_col].iloc[-1], color=color, linestyle='--', alpha=0.7, label=f'Support {angle}° Level {level}')
            if res_col in result.columns:
                color = colors.get(angle, 'blue')
                ax.axhline(y=result[res_col].iloc[-1], color=color, linestyle='-', alpha=0.7, label=f'Resistance {angle}° Level {level}')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.set_title(title)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=45)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig

def plot_gann_time_cycles(data: pd.DataFrame, result: pd.DataFrame, cycle_lengths: List[int]=[30, 60, 90], max_cycles: int=2, title: str='Gann Time Cycles', figsize: Tuple[int, int]=(12, 8), save_path: Optional[str]=None) -> plt.Figure:
    """
    Plot Gann time cycles on a price chart.
    
    Args:
        data: Original OHLCV data
        result: DataFrame with Gann time cycle calculations
        cycle_lengths: List of cycle lengths to plot
        max_cycles: Maximum number of cycles to plot
        title: Plot title
        figsize: Figure size
        save_path: Path to save the figure (if None, figure is not saved)
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(data.index, data['close'], color='black', alpha=0.5, label='Close Price')
    start_idx = result.loc[result['gann_time_cycle_start']].index[0]
    ax.axvline(x=start_idx, color='red', linestyle='-', label='Starting Point')
    colors = {30: '#1f77b4', 60: '#ff7f0e', 90: '#2ca02c', 120: '#d62728', 180: '#9467bd', 270: '#8c564b', 360: '#e377c2'}
    for length in cycle_lengths:
        for i in range(1, max_cycles + 1):
            col = f'gann_time_cycle_{length}_{i}'
            if col in result.columns:
                cycle_points = result.loc[result[col]].index
                if len(cycle_points) > 0:
                    color = colors.get(length, 'blue')
                    for point in cycle_points:
                        ax.axvline(x=point, color=color, linestyle='--', alpha=0.7)
                        y_pos = ax.get_ylim()[0] + (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.05 * i
                        ax.annotate(f'{length} days', xy=(point, y_pos), xytext=(point, y_pos), ha='center', va='bottom', rotation=90, color=color)
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.set_title(title)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=45)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig

def plot_gann_box(data: pd.DataFrame, result: pd.DataFrame, title: str='Gann Box', figsize: Tuple[int, int]=(12, 8), save_path: Optional[str]=None) -> plt.Figure:
    """
    Plot Gann box on a price chart.
    
    Args:
        data: Original OHLCV data
        result: DataFrame with Gann box calculations
        title: Plot title
        figsize: Figure size
        save_path: Path to save the figure (if None, figure is not saved)
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(data.index, data['close'], color='black', alpha=0.5, label='Close Price')
    start_idx = result.loc[result['gann_box_start_idx']].index[0]
    end_idx = result.loc[result['gann_box_end_idx']].index[0]
    start_price = result.loc[start_idx, 'gann_box_start_price']
    end_price = result.loc[end_idx, 'gann_box_end_price']
    ax.scatter([start_idx], [start_price], color='green', s=100, zorder=5, label='Start Point')
    ax.scatter([end_idx], [end_price], color='red', s=100, zorder=5, label='End Point')
    start_pos = data.index.get_loc(start_idx)
    end_pos = data.index.get_loc(end_idx)
    ax.plot([start_idx, end_idx], [start_price, start_price], 'b-', alpha=0.7)
    ax.plot([start_idx, end_idx], [end_price, end_price], 'b-', alpha=0.7)
    ax.plot([start_idx, start_idx], [start_price, end_price], 'b-', alpha=0.7)
    ax.plot([end_idx, end_idx], [start_price, end_price], 'b-', alpha=0.7)
    price_division_cols = [col for col in result.columns if col.startswith('gann_box_price_')]
    for col in price_division_cols:
        price_level = result[col].iloc[0]
        ax.plot([start_idx, end_idx], [price_level, price_level], 'g--', alpha=0.5, label=f'Price Division {col.split('_')[-1]}%')
    time_division_cols = [col for col in result.columns if col.startswith('gann_box_time_')]
    for col in time_division_cols:
        time_points = result.loc[result[col]].index
        if len(time_points) > 0:
            for point in time_points:
                ax.plot([point, point], [start_price, end_price], 'r--', alpha=0.5)
                y_pos = start_price + (end_price - start_price) * 0.5
                ax.annotate(f'Time {col.split('_')[-1]}%', xy=(point, y_pos), xytext=(point, y_pos), ha='center', va='bottom', rotation=90, color='red')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.set_title(title)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=45)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig

def angle_to_degrees(angle_type: str) -> float:
    """
    Convert angle type (e.g., '1x1', '1x2') to degrees.
    
    Args:
        angle_type: Angle type
        
    Returns:
        Angle in degrees
    """
    if 'x' not in angle_type:
        return float(angle_type)
    parts = angle_type.split('x')
    price_units = float(parts[0])
    time_units = float(parts[1])
    return math.degrees(math.atan(price_units / time_units))

def create_example_plots(save_dir: str='.') -> None:
    """
    Create example plots for all Gann tools.
    
    Args:
        save_dir: Directory to save the plots
    """
    from feature_store_service.indicators.gann import GannAngles, GannFan, GannSquare, GannTimeCycles, GannBox
    data = pd.DataFrame({'open': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109], 'high': [102, 103, 104, 105, 106, 107, 108, 109, 110, 111], 'low': [99, 100, 101, 102, 103, 104, 105, 106, 107, 108], 'close': [101, 102, 103, 104, 105, 106, 107, 108, 109, 110], 'volume': [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900]}, index=pd.date_range(start='2023-01-01', periods=10))
    gann_angles = GannAngles(pivot_type='swing_low', angle_types=['1x1', '1x2', '2x1'], lookback_period=5, price_scaling=1.0, projection_bars=5)
    angles_result = gann_angles.calculate(data)
    gann_fan = GannFan(pivot_type='swing_low', fan_angles=['1x1', '1x2', '2x1'], lookback_period=5, price_scaling=1.0, projection_bars=5)
    fan_result = gann_fan.calculate(data)
    gann_square = GannSquare(square_type='square_of_9', pivot_price=100.0, auto_detect_pivot=False, lookback_period=5, num_levels=2)
    square_result = gann_square.calculate(data)
    gann_time_cycles = GannTimeCycles(cycle_lengths=[2, 3, 4], starting_point_type='major_low', lookback_period=5, auto_detect_start=True, max_cycles=2)
    time_cycles_result = gann_time_cycles.calculate(data)
    gann_box = GannBox(start_pivot_type='major_low', end_pivot_type='major_high', lookback_period=5, price_divisions=[0.5], time_divisions=[0.5])
    box_result = gann_box.calculate(data)
    plot_gann_angles(data, angles_result, save_path=f'{save_dir}/gann_angles_example.png')
    plot_gann_fan(data, fan_result, save_path=f'{save_dir}/gann_fan_example.png')
    plot_gann_square(data, square_result, save_path=f'{save_dir}/gann_square_example.png')
    plot_gann_time_cycles(data, time_cycles_result, cycle_lengths=[2, 3, 4], save_path=f'{save_dir}/gann_time_cycles_example.png')
    plot_gann_box(data, box_result, save_path=f'{save_dir}/gann_box_example.png')