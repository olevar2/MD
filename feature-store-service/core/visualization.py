"""
Chart Pattern Visualization Module.

This module provides utilities for visualizing detected chart patterns.
"""
from typing import Dict, Any, List, Optional, Tuple, Union
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.dates import date2num
import mplfinance as mpf

def plot_chart_with_patterns(data: pd.DataFrame, patterns: Dict[str, List[Dict[str, Any]]], title: str='Chart Patterns', figsize: Tuple[int, int]=(12, 8), save_path: Optional[str]=None) -> plt.Figure:
    """
    Plot a price chart with detected patterns highlighted.
    
    Args:
        data: DataFrame with OHLCV data
        patterns: Dictionary of pattern types and their occurrences
        title: Title of the chart
        figsize: Figure size (width, height) in inches
        save_path: Path to save the figure (if None, the figure is not saved)
        
    Returns:
        Matplotlib figure object
    """
    if not isinstance(data.index, pd.DatetimeIndex):
        data = data.copy()
        data.index = pd.date_range(start='2020-01-01', periods=len(data))
    fig, ax = plt.subplots(figsize=figsize)
    mpf.plot(data, type='candle', style='yahoo', ax=ax)
    _add_patterns_to_chart(ax, data, patterns)
    ax.set_title(title, fontsize=14)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Price', fontsize=12)
    _add_pattern_legend(ax, patterns)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig

def plot_harmonic_pattern(data: pd.DataFrame, pattern: Dict[str, Any], title: str='Harmonic Pattern', figsize: Tuple[int, int]=(10, 6), save_path: Optional[str]=None) -> plt.Figure:
    """
    Plot a harmonic pattern with its Fibonacci ratios.
    
    Args:
        data: DataFrame with OHLCV data
        pattern: Dictionary with pattern information
        title: Title of the chart
        figsize: Figure size (width, height) in inches
        save_path: Path to save the figure (if None, the figure is not saved)
        
    Returns:
        Matplotlib figure object
    """
    points = pattern.get('points', {})
    if not points or 'X' not in points or 'A' not in points or ('B' not in points) or ('C' not in points) or ('D' not in points):
        raise ValueError('Pattern must contain X, A, B, C, and D points')
    x_idx, x_price = (points['X']['idx'], points['X']['price'])
    a_idx, a_price = (points['A']['idx'], points['A']['price'])
    b_idx, b_price = (points['B']['idx'], points['B']['price'])
    c_idx, c_price = (points['C']['idx'], points['C']['price'])
    d_idx, d_price = (points['D']['idx'], points['D']['price'])
    start_idx = max(0, x_idx - 5)
    end_idx = min(len(data), d_idx + 5)
    pattern_data = data.iloc[start_idx:end_idx].copy()
    if not isinstance(pattern_data.index, pd.DatetimeIndex):
        pattern_data.index = pd.date_range(start='2020-01-01', periods=len(pattern_data))
    fig, ax = plt.subplots(figsize=figsize)
    mpf.plot(pattern_data, type='candle', style='yahoo', ax=ax)
    x_idx -= start_idx
    a_idx -= start_idx
    b_idx -= start_idx
    c_idx -= start_idx
    d_idx -= start_idx
    dates = date2num(pattern_data.index.to_pydatetime())
    x_date = dates[x_idx]
    a_date = dates[a_idx]
    b_date = dates[b_idx]
    c_date = dates[c_idx]
    d_date = dates[d_idx]
    ax.plot([x_date, a_date], [x_price, a_price], 'b-', linewidth=2, label='XA')
    ax.plot([a_date, b_date], [a_price, b_price], 'g-', linewidth=2, label='AB')
    ax.plot([b_date, c_date], [b_price, c_price], 'r-', linewidth=2, label='BC')
    ax.plot([c_date, d_date], [c_price, d_price], 'c-', linewidth=2, label='CD')
    ax.plot(x_date, x_price, 'bo', markersize=8, label='X')
    ax.plot(a_date, a_price, 'go', markersize=8, label='A')
    ax.plot(b_date, b_price, 'ro', markersize=8, label='B')
    ax.plot(c_date, c_price, 'co', markersize=8, label='C')
    ax.plot(d_date, d_price, 'mo', markersize=8, label='D')
    ax.annotate('X', (x_date, x_price), xytext=(5, 5), textcoords='offset points')
    ax.annotate('A', (a_date, a_price), xytext=(5, 5), textcoords='offset points')
    ax.annotate('B', (b_date, b_price), xytext=(5, 5), textcoords='offset points')
    ax.annotate('C', (c_date, c_price), xytext=(5, 5), textcoords='offset points')
    ax.annotate('D', (d_date, d_price), xytext=(5, 5), textcoords='offset points')
    _add_fibonacci_ratios(ax, pattern, x_date, x_price, a_date, a_price, b_date, b_price, c_date, c_price, d_date, d_price)
    pattern_type = pattern.get('pattern_type', 'Unknown')
    direction = pattern.get('direction', 'Unknown')
    strength = pattern.get('strength', 0.0)
    ax.set_title(f'{title}: {pattern_type.capitalize()} ({direction.capitalize()}, Strength: {strength:.2f})', fontsize=14)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Price', fontsize=12)
    ax.legend(loc='upper left')
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig

def plot_candlestick_pattern(data: pd.DataFrame, pattern: Dict[str, Any], lookback: int=5, lookahead: int=5, title: str='Candlestick Pattern', figsize: Tuple[int, int]=(10, 6), save_path: Optional[str]=None) -> plt.Figure:
    """
    Plot a candlestick pattern with annotations.
    
    Args:
        data: DataFrame with OHLCV data
        pattern: Dictionary with pattern information
        lookback: Number of candles to include before the pattern
        lookahead: Number of candles to include after the pattern
        title: Title of the chart
        figsize: Figure size (width, height) in inches
        save_path: Path to save the figure (if None, the figure is not saved)
        
    Returns:
        Matplotlib figure object
    """
    pattern_idx = pattern.get('index', 0)
    start_idx = max(0, pattern_idx - lookback)
    end_idx = min(len(data), pattern_idx + lookahead + 1)
    pattern_data = data.iloc[start_idx:end_idx].copy()
    if not isinstance(pattern_data.index, pd.DatetimeIndex):
        pattern_data.index = pd.date_range(start='2020-01-01', periods=len(pattern_data))
    fig, ax = plt.subplots(figsize=figsize)
    mpf.plot(pattern_data, type='candle', style='yahoo', ax=ax)
    pattern_idx -= start_idx
    dates = date2num(pattern_data.index.to_pydatetime())
    pattern_date = dates[pattern_idx]
    pattern_type = pattern.get('pattern_type', 'Unknown')
    direction = pattern.get('direction', 'Unknown')
    if pattern_type == 'doji':
        _highlight_candle(ax, pattern_data, pattern_idx, 'blue', 'Doji')
    elif pattern_type == 'hammer' or pattern_type == 'hanging_man':
        color = 'green' if direction == 'bullish' else 'red'
        label = 'Hammer' if direction == 'bullish' else 'Hanging Man'
        _highlight_candle(ax, pattern_data, pattern_idx, color, label)
    elif pattern_type == 'engulfing':
        color = 'green' if direction == 'bullish' else 'red'
        _highlight_candle(ax, pattern_data, pattern_idx - 1, 'gray', 'Previous')
        _highlight_candle(ax, pattern_data, pattern_idx, color, 'Engulfing')
    strength = pattern.get('strength', 0.0)
    ax.set_title(f'{title}: {pattern_type.capitalize()} ({direction.capitalize()}, Strength: {strength:.2f})', fontsize=14)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Price', fontsize=12)
    ax.legend(loc='upper left')
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig

def add_patterns_to_chart(ax: plt.Axes, data: pd.DataFrame, patterns: Dict[str, List[Dict[str, Any]]]) -> None:
    """
    Add pattern markers to a chart.
    
    Args:
        ax: Matplotlib axis
        data: DataFrame with OHLCV data
        patterns: Dictionary of pattern types and their occurrences
    """
    dates = date2num(data.index.to_pydatetime())
    pattern_colors = {'head_and_shoulders': 'red', 'inverse_head_and_shoulders': 'green', 'double_top': 'red', 'double_bottom': 'green', 'triple_top': 'red', 'triple_bottom': 'green', 'triangle': 'blue', 'flag': 'purple', 'pennant': 'orange', 'wedge_rising': 'cyan', 'wedge_falling': 'magenta', 'rectangle': 'brown', 'gartley': 'darkgreen', 'butterfly': 'darkblue', 'bat': 'darkorange', 'crab': 'darkred', 'doji': 'black', 'hammer': 'green', 'hanging_man': 'red', 'engulfing': 'blue'}
    for pattern_type, pattern_list in patterns.items():
        for pattern in pattern_list:
            if 'start_idx' in pattern and 'end_idx' in pattern:
                start_idx = pattern['start_idx']
                end_idx = pattern['end_idx']
                if start_idx >= len(data) or end_idx >= len(data):
                    continue
                start_date = dates[start_idx]
                end_date = dates[end_idx]
                start_price = data['low'].iloc[start_idx]
                end_price = data['high'].iloc[end_idx]
                color = pattern_colors.get(pattern_type, 'gray')
                rect = patches.Rectangle((start_date, start_price), end_date - start_date, end_price - start_price, linewidth=1, edgecolor=color, facecolor='none', alpha=0.7, label=pattern_type.capitalize())
                ax.add_patch(rect)
                direction = pattern.get('direction', '')
                direction_str = f' ({direction.capitalize()})' if direction else ''
                ax.annotate(f'{pattern_type.capitalize()}{direction_str}', (start_date, start_price), xytext=(0, -15), textcoords='offset points', color=color, fontsize=8)
            elif 'index' in pattern:
                idx = pattern['index']
                if idx >= len(data):
                    continue
                date = dates[idx]
                price = data['high'].iloc[idx]
                color = pattern_colors.get(pattern_type, 'gray')
                direction = pattern.get('direction', '')
                marker = '^' if direction == 'bullish' else 'v' if direction == 'bearish' else 'o'
                ax.plot(date, price, marker=marker, markersize=10, color=color, alpha=0.7, label=pattern_type.capitalize())
                direction_str = f' ({direction.capitalize()})' if direction else ''
                ax.annotate(f'{pattern_type.capitalize()}{direction_str}', (date, price), xytext=(0, 10), textcoords='offset points', color=color, fontsize=8)

def add_pattern_legend(ax: plt.Axes, patterns: Dict[str, List[Dict[str, Any]]]) -> None:
    """
    Add a legend for the patterns.
    
    Args:
        ax: Matplotlib axis
        patterns: Dictionary of pattern types and their occurrences
    """
    pattern_types = list(patterns.keys())
    handles, labels = ax.get_legend_handles_labels()
    unique_labels = []
    unique_handles = []
    for handle, label in zip(handles, labels):
        if label not in unique_labels:
            unique_labels.append(label)
            unique_handles.append(handle)
    if unique_handles:
        ax.legend(unique_handles, unique_labels, loc='upper left')

def add_fibonacci_ratios(ax: plt.Axes, pattern: Dict[str, Any], x_date: float, x_price: float, a_date: float, a_price: float, b_date: float, b_price: float, c_date: float, c_price: float, d_date: float, d_price: float) -> None:
    """
    Add Fibonacci ratio annotations to a harmonic pattern.
    
    Args:
        ax: Matplotlib axis
        pattern: Dictionary with pattern information
        x_date, x_price: X point coordinates
        a_date, a_price: A point coordinates
        b_date, b_price: B point coordinates
        c_date, c_price: C point coordinates
        d_date, d_price: D point coordinates
    """
    pattern_type = pattern.get('pattern_type', 'Unknown')
    xa_diff = abs(a_price - x_price)
    ab_diff = abs(b_price - a_price)
    ab_ratio = ab_diff / xa_diff
    bc_diff = abs(c_price - b_price)
    bc_ratio = bc_diff / ab_diff
    cd_diff = abs(d_price - c_price)
    cd_ratio = cd_diff / bc_diff
    xd_diff = abs(d_price - x_price)
    xd_ratio = xd_diff / xa_diff
    ax.annotate(f'AB/XA: {ab_ratio:.3f}', ((a_date + b_date) / 2, (a_price + b_price) / 2), xytext=(10, 10), textcoords='offset points', bbox=dict(boxstyle='round,pad=0.3', fc='yellow', alpha=0.5))
    ax.annotate(f'BC/AB: {bc_ratio:.3f}', ((b_date + c_date) / 2, (b_price + c_price) / 2), xytext=(10, -20), textcoords='offset points', bbox=dict(boxstyle='round,pad=0.3', fc='yellow', alpha=0.5))
    ax.annotate(f'CD/BC: {cd_ratio:.3f}', ((c_date + d_date) / 2, (c_price + d_price) / 2), xytext=(10, 10), textcoords='offset points', bbox=dict(boxstyle='round,pad=0.3', fc='yellow', alpha=0.5))
    ax.annotate(f'XD/XA: {xd_ratio:.3f}', ((x_date + d_date) / 2, (x_price + d_price) / 2), xytext=(-40, -20), textcoords='offset points', bbox=dict(boxstyle='round,pad=0.3', fc='yellow', alpha=0.5))

def highlight_candle(ax: plt.Axes, data: pd.DataFrame, idx: int, color: str, label: str) -> None:
    """
    Highlight a candlestick.
    
    Args:
        ax: Matplotlib axis
        data: DataFrame with OHLCV data
        idx: Index of the candle to highlight
        color: Color of the highlight
        label: Label for the legend
    """
    date = date2num(data.index[idx].to_pydatetime())
    open_price = data['open'].iloc[idx]
    high_price = data['high'].iloc[idx]
    low_price = data['low'].iloc[idx]
    close_price = data['close'].iloc[idx]
    width = 0.8
    rect = patches.Rectangle((date - width / 2, low_price), width, high_price - low_price, linewidth=2, edgecolor=color, facecolor='none', alpha=0.7, label=label)
    ax.add_patch(rect)
    ax.annotate(label, (date, high_price), xytext=(0, 20), textcoords='offset points', arrowprops=dict(arrowstyle='->', color=color), color=color, fontsize=10)