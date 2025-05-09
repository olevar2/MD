"""
Ichimoku Pattern Utilities Module.

This module provides utility functions for Ichimoku pattern analysis.
"""

from typing import List, Dict, Any, Optional, Tuple, Union
import pandas as pd
import numpy as np

from feature_store_service.indicators.advanced_patterns.ichimoku.models import (
    IchimokuPatternType,
    IchimokuComponents,
    IchimokuPattern
)


def detect_tk_cross(
    data: pd.DataFrame,
    tenkan_col: str = "ichimoku_tenkan",
    kijun_col: str = "ichimoku_kijun",
    price_col: str = "close",
    lookback: int = 5
) -> List[IchimokuPattern]:
    """
    Detect Tenkan-Kijun (TK) crosses in Ichimoku data.
    
    Args:
        data: DataFrame with Ichimoku data
        tenkan_col: Column name for Tenkan-sen
        kijun_col: Column name for Kijun-sen
        price_col: Column name for price data
        lookback: Number of bars to look back for confirmation
        
    Returns:
        List of detected TK cross patterns
    """
    if len(data) < lookback + 1:
        return []
    
    patterns = []
    
    # Calculate crosses
    data['tk_diff'] = data[tenkan_col] - data[kijun_col]
    data['tk_diff_prev'] = data['tk_diff'].shift(1)
    data['tk_cross_up'] = (data['tk_diff'] > 0) & (data['tk_diff_prev'] <= 0)
    data['tk_cross_down'] = (data['tk_diff'] < 0) & (data['tk_diff_prev'] >= 0)
    
    # Find TK crosses
    for i in range(lookback, len(data)):
        # Check for bullish cross
        if data['tk_cross_up'].iloc[i-1]:
            # Confirm the cross with lookback period
            if all(data[tenkan_col].iloc[i-lookback:i-1] < data[kijun_col].iloc[i-lookback:i-1]):
                # Create pattern
                components = IchimokuComponents(
                    tenkan_sen=data[tenkan_col].iloc[i],
                    kijun_sen=data[kijun_col].iloc[i],
                    senkou_span_a=data.get("ichimoku_senkou_a", pd.Series([0] * len(data))).iloc[i],
                    senkou_span_b=data.get("ichimoku_senkou_b", pd.Series([0] * len(data))).iloc[i],
                    chikou_span=data.get("ichimoku_chikou", pd.Series([0] * len(data))).iloc[i]
                )
                
                # Calculate strength based on distance between lines and price position
                price = data[price_col].iloc[i]
                cloud_top = max(components.senkou_span_a, components.senkou_span_b)
                cloud_bottom = min(components.senkou_span_a, components.senkou_span_b)
                
                # Stronger if price is above the cloud
                strength = 0.7  # Base strength
                if price > cloud_top:
                    strength = 0.9
                elif price < cloud_bottom:
                    strength = 0.5
                
                # Calculate target and stop prices
                kijun = components.kijun_sen
                tenkan = components.tenkan_sen
                target_price = price + (price - kijun) * 2  # Project the same distance from Kijun
                stop_price = kijun - (tenkan - kijun) / 2  # Halfway between Tenkan and Kijun
                
                pattern = IchimokuPattern(
                    pattern_type=IchimokuPatternType.TK_CROSS,
                    start_index=i-lookback,
                    end_index=i,
                    direction="bullish",
                    strength=strength,
                    components=components,
                    target_price=target_price,
                    stop_price=stop_price
                )
                
                patterns.append(pattern)
        
        # Check for bearish cross
        elif data['tk_cross_down'].iloc[i-1]:
            # Confirm the cross with lookback period
            if all(data[tenkan_col].iloc[i-lookback:i-1] > data[kijun_col].iloc[i-lookback:i-1]):
                # Create pattern
                components = IchimokuComponents(
                    tenkan_sen=data[tenkan_col].iloc[i],
                    kijun_sen=data[kijun_col].iloc[i],
                    senkou_span_a=data.get("ichimoku_senkou_a", pd.Series([0] * len(data))).iloc[i],
                    senkou_span_b=data.get("ichimoku_senkou_b", pd.Series([0] * len(data))).iloc[i],
                    chikou_span=data.get("ichimoku_chikou", pd.Series([0] * len(data))).iloc[i]
                )
                
                # Calculate strength based on distance between lines and price position
                price = data[price_col].iloc[i]
                cloud_top = max(components.senkou_span_a, components.senkou_span_b)
                cloud_bottom = min(components.senkou_span_a, components.senkou_span_b)
                
                # Stronger if price is below the cloud
                strength = 0.7  # Base strength
                if price < cloud_bottom:
                    strength = 0.9
                elif price > cloud_top:
                    strength = 0.5
                
                # Calculate target and stop prices
                kijun = components.kijun_sen
                tenkan = components.tenkan_sen
                target_price = price - (kijun - price) * 2  # Project the same distance from Kijun
                stop_price = kijun + (kijun - tenkan) / 2  # Halfway between Tenkan and Kijun
                
                pattern = IchimokuPattern(
                    pattern_type=IchimokuPatternType.TK_CROSS,
                    start_index=i-lookback,
                    end_index=i,
                    direction="bearish",
                    strength=strength,
                    components=components,
                    target_price=target_price,
                    stop_price=stop_price
                )
                
                patterns.append(pattern)
    
    return patterns


def detect_kumo_breakout(
    data: pd.DataFrame,
    senkou_a_col: str = "ichimoku_senkou_a",
    senkou_b_col: str = "ichimoku_senkou_b",
    price_col: str = "close",
    lookback: int = 10
) -> List[IchimokuPattern]:
    """
    Detect Kumo (cloud) breakouts in Ichimoku data.
    
    Args:
        data: DataFrame with Ichimoku data
        senkou_a_col: Column name for Senkou Span A
        senkou_b_col: Column name for Senkou Span B
        price_col: Column name for price data
        lookback: Number of bars to look back for confirmation
        
    Returns:
        List of detected Kumo breakout patterns
    """
    if len(data) < lookback + 1:
        return []
    
    patterns = []
    
    # Calculate cloud top and bottom
    data['cloud_top'] = data[[senkou_a_col, senkou_b_col]].max(axis=1)
    data['cloud_bottom'] = data[[senkou_a_col, senkou_b_col]].min(axis=1)
    
    # Calculate breakouts
    data['above_cloud'] = data[price_col] > data['cloud_top']
    data['below_cloud'] = data[price_col] < data['cloud_bottom']
    data['above_cloud_prev'] = data['above_cloud'].shift(1)
    data['below_cloud_prev'] = data['below_cloud'].shift(1)
    
    # Find Kumo breakouts
    for i in range(lookback, len(data)):
        # Check for bullish breakout (price breaks above the cloud)
        if data['above_cloud'].iloc[i] and not data['above_cloud_prev'].iloc[i]:
            # Confirm the breakout with lookback period
            if all(data[price_col].iloc[i-lookback:i] >= data['cloud_bottom'].iloc[i-lookback:i]):
                # Create pattern
                components = IchimokuComponents(
                    tenkan_sen=data.get("ichimoku_tenkan", pd.Series([0] * len(data))).iloc[i],
                    kijun_sen=data.get("ichimoku_kijun", pd.Series([0] * len(data))).iloc[i],
                    senkou_span_a=data[senkou_a_col].iloc[i],
                    senkou_span_b=data[senkou_b_col].iloc[i],
                    chikou_span=data.get("ichimoku_chikou", pd.Series([0] * len(data))).iloc[i]
                )
                
                # Calculate strength based on cloud thickness and time spent near cloud
                cloud_thickness = abs(components.senkou_span_a - components.senkou_span_b)
                price = data[price_col].iloc[i]
                avg_price = data[price_col].iloc[i-lookback:i].mean()
                
                # Stronger if breakout is decisive and cloud is thick
                strength = 0.7  # Base strength
                if cloud_thickness > (price * 0.02):  # Cloud is thick (>2% of price)
                    strength += 0.1
                if (price - data['cloud_top'].iloc[i]) > (price * 0.01):  # Decisive breakout (>1% above cloud)
                    strength += 0.1
                
                # Calculate target and stop prices
                cloud_height = data['cloud_top'].iloc[i] - data['cloud_bottom'].iloc[i]
                target_price = price + cloud_height  # Project the cloud height from breakout point
                stop_price = data['cloud_bottom'].iloc[i]  # Stop at cloud bottom
                
                pattern = IchimokuPattern(
                    pattern_type=IchimokuPatternType.KUMO_BREAKOUT,
                    start_index=i-lookback,
                    end_index=i,
                    direction="bullish",
                    strength=min(1.0, strength),
                    components=components,
                    target_price=target_price,
                    stop_price=stop_price
                )
                
                patterns.append(pattern)
        
        # Check for bearish breakout (price breaks below the cloud)
        elif data['below_cloud'].iloc[i] and not data['below_cloud_prev'].iloc[i]:
            # Confirm the breakout with lookback period
            if all(data[price_col].iloc[i-lookback:i] <= data['cloud_top'].iloc[i-lookback:i]):
                # Create pattern
                components = IchimokuComponents(
                    tenkan_sen=data.get("ichimoku_tenkan", pd.Series([0] * len(data))).iloc[i],
                    kijun_sen=data.get("ichimoku_kijun", pd.Series([0] * len(data))).iloc[i],
                    senkou_span_a=data[senkou_a_col].iloc[i],
                    senkou_span_b=data[senkou_b_col].iloc[i],
                    chikou_span=data.get("ichimoku_chikou", pd.Series([0] * len(data))).iloc[i]
                )
                
                # Calculate strength based on cloud thickness and time spent near cloud
                cloud_thickness = abs(components.senkou_span_a - components.senkou_span_b)
                price = data[price_col].iloc[i]
                avg_price = data[price_col].iloc[i-lookback:i].mean()
                
                # Stronger if breakout is decisive and cloud is thick
                strength = 0.7  # Base strength
                if cloud_thickness > (price * 0.02):  # Cloud is thick (>2% of price)
                    strength += 0.1
                if (data['cloud_bottom'].iloc[i] - price) > (price * 0.01):  # Decisive breakout (>1% below cloud)
                    strength += 0.1
                
                # Calculate target and stop prices
                cloud_height = data['cloud_top'].iloc[i] - data['cloud_bottom'].iloc[i]
                target_price = price - cloud_height  # Project the cloud height from breakout point
                stop_price = data['cloud_top'].iloc[i]  # Stop at cloud top
                
                pattern = IchimokuPattern(
                    pattern_type=IchimokuPatternType.KUMO_BREAKOUT,
                    start_index=i-lookback,
                    end_index=i,
                    direction="bearish",
                    strength=min(1.0, strength),
                    components=components,
                    target_price=target_price,
                    stop_price=stop_price
                )
                
                patterns.append(pattern)
    
    return patterns


def detect_kumo_twist(
    data: pd.DataFrame,
    senkou_a_col: str = "ichimoku_senkou_a",
    senkou_b_col: str = "ichimoku_senkou_b",
    price_col: str = "close",
    lookback: int = 5
) -> List[IchimokuPattern]:
    """
    Detect Kumo (cloud) twists in Ichimoku data.
    
    Args:
        data: DataFrame with Ichimoku data
        senkou_a_col: Column name for Senkou Span A
        senkou_b_col: Column name for Senkou Span B
        price_col: Column name for price data
        lookback: Number of bars to look back for confirmation
        
    Returns:
        List of detected Kumo twist patterns
    """
    if len(data) < lookback + 1:
        return []
    
    patterns = []
    
    # Calculate cloud direction
    data['senkou_diff'] = data[senkou_a_col] - data[senkou_b_col]
    data['senkou_diff_prev'] = data['senkou_diff'].shift(1)
    data['kumo_twist_bullish'] = (data['senkou_diff'] > 0) & (data['senkou_diff_prev'] <= 0)
    data['kumo_twist_bearish'] = (data['senkou_diff'] < 0) & (data['senkou_diff_prev'] >= 0)
    
    # Find Kumo twists
    for i in range(lookback, len(data)):
        # Check for bullish twist (Senkou A crosses above Senkou B)
        if data['kumo_twist_bullish'].iloc[i-1]:
            # Confirm the twist with lookback period
            if all(data[senkou_a_col].iloc[i-lookback:i-1] < data[senkou_b_col].iloc[i-lookback:i-1]):
                # Create pattern
                components = IchimokuComponents(
                    tenkan_sen=data.get("ichimoku_tenkan", pd.Series([0] * len(data))).iloc[i],
                    kijun_sen=data.get("ichimoku_kijun", pd.Series([0] * len(data))).iloc[i],
                    senkou_span_a=data[senkou_a_col].iloc[i],
                    senkou_span_b=data[senkou_b_col].iloc[i],
                    chikou_span=data.get("ichimoku_chikou", pd.Series([0] * len(data))).iloc[i]
                )
                
                # Calculate strength based on price position relative to cloud
                price = data[price_col].iloc[i]
                cloud_top = max(components.senkou_span_a, components.senkou_span_b)
                cloud_bottom = min(components.senkou_span_a, components.senkou_span_b)
                
                # Stronger if price is above the cloud
                strength = 0.7  # Base strength
                if price > cloud_top:
                    strength = 0.9
                elif price < cloud_bottom:
                    strength = 0.5
                
                # Calculate target and stop prices
                cloud_height = cloud_top - cloud_bottom
                target_price = price + cloud_height  # Project the cloud height from current price
                stop_price = min(data[price_col].iloc[i-lookback:i])  # Stop at recent low
                
                pattern = IchimokuPattern(
                    pattern_type=IchimokuPatternType.KUMO_TWIST,
                    start_index=i-lookback,
                    end_index=i,
                    direction="bullish",
                    strength=strength,
                    components=components,
                    target_price=target_price,
                    stop_price=stop_price
                )
                
                patterns.append(pattern)
        
        # Check for bearish twist (Senkou A crosses below Senkou B)
        elif data['kumo_twist_bearish'].iloc[i-1]:
            # Confirm the twist with lookback period
            if all(data[senkou_a_col].iloc[i-lookback:i-1] > data[senkou_b_col].iloc[i-lookback:i-1]):
                # Create pattern
                components = IchimokuComponents(
                    tenkan_sen=data.get("ichimoku_tenkan", pd.Series([0] * len(data))).iloc[i],
                    kijun_sen=data.get("ichimoku_kijun", pd.Series([0] * len(data))).iloc[i],
                    senkou_span_a=data[senkou_a_col].iloc[i],
                    senkou_span_b=data[senkou_b_col].iloc[i],
                    chikou_span=data.get("ichimoku_chikou", pd.Series([0] * len(data))).iloc[i]
                )
                
                # Calculate strength based on price position relative to cloud
                price = data[price_col].iloc[i]
                cloud_top = max(components.senkou_span_a, components.senkou_span_b)
                cloud_bottom = min(components.senkou_span_a, components.senkou_span_b)
                
                # Stronger if price is below the cloud
                strength = 0.7  # Base strength
                if price < cloud_bottom:
                    strength = 0.9
                elif price > cloud_top:
                    strength = 0.5
                
                # Calculate target and stop prices
                cloud_height = cloud_top - cloud_bottom
                target_price = price - cloud_height  # Project the cloud height from current price
                stop_price = max(data[price_col].iloc[i-lookback:i])  # Stop at recent high
                
                pattern = IchimokuPattern(
                    pattern_type=IchimokuPatternType.KUMO_TWIST,
                    start_index=i-lookback,
                    end_index=i,
                    direction="bearish",
                    strength=strength,
                    components=components,
                    target_price=target_price,
                    stop_price=stop_price
                )
                
                patterns.append(pattern)
    
    return patterns


def detect_chikou_cross(
    data: pd.DataFrame,
    chikou_col: str = "ichimoku_chikou",
    price_col: str = "close",
    lookback: int = 5
) -> List[IchimokuPattern]:
    """
    Detect Chikou Span crosses in Ichimoku data.
    
    Args:
        data: DataFrame with Ichimoku data
        chikou_col: Column name for Chikou Span
        price_col: Column name for price data
        lookback: Number of bars to look back for confirmation
        
    Returns:
        List of detected Chikou cross patterns
    """
    if len(data) < lookback + 1:
        return []
    
    patterns = []
    
    # Calculate Chikou crosses
    # Note: Chikou Span is the current price shifted back 26 periods
    # So we need to compare it with the price from 26 periods ago
    data['price_26_ago'] = data[price_col].shift(-26)  # Price from 26 periods ago
    data['chikou_diff'] = data[chikou_col] - data['price_26_ago']
    data['chikou_diff_prev'] = data['chikou_diff'].shift(1)
    data['chikou_cross_up'] = (data['chikou_diff'] > 0) & (data['chikou_diff_prev'] <= 0)
    data['chikou_cross_down'] = (data['chikou_diff'] < 0) & (data['chikou_diff_prev'] >= 0)
    
    # Find Chikou crosses
    for i in range(lookback, len(data)):
        # Check for bullish cross (Chikou crosses above price)
        if data['chikou_cross_up'].iloc[i-1]:
            # Confirm the cross with lookback period
            if all(data[chikou_col].iloc[i-lookback:i-1] < data['price_26_ago'].iloc[i-lookback:i-1]):
                # Create pattern
                components = IchimokuComponents(
                    tenkan_sen=data.get("ichimoku_tenkan", pd.Series([0] * len(data))).iloc[i],
                    kijun_sen=data.get("ichimoku_kijun", pd.Series([0] * len(data))).iloc[i],
                    senkou_span_a=data.get("ichimoku_senkou_a", pd.Series([0] * len(data))).iloc[i],
                    senkou_span_b=data.get("ichimoku_senkou_b", pd.Series([0] * len(data))).iloc[i],
                    chikou_span=data[chikou_col].iloc[i]
                )
                
                # Calculate strength based on other Ichimoku components
                price = data[price_col].iloc[i]
                cloud_top = max(components.senkou_span_a, components.senkou_span_b)
                cloud_bottom = min(components.senkou_span_a, components.senkou_span_b)
                
                # Stronger if price is above the cloud and Tenkan > Kijun
                strength = 0.7  # Base strength
                if price > cloud_top:
                    strength += 0.1
                if components.tenkan_sen > components.kijun_sen:
                    strength += 0.1
                
                # Calculate target and stop prices
                kijun = components.kijun_sen
                target_price = price + (price - kijun) * 2  # Project the same distance from Kijun
                stop_price = kijun  # Stop at Kijun
                
                pattern = IchimokuPattern(
                    pattern_type=IchimokuPatternType.CHIKOU_CROSS,
                    start_index=i-lookback,
                    end_index=i,
                    direction="bullish",
                    strength=min(1.0, strength),
                    components=components,
                    target_price=target_price,
                    stop_price=stop_price
                )
                
                patterns.append(pattern)
        
        # Check for bearish cross (Chikou crosses below price)
        elif data['chikou_cross_down'].iloc[i-1]:
            # Confirm the cross with lookback period
            if all(data[chikou_col].iloc[i-lookback:i-1] > data['price_26_ago'].iloc[i-lookback:i-1]):
                # Create pattern
                components = IchimokuComponents(
                    tenkan_sen=data.get("ichimoku_tenkan", pd.Series([0] * len(data))).iloc[i],
                    kijun_sen=data.get("ichimoku_kijun", pd.Series([0] * len(data))).iloc[i],
                    senkou_span_a=data.get("ichimoku_senkou_a", pd.Series([0] * len(data))).iloc[i],
                    senkou_span_b=data.get("ichimoku_senkou_b", pd.Series([0] * len(data))).iloc[i],
                    chikou_span=data[chikou_col].iloc[i]
                )
                
                # Calculate strength based on other Ichimoku components
                price = data[price_col].iloc[i]
                cloud_top = max(components.senkou_span_a, components.senkou_span_b)
                cloud_bottom = min(components.senkou_span_a, components.senkou_span_b)
                
                # Stronger if price is below the cloud and Tenkan < Kijun
                strength = 0.7  # Base strength
                if price < cloud_bottom:
                    strength += 0.1
                if components.tenkan_sen < components.kijun_sen:
                    strength += 0.1
                
                # Calculate target and stop prices
                kijun = components.kijun_sen
                target_price = price - (kijun - price) * 2  # Project the same distance from Kijun
                stop_price = kijun  # Stop at Kijun
                
                pattern = IchimokuPattern(
                    pattern_type=IchimokuPatternType.CHIKOU_CROSS,
                    start_index=i-lookback,
                    end_index=i,
                    direction="bearish",
                    strength=min(1.0, strength),
                    components=components,
                    target_price=target_price,
                    stop_price=stop_price
                )
                
                patterns.append(pattern)
    
    return patterns