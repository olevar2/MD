"""
Market Manipulation Detection Module

This module provides functionality for detecting potential market manipulation patterns
in forex price and volume data, including unusual price/volume relationships,
stop hunting, and fake breakout patterns.
"""

from typing import Dict, List, Any, Union, Optional, Tuple
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from scipy.stats import zscore, norm
from scipy.signal import find_peaks, argrelextrema

from analysis_engine.analysis.base_analyzer import BaseAnalyzer
from analysis_engine.models.analysis_result import AnalysisResult
from analysis_engine.models.market_data import MarketData

logger = logging.getLogger(__name__)


class MarketManipulationAnalyzer(BaseAnalyzer):
    """
    Analyzer for detecting potential market manipulation patterns in forex markets.
    
    This analyzer identifies unusual price/volume relationships, stop hunting patterns,
    fake breakouts, and other common manipulation tactics. It provides confidence
    scoring and trade protection recommendations.
    """
    
    def __init__(self, parameters: Dict[str, Any] = None):
        """
        Initialize the market manipulation detector
        
        Args:
            parameters: Configuration parameters for the detector
        """
        default_params = {
            "volatility_window": 20,  # Window for calculating price volatility
            "volume_window": 20,  # Window for calculating volume baselines
            "volume_z_threshold": 2.0,  # Z-score threshold for abnormal volume
            "price_reversal_threshold": 0.5,  # % threshold for price reversals
            "stop_hunting_lookback": 30,  # Bars to look back for stop hunting patterns
            "stop_hunting_recovery": 0.5,  # % recovery after stop hunting
            "fake_breakout_threshold": 0.7,  # % threshold for fake breakout detection
            "support_resistance_lookback": 100,  # Bars to look back for S/R levels
            "confidence_high_threshold": 0.8,  # Threshold for high confidence manipulations
            "confidence_medium_threshold": 0.6,  # Threshold for medium confidence manipulations
            "sensitive_time_windows": [  # High-sensitivity time windows (UTC)
                {"start": "21:00", "end": "22:00", "description": "NY Close/Asian Open"},
                {"start": "07:00", "end": "08:00", "description": "London Open"},
                {"start": "12:00", "end": "13:00", "description": "NY Open"},
                {"start": "14:30", "end": "15:30", "description": "Major US News Releases"}
            ],
            "min_samples": 100  # Minimum samples needed for reliable detection
        }
        
        # Merge default parameters with provided parameters
        merged_params = {**default_params, **(parameters or {})}
        super().__init__("market_manipulation_detector", merged_params)
        
    def _preprocess_data(self, ohlcv_data: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess OHLCV data for manipulation detection
        
        Args:
            ohlcv_data: DataFrame with OHLCV data
            
        Returns:
            Preprocessed DataFrame with additional calculated fields
        """
        if len(ohlcv_data) < self.parameters["min_samples"]:
            logger.warning(f"Insufficient data points ({len(ohlcv_data)}) for reliable manipulation detection")
            return ohlcv_data
        
        df = ohlcv_data.copy()
        
        # Ensure we have the necessary columns
        required_columns = ['open', 'high', 'low', 'close']
        if not all(col in df.columns for col in required_columns):
            logger.error("Missing required OHLC columns for manipulation detection")
            return df
            
        # Check if volume data is available
        has_volume = 'volume' in df.columns
        
        # Calculate returns
        df['returns'] = df['close'].pct_change()
        
        # Calculate volatility
        df['volatility'] = df['returns'].rolling(self.parameters["volatility_window"]).std()
        
        # Calculate true range and ATR
        df['tr'] = np.maximum(
            np.maximum(
                df['high'] - df['low'],
                np.abs(df['high'] - df['close'].shift(1))
            ),
            np.abs(df['low'] - df['close'].shift(1))
        )
        df['atr'] = df['tr'].rolling(self.parameters["volatility_window"]).mean()
        
        # Calculate price ranges
        df['range'] = df['high'] - df['low']
        df['body'] = np.abs(df['close'] - df['open'])
        df['body_pct'] = df['body'] / df['range']
        df['upper_wick'] = df['high'] - np.maximum(df['open'], df['close'])
        df['lower_wick'] = np.minimum(df['open'], df['close']) - df['low']
        
        # Calculate if candle is bullish or bearish
        df['is_bullish'] = df['close'] > df['open']
        
        # Calculate volume metrics if volume data is available
        if has_volume:
            # Calculate volume Z-score
            df['volume_ma'] = df['volume'].rolling(self.parameters["volume_window"]).mean()
            df['volume_std'] = df['volume'].rolling(self.parameters["volume_window"]).std()
            df['volume_z'] = (df['volume'] - df['volume_ma']) / df['volume_std']
            
            # Calculate price/volume correlation
            df['price_volume_corr'] = df['returns'].rolling(self.parameters["volume_window"]).corr(
                df['volume'].pct_change()
            )
        
        # Create time-based features if timestamp is available
        if 'timestamp' in df.columns:
            df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
            df['minute'] = pd.to_datetime(df['timestamp']).dt.minute
            df['time_string'] = df['hour'].astype(str).str.zfill(2) + ':' + df['minute'].astype(str).str.zfill(2)
            
            # Flag sensitive time windows
            df['is_sensitive_time'] = False
            for window in self.parameters["sensitive_time_windows"]:
                mask = (df['time_string'] >= window["start"]) & (df['time_string'] <= window["end"])
                df.loc[mask, 'is_sensitive_time'] = True
        
        return df
        
    def _identify_support_resistance_levels(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Identify potential support and resistance levels
        
        Args:
            df: Preprocessed DataFrame with OHLCV data
            
        Returns:
            List of detected support/resistance levels
        """
        levels = []
        lookback = self.parameters["support_resistance_lookback"]
        
        if len(df) < lookback:
            return levels
            
        # Find local highs (resistance)
        high_idx = argrelextrema(df['high'].values, np.greater, order=5)[0]
        for idx in high_idx:
            if idx < len(df) - 5:  # Ensure we have some data after this point
                level_value = df.iloc[idx]['high']
                level_strength = self._calculate_level_strength(df, idx, level_value, is_support=False)
                
                # Only include significant levels
                if level_strength > 0.5:
                    levels.append({
                        "price": level_value,
                        "type": "resistance",
                        "strength": level_strength,
                        "index": idx
                    })
        
        # Find local lows (support)
        low_idx = argrelextrema(df['low'].values, np.less, order=5)[0]
        for idx in low_idx:
            if idx < len(df) - 5:  # Ensure we have some data after this point
                level_value = df.iloc[idx]['low']
                level_strength = self._calculate_level_strength(df, idx, level_value, is_support=True)
                
                # Only include significant levels
                if level_strength > 0.5:
                    levels.append({
                        "price": level_value,
                        "type": "support",
                        "strength": level_strength,
                        "index": idx
                    })
        
        # Group nearby levels
        grouped_levels = self._group_nearby_levels(levels, df['atr'].mean())
        
        return grouped_levels
    
    def _calculate_level_strength(self, df: pd.DataFrame, idx: int, level_price: float, is_support: bool) -> float:
        """
        Calculate the strength of a support/resistance level
        
        Args:
            df: DataFrame with price data
            idx: Index of the level in the DataFrame
            level_price: Price value of the level
            is_support: Whether this is a support (True) or resistance (False) level
            
        Returns:
            Strength score between 0 and 1
        """
        # Count touches before this point
        lookback_df = df.iloc[:idx+1]
        touches = 0
        
        price_range = lookback_df['high'].max() - lookback_df['low'].min()
        proximity_threshold = 0.001 * price_range  # 0.1% of the price range
        
        for i in range(len(lookback_df)):
            if is_support:
                price_diff = abs(lookback_df.iloc[i]['low'] - level_price)
                if price_diff <= proximity_threshold and i != idx:
                    touches += 1
            else:
                price_diff = abs(lookback_df.iloc[i]['high'] - level_price)
                if price_diff <= proximity_threshold and i != idx:
                    touches += 1
        
        # Calculate strength based on number of touches and recency
        recency_factor = 1.0 - (idx / len(df)) if len(df) > 0 else 0.5
        touch_factor = min(touches / 3, 1.0)  # Cap at 1.0
        
        # Combine factors
        strength = 0.6 * touch_factor + 0.4 * recency_factor
        
        return strength
        
    def _group_nearby_levels(self, levels: List[Dict[str, Any]], atr: float) -> List[Dict[str, Any]]:
        """
        Group nearby support/resistance levels
        
        Args:
            levels: List of detected levels
            atr: Average True Range value
            
        Returns:
            List of grouped levels
        """
        if not levels:
            return []
            
        # Sort levels by price
        sorted_levels = sorted(levels, key=lambda x: x["price"])
        
        # Group levels that are within 0.5*ATR of each other
        threshold = 0.5 * atr
        grouped = []
        current_group = [sorted_levels[0]]
        
        for i in range(1, len(sorted_levels)):
            if sorted_levels[i]["price"] - sorted_levels[i-1]["price"] < threshold:
                # Add to current group
                current_group.append(sorted_levels[i])
            else:
                # Process current group
                if current_group:
                    # Take the strongest level from the group
                    strongest = max(current_group, key=lambda x: x["strength"])
                    grouped.append(strongest)
                    
                # Start new group
                current_group = [sorted_levels[i]]
                
        # Process the last group
        if current_group:
            strongest = max(current_group, key=lambda x: x["strength"])
            grouped.append(strongest)
            
        return grouped
        
    def _detect_volume_anomalies(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Detect abnormal volume patterns
        
        Args:
            df: Preprocessed DataFrame with OHLCV data
            
        Returns:
            List of detected volume anomalies
        """
        anomalies = []
        
        if 'volume' not in df.columns:
            logger.warning("Volume data not available for volume anomaly detection")
            return anomalies
            
        if len(df) < self.parameters["volume_window"] + 5:
            logger.warning("Insufficient data for volume anomaly detection")
            return anomalies
            
        z_threshold = self.parameters["volume_z_threshold"]
        
        for i in range(self.parameters["volume_window"], len(df)):
            # Check for abnormal volume
            if abs(df.iloc[i]['volume_z']) > z_threshold:
                # Check for volume spike with price reversal (potential manipulation)
                price_change = df.iloc[i]['returns']
                next_price_change = df.iloc[i+1]['returns'] if i+1 < len(df) else 0
                
                # Calculate if the volume spike is accompanied by a price reversal
                reversal_detected = price_change * next_price_change < 0 and abs(next_price_change) > 0.001
                
                # Check if the volume spike is not in the expected direction
                unexpected_direction = (df.iloc[i]['is_bullish'] and df.iloc[i]['volume_z'] < -z_threshold) or \
                                      (not df.iloc[i]['is_bullish'] and df.iloc[i]['volume_z'] > z_threshold)
                
                # Calculate confidence score
                confidence = 0.5 + min(abs(df.iloc[i]['volume_z']) / 10, 0.3)  # Base confidence from volume size
                
                if reversal_detected:
                    confidence += 0.2  # Increase confidence if there's a reversal
                    
                if unexpected_direction:
                    confidence += 0.1  # Increase confidence if volume direction doesn't match price
                
                # Add time-based factor
                if 'is_sensitive_time' in df.columns and df.iloc[i]['is_sensitive_time']:
                    confidence += 0.1  # Higher confidence during sensitive market hours
                
                # Cap confidence at 1.0
                confidence = min(confidence, 1.0)
                
                anomaly = {
                    "index": i,
                    "timestamp": df.iloc[i].get('timestamp', i),
                    "type": "volume_anomaly",
                    "subtype": "reversal" if reversal_detected else "spike",
                    "price": df.iloc[i]['close'],
                    "volume": df.iloc[i]['volume'],
                    "volume_z": df.iloc[i]['volume_z'],
                    "confidence": confidence,
                    "description": f"{'Bearish' if df.iloc[i]['returns'] < 0 else 'Bullish'} volume anomaly"
                }
                
                anomalies.append(anomaly)
        
        return anomalies
    
    def _detect_stop_hunting(self, df: pd.DataFrame, support_resistance: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Detect potential stop hunting patterns
        
        Args:
            df: Preprocessed DataFrame with OHLCV data
            support_resistance: List of support and resistance levels
            
        Returns:
            List of detected stop hunting patterns
        """
        patterns = []
        lookback = self.parameters["stop_hunting_lookback"]
        recovery_threshold = self.parameters["stop_hunting_recovery"]
        
        if len(df) < lookback + 5:
            logger.warning("Insufficient data for stop hunting detection")
            return patterns
            
        for level in support_resistance:
            level_price = level["price"]
            level_idx = level["index"]
            
            # Only consider recent levels
            if len(df) - level_idx > lookback:
                continue
                
            # Check if price broke through the level significantly
            for i in range(level_idx + 1, min(level_idx + 10, len(df))):
                breakthrough = False
                reversal = False
                
                if level["type"] == "support":
                    # Check for break below support
                    if df.iloc[i]['low'] < level_price * 0.998:  # 0.2% below support
                        breakthrough = True
                        # Check for reversal back above support
                        for j in range(i+1, min(i+5, len(df))):
                            if df.iloc[j]['close'] > level_price * (1 + recovery_threshold/100):
                                reversal = True
                                reversal_idx = j
                                break
                else:  # resistance
                    # Check for break above resistance
                    if df.iloc[i]['high'] > level_price * 1.002:  # 0.2% above resistance
                        breakthrough = True
                        # Check for reversal back below resistance
                        for j in range(i+1, min(i+5, len(df))):
                            if df.iloc[j]['close'] < level_price * (1 - recovery_threshold/100):
                                reversal = True
                                reversal_idx = j
                                break
                                
                if breakthrough and reversal:
                    # Calculate confidence based on multiple factors
                    base_confidence = 0.6  # Base confidence for pattern match
                    
                    # Factor: Level strength
                    confidence = base_confidence + min(level["strength"] * 0.2, 0.2)
                    
                    # Factor: Speed of reversal
                    reversal_speed = 1.0 / (reversal_idx - i) if reversal_idx > i else 1.0
                    confidence += reversal_speed * 0.1
                    
                    # Factor: Volume on breakthrough (if available)
                    if 'volume_z' in df.columns and not pd.isna(df.iloc[i]['volume_z']):
                        if abs(df.iloc[i]['volume_z']) > 1.0:
                            confidence += min(abs(df.iloc[i]['volume_z']) * 0.05, 0.1)
                    
                    # Factor: Time-based sensitivity
                    if 'is_sensitive_time' in df.columns and df.iloc[i]['is_sensitive_time']:
                        confidence += 0.1
                    
                    # Cap confidence at 1.0
                    confidence = min(confidence, 1.0)
                    
                    pattern = {
                        "type": "stop_hunting",
                        "subtype": f"{level['type']}_violation",
                        "level_price": level_price,
                        "breakthrough_index": i,
                        "breakthrough_price": df.iloc[i]['low'] if level["type"] == "support" else df.iloc[i]['high'],
                        "reversal_index": reversal_idx,
                        "reversal_price": df.iloc[reversal_idx]['close'],
                        "confidence": confidence,
                        "level_strength": level["strength"],
                        "description": f"Potential stop hunting: {level['type']} violation with quick reversal"
                    }
                    
                    patterns.append(pattern)
                    break  # Found a pattern for this level, move to next level
        
        return patterns
        
    def _detect_fake_breakouts(self, df: pd.DataFrame, support_resistance: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Detect fake breakout patterns
        
        Args:
            df: Preprocessed DataFrame with OHLCV data
            support_resistance: List of support and resistance levels
            
        Returns:
            List of detected fake breakout patterns
        """
        patterns = []
        threshold = self.parameters["fake_breakout_threshold"]
        
        if len(df) < 20:
            logger.warning("Insufficient data for fake breakout detection")
            return patterns
            
        for level in support_resistance:
            level_price = level["price"]
            level_idx = level["index"]
            
            # Check if we have enough data after the level was established
            if level_idx >= len(df) - 5:
                continue
                
            # Check for breakout followed by failed follow-through
            for i in range(level_idx + 1, len(df) - 3):
                breakout = False
                fail = False
                
                if level["type"] == "resistance":
                    # Check for break above resistance
                    if df.iloc[i]['close'] > level_price * 1.001:  # 0.1% above resistance
                        breakout = True
                        # Check for failure to follow through
                        if df.iloc[i+1]['close'] < level_price or df.iloc[i+2]['close'] < level_price:
                            fail = True
                else:  # support
                    # Check for break below support
                    if df.iloc[i]['close'] < level_price * 0.999:  # 0.1% below support
                        breakout = True
                        # Check for failure to follow through
                        if df.iloc[i+1]['close'] > level_price or df.iloc[i+2]['close'] > level_price:
                            fail = True
                            
                if breakout and fail:
                    # Calculate confidence based on multiple factors
                    base_confidence = 0.6  # Base confidence for pattern match
                    
                    # Factor: Level strength
                    confidence = base_confidence + min(level["strength"] * 0.2, 0.2)
                    
                    # Factor: Initial breakout strength
                    breakout_strength = abs(df.iloc[i]['close'] - level_price) / df.iloc[i]['atr']
                    if breakout_strength > 0.3:
                        confidence += min(breakout_strength * 0.1, 0.1)
                    
                    # Factor: Volume on breakout (if available)
                    if 'volume_z' in df.columns and not pd.isna(df.iloc[i]['volume_z']):
                        if abs(df.iloc[i]['volume_z']) > 1.0:
                            confidence += min(abs(df.iloc[i]['volume_z']) * 0.05, 0.1)
                    
                    # Cap confidence at 1.0
                    confidence = min(confidence, 1.0)
                    
                    pattern = {
                        "type": "fake_breakout",
                        "subtype": f"{level['type']}_fake",
                        "level_price": level_price,
                        "breakout_index": i,
                        "breakout_price": df.iloc[i]['close'],
                        "failure_index": i+1 if df.iloc[i+1]['close'] < level_price else i+2,
                        "failure_price": df.iloc[i+1]['close'] if df.iloc[i+1]['close'] < level_price else df.iloc[i+2]['close'],
                        "confidence": confidence,
                        "level_strength": level["strength"],
                        "description": f"Potential fake breakout of {level['type']} level"
                    }
                    
                    patterns.append(pattern)
                    break  # Found a pattern for this level, move to next level
        
        return patterns
        
    def _identify_manipulation_clusters(self, all_patterns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Identify clusters of manipulation patterns that reinforce each other
        
        Args:
            all_patterns: List of all detected patterns
            
        Returns:
            List of clustered manipulation events
        """
        if not all_patterns or len(all_patterns) < 2:
            return []
            
        # Sort patterns by index/time
        sorted_patterns = sorted(all_patterns, key=lambda x: x.get('index', 0) if 'index' in x 
                                else x.get('breakthrough_index', 0) if 'breakthrough_index' in x
                                else x.get('breakout_index', 0))
        
        # Identify clusters of patterns that occur close to each other
        clusters = []
        current_cluster = [sorted_patterns[0]]
        cluster_threshold = 3  # Max bars between patterns to be considered a cluster
        
        for i in range(1, len(sorted_patterns)):
            current_idx = sorted_patterns[i].get('index', 0) if 'index' in sorted_patterns[i] else \
                          sorted_patterns[i].get('breakthrough_index', 0) if 'breakthrough_index' in sorted_patterns[i] else \
                          sorted_patterns[i].get('breakout_index', 0)
            
            prev_idx = sorted_patterns[i-1].get('index', 0) if 'index' in sorted_patterns[i-1] else \
                       sorted_patterns[i-1].get('breakthrough_index', 0) if 'breakthrough_index' in sorted_patterns[i-1] else \
                       sorted_patterns[i-1].get('breakout_index', 0)
            
            if current_idx - prev_idx <= cluster_threshold:
                # Add to current cluster
                current_cluster.append(sorted_patterns[i])
            else:
                # Process current cluster
                if len(current_cluster) > 1:
                    # Create cluster with combined confidence
                    combined_confidence = min(1.0, sum([p.get('confidence', 0.5) for p in current_cluster]) / len(current_cluster) + 0.1)
                    
                    cluster_event = {
                        "type": "manipulation_cluster",
                        "patterns": current_cluster,
                        "start_index": min([p.get('index', p.get('breakthrough_index', p.get('breakout_index', 0))) 
                                          for p in current_cluster]),
                        "pattern_count": len(current_cluster),
                        "confidence": combined_confidence,
                        "description": f"Cluster of {len(current_cluster)} potentially manipulative patterns"
                    }
                    clusters.append(cluster_event)
                
                # Start new cluster
                current_cluster = [sorted_patterns[i]]
        
        # Process the last cluster
        if len(current_cluster) > 1:
            combined_confidence = min(1.0, sum([p.get('confidence', 0.5) for p in current_cluster]) / len(current_cluster) + 0.1)
            
            cluster_event = {
                "type": "manipulation_cluster",
                "patterns": current_cluster,
                "start_index": min([p.get('index', p.get('breakthrough_index', p.get('breakout_index', 0))) 
                                  for p in current_cluster]),
                "pattern_count": len(current_cluster),
                "confidence": combined_confidence,
                "description": f"Cluster of {len(current_cluster)} potentially manipulative patterns"
            }
            clusters.append(cluster_event)
            
        return clusters
        
    def _generate_protection_recommendations(
        self, 
        patterns: List[Dict[str, Any]], 
        clusters: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Generate trade protection recommendations based on detected patterns
        
        Args:
            patterns: List of detected manipulation patterns
            clusters: List of identified pattern clusters
            
        Returns:
            List of protection recommendations
        """
        recommendations = []
        
        # Process high-confidence individual patterns
        for pattern in patterns:
            confidence = pattern.get('confidence', 0.0)
            if confidence >= self.parameters["confidence_high_threshold"]:
                rec = {
                    "type": "protection",
                    "trigger": pattern["type"],
                    "confidence": confidence,
                    "actions": []
                }
                
                # Add action based on pattern type
                if pattern["type"] == "stop_hunting":
                    rec["actions"].append({
                        "action": "widen_stops",
                        "description": "Widen stop losses to avoid stop hunting",
                        "details": f"Consider widening stops by 1.5x ATR around {pattern['level_price']} level"
                    })
                    rec["actions"].append({
                        "action": "time_exit",
                        "description": "Use time-based exits instead of price-based stops",
                        "details": "Consider implementing time-based exits for short-term trades"
                    })
                
                elif pattern["type"] == "fake_breakout":
                    rec["actions"].append({
                        "action": "delayed_entry",
                        "description": "Delay breakout entries for confirmation",
                        "details": "Wait for 1-2 candle confirmation before entering breakout trades"
                    })
                    rec["actions"].append({
                        "action": "reduce_size",
                        "description": "Reduce position size on breakouts",
                        "details": "Consider reducing position size by 30-50% for breakout trades"
                    })
                
                elif pattern["type"] == "volume_anomaly":
                    rec["actions"].append({
                        "action": "volatility_adjustment",
                        "description": "Adjust for increased volatility",
                        "details": "Widen stops and take profits to account for abnormal volatility"
                    })
                
                recommendations.append(rec)
        
        # Process clusters (higher priority)
        for cluster in clusters:
            confidence = cluster.get('confidence', 0.0)
            if confidence >= self.parameters["confidence_medium_threshold"]:
                rec = {
                    "type": "protection_cluster",
                    "trigger": "manipulation_cluster",
                    "confidence": confidence,
                    "pattern_count": cluster["pattern_count"],
                    "actions": [{
                        "action": "trading_pause",
                        "description": "Consider pausing trading temporarily",
                        "details": f"Multiple manipulation patterns detected with {confidence:.1%} confidence"
                    }]
                }
                
                # Add more specific actions based on cluster composition
                pattern_types = set([p["type"] for p in cluster["patterns"]])
                
                if "stop_hunting" in pattern_types:
                    rec["actions"].append({
                        "action": "alternative_stops",
                        "description": "Use mental stops instead of market stops",
                        "details": "Consider using mental stops or stop-limit orders at non-obvious levels"
                    })
                
                if "fake_breakout" in pattern_types:
                    rec["actions"].append({
                        "action": "fade_breakouts",
                        "description": "Consider fading weak breakouts",
                        "details": "Look for opportunities to fade breakouts with poor follow-through"
                    })
                
                recommendations.append(rec)
        
        return recommendations

    def analyze(self, data: Dict[str, Any]) -> AnalysisResult:
        """
        Analyze market data for potential manipulation patterns
        
        Args:
            data: Dictionary with OHLCV data and metadata
            
        Returns:
            AnalysisResult with detected patterns
        """
        if not data or "ohlcv" not in data:
            return AnalysisResult(
                analyzer_name=self.name,
                result_data={"error": "No OHLCV data provided"},
                is_valid=False
            )
            
        # Extract OHLCV data
        ohlcv_data = pd.DataFrame(data["ohlcv"])
        
        if len(ohlcv_data) < self.parameters["min_samples"]:
            return AnalysisResult(
                analyzer_name=self.name,
                result_data={
                    "warning": f"Insufficient data points ({len(ohlcv_data)}) for reliable manipulation detection",
                    "detected_patterns": [],
                    "protection_recommendations": []
                },
                is_valid=True
            )
            
        # Preprocess data
        preprocessed_data = self._preprocess_data(ohlcv_data)
        
        # Identify support/resistance levels
        support_resistance_levels = self._identify_support_resistance_levels(preprocessed_data)
        
        # Detect different types of manipulation patterns
        volume_anomalies = self._detect_volume_anomalies(preprocessed_data)
        stop_hunting_patterns = self._detect_stop_hunting(preprocessed_data, support_resistance_levels)
        fake_breakout_patterns = self._detect_fake_breakouts(preprocessed_data, support_resistance_levels)
        
        # Combine all patterns
        all_patterns = volume_anomalies + stop_hunting_patterns + fake_breakout_patterns
        
        # Sort patterns by confidence
        all_patterns.sort(key=lambda x: x.get('confidence', 0), reverse=True)
        
        # Identify clusters of manipulation patterns
        manipulation_clusters = self._identify_manipulation_clusters(all_patterns)
        
        # Generate protection recommendations
        protection_recommendations = self._generate_protection_recommendations(all_patterns, manipulation_clusters)
        
        # Compile results
        result = {
            "detected_patterns": {
                "volume_anomalies": volume_anomalies,
                "stop_hunting": stop_hunting_patterns,
                "fake_breakouts": fake_breakout_patterns
            },
            "pattern_count": {
                "volume_anomalies": len(volume_anomalies),
                "stop_hunting": len(stop_hunting_patterns),
                "fake_breakouts": len(fake_breakout_patterns),
                "total": len(all_patterns)
            },
            "support_resistance": {
                "levels": support_resistance_levels,
                "count": len(support_resistance_levels)
            },
            "manipulation_clusters": manipulation_clusters,
            "protection_recommendations": protection_recommendations,
            "manipulation_likelihood": self._calculate_overall_manipulation_likelihood(all_patterns, manipulation_clusters)
        }
        
        return AnalysisResult(
            analyzer_name=self.name,
            result_data=result,
            is_valid=True
        )
        
    def _calculate_overall_manipulation_likelihood(
        self, 
        patterns: List[Dict[str, Any]], 
        clusters: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Calculate overall likelihood of market manipulation
        
        Args:
            patterns: List of detected patterns
            clusters: List of pattern clusters
            
        Returns:
            Dictionary with manipulation likelihood assessment
        """
        if not patterns:
            return {
                "level": "low",
                "score": 0.1,
                "explanation": "No manipulation patterns detected"
            }
            
        # Calculate base score from individual patterns
        pattern_scores = [p.get('confidence', 0.5) for p in patterns]
        base_score = sum(pattern_scores) / max(10, len(patterns))  # Normalize by dividing by max(10, count)
        
        # Amplify score based on clusters
        cluster_bonus = sum([c.get('confidence', 0.5) * 0.2 for c in clusters])
        
        # Calculate final score
        final_score = min(base_score + cluster_bonus, 1.0)
        
        # Determine level
        level = "low"
        if final_score >= 0.7:
            level = "high"
        elif final_score >= 0.4:
            level = "medium"
            
        # Generate explanation
        if level == "high":
            explanation = f"High likelihood of manipulation with {len(patterns)} suspicious patterns" + \
                          (f" and {len(clusters)} pattern clusters" if clusters else "")
        elif level == "medium":
            explanation = f"Some evidence of potential manipulation with {len(patterns)} suspicious patterns"
        else:
            explanation = f"Limited evidence of manipulation with {len(patterns)} isolated patterns"
            
        return {
            "level": level,
            "score": final_score,
            "explanation": explanation
        }
