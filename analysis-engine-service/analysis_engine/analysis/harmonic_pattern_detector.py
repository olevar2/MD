"""
HarmonicPatternDetector

This module provides advanced detection of harmonic price patterns including:
- Gartley patterns
- Butterfly patterns
- Bat patterns
- Crab patterns
- Shark patterns
- Cypher patterns

Part of Phase 4 implementation to enhance the adaptive trading capabilities.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime


@dataclass
class HarmonicPoint:
    """Represents a point in a harmonic pattern with price and time information."""
    price: float
    index: int  # Bar index in the dataset
    timestamp: datetime  # Actual timestamp


class HarmonicPatternAnalyzer:
    """
    Advanced detector for identifying and validating harmonic price patterns
    using Fibonacci ratios.
    """
    
    def __init__(self, logger=None):
        """Initialize the detector with default settings."""
        self.logger = logger or logging.getLogger(__name__)
        
        # Fibonacci ratios commonly used in harmonic patterns
        self.fibonacci_ratios = {
            "0.382": 0.382,
            "0.5": 0.5,
            "0.618": 0.618,
            "0.786": 0.786,
            "0.886": 0.886,
            "1.13": 1.13,
            "1.27": 1.27,
            "1.41": 1.41,
            "1.618": 1.618,
            "2.0": 2.0,
            "2.24": 2.24,
            "2.618": 2.618,
            "3.14": 3.14,
            "3.618": 3.618,
        }
        
        # Tolerance for Fibonacci ratio matching (percentage)
        self.ratio_tolerance = 0.03
        
        # Configure pattern definitions
        self.pattern_definitions = self._configure_pattern_definitions()
    
    def _configure_pattern_definitions(self) -> Dict[str, Dict[str, Tuple[float, float]]]:
        """Configure the pattern definitions with Fibonacci ratio ranges."""
        return {
            "gartley": {
                "XA_retracement": (self.fibonacci_ratios["0.618"] - self.ratio_tolerance, 
                                  self.fibonacci_ratios["0.618"] + self.ratio_tolerance),
                "AB_retracement": (self.fibonacci_ratios["0.382"] - self.ratio_tolerance, 
                                  self.fibonacci_ratios["0.886"] + self.ratio_tolerance),
                "BC_retracement": (self.fibonacci_ratios["0.382"] - self.ratio_tolerance, 
                                  self.fibonacci_ratios["0.886"] + self.ratio_tolerance),
                "CD_projection": (self.fibonacci_ratios["1.27"] - self.ratio_tolerance, 
                                 self.fibonacci_ratios["1.618"] + self.ratio_tolerance)
            },
            "butterfly": {
                "XA_retracement": (self.fibonacci_ratios["0.786"] - self.ratio_tolerance, 
                                  self.fibonacci_ratios["0.786"] + self.ratio_tolerance),
                "AB_retracement": (self.fibonacci_ratios["0.382"] - self.ratio_tolerance, 
                                  self.fibonacci_ratios["0.886"] + self.ratio_tolerance),
                "BC_retracement": (self.fibonacci_ratios["0.382"] - self.ratio_tolerance, 
                                  self.fibonacci_ratios["0.886"] + self.ratio_tolerance),
                "CD_projection": (self.fibonacci_ratios["1.618"] - self.ratio_tolerance, 
                                 self.fibonacci_ratios["2.24"] + self.ratio_tolerance)
            },
            "bat": {
                "XA_retracement": (self.fibonacci_ratios["0.382"] - self.ratio_tolerance, 
                                  self.fibonacci_ratios["0.5"] + self.ratio_tolerance),
                "AB_retracement": (self.fibonacci_ratios["0.382"] - self.ratio_tolerance, 
                                  self.fibonacci_ratios["0.618"] + self.ratio_tolerance),
                "BC_retracement": (self.fibonacci_ratios["0.382"] - self.ratio_tolerance, 
                                  self.fibonacci_ratios["0.886"] + self.ratio_tolerance),
                "CD_projection": (self.fibonacci_ratios["1.618"] - self.ratio_tolerance, 
                                 self.fibonacci_ratios["2.618"] + self.ratio_tolerance)
            },
            "crab": {
                "XA_retracement": (self.fibonacci_ratios["0.382"] - self.ratio_tolerance, 
                                  self.fibonacci_ratios["0.618"] + self.ratio_tolerance),
                "AB_retracement": (self.fibonacci_ratios["0.382"] - self.ratio_tolerance, 
                                  self.fibonacci_ratios["0.618"] + self.ratio_tolerance),
                "BC_retracement": (self.fibonacci_ratios["0.382"] - self.ratio_tolerance, 
                                  self.fibonacci_ratios["0.886"] + self.ratio_tolerance),
                "CD_projection": (self.fibonacci_ratios["2.24"] - self.ratio_tolerance, 
                                 self.fibonacci_ratios["3.618"] + self.ratio_tolerance)
            },
            "shark": {
                "XA_retracement": (self.fibonacci_ratios["0.5"] - self.ratio_tolerance, 
                                  self.fibonacci_ratios["0.886"] + self.ratio_tolerance),
                "AB_retracement": (self.fibonacci_ratios["1.13"] - self.ratio_tolerance, 
                                  self.fibonacci_ratios["1.618"] + self.ratio_tolerance),
                "BC_retracement": (self.fibonacci_ratios["0.886"] - self.ratio_tolerance, 
                                  self.fibonacci_ratios["1.13"] + self.ratio_tolerance),
                "CD_projection": (self.fibonacci_ratios["1.618"] - self.ratio_tolerance, 
                                 self.fibonacci_ratios["2.24"] + self.ratio_tolerance)
            },
            "cypher": {
                "XA_retracement": (self.fibonacci_ratios["0.382"] - self.ratio_tolerance, 
                                  self.fibonacci_ratios["0.618"] + self.ratio_tolerance),
                "AB_retracement": (self.fibonacci_ratios["0.382"] - self.ratio_tolerance, 
                                  self.fibonacci_ratios["0.618"] + self.ratio_tolerance),
                "BC_retracement": (self.fibonacci_ratios["1.13"] - self.ratio_tolerance, 
                                  self.fibonacci_ratios["1.41"] + self.ratio_tolerance),
                "CD_projection": (self.fibonacci_ratios["0.786"] - self.ratio_tolerance, 
                                 self.fibonacci_ratios["0.786"] + self.ratio_tolerance)
            }
        }
    
    def detect_harmonic_patterns(self, price_data: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Detect harmonic patterns in the given price data.
        
        Args:
            price_data: DataFrame with OHLC price data
            
        Returns:
            List of detected harmonic patterns with their properties
        """
        try:
            self.logger.info(f"Detecting harmonic patterns in data with {len(price_data)} bars")
            
            # Find swing highs and lows
            swing_points = self._find_swing_points(price_data)
            
            # Generate potential pattern combinations
            potential_patterns = self._generate_pattern_combinations(swing_points, price_data)
            
            # Validate patterns using Fibonacci ratios
            validated_patterns = self._validate_patterns(potential_patterns, price_data)
            
            # Add visualization data and additional details
            enhanced_patterns = self._enhance_patterns_with_metadata(validated_patterns, price_data)
            
            self.logger.info(f"Found {len(enhanced_patterns)} valid harmonic patterns")
            return enhanced_patterns
            
        except Exception as e:
            self.logger.error(f"Error detecting harmonic patterns: {str(e)}", exc_info=True)
            return []
    
    def _find_swing_points(self, price_data: pd.DataFrame, 
                          window: int = 5) -> Dict[str, List[Tuple[int, float]]]:
        """
        Find swing highs and lows in the price data.
        
        Args:
            price_data: DataFrame with OHLC price data
            window: Number of bars to consider for swing point detection
            
        Returns:
            Dictionary with swing highs and lows as lists of (index, price) tuples
        """
        highs = []
        lows = []
        
        # Need at least 2*window+1 bars
        if len(price_data) < 2 * window + 1:
            return {"highs": highs, "lows": lows}
        
        # Find swing highs
        for i in range(window, len(price_data) - window):
            is_swing_high = True
            for j in range(1, window + 1):
                if (price_data['high'].iloc[i] <= price_data['high'].iloc[i-j] or 
                    price_data['high'].iloc[i] <= price_data['high'].iloc[i+j]):
                    is_swing_high = False
                    break
            
            if is_swing_high:
                timestamp = price_data.index[i]
                highs.append((i, price_data['high'].iloc[i], timestamp))
        
        # Find swing lows
        for i in range(window, len(price_data) - window):
            is_swing_low = True
            for j in range(1, window + 1):
                if (price_data['low'].iloc[i] >= price_data['low'].iloc[i-j] or 
                    price_data['low'].iloc[i] >= price_data['low'].iloc[i+j]):
                    is_swing_low = False
                    break
            
            if is_swing_low:
                timestamp = price_data.index[i]
                lows.append((i, price_data['low'].iloc[i], timestamp))
        
        return {"highs": highs, "lows": lows}
    
    def _generate_pattern_combinations(
        self, 
        swing_points: Dict[str, List[Tuple[int, float, datetime]]], 
        price_data: pd.DataFrame
    ) -> List[Dict[str, Any]]:
        """
        Generate potential XABCD pattern combinations from swing points.
        
        Args:
            swing_points: Dictionary with swing highs and lows
            price_data: DataFrame with OHLC price data
            
        Returns:
            List of potential patterns with their points
        """
        potential_patterns = []
        
        # Combine swing highs and lows
        all_swings = [(idx, price, timestamp, "high") for idx, price, timestamp in swing_points["highs"]]
        all_swings.extend([(idx, price, timestamp, "low") for idx, price, timestamp in swing_points["lows"]])
        
        # Sort by bar index
        all_swings.sort(key=lambda x: x[0])
        
        # Need at least 5 swing points for XABCD
        if len(all_swings) < 5:
            return potential_patterns
        
        # Generate potential patterns (both bullish and bearish)
        for i in range(len(all_swings) - 4):
            # Bullish pattern: X(low)-A(high)-B(low)-C(high)-D(low)
            if (all_swings[i][3] == "low" and all_swings[i+1][3] == "high" and
                all_swings[i+2][3] == "low" and all_swings[i+3][3] == "high" and
                all_swings[i+4][3] == "low"):
                
                # Create bullish pattern
                bullish = {
                    "type": "unknown",  # Will be determined during validation
                    "direction": "bullish",
                    "points": {
                        "X": {"index": all_swings[i][0], "price": all_swings[i][1], "timestamp": all_swings[i][2]},
                        "A": {"index": all_swings[i+1][0], "price": all_swings[i+1][1], "timestamp": all_swings[i+1][2]},
                        "B": {"index": all_swings[i+2][0], "price": all_swings[i+2][1], "timestamp": all_swings[i+2][2]},
                        "C": {"index": all_swings[i+3][0], "price": all_swings[i+3][1], "timestamp": all_swings[i+3][2]},
                        "D": {"index": all_swings[i+4][0], "price": all_swings[i+4][1], "timestamp": all_swings[i+4][2]}
                    },
                    "completion_percentage": self._calculate_completion_percentage(
                        all_swings[i+4][0], all_swings[i+3][0], len(price_data)
                    )
                }
                potential_patterns.append(bullish)
            
            # Bearish pattern: X(high)-A(low)-B(high)-C(low)-D(high)
            elif (all_swings[i][3] == "high" and all_swings[i+1][3] == "low" and
                all_swings[i+2][3] == "high" and all_swings[i+3][3] == "low" and
                all_swings[i+4][3] == "high"):
                
                # Create bearish pattern
                bearish = {
                    "type": "unknown",  # Will be determined during validation
                    "direction": "bearish",
                    "points": {
                        "X": {"index": all_swings[i][0], "price": all_swings[i][1], "timestamp": all_swings[i][2]},
                        "A": {"index": all_swings[i+1][0], "price": all_swings[i+1][1], "timestamp": all_swings[i+1][2]},
                        "B": {"index": all_swings[i+2][0], "price": all_swings[i+2][1], "timestamp": all_swings[i+2][2]},
                        "C": {"index": all_swings[i+3][0], "price": all_swings[i+3][1], "timestamp": all_swings[i+3][2]},
                        "D": {"index": all_swings[i+4][0], "price": all_swings[i+4][1], "timestamp": all_swings[i+4][2]}
                    },
                    "completion_percentage": self._calculate_completion_percentage(
                        all_swings[i+4][0], all_swings[i+3][0], len(price_data)
                    )
                }
                potential_patterns.append(bearish)
        
        return potential_patterns
    
    def _calculate_completion_percentage(self, d_index: int, c_index: int, data_length: int) -> float:
        """
        Calculate how complete the pattern is, based on D point formation.
        
        Args:
            d_index: Index of D point
            c_index: Index of C point
            data_length: Total length of the price data
            
        Returns:
            Completion percentage (0.0-1.0)
        """
        # If D point is the last bar in the data, pattern is still forming
        if d_index >= data_length - 1:
            # Calculate progress from C to where D should be (using historical average CD leg length)
            avg_leg_length = 10  # This is a simplification; real implementation would calculate this
            expected_d_index = c_index + avg_leg_length
            
            # Calculate completion based on current position vs expected
            current_progress = data_length - 1 - c_index
            return min(1.0, current_progress / (expected_d_index - c_index))
        
        # If D point already exists, pattern is 100% complete
        return 1.0
    
    def _validate_patterns(
        self, 
        potential_patterns: List[Dict[str, Any]], 
        price_data: pd.DataFrame
    ) -> List[Dict[str, Any]]:
        """
        Validate potential patterns using Fibonacci ratio criteria.
        
        Args:
            potential_patterns: List of potential harmonic patterns
            price_data: DataFrame with OHLC price data
            
        Returns:
            List of validated patterns with pattern type and confidence score
        """
        validated_patterns = []
        
        for pattern in potential_patterns:
            points = pattern["points"]
            
            # Extract prices for calculations
            x_price = points["X"]["price"]
            a_price = points["A"]["price"]
            b_price = points["B"]["price"]
            c_price = points["C"]["price"]
            d_price = points["D"]["price"]
            
            # Calculate leg lengths
            xa_length = abs(a_price - x_price)
            ab_length = abs(b_price - a_price)
            bc_length = abs(c_price - b_price)
            cd_length = abs(d_price - c_price)
            
            # Calculate retracements and projections
            ab_xa_ratio = ab_length / xa_length if xa_length else 0
            bc_ab_ratio = bc_length / ab_length if ab_length else 0
            cd_bc_ratio = cd_length / bc_length if bc_length else 0
            xb_xa_ratio = abs(b_price - x_price) / xa_length if xa_length else 0
            ad_xa_ratio = abs(d_price - a_price) / xa_length if xa_length else 0
            
            # Calculate the best matching pattern based on Fibonacci ratios
            pattern_scores = {}
            
            for pattern_type, criteria in self.pattern_definitions.items():
                # Check each criterion against the calculated ratios
                matches = {
                    "XA_retracement": criteria["XA_retracement"][0] <= ab_xa_ratio <= criteria["XA_retracement"][1],
                    "AB_retracement": criteria["AB_retracement"][0] <= bc_ab_ratio <= criteria["AB_retracement"][1],
                    "BC_retracement": criteria["BC_retracement"][0] <= cd_bc_ratio <= criteria["BC_retracement"][1]
                }
                
                # Calculate how many criteria match
                match_count = sum(1 for match in matches.values() if match)
                match_ratio = match_count / len(matches)
                
                # Calculate distance from ideal ratios
                xab_distance = self._calculate_ratio_distance(ab_xa_ratio, sum(criteria["XA_retracement"]) / 2)
                abc_distance = self._calculate_ratio_distance(bc_ab_ratio, sum(criteria["AB_retracement"]) / 2)
                bcd_distance = self._calculate_ratio_distance(cd_bc_ratio, sum(criteria["BC_retracement"]) / 2)
                
                # Overall score (high is better)
                accuracy_score = 1.0 - (xab_distance + abc_distance + bcd_distance) / 3
                
                # Store the score for this pattern type
                pattern_scores[pattern_type] = {
                    "match_ratio": match_ratio,
                    "accuracy_score": accuracy_score,
                    "overall_score": match_ratio * accuracy_score,
                    "matches": matches
                }
            
            # Find the best matching pattern type
            best_pattern_type = max(pattern_scores.items(), key=lambda x: x[1]["overall_score"])
            pattern_type = best_pattern_type[0]
            score = best_pattern_type[1]["overall_score"]
            
            # Only include patterns with reasonable match score
            if score >= 0.6:  # 60% confidence threshold
                validated_pattern = {
                    **pattern,
                    "type": pattern_type,
                    "confidence_score": score,
                    "ratios": {
                        "AB/XA": ab_xa_ratio,
                        "BC/AB": bc_ab_ratio,
                        "CD/BC": cd_bc_ratio,
                        "XB/XA": xb_xa_ratio,
                        "AD/XA": ad_xa_ratio
                    },
                    "match_details": pattern_scores[pattern_type]["matches"]
                }
                validated_patterns.append(validated_pattern)
        
        # Sort by confidence score
        return sorted(validated_patterns, key=lambda x: x["confidence_score"], reverse=True)
    
    def _calculate_ratio_distance(self, actual: float, ideal: float) -> float:
        """Calculate how far an actual ratio is from the ideal ratio."""
        return abs(actual - ideal) / ideal if ideal else 1.0
    
    def _enhance_patterns_with_metadata(
        self, 
        patterns: List[Dict[str, Any]], 
        price_data: pd.DataFrame
    ) -> List[Dict[str, Any]]:
        """
        Enhance patterns with additional metadata for visualization and analysis.
        
        Args:
            patterns: List of validated harmonic patterns
            price_data: DataFrame with OHLC price data
            
        Returns:
            List of patterns with enhanced metadata
        """
        current_price = price_data["close"].iloc[-1]
        
        enhanced_patterns = []
        for pattern in patterns:
            # Calculate pattern age in bars
            pattern_age = len(price_data) - 1 - pattern["points"]["X"]["index"]
            
            # Determine if pattern is still active
            is_active = pattern["completion_percentage"] < 1.0 or pattern_age < 20  # Active if forming or recent
            
            # Calculate potential take profit and stop loss
            d_price = pattern["points"]["D"]["price"]
            x_price = pattern["points"]["X"]["price"]
            a_price = pattern["points"]["A"]["price"]
            c_price = pattern["points"]["C"]["price"]
            
            # Stop loss is typically just beyond the D point
            if pattern["direction"] == "bullish":
                stop_loss = d_price * 0.99  # 1% below D for bullish
                take_profit_near = a_price  # First target is near A point
                take_profit_far = x_price  # Extended target is near X point
            else:  # bearish
                stop_loss = d_price * 1.01  # 1% above D for bearish
                take_profit_near = a_price  # First target is near A point
                take_profit_far = x_price  # Extended target is near X point
            
            # Calculate risk-reward ratio
            risk = abs(current_price - stop_loss)
            reward_near = abs(current_price - take_profit_near)
            reward_far = abs(current_price - take_profit_far)
            risk_reward_near = reward_near / risk if risk else 0
            risk_reward_far = reward_far / risk if risk else 0
            
            # Create visualization data for chart plotting
            visualization_data = {
                "line_segments": [
                    {"from": {"x": pattern["points"]["X"]["index"], "y": x_price}, 
                     "to": {"x": pattern["points"]["A"]["index"], "y": a_price}},
                    {"from": {"x": pattern["points"]["A"]["index"], "y": a_price}, 
                     "to": {"x": pattern["points"]["B"]["index"], "y": pattern["points"]["B"]["price"]}},
                    {"from": {"x": pattern["points"]["B"]["index"], "y": pattern["points"]["B"]["price"]}, 
                     "to": {"x": pattern["points"]["C"]["index"], "y": c_price}},
                    {"from": {"x": pattern["points"]["C"]["index"], "y": c_price}, 
                     "to": {"x": pattern["points"]["D"]["index"], "y": d_price}}
                ],
                "labels": [
                    {"x": pattern["points"]["X"]["index"], "y": x_price, "text": "X"},
                    {"x": pattern["points"]["A"]["index"], "y": a_price, "text": "A"},
                    {"x": pattern["points"]["B"]["index"], "y": pattern["points"]["B"]["price"], "text": "B"},
                    {"x": pattern["points"]["C"]["index"], "y": c_price, "text": "C"},
                    {"x": pattern["points"]["D"]["index"], "y": d_price, "text": "D"}
                ]
            }
            
            # Add the enhanced data
            enhanced_pattern = {
                **pattern,
                "age_bars": pattern_age,
                "is_active": is_active,
                "trade_data": {
                    "stop_loss": stop_loss,
                    "take_profit_near": take_profit_near,
                    "take_profit_far": take_profit_far,
                    "risk_reward_near": risk_reward_near,
                    "risk_reward_far": risk_reward_far
                },
                "visualization_data": visualization_data
            }
            
            enhanced_patterns.append(enhanced_pattern)
        
        return enhanced_patterns
"""
