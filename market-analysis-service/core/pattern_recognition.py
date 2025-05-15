"""
Pattern Recognition module for Market Analysis Service.

This module provides algorithms for recognizing chart patterns in market data.
"""
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum
from datetime import datetime
from market_analysis_service.models.market_analysis_models import PatternType

logger = logging.getLogger(__name__)

class PatternRecognizer:
    """
    Class for recognizing chart patterns in market data.
    """
    
    def __init__(self):
        """
        Initialize the Pattern Recognizer.
        """
        self.available_patterns = self._get_available_patterns()
        
    def _get_available_patterns(self) -> List[Dict[str, Any]]:
        """
        Get available chart patterns for recognition.
        
        Returns:
            List of available patterns
        """
        patterns = []
        
        for pattern_type in PatternType:
            pattern_info = {
                "id": pattern_type.value,
                "name": pattern_type.name,
                "description": self._get_pattern_description(pattern_type),
                "min_bars": self._get_pattern_min_bars(pattern_type)
            }
            
            patterns.append(pattern_info)
            
        return patterns
        
    def _get_pattern_description(self, pattern_type: PatternType) -> str:
        """
        Get description for a pattern type.
        
        Args:
            pattern_type: Pattern type
            
        Returns:
            Pattern description
        """
        descriptions = {
            PatternType.HEAD_AND_SHOULDERS: "A reversal pattern with three peaks, the middle one being the highest",
            PatternType.INVERSE_HEAD_AND_SHOULDERS: "A reversal pattern with three troughs, the middle one being the lowest",
            PatternType.DOUBLE_TOP: "A reversal pattern with two peaks at approximately the same level",
            PatternType.DOUBLE_BOTTOM: "A reversal pattern with two troughs at approximately the same level",
            PatternType.TRIPLE_TOP: "A reversal pattern with three peaks at approximately the same level",
            PatternType.TRIPLE_BOTTOM: "A reversal pattern with three troughs at approximately the same level",
            PatternType.ASCENDING_TRIANGLE: "A continuation pattern with a flat top and rising bottom",
            PatternType.DESCENDING_TRIANGLE: "A continuation pattern with a flat bottom and falling top",
            PatternType.SYMMETRICAL_TRIANGLE: "A continuation pattern with converging trendlines",
            PatternType.FLAG: "A continuation pattern that appears as a small channel in the opposite direction of the trend",
            PatternType.PENNANT: "A continuation pattern similar to a symmetrical triangle but smaller",
            PatternType.WEDGE: "A pattern with converging trendlines, both moving in the same direction",
            PatternType.RECTANGLE: "A continuation pattern with horizontal support and resistance lines",
            PatternType.CUP_AND_HANDLE: "A continuation pattern resembling a cup with a handle",
            PatternType.CUSTOM: "Custom pattern defined by user parameters"
        }
        
        return descriptions.get(pattern_type, "Unknown pattern")
        
    def _get_pattern_min_bars(self, pattern_type: PatternType) -> int:
        """
        Get minimum number of bars required for a pattern type.
        
        Args:
            pattern_type: Pattern type
            
        Returns:
            Minimum number of bars
        """
        min_bars = {
            PatternType.HEAD_AND_SHOULDERS: 30,
            PatternType.INVERSE_HEAD_AND_SHOULDERS: 30,
            PatternType.DOUBLE_TOP: 20,
            PatternType.DOUBLE_BOTTOM: 20,
            PatternType.TRIPLE_TOP: 30,
            PatternType.TRIPLE_BOTTOM: 30,
            PatternType.ASCENDING_TRIANGLE: 15,
            PatternType.DESCENDING_TRIANGLE: 15,
            PatternType.SYMMETRICAL_TRIANGLE: 15,
            PatternType.FLAG: 10,
            PatternType.PENNANT: 10,
            PatternType.WEDGE: 15,
            PatternType.RECTANGLE: 15,
            PatternType.CUP_AND_HANDLE: 30,
            PatternType.CUSTOM: 10
        }
        
        return min_bars.get(pattern_type, 20)
        
    def recognize_patterns(
        self,
        data: pd.DataFrame,
        pattern_types: Optional[List[PatternType]] = None,
        min_confidence: float = 0.7
    ) -> List[Dict[str, Any]]:
        """
        Recognize chart patterns in market data.
        
        Args:
            data: Market data
            pattern_types: Types of patterns to recognize
            min_confidence: Minimum confidence level for pattern recognition
            
        Returns:
            List of recognized patterns
        """
        if pattern_types is None:
            pattern_types = list(PatternType)
            
        recognized_patterns = []
        
        for pattern_type in pattern_types:
            # Skip custom patterns if no parameters are provided
            if pattern_type == PatternType.CUSTOM:
                continue
                
            # Recognize pattern
            patterns = self._recognize_pattern(data, pattern_type, min_confidence)
            
            recognized_patterns.extend(patterns)
            
        # Sort patterns by confidence (descending)
        recognized_patterns.sort(key=lambda x: x["confidence"], reverse=True)
        
        return recognized_patterns
        
    def _recognize_pattern(
        self,
        data: pd.DataFrame,
        pattern_type: PatternType,
        min_confidence: float
    ) -> List[Dict[str, Any]]:
        """
        Recognize a specific pattern type in market data.
        
        Args:
            data: Market data
            pattern_type: Type of pattern to recognize
            min_confidence: Minimum confidence level for pattern recognition
            
        Returns:
            List of recognized patterns
        """
        recognized_patterns = []
        
        try:
            if pattern_type == PatternType.HEAD_AND_SHOULDERS:
                patterns = self._recognize_head_and_shoulders(data, min_confidence)
                recognized_patterns.extend(patterns)
                
            elif pattern_type == PatternType.INVERSE_HEAD_AND_SHOULDERS:
                patterns = self._recognize_inverse_head_and_shoulders(data, min_confidence)
                recognized_patterns.extend(patterns)
                
            elif pattern_type == PatternType.DOUBLE_TOP:
                patterns = self._recognize_double_top(data, min_confidence)
                recognized_patterns.extend(patterns)
                
            elif pattern_type == PatternType.DOUBLE_BOTTOM:
                patterns = self._recognize_double_bottom(data, min_confidence)
                recognized_patterns.extend(patterns)
                
            # Add more pattern recognition methods as needed
                
        except Exception as e:
            logger.error(f"Error recognizing {pattern_type.value} pattern: {e}")
            
        return recognized_patterns
        
    def _recognize_head_and_shoulders(
        self,
        data: pd.DataFrame,
        min_confidence: float
    ) -> List[Dict[str, Any]]:
        """
        Recognize Head and Shoulders pattern in market data.
        
        Args:
            data: Market data
            min_confidence: Minimum confidence level for pattern recognition
            
        Returns:
            List of recognized patterns
        """
        recognized_patterns = []
        
        # Ensure we have enough data
        if len(data) < 30:
            return recognized_patterns
            
        # Get high prices
        highs = data["high"].values
        
        # Find local maxima
        local_maxima_indices = self._find_local_maxima(highs, window=5)
        
        # Need at least 3 local maxima for Head and Shoulders
        if len(local_maxima_indices) < 3:
            return recognized_patterns
            
        # Check each possible combination of 3 consecutive local maxima
        for i in range(len(local_maxima_indices) - 2):
            left_shoulder_idx = local_maxima_indices[i]
            head_idx = local_maxima_indices[i + 1]
            right_shoulder_idx = local_maxima_indices[i + 2]
            
            # Check if the pattern is valid
            if self._is_valid_head_and_shoulders(highs, left_shoulder_idx, head_idx, right_shoulder_idx):
                # Calculate confidence
                confidence = self._calculate_head_and_shoulders_confidence(
                    highs, left_shoulder_idx, head_idx, right_shoulder_idx
                )
                
                if confidence >= min_confidence:
                    # Calculate target price and stop loss
                    neckline = self._calculate_head_and_shoulders_neckline(
                        data, left_shoulder_idx, head_idx, right_shoulder_idx
                    )
                    
                    pattern_height = highs[head_idx] - neckline
                    target_price = neckline - pattern_height
                    stop_loss = highs[head_idx]
                    
                    # Create pattern instance
                    pattern = {
                        "pattern_type": PatternType.HEAD_AND_SHOULDERS.value,
                        "start_index": left_shoulder_idx,
                        "end_index": right_shoulder_idx,
                        "confidence": confidence,
                        "target_price": target_price,
                        "stop_loss": stop_loss,
                        "risk_reward_ratio": (neckline - target_price) / (stop_loss - neckline) if stop_loss != neckline else 0,
                        "metadata": {
                            "left_shoulder_idx": int(left_shoulder_idx),
                            "head_idx": int(head_idx),
                            "right_shoulder_idx": int(right_shoulder_idx),
                            "neckline": float(neckline)
                        }
                    }
                    
                    recognized_patterns.append(pattern)
                    
        return recognized_patterns
        
    def _is_valid_head_and_shoulders(
        self,
        highs: np.ndarray,
        left_shoulder_idx: int,
        head_idx: int,
        right_shoulder_idx: int
    ) -> bool:
        """
        Check if a potential Head and Shoulders pattern is valid.
        
        Args:
            highs: High prices
            left_shoulder_idx: Index of left shoulder
            head_idx: Index of head
            right_shoulder_idx: Index of right shoulder
            
        Returns:
            True if the pattern is valid, False otherwise
        """
        # Head must be higher than both shoulders
        if highs[head_idx] <= highs[left_shoulder_idx] or highs[head_idx] <= highs[right_shoulder_idx]:
            return False
            
        # Shoulders should be at similar levels
        shoulder_diff = abs(highs[left_shoulder_idx] - highs[right_shoulder_idx])
        shoulder_avg = (highs[left_shoulder_idx] + highs[right_shoulder_idx]) / 2
        if shoulder_diff / shoulder_avg > 0.1:  # Shoulders should be within 10% of each other
            return False
            
        # Head should be between shoulders
        if not (left_shoulder_idx < head_idx < right_shoulder_idx):
            return False
            
        # Pattern should not be too compressed or too stretched
        if (right_shoulder_idx - left_shoulder_idx) < 10 or (right_shoulder_idx - left_shoulder_idx) > 100:
            return False
            
        return True
        
    def _calculate_head_and_shoulders_confidence(
        self,
        highs: np.ndarray,
        left_shoulder_idx: int,
        head_idx: int,
        right_shoulder_idx: int
    ) -> float:
        """
        Calculate confidence level for a Head and Shoulders pattern.
        
        Args:
            highs: High prices
            left_shoulder_idx: Index of left shoulder
            head_idx: Index of head
            right_shoulder_idx: Index of right shoulder
            
        Returns:
            Confidence level (0-1)
        """
        # Calculate confidence based on various factors
        
        # 1. Symmetry of shoulders
        shoulder_diff = abs(highs[left_shoulder_idx] - highs[right_shoulder_idx])
        shoulder_avg = (highs[left_shoulder_idx] + highs[right_shoulder_idx]) / 2
        symmetry_score = 1 - min(1, shoulder_diff / shoulder_avg)
        
        # 2. Height of head relative to shoulders
        head_height = highs[head_idx] - shoulder_avg
        relative_height_score = min(1, head_height / shoulder_avg)
        
        # 3. Spacing between shoulders
        ideal_spacing = 20  # Ideal number of bars between shoulders
        actual_spacing = right_shoulder_idx - left_shoulder_idx
        spacing_score = 1 - min(1, abs(actual_spacing - ideal_spacing) / ideal_spacing)
        
        # Combine scores with weights
        confidence = 0.5 * symmetry_score + 0.3 * relative_height_score + 0.2 * spacing_score
        
        return confidence
        
    def _calculate_head_and_shoulders_neckline(
        self,
        data: pd.DataFrame,
        left_shoulder_idx: int,
        head_idx: int,
        right_shoulder_idx: int
    ) -> float:
        """
        Calculate the neckline for a Head and Shoulders pattern.
        
        Args:
            data: Market data
            left_shoulder_idx: Index of left shoulder
            head_idx: Index of head
            right_shoulder_idx: Index of right shoulder
            
        Returns:
            Neckline price
        """
        # Find the lowest point between left shoulder and head
        left_trough_idx = left_shoulder_idx
        left_trough_price = data["low"].iloc[left_shoulder_idx]
        
        for i in range(left_shoulder_idx + 1, head_idx):
            if data["low"].iloc[i] < left_trough_price:
                left_trough_idx = i
                left_trough_price = data["low"].iloc[i]
                
        # Find the lowest point between head and right shoulder
        right_trough_idx = head_idx
        right_trough_price = data["low"].iloc[head_idx]
        
        for i in range(head_idx + 1, right_shoulder_idx):
            if data["low"].iloc[i] < right_trough_price:
                right_trough_idx = i
                right_trough_price = data["low"].iloc[i]
                
        # Calculate neckline as the line connecting the two troughs
        if right_trough_idx == left_trough_idx:
            return left_trough_price
            
        slope = (right_trough_price - left_trough_price) / (right_trough_idx - left_trough_idx)
        neckline = left_trough_price + slope * (right_shoulder_idx - left_trough_idx)
        
        return neckline
        
    def _recognize_inverse_head_and_shoulders(
        self,
        data: pd.DataFrame,
        min_confidence: float
    ) -> List[Dict[str, Any]]:
        """
        Recognize Inverse Head and Shoulders pattern in market data.
        
        Args:
            data: Market data
            min_confidence: Minimum confidence level for pattern recognition
            
        Returns:
            List of recognized patterns
        """
        recognized_patterns = []
        
        # Implementation similar to head and shoulders but using lows instead of highs
        # and inverting the logic
        
        return recognized_patterns
        
    def _recognize_double_top(
        self,
        data: pd.DataFrame,
        min_confidence: float
    ) -> List[Dict[str, Any]]:
        """
        Recognize Double Top pattern in market data.
        
        Args:
            data: Market data
            min_confidence: Minimum confidence level for pattern recognition
            
        Returns:
            List of recognized patterns
        """
        recognized_patterns = []
        
        # Implementation for double top pattern recognition
        
        return recognized_patterns
        
    def _recognize_double_bottom(
        self,
        data: pd.DataFrame,
        min_confidence: float
    ) -> List[Dict[str, Any]]:
        """
        Recognize Double Bottom pattern in market data.
        
        Args:
            data: Market data
            min_confidence: Minimum confidence level for pattern recognition
            
        Returns:
            List of recognized patterns
        """
        recognized_patterns = []
        
        # Implementation for double bottom pattern recognition
        
        return recognized_patterns
        
    def _find_local_maxima(
        self,
        data: np.ndarray,
        window: int = 5
    ) -> List[int]:
        """
        Find local maxima in data.
        
        Args:
            data: Data to find local maxima in
            window: Window size for local maxima detection
            
        Returns:
            Indices of local maxima
        """
        local_maxima = []
        
        for i in range(window, len(data) - window):
            is_local_max = True
            
            for j in range(1, window + 1):
                if data[i] <= data[i - j] or data[i] <= data[i + j]:
                    is_local_max = False
                    break
                    
            if is_local_max:
                local_maxima.append(i)
                
        return local_maxima
        
    def _find_local_minima(
        self,
        data: np.ndarray,
        window: int = 5
    ) -> List[int]:
        """
        Find local minima in data.
        
        Args:
            data: Data to find local minima in
            window: Window size for local minima detection
            
        Returns:
            Indices of local minima
        """
        local_minima = []
        
        for i in range(window, len(data) - window):
            is_local_min = True
            
            for j in range(1, window + 1):
                if data[i] >= data[i - j] or data[i] >= data[i + j]:
                    is_local_min = False
                    break
                    
            if is_local_min:
                local_minima.append(i)
                
        return local_minima