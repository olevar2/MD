"""
Advanced Signal Classification System

This module extends the central signal system with specialized
classification capabilities for technical indicator signals.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Callable, Any
import logging
from datetime import datetime
from dataclasses import dataclass, field

from ..analysis.signal_system import Signal, SignalType


class SignalClassifier:
    """
    Base class for technical indicator signal classifiers.
    
    Signal classifiers analyze indicator outputs and identify 
    actionable signals based on specific patterns or conditions.
    """
    
    def __init__(self, name: str):
        """
        Initialize signal classifier.
        
        Args:
            name: Classifier name
        """
        self.name = name
        self.logger = logging.getLogger(f"SignalClassifier.{name}")
    
    def classify(self, data: pd.DataFrame, **kwargs) -> List[Signal]:
        """
        Classify signals from indicator data.
        
        Args:
            data: DataFrame with indicator values
            **kwargs: Additional parameters
            
        Returns:
            List of identified signals
        """
        raise NotImplementedError("Subclasses must implement classify method")
    
    def get_required_columns(self) -> List[str]:
        """
        Get columns required for classification.
        
        Returns:
            List of required column names
        """
        raise NotImplementedError("Subclasses must implement get_required_columns method")


class MovingAverageCrossClassifier(SignalClassifier):
    """
    Classifier for moving average crossover signals.
    """
    
    def __init__(
        self, 
        name: str = "MA_Cross",
        fast_ma_column: str = "sma_10",
        slow_ma_column: str = "sma_50",
        price_column: str = "close",
        signal_window: int = 3  # Lookback window for confirmation
    ):
        """
        Initialize moving average cross classifier.
        
        Args:
            name: Classifier name
            fast_ma_column: Column name for fast moving average
            slow_ma_column: Column name for slow moving average
            price_column: Column name for price data
            signal_window: Window for signal confirmation
        """
        super().__init__(name)
        self.fast_ma_column = fast_ma_column
        self.slow_ma_column = slow_ma_column
        self.price_column = price_column
        self.signal_window = signal_window
    
    def get_required_columns(self) -> List[str]:
        """
        Get columns required for classification.
        
        Returns:
            List of required column names
        """
        return [self.fast_ma_column, self.slow_ma_column, self.price_column]
    
    def classify(self, data: pd.DataFrame, **kwargs) -> List[Signal]:
        """
        Identify moving average crossover signals.
        
        Args:
            data: DataFrame with indicator values
            **kwargs: Additional parameters
            
        Returns:
            List of identified signals
        """
        # Validate required columns
        required_columns = self.get_required_columns()
        for col in required_columns:
            if col not in data.columns:
                self.logger.error(f"Required column {col} not found in data")
                return []
        
        signals = []
        
        # Calculate crossovers
        cross_above = (data[self.fast_ma_column] > data[self.slow_ma_column]) & \
                      (data[self.fast_ma_column].shift(1) <= data[self.slow_ma_column].shift(1))
        
        cross_below = (data[self.fast_ma_column] < data[self.slow_ma_column]) & \
                      (data[self.fast_ma_column].shift(1) >= data[self.slow_ma_column].shift(1))
        
        # Find crossover points
        cross_above_idx = data.index[cross_above]
        cross_below_idx = data.index[cross_below]
        
        # Process buy signals (cross above)
        for idx in cross_above_idx:
            pos = data.index.get_loc(idx)
            
            # Skip if too close to the beginning for confirmation
            if pos < self.signal_window:
                continue
                
            # Calculate signal strength based on confirmation factors
            strength = self._calculate_signal_strength(data, pos, is_buy=True)
            
            signals.append(Signal(
                timestamp=idx,
                indicator_name=self.name,
                signal_type=SignalType.BUY if strength < 0.8 else SignalType.STRONG_BUY,
                strength=strength,
                price=data.loc[idx, self.price_column],
                metadata={
                    "fast_ma": data.loc[idx, self.fast_ma_column],
                    "slow_ma": data.loc[idx, self.slow_ma_column],
                    "ma_difference": data.loc[idx, self.fast_ma_column] - data.loc[idx, self.slow_ma_column]
                }
            ))
        
        # Process sell signals (cross below)
        for idx in cross_below_idx:
            pos = data.index.get_loc(idx)
            
            # Skip if too close to the beginning for confirmation
            if pos < self.signal_window:
                continue
                
            # Calculate signal strength based on confirmation factors
            strength = self._calculate_signal_strength(data, pos, is_buy=False)
            
            signals.append(Signal(
                timestamp=idx,
                indicator_name=self.name,
                signal_type=SignalType.SELL if strength < 0.8 else SignalType.STRONG_SELL,
                strength=strength,
                price=data.loc[idx, self.price_column],
                metadata={
                    "fast_ma": data.loc[idx, self.fast_ma_column],
                    "slow_ma": data.loc[idx, self.slow_ma_column],
                    "ma_difference": data.loc[idx, self.fast_ma_column] - data.loc[idx, self.slow_ma_column]
                }
            ))
        
        return signals
    
    def _calculate_signal_strength(self, data: pd.DataFrame, position: int, is_buy: bool) -> float:
        """
        Calculate signal strength based on multiple confirmation factors.
        
        Args:
            data: DataFrame with indicator values
            position: Position in DataFrame to calculate strength for
            is_buy: Whether this is a buy signal
            
        Returns:
            Signal strength (0.0 to 1.0)
        """
        # Get relevant data slices
        idx = data.index[position]
        lookback_slice = data.iloc[max(0, position - self.signal_window):position + 1]
        
        # Factor 1: Magnitude of crossover
        fast_ma = data.loc[idx, self.fast_ma_column]
        slow_ma = data.loc[idx, self.slow_ma_column]
        price = data.loc[idx, self.price_column]
        
        # Normalize the difference relative to price
        normalized_diff = abs(fast_ma - slow_ma) / price
        magnitude_factor = min(1.0, normalized_diff * 100)  # Scale appropriately
        
        # Factor 2: Price momentum
        price_change = (price - lookback_slice[self.price_column].iloc[0]) / lookback_slice[self.price_column].iloc[0]
        momentum_factor = min(1.0, abs(price_change) * 10)  # Scale appropriately
        
        # Adjust momentum based on direction
        if (is_buy and price_change < 0) or (not is_buy and price_change > 0):
            momentum_factor *= 0.5  # Reduce strength if price is moving contrary to signal
        
        # Factor 3: Volatility consideration
        if "atr_14" in data.columns:
            atr = data.loc[idx, "atr_14"]
            volatility_factor = min(1.0, atr / price * 100)  # Higher volatility = stronger signals
        else:
            volatility_factor = 0.5  # Default if ATR not available
        
        # Combine factors with weights
        strength = (0.5 * magnitude_factor + 0.3 * momentum_factor + 0.2 * volatility_factor)
        
        return min(1.0, max(0.1, strength))  # Ensure range is 0.1 to 1.0


class RSIClassifier(SignalClassifier):
    """
    Classifier for RSI signals.
    """
    
    def __init__(
        self, 
        name: str = "RSI",
        rsi_column: str = "rsi_14",
        price_column: str = "close",
        overbought_level: float = 70.0,
        oversold_level: float = 30.0,
        signal_window: int = 3  # Lookback window for confirmation
    ):
        """
        Initialize RSI classifier.
        
        Args:
            name: Classifier name
            rsi_column: Column name for RSI values
            price_column: Column name for price data
            overbought_level: RSI level for overbought condition
            oversold_level: RSI level for oversold condition
            signal_window: Window for signal confirmation
        """
        super().__init__(name)
        self.rsi_column = rsi_column
        self.price_column = price_column
        self.overbought_level = overbought_level
        self.oversold_level = oversold_level
        self.signal_window = signal_window
    
    def get_required_columns(self) -> List[str]:
        """
        Get columns required for classification.
        
        Returns:
            List of required column names
        """
        return [self.rsi_column, self.price_column]
    
    def classify(self, data: pd.DataFrame, **kwargs) -> List[Signal]:
        """
        Identify RSI signals.
        
        Args:
            data: DataFrame with indicator values
            **kwargs: Additional parameters
            
        Returns:
            List of identified signals
        """
        # Validate required columns
        required_columns = self.get_required_columns()
        for col in required_columns:
            if col not in data.columns:
                self.logger.error(f"Required column {col} not found in data")
                return []
        
        signals = []
        
        # Detect oversold conditions (buy signals)
        oversold_exit = (data[self.rsi_column] > self.oversold_level) & \
                        (data[self.rsi_column].shift(1) <= self.oversold_level)
        
        # Detect overbought conditions (sell signals)
        overbought_exit = (data[self.rsi_column] < self.overbought_level) & \
                          (data[self.rsi_column].shift(1) >= self.overbought_level)
        
        # Find signal points
        oversold_idx = data.index[oversold_exit]
        overbought_idx = data.index[overbought_exit]
        
        # Process buy signals (oversold exit)
        for idx in oversold_idx:
            pos = data.index.get_loc(idx)
            
            # Skip if too close to the beginning for confirmation
            if pos < self.signal_window:
                continue
                
            # Calculate signal strength based on how deeply oversold it was
            min_rsi = data[self.rsi_column].iloc[max(0, pos-self.signal_window):pos+1].min()
            
            # More oversold = stronger signal
            strength = (self.oversold_level - min_rsi) / self.oversold_level
            strength = min(1.0, max(0.1, strength))
            
            signals.append(Signal(
                timestamp=idx,
                indicator_name=self.name,
                signal_type=SignalType.BUY if strength < 0.8 else SignalType.STRONG_BUY,
                strength=strength,
                price=data.loc[idx, self.price_column],
                metadata={
                    "rsi_value": data.loc[idx, self.rsi_column],
                    "min_rsi": min_rsi,
                    "oversold_level": self.oversold_level
                }
            ))
        
        # Process sell signals (overbought exit)
        for idx in overbought_idx:
            pos = data.index.get_loc(idx)
            
            # Skip if too close to the beginning for confirmation
            if pos < self.signal_window:
                continue
                
            # Calculate signal strength based on how deeply overbought it was
            max_rsi = data[self.rsi_column].iloc[max(0, pos-self.signal_window):pos+1].max()
            
            # More overbought = stronger signal
            strength = (max_rsi - self.overbought_level) / (100 - self.overbought_level)
            strength = min(1.0, max(0.1, strength))
            
            signals.append(Signal(
                timestamp=idx,
                indicator_name=self.name,
                signal_type=SignalType.SELL if strength < 0.8 else SignalType.STRONG_SELL,
                strength=strength,
                price=data.loc[idx, self.price_column],
                metadata={
                    "rsi_value": data.loc[idx, self.rsi_column],
                    "max_rsi": max_rsi,
                    "overbought_level": self.overbought_level
                }
            ))
        
        return signals


@dataclass
class SignalGroup:
    """
    Represents a group of related signals within a time window.
    Used for concordance analysis between multiple indicators.
    """
    start_time: datetime
    end_time: datetime
    signals: List[Signal] = field(default_factory=list)
    primary_signal_type: Optional[SignalType] = None
    concordance_score: float = 0.0
    conflict_level: float = 0.0
    
    def add_signal(self, signal: Signal) -> None:
        """
        Add a signal to the group.
        
        Args:
            signal: Signal to add
        """
        self.signals.append(signal)
        
        # Update primary signal type if needed
        if not self.primary_signal_type:
            self.primary_signal_type = signal.signal_type
    
    @property
    def is_valid(self) -> bool:
        """
        Check if the signal group is valid (has signals).
        
        Returns:
            Whether the group is valid
        """
        return len(self.signals) > 0


class SignalConcordanceAnalyzer:
    """
    Analyzes agreement between signals from different indicators.
    """
    
    def __init__(
        self, 
        time_window_seconds: int = 3600,  # 1 hour default window
        min_signals_for_concordance: int = 2
    ):
        """
        Initialize concordance analyzer.
        
        Args:
            time_window_seconds: Window size in seconds for grouping signals
            min_signals_for_concordance: Minimum number of signals for valid concordance
        """
        self.time_window_seconds = time_window_seconds
        self.min_signals_for_concordance = min_signals_for_concordance
        self.logger = logging.getLogger("SignalConcordanceAnalyzer")
        
    def group_signals_by_time(self, signals: List[Signal]) -> List[SignalGroup]:
        """
        Group signals into time windows.
        
        Args:
            signals: List of signals to group
            
        Returns:
            List of signal groups
        """
        if not signals:
            return []
            
        # Sort signals by timestamp
        sorted_signals = sorted(signals, key=lambda s: s.timestamp)
        
        groups = []
        current_group = None
        
        for signal in sorted_signals:
            # If no current group, create one
            if current_group is None:
                current_group = SignalGroup(
                    start_time=signal.timestamp,
                    end_time=signal.timestamp,
                )
                current_group.add_signal(signal)
                continue
            
            # Check if signal is within time window
            time_diff = (signal.timestamp - current_group.end_time).total_seconds()
            
            if time_diff <= self.time_window_seconds:
                # Add to current group and extend end time
                current_group.add_signal(signal)
                current_group.end_time = max(current_group.end_time, signal.timestamp)
            else:
                # Close current group and start a new one
                groups.append(current_group)
                current_group = SignalGroup(
                    start_time=signal.timestamp,
                    end_time=signal.timestamp,
                )
                current_group.add_signal(signal)
        
        # Add last group if exists
        if current_group and current_group.is_valid:
            groups.append(current_group)
        
        return groups
        
    def analyze_concordance(self, signals: List[Signal]) -> List[SignalGroup]:
        """
        Analyze concordance between signals from different indicators.
        
        Args:
            signals: List of signals to analyze
            
        Returns:
            List of signal groups with concordance scores
        """
        # Group signals by time
        groups = self.group_signals_by_time(signals)
        
        # Calculate concordance for each group
        for group in groups:
            # Skip groups with too few signals
            if len(group.signals) < self.min_signals_for_concordance:
                continue
                
            group.concordance_score = self._calculate_concordance_score(group)
            group.conflict_level = self._calculate_conflict_level(group)
            
        return groups
        
    def _calculate_concordance_score(self, group: SignalGroup) -> float:
        """
        Calculate concordance score for a signal group.
        
        Args:
            group: Signal group to analyze
            
        Returns:
            Concordance score (0.0 to 1.0)
        """
        if len(group.signals) < self.min_signals_for_concordance:
            return 0.0
            
        # Count signal types
        signal_type_counts = {}
        
        for signal in group.signals:
            # Normalize signal type (combine strong and regular)
            normalized_type = SignalType.BUY if signal.signal_type in [SignalType.BUY, SignalType.STRONG_BUY] else \
                             SignalType.SELL if signal.signal_type in [SignalType.SELL, SignalType.STRONG_SELL] else \
                             SignalType.NEUTRAL
            
            # Count by indicator to avoid bias from multiple signals from same indicator
            key = (signal.indicator_name, normalized_type)
            
            if key not in signal_type_counts:
                signal_type_counts[key] = signal.strength
        
        # Group by signal type
        type_strengths = {}
        for (_, sig_type), strength in signal_type_counts.items():
            if sig_type not in type_strengths:
                type_strengths[sig_type] = []
            type_strengths[sig_type].append(strength)
        
        # Calculate weighted count for each type
        weighted_counts = {}
        for sig_type, strengths in type_strengths.items():
            weighted_counts[sig_type] = sum(strengths)
        
        if not weighted_counts:
            return 0.0
            
        # Find dominant signal type
        dominant_type = max(weighted_counts, key=weighted_counts.get)
        total_weight = sum(weighted_counts.values())
        
        # Concordance is the proportion of the dominant signal
        concordance = weighted_counts[dominant_type] / total_weight if total_weight > 0 else 0.0
        
        return concordance
        
    def _calculate_conflict_level(self, group: SignalGroup) -> float:
        """
        Calculate conflict level for a signal group.
        
        Args:
            group: Signal group to analyze
            
        Returns:
            Conflict level (0.0 to 1.0)
        """
        if len(group.signals) < self.min_signals_for_concordance:
            return 0.0
        
        # Group signals by type (buy vs sell)
        buy_signals = [s for s in group.signals if s.signal_type in [SignalType.BUY, SignalType.STRONG_BUY]]
        sell_signals = [s for s in group.signals if s.signal_type in [SignalType.SELL, SignalType.STRONG_SELL]]
        
        total_signals = len(buy_signals) + len(sell_signals)
        
        if total_signals < 2:
            return 0.0
            
        # Perfect conflict is when buy and sell signals are equal
        buy_ratio = len(buy_signals) / total_signals
        conflict = 1.0 - abs(buy_ratio - 0.5) * 2  # 0.5 means equal buy/sell = max conflict
        
        return conflict
