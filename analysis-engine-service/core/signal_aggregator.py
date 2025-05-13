"""
Signal Aggregator

This module provides an enhanced signal aggregation system that uses adaptive weights
from the AdaptiveLayer to combine signals from multiple sources with confidence scoring.
"""

import logging
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime, timedelta
from enum import Enum
import json

from analysis_engine.services.tool_effectiveness import MarketRegime, TimeFrame


class SignalDirection(str, Enum):
    """Possible directions for trading signals"""
    LONG = "long"
    SHORT = "short"
    NEUTRAL = "neutral"
    CONFLICTED = "conflicted"


class SignalStrength(str, Enum):
    """Signal strength classifications"""
    VERY_WEAK = "very_weak"    # 0-20% confidence
    WEAK = "weak"              # 20-40% confidence
    MODERATE = "moderate"      # 40-60% confidence
    STRONG = "strong"          # 60-80% confidence
    VERY_STRONG = "very_strong"  # 80-100% confidence


class SignalTimeframe(str, Enum):
    """Signal timeframe classifications"""
    IMMEDIATE = "immediate"    # Current bar/very short term (minutes)
    SHORT_TERM = "short_term"  # Short term (hours)
    MEDIUM_TERM = "medium_term"  # Medium term (days)
    LONG_TERM = "long_term"    # Long term (weeks)


class SignalAggregator:
    """
    Enhanced signal aggregator that combines signals from multiple sources
    using adaptive weights and provides confidence scoring.
    """
    
    def __init__(self, default_threshold: float = 0.6):
        """
        Initialize the signal aggregator.
        
        Args:
            default_threshold: Default threshold for generating signals (0.0-1.0)
        """
        self.logger = logging.getLogger(__name__)
        self.default_threshold = default_threshold
        
        # Track recent signals for persistence analysis
        self.recent_signals = {}
        self.signal_history = {}
        self.max_history_items = 50  # Maximum signals to keep per instrument/timeframe
        
    def aggregate_signals(
        self,
        signals: List[Dict[str, Any]],
        weights: Dict[str, float],
        market_regime: MarketRegime,
        signal_threshold: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Aggregate multiple trading signals using adaptive weights.
        
        Args:
            signals: List of signal dictionaries from various sources
            weights: Dictionary mapping signal source IDs to their weights
            market_regime: Current market regime
            signal_threshold: Threshold for generating a signal (optional)
            
        Returns:
            Dictionary containing aggregated signal information
        """
        if not signals:
            return self._create_neutral_signal(market_regime)
            
        if signal_threshold is None:
            signal_threshold = self.default_threshold
            
        # Group signals by timeframe
        timeframe_signals = self._group_signals_by_timeframe(signals)
        
        # Prepare aggregation results by timeframe
        timeframe_results = {}
        signals_by_source = {}
        
        # Track raw values for direction calculation
        raw_long_score = 0.0
        raw_short_score = 0.0
        weighted_count = 0
        
        # Process each timeframe separately
        for timeframe, tf_signals in timeframe_signals.items():
            # Skip empty timeframes
            if not tf_signals:
                continue
                
            tf_result = self._aggregate_timeframe_signals(
                tf_signals, weights, market_regime
            )
            
            timeframe_results[timeframe] = tf_result
            
            # Collect signals by source for the conflict resolution
            for signal in tf_signals:
                source = signal.get("source_id")
                if source:
                    if source not in signals_by_source:
                        signals_by_source[source] = []
                    signals_by_source[source].append(signal)
            
            # Add to raw direction scores, weighted by timeframe importance
            tf_weight = self._get_timeframe_weight(timeframe, market_regime)
            raw_long_score += tf_result["long_score"] * tf_weight
            raw_short_score += tf_result["short_score"] * tf_weight
            weighted_count += tf_weight
            
        # Normalize raw scores
        if weighted_count > 0:
            raw_long_score /= weighted_count
            raw_short_score /= weighted_count
            
        # Resolve conflicts between signal sources
        conflict_resolution = self._resolve_signal_conflicts(signals_by_source, weights)
        
        # Calculate final direction and confidence based on all timeframes
        direction, confidence, strength = self._calculate_final_direction(
            raw_long_score, raw_short_score, signal_threshold
        )
        
        # Get strongest timeframe signal for entry/exit points
        strongest_tf = self._get_strongest_timeframe_signal(timeframe_results)
        
        # Track the signal for persistence analysis
        symbol = signals[0].get("symbol") if signals else None
        timeframe_name = signals[0].get("timeframe") if signals else None
        
        if symbol and timeframe_name:
            self._track_signal(symbol, timeframe_name, direction, confidence, market_regime)
        
        # Create result dict
        result = {
            "direction": direction,
            "confidence": confidence,
            "strength": strength,
            "generated_at": datetime.utcnow().isoformat(),
            "market_regime": market_regime.value,
            "timeframe_signals": timeframe_results,
            "conflict_resolution": conflict_resolution,
            "strongest_timeframe": strongest_tf,
            "source_count": len(signals_by_source),
            "signal_count": len(signals),
            "signal_persistence": self._calculate_signal_persistence(symbol, timeframe_name, direction)
        }
        
        return result
        
    def _group_signals_by_timeframe(self, signals: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Group signals by their timeframe.
        
        Args:
            signals: List of signal dictionaries
            
        Returns:
            Dictionary mapping timeframes to lists of signals
        """
        result = {}
        
        for signal in signals:
            # Get timeframe, default to medium term if not specified
            tf = signal.get("timeframe", SignalTimeframe.MEDIUM_TERM.value)
            
            if tf not in result:
                result[tf] = []
                
            result[tf].append(signal)
            
        return result
        
    def _aggregate_timeframe_signals(
        self,
        signals: List[Dict[str, Any]],
        weights: Dict[str, float],
        market_regime: MarketRegime
    ) -> Dict[str, Any]:
        """
        Aggregate signals for a specific timeframe.
        
        Args:
            signals: List of signals for a timeframe
            weights: Dictionary mapping signal source IDs to weights
            market_regime: Current market regime
            
        Returns:
            Dictionary with aggregated signal information
        """
        long_score = 0.0
        short_score = 0.0
        total_weight = 0.0
        
        signal_details = []
        
        for signal in signals:
            source_id = signal.get("source_id", "unknown")
            source_weight = weights.get(source_id, 1.0)
            signal_strength = signal.get("strength", 0.5)
            
            # Get raw signal direction (-1 for short, 0 for neutral, 1 for long)
            direction = signal.get("direction", "neutral").lower()
            raw_direction = 1 if direction == "long" else (-1 if direction == "short" else 0)
            
            # Apply weight to signal
            weighted_strength = signal_strength * source_weight
            total_weight += source_weight
            
            # Add to appropriate score
            if raw_direction > 0:
                long_score += weighted_strength
            elif raw_direction < 0:
                short_score += weighted_strength
                
            # Store signal details for reference
            signal_details.append({
                "source_id": source_id,
                "direction": direction,
                "strength": signal_strength,
                "weight": source_weight,
                "weighted_contribution": weighted_strength * raw_direction
            })
            
        # Normalize scores
        if total_weight > 0:
            long_score /= total_weight
            short_score /= total_weight
            
        # Calculate net score and direction
        net_score = long_score - short_score
        
        if abs(net_score) < 0.2:  # Low conviction threshold
            direction = SignalDirection.NEUTRAL
            confidence = abs(net_score) * 2.5  # Scale 0-0.2 to 0-0.5
        elif net_score > 0:
            direction = SignalDirection.LONG
            confidence = min(1.0, net_score)
        else:
            direction = SignalDirection.SHORT
            confidence = min(1.0, abs(net_score))
            
        # Determine signal strength category
        if confidence < 0.2:
            strength = SignalStrength.VERY_WEAK
        elif confidence < 0.4:
            strength = SignalStrength.WEAK
        elif confidence < 0.6:
            strength = SignalStrength.MODERATE
        elif confidence < 0.8:
            strength = SignalStrength.STRONG
        else:
            strength = SignalStrength.VERY_STRONG
            
        return {
            "direction": direction,
            "confidence": confidence,
            "strength": strength,
            "long_score": long_score,
            "short_score": short_score,
            "net_score": net_score,
            "signal_details": signal_details,
            "signal_count": len(signals)
        }
        
    def _calculate_final_direction(
        self,
        long_score: float,
        short_score: float,
        threshold: float
    ) -> Tuple[str, float, str]:
        """
        Calculate final signal direction and confidence.
        
        Args:
            long_score: Aggregated long score (0.0-1.0)
            short_score: Aggregated short score (0.0-1.0)
            threshold: Signal generation threshold
            
        Returns:
            Tuple of (direction, confidence, strength)
        """
        # Calculate net score and confidence
        net_score = long_score - short_score
        
        # Determine direction based on threshold
        if abs(net_score) < threshold:
            direction = SignalDirection.NEUTRAL
            # Scale confidence within the neutral range
            confidence = abs(net_score) / threshold
        elif net_score > 0:
            direction = SignalDirection.LONG
            # Scale confidence for long signals
            confidence = min(1.0, net_score / (2.0 - threshold))
        else:
            direction = SignalDirection.SHORT
            # Scale confidence for short signals
            confidence = min(1.0, abs(net_score) / (2.0 - threshold))
            
        # Determine signal strength
        if confidence < 0.2:
            strength = SignalStrength.VERY_WEAK
        elif confidence < 0.4:
            strength = SignalStrength.WEAK
        elif confidence < 0.6:
            strength = SignalStrength.MODERATE
        elif confidence < 0.8:
            strength = SignalStrength.STRONG
        else:
            strength = SignalStrength.VERY_STRONG
            
        return direction, confidence, strength
        
    def _resolve_signal_conflicts(
        self,
        signals_by_source: Dict[str, List[Dict[str, Any]]],
        weights: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Resolve conflicts between signal sources.
        
        Args:
            signals_by_source: Dictionary mapping source IDs to their signals
            weights: Dictionary mapping source IDs to their weights
            
        Returns:
            Dictionary with conflict resolution information
        """
        if not signals_by_source:
            return {"conflicts_detected": False}
            
        # Count directions by source
        source_directions = {}
        for source_id, signals in signals_by_source.items():
            source_directions[source_id] = self._get_dominant_direction(signals)
            
        # Check for conflicts
        long_sources = [s for s, d in source_directions.items() if d == SignalDirection.LONG]
        short_sources = [s for s, d in source_directions.items() if d == SignalDirection.SHORT]
        
        has_conflict = bool(long_sources and short_sources)
        
        if not has_conflict:
            return {"conflicts_detected": False}
            
        # Resolve conflict by weight
        long_weight = sum(weights.get(s, 1.0) for s in long_sources)
        short_weight = sum(weights.get(s, 1.0) for s in short_sources)
        
        if long_weight > short_weight:
            winner = "long"
            confidence = long_weight / (long_weight + short_weight)
        elif short_weight > long_weight:
            winner = "short"
            confidence = short_weight / (long_weight + short_weight)
        else:
            winner = "neutral"
            confidence = 0.5
            
        return {
            "conflicts_detected": True,
            "long_sources": long_sources,
            "short_sources": short_sources,
            "long_weight": long_weight,
            "short_weight": short_weight,
            "winning_direction": winner,
            "resolution_confidence": confidence
        }
        
    def _get_dominant_direction(self, signals: List[Dict[str, Any]]) -> str:
        """
        Get the dominant direction from a list of signals.
        
        Args:
            signals: List of signals
            
        Returns:
            Dominant signal direction
        """
        if not signals:
            return SignalDirection.NEUTRAL
            
        # Count direction occurrences
        counts = {
            SignalDirection.LONG: 0,
            SignalDirection.SHORT: 0,
            SignalDirection.NEUTRAL: 0
        }
        
        # Sum signal strengths by direction
        for signal in signals:
            direction = signal.get("direction", "neutral").lower()
            strength = signal.get("strength", 0.5)
            
            if direction == "long":
                counts[SignalDirection.LONG] += strength
            elif direction == "short":
                counts[SignalDirection.SHORT] += strength
            else:
                counts[SignalDirection.NEUTRAL] += strength
                
        # Find max direction
        max_direction = max(counts.items(), key=lambda x: x[1])[0]
        return max_direction
        
    def _get_timeframe_weight(self, timeframe: str, market_regime: MarketRegime) -> float:
        """
        Get the importance weight for a timeframe based on market regime.
        
        Args:
            timeframe: Signal timeframe
            market_regime: Current market regime
            
        Returns:
            Weight for the timeframe (0.0-1.0)
        """
        # Default weights
        default_weights = {
            SignalTimeframe.IMMEDIATE.value: 0.2,
            SignalTimeframe.SHORT_TERM.value: 0.3,
            SignalTimeframe.MEDIUM_TERM.value: 0.3,
            SignalTimeframe.LONG_TERM.value: 0.2
        }
        
        # Regime-specific weights
        regime_weights = {
            MarketRegime.TRENDING_UP: {
                SignalTimeframe.IMMEDIATE.value: 0.1,
                SignalTimeframe.SHORT_TERM.value: 0.2,
                SignalTimeframe.MEDIUM_TERM.value: 0.4,
                SignalTimeframe.LONG_TERM.value: 0.3
            },
            MarketRegime.TRENDING_DOWN: {
                SignalTimeframe.IMMEDIATE.value: 0.1,
                SignalTimeframe.SHORT_TERM.value: 0.2,
                SignalTimeframe.MEDIUM_TERM.value: 0.4,
                SignalTimeframe.LONG_TERM.value: 0.3
            },
            MarketRegime.RANGING: {
                SignalTimeframe.IMMEDIATE.value: 0.3,
                SignalTimeframe.SHORT_TERM.value: 0.4,
                SignalTimeframe.MEDIUM_TERM.value: 0.2,
                SignalTimeframe.LONG_TERM.value: 0.1
            },
            MarketRegime.VOLATILE: {
                SignalTimeframe.IMMEDIATE.value: 0.4,
                SignalTimeframe.SHORT_TERM.value: 0.3,
                SignalTimeframe.MEDIUM_TERM.value: 0.2,
                SignalTimeframe.LONG_TERM.value: 0.1
            },
            MarketRegime.BREAKOUT: {
                SignalTimeframe.IMMEDIATE.value: 0.4,
                SignalTimeframe.SHORT_TERM.value: 0.3,
                SignalTimeframe.MEDIUM_TERM.value: 0.2,
                SignalTimeframe.LONG_TERM.value: 0.1
            },
            MarketRegime.REVERSAL: {
                SignalTimeframe.IMMEDIATE.value: 0.2,
                SignalTimeframe.SHORT_TERM.value: 0.3,
                SignalTimeframe.MEDIUM_TERM.value: 0.3,
                SignalTimeframe.LONG_TERM.value: 0.2
            }
        }
        
        # Get regime-specific weights or fall back to default
        weights = regime_weights.get(market_regime, default_weights)
        
        return weights.get(timeframe, 0.25)  # Default equal weight if unknown timeframe
        
    def _get_strongest_timeframe_signal(
        self,
        timeframe_signals: Dict[str, Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """
        Get the strongest timeframe signal for entry/exit points.
        
        Args:
            timeframe_signals: Dictionary mapping timeframes to signal results
            
        Returns:
            Dictionary with strongest timeframe information, or None if no signals
        """
        if not timeframe_signals:
            return None
            
        # Find timeframe with highest confidence
        strongest = max(
            timeframe_signals.items(),
            key=lambda x: x[1].get("confidence", 0)
        )
        
        timeframe = strongest[0]
        signal = strongest[1]
        
        return {
            "timeframe": timeframe,
            "direction": signal.get("direction"),
            "confidence": signal.get("confidence"),
            "strength": signal.get("strength")
        }
        
    def _track_signal(
        self, 
        symbol: str, 
        timeframe: str, 
        direction: str, 
        confidence: float, 
        market_regime: MarketRegime
    ) -> None:
        """
        Track a signal for persistence analysis.
        
        Args:
            symbol: Trading symbol
            timeframe: Signal timeframe
            direction: Signal direction
            confidence: Signal confidence
            market_regime: Current market regime
        """
        key = f"{symbol}:{timeframe}"
        
        signal_data = {
            "direction": direction,
            "confidence": confidence,
            "market_regime": market_regime.value,
            "timestamp": datetime.utcnow()
        }
        
        # Update recent signal
        self.recent_signals[key] = signal_data
        
        # Update history
        if key not in self.signal_history:
            self.signal_history[key] = []
            
        self.signal_history[key].append(signal_data)
        
        # Trim history if needed
        if len(self.signal_history[key]) > self.max_history_items:
            self.signal_history[key] = self.signal_history[key][-self.max_history_items:]
            
    def _calculate_signal_persistence(
        self, 
        symbol: Optional[str], 
        timeframe: Optional[str],
        current_direction: str
    ) -> Dict[str, Any]:
        """
        Calculate signal persistence metrics.
        
        Args:
            symbol: Trading symbol
            timeframe: Signal timeframe
            current_direction: Current signal direction
            
        Returns:
            Dictionary with persistence metrics
        """
        if not symbol or not timeframe:
            return {"persistence_score": 0, "consistent_bars": 0}
            
        key = f"{symbol}:{timeframe}"
        
        if key not in self.signal_history or len(self.signal_history[key]) < 2:
            return {"persistence_score": 0, "consistent_bars": 0}
            
        # Get signal history for this symbol/timeframe
        history = self.signal_history[key]
        
        # Count consecutive signals with the same direction
        consistent_bars = 0
        for signal in reversed(history[:-1]):  # Exclude current signal
            if signal["direction"] == current_direction:
                consistent_bars += 1
            else:
                break
                
        # Calculate persistence score (0-1)
        max_lookback = min(10, len(history) - 1)  # Look back up to 10 bars
        persistence_score = consistent_bars / max_lookback if max_lookback > 0 else 0
        
        return {
            "persistence_score": persistence_score,
            "consistent_bars": consistent_bars
        }
        
    def _create_neutral_signal(self, market_regime: MarketRegime) -> Dict[str, Any]:
        """
        Create a neutral signal when no inputs are available.
        
        Args:
            market_regime: Current market regime
            
        Returns:
            Dictionary with neutral signal information
        """
        return {
            "direction": SignalDirection.NEUTRAL,
            "confidence": 0.0,
            "strength": SignalStrength.VERY_WEAK,
            "generated_at": datetime.utcnow().isoformat(),
            "market_regime": market_regime.value,
            "timeframe_signals": {},
            "conflict_resolution": {"conflicts_detected": False},
            "strongest_timeframe": None,
            "source_count": 0,
            "signal_count": 0,
            "signal_persistence": {"persistence_score": 0, "consistent_bars": 0}
        }
