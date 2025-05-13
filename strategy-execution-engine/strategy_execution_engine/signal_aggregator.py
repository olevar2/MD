"""
Signal Aggregator

This module provides functionality to aggregate signals from various sources
(technical analysis, ML predictions, confluence analysis) and apply dynamically
adjusting weights based on tool effectiveness and market conditions.
"""

import logging
from typing import Dict, List, Any, Optional, Union, Tuple
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from enum import Enum
import asyncio

from strategy_execution_engine.adaptive_layer.adaptive_service import (
    AdaptiveLayerService, AdaptationContext, AdaptationLevel
)


class SignalDirection(Enum):
    """Direction of a trading signal"""
    LONG = "long"
    SHORT = "short"
    NEUTRAL = "neutral"


class SignalTimeframe(Enum):
    """Timeframe of a signal"""
    M1 = "1m"
    M5 = "5m"
    M15 = "15m"
    M30 = "30m"
    H1 = "1h"
    H4 = "4h"
    D1 = "1d"
    W1 = "1w"


class SignalSource(Enum):
    """Source of a trading signal"""
    TECHNICAL_ANALYSIS = "technical_analysis"
    MACHINE_LEARNING = "machine_learning"
    CONFLUENCE_ANALYSIS = "confluence_analysis"
    FUNDAMENTAL = "fundamental"
    SENTIMENT = "sentiment"
    MARKET_REGIME = "market_regime"
    PATTERN_RECOGNITION = "pattern_recognition"
    ECONOMIC_CALENDAR = "economic_calendar"


class Signal:
    """Represents a trading signal from any source"""
    
    def __init__(
        self,
        source_id: str,
        source_type: SignalSource,
        direction: SignalDirection,
        symbol: str,
        timeframe: SignalTimeframe,
        strength: float,  # 0.0 to 1.0
        timestamp: datetime,
        expiration: Optional[datetime] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
    """
      init  .
    
    Args:
        source_id: Description of source_id
        source_type: Description of source_type
        direction: Description of direction
        symbol: Description of symbol
        timeframe: Description of timeframe
        strength: Description of strength
        # 0.0 to 1.0
        timestamp: Description of # 0.0 to 1.0
        timestamp
        expiration: Description of expiration
        metadata: Description of metadata
        Any]]: Description of Any]]
    
    """

        self.source_id = source_id
        self.source_type = source_type
        self.direction = direction
        self.symbol = symbol
        self.timeframe = timeframe
        self.strength = max(0.0, min(1.0, strength))  # Clamp between 0 and 1
        self.timestamp = timestamp
        self.expiration = expiration
        self.metadata = metadata or {}
        
        # Will be filled by the aggregator
        self.adjusted_weight = 1.0
        self.adjusted_strength = self.strength
    
    def is_valid(self, current_time: datetime) -> bool:
        """Check if the signal is still valid (not expired)"""
        if not self.expiration:
            return True
        return current_time < self.expiration
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "source_id": self.source_id,
            "source_type": self.source_type.value,
            "direction": self.direction.value,
            "symbol": self.symbol,
            "timeframe": self.timeframe.value,
            "strength": self.strength,
            "adjusted_strength": self.adjusted_strength,
            "timestamp": self.timestamp.isoformat(),
            "expiration": self.expiration.isoformat() if self.expiration else None,
            "metadata": self.metadata,
            "adjusted_weight": self.adjusted_weight
        }


class AggregatedSignal:
    """Represents the final aggregated signal for a symbol"""
    
    def __init__(
        self,
        symbol: str,
        direction: SignalDirection,
        strength: float,
        confidence: float,  # 0.0 to 1.0
        contributing_signals: List[Signal],
        timestamp: datetime,
        timeframe: SignalTimeframe,
        market_context: Dict[str, Any]
    ):
    """
      init  .
    
    Args:
        symbol: Description of symbol
        direction: Description of direction
        strength: Description of strength
        confidence: Description of confidence
        # 0.0 to 1.0
        contributing_signals: Description of # 0.0 to 1.0
        contributing_signals
        timestamp: Description of timestamp
        timeframe: Description of timeframe
        market_context: Description of market_context
        Any]: Description of Any]
    
    """

        self.symbol = symbol
        self.direction = direction
        self.strength = strength
        self.confidence = confidence
        self.contributing_signals = contributing_signals
        self.timestamp = timestamp
        self.timeframe = timeframe
        self.market_context = market_context
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "symbol": self.symbol,
            "direction": self.direction.value,
            "strength": self.strength,
            "confidence": self.confidence,
            "contributing_signals_count": len(self.contributing_signals),
            "timestamp": self.timestamp.isoformat(),
            "timeframe": self.timeframe.value,
            "market_context": self.market_context
        }


class SignalAggregator:
    """
    Aggregates trading signals from multiple sources and applies dynamic
    weighting based on adaptive parameters.
    """
    
    def __init__(self, adaptive_layer_service: AdaptiveLayerService):
        """
        Initialize the Signal Aggregator
        
        Args:
            adaptive_layer_service: Service for getting adaptive parameters
        """
        self.adaptive_layer = adaptive_layer_service
        self.logger = logging.getLogger(__name__)
        
        # Default weights for different signal sources
        self.default_source_weights = {
            SignalSource.TECHNICAL_ANALYSIS: 1.0,
            SignalSource.MACHINE_LEARNING: 1.0,
            SignalSource.CONFLUENCE_ANALYSIS: 1.2,  # Higher weight for confluence
            SignalSource.FUNDAMENTAL: 0.8,
            SignalSource.SENTIMENT: 0.7,
            SignalSource.MARKET_REGIME: 0.9,
            SignalSource.PATTERN_RECOGNITION: 1.0,
            SignalSource.ECONOMIC_CALENDAR: 1.1
        }
        
        # Default weights for different timeframes
        self.default_timeframe_weights = {
            SignalTimeframe.M1: 0.6,
            SignalTimeframe.M5: 0.7,
            SignalTimeframe.M15: 0.8,
            SignalTimeframe.M30: 0.9,
            SignalTimeframe.H1: 1.0,
            SignalTimeframe.H4: 1.1,
            SignalTimeframe.D1: 1.2,
            SignalTimeframe.W1: 1.3
        }
        
        # Tool-specific weights (will be updated from adaptive layer)
        self.tool_weights = {}
        
    async def aggregate_signals(
        self,
        signals: List[Signal],
        symbol: str,
        target_timeframe: SignalTimeframe,
        current_market_regime: str,
        effectiveness_data: Optional[Dict[str, float]] = None,
        market_context: Optional[Dict[str, Any]] = None
    ) -> Optional[AggregatedSignal]:
        """
        Aggregate multiple signals into a single trading signal with direction and strength
        
        Args:
            signals: List of signals to aggregate
            symbol: The trading symbol
            target_timeframe: The timeframe for the aggregated signal
            current_market_regime: Current identified market regime
            effectiveness_data: Optional effectiveness scores for different tools
            market_context: Additional market context for decision making
            
        Returns:
            Aggregated signal or None if no valid signals
        """
        if not signals:
            return None
        
        current_time = datetime.now()
        valid_signals = [s for s in signals if s.is_valid(current_time)]
        
        if not valid_signals:
            return None
        
        # Get adaptive weights from the adaptive layer
        adaptive_weights = await self._get_adaptive_weights(current_market_regime)
        
        # Apply weights to each signal
        weighted_signals = []
        for signal in valid_signals:
            # Calculate adjusted weight for this signal
            adjusted_weight = self._calculate_signal_weight(
                signal,
                adaptive_weights,
                effectiveness_data,
                current_market_regime
            )
            
            # Update the signal's adjusted weight
            signal.adjusted_weight = adjusted_weight
            
            # Calculate adjusted strength
            signal.adjusted_strength = signal.strength * adjusted_weight
            
            weighted_signals.append(signal)
        
        # Calculate direction and strength based on weighted signals
        aggregated_direction, aggregated_strength, confidence = self._calculate_aggregated_direction(
            weighted_signals
        )
        
        # Create market context dict if not provided
        if not market_context:
            market_context = {
                "market_regime": current_market_regime,
                "analysis_timestamp": current_time.isoformat()
            }
        
        # Create aggregated signal
        return AggregatedSignal(
            symbol=symbol,
            direction=aggregated_direction,
            strength=aggregated_strength,
            confidence=confidence,
            contributing_signals=weighted_signals,
            timestamp=current_time,
            timeframe=target_timeframe,
            market_context=market_context
        )
        
    async def _get_adaptive_weights(
        self, 
        current_market_regime: str
    ) -> Dict[str, float]:
        """
        Get adaptive weights from the AdaptiveLayer
        
        Args:
            current_market_regime: Current identified market regime
            
        Returns:
            Dictionary of signal_source_id -> weight adjustment
        """
        # Request current adaptive parameters from the adaptive layer
        # These are tool-specific weights that have been adjusted based on performance
        parameters = await self.adaptive_layer.get_signal_weights(
            market_regime=current_market_regime
        )
        
        return parameters.get("signal_weights", {})
    
    def _calculate_signal_weight(
        self,
        signal: Signal,
        adaptive_weights: Dict[str, float],
        effectiveness_data: Optional[Dict[str, float]],
        current_market_regime: str
    ) -> float:
        """
        Calculate the final weight for a signal based on multiple factors
        
        Args:
            signal: The signal to weight
            adaptive_weights: Weights from the adaptive layer
            effectiveness_data: Tool effectiveness data if available
            current_market_regime: Current market regime
            
        Returns:
            The calculated weight for this signal
        """
        # Start with base weight of 1.0
        final_weight = 1.0
        
        # Apply source type weight
        source_type_weight = self.default_source_weights.get(signal.source_type, 1.0)
        final_weight *= source_type_weight
        
        # Apply timeframe weight
        timeframe_weight = self.default_timeframe_weights.get(signal.timeframe, 1.0)
        final_weight *= timeframe_weight
        
        # Apply tool-specific adaptive weight if available
        if signal.source_id in adaptive_weights:
            final_weight *= adaptive_weights[signal.source_id]
        
        # Apply effectiveness adjustment if available
        if effectiveness_data and signal.source_id in effectiveness_data:
            effectiveness = effectiveness_data[signal.source_id]
            # Scale effectiveness to reasonable range (e.g., 0.5-1.5)
            effectiveness_factor = 0.5 + effectiveness
            final_weight *= effectiveness_factor
        
        # Apply freshness decay - newer signals get higher weight
        age_minutes = (datetime.now() - signal.timestamp).total_seconds() / 60
        freshness_factor = max(0.5, 1.0 - (age_minutes / 1440))  # Decay over 24 hours
        final_weight *= freshness_factor
        
        # Account for market regime specific adjustments
        # Some signals perform better in trending vs ranging markets
        regime_adjustment = 1.0
        if "regime_specific_performance" in signal.metadata:
            regime_performances = signal.metadata["regime_specific_performance"]
            if current_market_regime in regime_performances:
                regime_adjustment = regime_performances[current_market_regime]
        final_weight *= regime_adjustment
        
        return final_weight
    
    def _calculate_aggregated_direction(
        self, 
        weighted_signals: List[Signal]
    ) -> Tuple[SignalDirection, float, float]:
        """
        Calculate the aggregated direction and strength from weighted signals
        
        Args:
            weighted_signals: List of signals with adjusted weights and strengths
            
        Returns:
            Tuple of (direction, strength, confidence)
        """
        # Calculate directional vote
        long_strength = sum(
            s.adjusted_strength for s in weighted_signals 
            if s.direction == SignalDirection.LONG
        )
        short_strength = sum(
            s.adjusted_strength for s in weighted_signals 
            if s.direction == SignalDirection.SHORT
        )
        
        # Calculate total vote strength
        total_strength = long_strength + short_strength
        if total_strength == 0:
            return SignalDirection.NEUTRAL, 0.0, 0.0
        
        # Determine direction
        if long_strength > short_strength:
            direction = SignalDirection.LONG
            strength = (long_strength - short_strength) / total_strength
        elif short_strength > long_strength:
            direction = SignalDirection.SHORT
            strength = (short_strength - long_strength) / total_strength
        else:
            direction = SignalDirection.NEUTRAL
            strength = 0.0
        
        # Calculate confidence based on:
        # 1. Agreement among signals
        # 2. Total number of signals
        # 3. Average signal strength
        
        # Agreement level (how unanimous is the direction)
        if total_strength > 0:
            if direction == SignalDirection.LONG:
                agreement = long_strength / total_strength
            elif direction == SignalDirection.SHORT:
                agreement = short_strength / total_strength
            else:
                agreement = 0.5  # Neutral has 50% agreement
        else:
            agreement = 0.0
        
        # Signal count factor (more signals = more confidence, up to a point)
        count = len(weighted_signals)
        count_factor = min(1.0, count / 5)  # Max out at 5 signals
        
        # Average strength
        avg_strength = total_strength / max(1, count)
        
        # Combined confidence calculation
        confidence = (agreement * 0.7) + (count_factor * 0.2) + (avg_strength * 0.1)
        confidence = min(1.0, max(0.0, confidence))
        
        return direction, min(1.0, strength), confidence
    
    def generate_explanation(self, aggregated_signal: AggregatedSignal) -> str:
        """
        Generate human-readable explanation of the aggregated signal
        
        Args:
            aggregated_signal: The aggregated signal to explain
            
        Returns:
            String explanation of how the signal was formed
        """
        explanation = []
        
        # Basic signal information
        explanation.append(
            f"Signal for {aggregated_signal.symbol} on {aggregated_signal.timeframe.value} timeframe: "
            f"{aggregated_signal.direction.value.upper()} with {aggregated_signal.strength:.2f} strength "
            f"and {aggregated_signal.confidence:.2f} confidence"
        )
        
        # Market context
        regime = aggregated_signal.market_context.get("market_regime", "Unknown")
        explanation.append(f"Current market regime: {regime}")
        
        # Contributing signals summary
        by_direction = {
            SignalDirection.LONG: [],
            SignalDirection.SHORT: [],
            SignalDirection.NEUTRAL: []
        }
        
        for signal in aggregated_signal.contributing_signals:
            by_direction[signal.direction].append(signal)
        
        # Sort signals by adjusted strength
        for direction in by_direction:
            by_direction[direction] = sorted(
                by_direction[direction],
                key=lambda s: s.adjusted_strength,
                reverse=True
            )
        
        # Add top signals from each direction
        explanation.append("\nTop contributing signals:")
        
        # Long signals
        if by_direction[SignalDirection.LONG]:
            explanation.append("\nLONG signals:")
            for signal in by_direction[SignalDirection.LONG][:3]:  # Top 3
                explanation.append(
                    f"  - {signal.source_type.value} ({signal.source_id}): "
                    f"Strength {signal.strength:.2f} with weight {signal.adjusted_weight:.2f}"
                )
        
        # Short signals
        if by_direction[SignalDirection.SHORT]:
            explanation.append("\nSHORT signals:")
            for signal in by_direction[SignalDirection.SHORT][:3]:  # Top 3
                explanation.append(
                    f"  - {signal.source_type.value} ({signal.source_id}): "
                    f"Strength {signal.strength:.2f} with weight {signal.adjusted_weight:.2f}"
                )
        
        # If there are conflicting signals, explain how the decision was made
        if by_direction[SignalDirection.LONG] and by_direction[SignalDirection.SHORT]:
            explanation.append(
                "\nNOTE: Conflicting signals were detected. The final direction was determined "
                "by the weighted strength of all signals."
            )
            
            long_strength = sum(s.adjusted_strength for s in by_direction[SignalDirection.LONG])
            short_strength = sum(s.adjusted_strength for s in by_direction[SignalDirection.SHORT])
            
            explanation.append(
                f"Total weighted LONG strength: {long_strength:.2f}, "
                f"SHORT strength: {short_strength:.2f}"
            )
        
        return "\n".join(explanation)
