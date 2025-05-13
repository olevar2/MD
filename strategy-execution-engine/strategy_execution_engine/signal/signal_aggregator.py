"""
Signal Aggregator Module for Forex Trading Platform

This module combines signals from multiple sources including technical analysis,
machine learning predictions, and adaptive parameters to generate final trading decisions.
"""
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Any, Optional, Union, Callable

logger = logging.getLogger(__name__)

class SignalAggregator:
    """
    Aggregates signals from multiple sources to produce final trading decisions
    
    This class takes signals from various sources (technical analysis tools,
    machine learning models, strategies) and produces a unified signal with
    confidence levels by applying weight-based aggregation methods.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the signal aggregator
        
        Args:
            config: Configuration dictionary for signal aggregation
        """
        self.config = config or {}
        self.default_weights = {
            "technical_analysis": 0.5,
            "machine_learning": 0.3,
            "market_regime": 0.1,
            "correlation": 0.1
        }
        
        # Load weights from config or use defaults
        self.signal_weights = self.config_manager.get('signal_weights', self.default_weights)
        self.regime_adjustments = self.config_manager.get('regime_adjustments', {})
        self.confidence_threshold = self.config_manager.get('confidence_threshold', 0.5)
        self.use_adaptive_weights = self.config_manager.get('use_adaptive_weights', True)
        
    def set_adaptive_weights(self, weights: Dict[str, float]) -> None:
        """
        Update signal weights with adaptive values from the adaptive layer
        
        Args:
            weights: Dictionary of weight adjustments from adaptive layer
        """
        if self.use_adaptive_weights and weights:
            logger.info(f"Updating signal weights with adaptive values: {weights}")
            self.signal_weights.update(weights)
            
    def _normalize_weights(self, weights: Dict[str, float]) -> Dict[str, float]:
        """
        Normalize weights to sum to 1.0
        
        Args:
            weights: Dictionary of signal category weights
            
        Returns:
            Normalized weights dictionary
        """
        total = sum(weights.values())
        if total == 0:
            return weights
        return {k: v / total for k, v in weights.items()}
            
    def aggregate_signals(
        self, 
        data: pd.DataFrame,
        technical_signals: Dict[str, pd.Series] = None,
        ml_predictions: pd.Series = None, 
        market_regime: str = None,
        correlation_signals: Dict[str, pd.Series] = None,
        news_events: List[Dict[str, Any]] = None
    ) -> pd.DataFrame:
        """
        Aggregate signals from multiple sources to produce a final signal
        
        Args:
            data: Base market data DataFrame
            technical_signals: Dictionary of technical signal Series
            ml_predictions: Machine learning prediction Series
            market_regime: Current market regime identifier
            correlation_signals: Dictionary of correlation-based signals
            news_events: List of news events during the period
            
        Returns:
            DataFrame with aggregated signals and confidence
        """
        result = data.copy()
        
        # Initialize columns for aggregated signal and confidence
        result['aggregated_signal'] = 0
        result['signal_confidence'] = 0.0
        
        # Apply regime-specific weight adjustments if available
        current_weights = self.signal_weights.copy()
        if market_regime and market_regime in self.regime_adjustments:
            regime_adj = self.regime_adjustments[market_regime]
            for k, v in regime_adj.items():
                if k in current_weights:
                    current_weights[k] *= v
        
        # Normalize weights
        current_weights = self._normalize_weights(current_weights)
        logger.debug(f"Using normalized weights: {current_weights}")
        
        # Process technical analysis signals
        if technical_signals:
            tech_signal = self._combine_technical_signals(technical_signals, result.index)
            result['technical_signal'] = tech_signal
            result['aggregated_signal'] += tech_signal * current_weights.get('technical_analysis', 0)
        
        # Process ML predictions
        if ml_predictions is not None:
            # Ensure the series has the right index
            aligned_ml = ml_predictions.reindex(result.index)
            result['ml_signal'] = aligned_ml
            result['aggregated_signal'] += aligned_ml * current_weights.get('machine_learning', 0)
        
        # Process correlation signals
        if correlation_signals:
            corr_signal = self._combine_correlation_signals(correlation_signals, result.index)
            result['correlation_signal'] = corr_signal
            result['aggregated_signal'] += corr_signal * current_weights.get('correlation', 0)
        
        # Calculate confidence based on signal strength and agreement
        self._calculate_confidence(result)
        
        # Apply news event filters if provided
        if news_events:
            self._apply_news_filter(result, news_events)
        
        # Convert aggregated signal to discrete values based on confidence threshold
        result['final_signal'] = 0
        result.loc[result['aggregated_signal'] > self.confidence_threshold, 'final_signal'] = 1
        result.loc[result['aggregated_signal'] < -self.confidence_threshold, 'final_signal'] = -1
        
        return result
    
    def _combine_technical_signals(self, signals: Dict[str, pd.Series], index: pd.Index) -> pd.Series:
        """
        Combine multiple technical signals into a single signal series
        
        Args:
            signals: Dictionary of technical signal series
            index: Target index for the result
            
        Returns:
            Combined technical signal series
        """
        if not signals:
            return pd.Series(0, index=index)
        
        # Get tool effectiveness weights if available in config
        tool_weights = self.config_manager.get('technical_tool_weights', {})
        
        # Sum all signals with their respective weights
        combined = pd.Series(0, index=index)
        count = 0
        
        for tool_name, signal in signals.items():
            # Get weight for this tool, default to 1.0 if not specified
            weight = tool_weights.get(tool_name, 1.0)
            
            # Align signal to target index
            aligned = signal.reindex(index, fill_value=0)
            
            # Add weighted signal to combined
            combined += aligned * weight
            count += weight
        
        # Normalize by total weight
        if count > 0:
            combined /= count
            
        return combined
    
    def _combine_correlation_signals(self, signals: Dict[str, pd.Series], index: pd.Index) -> pd.Series:
        """
        Combine correlation-based signals into a single signal series
        
        Args:
            signals: Dictionary of correlation signal series
            index: Target index for the result
            
        Returns:
            Combined correlation signal series
        """
        # Similar to technical signals but might use different combination logic
        if not signals:
            return pd.Series(0, index=index)
        
        # Default to simple average
        combined = pd.Series(0, index=index)
        for signal in signals.values():
            aligned = signal.reindex(index, fill_value=0)
            combined += aligned
            
        return combined / len(signals)
    
    def _calculate_confidence(self, data: pd.DataFrame) -> None:
        """
        Calculate confidence score based on signal agreement and strength
        
        Args:
            data: DataFrame with all signal components
        """
        # Initialize confidence column
        data['signal_confidence'] = abs(data['aggregated_signal'])
        
        # Increase confidence when multiple signals agree
        if 'technical_signal' in data.columns and 'ml_signal' in data.columns:
            # Check if technical and ML signals agree (both positive or both negative)
            agreement = (data['technical_signal'] * data['ml_signal']) > 0
            # Boost confidence when signals agree
            data.loc[agreement, 'signal_confidence'] *= 1.2
        
        # Cap confidence at 1.0
        data.loc[data['signal_confidence'] > 1.0, 'signal_confidence'] = 1.0
    
    def _apply_news_filter(self, data: pd.DataFrame, news_events: List[Dict[str, Any]]) -> None:
        """
        Filter signals around high-impact news events
        
        Args:
            data: DataFrame with signals
            news_events: List of news events with timestamp, impact, etc.
        """
        if not news_events:
            return
            
        # Sort events by timestamp
        sorted_events = sorted(news_events, key=lambda x: x['timestamp'])
        
        # Create a window around each high-impact event
        for event in sorted_events:
            if event.get('impact', '').lower() == 'high':
                event_time = pd.Timestamp(event['timestamp'])
                
                # Define window (default: 15 minutes before, 30 minutes after)
                before_window = self.config_manager.get('news_window_before', 15)  # minutes
                after_window = self.config_manager.get('news_window_after', 30)  # minutes
                
                start_window = event_time - pd.Timedelta(minutes=before_window)
                end_window = event_time + pd.Timedelta(minutes=after_window)
                
                # Filter out signals during news window
                mask = (data.index >= start_window) & (data.index <= end_window)
                data.loc[mask, 'final_signal'] = 0
                
                logger.info(f"Filtered signals around high-impact news event at {event_time}")
                
    def reset(self) -> None:
        """
        Reset the signal aggregator to initial state
        """
        self.signal_weights = self.config_manager.get('signal_weights', self.default_weights)
