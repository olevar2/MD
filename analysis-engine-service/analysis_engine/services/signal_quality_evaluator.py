"""
Signal Quality Evaluator

This module evaluates the quality of trading signals beyond basic win/loss metrics,
analyzing factors like entry timing, price action confirmation, and signal stability.
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Any, Union
from datetime import datetime

from analysis_engine.services.tool_effectiveness import SignalOutcome, MarketRegime, TimeFrame

class SignalQualityEvaluator:
    """
    Evaluates the quality of trading signals based on various factors
    beyond simple win/loss outcomes.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def evaluate_all_metrics(
        self, 
        outcomes: List[SignalOutcome],
        market_regime: Optional[MarketRegime] = None,
        timeframe: Optional[TimeFrame] = None
    ) -> Dict[str, Any]:
        """
        Calculate all signal quality metrics for a set of outcomes
        
        Args:
            outcomes: List of signal outcomes to evaluate
            market_regime: Optional filter by market regime
            timeframe: Optional filter by timeframe
            
        Returns:
            Dictionary with calculated quality metrics
        """
        try:
            if not outcomes:
                return {
                    "error": "No outcomes to evaluate",
                    "metrics": {},
                    "composite_quality_score": None
                }
                
            # Apply filters if provided
            filtered_outcomes = outcomes
            if market_regime:
                filtered_outcomes = [
                    o for o in filtered_outcomes 
                    if o.signal_event.market_context.get("regime") == market_regime.value
                ]
                
            if timeframe:
                filtered_outcomes = [
                    o for o in filtered_outcomes 
                    if o.signal_event.timeframe == timeframe.value
                ]
            
            # Calculate individual quality metrics
            entry_timing_score = self.evaluate_entry_timing(filtered_outcomes)
            signal_stability_score = self.evaluate_signal_stability(filtered_outcomes)
            price_action_alignment = self.evaluate_price_action_alignment(filtered_outcomes)
            reversal_potential = self.evaluate_reversal_potential(filtered_outcomes)
            false_signal_rate = self.evaluate_false_signal_rate(filtered_outcomes)
            
            # Calculate composite score (weighted average)
            weights = {
                "entry_timing": 0.25,
                "signal_stability": 0.20,
                "price_action_alignment": 0.25,
                "reversal_potential": 0.15,
                "false_signal_rate": 0.15
            }
            
            metrics = {
                "entry_timing": entry_timing_score,
                "signal_stability": signal_stability_score,
                "price_action_alignment": price_action_alignment,
                "reversal_potential": reversal_potential,
                "false_signal_rate": false_signal_rate
            }
            
            # Calculate weighted average for composite score
            score_values = []
            score_weights = []
            
            for metric_name, metric in metrics.items():
                if metric["value"] is not None:
                    score_values.append(metric["value"])
                    score_weights.append(weights[metric_name])
            
            if score_values and score_weights:
                composite_quality_score = sum(v * w for v, w in zip(score_values, score_weights)) / sum(score_weights)
            else:
                composite_quality_score = None
            
            return {
                "metrics": metrics,
                "composite_quality_score": composite_quality_score,
                "sample_size": len(filtered_outcomes)
            }
            
        except Exception as e:
            self.logger.error(f"Error evaluating signal quality: {str(e)}")
            return {
                "error": f"Error evaluating signal quality: {str(e)}",
                "metrics": {},
                "composite_quality_score": None
            }
            
    def evaluate_entry_timing(self, outcomes: List[SignalOutcome]) -> Dict[str, Any]:
        """
        Evaluate how well-timed the entry signals were
        
        This looks at how close to optimal entry points the signals were made
        by examining price movements following the signal.
        
        Args:
            outcomes: List of signal outcomes to evaluate
            
        Returns:
            Dictionary with entry timing quality assessment
        """
        try:
            if not outcomes:
                return {"value": None, "sample_size": 0}
            
            # Extract relevant data from outcomes
            entry_timing_scores = []
            
            for outcome in outcomes:
                # Skip outcomes without needed data
                if not outcome.additional_data:
                    continue
                    
                # Get price at signal and max adverse/favorable prices
                price_at_signal = outcome.additional_data.get("price_at_signal")
                max_adverse_price = outcome.max_adverse_price
                exit_price = outcome.exit_price
                
                if price_at_signal is None or max_adverse_price is None or exit_price is None:
                    continue
                
                # Calculate adverse movement before favorable movement
                if outcome.signal_event.direction.lower() == "buy":
                    # For buy signals, adverse movement is downward
                    adverse_move = (price_at_signal - max_adverse_price) / price_at_signal
                    total_move = abs((exit_price - price_at_signal) / price_at_signal)
                else:
                    # For sell signals, adverse movement is upward
                    adverse_move = (max_adverse_price - price_at_signal) / price_at_signal
                    total_move = abs((price_at_signal - exit_price) / price_at_signal)
                
                # Normalize to 0-1 score (0 = bad timing, 1 = perfect timing)
                # The less adverse movement before the eventual move in the expected direction, the better
                if total_move > 0:
                    timing_score = max(0, min(1, 1 - (adverse_move / (total_move + adverse_move))))
                    entry_timing_scores.append(timing_score)
            
            # Calculate average entry timing score
            if entry_timing_scores:
                avg_score = sum(entry_timing_scores) / len(entry_timing_scores)
                return {
                    "value": avg_score,
                    "sample_size": len(entry_timing_scores),
                    "description": "How well-timed the entry signals were (0-1)"
                }
            else:
                return {"value": None, "sample_size": 0}
                
        except Exception as e:
            self.logger.error(f"Error evaluating entry timing: {str(e)}")
            return {"value": None, "error": str(e), "sample_size": 0}
    
    def evaluate_signal_stability(self, outcomes: List[SignalOutcome]) -> Dict[str, Any]:
        """
        Evaluate the stability of signals over time
        
        This looks at whether signals maintain consistent direction/confidence
        without frequent reversals or contradictions.
        
        Args:
            outcomes: List of signal outcomes to evaluate
            
        Returns:
            Dictionary with signal stability assessment
        """
        try:
            if not outcomes or len(outcomes) < 2:
                return {"value": None, "sample_size": 0}
            
            # Sort outcomes by timestamp
            sorted_outcomes = sorted(outcomes, key=lambda o: o.signal_event.timestamp)
            
            # Count direction changes
            direction_changes = 0
            prev_direction = sorted_outcomes[0].signal_event.direction
            
            for outcome in sorted_outcomes[1:]:
                current_direction = outcome.signal_event.direction
                if current_direction != prev_direction:
                    direction_changes += 1
                prev_direction = current_direction
            
            # Calculate stability as lack of direction changes
            # (0 = unstable with many changes, 1 = very stable with no changes)
            max_possible_changes = len(sorted_outcomes) - 1
            if max_possible_changes > 0:
                stability_score = 1.0 - (direction_changes / max_possible_changes)
                
                return {
                    "value": stability_score,
                    "sample_size": len(sorted_outcomes),
                    "description": "Stability of signals without direction changes (0-1)"
                }
            else:
                return {"value": None, "sample_size": 0}
                
        except Exception as e:
            self.logger.error(f"Error evaluating signal stability: {str(e)}")
            return {"value": None, "error": str(e), "sample_size": 0}
    
    def evaluate_price_action_alignment(self, outcomes: List[SignalOutcome]) -> Dict[str, Any]:
        """
        Evaluate how well signals align with price action
        
        This assesses whether signals are aligned with the actual price movement
        patterns observed in the market.
        
        Args:
            outcomes: List of signal outcomes to evaluate
            
        Returns:
            Dictionary with price action alignment assessment
        """
        try:
            if not outcomes:
                return {"value": None, "sample_size": 0}
            
            alignment_scores = []
            
            for outcome in outcomes:
                if not outcome.additional_data:
                    continue
                    
                # Get price action context if available
                price_action_context = outcome.additional_data.get("price_action", {})
                if not price_action_context:
                    continue
                
                # Compare signal direction with price action trend
                signal_direction = outcome.signal_event.direction.lower()
                price_trend = price_action_context.get("trend", "").lower()
                
                if not price_trend:
                    continue
                
                # Score alignment (1 = perfect alignment, 0 = complete misalignment)
                if (signal_direction == "buy" and price_trend == "bullish") or \
                   (signal_direction == "sell" and price_trend == "bearish"):
                    alignment_scores.append(1.0)
                elif (signal_direction == "buy" and price_trend == "bearish") or \
                     (signal_direction == "sell" and price_trend == "bullish"):
                    alignment_scores.append(0.0)
                else:
                    # Neutral trend gets a middle score
                    alignment_scores.append(0.5)
            
            if alignment_scores:
                avg_alignment = sum(alignment_scores) / len(alignment_scores)
                return {
                    "value": avg_alignment,
                    "sample_size": len(alignment_scores),
                    "description": "Alignment between signals and price action (0-1)"
                }
            else:
                return {"value": None, "sample_size": 0}
                
        except Exception as e:
            self.logger.error(f"Error evaluating price action alignment: {str(e)}")
            return {"value": None, "error": str(e), "sample_size": 0}
    
    def evaluate_reversal_potential(self, outcomes: List[SignalOutcome]) -> Dict[str, Any]:
        """
        Evaluate ability to identify potential market reversals
        
        This assesses how well the signals can identify turning points in the market.
        
        Args:
            outcomes: List of signal outcomes to evaluate
            
        Returns:
            Dictionary with reversal potential assessment
        """
        try:
            if not outcomes:
                return {"value": None, "sample_size": 0}
            
            reversal_scores = []
            
            for outcome in outcomes:
                if not outcome.additional_data:
                    continue
                    
                # Check if this was a reversal signal
                is_reversal_signal = outcome.additional_data.get("is_reversal_signal", False)
                
                if not is_reversal_signal:
                    continue
                
                # For reversal signals, check if they were correct
                if outcome.success:
                    reversal_scores.append(1.0)
                else:
                    reversal_scores.append(0.0)
            
            if reversal_scores:
                avg_reversal_score = sum(reversal_scores) / len(reversal_scores)
                return {
                    "value": avg_reversal_score,
                    "sample_size": len(reversal_scores),
                    "description": "Accuracy of reversal signals (0-1)"
                }
            else:
                return {"value": None, "sample_size": 0}
                
        except Exception as e:
            self.logger.error(f"Error evaluating reversal potential: {str(e)}")
            return {"value": None, "error": str(e), "sample_size": 0}
    
    def evaluate_false_signal_rate(self, outcomes: List[SignalOutcome]) -> Dict[str, Any]:
        """
        Evaluate the rate of false signals
        
        This assesses how often signals are generated but quickly invalidated
        without reaching either a successful outcome or a stop loss.
        
        Args:
            outcomes: List of signal outcomes to evaluate
            
        Returns:
            Dictionary with false signal rate assessment
        """
        try:
            if not outcomes:
                return {"value": None, "sample_size": 0}
            
            # Count signals that were quickly invalidated (marked in additional_data)
            false_signals = 0
            
            for outcome in outcomes:
                if outcome.additional_data and outcome.additional_data.get("invalidated_quickly", False):
                    false_signals += 1
            
            # Calculate false signal rate and convert to quality score (invert: 1 = no false signals)
            false_signal_rate = false_signals / len(outcomes) if outcomes else 0
            quality_score = 1.0 - false_signal_rate
            
            return {
                "value": quality_score,
                "sample_size": len(outcomes),
                "description": "Lack of false/invalidated signals (0-1)"
            }
                
        except Exception as e:
            self.logger.error(f"Error evaluating false signal rate: {str(e)}")
            return {"value": None, "error": str(e), "sample_size": 0}
