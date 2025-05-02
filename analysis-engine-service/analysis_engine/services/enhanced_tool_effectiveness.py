"""
Tool Effectiveness Tracker

This module provides a comprehensive service for tracking and analyzing tool effectiveness metrics.
It integrates the core metrics calculation, signal quality evaluation, and error handling.
"""

import logging
import uuid
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta

from analysis_engine.services.tool_effectiveness import (
    SignalEvent, 
    SignalOutcome,
    MarketRegime,
    TimeFrame,
    WinRateMetric,
    ProfitFactorMetric
)
from analysis_engine.services.effectiveness_metrics import ExpectedPayoffMetric, ReliabilityByMarketRegimeMetric
from analysis_engine.services.signal_quality_evaluator import SignalQualityEvaluator

class EnhancedToolEffectivenessTracker:
    """
    Enhanced service for tracking and analyzing the effectiveness of trading tools
    across different market conditions with comprehensive metrics and error handling.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Initialize metric calculators
        self.win_rate_metric = WinRateMetric()
        self.profit_factor_metric = ProfitFactorMetric()
        self.expected_payoff_metric = ExpectedPayoffMetric()
        self.reliability_metric = ReliabilityByMarketRegimeMetric()
        
        # Initialize signal quality evaluator
        self.signal_quality_evaluator = SignalQualityEvaluator()
    
    def register_signal(self, signal_event: SignalEvent) -> str:
        """
        Register a new signal for tracking effectiveness
        
        Args:
            signal_event: The signal event to register
            
        Returns:
            str: The generated signal ID
        """
        try:
            # Generate a unique ID for the signal if not already present
            signal_id = getattr(signal_event, 'id', str(uuid.uuid4()))
            
            self.logger.info(f"Registered signal {signal_id} from tool '{signal_event.tool_name}' for {signal_event.symbol}")
            
            return signal_id
        except Exception as e:
            self.logger.error(f"Error registering signal: {str(e)}")
            raise RuntimeError(f"Failed to register signal: {str(e)}")
    
    def register_outcome(self, signal_outcome: SignalOutcome) -> None:
        """
        Register the outcome of a previously registered signal
        
        Args:
            signal_outcome: The outcome of a signal
        """
        try:
            signal_id = getattr(signal_outcome.signal_event, 'id', None)
            if signal_id is None:
                self.logger.warning("Signal outcome has no associated signal ID")
                return
            
            self.logger.info(
                f"Registered outcome for signal {signal_id}: {signal_outcome.outcome} "
                f"with profit/loss {signal_outcome.profit_loss}"
            )
        except Exception as e:
            self.logger.error(f"Error registering outcome: {str(e)}")
            raise RuntimeError(f"Failed to register outcome: {str(e)}")
    
    def calculate_metrics(
        self,
        outcomes: List[SignalOutcome],
        tool_name: Optional[str] = None,
        market_regime: Optional[MarketRegime] = None,
        timeframe: Optional[TimeFrame] = None,
        symbol: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Calculate effectiveness metrics for a set of signal outcomes
        
        Args:
            outcomes: List of signal outcomes
            tool_name: Optional filter by tool name
            market_regime: Optional filter by market regime
            timeframe: Optional filter by timeframe
            symbol: Optional filter by trading symbol
            start_date: Optional filter by start date
            end_date: Optional filter by end date
            
        Returns:
            Dictionary with calculated effectiveness metrics
        """
        try:
            # Filter by tool name if specified
            if tool_name:
                outcomes = [o for o in outcomes if o.signal_event.tool_name == tool_name]
            
            if not outcomes:
                self.logger.warning("No outcomes match criteria for metrics calculation")
                return {
                    "error": "No outcomes match the specified criteria",
                    "sample_size": 0,
                    "metrics": {}
                }
            
            # Calculate core metrics
            win_rate = self.win_rate_metric.calculate(
                outcomes, market_regime, timeframe, symbol, start_date, end_date
            )
            
            profit_factor = self.profit_factor_metric.calculate(
                outcomes, market_regime, timeframe, symbol, start_date, end_date
            )
            
            expected_payoff = self.expected_payoff_metric.calculate(
                outcomes, market_regime, timeframe, symbol, start_date, end_date
            )
            
            # Calculate reliability across market regimes
            reliability_by_regime = self.reliability_metric.calculate(
                outcomes, timeframe, symbol, start_date, end_date
            )
            
            # Calculate signal quality metrics
            signal_quality = self.signal_quality_evaluator.evaluate_all_metrics(
                outcomes, market_regime, timeframe
            )
            
            # Assemble the comprehensive metrics report
            metrics_report = {
                "sample_size": len(outcomes),
                "filtered_sample_size": win_rate["sample_size"],
                "time_range": {
                    "start_date": start_date.isoformat() if start_date else None,
                    "end_date": end_date.isoformat() if end_date else None
                },
                "filters": {
                    "tool_name": tool_name,
                    "market_regime": market_regime.value if market_regime else None,
                    "timeframe": timeframe.value if timeframe else None,
                    "symbol": symbol
                },
                "core_metrics": {
                    "win_rate": win_rate,
                    "profit_factor": profit_factor,
                    "expected_payoff": expected_payoff,
                    "reliability_by_regime": reliability_by_regime
                },
                "signal_quality_metrics": signal_quality,
                "composite_score": self._calculate_composite_score(
                    win_rate, profit_factor, expected_payoff, reliability_by_regime, signal_quality
                )
            }
            
            self.logger.info(
                f"Calculated metrics for {metrics_report['filtered_sample_size']} outcomes "
                f"with win rate: {win_rate['value']:.2f}, profit factor: {profit_factor['value']}"
            )
            
            return metrics_report
        except Exception as e:
            self.logger.error(f"Error calculating metrics: {str(e)}")
            return {
                "error": f"Failed to calculate metrics: {str(e)}",
                "sample_size": len(outcomes) if outcomes else 0,
                "metrics": {}
            }
    
    def analyze_tool_by_market_regime(
        self,
        outcomes: List[SignalOutcome],
        tool_name: str
    ) -> Dict[str, Any]:
        """
        Analyze how a specific tool performs across different market regimes
        
        Args:
            outcomes: List of signal outcomes
            tool_name: The name of the tool to analyze
            
        Returns:
            Dictionary with regime-specific metrics
        """
        try:
            # Filter outcomes for the specified tool
            tool_outcomes = [o for o in outcomes if o.signal_event.tool_name == tool_name]
            
            if not tool_outcomes:
                self.logger.warning(f"No outcomes found for tool '{tool_name}'")
                return {
                    "error": f"No outcomes found for tool '{tool_name}'",
                    "sample_size": 0,
                    "regimes": {}
                }
            
            # Get unique regimes in the outcomes
            regimes = set()
            for outcome in tool_outcomes:
                regime = outcome.signal_event.market_context.get("regime", MarketRegime.UNKNOWN.value)
                regimes.add(regime)
            
            # Calculate metrics for each regime
            regime_metrics = {}
            for regime in regimes:
                regime_outcomes = [
                    o for o in tool_outcomes 
                    if o.signal_event.market_context.get("regime", MarketRegime.UNKNOWN.value) == regime
                ]
                
                regime_metrics[regime] = {
                    "sample_size": len(regime_outcomes),
                    "win_rate": self.win_rate_metric.calculate(regime_outcomes),
                    "profit_factor": self.profit_factor_metric.calculate(regime_outcomes),
                    "expected_payoff": self.expected_payoff_metric.calculate(regime_outcomes)
                }
            
            # Calculate overall metrics
            overall_metrics = {
                "win_rate": self.win_rate_metric.calculate(tool_outcomes),
                "profit_factor": self.profit_factor_metric.calculate(tool_outcomes),
                "expected_payoff": self.expected_payoff_metric.calculate(tool_outcomes)
            }
            
            # Determine best and worst regimes based on win rate
            if len(regime_metrics) > 1:
                best_regime = max(
                    regime_metrics.items(), 
                    key=lambda x: x[1]["win_rate"]["value"] if x[1]["win_rate"]["value"] is not None else -1
                )[0]
                
                worst_regime = min(
                    regime_metrics.items(), 
                    key=lambda x: x[1]["win_rate"]["value"] if x[1]["win_rate"]["value"] is not None else float('inf')
                )[0]
            else:
                best_regime = list(regime_metrics.keys())[0] if regime_metrics else None
                worst_regime = None
            
            return {
                "tool_name": tool_name,
                "sample_size": len(tool_outcomes),
                "overall_metrics": overall_metrics,
                "regime_metrics": regime_metrics,
                "best_regime": best_regime,
                "worst_regime": worst_regime
            }
        except Exception as e:
            self.logger.error(f"Error analyzing tool by market regime: {str(e)}")
            return {
                "error": f"Failed to analyze tool by market regime: {str(e)}",
                "sample_size": len(outcomes) if outcomes else 0,
                "regimes": {}
            }
    
    def get_effectiveness_trend(
        self,
        outcomes: List[SignalOutcome],
        tool_name: Optional[str] = None,
        window_size: int = 20,
        step_size: int = 5
    ) -> Dict[str, Any]:
        """
        Calculate effectiveness metrics over time to identify trends
        
        Args:
            outcomes: List of signal outcomes
            tool_name: Optional filter by tool name
            window_size: Size of rolling window for calculations
            step_size: Number of outcomes to step between windows
            
        Returns:
            Dictionary with effectiveness trends
        """
        try:
            # Filter by tool name if specified
            if tool_name:
                outcomes = [o for o in outcomes if o.signal_event.tool_name == tool_name]
            
            if len(outcomes) < window_size:
                self.logger.warning(f"Not enough outcomes for trend analysis (need at least {window_size})")
                return {
                    "error": f"Not enough outcomes for trend analysis (need at least {window_size})",
                    "sample_size": len(outcomes),
                    "trends": {}
                }
            
            # Sort outcomes by timestamp
            sorted_outcomes = sorted(outcomes, key=lambda o: o.signal_event.timestamp)
            
            # Calculate metrics for rolling windows
            trend_data = []
            for i in range(0, len(sorted_outcomes) - window_size + 1, step_size):
                window = sorted_outcomes[i:i+window_size]
                end_date = window[-1].signal_event.timestamp
                
                win_rate = self.win_rate_metric.calculate(window)
                profit_factor = self.profit_factor_metric.calculate(window)
                expected_payoff = self.expected_payoff_metric.calculate(window)
                
                trend_data.append({
                    "end_date": end_date.isoformat(),
                    "window_size": len(window),
                    "metrics": {
                        "win_rate": win_rate["value"],
                        "profit_factor": profit_factor["value"],
                        "expected_payoff": expected_payoff["value"]
                    }
                })
            
            # Analyze trend direction
            if len(trend_data) > 1:
                first_win_rate = trend_data[0]["metrics"]["win_rate"]
                last_win_rate = trend_data[-1]["metrics"]["win_rate"]
                win_rate_trend = "improving" if last_win_rate > first_win_rate else "declining"
                
                first_pf = trend_data[0]["metrics"]["profit_factor"]
                last_pf = trend_data[-1]["metrics"]["profit_factor"]
                pf_trend = "improving" if last_pf > first_pf else "declining"
            else:
                win_rate_trend = "insufficient data"
                pf_trend = "insufficient data"
            
            return {
                "tool_name": tool_name,
                "sample_size": len(outcomes),
                "window_size": window_size,
                "step_size": step_size,
                "trend_points": len(trend_data),
                "trend_direction": {
                    "win_rate": win_rate_trend,
                    "profit_factor": pf_trend
                },
                "trend_data": trend_data
            }
        except Exception as e:
            self.logger.error(f"Error calculating effectiveness trend: {str(e)}")
            return {
                "error": f"Failed to calculate effectiveness trend: {str(e)}",
                "sample_size": len(outcomes) if outcomes else 0,
                "trends": {}
            }

    def _calculate_composite_score(
        self,
        win_rate: Dict[str, Any],
        profit_factor: Dict[str, Any],
        expected_payoff: Dict[str, Any],
        reliability: Dict[str, Any],
        signal_quality: Dict[str, Any]
    ) -> Optional[float]:
        """
        Calculate a composite effectiveness score based on all available metrics
        
        Args:
            win_rate: Win rate metric results
            profit_factor: Profit factor metric results
            expected_payoff: Expected payoff metric results
            reliability: Reliability by market regime metric results
            signal_quality: Signal quality evaluation results
            
        Returns:
            A composite score between 0 and 1, or None if calculation is not possible
        """
        try:
            # Collect all metric values that are not None
            metric_values = []
            
            # Add normalized win rate (already 0-1)
            if win_rate["value"] is not None:
                metric_values.append(win_rate["value"])
            
            # Add normalized profit factor (cap at 5 for normalization)
            if profit_factor["value"] is not None and profit_factor["value"] != float('inf'):
                pf_normalized = min(profit_factor["value"], 5) / 5.0
                metric_values.append(pf_normalized)
            
            # Add reliability score if available
            if reliability["value"] is not None:
                metric_values.append(reliability["value"])
            
            # Add signal quality composite score if available
            if signal_quality["composite_quality_score"] is not None:
                metric_values.append(signal_quality["composite_quality_score"])
            
            # Expected payoff requires special handling as it can be negative
            # We convert it to 0-1 scale using a sigmoid-like function
            if expected_payoff["value"] is not None:
                # Normalize to 0-1 range where 0.5 means breakeven
                ep_normalized = 1 / (1 + np.exp(-expected_payoff["value"] * 0.1))
                metric_values.append(ep_normalized)
            
            if not metric_values:
                return None
            
            # Calculate the average of all normalized metrics
            composite_score = sum(metric_values) / len(metric_values)
            return composite_score
        
        except Exception as e:
            self.logger.error(f"Error calculating composite score: {str(e)}")
            return None
