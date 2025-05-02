"""
Market Regime Analysis Service

This module provides functionality to analyze tool effectiveness across different market regimes,
helping to identify which tools perform best under specific market conditions.
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from analysis_engine.services.tool_effectiveness import (
    MarketRegime, 
    TimeFrame, 
    ToolEffectivenessMetric, 
    ToolEffectivenessTracker,
    SignalEvent,
    SignalOutcome,
    WinRateMetric,
    ProfitFactorMetric,
    ExpectedPayoffMetric
)

from analysis_engine.repositories.tool_effectiveness_repository import ToolEffectivenessRepository


class MarketRegimeAnalysisService:
    """Service for analyzing tool effectiveness across different market regimes"""
    
    def __init__(self, repository: ToolEffectivenessRepository):
        self.repository = repository
        self.logger = logging.getLogger(__name__)
        
    def get_regime_performance_matrix(
        self, 
        tool_id: str,
        timeframe: Optional[str] = None,
        instrument: Optional[str] = None,
        from_date: Optional[datetime] = None,
        to_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Generate a performance matrix showing how a tool performs across different market regimes
        
        Args:
            tool_id: ID of the trading tool to analyze
            timeframe: Optional filter by timeframe
            instrument: Optional filter by trading instrument
            from_date: Optional start date for analysis
            to_date: Optional end date for analysis
            
        Returns:
            Dictionary with performance metrics by market regime
        """
        # Get all signals for this tool with the specified filters
        signals = self.repository.get_signals(
            tool_id=tool_id,
            timeframe=timeframe,
            instrument=instrument,
            from_date=from_date,
            to_date=to_date
        )
        
        if not signals:
            self.logger.warning(f"No signals found for tool {tool_id} with the specified filters")
            return {
                "tool_id": tool_id,
                "regimes": {},
                "sample_size": 0,
                "timestamp": datetime.now().isoformat()
            }
        
        # Group signals by market regime
        regimes = {}
        signal_ids_by_regime = {}
        
        for signal in signals:
            regime = signal.market_regime
            if regime not in regimes:
                regimes[regime] = []
                signal_ids_by_regime[regime] = []
                
            regimes[regime].append(signal)
            signal_ids_by_regime[regime].append(signal.signal_id)
        
        # Calculate metrics for each regime
        regime_metrics = {}
        
        for regime, regime_signals in regimes.items():
            # Get outcomes for these signals
            signal_ids = signal_ids_by_regime[regime]
            outcomes = []
            
            for signal_id in signal_ids:
                outcomes.extend(self.repository.get_outcomes_for_signal(signal_id))
            
            # Skip if no outcomes found
            if not outcomes:
                continue
                
            # Calculate win rate
            success_count = sum(1 for o in outcomes if o.success)
            win_rate = (success_count / len(outcomes)) * 100 if outcomes else 0
            
            # Calculate profit factor if profit data available
            profitable_outcomes = [o for o in outcomes if o.realized_profit > 0]
            losing_outcomes = [o for o in outcomes if o.realized_profit < 0]
            
            profit_sum = sum(o.realized_profit for o in profitable_outcomes)
            loss_sum = abs(sum(o.realized_profit for o in losing_outcomes))
            profit_factor = profit_sum / loss_sum if loss_sum > 0 else float('inf')
            
            # Calculate expected payoff
            expected_payoff = sum(o.realized_profit for o in outcomes) / len(outcomes) if outcomes else 0
            
            # Store metrics for this regime
            regime_metrics[regime] = {
                "win_rate": win_rate,
                "profit_factor": profit_factor if profit_factor != float('inf') else None,
                "expected_payoff": expected_payoff,
                "sample_size": len(outcomes),
                "signal_count": len(regime_signals),
                "outcome_count": len(outcomes)
            }
        
        # Calculate overall performance metrics
        all_outcomes = []
        for regime_signal_ids in signal_ids_by_regime.values():
            for signal_id in regime_signal_ids:
                all_outcomes.extend(self.repository.get_outcomes_for_signal(signal_id))
        
        overall_success_count = sum(1 for o in all_outcomes if o.success)
        overall_win_rate = (overall_success_count / len(all_outcomes)) * 100 if all_outcomes else 0
        
        return {
            "tool_id": tool_id,
            "regimes": regime_metrics,
            "overall_win_rate": overall_win_rate,
            "sample_size": len(all_outcomes),
            "signal_count": len(signals),
            "timestamp": datetime.now().isoformat()
        }
    
    def find_optimal_market_conditions(
        self,
        tool_id: str,
        min_sample_size: int = 10,
        timeframe: Optional[str] = None,
        instrument: Optional[str] = None,
        from_date: Optional[datetime] = None,
        to_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Find the optimal market conditions for a specific tool
        
        Args:
            tool_id: ID of the trading tool to analyze
            min_sample_size: Minimum number of signals required for reliable analysis
            timeframe: Optional filter by timeframe
            instrument: Optional filter by trading instrument
            from_date: Optional start date for analysis
            to_date: Optional end date for analysis
            
        Returns:
            Dictionary with optimal market conditions for the tool
        """
        # Get performance matrix
        performance_matrix = self.get_regime_performance_matrix(
            tool_id=tool_id,
            timeframe=timeframe,
            instrument=instrument,
            from_date=from_date,
            to_date=to_date
        )
        
        # Find best regime based on win rate with minimum sample size
        best_regime = None
        best_win_rate = 0
        
        for regime, metrics in performance_matrix.get("regimes", {}).items():
            if metrics.get("sample_size", 0) >= min_sample_size:
                win_rate = metrics.get("win_rate", 0)
                if win_rate > best_win_rate:
                    best_win_rate = win_rate
                    best_regime = regime
        
        # Find worst regime based on win rate with minimum sample size
        worst_regime = None
        worst_win_rate = 100  # Start with 100%
        
        for regime, metrics in performance_matrix.get("regimes", {}).items():
            if metrics.get("sample_size", 0) >= min_sample_size:
                win_rate = metrics.get("win_rate", 0)
                if win_rate < worst_win_rate:
                    worst_win_rate = win_rate
                    worst_regime = regime
        
        # Calculate regime reliability variance (consistency across regimes)
        win_rates = [
            metrics.get("win_rate", 0) 
            for metrics in performance_matrix.get("regimes", {}).values()
            if metrics.get("sample_size", 0) >= min_sample_size
        ]
        
        regime_variance = np.var(win_rates) if len(win_rates) > 1 else 0
        
        # Determine if tool is regime-specific or general purpose
        is_regime_specific = regime_variance > 15  # Variance threshold for regime specificity
        
        return {
            "tool_id": tool_id,
            "best_regime": {
                "regime": best_regime,
                "win_rate": best_win_rate,
                "metrics": performance_matrix.get("regimes", {}).get(best_regime, {}) if best_regime else None
            },
            "worst_regime": {
                "regime": worst_regime,
                "win_rate": worst_win_rate,
                "metrics": performance_matrix.get("regimes", {}).get(worst_regime, {}) if worst_regime else None
            },
            "is_regime_specific": is_regime_specific,
            "regime_variance": regime_variance,
            "performance_matrix": performance_matrix,
            "timestamp": datetime.now().isoformat()
        }
    
    def compute_tool_complementarity(
        self,
        tool_ids: List[str],
        timeframe: Optional[str] = None,
        instrument: Optional[str] = None,
        from_date: Optional[datetime] = None,
        to_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Analyze how well different tools complement each other across market regimes
        
        Args:
            tool_ids: List of tool IDs to analyze
            timeframe: Optional filter by timeframe
            instrument: Optional filter by trading instrument
            from_date: Optional start date for analysis
            to_date: Optional end date for analysis
            
        Returns:
            Dictionary with complementarity analysis results
        """
        # Get performance matrix for each tool
        tool_matrices = {}
        for tool_id in tool_ids:
            matrix = self.get_regime_performance_matrix(
                tool_id=tool_id,
                timeframe=timeframe,
                instrument=instrument,
                from_date=from_date,
                to_date=to_date
            )
            tool_matrices[tool_id] = matrix
        
        # Find complementary tools for each market regime
        regime_coverage = {}
        all_regimes = set()
        
        # Collect all regimes from all tools
        for matrix in tool_matrices.values():
            for regime in matrix.get("regimes", {}).keys():
                all_regimes.add(regime)
        
        # For each regime, find the best tool
        for regime in all_regimes:
            best_tool = None
            best_win_rate = 0
            
            for tool_id, matrix in tool_matrices.items():
                if regime in matrix.get("regimes", {}):
                    win_rate = matrix["regimes"][regime].get("win_rate", 0)
                    sample_size = matrix["regimes"][regime].get("sample_size", 0)
                    
                    # Only consider tools with sufficient sample size
                    if sample_size >= 10 and win_rate > best_win_rate:
                        best_win_rate = win_rate
                        best_tool = tool_id
            
            regime_coverage[regime] = {
                "best_tool": best_tool,
                "win_rate": best_win_rate
            }
        
        # Calculate overall coverage and gaps
        covered_regimes = [regime for regime, data in regime_coverage.items() if data["best_tool"] is not None]
        coverage_percentage = (len(covered_regimes) / len(all_regimes)) * 100 if all_regimes else 0
        
        # Find optimal tool combination
        optimal_combination = self._find_minimal_tool_set(regime_coverage, tool_matrices)
        
        return {
            "tool_ids": tool_ids,
            "regime_coverage": regime_coverage,
            "all_regimes": list(all_regimes),
            "coverage_percentage": coverage_percentage,
            "uncovered_regimes": [r for r in all_regimes if r not in covered_regimes],
            "optimal_combination": optimal_combination,
            "timestamp": datetime.now().isoformat()
        }
    
    def _find_minimal_tool_set(self, regime_coverage, tool_matrices):
        """Find the minimal set of tools that provides the best coverage"""
        # Count how many regimes each tool covers as the best tool
        tool_regime_counts = {}
        for regime, data in regime_coverage.items():
            tool = data["best_tool"]
            if tool:
                if tool not in tool_regime_counts:
                    tool_regime_counts[tool] = {
                        "regimes": [],
                        "count": 0
                    }
                tool_regime_counts[tool]["regimes"].append(regime)
                tool_regime_counts[tool]["count"] += 1
        
        # Sort tools by number of regimes covered
        sorted_tools = sorted(
            tool_regime_counts.items(),
            key=lambda x: x[1]["count"],
            reverse=True
        )
        
        # Take the minimal set that covers all regimes
        covered_regimes = set()
        optimal_combination = []
        
        for tool, data in sorted_tools:
            new_regimes = set(data["regimes"]) - covered_regimes
            if new_regimes:
                optimal_combination.append({
                    "tool_id": tool,
                    "additional_regimes_covered": list(new_regimes),
                    "performance_matrix": tool_matrices.get(tool, {})
                })
                covered_regimes.update(new_regimes)
        
        return optimal_combination
    
    def generate_performance_report(
        self,
        timeframe: Optional[str] = None,
        instrument: Optional[str] = None,
        from_date: Optional[datetime] = None,
        to_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Generate a comprehensive performance report for all tools
        
        Args:
            timeframe: Optional filter by timeframe
            instrument: Optional filter by trading instrument
            from_date: Optional start date for analysis
            to_date: Optional end date for analysis
            
        Returns:
            Dictionary with comprehensive performance report
        """
        # Get all tools from repository
        tools = self.repository.get_tools()
        
        if not tools:
            self.logger.warning("No trading tools found in the repository")
            return {
                "tools": [],
                "timestamp": datetime.now().isoformat()
            }
        
        # Generate report for each tool
        tool_reports = []
        all_regimes = set()
        
        for tool in tools:
            # Get optimal market conditions for this tool
            tool_report = self.find_optimal_market_conditions(
                tool_id=tool.tool_id,
                timeframe=timeframe,
                instrument=instrument,
                from_date=from_date,
                to_date=to_date
            )
            
            # Add to report
            tool_reports.append({
                "tool_id": tool.tool_id,
                "tool_name": tool.name,
                "tool_description": tool.description,
                "best_regime": tool_report.get("best_regime", {}),
                "worst_regime": tool_report.get("worst_regime", {}),
                "is_regime_specific": tool_report.get("is_regime_specific", False),
                "regime_variance": tool_report.get("regime_variance", 0),
                "overall_win_rate": tool_report["performance_matrix"].get("overall_win_rate", 0),
                "sample_size": tool_report["performance_matrix"].get("sample_size", 0)
            })
            
            # Collect all regimes
            for regime in tool_report["performance_matrix"].get("regimes", {}).keys():
                all_regimes.add(regime)
        
        # Sort tools by overall win rate
        sorted_tools = sorted(
            tool_reports,
            key=lambda x: x.get("overall_win_rate", 0),
            reverse=True
        )
        
        # Find best tool for each regime
        best_tools_by_regime = {}
        
        for regime in all_regimes:
            best_tool = None
            best_win_rate = 0
            
            for tool_report in tool_reports:
                performance_matrix = self.get_regime_performance_matrix(tool_report["tool_id"])
                if regime in performance_matrix.get("regimes", {}):
                    win_rate = performance_matrix["regimes"][regime].get("win_rate", 0)
                    sample_size = performance_matrix["regimes"][regime].get("sample_size", 0)
                    
                    if sample_size >= 10 and win_rate > best_win_rate:
                        best_win_rate = win_rate
                        best_tool = tool_report["tool_id"]
            
            best_tools_by_regime[regime] = {
                "tool_id": best_tool,
                "win_rate": best_win_rate
            }
        
        # Calculate regime coverage
        covered_regimes = [r for r, data in best_tools_by_regime.items() if data["tool_id"] is not None]
        coverage_percentage = (len(covered_regimes) / len(all_regimes)) * 100 if all_regimes else 0
        
        return {
            "tools": sorted_tools,
            "regimes": list(all_regimes),
            "best_tools_by_regime": best_tools_by_regime,
            "covered_regimes": covered_regimes,
            "uncovered_regimes": [r for r in all_regimes if r not in covered_regimes],
            "coverage_percentage": coverage_percentage,
            "timestamp": datetime.now().isoformat()
        }
    
    def recommend_tools_for_current_regime(
        self,
        current_regime: str,
        instrument: Optional[str] = None,
        timeframe: Optional[str] = None,
        min_sample_size: int = 10,
        min_win_rate: float = 50.0,
        top_n: int = 3
    ) -> Dict[str, Any]:
        """
        Recommend the best trading tools for the current market regime
        
        Args:
            current_regime: The current market regime (e.g., 'trending_up', 'ranging')
            instrument: Optional filter by trading instrument
            timeframe: Optional filter by timeframe
            min_sample_size: Minimum number of signals required for reliable analysis
            min_win_rate: Minimum win rate percentage to consider a tool viable
            top_n: Number of top tools to recommend
            
        Returns:
            Dictionary with tool recommendations for the current regime
        """
        # Get all tools from repository
        tools = self.repository.get_tools()
        
        if not tools:
            self.logger.warning("No trading tools found in the repository")
            return {
                "regime": current_regime,
                "recommendations": [],
                "timestamp": datetime.now().isoformat()
            }
        
        # Analyze each tool for this specific regime
        regime_tools = []
        
        for tool in tools:
            performance_matrix = self.get_regime_performance_matrix(
                tool_id=tool.tool_id,
                timeframe=timeframe,
                instrument=instrument
            )
            
            # Check if this tool has data for the current regime
            if current_regime in performance_matrix.get("regimes", {}):
                regime_metrics = performance_matrix["regimes"][current_regime]
                
                # Only consider tools with sufficient sample size and minimum win rate
                if (regime_metrics.get("sample_size", 0) >= min_sample_size and 
                    regime_metrics.get("win_rate", 0) >= min_win_rate):
                    
                    # Add to list of viable tools
                    regime_tools.append({
                        "tool_id": tool.tool_id,
                        "tool_name": tool.name,
                        "tool_description": tool.description,
                        "win_rate": regime_metrics.get("win_rate", 0),
                        "profit_factor": regime_metrics.get("profit_factor"),
                        "expected_payoff": regime_metrics.get("expected_payoff", 0),
                        "sample_size": regime_metrics.get("sample_size", 0),
                        "category": tool.category
                    })
        
        # Sort tools by win rate (primary) and then by profit factor (secondary)
        sorted_tools = sorted(
            regime_tools,
            key=lambda x: (x.get("win_rate", 0), x.get("profit_factor") or 0),
            reverse=True
        )
        
        # Take top N tools
        top_tools = sorted_tools[:top_n] if len(sorted_tools) > top_n else sorted_tools
        
        # Group tools by category
        tools_by_category = {}
        for tool in regime_tools:
            category = tool.get("category", "Uncategorized")
            if category not in tools_by_category:
                tools_by_category[category] = []
            tools_by_category[category].append(tool)
        
        # Get top tool for each category
        top_by_category = {}
        for category, cat_tools in tools_by_category.items():
            sorted_cat_tools = sorted(
                cat_tools,
                key=lambda x: (x.get("win_rate", 0), x.get("profit_factor") or 0),
                reverse=True
            )
            top_by_category[category] = sorted_cat_tools[0] if sorted_cat_tools else None
        
        return {
            "regime": current_regime,
            "recommendations": top_tools,
            "total_viable_tools": len(regime_tools),
            "top_by_category": top_by_category,
            "timestamp": datetime.now().isoformat()
        }
    
    def analyze_effectiveness_trends(
        self,
        tool_id: str,
        timeframe: Optional[str] = None,
        instrument: Optional[str] = None,
        period_days: int = 30,
        look_back_periods: int = 6
    ) -> Dict[str, Any]:
        """
        Analyze how the effectiveness of a tool has changed over time
        
        Args:
            tool_id: ID of the trading tool to analyze
            timeframe: Optional filter by timeframe
            instrument: Optional filter by trading instrument
            period_days: Number of days in each analysis period
            look_back_periods: Number of historical periods to analyze
            
        Returns:
            Dictionary with effectiveness trend analysis results
        """
        # Calculate the date ranges for each period
        end_date = datetime.now()
        period_ranges = []
        
        for i in range(look_back_periods):
            period_end = end_date - timedelta(days=i*period_days)
            period_start = period_end - timedelta(days=period_days)
            period_ranges.append((period_start, period_end))
        
        # Get metrics for each period
        period_metrics = []
        
        for period_idx, (start_date, end_date) in enumerate(period_ranges):
            period_name = f"Period {look_back_periods - period_idx}"
            
            # Get performance for this period
            performance = self.get_regime_performance_matrix(
                tool_id=tool_id,
                timeframe=timeframe,
                instrument=instrument,
                from_date=start_date,
                to_date=end_date
            )
            
            # Add period data
            period_metrics.append({
                "period_name": period_name,
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "overall_win_rate": performance.get("overall_win_rate", 0),
                "sample_size": performance.get("sample_size", 0),
                "regime_metrics": performance.get("regimes", {}),
            })
        
        # Calculate win rate trend
        win_rate_trend = []
        for period in period_metrics:
            if period["sample_size"] >= 5:  # Only consider periods with sufficient data
                win_rate_trend.append({
                    "period": period["period_name"],
                    "win_rate": period["overall_win_rate"]
                })
        
        # Calculate trend direction
        trend_direction = "stable"
        if len(win_rate_trend) >= 3:
            # Get the three most recent periods
            recent_rates = [p["win_rate"] for p in win_rate_trend[:3]]
            
            if all(recent_rates[i] > recent_rates[i+1] for i in range(len(recent_rates)-1)):
                trend_direction = "improving"
            elif all(recent_rates[i] < recent_rates[i+1] for i in range(len(recent_rates)-1)):
                trend_direction = "deteriorating"
        
        # Calculate regime consistency
        regime_consistency = {}
        
        for period in period_metrics:
            for regime, metrics in period.get("regime_metrics", {}).items():
                if regime not in regime_consistency:
                    regime_consistency[regime] = []
                
                if metrics.get("sample_size", 0) >= 5:  # Only consider regimes with sufficient data
                    regime_consistency[regime].append({
                        "period": period["period_name"],
                        "win_rate": metrics.get("win_rate", 0)
                    })
        
        # Calculate variance in win rate for each regime
        regime_variance = {}
        for regime, data in regime_consistency.items():
            if len(data) >= 3:
                win_rates = [item["win_rate"] for item in data]
                regime_variance[regime] = {
                    "variance": np.var(win_rates),
                    "is_consistent": np.var(win_rates) < 10  # Threshold for consistency
                }
        
        return {
            "tool_id": tool_id,
            "period_metrics": period_metrics,
            "win_rate_trend": win_rate_trend,
            "trend_direction": trend_direction,
            "regime_consistency": {k: v for k, v in regime_consistency.items() if k in regime_variance},
            "regime_variance": regime_variance,
            "timestamp": datetime.now().isoformat()
        }
    
    def get_underperforming_tools(
        self,
        win_rate_threshold: float = 50.0,
        min_sample_size: int = 20,
        timeframe: Optional[str] = None,
        instrument: Optional[str] = None,
        from_date: Optional[datetime] = None,
        to_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Identify underperforming trading tools that may need optimization or retirement
        
        Args:
            win_rate_threshold: Win rate threshold below which a tool is considered underperforming
            min_sample_size: Minimum sample size for reliable analysis
            timeframe: Optional filter by timeframe
            instrument: Optional filter by trading instrument
            from_date: Optional start date for analysis
            to_date: Optional end date for analysis
            
        Returns:
            Dictionary with information about underperforming tools
        """
        # Get all tools from repository
        tools = self.repository.get_tools()
        
        if not tools:
            self.logger.warning("No trading tools found in the repository")
            return {
                "underperforming_tools": [],
                "timestamp": datetime.now().isoformat()
            }
        
        # Check each tool's performance
        underperforming = []
        
        for tool in tools:
            # Get performance matrix
            performance = self.get_regime_performance_matrix(
                tool_id=tool.tool_id,
                timeframe=timeframe,
                instrument=instrument,
                from_date=from_date,
                to_date=to_date
            )
            
            # Check if tool has sufficient data and is underperforming
            overall_win_rate = performance.get("overall_win_rate", 0)
            sample_size = performance.get("sample_size", 0)
            
            if sample_size >= min_sample_size and overall_win_rate < win_rate_threshold:
                # Check if the tool performs well in any regime
                has_good_regime = False
                best_regime = None
                best_win_rate = 0
                
                for regime, metrics in performance.get("regimes", {}).items():
                    if (metrics.get("sample_size", 0) >= min_sample_size / 2 and 
                        metrics.get("win_rate", 0) >= win_rate_threshold):
                        has_good_regime = True
                        if metrics.get("win_rate", 0) > best_win_rate:
                            best_win_rate = metrics.get("win_rate", 0)
                            best_regime = regime
                
                # Add to underperforming list
                underperforming.append({
                    "tool_id": tool.tool_id,
                    "tool_name": tool.name,
                    "overall_win_rate": overall_win_rate,
                    "sample_size": sample_size,
                    "has_good_regime": has_good_regime,
                    "best_regime": best_regime,
                    "best_regime_win_rate": best_win_rate if best_regime else None,
                    "recommendation": "Optimize for specific regime" if has_good_regime else "Consider retirement"
                })
        
        # Sort by overall win rate (ascending)
        sorted_underperforming = sorted(
            underperforming,
            key=lambda x: x.get("overall_win_rate", 0)
        )
        
        return {
            "underperforming_tools": sorted_underperforming,
            "count": len(sorted_underperforming),
            "timestamp": datetime.now().isoformat()
        }
