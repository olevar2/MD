"""
Tool Effectiveness Dashboard Generator

This module provides functionality to generate dashboard data and visualizations 
for the Tool Effectiveness metrics framework.
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import json
import pandas as pd
import numpy as np

from analysis_engine.repositories.tool_effectiveness_repository import ToolEffectivenessRepository
from analysis_engine.services.market_regime_analysis import MarketRegimeAnalysisService


class ToolEffectivenessDashboardGenerator:
    """Generates dashboard data for effectiveness visualization"""
    
    def __init__(self, repository: ToolEffectivenessRepository):
        self.repository = repository
        self.market_regime_service = MarketRegimeAnalysisService(repository)
        self.logger = logging.getLogger(__name__)
    
    def generate_overview_data(self) -> Dict[str, Any]:
        """
        Generate overview data for the effectiveness dashboard
        
        Returns:
            Dictionary with overview metrics for all tools
        """
        # Get all tools from repository
        tools = self.repository.get_tools()
        
        if not tools:
            self.logger.warning("No trading tools found in the repository")
            return {
                "tools_count": 0,
                "signals_count": 0,
                "outcomes_count": 0,
                "timestamp": datetime.now().isoformat()
            }
        
        # Count signals and outcomes
        signals_count = 0
        outcomes_count = 0
        success_count = 0
        
        for tool in tools:
            # Get signals for this tool
            tool_signals = self.repository.get_signals(tool_id=tool.tool_id)
            signals_count += len(tool_signals)
            
            # Get outcomes for these signals
            for signal in tool_signals:
                signal_outcomes = self.repository.get_outcomes_for_signal(signal.signal_id)
                outcomes_count += len(signal_outcomes)
                success_count += sum(1 for o in signal_outcomes if o.success)
        
        # Calculate overall success rate
        overall_success_rate = (success_count / outcomes_count) * 100 if outcomes_count > 0 else 0
        
        # Get the top 5 performing tools based on win rate
        top_tools = []
        for tool in tools:
            win_rate, signal_count, outcome_count = self.repository.get_tool_win_rate(tool.tool_id)
            if outcome_count >= 10:  # Minimum sample size for reliable stats
                top_tools.append({
                    "tool_id": tool.tool_id,
                    "name": tool.name,
                    "win_rate": win_rate,
                    "signal_count": signal_count,
                    "outcome_count": outcome_count
                })
        
        # Sort by win rate
        top_tools = sorted(top_tools, key=lambda x: x["win_rate"], reverse=True)[:5]
        
        return {
            "tools_count": len(tools),
            "signals_count": signals_count,
            "outcomes_count": outcomes_count,
            "overall_success_rate": overall_success_rate,
            "top_performing_tools": top_tools,
            "timestamp": datetime.now().isoformat()
        }
    
    def generate_tool_comparison_data(
        self,
        tool_ids: Optional[List[str]] = None,
        timeframe: Optional[str] = None,
        instrument: Optional[str] = None,
        from_date: Optional[datetime] = None,
        to_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Generate comparison data for multiple tools
        
        Args:
            tool_ids: Optional list of tool IDs to compare. If None, includes all tools.
            timeframe: Optional filter by timeframe
            instrument: Optional filter by trading instrument
            from_date: Optional start date for analysis
            to_date: Optional end date for analysis
            
        Returns:
            Dictionary with comparison data
        """
        # Get tools to compare
        if tool_ids:
            tools = [self.repository.get_tool(tool_id) for tool_id in tool_ids]
            tools = [t for t in tools if t]  # Filter out None values
        else:
            tools = self.repository.get_tools()
        
        if not tools:
            self.logger.warning("No tools found for comparison")
            return {
                "tools": [],
                "metrics": [],
                "timestamp": datetime.now().isoformat()
            }
        
        # Prepare comparison data
        comparison_data = []
        
        for tool in tools:
            # Get win rate
            win_rate, signal_count, outcome_count = self.repository.get_tool_win_rate(
                tool_id=tool.tool_id,
                timeframe=timeframe,
                instrument=instrument,
                from_date=from_date,
                to_date=to_date
            )
            
            # Skip tools with insufficient data
            if outcome_count < 5:
                continue
                
            # Get latest metrics for this tool
            latest_metrics = self.repository.get_latest_tool_metrics(tool.tool_id)
            
            # Prepare tool data
            tool_data = {
                "tool_id": tool.tool_id,
                "name": tool.name,
                "description": tool.description,
                "win_rate": win_rate,
                "profit_factor": latest_metrics.get("profit_factor"),
                "expected_payoff": latest_metrics.get("expected_payoff"),
                "signal_count": signal_count,
                "outcome_count": outcome_count
            }
            
            comparison_data.append(tool_data)
        
        # Sort by win rate
        comparison_data = sorted(comparison_data, key=lambda x: x["win_rate"], reverse=True)
        
        return {
            "tools": comparison_data,
            "metrics": ["win_rate", "profit_factor", "expected_payoff"],
            "timestamp": datetime.now().isoformat()
        }
    
    def generate_regime_analysis_data(
        self,
        tool_id: str,
        timeframe: Optional[str] = None,
        instrument: Optional[str] = None,
        from_date: Optional[datetime] = None,
        to_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Generate market regime analysis data for a specific tool
        
        Args:
            tool_id: ID of the trading tool to analyze
            timeframe: Optional filter by timeframe
            instrument: Optional filter by trading instrument
            from_date: Optional start date for analysis
            to_date: Optional end date for analysis
            
        Returns:
            Dictionary with regime analysis data
        """
        # Get tool details
        tool = self.repository.get_tool(tool_id)
        
        if not tool:
            self.logger.warning(f"Tool {tool_id} not found")
            return {
                "tool_id": tool_id,
                "regimes": {},
                "timestamp": datetime.now().isoformat()
            }
        
        # Get regime performance matrix
        performance_matrix = self.market_regime_service.get_regime_performance_matrix(
            tool_id=tool_id,
            timeframe=timeframe,
            instrument=instrument,
            from_date=from_date,
            to_date=to_date
        )
        
        # Find optimal market conditions
        optimal_conditions = self.market_regime_service.find_optimal_market_conditions(
            tool_id=tool_id,
            timeframe=timeframe,
            instrument=instrument,
            from_date=from_date,
            to_date=to_date
        )
        
        # Format data for visualization
        regime_data = []
        for regime, metrics in performance_matrix.get("regimes", {}).items():
            regime_data.append({
                "regime": regime,
                "win_rate": metrics.get("win_rate", 0),
                "profit_factor": metrics.get("profit_factor"),
                "expected_payoff": metrics.get("expected_payoff", 0),
                "sample_size": metrics.get("sample_size", 0)
            })
        
        # Sort by win rate
        regime_data = sorted(regime_data, key=lambda x: x["win_rate"], reverse=True)
        
        return {
            "tool_id": tool_id,
            "tool_name": tool.name,
            "tool_description": tool.description,
            "regime_data": regime_data,
            "is_regime_specific": optimal_conditions.get("is_regime_specific", False),
            "best_regime": optimal_conditions.get("best_regime", {}),
            "worst_regime": optimal_conditions.get("worst_regime", {}),
            "regime_variance": optimal_conditions.get("regime_variance", 0),
            "timestamp": datetime.now().isoformat()
        }
    
    def generate_performance_trends(
        self,
        tool_id: str,
        period_days: int = 90,
        interval_days: int = 7
    ) -> Dict[str, Any]:
        """
        Generate performance trend data over time for a specific tool
        
        Args:
            tool_id: ID of the trading tool to analyze
            period_days: Number of days to analyze
            interval_days: Size of each interval in days
            
        Returns:
            Dictionary with trend data
        """
        # Get tool details
        tool = self.repository.get_tool(tool_id)
        
        if not tool:
            self.logger.warning(f"Tool {tool_id} not found")
            return {
                "tool_id": tool_id,
                "trends": [],
                "timestamp": datetime.now().isoformat()
            }
        
        # Calculate date ranges for analysis
        end_date = datetime.now()
        start_date = end_date - timedelta(days=period_days)
        
        # Get all metrics for this tool in the date range
        metrics = self.repository.get_effectiveness_metrics(
            tool_id=tool_id,
            from_date=start_date,
            to_date=end_date
        )
        
        # Group metrics by date and type
        metrics_by_date = {}
        
        for metric in metrics:
            # Use the end date of the metric period
            date_key = metric.end_date.strftime("%Y-%m-%d")
            if date_key not in metrics_by_date:
                metrics_by_date[date_key] = {}
                
            metrics_by_date[date_key][metric.metric_type] = metric.value
        
        # Generate trend data for each interval
        trend_data = []
        current_date = start_date
        
        while current_date <= end_date:
            interval_end = min(current_date + timedelta(days=interval_days), end_date)
            
            # Calculate win rate for this interval
            interval_signals = self.repository.get_signals(
                tool_id=tool_id,
                from_date=current_date,
                to_date=interval_end
            )
            
            signal_ids = [s.signal_id for s in interval_signals]
            outcomes = []
            
            for signal_id in signal_ids:
                outcomes.extend(self.repository.get_outcomes_for_signal(signal_id))
            
            success_count = sum(1 for o in outcomes if o.success)
            win_rate = (success_count / len(outcomes) * 100) if outcomes else None
            
            # Add entry for this interval
            trend_data.append({
                "interval_start": current_date.isoformat(),
                "interval_end": interval_end.isoformat(),
                "win_rate": win_rate,
                "signal_count": len(interval_signals),
                "outcome_count": len(outcomes)
            })
            
            # Move to next interval
            current_date = interval_end + timedelta(days=1)
        
        return {
            "tool_id": tool_id,
            "tool_name": tool.name,
            "period_days": period_days,
            "interval_days": interval_days,
            "trends": trend_data,
            "timestamp": datetime.now().isoformat()
        }
    
    def generate_dashboard_data(
        self,
        timeframe: Optional[str] = None,
        instrument: Optional[str] = None,
        from_date: Optional[datetime] = None,
        to_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Generate complete dashboard data for all tools and metrics
        
        Args:
            timeframe: Optional filter by timeframe
            instrument: Optional filter by trading instrument
            from_date: Optional start date for analysis
            to_date: Optional end date for analysis
            
        Returns:
            Dictionary with complete dashboard data
        """
        # Get overview data
        overview_data = self.generate_overview_data()
        
        # Get tool comparison data
        comparison_data = self.generate_tool_comparison_data(
            timeframe=timeframe,
            instrument=instrument,
            from_date=from_date,
            to_date=to_date
        )
        
        # Get tools
        tools = self.repository.get_tools()
        
        # Generate regime analysis for each tool with sufficient data
        regime_analyses = []
        performance_trends = []
        
        for tool in tools:
            # Check if tool has sufficient data
            _, signal_count, outcome_count = self.repository.get_tool_win_rate(tool.tool_id)
            
            if outcome_count >= 10:  # Minimum sample size for reliable analysis
                # Generate regime analysis
                regime_analysis = self.generate_regime_analysis_data(
                    tool_id=tool.tool_id,
                    timeframe=timeframe,
                    instrument=instrument,
                    from_date=from_date,
                    to_date=to_date
                )
                regime_analyses.append(regime_analysis)
                
                # Generate performance trends
                trend_data = self.generate_performance_trends(tool.tool_id)
                performance_trends.append(trend_data)
        
        # Calculate tool complementarity
        tool_ids = [tool.tool_id for tool in tools]
        complementarity = self.market_regime_service.compute_tool_complementarity(
            tool_ids=tool_ids,
            timeframe=timeframe,
            instrument=instrument,
            from_date=from_date,
            to_date=to_date
        )
        
        # Collect available filters
        timeframes = set()
        instruments = set()
        regimes = set()
        
        for tool in tools:
            signals = self.repository.get_signals(tool_id=tool.tool_id)
            for signal in signals:
                if signal.timeframe:
                    timeframes.add(signal.timeframe)
                if signal.instrument:
                    instruments.add(signal.instrument)
                if signal.market_regime:
                    regimes.add(signal.market_regime)
        
        # Create complete dashboard data
        dashboard_data = {
            "generated_at": datetime.now().isoformat(),
            "overview": overview_data,
            "tool_comparison": comparison_data,
            "regime_analyses": regime_analyses,
            "performance_trends": performance_trends,
            "tool_complementarity": complementarity,
            "filters": {
                "timeframes": list(timeframes),
                "instruments": list(instruments),
                "regimes": list(regimes)
            }
        }
        
        return dashboard_data
    
    def export_dashboard_json(self, output_path: str) -> bool:
        """
        Export complete dashboard data to a JSON file
        
        Args:
            output_path: Path where the JSON file will be saved
            
        Returns:
            True if successful, False otherwise
        """
        try:
            dashboard_data = self.generate_dashboard_data()
            
            with open(output_path, 'w') as f:
                json.dump(dashboard_data, f, indent=2, default=str)
                
            self.logger.info(f"Dashboard data exported to {output_path}")
            return True
        except Exception as e:
            self.logger.error(f"Error exporting dashboard data: {str(e)}")
            return False
