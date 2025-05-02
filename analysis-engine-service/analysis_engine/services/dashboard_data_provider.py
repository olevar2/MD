"""
Dashboard Data Provider

This module provides aggregated effectiveness metrics data for dashboards,
offering trend analysis, tool comparisons, and market regime insights.
"""

import logging
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
from sqlalchemy.orm import Session

from analysis_engine.repositories.tool_effectiveness_repository import ToolEffectivenessRepository
from analysis_engine.services.enhanced_tool_effectiveness import EnhancedToolEffectivenessTracker
from analysis_engine.services.tool_effectiveness import MarketRegime, TimeFrame


class DashboardDataProvider:
    """
    Provides aggregated data for tool effectiveness dashboards
    """
    
    def __init__(self, db: Session):
        self.db = db
        self.repository = ToolEffectivenessRepository(db)
        self.tracker = EnhancedToolEffectivenessTracker()
        self.logger = logging.getLogger(__name__)
        
    def get_summary_data(self) -> Dict[str, Any]:
        """
        Get summary data for dashboard overview
        
        Returns:
            Dictionary with summary statistics for all tools
        """
        try:
            # Get tool statistics
            tools = self.repository.get_all_tools()
            
            # Get last 24 hours performance
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(hours=24)
            
            recent_signals = self.repository.count_signals(start_date=start_date)
            recent_outcomes = self.repository.count_outcomes(start_date=start_date)
            
            # Get overall win rate
            overall_win_rate = self.repository.get_overall_win_rate()
            
            # Get top performers
            top_performers = self._get_top_performers(limit=5)
            
            # Prepare summary data
            summary = {
                "total_tools": len(tools),
                "active_tools": sum(1 for t in tools if t.is_active),
                "recent_signals": recent_signals,
                "recent_outcomes": recent_outcomes,
                "overall_win_rate": overall_win_rate,
                "top_performers": top_performers,
                "generated_at": datetime.utcnow().isoformat()
            }
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error generating dashboard summary data: {str(e)}")
            return {
                "error": f"Failed to generate summary data: {str(e)}",
                "generated_at": datetime.utcnow().isoformat()
            }
            
    def get_tool_comparison_data(
        self, 
        tool_ids: Optional[List[str]] = None,
        lookback_days: int = 30
    ) -> Dict[str, Any]:
        """
        Get comparative data for multiple tools
        
        Args:
            tool_ids: List of tool IDs to compare, or None for all tools
            lookback_days: Number of days to look back
            
        Returns:
            Dictionary with comparative metrics for specified tools
        """
        try:
            # Set time range
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(days=lookback_days)
            
            # If no tools specified, get the top 10 by activity
            if not tool_ids:
                most_active_tools = self.repository.get_most_active_tools(
                    start_date=start_date,
                    limit=10
                )
                tool_ids = [t.tool_id for t in most_active_tools]
                
            if not tool_ids:
                return {
                    "message": "No tools found for comparison",
                    "tools": [],
                    "lookback_days": lookback_days,
                    "generated_at": datetime.utcnow().isoformat()
                }
                
            # Get metrics for each tool
            tools_data = []
            
            for tool_id in tool_ids:
                # Get win rate metrics
                win_rate_metrics = self.repository.get_tool_metrics(
                    tool_id=tool_id,
                    metric_type="win_rate",
                    start_date=start_date,
                    end_date=end_date,
                    order_by_date=True
                )
                
                # Get profit factor metrics
                profit_factor_metrics = self.repository.get_tool_metrics(
                    tool_id=tool_id,
                    metric_type="profit_factor",
                    start_date=start_date,
                    end_date=end_date,
                    order_by_date=True
                )
                
                # Get latest composite score
                composite_score = self.repository.get_latest_composite_score(tool_id)
                
                # Get tool info
                tool_info = self.repository.get_tool(tool_id)
                
                # Format tool data
                tool_data = {
                    "tool_id": tool_id,
                    "name": tool_info.name if tool_info else tool_id,
                    "description": tool_info.description if tool_info else "",
                    "composite_score": composite_score,
                    "win_rate": {
                        "trend": [
                            {
                                "date": m.created_at.isoformat(),
                                "value": m.value,
                                "sample_size": m.sample_size
                            }
                            for m in win_rate_metrics
                        ],
                        "latest": win_rate_metrics[-1].value if win_rate_metrics else None
                    },
                    "profit_factor": {
                        "trend": [
                            {
                                "date": m.created_at.isoformat(),
                                "value": m.value,
                                "sample_size": m.sample_size
                            }
                            for m in profit_factor_metrics
                        ],
                        "latest": profit_factor_metrics[-1].value if profit_factor_metrics else None
                    }
                }
                
                tools_data.append(tool_data)
                
            # Return comparative data
            return {
                "tools": tools_data,
                "lookback_days": lookback_days,
                "tool_count": len(tools_data),
                "generated_at": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error generating tool comparison data: {str(e)}")
            return {
                "error": f"Failed to generate comparison data: {str(e)}",
                "tools": [],
                "lookback_days": lookback_days,
                "generated_at": datetime.utcnow().isoformat()
            }
            
    def get_regime_effectiveness_data(
        self, 
        tool_id: str,
        lookback_days: int = 90
    ) -> Dict[str, Any]:
        """
        Get effectiveness data across different market regimes
        
        Args:
            tool_id: Tool ID to analyze
            lookback_days: Number of days to look back
            
        Returns:
            Dictionary with regime-specific effectiveness data
        """
        try:
            # Set time range
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(days=lookback_days)
            
            # Get outcomes for this tool
            outcomes = self.repository.get_outcomes(
                tool_id=tool_id,
                start_date=start_date,
                end_date=end_date
            )
            
            if not outcomes:
                return {
                    "tool_id": tool_id,
                    "message": "No outcomes found for analysis",
                    "regimes": {},
                    "lookback_days": lookback_days,
                    "generated_at": datetime.utcnow().isoformat()
                }
                
            # Convert to domain models
            domain_outcomes = []
            for outcome in outcomes:
                signal = outcome.signal
                
                signal_event = {
                    "id": signal.signal_id,
                    "tool_name": signal.tool_id,
                    "signal_type": signal.signal_type,
                    "direction": signal.additional_data.get("direction", "unknown") if signal.additional_data else "unknown",
                    "strength": signal.confidence,
                    "timestamp": signal.timestamp,
                    "symbol": signal.instrument,
                    "timeframe": signal.timeframe,
                    "price_at_signal": signal.additional_data.get("price_at_signal") if signal.additional_data else None,
                    "market_context": {"regime": signal.market_regime}
                }
                
                domain_outcome = {
                    "signal_event": signal_event,
                    "outcome": "success" if outcome.success else "failure",
                    "exit_price": outcome.additional_data.get("exit_price") if outcome.additional_data else None,
                    "exit_timestamp": outcome.exit_timestamp,
                    "profit_loss": outcome.realized_profit,
                    "max_favorable_price": outcome.additional_data.get("max_favorable_price") if outcome.additional_data else None,
                    "max_adverse_price": outcome.additional_data.get("max_adverse_price") if outcome.additional_data else None
                }
                
                domain_outcomes.append(domain_outcome)
                
            # Get regime analysis
            regime_analysis = self.tracker.analyze_tool_by_market_regime(
                outcomes=domain_outcomes,
                tool_name=tool_id
            )
            
            # Get signal quality metrics for each regime
            regime_quality_metrics = {}
            
            for regime in MarketRegime:
                regime_outcomes = [
                    o for o in domain_outcomes
                    if o["signal_event"]["market_context"]["regime"] == regime.value
                ]
                
                if regime_outcomes:
                    quality_metrics = self.tracker.signal_quality_evaluator.evaluate_all_metrics(regime_outcomes)
                    regime_quality_metrics[regime.value] = quality_metrics
            
            # Return regime effectiveness data
            return {
                "tool_id": tool_id,
                "sample_size": len(domain_outcomes),
                "lookback_days": lookback_days,
                "regime_analysis": regime_analysis,
                "regime_quality_metrics": regime_quality_metrics,
                "generated_at": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error generating regime effectiveness data: {str(e)}")
            return {
                "error": f"Failed to generate regime data: {str(e)}",
                "tool_id": tool_id,
                "lookback_days": lookback_days,
                "generated_at": datetime.utcnow().isoformat()
            }
    
    def _get_top_performers(self, limit: int = 5) -> List[Dict[str, Any]]:
        """Get top performing tools based on composite score"""
        try:
            top_tools = self.repository.get_top_tools_by_composite_score(limit=limit)
            
            return [
                {
                    "tool_id": t.tool_id,
                    "name": t.name,
                    "composite_score": t.latest_score,
                    "win_rate": t.latest_win_rate
                }
                for t in top_tools
            ]
            
        except Exception as e:
            self.logger.error(f"Error getting top performers: {str(e)}")
            return []
