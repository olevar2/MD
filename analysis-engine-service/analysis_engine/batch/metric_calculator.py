"""
Metric Calculator

This module provides batch processing capabilities for calculating
and updating effectiveness metrics across multiple tools and timeframes.
"""

import time
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Set
import concurrent.futures
import logging
from sqlalchemy.orm import Session

from analysis_engine.repositories.tool_effectiveness_repository import ToolEffectivenessRepository


logger = logging.getLogger(__name__)


class MetricBatchCalculator:
    """
    Batch calculator for tool effectiveness metrics
    """
    
    def __init__(self, db: Session, max_workers: int = 4):
        self.repository = ToolEffectivenessRepository(db)
        self.max_workers = max_workers
    
    def recalculate_all_metrics(
        self,
        days_back: int = 90,
        tool_ids: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Recalculate metrics for all tools or specified tools"""
        start_time = time.time()
        
        # Get tools to process
        if tool_ids:
            tools = [self.repository.get_tool(tool_id) for tool_id in tool_ids]
            tools = [t for t in tools if t is not None]
        else:
            tools = self.repository.get_tools(limit=1000)
        
        if not tools:
            return {
                "status": "completed",
                "message": "No tools found for processing",
                "duration_seconds": time.time() - start_time,
                "tools_processed": 0
            }
        
        # Process tool metrics in parallel
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_tool = {
                executor.submit(self._recalculate_tool_metrics, tool.tool_id, days_back): tool.tool_id
                for tool in tools
            }
            
            for future in concurrent.futures.as_completed(future_to_tool):
                tool_id = future_to_tool[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logger.error(f"Error processing tool {tool_id}: {e}")
                    results.append({
                        "tool_id": tool_id,
                        "status": "error",
                        "error": str(e)
                    })
        
        return {
            "status": "completed",
            "duration_seconds": time.time() - start_time,
            "tools_processed": len(tools),
            "results": results
        }
    
    def _recalculate_tool_metrics(self, tool_id: str, days_back: int) -> Dict[str, Any]:
        """Recalculate metrics for a specific tool"""
        start_time = time.time()
        from_date = datetime.utcnow() - timedelta(days=days_back)
        
        tool = self.repository.get_tool(tool_id)
        if not tool:
            return {"tool_id": tool_id, "status": "error", "error": "Tool not found"}
        
        # Get all signals for this tool
        signals = self.repository.get_signals(
            tool_id=tool_id,
            from_date=from_date,
            limit=10000
        )
        
        if not signals:
            return {
                "tool_id": tool_id,
                "status": "completed",
                "message": "No signals found for processing",
                "signals_processed": 0,
                "duration_seconds": time.time() - start_time
            }
        
        # Process each signal
        processed_count = 0
        for signal in signals:
            try:
                self.repository._calculate_and_store_metrics(signal.signal_id)
                processed_count += 1
            except Exception as e:
                logger.error(f"Error processing signal {signal.signal_id}: {e}")
        
        return {
            "tool_id": tool_id,
            "tool_name": tool.name,
            "status": "completed",
            "signals_processed": processed_count,
            "duration_seconds": time.time() - start_time
        }
    
    def generate_periodic_reports(self, report_type: str = "monthly") -> Dict[str, Any]:
        """Generate periodic reports for all tools"""
        start_time = time.time()
        
        # Get all tools
        tools = self.repository.get_tools(limit=1000)
        
        if not tools:
            return {
                "status": "completed",
                "message": "No tools found for processing",
                "duration_seconds": time.time() - start_time,
                "reports_generated": 0
            }
        
        # Generate reports in parallel
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_tool = {
                executor.submit(self._generate_tool_report, tool.tool_id, report_type): tool.tool_id
                for tool in tools
            }
            
            for future in concurrent.futures.as_completed(future_to_tool):
                tool_id = future_to_tool[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logger.error(f"Error generating report for tool {tool_id}: {e}")
                    results.append({
                        "tool_id": tool_id,
                        "status": "error",
                        "error": str(e)
                    })
        
        return {
            "status": "completed",
            "duration_seconds": time.time() - start_time,
            "reports_generated": len(results),
            "results": results
        }
    
    def _generate_tool_report(self, tool_id: str, report_type: str) -> Dict[str, Any]:
        """Generate a report for a specific tool"""
        start_time = time.time()
        
        tool = self.repository.get_tool(tool_id)
        if not tool:
            return {"tool_id": tool_id, "status": "error", "error": "Tool not found"}
        
        # Define date range based on report type
        now = datetime.utcnow()
        if report_type == "weekly":
            from_date = now - timedelta(days=7)
            period = "week"
        elif report_type == "monthly":
            from_date = now - timedelta(days=30)
            period = "month"
        elif report_type == "quarterly":
            from_date = now - timedelta(days=90)
            period = "quarter"
        else:
            from_date = now - timedelta(days=30)
            period = "month"
        
        # Get signals and outcomes
        signals = self.repository.get_signals(
            tool_id=tool_id,
            from_date=from_date,
            limit=10000
        )
        
        outcomes = self.repository.get_outcomes(
            tool_id=tool_id,
            from_date=from_date,
            limit=10000
        )
        
        # Generate report data
        total_signals = len(signals)
        total_outcomes = len(outcomes)
        win_count = len([o for o in outcomes if o.pnl > 0])
        win_rate = win_count / total_outcomes if total_outcomes > 0 else 0
        total_pnl = sum(o.pnl for o in outcomes)
        
        # Get metrics
        metrics = self.repository.get_latest_metrics_for_tool(tool_id)
        metrics_dict = {m.metric_type: m.metric_value for m in metrics}
        
        # Create report
        report_data = {
            "report_id": str(tool_id) + "_" + report_type + "_" + now.strftime("%Y%m%d"),
            "tool_id": tool_id,
            "tool_name": tool.name,
            "report_type": report_type,
            "period": period,
            "generated_at": now,
            "period_start": from_date,
            "period_end": now,
            "summary": {
                "total_signals": total_signals,
                "total_outcomes": total_outcomes,
                "win_rate": win_rate,
                "total_pnl": total_pnl
            },
            "metrics": metrics_dict
        }
        
        # Save report
        self.repository.create_report({
            "report_id": report_data["report_id"],
            "tool_id": tool_id,
            "report_type": report_type,
            "timestamp": now,
            "report_data": report_data
        })
        
        return {
            "tool_id": tool_id,
            "tool_name": tool.name,
            "report_id": report_data["report_id"],
            "report_type": report_type,
            "status": "completed",
            "duration_seconds": time.time() - start_time
        }
