"""
Reconciliation reporting implementation.

This module provides functionality for generating reports from reconciliation data,
including summary reports, detailed reports, and chart reports.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from collections import defaultdict

from core.base import ReconciliationBase
from core_foundations.utils.logger import get_logger

logger = get_logger(__name__)


class ReconciliationReporting(ReconciliationBase):
    """
    Reconciliation reporting implementation.
    """
    
    def generate_summary_report(
        self,
        account_id: str,
        reconciliations: List[Dict[str, Any]],
        start_date: datetime,
        end_date: datetime
    ) -> Dict[str, Any]:
        """Generate a summary report of historical reconciliations"""
        total_count = len(reconciliations)
        
        # Calculate summary statistics
        discrepancy_counts = [r.get("discrepancies", {}).get("total_count", 0) for r in reconciliations]
        avg_discrepancies = sum(discrepancy_counts) / total_count if total_count > 0 else 0
        
        # Count reconciliations by status
        status_counts = defaultdict(int)
        for r in reconciliations:
            status_counts[r.get("status", "unknown")] += 1
        
        # Generate monthly/weekly summaries
        monthly_summary = self._generate_time_period_summary(reconciliations, "month")
        weekly_summary = self._generate_time_period_summary(reconciliations, "week")
        
        return {
            "account_id": account_id,
            "period": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat(),
                "days": (end_date - start_date).days
            },
            "total_reconciliations": total_count,
            "average_discrepancies": avg_discrepancies,
            "max_discrepancies": max(discrepancy_counts) if discrepancy_counts else 0,
            "status_summary": dict(status_counts),
            "monthly_summary": monthly_summary,
            "weekly_summary": weekly_summary,
            "generation_time": datetime.utcnow().isoformat()
        }
    
    def _generate_time_period_summary(
        self,
        reconciliations: List[Dict[str, Any]],
        period_type: str
    ) -> List[Dict[str, Any]]:
        """Generate a summary grouped by time period (week/month)"""
        period_summary = defaultdict(lambda: {"count": 0, "discrepancies": 0})
        
        for r in reconciliations:
            timestamp = r.get("start_time")
            if not timestamp:
                continue
                
            if period_type == "month":
                period_key = timestamp.strftime("%Y-%m")
            elif period_type == "week":
                # ISO week number (1-53)
                period_key = f"{timestamp.year}-W{timestamp.isocalendar()[1]:02d}"
            else:
                continue
                
            period_summary[period_key]["count"] += 1
            period_summary[period_key]["discrepancies"] += r.get("discrepancies", {}).get("total_count", 0)
        
        # Calculate averages and format for output
        result = []
        for period, data in sorted(period_summary.items()):
            result.append({
                "period": period,
                "reconciliation_count": data["count"],
                "total_discrepancies": data["discrepancies"],
                "average_discrepancies": data["discrepancies"] / data["count"] if data["count"] > 0 else 0
            })
            
        return result
    
    def generate_detailed_report(
        self,
        account_id: str,
        reconciliations: List[Dict[str, Any]],
        start_date: datetime,
        end_date: datetime
    ) -> Dict[str, Any]:
        """Generate a detailed report of historical reconciliations"""
        # Similar to summary but with additional details
        summary = self.generate_summary_report(account_id, reconciliations, start_date, end_date)
        
        # Add field-specific analysis
        field_analysis = defaultdict(lambda: {"occurrences": 0, "total_difference": 0})
        
        for r in reconciliations:
            for disc in r.get("discrepancies", {}).get("details", []):
                field = disc.get("field")
                if field:
                    field_analysis[field]["occurrences"] += 1
                    field_analysis[field]["total_difference"] += abs(disc.get("absolute_difference", 0))
        
        # Format for output
        fields_report = []
        for field, data in field_analysis.items():
            fields_report.append({
                "field": field,
                "occurrence_count": data["occurrences"],
                "occurrence_percentage": data["occurrences"] / len(reconciliations) * 100 if reconciliations else 0,
                "average_difference": data["total_difference"] / data["occurrences"] if data["occurrences"] > 0 else 0
            })
        
        # Sort by occurrence count
        fields_report.sort(key=lambda x: x["occurrence_count"], reverse=True)
        
        # Combine with summary
        detailed_report = {**summary, "field_analysis": fields_report}
        
        # Add sample reconciliations with most discrepancies
        top_discrepancy_samples = sorted(
            reconciliations,
            key=lambda x: x.get("discrepancies", {}).get("total_count", 0),
            reverse=True
        )[:5]
        
        detailed_report["top_discrepancy_samples"] = [
            {
                "reconciliation_id": r.get("reconciliation_id"),
                "timestamp": r.get("start_time").isoformat() if r.get("start_time") else None,
                "discrepancy_count": r.get("discrepancies", {}).get("total_count", 0),
                "status": r.get("status")
            }
            for r in top_discrepancy_samples
        ]
        
        return detailed_report
    
    async def generate_chart_report(
        self,
        account_id: str,
        reconciliations: List[Dict[str, Any]],
        start_date: datetime,
        end_date: datetime
    ) -> Dict[str, Any]:
        """Generate a report with charts of historical reconciliations"""
        # Generate the summary data first
        summary = self.generate_summary_report(account_id, reconciliations, start_date, end_date)
        
        # Prepare data for charting
        chart_data = []
        for r in sorted(reconciliations, key=lambda x: x.get("start_time", datetime.min)):
            timestamp = r.get("start_time")
            if timestamp:
                chart_data.append({
                    "timestamp": timestamp,
                    "discrepancy_count": r.get("discrepancies", {}).get("total_count", 0)
                })
        
        # Create time series chart of discrepancies
        time_series_chart = self._create_time_series_chart(chart_data)
        
        # Create field frequency chart
        field_frequency_chart = self._create_field_frequency_chart(reconciliations)
        
        # Add charts to the report
        chart_report = {
            **summary,
            "charts": {
                "time_series": time_series_chart,
                "field_frequency": field_frequency_chart
            }
        }
        
        return chart_report
    
    def _create_time_series_chart(self, chart_data: List[Dict[str, Any]]) -> str:
        """
        Create a time series chart of discrepancies over time.
        
        In a real implementation, this would generate an actual chart.
        For this demonstration, we return a placeholder string that would represent the chart data.
        """
        return "time_series_chart_data_placeholder"
    
    def _create_field_frequency_chart(self, reconciliations: List[Dict[str, Any]]) -> str:
        """
        Create a chart showing frequency of discrepancies by field.
        
        In a real implementation, this would generate an actual chart.
        For this demonstration, we return a placeholder string that would represent the chart data.
        """
        return "field_frequency_chart_data_placeholder"