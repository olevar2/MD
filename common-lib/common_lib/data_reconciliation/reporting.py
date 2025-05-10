"""
Reporting utilities for data reconciliation.

This module provides utilities for generating reports and metrics
from reconciliation results.
"""

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import pandas as pd
import numpy as np
import logging

from common_lib.data_reconciliation.base import (
    Discrepancy,
    DiscrepancyResolution,
    ReconciliationResult,
    ReconciliationSeverity,
    ReconciliationStatus,
)

logger = logging.getLogger(__name__)


class ReconciliationMetrics:
    """Metrics calculated from reconciliation results."""

    def __init__(self, results: List[ReconciliationResult]):
        """
        Initialize reconciliation metrics.

        Args:
            results: List of reconciliation results to analyze
        """
        self.results = results
        self.calculate_metrics()

    def calculate_metrics(self) -> None:
        """Calculate metrics from the reconciliation results."""
        # Basic counts
        self.total_reconciliations = len(self.results)
        self.successful_reconciliations = sum(1 for r in self.results if r.status == ReconciliationStatus.COMPLETED)
        self.failed_reconciliations = sum(1 for r in self.results if r.status == ReconciliationStatus.FAILED)
        self.partial_reconciliations = sum(1 for r in self.results if r.status == ReconciliationStatus.PARTIALLY_COMPLETED)

        # Discrepancy metrics
        self.total_discrepancies = sum(r.discrepancy_count for r in self.results)
        self.total_resolutions = sum(r.resolution_count for r in self.results)

        if self.total_discrepancies > 0:
            self.overall_resolution_rate = (self.total_resolutions / self.total_discrepancies) * 100
        else:
            self.overall_resolution_rate = 100.0

        # Severity breakdown
        self.severity_counts = {
            ReconciliationSeverity.CRITICAL: 0,
            ReconciliationSeverity.HIGH: 0,
            ReconciliationSeverity.MEDIUM: 0,
            ReconciliationSeverity.LOW: 0,
            ReconciliationSeverity.INFO: 0,
        }

        for result in self.results:
            for discrepancy in result.discrepancies:
                self.severity_counts[discrepancy.severity] += 1

        # Performance metrics
        durations = [r.duration_seconds for r in self.results if r.duration_seconds is not None]
        if durations:
            self.avg_duration = sum(durations) / len(durations)
            self.min_duration = min(durations)
            self.max_duration = max(durations)
        else:
            self.avg_duration = None
            self.min_duration = None
            self.max_duration = None

        # Field metrics
        self.field_counts = {}
        for result in self.results:
            for discrepancy in result.discrepancies:
                field = discrepancy.field
                if field not in self.field_counts:
                    self.field_counts[field] = 0
                self.field_counts[field] += 1

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary representation."""
        return {
            "total_reconciliations": self.total_reconciliations,
            "successful_reconciliations": self.successful_reconciliations,
            "failed_reconciliations": self.failed_reconciliations,
            "partial_reconciliations": self.partial_reconciliations,
            "total_discrepancies": self.total_discrepancies,
            "total_resolutions": self.total_resolutions,
            "overall_resolution_rate": self.overall_resolution_rate,
            "severity_counts": {s.name: c for s, c in self.severity_counts.items()},
            "performance": {
                "avg_duration": self.avg_duration,
                "min_duration": self.min_duration,
                "max_duration": self.max_duration,
            },
            "field_counts": self.field_counts,
        }


class DiscrepancyReport:
    """Detailed report on discrepancies."""

    def __init__(self, discrepancies: List[Discrepancy], resolutions: Optional[List[DiscrepancyResolution]] = None):
        """
        Initialize discrepancy report.

        Args:
            discrepancies: List of discrepancies to report on
            resolutions: Optional list of resolutions for the discrepancies
        """
        self.discrepancies = discrepancies
        self.resolutions = resolutions or []
        self.generate_report()

    def generate_report(self) -> None:
        """Generate the discrepancy report."""
        # Create a dictionary of resolutions for easy lookup
        resolution_map = {}
        for resolution in self.resolutions:
            resolution_map[resolution.discrepancy.discrepancy_id] = resolution

        # Create detailed report entries
        self.entries = []
        for discrepancy in self.discrepancies:
            entry = {
                "discrepancy_id": discrepancy.discrepancy_id,
                "field": discrepancy.field,
                "severity": discrepancy.severity.name,
                "timestamp": discrepancy.timestamp.isoformat(),
                "sources": discrepancy.sources,
                "statistics": {
                    "min_value": discrepancy.min_value,
                    "max_value": discrepancy.max_value,
                    "mean_value": discrepancy.mean_value,
                    "median_value": discrepancy.median_value,
                    "std_dev": discrepancy.std_dev,
                    "range_pct": discrepancy.range_pct,
                },
                "resolved": discrepancy.discrepancy_id in resolution_map,
            }

            # Add resolution details if available
            if discrepancy.discrepancy_id in resolution_map:
                resolution = resolution_map[discrepancy.discrepancy_id]
                entry["resolution"] = {
                    "resolved_value": resolution.resolved_value,
                    "strategy": resolution.strategy.name,
                    "resolution_source": resolution.resolution_source,
                    "timestamp": resolution.timestamp.isoformat(),
                }

            self.entries.append(entry)

        # Calculate summary statistics
        self.total_discrepancies = len(self.discrepancies)
        self.resolved_discrepancies = len(self.resolutions)

        if self.total_discrepancies > 0:
            self.resolution_rate = (self.resolved_discrepancies / self.total_discrepancies) * 100
        else:
            self.resolution_rate = 100.0

        # Severity breakdown
        self.severity_counts = {
            ReconciliationSeverity.CRITICAL: 0,
            ReconciliationSeverity.HIGH: 0,
            ReconciliationSeverity.MEDIUM: 0,
            ReconciliationSeverity.LOW: 0,
            ReconciliationSeverity.INFO: 0,
        }

        for discrepancy in self.discrepancies:
            self.severity_counts[discrepancy.severity] += 1

        # Field breakdown
        self.field_counts = {}
        for discrepancy in self.discrepancies:
            field = discrepancy.field
            if field not in self.field_counts:
                self.field_counts[field] = 0
            self.field_counts[field] += 1

    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary representation."""
        return {
            "total_discrepancies": self.total_discrepancies,
            "resolved_discrepancies": self.resolved_discrepancies,
            "resolution_rate": self.resolution_rate,
            "severity_counts": {s.name: c for s, c in self.severity_counts.items()},
            "field_counts": self.field_counts,
            "entries": self.entries,
        }

    def to_dataframe(self) -> pd.DataFrame:
        """Convert report to DataFrame representation."""
        return pd.DataFrame(self.entries)


class ReconciliationSummary:
    """Summary of reconciliation results over time."""

    def __init__(self, results: List[ReconciliationResult], time_window: Optional[timedelta] = None):
        """
        Initialize reconciliation summary.

        Args:
            results: List of reconciliation results to summarize
            time_window: Optional time window for filtering results
        """
        self.results = results
        self.time_window = time_window
        self.generate_summary()

    def generate_summary(self) -> None:
        """Generate the reconciliation summary."""
        # Filter results by time window if specified
        if self.time_window:
            cutoff_time = datetime.utcnow() - self.time_window
            filtered_results = [r for r in self.results if r.start_time >= cutoff_time]
        else:
            filtered_results = self.results

        # Calculate metrics
        self.metrics = ReconciliationMetrics(filtered_results)

        # Group results by day
        self.daily_metrics = {}
        for result in filtered_results:
            day = result.start_time.date()
            if day not in self.daily_metrics:
                self.daily_metrics[day] = []
            self.daily_metrics[day].append(result)

        # Calculate daily metrics
        self.daily_summaries = {}
        for day, day_results in self.daily_metrics.items():
            self.daily_summaries[day] = ReconciliationMetrics(day_results).to_dict()

        # Calculate trend metrics
        if len(self.daily_summaries) > 1:
            days = sorted(self.daily_summaries.keys())
            first_day = days[0]
            last_day = days[-1]

            first_metrics = self.daily_summaries[first_day]
            last_metrics = self.daily_summaries[last_day]

            # Calculate change in discrepancy rate
            if first_metrics["total_reconciliations"] > 0 and last_metrics["total_reconciliations"] > 0:
                first_rate = first_metrics["total_discrepancies"] / first_metrics["total_reconciliations"]
                last_rate = last_metrics["total_discrepancies"] / last_metrics["total_reconciliations"]
                self.discrepancy_rate_change = ((last_rate - first_rate) / first_rate) * 100 if first_rate > 0 else 0
            else:
                self.discrepancy_rate_change = 0

            # Calculate change in resolution rate
            if first_metrics["total_discrepancies"] > 0 and last_metrics["total_discrepancies"] > 0:
                first_res_rate = first_metrics["total_resolutions"] / first_metrics["total_discrepancies"]
                last_res_rate = last_metrics["total_resolutions"] / last_metrics["total_discrepancies"]
                self.resolution_rate_change = ((last_res_rate - first_res_rate) / first_res_rate) * 100 if first_res_rate > 0 else 0
            else:
                self.resolution_rate_change = 0
        else:
            self.discrepancy_rate_change = 0
            self.resolution_rate_change = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert summary to dictionary representation."""
        return {
            "overall_metrics": self.metrics.to_dict(),
            "daily_summaries": {str(day): metrics for day, metrics in self.daily_summaries.items()},
            "trends": {
                "discrepancy_rate_change": self.discrepancy_rate_change,
                "resolution_rate_change": self.resolution_rate_change,
            },
            "time_window": str(self.time_window) if self.time_window else "all",
        }


class ReconciliationReport:
    """Comprehensive report on reconciliation processes."""

    def __init__(
        self,
        result: ReconciliationResult,
        include_details: bool = True
    ):
        """
        Initialize reconciliation report.

        Args:
            result: Reconciliation result to report on
            include_details: Whether to include detailed discrepancy information
        """
        self.result = result
        self.include_details = include_details
        self.generate_report()

    def generate_report(self) -> None:
        """Generate the reconciliation report."""
        # Basic information
        self.reconciliation_id = self.result.reconciliation_id
        self.status = self.result.status
        self.start_time = self.result.start_time
        self.end_time = self.result.end_time
        self.duration_seconds = self.result.duration_seconds

        # Configuration summary
        self.config_summary = {
            "sources": [s.source_id for s in self.result.config.sources],
            "strategy": self.result.config.strategy.name,
            "tolerance": self.result.config.tolerance,
            "auto_resolve": self.result.config.auto_resolve,
        }

        # Discrepancy summary
        self.discrepancy_count = self.result.discrepancy_count
        self.resolution_count = self.result.resolution_count
        self.resolution_rate = self.result.resolution_rate

        # Severity breakdown
        self.severity_counts = {
            ReconciliationSeverity.CRITICAL: 0,
            ReconciliationSeverity.HIGH: 0,
            ReconciliationSeverity.MEDIUM: 0,
            ReconciliationSeverity.LOW: 0,
            ReconciliationSeverity.INFO: 0,
        }

        for discrepancy in self.result.discrepancies:
            self.severity_counts[discrepancy.severity] += 1

        # Field breakdown
        self.field_counts = {}
        for discrepancy in self.result.discrepancies:
            field = discrepancy.field
            if field not in self.field_counts:
                self.field_counts[field] = 0
            self.field_counts[field] += 1

        # Detailed discrepancy report if requested
        if self.include_details:
            self.discrepancy_report = DiscrepancyReport(
                self.result.discrepancies,
                self.result.resolutions
            )
        else:
            self.discrepancy_report = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary representation."""
        report = {
            "reconciliation_id": self.reconciliation_id,
            "status": self.status.name,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_seconds": self.duration_seconds,
            "config_summary": self.config_summary,
            "discrepancy_count": self.discrepancy_count,
            "resolution_count": self.resolution_count,
            "resolution_rate": self.resolution_rate,
            "severity_counts": {s.name: c for s, c in self.severity_counts.items()},
            "field_counts": self.field_counts,
        }

        if self.include_details and self.discrepancy_report:
            report["discrepancy_details"] = self.discrepancy_report.to_dict()

        return report
