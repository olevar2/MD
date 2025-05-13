"""
Dashboard API for Data Reconciliation.

This module provides API endpoints for the reconciliation dashboard.
"""

import logging
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union

from fastapi import APIRouter, Depends, HTTPException, Query, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import pandas as pd
import io

from data_management_service.reconciliation.models import (
    ReconciliationStatus,
    ReconciliationSeverity,
    ReconciliationType
)
from data_management_service.reconciliation.service import ReconciliationService
from data_management_service.reconciliation.api import get_reconciliation_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/reconciliation/dashboard", tags=["reconciliation-dashboard"])


# Response models
class DashboardSummary(BaseModel):
    """Summary statistics for the dashboard."""
    total_configs: int
    active_configs: int
    total_tasks_24h: int
    completed_tasks_24h: int
    failed_tasks_24h: int
    pending_tasks_24h: int
    total_issues_24h: int
    error_issues_24h: int
    warning_issues_24h: int
    info_issues_24h: int
    match_percentage_24h: float


class TimeSeriesPoint(BaseModel):
    """Point in a time series."""
    timestamp: datetime
    value: float


class TimeSeries(BaseModel):
    """Time series data."""
    name: str
    data: List[TimeSeriesPoint]


class IssuesByField(BaseModel):
    """Issues grouped by field."""
    field: str
    error_count: int
    warning_count: int
    info_count: int
    total_count: int


class IssuesBySeverity(BaseModel):
    """Issues grouped by severity."""
    severity: str
    count: int


class ConfigPerformance(BaseModel):
    """Performance metrics for a configuration."""
    config_id: str
    name: str
    match_percentage: float
    issue_count: int
    last_run: datetime


# API endpoints
@router.get("/summary", response_model=DashboardSummary)
async def get_dashboard_summary(
    service: ReconciliationService = Depends(get_reconciliation_service)
) -> DashboardSummary:
    """Get summary statistics for the dashboard."""
    try:
        # Get time range
        now = datetime.utcnow()
        start_date = now - timedelta(hours=24)

        # Get configurations
        configs = await service.get_configs()
        active_configs = [config for config in configs if config.enabled]

        # Get tasks
        tasks = await service.get_tasks(start_date=start_date, end_date=now)
        completed_tasks = [task for task in tasks if task.status == ReconciliationStatus.COMPLETED]
        failed_tasks = [task for task in tasks if task.status == ReconciliationStatus.FAILED]
        pending_tasks = [task for task in tasks if task.status == ReconciliationStatus.PENDING]

        # Get results
        results = await service.get_results(start_date=start_date, end_date=now)

        # Count issues
        total_issues = 0
        error_issues = 0
        warning_issues = 0
        info_issues = 0

        for result in results:
            issues = result.issues
            total_issues += len(issues)
            error_issues += len([i for i in issues if i.severity == ReconciliationSeverity.ERROR])
            warning_issues += len([i for i in issues if i.severity == ReconciliationSeverity.WARNING])
            info_issues += len([i for i in issues if i.severity == ReconciliationSeverity.INFO])

        # Calculate match percentage
        total_records = sum(result.total_records for result in results)
        matched_records = sum(result.matched_records for result in results)
        match_percentage = matched_records / total_records * 100 if total_records > 0 else 0

        return DashboardSummary(
            total_configs=len(configs),
            active_configs=len(active_configs),
            total_tasks_24h=len(tasks),
            completed_tasks_24h=len(completed_tasks),
            failed_tasks_24h=len(failed_tasks),
            pending_tasks_24h=len(pending_tasks),
            total_issues_24h=total_issues,
            error_issues_24h=error_issues,
            warning_issues_24h=warning_issues,
            info_issues_24h=info_issues,
            match_percentage_24h=match_percentage
        )
    except Exception as e:
        logger.error(f"Failed to get dashboard summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/time-series/match-percentage", response_model=TimeSeries)
async def get_match_percentage_time_series(
    days: int = Query(7, description="Number of days to include"),
    config_id: Optional[str] = Query(None, description="Filter by config ID"),
    service: ReconciliationService = Depends(get_reconciliation_service)
) -> TimeSeries:
    """Get time series data for match percentage."""
    try:
        # Get time range
        now = datetime.utcnow()
        start_date = now - timedelta(days=days)

        # Get results
        results = await service.get_results(
            config_id=config_id,
            start_date=start_date,
            end_date=now,
            status=ReconciliationStatus.COMPLETED
        )

        # Group results by day
        results_by_day = {}
        for result in results:
            day = result.start_time.replace(hour=0, minute=0, second=0, microsecond=0)
            if day not in results_by_day:
                results_by_day[day] = []

            results_by_day[day].append(result)

        # Calculate match percentage for each day
        data = []
        for day, day_results in sorted(results_by_day.items()):
            total_records = sum(result.total_records for result in day_results)
            matched_records = sum(result.matched_records for result in day_results)
            match_percentage = matched_records / total_records * 100 if total_records > 0 else 0

            data.append(TimeSeriesPoint(timestamp=day, value=match_percentage))

        return TimeSeries(name="Match Percentage", data=data)
    except Exception as e:
        logger.error(f"Failed to get match percentage time series: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/time-series/issue-count", response_model=List[TimeSeries])
async def get_issue_count_time_series(
    days: int = Query(7, description="Number of days to include"),
    config_id: Optional[str] = Query(None, description="Filter by config ID"),
    service: ReconciliationService = Depends(get_reconciliation_service)
) -> List[TimeSeries]:
    """Get time series data for issue counts by severity."""
    try:
        # Get time range
        now = datetime.utcnow()
        start_date = now - timedelta(days=days)

        # Get results
        results = await service.get_results(
            config_id=config_id,
            start_date=start_date,
            end_date=now,
            status=ReconciliationStatus.COMPLETED
        )

        # Group results by day
        results_by_day = {}
        for result in results:
            day = result.start_time.replace(hour=0, minute=0, second=0, microsecond=0)
            if day not in results_by_day:
                results_by_day[day] = []

            results_by_day[day].append(result)

        # Initialize time series
        error_series = TimeSeries(name="Error Issues", data=[])
        warning_series = TimeSeries(name="Warning Issues", data=[])
        info_series = TimeSeries(name="Info Issues", data=[])

        # Calculate issue counts for each day
        for day, day_results in sorted(results_by_day.items()):
            error_count = 0
            warning_count = 0
            info_count = 0

            for result in day_results:
                error_count += len([i for i in result.issues if i.severity == ReconciliationSeverity.ERROR])
                warning_count += len([i for i in result.issues if i.severity == ReconciliationSeverity.WARNING])
                info_count += len([i for i in result.issues if i.severity == ReconciliationSeverity.INFO])

            error_series.data.append(TimeSeriesPoint(timestamp=day, value=error_count))
            warning_series.data.append(TimeSeriesPoint(timestamp=day, value=warning_count))
            info_series.data.append(TimeSeriesPoint(timestamp=day, value=info_count))

        return [error_series, warning_series, info_series]
    except Exception as e:
        logger.error(f"Failed to get issue count time series: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/issues-by-field", response_model=List[IssuesByField])
async def get_issues_by_field(
    days: int = Query(7, description="Number of days to include"),
    config_id: Optional[str] = Query(None, description="Filter by config ID"),
    limit: int = Query(10, description="Maximum number of fields to return"),
    service: ReconciliationService = Depends(get_reconciliation_service)
) -> List[IssuesByField]:
    """Get issues grouped by field."""
    try:
        # Get time range
        now = datetime.utcnow()
        start_date = now - timedelta(days=days)

        # Get results
        results = await service.get_results(
            config_id=config_id,
            start_date=start_date,
            end_date=now,
            status=ReconciliationStatus.COMPLETED
        )

        # Group issues by field
        issues_by_field = {}

        for result in results:
            for issue in result.issues:
                field = issue.field

                if field not in issues_by_field:
                    issues_by_field[field] = {
                        "error_count": 0,
                        "warning_count": 0,
                        "info_count": 0,
                        "total_count": 0
                    }

                issues_by_field[field]["total_count"] += 1

                if issue.severity == ReconciliationSeverity.ERROR:
                    issues_by_field[field]["error_count"] += 1
                elif issue.severity == ReconciliationSeverity.WARNING:
                    issues_by_field[field]["warning_count"] += 1
                elif issue.severity == ReconciliationSeverity.INFO:
                    issues_by_field[field]["info_count"] += 1

        # Sort by total count and limit
        sorted_fields = sorted(
            issues_by_field.items(),
            key=lambda x: x[1]["total_count"],
            reverse=True
        )[:limit]

        # Convert to response model
        response = []
        for field, counts in sorted_fields:
            response.append(IssuesByField(
                field=field,
                error_count=counts["error_count"],
                warning_count=counts["warning_count"],
                info_count=counts["info_count"],
                total_count=counts["total_count"]
            ))

        return response
    except Exception as e:
        logger.error(f"Failed to get issues by field: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/issues-by-severity", response_model=List[IssuesBySeverity])
async def get_issues_by_severity(
    days: int = Query(7, description="Number of days to include"),
    config_id: Optional[str] = Query(None, description="Filter by config ID"),
    service: ReconciliationService = Depends(get_reconciliation_service)
) -> List[IssuesBySeverity]:
    """Get issues grouped by severity."""
    try:
        # Get time range
        now = datetime.utcnow()
        start_date = now - timedelta(days=days)

        # Get results
        results = await service.get_results(
            config_id=config_id,
            start_date=start_date,
            end_date=now,
            status=ReconciliationStatus.COMPLETED
        )

        # Count issues by severity
        error_count = 0
        warning_count = 0
        info_count = 0

        for result in results:
            for issue in result.issues:
                if issue.severity == ReconciliationSeverity.ERROR:
                    error_count += 1
                elif issue.severity == ReconciliationSeverity.WARNING:
                    warning_count += 1
                elif issue.severity == ReconciliationSeverity.INFO:
                    info_count += 1

        return [
            IssuesBySeverity(severity="ERROR", count=error_count),
            IssuesBySeverity(severity="WARNING", count=warning_count),
            IssuesBySeverity(severity="INFO", count=info_count)
        ]
    except Exception as e:
        logger.error(f"Failed to get issues by severity: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/config-performance", response_model=List[ConfigPerformance])
async def get_config_performance(
    days: int = Query(7, description="Number of days to include"),
    limit: int = Query(10, description="Maximum number of configurations to return"),
    service: ReconciliationService = Depends(get_reconciliation_service)
) -> List[ConfigPerformance]:
    """Get performance metrics for configurations."""
    try:
        # Get time range
        now = datetime.utcnow()
        start_date = now - timedelta(days=days)

        # Get configurations
        configs = await service.get_configs(enabled=True)

        # Get results for each configuration
        config_performance = []

        for config in configs:
            results = await service.get_results(
                config_id=config.config_id,
                start_date=start_date,
                end_date=now,
                status=ReconciliationStatus.COMPLETED
            )

            if not results:
                continue

            # Calculate metrics
            total_records = sum(result.total_records for result in results)
            matched_records = sum(result.matched_records for result in results)
            match_percentage = matched_records / total_records * 100 if total_records > 0 else 0
            issue_count = sum(len(result.issues) for result in results)
            last_run = max(result.start_time for result in results)

            config_performance.append(ConfigPerformance(
                config_id=config.config_id,
                name=config.name,
                match_percentage=match_percentage,
                issue_count=issue_count,
                last_run=last_run
            ))

        # Sort by match percentage (ascending) and limit
        sorted_performance = sorted(
            config_performance,
            key=lambda x: x.match_percentage
        )[:limit]

        return sorted_performance
    except Exception as e:
        logger.error(f"Failed to get config performance: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/results/{result_id}/report")
async def download_reconciliation_report(
    result_id: str,
    service: ReconciliationService = Depends(get_reconciliation_service)
):
    """Download a reconciliation report as CSV."""
    try:
        # Get result
        result = await service.get_result(result_id)

        if not result:
            raise HTTPException(status_code=404, detail=f"Result {result_id} not found")

        # Get configuration
        config = await service.get_config(result.config_id)

        if not config:
            raise HTTPException(status_code=404, detail=f"Configuration {result.config_id} not found")

        # Create report data
        report_data = {
            "metadata": {
                "result_id": result.result_id,
                "config_id": result.config_id,
                "config_name": config.name,
                "reconciliation_type": config.reconciliation_type,
                "primary_source": config.primary_source.source_id,
                "secondary_source": config.secondary_source.source_id if config.secondary_source else None,
                "start_time": result.start_time.isoformat(),
                "end_time": result.end_time.isoformat() if result.end_time else None,
                "status": result.status,
                "total_records": result.total_records,
                "matched_records": result.matched_records,
                "match_percentage": result.matched_records / result.total_records * 100 if result.total_records > 0 else 0,
                "total_issues": len(result.issues)
            },
            "issues": [
                {
                    "field": issue.field,
                    "severity": issue.severity,
                    "description": issue.description,
                    "primary_value": issue.primary_value,
                    "secondary_value": issue.secondary_value,
                    "record_id": issue.record_id
                }
                for issue in result.issues
            ]
        }

        # Determine format based on accept header
        format_type = "csv"  # Default to CSV

        if format_type == "csv":
            # Create CSV
            if not result.issues:
                # Create empty DataFrame with columns
                df = pd.DataFrame(columns=["field", "severity", "description", "primary_value", "secondary_value", "record_id"])
            else:
                df = pd.DataFrame(report_data["issues"])

            # Create buffer
            buffer = io.StringIO()

            # Write metadata
            buffer.write("# Reconciliation Report\n")
            buffer.write(f"# Result ID: {report_data['metadata']['result_id']}\n")
            buffer.write(f"# Configuration: {report_data['metadata']['config_name']} ({report_data['metadata']['config_id']})\n")
            buffer.write(f"# Type: {report_data['metadata']['reconciliation_type']}\n")
            buffer.write(f"# Primary Source: {report_data['metadata']['primary_source']}\n")

            if report_data['metadata']['secondary_source']:
                buffer.write(f"# Secondary Source: {report_data['metadata']['secondary_source']}\n")

            buffer.write(f"# Start Time: {report_data['metadata']['start_time']}\n")

            if report_data['metadata']['end_time']:
                buffer.write(f"# End Time: {report_data['metadata']['end_time']}\n")

            buffer.write(f"# Status: {report_data['metadata']['status']}\n")
            buffer.write(f"# Total Records: {report_data['metadata']['total_records']}\n")
            buffer.write(f"# Matched Records: {report_data['metadata']['matched_records']}\n")
            buffer.write(f"# Match Percentage: {report_data['metadata']['match_percentage']:.2f}%\n")
            buffer.write(f"# Total Issues: {report_data['metadata']['total_issues']}\n")
            buffer.write("#\n")

            # Write data
            df.to_csv(buffer, index=False)

            # Reset buffer position
            buffer.seek(0)

            # Return response
            return StreamingResponse(
                iter([buffer.getvalue()]),
                media_type="text/csv",
                headers={
                    "Content-Disposition": f"attachment; filename=reconciliation_report_{result_id}.csv"
                }
            )
        else:
            # Return JSON
            return Response(
                content=json.dumps(report_data, default=str),
                media_type="application/json",
                headers={
                    "Content-Disposition": f"attachment; filename=reconciliation_report_{result_id}.json"
                }
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to download reconciliation report: {e}")
        raise HTTPException(status_code=500, detail=str(e))
