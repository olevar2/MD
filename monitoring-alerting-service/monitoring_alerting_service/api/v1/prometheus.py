"""
Prometheus API for Monitoring Alerting Service.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from typing import Dict, Any, List, Optional

router = APIRouter()

@router.get("/targets")
async def get_targets():
    """Get all Prometheus targets."""
    return {
        "status": "success",
        "message": "Prometheus targets retrieved successfully",
        "data": [
            {
                "target": "ml-workbench-service:8030",
                "labels": {"service": "ml-workbench-service", "environment": "production"},
                "health": "up",
                "last_scrape": "2025-05-18T12:00:00Z",
            },
            {
                "target": "monitoring-alerting-service:8009",
                "labels": {"service": "monitoring-alerting-service", "environment": "production"},
                "health": "up",
                "last_scrape": "2025-05-18T12:00:00Z",
            },
            {
                "target": "data-pipeline-service:8010",
                "labels": {"service": "data-pipeline-service", "environment": "production"},
                "health": "up",
                "last_scrape": "2025-05-18T12:00:00Z",
            },
            {
                "target": "ml-integration-service:8020",
                "labels": {"service": "ml-integration-service", "environment": "production"},
                "health": "up",
                "last_scrape": "2025-05-18T12:00:00Z",
            },
        ],
    }

@router.get("/rules")
async def get_rules():
    """Get all Prometheus rules."""
    return {
        "status": "success",
        "message": "Prometheus rules retrieved successfully",
        "data": [
            {
                "name": "HighCPUUsage",
                "query": 'cpu_usage_percent > 90',
                "duration": "5m",
                "labels": {"severity": "critical"},
                "annotations": {"summary": "High CPU usage", "description": "CPU usage is above 90%"},
            },
            {
                "name": "HighMemoryUsage",
                "query": 'memory_usage_percent > 90',
                "duration": "5m",
                "labels": {"severity": "critical"},
                "annotations": {"summary": "High memory usage", "description": "Memory usage is above 90%"},
            },
            {
                "name": "HighLatency",
                "query": 'http_request_duration_seconds{quantile="0.95"} > 1',
                "duration": "5m",
                "labels": {"severity": "warning"},
                "annotations": {"summary": "High latency", "description": "95th percentile latency is above 1 second"},
            },
        ],
    }