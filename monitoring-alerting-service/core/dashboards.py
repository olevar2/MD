"""
Dashboards API for Monitoring Alerting Service.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from typing import Dict, Any, List, Optional
import uuid

router = APIRouter()

@router.get("/")
async def get_dashboards():
    """Get all dashboards."""
    return {
        "status": "success",
        "message": "Dashboards retrieved successfully",
        "data": [],
    }

@router.post("/")
async def create_dashboard(dashboard_data: Dict[str, Any]):
    """Create a new dashboard."""
    uid = str(uuid.uuid4())
    return {
        "status": "success",
        "message": "Dashboard created successfully",
        "data": {
            "uid": uid,
            "title": dashboard_data.get("title"),
            "description": dashboard_data.get("description"),
            "created_by": dashboard_data.get("created_by"),
            "tags": dashboard_data.get("tags"),
            "data": dashboard_data.get("data"),
        },
    }

@router.get("/{dashboard_uid}")
async def get_dashboard(dashboard_uid: str):
    """Get a dashboard by UID."""
    return {
        "status": "success",
        "message": "Dashboard retrieved successfully",
        "data": {
            "uid": dashboard_uid,
            "title": "Test Dashboard",
            "description": "Test dashboard",
            "created_by": "test_user",
            "tags": ["test", "dashboard"],
            "data": {
                "panels": [
                    {
                        "id": 1,
                        "title": "CPU Usage",
                        "type": "graph",
                        "datasource": "prometheus",
                        "targets": [
                            {
                                "expr": 'cpu_usage_percent{service="trading-gateway"}',
                                "legendFormat": "CPU Usage",
                            }
                        ],
                    },
                ],
            },
        },
    }