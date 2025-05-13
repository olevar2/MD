"""
Grafana API for Monitoring Alerting Service.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from typing import Dict, Any, List, Optional

router = APIRouter()

@router.get("/dashboards")
async def get_dashboards():
    """Get all Grafana dashboards."""
    return {
        "status": "success",
        "message": "Grafana dashboards retrieved successfully",
        "data": [
            {
                "uid": "forex-trading-platform",
                "title": "Forex Trading Platform",
                "url": "/d/forex-trading-platform",
                "tags": ["forex", "trading"],
                "version": 1,
            },
            {
                "uid": "ml-workbench",
                "title": "ML Workbench",
                "url": "/d/ml-workbench",
                "tags": ["ml", "workbench"],
                "version": 1,
            },
            {
                "uid": "data-pipeline",
                "title": "Data Pipeline",
                "url": "/d/data-pipeline",
                "tags": ["data", "pipeline"],
                "version": 1,
            },
            {
                "uid": "ml-integration",
                "title": "ML Integration",
                "url": "/d/ml-integration",
                "tags": ["ml", "integration"],
                "version": 1,
            },
        ],
    }

@router.get("/datasources")
async def get_datasources():
    """Get all Grafana datasources."""
    return {
        "status": "success",
        "message": "Grafana datasources retrieved successfully",
        "data": [
            {
                "id": 1,
                "name": "Prometheus",
                "type": "prometheus",
                "url": "http://prometheus:9090",
                "isDefault": True,
            },
            {
                "id": 2,
                "name": "Loki",
                "type": "loki",
                "url": "http://loki:3100",
                "isDefault": False,
            },
            {
                "id": 3,
                "name": "Jaeger",
                "type": "jaeger",
                "url": "http://jaeger:16686",
                "isDefault": False,
            },
        ],
    }

@router.get("/users")
async def get_users():
    """Get all Grafana users."""
    return {
        "status": "success",
        "message": "Grafana users retrieved successfully",
        "data": [
            {
                "id": 1,
                "login": "admin",
                "email": "admin@example.com",
                "name": "Admin",
                "isAdmin": True,
            },
            {
                "id": 2,
                "login": "viewer",
                "email": "viewer@example.com",
                "name": "Viewer",
                "isAdmin": False,
            },
        ],
    }