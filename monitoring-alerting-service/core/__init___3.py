"""
API v1 package for Monitoring Alerting Service.
"""

from core.alerts import router as alerts_router
from core.dashboards import router as dashboards_router
from core.prometheus import router as prometheus_router
from core.alertmanager import router as alertmanager_router
from core.grafana import router as grafana_router
from core.notifications import router as notifications_router