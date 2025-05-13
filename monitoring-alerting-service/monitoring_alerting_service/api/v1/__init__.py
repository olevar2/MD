"""
API v1 package for Monitoring Alerting Service.
"""

from monitoring_alerting_service.api.v1.alerts import router as alerts_router
from monitoring_alerting_service.api.v1.dashboards import router as dashboards_router
from monitoring_alerting_service.api.v1.prometheus import router as prometheus_router
from monitoring_alerting_service.api.v1.alertmanager import router as alertmanager_router
from monitoring_alerting_service.api.v1.grafana import router as grafana_router
from monitoring_alerting_service.api.v1.notifications import router as notifications_router