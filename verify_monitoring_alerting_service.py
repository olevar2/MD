#!/usr/bin/env python
"""
Verification script for Monitoring Alerting Service.

This script verifies that the standardized modules in the Monitoring Alerting Service
exist and have the required attributes.
"""

import os
import sys
import importlib
import inspect
import logging
from typing import Dict, Any, List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("verify_monitoring_alerting_service")

# Add service path to sys.path
service_path = os.path.join(os.getcwd(), "monitoring-alerting-service")
if service_path not in sys.path:
    sys.path.append(service_path)

# Import modules
try:
    import monitoring_alerting_service.config.standardized_config as config_module
    import monitoring_alerting_service.logging_setup as logging_module
    import monitoring_alerting_service.monitoring as monitoring_module
    import monitoring_alerting_service.error_handlers as error_handlers_module
    import monitoring_alerting_service.service_clients as service_clients_module
    import monitoring_alerting_service.database as database_module
    import monitoring_alerting_service.main as main_module
    import monitoring_alerting_service.api.v1 as api_v1_module
    
    logger.info("All modules imported successfully")
    
    # Verify config module
    logger.info("Verifying config module")
    assert hasattr(config_module, "settings"), "Config module should have settings attribute"
    assert hasattr(config_module, "get_settings"), "Config module should have get_settings function"
    assert hasattr(config_module, "get_db_url"), "Config module should have get_db_url function"
    assert hasattr(config_module, "get_api_settings"), "Config module should have get_api_settings function"
    assert hasattr(config_module, "get_security_settings"), "Config module should have get_security_settings function"
    assert hasattr(config_module, "get_monitoring_settings"), "Config module should have get_monitoring_settings function"
    logger.info("Config module verified successfully")
    
    # Verify logging module
    logger.info("Verifying logging module")
    assert hasattr(logging_module, "configure_logging"), "Logging module should have configure_logging function"
    assert hasattr(logging_module, "get_logger"), "Logging module should have get_logger function"
    assert hasattr(logging_module, "set_correlation_id"), "Logging module should have set_correlation_id function"
    assert hasattr(logging_module, "get_correlation_id"), "Logging module should have get_correlation_id function"
    logger.info("Logging module verified successfully")
    
    # Verify monitoring module
    logger.info("Verifying monitoring module")
    assert hasattr(monitoring_module, "setup_monitoring"), "Monitoring module should have setup_monitoring function"
    assert hasattr(monitoring_module, "register_health_check"), "Monitoring module should have register_health_check function"
    assert hasattr(monitoring_module, "health_check"), "Monitoring module should have health_check attribute"
    assert hasattr(monitoring_module, "metrics_registry"), "Monitoring module should have metrics_registry attribute"
    assert hasattr(monitoring_module, "start_metrics_collection"), "Monitoring module should have start_metrics_collection function"
    assert hasattr(monitoring_module, "stop_metrics_collection"), "Monitoring module should have stop_metrics_collection function"
    assert hasattr(monitoring_module, "track_database_query"), "Monitoring module should have track_database_query decorator"
    assert hasattr(monitoring_module, "track_service_client_request"), "Monitoring module should have track_service_client_request decorator"
    logger.info("Monitoring module verified successfully")
    
    # Verify error handlers module
    logger.info("Verifying error handlers module")
    assert hasattr(error_handlers_module, "setup_error_handlers"), "Error handlers module should have setup_error_handlers function"
    assert hasattr(error_handlers_module, "get_error_response"), "Error handlers module should have get_error_response function"
    logger.info("Error handlers module verified successfully")
    
    # Verify service clients module
    logger.info("Verifying service clients module")
    assert hasattr(service_clients_module, "BaseServiceClient"), "Service clients module should have BaseServiceClient class"
    assert hasattr(service_clients_module, "close_all_clients"), "Service clients module should have close_all_clients function"
    logger.info("Service clients module verified successfully")
    
    # Verify database module
    logger.info("Verifying database module")
    assert hasattr(database_module, "Base"), "Database module should have Base class"
    assert hasattr(database_module, "init_db"), "Database module should have init_db function"
    assert hasattr(database_module, "create_tables"), "Database module should have create_tables function"
    assert hasattr(database_module, "get_sync_session"), "Database module should have get_sync_session function"
    assert hasattr(database_module, "get_async_session"), "Database module should have get_async_session function"
    assert hasattr(database_module, "BaseRepository"), "Database module should have BaseRepository class"
    logger.info("Database module verified successfully")
    
    # Verify main module
    logger.info("Verifying main module")
    assert hasattr(main_module, "app"), "Main module should have app attribute"
    assert hasattr(main_module, "startup_event"), "Main module should have startup_event function"
    assert hasattr(main_module, "shutdown_event"), "Main module should have shutdown_event function"
    logger.info("Main module verified successfully")
    
    # Verify API endpoints
    logger.info("Verifying API endpoints")
    assert hasattr(api_v1_module, "alerts_router"), "API v1 module should have alerts_router attribute"
    assert hasattr(api_v1_module, "dashboards_router"), "API v1 module should have dashboards_router attribute"
    assert hasattr(api_v1_module, "prometheus_router"), "API v1 module should have prometheus_router attribute"
    assert hasattr(api_v1_module, "alertmanager_router"), "API v1 module should have alertmanager_router attribute"
    assert hasattr(api_v1_module, "grafana_router"), "API v1 module should have grafana_router attribute"
    assert hasattr(api_v1_module, "notifications_router"), "API v1 module should have notifications_router attribute"
    logger.info("API endpoints verified successfully")
    
    logger.info("All modules verified successfully")
except ImportError as e:
    logger.error(f"Error importing modules: {str(e)}")
except AssertionError as e:
    logger.error(f"Assertion error: {str(e)}")
except Exception as e:
    logger.error(f"Error: {str(e)}")