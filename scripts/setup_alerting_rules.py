#!/usr/bin/env python3
"""
Script to set up alerting rules for the Forex Trading Platform.

This script creates Prometheus alerting rules and Alertmanager configuration.
"""

import os
import sys
import yaml
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('setup_alerting.log')
    ]
)
logger = logging.getLogger(__name__)

# Prometheus configuration
PROMETHEUS_CONFIG = {
    'rules_dir': os.environ.get('PROMETHEUS_RULES_DIR', 'monitoring/prometheus/rules'),
    'alertmanager_config': os.environ.get('ALERTMANAGER_CONFIG', 'monitoring/alertmanager/config.yml')
}

# Service configuration
SERVICE_CONFIG = {
    'trading-gateway-service': {
        'port': 8001,
        'critical': True  # Is this service critical?
    },
    'portfolio-management-service': {
        'port': 8002,
        'critical': True
    },
    'risk-management-service': {
        'port': 8003,
        'critical': True
    },
    'data-pipeline-service': {
        'port': 8004,
        'critical': True
    },
    'feature-store-service': {
        'port': 8005,
        'critical': False
    },
    'ml-integration-service': {
        'port': 8006,
        'critical': False
    },
    'ml-workbench-service': {
        'port': 8007,
        'critical': False
    },
    'monitoring-alerting-service': {
        'port': 8008,
        'critical': True
    }
}

# Alert configuration
ALERT_CONFIG = {
    'service_down': {
        'name': 'ServiceDown',
        'expr': 'up == 0',
        'for': '1m',
        'labels': {
            'severity': 'critical'
        },
        'annotations': {
            'summary': 'Service {{ $labels.instance }} is down',
            'description': 'Service {{ $labels.instance }} has been down for more than 1 minute.'
        }
    },
    'high_error_rate': {
        'name': 'HighErrorRate',
        'expr': 'sum(rate(http_requests_total{status=~"5.."}[5m])) by (service) / sum(rate(http_requests_total[5m])) by (service) > 0.05',
        'for': '5m',
        'labels': {
            'severity': 'warning'
        },
        'annotations': {
            'summary': 'High error rate for {{ $labels.service }}',
            'description': 'Service {{ $labels.service }} has a high HTTP error rate (> 5%) for more than 5 minutes.'
        }
    },
    'high_latency': {
        'name': 'HighLatency',
        'expr': 'histogram_quantile(0.95, sum(rate(http_request_duration_seconds_bucket[5m])) by (le, service)) > 1',
        'for': '5m',
        'labels': {
            'severity': 'warning'
        },
        'annotations': {
            'summary': 'High latency for {{ $labels.service }}',
            'description': 'Service {{ $labels.service }} has a 95th percentile latency > 1s for more than 5 minutes.'
        }
    },
    'high_cpu_usage': {
        'name': 'HighCPUUsage',
        'expr': 'rate(process_cpu_seconds_total[1m]) > 0.8',
        'for': '5m',
        'labels': {
            'severity': 'warning'
        },
        'annotations': {
            'summary': 'High CPU usage for {{ $labels.service }}',
            'description': 'Service {{ $labels.service }} has CPU usage > 80% for more than 5 minutes.'
        }
    },
    'high_memory_usage': {
        'name': 'HighMemoryUsage',
        'expr': 'process_resident_memory_bytes / process_resident_memory_bytes{quantile="0.99"} > 0.8',
        'for': '5m',
        'labels': {
            'severity': 'warning'
        },
        'annotations': {
            'summary': 'High memory usage for {{ $labels.service }}',
            'description': 'Service {{ $labels.service }} has memory usage > 80% for more than 5 minutes.'
        }
    },
    'instance_missing': {
        'name': 'InstanceMissing',
        'expr': 'absent(up{job="forex-trading-platform"})',
        'for': '5m',
        'labels': {
            'severity': 'critical'
        },
        'annotations': {
            'summary': 'Instance missing',
            'description': 'Instance has disappeared from Prometheus target discovery.'
        }
    },
    'high_request_rate': {
        'name': 'HighRequestRate',
        'expr': 'sum(rate(http_requests_total[1m])) by (service) > 100',
        'for': '5m',
        'labels': {
            'severity': 'warning'
        },
        'annotations': {
            'summary': 'High request rate for {{ $labels.service }}',
            'description': 'Service {{ $labels.service }} is experiencing high request rate (> 100 req/s) for more than 5 minutes.'
        }
    },
    'business_operation_errors': {
        'name': 'BusinessOperationErrors',
        'expr': 'sum(rate(business_operation_total{status="error"}[5m])) by (service, operation) / sum(rate(business_operation_total[5m])) by (service, operation) > 0.05',
        'for': '5m',
        'labels': {
            'severity': 'warning'
        },
        'annotations': {
            'summary': 'High business operation error rate for {{ $labels.service }} - {{ $labels.operation }}',
            'description': 'Service {{ $labels.service }} has a high error rate (> 5%) for business operation {{ $labels.operation }} for more than 5 minutes.'
        }
    }
}

def create_prometheus_rules() -> bool:
    """
    Simulate creating Prometheus alerting rules.

    Returns:
        Whether the rules were created successfully
    """
    logger.info("Creating Prometheus alerting rules")

    # Create rules directory with absolute path
    rules_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'monitoring', 'prometheus', 'rules')
    os.makedirs(rules_dir, exist_ok=True)

    # Create rules file
    rules_file = os.path.join(rules_dir, 'forex_trading_platform_alerts.yml')

    # Create rules
    rules = {
        'groups': [
            {
                'name': 'forex_trading_platform',
                'rules': []
            }
        ]
    }

    # Add service-specific rules
    for service_name, service_config in SERVICE_CONFIG.items():
        # Service down alert
        service_down_alert = ALERT_CONFIG['service_down'].copy()
        service_down_alert['name'] = f"{service_name.replace('-', '_').title()}Down"
        service_down_alert['expr'] = f'up{{service="{service_name}"}} == 0'

        # Set severity based on service criticality
        if service_config['critical']:
            service_down_alert['labels']['severity'] = 'critical'
        else:
            service_down_alert['labels']['severity'] = 'warning'

        service_down_alert['annotations']['summary'] = f"Service {service_name} is down"
        service_down_alert['annotations']['description'] = f"Service {service_name} has been down for more than 1 minute."

        rules['groups'][0]['rules'].append(service_down_alert)

        # High error rate alert
        high_error_rate_alert = ALERT_CONFIG['high_error_rate'].copy()
        high_error_rate_alert['name'] = f"{service_name.replace('-', '_').title()}HighErrorRate"
        high_error_rate_alert['expr'] = f'sum(rate(http_requests_total{{service="{service_name}", status=~"5.."}}[5m])) / sum(rate(http_requests_total{{service="{service_name}"}}[5m])) > 0.05'

        rules['groups'][0]['rules'].append(high_error_rate_alert)

    # Add general rules
    rules['groups'][0]['rules'].append(ALERT_CONFIG['high_cpu_usage'])
    rules['groups'][0]['rules'].append(ALERT_CONFIG['high_memory_usage'])
    rules['groups'][0]['rules'].append(ALERT_CONFIG['instance_missing'])

    # Write rules to file
    try:
        with open(rules_file, 'w') as f:
            yaml.dump(rules, f, default_flow_style=False)

        logger.info(f"Created Prometheus alerting rules at {rules_file}")
        return True

    except Exception as e:
        logger.error(f"Error creating Prometheus alerting rules: {str(e)}")
        return False

def create_alertmanager_config() -> bool:
    """
    Simulate creating Alertmanager configuration.

    Returns:
        Whether the configuration was created successfully
    """
    logger.info("Creating Alertmanager configuration")

    # Create config directory with absolute path
    config_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'monitoring', 'alertmanager')
    os.makedirs(config_dir, exist_ok=True)

    # Create config file
    config_file = os.path.join(config_dir, 'config.yml')

    # Create config
    config = {
        'global': {
            'resolve_timeout': '5m',
            'smtp_smarthost': 'smtp.example.com:587',
            'smtp_from': 'alertmanager@example.com',
            'smtp_auth_username': 'alertmanager',
            'smtp_auth_password': 'password',
            'smtp_require_tls': True
        },
        'route': {
            'group_by': ['alertname', 'service'],
            'group_wait': '30s',
            'group_interval': '5m',
            'repeat_interval': '4h',
            'receiver': 'email-notifications',
            'routes': [
                {
                    'match': {
                        'severity': 'critical'
                    },
                    'receiver': 'pager-duty-critical',
                    'repeat_interval': '1h'
                }
            ]
        },
        'receivers': [
            {
                'name': 'email-notifications',
                'email_configs': [
                    {
                        'to': 'alerts@example.com',
                        'send_resolved': True
                    }
                ]
            },
            {
                'name': 'pager-duty-critical',
                'pagerduty_configs': [
                    {
                        'service_key': 'your-pagerduty-service-key',
                        'description': '{{ .CommonAnnotations.summary }}',
                        'details': {
                            'firing': '{{ .Alerts.Firing | len }}',
                            'resolved': '{{ .Alerts.Resolved | len }}',
                            'instances': '{{ range .Alerts }}{{ .Labels.instance }} {{ end }}'
                        },
                        'severity': 'critical',
                        'send_resolved': True
                    }
                ]
            }
        ],
        'inhibit_rules': [
            {
                'source_match': {
                    'severity': 'critical'
                },
                'target_match': {
                    'severity': 'warning'
                },
                'equal': ['alertname', 'service']
            }
        ]
    }

    # Write config to file
    try:
        with open(config_file, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)

        logger.info(f"Created Alertmanager configuration at {config_file}")
        return True

    except Exception as e:
        logger.error(f"Error creating Alertmanager configuration: {str(e)}")
        return False

def main():
    """Main function to set up alerting rules."""
    logger.info("Starting setup of alerting rules")

    try:
        # Create Prometheus rules
        if not create_prometheus_rules():
            logger.error("Failed to create Prometheus alerting rules")
            return 1

        # Create Alertmanager config
        if not create_alertmanager_config():
            logger.error("Failed to create Alertmanager configuration")
            return 1

        logger.info("Successfully set up alerting rules")
        return 0

    except Exception as e:
        logger.error(f"Error setting up alerting rules: {str(e)}")
        return 1

if __name__ == '__main__':
    sys.exit(main())
