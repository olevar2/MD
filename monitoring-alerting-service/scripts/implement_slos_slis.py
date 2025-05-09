#!/usr/bin/env python3
"""
SLO and SLI Implementation Script

This script implements Service Level Objectives (SLOs) and Service Level Indicators (SLIs)
for the forex trading platform. It creates SLO definitions, Prometheus recording rules,
and Grafana dashboards.
"""

import os
import sys
import argparse
import logging
import json
import yaml
from pathlib import Path
import subprocess

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("slo-sli-implementation")

# Service Level Objectives
SLO_DEFINITIONS = [
    {
        "name": "trading_gateway_availability",
        "description": "Trading Gateway API availability",
        "service": "trading-gateway-service",
        "sli": {
            "metric_name": "http_request_availability",
            "metric_query": 'sum(rate(http_requests_total{service="trading-gateway-service",status_code=~"[2345].."}[5m])) / sum(rate(http_requests_total{service="trading-gateway-service"}[5m]))',
            "metric_type": "availability"
        },
        "slo": {
            "target": 0.995,  # 99.5% availability
            "window": "30d",
            "error_budget": 0.005  # 0.5% error budget
        },
        "alerting": {
            "burn_rate_thresholds": [
                {
                    "name": "1h_burn_rate_high",
                    "window": "1h",
                    "burn_rate": 14.4,  # Burn 14.4x error budget (100% in ~1h)
                    "alert": {
                        "name": "TradingGatewayHighErrorBudgetBurn1h",
                        "severity": "critical",
                        "description": "Trading Gateway is burning error budget 14.4x faster than normal (100% in ~1h)"
                    }
                },
                {
                    "name": "6h_burn_rate_high",
                    "window": "6h",
                    "burn_rate": 6,  # Burn 6x error budget (100% in ~5h)
                    "alert": {
                        "name": "TradingGatewayHighErrorBudgetBurn6h",
                        "severity": "warning",
                        "description": "Trading Gateway is burning error budget 6x faster than normal (100% in ~5h)"
                    }
                }
            ]
        }
    },
    {
        "name": "trading_gateway_latency",
        "description": "Trading Gateway API latency",
        "service": "trading-gateway-service",
        "sli": {
            "metric_name": "http_request_latency",
            "metric_query": 'histogram_quantile(0.95, sum(rate(http_request_duration_seconds_bucket{service="trading-gateway-service"}[5m])) by (le))',
            "metric_type": "latency"
        },
        "slo": {
            "target": 0.95,  # 95% of requests under threshold
            "threshold": 0.2,  # 200ms
            "window": "30d",
            "error_budget": 0.05  # 5% error budget
        },
        "alerting": {
            "burn_rate_thresholds": [
                {
                    "name": "1h_burn_rate_high",
                    "window": "1h",
                    "burn_rate": 14.4,
                    "alert": {
                        "name": "TradingGatewayHighLatencyBudgetBurn1h",
                        "severity": "critical",
                        "description": "Trading Gateway is burning latency budget 14.4x faster than normal (100% in ~1h)"
                    }
                },
                {
                    "name": "6h_burn_rate_high",
                    "window": "6h",
                    "burn_rate": 6,
                    "alert": {
                        "name": "TradingGatewayHighLatencyBudgetBurn6h",
                        "severity": "warning",
                        "description": "Trading Gateway is burning latency budget 6x faster than normal (100% in ~5h)"
                    }
                }
            ]
        }
    },
    {
        "name": "order_execution_success",
        "description": "Order execution success rate",
        "service": "trading-gateway-service",
        "sli": {
            "metric_name": "order_execution_success",
            "metric_query": 'sum(rate(order_execution_success_total{service="trading-gateway-service"}[5m])) / sum(rate(order_execution_total{service="trading-gateway-service"}[5m]))',
            "metric_type": "success_rate"
        },
        "slo": {
            "target": 0.995,  # 99.5% success rate
            "window": "30d",
            "error_budget": 0.005  # 0.5% error budget
        },
        "alerting": {
            "burn_rate_thresholds": [
                {
                    "name": "1h_burn_rate_high",
                    "window": "1h",
                    "burn_rate": 14.4,
                    "alert": {
                        "name": "OrderExecutionHighErrorBudgetBurn1h",
                        "severity": "critical",
                        "description": "Order execution is burning error budget 14.4x faster than normal (100% in ~1h)"
                    }
                },
                {
                    "name": "6h_burn_rate_high",
                    "window": "6h",
                    "burn_rate": 6,
                    "alert": {
                        "name": "OrderExecutionHighErrorBudgetBurn6h",
                        "severity": "warning",
                        "description": "Order execution is burning error budget 6x faster than normal (100% in ~5h)"
                    }
                }
            ]
        }
    },
    {
        "name": "ml_model_inference_latency",
        "description": "ML model inference latency",
        "service": "ml-integration-service",
        "sli": {
            "metric_name": "model_inference_latency",
            "metric_query": 'histogram_quantile(0.95, sum(rate(model_inference_duration_seconds_bucket{service="ml-integration-service"}[5m])) by (le))',
            "metric_type": "latency"
        },
        "slo": {
            "target": 0.95,  # 95% of inferences under threshold
            "threshold": 0.1,  # 100ms
            "window": "30d",
            "error_budget": 0.05  # 5% error budget
        },
        "alerting": {
            "burn_rate_thresholds": [
                {
                    "name": "1h_burn_rate_high",
                    "window": "1h",
                    "burn_rate": 14.4,
                    "alert": {
                        "name": "MLInferenceHighLatencyBudgetBurn1h",
                        "severity": "critical",
                        "description": "ML inference is burning latency budget 14.4x faster than normal (100% in ~1h)"
                    }
                },
                {
                    "name": "6h_burn_rate_high",
                    "window": "6h",
                    "burn_rate": 6,
                    "alert": {
                        "name": "MLInferenceHighLatencyBudgetBurn6h",
                        "severity": "warning",
                        "description": "ML inference is burning latency budget 6x faster than normal (100% in ~5h)"
                    }
                }
            ]
        }
    },
    {
        "name": "strategy_execution_latency",
        "description": "Strategy execution latency",
        "service": "strategy-execution-engine",
        "sli": {
            "metric_name": "strategy_execution_latency",
            "metric_query": 'histogram_quantile(0.95, sum(rate(strategy_execution_duration_seconds_bucket{service="strategy-execution-engine"}[5m])) by (le))',
            "metric_type": "latency"
        },
        "slo": {
            "target": 0.95,  # 95% of executions under threshold
            "threshold": 0.5,  # 500ms
            "window": "30d",
            "error_budget": 0.05  # 5% error budget
        },
        "alerting": {
            "burn_rate_thresholds": [
                {
                    "name": "1h_burn_rate_high",
                    "window": "1h",
                    "burn_rate": 14.4,
                    "alert": {
                        "name": "StrategyExecutionHighLatencyBudgetBurn1h",
                        "severity": "critical",
                        "description": "Strategy execution is burning latency budget 14.4x faster than normal (100% in ~1h)"
                    }
                },
                {
                    "name": "6h_burn_rate_high",
                    "window": "6h",
                    "burn_rate": 6,
                    "alert": {
                        "name": "StrategyExecutionHighLatencyBudgetBurn6h",
                        "severity": "warning",
                        "description": "Strategy execution is burning latency budget 6x faster than normal (100% in ~5h)"
                    }
                }
            ]
        }
    }
]

def create_slo_document():
    """Create the SLO document."""
    logger.info("Creating SLO document")
    
    # Create directory if it doesn't exist
    docs_dir = Path("monitoring-alerting-service/docs")
    docs_dir.mkdir(exist_ok=True)
    
    # Create SLO document
    slo_file = docs_dir / "service_level_objectives.json"
    with open(slo_file, "w") as f:
        json.dump(SLO_DEFINITIONS, f, indent=2)
    
    # Create Markdown version
    markdown_file = docs_dir / "service_level_objectives.md"
    with open(markdown_file, "w") as f:
        f.write("# Service Level Objectives (SLOs)\n\n")
        f.write("This document defines the Service Level Objectives (SLOs) and Service Level Indicators (SLIs) for the forex trading platform.\n\n")
        
        for slo in SLO_DEFINITIONS:
            f.write(f"## {slo['name']}\n\n")
            f.write(f"**Description**: {slo['description']}\n\n")
            f.write(f"**Service**: {slo['service']}\n\n")
            
            f.write("### Service Level Indicator (SLI)\n\n")
            f.write(f"**Metric Name**: {slo['sli']['metric_name']}\n\n")
            f.write(f"**Metric Type**: {slo['sli']['metric_type']}\n\n")
            f.write(f"**Metric Query**:\n```\n{slo['sli']['metric_query']}\n```\n\n")
            
            f.write("### Service Level Objective (SLO)\n\n")
            f.write(f"**Target**: {slo['slo']['target'] * 100}%\n\n")
            
            if "threshold" in slo['slo']:
                if slo['sli']['metric_type'] == "latency":
                    f.write(f"**Threshold**: {slo['slo']['threshold'] * 1000}ms\n\n")
                else:
                    f.write(f"**Threshold**: {slo['slo']['threshold']}\n\n")
            
            f.write(f"**Window**: {slo['slo']['window']}\n\n")
            f.write(f"**Error Budget**: {slo['slo']['error_budget'] * 100}%\n\n")
            
            f.write("### Alerting\n\n")
            f.write("| Alert | Window | Burn Rate | Severity |\n")
            f.write("|-------|--------|-----------|----------|\n")
            
            for threshold in slo['alerting']['burn_rate_thresholds']:
                f.write(f"| {threshold['alert']['name']} | {threshold['window']} | {threshold['burn_rate']}x | {threshold['alert']['severity']} |\n")
            
            f.write("\n")
    
    logger.info(f"SLO document saved to {slo_file} and {markdown_file}")
    return slo_file, markdown_file

def create_prometheus_recording_rules():
    """Create Prometheus recording rules for SLIs."""
    logger.info("Creating Prometheus recording rules")
    
    # Create directory if it doesn't exist
    config_dir = Path("monitoring-alerting-service/config")
    config_dir.mkdir(exist_ok=True)
    
    # Create recording rules
    rules = {
        "groups": [
            {
                "name": "sli_recording_rules",
                "rules": []
            }
        ]
    }
    
    for slo in SLO_DEFINITIONS:
        # SLI recording rule
        rules["groups"][0]["rules"].append({
            "record": f"sli:{slo['sli']['metric_name']}:ratio_rate5m",
            "expr": slo['sli']['metric_query']
        })
        
        # Error budget recording rule
        if slo['sli']['metric_type'] == "availability" or slo['sli']['metric_type'] == "success_rate":
            # For availability/success rate, error = 1 - actual
            rules["groups"][0]["rules"].append({
                "record": f"sli:{slo['sli']['metric_name']}:error_rate5m",
                "expr": f"1 - ({slo['sli']['metric_query']})"
            })
        elif slo['sli']['metric_type'] == "latency":
            # For latency, error = actual > threshold
            rules["groups"][0]["rules"].append({
                "record": f"sli:{slo['sli']['metric_name']}:error_rate5m",
                "expr": f"({slo['sli']['metric_query']}) > {slo['slo']['threshold']}"
            })
        
        # Error budget burn rate recording rules
        for threshold in slo['alerting']['burn_rate_thresholds']:
            window_seconds = 0
            if threshold['window'] == "1h":
                window_seconds = 3600
            elif threshold['window'] == "6h":
                window_seconds = 21600
            elif threshold['window'] == "1d":
                window_seconds = 86400
            
            if window_seconds > 0:
                rules["groups"][0]["rules"].append({
                    "record": f"sli:{slo['sli']['metric_name']}:error_budget_burn_rate{threshold['window']}",
                    "expr": f"sum(rate(sli:{slo['sli']['metric_name']}:error_rate5m[{threshold['window']}])) / ({slo['slo']['error_budget']} / {window_seconds})"
                })
    
    # Save recording rules
    rules_file = config_dir / "slo_recording_rules.yml"
    with open(rules_file, "w") as f:
        yaml.dump(rules, f, default_flow_style=False)
    
    logger.info(f"Prometheus recording rules saved to {rules_file}")
    return rules_file

def create_prometheus_alerting_rules():
    """Create Prometheus alerting rules for SLOs."""
    logger.info("Creating Prometheus alerting rules")
    
    # Create directory if it doesn't exist
    config_dir = Path("monitoring-alerting-service/config")
    config_dir.mkdir(exist_ok=True)
    
    # Create alerting rules
    rules = {
        "groups": [
            {
                "name": "slo_alerting_rules",
                "rules": []
            }
        ]
    }
    
    for slo in SLO_DEFINITIONS:
        for threshold in slo['alerting']['burn_rate_thresholds']:
            rules["groups"][0]["rules"].append({
                "alert": threshold['alert']['name'],
                "expr": f"sli:{slo['sli']['metric_name']}:error_budget_burn_rate{threshold['window']} > {threshold['burn_rate']}",
                "for": "5m",
                "labels": {
                    "severity": threshold['alert']['severity'],
                    "service": slo['service'],
                    "slo": slo['name']
                },
                "annotations": {
                    "summary": f"High error budget burn rate for {slo['description']}",
                    "description": threshold['alert']['description']
                }
            })
    
    # Save alerting rules
    rules_file = config_dir / "slo_alerting_rules.yml"
    with open(rules_file, "w") as f:
        yaml.dump(rules, f, default_flow_style=False)
    
    logger.info(f"Prometheus alerting rules saved to {rules_file}")
    return rules_file

def create_grafana_dashboard():
    """Create Grafana dashboard for SLOs."""
    logger.info("Creating Grafana dashboard")
    
    # Create directory if it doesn't exist
    dashboard_dir = Path("infrastructure/docker/grafana/dashboards")
    dashboard_dir.mkdir(exist_ok=True, parents=True)
    
    # Create dashboard
    dashboard = {
        "annotations": {
            "list": [
                {
                    "builtIn": 1,
                    "datasource": "-- Grafana --",
                    "enable": True,
                    "hide": True,
                    "iconColor": "rgba(0, 211, 255, 1)",
                    "name": "Annotations & Alerts",
                    "type": "dashboard"
                }
            ]
        },
        "editable": True,
        "gnetId": None,
        "graphTooltip": 0,
        "id": None,
        "links": [],
        "panels": [],
        "refresh": "10s",
        "schemaVersion": 27,
        "style": "dark",
        "tags": ["slo", "sli", "forex-platform"],
        "templating": {
            "list": []
        },
        "time": {
            "from": "now-6h",
            "to": "now"
        },
        "timepicker": {},
        "timezone": "",
        "title": "Forex Platform SLOs",
        "uid": "forex-slos",
        "version": 1
    }
    
    # Add header row
    dashboard["panels"].append({
        "collapsed": False,
        "datasource": None,
        "gridPos": {
            "h": 1,
            "w": 24,
            "x": 0,
            "y": 0
        },
        "id": 1,
        "panels": [],
        "title": "Service Level Objectives",
        "type": "row"
    })
    
    # Add SLO panels
    panel_id = 2
    y_pos = 1
    
    for slo in SLO_DEFINITIONS:
        # Add SLO status panel
        dashboard["panels"].append({
            "datasource": "Prometheus",
            "fieldConfig": {
                "defaults": {
                    "color": {
                        "mode": "thresholds"
                    },
                    "mappings": [],
                    "thresholds": {
                        "mode": "absolute",
                        "steps": [
                            {
                                "color": "green",
                                "value": None
                            },
                            {
                                "color": "red",
                                "value": 1
                            }
                        ]
                    },
                    "unit": "percentunit"
                },
                "overrides": []
            },
            "gridPos": {
                "h": 8,
                "w": 12,
                "x": 0,
                "y": y_pos
            },
            "id": panel_id,
            "options": {
                "colorMode": "value",
                "graphMode": "area",
                "justifyMode": "auto",
                "orientation": "auto",
                "reduceOptions": {
                    "calcs": [
                        "lastNotNull"
                    ],
                    "fields": "",
                    "values": False
                },
                "text": {},
                "textMode": "auto"
            },
            "pluginVersion": "7.5.7",
            "targets": [
                {
                    "expr": f"sli:{slo['sli']['metric_name']}:ratio_rate5m",
                    "interval": "",
                    "legendFormat": "",
                    "refId": "A"
                }
            ],
            "title": f"{slo['description']} - Current",
            "type": "stat"
        })
        panel_id += 1
        
        # Add SLO burn rate panel
        dashboard["panels"].append({
            "datasource": "Prometheus",
            "fieldConfig": {
                "defaults": {
                    "color": {
                        "mode": "thresholds"
                    },
                    "mappings": [],
                    "thresholds": {
                        "mode": "absolute",
                        "steps": [
                            {
                                "color": "green",
                                "value": None
                            },
                            {
                                "color": "yellow",
                                "value": 1
                            },
                            {
                                "color": "red",
                                "value": 10
                            }
                        ]
                    }
                },
                "overrides": []
            },
            "gridPos": {
                "h": 8,
                "w": 12,
                "x": 12,
                "y": y_pos
            },
            "id": panel_id,
            "options": {
                "colorMode": "value",
                "graphMode": "area",
                "justifyMode": "auto",
                "orientation": "auto",
                "reduceOptions": {
                    "calcs": [
                        "lastNotNull"
                    ],
                    "fields": "",
                    "values": False
                },
                "text": {},
                "textMode": "auto"
            },
            "pluginVersion": "7.5.7",
            "targets": [
                {
                    "expr": f"sli:{slo['sli']['metric_name']}:error_budget_burn_rate1h",
                    "interval": "",
                    "legendFormat": "1h",
                    "refId": "A"
                },
                {
                    "expr": f"sli:{slo['sli']['metric_name']}:error_budget_burn_rate6h",
                    "interval": "",
                    "legendFormat": "6h",
                    "refId": "B"
                }
            ],
            "title": f"{slo['description']} - Error Budget Burn Rate",
            "type": "stat"
        })
        panel_id += 1
        
        # Add SLO trend panel
        dashboard["panels"].append({
            "aliasColors": {},
            "bars": False,
            "dashLength": 10,
            "dashes": False,
            "datasource": "Prometheus",
            "fieldConfig": {
                "defaults": {
                    "unit": "percentunit"
                },
                "overrides": []
            },
            "fill": 1,
            "fillGradient": 0,
            "gridPos": {
                "h": 8,
                "w": 24,
                "x": 0,
                "y": y_pos + 8
            },
            "hiddenSeries": False,
            "id": panel_id,
            "legend": {
                "avg": False,
                "current": False,
                "max": False,
                "min": False,
                "show": True,
                "total": False,
                "values": False
            },
            "lines": True,
            "linewidth": 1,
            "nullPointMode": "null",
            "options": {
                "alertThreshold": True
            },
            "percentage": False,
            "pluginVersion": "7.5.7",
            "pointradius": 2,
            "points": False,
            "renderer": "flot",
            "seriesOverrides": [],
            "spaceLength": 10,
            "stack": False,
            "steppedLine": False,
            "targets": [
                {
                    "expr": f"sli:{slo['sli']['metric_name']}:ratio_rate5m",
                    "interval": "",
                    "legendFormat": "Actual",
                    "refId": "A"
                },
                {
                    "expr": str(slo['slo']['target']),
                    "interval": "",
                    "legendFormat": "Target",
                    "refId": "B"
                }
            ],
            "thresholds": [],
            "timeFrom": None,
            "timeRegions": [],
            "timeShift": None,
            "title": f"{slo['description']} - Trend",
            "tooltip": {
                "shared": True,
                "sort": 0,
                "value_type": "individual"
            },
            "type": "graph",
            "xaxis": {
                "buckets": None,
                "mode": "time",
                "name": None,
                "show": True,
                "values": []
            },
            "yaxes": [
                {
                    "format": "percentunit",
                    "label": None,
                    "logBase": 1,
                    "max": "1",
                    "min": "0",
                    "show": True
                },
                {
                    "format": "short",
                    "label": None,
                    "logBase": 1,
                    "max": None,
                    "min": None,
                    "show": True
                }
            ],
            "yaxis": {
                "align": False,
                "alignLevel": None
            }
        })
        panel_id += 1
        
        y_pos += 16
    
    # Save dashboard
    dashboard_file = dashboard_dir / "slo_dashboard.json"
    with open(dashboard_file, "w") as f:
        json.dump(dashboard, f, indent=2)
    
    logger.info(f"Grafana dashboard saved to {dashboard_file}")
    return dashboard_file

def create_documentation():
    """Create documentation for SLOs and SLIs."""
    logger.info("Creating documentation")
    
    # Create directory if it doesn't exist
    docs_dir = Path("monitoring-alerting-service/docs")
    docs_dir.mkdir(exist_ok=True)
    
    # Create documentation file
    docs_file = docs_dir / "slo_sli_implementation.md"
    
    docs_content = """# SLO and SLI Implementation

This document describes the implementation of Service Level Objectives (SLOs) and Service Level Indicators (SLIs) for the forex trading platform.

## Overview

Service Level Objectives (SLOs) are targets for the level of service that we aim to provide to our users. Service Level Indicators (SLIs) are the metrics that we use to measure our service level.

## SLO Definitions

The SLO definitions are stored in `monitoring-alerting-service/docs/service_level_objectives.json` and `monitoring-alerting-service/docs/service_level_objectives.md`.

## Implementation

The SLOs and SLIs are implemented using the following components:

### Prometheus Recording Rules

Recording rules are used to pre-compute SLIs and error budget burn rates. These rules are stored in `monitoring-alerting-service/config/slo_recording_rules.yml`.

### Prometheus Alerting Rules

Alerting rules are used to trigger alerts when error budget burn rates exceed thresholds. These rules are stored in `monitoring-alerting-service/config/slo_alerting_rules.yml`.

### Grafana Dashboard

A Grafana dashboard is used to visualize SLOs and SLIs. The dashboard is stored in `infrastructure/docker/grafana/dashboards/slo_dashboard.json`.

## Error Budget Burn Rate

Error budget burn rate is a measure of how quickly we are consuming our error budget. It is calculated as:

```
error_budget_burn_rate = error_rate / (error_budget / window_seconds)
```

Where:
- `error_rate` is the rate of errors (e.g., failed requests, high latency)
- `error_budget` is the allowed error rate (e.g., 0.5% for 99.5% availability)
- `window_seconds` is the duration of the window in seconds

A burn rate of 1 means we are consuming our error budget at the expected rate. A burn rate of 10 means we are consuming our error budget 10 times faster than expected.

## Alerting

Alerts are triggered when error budget burn rates exceed thresholds. The following thresholds are used:

- **Critical**: Burn rate > 14.4 over 1 hour (100% of error budget in ~1 hour)
- **Warning**: Burn rate > 6 over 6 hours (100% of error budget in ~5 hours)

## Adding New SLOs

To add a new SLO:

1. Add the SLO definition to `SLO_DEFINITIONS` in `monitoring-alerting-service/scripts/implement_slos_slis.py`
2. Run the script to update the SLO document, recording rules, alerting rules, and Grafana dashboard
"""
    
    with open(docs_file, "w") as f:
        f.write(docs_content)
    
    logger.info(f"Documentation saved to {docs_file}")
    return docs_file

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Implement SLOs and SLIs")
    args = parser.parse_args()
    
    # Create SLO document
    slo_file, markdown_file = create_slo_document()
    
    # Create Prometheus recording rules
    recording_rules_file = create_prometheus_recording_rules()
    
    # Create Prometheus alerting rules
    alerting_rules_file = create_prometheus_alerting_rules()
    
    # Create Grafana dashboard
    dashboard_file = create_grafana_dashboard()
    
    # Create documentation
    docs_file = create_documentation()
    
    logger.info("SLO and SLI implementation completed")
    logger.info(f"SLO document: {slo_file}")
    logger.info(f"Markdown document: {markdown_file}")
    logger.info(f"Recording rules: {recording_rules_file}")
    logger.info(f"Alerting rules: {alerting_rules_file}")
    logger.info(f"Grafana dashboard: {dashboard_file}")
    logger.info(f"Documentation: {docs_file}")

if __name__ == "__main__":
    main()
