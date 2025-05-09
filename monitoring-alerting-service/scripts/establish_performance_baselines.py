#!/usr/bin/env python3
"""
Performance Baseline Establishment Script

This script runs performance tests and establishes baseline metrics for all services
in the forex trading platform. It creates baseline metrics for API performance,
resource usage, and business metrics.
"""

import os
import sys
import argparse
import logging
import json
import time
import datetime
import statistics
import requests
import yaml
from pathlib import Path
import subprocess
from concurrent.futures import ThreadPoolExecutor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("performance-baseline-establishment")

# Service configuration
SERVICES = [
    {
        "name": "trading-gateway-service",
        "base_url": "http://localhost:8001",
        "endpoints": [
            {"path": "/api/v1/orders", "method": "GET", "name": "list_orders"},
            {"path": "/api/v1/orders", "method": "POST", "name": "create_order"},
            {"path": "/api/v1/positions", "method": "GET", "name": "list_positions"}
        ],
        "business_metrics": [
            "order_execution_time",
            "slippage_bps",
            "fill_rate"
        ]
    },
    {
        "name": "analysis-engine-service",
        "base_url": "http://localhost:8002",
        "endpoints": [
            {"path": "/api/v1/analysis/market", "method": "GET", "name": "market_analysis"},
            {"path": "/api/v1/analysis/patterns", "method": "GET", "name": "pattern_detection"},
            {"path": "/api/v1/analysis/signals", "method": "GET", "name": "signal_generation"}
        ],
        "business_metrics": [
            "pattern_recognition_accuracy",
            "signal_quality_score",
            "market_regime_detection_confidence"
        ]
    },
    {
        "name": "feature-store-service",
        "base_url": "http://localhost:8003",
        "endpoints": [
            {"path": "/api/v1/features", "method": "GET", "name": "list_features"},
            {"path": "/api/v1/features/calculate", "method": "POST", "name": "calculate_features"},
            {"path": "/api/v1/features/batch", "method": "POST", "name": "batch_calculate"}
        ],
        "business_metrics": [
            "feature_calculation_time",
            "cache_hit_rate",
            "data_freshness"
        ]
    },
    {
        "name": "ml-integration-service",
        "base_url": "http://localhost:8004",
        "endpoints": [
            {"path": "/api/v1/models", "method": "GET", "name": "list_models"},
            {"path": "/api/v1/models/predict", "method": "POST", "name": "model_prediction"},
            {"path": "/api/v1/models/batch-predict", "method": "POST", "name": "batch_prediction"}
        ],
        "business_metrics": [
            "model_inference_time",
            "prediction_accuracy",
            "model_confidence"
        ]
    },
    {
        "name": "strategy-execution-engine",
        "base_url": "http://localhost:8005",
        "endpoints": [
            {"path": "/api/v1/strategies", "method": "GET", "name": "list_strategies"},
            {"path": "/api/v1/strategies/execute", "method": "POST", "name": "execute_strategy"},
            {"path": "/api/v1/strategies/backtest", "method": "POST", "name": "backtest_strategy"}
        ],
        "business_metrics": [
            "strategy_execution_time",
            "strategy_win_rate",
            "strategy_sharpe_ratio"
        ]
    },
    {
        "name": "data-pipeline-service",
        "base_url": "http://localhost:8006",
        "endpoints": [
            {"path": "/api/v1/data/market", "method": "GET", "name": "get_market_data"},
            {"path": "/api/v1/data/process", "method": "POST", "name": "process_data"},
            {"path": "/api/v1/data/status", "method": "GET", "name": "pipeline_status"}
        ],
        "business_metrics": [
            "data_processing_time",
            "data_quality_score",
            "pipeline_throughput"
        ]
    }
]

# Test scenarios
TEST_SCENARIOS = [
    {
        "name": "normal_load",
        "description": "Normal trading hours with moderate market activity",
        "concurrent_users": 10,
        "duration_seconds": 60,
        "requests_per_second": 10
    },
    {
        "name": "high_load",
        "description": "Market open with high activity",
        "concurrent_users": 50,
        "duration_seconds": 60,
        "requests_per_second": 50
    },
    {
        "name": "peak_load",
        "description": "News event with very high activity",
        "concurrent_users": 100,
        "duration_seconds": 30,
        "requests_per_second": 100
    }
]

def run_endpoint_test(service, endpoint, scenario):
    """Run a performance test for a specific endpoint."""
    logger.info(f"Testing {service['name']} - {endpoint['name']} under {scenario['name']} scenario")
    
    url = f"{service['base_url']}{endpoint['path']}"
    method = endpoint['method']
    
    # Prepare test data
    if method == "POST":
        # Generate dummy data based on endpoint
        if "order" in endpoint['name']:
            data = {
                "symbol": "EUR/USD",
                "side": "BUY",
                "quantity": 10000,
                "order_type": "MARKET"
            }
        elif "feature" in endpoint['name']:
            data = {
                "symbol": "EUR/USD",
                "features": ["rsi", "macd", "bollinger_bands"],
                "timeframe": "1h",
                "start_time": "2025-01-01T00:00:00Z",
                "end_time": "2025-01-02T00:00:00Z"
            }
        elif "model" in endpoint['name']:
            data = {
                "model_id": "forex-prediction-v1",
                "features": {
                    "rsi": 65.5,
                    "macd": 0.0023,
                    "bollinger_width": 0.0134
                }
            }
        elif "strateg" in endpoint['name']:
            data = {
                "strategy_id": "momentum-v1",
                "symbol": "EUR/USD",
                "timeframe": "1h",
                "parameters": {
                    "rsi_period": 14,
                    "rsi_overbought": 70,
                    "rsi_oversold": 30
                }
            }
        elif "data" in endpoint['name']:
            data = {
                "symbol": "EUR/USD",
                "timeframe": "1h",
                "start_time": "2025-01-01T00:00:00Z",
                "end_time": "2025-01-02T00:00:00Z"
            }
        else:
            data = {}
    else:
        data = None
    
    # Run test
    latencies = []
    errors = 0
    
    # Simulate concurrent users
    def make_request():
        try:
            start_time = time.time()
            
            if method == "GET":
                response = requests.get(url, timeout=10)
            elif method == "POST":
                response = requests.post(url, json=data, timeout=10)
            
            end_time = time.time()
            latency = (end_time - start_time) * 1000  # Convert to ms
            
            if response.status_code >= 400:
                logger.warning(f"Request failed with status {response.status_code}")
                errors += 1
            else:
                latencies.append(latency)
        
        except Exception as e:
            logger.error(f"Request failed: {e}")
            errors += 1
    
    # Run requests in parallel
    with ThreadPoolExecutor(max_workers=scenario['concurrent_users']) as executor:
        for _ in range(scenario['requests_per_second'] * scenario['duration_seconds']):
            executor.submit(make_request)
            time.sleep(1 / scenario['requests_per_second'])
    
    # Calculate metrics
    if latencies:
        p50 = statistics.median(latencies)
        p95 = statistics.quantiles(latencies, n=20)[18]  # 95th percentile
        p99 = statistics.quantiles(latencies, n=100)[98]  # 99th percentile
        
        return {
            "service": service['name'],
            "endpoint": endpoint['name'],
            "scenario": scenario['name'],
            "latency": {
                "p50": p50,
                "p95": p95,
                "p99": p99,
                "min": min(latencies),
                "max": max(latencies),
                "mean": statistics.mean(latencies)
            },
            "throughput": len(latencies) / scenario['duration_seconds'],
            "error_rate": errors / (len(latencies) + errors) if (len(latencies) + errors) > 0 else 0
        }
    else:
        logger.error(f"No successful requests for {service['name']} - {endpoint['name']}")
        return {
            "service": service['name'],
            "endpoint": endpoint['name'],
            "scenario": scenario['name'],
            "latency": {
                "p50": None,
                "p95": None,
                "p99": None,
                "min": None,
                "max": None,
                "mean": None
            },
            "throughput": 0,
            "error_rate": 1.0
        }

def generate_baseline_document(results):
    """Generate a baseline document from test results."""
    logger.info("Generating baseline document")
    
    # Create baseline document
    baseline = {
        "generated_at": datetime.datetime.now().isoformat(),
        "test_scenarios": TEST_SCENARIOS,
        "services": {}
    }
    
    # Process results
    for result in results:
        service_name = result['service']
        endpoint_name = result['endpoint']
        scenario_name = result['scenario']
        
        if service_name not in baseline['services']:
            baseline['services'][service_name] = {
                "api_performance": {},
                "resource_usage": {
                    "cpu_usage": {
                        "baseline": 30,
                        "warning": 70,
                        "critical": 90
                    },
                    "memory_usage": {
                        "baseline": 40,
                        "warning": 80,
                        "critical": 90
                    }
                },
                "business_metrics": {}
            }
        
        if endpoint_name not in baseline['services'][service_name]['api_performance']:
            baseline['services'][service_name]['api_performance'][endpoint_name] = {}
        
        baseline['services'][service_name]['api_performance'][endpoint_name][scenario_name] = {
            "latency": {
                "p50": result['latency']['p50'],
                "p95": result['latency']['p95'],
                "p99": result['latency']['p99']
            },
            "throughput": result['throughput'],
            "error_rate": result['error_rate'],
            "alert_thresholds": {
                "latency": {
                    "warning": result['latency']['p95'] * 1.5 if result['latency']['p95'] else 1000,
                    "critical": result['latency']['p95'] * 3 if result['latency']['p95'] else 2000
                },
                "throughput": {
                    "warning": result['throughput'] * 0.7 if result['throughput'] else 1,
                    "critical": result['throughput'] * 0.5 if result['throughput'] else 0.5
                },
                "error_rate": {
                    "warning": max(0.02, result['error_rate'] * 2),
                    "critical": max(0.05, result['error_rate'] * 5)
                }
            }
        }
    
    # Add business metrics
    for service in SERVICES:
        if service['name'] in baseline['services']:
            for metric in service['business_metrics']:
                baseline['services'][service['name']]['business_metrics'][metric] = {
                    "baseline": 0.8,  # Default good value
                    "warning": 0.6,
                    "critical": 0.5
                }
    
    return baseline

def save_baseline_document(baseline):
    """Save the baseline document to a file."""
    logger.info("Saving baseline document")
    
    # Create directory if it doesn't exist
    baseline_dir = Path("monitoring-alerting-service/docs")
    baseline_dir.mkdir(exist_ok=True)
    
    # Save as JSON
    baseline_file = baseline_dir / "performance_baselines.json"
    with open(baseline_file, "w") as f:
        json.dump(baseline, f, indent=2)
    
    # Save as Markdown
    markdown_file = baseline_dir / "performance_baselines.md"
    with open(markdown_file, "w") as f:
        f.write("# Performance Baselines\n\n")
        f.write("This document defines the performance baselines for all services in the forex trading platform. ")
        f.write("These baselines are used to set appropriate thresholds for alerts and to track performance improvements over time.\n\n")
        
        f.write("## Test Scenarios\n\n")
        for scenario in TEST_SCENARIOS:
            f.write(f"### Scenario: {scenario['name']}\n\n")
            f.write(f"- **Description**: {scenario['description']}\n")
            f.write(f"- **Concurrent Users**: {scenario['concurrent_users']}\n")
            f.write(f"- **Duration**: {scenario['duration_seconds']} seconds\n")
            f.write(f"- **Requests Per Second**: {scenario['requests_per_second']}\n\n")
        
        f.write("## Service Baselines\n\n")
        for service_name, service_data in baseline['services'].items():
            f.write(f"### {service_name}\n\n")
            
            f.write("#### API Performance\n\n")
            f.write("| Endpoint | Scenario | p50 Latency | p95 Latency | p99 Latency | Throughput | Error Rate |\n")
            f.write("|----------|----------|------------|------------|------------|------------|------------|\n")
            
            for endpoint_name, endpoint_data in service_data['api_performance'].items():
                for scenario_name, scenario_data in endpoint_data.items():
                    p50 = f"{scenario_data['latency']['p50']:.2f}ms" if scenario_data['latency']['p50'] else "N/A"
                    p95 = f"{scenario_data['latency']['p95']:.2f}ms" if scenario_data['latency']['p95'] else "N/A"
                    p99 = f"{scenario_data['latency']['p99']:.2f}ms" if scenario_data['latency']['p99'] else "N/A"
                    throughput = f"{scenario_data['throughput']:.2f} req/s" if scenario_data['throughput'] else "N/A"
                    error_rate = f"{scenario_data['error_rate']:.2%}" if scenario_data['error_rate'] is not None else "N/A"
                    
                    f.write(f"| {endpoint_name} | {scenario_name} | {p50} | {p95} | {p99} | {throughput} | {error_rate} |\n")
            
            f.write("\n#### Resource Usage\n\n")
            f.write("| Metric | Baseline | Warning Threshold | Critical Threshold |\n")
            f.write("|--------|----------|-------------------|--------------------|\n")
            
            for metric_name, metric_data in service_data['resource_usage'].items():
                f.write(f"| {metric_name} | {metric_data['baseline']}% | {metric_data['warning']}% | {metric_data['critical']}% |\n")
            
            f.write("\n#### Business Metrics\n\n")
            f.write("| Metric | Baseline | Warning Threshold | Critical Threshold |\n")
            f.write("|--------|----------|-------------------|--------------------|\n")
            
            for metric_name, metric_data in service_data['business_metrics'].items():
                f.write(f"| {metric_name} | {metric_data['baseline']} | {metric_data['warning']} | {metric_data['critical']} |\n")
            
            f.write("\n")
    
    logger.info(f"Baseline document saved to {baseline_file} and {markdown_file}")
    return baseline_file, markdown_file

def generate_prometheus_rules(baseline):
    """Generate Prometheus alerting rules from the baseline document."""
    logger.info("Generating Prometheus alerting rules")
    
    rules = {
        "groups": []
    }
    
    # API performance rules
    api_rules = {
        "name": "api_performance",
        "rules": []
    }
    
    for service_name, service_data in baseline['services'].items():
        for endpoint_name, endpoint_data in service_data['api_performance'].items():
            # Use normal_load scenario as the baseline
            if "normal_load" in endpoint_data:
                scenario_data = endpoint_data["normal_load"]
                
                # Latency rule
                if scenario_data['latency']['p95']:
                    api_rules["rules"].append({
                        "alert": f"{service_name}_{endpoint_name}_high_latency",
                        "expr": f'histogram_quantile(0.95, sum(rate(http_request_duration_seconds_bucket{{service="{service_name}",endpoint="{endpoint_name}"}}[5m])) by (le)) > {scenario_data["alert_thresholds"]["latency"]["warning"] / 1000}',
                        "for": "5m",
                        "labels": {
                            "severity": "warning",
                            "service": service_name,
                            "endpoint": endpoint_name
                        },
                        "annotations": {
                            "summary": f"High latency on {service_name} - {endpoint_name}",
                            "description": f"95th percentile latency for {service_name} - {endpoint_name} is above {scenario_data['alert_thresholds']['latency']['warning']}ms for more than 5 minutes"
                        }
                    })
                    
                    api_rules["rules"].append({
                        "alert": f"{service_name}_{endpoint_name}_critical_latency",
                        "expr": f'histogram_quantile(0.95, sum(rate(http_request_duration_seconds_bucket{{service="{service_name}",endpoint="{endpoint_name}"}}[5m])) by (le)) > {scenario_data["alert_thresholds"]["latency"]["critical"] / 1000}',
                        "for": "5m",
                        "labels": {
                            "severity": "critical",
                            "service": service_name,
                            "endpoint": endpoint_name
                        },
                        "annotations": {
                            "summary": f"Critical latency on {service_name} - {endpoint_name}",
                            "description": f"95th percentile latency for {service_name} - {endpoint_name} is above {scenario_data['alert_thresholds']['latency']['critical']}ms for more than 5 minutes"
                        }
                    })
                
                # Error rate rule
                api_rules["rules"].append({
                    "alert": f"{service_name}_{endpoint_name}_high_error_rate",
                    "expr": f'sum(rate(http_request_errors_total{{service="{service_name}",endpoint="{endpoint_name}"}}[5m])) / sum(rate(http_requests_total{{service="{service_name}",endpoint="{endpoint_name}"}}[5m])) > {scenario_data["alert_thresholds"]["error_rate"]["warning"]}',
                    "for": "5m",
                    "labels": {
                        "severity": "warning",
                        "service": service_name,
                        "endpoint": endpoint_name
                    },
                    "annotations": {
                        "summary": f"High error rate on {service_name} - {endpoint_name}",
                        "description": f"Error rate for {service_name} - {endpoint_name} is above {scenario_data['alert_thresholds']['error_rate']['warning']:.2%} for more than 5 minutes"
                    }
                })
                
                api_rules["rules"].append({
                    "alert": f"{service_name}_{endpoint_name}_critical_error_rate",
                    "expr": f'sum(rate(http_request_errors_total{{service="{service_name}",endpoint="{endpoint_name}"}}[5m])) / sum(rate(http_requests_total{{service="{service_name}",endpoint="{endpoint_name}"}}[5m])) > {scenario_data["alert_thresholds"]["error_rate"]["critical"]}',
                    "for": "5m",
                    "labels": {
                        "severity": "critical",
                        "service": service_name,
                        "endpoint": endpoint_name
                    },
                    "annotations": {
                        "summary": f"Critical error rate on {service_name} - {endpoint_name}",
                        "description": f"Error rate for {service_name} - {endpoint_name} is above {scenario_data['alert_thresholds']['error_rate']['critical']:.2%} for more than 5 minutes"
                    }
                })
    
    rules["groups"].append(api_rules)
    
    # Resource usage rules
    resource_rules = {
        "name": "resource_usage",
        "rules": []
    }
    
    for service_name, service_data in baseline['services'].items():
        # CPU usage rule
        resource_rules["rules"].append({
            "alert": f"{service_name}_high_cpu_usage",
            "expr": f'process_cpu_seconds_total{{service="{service_name}"}} > {service_data["resource_usage"]["cpu_usage"]["warning"]}',
            "for": "5m",
            "labels": {
                "severity": "warning",
                "service": service_name
            },
            "annotations": {
                "summary": f"High CPU usage on {service_name}",
                "description": f"CPU usage for {service_name} is above {service_data['resource_usage']['cpu_usage']['warning']}% for more than 5 minutes"
            }
        })
        
        resource_rules["rules"].append({
            "alert": f"{service_name}_critical_cpu_usage",
            "expr": f'process_cpu_seconds_total{{service="{service_name}"}} > {service_data["resource_usage"]["cpu_usage"]["critical"]}',
            "for": "5m",
            "labels": {
                "severity": "critical",
                "service": service_name
            },
            "annotations": {
                "summary": f"Critical CPU usage on {service_name}",
                "description": f"CPU usage for {service_name} is above {service_data['resource_usage']['cpu_usage']['critical']}% for more than 5 minutes"
            }
        })
        
        # Memory usage rule
        resource_rules["rules"].append({
            "alert": f"{service_name}_high_memory_usage",
            "expr": f'process_resident_memory_bytes{{service="{service_name}"}} / process_virtual_memory_bytes{{service="{service_name}"}} > {service_data["resource_usage"]["memory_usage"]["warning"] / 100}',
            "for": "5m",
            "labels": {
                "severity": "warning",
                "service": service_name
            },
            "annotations": {
                "summary": f"High memory usage on {service_name}",
                "description": f"Memory usage for {service_name} is above {service_data['resource_usage']['memory_usage']['warning']}% for more than 5 minutes"
            }
        })
        
        resource_rules["rules"].append({
            "alert": f"{service_name}_critical_memory_usage",
            "expr": f'process_resident_memory_bytes{{service="{service_name}"}} / process_virtual_memory_bytes{{service="{service_name}"}} > {service_data["resource_usage"]["memory_usage"]["critical"] / 100}',
            "for": "5m",
            "labels": {
                "severity": "critical",
                "service": service_name
            },
            "annotations": {
                "summary": f"Critical memory usage on {service_name}",
                "description": f"Memory usage for {service_name} is above {service_data['resource_usage']['memory_usage']['critical']}% for more than 5 minutes"
            }
        })
    
    rules["groups"].append(resource_rules)
    
    # Business metrics rules
    business_rules = {
        "name": "business_metrics",
        "rules": []
    }
    
    for service_name, service_data in baseline['services'].items():
        for metric_name, metric_data in service_data['business_metrics'].items():
            business_rules["rules"].append({
                "alert": f"{service_name}_{metric_name}_warning",
                "expr": f'{metric_name}{{service="{service_name}"}} < {metric_data["warning"]}',
                "for": "15m",
                "labels": {
                    "severity": "warning",
                    "service": service_name,
                    "metric": metric_name
                },
                "annotations": {
                    "summary": f"Low {metric_name} on {service_name}",
                    "description": f"{metric_name} for {service_name} is below {metric_data['warning']} for more than 15 minutes"
                }
            })
            
            business_rules["rules"].append({
                "alert": f"{service_name}_{metric_name}_critical",
                "expr": f'{metric_name}{{service="{service_name}"}} < {metric_data["critical"]}',
                "for": "15m",
                "labels": {
                    "severity": "critical",
                    "service": service_name,
                    "metric": metric_name
                },
                "annotations": {
                    "summary": f"Critical {metric_name} on {service_name}",
                    "description": f"{metric_name} for {service_name} is below {metric_data['critical']} for more than 15 minutes"
                }
            })
    
    rules["groups"].append(business_rules)
    
    # Save rules to file
    rules_dir = Path("monitoring-alerting-service/config")
    rules_dir.mkdir(exist_ok=True)
    
    rules_file = rules_dir / "performance_alert_rules.yml"
    with open(rules_file, "w") as f:
        yaml.dump(rules, f, default_flow_style=False)
    
    logger.info(f"Prometheus alerting rules saved to {rules_file}")
    return rules_file

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Establish performance baselines")
    parser.add_argument("--service", help="Establish baselines for a specific service")
    parser.add_argument("--all", action="store_true", help="Establish baselines for all services")
    parser.add_argument("--skip-tests", action="store_true", help="Skip performance tests and use default values")
    args = parser.parse_args()
    
    if not args.service and not args.all:
        parser.print_help()
        return
    
    services_to_process = []
    if args.service:
        for service in SERVICES:
            if service["name"] == args.service:
                services_to_process.append(service)
                break
        if not services_to_process:
            logger.error(f"Service {args.service} not found")
            return
    elif args.all:
        services_to_process = SERVICES
    
    results = []
    
    if not args.skip_tests:
        # Run performance tests
        for service in services_to_process:
            for endpoint in service["endpoints"]:
                for scenario in TEST_SCENARIOS:
                    result = run_endpoint_test(service, endpoint, scenario)
                    results.append(result)
    else:
        # Use default values
        logger.info("Skipping tests and using default values")
        for service in services_to_process:
            for endpoint in service["endpoints"]:
                for scenario in TEST_SCENARIOS:
                    results.append({
                        "service": service['name'],
                        "endpoint": endpoint['name'],
                        "scenario": scenario['name'],
                        "latency": {
                            "p50": 100,
                            "p95": 200,
                            "p99": 500,
                            "min": 50,
                            "max": 1000,
                            "mean": 150
                        },
                        "throughput": scenario['requests_per_second'],
                        "error_rate": 0.01
                    })
    
    # Generate baseline document
    baseline = generate_baseline_document(results)
    
    # Save baseline document
    baseline_file, markdown_file = save_baseline_document(baseline)
    
    # Generate Prometheus rules
    rules_file = generate_prometheus_rules(baseline)
    
    logger.info("Performance baseline establishment completed")
    logger.info(f"Baseline document: {baseline_file}")
    logger.info(f"Markdown document: {markdown_file}")
    logger.info(f"Prometheus rules: {rules_file}")

if __name__ == "__main__":
    main()
