#!/usr/bin/env python3
"""
Regular Performance Testing Setup Script

This script sets up regular performance testing for the forex trading platform.
It creates a scheduled job that runs performance tests and compares the results
against the established baselines.
"""

import os
import sys
import argparse
import logging
import json
import yaml
import datetime
from pathlib import Path
import subprocess

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("regular-performance-testing-setup")

# Default configuration
DEFAULT_CONFIG = {
    "schedule": {
        "daily_test": {
            "cron": "0 1 * * *",  # Run at 1:00 AM every day
            "test_type": "normal",
            "services": "all"
        },
        "weekly_test": {
            "cron": "0 2 * * 0",  # Run at 2:00 AM every Sunday
            "test_type": "comprehensive",
            "services": "all"
        },
        "monthly_test": {
            "cron": "0 3 1 * *",  # Run at 3:00 AM on the 1st of every month
            "test_type": "full",
            "services": "all"
        }
    },
    "test_types": {
        "normal": {
            "scenarios": ["normal_load"],
            "duration": 60,
            "compare_to_baseline": True,
            "alert_on_regression": True
        },
        "comprehensive": {
            "scenarios": ["normal_load", "high_load"],
            "duration": 120,
            "compare_to_baseline": True,
            "alert_on_regression": True
        },
        "full": {
            "scenarios": ["normal_load", "high_load", "peak_load"],
            "duration": 300,
            "compare_to_baseline": True,
            "alert_on_regression": True
        }
    },
    "reporting": {
        "store_results": True,
        "results_retention_days": 90,
        "generate_report": True,
        "send_report_email": True,
        "email_recipients": ["forex-platform-team@example.com"]
    },
    "alerting": {
        "regression_threshold_warning": 1.2,  # 20% worse than baseline
        "regression_threshold_critical": 1.5,  # 50% worse than baseline
        "alert_channels": ["email", "slack"]
    }
}

def create_config_file(config=None):
    """Create the configuration file for regular performance testing."""
    logger.info("Creating configuration file")
    
    if config is None:
        config = DEFAULT_CONFIG
    
    # Create directory if it doesn't exist
    config_dir = Path("monitoring-alerting-service/config")
    config_dir.mkdir(exist_ok=True)
    
    # Save as YAML
    config_file = config_dir / "regular_performance_testing.yml"
    with open(config_file, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    
    logger.info(f"Configuration file saved to {config_file}")
    return config_file

def create_performance_test_script():
    """Create the performance test script."""
    logger.info("Creating performance test script")
    
    # Create directory if it doesn't exist
    scripts_dir = Path("monitoring-alerting-service/scripts")
    scripts_dir.mkdir(exist_ok=True)
    
    # Create script
    script_file = scripts_dir / "run_performance_test.py"
    
    script_content = """#!/usr/bin/env python3
\"\"\"
Performance Test Runner Script

This script runs performance tests for the forex trading platform and compares
the results against the established baselines.
\"\"\"

import os
import sys
import argparse
import logging
import json
import yaml
import datetime
import time
import statistics
import requests
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import subprocess
from concurrent.futures import ThreadPoolExecutor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("performance-test-runner")

def load_config():
    """
    Load config.
    
    """

    \"\"\"Load the configuration file.\"\"\"
    config_file = Path("monitoring-alerting-service/config/regular_performance_testing.yml")
    
    if not config_file.exists():
        logger.error(f"Configuration file {config_file} not found")
        sys.exit(1)
    
    with open(config_file, "r") as f:
        return yaml.safe_load(f)

def load_baseline():
    """
    Load baseline.
    
    """

    \"\"\"Load the baseline document.\"\"\"
    baseline_file = Path("monitoring-alerting-service/docs/performance_baselines.json")
    
    if not baseline_file.exists():
        logger.error(f"Baseline file {baseline_file} not found")
        sys.exit(1)
    
    with open(baseline_file, "r") as f:
        return json.load(f)

def run_performance_test(test_type, services="all"):
    """
    Run performance test.
    
    Args:
        test_type: Description of test_type
        services: Description of services
    
    """

    \"\"\"Run a performance test.\"\"\"
    logger.info(f"Running {test_type} performance test for {services}")
    
    config = load_config()
    baseline = load_baseline()
    
    if test_type not in config["test_types"]:
        logger.error(f"Test type {test_type} not found in configuration")
        return None
    
    test_config = config["test_types"][test_type]
    
    # Determine which services to test
    if services == "all":
        services_to_test = list(baseline["services"].keys())
    else:
        services_to_test = services.split(",")
    
    # Run the performance test
    cmd = [
        sys.executable,
        "monitoring-alerting-service/scripts/establish_performance_baselines.py",
        "--skip-tests"  # For now, we'll skip actual tests and use default values
    ]
    
    if services != "all":
        cmd.extend(["--service", services])
    else:
        cmd.append("--all")
    
    logger.info(f"Running command: {' '.join(cmd)}")
    
    try:
        subprocess.run(cmd, check=True)
        logger.info("Performance test completed successfully")
    except subprocess.CalledProcessError as e:
        logger.error(f"Performance test failed: {e}")
        return None
    
    # Load the results
    results_file = Path("monitoring-alerting-service/docs/performance_baselines.json")
    
    if not results_file.exists():
        logger.error(f"Results file {results_file} not found")
        return None
    
    with open(results_file, "r") as f:
        results = json.load(f)
    
    # Compare results to baseline
    if test_config["compare_to_baseline"]:
        comparison = compare_to_baseline(results, baseline)
        
        # Generate report
        if config["reporting"]["generate_report"]:
            report_file = generate_report(results, baseline, comparison)
            logger.info(f"Report generated: {report_file}")
        
        # Check for regressions
        if test_config["alert_on_regression"]:
            check_for_regressions(comparison, config["alerting"])
    
    return results

def compare_to_baseline(results, baseline):
    """
    Compare to baseline.
    
    Args:
        results: Description of results
        baseline: Description of baseline
    
    """

    \"\"\"Compare test results to baseline.\"\"\"
    logger.info("Comparing results to baseline")
    
    comparison = {
        "services": {}
    }
    
    for service_name, service_data in results["services"].items():
        if service_name not in baseline["services"]:
            logger.warning(f"Service {service_name} not found in baseline, skipping")
            continue
        
        comparison["services"][service_name] = {
            "api_performance": {},
            "resource_usage": {},
            "business_metrics": {}
        }
        
        # Compare API performance
        for endpoint_name, endpoint_data in service_data["api_performance"].items():
            if endpoint_name not in baseline["services"][service_name]["api_performance"]:
                logger.warning(f"Endpoint {endpoint_name} not found in baseline for {service_name}, skipping")
                continue
            
            comparison["services"][service_name]["api_performance"][endpoint_name] = {}
            
            for scenario_name, scenario_data in endpoint_data.items():
                if scenario_name not in baseline["services"][service_name]["api_performance"][endpoint_name]:
                    logger.warning(f"Scenario {scenario_name} not found in baseline for {service_name}/{endpoint_name}, skipping")
                    continue
                
                baseline_scenario = baseline["services"][service_name]["api_performance"][endpoint_name][scenario_name]
                
                # Calculate percentage changes
                latency_p50_change = None
                if scenario_data["latency"]["p50"] and baseline_scenario["latency"]["p50"]:
                    latency_p50_change = scenario_data["latency"]["p50"] / baseline_scenario["latency"]["p50"]
                
                latency_p95_change = None
                if scenario_data["latency"]["p95"] and baseline_scenario["latency"]["p95"]:
                    latency_p95_change = scenario_data["latency"]["p95"] / baseline_scenario["latency"]["p95"]
                
                latency_p99_change = None
                if scenario_data["latency"]["p99"] and baseline_scenario["latency"]["p99"]:
                    latency_p99_change = scenario_data["latency"]["p99"] / baseline_scenario["latency"]["p99"]
                
                throughput_change = None
                if scenario_data["throughput"] and baseline_scenario["throughput"]:
                    throughput_change = scenario_data["throughput"] / baseline_scenario["throughput"]
                
                error_rate_change = None
                if scenario_data["error_rate"] is not None and baseline_scenario["error_rate"] is not None:
                    if baseline_scenario["error_rate"] > 0:
                        error_rate_change = scenario_data["error_rate"] / baseline_scenario["error_rate"]
                    else:
                        error_rate_change = 1.0 if scenario_data["error_rate"] == 0 else float('inf')
                
                comparison["services"][service_name]["api_performance"][endpoint_name][scenario_name] = {
                    "latency": {
                        "p50": {
                            "baseline": baseline_scenario["latency"]["p50"],
                            "current": scenario_data["latency"]["p50"],
                            "change": latency_p50_change
                        },
                        "p95": {
                            "baseline": baseline_scenario["latency"]["p95"],
                            "current": scenario_data["latency"]["p95"],
                            "change": latency_p95_change
                        },
                        "p99": {
                            "baseline": baseline_scenario["latency"]["p99"],
                            "current": scenario_data["latency"]["p99"],
                            "change": latency_p99_change
                        }
                    },
                    "throughput": {
                        "baseline": baseline_scenario["throughput"],
                        "current": scenario_data["throughput"],
                        "change": throughput_change
                    },
                    "error_rate": {
                        "baseline": baseline_scenario["error_rate"],
                        "current": scenario_data["error_rate"],
                        "change": error_rate_change
                    }
                }
    
    return comparison

def generate_report(results, baseline, comparison):
    """
    Generate report.
    
    Args:
        results: Description of results
        baseline: Description of baseline
        comparison: Description of comparison
    
    """

    \"\"\"Generate a performance test report.\"\"\"
    logger.info("Generating performance test report")
    
    # Create directory if it doesn't exist
    reports_dir = Path("monitoring-alerting-service/reports")
    reports_dir.mkdir(exist_ok=True)
    
    # Create report file
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = reports_dir / f"performance_test_report_{timestamp}.md"
    
    with open(report_file, "w") as f:
        f.write("# Performance Test Report\n\n")
        f.write(f"**Date:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## Summary\n\n")
        f.write("This report compares the current performance test results against the established baselines.\n\n")
        
        f.write("## Service Performance\n\n")
        
        for service_name, service_data in comparison["services"].items():
            f.write(f"### {service_name}\n\n")
            
            f.write("#### API Performance\n\n")
            f.write("| Endpoint | Scenario | Metric | Baseline | Current | Change |\n")
            f.write("|----------|----------|--------|----------|---------|--------|\n")
            
            for endpoint_name, endpoint_data in service_data["api_performance"].items():
                for scenario_name, scenario_data in endpoint_data.items():
                    # Latency p95
                    baseline_p95 = f"{scenario_data['latency']['p95']['baseline']:.2f}ms" if scenario_data['latency']['p95']['baseline'] else "N/A"
                    current_p95 = f"{scenario_data['latency']['p95']['current']:.2f}ms" if scenario_data['latency']['p95']['current'] else "N/A"
                    
                    if scenario_data['latency']['p95']['change']:
                        if scenario_data['latency']['p95']['change'] > 1.0:
                            change_p95 = f"🔴 +{(scenario_data['latency']['p95']['change'] - 1) * 100:.1f}%"
                        elif scenario_data['latency']['p95']['change'] < 1.0:
                            change_p95 = f"🟢 {(scenario_data['latency']['p95']['change'] - 1) * 100:.1f}%"
                        else:
                            change_p95 = "0%"
                    else:
                        change_p95 = "N/A"
                    
                    f.write(f"| {endpoint_name} | {scenario_name} | Latency (p95) | {baseline_p95} | {current_p95} | {change_p95} |\n")
                    
                    # Throughput
                    baseline_throughput = f"{scenario_data['throughput']['baseline']:.2f} req/s" if scenario_data['throughput']['baseline'] else "N/A"
                    current_throughput = f"{scenario_data['throughput']['current']:.2f} req/s" if scenario_data['throughput']['current'] else "N/A"
                    
                    if scenario_data['throughput']['change']:
                        if scenario_data['throughput']['change'] < 1.0:
                            change_throughput = f"🔴 {(scenario_data['throughput']['change'] - 1) * 100:.1f}%"
                        elif scenario_data['throughput']['change'] > 1.0:
                            change_throughput = f"🟢 +{(scenario_data['throughput']['change'] - 1) * 100:.1f}%"
                        else:
                            change_throughput = "0%"
                    else:
                        change_throughput = "N/A"
                    
                    f.write(f"| {endpoint_name} | {scenario_name} | Throughput | {baseline_throughput} | {current_throughput} | {change_throughput} |\n")
                    
                    # Error rate
                    baseline_error = f"{scenario_data['error_rate']['baseline']:.2%}" if scenario_data['error_rate']['baseline'] is not None else "N/A"
                    current_error = f"{scenario_data['error_rate']['current']:.2%}" if scenario_data['error_rate']['current'] is not None else "N/A"
                    
                    if scenario_data['error_rate']['change']:
                        if scenario_data['error_rate']['change'] > 1.0:
                            change_error = f"🔴 +{(scenario_data['error_rate']['change'] - 1) * 100:.1f}%"
                        elif scenario_data['error_rate']['change'] < 1.0:
                            change_error = f"🟢 {(scenario_data['error_rate']['change'] - 1) * 100:.1f}%"
                        else:
                            change_error = "0%"
                    else:
                        change_error = "N/A"
                    
                    f.write(f"| {endpoint_name} | {scenario_name} | Error Rate | {baseline_error} | {current_error} | {change_error} |\n")
            
            f.write("\n")
    
    logger.info(f"Report saved to {report_file}")
    return report_file

def check_for_regressions(comparison, alerting_config):
    """
    Check for regressions.
    
    Args:
        comparison: Description of comparison
        alerting_config: Description of alerting_config
    
    """

    \"\"\"Check for performance regressions and trigger alerts if necessary.\"\"\"
    logger.info("Checking for performance regressions")
    
    warning_threshold = alerting_config["regression_threshold_warning"]
    critical_threshold = alerting_config["regression_threshold_critical"]
    
    regressions = {
        "warning": [],
        "critical": []
    }
    
    for service_name, service_data in comparison["services"].items():
        for endpoint_name, endpoint_data in service_data["api_performance"].items():
            for scenario_name, scenario_data in endpoint_data.items():
                # Check latency
                if scenario_data["latency"]["p95"]["change"] and scenario_data["latency"]["p95"]["change"] > warning_threshold:
                    regression = {
                        "service": service_name,
                        "endpoint": endpoint_name,
                        "scenario": scenario_name,
                        "metric": "latency_p95",
                        "baseline": scenario_data["latency"]["p95"]["baseline"],
                        "current": scenario_data["latency"]["p95"]["current"],
                        "change": scenario_data["latency"]["p95"]["change"]
                    }
                    
                    if scenario_data["latency"]["p95"]["change"] > critical_threshold:
                        regressions["critical"].append(regression)
                    else:
                        regressions["warning"].append(regression)
                
                # Check throughput
                if scenario_data["throughput"]["change"] and scenario_data["throughput"]["change"] < (1 / warning_threshold):
                    regression = {
                        "service": service_name,
                        "endpoint": endpoint_name,
                        "scenario": scenario_name,
                        "metric": "throughput",
                        "baseline": scenario_data["throughput"]["baseline"],
                        "current": scenario_data["throughput"]["current"],
                        "change": scenario_data["throughput"]["change"]
                    }
                    
                    if scenario_data["throughput"]["change"] < (1 / critical_threshold):
                        regressions["critical"].append(regression)
                    else:
                        regressions["warning"].append(regression)
                
                # Check error rate
                if scenario_data["error_rate"]["change"] and scenario_data["error_rate"]["change"] > warning_threshold:
                    regression = {
                        "service": service_name,
                        "endpoint": endpoint_name,
                        "scenario": scenario_name,
                        "metric": "error_rate",
                        "baseline": scenario_data["error_rate"]["baseline"],
                        "current": scenario_data["error_rate"]["current"],
                        "change": scenario_data["error_rate"]["change"]
                    }
                    
                    if scenario_data["error_rate"]["change"] > critical_threshold:
                        regressions["critical"].append(regression)
                    else:
                        regressions["warning"].append(regression)
    
    # Log regressions
    if regressions["warning"] or regressions["critical"]:
        logger.warning(f"Found {len(regressions['warning'])} warning and {len(regressions['critical'])} critical regressions")
        
        # TODO: Send alerts
        # This would integrate with your alerting system (email, Slack, etc.)
        
        # For now, just log the regressions
        for severity, regression_list in regressions.items():
            for regression in regression_list:
                logger.warning(
                    f"{severity.upper()} regression: {regression['service']} - {regression['endpoint']} - "
                    f"{regression['scenario']} - {regression['metric']} - "
                    f"Baseline: {regression['baseline']}, Current: {regression['current']}, "
                    f"Change: {regression['change']:.2f}x"
                )
    else:
        logger.info("No performance regressions found")

def main():
    """
    Main.
    
    """

    \"\"\"Main function.\"\"\"
    parser = argparse.ArgumentParser(description="Run performance tests")
    parser.add_argument("--type", choices=["normal", "comprehensive", "full"], default="normal", help="Test type")
    parser.add_argument("--services", default="all", help="Comma-separated list of services to test")
    args = parser.parse_args()
    
    run_performance_test(args.type, args.services)

if __name__ == "__main__":
    main()
"""
    
    with open(script_file, "w") as f:
        f.write(script_content)
    
    # Make the script executable
    script_file.chmod(0o755)
    
    logger.info(f"Performance test script saved to {script_file}")
    return script_file

def create_cron_jobs(config_file):
    """Create cron jobs for regular performance testing."""
    logger.info("Creating cron jobs")
    
    # Load configuration
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
    
    # Create crontab entries
    crontab_entries = []
    
    for job_name, job_config in config["schedule"].items():
        cron_expression = job_config["cron"]
        test_type = job_config["test_type"]
        services = job_config["services"]
        
        command = f"cd {os.getcwd()} && python3 monitoring-alerting-service/scripts/run_performance_test.py --type {test_type} --services {services} >> /var/log/forex-platform/performance-tests.log 2>&1"
        
        crontab_entries.append(f"{cron_expression} {command} # {job_name}")
    
    # Write crontab file
    crontab_file = Path("monitoring-alerting-service/config/performance_test_crontab")
    with open(crontab_file, "w") as f:
        f.write("# Forex Platform Performance Test Cron Jobs\n")
        f.write("# These jobs run regular performance tests and compare against baselines\n\n")
        f.write("\n".join(crontab_entries))
        f.write("\n")
    
    logger.info(f"Crontab file saved to {crontab_file}")
    
    # Print instructions for installing crontab
    logger.info("To install the crontab, run:")
    logger.info(f"  crontab {crontab_file}")
    
    return crontab_file

def create_documentation():
    """Create documentation for regular performance testing."""
    logger.info("Creating documentation")
    
    # Create directory if it doesn't exist
    docs_dir = Path("monitoring-alerting-service/docs")
    docs_dir.mkdir(exist_ok=True)
    
    # Create documentation file
    docs_file = docs_dir / "regular_performance_testing.md"
    
    docs_content = """# Regular Performance Testing

This document describes the regular performance testing setup for the forex trading platform.

## Overview

Regular performance testing is essential to ensure that the platform maintains its performance characteristics over time. It helps to:

- Detect performance regressions early
- Validate that performance improvements are maintained
- Establish trends in performance over time
- Provide data for capacity planning

## Test Types

The following test types are defined:

### Normal Test

- Runs daily
- Tests normal load scenarios
- Duration: 60 seconds
- Compares results to baseline
- Alerts on regressions

### Comprehensive Test

- Runs weekly
- Tests normal and high load scenarios
- Duration: 120 seconds
- Compares results to baseline
- Alerts on regressions

### Full Test

- Runs monthly
- Tests normal, high, and peak load scenarios
- Duration: 300 seconds
- Compares results to baseline
- Alerts on regressions

## Schedule

- Daily test: Runs at 1:00 AM every day
- Weekly test: Runs at 2:00 AM every Sunday
- Monthly test: Runs at 3:00 AM on the 1st of every month

## Reporting

Performance test results are stored and retained for 90 days. A report is generated after each test and sent to the forex platform team.

## Alerting

Alerts are triggered when performance regressions are detected:

- Warning: Performance is 20% worse than baseline
- Critical: Performance is 50% worse than baseline

Alerts are sent via email and Slack.

## Configuration

The configuration for regular performance testing is stored in `monitoring-alerting-service/config/regular_performance_testing.yml`.

## Running Tests Manually

To run a performance test manually, use the following command:

```bash
python3 monitoring-alerting-service/scripts/run_performance_test.py --type [normal|comprehensive|full] --services [all|service1,service2,...]
```

For example, to run a comprehensive test for the trading-gateway-service:

```bash
python3 monitoring-alerting-service/scripts/run_performance_test.py --type comprehensive --services trading-gateway-service
```

## Viewing Results

Performance test reports are stored in `monitoring-alerting-service/reports/`.
"""
    
    with open(docs_file, "w") as f:
        f.write(docs_content)
    
    logger.info(f"Documentation saved to {docs_file}")
    return docs_file

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Set up regular performance testing")
    parser.add_argument("--config", help="Path to custom configuration file")
    args = parser.parse_args()
    
    # Create configuration file
    if args.config:
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)
        config_file = create_config_file(config)
    else:
        config_file = create_config_file()
    
    # Create performance test script
    script_file = create_performance_test_script()
    
    # Create cron jobs
    crontab_file = create_cron_jobs(config_file)
    
    # Create documentation
    docs_file = create_documentation()
    
    logger.info("Regular performance testing setup completed")
    logger.info(f"Configuration: {config_file}")
    logger.info(f"Test script: {script_file}")
    logger.info(f"Crontab: {crontab_file}")
    logger.info(f"Documentation: {docs_file}")

if __name__ == "__main__":
    main()
