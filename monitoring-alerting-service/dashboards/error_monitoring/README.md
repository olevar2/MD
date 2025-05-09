# Error Monitoring Dashboard

This dashboard provides comprehensive monitoring of errors across the Forex Trading Platform.

## Overview

The Error Monitoring Dashboard visualizes error metrics collected from all services in the platform. It helps identify error patterns, track error rates, and monitor the health of the system.

## Features

- **Error Rate by Service**: Visualizes the error rate for each service over time
- **Top Error Codes**: Shows the most frequent error codes across the platform
- **Top Services with Errors**: Identifies services with the highest error counts
- **Error Rate by Error Code**: Tracks error rates by error code over time
- **Recent Error Logs**: Displays recent error logs from all services
- **HTTP 5xx Error Rate**: Shows the rate of HTTP 5xx errors by service
- **Circuit Breaker Status**: Displays the current status of circuit breakers across services

## How to Use

1. **Filter by Service**: Use the "Service" dropdown to filter the dashboard by specific services
2. **Filter by Error Code**: Use the "Error Code" dropdown to filter by specific error codes
3. **Time Range**: Adjust the time range to view errors over different periods
4. **Drill Down**: Click on specific errors or services to drill down for more details

## Alert Rules

The dashboard is integrated with the following alert rules:

- **High Error Rate**: Triggers when a service has a high error rate (> 0.1 errors/s) for 5 minutes
- **Critical Error Rate**: Triggers when a service has a critical error rate (> 1 errors/s) for 2 minutes
- **HTTP 5xx Error Rate**: Triggers when a service has a high rate of HTTP 5xx errors (> 0.1 errors/s) for 5 minutes
- **Circuit Breaker Open**: Triggers when a circuit breaker is open for 1 minute
- **Data Validation Errors**: Triggers when a service has a high rate of data validation errors (> 0.2 errors/s) for 5 minutes
- **Service Unavailable Errors**: Triggers when a service reports service unavailable errors (> 0.1 errors/s) for 2 minutes
- **Trading Errors**: Triggers when a service reports trading errors (> 0.05 errors/s) for 2 minutes
- **Backtest Errors**: Triggers when a service reports backtest errors (> 0.1 errors/s) for 5 minutes
- **Analysis Errors**: Triggers when a service reports analysis errors (> 0.1 errors/s) for 5 minutes
- **Authentication Errors**: Triggers when a service reports authentication errors (> 0.2 errors/s) for 5 minutes
- **Security Errors**: Triggers when a service reports security errors (> 0.1 errors/s) for 2 minutes

## Notifications

Alerts can be configured to send notifications through the following channels:

- **Email**: Sends detailed error information via email
- **Slack**: Sends error notifications to a Slack channel
- **Microsoft Teams**: Sends error notifications to a Microsoft Teams channel

## Integration

The dashboard integrates with the following components:

- **Prometheus**: Collects error metrics from all services
- **Grafana**: Visualizes error metrics and provides alerting
- **Elasticsearch**: Stores and indexes error logs
- **Alertmanager**: Manages alert notifications

## Maintenance

To update the dashboard:

1. Edit the `error_dashboard.json` file
2. Import the updated dashboard into Grafana
3. Update alert rules as needed

## Troubleshooting

If the dashboard is not showing data:

1. Check that Prometheus is collecting metrics from all services
2. Verify that services are properly instrumented to report error metrics
3. Check Grafana's data source configuration
4. Ensure that the error metrics exporter is properly initialized in all services
