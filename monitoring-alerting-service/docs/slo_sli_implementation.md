# SLO and SLI Implementation

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
