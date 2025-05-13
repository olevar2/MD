"""
Correlation Analysis Metrics Exporter

This module exports advanced correlation analysis metrics to Prometheus for monitoring and alerting.
It periodically fetches metrics from the analysis engine API and converts them
to Prometheus metrics for visualization in Grafana.
"""
import time
import logging
import requests
from typing import Dict, List, Any, Optional
import os
import json
from datetime import datetime, timedelta
from prometheus_client import start_http_server, Gauge, Counter, Summary, Histogram
logging.basicConfig(level=logging.INFO, format=
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
ANALYSIS_ENGINE_BASE_URL = os.environ.get('ANALYSIS_ENGINE_BASE_URL',
    'http://localhost:8000')
METRICS_PORT = int(os.environ.get('METRICS_PORT', 8003))
METRICS_UPDATE_INTERVAL = int(os.environ.get('METRICS_UPDATE_INTERVAL', 60))
CORRELATION_STRENGTH = Gauge('correlation_analysis_strength',
    'Strength of correlation between assets', ['asset_pair', 'timeframe'])
CORRELATION_STABILITY = Gauge('correlation_stability_score',
    'Stability of correlation over time', ['asset_pair', 'timeframe'])
CORRELATION_LAG_TIME = Gauge('correlation_lead_lag_time',
    'Lead-lag relationship time in minutes', ['leading_asset',
    'lagging_asset', 'timeframe'])
CORRELATION_BREAKDOWN_ALERTS = Counter('correlation_breakdown_alerts_total',
    'Number of correlation breakdown alerts generated', ['asset_pair',
    'severity'])
CORRELATION_PROCESSING_TIME = Summary('correlation_processing_time_seconds',
    'Time spent processing correlation analysis', ['analysis_type',
    'timeframe'])
CORRELATION_ANOMALY_SCORE = Gauge('correlation_anomaly_score',
    'Anomaly score for correlation patterns', ['asset_pair', 'timeframe'])
CORRELATION_REQUEST_FAILURES = Counter('correlation_request_failures_total',
    'Number of failed correlation analysis API requests', ['endpoint',
    'error_type'])
CORRELATION_PATTERN_COUNT = Counter('correlation_patterns_detected_total',
    'Number of correlation patterns detected', ['pattern_type', 'asset_pair'])


from monitoring_alerting_service.error.exceptions_bridge import (
    with_exception_handling,
    async_with_exception_handling,
    ForexTradingPlatformError,
    ServiceError,
    DataError,
    ValidationError
)

class CorrelationMetricsExporter:
    """Exports correlation analysis metrics to Prometheus."""

    def __init__(self, base_url=ANALYSIS_ENGINE_BASE_URL, update_interval=
        METRICS_UPDATE_INTERVAL):
        """Initialize the correlation metrics exporter."""
        self.base_url = base_url
        self.update_interval = update_interval
        self.session = requests.Session()
        self.headers = {'Content-Type': 'application/json'}

    def start_metrics_server(self, port=METRICS_PORT):
        """Start the Prometheus metrics server."""
        start_http_server(port)
        logger.info(f'Started Prometheus metrics server on port {port}')

    @with_exception_handling
    def fetch_correlation_metrics(self) ->Optional[Dict[str, Any]]:
        """Fetch correlation analysis metrics from the API."""
        try:
            response = self.session.get(
                f'{self.base_url}/api/v1/correlation/metrics', headers=self
                .headers, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.error(f'Failed to fetch correlation metrics: {str(e)}')
            CORRELATION_REQUEST_FAILURES.labels(endpoint='metrics',
                error_type=type(e).__name__).inc()
            return None

    @with_exception_handling
    def fetch_lead_lag_metrics(self) ->Optional[Dict[str, Any]]:
        """Fetch lead-lag relationship metrics from the API."""
        try:
            response = self.session.get(
                f'{self.base_url}/api/v1/correlation/lead-lag/metrics',
                headers=self.headers, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.error(f'Failed to fetch lead-lag metrics: {str(e)}')
            CORRELATION_REQUEST_FAILURES.labels(endpoint='lead-lag',
                error_type=type(e).__name__).inc()
            return None

    @with_exception_handling
    def fetch_breakdown_alerts(self) ->Optional[Dict[str, Any]]:
        """Fetch correlation breakdown alerts from the API."""
        try:
            response = self.session.get(
                f'{self.base_url}/api/v1/correlation/breakdown/metrics',
                headers=self.headers, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.error(f'Failed to fetch breakdown metrics: {str(e)}')
            CORRELATION_REQUEST_FAILURES.labels(endpoint='breakdown',
                error_type=type(e).__name__).inc()
            return None

    def update_correlation_metrics(self):
        """Update correlation metrics in Prometheus."""
        metrics = self.fetch_correlation_metrics()
        if metrics:
            for asset_pair, timeframes in metrics.get('correlation_data', {}
                ).items():
                for timeframe, data in timeframes.items():
                    CORRELATION_STRENGTH.labels(asset_pair=asset_pair,
                        timeframe=timeframe).set(data.get('strength', 0))
                    CORRELATION_STABILITY.labels(asset_pair=asset_pair,
                        timeframe=timeframe).set(data.get('stability', 0))
                    CORRELATION_ANOMALY_SCORE.labels(asset_pair=asset_pair,
                        timeframe=timeframe).set(data.get('anomaly_score', 0))
            for analysis_type, timeframes in metrics.get('processing_times', {}
                ).items():
                for timeframe, time_data in timeframes.items():
                    CORRELATION_PROCESSING_TIME.labels(analysis_type=
                        analysis_type, timeframe=timeframe).observe(time_data
                        .get('avg_seconds', 0))
            for pattern_type, asset_pairs in metrics.get('pattern_counts', {}
                ).items():
                for asset_pair, count in asset_pairs.items():
                    CORRELATION_PATTERN_COUNT.labels(pattern_type=
                        pattern_type, asset_pair=asset_pair).inc(count)
        lead_lag_data = self.fetch_lead_lag_metrics()
        if lead_lag_data:
            for leading_asset, lagging_assets in lead_lag_data.get(
                'lead_lag_relationships', {}).items():
                for lagging_asset, timeframes in lagging_assets.items():
                    for timeframe, lag_time in timeframes.items():
                        CORRELATION_LAG_TIME.labels(leading_asset=
                            leading_asset, lagging_asset=lagging_asset,
                            timeframe=timeframe).set(lag_time)
        breakdown_data = self.fetch_breakdown_alerts()
        if breakdown_data:
            for asset_pair, severities in breakdown_data.get('breakdown_alerts'
                , {}).items():
                for severity, count in severities.items():
                    CORRELATION_BREAKDOWN_ALERTS.labels(asset_pair=
                        asset_pair, severity=severity).inc(count)

    @with_exception_handling
    def run_metrics_loop(self):
        """Run the metrics export loop."""
        while True:
            try:
                logger.info('Updating correlation metrics...')
                self.update_correlation_metrics()
                logger.info('Correlation metrics updated successfully')
            except Exception as e:
                logger.error(f'Error updating correlation metrics: {str(e)}')
            time.sleep(self.update_interval)


def main():
    """Main entry point for the correlation metrics exporter."""
    exporter = CorrelationMetricsExporter()
    exporter.start_metrics_server()
    logger.info('Correlation metrics exporter started')
    exporter.run_metrics_loop()


if __name__ == '__main__':
    main()
