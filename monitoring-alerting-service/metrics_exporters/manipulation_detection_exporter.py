"""
Market Manipulation Detection Metrics Exporter

This module exports metrics related to market manipulation detection analytics to the monitoring system.
It tracks usage patterns, detection frequencies, confidence scores, and performance metrics.
"""
import logging
import time
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import json
import os
import requests
from prometheus_client import Counter, Gauge, Histogram, Summary, push_to_gateway
from prometheus_client.exposition import basic_auth_handler
logger = logging.getLogger(__name__)
PROMETHEUS_GATEWAY = os.environ.get('PROMETHEUS_GATEWAY', 'localhost:9091')
PUSH_INTERVAL_SECONDS = int(os.environ.get('METRICS_PUSH_INTERVAL', '30'))
AUTH_USERNAME = os.environ.get('METRICS_AUTH_USERNAME', '')
AUTH_PASSWORD = os.environ.get('METRICS_AUTH_PASSWORD', '')
MANIPULATION_DETECTION_TOTAL = Counter('forex_manipulation_detection_total',
    'Total number of market manipulation detection checks performed', [
    'currency_pair', 'timeframe'])
MANIPULATION_ALERTS_TOTAL = Counter('forex_manipulation_alerts_total',
    'Total number of manipulation alerts raised', ['currency_pair',
    'pattern_type', 'severity'])
MANIPULATION_FALSE_POSITIVES = Counter('forex_manipulation_false_positives',
    'Number of false positive manipulation detections', ['currency_pair',
    'pattern_type'])
MANIPULATION_CONFIDENCE_SCORE = Gauge('forex_manipulation_confidence_score',
    'Confidence score of the latest manipulation detection', [
    'currency_pair', 'pattern_type'])
MANIPULATION_IMPACT_ESTIMATE = Gauge('forex_manipulation_impact_estimate',
    'Estimated market impact of detected manipulation', ['currency_pair'])
MANIPULATION_DETECTION_LATENCY = Histogram(
    'forex_manipulation_detection_latency_seconds',
    'Time taken to complete manipulation detection analysis', [
    'pattern_type'], buckets=(0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0))
MANIPULATION_PATTERN_COMPLEXITY = Summary(
    'forex_manipulation_pattern_complexity',
    'Complexity score of detected manipulation patterns', ['pattern_type'])
MANIPULATION_ML_MODEL_ACCURACY = Gauge('forex_manipulation_ml_model_accuracy',
    'Accuracy score of machine learning model for manipulation detection',
    ['model_version', 'currency_pair'])
MANIPULATION_ANOMALY_SCORE = Gauge('forex_manipulation_anomaly_score',
    'Anomaly score for potential market manipulation', ['currency_pair',
    'anomaly_type'])
MANIPULATION_DETECTION_RELIABILITY = Gauge(
    'forex_manipulation_detection_reliability',
    'Reliability score of manipulation detection system', [
    'detection_algorithm'])
MANIPULATION_COLLABORATIVE_PATTERN = Counter(
    'forex_manipulation_collaborative_pattern',
    'Count of detected collaborative manipulation patterns', [
    'pattern_signature', 'actor_count'])
RISK_MGMT_API_URL = os.environ.get('RISK_MGMT_API_URL',
    'http://risk-management-service:8000')


from monitoring_alerting_service.error.exceptions_bridge import (
    with_exception_handling,
    async_with_exception_handling,
    ForexTradingPlatformError,
    ServiceError,
    DataError,
    ValidationError
)

def auth_handler(url, method, timeout, headers, data):
    """
    Authentication handler for Prometheus push gateway
    """
    return basic_auth_handler(url, method, timeout, headers, data,
        AUTH_USERNAME, AUTH_PASSWORD)


@with_exception_handling
def fetch_and_export_metrics():
    """
    Fetch market manipulation detection metrics and export them to Prometheus
    """
    try:
        response = requests.get(
            f'{RISK_MGMT_API_URL}/api/v1/manipulation/metrics', headers={
            'Content-Type': 'application/json'})
        if response.status_code == 200:
            metrics_data = response.json()
            for detection in metrics_data.get('detection_counts', []):
                MANIPULATION_DETECTION_TOTAL.labels(currency_pair=detection
                    .get('currency_pair', 'unknown'), timeframe=detection.
                    get('timeframe', 'unknown')).inc(detection.get('count', 0))
            for alert in metrics_data.get('alerts', []):
                MANIPULATION_ALERTS_TOTAL.labels(currency_pair=alert.get(
                    'currency_pair', 'unknown'), pattern_type=alert.get(
                    'pattern_type', 'unknown'), severity=alert.get(
                    'severity', 'medium')).inc(alert.get('count', 0))
            for fp in metrics_data.get('false_positives', []):
                MANIPULATION_FALSE_POSITIVES.labels(currency_pair=fp.get(
                    'currency_pair', 'unknown'), pattern_type=fp.get(
                    'pattern_type', 'unknown')).inc(fp.get('count', 0))
            for confidence in metrics_data.get('confidence_scores', []):
                MANIPULATION_CONFIDENCE_SCORE.labels(currency_pair=
                    confidence.get('currency_pair', 'unknown'),
                    pattern_type=confidence.get('pattern_type', 'unknown')
                    ).set(confidence.get('score', 0))
            for impact in metrics_data.get('impact_estimates', []):
                MANIPULATION_IMPACT_ESTIMATE.labels(currency_pair=impact.
                    get('currency_pair', 'unknown')).set(impact.get(
                    'estimate', 0))
            for latency in metrics_data.get('detection_latencies', []):
                MANIPULATION_DETECTION_LATENCY.labels(pattern_type=latency.
                    get('pattern_type', 'unknown')).observe(latency.get(
                    'duration_seconds', 0))
            for complexity in metrics_data.get('pattern_complexities', []):
                MANIPULATION_PATTERN_COMPLEXITY.labels(pattern_type=
                    complexity.get('pattern_type', 'unknown')).observe(
                    complexity.get('complexity_score', 0))
            for ml_model in metrics_data.get('ml_models', []):
                MANIPULATION_ML_MODEL_ACCURACY.labels(model_version=
                    ml_model.get('version', 'unknown'), currency_pair=
                    ml_model.get('currency_pair', 'unknown')).set(ml_model.
                    get('accuracy', 0))
            for anomaly in metrics_data.get('anomalies', []):
                MANIPULATION_ANOMALY_SCORE.labels(currency_pair=anomaly.get
                    ('currency_pair', 'unknown'), anomaly_type=anomaly.get(
                    'type', 'unknown')).set(anomaly.get('score', 0))
            for reliability in metrics_data.get('reliability_scores', []):
                MANIPULATION_DETECTION_RELIABILITY.labels(detection_algorithm
                    =reliability.get('algorithm', 'unknown')).set(reliability
                    .get('score', 0))
            for pattern in metrics_data.get('collaborative_patterns', []):
                MANIPULATION_COLLABORATIVE_PATTERN.labels(pattern_signature
                    =pattern.get('signature', 'unknown'), actor_count=
                    pattern.get('actor_count', 'unknown')).inc(pattern.get(
                    'count', 0))
            logger.info(
                f'Successfully updated market manipulation metrics at {datetime.now().isoformat()}'
                )
        else:
            logger.error(
                f'Failed to fetch market manipulation metrics: {response.status_code} - {response.text}'
                )
    except Exception as e:
        logger.error(f'Error fetching market manipulation metrics: {str(e)}')


@with_exception_handling
def push_metrics_to_gateway():
    """
    Push all collected metrics to the Prometheus push gateway
    """
    try:
        job_name = 'manipulation_detection_exporter'
        instance_id = os.environ.get('INSTANCE_ID', 'default')
        handler = auth_handler if AUTH_USERNAME and AUTH_PASSWORD else None
        push_to_gateway(PROMETHEUS_GATEWAY, job=job_name, registry=None,
            handler=handler, grouping_key={'instance': instance_id})
        logger.info(
            f'Successfully pushed metrics to gateway at {datetime.now().isoformat()}'
            )
    except Exception as e:
        logger.error(f'Error pushing metrics to gateway: {str(e)}')


@with_exception_handling
def collect_and_push_metrics():
    """
    Collect and push metrics to the gateway in a single operation
    """
    try:
        fetch_and_export_metrics()
        push_metrics_to_gateway()
    except Exception as e:
        logger.error(f'Error in metrics collection and export cycle: {str(e)}')


@with_exception_handling
def run_metrics_exporter():
    """
    Main function to run the metrics exporter in a continuous loop
    """
    logger.info(f'Starting Market Manipulation Detection Metrics Exporter')
    logger.info(f'Push interval set to {PUSH_INTERVAL_SECONDS} seconds')
    logger.info(f'Prometheus gateway: {PROMETHEUS_GATEWAY}')
    while True:
        try:
            collect_and_push_metrics()
            time.sleep(PUSH_INTERVAL_SECONDS)
        except KeyboardInterrupt:
            logger.info('Metrics exporter stopped by user')
            break
        except Exception as e:
            logger.error(f'Unexpected error in metrics exporter: {str(e)}')
            time.sleep(5)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format=
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    run_metrics_exporter()


def auth_handler(url, method, timeout, headers, data):
    """Custom auth handler for Prometheus push gateway"""
    if AUTH_USERNAME and AUTH_PASSWORD:
        return basic_auth_handler(url, method, timeout, headers, data,
            AUTH_USERNAME, AUTH_PASSWORD)
    return requests.request(method, url, data=data, headers=headers,
        timeout=timeout)


MANIPULATION_TYPES_DETECTED = Counter('forex_manipulation_types_detected',
    'Number of specific manipulation types detected', ['currency_pair',
    'manipulation_type', 'severity'])
API_REQUESTS = Counter('forex_manipulation_api_requests_total',
    'Total number of API requests to manipulation detection endpoints', [
    'endpoint', 'status_code'])
CONFIDENCE_SCORE = Gauge('forex_manipulation_confidence_score',
    'Current average confidence score of manipulation detection by type', [
    'manipulation_type'])
DETECTION_SYSTEM_HEALTH = Gauge('forex_manipulation_system_health',
    'Health status of the manipulation detection system (1=healthy, 0=unhealthy)'
    , ['component'])
DETECTION_LATENCY = Histogram('forex_manipulation_detection_latency_seconds',
    'Time taken to perform manipulation detection analysis', [
    'detection_type'], buckets=(0.05, 0.1, 0.25, 0.5, 0.75, 1.0, 2.0, 5.0, 
    10.0))
CONFIDENCE_SCORE_DISTRIBUTION = Histogram(
    'forex_manipulation_confidence_score_distribution',
    'Distribution of confidence scores for detected manipulations', [
    'manipulation_type'], buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 
    0.9, 0.95, 0.99))
DETECTION_PROCESSING_TIME = Summary(
    'forex_manipulation_processing_time_seconds',
    'Time spent processing manipulation detection requests', ['detection_type']
    )


class MarketManipulationMetricsExporter:
    """
    Exports metrics related to market manipulation detection for monitoring and alerting.
    """

    def __init__(self):
        """Initialize the metrics exporter."""
        self.last_push_time = datetime.now()
        self.manipulation_stats = {'stop_hunting': {'count': 0,
            'avg_confidence': 0.0}, 'fake_breakout': {'count': 0,
            'avg_confidence': 0.0}, 'volume_anomaly': {'count': 0,
            'avg_confidence': 0.0}}
        for component in ['detector', 'api', 'database']:
            DETECTION_SYSTEM_HEALTH.labels(component=component).set(1.0)

    def record_detection_check(self, currency_pair: str, timeframe: str
        ) ->None:
        """
        Record that a manipulation detection check was performed.
        
        Args:
            currency_pair: The currency pair being checked (e.g., 'EUR/USD')
            timeframe: The timeframe of the analysis (e.g., 'H1')
        """
        MANIPULATION_DETECTION_TOTAL.labels(currency_pair=currency_pair,
            timeframe=timeframe).inc()

    def record_detection(self, currency_pair: str, manipulation_type: str,
        confidence: float, processing_time: float) ->None:
        """
        Record a detected manipulation pattern.
        
        Args:
            currency_pair: The currency pair (e.g., 'EUR/USD')
            manipulation_type: The type of manipulation detected
            confidence: Confidence score (0.0 to 1.0)
            processing_time: Time taken to process the detection in seconds
        """
        severity = ('high' if confidence > 0.8 else 'medium' if confidence >
            0.5 else 'low')
        MANIPULATION_TYPES_DETECTED.labels(currency_pair=currency_pair,
            manipulation_type=manipulation_type, severity=severity).inc()
        CONFIDENCE_SCORE.labels(manipulation_type=manipulation_type).set(
            confidence)
        CONFIDENCE_SCORE_DISTRIBUTION.labels(manipulation_type=
            manipulation_type).observe(confidence)
        DETECTION_PROCESSING_TIME.labels(detection_type=manipulation_type
            ).observe(processing_time)
        if manipulation_type in self.manipulation_stats:
            stats = self.manipulation_stats[manipulation_type]
            stats['avg_confidence'] = (stats['avg_confidence'] * stats[
                'count'] + confidence) / (stats['count'] + 1)
            stats['count'] += 1

    def record_api_request(self, endpoint: str, status_code: int) ->None:
        """
        Record an API request to manipulation detection endpoints.
        
        Args:
            endpoint: The API endpoint path
            status_code: HTTP status code of the response
        """
        API_REQUESTS.labels(endpoint=endpoint, status_code=str(status_code)
            ).inc()

    def record_detection_latency(self, detection_type: str, latency: float
        ) ->None:
        """
        Record the latency of a manipulation detection operation.
        
        Args:
            detection_type: The type of detection performed
            latency: Latency in seconds
        """
        DETECTION_LATENCY.labels(detection_type=detection_type).observe(latency
            )

    def update_system_health(self, component: str, is_healthy: bool) ->None:
        """
        Update the health status of a system component.
        
        Args:
            component: The component being monitored (e.g., 'detector', 'api', 'database')
            is_healthy: Whether the component is currently healthy
        """
        DETECTION_SYSTEM_HEALTH.labels(component=component).set(1.0 if
            is_healthy else 0.0)

    @with_exception_handling
    def push_metrics(self) ->bool:
        """
        Push metrics to Prometheus gateway if the push interval has elapsed.
        
        Returns:
            bool: True if metrics were pushed, False otherwise
        """
        now = datetime.now()
        if (now - self.last_push_time).total_seconds() < PUSH_INTERVAL_SECONDS:
            return False
        try:
            auth_handler = None
            if AUTH_USERNAME and AUTH_PASSWORD:

                def auth_handler(url, method, timeout, headers, data):
    """
    Auth handler.
    
    Args:
        url: Description of url
        method: Description of method
        timeout: Description of timeout
        headers: Description of headers
        data: Description of data
    
    """

                    return basic_auth_handler(url, method, timeout, headers,
                        data, AUTH_USERNAME, AUTH_PASSWORD)
            push_to_gateway(PROMETHEUS_GATEWAY, job=
                'market_manipulation_detection', handler=auth_handler)
            self.last_push_time = now
            logger.info(
                f'Successfully pushed market manipulation metrics to gateway')
            return True
        except Exception as e:
            logger.error(f'Failed to push metrics to gateway: {e}')
            return False

    def periodic_metrics_update(self) ->None:
        """Run periodic updates for metrics that need refreshing."""
        for manipulation_type, stats in self.manipulation_stats.items():
            if stats['count'] > 0:
                CONFIDENCE_SCORE.labels(manipulation_type=manipulation_type
                    ).set(stats['avg_confidence'])
        self.push_metrics()


metrics_exporter = MarketManipulationMetricsExporter()
