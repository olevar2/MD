"""
NLP Analysis Metrics Exporter

This module exports NLP analysis metrics to Prometheus for monitoring and alerting.
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

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
ANALYSIS_ENGINE_BASE_URL = os.environ.get('ANALYSIS_ENGINE_BASE_URL', 'http://localhost:8000')
METRICS_PORT = int(os.environ.get('METRICS_PORT', 8002))
METRICS_UPDATE_INTERVAL = int(os.environ.get('METRICS_UPDATE_INTERVAL', 60))  # seconds

# Define Prometheus metrics
# NLP Analysis metrics
nlp_requests_total = Counter(
    'nlp_analysis_requests_total', 
    'Total number of NLP analysis requests',
    ['type', 'status']
)

nlp_processing_time = Summary(
    'nlp_analysis_processing_seconds', 
    'Time spent processing NLP analysis requests',
    ['type']
)

sentiment_score = Gauge(
    'nlp_sentiment_score',
    'Current sentiment score from NLP analysis',
    ['currency_pair', 'source_type']
)

entity_detection_count = Counter(
    'nlp_entity_detection_total',
    'Total number of entities detected in text',
    ['entity_type']
)

news_impact_score = Gauge(
    'news_impact_score',
    'Impact score of news on trading decisions',
    ['currency_pair', 'impact_type']
)

economic_report_accuracy = Gauge(
    'economic_report_parsing_accuracy',
    'Accuracy of economic report parsing',
    ['report_type']
)

nlp_error_count = Counter(
    'nlp_analysis_errors_total',
    'Total number of errors in NLP processing',
    ['error_type']
)
nlp_requests_total = Counter(
    'nlp_analysis_requests_total', 
    'Total number of NLP analysis requests',
    ['type', 'status']
)

nlp_processing_time = Summary(
    'nlp_analysis_processing_seconds', 
    'Time spent processing NLP analysis requests',
    ['type']
)

sentiment_score = Gauge(
    'nlp_sentiment_score',
    'Current sentiment score from NLP analysis',
    ['currency_pair', 'source_type']
)

entity_detection_count = Counter(
    'nlp_entity_detection_total',
    'Total number of entities detected in text',
    ['entity_type']
)

news_impact_score = Gauge(
    'news_impact_score',
    'Impact score of news on trading decisions',
    ['currency_pair', 'impact_type']
)

economic_report_accuracy = Gauge(
    'economic_report_parsing_accuracy',
    'Accuracy of economic report parsing',
    ['report_type']
)

nlp_error_count = Counter(
    'nlp_analysis_errors_total',
    'Total number of errors in NLP processing',
    ['error_type']
)

def fetch_and_export_metrics():
    """
    Fetch metrics from the Analysis Engine API and export them to Prometheus
    """
    try:
        # Fetch NLP analysis usage statistics
        response = requests.get(
            f"{ANALYSIS_ENGINE_BASE_URL}/api/v1/nlp/metrics",
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            metrics = response.json()
            
            # Update request counters
            for request_type, counts in metrics.get('requests', {}).items():
                nlp_requests_total.labels(type=request_type, status='success').inc(counts.get('success', 0))
                nlp_requests_total.labels(type=request_type, status='failure').inc(counts.get('failure', 0))
            
            # Update processing times
            for analysis_type, time_value in metrics.get('processing_times', {}).items():
                nlp_processing_time.labels(type=analysis_type).observe(time_value)
            
            # Update sentiment scores
            for pair_data in metrics.get('sentiment_scores', []):
                sentiment_score.labels(
                    currency_pair=pair_data.get('currency_pair', 'unknown'),
                    source_type=pair_data.get('source_type', 'unknown')
                ).set(pair_data.get('score', 0))
                
            # Update news impact scores
            for impact_data in metrics.get('news_impacts', []):
                news_impact_score.labels(
                    currency_pair=impact_data.get('currency_pair', 'unknown'),
                    impact_type=impact_data.get('impact_type', 'unknown')
                ).set(impact_data.get('score', 0))
                
            # Update economic report accuracy
            for report_data in metrics.get('economic_reports', []):
                economic_report_accuracy.labels(
                    report_type=report_data.get('report_type', 'unknown')
                ).set(report_data.get('accuracy', 0))
                
            # Update entity detection counts
            for entity_type, count in metrics.get('entity_counts', {}).items():
                entity_detection_count.labels(entity_type=entity_type).inc(count)
                
            # Log successful metrics update
            logger.info(f"Successfully updated NLP analysis metrics at {datetime.now().isoformat()}")
                
        else:
            logger.error(f"Failed to fetch NLP metrics: {response.status_code} - {response.text}")
            nlp_error_count.labels(error_type='api_fetch_failure').inc()
    except Exception as e:
        logger.error(f"Error fetching NLP metrics: {str(e)}")
        nlp_error_count.labels(error_type='exception').inc()

def main():
    """
    Start the HTTP server and periodically fetch and export metrics
    """
    # Start Prometheus HTTP server
    start_http_server(METRICS_PORT)
    logger.info(f"NLP Analysis metrics exporter started on port {METRICS_PORT}")
    
    # Initial fetch of metrics
    fetch_and_export_metrics()
    
    # Continuously fetch and export metrics at the specified interval
    while True:
        time.sleep(METRICS_UPDATE_INTERVAL)
        fetch_and_export_metrics()

if __name__ == "__main__":
    main()
NLP_SENTIMENT_SCORE = Gauge('nlp_sentiment_score', 
                           'Sentiment score from news analysis', 
                           ['currency_pair', 'source_type', 'timeframe'])

NLP_ENTITY_COUNT = Counter('nlp_entity_extraction_count', 
                          'Number of entities extracted from text', 
                          ['entity_type', 'source_type'])

NLP_ECONOMIC_IMPACT = Gauge('nlp_economic_report_impact', 
                           'Impact score of economic reports', 
                           ['currency_pair', 'report_type', 'timeframe'])

NLP_PROCESSING_TIME = Summary('nlp_processing_time_seconds', 
                              'Time spent processing NLP tasks', 
                              ['task_type', 'source_type'])

NLP_RELIABILITY_SCORE = Gauge('nlp_reliability_score',
                             'Reliability score of NLP analysis',
                             ['source_type', 'currency_pair'])

NLP_REQUEST_FAILURES = Counter('nlp_request_failures_total',
                              'Number of failed NLP analysis API requests',
                              ['endpoint', 'error_type'])

NLP_NEWS_COUNT = Counter('nlp_news_items_processed_total',
                        'Number of news items processed',
                        ['source', 'relevance_level'])

NLP_ECONOMIC_REPORT_COUNT = Counter('nlp_economic_reports_processed_total',
                                   'Number of economic reports processed',
                                   ['report_type', 'importance'])


class NlpMetricsExporter:
    """Exports NLP analysis metrics to Prometheus."""
    
    def __init__(self, base_url=ANALYSIS_ENGINE_BASE_URL, update_interval=METRICS_UPDATE_INTERVAL):
        """Initialize the NLP metrics exporter."""
        self.base_url = base_url
        self.update_interval = update_interval
        self.session = requests.Session()
        self.headers = {"Content-Type": "application/json"}
    
    def start_metrics_server(self, port=METRICS_PORT):
        """Start the Prometheus metrics server."""
        start_http_server(port)
        logger.info(f"Started Prometheus metrics server on port {port}")
    
    def fetch_nlp_metrics(self) -> Optional[Dict[str, Any]]:
        """Fetch NLP analysis metrics from the API."""
        try:
            response = self.session.get(
                f"{self.base_url}/api/v1/nlp/metrics",
                headers=self.headers,
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.error(f"Failed to fetch NLP metrics: {str(e)}")
            NLP_REQUEST_FAILURES.labels(endpoint="metrics", error_type=type(e).__name__).inc()
            return None
    
    def fetch_sentiment_analysis(self) -> Optional[Dict[str, Any]]:
        """Fetch sentiment analysis metrics from the API."""
        try:
            response = self.session.get(
                f"{self.base_url}/api/v1/nlp/sentiment/metrics",
                headers=self.headers,
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.error(f"Failed to fetch sentiment metrics: {str(e)}")
            NLP_REQUEST_FAILURES.labels(endpoint="sentiment", error_type=type(e).__name__).inc()
            return None
    
    def fetch_economic_impact(self) -> Optional[Dict[str, Any]]:
        """Fetch economic report impact metrics from the API."""
        try:
            response = self.session.get(
                f"{self.base_url}/api/v1/nlp/economic/metrics",
                headers=self.headers,
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.error(f"Failed to fetch economic impact metrics: {str(e)}")
            NLP_REQUEST_FAILURES.labels(endpoint="economic", error_type=type(e).__name__).inc()
            return None
    
    def update_nlp_metrics(self):
        """Update NLP metrics in Prometheus."""
        # Fetch general NLP metrics
        metrics = self.fetch_nlp_metrics()
        
        if metrics:
            # Update entity extraction counts
            for entity_type, data in metrics.get('entity_counts', {}).items():
                for source, count in data.items():
                    NLP_ENTITY_COUNT.labels(entity_type=entity_type, source_type=source).inc(count)
            
            # Update processing times
            for task, times in metrics.get('processing_times', {}).items():
                for source, time_data in times.items():
                    NLP_PROCESSING_TIME.labels(task_type=task, source_type=source).observe(time_data['avg_seconds'])
            
            # Update reliability scores
            for source, scores in metrics.get('reliability_scores', {}).items():
                for currency_pair, score in scores.items():
                    NLP_RELIABILITY_SCORE.labels(source_type=source, currency_pair=currency_pair).set(score)
        
        # Fetch sentiment analysis metrics
        sentiment_data = self.fetch_sentiment_analysis()
        
        if sentiment_data:
            for currency_pair, sources in sentiment_data.get('sentiment_scores', {}).items():
                for source, timeframes in sources.items():
                    for timeframe, score in timeframes.items():
                        NLP_SENTIMENT_SCORE.labels(
                            currency_pair=currency_pair, 
                            source_type=source, 
                            timeframe=timeframe
                        ).set(score)
            
            # Update news count
            for source, levels in sentiment_data.get('news_counts', {}).items():
                for level, count in levels.items():
                    NLP_NEWS_COUNT.labels(source=source, relevance_level=level).inc(count)
        
        # Fetch economic impact metrics
        economic_data = self.fetch_economic_impact()
        
        if economic_data:
            for currency_pair, reports in economic_data.get('impact_scores', {}).items():
                for report_type, timeframes in reports.items():
                    for timeframe, score in timeframes.items():
                        NLP_ECONOMIC_IMPACT.labels(
                            currency_pair=currency_pair, 
                            report_type=report_type, 
                            timeframe=timeframe
                        ).set(score)
            
            # Update economic report count
            for report_type, importance_levels in economic_data.get('report_counts', {}).items():
                for importance, count in importance_levels.items():
                    NLP_ECONOMIC_REPORT_COUNT.labels(
                        report_type=report_type, 
                        importance=importance
                    ).inc(count)
    
    def run_metrics_loop(self):
        """Run the metrics export loop."""
        while True:
            try:
                logger.info("Updating NLP metrics...")
                self.update_nlp_metrics()
                logger.info("NLP metrics updated successfully")
            except Exception as e:
                logger.error(f"Error updating NLP metrics: {str(e)}")
            
            time.sleep(self.update_interval)


def main():
    """Main entry point for the NLP metrics exporter."""
    exporter = NlpMetricsExporter()
    exporter.start_metrics_server()
    logger.info("NLP metrics exporter started")
    exporter.run_metrics_loop()


if __name__ == "__main__":
    main()
