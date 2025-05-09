"""
Locust load testing file for forex trading platform services.

This file defines the load testing behavior for different services.
"""

import json
import os
import random
import time
from typing import Dict, List, Any, Optional

import pandas as pd
from locust import HttpUser, task, between, events

# Get service and endpoints from environment variables
SERVICE = os.environ.get("SERVICE", "analysis-engine-service")
ENDPOINTS = json.loads(os.environ.get("ENDPOINTS", "[]"))
DATA_FILE = os.environ.get("DATA_FILE", "normal_trading_data.csv")

# Load test data
def load_test_data() -> pd.DataFrame:
    """Load test data from CSV file."""
    data_path = os.path.join(os.path.dirname(__file__), "data", DATA_FILE)
    if not os.path.exists(data_path):
        print(f"Warning: Test data file not found: {data_path}")
        return pd.DataFrame()
    
    return pd.read_csv(data_path)

# Try to load test data
try:
    TEST_DATA = load_test_data()
except Exception as e:
    print(f"Error loading test data: {e}")
    TEST_DATA = pd.DataFrame()

# Define request payloads for each service and endpoint
def get_payload(service: str, endpoint: str) -> Dict[str, Any]:
    """Get request payload for a service and endpoint."""
    if not TEST_DATA.empty:
        # Use test data if available
        row = TEST_DATA.sample(1).iloc[0]
        
        if service == "analysis-engine-service":
            if endpoint == "/api/v1/analysis/market":
                return {
                    "instrument": row.get("instrument", "EUR_USD"),
                    "timeframe": row.get("timeframe", "1h"),
                    "start_time": row.get("start_time", "2023-01-01T00:00:00Z"),
                    "end_time": row.get("end_time", "2023-01-02T00:00:00Z")
                }
            elif endpoint == "/api/v1/analysis/patterns":
                return {
                    "instrument": row.get("instrument", "EUR_USD"),
                    "timeframe": row.get("timeframe", "1h"),
                    "pattern_types": ["double_top", "double_bottom", "head_and_shoulders"],
                    "start_time": row.get("start_time", "2023-01-01T00:00:00Z"),
                    "end_time": row.get("end_time", "2023-01-02T00:00:00Z")
                }
            elif endpoint == "/api/v1/analysis/signals":
                return {
                    "instrument": row.get("instrument", "EUR_USD"),
                    "timeframe": row.get("timeframe", "1h"),
                    "signal_types": ["trend", "reversal", "breakout"],
                    "start_time": row.get("start_time", "2023-01-01T00:00:00Z"),
                    "end_time": row.get("end_time", "2023-01-02T00:00:00Z")
                }
            elif endpoint == "/api/v1/analysis/regime":
                return {
                    "instrument": row.get("instrument", "EUR_USD"),
                    "timeframe": row.get("timeframe", "1h"),
                    "start_time": row.get("start_time", "2023-01-01T00:00:00Z"),
                    "end_time": row.get("end_time", "2023-01-02T00:00:00Z")
                }
        
        elif service == "trading-gateway-service":
            if endpoint == "/api/v1/orders":
                return {
                    "instrument": row.get("instrument", "EUR_USD"),
                    "order_type": "MARKET",
                    "side": row.get("side", "BUY"),
                    "quantity": row.get("quantity", 10000),
                    "price": row.get("price", 1.1000),
                    "time_in_force": "GTC"
                }
            elif endpoint == "/api/v1/positions":
                return {}  # GET request, no payload
            elif endpoint == "/api/v1/executions":
                return {}  # GET request, no payload
            elif endpoint == "/api/v1/market-data":
                return {
                    "instrument": row.get("instrument", "EUR_USD"),
                    "timeframe": row.get("timeframe", "1h"),
                    "count": 100
                }
        
        elif service == "feature-store-service":
            if endpoint == "/api/v1/features":
                return {
                    "feature_names": ["price_momentum", "volume_trend", "volatility"],
                    "instrument": row.get("instrument", "EUR_USD"),
                    "timeframe": row.get("timeframe", "1h"),
                    "start_time": row.get("start_time", "2023-01-01T00:00:00Z"),
                    "end_time": row.get("end_time", "2023-01-02T00:00:00Z")
                }
            elif endpoint == "/api/v1/feature-sets":
                return {}  # GET request, no payload
            elif endpoint == "/api/v1/data-sources":
                return {}  # GET request, no payload
            elif endpoint == "/api/v1/calculations":
                return {
                    "feature_name": "custom_feature",
                    "formula": "close - open",
                    "instrument": row.get("instrument", "EUR_USD"),
                    "timeframe": row.get("timeframe", "1h"),
                    "start_time": row.get("start_time", "2023-01-01T00:00:00Z"),
                    "end_time": row.get("end_time", "2023-01-02T00:00:00Z")
                }
        
        elif service == "ml-integration-service":
            if endpoint == "/api/v1/models":
                return {}  # GET request, no payload
            elif endpoint == "/api/v1/predictions":
                return {
                    "model_id": "price_prediction_v1",
                    "features": {
                        "price_momentum": row.get("price_momentum", 0.5),
                        "volume_trend": row.get("volume_trend", 0.2),
                        "volatility": row.get("volatility", 0.3)
                    },
                    "instrument": row.get("instrument", "EUR_USD"),
                    "timeframe": row.get("timeframe", "1h")
                }
            elif endpoint == "/api/v1/training":
                return {
                    "model_id": "price_prediction_v1",
                    "feature_set_id": "price_prediction_features",
                    "training_params": {
                        "epochs": 100,
                        "batch_size": 32,
                        "learning_rate": 0.001
                    }
                }
            elif endpoint == "/api/v1/evaluation":
                return {
                    "model_id": "price_prediction_v1",
                    "feature_set_id": "price_prediction_features",
                    "start_time": row.get("start_time", "2023-01-01T00:00:00Z"),
                    "end_time": row.get("end_time", "2023-01-02T00:00:00Z")
                }
        
        elif service == "strategy-execution-engine":
            if endpoint == "/api/v1/strategies":
                return {}  # GET request, no payload
            elif endpoint == "/api/v1/backtests":
                return {
                    "strategy_id": "trend_following_v1",
                    "instrument": row.get("instrument", "EUR_USD"),
                    "timeframe": row.get("timeframe", "1h"),
                    "start_time": row.get("start_time", "2023-01-01T00:00:00Z"),
                    "end_time": row.get("end_time", "2023-01-02T00:00:00Z"),
                    "initial_capital": 10000
                }
            elif endpoint == "/api/v1/executions":
                return {
                    "strategy_id": "trend_following_v1",
                    "instrument": row.get("instrument", "EUR_USD"),
                    "side": row.get("side", "BUY"),
                    "quantity": row.get("quantity", 10000),
                    "price": row.get("price", 1.1000)
                }
            elif endpoint == "/api/v1/performance":
                return {
                    "strategy_id": "trend_following_v1",
                    "start_time": row.get("start_time", "2023-01-01T00:00:00Z"),
                    "end_time": row.get("end_time", "2023-01-02T00:00:00Z")
                }
        
        elif service == "data-pipeline-service":
            if endpoint == "/api/v1/pipelines":
                return {}  # GET request, no payload
            elif endpoint == "/api/v1/data-sources":
                return {}  # GET request, no payload
            elif endpoint == "/api/v1/transformations":
                return {}  # GET request, no payload
            elif endpoint == "/api/v1/jobs":
                return {
                    "pipeline_id": "market_data_ingestion",
                    "parameters": {
                        "instrument": row.get("instrument", "EUR_USD"),
                        "timeframe": row.get("timeframe", "1h"),
                        "start_time": row.get("start_time", "2023-01-01T00:00:00Z"),
                        "end_time": row.get("end_time", "2023-01-02T00:00:00Z")
                    }
                }
    
    # Default payloads if test data is not available
    if service == "analysis-engine-service":
        if endpoint == "/api/v1/analysis/market":
            return {
                "instrument": "EUR_USD",
                "timeframe": "1h",
                "start_time": "2023-01-01T00:00:00Z",
                "end_time": "2023-01-02T00:00:00Z"
            }
        elif endpoint == "/api/v1/analysis/patterns":
            return {
                "instrument": "EUR_USD",
                "timeframe": "1h",
                "pattern_types": ["double_top", "double_bottom", "head_and_shoulders"],
                "start_time": "2023-01-01T00:00:00Z",
                "end_time": "2023-01-02T00:00:00Z"
            }
        elif endpoint == "/api/v1/analysis/signals":
            return {
                "instrument": "EUR_USD",
                "timeframe": "1h",
                "signal_types": ["trend", "reversal", "breakout"],
                "start_time": "2023-01-01T00:00:00Z",
                "end_time": "2023-01-02T00:00:00Z"
            }
        elif endpoint == "/api/v1/analysis/regime":
            return {
                "instrument": "EUR_USD",
                "timeframe": "1h",
                "start_time": "2023-01-01T00:00:00Z",
                "end_time": "2023-01-02T00:00:00Z"
            }
    
    # Add default payloads for other services...
    
    return {}

# Define HTTP methods for each endpoint
def get_http_method(endpoint: str) -> str:
    """Get HTTP method for an endpoint."""
    if endpoint.endswith("/orders") or endpoint.endswith("/calculations") or endpoint.endswith("/predictions") or endpoint.endswith("/training") or endpoint.endswith("/backtests") or endpoint.endswith("/executions") or endpoint.endswith("/jobs"):
        return "POST"
    else:
        return "GET"

class ForexPlatformUser(HttpUser):
    """Locust user for forex trading platform services."""
    
    wait_time = between(1, 3)
    
    def on_start(self):
        """Initialize user."""
        self.service = SERVICE
        self.endpoints = ENDPOINTS
    
    @task
    def request_endpoint(self):
        """Make a request to a random endpoint."""
        if not self.endpoints:
            return
        
        endpoint = random.choice(self.endpoints)
        method = get_http_method(endpoint)
        payload = get_payload(self.service, endpoint)
        
        if method == "GET":
            self.client.get(endpoint)
        else:
            self.client.post(endpoint, json=payload)

@events.init.add_listener
def on_locust_init(environment, **kwargs):
    """Initialize locust environment."""
    print(f"Initializing load test for {SERVICE}")
    print(f"Endpoints: {ENDPOINTS}")
    print(f"Data file: {DATA_FILE}")

@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    """Handle test start event."""
    print(f"Starting load test for {SERVICE}")

@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    """Handle test stop event."""
    print(f"Load test for {SERVICE} completed")
