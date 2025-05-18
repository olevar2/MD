"""
Load tests for causal analysis service using Locust.
"""
from locust import HttpUser, task, between

class CausalAnalysisUser(HttpUser):
    """
    Simulated user for load testing the causal analysis service.
    """
    # Wait between 1 and 5 seconds between tasks
    wait_time = between(1, 5)
    
    @task(1)
    def health_check(self):
        """Check the health endpoint."""
        self.client.get("/health")
    
    @task(10)
    def causal_analysis(self):
        """Perform causal analysis."""
        # Sample data for testing
        test_data = {
            "variables": ["price", "volume", "volatility"],
            "data": [
                {"price": 100, "volume": 1000, "volatility": 0.1},
                {"price": 101, "volume": 1100, "volatility": 0.2},
                {"price": 102, "volume": 1200, "volatility": 0.15},
                {"price": 103, "volume": 1300, "volatility": 0.25},
                {"price": 104, "volume": 1400, "volatility": 0.3}
            ],
            "method": "pc"
        }
        
        # Make a request to the causal analysis endpoint
        self.client.post("/api/v1/causal-analysis", json=test_data)
    
    @task(5)
    def complex_analysis(self):
        """Perform a more complex analysis."""
        # Sample data for testing with more variables and data points
        test_data = {
            "variables": ["price", "volume", "volatility", "open", "close", "high", "low"],
            "data": [
                {"price": 100, "volume": 1000, "volatility": 0.1, "open": 99, "close": 101, "high": 102, "low": 98},
                {"price": 101, "volume": 1100, "volatility": 0.2, "open": 100, "close": 102, "high": 103, "low": 99},
                {"price": 102, "volume": 1200, "volatility": 0.15, "open": 101, "close": 103, "high": 104, "low": 100},
                {"price": 103, "volume": 1300, "volatility": 0.25, "open": 102, "close": 104, "high": 105, "low": 101},
                {"price": 104, "volume": 1400, "volatility": 0.3, "open": 103, "close": 105, "high": 106, "low": 102},
                {"price": 105, "volume": 1500, "volatility": 0.2, "open": 104, "close": 106, "high": 107, "low": 103},
                {"price": 106, "volume": 1600, "volatility": 0.1, "open": 105, "close": 107, "high": 108, "low": 104},
                {"price": 107, "volume": 1700, "volatility": 0.3, "open": 106, "close": 108, "high": 109, "low": 105},
                {"price": 108, "volume": 1800, "volatility": 0.25, "open": 107, "close": 109, "high": 110, "low": 106},
                {"price": 109, "volume": 1900, "volatility": 0.15, "open": 108, "close": 110, "high": 111, "low": 107}
            ],
            "method": "pc"
        }
        
        # Make a request to the causal analysis endpoint
        self.client.post("/api/v1/causal-analysis", json=test_data)
