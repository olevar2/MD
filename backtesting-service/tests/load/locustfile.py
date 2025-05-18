"""
Load tests for backtesting service using Locust.
"""
from locust import HttpUser, task, between

class BacktestingUser(HttpUser):
    """
    Simulated user for load testing the backtesting service.
    """
    # Wait between 1 and 5 seconds between tasks
    wait_time = between(1, 5)
    
    @task(1)
    def health_check(self):
        """Check the health endpoint."""
        self.client.get("/health")
    
    @task(10)
    def simple_backtest(self):
        """Perform a simple backtest."""
        # Sample data for testing
        test_data = {
            "strategy_id": "test_strategy",
            "start_date": "2023-01-01",
            "end_date": "2023-01-31",
            "symbol": "EURUSD",
            "timeframe": "1h",
            "parameters": {
                "risk_reward_ratio": 2.0,
                "stop_loss_pips": 20,
                "take_profit_pips": 40
            }
        }
        
        # Make a request to the backtesting endpoint
        self.client.post("/api/v1/backtest", json=test_data)
