# filepath: d:\MD\forex_trading_platform\testing\stress_testing\locustfile_feedback.py
"""
Locust file for stress testing the feedback collection system.
"""

import time
import random
from locust import HttpUser, task, between

class FeedbackUser(HttpUser):
    """Simulates a user sending feedback events."""
    wait_time = between(0.5, 2.5) # Wait time between tasks

    # TODO: Update host URL to the feedback ingestion endpoint
    # host = "http://your-feedback-api-endpoint.com"

    @task
    def send_feedback_event(self):
        """Simulates sending a single feedback event."""
        strategy_id = f"strat_{random.randint(1, 100)}"
        trade_id = f"trade_{random.randint(10000, 99999)}"
        pnl = random.uniform(-100, 150)
        slippage = random.uniform(0.0001, 0.005)

        feedback_payload = {
            "event_id": f"evt_{random.randint(100000, 999999)}",
            "trade_id": trade_id,
            "strategy_id": strategy_id,
            "outcome": {"pnl": pnl, "slippage": slippage},
            "parameters_used": {"paramA": random.randint(1, 10), "paramB": random.random()},
            "timestamp": time.time()
        }

        # TODO: Adjust the endpoint path and method (POST/PUT etc.)
        endpoint = "/api/v1/feedback"
        try:
            with self.client.post(endpoint, json=feedback_payload, catch_response=True) as response:
                if response.status_code == 200 or response.status_code == 201 or response.status_code == 202:
                    response.success()
                else:
                    response.failure(f"Failed request to {endpoint}. Status: {response.status_code}")
        except Exception as e:
            # Log exception during the request
            # You might need to configure Locust logging for this
            print(f"Request to {endpoint} failed with exception: {e}")
            # Optionally mark as failure
            # self.environment.events.request_failure.fire(request_type="POST", name=endpoint, response_time=0, exception=e)
            pass # Avoid stopping the test run completely on single errors

    # TODO: Add other tasks if there are different types of feedback or related actions
    # @task(2) # Example: Make this task twice as likely
    # def send_complex_feedback(self):
    """
    Send complex feedback.
    
    """

    #     pass

# To run this test:
# 1. Install Locust: pip install locust
# 2. Run Locust: locust -f locustfile_feedback.py --host=http://your-target-host.com
# 3. Open the Locust web UI (usually http://localhost:8089) and start the test.

print("Locust file for feedback stress testing created. Remember to update the host URL and endpoint.")
