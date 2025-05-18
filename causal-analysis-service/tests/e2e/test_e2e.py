"""
End-to-end tests for causal analysis service.
"""
import pytest
import os
import httpx

# Get service URL from environment variables
SERVICE_URL = os.getenv("SERVICE_URL", "http://localhost:8000")

@pytest.mark.asyncio
async def test_end_to_end_workflow():
    """Test the complete workflow of the causal analysis service."""
    async with httpx.AsyncClient() as client:
        # Step 1: Check if the service is healthy
        response = await client.get(f"{SERVICE_URL}/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}
        
        # Step 2: Submit a causal analysis request
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
        
        response = await client.post(f"{SERVICE_URL}/api/v1/causal-analysis", json=test_data)
        assert response.status_code == 200
        result = response.json()
        
        # Step 3: Verify the result structure
        assert "edges" in result
        assert "nodes" in result
        
        # Step 4: Use the result to make another request (e.g., get detailed analysis)
        # This is a placeholder for a more complex workflow
        # In a real test, you would make additional requests based on the previous results
        
        # Step 5: Verify that the service can handle errors gracefully
        invalid_data = {
            "variables": ["price"],
            "data": [],
            "method": "invalid_method"
        }
        
        response = await client.post(f"{SERVICE_URL}/api/v1/causal-analysis", json=invalid_data)
        # The service should return a 4xx status code for invalid input
        assert 400 <= response.status_code < 500
