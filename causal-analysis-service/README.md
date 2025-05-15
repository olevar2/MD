# Causal Analysis Service

This service provides causal analysis capabilities for the forex trading platform, including causal graph generation, intervention effect analysis, and counterfactual scenario generation.

## Features

- **Causal Graph Generation**: Discover causal relationships in market data
- **Intervention Effect Analysis**: Analyze the effect of interventions on the market
- **Counterfactual Scenario Generation**: Generate "what-if" scenarios for risk assessment
- **Currency Pair Relationship Analysis**: Discover causal relationships between currency pairs
- **Regime Change Driver Analysis**: Identify factors that drive market regime changes
- **Trading Signal Enhancement**: Enhance trading signals with causal insights
- **Correlation Breakdown Risk Assessment**: Assess the risk of correlation breakdown between assets

## API Endpoints

- `POST /api/v1/causal-graph`: Generate a causal graph from the provided data
- `POST /api/v1/intervention-effect`: Analyze the effect of an intervention on the system
- `POST /api/v1/counterfactual-scenario`: Generate a counterfactual scenario based on the intervention
- `POST /api/v1/currency-pair-relationships`: Discover causal relationships between currency pairs
- `POST /api/v1/regime-change-drivers`: Discover causal factors that drive market regime changes
- `POST /api/v1/enhance-trading-signals`: Enhance trading signals with causal insights
- `POST /api/v1/correlation-breakdown-risk`: Assess correlation breakdown risk between assets
- `GET /health`: Health check endpoint

## Installation

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run the service: `python main.py`

## Usage

```python
import httpx

async def generate_causal_graph():
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:8000/api/v1/causal-graph",
            json={
                "data": {
                    "price": [100, 101, 102, 103, 104],
                    "volume": [1000, 1100, 900, 1200, 1000],
                    "sentiment": [0.5, 0.6, 0.4, 0.7, 0.5]
                }
            }
        )
        return response.json()
```

## Development

1. Install development dependencies: `pip install -r requirements-dev.txt`
2. Run tests: `pytest`
3. Run linting: `flake8`

## Configuration

The service can be configured using environment variables:

- `LOG_LEVEL`: Logging level (default: INFO)
- `PORT`: Port to run the service on (default: 8000)
- `HOST`: Host to run the service on (default: 0.0.0.0)

## License

This project is licensed under the MIT License - see the LICENSE file for details.