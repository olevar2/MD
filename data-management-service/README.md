# Historical Data Management Service

A comprehensive service for managing historical data for forex trading.

## Features

- **Point-in-time accurate historical data storage**: Store and retrieve historical data with point-in-time accuracy.
- **Immutable historical records with corrections**: Maintain immutability of historical records while allowing for corrections.
- **Efficient storage and retrieval for time-series data**: Optimized for time-series data storage and retrieval.
- **Support for tick-level historical data**: Handle high-volume tick-level data efficiently.
- **Data versioning for tracking corrections and updates**: Track all changes to historical data.
- **Specialized datasets for ML training**: Create custom datasets for machine learning models.
- **Consistent data for backtesting and live trading**: Ensure backtesting uses the same historical data as live trading.

## Data Types

The service supports the following types of historical data:

- **OHLCV Data**: Open, High, Low, Close, Volume data for different timeframes.
- **Tick Data**: Bid/Ask prices and volumes at the tick level.
- **Alternative Data**: Any other type of data that can be associated with a symbol and timestamp.

## Architecture

The service is built with a layered architecture:

- **API Layer**: FastAPI endpoints for interacting with the service.
- **Service Layer**: Business logic for managing historical data.
- **Repository Layer**: Data access layer for storing and retrieving data.
- **Model Layer**: Data models and schemas.

## Getting Started

### Prerequisites

- Python 3.9 or higher
- PostgreSQL 13 or higher (with TimescaleDB extension recommended)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-org/forex-trading-platform.git
   cd forex-trading-platform/data-management-service
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up the database:
   ```bash
   # Create the database
   createdb forex_platform

   # Install TimescaleDB extension (optional but recommended)
   psql -d forex_platform -c "CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;"
   ```

4. Set up environment variables:
   ```bash
   # Copy the example environment file
   cp .env.example .env

   # Edit the .env file with your configuration
   # Replace the placeholder values with your actual configuration
   ```

5. Run the service:
   ```bash
   python -m data_management_service.main
   ```

### Docker

You can also run the service using Docker:

```bash
# Build the Docker image
docker build -t historical-data-service .

# Run the container
docker run -p 8000:8000 historical-data-service
```

## API Documentation

Once the service is running, you can access the API documentation at:

```
http://localhost:8000/docs
```

## Usage Examples

### Storing OHLCV Data

```python
import requests
import datetime

# Store OHLCV data
response = requests.post(
    "http://localhost:8000/historical/ohlcv",
    json={
        "symbol": "EURUSD",
        "timeframe": "1h",
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "open_price": 1.1234,
        "high_price": 1.1256,
        "low_price": 1.1222,
        "close_price": 1.1245,
        "volume": 1000.0,
        "source_id": "provider1"
    }
)

print(response.json())
```

### Retrieving OHLCV Data

```python
import requests
import datetime

# Get OHLCV data
start_time = datetime.datetime.utcnow() - datetime.timedelta(days=7)
end_time = datetime.datetime.utcnow()

response = requests.get(
    "http://localhost:8000/historical/ohlcv",
    params={
        "symbols": "EURUSD,GBPUSD",
        "timeframe": "1h",
        "start_timestamp": start_time.isoformat(),
        "end_timestamp": end_time.isoformat(),
        "format": "json"
    }
)

print(response.json())
```

### Creating a Correction

```python
import requests

# Create a correction
response = requests.post(
    "http://localhost:8000/historical/correction",
    json={
        "original_record_id": "123e4567-e89b-12d3-a456-426614174000",
        "correction_data": {
            "data": {
                "close": 1.1246  # Corrected close price
            }
        },
        "correction_type": "PROVIDER_CORRECTION",
        "correction_reason": "Provider sent corrected data",
        "corrected_by": "system",
        "source_type": "OHLCV"
    }
)

print(response.json())
```

### Creating an ML Dataset

```python
import requests
import datetime

# Create an ML dataset
response = requests.post(
    "http://localhost:8000/historical/ml-dataset",
    json={
        "dataset_id": "my-dataset-001",
        "name": "EURUSD Prediction Dataset",
        "symbols": ["EURUSD"],
        "timeframes": ["1h"],
        "start_timestamp": (datetime.datetime.utcnow() - datetime.timedelta(days=30)).isoformat(),
        "end_timestamp": datetime.datetime.utcnow().isoformat(),
        "features": ["open", "high", "low", "close", "volume"],
        "transformations": [
            {
                "type": "add_technical_indicator",
                "indicator": "sma",
                "params": {"period": 14}
            },
            {
                "type": "add_target",
                "target_type": "future_return",
                "periods": 1
            }
        ],
        "validation_split": 0.2,
        "test_split": 0.1
    },
    params={"format": "csv"}
)

# Save the CSV data
with open("dataset.csv", "w") as f:
    f.write(response.text)
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
