# Interface-Based Adapter Pattern Implementation Guide

## Overview

This guide provides a detailed approach to implementing the interface-based adapter pattern in the Forex Trading Platform. The pattern helps eliminate direct dependencies between services, making the system more maintainable, testable, and flexible.

## Key Concepts

### Interface

An interface defines a contract that implementations must adhere to. In Python, we use abstract base classes (ABC) to define interfaces.

```python
from abc import ABC, abstractmethod

class IDataService(ABC):
    @abstractmethod
    def get_market_data(self, symbol: str, timeframe: str, start_date: str, end_date: str):
        """Get market data for a symbol."""
        pass
```

### Adapter

An adapter implements an interface and wraps a concrete implementation, often translating between different interfaces.

```python
class DataServiceAdapter(IDataService):
    def __init__(self, data_service_client):
        self.client = data_service_client
    
    def get_market_data(self, symbol: str, timeframe: str, start_date: str, end_date: str):
        # Adapt the client's interface to match the IDataService interface
        return self.client.fetch_market_data(
            instrument=symbol,
            period=timeframe,
            from_date=start_date,
            to_date=end_date
        )
```

### Factory

A factory creates and returns adapter instances, often handling configuration and dependency injection.

```python
class DataServiceAdapterFactory:
    @staticmethod
    def create(config=None):
        client = DataServiceClient(config)
        return DataServiceAdapter(client)
```

## Implementation Steps

### 1. Define Interfaces in common-lib

All interfaces should be defined in the common-lib package to ensure they're accessible to all services.

#### Directory Structure

```
common-lib/
  common_lib/
    interfaces/
      __init__.py
      data_service.py
      analysis_service.py
      trading_service.py
      ...
```

#### Interface Definition

Each interface file should define one or more related interfaces:

```python
# common_lib/interfaces/data_service.py
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional

class IDataService(ABC):
    @abstractmethod
    def get_market_data(self, symbol: str, timeframe: str, start_date: str, end_date: str) -> Dict[str, Any]:
        """Get market data for a symbol."""
        pass
    
    @abstractmethod
    def get_symbols(self) -> List[str]:
        """Get available symbols."""
        pass
```

### 2. Implement Adapters in common-lib

Adapters should also be implemented in the common-lib package to ensure they're accessible to all services.

#### Directory Structure

```
common-lib/
  common_lib/
    adapters/
      __init__.py
      data_service_adapter.py
      analysis_service_adapter.py
      trading_service_adapter.py
      ...
```

#### Adapter Implementation

Each adapter file should implement one or more adapters for a specific service:

```python
# common_lib/adapters/data_service_adapter.py
from typing import Dict, List, Any, Optional
from common_lib.interfaces.data_service import IDataService
from data_pipeline_service.client import DataPipelineClient

class DataPipelineServiceAdapter(IDataService):
    def __init__(self, client: Optional[DataPipelineClient] = None, config: Optional[Dict[str, Any]] = None):
        self.client = client or DataPipelineClient(config)
    
    def get_market_data(self, symbol: str, timeframe: str, start_date: str, end_date: str) -> Dict[str, Any]:
        return self.client.fetch_market_data(
            instrument=symbol,
            period=timeframe,
            from_date=start_date,
            to_date=end_date
        )
    
    def get_symbols(self) -> List[str]:
        return self.client.list_instruments()
```

### 3. Implement Adapter Factories in common-lib

Factories should be implemented in the common-lib package to provide a consistent way to create adapters.

#### Directory Structure

```
common-lib/
  common_lib/
    factories/
      __init__.py
      data_service_factory.py
      analysis_service_factory.py
      trading_service_factory.py
      ...
```

#### Factory Implementation

Each factory file should implement one or more factories for a specific service:

```python
# common_lib/factories/data_service_factory.py
from typing import Dict, Any, Optional
from common_lib.interfaces.data_service import IDataService
from common_lib.adapters.data_service_adapter import DataPipelineServiceAdapter
from data_pipeline_service.client import DataPipelineClient

class DataServiceFactory:
    @staticmethod
    def create_data_pipeline_adapter(config: Optional[Dict[str, Any]] = None) -> IDataService:
        client = DataPipelineClient(config)
        return DataPipelineServiceAdapter(client)
```

### 4. Update Service Clients to Use Adapters

Service clients should be updated to use adapters instead of direct dependencies.

#### Before

```python
from data_pipeline_service.client import DataPipelineClient

class MarketAnalyzer:
    def __init__(self, config=None):
        self.data_client = DataPipelineClient(config)
    
    def analyze_market(self, symbol, timeframe):
        data = self.data_client.fetch_market_data(
            instrument=symbol,
            period=timeframe,
            from_date="2023-01-01",
            to_date="2023-12-31"
        )
        # Analyze data
        return analysis_result
```

#### After

```python
from common_lib.interfaces.data_service import IDataService
from common_lib.factories.data_service_factory import DataServiceFactory

class MarketAnalyzer:
    def __init__(self, data_service: IDataService = None, config=None):
        self.data_service = data_service or DataServiceFactory.create_data_pipeline_adapter(config)
    
    def analyze_market(self, symbol, timeframe):
        data = self.data_service.get_market_data(
            symbol=symbol,
            timeframe=timeframe,
            start_date="2023-01-01",
            end_date="2023-12-31"
        )
        # Analyze data
        return analysis_result
```

### 5. Implement Dependency Injection

Use dependency injection to provide adapters to services.

#### Dependency Injection Container

```python
# common_lib/di/container.py
from typing import Dict, Any
from common_lib.factories.data_service_factory import DataServiceFactory
from common_lib.factories.analysis_service_factory import AnalysisServiceFactory
from common_lib.factories.trading_service_factory import TradingServiceFactory

class DIContainer:
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.services = {}
    
    def get_data_service(self):
        if 'data_service' not in self.services:
            self.services['data_service'] = DataServiceFactory.create_data_pipeline_adapter(self.config.get('data_service'))
        return self.services['data_service']
    
    def get_analysis_service(self):
        if 'analysis_service' not in self.services:
            self.services['analysis_service'] = AnalysisServiceFactory.create_analysis_engine_adapter(self.config.get('analysis_service'))
        return self.services['analysis_service']
    
    def get_trading_service(self):
        if 'trading_service' not in self.services:
            self.services['trading_service'] = TradingServiceFactory.create_trading_gateway_adapter(self.config.get('trading_service'))
        return self.services['trading_service']
```

#### Using the Container

```python
from common_lib.di.container import DIContainer

# Create the container
container = DIContainer(config={
    'data_service': {'base_url': 'http://data-service:8000'},
    'analysis_service': {'base_url': 'http://analysis-service:8000'},
    'trading_service': {'base_url': 'http://trading-service:8000'}
})

# Get services
data_service = container.get_data_service()
analysis_service = container.get_analysis_service()
trading_service = container.get_trading_service()

# Use services
market_data = data_service.get_market_data('EURUSD', '1h', '2023-01-01', '2023-12-31')
analysis_result = analysis_service.analyze_market_data(market_data)
trade_result = trading_service.execute_trade('EURUSD', 'BUY', 1.0, 1.1, 0.9)
```

## Testing

### Unit Testing Adapters

```python
import unittest
from unittest.mock import Mock
from common_lib.adapters.data_service_adapter import DataPipelineServiceAdapter

class TestDataPipelineServiceAdapter(unittest.TestCase):
    def setUp(self):
        self.mock_client = Mock()
        self.adapter = DataPipelineServiceAdapter(self.mock_client)
    
    def test_get_market_data(self):
        # Arrange
        self.mock_client.fetch_market_data.return_value = {'data': 'test'}
        
        # Act
        result = self.adapter.get_market_data('EURUSD', '1h', '2023-01-01', '2023-12-31')
        
        # Assert
        self.assertEqual(result, {'data': 'test'})
        self.mock_client.fetch_market_data.assert_called_once_with(
            instrument='EURUSD',
            period='1h',
            from_date='2023-01-01',
            to_date='2023-12-31'
        )
```

### Integration Testing

```python
import unittest
from common_lib.factories.data_service_factory import DataServiceFactory

class TestDataServiceIntegration(unittest.TestCase):
    def setUp(self):
        self.data_service = DataServiceFactory.create_data_pipeline_adapter({
            'base_url': 'http://localhost:8000'
        })
    
    def test_get_market_data_integration(self):
        # Act
        result = self.data_service.get_market_data('EURUSD', '1h', '2023-01-01', '2023-01-02')
        
        # Assert
        self.assertIsNotNone(result)
        self.assertIn('open', result)
        self.assertIn('high', result)
        self.assertIn('low', result)
        self.assertIn('close', result)
```

## Best Practices

1. **Define Clear Interfaces**: Interfaces should be clear, concise, and focused on a specific responsibility.
2. **Use Dependency Injection**: Inject adapters into services rather than creating them directly.
3. **Factory Pattern**: Use factories to create adapters, handling configuration and dependencies.
4. **Versioning**: Version interfaces to handle changes over time.
5. **Error Handling**: Implement consistent error handling in adapters.
6. **Testing**: Test adapters thoroughly, both in isolation and integration.
7. **Documentation**: Document interfaces, adapters, and factories clearly.

## Example Implementation

### Interface

```python
# common_lib/interfaces/market_data.py
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from datetime import datetime

class IMarketDataService(ABC):
    @abstractmethod
    def get_ohlcv(self, symbol: str, timeframe: str, start_date: datetime, end_date: Optional[datetime] = None) -> Dict[str, Any]:
        """Get OHLCV data for a symbol."""
        pass
    
    @abstractmethod
    def get_available_symbols(self) -> List[str]:
        """Get list of available symbols."""
        pass
    
    @abstractmethod
    def get_latest_price(self, symbol: str) -> float:
        """Get latest price for a symbol."""
        pass
```

### Adapter

```python
# common_lib/adapters/market_data_adapter.py
from typing import Dict, List, Any, Optional
from datetime import datetime
from common_lib.interfaces.market_data import IMarketDataService
from data_pipeline_service.client import DataPipelineClient

class DataPipelineMarketDataAdapter(IMarketDataService):
    def __init__(self, client: Optional[DataPipelineClient] = None, config: Optional[Dict[str, Any]] = None):
        self.client = client or DataPipelineClient(config)
    
    def get_ohlcv(self, symbol: str, timeframe: str, start_date: datetime, end_date: Optional[datetime] = None) -> Dict[str, Any]:
        return self.client.fetch_market_data(
            instrument=symbol,
            period=timeframe,
            from_date=start_date.strftime('%Y-%m-%d'),
            to_date=end_date.strftime('%Y-%m-%d') if end_date else None
        )
    
    def get_available_symbols(self) -> List[str]:
        return self.client.list_instruments()
    
    def get_latest_price(self, symbol: str) -> float:
        return self.client.get_current_price(instrument=symbol)
```

### Factory

```python
# common_lib/factories/market_data_factory.py
from typing import Dict, Any, Optional
from common_lib.interfaces.market_data import IMarketDataService
from common_lib.adapters.market_data_adapter import DataPipelineMarketDataAdapter
from data_pipeline_service.client import DataPipelineClient

class MarketDataFactory:
    @staticmethod
    def create_data_pipeline_adapter(config: Optional[Dict[str, Any]] = None) -> IMarketDataService:
        client = DataPipelineClient(config)
        return DataPipelineMarketDataAdapter(client)
```

### Usage

```python
from common_lib.factories.market_data_factory import MarketDataFactory
from datetime import datetime, timedelta

# Create the adapter
market_data_service = MarketDataFactory.create_data_pipeline_adapter({
    'base_url': 'http://data-pipeline-service:8000'
})

# Use the adapter
start_date = datetime.now() - timedelta(days=30)
end_date = datetime.now()
ohlcv_data = market_data_service.get_ohlcv('EURUSD', '1h', start_date, end_date)
available_symbols = market_data_service.get_available_symbols()
latest_price = market_data_service.get_latest_price('EURUSD')

print(f"Latest EURUSD price: {latest_price}")
print(f"Available symbols: {available_symbols}")
print(f"OHLCV data points: {len(ohlcv_data)}")
```
