# Interface-Based Decoupling

This document explains the interface-based decoupling approach used in the Analysis Engine Service to break circular dependencies between services.

## Overview

The Analysis Engine Service has dependencies on several other services in the Forex Trading Platform:

- Trading Gateway Service
- ML Integration Service
- ML Workbench Service
- Risk Management Service
- Feature Store Service

To avoid direct dependencies and potential circular dependencies, we use the interface-based adapter pattern. This pattern involves:

1. Defining interfaces in a common library (`common-lib`)
2. Implementing adapters for these interfaces in the common library
3. Using these adapters in the Analysis Engine Service to interact with other services

## Benefits

- **Decoupling**: Services are decoupled from each other, reducing tight coupling
- **Testability**: Services can be tested in isolation using mock adapters
- **Flexibility**: Implementations can be changed without affecting consumers
- **Consistency**: Common interfaces ensure consistent interaction patterns
- **Circular Dependency Resolution**: Breaks circular dependencies between services

## Implementation

### 1. Common Interfaces

Interfaces are defined in `common-lib/common_lib/interfaces/`:

- `trading_gateway.py`: Interfaces for Trading Gateway Service
- `ml_integration.py`: Interfaces for ML Integration Service
- `ml_workbench.py`: Interfaces for ML Workbench Service
- `risk_management.py`: Interfaces for Risk Management Service
- `feature_store.py`: Interfaces for Feature Store Service

Example interface:

```python
class ITradingGateway(ABC):
    """Interface for trading gateway functionality."""
    
    @abstractmethod
    async def place_order(self, order: Order) -> ExecutionReport:
        """Place a trading order."""
        pass
    
    @abstractmethod
    async def cancel_order(self, order_id: str) -> ExecutionReport:
        """Cancel an existing order."""
        pass
    
    # Other methods...
```

### 2. Adapter Implementations

Adapters are implemented in `common-lib/common_lib/adapters/`:

- `trading_gateway_adapter.py`: Adapter for Trading Gateway Service
- `ml_integration_adapter.py`: Adapter for ML Integration Service
- `ml_workbench_adapter.py`: Adapter for ML Workbench Service
- `risk_management_adapter.py`: Adapter for Risk Management Service
- `feature_store_adapter.py`: Adapter for Feature Store Service

Example adapter:

```python
class TradingGatewayAdapter(ITradingGateway):
    """Adapter implementation for the Trading Gateway Service."""
    
    def __init__(self, config: Union[ServiceClientConfig, Dict[str, Any]]):
        """Initialize the adapter with a service client configuration."""
        if isinstance(config, dict):
            config = ServiceClientConfig(**config)
        
        self.client = AsyncHTTPServiceClient(config)
        self.logger = logger
    
    @with_circuit_breaker("trading_gateway.place_order")
    @with_retry(max_retries=3, retry_delay=1.0)
    @with_timeout(timeout_seconds=10.0)
    async def place_order(self, order: Order) -> ExecutionReport:
        """Place a trading order."""
        try:
            response = await self.client.post(
                "/api/v1/orders",
                data=order.dict() if hasattr(order, "dict") else order
            )
            return ExecutionReport(**response)
        except Exception as e:
            self.logger.error(f"Error placing order: {str(e)}")
            raise ServiceError(f"Failed to place order: {str(e)}")
    
    # Other methods...
```

### 3. Adapter Factory

The Analysis Engine Service uses an adapter factory to create adapters for the services it depends on:

```python
class CommonAdapterFactory:
    """Factory for creating service adapters using common interfaces."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the adapter factory."""
        self.config = config or {}
        self.service_config = ServiceConfig()
        self.adapters: Dict[Type, Any] = {}
        self.logger = logger
    
    def get_adapter(self, interface_type: Type[T]) -> T:
        """Get an adapter for the specified interface type."""
        # Check if we already have an adapter for this interface
        if interface_type in self.adapters:
            return cast(T, self.adapters[interface_type])
        
        # Create a new adapter based on the interface type
        adapter = self._create_adapter(interface_type)
        self.adapters[interface_type] = adapter
        return adapter
    
    def _create_adapter(self, interface_type: Type[T]) -> T:
        """Create an adapter for the specified interface type."""
        # Trading Gateway interfaces
        if issubclass(interface_type, ITradingGateway):
            return cast(T, self._create_trading_gateway_adapter())
        
        # Other interface types...
        
        raise ValueError(f"No adapter available for interface type: {interface_type.__name__}")
    
    # Methods to create specific adapters...
```

### 4. Service Dependencies

The Analysis Engine Service uses a service dependencies module to provide adapters to the API endpoints:

```python
class ServiceDependencies:
    """Service dependencies for the Analysis Engine Service."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize service dependencies."""
        self.config = config or {}
        self.adapter_factory = get_common_adapter_factory()
        self.logger = logger
    
    async def get_trading_gateway(self) -> ITradingGateway:
        """Get the Trading Gateway adapter."""
        return self.adapter_factory.get_adapter(ITradingGateway)
    
    # Other methods to get adapters...
```

### 5. API Endpoints

API endpoints use the service dependencies to interact with other services:

```python
@router.get("/market-overview/{symbol}")
async def get_market_overview(
    symbol: str = Path(..., description="Trading symbol"),
    timeframe: str = Query("1h", description="Timeframe for analysis"),
    lookback_days: int = Query(7, description="Number of days to look back"),
    trading_gateway: TradingGatewayDep = None,
    feature_provider: FeatureProviderDep = None,
    ml_model_registry: MLModelRegistryDep = None,
    risk_manager: RiskManagerDep = None
):
    """Get a comprehensive market overview for a symbol."""
    try:
        # Get market data from Trading Gateway
        market_data = await trading_gateway.get_market_data(
            symbol=symbol,
            timeframe=timeframe,
            start_time=start_time,
            end_time=end_time
        )
        
        # Get feature data from Feature Store
        features = await feature_provider.get_available_features()
        
        # Get model predictions from ML Integration
        models = await ml_model_registry.list_models()
        
        # Get risk assessment from Risk Management
        position_risk = await risk_manager.get_position_risk(symbol)
        
        # Combine all data into a comprehensive market overview
        return {
            # Result data...
        }
    except Exception as e:
        logger.error(f"Error in get_market_overview: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error generating market overview: {str(e)}"
        )
```

## Usage Examples

See the following files for examples of how to use the interface-based decoupling approach:

- `analysis-engine-service/examples/common_adapter_usage.py`: Examples of using the common adapters
- `analysis-engine-service/api/v1/integrated_analysis.py`: API endpoints that use the common adapters

## Testing

The interface-based decoupling approach makes it easy to test the Analysis Engine Service in isolation by using mock adapters:

```python
class MockTradingGatewayAdapter(ITradingGateway):
    """Mock adapter for testing."""
    
    async def place_order(self, order: Order) -> ExecutionReport:
        """Mock implementation of place_order."""
        return ExecutionReport(
            order_id="mock-order-id",
            status="FILLED",
            filled_quantity=order.quantity,
            average_price=100.0
        )
    
    # Other mock methods...
```

## Conclusion

The interface-based decoupling approach provides a clean and maintainable way to break circular dependencies between services in the Forex Trading Platform. By defining interfaces in a common library and implementing adapters for these interfaces, services can interact with each other without direct dependencies.
