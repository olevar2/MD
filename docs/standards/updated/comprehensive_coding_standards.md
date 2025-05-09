# Comprehensive Forex Trading Platform Coding Standards

## Overview

This document defines the comprehensive coding standards for the Forex Trading Platform. These standards are designed to ensure consistency, maintainability, and alignment with domain concepts across all services.

## Core Principles

1. **Domain-Driven Design**: Code should reflect the ubiquitous language of forex trading
2. **Consistency**: Follow consistent patterns across all services
3. **Readability**: Code should be self-documenting and easy to understand
4. **Maintainability**: Design for future changes and extensions
5. **Testability**: Code should be designed for comprehensive testing

## Domain Language Alignment

### Domain Concepts and Terminology

All code should use consistent terminology aligned with forex trading domain concepts:

| Domain Term | Definition | Code Usage |
|-------------|------------|------------|
| Pip | Smallest price movement in an exchange rate | Use `pip_value` not `tick_size` or `minimum_movement` |
| Lot | Standard unit of trading | Use `lot_size` not `position_size` or `trade_size` |
| Spread | Difference between bid and ask price | Use `spread` not `bid_ask_difference` |
| Timeframe | Time interval for price data | Use `timeframe` not `interval` or `period` |
| Order | Instruction to execute a trade | Use `order` not `trade_request` |
| Position | Open trade in the market | Use `position` not `open_trade` |
| Indicator | Technical analysis tool | Use `indicator` not `signal_generator` |
| Pattern | Chart pattern formation | Use `pattern` not `formation` |
| Signal | Trading recommendation | Use `signal` not `recommendation` or `alert` |
| Execution | Process of filling an order | Use `execution` not `fill` or `processing` |

### Naming Conventions with Domain Context

#### Python

```python
# GOOD - Domain-aligned naming
def calculate_pip_value(currency_pair: str, lot_size: float) -> float:
    """Calculate the monetary value of a pip for a given currency pair and lot size."""
    # Implementation

# BAD - Generic naming not aligned with domain
def calculate_value(pair: str, size: float) -> float:
    """Calculate value."""
    # Implementation
```

#### TypeScript/JavaScript

```typescript
// GOOD - Domain-aligned naming
function calculatePipValue(currencyPair: string, lotSize: number): number {
  // Implementation
}

// BAD - Generic naming not aligned with domain
function calculateValue(pair: string, size: number): number {
  // Implementation
}
```

## Language-Specific Standards

### Python Standards

#### Naming Conventions

* **Packages**: Lowercase, short names, no underscores (e.g., `indicators`, `patterns`)
* **Modules**: Lowercase with underscores (e.g., `fibonacci_tools.py`, `market_data_service.py`)
* **Classes**: CapWords/PascalCase (e.g., `FibonacciRetracement`, `MarketDataService`)
* **Functions/Methods**: Lowercase with underscores (e.g., `calculate_pip_value`, `get_market_data`)
* **Variables**: Lowercase with underscores (e.g., `currency_pair`, `lot_size`)
* **Constants**: All uppercase with underscores (e.g., `MAX_LEVERAGE`, `DEFAULT_TIMEFRAME`)
* **Type Variables**: CapWords with `_co` or `_contra` suffix for covariant or contravariant behavior

#### Code Style

* **Line Length**: 88 characters (Black default)
* **Indentation**: 4 spaces (no tabs)
* **String Quotes**: Prefer double quotes for docstrings, single quotes for regular strings
* **Imports**: Group imports in the following order:
  1. Standard library imports
  2. Related third-party imports
  3. Local application/library specific imports
* **Docstrings**: Use Google-style docstrings

```python
def calculate_pip_value(currency_pair: str, lot_size: float) -> float:
    """Calculate the monetary value of a pip for a given currency pair and lot size.

    Args:
        currency_pair: The currency pair (e.g., 'EUR/USD')
        lot_size: The size of the position in lots

    Returns:
        The monetary value of a pip in the account currency

    Raises:
        ValueError: If the currency pair is not supported
    """
    # Implementation
```

#### Type Annotations

* Use type annotations for all function parameters and return values
* Use `Optional[T]` for parameters that can be None
* Use `Union[T1, T2]` for parameters that can be multiple types
* Use `Literal` for parameters with a fixed set of string values
* Use `TypedDict` for dictionaries with known keys and value types
* Use `Protocol` for structural typing

```python
from typing import Optional, Union, Literal, TypedDict, Protocol

OrderType = Literal['MARKET', 'LIMIT', 'STOP', 'STOP_LIMIT']

class OrderParams(TypedDict):
    symbol: str
    order_type: OrderType
    side: Literal['BUY', 'SELL']
    quantity: float
    price: Optional[float]

def place_order(params: OrderParams) -> str:
    """Place a new order."""
    # Implementation
```

### JavaScript/TypeScript Standards

#### Naming Conventions

* **Files**: Lowercase with hyphens (e.g., `fibonacci-tools.ts`, `market-data-service.ts`)
* **Classes**: PascalCase (e.g., `FibonacciRetracement`, `MarketDataService`)
* **Functions/Methods**: camelCase (e.g., `calculatePipValue`, `getMarketData`)
* **Variables**: camelCase (e.g., `currencyPair`, `lotSize`)
* **Constants**: UPPER_CASE or PascalCase for exported constants (e.g., `MAX_LEVERAGE`, `DefaultTimeframe`)
* **Interfaces/Types**: PascalCase with no prefix (e.g., `OrderParams`, `TradeExecutionResult`)
* **Enums**: PascalCase (e.g., `OrderType`, `TimeFrame`)

#### Code Style

* **Line Length**: 100 characters
* **Indentation**: 2 spaces (no tabs)
* **String Quotes**: Single quotes for strings, backticks for templates
* **Semicolons**: Required
* **Imports**: Group imports in the following order:
  1. External libraries
  2. Internal modules
  3. Relative imports
* **JSDoc Comments**: Use JSDoc for documentation

```typescript
/**
 * Calculate the monetary value of a pip for a given currency pair and lot size.
 *
 * @param currencyPair - The currency pair (e.g., 'EUR/USD')
 * @param lotSize - The size of the position in lots
 * @returns The monetary value of a pip in the account currency
 * @throws {Error} If the currency pair is not supported
 */
function calculatePipValue(currencyPair: string, lotSize: number): number {
  // Implementation
}
```

#### TypeScript Types

* Use explicit types for all function parameters and return values
* Use interfaces for object shapes that will be implemented
* Use type aliases for unions, intersections, and complex types
* Use enums for fixed sets of values
* Use generics for reusable components

```typescript
enum OrderType {
  MARKET = 'MARKET',
  LIMIT = 'LIMIT',
  STOP = 'STOP',
  STOP_LIMIT = 'STOP_LIMIT'
}

enum OrderSide {
  BUY = 'BUY',
  SELL = 'SELL'
}

interface OrderParams {
  symbol: string;
  orderType: OrderType;
  side: OrderSide;
  quantity: number;
  price?: number;
}

function placeOrder(params: OrderParams): string {
  // Implementation
}
```

## Service Structure Standards

### Python Service Structure

```
service-name/
├── service_name/                # Main package
│   ├── __init__.py              # Package initialization
│   ├── main.py                  # Application entry point
│   ├── config.py                # Configuration
│   ├── constants.py             # Constants and enums
│   ├── api/                     # API endpoints
│   │   ├── __init__.py
│   │   ├── routes.py            # Route definitions
│   │   ├── models.py            # API models (request/response)
│   │   └── dependencies.py      # API dependencies
│   ├── domain/                  # Domain models and logic
│   │   ├── __init__.py
│   │   ├── models.py            # Domain entities
│   │   └── services.py          # Domain services
│   ├── adapters/                # External service adapters
│   │   ├── __init__.py
│   │   └── service_adapter.py   # Adapter implementation
│   ├── infrastructure/          # Infrastructure concerns
│   │   ├── __init__.py
│   │   ├── database/            # Database access
│   │   ├── messaging/           # Message bus integration
│   │   └── logging/             # Logging configuration
│   └── utils/                   # Utility functions
├── tests/                       # Test directory
│   ├── unit/                    # Unit tests
│   ├── integration/             # Integration tests
│   └── conftest.py              # Test fixtures
├── pyproject.toml               # Project configuration
├── README.md                    # Project documentation
└── Makefile                     # Build and development tasks
```

### JavaScript/TypeScript Frontend Structure

```
ui-service/
├── src/                         # Source code
│   ├── components/              # React components
│   │   ├── common/              # Shared components
│   │   ├── charts/              # Chart components
│   │   └── trading/             # Trading components
│   ├── hooks/                   # Custom React hooks
│   │   ├── useMarketData.ts
│   │   └── useOrderManagement.ts
│   ├── services/                # API services
│   │   ├── marketDataService.ts
│   │   └── orderService.ts
│   ├── store/                   # State management
│   │   ├── slices/              # Redux slices
│   │   └── store.ts             # Store configuration
│   ├── utils/                   # Utility functions
│   │   ├── formatting.ts
│   │   └── calculations.ts
│   ├── types/                   # TypeScript type definitions
│   │   ├── marketData.ts
│   │   └── orders.ts
│   ├── constants/               # Application constants
│   │   ├── timeframes.ts
│   │   └── orderTypes.ts
│   └── App.tsx                  # Application entry point
├── public/                      # Static assets
├── tests/                       # Test directory
│   ├── unit/                    # Unit tests
│   └── integration/             # Integration tests
├── package.json                 # Project configuration
├── tsconfig.json                # TypeScript configuration
├── .eslintrc.js                 # ESLint configuration
└── README.md                    # Project documentation
```

## Automated Tools and Enforcement

### Python Tools

1. **Black**: Code formatting with line length of 88
2. **isort**: Import sorting with Black compatibility
3. **flake8**: Linting with custom rules
4. **mypy**: Static type checking
5. **pylint**: Advanced static analysis

### JavaScript/TypeScript Tools

1. **Prettier**: Code formatting
2. **ESLint**: Linting with custom rules
3. **TypeScript**: Static type checking

### CI/CD Integration

1. **Pre-commit Hooks**: Run linting and formatting before commits
2. **CI Checks**: Enforce standards in pull requests
3. **Automated Reports**: Generate code quality reports

## Domain-Specific Coding Patterns

### Technical Analysis Patterns

```python
class Indicator:
    """Base class for all technical indicators."""

    def __init__(self, input_data: pd.DataFrame, params: Dict[str, Any]):
        """Initialize the indicator with input data and parameters."""
        self.input_data = input_data
        self.params = params
        self.result = None

    def calculate(self) -> pd.DataFrame:
        """Calculate the indicator values."""
        raise NotImplementedError("Subclasses must implement calculate()")

    def is_signal_generated(self) -> bool:
        """Check if the indicator generates a trading signal."""
        raise NotImplementedError("Subclasses must implement is_signal_generated()")
```

### Order Management Patterns

```python
class OrderService:
    """Service for managing trading orders."""

    def __init__(self, broker_adapter: BrokerAdapter, risk_manager: RiskManager):
        """Initialize the order service with dependencies."""
        self.broker_adapter = broker_adapter
        self.risk_manager = risk_manager

    async def place_order(self, order: Order) -> OrderResult:
        """Place an order with the broker after risk checks."""
        # Validate order
        if not self._validate_order(order):
            raise InvalidOrderError(f"Invalid order: {order}")

        # Apply risk management
        approved_order = self.risk_manager.apply_risk_rules(order)

        # Execute order
        try:
            result = await self.broker_adapter.execute_order(approved_order)
            return OrderResult(
                order_id=result.order_id,
                status=result.status,
                filled_price=result.filled_price,
                filled_quantity=result.filled_quantity,
                timestamp=result.timestamp
            )
        except BrokerError as e:
            raise OrderExecutionError(f"Failed to execute order: {e}")
```

### Market Data Patterns

```python
class MarketDataService:
    """Service for retrieving and processing market data."""

    def __init__(self, data_provider: DataProvider, cache_manager: CacheManager):
        """Initialize the market data service with dependencies."""
        self.data_provider = data_provider
        self.cache_manager = cache_manager
        self.logger = logging.getLogger(__name__)

    async def get_ohlcv_data(
        self,
        symbol: str,
        timeframe: Timeframe,
        start_time: datetime,
        end_time: Optional[datetime] = None,
        include_current_candle: bool = False
    ) -> pd.DataFrame:
        """
        Get OHLCV (Open, High, Low, Close, Volume) data for a symbol.

        Args:
            symbol: The trading symbol (e.g., 'EUR/USD')
            timeframe: The timeframe for the data
            start_time: The start time for the data
            end_time: The end time for the data (defaults to current time)
            include_current_candle: Whether to include the current incomplete candle

        Returns:
            DataFrame with OHLCV data

        Raises:
            MarketDataError: If data cannot be retrieved
        """
        # Validate inputs
        self._validate_inputs(symbol, timeframe, start_time, end_time)

        # Check cache first
        cache_key = self._generate_cache_key(symbol, timeframe, start_time, end_time, include_current_candle)
        cached_data = self.cache_manager.get(cache_key)
        if cached_data is not None:
            self.logger.debug(f"Retrieved cached data for {cache_key}")
            return cached_data

        # Fetch from provider if not in cache
        try:
            data = await self.data_provider.get_historical_data(
                symbol=symbol,
                timeframe=timeframe,
                start_time=start_time,
                end_time=end_time,
                include_current_candle=include_current_candle
            )

            # Process data
            processed_data = self._process_ohlcv_data(data)

            # Cache the result
            self.cache_manager.set(cache_key, processed_data, ttl=self._get_cache_ttl(timeframe))

            return processed_data
        except Exception as e:
            self.logger.error(f"Failed to retrieve market data: {e}")
            raise MarketDataError(f"Failed to retrieve market data for {symbol}: {e}")
```

### Machine Learning Patterns

```python
class ModelService:
    """Service for managing machine learning models."""

    def __init__(
        self,
        model_registry: ModelRegistry,
        feature_store: FeatureStore,
        metrics_collector: MetricsCollector
    ):
        """Initialize the model service with dependencies."""
        self.model_registry = model_registry
        self.feature_store = feature_store
        self.metrics_collector = metrics_collector
        self.logger = logging.getLogger(__name__)

    async def predict(
        self,
        model_id: str,
        features: Dict[str, Any],
        version: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate predictions using a model.

        Args:
            model_id: The ID of the model to use
            features: The features to use for prediction
            version: Optional model version (defaults to latest production version)

        Returns:
            Prediction results

        Raises:
            ModelNotFoundError: If the model cannot be found
            ModelVersionNotFoundError: If the specified version cannot be found
            ModelPredictionError: If prediction fails
        """
        # Get model
        try:
            model = await self.model_registry.get_model(model_id, version)
        except Exception as e:
            self.logger.error(f"Failed to get model {model_id} (version {version}): {e}")
            raise ModelNotFoundError(f"Failed to get model {model_id} (version {version}): {e}")

        # Validate features
        self._validate_features(features, model.feature_schema)

        # Generate prediction
        start_time = time.time()
        try:
            prediction = model.predict(features)

            # Record metrics
            prediction_time = time.time() - start_time
            self.metrics_collector.record_prediction_time(model_id, prediction_time)
            self.metrics_collector.increment_prediction_count(model_id)

            return {
                "model_id": model_id,
                "model_version": model.version,
                "prediction": prediction,
                "prediction_time": prediction_time
            }
        except Exception as e:
            self.logger.error(f"Failed to generate prediction with model {model_id}: {e}")
            self.metrics_collector.increment_prediction_error_count(model_id)
            raise ModelPredictionError(f"Failed to generate prediction with model {model_id}: {e}")
```

### Multi-Timeframe Analysis Patterns

```python
class MultiTimeframeAnalyzer:
    """Analyzer for multi-timeframe analysis."""

    def __init__(
        self,
        market_data_service: MarketDataService,
        timeframes: List[Timeframe],
        indicators: Dict[Timeframe, List[Indicator]]
    ):
        """
        Initialize the multi-timeframe analyzer.

        Args:
            market_data_service: Service for retrieving market data
            timeframes: List of timeframes to analyze, in order of priority
            indicators: Dictionary mapping timeframes to lists of indicators
        """
        self.market_data_service = market_data_service
        self.timeframes = timeframes
        self.indicators = indicators
        self.logger = logging.getLogger(__name__)

    async def analyze(
        self,
        symbol: str,
        start_time: datetime,
        end_time: Optional[datetime] = None
    ) -> Dict[Timeframe, Dict[str, pd.DataFrame]]:
        """
        Perform multi-timeframe analysis for a symbol.

        Args:
            symbol: The trading symbol (e.g., 'EUR/USD')
            start_time: The start time for the analysis
            end_time: The end time for the analysis (defaults to current time)

        Returns:
            Dictionary mapping timeframes to dictionaries of indicator results

        Raises:
            MultiTimeframeAnalysisError: If analysis fails
        """
        results = {}

        # Analyze each timeframe
        for timeframe in self.timeframes:
            try:
                # Get market data for this timeframe
                data = await self.market_data_service.get_ohlcv_data(
                    symbol=symbol,
                    timeframe=timeframe,
                    start_time=start_time,
                    end_time=end_time
                )

                # Calculate indicators for this timeframe
                timeframe_results = {}
                for indicator in self.indicators.get(timeframe, []):
                    indicator_instance = indicator(data, {})
                    result = indicator_instance.calculate()
                    timeframe_results[indicator_instance.__class__.__name__] = result

                results[timeframe] = timeframe_results
            except Exception as e:
                self.logger.error(f"Failed to analyze timeframe {timeframe}: {e}")
                raise MultiTimeframeAnalysisError(f"Failed to analyze timeframe {timeframe}: {e}")

        return results

    def get_consolidated_signals(
        self,
        results: Dict[Timeframe, Dict[str, pd.DataFrame]]
    ) -> Dict[str, Any]:
        """
        Get consolidated signals from multi-timeframe analysis.

        Args:
            results: Results from multi-timeframe analysis

        Returns:
            Dictionary of consolidated signals
        """
        signals = {}

        # Process signals from each timeframe, giving priority to higher timeframes
        for timeframe in self.timeframes:
            timeframe_results = results.get(timeframe, {})
            for indicator_name, result in timeframe_results.items():
                # Apply signal generation logic
                signal = self._generate_signal(indicator_name, result, timeframe)

                # Add to signals if not already present from a higher timeframe
                if signal and indicator_name not in signals:
                    signals[indicator_name] = {
                        "timeframe": timeframe,
                        "signal": signal,
                        "strength": self._calculate_signal_strength(signal, timeframe)
                    }

        return signals
```

## Examples of Correct and Incorrect Implementations

### Domain Model Examples

#### Correct Implementation

```python
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Optional, List, Literal

OrderSide = Literal['BUY', 'SELL']
OrderType = Literal['MARKET', 'LIMIT', 'STOP', 'STOP_LIMIT']
OrderStatus = Literal['PENDING', 'FILLED', 'PARTIALLY_FILLED', 'CANCELLED', 'REJECTED']

@dataclass
class Order:
    """Represents a trading order in the forex market."""

    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: Decimal
    price: Optional[Decimal] = None
    stop_price: Optional[Decimal] = None
    time_in_force: str = 'GTC'
    order_id: Optional[str] = None
    status: OrderStatus = 'PENDING'
    created_at: datetime = datetime.utcnow()
    updated_at: Optional[datetime] = None

    def is_valid(self) -> bool:
        """Check if the order is valid based on its type and parameters."""
        if self.order_type in ('LIMIT', 'STOP_LIMIT') and self.price is None:
            return False
        if self.order_type in ('STOP', 'STOP_LIMIT') and self.stop_price is None:
            return False
        return True
```

#### Incorrect Implementation

```python
class Order:
    """Order class."""

    def __init__(self, data):
        """Initialize with data."""
        self.data = data

    def process(self):
        """Process the order."""
        # Implementation with no validation or domain logic
        pass
```

### Service Implementation Examples

#### Correct Implementation

```python
class MarketDataService:
    """Service for retrieving and processing market data."""

    def __init__(self, data_provider: DataProvider, cache_manager: CacheManager):
        """Initialize the market data service with dependencies."""
        self.data_provider = data_provider
        self.cache_manager = cache_manager
        self.logger = logging.getLogger(__name__)

    async def get_ohlcv_data(
        self,
        symbol: str,
        timeframe: Timeframe,
        start_time: datetime,
        end_time: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Get OHLCV (Open, High, Low, Close, Volume) data for a symbol.

        Args:
            symbol: The trading symbol (e.g., 'EUR/USD')
            timeframe: The timeframe for the data
            start_time: The start time for the data
            end_time: The end time for the data (defaults to current time)

        Returns:
            DataFrame with OHLCV data

        Raises:
            MarketDataError: If data cannot be retrieved
        """
        # Check cache first
        cache_key = f"{symbol}_{timeframe}_{start_time.isoformat()}_{end_time.isoformat() if end_time else 'now'}"
        cached_data = self.cache_manager.get(cache_key)
        if cached_data is not None:
            self.logger.debug(f"Retrieved cached data for {cache_key}")
            return cached_data

        # Fetch from provider if not in cache
        try:
            data = await self.data_provider.get_historical_data(
                symbol=symbol,
                timeframe=timeframe,
                start_time=start_time,
                end_time=end_time
            )

            # Process data
            processed_data = self._process_ohlcv_data(data)

            # Cache the result
            self.cache_manager.set(cache_key, processed_data, ttl=3600)  # Cache for 1 hour

            return processed_data
        except Exception as e:
            self.logger.error(f"Failed to retrieve market data: {e}")
            raise MarketDataError(f"Failed to retrieve market data for {symbol}: {e}")

    def _process_ohlcv_data(self, data: List[Dict[str, Any]]) -> pd.DataFrame:
        """Process raw OHLCV data into a pandas DataFrame."""
        # Implementation
```

#### Incorrect Implementation

```python
class MarketDataService:
    """Market data service."""

    def get_data(self, symbol, tf, start, end=None):
        """Get data."""
        # No error handling
        # No caching strategy
        # No proper parameter validation
        # No domain-specific terminology
        data = requests.get(f"https://api.example.com/data?symbol={symbol}&tf={tf}&start={start}&end={end}")
        return data.json()
```

## Migration Strategy

1. **Phased Approach**
   - Start with critical services
   - Apply standards to new code first
   - Gradually refactor existing code

2. **Documentation and Training**
   - Provide comprehensive documentation
   - Conduct knowledge sharing sessions
   - Create examples of correct implementations

3. **External Libraries**
   - Document integration patterns for external libraries
   - Create wrappers to maintain consistent interfaces
   - Isolate external dependencies in adapter layers

## Review and Evolution

These standards should evolve with the platform:

1. **Regular Reviews**
   - Schedule quarterly reviews of standards
   - Collect feedback from development teams
   - Update standards based on lessons learned

2. **Metrics and Monitoring**
   - Track adherence to standards
   - Monitor impact on development velocity
   - Measure code quality improvements

3. **Continuous Improvement**
   - Refine standards based on project needs
   - Incorporate new best practices
   - Remove outdated or unnecessary rules