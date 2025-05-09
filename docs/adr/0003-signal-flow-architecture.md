# ADR-0003: Signal Flow Architecture

## Status

Accepted

## Context

The communication between analysis-engine-service and strategy-execution-engine was problematic:

1. Direct service dependencies creating cycles
2. Inconsistent signal formats
3. Lack of validation and quality checks
4. No clear aggregation strategy
5. Missing monitoring and tracing

## Decision

We implemented a clear signal flow architecture with:

### 1. Core Domain Models
```python
class SignalCategory(str, Enum):
    TECHNICAL_ANALYSIS = "technical_analysis"
    MACHINE_LEARNING = "machine_learning"
    MARKET_SENTIMENT = "market_sentiment"
    ECONOMIC_INDICATORS = "economic_indicators"
    CORRELATION_SIGNALS = "correlation_signals"

class SignalFlow(BaseModel):
    signal_id: str
    generated_at: datetime
    symbol: str
    timeframe: str
    category: SignalCategory
    source: SignalSource
    direction: str
    strength: SignalStrength
    confidence: float
    priority: SignalPriority
    # ... additional fields
```

### 2. Architecture
```
common-lib/
└── signal_flow/
    ├── model.py      # Domain models
    ├── interface.py  # Core interfaces
    └── utils.py      # Shared utilities

strategy-execution-engine/
└── signal_flow/
    ├── strategy_signal_manager.py
    ├── signal_validator.py
    └── signal_aggregator.py
```

### 3. Components
1. Signal Flow Manager - Controls signal lifecycle
2. Signal Validator - Validates incoming signals
3. Signal Aggregator - Combines multiple signals
4. Signal Monitor - Tracks signal performance
5. Signal Executor - Executes trading decisions

### 4. Flow States
```
Generated -> Validated -> Aggregated -> Risk Checked -> Executing -> Completed
                     └-> Rejected
                                                    └-> Expired
```

## Consequences

### Positive
1. Clear signal flow between services
2. Consistent validation and aggregation
3. Better monitoring and tracing
4. Improved error handling
5. Enhanced testability

### Negative
1. Additional complexity in signal processing
2. Need for more comprehensive testing
3. Increased latency in signal processing
4. More complex deployment
5. Learning curve for developers

## Technical Implementation

### 1. Signal Processing
- Asynchronous processing using asyncio
- Clear validation rules
- Configurable aggregation strategies
- Comprehensive error handling

### 2. Monitoring
- Signal flow metrics
- Performance monitoring
- Error tracking
- Success rate analysis

### 3. Integration
- Event-based communication
- Clear interface contracts
- Circuit breaker implementation
- Graceful degradation

## Migration Plan

1. Implement new signal flow architecture
2. Add monitoring and validation
3. Migrate existing signals
4. Remove old dependencies
5. Monitor and optimize

## Validation

Architecture must demonstrate:
1. No circular dependencies
2. Clear signal traceability
3. Consistent validation
4. Reliable aggregation
5. Performance within latency targets
