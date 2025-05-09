# ADR-0001: Service Communication Patterns

## Status

Accepted

## Context

The Forex Trading Platform consists of 28 microservices that need to communicate effectively while maintaining loose coupling. The initial implementation led to several issues:

1. Direct imports between services creating tight coupling
2. Circular dependencies between services
3. Inconsistent communication patterns
4. Lack of resilience in service communication
5. Difficulty in evolving services independently

## Decision

We will implement the following service communication patterns:

### 1. Event-Driven Communication
- Use an event bus for asynchronous communication
- Define clear event schemas in common-lib
- Implement at-least-once delivery
- Use dead letter queues for failed messages

### 2. Interface-Based Communication
- Define interfaces in common-lib
- Implement concrete adapters in each service
- Use dependency injection for flexibility
- Support multiple implementations (e.g., HTTP, gRPC)

### 3. Communication Layers
```
┌──────────────────────────────────────────┐
│              Common Lib                  │
├──────────────────────────────────────────┤
│ ┌─────────────┐  ┌────────────────────┐ │
│ │  Interfaces │  │   Event Schemas    │ │
│ └─────────────┘  └────────────────────┘ │
└──────────────────────────────────────────┘
              ▲              ▲
              │              │
┌─────────────┴──┐    ┌─────┴────────┐
│    Adapters    │    │  Event Bus   │
└─────────────────┘    └─────────────┘
              ▲              ▲
              │              │
┌─────────────┴──────────────┴────────┐
│           Services                   │
└────────────────────────────────────────┘
```

### 4. Error Handling
- Implement circuit breakers
- Use timeouts and retries
- Provide fallback mechanisms
- Monitor communication health

### 5. Versioning
- Version interfaces and events
- Support backward compatibility
- Implement graceful degradation
- Use feature flags for transitions

## Consequences

### Positive
1. Reduced coupling between services
2. Improved resilience and fault tolerance
3. Easier service evolution and maintenance
4. Better scalability and performance
5. Clear communication boundaries

### Negative
1. Increased initial development complexity
2. Need for additional infrastructure (event bus)
3. More complex testing requirements
4. Additional monitoring needs
5. Learning curve for developers

## Compliance

Each service must:
1. Implement communication through defined interfaces
2. Use the event bus for asynchronous operations
3. Include proper error handling and monitoring
4. Follow versioning guidelines
5. Document its communication patterns

## Implementation Notes

1. Use common-lib for shared interfaces and models
2. Implement circuit breakers using resilience4j
3. Use OpenTelemetry for tracing
4. Monitor with Prometheus and Grafana
5. Document with OpenAPI/AsyncAPI
