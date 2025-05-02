# Analysis Engine Service Architecture

## Current Architecture

### Core Components

1. **Base Classes**
   - `BaseAnalyzer`: Abstract base class for all analysis components
   - `BaseNLPAnalyzer`: Specialized base class for NLP analysis
   - `AdvancedAnalysisBase`: Base class for advanced technical analysis

2. **Analysis Components**
   - Confluence Analysis
   - Multi-Timeframe Analysis
   - Market Regime Analysis
   - Volume/Volatility Analysis
   - NLP Analysis
   - Fibonacci Analysis

3. **Services**
   - `AnalysisService`: Central service for managing analyzers
   - `AnalysisIntegrationService`: Integrates signals from different components
   - `AnalysisEngineMonitoring`: Performance monitoring

### Current Issues

1. **Inconsistent Base Classes**
   - Multiple base classes with overlapping functionality
   - Inconsistent inheritance patterns
   - Duplicate monitoring and logging code

2. **Service Layer Inconsistencies**
   - Mixed responsibilities between services
   - Inconsistent error handling
   - Duplicate initialization code

3. **Component Integration**
   - Ad-hoc integration patterns
   - Inconsistent data flow
   - Lack of clear component boundaries

4. **Monitoring and Logging**
   - Multiple monitoring implementations
   - Inconsistent metrics collection
   - Scattered logging code

## Proposed Architecture

### 1. Unified Base Classes

```python
class BaseComponent(ABC):
    """Base class for all components with common functionality"""

    def __init__(self, name: str, parameters: Dict[str, Any] = None):
        self.name = name
        self.parameters = parameters or {}
        self.monitor = ComponentMonitor(name)
        self.logger = logging.getLogger(f"{__name__}.{name}")

    @abstractmethod
    def execute(self, data: Any) -> Any:
        """Execute component logic"""
        pass

    def get_metrics(self) -> Dict[str, Any]:
        """Get component metrics"""
        return self.monitor.get_metrics()

class BaseAnalyzer(BaseComponent):
    """Base class for all analyzers"""

    def __init__(self, name: str, parameters: Dict[str, Any] = None):
        super().__init__(name, parameters)
        self.effectiveness_tracker = EffectivenessTracker(name)

    @abstractmethod
    def analyze(self, data: Any) -> AnalysisResult:
        """Perform analysis"""
        pass

    def execute(self, data: Any) -> AnalysisResult:
        """Execute analysis with monitoring"""
        with self.monitor.track_operation("analyze"):
            return self.analyze(data)

class BaseService(BaseComponent):
    """Base class for all services"""

    def __init__(self, name: str, parameters: Dict[str, Any] = None):
        super().__init__(name, parameters)
        self.dependencies = {}

    def register_dependency(self, name: str, component: BaseComponent):
        """Register a component dependency"""
        self.dependencies[name] = component

    def get_dependency(self, name: str) -> Optional[BaseComponent]:
        """Get a registered dependency"""
        return self.dependencies.get(name)
```

### 2. Standardized Service Layer

```python
class AnalysisService(BaseService):
    """Central service for managing analyzers"""

    def __init__(self, parameters: Dict[str, Any] = None):
        super().__init__("analysis_service", parameters)
        self.analyzers = {}

    def register_analyzer(self, name: str, analyzer: BaseAnalyzer):
        """Register an analyzer"""
        self.analyzers[name] = analyzer
        self.register_dependency(f"analyzer_{name}", analyzer)

    async def analyze(self,
                     data: Any,
                     analyzers: List[str] = None) -> Dict[str, AnalysisResult]:
        """Run analysis with specified analyzers"""
        results = {}
        analyzers = analyzers or list(self.analyzers.keys())

        for name in analyzers:
            if analyzer := self.analyzers.get(name):
                try:
                    results[name] = await analyzer.execute(data)
                except Exception as e:
                    self.logger.error(f"Error in analyzer {name}: {e}")
                    results[name] = AnalysisResult(
                        analyzer_name=name,
                        error=str(e),
                        is_valid=False
                    )

        return results

class IntegrationService(BaseService):
    """Service for integrating analysis results"""

    def __init__(self, parameters: Dict[str, Any] = None):
        super().__init__("integration_service", parameters)
        self.integrators = {}

    def register_integrator(self, name: str, integrator: BaseComponent):
        """Register an integrator"""
        self.integrators[name] = integrator
        self.register_dependency(f"integrator_{name}", integrator)

    async def integrate(self,
                       results: Dict[str, AnalysisResult],
                       integrators: List[str] = None) -> Dict[str, Any]:
        """Integrate analysis results"""
        integrated = {}
        integrators = integrators or list(self.integrators.keys())

        for name in integrators:
            if integrator := self.integrators.get(name):
                try:
                    integrated[name] = await integrator.execute(results)
                except Exception as e:
                    self.logger.error(f"Error in integrator {name}: {e}")
                    integrated[name] = {
                        "error": str(e),
                        "is_valid": False
                    }

        return integrated
```

### 3. Standardized Component Integration

```python
class ComponentRegistry:
    """Registry for managing component dependencies"""

    def __init__(self):
        self.components = {}
        self.dependencies = {}

    def register(self, name: str, component: BaseComponent):
        """Register a component"""
        self.components[name] = component

    def register_dependency(self, component: str, dependency: str):
        """Register a component dependency"""
        if component not in self.dependencies:
            self.dependencies[component] = set()
        self.dependencies[component].add(dependency)

    def get_dependencies(self, component: str) -> Set[str]:
        """Get component dependencies"""
        return self.dependencies.get(component, set())

    def validate_dependencies(self) -> bool:
        """Validate component dependencies"""
        for component, deps in self.dependencies.items():
            for dep in deps:
                if dep not in self.components:
                    return False
        return True

class ComponentFactory:
    """Factory for creating components"""

    def __init__(self, registry: ComponentRegistry):
        self.registry = registry

    def create_analyzer(self,
                       name: str,
                       analyzer_class: Type[BaseAnalyzer],
                       parameters: Dict[str, Any] = None) -> BaseAnalyzer:
        """Create an analyzer instance"""
        analyzer = analyzer_class(name, parameters)
        self.registry.register(name, analyzer)
        return analyzer

    def create_service(self,
                      name: str,
                      service_class: Type[BaseService],
                      parameters: Dict[str, Any] = None) -> BaseService:
        """Create a service instance"""
        service = service_class(name, parameters)
        self.registry.register(name, service)
        return service
```

### 4. Unified Monitoring and Logging

```python
class ComponentMonitor:
    """Unified monitoring for components"""

    def __init__(self, component_name: str):
        self.name = component_name
        self.metrics = {
            "execution_count": 0,
            "total_time": 0,
            "error_count": 0,
            "last_execution": None
        }

    def track_operation(self, operation: str):
        """Track an operation execution"""
        return OperationTracker(self, operation)

    def record_execution(self,
                        operation: str,
                        duration: float,
                        error: Optional[Exception] = None):
        """Record operation execution"""
        self.metrics["execution_count"] += 1
        self.metrics["total_time"] += duration
        self.metrics["last_execution"] = datetime.now()

        if error:
            self.metrics["error_count"] += 1

    def get_metrics(self) -> Dict[str, Any]:
        """Get component metrics"""
        return {
            **self.metrics,
            "average_time": (
                self.metrics["total_time"] / self.metrics["execution_count"]
                if self.metrics["execution_count"] > 0
                else 0
            ),
            "error_rate": (
                self.metrics["error_count"] / self.metrics["execution_count"]
                if self.metrics["execution_count"] > 0
                else 0
            )
        }

class OperationTracker:
    """Context manager for tracking operations"""

    def __init__(self, monitor: ComponentMonitor, operation: str):
        self.monitor = monitor
        self.operation = operation
        self.start_time = None

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.perf_counter() - self.start_time
        self.monitor.record_execution(
            self.operation,
            duration,
            exc_val if exc_type else None
        )
```

## Implementation Plan

1. **Phase 1: Base Classes**
   - Create new base classes
   - Migrate existing analyzers to new base classes
   - Update service layer to use new base classes

2. **Phase 2: Service Layer**
   - Implement new service classes
   - Migrate existing services to new structure
   - Update dependency injection

3. **Phase 3: Component Integration**
   - Implement component registry
   - Create component factory
   - Update component initialization

4. **Phase 4: Monitoring and Logging**
   - Implement unified monitoring
   - Update logging configuration
   - Migrate existing monitoring code

## Benefits

1. **Improved Maintainability**
   - Consistent patterns across components
   - Clear component boundaries
   - Standardized error handling

2. **Better Performance**
   - Unified monitoring
   - Optimized component initialization
   - Efficient dependency management

3. **Enhanced Reliability**
   - Standardized error handling
   - Better dependency validation
   - Improved logging and monitoring

4. **Easier Development**
   - Clear component interfaces
   - Simplified component creation
   - Better code organization

## Migration Strategy

1. **Incremental Migration**
   - Migrate one component at a time
   - Maintain backward compatibility
   - Update tests as components are migrated

2. **Testing Strategy**
   - Unit tests for new base classes
   - Integration tests for services
   - Performance tests for monitoring

3. **Documentation**
   - Update API documentation
   - Create migration guides
   - Document new patterns
   - Document async patterns (see [Async Patterns](./docs/async_patterns.md))

4. **Monitoring**
   - Track migration progress
   - Monitor performance impact
   - Validate improvements