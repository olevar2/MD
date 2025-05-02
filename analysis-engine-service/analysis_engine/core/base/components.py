"""
Core Components Module

This module provides the base classes and abstractions for the analysis engine service.
It defines the fundamental interfaces and base implementations that all analyzers
and services should extend.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import logging
import time
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class AnalysisResult:
    """
    Standard container for analysis results
    """
    analyzer_name: str
    result: Dict[str, Any]
    metadata: Optional[Dict[str, Any]] = None
    is_valid: bool = True
    execution_time: Optional[float] = None
    timestamp: Optional[str] = None

    def __post_init__(self):
        """Set defaults for optional fields"""
        if self.metadata is None:
            self.metadata = {}
        if self.timestamp is None:
            self.timestamp = datetime.utcnow().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            "analyzer_name": self.analyzer_name,
            "result": self.result,
            "metadata": self.metadata,
            "is_valid": self.is_valid,
            "execution_time": self.execution_time,
            "timestamp": self.timestamp
        }

class BaseComponent(ABC):
    """
    Base class for all components in the analysis engine
    
    Provides:
    - Initialization with parameters
    - Performance monitoring
    - Logging setup
    - Error handling patterns
    """
    
    def __init__(self, name: str, parameters: Optional[Dict[str, Any]] = None):
        """
        Initialize base component
        
        Args:
            name: Component identifier
            parameters: Optional configuration parameters
        """
        self.name = name
        self.parameters = parameters or {}
        self.logger = logging.getLogger(f"{__name__}.{name}")
        self._execution_count = 0
        self._total_execution_time = 0.0
        self._last_execution_time = None
        
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get component performance metrics"""
        avg_execution_time = (
            self._total_execution_time / self._execution_count 
            if self._execution_count > 0 else 0
        )
        
        return {
            "name": self.name,
            "execution_count": self._execution_count,
            "total_execution_time": self._total_execution_time,
            "average_execution_time": avg_execution_time,
            "last_execution_time": self._last_execution_time
        }
    
    def _update_metrics(self, execution_time: float):
        """Update performance metrics"""
        self._execution_count += 1
        self._total_execution_time += execution_time
        self._last_execution_time = execution_time

class BaseAnalyzer(BaseComponent):
    """
    Base class for all analyzers
    
    Provides:
    - Standard analysis interface
    - Performance monitoring
    - Result validation
    - Error handling
    """
    
    @abstractmethod
    async def analyze(self, data: Dict[str, Any]) -> AnalysisResult:
        """
        Perform analysis on input data
        
        Args:
            data: Input data for analysis
            
        Returns:
            AnalysisResult containing analysis output
        """
        pass
    
    async def execute(self, data: Dict[str, Any]) -> AnalysisResult:
        """
        Execute analysis with performance monitoring
        
        Args:
            data: Input data for analysis
            
        Returns:
            AnalysisResult with execution metrics
        """
        start_time = time.time()
        
        try:
            result = await self.analyze(data)
            execution_time = time.time() - start_time
            
            self._update_metrics(execution_time)
            result.execution_time = execution_time
            
            return result
            
        except Exception as e:
            self.logger.error(
                f"Error in analyzer {self.name}: {str(e)}", 
                exc_info=True
            )
            execution_time = time.time() - start_time
            
            return AnalysisResult(
                analyzer_name=self.name,
                result={"error": str(e)},
                is_valid=False,
                execution_time=execution_time,
                metadata={
                    "error_type": type(e).__name__,
                    "input_data_keys": list(data.keys())
                }
            )

class BaseService(BaseComponent):
    """
    Base class for all services
    
    Provides:
    - Standard service interface
    - Resource management
    - Health check capabilities
    - Dependency injection support
    """
    
    def __init__(
        self,
        name: str,
        parameters: Optional[Dict[str, Any]] = None,
        dependencies: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize service
        
        Args:
            name: Service identifier
            parameters: Optional configuration parameters
            dependencies: Optional service dependencies
        """
        super().__init__(name, parameters)
        self.dependencies = dependencies or {}
        self._is_healthy = True
        self._health_check_errors = []
        
    async def initialize(self):
        """Initialize service resources"""
        pass
    
    async def cleanup(self):
        """Cleanup service resources"""
        pass
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get service health status"""
        return {
            "name": self.name,
            "is_healthy": self._is_healthy,
            "errors": self._health_check_errors,
            "metrics": self.get_performance_metrics()
        }
    
    async def health_check(self) -> bool:
        """Perform service health check"""
        try:
            # Basic health check
            self._is_healthy = True
            self._health_check_errors = []
            return True
        except Exception as e:
            self._is_healthy = False
            self._health_check_errors.append(str(e))
            return False

class AnalysisService(BaseService):
    """
    Service for managing and executing analysis components
    
    Provides:
    - Analyzer registration and management
    - Parallel execution capabilities
    - Result aggregation
    - Error handling and recovery
    """
    
    def __init__(
        self,
        name: str = "AnalysisService",
        parameters: Optional[Dict[str, Any]] = None,
        dependencies: Optional[Dict[str, Any]] = None
    ):
        """Initialize analysis service"""
        super().__init__(name, parameters, dependencies)
        self.analyzers: Dict[str, BaseAnalyzer] = {}
        
    def register_analyzer(self, analyzer: BaseAnalyzer):
        """Register an analyzer with the service"""
        self.analyzers[analyzer.name] = analyzer
        self.logger.info(f"Registered analyzer: {analyzer.name}")
        
    def get_analyzer(self, name: str) -> Optional[BaseAnalyzer]:
        """Get analyzer by name"""
        return self.analyzers.get(name)
        
    async def execute_analysis(
        self,
        analyzer_name: str,
        data: Dict[str, Any]
    ) -> AnalysisResult:
        """
        Execute specific analyzer
        
        Args:
            analyzer_name: Name of analyzer to execute
            data: Input data for analysis
            
        Returns:
            Analysis result from specified analyzer
        """
        analyzer = self.get_analyzer(analyzer_name)
        if not analyzer:
            return AnalysisResult(
                analyzer_name=analyzer_name,
                result={"error": f"Analyzer {analyzer_name} not found"},
                is_valid=False
            )
            
        return await analyzer.execute(data)
        
    async def execute_multiple(
        self,
        analyzer_names: List[str],
        data: Dict[str, Any]
    ) -> List[AnalysisResult]:
        """
        Execute multiple analyzers in parallel
        
        Args:
            analyzer_names: List of analyzers to execute
            data: Input data for analysis
            
        Returns:
            List of analysis results
        """
        import asyncio
        
        tasks = [
            self.execute_analysis(name, data)
            for name in analyzer_names
        ]
        
        return await asyncio.gather(*tasks)
        
    async def health_check(self) -> bool:
        """Perform health check on service and all analyzers"""
        try:
            # Check base service health
            service_healthy = await super().health_check()
            
            # Check all analyzers
            analyzer_status = []
            for name, analyzer in self.analyzers.items():
                try:
                    # Basic analyzer health check
                    metrics = analyzer.get_performance_metrics()
                    analyzer_status.append({
                        "name": name,
                        "healthy": True,
                        "metrics": metrics
                    })
                except Exception as e:
                    analyzer_status.append({
                        "name": name,
                        "healthy": False,
                        "error": str(e)
                    })
            
            self._health_check_errors = [
                f"Analyzer {status['name']} unhealthy: {status['error']}"
                for status in analyzer_status
                if not status["healthy"]
            ]
            
            return service_healthy and all(
                status["healthy"] for status in analyzer_status
            )
            
        except Exception as e:
            self._is_healthy = False
            self._health_check_errors.append(str(e))
            return False 