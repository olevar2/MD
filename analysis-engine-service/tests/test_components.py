"""
Tests for the base component classes.
"""

import pytest
from datetime import datetime
from typing import Any, Dict

from analysis_engine.core.base.components import (
    BaseComponent,
    BaseAnalyzer,
    BaseService,
    ComponentMonitor,
    OperationTracker,
    EffectivenessTracker,
    AnalysisResult
)

class TestComponent(BaseComponent):
    """Test implementation of BaseComponent"""
    
    def execute(self, data: Any) -> Any:
        return {"processed": data}

class TestAnalyzer(BaseAnalyzer):
    """Test implementation of BaseAnalyzer"""
    
    def analyze(self, data: Any) -> AnalysisResult:
        return AnalysisResult(
            analyzer_name=self.name,
            result={"analyzed": data}
        )

class TestService(BaseService):
    """Test implementation of BaseService"""
    
    def execute(self, data: Any) -> Any:
        return {"serviced": data}

def test_base_component_initialization():
    """Test BaseComponent initialization"""
    component = TestComponent("test_component", {"param": "value"})
    
    assert component.name == "test_component"
    assert component.parameters == {"param": "value"}
    assert isinstance(component.monitor, ComponentMonitor)
    assert component.logger.name.endswith("test_component")

def test_base_component_execution():
    """Test BaseComponent execution"""
    component = TestComponent("test_component")
    result = component.execute({"test": "data"})
    
    assert result == {"processed": {"test": "data"}}
    metrics = component.get_metrics()
    assert metrics["execution_count"] == 1
    assert metrics["error_count"] == 0

def test_base_analyzer_initialization():
    """Test BaseAnalyzer initialization"""
    analyzer = TestAnalyzer("test_analyzer")
    
    assert analyzer.name == "test_analyzer"
    assert isinstance(analyzer.effectiveness_tracker, EffectivenessTracker)

def test_base_analyzer_execution():
    """Test BaseAnalyzer execution"""
    analyzer = TestAnalyzer("test_analyzer")
    result = analyzer.execute({"test": "data"})
    
    assert isinstance(result, AnalysisResult)
    assert result.analyzer_name == "test_analyzer"
    assert result.result == {"analyzed": {"test": "data"}}
    assert result.is_valid is True

def test_base_service_initialization():
    """Test BaseService initialization"""
    service = TestService("test_service")
    
    assert service.name == "test_service"
    assert service.dependencies == {}

def test_base_service_dependency_management():
    """Test BaseService dependency management"""
    service = TestService("test_service")
    component = TestComponent("test_component")
    
    service.register_dependency("test_dep", component)
    assert service.get_dependency("test_dep") == component
    assert service.get_dependency("nonexistent") is None

def test_component_monitor():
    """Test ComponentMonitor functionality"""
    monitor = ComponentMonitor("test_monitor")
    
    with monitor.track_operation("test_op"):
        pass
    
    metrics = monitor.get_metrics()
    assert metrics["execution_count"] == 1
    assert metrics["error_count"] == 0
    assert metrics["average_time"] >= 0
    assert metrics["error_rate"] == 0

def test_operation_tracker():
    """Test OperationTracker functionality"""
    monitor = ComponentMonitor("test_monitor")
    tracker = OperationTracker(monitor, "test_op")
    
    with tracker:
        pass
    
    metrics = monitor.get_metrics()
    assert metrics["execution_count"] == 1
    assert metrics["error_count"] == 0

def test_effectiveness_tracker():
    """Test EffectivenessTracker functionality"""
    tracker = EffectivenessTracker("test_analyzer")
    
    tracker.record_prediction(True, 1.0)
    tracker.record_prediction(False, 0.5)
    
    metrics = tracker.get_effectiveness()
    assert metrics["total_predictions"] == 2
    assert metrics["accuracy"] == 0.5
    assert metrics["average_impact"] == 0.75

def test_analysis_result():
    """Test AnalysisResult functionality"""
    result = AnalysisResult(
        analyzer_name="test_analyzer",
        result={"test": "data"},
        metadata={"confidence": 0.95}
    )
    
    assert result.analyzer_name == "test_analyzer"
    assert result.result == {"test": "data"}
    assert result.is_valid is True
    assert result.error is None
    assert result.metadata == {"confidence": 0.95}
    
    result_dict = result.to_dict()
    assert result_dict["analyzer_name"] == "test_analyzer"
    assert result_dict["result"] == {"test": "data"}
    assert result_dict["is_valid"] is True
    assert result_dict["error"] is None
    assert result_dict["metadata"] == {"confidence": 0.95}
    assert "timestamp" in result_dict 