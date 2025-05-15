import pytest
import asyncio
from unittest.mock import MagicMock, patch, AsyncMock
from datetime import datetime

from causal_analysis_service.services.causal_analysis_service import CausalAnalysisService
from causal_analysis_service.models.causal_models import (
    CausalGraphRequest,
    CausalGraphResponse,
    InterventionEffectRequest,
    InterventionEffectResponse,
    CounterfactualScenarioRequest,
    CounterfactualScenarioResponse,
    CausalAlgorithm
)

@pytest.fixture
def feature_store_adapter():
    adapter = AsyncMock()
    adapter.get_feature_data = AsyncMock()
    adapter.get_feature_metadata = AsyncMock()
    adapter.get_feature_correlations = AsyncMock()
    return adapter

@pytest.fixture
def analysis_coordinator_adapter():
    adapter = AsyncMock()
    adapter.notify_causal_graph_created = AsyncMock()
    adapter.get_analysis_context = AsyncMock()
    adapter.update_analysis_status = AsyncMock()
    return adapter

@pytest.fixture
def causal_graph_repository():
    repository = AsyncMock()
    repository.save_graph = AsyncMock()
    repository.get_graph = AsyncMock()
    repository.delete_graph = AsyncMock()
    repository.list_graphs = AsyncMock()
    return repository

@pytest.fixture
def causal_analysis_service(feature_store_adapter, analysis_coordinator_adapter, causal_graph_repository):
    return CausalAnalysisService(
        feature_store_adapter=feature_store_adapter,
        analysis_coordinator_adapter=analysis_coordinator_adapter,
        causal_graph_repository=causal_graph_repository
    )

@pytest.mark.asyncio
async def test_generate_causal_graph(causal_analysis_service, causal_graph_repository, analysis_coordinator_adapter):
    # Arrange
    request = CausalGraphRequest(
        data={
            "x": [1.0, 2.0, 3.0],
            "y": [2.0, 4.0, 6.0],
            "z": [3.0, 6.0, 9.0]
        },
        algorithm=CausalAlgorithm.PC,
        significance_level=0.05,
        max_lag=5
    )
    
    # Mock the UUID generation
    with patch('uuid.uuid4', return_value='test-uuid'):
        # Act
        response = await causal_analysis_service.generate_causal_graph(request)
        
        # Assert
        assert isinstance(response, CausalGraphResponse)
        assert response.algorithm_used == CausalAlgorithm.PC
        assert isinstance(response.execution_time_ms, int)
        assert isinstance(response.timestamp, datetime)
        
        # Verify repository and adapter calls
        causal_graph_repository.save_graph.assert_called_once()
        analysis_coordinator_adapter.notify_causal_graph_created.assert_called_once_with('test-uuid')

@pytest.mark.asyncio
async def test_calculate_intervention_effect(causal_analysis_service, causal_graph_repository):
    # Arrange
    graph_data = CausalGraphResponse(
        graph={"x": ["y"], "y": ["z"], "z": []},
        edge_weights={"x": {"y": 0.5}, "y": {"z": 0.7}, "z": {}},
        node_metadata={"x": {}, "y": {}, "z": {}},
        algorithm_used=CausalAlgorithm.PC,
        execution_time_ms=100,
        timestamp=datetime.now()
    )
    causal_graph_repository.get_graph.return_value = graph_data
    
    request = InterventionEffectRequest(
        graph_id="test-graph-id",
        intervention_variable="x",
        intervention_value=2.0,
        target_variables=["y", "z"]
    )
    
    # Act
    response = await causal_analysis_service.calculate_intervention_effect(request)
    
    # Assert
    assert isinstance(response, InterventionEffectResponse)
    assert isinstance(response.execution_time_ms, int)
    assert isinstance(response.timestamp, datetime)
    
    # Verify repository calls
    causal_graph_repository.get_graph.assert_called_once_with("test-graph-id")

@pytest.mark.asyncio
async def test_generate_counterfactual_scenario(causal_analysis_service, causal_graph_repository):
    # Arrange
    graph_data = CausalGraphResponse(
        graph={"x": ["y"], "y": ["z"], "z": []},
        edge_weights={"x": {"y": 0.5}, "y": {"z": 0.7}, "z": {}},
        node_metadata={"x": {}, "y": {}, "z": {}},
        algorithm_used=CausalAlgorithm.PC,
        execution_time_ms=100,
        timestamp=datetime.now()
    )
    causal_graph_repository.get_graph.return_value = graph_data
    
    request = CounterfactualScenarioRequest(
        graph_id="test-graph-id",
        factual_values={"x": 1.0, "y": 2.0, "z": 3.0},
        intervention_variables={"x": 2.0},
        target_variables=["y", "z"]
    )
    
    # Act
    response = await causal_analysis_service.generate_counterfactual_scenario(request)
    
    # Assert
    assert isinstance(response, CounterfactualScenarioResponse)
    assert isinstance(response.execution_time_ms, int)
    assert isinstance(response.timestamp, datetime)
    
    # Verify repository calls
    causal_graph_repository.get_graph.assert_called_once_with("test-graph-id")