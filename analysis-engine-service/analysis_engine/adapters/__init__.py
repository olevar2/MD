"""
Adapters package initialization.

Exports adapter components for use across the analysis engine service.
"""
from analysis_engine.adapters.data_pipeline_adapter import TickDataServiceAdapter
from analysis_engine.adapters.strategy_execution_adapter import (
    StrategyExecutorAdapter,
    SignalAggregatorAdapter,
    StrategyEvaluatorAdapter
)
from analysis_engine.adapters.tool_effectiveness_adapter import ToolEffectivenessTrackerAdapter
from analysis_engine.adapters.enhanced_tool_effectiveness_adapter import (
    EnhancedToolEffectivenessTrackerAdapter,
    AdaptiveLayerServiceAdapter
)
from analysis_engine.adapters.model_feedback_adapter import ModelTrainingFeedbackAdapter
from analysis_engine.adapters.ml_integration_adapter import (
    MLModelConnectorAdapter,
    ExplanationGeneratorAdapter,
    UserPreferenceManagerAdapter
)
from analysis_engine.adapters.multi_asset_adapter import (
    MultiAssetServiceAdapter,
    AssetRegistryAdapter
)
from analysis_engine.adapters.adaptive_strategy_adapter import AdaptiveStrategyServiceAdapter

__all__ = [
    'TickDataServiceAdapter',
    'StrategyExecutorAdapter',
    'SignalAggregatorAdapter',
    'StrategyEvaluatorAdapter',
    'ToolEffectivenessTrackerAdapter',
    'EnhancedToolEffectivenessTrackerAdapter',
    'AdaptiveLayerServiceAdapter',
    'AdaptiveStrategyServiceAdapter',
    'ModelTrainingFeedbackAdapter',
    'MLModelConnectorAdapter',
    'ExplanationGeneratorAdapter',
    'UserPreferenceManagerAdapter',
    'MultiAssetServiceAdapter',
    'AssetRegistryAdapter'
]
