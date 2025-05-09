"""
Adapters package for ML workbench service.

This package contains adapter implementations for interfaces
to break circular dependencies between services.
"""

from ml_workbench_service.adapters.analysis_adapters import (
    AnalysisEngineAdapter,
    MarketRegimeAnalyzerAdapter,
    MultiAssetAnalyzerAdapter
)
from ml_workbench_service.adapters.simulation_adapters import (
    MarketRegimeSimulatorAdapter,
    BrokerSimulatorAdapter
)
from ml_workbench_service.adapters.risk_adapters import RiskManagerAdapter
from ml_workbench_service.adapters.risk_optimizer_adapter import (
    RiskParametersAdapter,
    RiskParameterOptimizerAdapter
)
from ml_workbench_service.adapters.trading_feedback_adapter import (
    TradingFeedbackCollectorAdapter,
    ModelTrainingFeedbackIntegratorAdapter
)

__all__ = [
    'AnalysisEngineAdapter',
    'MarketRegimeAnalyzerAdapter',
    'MultiAssetAnalyzerAdapter',
    'MarketRegimeSimulatorAdapter',
    'BrokerSimulatorAdapter',
    'RiskManagerAdapter',
    'RiskParametersAdapter',
    'RiskParameterOptimizerAdapter',
    'TradingFeedbackCollectorAdapter',
    'ModelTrainingFeedbackIntegratorAdapter'
]
"""
