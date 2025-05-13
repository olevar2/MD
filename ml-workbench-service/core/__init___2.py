"""
Adapters package for ML workbench service.

This package contains adapter implementations for interfaces
to break circular dependencies between services.
"""

from adapters.analysis_adapters import (
    AnalysisEngineAdapter,
    MarketRegimeAnalyzerAdapter,
    MultiAssetAnalyzerAdapter
)
from adapters.simulation_adapters import (
    MarketRegimeSimulatorAdapter,
    BrokerSimulatorAdapter
)
from adapters.risk_adapters import RiskManagerAdapter
from adapters.risk_optimizer_adapter import (
    RiskParametersAdapter,
    RiskParameterOptimizerAdapter
)
from adapters.trading_feedback_adapter import (
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
""""""
