"""
Write repositories for the Causal Analysis Service.

This module provides the write repositories for the Causal Analysis Service.
"""
from causal_analysis_service.repositories.write_repositories.causal_graph_write_repository import CausalGraphWriteRepository
from causal_analysis_service.repositories.write_repositories.intervention_effect_write_repository import InterventionEffectWriteRepository
from causal_analysis_service.repositories.write_repositories.counterfactual_write_repository import CounterfactualWriteRepository
from causal_analysis_service.repositories.write_repositories.currency_pair_relationship_write_repository import CurrencyPairRelationshipWriteRepository
from causal_analysis_service.repositories.write_repositories.regime_change_driver_write_repository import RegimeChangeDriverWriteRepository
from causal_analysis_service.repositories.write_repositories.correlation_breakdown_risk_write_repository import CorrelationBreakdownRiskWriteRepository

__all__ = [
    'CausalGraphWriteRepository',
    'InterventionEffectWriteRepository',
    'CounterfactualWriteRepository',
    'CurrencyPairRelationshipWriteRepository',
    'RegimeChangeDriverWriteRepository',
    'CorrelationBreakdownRiskWriteRepository'
]