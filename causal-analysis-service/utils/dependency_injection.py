"""
Dependency injection module for the Causal Analysis Service.

This module provides the dependency injection for the Causal Analysis Service.
"""
import logging
from typing import Optional

from common_lib.cqrs.commands import CommandBus
from common_lib.cqrs.queries import QueryBus
from causal_analysis_service.cqrs.handlers.command_handlers import (
    GenerateCausalGraphCommandHandler,
    AnalyzeInterventionEffectCommandHandler,
    GenerateCounterfactualScenarioCommandHandler,
    DiscoverCurrencyPairRelationshipsCommandHandler,
    DiscoverRegimeChangeDriversCommandHandler,
    EnhanceTradingSignalsCommandHandler,
    AssessCorrelationBreakdownRiskCommandHandler
)
from causal_analysis_service.cqrs.handlers.query_handlers import (
    GetCausalGraphQueryHandler,
    GetInterventionEffectQueryHandler,
    GetCounterfactualScenarioQueryHandler,
    GetCurrencyPairRelationshipsQueryHandler,
    GetRegimeChangeDriversQueryHandler,
    GetCorrelationBreakdownRiskQueryHandler
)
from causal_analysis_service.cqrs.commands import (
    GenerateCausalGraphCommand,
    AnalyzeInterventionEffectCommand,
    GenerateCounterfactualScenarioCommand,
    DiscoverCurrencyPairRelationshipsCommand,
    DiscoverRegimeChangeDriversCommand,
    EnhanceTradingSignalsCommand,
    AssessCorrelationBreakdownRiskCommand
)
from causal_analysis_service.cqrs.queries import (
    GetCausalGraphQuery,
    GetInterventionEffectQuery,
    GetCounterfactualScenarioQuery,
    GetCurrencyPairRelationshipsQuery,
    GetRegimeChangeDriversQuery,
    GetCorrelationBreakdownRiskQuery
)
from causal_analysis_service.repositories.read_repositories import (
    CausalGraphReadRepository,
    InterventionEffectReadRepository,
    CounterfactualReadRepository,
    CurrencyPairRelationshipReadRepository,
    RegimeChangeDriverReadRepository,
    CorrelationBreakdownRiskReadRepository
)
from causal_analysis_service.repositories.write_repositories import (
    CausalGraphWriteRepository,
    InterventionEffectWriteRepository,
    CounterfactualWriteRepository,
    CurrencyPairRelationshipWriteRepository,
    RegimeChangeDriverWriteRepository,
    CorrelationBreakdownRiskWriteRepository
)
from causal_analysis_service.services.causal_service import CausalService

logger = logging.getLogger(__name__)

# Singleton instances
_command_bus: Optional[CommandBus] = None
_query_bus: Optional[QueryBus] = None


def get_command_bus() -> CommandBus:
    """
    Get the command bus.
    
    Returns:
        The command bus
    """
    global _command_bus
    
    if _command_bus is None:
        _command_bus = CommandBus()
        
        # Create repositories
        causal_graph_write_repository = CausalGraphWriteRepository()
        intervention_effect_write_repository = InterventionEffectWriteRepository()
        counterfactual_write_repository = CounterfactualWriteRepository()
        currency_pair_relationship_write_repository = CurrencyPairRelationshipWriteRepository()
        regime_change_driver_write_repository = RegimeChangeDriverWriteRepository()
        correlation_breakdown_risk_write_repository = CorrelationBreakdownRiskWriteRepository()
        
        # Create service
        causal_service = CausalService()
        
        # Create command handlers
        generate_causal_graph_handler = GenerateCausalGraphCommandHandler(
            causal_service=causal_service,
            repository=causal_graph_write_repository
        )
        analyze_intervention_effect_handler = AnalyzeInterventionEffectCommandHandler(
            causal_service=causal_service,
            repository=intervention_effect_write_repository
        )
        generate_counterfactual_scenario_handler = GenerateCounterfactualScenarioCommandHandler(
            causal_service=causal_service,
            repository=counterfactual_write_repository
        )
        discover_currency_pair_relationships_handler = DiscoverCurrencyPairRelationshipsCommandHandler(
            causal_service=causal_service,
            repository=currency_pair_relationship_write_repository
        )
        discover_regime_change_drivers_handler = DiscoverRegimeChangeDriversCommandHandler(
            causal_service=causal_service,
            repository=regime_change_driver_write_repository
        )
        enhance_trading_signals_handler = EnhanceTradingSignalsCommandHandler(
            causal_service=causal_service
        )
        assess_correlation_breakdown_risk_handler = AssessCorrelationBreakdownRiskCommandHandler(
            causal_service=causal_service,
            repository=correlation_breakdown_risk_write_repository
        )
        
        # Register command handlers
        _command_bus.register_handler(GenerateCausalGraphCommand, generate_causal_graph_handler)
        _command_bus.register_handler(AnalyzeInterventionEffectCommand, analyze_intervention_effect_handler)
        _command_bus.register_handler(GenerateCounterfactualScenarioCommand, generate_counterfactual_scenario_handler)
        _command_bus.register_handler(DiscoverCurrencyPairRelationshipsCommand, discover_currency_pair_relationships_handler)
        _command_bus.register_handler(DiscoverRegimeChangeDriversCommand, discover_regime_change_drivers_handler)
        _command_bus.register_handler(EnhanceTradingSignalsCommand, enhance_trading_signals_handler)
        _command_bus.register_handler(AssessCorrelationBreakdownRiskCommand, assess_correlation_breakdown_risk_handler)
    
    return _command_bus


def get_query_bus() -> QueryBus:
    """
    Get the query bus.
    
    Returns:
        The query bus
    """
    global _query_bus
    
    if _query_bus is None:
        _query_bus = QueryBus()
        
        # Create repositories
        causal_graph_read_repository = CausalGraphReadRepository()
        intervention_effect_read_repository = InterventionEffectReadRepository()
        counterfactual_read_repository = CounterfactualReadRepository()
        currency_pair_relationship_read_repository = CurrencyPairRelationshipReadRepository()
        regime_change_driver_read_repository = RegimeChangeDriverReadRepository()
        correlation_breakdown_risk_read_repository = CorrelationBreakdownRiskReadRepository()
        
        # Create query handlers
        get_causal_graph_handler = GetCausalGraphQueryHandler(
            repository=causal_graph_read_repository
        )
        get_intervention_effect_handler = GetInterventionEffectQueryHandler(
            repository=intervention_effect_read_repository
        )
        get_counterfactual_scenario_handler = GetCounterfactualScenarioQueryHandler(
            repository=counterfactual_read_repository
        )
        get_currency_pair_relationships_handler = GetCurrencyPairRelationshipsQueryHandler(
            repository=currency_pair_relationship_read_repository
        )
        get_regime_change_drivers_handler = GetRegimeChangeDriversQueryHandler(
            repository=regime_change_driver_read_repository
        )
        get_correlation_breakdown_risk_handler = GetCorrelationBreakdownRiskQueryHandler(
            repository=correlation_breakdown_risk_read_repository
        )
        
        # Register query handlers
        _query_bus.register_handler(GetCausalGraphQuery, get_causal_graph_handler)
        _query_bus.register_handler(GetInterventionEffectQuery, get_intervention_effect_handler)
        _query_bus.register_handler(GetCounterfactualScenarioQuery, get_counterfactual_scenario_handler)
        _query_bus.register_handler(GetCurrencyPairRelationshipsQuery, get_currency_pair_relationships_handler)
        _query_bus.register_handler(GetRegimeChangeDriversQuery, get_regime_change_drivers_handler)
        _query_bus.register_handler(GetCorrelationBreakdownRiskQuery, get_correlation_breakdown_risk_handler)
    
    return _query_bus