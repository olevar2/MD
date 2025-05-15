"""
Query handlers for the Causal Analysis Service.

This module provides the query handlers for the Causal Analysis Service.
"""
import logging
from typing import Optional

from common_lib.cqrs.queries import QueryHandler
from causal_analysis_service.cqrs.queries import (
    GetCausalGraphQuery,
    GetInterventionEffectQuery,
    GetCounterfactualScenarioQuery,
    GetCurrencyPairRelationshipsQuery,
    GetRegimeChangeDriversQuery,
    GetCorrelationBreakdownRiskQuery
)
from causal_analysis_service.models.causal_models import (
    CausalGraphResponse,
    InterventionEffectResponse,
    CounterfactualResponse,
    CurrencyPairRelationshipResponse,
    RegimeChangeDriverResponse,
    CorrelationBreakdownRiskResponse
)
from causal_analysis_service.repositories.read_repositories import (
    CausalGraphReadRepository,
    InterventionEffectReadRepository,
    CounterfactualReadRepository,
    CurrencyPairRelationshipReadRepository,
    RegimeChangeDriverReadRepository,
    CorrelationBreakdownRiskReadRepository
)

logger = logging.getLogger(__name__)


class GetCausalGraphQueryHandler(QueryHandler[GetCausalGraphQuery, Optional[CausalGraphResponse]]):
    """Handler for GetCausalGraphQuery."""
    
    def __init__(self, repository: CausalGraphReadRepository):
        """
        Initialize the handler.
        
        Args:
            repository: The repository
        """
        self.repository = repository
    
    async def handle(self, query: GetCausalGraphQuery) -> Optional[CausalGraphResponse]:
        """
        Handle the query.
        
        Args:
            query: The query
            
        Returns:
            The causal graph response or None if not found
        """
        logger.info(f"Handling GetCausalGraphQuery: {query}")
        
        return await self.repository.get_by_id(query.graph_id)


class GetInterventionEffectQueryHandler(QueryHandler[GetInterventionEffectQuery, Optional[InterventionEffectResponse]]):
    """Handler for GetInterventionEffectQuery."""
    
    def __init__(self, repository: InterventionEffectReadRepository):
        """
        Initialize the handler.
        
        Args:
            repository: The repository
        """
        self.repository = repository
    
    async def handle(self, query: GetInterventionEffectQuery) -> Optional[InterventionEffectResponse]:
        """
        Handle the query.
        
        Args:
            query: The query
            
        Returns:
            The intervention effect response or None if not found
        """
        logger.info(f"Handling GetInterventionEffectQuery: {query}")
        
        return await self.repository.get_by_id(query.effect_id)


class GetCounterfactualScenarioQueryHandler(QueryHandler[GetCounterfactualScenarioQuery, Optional[CounterfactualResponse]]):
    """Handler for GetCounterfactualScenarioQuery."""
    
    def __init__(self, repository: CounterfactualReadRepository):
        """
        Initialize the handler.
        
        Args:
            repository: The repository
        """
        self.repository = repository
    
    async def handle(self, query: GetCounterfactualScenarioQuery) -> Optional[CounterfactualResponse]:
        """
        Handle the query.
        
        Args:
            query: The query
            
        Returns:
            The counterfactual response or None if not found
        """
        logger.info(f"Handling GetCounterfactualScenarioQuery: {query}")
        
        return await self.repository.get_by_id(query.counterfactual_id)


class GetCurrencyPairRelationshipsQueryHandler(QueryHandler[GetCurrencyPairRelationshipsQuery, Optional[CurrencyPairRelationshipResponse]]):
    """Handler for GetCurrencyPairRelationshipsQuery."""
    
    def __init__(self, repository: CurrencyPairRelationshipReadRepository):
        """
        Initialize the handler.
        
        Args:
            repository: The repository
        """
        self.repository = repository
    
    async def handle(self, query: GetCurrencyPairRelationshipsQuery) -> Optional[CurrencyPairRelationshipResponse]:
        """
        Handle the query.
        
        Args:
            query: The query
            
        Returns:
            The currency pair relationship response or None if not found
        """
        logger.info(f"Handling GetCurrencyPairRelationshipsQuery: {query}")
        
        return await self.repository.get_by_id(query.relationship_id)


class GetRegimeChangeDriversQueryHandler(QueryHandler[GetRegimeChangeDriversQuery, Optional[RegimeChangeDriverResponse]]):
    """Handler for GetRegimeChangeDriversQuery."""
    
    def __init__(self, repository: RegimeChangeDriverReadRepository):
        """
        Initialize the handler.
        
        Args:
            repository: The repository
        """
        self.repository = repository
    
    async def handle(self, query: GetRegimeChangeDriversQuery) -> Optional[RegimeChangeDriverResponse]:
        """
        Handle the query.
        
        Args:
            query: The query
            
        Returns:
            The regime change driver response or None if not found
        """
        logger.info(f"Handling GetRegimeChangeDriversQuery: {query}")
        
        return await self.repository.get_by_id(query.driver_id)


class GetCorrelationBreakdownRiskQueryHandler(QueryHandler[GetCorrelationBreakdownRiskQuery, Optional[CorrelationBreakdownRiskResponse]]):
    """Handler for GetCorrelationBreakdownRiskQuery."""
    
    def __init__(self, repository: CorrelationBreakdownRiskReadRepository):
        """
        Initialize the handler.
        
        Args:
            repository: The repository
        """
        self.repository = repository
    
    async def handle(self, query: GetCorrelationBreakdownRiskQuery) -> Optional[CorrelationBreakdownRiskResponse]:
        """
        Handle the query.
        
        Args:
            query: The query
            
        Returns:
            The correlation breakdown risk response or None if not found
        """
        logger.info(f"Handling GetCorrelationBreakdownRiskQuery: {query}")
        
        return await self.repository.get_by_id(query.risk_id)