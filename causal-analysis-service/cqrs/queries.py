"""
Query models for the Causal Analysis Service.

This module provides the query models for the Causal Analysis Service.
"""
from typing import Optional
from pydantic import BaseModel

from common_lib.cqrs.queries import Query


class GetCausalGraphQuery(Query):
    """Query to get a causal graph."""
    graph_id: str


class GetInterventionEffectQuery(Query):
    """Query to get an intervention effect."""
    effect_id: str


class GetCounterfactualScenarioQuery(Query):
    """Query to get a counterfactual scenario."""
    counterfactual_id: str


class GetCurrencyPairRelationshipsQuery(Query):
    """Query to get currency pair relationships."""
    relationship_id: str


class GetRegimeChangeDriversQuery(Query):
    """Query to get regime change drivers."""
    driver_id: str


class GetCorrelationBreakdownRiskQuery(Query):
    """Query to get correlation breakdown risk."""
    risk_id: str