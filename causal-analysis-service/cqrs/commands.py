"""
Command models for the Causal Analysis Service.

This module provides the command models for the Causal Analysis Service.
"""
from datetime import datetime
from typing import Dict, List, Optional, Any
from pydantic import BaseModel

from common_lib.cqrs.commands import Command


class GenerateCausalGraphCommand(Command):
    """Command to generate a causal graph."""
    symbol: str
    timeframe: str
    start_date: datetime
    end_date: Optional[datetime] = None
    algorithm: str = "granger"
    parameters: Optional[Dict[str, Any]] = None


class AnalyzeInterventionEffectCommand(Command):
    """Command to analyze intervention effect."""
    symbol: str
    timeframe: str
    start_date: datetime
    end_date: Optional[datetime] = None
    treatment: str
    outcome: str
    confounders: Optional[List[str]] = None
    algorithm: str = "dowhy"
    parameters: Optional[Dict[str, Any]] = None


class GenerateCounterfactualScenarioCommand(Command):
    """Command to generate counterfactual scenario."""
    symbol: str
    timeframe: str
    start_date: datetime
    end_date: Optional[datetime] = None
    intervention: Dict[str, Any]
    target_variables: List[str]
    algorithm: str = "counterfactual"
    parameters: Optional[Dict[str, Any]] = None


class DiscoverCurrencyPairRelationshipsCommand(Command):
    """Command to discover currency pair relationships."""
    symbols: List[str]
    timeframe: str
    start_date: datetime
    end_date: Optional[datetime] = None
    variables: Optional[List[str]] = None
    algorithm: str = "granger"
    parameters: Optional[Dict[str, Any]] = None


class DiscoverRegimeChangeDriversCommand(Command):
    """Command to discover regime change drivers."""
    symbol: str
    timeframe: str
    start_date: datetime
    end_date: Optional[datetime] = None
    regime_variable: str
    potential_drivers: Optional[List[str]] = None
    algorithm: str = "dowhy"
    parameters: Optional[Dict[str, Any]] = None


class EnhanceTradingSignalsCommand(Command):
    """Command to enhance trading signals."""
    market_data: Dict[str, Any]
    signals: List[Dict[str, Any]]
    parameters: Optional[Dict[str, Any]] = None


class AssessCorrelationBreakdownRiskCommand(Command):
    """Command to assess correlation breakdown risk."""
    symbols: List[str]
    timeframe: str
    start_date: datetime
    end_date: Optional[datetime] = None
    stress_scenarios: Optional[List[Dict[str, Any]]] = None
    algorithm: str = "correlation"
    parameters: Optional[Dict[str, Any]] = None