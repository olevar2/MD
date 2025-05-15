"""
Command handlers for the Causal Analysis Service.

This module provides the command handlers for the Causal Analysis Service.
"""
import logging
from typing import Dict, Any

from common_lib.cqrs.commands import CommandHandler
from causal_analysis_service.cqrs.commands import (
    GenerateCausalGraphCommand,
    AnalyzeInterventionEffectCommand,
    GenerateCounterfactualScenarioCommand,
    DiscoverCurrencyPairRelationshipsCommand,
    DiscoverRegimeChangeDriversCommand,
    EnhanceTradingSignalsCommand,
    AssessCorrelationBreakdownRiskCommand
)
from causal_analysis_service.models.causal_models import (
    CausalGraphResponse,
    InterventionEffectResponse,
    CounterfactualResponse,
    CurrencyPairRelationshipResponse,
    RegimeChangeDriverResponse,
    TradingSignalEnhancementResponse,
    CorrelationBreakdownRiskResponse
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


class GenerateCausalGraphCommandHandler(CommandHandler[GenerateCausalGraphCommand, str]):
    """Handler for GenerateCausalGraphCommand."""
    
    def __init__(self, causal_service: CausalService, repository: CausalGraphWriteRepository):
        """
        Initialize the handler.
        
        Args:
            causal_service: The causal service
            repository: The repository
        """
        self.causal_service = causal_service
        self.repository = repository
    
    async def handle(self, command: GenerateCausalGraphCommand) -> str:
        """
        Handle the command.
        
        Args:
            command: The command
            
        Returns:
            The ID of the generated causal graph
        """
        logger.info(f"Handling GenerateCausalGraphCommand: {command}")
        
        # Fetch market data and generate causal graph
        data = await self.causal_service._fetch_market_data(
            symbol=command.symbol,
            timeframe=command.timeframe,
            start_date=command.start_date,
            end_date=command.end_date
        )
        
        # Select algorithm
        algorithm = self.causal_service._select_algorithm(command.algorithm)
        
        # Set algorithm parameters
        if command.parameters:
            algorithm.config.update(command.parameters)
        
        # Discover causal relationships
        causal_graph = algorithm.discover_causal_relationships(data)
        
        # Save causal graph
        metadata = {
            'algorithm': command.algorithm,
            'parameters': command.parameters or {},
            'symbol': command.symbol,
            'timeframe': command.timeframe,
            'start_date': command.start_date,
            'end_date': command.end_date
        }
        
        graph_id = await self.repository.add(causal_graph, metadata)
        
        return graph_id


class AnalyzeInterventionEffectCommandHandler(CommandHandler[AnalyzeInterventionEffectCommand, str]):
    """Handler for AnalyzeInterventionEffectCommand."""
    
    def __init__(self, causal_service: CausalService, repository: InterventionEffectWriteRepository):
        """
        Initialize the handler.
        
        Args:
            causal_service: The causal service
            repository: The repository
        """
        self.causal_service = causal_service
        self.repository = repository
    
    async def handle(self, command: AnalyzeInterventionEffectCommand) -> str:
        """
        Handle the command.
        
        Args:
            command: The command
            
        Returns:
            The ID of the analyzed intervention effect
        """
        logger.info(f"Handling AnalyzeInterventionEffectCommand: {command}")
        
        # Fetch market data
        data = await self.causal_service._fetch_market_data(
            symbol=command.symbol,
            timeframe=command.timeframe,
            start_date=command.start_date,
            end_date=command.end_date
        )
        
        # Estimate causal effect
        effect_data = self.causal_service.dowhy_algorithm.estimate_causal_effect(
            data=data,
            treatment=command.treatment,
            outcome=command.outcome,
            confounders=command.confounders
        )
        
        # Save intervention effect
        metadata = {
            'algorithm': command.algorithm,
            'parameters': command.parameters or {},
            'symbol': command.symbol,
            'timeframe': command.timeframe,
            'start_date': command.start_date,
            'end_date': command.end_date,
            'treatment': command.treatment,
            'outcome': command.outcome
        }
        
        effect_id = await self.repository.add(effect_data, metadata)
        
        return effect_id


class GenerateCounterfactualScenarioCommandHandler(CommandHandler[GenerateCounterfactualScenarioCommand, str]):
    """Handler for GenerateCounterfactualScenarioCommand."""
    
    def __init__(self, causal_service: CausalService, repository: CounterfactualWriteRepository):
        """
        Initialize the handler.
        
        Args:
            causal_service: The causal service
            repository: The repository
        """
        self.causal_service = causal_service
        self.repository = repository
    
    async def handle(self, command: GenerateCounterfactualScenarioCommand) -> str:
        """
        Handle the command.
        
        Args:
            command: The command
            
        Returns:
            The ID of the generated counterfactual scenario
        """
        logger.info(f"Handling GenerateCounterfactualScenarioCommand: {command}")
        
        # Fetch market data
        data = await self.causal_service._fetch_market_data(
            symbol=command.symbol,
            timeframe=command.timeframe,
            start_date=command.start_date,
            end_date=command.end_date
        )
        
        # Generate counterfactual
        counterfactual_data = self.causal_service.counterfactual_algorithm.generate_counterfactual(
            data=data,
            intervention=command.intervention,
            target_variables=command.target_variables
        )
        
        # Extract counterfactual values
        counterfactual_values = {}
        for target in command.target_variables:
            if target in counterfactual_data.columns:
                counterfactual_values[target] = counterfactual_data[target].tolist()
        
        # Save counterfactual
        metadata = {
            'algorithm': command.algorithm,
            'parameters': command.parameters or {},
            'symbol': command.symbol,
            'timeframe': command.timeframe,
            'start_date': command.start_date,
            'end_date': command.end_date,
            'intervention': command.intervention,
            'target_variables': command.target_variables
        }
        
        counterfactual_id = await self.repository.add(
            {'counterfactual_values': counterfactual_values},
            metadata
        )
        
        return counterfactual_id


class DiscoverCurrencyPairRelationshipsCommandHandler(CommandHandler[DiscoverCurrencyPairRelationshipsCommand, str]):
    """Handler for DiscoverCurrencyPairRelationshipsCommand."""
    
    def __init__(self, causal_service: CausalService, repository: CurrencyPairRelationshipWriteRepository):
        """
        Initialize the handler.
        
        Args:
            causal_service: The causal service
            repository: The repository
        """
        self.causal_service = causal_service
        self.repository = repository
    
    async def handle(self, command: DiscoverCurrencyPairRelationshipsCommand) -> str:
        """
        Handle the command.
        
        Args:
            command: The command
            
        Returns:
            The ID of the discovered currency pair relationships
        """
        logger.info(f"Handling DiscoverCurrencyPairRelationshipsCommand: {command}")
        
        # Fetch market data for all symbols
        import pandas as pd
        all_data = {}
        for symbol in command.symbols:
            data = await self.causal_service._fetch_market_data(
                symbol=symbol,
                timeframe=command.timeframe,
                start_date=command.start_date,
                end_date=command.end_date
            )
            
            # Extract relevant variables
            if command.variables:
                data = data[command.variables]
            else:
                # Use close price by default
                data = data[['close']]
            
            # Rename columns to include symbol
            data = data.rename(columns={col: f"{symbol}_{col}" for col in data.columns})
            
            all_data[symbol] = data
        
        # Combine data
        combined_data = pd.concat(all_data.values(), axis=1)
        
        # Select algorithm
        algorithm = self.causal_service._select_algorithm(command.algorithm)
        
        # Set algorithm parameters
        if command.parameters:
            algorithm.config.update(command.parameters)
        
        # Discover causal relationships
        causal_graph = algorithm.discover_causal_relationships(combined_data)
        
        # Save currency pair relationship
        metadata = {
            'algorithm': command.algorithm,
            'parameters': command.parameters or {},
            'symbols': command.symbols,
            'timeframe': command.timeframe,
            'start_date': command.start_date,
            'end_date': command.end_date,
            'variables': command.variables
        }
        
        relationship_id = await self.repository.add(causal_graph, metadata)
        
        return relationship_id


class DiscoverRegimeChangeDriversCommandHandler(CommandHandler[DiscoverRegimeChangeDriversCommand, str]):
    """Handler for DiscoverRegimeChangeDriversCommand."""
    
    def __init__(self, causal_service: CausalService, repository: RegimeChangeDriverWriteRepository):
        """
        Initialize the handler.
        
        Args:
            causal_service: The causal service
            repository: The repository
        """
        self.causal_service = causal_service
        self.repository = repository
    
    async def handle(self, command: DiscoverRegimeChangeDriversCommand) -> str:
        """
        Handle the command.
        
        Args:
            command: The command
            
        Returns:
            The ID of the discovered regime change drivers
        """
        logger.info(f"Handling DiscoverRegimeChangeDriversCommand: {command}")
        
        # Fetch market data
        data = await self.causal_service._fetch_market_data(
            symbol=command.symbol,
            timeframe=command.timeframe,
            start_date=command.start_date,
            end_date=command.end_date
        )
        
        # Ensure regime variable exists
        if command.regime_variable not in data.columns:
            # If not, we need to compute it
            # This is a placeholder for actual regime detection
            from sklearn.cluster import KMeans
            
            # Use price and volatility features for regime detection
            features = data[['close', 'high', 'low']].copy()
            features['volatility'] = (data['high'] - data['low']) / data['close']
            features['returns'] = data['close'].pct_change().fillna(0)
            
            # Normalize features
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features)
            
            # Cluster into regimes
            kmeans = KMeans(n_clusters=3, random_state=42)
            data[command.regime_variable] = kmeans.fit_predict(features_scaled)
        
        # Select potential drivers
        if command.potential_drivers:
            driver_data = data[[command.regime_variable] + command.potential_drivers]
        else:
            # Use all numeric columns as potential drivers
            numeric_columns = data.select_dtypes(include=['number']).columns
            driver_data = data[[command.regime_variable] + [col for col in numeric_columns if col != command.regime_variable]]
        
        # Use DoWhy for causal discovery
        drivers = []
        
        for col in driver_data.columns:
            if col == command.regime_variable:
                continue
            
            try:
                effect = self.causal_service.dowhy_algorithm.estimate_causal_effect(
                    data=driver_data,
                    treatment=col,
                    outcome=command.regime_variable
                )
                
                drivers.append({
                    'variable': col,
                    'effect_size': effect['causal_effect'],
                    'p_value': effect.get('p_value'),
                    'confidence_interval': effect.get('confidence_interval')
                })
            except Exception as e:
                logger.warning(f"Error estimating causal effect for {col}: {e}")
        
        # Sort drivers by absolute effect size
        drivers.sort(key=lambda x: abs(x['effect_size']), reverse=True)
        
        # Save regime change drivers
        metadata = {
            'algorithm': command.algorithm,
            'parameters': command.parameters or {},
            'symbol': command.symbol,
            'timeframe': command.timeframe,
            'start_date': command.start_date,
            'end_date': command.end_date,
            'regime_variable': command.regime_variable
        }
        
        driver_id = await self.repository.add({'drivers': drivers}, metadata)
        
        return driver_id


class EnhanceTradingSignalsCommandHandler(CommandHandler[EnhanceTradingSignalsCommand, TradingSignalEnhancementResponse]):
    """Handler for EnhanceTradingSignalsCommand."""
    
    def __init__(self, causal_service: CausalService):
        """
        Initialize the handler.
        
        Args:
            causal_service: The causal service
        """
        self.causal_service = causal_service
    
    async def handle(self, command: EnhanceTradingSignalsCommand) -> TradingSignalEnhancementResponse:
        """
        Handle the command.
        
        Args:
            command: The command
            
        Returns:
            The enhanced trading signals
        """
        logger.info(f"Handling EnhanceTradingSignalsCommand")
        
        # Convert market data to DataFrame
        import pandas as pd
        market_data = pd.DataFrame(command.market_data)
        
        # Discover causal relationships
        causal_graph = self.causal_service.granger_algorithm.discover_causal_relationships(market_data)
        
        # Enhance trading signals
        enhanced_signals = []
        
        for signal in command.signals:
            enhanced_signal = signal.copy()
            
            # Add causal factors
            causal_factors = []
            for node in causal_graph.nodes():
                if causal_graph.has_edge(node, 'price'):
                    causal_factors.append({
                        'factor': node,
                        'effect': causal_graph[node]['price']['weight']
                    })
            
            enhanced_signal['causal_factors'] = causal_factors
            
            # Add confidence adjustment based on causal factors
            confidence_adjustment = 0.0
            for factor in causal_factors:
                if factor['effect'] > 0.5:
                    confidence_adjustment += 0.1
                elif factor['effect'] > 0.3:
                    confidence_adjustment += 0.05
            
            enhanced_signal['confidence'] = min(1.0, (enhanced_signal.get('confidence', 0.5) + confidence_adjustment))
            
            # Add expected duration based on causal analysis
            # This is a placeholder for actual duration estimation
            enhanced_signal['expected_duration'] = '4h'
            
            # Add conflicting signals
            conflicting_signals = []
            for factor in causal_factors:
                if factor['effect'] < -0.3:
                    conflicting_signals.append(factor['factor'])
            
            enhanced_signal['conflicting_signals'] = conflicting_signals
            
            enhanced_signals.append(enhanced_signal)
        
        return TradingSignalEnhancementResponse(
            enhanced_signals=enhanced_signals,
            count=len(enhanced_signals),
            causal_factors_considered=['volatility', 'trend', 'sentiment', 'correlations']
        )


class AssessCorrelationBreakdownRiskCommandHandler(CommandHandler[AssessCorrelationBreakdownRiskCommand, str]):
    """Handler for AssessCorrelationBreakdownRiskCommand."""
    
    def __init__(self, causal_service: CausalService, repository: CorrelationBreakdownRiskWriteRepository):
        """
        Initialize the handler.
        
        Args:
            causal_service: The causal service
            repository: The repository
        """
        self.causal_service = causal_service
        self.repository = repository
    
    async def handle(self, command: AssessCorrelationBreakdownRiskCommand) -> str:
        """
        Handle the command.
        
        Args:
            command: The command
            
        Returns:
            The ID of the assessed correlation breakdown risk
        """
        logger.info(f"Handling AssessCorrelationBreakdownRiskCommand: {command}")
        
        # Fetch market data for all symbols
        import pandas as pd
        all_data = {}
        for symbol in command.symbols:
            data = await self.causal_service._fetch_market_data(
                symbol=symbol,
                timeframe=command.timeframe,
                start_date=command.start_date,
                end_date=command.end_date
            )
            
            # Use close price
            all_data[symbol] = data['close']
        
        # Combine data
        combined_data = pd.DataFrame(all_data)
        
        # Calculate baseline correlations
        baseline_correlations = {}
        correlation_matrix = combined_data.corr()
        
        for symbol1 in command.symbols:
            baseline_correlations[symbol1] = {}
            for symbol2 in command.symbols:
                if symbol1 != symbol2:
                    baseline_correlations[symbol1][symbol2] = correlation_matrix.loc[symbol1, symbol2]
        
        # Calculate stress correlations
        stress_correlations = {}
        breakdown_risk_scores = {}
        
        if command.stress_scenarios:
            for i, scenario in enumerate(command.stress_scenarios):
                scenario_name = scenario.get('name', f"scenario_{i}")
                stress_data = combined_data.copy()
                
                # Apply stress scenario
                for symbol, change in scenario.get('changes', {}).items():
                    if symbol in stress_data.columns:
                        stress_data[symbol] = stress_data[symbol] * (1 + change)
                
                # Calculate correlations under stress
                stress_correlation_matrix = stress_data.corr()
                
                stress_correlations[scenario_name] = {}
                breakdown_risk_scores[scenario_name] = {}
                
                for symbol1 in command.symbols:
                    stress_correlations[scenario_name][symbol1] = {}
                    breakdown_risk_scores[scenario_name][symbol1] = {}
                    
                    for symbol2 in command.symbols:
                        if symbol1 != symbol2:
                            stress_corr = stress_correlation_matrix.loc[symbol1, symbol2]
                            baseline_corr = baseline_correlations[symbol1][symbol2]
                            
                            stress_correlations[scenario_name][symbol1][symbol2] = stress_corr
                            
                            # Calculate breakdown risk score
                            breakdown_risk = abs(baseline_corr - stress_corr) / (abs(baseline_corr) + 0.01)
                            breakdown_risk_scores[scenario_name][symbol1][symbol2] = breakdown_risk
        
        # Save correlation breakdown risk
        risk_data = {
            'symbols': command.symbols,
            'baseline_correlations': baseline_correlations,
            'stress_correlations': stress_correlations,
            'breakdown_risk_scores': breakdown_risk_scores
        }
        
        metadata = {
            'algorithm': command.algorithm,
            'parameters': command.parameters or {},
            'symbols': command.symbols,
            'timeframe': command.timeframe,
            'start_date': command.start_date,
            'end_date': command.end_date
        }
        
        risk_id = await self.repository.add(risk_data, metadata)
        
        return risk_id