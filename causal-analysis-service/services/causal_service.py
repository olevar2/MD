"""
Causal Service

This module provides the service layer for causal analysis.
"""
import logging
import pandas as pd
import networkx as nx
from datetime import datetime
from typing import Dict, List, Optional, Any, Union

from causal_analysis_service.core.algorithms import (
    GrangerCausalityAlgorithm,
    PCAlgorithm,
    DoWhyAlgorithm,
    CounterfactualAnalysisAlgorithm
)
from causal_analysis_service.repositories.causal_repository import CausalRepository
from causal_analysis_service.utils.validation import (
    validate_causal_graph_request,
    validate_intervention_effect_request,
    validate_counterfactual_request
)
from causal_analysis_service.models.causal_models import (
    CausalGraphRequest,
    CausalGraphResponse,
    InterventionEffectRequest,
    InterventionEffectResponse,
    CounterfactualRequest,
    CounterfactualResponse,
    CurrencyPairRelationshipRequest,
    CurrencyPairRelationshipResponse,
    RegimeChangeDriverRequest,
    RegimeChangeDriverResponse,
    TradingSignalEnhancementRequest,
    TradingSignalEnhancementResponse,
    CorrelationBreakdownRiskRequest,
    CorrelationBreakdownRiskResponse
)

logger = logging.getLogger(__name__)

class CausalService:
    """
    Service for causal analysis.
    """
    def __init__(self, repository: Optional[CausalRepository] = None, data_client=None):
        """
        Initialize the causal service.
        
        Args:
            repository: Repository for storing and retrieving causal analysis results
            data_client: Client for retrieving market data
        """
        self.repository = repository or CausalRepository()
        self.data_client = data_client
        
        # Initialize algorithms
        self.granger_algorithm = GrangerCausalityAlgorithm()
        self.pc_algorithm = PCAlgorithm()
        self.dowhy_algorithm = DoWhyAlgorithm()
        self.counterfactual_algorithm = CounterfactualAnalysisAlgorithm()
    
    async def generate_causal_graph(self, request: CausalGraphRequest) -> CausalGraphResponse:
        """
        Generate a causal graph from market data.
        
        Args:
            request: Causal graph request
            
        Returns:
            Causal graph response
        """
        # Validate request
        validate_causal_graph_request(request.dict())
        
        # Fetch market data
        data = await self._fetch_market_data(
            symbol=request.symbol,
            timeframe=request.timeframe,
            start_date=request.start_date,
            end_date=request.end_date
        )
        
        # Select algorithm
        algorithm = self._select_algorithm(request.algorithm)
        
        # Set algorithm parameters
        if request.parameters:
            algorithm.config.update(request.parameters)
        
        # Discover causal relationships
        causal_graph = algorithm.discover_causal_relationships(data)
        
        # Save causal graph
        metadata = {
            'algorithm': request.algorithm,
            'parameters': request.parameters or {},
            'symbol': request.symbol,
            'timeframe': request.timeframe,
            'start_date': request.start_date,
            'end_date': request.end_date
        }
        
        graph_id = await self.repository.save_causal_graph(causal_graph, metadata)
        
        # Retrieve saved graph
        graph_response = await self.repository.get_causal_graph(graph_id)
        
        return graph_response
    
    async def analyze_intervention_effect(self, request: InterventionEffectRequest) -> InterventionEffectResponse:
        """
        Analyze the effect of an intervention on the system.
        
        Args:
            request: Intervention effect request
            
        Returns:
            Intervention effect response
        """
        # Validate request
        validate_intervention_effect_request(request.dict())
        
        # Fetch market data
        data = await self._fetch_market_data(
            symbol=request.symbol,
            timeframe=request.timeframe,
            start_date=request.start_date,
            end_date=request.end_date
        )
        
        # Estimate causal effect
        effect_data = self.dowhy_algorithm.estimate_causal_effect(
            data=data,
            treatment=request.treatment,
            outcome=request.outcome,
            confounders=request.confounders
        )
        
        # Save intervention effect
        metadata = {
            'algorithm': request.algorithm,
            'parameters': request.parameters or {},
            'symbol': request.symbol,
            'timeframe': request.timeframe,
            'start_date': request.start_date,
            'end_date': request.end_date,
            'treatment': request.treatment,
            'outcome': request.outcome
        }
        
        effect_id = await self.repository.save_intervention_effect(effect_data, metadata)
        
        # Retrieve saved effect
        effect_response = await self.repository.get_intervention_effect(effect_id)
        
        return effect_response
    
    async def generate_counterfactual_scenario(self, request: CounterfactualRequest) -> CounterfactualResponse:
        """
        Generate a counterfactual scenario based on the intervention.
        
        Args:
            request: Counterfactual request
            
        Returns:
            Counterfactual response
        """
        # Validate request
        validate_counterfactual_request(request.dict())
        
        # Fetch market data
        data = await self._fetch_market_data(
            symbol=request.symbol,
            timeframe=request.timeframe,
            start_date=request.start_date,
            end_date=request.end_date
        )
        
        # Generate counterfactual
        counterfactual_data = self.counterfactual_algorithm.generate_counterfactual(
            data=data,
            intervention=request.intervention,
            target_variables=request.target_variables
        )
        
        # Extract counterfactual values
        counterfactual_values = {}
        for target in request.target_variables:
            if target in counterfactual_data.columns:
                counterfactual_values[target] = counterfactual_data[target].tolist()
        
        # Save counterfactual
        metadata = {
            'algorithm': request.algorithm,
            'parameters': request.parameters or {},
            'symbol': request.symbol,
            'timeframe': request.timeframe,
            'start_date': request.start_date,
            'end_date': request.end_date,
            'intervention': request.intervention,
            'target_variables': request.target_variables
        }
        
        counterfactual_id = await self.repository.save_counterfactual(
            {'counterfactual_values': counterfactual_values},
            metadata
        )
        
        # Retrieve saved counterfactual
        counterfactual_response = await self.repository.get_counterfactual(counterfactual_id)
        
        return counterfactual_response
    
    async def discover_currency_pair_relationships(self, request: CurrencyPairRelationshipRequest) -> CurrencyPairRelationshipResponse:
        """
        Discover causal relationships between currency pairs.
        
        Args:
            request: Currency pair relationship request
            
        Returns:
            Currency pair relationship response
        """
        # Fetch market data for all symbols
        all_data = {}
        for symbol in request.symbols:
            data = await self._fetch_market_data(
                symbol=symbol,
                timeframe=request.timeframe,
                start_date=request.start_date,
                end_date=request.end_date
            )
            
            # Extract relevant variables
            if request.variables:
                data = data[request.variables]
            else:
                # Use close price by default
                data = data[['close']]
            
            # Rename columns to include symbol
            data = data.rename(columns={col: f"{symbol}_{col}" for col in data.columns})
            
            all_data[symbol] = data
        
        # Combine data
        combined_data = pd.concat(all_data.values(), axis=1)
        
        # Select algorithm
        algorithm = self._select_algorithm(request.algorithm)
        
        # Set algorithm parameters
        if request.parameters:
            algorithm.config.update(request.parameters)
        
        # Discover causal relationships
        causal_graph = algorithm.discover_causal_relationships(combined_data)
        
        # Save currency pair relationship
        metadata = {
            'algorithm': request.algorithm,
            'parameters': request.parameters or {},
            'symbols': request.symbols,
            'timeframe': request.timeframe,
            'start_date': request.start_date,
            'end_date': request.end_date,
            'variables': request.variables
        }
        
        relationship_id = await self.repository.save_currency_pair_relationship(causal_graph, metadata)
        
        # Retrieve saved relationship
        relationship_response = await self.repository.get_currency_pair_relationship(relationship_id)
        
        return relationship_response
    
    async def discover_regime_change_drivers(self, request: RegimeChangeDriverRequest) -> RegimeChangeDriverResponse:
        """
        Discover causal factors that drive market regime changes.
        
        Args:
            request: Regime change driver request
            
        Returns:
            Regime change driver response
        """
        # Fetch market data
        data = await self._fetch_market_data(
            symbol=request.symbol,
            timeframe=request.timeframe,
            start_date=request.start_date,
            end_date=request.end_date
        )
        
        # Ensure regime variable exists
        if request.regime_variable not in data.columns:
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
            data[request.regime_variable] = kmeans.fit_predict(features_scaled)
        
        # Select potential drivers
        if request.potential_drivers:
            driver_data = data[[request.regime_variable] + request.potential_drivers]
        else:
            # Use all numeric columns as potential drivers
            numeric_columns = data.select_dtypes(include=['number']).columns
            driver_data = data[[request.regime_variable] + [col for col in numeric_columns if col != request.regime_variable]]
        
        # Use DoWhy for causal discovery
        drivers = []
        
        for col in driver_data.columns:
            if col == request.regime_variable:
                continue
            
            try:
                effect = self.dowhy_algorithm.estimate_causal_effect(
                    data=driver_data,
                    treatment=col,
                    outcome=request.regime_variable
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
            'algorithm': request.algorithm,
            'parameters': request.parameters or {},
            'symbol': request.symbol,
            'timeframe': request.timeframe,
            'start_date': request.start_date,
            'end_date': request.end_date,
            'regime_variable': request.regime_variable
        }
        
        driver_id = await self.repository.save_regime_change_driver({'drivers': drivers}, metadata)
        
        # Retrieve saved drivers
        driver_response = await self.repository.get_regime_change_driver(driver_id)
        
        return driver_response
    
    async def enhance_trading_signals(self, request: TradingSignalEnhancementRequest) -> TradingSignalEnhancementResponse:
        """
        Enhance trading signals with causal insights.
        
        Args:
            request: Trading signal enhancement request
            
        Returns:
            Trading signal enhancement response
        """
        # Convert market data to DataFrame
        market_data = pd.DataFrame(request.market_data)
        
        # Discover causal relationships
        causal_graph = self.granger_algorithm.discover_causal_relationships(market_data)
        
        # Enhance trading signals
        enhanced_signals = []
        
        for signal in request.signals:
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
    
    async def assess_correlation_breakdown_risk(self, request: CorrelationBreakdownRiskRequest) -> CorrelationBreakdownRiskResponse:
        """
        Assess correlation breakdown risk between assets.
        
        Args:
            request: Correlation breakdown risk request
            
        Returns:
            Correlation breakdown risk response
        """
        # Fetch market data for all symbols
        all_data = {}
        for symbol in request.symbols:
            data = await self._fetch_market_data(
                symbol=symbol,
                timeframe=request.timeframe,
                start_date=request.start_date,
                end_date=request.end_date
            )
            
            # Use close price
            all_data[symbol] = data['close']
        
        # Combine data
        combined_data = pd.DataFrame(all_data)
        
        # Calculate baseline correlations
        baseline_correlations = {}
        correlation_matrix = combined_data.corr()
        
        for symbol1 in request.symbols:
            baseline_correlations[symbol1] = {}
            for symbol2 in request.symbols:
                if symbol1 != symbol2:
                    baseline_correlations[symbol1][symbol2] = correlation_matrix.loc[symbol1, symbol2]
        
        # Calculate stress correlations
        stress_correlations = {}
        breakdown_risk_scores = {}
        
        if request.stress_scenarios:
            for i, scenario in enumerate(request.stress_scenarios):
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
                
                for symbol1 in request.symbols:
                    stress_correlations[scenario_name][symbol1] = {}
                    breakdown_risk_scores[scenario_name][symbol1] = {}
                    
                    for symbol2 in request.symbols:
                        if symbol1 != symbol2:
                            stress_corr = stress_correlation_matrix.loc[symbol1, symbol2]
                            baseline_corr = baseline_correlations[symbol1][symbol2]
                            
                            stress_correlations[scenario_name][symbol1][symbol2] = stress_corr
                            
                            # Calculate breakdown risk score
                            breakdown_risk = abs(baseline_corr - stress_corr) / (abs(baseline_corr) + 0.01)
                            breakdown_risk_scores[scenario_name][symbol1][symbol2] = breakdown_risk
        
        # Generate a unique ID for the risk assessment
        import uuid
        risk_id = str(uuid.uuid4())
        
        return CorrelationBreakdownRiskResponse(
            risk_id=risk_id,
            symbols=request.symbols,
            baseline_correlations=baseline_correlations,
            stress_correlations=stress_correlations,
            breakdown_risk_scores=breakdown_risk_scores,
            created_at=datetime.now(),
            algorithm=request.algorithm,
            parameters=request.parameters or {}
        )
    
    async def _fetch_market_data(self, symbol: str, timeframe: str, 
                               start_date: datetime, end_date: Optional[datetime] = None) -> pd.DataFrame:
        """
        Fetch market data for a symbol.
        
        Args:
            symbol: The currency pair or symbol
            timeframe: The timeframe for the data
            start_date: The start date
            end_date: The end date
            
        Returns:
            DataFrame containing market data
        """
        if self.data_client:
            # Use data client to fetch data
            data = await self.data_client.get_market_data(
                symbol=symbol,
                timeframe=timeframe,
                start_date=start_date,
                end_date=end_date
            )
            return data
        else:
            # Generate mock data for testing
            return self._generate_mock_data(
                symbol=symbol,
                timeframe=timeframe,
                start_date=start_date,
                end_date=end_date
            )
    
    def _generate_mock_data(self, symbol: str, timeframe: str, 
                          start_date: datetime, end_date: Optional[datetime] = None) -> pd.DataFrame:
        """
        Generate mock market data for testing.
        
        Args:
            symbol: The currency pair or symbol
            timeframe: The timeframe for the data
            start_date: The start date
            end_date: The end date
            
        Returns:
            DataFrame containing mock market data
        """
        import numpy as np
        
        # Set end date if not provided
        if end_date is None:
            end_date = datetime.now()
        
        # Generate date range
        if timeframe == '1d':
            date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        elif timeframe == '1h':
            date_range = pd.date_range(start=start_date, end=end_date, freq='H')
        elif timeframe == '4h':
            date_range = pd.date_range(start=start_date, end=end_date, freq='4H')
        else:
            # Default to daily
            date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # Generate price data
        n = len(date_range)
        close = np.random.normal(loc=100, scale=1, size=n).cumsum() + 1000
        
        # Generate OHLCV data
        high = close * (1 + np.random.uniform(0, 0.01, size=n))
        low = close * (1 - np.random.uniform(0, 0.01, size=n))
        open_price = low + np.random.uniform(0, 1, size=n) * (high - low)
        volume = np.random.uniform(1000, 5000, size=n)
        
        # Generate additional features
        volatility = (high - low) / close
        trend = np.convolve(close, np.ones(5)/5, mode='same') - np.convolve(close, np.ones(20)/20, mode='same')
        momentum = np.concatenate([[0], np.diff(close)])
        
        # Create DataFrame
        data = pd.DataFrame({
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume,
            'volatility': volatility,
            'trend': trend,
            'momentum': momentum
        }, index=date_range)
        
        return data
    
    def _select_algorithm(self, algorithm_name: str):
        """
        Select a causal discovery algorithm.
        
        Args:
            algorithm_name: Name of the algorithm
            
        Returns:
            Causal discovery algorithm
        """
        if algorithm_name == 'granger':
            return self.granger_algorithm
        elif algorithm_name == 'pc':
            return self.pc_algorithm
        elif algorithm_name == 'dowhy':
            return self.dowhy_algorithm
        elif algorithm_name == 'counterfactual':
            return self.counterfactual_algorithm
        else:
            # Default to Granger causality
            logger.warning(f"Unknown algorithm: {algorithm_name}. Using Granger causality.")
            return self.granger_algorithm