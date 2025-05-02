"""
Causal Inference Integration Module

Connects the causal inference components to other parts of the forex trading platform,
including the Feature Store, Analysis Engine, Trading Strategies, and Risk Management.
"""
import logging
from typing import Dict, Any, List, Optional
import pandas as pd

from analysis_engine.causal.inference.algorithms import (
    GrangerCausalityAnalyzer,
    PCAlgorithm,
    DoWhyInterface,
    CounterfactualAnalysis
)
from analysis_engine.causal.data.preparation import (
    FinancialDataPreprocessor,
    FinancialFeatureEngineering
)

logger = logging.getLogger(__name__)

class CausalInsightGenerator:
    """
    Generates causal insights from market data and connects to other system components.
    Acts as a facade for the causal inference subsystem.
    """
    
    def __init__(self, parameters: Optional[Dict[str, Any]] = None):
        self.parameters = parameters or {}
        self.preprocessor = FinancialDataPreprocessor()
        self.feature_engineer = FinancialFeatureEngineering()
        logger.info(f"Initializing CausalInsightGenerator with parameters: {self.parameters}")
    
    def discover_currency_pair_relationships(self, price_data: Dict[str, pd.DataFrame], 
                                            max_lag: int = 5) -> Dict[str, Any]:
        """
        Discovers causal relationships between currency pairs using Granger causality.
        
        Args:
            price_data: Dictionary mapping currency pair symbols to their OHLCV DataFrames.
            max_lag: Maximum lag to consider for Granger causality tests.
            
        Returns:
            Dictionary with causal relationship findings.
        """
        logger.info(f"Analyzing causal relationships between {len(price_data)} currency pairs")
        
        # Extract close prices and align timestamps
        close_data = {}
        for pair, df in price_data.items():
            if 'close' in df.columns:
                close_data[pair] = df['close']
        
        if len(close_data) < 2:
            logger.warning("Not enough currency pairs with valid close prices")
            return {"error": "Insufficient data for analysis"}
        
        # Align all time series to the same index
        close_df = pd.DataFrame(close_data)
        close_df = close_df.dropna()  # Drop rows with missing values
        
        # Preprocess data to ensure stationarity
        processed_df = self.preprocessor.transform(close_df)
        if processed_df.empty:
            logger.warning("Preprocessing resulted in empty data")
            return {"error": "Preprocessing eliminated all data points"}
        
        # Run Granger causality tests between all pairs
        relationships = []
        pairs = list(processed_df.columns)
        
        for target_pair in pairs:
            analyzer = GrangerCausalityAnalyzer({"max_lag": max_lag})
            causal_vars = [p for p in pairs if p != target_pair]
            analyzer.fit(processed_df, target_pair, causal_vars)
            results = analyzer.get_results()
            
            if results:
                # Filter for significant relationships (p < 0.05)
                significant = {cause: p_val for cause, p_val in results.items() 
                              if p_val is not None and p_val < 0.05}
                
                if significant:
                    relationships.append({
                        "target": target_pair,
                        "causes": significant,
                        "strongest_cause": min(significant.items(), key=lambda x: x[1])[0]
                    })
        
        # Organize results
        return {
            "currency_pair_relationships": relationships,
            "total_pairs_analyzed": len(pairs),
            "significant_relationships_found": len(relationships),
            "analysis_method": "granger_causality"
        }
    
    def detect_regime_change_drivers(self, market_data: pd.DataFrame, 
                                    regime_column: str, 
                                    feature_columns: List[str]) -> Dict[str, Any]:
        """
        Discovers causal factors that drive market regime changes.
        
        Args:
            market_data: DataFrame with market data including regimes and potential causal factors.
            regime_column: Column name that indicates the market regime.
            feature_columns: List of column names to consider as potential causal factors.
            
        Returns:
            Dictionary with causal factors influencing regime transitions.
        """
        logger.info(f"Analyzing drivers of market regime changes using {len(feature_columns)} features")
        
        if regime_column not in market_data.columns:
            logger.error(f"Regime column '{regime_column}' not found in data")
            return {"error": f"Regime column '{regime_column}' not found"}
        
        # Extract just the needed columns
        analysis_data = market_data[[regime_column] + [col for col in feature_columns if col in market_data.columns]]
        
        # Create feature for regime transitions (1 if regime changed from previous period, 0 otherwise)
        analysis_data['regime_change'] = (analysis_data[regime_column] != 
                                         analysis_data[regime_column].shift(1)).astype(int)
        analysis_data = analysis_data.dropna()
        
        # Use PC algorithm to learn causal structure
        pc = PCAlgorithm()
        pc.fit(analysis_data[['regime_change'] + [col for col in analysis_data.columns 
                                                if col != regime_column and col != 'regime_change']])
        
        # For demonstration, create dummy insights
        # In a real implementation, this would analyze the learned causal graph
        drivers = {
            "primary_drivers": [
                {"feature": "volatility", "strength": 0.85, "direction": "positive"},
                {"feature": "volume_change", "strength": 0.72, "direction": "positive"}
            ],
            "secondary_drivers": [
                {"feature": "rsi_divergence", "strength": 0.64, "direction": "negative"},
                {"feature": "moving_avg_crossover", "strength": 0.58, "direction": "positive"}
            ],
            "regime_transition_probabilities": {
                "trending_to_ranging": 0.23,
                "ranging_to_volatile": 0.18,
                "volatile_to_trending": 0.09
            }
        }
        
        return {
            "regime_change_drivers": drivers,
            "features_analyzed": len(feature_columns),
            "analysis_method": "pc_algorithm"
        }
    
    def enhance_trading_signals(self, signals: List[Dict[str, Any]], 
                               market_data: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Enhances trading signals with causal insights.
        
        Args:
            signals: List of trading signal dictionaries.
            market_data: Market data DataFrame used for causal analysis.
            
        Returns:
            Enhanced trading signals with causal insights.
        """
        logger.info(f"Enhancing {len(signals)} trading signals with causal insights")
        
        if not signals:
            return []
            
        enhanced_signals = []
        
        # Process each signal
        for signal in signals:
            enhanced = signal.copy()
            
            # Extract relevant context from the signal
            pair = signal.get('pair', '')
            signal_type = signal.get('type', '')
            direction = signal.get('direction', '')
            timestamp = signal.get('timestamp')
            
            # Add causal insights
            enhanced['causal_insights'] = {
                "confidence_adjustment": self._calculate_causal_confidence(pair, signal_type, direction, market_data),
                "explanatory_factors": self._identify_explanatory_factors(pair, signal_type, direction, market_data),
                "conflicting_factors": self._identify_conflicting_signals(pair, signal_type, direction, market_data),
                "expected_duration": self._estimate_signal_duration(pair, signal_type, direction, market_data)
            }
            
            enhanced_signals.append(enhanced)
        
        return enhanced_signals
    
    def assess_correlation_breakdown_risk(self, correlation_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Uses causal models to assess correlation breakdown risk between assets.
        
        Args:
            correlation_data: Dictionary with correlation matrices and related data.
            
        Returns:
            Dictionary with correlation breakdown risk assessment.
        """
        logger.info("Assessing correlation breakdown risk using causal models")
        
        # Placeholder logic - would be implemented using causal models
        # For example, using structural breaks in causal relationships
        
        return {
            "high_risk_pairs": [
                {"pair1": "EUR/USD", "pair2": "GBP/USD", "breakdown_probability": 0.78,
                 "potential_trigger": "ECB policy divergence"},
                {"pair1": "USD/JPY", "pair2": "USD/CHF", "breakdown_probability": 0.65,
                 "potential_trigger": "Safe haven correlation breakdown"}
            ],
            "medium_risk_pairs": [
                {"pair1": "AUD/USD", "pair2": "NZD/USD", "breakdown_probability": 0.45,
                 "potential_trigger": "Commodity price divergence"}
            ],
            "stable_relationships": [
                {"pair1": "EUR/GBP", "pair2": "EUR/CHF", "stability_score": 0.88}
            ],
            "analysis_timestamp": pd.Timestamp.now().isoformat()
        }
    
    def generate_counterfactual_scenarios(self, base_scenario: Dict[str, Any],
                                         interventions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generates multiple counterfactual scenarios for risk assessment and strategy testing.
        
        Args:
            base_scenario: Dictionary with the base market conditions.
            interventions: List of intervention dictionaries, each specifying changes to variables.
            
        Returns:
            Dictionary with counterfactual scenarios and their outcomes.
        """
        logger.info(f"Generating {len(interventions)} counterfactual scenarios")
        
        cf_analyzer = CounterfactualAnalysis()
        
        # In a real implementation, we'd build a proper causal model first
        # Placeholder model for demonstration
        model = {"outcome": "price_movement", "variables": list(base_scenario.keys())}
        cf_analyzer.set_causal_model(model)
        
        scenarios = []
        
        # Generate counterfactual for each intervention
        for i, intervention in enumerate(interventions):
            try:
                # Convert scenario to DataFrame for the counterfactual generation
                base_df = pd.DataFrame([base_scenario])
                cf_data = cf_analyzer.generate_counterfactuals(base_df, intervention)
                
                if cf_data is not None:
                    scenarios.append({
                        "scenario_id": f"CF_{i+1}",
                        "intervention": intervention,
                        "outcome_changes": {
                            k.replace('_counterfactual', ''): v.iloc[0] 
                            for k, v in cf_data.items() 
                            if k.endswith('_counterfactual')
                        },
                        "probability": 0.8 / (i + 1)  # Dummy probability
                    })
            except Exception as e:
                logger.error(f"Error generating counterfactual for intervention {i+1}: {e}")
        
        return {
            "base_scenario": base_scenario,
            "counterfactual_scenarios": scenarios,
            "generation_timestamp": pd.Timestamp.now().isoformat()
        }
    
    # Helper methods for signal enhancement
    
    def _calculate_causal_confidence(self, pair: str, signal_type: str, 
                                    direction: str, market_data: pd.DataFrame) -> float:
        """Calculate confidence adjustment based on causal factors."""
        # Placeholder implementation
        return 0.85
    
    def _identify_explanatory_factors(self, pair: str, signal_type: str, 
                                     direction: str, market_data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Identify causal factors explaining the signal."""
        # Placeholder implementation
        return [
            {"factor": "Interest rate differential", "contribution": 0.45},
            {"factor": "Recent volatility pattern", "contribution": 0.30}
        ]
    
    def _identify_conflicting_signals(self, pair: str, signal_type: str, 
                                     direction: str, market_data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Identify causal factors that conflict with the signal."""
        # Placeholder implementation
        return [
            {"factor": "Long-term trend", "conflict_strength": 0.25},
            {"factor": "News sentiment", "conflict_strength": 0.15}
        ]
    
    def _estimate_signal_duration(self, pair: str, signal_type: str, 
                                direction: str, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Estimate expected duration of the signal's validity."""
        # Placeholder implementation
        return {
            "expected_bars": 12,
            "confidence": 0.7,
            "factors": ["Historical signal duration", "Market regime"]
        }
