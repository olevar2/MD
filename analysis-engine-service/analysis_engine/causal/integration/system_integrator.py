"""
Confidence Scoring and System Integration Module

Implements a confidence scoring system for causal relationships and
provides integration points with the trading decision system.
"""

import logging
from typing import Dict, List, Optional, Tuple, Any, Union
import pandas as pd
import numpy as np
from dataclasses import dataclass
import networkx as nx
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LassoCV
from sklearn.metrics import r2_score

from ..detection.relationship_detector import CausalRelationshipAnalyzer
from ..graph.causal_graph_generator import CausalGraphGenerator

logger = logging.getLogger(__name__)

@dataclass
class CausalConfidenceScore:
    """Represents a confidence score for a causal relationship or insight."""
    score: float  # Overall confidence score (0-1)
    components: Dict[str, float]  # Individual component scores
    supporting_evidence: Dict[str, Any]  # Evidence supporting the scores
    timestamp: pd.Timestamp  # When the score was calculated

class CausalSystemIntegrator:
    """
    Integrates causal insights with the trading system and maintains
    confidence scoring for relationships and decisions.
    """
    
    def __init__(self,
                 detector: Optional[CausalRelationshipAnalyzer] = None,
                 graph_generator: Optional[CausalGraphGenerator] = None):
        self.detector = detector or CausalRelationshipAnalyzer()
        self.graph_generator = graph_generator or CausalGraphGenerator(self.detector)
        self.confidence_threshold = 0.7
        self.min_validation_periods = 3
        
    def calculate_relationship_confidence(self,
                                         data: pd.DataFrame,
                                         cause: str,
                                         effect: str,
                                         validation_windows: int = 3) -> CausalConfidenceScore:
        """
        Calculates a comprehensive confidence score for a causal relationship.
        
        Args:
            data: Time series data containing the variables
            cause: Name of the cause variable
            effect: Name of the effect variable
            validation_windows: Number of time windows for validation
            
        Returns:
            CausalConfidenceScore object with detailed metrics
        """
        components = {}
        evidence = {}
        
        # 1. Statistical significance score with multiple tests
        relationship = self.detector.validate_relationship(data, cause, effect)
        statistical_scores = []
        
        # Granger causality
        if 'granger' in relationship['validation_details']:
            granger_result = relationship['validation_details']['granger']
            statistical_scores.append(1 - granger_result.get('p_value', 1.0))
        
        # Cross-correlation
        if 'cross_correlation' in relationship['validation_details']:
            cc_result = relationship['validation_details']['cross_correlation']
            statistical_scores.append(cc_result.get('max_correlation', 0.0))
        
        # Mutual information
        if 'mutual_information' in relationship['validation_details']:
            mi_result = relationship['validation_details']['mutual_information']
            statistical_scores.append(mi_result.get('score', 0.0))
        
        components['statistical_significance'] = np.mean(statistical_scores) if statistical_scores else 0.0
        evidence['statistical_tests'] = relationship['validation_details']
        
        # 2. Temporal stability score with dynamic window sizing
        splits = TimeSeriesSplit(n_splits=validation_windows)
        stability_scores = []
        window_sizes = []
        
        for train_idx, test_idx in splits.split(data):
            train_data = data.iloc[train_idx]
            test_data = data.iloc[test_idx]
            window_sizes.append(len(test_data))
            
            # Check relationship stability
            train_valid = self.detector.validate_relationship(train_data, cause, effect)
            test_valid = self.detector.validate_relationship(test_data, cause, effect)
            
            # Weight score by window size
            stability = 1.0 if train_valid['is_valid'] == test_valid['is_valid'] else 0.0
            stability_scores.append(stability * len(test_data))
        
        components['temporal_stability'] = np.sum(stability_scores) / np.sum(window_sizes)
        evidence['stability_scores'] = {
            'scores': stability_scores,
            'window_sizes': window_sizes
        }
        
        # 3. Predictive power score with advanced metrics
        try:
            prediction_scores = []
            feature_importance = []
            
            for train_idx, test_idx in splits.split(data):
                X_train = data[cause].iloc[train_idx].values.reshape(-1, 1)
                y_train = data[effect].iloc[train_idx]
                X_test = data[cause].iloc[test_idx].values.reshape(-1, 1)
                y_test = data[effect].iloc[test_idx]
                
                # Try multiple models and ensemble their predictions
                models = [
                    LinearRegression(),
                    RandomForestRegressor(n_estimators=100),
                    LassoCV(cv=5)
                ]
                
                model_scores = []
                for model in models:
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    
                    # Calculate multiple metrics
                    r2 = max(0, r2_score(y_test, y_pred))
                    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                    mae = mean_absolute_error(y_test, y_pred)
                    
                    # Combine metrics into a single score
                    model_score = (r2 + (1 / (1 + rmse)) + (1 / (1 + mae))) / 3
                    model_scores.append(model_score)
                    
                    # Get feature importance if available
                    if hasattr(model, 'feature_importances_'):
                        feature_importance.append(model.feature_importances_[0])
                    elif hasattr(model, 'coef_'):
                        feature_importance.append(abs(model.coef_[0]))
                
                prediction_scores.append(np.mean(model_scores))
            
            components['predictive_power'] = np.mean(prediction_scores)
            evidence['prediction_scores'] = {
                'scores': prediction_scores,
                'feature_importance': np.mean(feature_importance) if feature_importance else None
            }
            
        except Exception as e:
            logger.warning(f"Error calculating predictive power: {str(e)}")
            components['predictive_power'] = 0.0
        
        # 4. Graph consistency score with network analysis
        graph, validation = self.graph_generator.generate_validated_graph(data)
        
        if graph.has_edge(cause, effect):
            edge_data = graph.get_edge_data(cause, effect)
            # Calculate edge importance using network centrality
            centrality = nx.edge_betweenness_centrality(graph)
            edge_importance = centrality.get((cause, effect), 0.0)
            
            components['graph_consistency'] = (
                edge_data.get('weight', 0.0) * 0.7 +  # Direct edge weight
                edge_importance * 0.3                  # Edge importance in network
            )
        else:
            components['graph_consistency'] = 0.0
          evidence['graph_validation'] = validation.validation_details
        
        # 5. Add robustness score
        robustness_score, robustness_evidence = self._calculate_robustness_score(data, cause, effect)
        components['robustness'] = robustness_score
        evidence['robustness'] = robustness_evidence
        
        # 6. Add historical performance score
        hist_score, hist_evidence = self._calculate_historical_performance(cause, effect)
        components['historical_performance'] = hist_score
        evidence['historical_performance'] = hist_evidence
        
        # Calculate weighted confidence score with enhanced components
        weights = {
            'statistical_significance': 0.22,
            'temporal_stability': 0.22,
            'predictive_power': 0.15,
            'graph_consistency': 0.15,
            'robustness': 0.13,
            'historical_performance': 0.13
        }
        
        # Adaptive weights based on data quality
        if data.isnull().sum().sum() / (data.shape[0] * data.shape[1]) > 0.1:
            # Reduce weight of statistical tests if data has many missing values
            weights['statistical_significance'] *= 0.8
            weights['temporal_stability'] *= 1.2
        
        # Normalize weights
        weight_sum = sum(weights.values())
        weights = {k: v/weight_sum for k, v in weights.items()}
        
        overall_score = sum(
            score * weights[component]
            for component, score in components.items()
        )
        
        return CausalConfidenceScore(
            score=overall_score,
            components=components,
            supporting_evidence=evidence,
            timestamp=pd.Timestamp.now()
        )
    
    def enhance_trading_decision(self,
                                 decision: Dict[str, Any],
                                 market_data: pd.DataFrame,
                                 confidence_threshold: Optional[float] = None) -> Dict[str, Any]:
        """
        Enhances a trading decision with causal insights.
        
        Args:
            decision: Original trading decision dictionary
            market_data: Recent market data
            confidence_threshold: Optional override for confidence threshold
            
        Returns:
            Enhanced trading decision with causal insights
        """
        threshold = confidence_threshold or self.confidence_threshold
        enhanced = decision.copy()
        causal_insights = []
        
        # Generate causal graph for current market conditions
        graph, validation = self.graph_generator.generate_validated_graph(market_data)
        
        # Add causal validation to the decision
        enhanced['causal_validation'] = {
            'graph_confidence': validation.confidence_score,
            'validated': validation.is_valid,
            'failed_checks': validation.failed_checks
        }
        
        # Analyze each factor in the decision
        for factor, impact in decision.get('factors', {}).items():
            if factor in market_data.columns:
                # Find causal paths to the target variable
                target_var = decision.get('target_variable', 'price')
                if target_var in market_data.columns:
                    confidence = self.calculate_relationship_confidence(
                        market_data, factor, target_var
                    )
                    
                    if confidence.score >= threshold:
                        causal_insights.append({
                            'factor': factor,
                            'confidence': confidence.score,
                            'impact': impact,
                            'evidence': confidence.components
                        })
        
        enhanced['causal_insights'] = causal_insights
        
        # Adjust decision confidence based on causal validation
        original_confidence = decision.get('confidence', 0.5)
        causal_factor = np.mean([insight['confidence'] for insight in causal_insights]) if causal_insights else 0.5
        enhanced['confidence'] = (original_confidence + causal_factor) / 2
        
        # Add recommendation for decision adjustment if needed
        if enhanced['confidence'] < threshold and decision.get('action') != 'hold':
            enhanced['recommendation'] = {
                'action': 'reduce_position',
                'reason': 'insufficient_causal_confidence',
                'details': {
                    'original_confidence': original_confidence,
                    'causal_confidence': causal_factor,
                    'threshold': threshold
                }
            }
        
        return enhanced
    
    def track_performance_impact(self,
                                 decisions: List[Dict[str, Any]],
                                 outcomes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Tracks the impact of causal insights on decision performance.
        
        Args:
            decisions: List of enhanced trading decisions
            outcomes: List of corresponding trading outcomes
            
        Returns:
            Performance impact analysis
        """
        if not decisions or not outcomes or len(decisions) != len(outcomes):
            raise ValueError("Decisions and outcomes must be non-empty and match in length")
            
        impact_analysis = {
            'overall_improvement': 0.0,
            'confidence_correlation': 0.0,
            'insight_value': {},
            'validation_summary': {
                'total_decisions': len(decisions),
                'high_confidence_success_rate': 0.0,
                'low_confidence_success_rate': 0.0
            }
        }
        
        # Calculate success rates by confidence level
        high_conf_outcomes = []
        low_conf_outcomes = []
        
        for decision, outcome in zip(decisions, outcomes):
            success = outcome.get('success', False)
            confidence = decision.get('confidence', 0.0)
            
            if confidence >= self.confidence_threshold:
                high_conf_outcomes.append(success)
            else:
                low_conf_outcomes.append(success)
                
        if high_conf_outcomes:
            impact_analysis['validation_summary']['high_confidence_success_rate'] = \
                sum(high_conf_outcomes) / len(high_conf_outcomes)
                
        if low_conf_outcomes:
            impact_analysis['validation_summary']['low_confidence_success_rate'] = \
                sum(low_conf_outcomes) / len(low_conf_outcomes)
        
        # Calculate correlation between confidence and outcome
        confidences = [d.get('confidence', 0.0) for d in decisions]
        successes = [1.0 if o.get('success', False) else 0.0 for o in outcomes]
        
        if len(set(confidences)) > 1 and len(set(successes)) > 1:
            impact_analysis['confidence_correlation'] = np.corrcoef(confidences, successes)[0, 1]
        
        # Analyze value of different causal insights
        insight_outcomes = {}
        
        for decision, outcome in zip(decisions, outcomes):
            for insight in decision.get('causal_insights', []):
                factor = insight['factor']
                if factor not in insight_outcomes:
                    insight_outcomes[factor] = []
                    
                insight_outcomes[factor].append({
                    'confidence': insight['confidence'],
                    'success': outcome.get('success', False),
                    'pnl': outcome.get('pnl', 0.0)
                })
        
        # Calculate insight value metrics
        for factor, outcomes in insight_outcomes.items():
            impact_analysis['insight_value'][factor] = {
                'success_rate': sum(o['success'] for o in outcomes) / len(outcomes),
                'avg_pnl': np.mean([o['pnl'] for o in outcomes]),
                'confidence_correlation': np.corrcoef(
                    [o['confidence'] for o in outcomes],
                    [o['pnl'] for o in outcomes]
                )[0, 1] if len(outcomes) > 1 else 0.0
            }
        
        # Calculate overall improvement
        baseline_success_rate = impact_analysis['validation_summary']['low_confidence_success_rate']
        enhanced_success_rate = impact_analysis['validation_summary']['high_confidence_success_rate']
        impact_analysis['overall_improvement'] = enhanced_success_rate - baseline_success_rate
        
        return impact_analysis
    
    def _calculate_robustness_score(self, data: pd.DataFrame, cause: str, effect: str) -> Tuple[float, Dict[str, Any]]:
        """
        Calculates how robust the causal relationship is under different conditions.
        
        Args:
            data: DataFrame with time series data
            cause: Name of the cause variable
            effect: Name of the effect variable
            
        Returns:
            Tuple of (robustness_score, evidence_dictionary)
        """
        evidence = {}
        scores = []

        # Test with different subsets of data
        for sample_size in [0.8, 0.6, 0.4]:
            sample_scores = []
            for _ in range(5):  # Run multiple trials
                sample_data = data.sample(frac=sample_size)
                relationship = self.detector.validate_relationship(sample_data, cause, effect)
                sample_scores.append(relationship['confidence_score'])
            scores.append(np.mean(sample_scores))
        
        # Test with added noise
        noise_levels = [0.01, 0.05, 0.1]
        for noise_level in noise_levels:
            noisy_data = data.copy()
            noise = np.random.normal(0, noise_level, size=len(data))
            noisy_data[effect] = noisy_data[effect] + noise
            relationship = self.detector.validate_relationship(noisy_data, cause, effect)
            scores.append(relationship['confidence_score'])

        evidence['subsample_scores'] = scores[:3]
        evidence['noise_scores'] = scores[3:]
        
        # Weight recent scores more heavily
        weighted_score = np.average(scores, weights=np.linspace(0.5, 1.0, len(scores)))
        
        return weighted_score, evidence

    def _calculate_historical_performance(self, cause: str, effect: str) -> Tuple[float, Dict[str, Any]]:
        """
        Calculates a score based on historical trading performance using this relationship.
        
        Args:
            cause: Name of the cause variable
            effect: Name of the effect variable
            
        Returns:
            Tuple of (performance_score, evidence_dictionary)
        """
        try:
            # Retrieve historical decisions and outcomes involving this relationship
            decisions = self._get_historical_decisions(cause, effect)
            if not decisions:
                return 0.0, {'error': 'No historical data available'}

            performance_metrics = []
            confidence_pnl_correlation = []

            for decision in decisions:
                if 'causal_insights' in decision:
                    for insight in decision['causal_insights']:
                        if insight['factor'] == cause and decision.get('outcome'):
                            # Calculate success rate
                            success = decision['outcome'].get('success', False)
                            performance_metrics.append(success)
                            
                            # Calculate correlation between confidence and PnL
                            confidence = insight.get('confidence', 0)
                            pnl = decision['outcome'].get('pnl', 0)
                            if pnl != 0:  # Avoid cases with no PnL
                                confidence_pnl_correlation.append((confidence, pnl))

            evidence = {
                'success_rate': np.mean(performance_metrics) if performance_metrics else 0,
                'sample_size': len(performance_metrics),
                'confidence_pnl_correlation': (
                    np.corrcoef(list(zip(*confidence_pnl_correlation)))[0, 1]
                    if confidence_pnl_correlation else 0
                )
            }

            # Combine metrics into final score
            if performance_metrics:
                final_score = (
                    evidence['success_rate'] * 0.7 +
                    abs(evidence['confidence_pnl_correlation']) * 0.3
                )
            else:
                final_score = 0.0

            return final_score, evidence

        except Exception as e:
            logger.error(f"Error calculating historical performance: {str(e)}")
            return 0.0, {'error': str(e)}

    def _get_historical_decisions(self, cause: str, effect: str) -> List[Dict[str, Any]]:
        """
        Retrieves historical trading decisions involving the given causal relationship.
        
        Args:
            cause: Name of the cause variable
            effect: Name of the effect variable
            
        Returns:
            List of historical decision dictionaries with outcomes
        """
        # TODO: Implement connection to historical decision database
        # This is currently a placeholder implementation
        return []
