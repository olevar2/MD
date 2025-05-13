"""
Causal Inference Service

This module provides a service interface that makes causal inference capabilities
accessible to trading strategies and other components of the forex trading platform.
"""
import logging
from typing import Dict, List, Optional, Tuple, Union, Any
import pandas as pd
import networkx as nx
from datetime import datetime, timedelta
from analysis_engine.causal.inference.algorithms import PCAlgorithm, GrangerCausalityAnalyzer, DoWhyInterface, CounterfactualAnalysis
from analysis_engine.causal.data.preparation import FinancialDataPreprocessor, FinancialFeatureEngineering
from analysis_engine.causal.visualization.relationship_graph import CausalGraphVisualizer, CausalEffectVisualizer, CounterfactualVisualizer
from analysis_engine.causal.testing.algorithm_validation import ForexCausalValidation
logger = logging.getLogger(__name__)
from analysis_engine.core.exceptions_bridge import with_exception_handling, async_with_exception_handling, ForexTradingPlatformError, ServiceError, DataError, ValidationError


from analysis_engine.resilience.utils import (
    with_resilience,
    with_analysis_resilience,
    with_database_resilience
)

class CausalInferenceService:
    """
    Service for performing causal inference on forex market data.
    
    This service provides methods to discover causal relationships between
    currency pairs, estimate causal effects, and generate counterfactual
    scenarios for decision-making in trading strategies.
    """

    def __init__(self, config: Dict[str, Any]=None):
        """
        Initialize the causal inference service.
        
        Args:
            config: Configuration parameters for the service
        """
        self.config = config or {}
        self.max_lag = self.config_manager.get('max_lag', 10)
        self.significance_level = self.config_manager.get('significance_level', 0.05)
        self.cache_ttl = self.config_manager.get('cache_ttl_hours', 24)
        self.granger_analyzer = GrangerCausalityAnalyzer(max_lag=self.
            max_lag, alpha=self.significance_level)
        self.pc_algorithm = PCAlgorithm(alpha=self.significance_level)
        self.data_preprocessor = FinancialDataPreprocessor()
        self.feature_engineering = FinancialFeatureEngineering()
        self.dowhy_interface = DoWhyInterface()
        self.counterfactual_analyzer = CounterfactualAnalysis()
        self.graph_visualizer = CausalGraphVisualizer()
        self.effect_visualizer = CausalEffectVisualizer()
        self.counterfactual_visualizer = CounterfactualVisualizer()
        self.validator = ForexCausalValidation()
        self.causal_graph_cache = {}
        self.last_update_time = {}

    def discover_causal_structure(self, data: pd.DataFrame, method: str=
        'granger', force_refresh: bool=False, cache_key: Optional[str]=None
        ) ->nx.DiGraph:
        """
        Discover causal structure in the provided financial data.
        
        Args:
            data: DataFrame containing financial time series data
            method: Causality detection method ('granger' or 'pc')
            force_refresh: Whether to force a refresh of cached results
            cache_key: Optional key to use for caching results
            
        Returns:
            NetworkX DiGraph representing the discovered causal structure
        """
        if cache_key and not force_refresh:
            if cache_key in self.causal_graph_cache:
                last_update = self.last_update_time.get(cache_key, datetime.min
                    )
                if datetime.now() - last_update < timedelta(hours=self.
                    cache_ttl):
                    logger.info(f'Using cached causal graph for {cache_key}')
                    return self.causal_graph_cache[cache_key]
        if method == 'granger':
            processed_data = self.data_preprocessor.prepare_data(data)
            causal_graph = self.granger_analyzer.get_causal_graph(
                processed_data)
        elif method == 'pc':
            processed_data = self.data_preprocessor.scale_data(data)
            causal_graph = self.pc_algorithm.fit(processed_data)
        else:
            raise ValueError(
                f'Unsupported causality detection method: {method}')
        if cache_key:
            self.causal_graph_cache[cache_key] = causal_graph
            self.last_update_time[cache_key] = datetime.now()
        return causal_graph

    def estimate_causal_effect(self, data: pd.DataFrame, treatment: str,
        outcome: str, causal_graph: Optional[nx.DiGraph]=None, confounders:
        Optional[List[str]]=None) ->Dict[str, Any]:
        """
        Estimate the causal effect of one variable on another.
        
        Args:
            data: DataFrame containing financial time series data
            treatment: Treatment variable (cause)
            outcome: Outcome variable (effect)
            causal_graph: Optional causal graph to use
            confounders: Optional list of confounding variables
            
        Returns:
            Dictionary with estimated causal effect and confidence intervals
        """
        if causal_graph is None and confounders is None:
            causal_graph = self.discover_causal_structure(data)
        identification_results = self.dowhy_interface.identify_causal_effect(
            data=data, treatment=treatment, outcome=outcome, graph=
            causal_graph, confounders=confounders)
        estimation_results = self.dowhy_interface.estimate_effect()
        return {'identification': identification_results, 'estimation':
            estimation_results}

    def generate_counterfactuals(self, data: pd.DataFrame, target: str,
        interventions: Dict[str, Dict[str, float]], features: Optional[List
        [str]]=None) ->Dict[str, pd.DataFrame]:
        """
        Generate counterfactual scenarios for different interventions.
        
        Args:
            data: DataFrame containing financial time series data
            target: Target variable to predict counterfactuals for
            interventions: Dictionary mapping scenario names to intervention values
            features: Optional list of features to use in the model
            
        Returns:
            Dictionary mapping scenario names to counterfactual DataFrames
        """
        if features is None:
            features = [col for col in data.columns if col != target]
        self.counterfactual_analyzer.fit(data, target, features)
        counterfactuals = {}
        for scenario_name, intervention_values in interventions.items():
            cf_data = self.counterfactual_analyzer.generate_counterfactual(data
                , intervention_values)
            counterfactuals[scenario_name] = cf_data
        return counterfactuals

    @with_resilience('validate_causal_discovery')
    def validate_causal_discovery(self, algorithm, known_relationships: nx.
        DiGraph, data: Optional[pd.DataFrame]=None, synthetic_params:
        Optional[Dict[str, Any]]=None) ->Dict[str, float]:
        """
        Validate causal discovery algorithms against known relationships.
        
        Args:
            algorithm: Causal discovery algorithm to validate
            known_relationships: Graph of known causal relationships
            data: Optional real data to use for validation
            synthetic_params: Parameters for synthetic data generation
            
        Returns:
            Dictionary with validation metrics
        """
        if data is not None:
            processed_data = self.data_preprocessor.prepare_data(data)
            discovered_graph = self.discover_causal_structure(processed_data)
            precision, recall, f1 = self._calculate_graph_metrics(
                known_relationships, discovered_graph)
            return {'precision': precision, 'recall': recall, 'f1_score':
                f1, 'data_source': 'real'}
        else:
            params = synthetic_params or {'n_currencies': 8, 'n_samples': 500}
            validation_results = self.validator.validate_with_synthetic_forex(
                algorithm=algorithm, n_datasets=5, n_currencies=params.get(
                'n_currencies', 8), n_samples=params.get('n_samples', 500))
            return {'precision': validation_results['avg_precision'],
                'recall': validation_results['avg_recall'], 'f1_score':
                validation_results['avg_f1'], 'data_source': 'synthetic'}

    def _calculate_graph_metrics(self, true_graph: nx.DiGraph,
        discovered_graph: nx.DiGraph) ->Tuple[float, float, float]:
        """Calculate precision, recall and F1 score between two causal graphs."""
        true_edges = set(true_graph.edges())
        discovered_edges = set(discovered_graph.edges())
        if not discovered_edges:
            precision = 1.0 if not true_edges else 0.0
        else:
            precision = len(true_edges.intersection(discovered_edges)) / len(
                discovered_edges)
        if not true_edges:
            recall = 1.0 if not discovered_edges else 0.0
        else:
            recall = len(true_edges.intersection(discovered_edges)) / len(
                true_edges)
        f1 = 2 * precision * recall / (precision + recall
            ) if precision + recall > 0 else 0.0
        return precision, recall, f1

    @with_resilience('validate_relationship')
    @with_exception_handling
    def validate_relationship(self, data: pd.DataFrame, cause: str, effect:
        str, methods: Optional[List[str]]=None, confidence_threshold: float=0.7
        ) ->Dict[str, Any]:
        """
        Validates a causal relationship using multiple validation methods.
        
        Args:
            data: DataFrame containing time series data
            cause: Hypothesized cause variable
            effect: Hypothesized effect variable
            methods: List of validation methods to use
            confidence_threshold: Threshold for confidence score
            
        Returns:
            Dictionary containing validation results
        """
        if methods is None:
            methods = ['granger', 'cross_correlation', 'mutual_information']
        results = {'is_valid': False, 'confidence_score': 0.0,
            'validation_details': {}, 'failed_checks': []}
        try:
            if 'granger' in methods:
                granger_result = self.granger_analyzer.test_causality(data[
                    cause], data[effect])
                results['validation_details']['granger'] = granger_result
                if not granger_result['is_significant']:
                    results['failed_checks'].append('granger_causality')
            if 'cross_correlation' in methods:
                from scipy import signal
                ccf = signal.correlate(data[cause].fillna(0), data[effect].
                    fillna(0), mode='full')
                lags = signal.correlation_lags(len(data[cause]), len(data[
                    effect]))
                max_corr = np.max(np.abs(ccf))
                max_lag = lags[np.argmax(np.abs(ccf))]
                cross_corr_result = {'max_correlation': float(max_corr),
                    'lag': int(max_lag), 'is_significant': max_corr > 0.3}
                results['validation_details']['cross_correlation'
                    ] = cross_corr_result
                if not cross_corr_result['is_significant']:
                    results['failed_checks'].append('cross_correlation')
            if 'mutual_information' in methods:
                from sklearn.feature_selection import mutual_info_regression
                X = data[cause].values.reshape(-1, 1)
                y = data[effect].values
                mi_score = float(mutual_info_regression(X, y)[0])
                mi_result = {'score': mi_score, 'is_significant': mi_score >
                    0.1}
                results['validation_details']['mutual_information'] = mi_result
                if not mi_result['is_significant']:
                    results['failed_checks'].append('mutual_information')
            total_methods = len(methods)
            passed_methods = total_methods - len(results['failed_checks'])
            results['confidence_score'] = passed_methods / total_methods
            results['is_valid'] = results['confidence_score'
                ] >= confidence_threshold and len(results['failed_checks']
                ) < total_methods / 2
            return results
        except Exception as e:
            logger.error(f'Error in relationship validation: {e}')
            return results

    @with_resilience('validate_multiple_relationships')
    def validate_multiple_relationships(self, data: pd.DataFrame,
        relationships: List[Tuple[str, str]], **kwargs) ->Dict[Tuple[str,
        str], Dict[str, Any]]:
        """
        Validates multiple causal relationships in parallel.
        
        Args:
            data: DataFrame containing time series data
            relationships: List of (cause, effect) tuples to validate
            **kwargs: Additional arguments for validate_relationship
            
        Returns:
            Dictionary mapping relationships to their validation results
        """
        results = {}
        for cause, effect in relationships:
            results[cause, effect] = self.validate_relationship(data, cause,
                effect, **kwargs)
        return results
