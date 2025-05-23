"""
Causal Graph Generation Module

Implements causal graph generation and statistical validation for forex market data.
"""
import logging
from typing import Dict, List, Optional, Tuple, Any
import networkx as nx
import pandas as pd
import numpy as np
from scipy import stats
from dataclasses import dataclass
from ..detection.relationship_detector import CausalRelationshipAnalyzer
logger = logging.getLogger(__name__)


from analysis_engine.resilience.utils import (
    with_resilience,
    with_analysis_resilience,
    with_database_resilience
)

@dataclass
class GraphValidationResult:
    """Contains validation results for a causal graph."""
    is_valid: bool
    confidence_score: float
    validation_details: Dict[str, Any]
    failed_checks: List[str]


class CausalGraphGenerator:
    """Generates and validates causal graphs for forex market relationships."""

    def __init__(self, detector: Optional[CausalRelationshipAnalyzer]=None,
        significance_level: float=0.05):
    """
      init  .
    
    Args:
        detector: Description of detector
        significance_level: Description of significance_level
    
    """

        self.detector = detector or CausalRelationshipAnalyzer(
            significance_level)
        self.significance_level = significance_level

    def generate_graph(self, data: pd.DataFrame, focus_variables: Optional[
        List[str]]=None, min_confidence: float=0.6) ->nx.DiGraph:
        """
        Generates a causal graph from forex market data.
        
        Args:
            data: DataFrame containing market variables
            focus_variables: Optional list of variables to focus on
            min_confidence: Minimum confidence threshold for including edges
            
        Returns:
            NetworkX DiGraph representing causal relationships
        """
        variables = focus_variables or list(data.columns)
        graph = nx.DiGraph()
        for var in variables:
            graph.add_node(var, type='variable')
        for effect in variables:
            causes = [v for v in variables if v != effect]
            causality_results = self.detector.detect_granger_causality(data,
                effect, causes)
            for cause, result in causality_results.items():
                if result['is_significant']:
                    weight = 1 - result['p_value']
                    if weight >= min_confidence:
                        graph.add_edge(cause, effect, weight=weight, lag=
                            result['best_lag'], p_value=result['p_value'],
                            f_stat=result['f_stat'])
        return graph

    @with_resilience('validate_graph')
    def validate_graph(self, graph: nx.DiGraph, data: pd.DataFrame
        ) ->GraphValidationResult:
        """
        Validates a causal graph using multiple statistical tests.
        
        Args:
            graph: NetworkX DiGraph to validate
            data: DataFrame containing the original data
            
        Returns:
            GraphValidationResult containing validation metrics
        """
        validation_details = {}
        failed_checks = []
        cycles = list(nx.simple_cycles(graph))
        validation_details['cycles'] = {'found': len(cycles), 'cycles': cycles}
        if len(cycles) > 0:
            for cycle in cycles:
                cycle_valid = all(graph[cycle[i]][cycle[(i + 1) % len(cycle
                    )]]['weight'] > 0.8 for i in range(len(cycle)))
                if not cycle_valid:
                    failed_checks.append('cycle_validation')
                    break
        edge_strengths = []
        for u, v, data in graph.edges(data=True):
            edge_strengths.append(data['weight'])
        validation_details['edge_strengths'] = {'mean': np.mean(
            edge_strengths), 'std': np.std(edge_strengths), 'min': np.min(
            edge_strengths), 'max': np.max(edge_strengths)}
        if validation_details['edge_strengths']['std'] > 0.5:
            failed_checks.append('edge_strength_consistency')
        temporal_violations = 0
        for u, v, data in graph.edges(data=True):
            windows = np.array_split(data, 3)
            violation = False
            for window in windows:
                result = self.detector.validate_relationship(pd.DataFrame(
                    window, columns=data.columns), u, v)
                if not result['is_valid']:
                    violation = True
                    break
            if violation:
                temporal_violations += 1
        validation_details['temporal_consistency'] = {'violations':
            temporal_violations, 'total_edges': graph.number_of_edges(),
            'violation_ratio': temporal_violations / max(1, graph.
            number_of_edges())}
        if validation_details['temporal_consistency']['violation_ratio'] > 0.3:
            failed_checks.append('temporal_consistency')
        num_checks = 3
        confidence_score = 1.0 - len(failed_checks) / num_checks
        return GraphValidationResult(is_valid=len(failed_checks) == 0,
            confidence_score=confidence_score, validation_details=
            validation_details, failed_checks=failed_checks)

    def generate_validated_graph(self, data: pd.DataFrame, focus_variables:
        Optional[List[str]]=None, min_confidence: float=0.6) ->Tuple[nx.
        DiGraph, GraphValidationResult]:
        """
        Generates and validates a causal graph in one step.
        
        Args:
            data: DataFrame containing market variables
            focus_variables: Optional list of variables to focus on
            min_confidence: Minimum confidence threshold for including edges
            
        Returns:
            Tuple of (validated graph, validation result)
        """
        graph = self.generate_graph(data, focus_variables, min_confidence)
        validation_result = self.validate_graph(graph, data)
        if (not validation_result.is_valid and validation_result.
            confidence_score > 0.3):
            if 'edge_strength_consistency' in validation_result.failed_checks:
                edges_to_remove = [(u, v) for u, v, d in graph.edges(data=
                    True) if d['weight'] < min_confidence]
                graph.remove_edges_from(edges_to_remove)
            if 'temporal_consistency' in validation_result.failed_checks:
                for u, v, d in list(graph.edges(data=True)):
                    result = self.detector.validate_relationship(data, u, v)
                    if not result['is_valid']:
                        graph.remove_edge(u, v)
            validation_result = self.validate_graph(graph, data)
        return graph, validation_result
