"""
Causal Discovery Algorithm Module

This module provides algorithms for discovering causal relationships in financial data.
"""
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import networkx as nx

from causal_analysis_service.core.algorithms.base import BaseCausalAlgorithm
from causal_analysis_service.utils.validation import validate_data_for_causal_analysis

logger = logging.getLogger(__name__)

class CausalDiscoveryAlgorithm(BaseCausalAlgorithm):
    """
    Base class for causal discovery algorithms.
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the causal discovery algorithm.
        
        Args:
            config: Configuration parameters for the algorithm
        """
        self.config = config or {}
        self.significance_level = self.config.get('significance_level', 0.05)
        self.max_lag = self.config.get('max_lag', 10)
        
    def discover_causal_relationships(self, data: pd.DataFrame) -> nx.DiGraph:
        """
        Discover causal relationships in the data.
        
        Args:
            data: DataFrame containing time series data
            
        Returns:
            A directed graph representing causal relationships
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def _preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess data for causal discovery.
        
        Args:
            data: DataFrame containing time series data
            
        Returns:
            Preprocessed DataFrame
        """
        # Validate data
        validate_data_for_causal_analysis(data)
        
        # Handle missing values
        data = data.fillna(method='ffill').fillna(method='bfill')
        
        # Ensure stationarity
        data = self._ensure_stationarity(data)
        
        return data
    
    def _ensure_stationarity(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Ensure time series are stationary by differencing if necessary.
        
        Args:
            data: DataFrame containing time series data
            
        Returns:
            DataFrame with stationary time series
        """
        from statsmodels.tsa.stattools import adfuller
        
        stationary_data = data.copy()
        
        for column in data.columns:
            # Skip non-numeric columns
            if not np.issubdtype(data[column].dtype, np.number):
                continue
                
            # Check stationarity with Augmented Dickey-Fuller test
            adf_result = adfuller(data[column].dropna())
            p_value = adf_result[1]
            
            # If not stationary, take first difference
            if p_value > 0.05:
                stationary_data[column] = data[column].diff().fillna(0)
                
                # Check again
                adf_result = adfuller(stationary_data[column].dropna())
                p_value = adf_result[1]
                
                # If still not stationary, take second difference
                if p_value > 0.05:
                    stationary_data[column] = stationary_data[column].diff().fillna(0)
        
        return stationary_data
    
    def _create_graph_from_adjacency_matrix(self, adjacency_matrix: np.ndarray, 
                                           variable_names: List[str]) -> nx.DiGraph:
        """
        Create a directed graph from an adjacency matrix.
        
        Args:
            adjacency_matrix: Adjacency matrix representing causal relationships
            variable_names: Names of variables corresponding to matrix indices
            
        Returns:
            A directed graph representing causal relationships
        """
        G = nx.DiGraph()
        
        # Add nodes
        for name in variable_names:
            G.add_node(name)
        
        # Add edges
        n = len(variable_names)
        for i in range(n):
            for j in range(n):
                if adjacency_matrix[i, j] != 0:
                    G.add_edge(variable_names[i], variable_names[j], weight=adjacency_matrix[i, j])
        
        return G


class GrangerCausalityAlgorithm(CausalDiscoveryAlgorithm):
    """
    Granger causality test for causal discovery.
    """
    def discover_causal_relationships(self, data: pd.DataFrame) -> nx.DiGraph:
        """
        Discover causal relationships using Granger causality tests.
        
        Args:
            data: DataFrame containing time series data
            
        Returns:
            A directed graph representing causal relationships
        """
        from statsmodels.tsa.stattools import grangercausalitytests
        
        # Preprocess data
        data = self._preprocess_data(data)
        
        # Get numeric columns only
        numeric_data = data.select_dtypes(include=[np.number])
        variable_names = numeric_data.columns.tolist()
        n = len(variable_names)
        
        # Initialize adjacency matrix
        adjacency_matrix = np.zeros((n, n))
        
        # Perform Granger causality tests
        for i in range(n):
            for j in range(n):
                if i != j:  # Skip self-causality
                    # Prepare data for test
                    y = numeric_data.iloc[:, j]
                    x = numeric_data.iloc[:, i]
                    test_data = pd.concat([y, x], axis=1)
                    
                    # Run Granger causality test
                    try:
                        test_result = grangercausalitytests(test_data, maxlag=self.max_lag, verbose=False)
                        
                        # Check if any lag shows causality
                        for lag in range(1, self.max_lag + 1):
                            # Use F-test p-value
                            p_value = test_result[lag][0]['ssr_ftest'][1]
                            if p_value < self.significance_level:
                                # Variable i Granger-causes variable j
                                adjacency_matrix[i, j] = 1 - p_value  # Weight is 1 - p_value
                                break
                    except Exception as e:
                        logger.warning(f"Granger causality test failed for {variable_names[i]} -> {variable_names[j]}: {e}")
        
        # Create graph from adjacency matrix
        causal_graph = self._create_graph_from_adjacency_matrix(adjacency_matrix, variable_names)
        
        return causal_graph


class PCAlgorithm(CausalDiscoveryAlgorithm):
    """
    PC algorithm for causal discovery.
    """
    def discover_causal_relationships(self, data: pd.DataFrame) -> nx.DiGraph:
        """
        Discover causal relationships using the PC algorithm.
        
        Args:
            data: DataFrame containing time series data
            
        Returns:
            A directed graph representing causal relationships
        """
        try:
            import causallearn
            from causallearn.search.ConstraintBased.PC import pc
        except ImportError:
            logger.error("causallearn package is required for PC algorithm")
            raise ImportError("causallearn package is required for PC algorithm")
        
        # Preprocess data
        data = self._preprocess_data(data)
        
        # Get numeric columns only
        numeric_data = data.select_dtypes(include=[np.number])
        variable_names = numeric_data.columns.tolist()
        
        # Convert to numpy array
        data_array = numeric_data.values
        
        # Run PC algorithm
        try:
            pc_result = pc(data_array, alpha=self.significance_level)
            adjacency_matrix = pc_result.G.graph
            
            # Create graph from adjacency matrix
            causal_graph = self._create_graph_from_adjacency_matrix(adjacency_matrix, variable_names)
            
            return causal_graph
        except Exception as e:
            logger.error(f"PC algorithm failed: {e}")
            # Fallback to Granger causality
            logger.info("Falling back to Granger causality")
            granger_algorithm = GrangerCausalityAlgorithm(self.config)
            return granger_algorithm.discover_causal_relationships(data)


class DoWhyAlgorithm(CausalDiscoveryAlgorithm):
    """
    DoWhy-based algorithm for causal discovery and effect estimation.
    """
    def discover_causal_relationships(self, data: pd.DataFrame) -> nx.DiGraph:
        """
        Discover causal relationships using DoWhy.
        
        Args:
            data: DataFrame containing time series data
            
        Returns:
            A directed graph representing causal relationships
        """
        try:
            import dowhy
            from dowhy import CausalModel
        except ImportError:
            logger.error("dowhy package is required for DoWhy algorithm")
            raise ImportError("dowhy package is required for DoWhy algorithm")
        
        # Preprocess data
        data = self._preprocess_data(data)
        
        # Get numeric columns only
        numeric_data = data.select_dtypes(include=[np.number])
        variable_names = numeric_data.columns.tolist()
        n = len(variable_names)
        
        # Initialize adjacency matrix
        adjacency_matrix = np.zeros((n, n))
        
        # Use Granger causality to initialize the graph
        granger_algorithm = GrangerCausalityAlgorithm(self.config)
        initial_graph = granger_algorithm.discover_causal_relationships(numeric_data)
        
        # Convert NetworkX graph to adjacency matrix
        for i, source in enumerate(variable_names):
            for j, target in enumerate(variable_names):
                if initial_graph.has_edge(source, target):
                    adjacency_matrix[i, j] = initial_graph[source][target]['weight']
        
        # Create graph from adjacency matrix
        causal_graph = self._create_graph_from_adjacency_matrix(adjacency_matrix, variable_names)
        
        return causal_graph
    
    def estimate_causal_effect(self, data: pd.DataFrame, treatment: str, 
                              outcome: str, confounders: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Estimate the causal effect of a treatment on an outcome.
        
        Args:
            data: DataFrame containing time series data
            treatment: Name of the treatment variable
            outcome: Name of the outcome variable
            confounders: List of confounding variables
            
        Returns:
            Dictionary containing causal effect estimates
        """
        try:
            import dowhy
            from dowhy import CausalModel
        except ImportError:
            logger.error("dowhy package is required for DoWhy algorithm")
            raise ImportError("dowhy package is required for DoWhy algorithm")
        
        # Preprocess data
        data = self._preprocess_data(data)
        
        # Identify confounders if not provided
        if confounders is None:
            # Use all other variables as potential confounders
            confounders = [col for col in data.columns if col != treatment and col != outcome]
        
        # Create causal model
        model = CausalModel(
            data=data,
            treatment=treatment,
            outcome=outcome,
            common_causes=confounders
        )
        
        # Identify estimand
        identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)
        
        # Estimate effect
        estimate = model.estimate_effect(identified_estimand,
                                        method_name="backdoor.linear_regression")
        
        # Refute estimate
        refutation_results = []
        refutation_methods = ["random_common_cause", "placebo_treatment_refuter"]
        
        for method in refutation_methods:
            try:
                refutation = model.refute_estimate(identified_estimand, estimate, method_name=method)
                refutation_results.append({
                    "method": method,
                    "refutation_result": refutation
                })
            except Exception as e:
                logger.warning(f"Refutation method {method} failed: {e}")
        
        # Return results
        return {
            "causal_effect": estimate.value,
            "confidence_interval": estimate.get_confidence_intervals(),
            "p_value": estimate.get_significance_test_results(),
            "refutation_results": refutation_results
        }


class CounterfactualAnalysisAlgorithm(CausalDiscoveryAlgorithm):
    """
    Algorithm for counterfactual analysis.
    """
    def generate_counterfactual(self, data: pd.DataFrame, intervention: Dict[str, Any], 
                               target_variables: List[str]) -> pd.DataFrame:
        """
        Generate counterfactual scenarios based on interventions.
        
        Args:
            data: DataFrame containing time series data
            intervention: Dictionary mapping variable names to intervention values
            target_variables: List of variables to predict counterfactual values for
            
        Returns:
            DataFrame containing counterfactual predictions
        """
        # Preprocess data
        data = self._preprocess_data(data)
        
        # Discover causal graph
        causal_graph = self.discover_causal_relationships(data)
        
        # Create counterfactual DataFrame
        counterfactual_data = data.copy()
        
        # Apply interventions
        for variable, value in intervention.items():
            if variable in counterfactual_data.columns:
                counterfactual_data[variable] = value
        
        # Propagate effects through the causal graph
        for target in target_variables:
            if target in intervention:
                continue  # Skip if target is directly intervened
                
            # Find all parents of the target in the causal graph
            parents = list(causal_graph.predecessors(target))
            
            if not parents:
                continue  # Skip if no parents
                
            # Create a simple linear model to predict the target
            from sklearn.linear_model import LinearRegression
            
            # Prepare training data
            X_train = data[parents]
            y_train = data[target]
            
            # Train model
            model = LinearRegression()
            model.fit(X_train, y_train)
            
            # Predict counterfactual values
            X_cf = counterfactual_data[parents]
            counterfactual_data[target] = model.predict(X_cf)
        
        return counterfactual_data