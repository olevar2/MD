"""
Core algorithms for Causal Inference Proof of Concept (POC).

This module implements various algorithms for causal discovery,
conditional independence testing, causal effect estimation, and
counterfactual analysis, specifically tailored for financial time-series data.
"""
import pandas as pd
import numpy as np
import logging
from scipy.stats import power_divergence, chi2_contingency
from sklearn.kernel_ridge import KernelRidge
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV
from sklearn.neighbors import NearestNeighbors
logger = logging.getLogger(__name__)


from analysis_engine.core.exceptions_bridge import (
    with_exception_handling,
    async_with_exception_handling,
    ForexTradingPlatformError,
    ServiceError,
    DataError,
    ValidationError
)

class CausalGraph:
    """
    Base class for representing causal graphs.
    Might use networkx or a custom implementation.
    """

    def __init__(self, nodes=None, edges=None):
        self.nodes = nodes if nodes is not None else []
        self.edges = edges if edges is not None else []
        pass

    def add_node(self, node):
        """Adds a node to the graph."""
        if node not in self.nodes:
            self.nodes.append(node)
        else:
            logger.warning(f'Node {node} already exists in the graph.')

    def add_edge(self, u, v, edge_type='directed'):
        """Adds an edge between nodes u and v."""
        if u not in self.nodes:
            self.add_node(u)
        if v not in self.nodes:
            self.add_node(v)
        edge = u, v, edge_type
        if edge not in self.edges:
            self.edges.append(edge)
        else:
            logger.warning(f'Edge {edge} already exists in the graph.')

    def __str__(self):
        return f'CausalGraph(nodes={len(self.nodes)}, edges={len(self.edges)})'


def pc_algorithm(data: pd.DataFrame, **kwargs):
    """
    Implements the PC algorithm for causal structure learning.
    Handles time-series considerations if necessary.
    Requires conditional independence tests.
    """
    logger.warning('PC algorithm is not yet implemented.')
    raise NotImplementedError(
        'PC algorithm requires implementation using external libraries like causal-learn.'
        )


def fci_algorithm(data: pd.DataFrame, **kwargs):
    """
    Implements the Fast Causal Inference (FCI) algorithm,
    which can handle latent confounders.
    """
    logger.warning('FCI algorithm is not yet implemented.')
    raise NotImplementedError(
        'FCI algorithm requires implementation using external libraries like causal-learn.'
        )


def lingam_algorithm(data: pd.DataFrame, **kwargs):
    """
    Implements the LiNGAM (Linear Non-Gaussian Acyclic Model) algorithm.
    Assumes linear relationships and non-Gaussian noise.
    """
    logger.warning('LiNGAM algorithm is not yet implemented.')
    raise NotImplementedError(
        'LiNGAM algorithm requires implementation using external libraries like causal-learn or lingam.'
        )


@with_exception_handling
def g_test(data: pd.DataFrame, x: str, y: str, condition_set: list=None,
    lambda_='log-likelihood', **kwargs):
    """
    Performs a G-test (log-likelihood ratio test) for conditional independence.
    Suitable for categorical data or discretized continuous data.

    Args:
        data (pd.DataFrame): DataFrame containing the variables.
        x (str): Name of the first variable column.
        y (str): Name of the second variable column.
        condition_set (list, optional): List of conditioning variable column names. Defaults to None.
        lambda_ (str, optional): The statistic to compute ('log-likelihood', 'pearson', etc.). Defaults to 'log-likelihood'.

    Returns:
        float: The p-value of the test. Returns 1.0 if an error occurs or data is insufficient.
    """
    logger.debug(f'Performing G-test for {x} _||_ {y} | {condition_set}')
    if x not in data.columns or y not in data.columns:
        logger.error(f"Variable(s) '{x}' or '{y}' not found in data.")
        return 1.0
    if condition_set and not all(c in data.columns for c in condition_set):
        logger.error('One or more conditioning variables not found in data.')
        return 1.0
    try:
        if not condition_set:
            contingency_table = pd.crosstab(data[x], data[y])
            if contingency_table.size == 0 or contingency_table.sum().sum(
                ) == 0:
                logger.warning(
                    f'Empty contingency table for G-test({x}, {y}). Returning p=1.0'
                    )
                return 1.0
            g_stat, p_value, dof, expected = power_divergence(
                contingency_table + 1e-10, lambda_=lambda_)
        else:
            total_g_stat = 0
            total_dof = 0
            valid_strata = 0
            grouped = data.groupby(condition_set)
            for _, group_data in grouped:
                if len(group_data) < 2:
                    continue
                contingency_table = pd.crosstab(group_data[x], group_data[y])
                if contingency_table.shape[0] < 2 or contingency_table.shape[1
                    ] < 2:
                    continue
                if contingency_table.sum().sum() == 0:
                    continue
                g, p, dof, expected = power_divergence(contingency_table + 
                    1e-10, lambda_=lambda_)
                total_g_stat += g
                total_dof += dof
                valid_strata += 1
            if valid_strata == 0 or total_dof <= 0:
                logger.warning(
                    f'Insufficient data or degrees of freedom for conditional G-test({x}, {y} | {condition_set}). Returning p=1.0'
                    )
                return 1.0
            from scipy.stats import chi2
            p_value = chi2.sf(total_g_stat, total_dof)
        logger.debug(f'G-test({x}, {y} | {condition_set}) p-value: {p_value}')
        return p_value
    except Exception as e:
        logger.exception(
            f'Error during G-test for {x}, {y} | {condition_set}: {e}')
        return 1.0


@with_exception_handling
def chi_squared_test(data: pd.DataFrame, x: str, y: str, condition_set:
    list=None, **kwargs):
    """
    Performs a Chi-squared test for conditional independence.
    Suitable for categorical data or discretized continuous data.
    """
    logger.debug(
        f'Performing Chi-squared test for {x} _||_ {y} | {condition_set}')
    if x not in data.columns or y not in data.columns:
        logger.error(f"Variable(s) '{x}' or '{y}' not found in data.")
        return 1.0
    if condition_set and not all(c in data.columns for c in condition_set):
        logger.error('One or more conditioning variables not found in data.')
        return 1.0
    try:
        if condition_set:
            total_chi2_stat = 0
            total_dof = 0
            valid_strata = 0
            grouped = data.groupby(condition_set)
            for _, group_data in grouped:
                if len(group_data) < 2:
                    continue
                contingency = pd.crosstab(group_data[x], group_data[y])
                if contingency.shape[0] < 2 or contingency.shape[1] < 2:
                    continue
                if contingency.sum().sum() == 0:
                    continue
                try:
                    chi2_stat, p, dof, expected = chi2_contingency(contingency,
                        correction=False)
                    if dof > 0:
                        total_chi2_stat += chi2_stat
                        total_dof += dof
                        valid_strata += 1
                    else:
                        logger.debug(
                            f'Skipping stratum in Chi2 test due to zero DoF for {x}, {y} | {condition_set}'
                            )
                except ValueError as ve:
                    logger.warning(
                        f'Skipping stratum due to ValueError in chi2_contingency: {ve}'
                        )
                    continue
            if valid_strata == 0 or total_dof <= 0:
                logger.warning(
                    f'Insufficient data or degrees of freedom ({total_dof}) after processing strata for conditional Chi2-test({x}, {y} | {condition_set}). Returning p=1.0'
                    )
                return 1.0
            from scipy.stats import chi2
            p_value = chi2.sf(total_chi2_stat, total_dof)
        else:
            contingency = pd.crosstab(data[x], data[y])
            if contingency.size == 0 or contingency.sum().sum() == 0:
                logger.warning(
                    f'Empty contingency table for Chi2-test({x}, {y}). Returning p=1.0'
                    )
                return 1.0
            if contingency.shape[0] < 2 or contingency.shape[1] < 2:
                logger.warning(
                    f'Degenerate contingency table ({contingency.shape}) for Chi2-test({x}, {y}). Returning p=1.0'
                    )
                return 1.0
            try:
                chi2_stat, p_value, dof, expected = chi2_contingency(
                    contingency, correction=False)
                if dof <= 0:
                    logger.warning(
                        f'Zero degrees of freedom for Chi2-test({x}, {y}). Returning p=1.0'
                        )
                    return 1.0
            except ValueError as ve:
                logger.warning(
                    f'ValueError during unconditional chi2_contingency for {x}, {y}: {ve}. Returning p=1.0'
                    )
                return 1.0
        logger.debug(
            f'Chi-squared test({x}, {y} | {condition_set}) p-value: {p_value:.4f}, Chi2 Stat: {total_chi2_stat if condition_set else chi2_stat:.4f}, DoF: {total_dof if condition_set else dof}'
            )
        return np.clip(p_value, 0.0, 1.0)
    except Exception as e:
        logger.exception(
            f'Error in chi-squared test for {x}, {y} | {condition_set}: {e}')
        return 1.0


@with_exception_handling
def kernel_based_ci_test(data: pd.DataFrame, x: str, y: str, condition_set:
    list=None, kernel='rbf', gamma=None, alpha=0.1, **kwargs):
    """
    Performs kernel-based conditional independence test (HSIC variant approximation).
    Regresses X on Z and Y on Z using Kernel Ridge Regression, then tests
    independence of residuals using Hilbert-Schmidt Independence Criterion (HSIC)
    or an approximation like K-S test on residuals.

    Suitable for continuous data with potentially non-linear relationships.

    Args:
        data (pd.DataFrame): DataFrame containing the variables.
        x (str): Name of the first variable column.
        y (str): Name of the second variable column.
        condition_set (list, optional): List of conditioning variable column names. Defaults to None.
        kernel (str, optional): Kernel function for KernelRidge. Defaults to 'rbf'.
        gamma (float, optional): Gamma parameter for RBF kernel. Defaults to None (sklearn default).
        alpha (float, optional): Regularization strength for KernelRidge. Defaults to 0.1.

    Returns:
        float: The p-value of the independence test on residuals. Returns 1.0 if an error occurs.
    """
    from scipy.stats import kstest
    logger.debug(
        f'Performing Kernel CI test for {x} _||_ {y} | {condition_set}')
    required_cols = [x, y] + (condition_set or [])
    if not all(c in data.columns for c in required_cols):
        missing = [c for c in required_cols if c not in data.columns]
        logger.error(
            f'Variable(s) {missing} not found in data for Kernel CI test.')
        return 1.0
    data_clean = data[required_cols].dropna()
    if data_clean.shape[0] < 5:
        logger.warning(
            f'Insufficient non-NaN samples ({data_clean.shape[0]}) for Kernel CI test ({x}, {y} | {condition_set}). Returning p=1.0'
            )
        return 1.0
    try:
        x_data = data_clean[x].values.reshape(-1, 1)
        y_data = data_clean[y].values.reshape(-1, 1)
        if condition_set:
            z_data = data_clean[condition_set].values
            if z_data.ndim == 1:
                z_data = z_data.reshape(-1, 1)
            if z_data.shape[0] < 2:
                logger.warning(
                    f'Insufficient samples ({z_data.shape[0]}) after NaN removal for Kernel Ridge in Kernel CI test. Returning p=1.0'
                    )
                return 1.0
            z_scaler = StandardScaler()
            z_data_scaled = z_scaler.fit_transform(z_data)
            try:
                kr_x = KernelRidge(alpha=alpha, kernel=kernel, gamma=gamma)
                x_pred = kr_x.fit(z_data_scaled, x_data).predict(z_data_scaled)
                x_residual = x_data - x_pred
                kr_y = KernelRidge(alpha=alpha, kernel=kernel, gamma=gamma)
                y_pred = kr_y.fit(z_data_scaled, y_data).predict(z_data_scaled)
                y_residual = y_data - y_pred
            except np.linalg.LinAlgError as lae:
                logger.warning(
                    f'LinAlgError during KernelRidge fitting (likely singular matrix): {lae}. Returning p=1.0'
                    )
                return 1.0
            except ValueError as ve:
                logger.warning(
                    f'ValueError during KernelRidge fitting: {ve}. Returning p=1.0'
                    )
                return 1.0
        else:
            x_residual = x_data
            y_residual = y_data
        if x_residual.shape[0] < 2 or y_residual.shape[0] < 2:
            logger.warning(
                f'Insufficient residual samples ({x_residual.shape[0]}) for K-S test. Returning p=1.0'
                )
            return 1.0
        x_resid_scaled = StandardScaler().fit_transform(x_residual).ravel()
        y_resid_scaled = StandardScaler().fit_transform(y_residual).ravel()
        statistic, p_value = kstest(x_resid_scaled, y_resid_scaled)
        logger.debug(
            f'Kernel CI test ({x}, {y} | {condition_set}) K-S p-value on residuals: {p_value:.4f}, Stat: {statistic:.4f}'
            )
        return np.clip(p_value, 0.0, 1.0)
    except Exception as e:
        logger.exception(
            f'Error in kernel-based CI test for {x}, {y} | {condition_set}: {e}'
            )
        return 1.0


@with_exception_handling
def estimate_causal_effect_regression(data: pd.DataFrame, graph:
    CausalGraph, treatment: str, outcome: str, **kwargs):
    """
    Estimates causal effect using regression-based methods,
    adjusting for confounders identified in the graph.
    Note: Current implementation uses all other variables as potential confounders
    and returns the effect on the *scaled* treatment variable.
    """
    logger.debug(
        f'Estimating causal effect via regression: {treatment} -> {outcome}')
    if not isinstance(graph, CausalGraph):
        logger.error(
            'Invalid graph object passed to estimate_causal_effect_regression.'
            )
        return None
    if treatment not in data.columns or outcome not in data.columns:
        logger.error(
            f"Treatment '{treatment}' or Outcome '{outcome}' not found in data."
            )
        return None
    try:
        potential_confounders = [n for n in graph.nodes if n != treatment and
            n != outcome and n in data.columns]
        if not potential_confounders:
            logger.warning(
                f'No potential confounders identified for regression adjustment between {treatment} and {outcome}. Effect estimate might be biased.'
                )
        else:
            logger.debug(
                f'Using potential confounders for adjustment: {potential_confounders}'
                )
        features = [treatment] + potential_confounders
        X = data[features].copy()
        y = data[outcome].copy()
        combined = pd.concat([X, y], axis=1)
        combined_clean = combined.dropna()
        if combined_clean.shape[0] < max(10, len(features) + 1):
            logger.error(
                f'Insufficient non-NaN samples ({combined_clean.shape[0]}) for regression estimation.'
                )
            return None
        X_clean = combined_clean[features]
        y_clean = combined_clean[outcome]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_clean)
        X_scaled_df = pd.DataFrame(X_scaled, columns=X_clean.columns, index
            =X_clean.index)
        model = LassoCV(cv=5, random_state=42, max_iter=2000)
        model.fit(X_scaled_df, y_clean)
        try:
            treatment_index = X_clean.columns.get_loc(treatment)
        except KeyError:
            logger.error(
                f"Treatment variable '{treatment}' not found in columns after processing."
                )
            return None
        effect_on_scaled = model.coef_[treatment_index]
        r2 = model.score(X_scaled_df, y_clean)
        logger.debug(
            f'Regression estimation completed. Effect (on scaled treatment): {effect_on_scaled:.4f}, R2: {r2:.4f}'
            )
        return {'effect_on_scaled_treatment': effect_on_scaled,
            'confounders_used': potential_confounders, 'model': 'LassoCV',
            'r2_score': r2, 'intercept': model.intercept_, 'n_samples_used':
            X_scaled_df.shape[0]}
    except Exception as e:
        logger.exception(
            f'Error in causal effect estimation via regression: {e}')
        return None


@with_exception_handling
def estimate_causal_effect_matching(data: pd.DataFrame, graph: CausalGraph,
    treatment: str, outcome: str, n_neighbors=5, **kwargs):
    """
    Estimates causal effect using nearest neighbor matching on confounders.
    Assumes binary treatment (0 or 1) for simplicity in this implementation.
    Note: Current implementation uses all other variables as potential confounders.
    """
    logger.debug(
        f'Estimating causal effect via matching: {treatment} -> {outcome}')
    if not isinstance(graph, CausalGraph):
        logger.error(
            'Invalid graph object passed to estimate_causal_effect_matching.')
        return None
    if treatment not in data.columns or outcome not in data.columns:
        logger.error(
            f"Treatment '{treatment}' or Outcome '{outcome}' not found in data."
            )
        return None
    unique_treatments = data[treatment].unique()
    if not np.all(np.isin(unique_treatments, [0, 1])):
        logger.warning(
            f'Matching implementation currently assumes binary treatment (0, 1). Found values: {unique_treatments}. Results may be unreliable.'
            )
    try:
        potential_confounders = [n for n in graph.nodes if n != treatment and
            n != outcome and n in data.columns]
        if not potential_confounders:
            logger.warning(
                f'No potential confounders identified for matching between {treatment} and {outcome}. Performing simple group comparison.'
                )
        else:
            logger.debug(
                f'Using potential confounders for matching: {potential_confounders}'
                )
        cols_to_use = [treatment, outcome] + potential_confounders
        data_clean = data[cols_to_use].dropna()
        treated = data_clean[data_clean[treatment] == 1]
        control = data_clean[data_clean[treatment] == 0]
        if treated.empty or control.empty:
            logger.error(
                f'One or both treatment/control groups are empty after NaN removal ({len(treated)} treated, {len(control)} control). Cannot perform matching.'
                )
            return None
        if len(treated) < n_neighbors or len(control) < n_neighbors:
            logger.warning(
                f'Insufficient samples in treated ({len(treated)}) or control ({len(control)}) group for k={n_neighbors} neighbors after NaN removal.'
                )
            if len(control) < n_neighbors:
                logger.error(
                    f'Control group size ({len(control)}) is smaller than n_neighbors ({n_neighbors}). Cannot perform matching.'
                    )
                return None
        if not potential_confounders:
            treated_mean = treated[outcome].mean()
            control_mean = control[outcome].mean()
            ate = treated_mean - control_mean
            logger.info(f'No confounders used. Simple ATE: {ate:.4f}')
            return {'ate': ate, 'method':
                'Simple Group Mean Difference (No Confounders)',
                'n_treated': len(treated), 'n_control': len(control),
                'n_samples_used': len(data_clean)}
        scaler = StandardScaler()
        confounders_treated_scaled = scaler.fit_transform(treated[
            potential_confounders])
        confounders_control_scaled = scaler.transform(control[
            potential_confounders])
        nn = NearestNeighbors(n_neighbors=n_neighbors, metric='minkowski', p=2)
        nn.fit(confounders_control_scaled)
        distances, indices = nn.kneighbors(confounders_treated_scaled)
        effects = []
        for i in range(len(treated)):
            treated_outcome = treated.iloc[i][outcome]
            matched_control_indices_in_control_df = indices[i]
            matched_control_outcomes = control.iloc[
                matched_control_indices_in_control_df][outcome].mean()
            effect = treated_outcome - matched_control_outcomes
            effects.append(effect)
        att = np.mean(effects)
        logger.debug(f'Matching estimation completed. ATT: {att:.4f}')
        return {'att': att, 'confounders_used': potential_confounders,
            'method': f'Nearest Neighbor Matching (k={n_neighbors})',
            'n_treated': len(treated), 'n_control': len(control),
            'n_samples_used': len(data_clean)}
    except Exception as e:
        logger.exception(f'Error in causal effect estimation via matching: {e}'
            )
        return None


def estimate_counterfactual_outcome(data: pd.DataFrame, graph: CausalGraph,
    individual_data: dict, intervention: dict, **kwargs):
    """
    Estimates the outcome for an individual under a hypothetical intervention.
    Requires a causal model (e.g., from effect estimation).
    """
    logger.warning('Counterfactual outcome estimation is not yet implemented.')
    raise NotImplementedError(
        'Counterfactual outcome estimation requires a causal model and specific implementation steps.'
        )


def handle_time_series_data(data: pd.DataFrame, lag: int=1):
    """
    Prepares time-series data for causal analysis (e.g., creating lagged variables).
    """
    print(f'Handling time-series data with lag {lag}')
    data_lagged = data.copy()
    for col in data.columns:
        for i in range(1, lag + 1):
            data_lagged[f'{col}_lag{i}'] = data[col].shift(i)
    return data_lagged.dropna()
