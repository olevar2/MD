"""
Implementation of causal inference algorithms using DoWhy and EconML.
"""
import logging
import numpy as np
import pandas as pd
import networkx as nx
from typing import Dict, List, Optional, Tuple, Any
from sklearn.linear_model import LassoCV, LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

# Import DoWhy and EconML libraries
try:
    from dowhy import CausalModel
    # Default to linear regression for estimation if method not specified
    DEFAULT_DOWHY_METHOD = "backdoor.linear_regression"
    CAUSAL_LIBRARIES_AVAILABLE = True
except ImportError:
    DEFAULT_DOWHY_METHOD = None
    CAUSAL_LIBRARIES_AVAILABLE = False

try:
    from econml.dml import CausalForestDML
    ECONML_AVAILABLE = True
except ImportError:
    ECONML_AVAILABLE = False
    # Handle EconML specific import error if necessary
    pass

# Import causal discovery libraries
try:
    import cdt
    from cdt.causality.graph import PC, FCI
    cdt_available = True
except ImportError:
    cdt_available = False
    PC = None
    FCI = None
    logging.warning("CDT library not found. PC and FCI algorithms will not be available.")

try:
    import lingam
    from lingam import DirectLiNGAM, ICALiNGAM
    import networkx as nx # networkx is often used with lingam
    lingam_available = True
except ImportError:
    lingam_available = False
    DirectLiNGAM = None
    ICALiNGAM = None
    nx = None
    logging.warning("LiNGAM library not found. LiNGAM algorithms will not be available.")

# Import EconML if available for heterogeneous effect estimation
try:
    from econml.dml import CausalForestDML
    econml_available = True
except ImportError:
    CausalForestDML = None
    econml_available = False
    logging.warning("EconML library not found. estimate_heterogeneous_effect_econml will not be available.")


logger = logging.getLogger(__name__)

class CausalAlgorithms:
    """
    Applies causal inference algorithms to financial data.
    """

    def __init__(self):
        logger.info("Initializing CausalAlgorithms...")
        if not CAUSAL_LIBRARIES_AVAILABLE:
            logger.warning("DoWhy or EconML library not found. Causal inference features will be limited.")
        # No specific initialization needed for now

    def estimate_effect_dowhy(self, data: pd.DataFrame, treatment: str, outcome: str, common_causes: list[str], **kwargs) -> dict:
        """
        Estimates the causal effect using the DoWhy library.

        Args:
            data: DataFrame containing the data.
            treatment: Name of the treatment variable column.
            outcome: Name of the outcome variable column.
            common_causes: List of column names representing common causes (confounders).
            **kwargs: Additional arguments for CausalModel (e.g., graph, instruments, method_name, run_refuters).

        Returns:
            A dictionary containing the estimated causal effect and diagnostics.
        """
        if not CAUSAL_LIBRARIES_AVAILABLE or not DEFAULT_DOWHY_METHOD:
             logger.error("DoWhy library not installed. Cannot perform estimation.")
             return {"error": "DoWhy library not installed."}

        logger.info(f"Estimating causal effect (DoWhy): Treatment='{treatment}', Outcome='{outcome}'")
        try:
            # Ensure data types are suitable (e.g., numeric for regression)
            # Consider adding data validation/preprocessing steps here
            data = data.copy()
            for col in [treatment, outcome] + common_causes:
                if col not in data.columns:
                    raise ValueError(f"Column '{col}' not found in DataFrame.")
                # Attempt conversion to numeric, handling potential errors
                try:
                    data[col] = pd.to_numeric(data[col], errors='raise')
                except (ValueError, TypeError):
                    logger.warning(f"Column '{col}' could not be converted to numeric. DoWhy might fail if non-numeric types are used inappropriately.")

            # Handle potential NaN values - strategy might depend on context (e.g., imputation, removal)
            if data[[treatment, outcome] + common_causes].isnull().any().any():
                logger.warning("Data contains NaN values. Attempting to drop rows with NaNs in relevant columns.")
                data.dropna(subset=[treatment, outcome] + common_causes, inplace=True)
                if data.empty:
                    raise ValueError("Data became empty after dropping NaN values.")

            # 1. Create CausalModel
            model = CausalModel(
                data=data,
                treatment=treatment,
                outcome=outcome,
                common_causes=common_causes,
                graph=kwargs.get('graph') # Optional causal graph as NetworkX object or DOT string
            )

            # 2. Identify causal effect
            # proceed_when_unidentifiable=True allows estimation even if graph is missing/incomplete
            identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)
            logger.debug(f"Identified estimand: {identified_estimand}")

            # 3. Estimate causal effect using a suitable method
            method_name = kwargs.get("method_name", DEFAULT_DOWHY_METHOD)
            logger.info(f"Using estimation method: {method_name}")
            estimate = model.estimate_effect(
                identified_estimand,
                method_name=method_name,
                target_units=kwargs.get("target_units", "ate"), # Average Treatment Effect
                method_params=kwargs.get("method_params", {})
            )
            estimate_value = estimate.value
            logger.info(f"Estimated causal effect: {estimate_value}")

            # 4. Refute the estimate (optional but recommended)
            refutation_results = {}
            if kwargs.get("run_refuters", False): # Default to False unless explicitly requested
                logger.info("Running refutation tests...")
                try:
                    refute_placebo = model.refute_estimate(identified_estimand, estimate, method_name="placebo_treatment_refuter")
                    logger.debug(f"Placebo Treatment Refuter result: {refute_placebo}")
                    refutation_results["placebo_treatment"] = str(refute_placebo)
                except Exception as ref_err:
                    logger.warning(f"Placebo Treatment Refuter failed: {ref_err}")
                    refutation_results["placebo_treatment"] = f"Error: {ref_err}"

                try:
                    refute_random = model.refute_estimate(identified_estimand, estimate, method_name="random_common_cause")
                    logger.debug(f"Random Common Cause Refuter result: {refute_random}")
                    refutation_results["random_common_cause"] = str(refute_random)
                except Exception as ref_err:
                    logger.warning(f"Random Common Cause Refuter failed: {ref_err}")
                    refutation_results["random_common_cause"] = f"Error: {ref_err}"
                logger.info("Refutation tests complete.")

            results = {
                "treatment": treatment,
                "outcome": outcome,
                "method": f"DoWhy ({method_name})",
                "estimated_effect": estimate_value,
                "estimand": str(identified_estimand),
                "details": str(estimate),
                "refutation": refutation_results
            }

            logger.info(f"DoWhy estimation complete. Estimated effect: {estimate_value}")
            return results

        except ImportError:
             # This check is redundant due to the initial check, but kept for safety
             logger.error("DoWhy library not installed. Cannot perform estimation.")
             return {"error": "DoWhy library not installed."}
        except Exception as e:
            logger.error(f"Error during DoWhy causal effect estimation: {e}", exc_info=True)
            return {"error": str(e), "details": traceback.format_exc()}

    def estimate_heterogeneous_effect_econml(
        self,
        data: pd.DataFrame,
        treatment: str,
        outcome: str,
        features: List[str],
        controls: Optional[List[str]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Estimates heterogeneous treatment effects using the EconML library,
        primarily focusing on Double Machine Learning (DML) approaches like CausalForestDML.

        Args:
            data: DataFrame containing the data.
            treatment: Name of the treatment variable column.
            outcome: Name of the outcome variable column.
            features: List of column names for features (X) potentially influencing heterogeneity.
                      These variables are used to model how the treatment effect varies.
            controls: Optional list of column names for control variables (W).
                      These are confounders that affect both treatment and outcome.
            **kwargs: Additional arguments for the EconML estimator.
                      Supported kwargs include:
                      - estimator: Name of the EconML estimator (default: "CausalForestDML").
                      - model_y: Model for the outcome nuisance function (default: RandomForestRegressor).
                      - model_t: Model for the treatment nuisance function (default: RandomForestRegressor).
                      - n_estimators: Number of trees for forest-based models (default: 100).
                      - min_samples_leaf: Min samples per leaf for forest models (default: 10).
                      - cv: Number of cross-fitting folds (default: 5).
                      - discrete_treatment: Whether the treatment is discrete (default: False).
                      - Other parameters specific to the chosen EconML estimator.

        Returns:
            A dictionary containing the estimated heterogeneous effects, model details,
            and potentially feature importances.
        """
        if not ECONML_AVAILABLE:
            logger.error("EconML library not installed. Cannot perform heterogeneous effect estimation.")
            return {"error": "EconML library not installed."}

        logger.info(f"Estimating heterogeneous effects (EconML): Treatment='{treatment}', Outcome='{outcome}'")
        try:
            data = data.copy()
            required_cols = [outcome, treatment] + features + (controls if controls else [])

            # --- Data Validation and Preparation ---
            for col in required_cols:
                if col not in data.columns:
                    raise ValueError(f"Column '{col}' not found in DataFrame.")
                try:
                    # Ensure numeric types where expected (outcome, treatment, features, controls)
                    data[col] = pd.to_numeric(data[col], errors='raise')
                except (ValueError, TypeError):
                    logger.warning(f"Column '{col}' could not be converted to numeric. EconML might fail.")

            if data[required_cols].isnull().any().any():
                logger.warning("Data contains NaN values in required columns. Dropping rows with NaNs.")
                data.dropna(subset=required_cols, inplace=True)
                if data.empty:
                    raise ValueError("Data became empty after dropping NaN values.")

            # Prepare data splits (Y, T, X, W)
            Y = data[outcome]
            T = data[treatment]
            X = data[features]
            W = data[controls] if controls else None

            # Standardize features (X) and controls (W) - often improves performance
            scaler_X = StandardScaler()
            X_scaled = scaler_X.fit_transform(X)
            X_scaled_df = pd.DataFrame(X_scaled, index=X.index, columns=X.columns)

            W_scaled_df = None
            if W is not None:
                scaler_W = StandardScaler()
                W_scaled = scaler_W.fit_transform(W)
                W_scaled_df = pd.DataFrame(W_scaled, index=W.index, columns=W.columns)

            # --- Estimator Initialization ---
            estimator_name = kwargs.get("estimator", "CausalForestDML")
            logger.info(f"Using EconML estimator: {estimator_name}")

            # Default models (can be overridden by kwargs)
            default_model_y = RandomForestRegressor(n_estimators=kwargs.get("n_estimators", 100), min_samples_leaf=kwargs.get("min_samples_leaf", 10), random_state=42)
            default_model_t = RandomForestRegressor(n_estimators=kwargs.get("n_estimators", 100), min_samples_leaf=kwargs.get("min_samples_leaf", 10), random_state=42)

            if estimator_name == "CausalForestDML":
                est = CausalForestDML(
                    model_y=kwargs.get("model_y", default_model_y),
                    model_t=kwargs.get("model_t", default_model_t),
                    discrete_treatment=kwargs.get("discrete_treatment", False),
                    n_estimators=kwargs.get("n_estimators", 100),
                    min_samples_leaf=kwargs.get("min_samples_leaf", 10),
                    max_depth=kwargs.get("max_depth", None),
                    cv=kwargs.get("cv", TimeSeriesSplit(n_splits=kwargs.get("n_splits", 5))), # Use TimeSeriesSplit for time series data
                    random_state=kwargs.get("random_state", 42),
                    # Pass other CausalForestDML specific params from kwargs
                    **{k: v for k, v in kwargs.items() if k in CausalForestDML.__init__.__code__.co_varnames}
                )
            # Add other estimators here (e.g., LinearDML, NonParamDML) if needed
            # elif estimator_name == "LinearDML":
            #     est = LinearDML(...)
            else:
                raise ValueError(f"Unsupported EconML estimator: {estimator_name}")

            # --- Model Fitting ---
            logger.info("Fitting the EconML model...")
            # Use scaled features and controls for fitting
            est.fit(Y, T, X=X_scaled_df, W=W_scaled_df, inference="auto") # Use 'auto' or specify inference method
            logger.info("Model fitting complete.")

            # --- Effect Estimation ---
            # Estimate effects using the original (or scaled) features X
            # Using scaled X as the model was trained on it
            marginal_effects = est.effect(X_scaled_df)
            average_effect = np.mean(marginal_effects)
            logger.info(f"Estimated average heterogeneous effect: {average_effect:.4f}")

            # Get feature importances if available
            feature_importances = None
            if hasattr(est, 'feature_importances_'):
                try:
                    # Use the feature names from the scaled DataFrame
                    feature_importances = dict(zip(X_scaled_df.columns, est.feature_importances_()))
                    logger.debug(f"Feature importances: {feature_importances}")
                except Exception as fe_err:
                    logger.warning(f"Could not retrieve feature importances: {fe_err}")

            # --- Results --- 
            results = {
                "treatment": treatment,
                "outcome": outcome,
                "method": f"EconML ({estimator_name})",
                "average_heterogeneous_effect": float(average_effect),
                "estimator_details": str(est),
                "feature_importances": feature_importances,
                "n_samples_used": len(data),
                # Optionally return the fitted estimator itself
                # "model": est
            }

            logger.info(f"EconML estimation complete. Average effect: {average_effect:.4f}")
            return results

        except ImportError:
            # Redundant check, but safe
            logger.error("EconML library not installed. Cannot perform estimation.")
            return {"error": "EconML library not installed."}
        except Exception as e:
            logger.error(f"Error during EconML causal effect estimation: {e}", exc_info=True)
            return {"error": str(e), "details": traceback.format_exc()}

    # Removed redundant chi_squared_test - it exists in causal/algorithms.py

    def pc_algorithm(data: pd.DataFrame, alpha: float = 0.05, ci_test: str = 'gaussian', verbose: bool = False, **kwargs) -> Optional[nx.DiGraph]:
        """Performs causal discovery using the PC algorithm.

        Args:
            data (pd.DataFrame): Observational data (variables as columns).
            alpha (float): Significance level for conditional independence tests.
            ci_test (str): The conditional independence test to use ('gaussian', 'fisherz', etc.).
                           Refer to CDT documentation for available tests.
            verbose (bool): If True, prints detailed output during execution.
            **kwargs: Additional arguments passed to the cdt.causality.graph.PC algorithm.

        Returns:
            Optional[nx.DiGraph]: The estimated causal graph (Directed Acyclic Graph) as a NetworkX object,
                                 or None if CDT is not available or an error occurs.
        """
        if not cdt_available:
            logger.error("CDT library is not installed. Cannot run PC algorithm.")
            return None

        logger.info(f"Running PC algorithm with alpha={alpha}, ci_test={ci_test}")
        try:
            # Ensure data is in the correct format (Pandas DataFrame)
            if not isinstance(data, pd.DataFrame):
                logger.warning("Input data is not a Pandas DataFrame. Attempting conversion.")
                data = pd.DataFrame(data)

            # Initialize PC algorithm
            # Available CI tests might depend on the CDT version and installed dependencies (like R)
            # Common tests: 'gaussian', 'fisherz', 'gsq' (requires R/pcalg)
            model_pc = PC(CItest=ci_test, alpha=alpha, verbose=verbose, **kwargs)

            # Learn the causal graph
            # The predict method returns a NetworkX DiGraph
            graph = model_pc.predict(data)

            logger.info("PC algorithm finished successfully.")
            return graph

        except ImportError as ie:
            logger.error(f"ImportError during PC execution, possibly missing R dependencies for CI test '{ci_test}': {ie}", exc_info=True)
            return None
        except Exception as e:
            logger.error(f"Error running PC algorithm: {e}", exc_info=True)
            return None

    def fci_algorithm(data: pd.DataFrame, alpha: float = 0.05, ci_test: str = 'gaussian', verbose: bool = False, **kwargs) -> Optional[nx.DiGraph]:
        """Performs causal discovery using the FCI algorithm (handles latent confounders).

        Args:
            data (pd.DataFrame): Observational data (variables as columns).
            alpha (float): Significance level for conditional independence tests.
            ci_test (str): The conditional independence test to use ('gaussian', 'fisherz', etc.).
                           Refer to CDT documentation for available tests.
            verbose (bool): If True, prints detailed output during execution.
            **kwargs: Additional arguments passed to the cdt.causality.graph.FCI algorithm.

        Returns:
            Optional[nx.DiGraph]: The estimated causal graph (Partial Ancestral Graph - PAG) as a NetworkX object,
                                 representing equivalence classes. Edges have specific meanings:
                                 - o-> : Possible causal direction
                                 - <-> : Confounding arc
                                 - o-o : Undetermined relationship (confounding or causal)
                                 - --> : Determined causal direction
                                 Returns None if CDT is not available or an error occurs.
        """
        if not cdt_available:
            logger.error("CDT library is not installed. Cannot run FCI algorithm.")
            return None

        logger.info(f"Running FCI algorithm with alpha={alpha}, ci_test={ci_test}")
        try:
            # Ensure data is in the correct format (Pandas DataFrame)
            if not isinstance(data, pd.DataFrame):
                logger.warning("Input data is not a Pandas DataFrame. Attempting conversion.")
                data = pd.DataFrame(data)

            # Initialize FCI algorithm
            model_fci = FCI(CItest=ci_test, alpha=alpha, verbose=verbose, **kwargs)

            # Learn the causal graph (PAG)
            graph = model_fci.predict(data)

            logger.info("FCI algorithm finished successfully.")
            return graph

        except ImportError as ie:
            logger.error(f"ImportError during FCI execution, possibly missing R dependencies for CI test '{ci_test}': {ie}", exc_info=True)
            return None
        except Exception as e:
            logger.error(f"Error running FCI algorithm: {e}", exc_info=True)
            return None

    def lingam_algorithm(data: pd.DataFrame, measure: str = 'pwling', algorithm: str = 'direct', verbose: bool = False, **kwargs) -> Optional[Dict[str, Any]]:
        """Performs causal discovery using LiNGAM algorithms (DirectLiNGAM or ICALiNGAM).

        Assumes linear non-Gaussian acyclic models.

        Args:
            data (pd.DataFrame): Observational data (variables as columns).
            measure (str): Measure to evaluate independence for DirectLiNGAM ('pwling', 'kernel').
                           Defaults to 'pwling'. Ignored if algorithm='ica'.
            algorithm (str): Which LiNGAM variant to use ('direct' or 'ica'). Defaults to 'direct'.
            verbose (bool): If True, prints detailed output during execution (if supported by the algorithm).
            **kwargs: Additional arguments passed to the chosen LiNGAM algorithm constructor.

        Returns:
            Optional[Dict[str, Any]]: A dictionary containing:
                - 'causal_order': List of variable indices in causal order.
                - 'adjacency_matrix': The estimated weighted adjacency matrix (NumPy array).
                - 'graph': NetworkX DiGraph representation of the adjacency matrix (optional).
              Returns None if LiNGAM is not available or an error occurs.
        """
        if not lingam_available:
            logger.error("LiNGAM library is not installed. Cannot run LiNGAM algorithm.")
            return None

        logger.info(f"Running LiNGAM algorithm (variant: {algorithm}, measure: {measure if algorithm=='direct' else 'N/A'})")
        try:
            # Ensure data is in the correct format (Pandas DataFrame or NumPy array)
            if isinstance(data, pd.DataFrame):
                column_names = data.columns.tolist()
                data_np = data.values
            elif isinstance(data, np.ndarray):
                data_np = data
                column_names = [f'x{i}' for i in range(data.shape[1])]
            else:
                logger.error("Input data must be a Pandas DataFrame or NumPy array.")
                return None

            # Initialize LiNGAM algorithm
            if algorithm.lower() == 'direct':
                model = DirectLiNGAM(measure=measure, **kwargs)
            elif algorithm.lower() == 'ica':
                model = ICALiNGAM(**kwargs)
            else:
                logger.error(f"Unsupported LiNGAM algorithm: {algorithm}. Choose 'direct' or 'ica'.")
                return None

            # Fit the model
            model.fit(data_np)

            # Extract results
            causal_order = model.causal_order_
            adjacency_matrix = model.adjacency_matrix_

            results = {
                'causal_order': causal_order,
                'adjacency_matrix': adjacency_matrix
            }

            # Create NetworkX graph if networkx is available
            if nx:
                try:
                    # Create graph from adjacency matrix, mapping indices to column names
                    adj_df = pd.DataFrame(adjacency_matrix, index=column_names, columns=column_names)
                    # Create graph from the pandas DataFrame to preserve edge weights
                    graph = nx.from_pandas_adjacency(adj_df, create_using=nx.DiGraph())
                    results['graph'] = graph
                except Exception as graph_err:
                    logger.warning(f"Could not create NetworkX graph from LiNGAM results: {graph_err}")
                    results['graph'] = None
            else:
                 results['graph'] = None

            logger.info("LiNGAM algorithm finished successfully.")
            return results

        except Exception as e:
            logger.error(f"Error running LiNGAM algorithm: {e}", exc_info=True)
            return None

    # --- Heterogeneous Treatment Effect Estimation ---

    def estimate_heterogeneous_effect_econml(
        data: pd.DataFrame,
        outcome_col: str,
        treatment_col: str,
        covariate_cols: list[str],
        instrument_col: Optional[str] = None, # Optional: For IV estimation
        n_splits: int = 5,
        n_estimators: int = 100,
        min_samples_leaf: int = 10,
        max_depth: Optional[int] = 10,
        random_state: Optional[int] = None
    ) -> Optional[Dict[str, Any]]:
        """Estimates heterogeneous treatment effects using EconML's CausalForestDML.

        Args:
            data (pd.DataFrame): DataFrame containing outcome, treatment, covariates, and optionally instrument.
            outcome_col (str): Name of the outcome variable column.
            treatment_col (str): Name of the treatment variable column.
            covariate_cols (list[str]): List of names of covariate columns (features W).
            instrument_col (Optional[str]): Name of the instrumental variable column (Z). If None, assumes unconfoundedness.
            n_splits (int): Number of cross-fitting folds for DML.
            n_estimators (int): Number of trees in the causal forest.
            min_samples_leaf (int): Minimum number of samples per leaf in the forest trees.
            max_depth (Optional[int]): Maximum depth of the trees in the forest.
            random_state (Optional[int]): Random seed for reproducibility.

        Returns:
            Optional[Dict[str, Any]]: A dictionary containing:
                - 'average_treatment_effect': The estimated average treatment effect (ATE).
                - 'conditional_effects': Estimated effects for each sample (CATE).
                - 'feature_importances': Importance of covariates in explaining effect heterogeneity (if available).
                - 'model_details': Information about the fitted CausalForestDML model.
              Returns None if EconML is not available or an error occurs.
        """
        if not econml_available:
            logger.error("EconML library is not installed. Cannot estimate heterogeneous effects.")
            return None

        logger.info("Estimating heterogeneous treatment effects using CausalForestDML.")

        try:
            # --- 1. Data Preparation ---
            df = data.copy()

            # Check for required columns
            required_cols = [outcome_col, treatment_col] + covariate_cols
            if instrument_col:
                required_cols.append(instrument_col)
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                logger.error(f"Missing required columns in data: {missing_cols}")
                return None

            # Handle NaNs (simple imputation for demonstration, consider more robust methods)
            if df[required_cols].isnull().any().any():
                logger.warning("NaN values found in required columns. Filling with median/mode.")
                for col in required_cols:
                    if df[col].isnull().any():
                        if pd.api.types.is_numeric_dtype(df[col]):
                            df[col] = df[col].fillna(df[col].median())
                        else:
                            df[col] = df[col].fillna(df[col].mode()[0])

            Y = df[outcome_col].values
            T = df[treatment_col].values
            W = df[covariate_cols].values
            Z = df[instrument_col].values if instrument_col else None

            # Standardize covariates (often beneficial for forest-based methods)
            scaler_W = StandardScaler()
            W_scaled = scaler_W.fit_transform(W)

            # --- 2. Model Initialization ---
            # Use TimeSeriesSplit for cross-fitting if data has temporal order
            # Adjust n_splits if dataset size is small
            actual_n_splits = min(n_splits, len(df) // 2) # Ensure at least 2 samples per split
            if actual_n_splits < 2:
                 logger.error(f"Insufficient data for {n_splits} splits. Need at least {2*n_splits} samples.")
                 return None
            if actual_n_splits != n_splits:
                logger.warning(f"Reduced n_splits from {n_splits} to {actual_n_splits} due to data size.")

            cv = TimeSeriesSplit(n_splits=actual_n_splits)

            # Initialize CausalForestDML
            # Models for E[Y|X] (model_y) and E[T|X] (model_t)
            # Default is GradientBoostingRegressor, which often works well.
            est = CausalForestDML(
                n_estimators=n_estimators,
                min_samples_leaf=min_samples_leaf,
                max_depth=max_depth,
                cv=cv,
                discrete_treatment=pd.api.types.is_categorical_dtype(T) or len(np.unique(T)) < 10, # Heuristic
                random_state=random_state
            )

            # --- 3. Model Fitting ---
            logger.info("Fitting CausalForestDML model...")
            if Z is not None:
                est.fit(Y, T, X=W_scaled, W=None, Z=Z) # Use IV
            else:
                est.fit(Y, T, X=W_scaled, W=None) # Assume unconfoundedness
            logger.info("Model fitting complete.")

            # --- 4. Effect Estimation ---
            ate = est.ate(X=W_scaled)
            conditional_effects = est.effect(X=W_scaled)

            # --- 5. Feature Importances (for heterogeneity) ---
            try:
                # Access feature importances from the final model trained to predict effects
                feature_importances = est.feature_importances_
                importance_dict = dict(zip(covariate_cols, feature_importances))
            except AttributeError:
                logger.warning("Feature importances not available for this CausalForestDML configuration.")
                importance_dict = None

            # --- 6. Results ---
            results = {
                'average_treatment_effect': ate,
                'conditional_effects': conditional_effects,
                'feature_importances': importance_dict,
                'model_details': {
                    'model_type': 'CausalForestDML',
                    'n_estimators': n_estimators,
                    'min_samples_leaf': min_samples_leaf,
                    'max_depth': max_depth,
                    'n_splits': actual_n_splits,
                    'instrument_used': instrument_col is not None,
                    'random_state': random_state
                }
            }

            logger.info(f"Estimated Average Treatment Effect (ATE): {ate:.4f}")
            return results

        except Exception as e:
            logger.error(f"Error estimating heterogeneous effects with EconML: {e}", exc_info=True)
            return None

    # --- Other Potential Causal Inference Functions (Placeholders) ---

    def estimate_average_treatment_effect(*args, **kwargs):
        """Placeholder for estimating Average Treatment Effect (ATE) using various methods."""
        logger.warning("estimate_average_treatment_effect is not fully implemented.")
        raise NotImplementedError("Function to estimate ATE needs implementation (e.g., using propensity scores, regression adjustment, etc.)")

    def sensitivity_analysis(*args, **kwargs):
        """Placeholder for performing sensitivity analysis on causal estimates."""
        logger.warning("sensitivity_analysis is not fully implemented.")
        raise NotImplementedError("Function for sensitivity analysis (e.g., Rosenbaum bounds, omitted variable bias) needs implementation.")

    # Add traceback for better error reporting
