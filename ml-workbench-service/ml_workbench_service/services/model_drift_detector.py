"""
Model Drift Detection Module

This module implements tools for detecting concept drift, feature drift, and model drift 
to trigger retraining when model performance degrades due to changing market conditions.
"""

from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime, timedelta
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64
from scipy import stats
from pydantic import BaseModel, Field

from ml_workbench_service.model_registry.model_registry_service import ModelRegistryService
from ml_workbench_service.model_registry.registry import ModelStatus
from core_foundations.utils.logger import get_logger

logger = get_logger(__name__)


class DriftMetrics(BaseModel):
    """Metrics for quantifying drift between distributions."""
    
    ks_statistic: float  # Kolmogorov-Smirnov statistic
    ks_p_value: float  # p-value for KS test
    psi: Optional[float] = None  # Population Stability Index
    js_divergence: Optional[float] = None  # Jensen-Shannon divergence
    wasserstein_distance: Optional[float] = None  # Wasserstein distance (Earth Mover's)
    hellinger_distance: Optional[float] = None  # Hellinger distance
    mean_difference: float  # Difference in means
    std_difference: float  # Difference in standard deviations
    drift_detected: bool  # Overall drift detection flag


class FeatureDriftResult(BaseModel):
    """Drift detection results for a single feature."""
    
    feature_name: str
    metrics: DriftMetrics
    drift_severity: float  # 0-1 scale indicating severity of drift
    visualization_data: Optional[Dict[str, Any]] = None


class ModelPerformanceDrift(BaseModel):
    """Model performance drift metrics."""
    
    model_id: str
    version_id: str
    metric_name: str
    reference_value: float
    current_value: float
    relative_change: float
    absolute_change: float
    drift_detected: bool
    drift_severity: float


class DriftAnalysisResult(BaseModel):
    """Complete drift analysis results."""
    
    model_id: str
    version_id: str
    analysis_time: datetime = Field(default_factory=datetime.utcnow)
    feature_drift: List[FeatureDriftResult]
    performance_drift: Optional[ModelPerformanceDrift] = None
    overall_drift_detected: bool
    overall_drift_severity: float
    recommendation: str
    retraining_required: bool
    visualization_data: Optional[Dict[str, Any]] = None


class ModelDriftDetector:
    """
    Model Drift Detector that monitors concept drift, feature drift, and model performance drift.
    Triggers retraining when significant drift is detected.
    """
    
    def __init__(self, model_registry_service: ModelRegistryService):
        """Initialize with a model registry service."""
        self.model_registry = model_registry_service
        self.drift_history: Dict[str, Dict[str, List[DriftAnalysisResult]]] = {}
        
        # Configure default thresholds
        self.default_config = {
            'ks_threshold': 0.1,  # KS statistic threshold for drift detection
            'psi_threshold': 0.2,  # PSI threshold
            'p_value_threshold': 0.05,  # p-value threshold
            'performance_drop_threshold': 0.1,  # Relative performance drop threshold
            'severity_weights': {  # Weights for calculating overall severity
                'ks_statistic': 0.3,
                'psi': 0.3,
                'performance': 0.4
            }
        }
        
    def analyze_drift(
        self,
        model_id: str,
        version_id: str,
        reference_data: pd.DataFrame,
        current_data: pd.DataFrame,
        target_column: Optional[str] = None,
        reference_predictions: Optional[np.ndarray] = None,
        current_predictions: Optional[np.ndarray] = None,
        reference_metrics: Optional[Dict[str, float]] = None,
        current_metrics: Optional[Dict[str, float]] = None,
        config: Optional[Dict[str, Any]] = None
    ) -> DriftAnalysisResult:
        """
        Analyze drift between reference data (training data or baseline period)
        and current data (new data being evaluated).
        
        Args:
            model_id: ID of the model
            version_id: ID of the model version
            reference_data: Reference/baseline data (e.g., training data)
            current_data: Current data to check for drift
            target_column: Target/label column name (if available)
            reference_predictions: Model predictions on reference data
            current_predictions: Model predictions on current data
            reference_metrics: Performance metrics on reference data
            current_metrics: Performance metrics on current data
            config: Configuration overrides for thresholds and parameters
            
        Returns:
            DriftAnalysisResult: Complete drift analysis results
        """
        try:
            # Use default config if not provided
            if config is None:
                config = self.default_config.copy()
            else:
                # Merge with defaults for any missing values
                merged_config = self.default_config.copy()
                merged_config.update(config)
                config = merged_config
                
            # Get model version from registry
            model_version = self.model_registry.get_model_version(model_id, version_id)
            
            # Analyze feature drift
            feature_drift_results = []
            for column in reference_data.columns:
                # Skip target column if specified
                if target_column and column == target_column:
                    continue
                    
                # Skip non-numeric columns for now
                if not pd.api.types.is_numeric_dtype(reference_data[column]):
                    continue
                
                # Get reference and current distributions for the feature
                ref_dist = reference_data[column].values
                curr_dist = current_data[column].values
                
                # Compute drift metrics
                drift_metrics = self._compute_drift_metrics(ref_dist, curr_dist, config)
                
                # Calculate drift severity
                severity = self._calculate_drift_severity(drift_metrics, config)
                
                # Create visualization
                viz_data = self._create_distribution_visualization(ref_dist, curr_dist, column)
                
                # Create feature drift result
                feature_drift = FeatureDriftResult(
                    feature_name=column,
                    metrics=drift_metrics,
                    drift_severity=severity,
                    visualization_data=viz_data
                )
                
                feature_drift_results.append(feature_drift)
            
            # Analyze performance drift if metrics provided
            performance_drift = None
            if reference_metrics and current_metrics:
                # Use first provided metric for performance drift analysis
                key_metric = list(reference_metrics.keys())[0]
                if key_metric in current_metrics:
                    performance_drift = self._analyze_performance_drift(
                        model_id, version_id, key_metric,
                        reference_metrics[key_metric], 
                        current_metrics[key_metric],
                        config
                    )
            
            # Calculate overall drift metrics
            feature_severities = [f.drift_severity for f in feature_drift_results]
            avg_feature_severity = sum(feature_severities) / len(feature_severities) if feature_severities else 0
            
            overall_severity = avg_feature_severity
            if performance_drift:
                # Weight feature drift and performance drift
                weights = config['severity_weights']
                overall_severity = (
                    weights['ks_statistic'] * avg_feature_severity +
                    weights['performance'] * performance_drift.drift_severity
                )
            
            # Determine if retraining is required
            overall_drift_detected = overall_severity >= 0.5  # 0.5 threshold for overall drift
            retraining_required = overall_severity >= 0.7  # 0.7 threshold for retraining
            
            # Generate recommendation
            recommendation = self._generate_recommendation(
                overall_drift_detected, retraining_required, feature_drift_results, performance_drift
            )
            
            # Create overall visualization
            overall_viz = self._create_overall_drift_visualization(
                feature_drift_results, performance_drift, overall_severity
            )
            
            # Create result object
            result = DriftAnalysisResult(
                model_id=model_id,
                version_id=version_id,
                feature_drift=feature_drift_results,
                performance_drift=performance_drift,
                overall_drift_detected=overall_drift_detected,
                overall_drift_severity=overall_severity,
                recommendation=recommendation,
                retraining_required=retraining_required,
                visualization_data=overall_viz
            )
            
            # Store result in history
            if model_id not in self.drift_history:
                self.drift_history[model_id] = {}
            if version_id not in self.drift_history[model_id]:
                self.drift_history[model_id][version_id] = []
            self.drift_history[model_id][version_id].append(result)
            
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing drift: {str(e)}")
            raise
            
    def _compute_drift_metrics(
        self, reference_dist: np.ndarray, current_dist: np.ndarray, config: Dict[str, Any]
    ) -> DriftMetrics:
        """Compute drift metrics between two distributions."""
        try:
            # Remove NaN values
            reference_dist = reference_dist[~np.isnan(reference_dist)]
            current_dist = current_dist[~np.isnan(current_dist)]
            
            # Compute Kolmogorov-Smirnov test
            ks_stat, p_value = stats.ks_2samp(reference_dist, current_dist)
            
            # Compute Population Stability Index (PSI)
            psi = self._calculate_psi(reference_dist, current_dist)
            
            # Compute differences in basic statistics
            mean_diff = abs(np.mean(current_dist) - np.mean(reference_dist))
            std_diff = abs(np.std(current_dist) - np.std(reference_dist))
            
            # Determine if drift is detected based on thresholds
            drift_detected = (
                ks_stat > config['ks_threshold'] or 
                p_value < config['p_value_threshold'] or
                psi > config['psi_threshold']
            )
            
            return DriftMetrics(
                ks_statistic=float(ks_stat),
                ks_p_value=float(p_value),
                psi=float(psi),
                mean_difference=float(mean_diff),
                std_difference=float(std_diff),
                drift_detected=drift_detected
            )
            
        except Exception as e:
            logger.error(f"Error computing drift metrics: {str(e)}")
            raise
            
    def _calculate_psi(self, reference_dist: np.ndarray, current_dist: np.ndarray, bins: int = 10) -> float:
        """Calculate Population Stability Index between two distributions."""
        try:
            # Create bins based on reference distribution
            min_val = min(np.min(reference_dist), np.min(current_dist))
            max_val = max(np.max(reference_dist), np.max(current_dist))
            
            # Add small epsilon to avoid issues with max_val == min_val
            epsilon = 1e-10
            bin_edges = np.linspace(min_val, max_val + epsilon, bins + 1)
            
            # Calculate histograms
            ref_counts, _ = np.histogram(reference_dist, bins=bin_edges)
            curr_counts, _ = np.histogram(current_dist, bins=bin_edges)
            
            # Convert to percentages and add small epsilon to avoid division by zero
            epsilon = 1e-10
            ref_pct = ref_counts / np.sum(ref_counts) + epsilon
            curr_pct = curr_counts / np.sum(curr_counts) + epsilon
            
            # Calculate PSI
            psi_values = (curr_pct - ref_pct) * np.log(curr_pct / ref_pct)
            psi = np.sum(psi_values)
            
            return float(psi)
            
        except Exception as e:
            logger.error(f"Error calculating PSI: {str(e)}")
            return 0.0
            
    def _calculate_drift_severity(self, metrics: DriftMetrics, config: Dict[str, Any]) -> float:
        """Calculate drift severity on a 0-1 scale."""
        # Normalize KS statistic (0-1)
        ks_severity = min(1.0, metrics.ks_statistic / 0.5)
        
        # Normalize PSI (0-1), considering PSI > 0.25 as severe
        psi_severity = min(1.0, metrics.psi / 0.25) if metrics.psi is not None else 0.0
        
        # Normalize p-value (1 - p_value, capped at 1.0)
        p_value_severity = min(1.0, 1.0 - metrics.ks_p_value)
        
        # Calculate weighted severity
        weights = config['severity_weights']
        severity = (
            weights['ks_statistic'] * ks_severity + 
            weights['psi'] * psi_severity +
            (1.0 - weights['ks_statistic'] - weights['psi']) * p_value_severity
        )
        
        return min(1.0, max(0.0, severity))
            
    def _analyze_performance_drift(
        self, model_id: str, version_id: str, metric_name: str,
        reference_value: float, current_value: float, config: Dict[str, Any]
    ) -> ModelPerformanceDrift:
        """Analyze performance drift for a model."""
        # Calculate changes
        absolute_change = current_value - reference_value
        
        # For metrics where higher is better (e.g., accuracy)
        # Negative change indicates performance drop
        is_higher_better = True  # Default assumption
        
        # For metrics where lower is better (e.g., error rates, loss)
        # Need to handle differently
        if metric_name.lower() in ['loss', 'mse', 'rmse', 'mae', 'error']:
            is_higher_better = False
            absolute_change = -absolute_change  # Invert so negative still means degradation
            
        # Calculate relative change
        if reference_value != 0:
            relative_change = absolute_change / abs(reference_value)
        else:
            relative_change = 0.0 if absolute_change == 0 else 1.0
            
        # Determine if drift detected
        if is_higher_better:
            drift_detected = relative_change < -config['performance_drop_threshold']
        else:
            drift_detected = relative_change > config['performance_drop_threshold']
            
        # Calculate severity (0-1 scale)
        if is_higher_better:
            severity = min(1.0, max(0.0, -relative_change * 2))  # Scale to 0-1
        else:
            severity = min(1.0, max(0.0, relative_change * 2))  # Scale to 0-1
            
        return ModelPerformanceDrift(
            model_id=model_id,
            version_id=version_id,
            metric_name=metric_name,
            reference_value=reference_value,
            current_value=current_value,
            relative_change=relative_change,
            absolute_change=absolute_change,
            drift_detected=drift_detected,
            drift_severity=severity
        )
        
    def _generate_recommendation(
        self, 
        drift_detected: bool, 
        retraining_required: bool,
        feature_drift: List[FeatureDriftResult],
        performance_drift: Optional[ModelPerformanceDrift]
    ) -> str:
        """Generate recommendation based on drift analysis."""
        if not drift_detected:
            return "No significant drift detected. Model is performing as expected."
            
        if not retraining_required:
            return (
                "Moderate drift detected but not severe enough to require immediate action. "
                "Monitor the model closely in the coming period."
            )
            
        # Get most drifted features
        feature_drift.sort(key=lambda x: x.drift_severity, reverse=True)
        top_drifted = [f.feature_name for f in feature_drift[:3]]
        
        recommendation = (
            f"Significant drift detected. Retraining is recommended. "
            f"Most affected features: {', '.join(top_drifted)}. "
        )
        
        if performance_drift:
            recommendation += (
                f"Performance degradation: {performance_drift.metric_name} has changed by "
                f"{performance_drift.relative_change:.2%}."
            )
            
        return recommendation
        
    def _create_distribution_visualization(
        self, reference_dist: np.ndarray, current_dist: np.ndarray, feature_name: str
    ) -> Dict[str, Any]:
        """Create visualization comparing distributions."""
        try:
            plt.figure(figsize=(10, 6))
            
            # Create histograms
            bins = min(20, int(np.sqrt(len(reference_dist))))
            plt.hist(reference_dist, bins=bins, alpha=0.5, label='Reference')
            plt.hist(current_dist, bins=bins, alpha=0.5, label='Current')
            
            plt.xlabel(feature_name)
            plt.ylabel('Frequency')
            plt.title(f'Distribution Comparison - {feature_name}')
            plt.legend()
            
            # Save plot to base64 string
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            img_str = base64.b64encode(buf.read()).decode('utf-8')
            plt.close()
            
            # Create visualization data
            visualization = {
                'plot_type': 'histogram',
                'base64_image': img_str,
                'feature_name': feature_name,
                'data': {
                    'reference': reference_dist.tolist()[:100],  # Limit data size
                    'current': current_dist.tolist()[:100]  # Limit data size
                }
            }
            
            return visualization
            
        except Exception as e:
            logger.error(f"Error creating distribution visualization: {str(e)}")
            return {}
            
    def _create_overall_drift_visualization(
        self,
        feature_drift: List[FeatureDriftResult],
        performance_drift: Optional[ModelPerformanceDrift],
        overall_severity: float
    ) -> Dict[str, Any]:
        """Create overall drift visualization."""
        try:
            plt.figure(figsize=(12, 8))
            
            # Sort features by severity
            sorted_features = sorted(feature_drift, key=lambda x: x.drift_severity, reverse=True)
            
            # Bar chart of top features by drift severity
            top_n = min(10, len(sorted_features))
            feature_names = [f.feature_name for f in sorted_features[:top_n]]
            severities = [f.drift_severity for f in sorted_features[:top_n]]
            
            plt.barh(feature_names, severities)
            plt.xlabel('Drift Severity')
            plt.ylabel('Feature')
            plt.title('Feature Drift Severity')
            plt.xlim(0, 1)
            
            # Add overall severity line
            plt.axvline(x=overall_severity, color='red', linestyle='--', 
                       label=f'Overall Severity: {overall_severity:.2f}')
            plt.legend()
            
            # Save plot to base64 string
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            img_str = base64.b64encode(buf.read()).decode('utf-8')
            plt.close()
            
            # Create overall visualization data
            visualization = {
                'plot_type': 'drift_summary',
                'base64_image': img_str,
                'overall_severity': overall_severity,
                'data': {
                    'feature_names': feature_names,
                    'severities': severities
                }
            }
            
            return visualization
            
        except Exception as e:
            logger.error(f"Error creating overall drift visualization: {str(e)}")
            return {}
    
    def check_for_retraining_triggers(
        self, model_id: str, version_id: str, 
        drift_threshold: float = 0.7, 
        staleness_days: int = 30
    ) -> Dict[str, Any]:
        """
        Check if a model version needs retraining based on various triggers.
        
        Args:
            model_id: ID of the model
            version_id: ID of the model version
            drift_threshold: Threshold for drift severity to trigger retraining
            staleness_days: Days after which a model is considered stale
            
        Returns:
            dict: Retraining recommendation with reasons
        """
        try:
            triggers = []
            
            # Get model version from registry
            model_version = self.model_registry.get_model_version(model_id, version_id)
            
            # Check drift history
            if (model_id in self.drift_history and 
                version_id in self.drift_history[model_id]):
                recent_drift = self.drift_history[model_id][version_id][-1]
                if recent_drift.overall_drift_severity >= drift_threshold:
                    triggers.append({
                        "type": "drift",
                        "severity": recent_drift.overall_drift_severity,
                        "details": recent_drift.recommendation
                    })
            
            # Check model staleness
            if model_version:
                days_since_creation = (datetime.utcnow() - model_version.created_at).days
                if days_since_creation > staleness_days:
                    triggers.append({
                        "type": "staleness",
                        "days": days_since_creation,
                        "details": f"Model is {days_since_creation} days old, exceeding the {staleness_days} day threshold"
                    })
            
            # Generate recommendation
            retraining_needed = len(triggers) > 0
            
            return {
                "model_id": model_id,
                "version_id": version_id,
                "retraining_needed": retraining_needed,
                "triggers": triggers,
                "recommendation": self._generate_retraining_recommendation(triggers) if triggers else "No retraining needed at this time."
            }
            
        except Exception as e:
            logger.error(f"Error checking retraining triggers: {str(e)}")
            return {
                "model_id": model_id,
                "version_id": version_id,
                "retraining_needed": False,
                "error": str(e)
            }
    
    def _generate_retraining_recommendation(self, triggers: List[Dict[str, Any]]) -> str:
        """Generate a detailed retraining recommendation."""
        if not triggers:
            return "No retraining needed at this time."
            
        trigger_texts = []
        for trigger in triggers:
            if trigger["type"] == "drift":
                trigger_texts.append(
                    f"Drift detected with severity {trigger['severity']:.2f}. " +
                    trigger["details"]
                )
            elif trigger["type"] == "staleness":
                trigger_texts.append(
                    f"Model staleness: {trigger['details']}"
                )
            else:
                trigger_texts.append(trigger["details"])
                
        recommendation = "Model retraining recommended due to: " + "; ".join(trigger_texts)
        return recommendation
