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
from services.model_registry_service import ModelRegistryService
from models.registry import ModelStatus
from core_foundations.utils.logger import get_logger
logger = get_logger(__name__)


from ml_workbench_service.error.exceptions_bridge import (
    with_exception_handling,
    async_with_exception_handling,
    ForexTradingPlatformError,
    ServiceError,
    DataError,
    ValidationError
)

class DriftMetrics(BaseModel):
    """Metrics for quantifying drift between distributions."""
    ks_statistic: float
    ks_p_value: float
    psi: Optional[float] = None
    js_divergence: Optional[float] = None
    wasserstein_distance: Optional[float] = None
    hellinger_distance: Optional[float] = None
    mean_difference: float
    std_difference: float
    drift_detected: bool


class FeatureDriftResult(BaseModel):
    """Drift detection results for a single feature."""
    feature_name: str
    metrics: DriftMetrics
    drift_severity: float
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
        self.drift_history: Dict[str, Dict[str, List[DriftAnalysisResult]]] = {
            }
        self.default_config = {'ks_threshold': 0.1, 'psi_threshold': 0.2,
            'p_value_threshold': 0.05, 'performance_drop_threshold': 0.1,
            'severity_weights': {'ks_statistic': 0.3, 'psi': 0.3,
            'performance': 0.4}}

    @with_exception_handling
    def analyze_drift(self, model_id: str, version_id: str, reference_data:
        pd.DataFrame, current_data: pd.DataFrame, target_column: Optional[
        str]=None, reference_predictions: Optional[np.ndarray]=None,
        current_predictions: Optional[np.ndarray]=None, reference_metrics:
        Optional[Dict[str, float]]=None, current_metrics: Optional[Dict[str,
        float]]=None, config: Optional[Dict[str, Any]]=None
        ) ->DriftAnalysisResult:
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
            if config is None:
                config = self.default_config.copy()
            else:
                merged_config = self.default_config.copy()
                merged_config.update(config)
                config = merged_config
            model_version = self.model_registry.get_model_version(model_id,
                version_id)
            feature_drift_results = []
            for column in reference_data.columns:
                if target_column and column == target_column:
                    continue
                if not pd.api.types.is_numeric_dtype(reference_data[column]):
                    continue
                ref_dist = reference_data[column].values
                curr_dist = current_data[column].values
                drift_metrics = self._compute_drift_metrics(ref_dist,
                    curr_dist, config)
                severity = self._calculate_drift_severity(drift_metrics, config
                    )
                viz_data = self._create_distribution_visualization(ref_dist,
                    curr_dist, column)
                feature_drift = FeatureDriftResult(feature_name=column,
                    metrics=drift_metrics, drift_severity=severity,
                    visualization_data=viz_data)
                feature_drift_results.append(feature_drift)
            performance_drift = None
            if reference_metrics and current_metrics:
                key_metric = list(reference_metrics.keys())[0]
                if key_metric in current_metrics:
                    performance_drift = self._analyze_performance_drift(
                        model_id, version_id, key_metric, reference_metrics
                        [key_metric], current_metrics[key_metric], config)
            feature_severities = [f.drift_severity for f in
                feature_drift_results]
            avg_feature_severity = sum(feature_severities) / len(
                feature_severities) if feature_severities else 0
            overall_severity = avg_feature_severity
            if performance_drift:
                weights = config['severity_weights']
                overall_severity = weights['ks_statistic'
                    ] * avg_feature_severity + weights['performance'
                    ] * performance_drift.drift_severity
            overall_drift_detected = overall_severity >= 0.5
            retraining_required = overall_severity >= 0.7
            recommendation = self._generate_recommendation(
                overall_drift_detected, retraining_required,
                feature_drift_results, performance_drift)
            overall_viz = self._create_overall_drift_visualization(
                feature_drift_results, performance_drift, overall_severity)
            result = DriftAnalysisResult(model_id=model_id, version_id=
                version_id, feature_drift=feature_drift_results,
                performance_drift=performance_drift, overall_drift_detected
                =overall_drift_detected, overall_drift_severity=
                overall_severity, recommendation=recommendation,
                retraining_required=retraining_required, visualization_data
                =overall_viz)
            if model_id not in self.drift_history:
                self.drift_history[model_id] = {}
            if version_id not in self.drift_history[model_id]:
                self.drift_history[model_id][version_id] = []
            self.drift_history[model_id][version_id].append(result)
            return result
        except Exception as e:
            logger.error(f'Error analyzing drift: {str(e)}')
            raise

    @with_exception_handling
    def _compute_drift_metrics(self, reference_dist: np.ndarray,
        current_dist: np.ndarray, config: Dict[str, Any]) ->DriftMetrics:
        """Compute drift metrics between two distributions."""
        try:
            reference_dist = reference_dist[~np.isnan(reference_dist)]
            current_dist = current_dist[~np.isnan(current_dist)]
            ks_stat, p_value = stats.ks_2samp(reference_dist, current_dist)
            psi = self._calculate_psi(reference_dist, current_dist)
            mean_diff = abs(np.mean(current_dist) - np.mean(reference_dist))
            std_diff = abs(np.std(current_dist) - np.std(reference_dist))
            drift_detected = ks_stat > config['ks_threshold'
                ] or p_value < config['p_value_threshold'] or psi > config[
                'psi_threshold']
            return DriftMetrics(ks_statistic=float(ks_stat), ks_p_value=
                float(p_value), psi=float(psi), mean_difference=float(
                mean_diff), std_difference=float(std_diff), drift_detected=
                drift_detected)
        except Exception as e:
            logger.error(f'Error computing drift metrics: {str(e)}')
            raise

    @with_exception_handling
    def _calculate_psi(self, reference_dist: np.ndarray, current_dist: np.
        ndarray, bins: int=10) ->float:
        """Calculate Population Stability Index between two distributions."""
        try:
            min_val = min(np.min(reference_dist), np.min(current_dist))
            max_val = max(np.max(reference_dist), np.max(current_dist))
            epsilon = 1e-10
            bin_edges = np.linspace(min_val, max_val + epsilon, bins + 1)
            ref_counts, _ = np.histogram(reference_dist, bins=bin_edges)
            curr_counts, _ = np.histogram(current_dist, bins=bin_edges)
            epsilon = 1e-10
            ref_pct = ref_counts / np.sum(ref_counts) + epsilon
            curr_pct = curr_counts / np.sum(curr_counts) + epsilon
            psi_values = (curr_pct - ref_pct) * np.log(curr_pct / ref_pct)
            psi = np.sum(psi_values)
            return float(psi)
        except Exception as e:
            logger.error(f'Error calculating PSI: {str(e)}')
            return 0.0

    def _calculate_drift_severity(self, metrics: DriftMetrics, config: Dict
        [str, Any]) ->float:
        """Calculate drift severity on a 0-1 scale."""
        ks_severity = min(1.0, metrics.ks_statistic / 0.5)
        psi_severity = min(1.0, metrics.psi / 0.25
            ) if metrics.psi is not None else 0.0
        p_value_severity = min(1.0, 1.0 - metrics.ks_p_value)
        weights = config['severity_weights']
        severity = weights['ks_statistic'] * ks_severity + weights['psi'
            ] * psi_severity + (1.0 - weights['ks_statistic'] - weights['psi']
            ) * p_value_severity
        return min(1.0, max(0.0, severity))

    def _analyze_performance_drift(self, model_id: str, version_id: str,
        metric_name: str, reference_value: float, current_value: float,
        config: Dict[str, Any]) ->ModelPerformanceDrift:
        """Analyze performance drift for a model."""
        absolute_change = current_value - reference_value
        is_higher_better = True
        if metric_name.lower() in ['loss', 'mse', 'rmse', 'mae', 'error']:
            is_higher_better = False
            absolute_change = -absolute_change
        if reference_value != 0:
            relative_change = absolute_change / abs(reference_value)
        else:
            relative_change = 0.0 if absolute_change == 0 else 1.0
        if is_higher_better:
            drift_detected = relative_change < -config[
                'performance_drop_threshold']
        else:
            drift_detected = relative_change > config[
                'performance_drop_threshold']
        if is_higher_better:
            severity = min(1.0, max(0.0, -relative_change * 2))
        else:
            severity = min(1.0, max(0.0, relative_change * 2))
        return ModelPerformanceDrift(model_id=model_id, version_id=
            version_id, metric_name=metric_name, reference_value=
            reference_value, current_value=current_value, relative_change=
            relative_change, absolute_change=absolute_change,
            drift_detected=drift_detected, drift_severity=severity)

    def _generate_recommendation(self, drift_detected: bool,
        retraining_required: bool, feature_drift: List[FeatureDriftResult],
        performance_drift: Optional[ModelPerformanceDrift]) ->str:
        """Generate recommendation based on drift analysis."""
        if not drift_detected:
            return (
                'No significant drift detected. Model is performing as expected.'
                )
        if not retraining_required:
            return (
                'Moderate drift detected but not severe enough to require immediate action. Monitor the model closely in the coming period.'
                )
        feature_drift.sort(key=lambda x: x.drift_severity, reverse=True)
        top_drifted = [f.feature_name for f in feature_drift[:3]]
        recommendation = (
            f"Significant drift detected. Retraining is recommended. Most affected features: {', '.join(top_drifted)}. "
            )
        if performance_drift:
            recommendation += (
                f'Performance degradation: {performance_drift.metric_name} has changed by {performance_drift.relative_change:.2%}.'
                )
        return recommendation

    @with_exception_handling
    def _create_distribution_visualization(self, reference_dist: np.ndarray,
        current_dist: np.ndarray, feature_name: str) ->Dict[str, Any]:
        """Create visualization comparing distributions."""
        try:
            plt.figure(figsize=(10, 6))
            bins = min(20, int(np.sqrt(len(reference_dist))))
            plt.hist(reference_dist, bins=bins, alpha=0.5, label='Reference')
            plt.hist(current_dist, bins=bins, alpha=0.5, label='Current')
            plt.xlabel(feature_name)
            plt.ylabel('Frequency')
            plt.title(f'Distribution Comparison - {feature_name}')
            plt.legend()
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            img_str = base64.b64encode(buf.read()).decode('utf-8')
            plt.close()
            visualization = {'plot_type': 'histogram', 'base64_image':
                img_str, 'feature_name': feature_name, 'data': {'reference':
                reference_dist.tolist()[:100], 'current': current_dist.
                tolist()[:100]}}
            return visualization
        except Exception as e:
            logger.error(f'Error creating distribution visualization: {str(e)}'
                )
            return {}

    @with_exception_handling
    def _create_overall_drift_visualization(self, feature_drift: List[
        FeatureDriftResult], performance_drift: Optional[
        ModelPerformanceDrift], overall_severity: float) ->Dict[str, Any]:
        """Create overall drift visualization."""
        try:
            plt.figure(figsize=(12, 8))
            sorted_features = sorted(feature_drift, key=lambda x: x.
                drift_severity, reverse=True)
            top_n = min(10, len(sorted_features))
            feature_names = [f.feature_name for f in sorted_features[:top_n]]
            severities = [f.drift_severity for f in sorted_features[:top_n]]
            plt.barh(feature_names, severities)
            plt.xlabel('Drift Severity')
            plt.ylabel('Feature')
            plt.title('Feature Drift Severity')
            plt.xlim(0, 1)
            plt.axvline(x=overall_severity, color='red', linestyle='--',
                label=f'Overall Severity: {overall_severity:.2f}')
            plt.legend()
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            img_str = base64.b64encode(buf.read()).decode('utf-8')
            plt.close()
            visualization = {'plot_type': 'drift_summary', 'base64_image':
                img_str, 'overall_severity': overall_severity, 'data': {
                'feature_names': feature_names, 'severities': severities}}
            return visualization
        except Exception as e:
            logger.error(
                f'Error creating overall drift visualization: {str(e)}')
            return {}

    @with_exception_handling
    def check_for_retraining_triggers(self, model_id: str, version_id: str,
        drift_threshold: float=0.7, staleness_days: int=30) ->Dict[str, Any]:
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
            model_version = self.model_registry.get_model_version(model_id,
                version_id)
            if (model_id in self.drift_history and version_id in self.
                drift_history[model_id]):
                recent_drift = self.drift_history[model_id][version_id][-1]
                if recent_drift.overall_drift_severity >= drift_threshold:
                    triggers.append({'type': 'drift', 'severity':
                        recent_drift.overall_drift_severity, 'details':
                        recent_drift.recommendation})
            if model_version:
                days_since_creation = (datetime.utcnow() - model_version.
                    created_at).days
                if days_since_creation > staleness_days:
                    triggers.append({'type': 'staleness', 'days':
                        days_since_creation, 'details':
                        f'Model is {days_since_creation} days old, exceeding the {staleness_days} day threshold'
                        })
            retraining_needed = len(triggers) > 0
            return {'model_id': model_id, 'version_id': version_id,
                'retraining_needed': retraining_needed, 'triggers':
                triggers, 'recommendation': self.
                _generate_retraining_recommendation(triggers) if triggers else
                'No retraining needed at this time.'}
        except Exception as e:
            logger.error(f'Error checking retraining triggers: {str(e)}')
            return {'model_id': model_id, 'version_id': version_id,
                'retraining_needed': False, 'error': str(e)}

    def _generate_retraining_recommendation(self, triggers: List[Dict[str,
        Any]]) ->str:
        """Generate a detailed retraining recommendation."""
        if not triggers:
            return 'No retraining needed at this time.'
        trigger_texts = []
        for trigger in triggers:
            if trigger['type'] == 'drift':
                trigger_texts.append(
                    f"Drift detected with severity {trigger['severity']:.2f}. "
                     + trigger['details'])
            elif trigger['type'] == 'staleness':
                trigger_texts.append(f"Model staleness: {trigger['details']}")
            else:
                trigger_texts.append(trigger['details'])
        recommendation = 'Model retraining recommended due to: ' + '; '.join(
            trigger_texts)
        return recommendation
