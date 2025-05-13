"""
ML Integration for Data Reconciliation.

This module provides integration between the Data Reconciliation system and ML components.
It enables validation of ML features, detection of data drift, and monitoring of model inputs.
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union

from models.models_2 import (
    ReconciliationConfig,
    ReconciliationTask,
    ReconciliationResult,
    ReconciliationIssue,
    ReconciliationStatus,
    ReconciliationSeverity,
    ReconciliationType,
    ReconciliationRule,
    DataSourceConfig
)
from services.service_2 import ReconciliationService

logger = logging.getLogger(__name__)


class MLReconciliationService:
    """Service for ML-specific reconciliation tasks."""
    
    def __init__(self, reconciliation_service: ReconciliationService):
        """
        Initialize the service.
        
        Args:
            reconciliation_service: Base reconciliation service
        """
        self.reconciliation_service = reconciliation_service
    
    async def validate_feature_data(
        self,
        feature_name: str,
        source_data: pd.DataFrame,
        expected_data: pd.DataFrame,
        tolerance: float = 0.0001,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Validate feature data against expected values.
        
        Args:
            feature_name: Name of the feature
            source_data: Source data with the feature
            expected_data: Expected data with the feature
            tolerance: Tolerance for numerical comparisons
            metadata: Additional metadata
            
        Returns:
            Result ID
        """
        # Create configuration
        config_id = await self.reconciliation_service.create_config(
            name=f"Feature Validation - {feature_name}",
            reconciliation_type=ReconciliationType.CROSS_SOURCE,
            primary_source=DataSourceConfig(
                source_id="feature_data",
                source_type="feature",
                query_params={},
                metadata={"feature_name": feature_name}
            ),
            secondary_source=DataSourceConfig(
                source_id="expected_data",
                source_type="feature",
                query_params={},
                metadata={"feature_name": feature_name}
            ),
            rules=[
                ReconciliationRule(
                    name=f"{feature_name} Validation",
                    field=feature_name,
                    comparison_type="tolerance",
                    parameters={"tolerance": tolerance},
                    severity=ReconciliationSeverity.ERROR
                )
            ],
            description=f"Validate {feature_name} feature data",
            metadata=metadata or {"type": "feature_validation"}
        )
        
        # Schedule task
        task_id = await self.reconciliation_service.schedule_task(
            config_id=config_id,
            scheduled_time=datetime.utcnow(),
            metadata={"source": "ml_integration"}
        )
        
        # Run task
        result_id = await self.reconciliation_service.run_task(task_id)
        
        return result_id
    
    async def detect_data_drift(
        self,
        feature_names: List[str],
        current_data: pd.DataFrame,
        reference_data: pd.DataFrame,
        drift_threshold: float = 0.05,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Detect data drift between current and reference data.
        
        Args:
            feature_names: Names of features to check
            current_data: Current data
            reference_data: Reference data (e.g., training data)
            drift_threshold: Threshold for drift detection
            metadata: Additional metadata
            
        Returns:
            Result ID
        """
        # Create configuration
        config_id = await self.reconciliation_service.create_config(
            name="Data Drift Detection",
            reconciliation_type=ReconciliationType.TEMPORAL,
            primary_source=DataSourceConfig(
                source_id="current_data",
                source_type="feature",
                query_params={},
                metadata={"features": feature_names}
            ),
            secondary_source=DataSourceConfig(
                source_id="reference_data",
                source_type="feature",
                query_params={},
                metadata={"features": feature_names}
            ),
            rules=[
                ReconciliationRule(
                    name=f"{feature} Drift Detection",
                    field=feature,
                    comparison_type="distribution",
                    parameters={"threshold": drift_threshold, "method": "ks_test"},
                    severity=ReconciliationSeverity.WARNING
                )
                for feature in feature_names
            ],
            description="Detect data drift in features",
            metadata=metadata or {"type": "data_drift_detection"}
        )
        
        # Schedule task
        task_id = await self.reconciliation_service.schedule_task(
            config_id=config_id,
            scheduled_time=datetime.utcnow(),
            metadata={"source": "ml_integration"}
        )
        
        # Run task
        result_id = await self.reconciliation_service.run_task(task_id)
        
        return result_id
    
    async def validate_model_inputs(
        self,
        model_name: str,
        input_data: pd.DataFrame,
        feature_specs: Dict[str, Dict[str, Any]],
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Validate model inputs against specifications.
        
        Args:
            model_name: Name of the model
            input_data: Input data for the model
            feature_specs: Specifications for each feature
            metadata: Additional metadata
            
        Returns:
            Result ID
        """
        # Create rules based on feature specifications
        rules = []
        
        for feature, specs in feature_specs.items():
            # Range validation
            if "min_value" in specs or "max_value" in specs:
                min_value = specs.get("min_value")
                max_value = specs.get("max_value")
                
                rule = ReconciliationRule(
                    name=f"{feature} Range Validation",
                    field=feature,
                    comparison_type="range",
                    parameters={"min_value": min_value, "max_value": max_value},
                    severity=ReconciliationSeverity.ERROR
                )
                
                rules.append(rule)
            
            # Type validation
            if "data_type" in specs:
                rule = ReconciliationRule(
                    name=f"{feature} Type Validation",
                    field=feature,
                    comparison_type="type",
                    parameters={"expected_type": specs["data_type"]},
                    severity=ReconciliationSeverity.ERROR
                )
                
                rules.append(rule)
            
            # Null validation
            if "allow_null" in specs:
                rule = ReconciliationRule(
                    name=f"{feature} Null Validation",
                    field=feature,
                    comparison_type="null",
                    parameters={"allow_null": specs["allow_null"]},
                    severity=ReconciliationSeverity.ERROR
                )
                
                rules.append(rule)
        
        # Create configuration
        config_id = await self.reconciliation_service.create_config(
            name=f"Model Input Validation - {model_name}",
            reconciliation_type=ReconciliationType.CUSTOM,
            primary_source=DataSourceConfig(
                source_id="model_input",
                source_type="feature",
                query_params={},
                metadata={"model_name": model_name}
            ),
            rules=rules,
            description=f"Validate inputs for {model_name} model",
            metadata=metadata or {"type": "model_input_validation"}
        )
        
        # Schedule task
        task_id = await self.reconciliation_service.schedule_task(
            config_id=config_id,
            scheduled_time=datetime.utcnow(),
            metadata={"source": "ml_integration"}
        )
        
        # Run task
        result_id = await self.reconciliation_service.run_task(task_id)
        
        return result_id
    
    async def compare_model_predictions(
        self,
        model_a_name: str,
        model_b_name: str,
        model_a_predictions: pd.DataFrame,
        model_b_predictions: pd.DataFrame,
        tolerance: float = 0.01,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Compare predictions from two models.
        
        Args:
            model_a_name: Name of model A
            model_b_name: Name of model B
            model_a_predictions: Predictions from model A
            model_b_predictions: Predictions from model B
            tolerance: Tolerance for numerical comparisons
            metadata: Additional metadata
            
        Returns:
            Result ID
        """
        # Create configuration
        config_id = await self.reconciliation_service.create_config(
            name=f"Model Comparison - {model_a_name} vs {model_b_name}",
            reconciliation_type=ReconciliationType.CROSS_SOURCE,
            primary_source=DataSourceConfig(
                source_id=model_a_name,
                source_type="prediction",
                query_params={},
                metadata={"model_name": model_a_name}
            ),
            secondary_source=DataSourceConfig(
                source_id=model_b_name,
                source_type="prediction",
                query_params={},
                metadata={"model_name": model_b_name}
            ),
            rules=[
                ReconciliationRule(
                    name="Prediction Comparison",
                    field="prediction",
                    comparison_type="tolerance",
                    parameters={"tolerance": tolerance},
                    severity=ReconciliationSeverity.WARNING
                )
            ],
            description=f"Compare predictions from {model_a_name} and {model_b_name}",
            metadata=metadata or {"type": "model_comparison"}
        )
        
        # Schedule task
        task_id = await self.reconciliation_service.schedule_task(
            config_id=config_id,
            scheduled_time=datetime.utcnow(),
            metadata={"source": "ml_integration"}
        )
        
        # Run task
        result_id = await self.reconciliation_service.run_task(task_id)
        
        return result_id
