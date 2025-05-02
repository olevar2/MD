"""
ML Workbench Service API v1

This package provides API endpoints for the ML Workbench Service,
including model registry, training, serving, monitoring, and transfer learning.
"""

from ml_workbench_service.api.v1.model_registry import router as model_registry_router
from ml_workbench_service.api.v1.model_training import router as model_training_router
from ml_workbench_service.api.v1.model_serving import router as model_serving_router
from ml_workbench_service.api.v1.model_monitoring import router as model_monitoring_router

# New in Phase 7
from ml_workbench_service.api.v1.transfer_learning import router as transfer_learning_router

__all__ = [
    'model_registry_router',
    'model_training_router',
    'model_serving_router',
    'model_monitoring_router',
    'transfer_learning_router'
]
