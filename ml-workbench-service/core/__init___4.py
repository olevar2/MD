"""
API v1 package for ML Workbench Service.
"""

from core.model_registry import router as model_registry_router
from core.model_training import router as model_training_router
from core.model_serving import router as model_serving_router
from core.model_monitoring import router as model_monitoring_router
from core.transfer_learning import router as transfer_learning_router