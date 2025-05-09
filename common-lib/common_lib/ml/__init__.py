"""
Machine Learning package for common interfaces and models.
"""

from .interfaces import (
    ModelType,
    ModelFramework,
    IMLModelProvider,
    IRLModelTrainer
)

# Import model feedback interfaces
from .model_feedback_interfaces import (
    FeedbackSource,
    FeedbackCategory,
    FeedbackSeverity,
    ModelFeedback,
    IModelFeedbackProcessor,
    IModelTrainingFeedbackIntegrator
)

# Import existing interfaces if they exist
try:
    from .rl_effectiveness_interfaces import IRLModelEffectivenessAnalyzer
    __all__ = [
        'ModelType',
        'ModelFramework',
        'IMLModelProvider',
        'IRLModelTrainer',
        'IRLModelEffectivenessAnalyzer',
        'FeedbackSource',
        'FeedbackCategory',
        'FeedbackSeverity',
        'ModelFeedback',
        'IModelFeedbackProcessor',
        'IModelTrainingFeedbackIntegrator'
    ]
except ImportError:
    __all__ = [
        'ModelType',
        'ModelFramework',
        'IMLModelProvider',
        'IRLModelTrainer',
        'FeedbackSource',
        'FeedbackCategory',
        'FeedbackSeverity',
        'ModelFeedback',
        'IModelFeedbackProcessor',
        'IModelTrainingFeedbackIntegrator'
    ]
