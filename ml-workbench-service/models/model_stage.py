"""
Model Stage

This module defines the possible stages in a model's lifecycle.
"""

from enum import Enum

class ModelStage(Enum):
    """
    Enum representing the possible stages in a model's lifecycle.
    """
    DEVELOPMENT = "development"  # Model is under development/testing
    STAGING = "staging"          # Model is in staging/validation
    PRODUCTION = "production"    # Model is in production use
    ARCHIVED = "archived"        # Model is archived/deprecated
