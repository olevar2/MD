"""
Resource Allocation Policies Package

This package provides different policies for allocating resources to services.
"""

from . import fixed
from . import dynamic
from . import priority
from . import adaptive
from . import elastic

# Map policy types to allocation functions
POLICY_ALLOCATORS = {
    "fixed": fixed.allocate,
    "dynamic": dynamic.allocate,
    "priority": priority.allocate,
    "adaptive": adaptive.allocate,
    "elastic": elastic.allocate
}

__all__ = [
    'fixed',
    'dynamic',
    'priority',
    'adaptive',
    'elastic',
    'POLICY_ALLOCATORS'
]