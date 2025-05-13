"""
Tool Effectiveness Module

This module provides interfaces and utilities for tracking and analyzing
the effectiveness of trading tools across different market conditions.
"""

from .interfaces import (
    MarketRegimeEnum,
    TimeFrameEnum,
    SignalEvent,
    SignalOutcome,
    IToolEffectivenessMetric,
    IToolEffectivenessTracker,
    IToolEffectivenessRepository
)

from .enhanced_interfaces import (
    IEnhancedToolEffectivenessTracker,
    IAdaptiveLayerService
)

__all__ = [
    'MarketRegimeEnum',
    'TimeFrameEnum',
    'SignalEvent',
    'SignalOutcome',
    'IToolEffectivenessMetric',
    'IToolEffectivenessTracker',
    'IToolEffectivenessRepository',
    'IEnhancedToolEffectivenessTracker',
    'IAdaptiveLayerService'
]
""""""
