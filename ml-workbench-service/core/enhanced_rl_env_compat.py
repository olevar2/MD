"""
Backward Compatibility Module for Enhanced RL Environment

This module provides backward compatibility for code that uses the original
EnhancedForexTradingEnv class.
"""

import warnings
from typing import Dict, List, Tuple, Optional, Any, Union, Callable

from .environment.forex_environment import ForexTradingEnvironment


class EnhancedForexTradingEnv(ForexTradingEnvironment):
    """
    Backward compatibility class for the original EnhancedForexTradingEnv.
    
    This class inherits from the new ForexTradingEnvironment and provides
    the same interface as the original EnhancedForexTradingEnv class.
    """
    
    def __init__(self, *args, **kwargs):
        """
        Initialize the backward compatibility class.
        
        Args:
            *args: Positional arguments to pass to ForexTradingEnvironment
            **kwargs: Keyword arguments to pass to ForexTradingEnvironment
        """
        warnings.warn(
            "EnhancedForexTradingEnv is deprecated and will be removed in a future version. "
            "Use ForexTradingEnvironment instead.",
            DeprecationWarning,
            stacklevel=2
        )
        super().__init__(*args, **kwargs)