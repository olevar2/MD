#!/usr/bin/env python3
"""
Adapter factory for ml-workbench service.
"""

from typing import Dict, List, Optional, Any

from .risk_manager_adapter import RiskManagerAdapter

class AdapterFactory:
    """
    Factory for creating adapter instances.
    """
    
    @classmethod
    def get_risk_manager_adapter(cls) -> RiskManagerAdapter:
        """Get an instance of RiskManagerAdapter."""
        return RiskManagerAdapter()
