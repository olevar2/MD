"""
Configuration schema for the service.
"""

from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any


class ServiceConfig(BaseModel):
    """
    Service-specific configuration.
    """
    
    name: str = Field(..., description="Service name")
    version: str = Field(..., description="Service version")
    debug: bool = Field(False, description="Debug mode")
    log_level: str = Field("INFO", description="Logging level")
