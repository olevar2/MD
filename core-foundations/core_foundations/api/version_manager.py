"""
API Version Management System.
Handles API versioning, deprecation, and compatibility across services.
"""
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel


class VersionStatus(str, Enum):
    """
    VersionStatus class that inherits from str, Enum.
    
    Attributes:
        Add attributes here
    """

    CURRENT = "current"
    DEPRECATED = "deprecated"
    SUNSET = "sunset"


class ApiVersion(BaseModel):
    """
    ApiVersion class that inherits from BaseModel.
    
    Attributes:
        Add attributes here
    """

    version: str
    status: VersionStatus
    released_date: datetime
    deprecation_date: Optional[datetime]
    sunset_date: Optional[datetime]
    breaking_changes: List[str]
    migration_guide: Optional[str]


class ApiVersionManager:
    def __init__(self):
        self._versions: Dict[str, ApiVersion] = {}
        self._current_version: str = ""

    def register_version(self, version: ApiVersion) -> None:
        """Register a new API version."""
        self._versions[version.version] = version
        if version.status == VersionStatus.CURRENT:
            self._current_version = version.version

    def get_version(self, version: str) -> Optional[ApiVersion]:
        """Get details for a specific version."""
        return self._versions.get(version)

    def get_current_version(self) -> ApiVersion:
        """Get the current API version."""
        return self._versions[self._current_version]

    def list_versions(self) -> List[ApiVersion]:
        """List all API versions."""
        return list(self._versions.values())

    def deprecate_version(
        self, version: str, deprecation_date: datetime, sunset_date: datetime
    ) -> None:
        """Mark a version as deprecated."""
        if version in self._versions:
            api_version = self._versions[version]
            api_version.status = VersionStatus.DEPRECATED
            api_version.deprecation_date = deprecation_date
            api_version.sunset_date = sunset_date

    def is_version_supported(self, version: str) -> bool:
        """Check if a version is still supported."""
        if version not in self._versions:
            return False
        api_version = self._versions[version]
        return api_version.status != VersionStatus.SUNSET
