"""
Centralized configuration management system with hierarchical overrides and validation.
"""
from dataclasses import dataclass
from datetime import datetime
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml
from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)

@dataclass
class ConfigAuditEntry:
    timestamp: datetime
    user: str
    component: str
    changes: Dict[str, Any]
    previous_values: Dict[str, Any]

class ConfigManager:
    def __init__(self, base_config_path: str):
        self.base_config_path = Path(base_config_path)
        self.config_cache = {}
        self.audit_log: List[ConfigAuditEntry] = []
        self._load_base_configs()

    def _load_base_configs(self):
        """Load all base configuration files."""
        for config_file in self.base_config_path.glob("**/*.yaml"):
            component = str(config_file.relative_to(self.base_config_path).parent)
            try:
                with open(config_file, 'r') as f:
                    self.config_cache[component] = yaml.safe_load(f)
            except Exception as e:
                logger.error(f"Failed to load config for {component}: {e}")

    async def get_config(self, component: str, environment: str) -> Dict[str, Any]:
        """Get configuration with environment-specific overrides."""
        base_config = self.config_cache.get(component, {})
        env_config = self._load_environment_config(component, environment)
        
        # Deep merge base config with environment overrides
        return self._deep_merge(base_config, env_config)

    def _load_environment_config(self, component: str, environment: str) -> Dict[str, Any]:
        """Load environment-specific configuration."""
        env_file = self.base_config_path / component / f"{environment}.yaml"
        if not env_file.exists():
            return {}
            
        try:
            with open(env_file, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Failed to load environment config for {component}/{environment}: {e}")
            return {}

    def _deep_merge(self, base: Dict, override: Dict) -> Dict:
        """Deep merge two configuration dictionaries."""
        result = base.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        return result

    async def validate_config(self, component: str, config: Dict[str, Any], 
                            schema_class: BaseModel) -> tuple[bool, Optional[str]]:
        """Validate configuration against a Pydantic schema."""
        try:
            schema_class(**config)
            return True, None
        except ValidationError as e:
            return False, str(e)

    async def update_config(self, component: str, changes: Dict[str, Any], 
                          user: str, validate: bool = True) -> Dict[str, Any]:
        """Update configuration with audit trail."""
        current_config = self.config_cache.get(component, {})
        previous_values = {k: current_config.get(k) for k in changes.keys()}
        
        # Create audit entry
        audit_entry = ConfigAuditEntry(
            timestamp=datetime.utcnow(),
            user=user,
            component=component,
            changes=changes,
            previous_values=previous_values
        )
        
        # Apply changes
        new_config = self._deep_merge(current_config, changes)
        
        # Store audit entry
        self.audit_log.append(audit_entry)
        
        # Update cache
        self.config_cache[component] = new_config
        
        return new_config

    async def get_audit_trail(self, component: Optional[str] = None, 
                            start_time: Optional[datetime] = None,
                            end_time: Optional[datetime] = None) -> List[ConfigAuditEntry]:
        """Get configuration audit trail with optional filters."""
        filtered_log = self.audit_log

        if component:
            filtered_log = [entry for entry in filtered_log 
                          if entry.component == component]

        if start_time:
            filtered_log = [entry for entry in filtered_log 
                          if entry.timestamp >= start_time]

        if end_time:
            filtered_log = [entry for entry in filtered_log 
                          if entry.timestamp <= end_time]

        return filtered_log
