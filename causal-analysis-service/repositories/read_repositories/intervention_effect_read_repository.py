"""
Intervention effect read repository.

This module provides the read repository for intervention effects.
"""
import logging
import json
import os
from typing import Dict, List, Optional, Any
from datetime import datetime

from common_lib.cqrs.repositories import ReadRepository
from causal_analysis_service.models.causal_models import InterventionEffect
from common_lib.caching.decorators import cached
from causal_analysis_service.utils.cache_factory import cache_factory

logger = logging.getLogger(__name__)


class InterventionEffectReadRepository(ReadRepository):
    """
    Read repository for intervention effects.
    """
    
    def __init__(self, storage_path: Optional[str] = None):
        """
        Initialize the repository with a storage path.
        
        Args:
            storage_path: Path to the storage directory
        """
        self.storage_path = storage_path or os.environ.get("INTERVENTION_EFFECT_STORAGE_PATH", "./data/intervention_effects")
        os.makedirs(self.storage_path, exist_ok=True)
        self.effects: Dict[str, InterventionEffect] = {}
        self.cache = cache_factory.get_cache()
    
    @cached(cache_factory.get_cache(), "intervention_effect", ttl=3600)
    async def get_by_id(self, id: str) -> Optional[InterventionEffect]:
        """
        Get an intervention effect by ID.
        
        Args:
            id: Intervention effect ID
            
        Returns:
            The intervention effect or None if not found
        """
        logger.info(f"Getting intervention effect with ID: {id}")
        
        # Check in-memory cache first
        if id in self.effects:
            logger.debug(f"Found intervention effect {id} in memory cache")
            return self.effects[id]
        
        # Check file storage
        file_path = os.path.join(self.storage_path, f"{id}.json")
        if os.path.exists(file_path):
            try:
                with open(file_path, "r") as f:
                    effect_data = json.load(f)
                
                # Convert to InterventionEffect
                effect = InterventionEffect(**effect_data)
                
                # Cache in memory
                self.effects[id] = effect
                
                logger.debug(f"Loaded intervention effect {id} from file storage")
                return effect
            except Exception as e:
                logger.error(f"Error loading intervention effect {id} from file: {e}")
        
        logger.warning(f"Intervention effect {id} not found")
        return None
    
    async def get_all(self) -> List[InterventionEffect]:
        """
        Get all intervention effects.
        
        Returns:
            List of all intervention effects
        """
        logger.info("Getting all intervention effects")
        
        effects = []
        
        # Get from file storage
        for filename in os.listdir(self.storage_path):
            if filename.endswith(".json"):
                try:
                    effect_id = filename[:-5]  # Remove .json extension
                    effect = await self.get_by_id(effect_id)
                    if effect:
                        effects.append(effect)
                except Exception as e:
                    logger.error(f"Error loading intervention effect from {filename}: {e}")
        
        return effects
    
    async def get_by_criteria(self, criteria: Dict[str, Any]) -> List[InterventionEffect]:
        """
        Get intervention effects by criteria.
        
        Args:
            criteria: Criteria to filter by
            
        Returns:
            List of intervention effects matching the criteria
        """
        logger.info(f"Getting intervention effects with criteria: {criteria}")
        
        # Get all intervention effects
        all_effects = await self.get_all()
        
        # Filter by criteria
        filtered_effects = []
        for effect in all_effects:
            match = True
            for key, value in criteria.items():
                if hasattr(effect, key):
                    if getattr(effect, key) != value:
                        match = False
                        break
                else:
                    match = False
                    break
            
            if match:
                filtered_effects.append(effect)
        
        return filtered_effects