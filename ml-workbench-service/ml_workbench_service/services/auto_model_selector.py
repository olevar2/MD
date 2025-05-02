"""
Automated Model Selection Module

This module implements a system to automatically select the best ML model
based on predicted market conditions and historical performance.
"""

from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime
import logging
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from ml_workbench_service.model_registry.model_registry_service import ModelRegistryService
from ml_workbench_service.model_registry.registry import ModelType, ModelStatus
from core_foundations.utils.logger import get_logger

logger = get_logger(__name__)

class MarketCondition(BaseModel):
    """Represents a market condition or regime."""
    
    condition_id: str
    name: str
    description: Optional[str] = None
    features: Dict[str, float]
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ModelPerformanceScore(BaseModel):
    """Model performance score for a given market condition."""
    
    model_id: str
    version_id: str
    market_condition_id: str
    score: float
    confidence: float
    last_evaluated: datetime = Field(default_factory=datetime.utcnow)
    metrics: Dict[str, float] = Field(default_factory=dict)


class ModelSelectionCriteria(BaseModel):
    """Criteria for selecting models."""
    
    model_type: Optional[ModelType] = None
    minimum_score: float = 0.0
    minimum_confidence: float = 0.0
    only_production_models: bool = True
    prefer_recent_models: bool = True
    ensemble_selection: bool = False
    ensemble_size: int = 3
    custom_filters: Dict[str, Any] = Field(default_factory=dict)


class AutoModelSelector:
    """
    Automated Model Selector that chooses the best model based on:
    1. Current/predicted market conditions
    2. Historical model performance in similar conditions
    3. Model freshness and status
    4. User-defined selection criteria
    """
    
    def __init__(self, model_registry_service: ModelRegistryService):
        """Initialize with a model registry service."""
        self.model_registry = model_registry_service
        self.performance_history: Dict[str, Dict[str, List[ModelPerformanceScore]]] = {}
        
    def register_market_condition(self, condition: MarketCondition) -> str:
        """Register a new market condition."""
        # In a production system, you would store this in a database
        logger.info(f"Registering market condition: {condition.name}")
        return condition.condition_id
    
    def register_performance_score(self, score: ModelPerformanceScore) -> None:
        """Register a performance score for a model under specific market conditions."""
        # In a production system, you would store this in a database
        model_id = score.model_id
        condition_id = score.market_condition_id
        
        if model_id not in self.performance_history:
            self.performance_history[model_id] = {}
        
        if condition_id not in self.performance_history[model_id]:
            self.performance_history[model_id][condition_id] = []
        
        self.performance_history[model_id][condition_id].append(score)
        logger.debug(f"Registered performance score {score.score} for model {model_id} under condition {condition_id}")
        
    def compute_model_compatibility(self, model_id: str, version_id: str, 
                                   market_condition: MarketCondition) -> float:
        """
        Compute a compatibility score between a model and a market condition.
        Higher score means the model is more suitable for this condition.
        """
        # This is a simplified implementation
        # In a real system, you would use historical performance and similarity metrics
        
        # Get performance history for this model under similar conditions
        compatibility_score = 0.0
        
        if model_id in self.performance_history:
            # Look for exact condition match
            if market_condition.condition_id in self.performance_history[model_id]:
                scores = [s.score for s in self.performance_history[model_id][market_condition.condition_id]]
                if scores:
                    compatibility_score = sum(scores) / len(scores)
            else:
                # Find most similar condition based on features
                max_similarity = -1
                most_similar_scores = []
                
                for cond_id, scores_list in self.performance_history[model_id].items():
                    # We would actually fetch condition features from a database
                    # This is a placeholder
                    similarity = 0.5  # Placeholder for feature similarity calculation
                    if similarity > max_similarity:
                        max_similarity = similarity
                        most_similar_scores = [s.score for s in scores_list]
                
                if most_similar_scores:
                    compatibility_score = (sum(most_similar_scores) / len(most_similar_scores)) * max_similarity
        
        return max(0.0, min(1.0, compatibility_score))  # Normalize to 0-1
        
    def select_best_model(self, 
                          market_condition: MarketCondition,
                          criteria: ModelSelectionCriteria = None) -> Tuple[Optional[str], Optional[str], float]:
        """
        Select the best model for the given market condition.
        
        Args:
            market_condition: The current or predicted market condition
            criteria: Selection criteria to use
            
        Returns:
            tuple: (model_id, version_id, confidence_score)
        """
        if criteria is None:
            criteria = ModelSelectionCriteria()
            
        # Fetch eligible models from registry
        filters = {}
        if criteria.model_type:
            filters["model_type"] = criteria.model_type
            
        status_filter = [ModelStatus.PRODUCTION] if criteria.only_production_models else None
        
        try:
            models, _ = self.model_registry.list_models(filters=filters)
            
            # Score all eligible models
            model_scores = []
            for model in models:
                # Get versions based on status filter
                versions, _ = self.model_registry.list_model_versions(
                    model_id=model.model_id,
                    filters={"status": status_filter} if status_filter else {}
                )
                
                # Skip if no eligible versions
                if not versions:
                    continue
                    
                # Sort by version number if preferring recent models
                if criteria.prefer_recent_models:
                    versions.sort(key=lambda v: v.version_number, reverse=True)
                
                # Evaluate each version
                for version in versions:
                    compatibility = self.compute_model_compatibility(
                        model.model_id, version.version_id, market_condition
                    )
                    
                    # Ensure minimum criteria are met
                    if compatibility >= criteria.minimum_score:
                        confidence = 0.8  # Placeholder for confidence calculation
                        if confidence >= criteria.minimum_confidence:
                            model_scores.append((model.model_id, version.version_id, compatibility, confidence))
                            
            # Handle no matching models
            if not model_scores:
                logger.warning(f"No suitable models found for market condition {market_condition.condition_id}")
                return None, None, 0.0
                
            # Sort by compatibility score
            model_scores.sort(key=lambda x: x[2], reverse=True)
            
            # Handle ensemble selection
            if criteria.ensemble_selection:
                # For now just return the top model, but this is where you'd implement ensemble logic
                # by taking the top criteria.ensemble_size models
                pass
            
            # Return best model
            best_model_id, best_version_id, compatibility, confidence = model_scores[0]
            logger.info(f"Selected model {best_model_id}:{best_version_id} with score {compatibility:.2f}")
            return best_model_id, best_version_id, confidence
            
        except Exception as e:
            logger.error(f"Error selecting best model: {str(e)}")
            return None, None, 0.0
            
    def select_ensemble(self,
                        market_condition: MarketCondition,
                        ensemble_size: int = 3,
                        criteria: ModelSelectionCriteria = None) -> List[Tuple[str, str, float]]:
        """
        Select an ensemble of models for the given market condition.
        
        Args:
            market_condition: The current or predicted market condition
            ensemble_size: Number of models to include in the ensemble
            criteria: Selection criteria to use
            
        Returns:
            list: List of (model_id, version_id, weight) tuples for the ensemble
        """
        if criteria is None:
            criteria = ModelSelectionCriteria(ensemble_selection=True, ensemble_size=ensemble_size)
        else:
            criteria.ensemble_selection = True
            criteria.ensemble_size = ensemble_size
            
        # Use the same logic as select_best_model but return multiple models
        # This is a simplified implementation
        
        try:
            # Follow similar flow to select_best_model
            models, _ = self.model_registry.list_models(filters={})
            
            # Score all eligible models (simplified)
            model_scores = []
            for model in models:
                versions, _ = self.model_registry.list_model_versions(model_id=model.model_id)
                
                if not versions:
                    continue
                    
                # Get latest version for simplicity
                version = versions[0]
                
                compatibility = self.compute_model_compatibility(
                    model.model_id, version.version_id, market_condition
                )
                
                if compatibility >= criteria.minimum_score:
                    model_scores.append((model.model_id, version.version_id, compatibility))
            
            # Sort and take top ensemble_size models
            model_scores.sort(key=lambda x: x[2], reverse=True)
            top_models = model_scores[:ensemble_size]
            
            # Calculate weights based on scores
            if top_models:
                total_score = sum(score for _, _, score in top_models)
                if total_score > 0:
                    ensemble = [(model_id, version_id, score/total_score) for model_id, version_id, score in top_models]
                    return ensemble
            
            return []
            
        except Exception as e:
            logger.error(f"Error selecting model ensemble: {str(e)}")
            return []
