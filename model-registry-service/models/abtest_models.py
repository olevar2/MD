"""
API request/response models for A/B testing.
"""
from typing import Dict, List, Optional
from datetime import datetime
from pydantic import BaseModel, Field

class ABTestCreate(BaseModel):
    """Request model for creating an A/B test"""
    version_ids: List[str] = Field(..., description="IDs of model versions to test")
    traffic_split: List[float] = Field(..., description="Traffic split ratios (must sum to 1.0)")
    duration_days: Optional[int] = Field(None, description="Duration of test in days")
    description: Optional[str] = Field(None, description="Description of the A/B test")
    
class ABTestUpdate(BaseModel):
    """Request model for updating an A/B test"""
    traffic_split: Optional[List[float]] = Field(None, description="Updated traffic split ratios")
    status: Optional[str] = Field(None, description="New test status (active/completed/cancelled)")

class ABTestResponse(BaseModel):
    """Response model for A/B test details"""
    test_id: str
    model_id: str
    version_ids: List[str]
    traffic_split: List[float]
    start_time: datetime
    end_time: Optional[datetime]
    status: str
    description: Optional[str]

class ABTestList(BaseModel):
    """Response model for listing A/B tests"""
    tests: List[ABTestResponse]
