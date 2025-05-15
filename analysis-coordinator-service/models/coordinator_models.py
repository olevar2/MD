from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional
from enum import Enum
from datetime import datetime
import uuid

class AnalysisServiceType(str, Enum):
    MARKET_ANALYSIS = "market_analysis"
    CAUSAL_ANALYSIS = "causal_analysis"
    BACKTESTING = "backtesting"
    FEATURE_STORE = "feature_store"
    ML_INTEGRATION = "ml_integration"

class AnalysisTaskStatusEnum(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class IntegratedAnalysisRequest(BaseModel):
    symbol: str
    timeframe: str
    start_date: datetime
    end_date: Optional[datetime] = None
    services: List[AnalysisServiceType]
    parameters: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        schema_extra = {
            "example": {
                "symbol": "EURUSD",
                "timeframe": "1h",
                "start_date": "2025-01-01T00:00:00Z",
                "end_date": "2025-01-31T23:59:59Z",
                "services": ["market_analysis", "causal_analysis"],
                "parameters": {
                    "market_analysis": {
                        "patterns": ["head_and_shoulders", "double_top"],
                        "support_resistance": True,
                        "market_regime": True
                    },
                    "causal_analysis": {
                        "variables": ["price", "volume", "volatility"],
                        "max_lag": 5
                    }
                }
            }
        }

class IntegratedAnalysisResponse(BaseModel):
    task_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    status: AnalysisTaskStatusEnum = AnalysisTaskStatusEnum.PENDING
    created_at: datetime = Field(default_factory=datetime.utcnow)
    estimated_completion_time: Optional[datetime] = None
    
    class Config:
        schema_extra = {
            "example": {
                "task_id": "123e4567-e89b-12d3-a456-426614174000",
                "status": "pending",
                "created_at": "2025-05-22T10:30:00Z",
                "estimated_completion_time": "2025-05-22T10:35:00Z"
            }
        }

class AnalysisTaskRequest(BaseModel):
    service_type: AnalysisServiceType
    symbol: str
    timeframe: str
    start_date: datetime
    end_date: Optional[datetime] = None
    parameters: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        schema_extra = {
            "example": {
                "service_type": "market_analysis",
                "symbol": "EURUSD",
                "timeframe": "1h",
                "start_date": "2025-01-01T00:00:00Z",
                "end_date": "2025-01-31T23:59:59Z",
                "parameters": {
                    "patterns": ["head_and_shoulders", "double_top"],
                    "support_resistance": True,
                    "market_regime": True
                }
            }
        }

class AnalysisTaskResponse(BaseModel):
    task_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    service_type: AnalysisServiceType
    status: AnalysisTaskStatusEnum = AnalysisTaskStatusEnum.PENDING
    created_at: datetime = Field(default_factory=datetime.utcnow)
    estimated_completion_time: Optional[datetime] = None
    
    class Config:
        schema_extra = {
            "example": {
                "task_id": "123e4567-e89b-12d3-a456-426614174000",
                "service_type": "market_analysis",
                "status": "pending",
                "created_at": "2025-05-22T10:30:00Z",
                "estimated_completion_time": "2025-05-22T10:35:00Z"
            }
        }

class AnalysisTaskStatus(BaseModel):
    task_id: str
    service_type: AnalysisServiceType
    status: AnalysisTaskStatusEnum
    created_at: datetime
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    progress: Optional[float] = None
    message: Optional[str] = None
    
    class Config:
        schema_extra = {
            "example": {
                "task_id": "123e4567-e89b-12d3-a456-426614174000",
                "service_type": "market_analysis",
                "status": "running",
                "created_at": "2025-05-22T10:30:00Z",
                "updated_at": "2025-05-22T10:32:00Z",
                "progress": 0.65,
                "message": "Processing data for EURUSD"
            }
        }

class AnalysisTaskResult(BaseModel):
    task_id: str
    service_type: AnalysisServiceType
    status: AnalysisTaskStatusEnum
    created_at: datetime
    completed_at: Optional[datetime] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    
    class Config:
        schema_extra = {
            "example": {
                "task_id": "123e4567-e89b-12d3-a456-426614174000",
                "service_type": "market_analysis",
                "status": "completed",
                "created_at": "2025-05-22T10:30:00Z",
                "completed_at": "2025-05-22T10:35:00Z",
                "result": {
                    "patterns": [
                        {"type": "head_and_shoulders", "start": "2025-01-05T00:00:00Z", "end": "2025-01-10T00:00:00Z", "confidence": 0.85},
                        {"type": "double_top", "start": "2025-01-20T00:00:00Z", "end": "2025-01-25T00:00:00Z", "confidence": 0.92}
                    ],
                    "support_resistance": [
                        {"type": "support", "level": 1.1050, "strength": 0.78},
                        {"type": "resistance", "level": 1.1200, "strength": 0.85}
                    ],
                    "market_regime": {"type": "trending", "direction": "bullish", "strength": 0.72}
                },
                "error": None
            }
        }