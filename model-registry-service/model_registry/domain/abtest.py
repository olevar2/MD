"""
A/B testing functionality for model versions.
"""
from typing import Dict, List, Optional
from datetime import datetime
from pydantic import BaseModel
import random

from model_registry.domain.model import ModelStage
from model_registry.core.exceptions import ModelRegistryError, ModelVersionNotFoundError

class ABTest:
    """Represents an A/B test between model versions"""
    def __init__(
        self,
        test_id: str,
        model_id: str,
        version_ids: List[str],
        traffic_split: List[float],
        start_time: datetime,
        end_time: Optional[datetime] = None,
        status: str = "active"
    ):
        self.test_id = test_id
        self.model_id = model_id
        self.version_ids = version_ids
        self.traffic_split = traffic_split
        self.start_time = start_time
        self.end_time = end_time
        self.status = status
        self._validate()

    def _validate(self):
        if len(self.version_ids) != len(self.traffic_split):
            raise ValueError("Number of versions must match number of traffic split values")
        if abs(sum(self.traffic_split) - 1.0) > 0.0001:
            raise ValueError("Traffic split values must sum to 1.0")
        if any(p < 0 or p > 1 for p in self.traffic_split):
            raise ValueError("Traffic split values must be between 0 and 1")

    def get_version_for_request(self) -> str:
        """Get version ID for a request based on traffic split"""
        if self.status != "active":
            # Default to first version if test is not active
            return self.version_ids[0]
            
        rand = random.random()
        cumsum = 0
        for version_id, split in zip(self.version_ids, self.traffic_split):
            cumsum += split
            if rand < cumsum:
                return version_id
        return self.version_ids[-1]

    def to_dict(self) -> Dict:
        """Convert to dictionary representation"""
        return {
            "test_id": self.test_id,
            "model_id": self.model_id,
            "version_ids": self.version_ids,
            "traffic_split": self.traffic_split,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "status": self.status
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "ABTest":
        """Create from dictionary representation"""
        return cls(
            test_id=data["test_id"],
            model_id=data["model_id"],
            version_ids=data["version_ids"],
            traffic_split=data["traffic_split"],
            start_time=datetime.fromisoformat(data["start_time"]),
            end_time=datetime.fromisoformat(data["end_time"]) if data.get("end_time") else None,
            status=data["status"]
        )
