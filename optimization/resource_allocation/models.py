"""
Resource Allocation Models

This module defines data models and enums used in resource allocation.
"""

from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
from datetime import datetime


class ResourceType(Enum):
    """Enum representing different resource types."""
    CPU = auto()
    MEMORY = auto()
    GPU = auto()
    STORAGE = auto()
    NETWORK = auto()


class ServicePriority(Enum):
    """Enum representing service priority levels."""
    CRITICAL = auto()
    HIGH = auto()
    MEDIUM = auto()
    LOW = auto()
    BACKGROUND = auto()


class ResourcePolicy(Enum):
    """Enum representing resource allocation policies."""
    FIXED = auto()
    DYNAMIC = auto()
    PRIORITY_BASED = auto()
    ADAPTIVE = auto()
    ELASTIC = auto()


@dataclass
class ServiceResourceConfig:
    """Data model for service resource configuration."""
    service_name: str
    priority: ServicePriority
    policy: ResourcePolicy
    resources: Dict[ResourceType, Dict[str, Union[int, float, str]]]
    constraints: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the configuration to a dictionary."""
        return {
            'service_name': self.service_name,
            'priority': self.priority.name,
            'policy': self.policy.name,
            'resources': {r.name: v for r, v in self.resources.items()},
            'constraints': self.constraints or {},
            'metadata': self.metadata or {}
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ServiceResourceConfig':
        """Create a configuration from a dictionary."""
        resources = {
            ResourceType[k]: v for k, v in data.get('resources', {}).items()
        }
        
        return cls(
            service_name=data['service_name'],
            priority=ServicePriority[data['priority']],
            policy=ResourcePolicy[data['policy']],
            resources=resources,
            constraints=data.get('constraints'),
            metadata=data.get('metadata')
        )


@dataclass
class ResourceUtilization:
    """Data model for resource utilization."""
    service_name: str
    timestamp: datetime
    cpu_usage: float  # Percentage (0-100)
    memory_usage: float  # Percentage (0-100)
    gpu_usage: Optional[float] = None  # Percentage (0-100)
    storage_usage: Optional[float] = None  # Percentage (0-100)
    network_usage: Optional[float] = None  # Mbps
    additional_metrics: Optional[Dict[str, float]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the utilization to a dictionary."""
        result = {
            'service_name': self.service_name,
            'timestamp': self.timestamp.isoformat(),
            'cpu_usage': self.cpu_usage,
            'memory_usage': self.memory_usage
        }
        
        if self.gpu_usage is not None:
            result['gpu_usage'] = self.gpu_usage
            
        if self.storage_usage is not None:
            result['storage_usage'] = self.storage_usage
            
        if self.network_usage is not None:
            result['network_usage'] = self.network_usage
            
        if self.additional_metrics:
            result['additional_metrics'] = self.additional_metrics
            
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ResourceUtilization':
        """Create a utilization from a dictionary."""
        return cls(
            service_name=data['service_name'],
            timestamp=datetime.fromisoformat(data['timestamp']),
            cpu_usage=data['cpu_usage'],
            memory_usage=data['memory_usage'],
            gpu_usage=data.get('gpu_usage'),
            storage_usage=data.get('storage_usage'),
            network_usage=data.get('network_usage'),
            additional_metrics=data.get('additional_metrics')
        )


@dataclass
class ScalingDecision:
    """Data model for scaling decisions."""
    service_name: str
    timestamp: datetime
    resource_type: ResourceType
    current_value: Union[int, float, str]
    target_value: Union[int, float, str]
    reason: str
    confidence: float  # 0.0 to 1.0
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the decision to a dictionary."""
        return {
            'service_name': self.service_name,
            'timestamp': self.timestamp.isoformat(),
            'resource_type': self.resource_type.name,
            'current_value': self.current_value,
            'target_value': self.target_value,
            'reason': self.reason,
            'confidence': self.confidence,
            'metadata': self.metadata or {}
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ScalingDecision':
        """Create a decision from a dictionary."""
        return cls(
            service_name=data['service_name'],
            timestamp=datetime.fromisoformat(data['timestamp']),
            resource_type=ResourceType[data['resource_type']],
            current_value=data['current_value'],
            target_value=data['target_value'],
            reason=data['reason'],
            confidence=data['confidence'],
            metadata=data.get('metadata')
        )


@dataclass
class ResourceAllocationResult:
    """Data model for allocation results."""
    service_name: str
    timestamp: datetime
    success: bool
    resources: Dict[ResourceType, Dict[str, Any]]
    decisions: List[ScalingDecision] = field(default_factory=list)
    error: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the result to a dictionary."""
        return {
            'service_name': self.service_name,
            'timestamp': self.timestamp.isoformat(),
            'success': self.success,
            'resources': {r.name: v for r, v in self.resources.items()},
            'decisions': [d.to_dict() for d in self.decisions],
            'error': self.error,
            'warnings': self.warnings,
            'metadata': self.metadata or {}
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ResourceAllocationResult':
        """Create a result from a dictionary."""
        resources = {
            ResourceType[k]: v for k, v in data.get('resources', {}).items()
        }
        
        decisions = [
            ScalingDecision.from_dict(d) for d in data.get('decisions', [])
        ]
        
        return cls(
            service_name=data['service_name'],
            timestamp=datetime.fromisoformat(data['timestamp']),
            success=data['success'],
            resources=resources,
            decisions=decisions,
            error=data.get('error'),
            warnings=data.get('warnings', []),
            metadata=data.get('metadata')
        )