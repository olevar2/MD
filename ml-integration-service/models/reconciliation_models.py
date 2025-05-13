"""
Database models for reconciliation results.

This module defines the database models for storing reconciliation results,
including reconciliation processes, discrepancies, and resolutions.
"""

from typing import Dict, List, Any, Optional
from datetime import datetime
import uuid
from sqlalchemy import (
    Column, String, Integer, Float, Boolean, DateTime, 
    ForeignKey, Enum, JSON, Text, Table
)
from sqlalchemy.orm import relationship
from sqlalchemy.ext.declarative import declarative_base

from common_lib.data_reconciliation import (
    ReconciliationStatus,
    ReconciliationSeverity,
    ReconciliationStrategy,
    DataSourceType,
)

Base = declarative_base()


class ReconciliationProcess(Base):
    """Database model for reconciliation processes."""
    
    __tablename__ = "reconciliation_processes"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    model_id = Column(String(255), nullable=True)
    version = Column(String(255), nullable=True)
    reconciliation_type = Column(String(50), nullable=False)
    status = Column(Enum(ReconciliationStatus), nullable=False)
    strategy = Column(Enum(ReconciliationStrategy), nullable=False)
    tolerance = Column(Float, nullable=False)
    auto_resolve = Column(Boolean, nullable=False)
    notification_threshold = Column(Enum(ReconciliationSeverity), nullable=False)
    start_time = Column(DateTime, nullable=False)
    end_time = Column(DateTime, nullable=True)
    duration_seconds = Column(Float, nullable=True)
    discrepancy_count = Column(Integer, nullable=False, default=0)
    resolution_count = Column(Integer, nullable=False, default=0)
    resolution_rate = Column(Float, nullable=False, default=0.0)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    discrepancies = relationship("Discrepancy", back_populates="reconciliation_process", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<ReconciliationProcess(id='{self.id}', status='{self.status}', discrepancy_count={self.discrepancy_count})>"


class Discrepancy(Base):
    """Database model for discrepancies found during reconciliation."""
    
    __tablename__ = "discrepancies"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    reconciliation_id = Column(String(36), ForeignKey("reconciliation_processes.id"), nullable=False)
    field_name = Column(String(255), nullable=False)
    severity = Column(Enum(ReconciliationSeverity), nullable=False)
    source_1_id = Column(String(255), nullable=False)
    source_1_value = Column(Text, nullable=True)
    source_2_id = Column(String(255), nullable=False)
    source_2_value = Column(Text, nullable=True)
    difference = Column(Float, nullable=True)
    resolved = Column(Boolean, nullable=False, default=False)
    resolution_strategy = Column(Enum(ReconciliationStrategy), nullable=True)
    resolved_value = Column(Text, nullable=True)
    resolution_time = Column(DateTime, nullable=True)
    metadata = Column(JSON, nullable=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    reconciliation_process = relationship("ReconciliationProcess", back_populates="discrepancies")
    
    def __repr__(self):
        return f"<Discrepancy(id='{self.id}', field_name='{self.field_name}', severity='{self.severity}', resolved={self.resolved})>"


class DataSourceConfig(Base):
    """Database model for data source configurations."""
    
    __tablename__ = "data_source_configs"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    source_id = Column(String(255), nullable=False, unique=True)
    name = Column(String(255), nullable=False)
    source_type = Column(Enum(DataSourceType), nullable=False)
    priority = Column(Integer, nullable=False)
    connection_string = Column(String(1024), nullable=True)
    api_url = Column(String(1024), nullable=True)
    credentials_id = Column(String(255), nullable=True)
    metadata = Column(JSON, nullable=True)
    active = Column(Boolean, nullable=False, default=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def __repr__(self):
    """
      repr  .
    
    """

        return f"<DataSourceConfig(id='{self.id}', name='{self.name}', source_type='{self.source_type}')>"


# Association table for reconciliation processes and data sources
reconciliation_data_sources = Table(
    "reconciliation_data_sources",
    Base.metadata,
    Column("reconciliation_id", String(36), ForeignKey("reconciliation_processes.id"), primary_key=True),
    Column("data_source_id", String(36), ForeignKey("data_source_configs.id"), primary_key=True),
)


# Create database tables
def create_tables(engine):
    """
    Create database tables.
    
    Args:
        engine: SQLAlchemy engine
    """
    Base.metadata.create_all(engine)
