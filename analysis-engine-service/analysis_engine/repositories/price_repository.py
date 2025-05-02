"""
Price Repository

This module provides repository access for price data,
supporting the market regime analysis and other price-dependent components.
"""

from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from sqlalchemy.orm import Session
from sqlalchemy import func, and_, desc

from analysis_engine.db.models import PriceData


class PriceRepository:
    """
    Repository for accessing price data in the database
    """
    
    def __init__(self, db: Session):
        self.db = db
    
    def get_prices(
        self,
        instrument: str,
        timeframe: str,
        from_date: Optional[datetime] = None,
        to_date: Optional[datetime] = None,
        limit: int = 1000
    ) -> List[PriceData]:
        """Get price data with optional date filtering"""
        query = self.db.query(PriceData)\
            .filter(PriceData.instrument == instrument)\
            .filter(PriceData.timeframe == timeframe)
        
        if from_date:
            query = query.filter(PriceData.timestamp >= from_date)
        if to_date:
            query = query.filter(PriceData.timestamp <= to_date)
        
        return query.order_by(PriceData.timestamp).limit(limit).all()
    
    def get_latest_price(
        self,
        instrument: str,
        timeframe: str
    ) -> Optional[PriceData]:
        """Get the latest price for an instrument and timeframe"""
        return self.db.query(PriceData)\
            .filter(PriceData.instrument == instrument)\
            .filter(PriceData.timeframe == timeframe)\
            .order_by(desc(PriceData.timestamp))\
            .first()
    
    def get_price_statistics(
        self,
        instrument: str,
        timeframe: str,
        from_date: Optional[datetime] = None,
        to_date: Optional[datetime] = None
    ) -> Dict[str, float]:
        """Get price statistics for a specific instrument and timeframe"""
        query = self.db.query(
            func.min(PriceData.low).label('min_price'),
            func.max(PriceData.high).label('max_price'),
            func.avg(PriceData.close).label('avg_price'),
            func.avg(PriceData.high - PriceData.low).label('avg_range')
        )\
        .filter(PriceData.instrument == instrument)\
        .filter(PriceData.timeframe == timeframe)
        
        if from_date:
            query = query.filter(PriceData.timestamp >= from_date)
        if to_date:
            query = query.filter(PriceData.timestamp <= to_date)
            
        result = query.first()
        
        if not result:
            return {
                'min_price': 0,
                'max_price': 0,
                'avg_price': 0,
                'avg_range': 0
            }
            
        return {
            'min_price': float(result.min_price) if result.min_price else 0,
            'max_price': float(result.max_price) if result.max_price else 0,
            'avg_price': float(result.avg_price) if result.avg_price else 0,
            'avg_range': float(result.avg_range) if result.avg_range else 0
        }
    
    def store_price(self, price_data: Dict[str, Any]) -> PriceData:
        """Store a new price data point"""
        price = PriceData(**price_data)
        self.db.add(price)
        self.db.commit()
        self.db.refresh(price)
        return price
    
    def batch_store_prices(self, price_data_list: List[Dict[str, Any]]) -> int:
        """Store multiple price data points in batch"""
        prices = [PriceData(**data) for data in price_data_list]
        self.db.add_all(prices)
        self.db.commit()
        return len(prices)
