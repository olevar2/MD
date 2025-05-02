"""
Tool Effectiveness Repository

This module provides persistence and retrieval functionality for tool
effectiveness data, supporting both in-memory and database storage options.
"""

from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import logging
import json
import aioredis
import motor.motor_asyncio
from bson import ObjectId

from analysis_engine.services.tool_effectiveness import (
    MarketRegime,
    PredictionResult,
    ToolEffectivenessTracker
)

logger = logging.getLogger(__name__)

class ToolEffectivenessRepository:
    """
    Repository for storing and retrieving tool effectiveness data
    
    Features:
    - Multiple storage backends (in-memory, Redis, MongoDB)
    - Async operations
    - Data persistence
    - Efficient querying
    - Automatic cleanup
    """
    
    def __init__(
        self,
        storage_type: str = "memory",
        redis_url: Optional[str] = None,
        mongo_url: Optional[str] = None,
        cleanup_interval: int = 86400  # 24 hours
    ):
        """
        Initialize repository
        
        Args:
            storage_type: Type of storage backend ("memory", "redis", "mongo")
            redis_url: Optional Redis connection URL
            mongo_url: Optional MongoDB connection URL
            cleanup_interval: Interval for data cleanup in seconds
        """
        self.storage_type = storage_type
        self.redis_url = redis_url
        self.mongo_url = mongo_url
        self.cleanup_interval = cleanup_interval
        self.logger = logging.getLogger(f"{__name__}.ToolEffectivenessRepository")
        
        # In-memory storage
        self.memory_storage: Dict[str, Any] = {
            "predictions": {},
            "effectiveness": {},
            "metadata": {}
        }
        
        # Connections
        self.redis: Optional[aioredis.Redis] = None
        self.mongo: Optional[motor.motor_asyncio.AsyncIOMotorClient] = None
        
    async def initialize(self):
        """Initialize storage backend connections"""
        try:
            if self.storage_type == "redis" and self.redis_url:
                self.redis = await aioredis.create_redis_pool(self.redis_url)
                self.logger.info("Connected to Redis")
                
            elif self.storage_type == "mongo" and self.mongo_url:
                self.mongo = motor.motor_asyncio.AsyncIOMotorClient(self.mongo_url)
                self.logger.info("Connected to MongoDB")
                
        except Exception as e:
            self.logger.error(f"Error initializing storage: {str(e)}")
            raise
            
    async def cleanup(self):
        """Cleanup connections and resources"""
        try:
            if self.redis:
                self.redis.close()
                await self.redis.wait_closed()
                self.logger.info("Closed Redis connection")
                
        except Exception as e:
            self.logger.error(f"Error during cleanup: {str(e)}")
            
    async def store_prediction(self, prediction: PredictionResult):
        """
        Store a prediction result
        
        Args:
            prediction: Prediction result to store
        """
        try:
            if self.storage_type == "memory":
                await self._store_prediction_memory(prediction)
            elif self.storage_type == "redis":
                await self._store_prediction_redis(prediction)
            elif self.storage_type == "mongo":
                await self._store_prediction_mongo(prediction)
                
        except Exception as e:
            self.logger.error(f"Error storing prediction: {str(e)}")
            
    async def _store_prediction_memory(self, prediction: PredictionResult):
        """Store prediction in memory"""
        if prediction.tool_id not in self.memory_storage["predictions"]:
            self.memory_storage["predictions"][prediction.tool_id] = []
            
        self.memory_storage["predictions"][prediction.tool_id].append(
            prediction.to_dict()
        )
        
    async def _store_prediction_redis(self, prediction: PredictionResult):
        """Store prediction in Redis"""
        if not self.redis:
            raise RuntimeError("Redis connection not initialized")
            
        key = f"predictions:{prediction.tool_id}"
        value = json.dumps(prediction.to_dict())
        
        await self.redis.rpush(key, value)
        
    async def _store_prediction_mongo(self, prediction: PredictionResult):
        """Store prediction in MongoDB"""
        if not self.mongo:
            raise RuntimeError("MongoDB connection not initialized")
            
        collection = self.mongo.analysis_engine.predictions
        await collection.insert_one(prediction.to_dict())
        
    async def get_predictions(
        self,
        tool_id: str,
        market_regime: Optional[MarketRegime] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: Optional[int] = None
    ) -> List[PredictionResult]:
        """
        Get prediction history
        
        Args:
            tool_id: Tool identifier
            market_regime: Optional market regime filter
            start_time: Optional start time filter
            end_time: Optional end time filter
            limit: Optional limit on number of results
            
        Returns:
            List of prediction results
        """
        try:
            if self.storage_type == "memory":
                return await self._get_predictions_memory(
                    tool_id,
                    market_regime,
                    start_time,
                    end_time,
                    limit
                )
            elif self.storage_type == "redis":
                return await self._get_predictions_redis(
                    tool_id,
                    market_regime,
                    start_time,
                    end_time,
                    limit
                )
            elif self.storage_type == "mongo":
                return await self._get_predictions_mongo(
                    tool_id,
                    market_regime,
                    start_time,
                    end_time,
                    limit
                )
                
            return []
            
        except Exception as e:
            self.logger.error(f"Error getting predictions: {str(e)}")
            return []
            
    async def _get_predictions_memory(
        self,
        tool_id: str,
        market_regime: Optional[MarketRegime] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: Optional[int] = None
    ) -> List[PredictionResult]:
        """Get predictions from memory"""
        if tool_id not in self.memory_storage["predictions"]:
            return []
            
        predictions = self.memory_storage["predictions"][tool_id]
        
        # Apply filters
        filtered_predictions = []
        for pred_dict in predictions:
            # Convert stored dict back to PredictionResult
            pred = PredictionResult(
                tool_id=pred_dict["tool_id"],
                timestamp=datetime.fromisoformat(pred_dict["timestamp"]),
                market_regime=MarketRegime(pred_dict["market_regime"]),
                prediction=pred_dict["prediction"],
                actual_outcome=pred_dict.get("actual_outcome"),
                confidence=pred_dict.get("confidence", 0.0),
                impact=pred_dict.get("impact", 0.0),
                metadata=pred_dict.get("metadata")
            )
            
            if market_regime and pred.market_regime != market_regime:
                continue
                
            if start_time and pred.timestamp < start_time:
                continue
                
            if end_time and pred.timestamp > end_time:
                continue
                
            filtered_predictions.append(pred)
            
        if limit:
            filtered_predictions = filtered_predictions[-limit:]
            
        return filtered_predictions
        
    async def _get_predictions_redis(
        self,
        tool_id: str,
        market_regime: Optional[MarketRegime] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: Optional[int] = None
    ) -> List[PredictionResult]:
        """Get predictions from Redis"""
        if not self.redis:
            raise RuntimeError("Redis connection not initialized")
            
        key = f"predictions:{tool_id}"
        raw_predictions = await self.redis.lrange(key, 0, -1)
        
        predictions = []
        for raw_pred in raw_predictions:
            pred_dict = json.loads(raw_pred)
            pred = PredictionResult(
                tool_id=pred_dict["tool_id"],
                timestamp=datetime.fromisoformat(pred_dict["timestamp"]),
                market_regime=MarketRegime(pred_dict["market_regime"]),
                prediction=pred_dict["prediction"],
                actual_outcome=pred_dict.get("actual_outcome"),
                confidence=pred_dict.get("confidence", 0.0),
                impact=pred_dict.get("impact", 0.0),
                metadata=pred_dict.get("metadata")
            )
            
            if market_regime and pred.market_regime != market_regime:
                continue
                
            if start_time and pred.timestamp < start_time:
                continue
                
            if end_time and pred.timestamp > end_time:
                continue
                
            predictions.append(pred)
            
        if limit:
            predictions = predictions[-limit:]
            
        return predictions
        
    async def _get_predictions_mongo(
        self,
        tool_id: str,
        market_regime: Optional[MarketRegime] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: Optional[int] = None
    ) -> List[PredictionResult]:
        """Get predictions from MongoDB"""
        if not self.mongo:
            raise RuntimeError("MongoDB connection not initialized")
            
        collection = self.mongo.analysis_engine.predictions
        
        # Build query
        query = {"tool_id": tool_id}
        
        if market_regime:
            query["market_regime"] = market_regime.value
            
        if start_time or end_time:
            query["timestamp"] = {}
            if start_time:
                query["timestamp"]["$gte"] = start_time
            if end_time:
                query["timestamp"]["$lte"] = end_time
                
        # Execute query
        cursor = collection.find(query)
        
        if limit:
            cursor = cursor.sort("timestamp", -1).limit(limit)
            
        predictions = []
        async for doc in cursor:
            pred = PredictionResult(
                tool_id=doc["tool_id"],
                timestamp=doc["timestamp"],
                market_regime=MarketRegime(doc["market_regime"]),
                prediction=doc["prediction"],
                actual_outcome=doc.get("actual_outcome"),
                confidence=doc.get("confidence", 0.0),
                impact=doc.get("impact", 0.0),
                metadata=doc.get("metadata")
            )
            predictions.append(pred)
            
        return predictions
        
    async def store_effectiveness(
        self,
        tool_id: str,
        market_regime: MarketRegime,
        effectiveness: float
    ):
        """
        Store tool effectiveness score
        
        Args:
            tool_id: Tool identifier
            market_regime: Market regime
            effectiveness: Effectiveness score
        """
        try:
            if self.storage_type == "memory":
                await self._store_effectiveness_memory(
                    tool_id,
                    market_regime,
                    effectiveness
                )
            elif self.storage_type == "redis":
                await self._store_effectiveness_redis(
                    tool_id,
                    market_regime,
                    effectiveness
                )
            elif self.storage_type == "mongo":
                await self._store_effectiveness_mongo(
                    tool_id,
                    market_regime,
                    effectiveness
                )
                
        except Exception as e:
            self.logger.error(f"Error storing effectiveness: {str(e)}")
            
    async def _store_effectiveness_memory(
        self,
        tool_id: str,
        market_regime: MarketRegime,
        effectiveness: float
    ):
        """Store effectiveness in memory"""
        if tool_id not in self.memory_storage["effectiveness"]:
            self.memory_storage["effectiveness"][tool_id] = {}
            
        self.memory_storage["effectiveness"][tool_id][market_regime.value] = {
            "score": effectiveness,
            "updated_at": datetime.now().isoformat()
        }
        
    async def _store_effectiveness_redis(
        self,
        tool_id: str,
        market_regime: MarketRegime,
        effectiveness: float
    ):
        """Store effectiveness in Redis"""
        if not self.redis:
            raise RuntimeError("Redis connection not initialized")
            
        key = f"effectiveness:{tool_id}:{market_regime.value}"
        value = json.dumps({
            "score": effectiveness,
            "updated_at": datetime.now().isoformat()
        })
        
        await self.redis.set(key, value)
        
    async def _store_effectiveness_mongo(
        self,
        tool_id: str,
        market_regime: MarketRegime,
        effectiveness: float
    ):
        """Store effectiveness in MongoDB"""
        if not self.mongo:
            raise RuntimeError("MongoDB connection not initialized")
            
        collection = self.mongo.analysis_engine.effectiveness
        
        await collection.update_one(
            {
                "tool_id": tool_id,
                "market_regime": market_regime.value
            },
            {
                "$set": {
                    "score": effectiveness,
                    "updated_at": datetime.now()
                }
            },
            upsert=True
        )
        
    async def get_effective_tools(
        self,
        market_regime: MarketRegime,
        min_score: float = 0.5,
        max_age: Optional[timedelta] = None
    ) -> Dict[str, float]:
        """
        Get effective tools for a market regime
        
        Args:
            market_regime: Market regime
            min_score: Minimum effectiveness score
            max_age: Maximum age of effectiveness data
            
        Returns:
            Dictionary mapping tool IDs to effectiveness scores
        """
        try:
            if self.storage_type == "memory":
                return await self._get_effective_tools_memory(
                    market_regime,
                    min_score,
                    max_age
                )
            elif self.storage_type == "redis":
                return await self._get_effective_tools_redis(
                    market_regime,
                    min_score,
                    max_age
                )
            elif self.storage_type == "mongo":
                return await self._get_effective_tools_mongo(
                    market_regime,
                    min_score,
                    max_age
                )
                
            return {}
            
        except Exception as e:
            self.logger.error(f"Error getting effective tools: {str(e)}")
            return {}
            
    async def _get_effective_tools_memory(
        self,
        market_regime: MarketRegime,
        min_score: float = 0.5,
        max_age: Optional[timedelta] = None
    ) -> Dict[str, float]:
        """Get effective tools from memory"""
        effective_tools = {}
        
        for tool_id, regimes in self.memory_storage["effectiveness"].items():
            if market_regime.value not in regimes:
                continue
                
            data = regimes[market_regime.value]
            score = data["score"]
            updated_at = datetime.fromisoformat(data["updated_at"])
            
            if score < min_score:
                continue
                
            if max_age and datetime.now() - updated_at > max_age:
                continue
                
            effective_tools[tool_id] = score
            
        return effective_tools
        
    async def _get_effective_tools_redis(
        self,
        market_regime: MarketRegime,
        min_score: float = 0.5,
        max_age: Optional[timedelta] = None
    ) -> Dict[str, float]:
        """Get effective tools from Redis"""
        if not self.redis:
            raise RuntimeError("Redis connection not initialized")
            
        effective_tools = {}
        
        # Get all tool keys
        pattern = f"effectiveness:*:{market_regime.value}"
        keys = await self.redis.keys(pattern)
        
        for key in keys:
            raw_data = await self.redis.get(key)
            if not raw_data:
                continue
                
            data = json.loads(raw_data)
            score = data["score"]
            updated_at = datetime.fromisoformat(data["updated_at"])
            
            if score < min_score:
                continue
                
            if max_age and datetime.now() - updated_at > max_age:
                continue
                
            tool_id = key.split(":")[1]
            effective_tools[tool_id] = score
            
        return effective_tools
        
    async def _get_effective_tools_mongo(
        self,
        market_regime: MarketRegime,
        min_score: float = 0.5,
        max_age: Optional[timedelta] = None
    ) -> Dict[str, float]:
        """Get effective tools from MongoDB"""
        if not self.mongo:
            raise RuntimeError("MongoDB connection not initialized")
            
        collection = self.mongo.analysis_engine.effectiveness
        
        # Build query
        query = {
            "market_regime": market_regime.value,
            "score": {"$gte": min_score}
        }
        
        if max_age:
            min_date = datetime.now() - max_age
            query["updated_at"] = {"$gte": min_date}
            
        effective_tools = {}
        async for doc in collection.find(query):
            effective_tools[doc["tool_id"]] = doc["score"]
            
        return effective_tools