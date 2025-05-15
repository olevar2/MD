"""
Analysis Repository for Market Analysis Service.

This module provides a repository for storing and retrieving market analysis data.
"""
import logging
import uuid
from typing import Dict, List, Any, Optional
from datetime import datetime
import json
import os
import pandas as pd
from common_lib.resilience.decorators import retry_with_backoff

logger = logging.getLogger(__name__)

class AnalysisRepository:
    """
    Repository for storing and retrieving market analysis data.
    """
    
    def __init__(self, data_dir: str = "/data/market-analysis"):
        """
        Initialize the Analysis Repository.
        
        Args:
            data_dir: Directory for storing analysis data
        """
        self.data_dir = data_dir
        os.makedirs(self.data_dir, exist_ok=True)
        
    @retry_with_backoff(max_retries=3, backoff_factor=1.5)
    async def save_analysis_result(
        self,
        analysis_type: str,
        symbol: str,
        timeframe: str,
        result: Dict[str, Any]
    ) -> str:
        """
        Save an analysis result.
        
        Args:
            analysis_type: Type of analysis
            symbol: Symbol analyzed
            timeframe: Timeframe analyzed
            result: Analysis result
            
        Returns:
            ID of the saved result
        """
        try:
            result_id = str(uuid.uuid4())
            
            # Create directory for analysis type if it doesn't exist
            analysis_dir = os.path.join(self.data_dir, analysis_type)
            os.makedirs(analysis_dir, exist_ok=True)
            
            # Create filename
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            filename = f"{symbol}_{timeframe}_{timestamp}_{result_id}.json"
            filepath = os.path.join(analysis_dir, filename)
            
            # Add metadata to result
            result_with_metadata = {
                "id": result_id,
                "analysis_type": analysis_type,
                "symbol": symbol,
                "timeframe": timeframe,
                "timestamp": datetime.now().isoformat(),
                "result": result
            }
            
            # Save result to file
            with open(filepath, "w") as f:
                json.dump(result_with_metadata, f, indent=2)
                
            logger.info(f"Saved {analysis_type} analysis result for {symbol} {timeframe} with ID {result_id}")
            
            return result_id
            
        except Exception as e:
            logger.error(f"Error saving analysis result: {e}")
            raise
            
    @retry_with_backoff(max_retries=3, backoff_factor=1.5)
    async def get_analysis_result(
        self,
        result_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Get an analysis result by ID.
        
        Args:
            result_id: ID of the result
            
        Returns:
            Analysis result or None if not found
        """
        try:
            # Search for the result in all analysis type directories
            for root, _, files in os.walk(self.data_dir):
                for file in files:
                    if result_id in file and file.endswith(".json"):
                        filepath = os.path.join(root, file)
                        
                        with open(filepath, "r") as f:
                            result = json.load(f)
                            
                        logger.info(f"Retrieved analysis result with ID {result_id}")
                        
                        return result
                        
            logger.warning(f"Analysis result with ID {result_id} not found")
            return None
            
        except Exception as e:
            logger.error(f"Error getting analysis result: {e}")
            raise
            
    @retry_with_backoff(max_retries=3, backoff_factor=1.5)
    async def get_latest_analysis_results(
        self,
        analysis_type: str,
        symbol: str,
        timeframe: str,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Get the latest analysis results for a symbol and timeframe.
        
        Args:
            analysis_type: Type of analysis
            symbol: Symbol to get results for
            timeframe: Timeframe to get results for
            limit: Maximum number of results to return
            
        Returns:
            List of analysis results
        """
        try:
            analysis_dir = os.path.join(self.data_dir, analysis_type)
            
            if not os.path.exists(analysis_dir):
                logger.warning(f"No {analysis_type} analysis results found")
                return []
                
            # Get all files for the symbol and timeframe
            files = [f for f in os.listdir(analysis_dir) if f.startswith(f"{symbol}_{timeframe}_") and f.endswith(".json")]
            
            # Sort files by timestamp (descending)
            files.sort(reverse=True)
            
            # Limit the number of files
            files = files[:limit]
            
            results = []
            for file in files:
                filepath = os.path.join(analysis_dir, file)
                
                with open(filepath, "r") as f:
                    result = json.load(f)
                    
                results.append(result)
                
            logger.info(f"Retrieved {len(results)} {analysis_type} analysis results for {symbol} {timeframe}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error getting latest analysis results: {e}")
            raise
            
    @retry_with_backoff(max_retries=3, backoff_factor=1.5)
    async def delete_analysis_result(
        self,
        result_id: str
    ) -> bool:
        """
        Delete an analysis result by ID.
        
        Args:
            result_id: ID of the result
            
        Returns:
            True if the result was deleted, False otherwise
        """
        try:
            # Search for the result in all analysis type directories
            for root, _, files in os.walk(self.data_dir):
                for file in files:
                    if result_id in file and file.endswith(".json"):
                        filepath = os.path.join(root, file)
                        
                        os.remove(filepath)
                        
                        logger.info(f"Deleted analysis result with ID {result_id}")
                        
                        return True
                        
            logger.warning(f"Analysis result with ID {result_id} not found for deletion")
            return False
            
        except Exception as e:
            logger.error(f"Error deleting analysis result: {e}")
            raise