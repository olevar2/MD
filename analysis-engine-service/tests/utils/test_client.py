"""
Test Client

This module provides a test client for end-to-end testing of the Analysis Engine.
"""

import json
import logging
import aiohttp
import pandas as pd
from typing import Dict, List, Any, Optional, Union

# Configure logging
logger = logging.getLogger(__name__)

class TestClient:
    """Test client for end-to-end testing."""
    
    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        api_key: Optional[str] = None,
        timeout: int = 30
    ):
        """
        Initialize the test client.
        
        Args:
            base_url: Base URL of the API
            api_key: API key for authentication
            timeout: Timeout in seconds for API requests
        """
        self.base_url = base_url
        self.api_key = api_key
        self.timeout = timeout
    
    async def _request(
        self,
        method: str,
        path: str,
        params: Optional[Dict[str, Any]] = None,
        data: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        Make an API request.
        
        Args:
            method: HTTP method
            path: API path
            params: Query parameters
            data: Request body
            headers: Request headers
            
        Returns:
            API response
        """
        # Build URL
        url = f"{self.base_url}{path}"
        
        # Set up headers
        headers = headers or {}
        if self.api_key:
            headers["X-API-Key"] = self.api_key
        
        # Set up timeout
        timeout = aiohttp.ClientTimeout(total=self.timeout)
        
        # Make request
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.request(
                method=method,
                url=url,
                params=params,
                json=data,
                headers=headers
            ) as response:
                # Check status code
                if response.status >= 400:
                    text = await response.text()
                    raise Exception(f"API request failed with status {response.status}: {text}")
                
                # Parse response
                return await response.json()
    
    async def upload_test_data(
        self,
        symbol: str,
        timeframe: str,
        data: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Upload test data to the server.
        
        Args:
            symbol: Currency pair
            timeframe: Timeframe
            data: Price data
            
        Returns:
            API response
        """
        # Convert DataFrame to dict
        data_dict = data.to_dict(orient="records")
        
        # Make request
        return await self._request(
            method="POST",
            path="/test/upload-data",
            data={
                "symbol": symbol,
                "timeframe": timeframe,
                "data": data_dict
            }
        )
    
    async def detect_confluence(
        self,
        symbol: str,
        timeframe: str,
        signal_type: str,
        signal_direction: str,
        use_currency_strength: bool = True,
        min_confirmation_strength: float = 0.3,
        related_pairs: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Detect confluence.
        
        Args:
            symbol: Currency pair
            timeframe: Timeframe
            signal_type: Signal type
            signal_direction: Signal direction
            use_currency_strength: Whether to use currency strength
            min_confirmation_strength: Minimum confirmation strength
            related_pairs: Related pairs
            
        Returns:
            API response
        """
        # Make request
        return await self._request(
            method="POST",
            path="/confluence",
            data={
                "symbol": symbol,
                "timeframe": timeframe,
                "signal_type": signal_type,
                "signal_direction": signal_direction,
                "use_currency_strength": use_currency_strength,
                "min_confirmation_strength": min_confirmation_strength,
                "related_pairs": related_pairs
            }
        )
    
    async def detect_confluence_ml(
        self,
        symbol: str,
        timeframe: str,
        signal_type: str,
        signal_direction: str,
        use_currency_strength: bool = True,
        min_confirmation_strength: float = 0.3,
        related_pairs: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Detect confluence using ML.
        
        Args:
            symbol: Currency pair
            timeframe: Timeframe
            signal_type: Signal type
            signal_direction: Signal direction
            use_currency_strength: Whether to use currency strength
            min_confirmation_strength: Minimum confirmation strength
            related_pairs: Related pairs
            
        Returns:
            API response
        """
        # Make request
        return await self._request(
            method="POST",
            path="/ml/confluence",
            data={
                "symbol": symbol,
                "timeframe": timeframe,
                "signal_type": signal_type,
                "signal_direction": signal_direction,
                "use_currency_strength": use_currency_strength,
                "min_confirmation_strength": min_confirmation_strength,
                "related_pairs": related_pairs
            }
        )
    
    async def analyze_divergence(
        self,
        symbol: str,
        timeframe: str,
        related_pairs: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Analyze divergence.
        
        Args:
            symbol: Currency pair
            timeframe: Timeframe
            related_pairs: Related pairs
            
        Returns:
            API response
        """
        # Make request
        return await self._request(
            method="POST",
            path="/divergence",
            data={
                "symbol": symbol,
                "timeframe": timeframe,
                "related_pairs": related_pairs
            }
        )
    
    async def analyze_divergence_ml(
        self,
        symbol: str,
        timeframe: str,
        related_pairs: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Analyze divergence using ML.
        
        Args:
            symbol: Currency pair
            timeframe: Timeframe
            related_pairs: Related pairs
            
        Returns:
            API response
        """
        # Make request
        return await self._request(
            method="POST",
            path="/ml/divergence",
            data={
                "symbol": symbol,
                "timeframe": timeframe,
                "related_pairs": related_pairs
            }
        )
    
    async def recognize_patterns(
        self,
        symbol: str,
        timeframe: str,
        window_size: int = 30
    ) -> Dict[str, Any]:
        """
        Recognize patterns.
        
        Args:
            symbol: Currency pair
            timeframe: Timeframe
            window_size: Window size
            
        Returns:
            API response
        """
        # Make request
        return await self._request(
            method="POST",
            path="/patterns",
            data={
                "symbol": symbol,
                "timeframe": timeframe,
                "window_size": window_size
            }
        )
    
    async def recognize_patterns_ml(
        self,
        symbol: str,
        timeframe: str,
        window_size: int = 30
    ) -> Dict[str, Any]:
        """
        Recognize patterns using ML.
        
        Args:
            symbol: Currency pair
            timeframe: Timeframe
            window_size: Window size
            
        Returns:
            API response
        """
        # Make request
        return await self._request(
            method="POST",
            path="/ml/patterns",
            data={
                "symbol": symbol,
                "timeframe": timeframe,
                "window_size": window_size
            }
        )
    
    async def get_currency_strength(
        self,
        timeframe: str,
        method: str = "combined"
    ) -> Dict[str, Any]:
        """
        Get currency strength.
        
        Args:
            timeframe: Timeframe
            method: Method
            
        Returns:
            API response
        """
        # Make request
        return await self._request(
            method="GET",
            path="/currency-strength",
            params={
                "timeframe": timeframe,
                "method": method
            }
        )
    
    async def get_related_pairs(
        self,
        symbol: str,
        min_correlation: float = 0.5,
        timeframe: str = "H1"
    ) -> Dict[str, Any]:
        """
        Get related pairs.
        
        Args:
            symbol: Currency pair
            min_correlation: Minimum correlation
            timeframe: Timeframe
            
        Returns:
            API response
        """
        # Make request
        return await self._request(
            method="GET",
            path=f"/related-pairs/{symbol}",
            params={
                "min_correlation": min_correlation,
                "timeframe": timeframe
            }
        )
    
    async def get_system_status(self) -> Dict[str, Any]:
        """
        Get system status.
        
        Returns:
            API response
        """
        # Make request
        return await self._request(
            method="GET",
            path="/system/status"
        )
    
    async def list_models(
        self,
        model_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        List ML models.
        
        Args:
            model_type: Model type
            
        Returns:
            API response
        """
        # Make request
        return await self._request(
            method="GET",
            path="/ml/models",
            params={
                "model_type": model_type
            }
        )
    
    async def get_model_info(
        self,
        model_name: str
    ) -> Dict[str, Any]:
        """
        Get ML model info.
        
        Args:
            model_name: Model name
            
        Returns:
            API response
        """
        # Make request
        return await self._request(
            method="GET",
            path=f"/ml/models/{model_name}"
        )
