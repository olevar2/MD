"""
Alternative Data Service.

This module provides the main service for the Alternative Data Integration framework.
"""
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from common_lib.exceptions import DataFetchError, DataProcessingError, DataValidationError
from adapters.adapter_factory_1 import MultiSourceAdapterFactory
from data_management_service.alternative.feature_extraction import feature_extractor_registry
from models.models import (
    AlternativeDataType,
    AlternativeDataSource,
    FeatureExtractionConfig
)
from services.service_1 import HistoricalDataService

logger = logging.getLogger(__name__)


class AlternativeDataService:
    """Service for managing alternative data."""

    def __init__(
        self,
        config: Dict[str, Any],
        historical_service: Optional[HistoricalDataService] = None
    ):
        """
        Initialize the alternative data service.

        Args:
            config: Configuration for the service
            historical_service: Optional historical data service for storing data
        """
        self.config = config
        self.historical_service = historical_service
        self.adapter_factory = MultiSourceAdapterFactory(config)
        
        logger.info("Initialized AlternativeDataService")

    async def get_data(
        self,
        data_type: str,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime,
        adapter_id: Optional[str] = None,
        use_cache: bool = True,
        **kwargs
    ) -> pd.DataFrame:
        """
        Get alternative data.

        Args:
            data_type: Type of alternative data
            symbols: List of symbols
            start_date: Start date
            end_date: End date
            adapter_id: Optional adapter ID to use
            use_cache: Whether to use cache
            **kwargs: Additional parameters

        Returns:
            DataFrame containing the alternative data
        """
        try:
            # Convert string to enum if needed
            if isinstance(data_type, str):
                try:
                    data_type_enum = AlternativeDataType(data_type)
                except ValueError:
                    raise ValueError(f"Unknown data type: {data_type}")
            else:
                data_type_enum = data_type
            
            # Get adapter
            if adapter_id:
                adapter = self.adapter_factory.get_adapter(adapter_id)
                
                # Check if adapter supports this data type
                if data_type_enum.value not in await adapter.get_available_data_types():
                    raise ValueError(f"Adapter {adapter_id} does not support data type {data_type}")
                
                # Get data from adapter
                data = await adapter.get_data(
                    data_type=data_type_enum.value,
                    symbols=symbols,
                    start_date=start_date,
                    end_date=end_date,
                    use_cache=use_cache,
                    **kwargs
                )
            else:
                # Get all adapters for this data type
                adapters = self.adapter_factory.get_adapters_by_type(data_type_enum)
                if not adapters:
                    raise ValueError(f"No adapters available for data type {data_type}")
                
                # Try each adapter until we get data
                data = pd.DataFrame()
                errors = []
                
                for adapter in adapters:
                    try:
                        data = await adapter.get_data(
                            data_type=data_type_enum.value,
                            symbols=symbols,
                            start_date=start_date,
                            end_date=end_date,
                            use_cache=use_cache,
                            **kwargs
                        )
                        
                        if not data.empty:
                            break
                    except Exception as e:
                        errors.append(f"Adapter {adapter.name}: {str(e)}")
                
                if data.empty and errors:
                    raise DataFetchError(f"Failed to get data from any adapter: {'; '.join(errors)}")
            
            return data
        except Exception as e:
            logger.error(f"Error getting alternative data: {str(e)}")
            raise

    async def extract_features(
        self,
        data: pd.DataFrame,
        data_type: str,
        feature_config: Dict[str, Any],
        **kwargs
    ) -> pd.DataFrame:
        """
        Extract features from alternative data.

        Args:
            data: DataFrame containing the alternative data
            data_type: Type of alternative data
            feature_config: Configuration for feature extraction
            **kwargs: Additional parameters

        Returns:
            DataFrame containing the extracted features
        """
        try:
            # Convert string to enum if needed
            if isinstance(data_type, str):
                try:
                    data_type_enum = AlternativeDataType(data_type)
                except ValueError:
                    raise ValueError(f"Unknown data type: {data_type}")
            else:
                data_type_enum = data_type
            
            # Get feature extractor
            extractor_classes = feature_extractor_registry.get_extractors(data_type_enum)
            if not extractor_classes:
                raise ValueError(f"No feature extractors available for data type {data_type}")
            
            # Create extractor
            extractor = feature_extractor_registry.create_extractor(data_type_enum, feature_config)
            
            # Extract features
            features = await extractor.extract_features(
                data=data,
                data_type=data_type_enum.value,
                feature_config=feature_config,
                **kwargs
            )
            
            return features
        except Exception as e:
            logger.error(f"Error extracting features: {str(e)}")
            raise

    async def get_and_extract_features(
        self,
        data_type: str,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime,
        feature_config: Dict[str, Any],
        adapter_id: Optional[str] = None,
        use_cache: bool = True,
        **kwargs
    ) -> pd.DataFrame:
        """
        Get alternative data and extract features.

        Args:
            data_type: Type of alternative data
            symbols: List of symbols
            start_date: Start date
            end_date: End date
            feature_config: Configuration for feature extraction
            adapter_id: Optional adapter ID to use
            use_cache: Whether to use cache
            **kwargs: Additional parameters

        Returns:
            DataFrame containing the extracted features
        """
        # Get data
        data = await self.get_data(
            data_type=data_type,
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            adapter_id=adapter_id,
            use_cache=use_cache,
            **kwargs
        )
        
        if data.empty:
            logger.warning(f"No data found for {data_type} {symbols} from {start_date} to {end_date}")
            return pd.DataFrame()
        
        # Extract features
        features = await self.extract_features(
            data=data,
            data_type=data_type,
            feature_config=feature_config,
            **kwargs
        )
        
        return features

    async def store_data(
        self,
        data: pd.DataFrame,
        data_type: str,
        source_id: str,
        **kwargs
    ) -> List[str]:
        """
        Store alternative data in the historical data service.

        Args:
            data: DataFrame containing the alternative data
            data_type: Type of alternative data
            source_id: Source identifier
            **kwargs: Additional parameters

        Returns:
            List of record IDs
        """
        if self.historical_service is None:
            raise ValueError("Historical data service not configured")
        
        if data.empty:
            return []
        
        # Check required columns
        required_columns = ["timestamp", "symbol"]
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns in data: {missing_columns}")
        
        # Store each row
        record_ids = []
        
        for _, row in data.iterrows():
            # Convert row to dict
            row_dict = row.to_dict()
            
            # Remove timestamp and symbol from data
            timestamp = row_dict.pop("timestamp")
            symbol = row_dict.pop("symbol")
            
            # Store in historical service
            record_id = await self.historical_service.store_alternative_data(
                symbol=symbol,
                timestamp=timestamp,
                data_type=data_type,
                data=row_dict,
                source_id=source_id,
                metadata=kwargs.get("metadata", {}),
                created_by=kwargs.get("created_by")
            )
            
            record_ids.append(record_id)
        
        return record_ids
