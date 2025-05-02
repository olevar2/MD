"""
Advanced indicator ML integration module.

This module enhances the ML integration capabilities for advanced indicators
from the Analysis Engine Service.
"""

from typing import Dict, List, Any, Optional, Union, Type, Tuple
import pandas as pd
import numpy as np
import logging
from datetime import datetime

from feature_store_service.indicators.ml_integration import (
    FeatureExtractor, FeatureType
)
from feature_store_service.indicators.base_indicator import BaseIndicator
from feature_store_service.indicators.advanced_indicator_adapter import AdvancedIndicatorAdapter

# Configure logging
logger = logging.getLogger(__name__)


class EnhancedFeatureExtractor(FeatureExtractor):
    """
    Enhanced feature extractor with support for advanced indicators.
    
    This class extends the standard FeatureExtractor to better support
    features from advanced technical analysis components.
    """
    
    def __init__(self, **kwargs):
        """Initialize with parent class constructor."""
        super().__init__(**kwargs)
        
    def extract_advanced_features(
        self,
        data: pd.DataFrame,
        advanced_indicators: Dict[str, Dict[str, Any]],
        include_pattern_signals: bool = True,
        include_confluence: bool = True,
        **kwargs
    ) -> pd.DataFrame:
        """
        Extract features specifically from advanced indicators.
        
        Args:
            data: DataFrame with price and indicator data
            advanced_indicators: Dictionary mapping indicator names to their info
            include_pattern_signals: Whether to include pattern signals as features
            include_confluence: Whether to include confluence metrics
            **kwargs: Additional parameters for feature extraction
            
        Returns:
            DataFrame with extracted features
        """
        # First use the standard feature extraction
        features = self.extract_features(data, advanced_indicators, **kwargs)
        
        # Add specialized advanced features
        
        # 1. Pattern Recognition Signals
        if include_pattern_signals:
            features = self._add_pattern_signals(features, data)
            
        # 2. Indicator Confluence
        if include_confluence:
            features = self._add_confluence_metrics(features, data, advanced_indicators)
            
        # 3. Market Structure Features
        features = self._add_market_structure_features(features, data)
        
        return features
    
    def _add_pattern_signals(self, features: pd.DataFrame, data: pd.DataFrame) -> pd.DataFrame:
        """Add features based on detected chart patterns."""
        # Find column names related to patterns
        pattern_columns = [
            col for col in data.columns 
            if any(pattern in col.lower() for pattern in 
                  ['pattern', 'harmonic', 'elliott', 'fib', 'gann'])
        ]
        
        # For each pattern column, extract features
        for col in pattern_columns:
            # Skip if column is all NaN
            if data[col].isna().all():
                continue
                
            # Convert pattern signals to numeric features when possible
            if data[col].dtype == 'object':
                try:
                    # For string columns, create binary features for each unique value
                    unique_values = data[col].dropna().unique()
                    if len(unique_values) < 10:  # Don't create too many features
                        for val in unique_values:
                            col_name = f"{col}_{val}".lower().replace(' ', '_')
                            features[col_name] = (data[col] == val).astype(float)
                except Exception as e:
                    logger.warning(f"Could not process pattern column {col}: {str(e)}")
            else:
                # For numeric pattern signals, just copy them
                features[col] = data[col]
                
        return features
    
    def _add_confluence_metrics(
        self, 
        features: pd.DataFrame, 
        data: pd.DataFrame,
        indicators: Dict[str, Dict[str, Any]]
    ) -> pd.DataFrame:
        """Add metrics measuring confluence between different indicators."""
        # Find trend columns (usually columns with 'trend', 'direction', etc.)
        trend_columns = [
            col for col in data.columns
            if any(trend in col.lower() for trend in ['trend', 'direction', 'signal'])
        ]
        
        if len(trend_columns) >= 2:
            # Calculate trend agreement ratio
            agreement_count = 0
            total_pairs = 0
            
            for i, col1 in enumerate(trend_columns):
                for col2 in trend_columns[i+1:]:
                    # Skip if either column has all NaN
                    if data[col1].isna().all() or data[col2].isna().all():
                        continue
                        
                    # Count agreements (both positive or both negative)
                    agreement = ((data[col1] > 0) & (data[col2] > 0)) | ((data[col1] < 0) & (data[col2] < 0))
                    agreement_count += agreement.sum()
                    total_pairs += (~data[col1].isna() & ~data[col2].isna()).sum()
            
            if total_pairs > 0:
                features['trend_agreement_ratio'] = agreement_count / total_pairs
                
        return features
    
    def _add_market_structure_features(self, features: pd.DataFrame, data: pd.DataFrame) -> pd.DataFrame:
        """Add features related to market structure analysis."""
        # Look for support/resistance columns
        sr_columns = [
            col for col in data.columns
            if any(sr in col.lower() for sr in ['support', 'resistance', 'pivot'])
        ]
        
        if sr_columns and 'close' in data.columns:
            # Calculate price position relative to closest support/resistance levels
            for col in sr_columns:
                try:
                    # Calculate distance to level as percentage of price
                    distance = (data[col] - data['close']) / data['close']
                    features[f"distance_to_{col}"] = distance
                    
                    # Create proximity feature (closer to 1 means price is closer to level)
                    proximity = 1 / (1 + abs(distance))
                    features[f"proximity_to_{col}"] = proximity
                except Exception:
                    pass
                    
        return features
        

class AdvancedFeatureSelection:
    """
    Enhanced feature selection for advanced indicators.
    
    This class implements specialized feature selection methods
    optimized for the complex features generated by advanced indicators.
    """
    
    def __init__(self, **kwargs):
        """Initialize the feature selector."""
        self.selected_features = []
        self.feature_scores = {}
        
    def select_advanced_features(
        self, 
        features: pd.DataFrame, 
        target: pd.Series,
        n_features: int = None,
        include_basic: bool = True,
        method: str = 'importance'
    ) -> pd.DataFrame:
        """
        Select optimal features with special handling for advanced indicators.
        
        Args:
            features: Feature DataFrame
            target: Target variable Series
            n_features: Number of features to select (None for auto)
            include_basic: Whether to include basic features
            method: Selection method ('importance', 'correlation', 'domain')
            
        Returns:
            DataFrame with selected features
        """
        if not include_basic:
            # Filter to only keep advanced indicator features
            advanced_prefixes = ['fib_', 'pattern_', 'elliott_', 'gann_', 'harmonic_', 'fractal_']
            advanced_columns = [
                col for col in features.columns
                if any(col.startswith(prefix) for prefix in advanced_prefixes)
            ]
            features = features[advanced_columns]
            
        # If we have very few features left, just return them all
        if features.shape[1] <= 5:
            self.selected_features = features.columns.tolist()
            return features
            
        # Select features by chosen method
        if method == 'importance':
            # Use Random Forest for feature importance
            try:
                from sklearn.ensemble import RandomForestRegressor
                
                # Create and fit a Random Forest model
                rf = RandomForestRegressor(n_estimators=100, random_state=42)
                rf.fit(features.fillna(0), target)
                
                # Get feature importances
                importances = rf.feature_importances_
                self.feature_scores = dict(zip(features.columns, importances))
                
                # Sort features by importance and select top n
                sorted_features = sorted(
                    self.feature_scores.items(), 
                    key=lambda x: x[1], 
                    reverse=True
                )
                
                # Determine how many features to keep
                if n_features is None:
                    n_features = max(5, int(features.shape[1] * 0.3))  # Default to 30%
                
                # Select top features
                self.selected_features = [f[0] for f in sorted_features[:n_features]]
                
                return features[self.selected_features]
                
            except ImportError:
                # Fall back to correlation method if sklearn not available
                logger.warning("RandomForest not available, falling back to correlation method")
                method = 'correlation'
                
        if method == 'correlation':
            # Calculate correlation with target
            correlations = features.apply(lambda x: x.corr(target))
            correlations = correlations.abs().sort_values(ascending=False)
            
            # Store scores
            self.feature_scores = dict(zip(correlations.index, correlations.values))
            
            # Determine how many features to keep
            if n_features is None:
                n_features = max(5, int(features.shape[1] * 0.3))  # Default to 30%
                
            # Select top features
            self.selected_features = correlations.index[:n_features].tolist()
            
            return features[self.selected_features]
            
        if method == 'domain':
            # Domain-specific selection based on indicator types
            selected = []
            
            # Always include important pattern signals
            pattern_cols = [
                col for col in features.columns 
                if any(pattern in col.lower() for pattern in 
                      ['pattern', 'harmonic', 'elliott', 'fib', 'gann'])
            ]
            selected.extend(pattern_cols[:min(len(pattern_cols), 5)])
            
            # Include some trend features
            trend_cols = [
                col for col in features.columns
                if any(trend in col.lower() for trend in ['trend', 'direction', 'signal'])
            ]
            selected.extend(trend_cols[:min(len(trend_cols), 5)])
            
            # Include some support/resistance features
            sr_cols = [
                col for col in features.columns
                if any(sr in col.lower() for sr in ['support', 'resistance', 'pivot'])
            ]
            selected.extend(sr_cols[:min(len(sr_cols), 5)])
            
            # Finalize selection
            self.selected_features = list(set(selected))
            
            # If we have too few features, add some more from what's available
            if len(self.selected_features) < 5 and features.shape[1] > 5:
                remaining = [col for col in features.columns if col not in self.selected_features]
                self.selected_features.extend(remaining[:min(len(remaining), 5)])
                
            return features[self.selected_features]
            
        # Default - return all features
        self.selected_features = features.columns.tolist()
        return features
