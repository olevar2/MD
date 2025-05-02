"""
Feature Store Integration Adapter

This module provides adapters to integrate analysis results with the feature store service,
allowing pattern-based features to be stored and served efficiently.
"""
import logging
import requests
import json
from typing import Dict, List, Any, Optional, Union
from datetime import datetime

from analysis_engine.models.analysis_result import AnalysisResult
from analysis_engine.utils.config_loader import get_config

logger = logging.getLogger(__name__)

class FeatureStoreAdapter:
    """
    Adapter for integrating analysis results with the feature store service.
    
    This class provides methods to:
    1. Convert analysis results to feature store compatible format
    2. Send features to the feature store service
    3. Register pattern-based features in the feature store
    """
    
    def __init__(self):
        """Initialize the feature store adapter with configuration"""
        self.config = get_config().get("feature_store", {})
        self.base_url = self.config.get("service_url", "http://localhost:8002/api/v1")
        self.timeout = self.config.get("timeout_seconds", 10)
        
        logger.info(f"Feature Store Adapter initialized with base URL: {self.base_url}")
    
    def convert_to_feature(
        self,
        analysis_result: AnalysisResult,
        instrument: str,
        timeframe: str
    ) -> Dict[str, Any]:
        """
        Convert an analysis result to a feature store compatible format
        
        Args:
            analysis_result: The analysis result to convert
            instrument: The trading instrument (e.g., "EUR/USD")
            timeframe: The timeframe of the analysis (e.g., "1h", "4h")
            
        Returns:
            Dictionary containing feature data
        """
        # Extract analyzer-specific feature data
        feature_data = self._extract_feature_data(analysis_result)
        
        # Create the feature structure
        feature = {
            "name": f"{analysis_result.analyzer_name}_{timeframe}",
            "source": "analysis_engine",
            "instrument": instrument,
            "timeframe": timeframe,
            "timestamp": datetime.now().isoformat(),
            "data": feature_data,
            "metadata": {
                "analyzer_name": analysis_result.analyzer_name,
                "analysis_id": analysis_result.metadata.get("execution_id") if hasattr(analysis_result, "metadata") else None,
                "analysis_timestamp": analysis_result.metadata.get("timestamp") if hasattr(analysis_result, "metadata") else None,
            }
        }
        
        return feature
    
    def _extract_feature_data(self, analysis_result: AnalysisResult) -> Dict[str, Any]:
        """
        Extract feature data from an analysis result based on analyzer type
        
        This method handles different types of analyzers and extracts the
        relevant feature data from their results.
        
        Args:
            analysis_result: The analysis result to extract data from
            
        Returns:
            Dictionary containing analyzer-specific feature data
        """
        analyzer_name = analysis_result.analyzer_name.lower()
        result_data = analysis_result.result_data
        
        # Pattern-based features
        if "pattern" in analyzer_name or "patterns" in result_data:
            return self._extract_pattern_features(result_data)
        
        # Regime-based features
        elif "regime" in analyzer_name:
            return self._extract_regime_features(result_data)
        
        # Correlation-based features
        elif "correlation" in analyzer_name:
            return self._extract_correlation_features(result_data)
        
        # Elliott Wave features
        elif "elliott" in analyzer_name:
            return self._extract_elliott_wave_features(result_data)
        
        # Multi-timeframe features
        elif "timeframe" in analyzer_name:
            return self._extract_mtf_features(result_data)
        
        # Fractal features
        elif "fractal" in analyzer_name:
            return self._extract_fractal_features(result_data)
        
        # Confluence features
        elif "confluence" in analyzer_name:
            return self._extract_confluence_features(result_data)
            
        # Default: return the full result data
        return result_data
    
    def _extract_pattern_features(self, result_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract pattern-related features"""
        features = {}
        
        # Extract pattern information
        if "patterns" in result_data:
            # Get basic pattern counts
            patterns = result_data["patterns"]
            features["pattern_count"] = len(patterns)
            
            # Group patterns by type
            pattern_types = {}
            for pattern in patterns:
                pattern_type = pattern.get("pattern_type", "unknown")
                if pattern_type not in pattern_types:
                    pattern_types[pattern_type] = []
                pattern_types[pattern_type].append(pattern)
            
            # Create features for each pattern type
            for pattern_type, patterns_of_type in pattern_types.items():
                features[f"{pattern_type}_count"] = len(patterns_of_type)
                
                # Get the most recent pattern of this type
                if patterns_of_type:
                    latest = max(patterns_of_type, key=lambda p: p.get("end_time", 0))
                    features[f"latest_{pattern_type}"] = {
                        "name": latest.get("pattern_name"),
                        "confidence": latest.get("confidence_level", 0),
                        "start_price": latest.get("start_price"),
                        "end_price": latest.get("end_price")
                    }
        
        # Extract signal information if available
        if "signals" in result_data:
            features["signals"] = result_data["signals"]
        
        return features
    
    def _extract_regime_features(self, result_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract market regime features"""
        features = {}
        
        # Extract current regime
        if "current_regime" in result_data:
            features["current_regime"] = result_data["current_regime"]
        
        # Extract regime probabilities if available
        if "regime_probabilities" in result_data:
            features["regime_probabilities"] = result_data["regime_probabilities"]
        
        # Extract regime duration if available
        if "regime_duration" in result_data:
            features["regime_duration"] = result_data["regime_duration"]
        
        # Extract trend strength if available
        if "trend_strength" in result_data:
            features["trend_strength"] = result_data["trend_strength"]
        
        # Extract volatility if available
        if "volatility" in result_data:
            features["volatility"] = result_data["volatility"]
            
        return features
    
    def _extract_correlation_features(self, result_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract correlation features"""
        features = {}
        
        # Extract correlation matrix if available
        if "correlation_matrix" in result_data:
            # We don't want to store the full matrix as a feature
            # Instead, extract key correlations
            corr_matrix = result_data["correlation_matrix"]
            
            if isinstance(corr_matrix, dict):
                # Extract highest correlations
                high_corr_pairs = []
                for pair1, correlations in corr_matrix.items():
                    for pair2, corr_value in correlations.items():
                        if pair1 != pair2 and abs(corr_value) > 0.7:  # High correlation threshold
                            high_corr_pairs.append({
                                "pair1": pair1,
                                "pair2": pair2,
                                "correlation": corr_value
                            })
                
                # Sort by absolute correlation value
                high_corr_pairs.sort(key=lambda x: abs(x["correlation"]), reverse=True)
                
                # Keep top 10 pairs
                features["high_correlations"] = high_corr_pairs[:10]
        
        # Extract correlation regime if available
        if "correlation_regime" in result_data:
            features["correlation_regime"] = result_data["correlation_regime"]
            
        return features
    
    def _extract_elliott_wave_features(self, result_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract Elliott Wave features"""
        features = {}
        
        # Extract counts of different wave types
        if "impulse_count" in result_data:
            features["impulse_count"] = result_data["impulse_count"]
        
        if "corrective_count" in result_data:
            features["corrective_count"] = result_data["corrective_count"]
        
        # Extract latest wave information
        if "latest_impulse" in result_data and result_data["latest_impulse"]:
            latest_impulse = result_data["latest_impulse"]
            features["latest_impulse"] = {
                "confidence": latest_impulse.get("confidence_level", 0),
                "pattern_type": latest_impulse.get("pattern_type"),
                "pattern_complete": latest_impulse.get("pattern_complete", False)
            }
            
            # Extract wave points if available
            if "wave_points" in latest_impulse:
                wave_points = latest_impulse["wave_points"]
                features["current_wave_position"] = max(wave_points.keys())
        
        if "latest_corrective" in result_data and result_data["latest_corrective"]:
            latest_corrective = result_data["latest_corrective"]
            features["latest_corrective"] = {
                "confidence": latest_corrective.get("confidence_level", 0),
                "pattern_type": latest_corrective.get("pattern_type"),
                "pattern_complete": latest_corrective.get("pattern_complete", False)
            }
            
        return features
    
    def _extract_mtf_features(self, result_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract multi-timeframe features"""
        features = {}
        
        # Extract trend alignment information
        if "trend_alignment" in result_data:
            features["trend_alignment"] = result_data["trend_alignment"]
        
        # Extract signal consistency across timeframes
        if "signal_consistency" in result_data:
            features["signal_consistency"] = result_data["signal_consistency"]
        
        # Extract dominant timeframe
        if "dominant_timeframe" in result_data:
            features["dominant_timeframe"] = result_data["dominant_timeframe"]
            
        return features
    
    def _extract_fractal_features(self, result_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract fractal-based features"""
        features = {}
        
        # Extract fractal dimension if available
        if "fractal_dimension" in result_data:
            features["fractal_dimension"] = result_data["fractal_dimension"]
        
        # Extract fractal patterns if available
        if "fractal_patterns" in result_data:
            features["fractal_pattern_count"] = len(result_data["fractal_patterns"])
            
            # Get the most recent fractal pattern
            if result_data["fractal_patterns"]:
                latest = max(result_data["fractal_patterns"], 
                            key=lambda p: p.get("end_time", 0))
                features["latest_fractal_pattern"] = {
                    "type": latest.get("type"),
                    "confidence": latest.get("confidence", 0)
                }
        
        return features
    
    def _extract_confluence_features(self, result_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract confluence features"""
        features = {}
        
        # Extract overall confluence score if available
        if "confluence_score" in result_data:
            features["confluence_score"] = result_data["confluence_score"]
        
        # Extract directional bias if available
        if "directional_bias" in result_data:
            features["directional_bias"] = result_data["directional_bias"]
        
        # Extract signal strengths if available
        if "signal_strengths" in result_data:
            # Only keep top signals
            sorted_signals = sorted(
                result_data["signal_strengths"].items(),
                key=lambda x: abs(x[1]),
                reverse=True
            )
            features["top_signals"] = dict(sorted_signals[:5])
            
        return features
    
    def register_pattern_based_feature(
        self,
        feature_name: str,
        feature_description: str,
        analyzer_name: str,
        parameters: Dict[str, Any] = None
    ) -> bool:
        """
        Register a pattern-based feature with the feature store
        
        Args:
            feature_name: Name of the feature
            feature_description: Description of the feature
            analyzer_name: Name of the analyzer that generates this feature
            parameters: Additional parameters for feature generation
            
        Returns:
            True if registration was successful, False otherwise
        """
        try:
            url = f"{self.base_url}/features/register"
            
            data = {
                "name": feature_name,
                "description": feature_description,
                "source": "analysis_engine",
                "analyzer": analyzer_name,
                "feature_type": "pattern_based",
                "parameters": parameters or {}
            }
            
            response = requests.post(
                url, 
                json=data,
                timeout=self.timeout
            )
            
            if response.status_code == 200 or response.status_code == 201:
                logger.info(f"Successfully registered pattern-based feature: {feature_name}")
                return True
            else:
                logger.error(f"Failed to register feature {feature_name}: {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Error registering pattern-based feature {feature_name}: {e}")
            return False
    
    def store_features(
        self,
        features: Union[Dict[str, Any], List[Dict[str, Any]]],
    ) -> bool:
        """
        Store features in the feature store
        
        Args:
            features: Dictionary or list of dictionaries with feature data
            
        Returns:
            True if storage was successful, False otherwise
        """
        try:
            url = f"{self.base_url}/features/store"
            
            # Convert single feature to list
            if isinstance(features, dict):
                features = [features]
                
            response = requests.post(
                url, 
                json={"features": features},
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                logger.info(f"Successfully stored {len(features)} features in feature store")
                return True
            else:
                logger.error(f"Failed to store features: {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Error storing features: {e}")
            return False
    
    def process_analysis_result(
        self,
        analysis_result: AnalysisResult,
        instrument: str,
        timeframe: str
    ) -> bool:
        """
        Process an analysis result and store derived features in the feature store
        
        Args:
            analysis_result: The analysis result to process
            instrument: The trading instrument (e.g., "EUR/USD")
            timeframe: The timeframe of the analysis (e.g., "1h", "4h")
            
        Returns:
            True if processing was successful, False otherwise
        """
        try:
            # Convert analysis result to feature
            feature = self.convert_to_feature(analysis_result, instrument, timeframe)
            
            # Store feature in feature store
            success = self.store_features(feature)
            
            return success
            
        except Exception as e:
            logger.error(f"Error processing analysis result: {e}")
            return False
