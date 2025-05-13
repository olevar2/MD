"""
Advanced Pattern Recognition Facade Module.

This module provides a unified interface to all advanced pattern recognition systems.
"""
from typing import Dict, List, Optional, Any, Union
import pandas as pd
from feature_store_service.indicators.base_indicator import BaseIndicator
from feature_store_service.indicators.advanced_patterns.base import AdvancedPatternType, AdvancedPatternRecognizer, PatternDirection, PatternStrength
from feature_store_service.indicators.advanced_patterns.renko import RenkoPatternRecognizer
from feature_store_service.indicators.advanced_patterns.ichimoku import IchimokuPatternRecognizer
from feature_store_service.indicators.advanced_patterns.wyckoff import WyckoffPatternRecognizer
from feature_store_service.indicators.advanced_patterns.heikin_ashi import HeikinAshiPatternRecognizer
from feature_store_service.indicators.advanced_patterns.vsa import VSAPatternRecognizer
from feature_store_service.indicators.advanced_patterns.market_profile import MarketProfileAnalyzer
from feature_store_service.indicators.advanced_patterns.point_and_figure import PointAndFigureAnalyzer
from feature_store_service.indicators.advanced_patterns.wolfe_wave import WolfeWaveDetector
from feature_store_service.indicators.advanced_patterns.pitchfork import PitchforkAnalyzer
from feature_store_service.indicators.advanced_patterns.divergence import DivergenceDetector


from feature_store_service.error.exceptions_bridge import (
    with_exception_handling,
    async_with_exception_handling,
    ForexTradingPlatformError,
    ServiceError,
    DataError,
    ValidationError
)

class AdvancedPatternFacade(BaseIndicator):
    """
    Unified interface for all advanced pattern recognition systems.
    
    This class provides a single entry point to access all advanced pattern
    recognition capabilities in the platform.
    """
    category = 'pattern'

    def __init__(self, pattern_types: Optional[List[str]]=None,
        lookback_period: int=100, sensitivity: float=0.75, **kwargs):
        """
        Initialize the advanced pattern facade.
        
        Args:
            pattern_types: List of pattern types to recognize (None = all patterns)
            lookback_period: Number of bars to look back for pattern recognition
            sensitivity: Sensitivity of pattern detection (0.0-1.0)
            **kwargs: Additional parameters for specific pattern recognizers
        """
        self.lookback_period = lookback_period
        self.sensitivity = sensitivity
        all_pattern_types = [pt.value for pt in AdvancedPatternType]
        if pattern_types is None:
            self.pattern_types = all_pattern_types
        else:
            self.pattern_types = [pt for pt in pattern_types if pt in
                all_pattern_types]
        self.recognizers = {}
        renko_patterns = [pt for pt in self.pattern_types if pt.startswith(
            'renko_')]
        if renko_patterns:
            self.recognizers['renko'] = RenkoPatternRecognizer(pattern_types
                =renko_patterns, lookback_period=lookback_period,
                sensitivity=sensitivity, **kwargs.get('renko_params', {}))
        ichimoku_patterns = [pt for pt in self.pattern_types if pt.
            startswith('ichimoku_')]
        if ichimoku_patterns:
            self.recognizers['ichimoku'] = IchimokuPatternRecognizer(
                pattern_types=ichimoku_patterns, lookback_period=
                lookback_period, sensitivity=sensitivity, **kwargs.get(
                'ichimoku_params', {}))
        wyckoff_patterns = [pt for pt in self.pattern_types if pt.
            startswith('wyckoff_')]
        if wyckoff_patterns:
            self.recognizers['wyckoff'] = WyckoffPatternRecognizer(
                pattern_types=wyckoff_patterns, lookback_period=
                lookback_period, sensitivity=sensitivity, **kwargs.get(
                'wyckoff_params', {}))
        heikin_ashi_patterns = [pt for pt in self.pattern_types if pt.
            startswith('heikin_ashi_')]
        if heikin_ashi_patterns:
            self.recognizers['heikin_ashi'] = HeikinAshiPatternRecognizer(
                pattern_types=heikin_ashi_patterns, lookback_period=
                lookback_period, sensitivity=sensitivity, **kwargs.get(
                'heikin_ashi_params', {}))
        vsa_patterns = [pt for pt in self.pattern_types if pt.startswith(
            'vsa_')]
        if vsa_patterns:
            self.recognizers['vsa'] = VSAPatternRecognizer(pattern_types=
                vsa_patterns, lookback_period=lookback_period, sensitivity=
                sensitivity, **kwargs.get('vsa_params', {}))
        market_profile_patterns = [pt for pt in self.pattern_types if pt.
            startswith('market_profile_')]
        if market_profile_patterns:
            self.recognizers['market_profile'] = MarketProfileAnalyzer(
                pattern_types=market_profile_patterns, lookback_period=
                lookback_period, sensitivity=sensitivity, **kwargs.get(
                'market_profile_params', {}))
        pnf_patterns = [pt for pt in self.pattern_types if pt.startswith(
            'pnf_')]
        if pnf_patterns:
            self.recognizers['point_and_figure'] = PointAndFigureAnalyzer(
                pattern_types=pnf_patterns, lookback_period=lookback_period,
                sensitivity=sensitivity, **kwargs.get('pnf_params', {}))
        wolfe_wave_patterns = [pt for pt in self.pattern_types if pt.
            startswith('wolfe_wave_')]
        if wolfe_wave_patterns:
            self.recognizers['wolfe_wave'] = WolfeWaveDetector(pattern_types
                =wolfe_wave_patterns, lookback_period=lookback_period,
                sensitivity=sensitivity, **kwargs.get('wolfe_wave_params', {}))
        pitchfork_patterns = [pt for pt in self.pattern_types if pt.
            startswith('pitchfork_')]
        if pitchfork_patterns:
            self.recognizers['pitchfork'] = PitchforkAnalyzer(pattern_types
                =pitchfork_patterns, lookback_period=lookback_period,
                sensitivity=sensitivity, **kwargs.get('pitchfork_params', {}))
        divergence_patterns = [pt for pt in self.pattern_types if pt.
            startswith('divergence_')]
        if divergence_patterns:
            self.recognizers['divergence'] = DivergenceDetector(pattern_types
                =divergence_patterns, lookback_period=lookback_period,
                sensitivity=sensitivity, **kwargs.get('divergence_params', {}))

    @with_exception_handling
    def calculate(self, data: pd.DataFrame) ->pd.DataFrame:
        """
        Calculate all advanced pattern recognition for the given data.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with pattern recognition values
        """
        required_cols = ['open', 'high', 'low', 'close']
        for col in required_cols:
            if col not in data.columns:
                raise ValueError(f"Data must contain '{col}' column")
        result = data.copy()
        for recognizer_name, recognizer in self.recognizers.items():
            try:
                result = recognizer.calculate(result)
            except Exception as e:
                print(f'Error calculating {recognizer_name} patterns: {str(e)}'
                    )
        pattern_cols = [col for col in result.columns if col.startswith(
            'pattern_')]
        if pattern_cols:
            result['has_advanced_pattern'] = (result[pattern_cols].sum(axis
                =1) > 0).astype(int)
        else:
            result['has_advanced_pattern'] = 0
        return result

    @with_exception_handling
    def find_patterns(self, data: pd.DataFrame, pattern_types: Optional[
        List[str]]=None) ->Dict[str, List[Dict[str, Any]]]:
        """
        Find advanced patterns in the given data.
        
        Args:
            data: DataFrame with OHLCV data
            pattern_types: List of pattern types to look for (None = all patterns)
            
        Returns:
            Dictionary of pattern types and their occurrences
        """
        if pattern_types is None:
            patterns_to_find = self.pattern_types
        else:
            patterns_to_find = [pt for pt in pattern_types if pt in self.
                pattern_types]
        patterns_dict = {pattern_type: [] for pattern_type in patterns_to_find}
        for recognizer_name, recognizer in self.recognizers.items():
            recognizer_patterns = [pt for pt in patterns_to_find if pt.
                startswith(recognizer_name.split('_')[0] + '_')]
            if recognizer_patterns:
                try:
                    recognizer_results = recognizer.find_patterns(data,
                        recognizer_patterns)
                    for pattern_type, patterns in recognizer_results.items():
                        if pattern_type in patterns_dict:
                            patterns_dict[pattern_type].extend(patterns)
                except Exception as e:
                    print(f'Error finding {recognizer_name} patterns: {str(e)}'
                        )
        return patterns_dict

    def get_supported_patterns(self) ->List[str]:
        """
        Get a list of all supported pattern types.
        
        Returns:
            List of supported pattern types
        """
        return self.pattern_types.copy()

    @classmethod
    def get_info(cls) ->Dict[str, Any]:
        """Get indicator information."""
        return {'name': 'Advanced Pattern Recognition', 'description':
            'Identifies advanced chart patterns for technical analysis',
            'category': cls.category, 'parameters': [{'name':
            'pattern_types', 'description':
            'List of pattern types to look for (None = all patterns)',
            'type': 'list', 'default': None}, {'name': 'lookback_period',
            'description':
            'Number of bars to look back for pattern recognition', 'type':
            'int', 'default': 100}, {'name': 'sensitivity', 'description':
            'Sensitivity of pattern detection (0.0-1.0)', 'type': 'float',
            'default': 0.75}]}
