"""
Multi-Timeframe Analyzer implementation.

This analyzer performs analysis across multiple timeframes to identify
significant patterns and levels, providing a comprehensive market view.
"""
from typing import Any, Dict, List, Optional
import numpy as np
from datetime import datetime
from analysis_engine.core.base.components import BaseAnalyzer, AnalysisResult
from analysis_engine.analysis.indicators import IndicatorClient
from analysis_engine.core.errors import ValidationError, AnalysisError
from analysis_engine.core.exceptions_bridge import with_exception_handling, async_with_exception_handling, ForexTradingPlatformError, ServiceError, DataError, ValidationError


from analysis_engine.resilience.utils import (
    with_resilience,
    with_analysis_resilience,
    with_database_resilience
)

class MultiTimeframeAnalyzer(BaseAnalyzer):
    """Analyzer for multi-timeframe market analysis"""

    def __init__(self, name: str='multi_timeframe_analyzer', parameters:
        Optional[Dict[str, Any]]=None):
        """Initialize the multi-timeframe analyzer"""
        default_params = {'correlation_threshold': 0.7, 'min_timeframes': 2,
            'trend_strength_threshold': 0.6, 'volume_threshold': 1.5}
        if parameters:
            default_params.update(parameters)
        super().__init__(name, default_params)
        self.indicator_client = IndicatorClient()

    @with_exception_handling
    def analyze(self, data: Dict[str, Any]) ->AnalysisResult:
        """
        Analyze market data across multiple timeframes.
        
        Args:
            data: Dictionary containing market data and parameters
                {
                    "symbol": str,
                    "timeframes": List[str],
                    "market_data": {
                        "M15": {
                            "open": List[float],
                            "high": List[float],
                            "low": List[float],
                            "close": List[float],
                            "volume": List[float],
                            "timestamp": List[str]
                        },
                        "H1": {
                            // Similar structure
                        },
                        // Other timeframes...
                    }
                }
                
        Returns:
            AnalysisResult containing multi-timeframe analysis
            
        Raises:
            ValidationError: If input data is invalid
            AnalysisError: If analysis fails
        """
        try:
            if not self.validate_data(data):
                raise ValidationError(message='Invalid input data', details
                    =self._get_validation_details(data))
            symbol = data['symbol']
            timeframes = data['timeframes']
            market_data = data['market_data']
            if len(timeframes) < self.parameters['min_timeframes']:
                raise ValidationError(message=
                    f"At least {self.parameters['min_timeframes']} timeframes required"
                    , details={'provided_timeframes': timeframes})
            timeframe_analysis = {}
            for tf in timeframes:
                if (tf_data := market_data.get(tf)):
                    try:
                        timeframe_analysis[tf] = self._analyze_timeframe(tf,
                            tf_data)
                    except Exception as e:
                        raise AnalysisError(message=
                            f'Failed to analyze timeframe {tf}', analyzer=
                            self.name) from e
            try:
                correlation_matrix = self._calculate_correlation_matrix(
                    timeframe_analysis)
            except Exception as e:
                raise AnalysisError(message=
                    'Failed to calculate correlation matrix', analyzer=self
                    .name) from e
            try:
                overall_assessment = self._determine_overall_assessment(
                    timeframe_analysis, correlation_matrix)
            except Exception as e:
                raise AnalysisError(message=
                    'Failed to determine overall assessment', analyzer=self
                    .name) from e
            result = {'timestamp': datetime.now().isoformat(), 'symbol':
                symbol, 'timeframe_analysis': timeframe_analysis,
                'correlation_matrix': correlation_matrix,
                'overall_assessment': overall_assessment}
            return AnalysisResult(analyzer_name=self.name, result=result,
                metadata={'timeframes': timeframes, 'analysis_count': len(
                timeframe_analysis)})
        except (ValidationError, AnalysisError):
            raise
        except Exception as e:
            raise AnalysisError(message=
                f'Unexpected error in multi-timeframe analysis: {str(e)}',
                analyzer=self.name) from e

    @with_resilience('validate_data')
    def validate_data(self, data: Dict[str, Any]) ->bool:
        """Validate input data"""
        required_fields = ['symbol', 'timeframes', 'market_data']
        if not all(field in data for field in required_fields):
            return False
        if not isinstance(data['timeframes'], list):
            return False
        if not isinstance(data['market_data'], dict):
            return False
        for tf in data['timeframes']:
            if tf not in data['market_data']:
                return False
            tf_data = data['market_data'][tf]
            required_ohlcv = ['open', 'high', 'low', 'close', 'volume']
            if not all(field in tf_data for field in required_ohlcv):
                return False
            lengths = [len(tf_data[field]) for field in required_ohlcv]
            if not all(length == lengths[0] for length in lengths):
                return False
        return True

    def _get_validation_details(self, data: Dict[str, Any]) ->Dict[str, Any]:
        """Get detailed validation information"""
        details = {'missing_fields': [], 'invalid_timeframes': [],
            'invalid_market_data': []}
        required_fields = ['symbol', 'timeframes', 'market_data']
        details['missing_fields'] = [field for field in required_fields if 
            field not in data]
        if 'timeframes' in data and not isinstance(data['timeframes'], list):
            details['invalid_timeframes'].append('timeframes must be a list')
        if 'market_data' in data:
            if not isinstance(data['market_data'], dict):
                details['invalid_market_data'].append(
                    'market_data must be a dictionary')
            else:
                for tf in data.get('timeframes', []):
                    if tf not in data['market_data']:
                        details['invalid_market_data'].append(
                            f'Missing data for timeframe {tf}')
                    else:
                        tf_data = data['market_data'][tf]
                        required_ohlcv = ['open', 'high', 'low', 'close',
                            'volume']
                        missing_fields = [field for field in required_ohlcv if
                            field not in tf_data]
                        if missing_fields:
                            details['invalid_market_data'].append(
                                f'Missing fields for timeframe {tf}: {missing_fields}'
                                )
        return details

    @with_exception_handling
    def _analyze_timeframe(self, timeframe: str, data: Dict[str, List[Any]]
        ) ->Dict[str, Any]:
        """Analyze a single timeframe"""
        try:
            trend, strength = self._calculate_trend(data)
            key_levels = self._identify_key_levels(data)
            volume_profile = self._calculate_volume_profile(data)
            return {'trend': trend, 'strength': strength, 'key_levels':
                key_levels, 'volume_profile': volume_profile}
        except Exception as e:
            raise AnalysisError(message=
                f'Failed to analyze timeframe {timeframe}', analyzer=self.name
                ) from e

    @with_exception_handling
    def _calculate_trend(self, data: Dict[str, List[Any]]) ->tuple[str, float]:
        """Calculate trend direction and strength"""
        try:
            closes = data['close']
            sma20 = np.mean(closes[-20:])
            sma50 = np.mean(closes[-50:])
            if sma20 > sma50:
                trend = 'bullish'
                strength = min(1.0, (sma20 - sma50) / sma50)
            else:
                trend = 'bearish'
                strength = min(1.0, (sma50 - sma20) / sma50)
            return trend, strength
        except Exception as e:
            raise AnalysisError(message='Failed to calculate trend',
                analyzer=self.name) from e

    @with_exception_handling
    def _identify_key_levels(self, data: Dict[str, List[Any]]) ->List[Dict[
        str, Any]]:
        """Identify key price levels"""
        try:
            return [{'price': 1.092, 'type': 'support', 'strength': 0.8}, {
                'price': 1.095, 'type': 'resistance', 'strength': 0.7}]
        except Exception as e:
            raise AnalysisError(message='Failed to identify key levels',
                analyzer=self.name) from e

    @with_exception_handling
    def _calculate_volume_profile(self, data: Dict[str, List[Any]]) ->Dict[
        str, Any]:
        """Calculate volume profile"""
        try:
            return {'volume_trend': 'increasing', 'relative_volume': 1.2,
                'volume_climax': False}
        except Exception as e:
            raise AnalysisError(message=
                'Failed to calculate volume profile', analyzer=self.name
                ) from e

    @with_exception_handling
    def _calculate_correlation_matrix(self, timeframe_analysis: Dict[str,
        Dict[str, Any]]) ->Dict[str, float]:
        """Calculate correlation between timeframes"""
        try:
            matrix = {}
            timeframes = list(timeframe_analysis.keys())
            for i, tf1 in enumerate(timeframes):
                for tf2 in timeframes[i + 1:]:
                    correlation = self._calculate_timeframe_correlation(
                        timeframe_analysis[tf1], timeframe_analysis[tf2])
                    matrix[f'{tf1}_{tf2}'] = correlation
            return matrix
        except Exception as e:
            raise AnalysisError(message=
                'Failed to calculate correlation matrix', analyzer=self.name
                ) from e

    @with_exception_handling
    def _calculate_timeframe_correlation(self, analysis1: Dict[str, Any],
        analysis2: Dict[str, Any]) ->float:
        """Calculate correlation between two timeframe analyses"""
        try:
            if analysis1['trend'] == analysis2['trend']:
                return 0.8
            return 0.3
        except Exception as e:
            raise AnalysisError(message=
                'Failed to calculate timeframe correlation', analyzer=self.name
                ) from e

    @with_exception_handling
    def _determine_overall_assessment(self, timeframe_analysis: Dict[str,
        Dict[str, Any]], correlation_matrix: Dict[str, float]) ->Dict[str, Any
        ]:
        """Determine overall market assessment"""
        try:
            trends = []
            strengths = []
            for tf, analysis in timeframe_analysis.items():
                trends.append(analysis['trend'])
                strengths.append(analysis['strength'])
            if trends.count('bullish') > trends.count('bearish'):
                trend = 'bullish'
            else:
                trend = 'bearish'
            strength = np.mean(strengths)
            confidence = np.mean(list(correlation_matrix.values()))
            key_levels = []
            for analysis in timeframe_analysis.values():
                key_levels.extend(analysis['key_levels'])
            return {'trend': trend, 'strength': strength, 'confidence':
                confidence, 'key_levels': key_levels}
        except Exception as e:
            raise AnalysisError(message=
                'Failed to determine overall assessment', analyzer=self.name
                ) from e
