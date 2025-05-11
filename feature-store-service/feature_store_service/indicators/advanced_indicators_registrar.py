"""
Advanced Technical Indicators Registrar Module.

This module registers all the advanced technical indicators with the indicator registry,
including both local indicators and those adapted from the analysis-engine-service.
"""
import logging
from typing import Dict, Type, Any, List, Optional
from feature_store_service.indicators.indicator_registry import IndicatorRegistry
from feature_store_service.indicators.advanced_indicator_adapter import AdvancedIndicatorAdapter, load_advanced_indicators
from feature_store_service.indicators.advanced_moving_averages import TripleExponentialMovingAverage as TEMA, DoubleExponentialMovingAverage as DEMA, HullMovingAverage as HullMA, KaufmanAdaptiveMovingAverage as KAMA, ZeroLagExponentialMovingAverage as ZLEMA, ArnaudLegouxMovingAverage as ALMA, JurikMovingAverage as JMA
from feature_store_service.indicators.advanced_oscillators import AwesomeOscillator, AcceleratorOscillator, UltimateOscillatorIndicator as UltimateOscillator, DeMarker, TRIXIndicatorImpl as TRIX, KSTIndicatorImpl as KST, ElderForceIndex, RelativeVigorIndex as RVI, FisherTransform, CoppockCurveIndicatorImpl as CoppockCurve, ChandeMomentumOscillator as CMO
from feature_store_service.indicators.volume_analysis import VWAPBands, MarketFacilitationIndex, VolumeZoneOscillator, EaseOfMovement, NVIAndPVI, DemandIndex, RelativeVolume, MoneyFlowIndex
from feature_store_service.indicators.volatility import DonchianChannels, PriceEnvelopes, VIXFixIndicator as VIXFix, HistoricalVolatility
from feature_store_service.indicators.advanced_price_indicators import IchimokuCloud, HeikinAshi, RenkoCharts, PointAndFigure
from feature_store_service.indicators.statistical_regression_indicators import StandardDeviationIndicator, LinearRegressionIndicator, LinearRegressionChannel, RSquaredIndicator
from feature_store_service.indicators.advanced_patterns import AdvancedPatternFacade, RenkoPatternRecognizer, IchimokuPatternRecognizer, WyckoffPatternRecognizer, HeikinAshiPatternRecognizer, VSAPatternRecognizer, MarketProfileAnalyzer, PointAndFigureAnalyzer, WolfeWaveDetector, PitchforkAnalyzer, DivergenceDetector
logger = logging.getLogger(__name__)

def register_advanced_indicators(registry: IndicatorRegistry) -> None:
    """
    Register all advanced technical indicators with the indicator registry.

    Args:
        registry: The indicator registry instance
    """
    registry.register_indicator(TEMA)
    registry.register_indicator(DEMA)
    registry.register_indicator(HullMA)
    registry.register_indicator(KAMA)
    registry.register_indicator(ZLEMA)
    registry.register_indicator(ALMA)
    registry.register_indicator(JMA)
    registry.register_indicator(AwesomeOscillator)
    registry.register_indicator(AcceleratorOscillator)
    registry.register_indicator(UltimateOscillator)
    registry.register_indicator(DeMarker)
    registry.register_indicator(TRIX)
    registry.register_indicator(KST)
    registry.register_indicator(ElderForceIndex)
    registry.register_indicator(RVI)
    registry.register_indicator(FisherTransform)
    registry.register_indicator(CoppockCurve)
    registry.register_indicator(CMO)
    registry.register_indicator(VWAPBands)
    registry.register_indicator(MarketFacilitationIndex)
    registry.register_indicator(VolumeZoneOscillator)
    registry.register_indicator(EaseOfMovement)
    registry.register_indicator(NVIAndPVI)
    registry.register_indicator(DemandIndex)
    registry.register_indicator(RelativeVolume)
    registry.register_indicator(MoneyFlowIndex)
    registry.register_indicator(DonchianChannels)
    registry.register_indicator(PriceEnvelopes)
    registry.register_indicator(VIXFix)
    registry.register_indicator(HistoricalVolatility)
    registry.register_indicator(IchimokuCloud)
    registry.register_indicator(HeikinAshi)
    registry.register_indicator(RenkoCharts)
    registry.register_indicator(PointAndFigure)
    registry.register_indicator(StandardDeviationIndicator)
    registry.register_indicator(LinearRegressionIndicator)
    registry.register_indicator(LinearRegressionChannel)
    registry.register_indicator(RSquaredIndicator)
    registry.register_indicator(AdvancedPatternFacade)
    registry.register_indicator(RenkoPatternRecognizer)
    registry.register_indicator(IchimokuPatternRecognizer)
    registry.register_indicator(WyckoffPatternRecognizer)
    registry.register_indicator(HeikinAshiPatternRecognizer)
    registry.register_indicator(VSAPatternRecognizer)
    registry.register_indicator(MarketProfileAnalyzer)
    registry.register_indicator(PointAndFigureAnalyzer)
    registry.register_indicator(WolfeWaveDetector)
    registry.register_indicator(PitchforkAnalyzer)
    registry.register_indicator(DivergenceDetector)
    register_analysis_engine_indicators(registry)

def register_analysis_engine_indicators(registry: IndicatorRegistry) -> None:
    """
    Register indicators from the Analysis Engine Service using adapters.

    This function discovers and registers advanced indicators from the Analysis Engine,
    making them available in the Feature Store Service through adapter classes.

    Args:
        registry: The indicator registry instance
    """
    try:
        logger.info('Loading indicators from Analysis Engine Service...')
        advanced_indicators = load_advanced_indicators()
        if not advanced_indicators:
            logger.warning('No indicators found in Analysis Engine Service')
            return
        logger.info(f'Found {len(advanced_indicators)} indicators in Analysis Engine Service')
        adapter_configs = {'FibonacciRetracement': {'name_prefix': 'fib'}, 'FibonacciExtension': {'name_prefix': 'fib'}, 'FibonacciArcs': {'name_prefix': 'fib'}, 'FibonacciFans': {'name_prefix': 'fib'}, 'FibonacciTimeZones': {'name_prefix': 'fib'}, 'GannFan': {'name_prefix': 'gann'}, 'GannGrid': {'name_prefix': 'gann'}, 'GannSquare': {'name_prefix': 'gann'}, 'HarmonicPatternFinder': {'name_prefix': 'harmonic'}, 'ElliottWaveAnalyzer': {'name_prefix': 'elliott'}, 'FractalIndicator': {'name_prefix': 'fractal'}, 'PivotPointCalculator': {'name_prefix': 'pivot'}, 'VolatilityAnalysis': {'name_prefix': 'volvol'}, 'ConfluenceDetector': {'name_prefix': 'confluence'}, 'MarketRegimeDetector': {'name_prefix': 'regime'}, 'MultiTimeframeAnalysis': {'name_prefix': 'mta'}, 'TimeCycleAnalysis': {'name_prefix': 'cycle'}, 'CurrencyCorrelation': {'name_prefix': 'corr'}}
        for indicator_name, indicator_class in advanced_indicators.items():
            if indicator_name in adapter_configs:
                config = adapter_configs[indicator_name]
                module_name_snake = _camel_to_snake(indicator_name).replace('_analyzer', '').replace('_indicator', '').replace('_calculator', '').replace('_detector', '').replace('_finder', '').replace('_analysis', '')
                if 'fibonacci' in module_name_snake:
                    module_name = 'fibonacci'
                elif 'gann' in module_name_snake:
                    module_name = 'gann_tools'
                elif 'harmonic' in module_name_snake:
                    module_name = 'harmonic_patterns'
                elif 'elliott' in module_name_snake:
                    module_name = 'elliott_wave'
                elif 'fractal' in module_name_snake:
                    module_name = 'fractal_geometry'
                elif 'pivot' in module_name_snake:
                    module_name = 'pivot_points'
                elif 'volvol' in config.get('name_prefix', ''):
                    module_name = 'volume_volatility'
                elif 'confluence' in config.get('name_prefix', ''):
                    module_name = 'confluence'
                elif 'regime' in config.get('name_prefix', ''):
                    module_name = 'market_regime'
                elif 'mta' in config.get('name_prefix', ''):
                    module_name = 'multi_timeframe'
                elif 'cycle' in config.get('name_prefix', ''):
                    module_name = 'time_cycle'
                elif 'corr' in config.get('name_prefix', ''):
                    module_name = 'currency_correlation'
                else:
                    module_name = module_name_snake
                module_path = f'analysis_engine.analysis.advanced_ta.{module_name}'
                try:
                    adapter_instance = AdvancedIndicatorAdapter(advanced_indicator_class=indicator_class, **config)
                    registry.register_indicator(adapter_instance)
                    logger.info(f'Registered adapted indicator: {adapter_instance.name}')
                except Exception as e:
                    logger.error(f'Failed to create or register adapter for {indicator_name} from {module_path}: {e}')
            else:
                logger.warning(f'No adapter configuration found for {indicator_name}. Skipping registration.')
    except ImportError as e:
        logger.error(f'Could not import analysis engine modules: {e}. Adapters not registered.')
    except Exception as e:
        logger.error(f'An unexpected error occurred during analysis engine indicator registration: {e}')

def camel_to_snake(name: str) -> str:
    """Convert CamelCase name to snake_case."""
    import re
    name = re.sub('(.)([A-Z][a-z]+)', '\\1_\\2', name)
    name = re.sub('([a-z0-9])([A-Z])', '\\1_\\2', name)
    return name.lower()