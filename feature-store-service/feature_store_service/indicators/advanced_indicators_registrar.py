"""
Advanced Technical Indicators Registrar Module.

This module registers all the advanced technical indicators with the indicator registry,
including both local indicators and those adapted from the analysis-engine-service.
"""

import logging
from typing import Dict, Type, Any, List, Optional

from feature_store_service.indicators.indicator_registry import IndicatorRegistry
from feature_store_service.indicators.advanced_indicator_adapter import (
    AdvancedIndicatorAdapter, load_advanced_indicators
)
# TODO: Uncomment when advanced_moving_averages.py is implemented
from feature_store_service.indicators.advanced_moving_averages import (
    TripleExponentialMovingAverage as TEMA, # Renamed for clarity
    DoubleExponentialMovingAverage as DEMA,
    HullMovingAverage as HullMA,
    KaufmanAdaptiveMovingAverage as KAMA,
    ZeroLagExponentialMovingAverage as ZLEMA,
    ArnaudLegouxMovingAverage as ALMA,
    JurikMovingAverage as JMA
)
# TODO: Uncomment when advanced_oscillators.py is implemented
from feature_store_service.indicators.advanced_oscillators import (
    AwesomeOscillator,
    AcceleratorOscillator,
    UltimateOscillatorIndicator as UltimateOscillator, # Renamed for clarity
    DeMarker,
    TRIXIndicatorImpl as TRIX, # Renamed for clarity
    KSTIndicatorImpl as KST, # Renamed for clarity
    ElderForceIndex,
    RelativeVigorIndex as RVI, # Renamed for clarity
    FisherTransform,
    CoppockCurveIndicatorImpl as CoppockCurve, # Renamed for clarity
    ChandeMomentumOscillator as CMO # Renamed for clarity
)
# TODO: Uncomment when volume_analysis.py is implemented
from feature_store_service.indicators.volume_analysis import (
    # VolumeProfile, # Omitted
    VWAPBands,
    MarketFacilitationIndex,
    VolumeZoneOscillator,
    EaseOfMovement,
    NVIAndPVI,
    DemandIndex,
    RelativeVolume,
    MoneyFlowIndex # Added MFI here
    # VolumeDelta # Omitted
)
# TODO: Uncomment when volatility.py is implemented with these indicators
from feature_store_service.indicators.volatility import (
    DonchianChannels, PriceEnvelopes, VIXFixIndicator as VIXFix, HistoricalVolatility
)
# TODO: Uncomment when advanced_price_indicators.py is implemented
from feature_store_service.indicators.advanced_price_indicators import (
    IchimokuCloud, HeikinAshi, RenkoCharts, PointAndFigure
)
# TODO: Uncomment when statistical_regression_indicators.py is implemented
from feature_store_service.indicators.statistical_regression_indicators import (
    StandardDeviationIndicator,
    LinearRegressionIndicator,
    LinearRegressionChannel,
    RSquaredIndicator # Added R-Squared
)
# Advanced Pattern Recognition
from feature_store_service.indicators.advanced_patterns import (
    AdvancedPatternFacade,
    RenkoPatternRecognizer,
    IchimokuPatternRecognizer,
    WyckoffPatternRecognizer,
    HeikinAshiPatternRecognizer,
    VSAPatternRecognizer,
    MarketProfileAnalyzer,
    PointAndFigureAnalyzer,
    WolfeWaveDetector,
    PitchforkAnalyzer,
    DivergenceDetector
)

# Configure logging
logger = logging.getLogger(__name__)


def register_advanced_indicators(registry: IndicatorRegistry) -> None:
    \"\"\"
    Register all advanced technical indicators with the indicator registry.

    Args:
        registry: The indicator registry instance
    \"\"\"
    # TODO: Uncomment these sections when the corresponding implementation files exist
    # Register Advanced Moving Averages
    # registry.register_indicator(TEMA)
    # registry.register_indicator(DEMA)
    # registry.register_indicator(HullMA)
    # registry.register_indicator(KAMA)
    # registry.register_indicator(ZLEMA)
    # registry.register_indicator(ALMA)
    # registry.register_indicator(JMA)
    registry.register_indicator(TEMA)
    registry.register_indicator(DEMA)
    registry.register_indicator(HullMA)
    registry.register_indicator(KAMA)
    registry.register_indicator(ZLEMA)
    registry.register_indicator(ALMA)
    registry.register_indicator(JMA)

    # Register Advanced Oscillators
    # registry.register_indicator(AwesomeOscillator)
    # registry.register_indicator(AcceleratorOscillator)
    # registry.register_indicator(UltimateOscillator)
    # registry.register_indicator(DeMarker)
    # registry.register_indicator(TRIX)
    # registry.register_indicator(KST)
    # registry.register_indicator(ElderForceIndex)
    # registry.register_indicator(RVI)
    # registry.register_indicator(FisherTransform)
    # registry.register_indicator(CoppockCurve)
    # registry.register_indicator(CMO)
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

    # Register Volume Analysis Indicators
    # registry.register_indicator(VolumeProfile) # Omitted
    # registry.register_indicator(VWAPBands)
    # registry.register_indicator(MarketFacilitationIndex)
    # registry.register_indicator(VolumeZoneOscillator)
    # registry.register_indicator(EaseOfMovement)
    # registry.register_indicator(NVIAndPVI)
    # registry.register_indicator(DemandIndex)
    # registry.register_indicator(RelativeVolume)
    # registry.register_indicator(MoneyFlowIndex) # Added MFI
    # registry.register_indicator(VolumeDelta) # Omitted
    # registry.register_indicator(VolumeProfile) # Omitted
    registry.register_indicator(VWAPBands)
    registry.register_indicator(MarketFacilitationIndex)
    registry.register_indicator(VolumeZoneOscillator)
    registry.register_indicator(EaseOfMovement)
    registry.register_indicator(NVIAndPVI)
    registry.register_indicator(DemandIndex)
    registry.register_indicator(RelativeVolume)
    registry.register_indicator(MoneyFlowIndex) # Added MFI
    # registry.register_indicator(VolumeDelta) # Omitted

    # Register Advanced Volatility Indicators
    registry.register_indicator(DonchianChannels)
    registry.register_indicator(PriceEnvelopes)
    registry.register_indicator(VIXFix)
    registry.register_indicator(HistoricalVolatility)

    # Register Advanced Price Indicators
    # registry.register_indicator(IchimokuCloud)
    # registry.register_indicator(HeikinAshi)
    # registry.register_indicator(RenkoCharts)
    # registry.register_indicator(PointAndFigure)
    registry.register_indicator(IchimokuCloud)
    registry.register_indicator(HeikinAshi)
    registry.register_indicator(RenkoCharts)
    registry.register_indicator(PointAndFigure)

    # Register Statistical and Regression Indicators
    # registry.register_indicator(StandardDeviationIndicator)
    # registry.register_indicator(LinearRegressionIndicator)
    # registry.register_indicator(LinearRegressionChannel)
    # registry.register_indicator(RSquaredIndicator) # Added R-Squared
    registry.register_indicator(StandardDeviationIndicator)
    registry.register_indicator(LinearRegressionIndicator)
    registry.register_indicator(LinearRegressionChannel)
    registry.register_indicator(RSquaredIndicator) # Added R-Squared

    # Register Advanced Pattern Recognition
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

    # Register Analysis Engine Adapted Indicators
    # This part remains active as it depends on external code via the adapter
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
        logger.info("Loading indicators from Analysis Engine Service...")
        # Load advanced indicators from the Analysis Engine Service
        advanced_indicators = load_advanced_indicators()
        if not advanced_indicators:
            logger.warning("No indicators found in Analysis Engine Service")
            return

        logger.info(f"Found {len(advanced_indicators)} indicators in Analysis Engine Service")

        # Map of indicator types to adapter configuration
        adapter_configs = {
            # Fibonacci indicators
            'FibonacciRetracement': {'name_prefix': 'fib'},
            'FibonacciExtension': {'name_prefix': 'fib'},
            'FibonacciArcs': {'name_prefix': 'fib'},
            'FibonacciFans': {'name_prefix': 'fib'}, # Added
            'FibonacciTimeZones': {'name_prefix': 'fib'}, # Added

            # Gann indicators
            'GannFan': {'name_prefix': 'gann'},
            'GannGrid': {'name_prefix': 'gann'},
            'GannSquare': {'name_prefix': 'gann'},

            # Harmonic patterns
            'HarmonicPatternFinder': {'name_prefix': 'harmonic'},

            # Elliott Wave
            'ElliottWaveAnalyzer': {'name_prefix': 'elliott'}, # Added

            # Fractal Geometry
            'FractalIndicator': {'name_prefix': 'fractal'}, # Added

            # Pivot Points
            'PivotPointCalculator': {'name_prefix': 'pivot'}, # Added

            # Volume/Volatility
            'VolatilityAnalysis': {'name_prefix': 'volvol'}, # Added

            # Confluence
            'ConfluenceDetector': {'name_prefix': 'confluence'}, # Added

            # Market Regime
            'MarketRegimeDetector': {'name_prefix': 'regime'}, # Added

            # Multi-Timeframe
            'MultiTimeframeAnalysis': {'name_prefix': 'mta'}, # Added

            # Time Cycle
            'TimeCycleAnalysis': {'name_prefix': 'cycle'}, # Added

            # Currency Correlation
            'CurrencyCorrelation': {'name_prefix': 'corr'}, # Added

            # Add more configurations as needed...
        }

        for indicator_name, indicator_class in advanced_indicators.items():
            if indicator_name in adapter_configs:
                config = adapter_configs[indicator_name]
                # Assuming the module path follows a pattern like:
                # analysis_engine.analysis.advanced_ta.<module_name>
                # We need to infer the module name from the class name or have a mapping
                # For simplicity, let's assume a direct mapping for now (e.g., FibonacciRetracement is in fibonacci.py)
                module_name_snake = _camel_to_snake(indicator_name).replace("_analyzer", "").replace("_indicator", "").replace("_calculator", "").replace("_detector", "").replace("_finder", "").replace("_analysis", "")
                # Handle specific cases
                if "fibonacci" in module_name_snake:
                    module_name = "fibonacci"
                elif "gann" in module_name_snake:
                    module_name = "gann_tools"
                elif "harmonic" in module_name_snake:
                    module_name = "harmonic_patterns"
                elif "elliott" in module_name_snake:
                    module_name = "elliott_wave"
                elif "fractal" in module_name_snake:
                    module_name = "fractal_geometry"
                elif "pivot" in module_name_snake:
                    module_name = "pivot_points"
                elif "volvol" in config.get('name_prefix', ""):
                    module_name = "volume_volatility"
                elif "confluence" in config.get('name_prefix', ""):
                    module_name = "confluence"
                elif "regime" in config.get('name_prefix', ""):
                    module_name = "market_regime"
                elif "mta" in config.get('name_prefix', ""):
                    module_name = "multi_timeframe"
                elif "cycle" in config.get('name_prefix', ""):
                    module_name = "time_cycle"
                elif "corr" in config.get('name_prefix', ""):
                    module_name = "currency_correlation"
                else:
                    # Default guess - might need refinement
                    module_name = module_name_snake

                module_path = f"analysis_engine.analysis.advanced_ta.{module_name}"

                try:
                    # Create adapter with specific config
                    # We pass the class directly now since load_advanced_indicators returns it
                    adapter_instance = AdvancedIndicatorAdapter(
                        advanced_indicator_class=indicator_class,
                        **config
                    )
                    registry.register_indicator(adapter_instance) # Register the instance
                    logger.info(f"Registered adapted indicator: {adapter_instance.name}")
                except Exception as e:
                    logger.error(f"Failed to create or register adapter for {indicator_name} from {module_path}: {e}")
            else:
                logger.warning(f"No adapter configuration found for {indicator_name}. Skipping registration.")

    except ImportError as e:
        logger.error(f"Could not import analysis engine modules: {e}. Adapters not registered.")
    except Exception as e:
        logger.error(f"An unexpected error occurred during analysis engine indicator registration: {e}")

def _camel_to_snake(name: str) -> str:
    """Convert CamelCase name to snake_case."""
    import re
    name = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    name = re.sub('([a-z0-9])([A-Z])', r'\1_\2', name)
    return name.lower()
