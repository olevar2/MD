"""
Analysis Service Module

This service provides a central interface for managing and executing
all technical analysis components.
"""
import logging
from typing import Dict, List, Any, Optional, Union, Type
import importlib
import inspect
import pkgutil
import time
import asyncio
import traceback
from datetime import datetime

from analysis_engine.models.analysis_result import AnalysisResult
from analysis_engine.models.market_data import MarketData
from analysis_engine.analysis.base_analyzer import BaseAnalyzer
from analysis_engine.analysis.advanced_ta.market_regime import MarketRegimeAnalyzer
from analysis_engine.analysis.advanced_ta.multi_timeframe import MultiTimeframeAnalyzer
from analysis_engine.analysis.advanced_ta.currency_correlation import CurrencyCorrelationAnalyzer
from analysis_engine.analysis.advanced_ta.time_cycle import TimeCycleAnalyzer
from analysis_engine.analysis.confluence_analyzer import ConfluenceAnalyzer
from analysis_engine.repositories.tool_effectiveness_repository import ToolEffectivenessRepository
from analysis_engine.core.exceptions_bridge import (
    AnalysisError,
    AnalyzerNotFoundError,
    InsufficientDataError,
    InvalidAnalysisParametersError,
    AnalysisTimeoutError,
    MarketRegimeError,
    generate_correlation_id,
    async_with_exception_handling,
    with_exception_handling
)
from analysis_engine.core.logging import get_logger

# Use enhanced logger
logger = get_logger(__name__)

class AnalysisService:
    """Service for managing and executing technical analysis components.

    Provides a central interface for working with various technical analysis
    components, handling initialization, execution, and performance tracking.

    Attributes:
        tool_effectiveness_repository (Optional[ToolEffectivenessRepository]): Repository
            for tracking tool effectiveness.
        analyzers (Dict[str, BaseAnalyzer]): Dictionary holding initialized analyzer instances.
        analyzer_classes (Dict[str, Type[BaseAnalyzer]]): Dictionary mapping analyzer names
            to their classes.
        _initialized (bool): Flag indicating if the service has been initialized.
    """
    """
    Service for managing and executing technical analysis components

    This service provides a central interface for working with various
    technical analysis components, handling initialization, execution,
    and performance tracking.
    """

    def __init__(
        self,
        tool_effectiveness_repository: Optional[ToolEffectivenessRepository] = None
    ):
        """Initializes the AnalysisService.

        Args:
            tool_effectiveness_repository: Optional repository for tracking tool effectiveness.
        """
        """
        Initialize the analysis service

        Args:
            tool_effectiveness_repository: Repository for tracking tool effectiveness
        """
        self.tool_effectiveness_repository = tool_effectiveness_repository
        self.analyzers = {}
        self.analyzer_classes = {
            "confluence": ConfluenceAnalyzer
        }

        # Initialize default analyzers - will be done asynchronously in initialize()
        self._initialized = False

    async def initialize(self):
        """Asynchronously initializes the service and its analyzers.

        This method must be called after creating an instance and before using
        the service. It initializes default analyzers.
        """
        """
        Asynchronously initialize the service

        This method must be called after creating an instance and before using the service.
        """
        if self._initialized:
            return

        await self._initialize_default_analyzers()
        self._initialized = True

    async def _initialize_default_analyzers(self) -> None:
        """Initializes the default analyzer instances required by the service.

        Currently initializes the ConfluenceAnalyzer.
        """
        """Initialize default analyzer instances"""
        # Initialize confluence analyzer
        self.analyzers["confluence"] = ConfluenceAnalyzer(
            tool_effectiveness_repository=self.tool_effectiveness_repository
        )

    async def _discover_analyzers(self):
        """Dynamically discovers available analyzer classes within the analysis package.

        Scans the `analysis_engine.analysis` package for classes inheriting
        from `BaseAnalyzer` and registers them in `analyzer_classes`.
        """
        """
        Dynamically discover available analyzers in the analysis module
        """
        try:
            # Import the analysis package
            import analysis_engine.analysis as analysis_package

            # Recursively walk through all modules
            for _, name, is_pkg in pkgutil.walk_packages(
                analysis_package.__path__,
                analysis_package.__name__ + "."
            ):
                if is_pkg:
                    continue

                try:
                    # Import the module
                    module = importlib.import_module(name)

                    # Find all analyzer classes in the module
                    for attr_name, attr_value in module.__dict__.items():
                        if (inspect.isclass(attr_value) and
                            issubclass(attr_value, BaseAnalyzer) and
                            attr_value != BaseAnalyzer):

                            # Register the analyzer class
                            analyzer_name = getattr(attr_value, 'name', attr_name.lower())
                            self.analyzer_classes[analyzer_name] = attr_value
                            logger.info(f"Discovered analyzer: {analyzer_name}")
                except Exception as e:
                    logger.warning(f"Error loading module {name}: {e}")
        except Exception as e:
            logger.error(f"Error discovering analyzers: {e}")

    async def _initialize_core_analyzers(self):
        """Initializes core analyzer instances with default settings.

        Instantiates essential analyzers like MarketRegimeAnalyzer,
        MultiTimeframeAnalyzer, etc.
        """
        """
        Initialize core analyzers with default settings
        """
        # Market Regime Analyzer
        self.analyzers["market_regime"] = MarketRegimeAnalyzer()

        # Multi-Timeframe Analyzer
        self.analyzers["multi_timeframe"] = MultiTimeframeAnalyzer()

        # Currency Correlation Analyzer
        self.analyzers["currency_correlation"] = CurrencyCorrelationAnalyzer()

        # Time Cycle Analyzer
        self.analyzers["time_cycle"] = TimeCycleAnalyzer()

        # Confluence Analyzer
        self.analyzers["confluence"] = ConfluenceAnalyzer()

        logger.info(f"Initialized {len(self.analyzers)} core analyzers")

    async def get_analyzer(self, name: str, parameters: Optional[Dict[str, Any]] = None) -> BaseAnalyzer:
        """Retrieves or initializes an analyzer instance by name.

        If the analyzer is already initialized, returns the existing instance.
        Otherwise, if the analyzer class is registered, initializes a new instance
        with the provided parameters.

        Args:
            name: The name of the analyzer to retrieve.
            parameters: Optional parameters to pass to the analyzer's constructor
                if it needs to be initialized.

        Returns:
            The analyzer instance.

        Raises:
            AnalyzerNotFoundError: If the analyzer is not found.
        """
        """
        Get an analyzer instance by name

        Args:
            name: Name of the analyzer
            parameters: Optional parameters for the analyzer

        Returns:
            Analyzer instance

        Raises:
            AnalyzerNotFoundError: If the analyzer is not found
        """
        if not self._initialized:
            await self.initialize()

        # Generate a correlation ID for this operation
        correlation_id = generate_correlation_id()

        # Check if analyzer is already initialized
        if name in self.analyzers:
            return self.analyzers[name]

        # Check if analyzer class is registered
        if name in self.analyzer_classes:
            try:
                analyzer_class = self.analyzer_classes[name]
                analyzer = analyzer_class(
                    tool_effectiveness_repository=self.tool_effectiveness_repository,
                    parameters=parameters
                )
                self.analyzers[name] = analyzer
                return analyzer
            except Exception as e:
                logger.error(f"Error initializing analyzer {name}: {str(e)}", exc_info=True)
                raise AnalysisError(
                    message=f"Failed to initialize analyzer '{name}'",
                    error_code="ANALYZER_INITIALIZATION_ERROR",
                    details={
                        "analyzer_name": name,
                        "parameters": parameters,
                        "error": str(e)
                    },
                    correlation_id=correlation_id
                )

        # Analyzer not found
        available_analyzers = list(self.analyzers.keys()) + [
            name for name in self.analyzer_classes.keys()
            if name not in self.analyzers
        ]

        logger.warning(
            f"Analyzer '{name}' not found. Available analyzers: {available_analyzers}",
            extra={"correlation_id": correlation_id}
        )

        raise AnalyzerNotFoundError(
            analyzer_name=name,
            available_analyzers=available_analyzers,
            correlation_id=correlation_id
        )

    @async_with_exception_handling
    async def run_analysis(
        self, analyzer_name: str, data: Union[MarketData, Dict[str, MarketData]]
    ) -> AnalysisResult:
        """Runs analysis using a specified analyzer.

        Retrieves the analyzer by name and executes its analysis logic
        on the provided market data.

        Args:
            analyzer_name: The name of the analyzer to run.
            data: The market data to analyze. Can be a single MarketData object
                or a dictionary for multi-timeframe analysis.

        Returns:
            An AnalysisResult object containing the analysis output.

        Raises:
            AnalyzerNotFoundError: If the analyzer is not found.
            InsufficientDataError: If there is insufficient data for analysis.
            AnalysisError: If an error occurs during analysis.
            AnalysisTimeoutError: If the analysis times out.
            InvalidAnalysisParametersError: If the analysis parameters are invalid.
        """
        # Generate a correlation ID for this operation
        correlation_id = generate_correlation_id()

        # Validate inputs
        if not analyzer_name:
            raise InvalidAnalysisParametersError(
                message="Analyzer name cannot be empty",
                error_code="EMPTY_ANALYZER_NAME",
                details={"analyzer_name": analyzer_name},
                correlation_id=correlation_id
            )

        if data is None:
            raise InvalidAnalysisParametersError(
                message="Data cannot be null",
                error_code="NULL_DATA",
                details={"analyzer_name": analyzer_name},
                correlation_id=correlation_id
            )

        try:
            # Get the analyzer (this will raise AnalyzerNotFoundError if not found)
            analyzer = await self.get_analyzer(analyzer_name)

            # Check if data is sufficient for single MarketData
            if isinstance(data, MarketData):
                if not hasattr(data, 'is_valid') or not callable(data.is_valid):
                    logger.warning(
                        f"MarketData object does not have is_valid method, cannot validate data for {analyzer_name}",
                        extra={
                            "correlation_id": correlation_id,
                            "analyzer_name": analyzer_name,
                            "data_type": type(data).__name__
                        }
                    )
                elif not data.is_valid():
                    symbol = getattr(data, "symbol", "unknown")
                    timeframe = getattr(data, "timeframe", "unknown")
                    available_points = len(data.close) if hasattr(data, "close") else 0

                    raise InsufficientDataError(
                        message=f"Insufficient data for {analyzer_name} analysis",
                        symbol=symbol,
                        timeframe=timeframe,
                        available_points=available_points,
                        required_points=getattr(analyzer, "min_data_points", "unknown"),
                        correlation_id=correlation_id
                    )

            # Check if data is sufficient for multi-timeframe analysis
            elif isinstance(data, dict):
                if not data:
                    raise InvalidAnalysisParametersError(
                        message="Empty timeframe dictionary provided for analysis",
                        error_code="EMPTY_TIMEFRAME_DICT",
                        details={"analyzer_name": analyzer_name},
                        correlation_id=correlation_id
                    )

                # Check each timeframe's data
                invalid_timeframes = []
                for timeframe, market_data in data.items():
                    if not hasattr(market_data, 'is_valid') or not callable(market_data.is_valid):
                        logger.warning(
                            f"MarketData object for timeframe {timeframe} does not have is_valid method",
                            extra={
                                "correlation_id": correlation_id,
                                "analyzer_name": analyzer_name,
                                "timeframe": timeframe
                            }
                        )
                        continue

                    if not market_data.is_valid():
                        symbol = getattr(market_data, "symbol", "unknown")
                        available_points = len(market_data.close) if hasattr(market_data, "close") else 0

                        invalid_timeframes.append({
                            "timeframe": timeframe,
                            "symbol": symbol,
                            "available_points": available_points
                        })

                if invalid_timeframes:
                    raise InsufficientDataError(
                        message=f"Insufficient data for {analyzer_name} analysis in {len(invalid_timeframes)} timeframes",
                        timeframes=invalid_timeframes,
                        correlation_id=correlation_id
                    )

            # Run the analysis with performance monitoring
            start_time = time.time()

            try:
                # Execute the analysis
                result = await analyzer.execute(data)
                execution_time = time.time() - start_time

                # Log performance metrics
                logger.debug(
                    f"Analysis completed: {analyzer_name}",
                    extra={
                        "correlation_id": correlation_id,
                        "analyzer_name": analyzer_name,
                        "execution_time": execution_time,
                        "is_valid": result.is_valid if hasattr(result, "is_valid") else True
                    }
                )

                # Add correlation ID to result metadata if not already present
                if result and hasattr(result, 'metadata') and 'correlation_id' not in result.metadata:
                    result.metadata['correlation_id'] = correlation_id

                return result

            except asyncio.TimeoutError:
                # Handle timeout specifically
                symbol = getattr(data, "symbol", "unknown") if isinstance(data, MarketData) else "multiple"
                timeframe = getattr(data, "timeframe", "unknown") if isinstance(data, MarketData) else "multiple"

                raise AnalysisTimeoutError(
                    message=f"Analysis timed out for {analyzer_name}",
                    analyzer_name=analyzer_name,
                    symbol=symbol,
                    timeframe=timeframe,
                    timeout_seconds=getattr(settings, 'ANALYSIS_TIMEOUT', 30),
                    correlation_id=correlation_id
                )

        except (AnalyzerNotFoundError, InsufficientDataError, InvalidAnalysisParametersError, AnalysisTimeoutError):
            # Re-raise these specific exceptions
            raise

        except Exception as e:
            # Extract symbol and timeframe information for error context
            symbol = "unknown"
            timeframe = "unknown"

            if isinstance(data, MarketData):
                symbol = getattr(data, "symbol", "unknown")
                timeframe = getattr(data, "timeframe", "unknown")
            elif isinstance(data, dict) and data:
                # For multi-timeframe, use the first timeframe's symbol
                first_timeframe = next(iter(data))
                first_data = data[first_timeframe]
                symbol = getattr(first_data, "symbol", "unknown")
                timeframe = "multiple"

            # Wrap other exceptions in AnalysisError
            logger.error(
                f"Error during analysis with {analyzer_name}: {str(e)}",
                extra={
                    "correlation_id": correlation_id,
                    "analyzer_name": analyzer_name,
                    "symbol": symbol,
                    "timeframe": timeframe
                },
                exc_info=True
            )

            raise AnalysisError(
                message=f"Error during analysis with {analyzer_name}",
                error_code="ANALYSIS_EXECUTION_ERROR",
                details={
                    "analyzer_name": analyzer_name,
                    "error": str(e),
                    "data_type": type(data).__name__,
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "traceback": traceback.format_exc()
                },
                correlation_id=correlation_id
            )

    async def run_multi_timeframe_analysis(
        self,
        market_data: Dict[str, MarketData],
        parameters: Dict[str, Any] = None
    ) -> AnalysisResult:
        """Runs multi-timeframe analysis using the dedicated analyzer.

        Args:
            market_data: A dictionary mapping timeframes (e.g., '1h', '4h') to
                MarketData objects.
            parameters: Optional parameters to pass to the MultiTimeframeAnalyzer.

        Returns:
            An AnalysisResult object containing the multi-timeframe analysis output.

        Raises:
            AnalyzerNotFoundError: If the multi-timeframe analyzer is not found.
            InsufficientDataError: If there is insufficient data for analysis.
            AnalysisError: If an error occurs during analysis.
        """
        """
        Run multi-timeframe analysis

        Args:
            market_data: Dictionary mapping timeframes to MarketData
            parameters: Optional parameters for the analyzer

        Returns:
            Analysis result

        Raises:
            AnalyzerNotFoundError: If the multi-timeframe analyzer is not found
            InsufficientDataError: If there is insufficient data for analysis
            AnalysisError: If an error occurs during analysis
        """
        # Generate a correlation ID for this operation
        correlation_id = generate_correlation_id()

        try:
            # Get the analyzer (this will raise AnalyzerNotFoundError if not found)
            analyzer = await self.get_analyzer("multi_timeframe", parameters)

            # Check if data is sufficient
            if not market_data or len(market_data) == 0:
                raise InsufficientDataError(
                    message="No timeframes provided for multi-timeframe analysis",
                    correlation_id=correlation_id
                )

            # Check each timeframe's data
            for timeframe, data in market_data.items():
                if not data.is_valid():
                    raise InsufficientDataError(
                        message=f"Insufficient data for timeframe {timeframe}",
                        timeframe=timeframe,
                        symbol=getattr(data, "symbol", None),
                        available_points=len(data.close) if hasattr(data, "close") else 0,
                        correlation_id=correlation_id
                    )

            # Run the analysis with performance monitoring
            start_time = time.time()
            result = await analyzer.execute(market_data)
            execution_time = time.time() - start_time

            # Log performance metrics
            logger.debug(
                "Multi-timeframe analysis completed",
                extra={
                    "correlation_id": correlation_id,
                    "timeframes": list(market_data.keys()),
                    "execution_time": execution_time,
                    "is_valid": result.is_valid if hasattr(result, "is_valid") else True
                }
            )

            return result

        except (AnalyzerNotFoundError, InsufficientDataError):
            # Re-raise these specific exceptions
            raise

        except Exception as e:
            # Wrap other exceptions in AnalysisError
            logger.error(
                f"Error during multi-timeframe analysis: {str(e)}",
                extra={"correlation_id": correlation_id},
                exc_info=True
            )

            raise AnalysisError(
                message="Error during multi-timeframe analysis",
                error_code="MULTI_TIMEFRAME_ANALYSIS_ERROR",
                details={
                    "error": str(e),
                    "timeframes": list(market_data.keys()) if market_data else []
                },
                correlation_id=correlation_id
            )

    async def run_confluence_analysis(
        self,
        market_data: Any,
        parameters: Optional[Dict[str, Any]] = None
    ) -> AnalysisResult:
        """Run confluence analysis on market data.

        Confluence analysis combines multiple technical indicators and patterns
        to identify high-probability trading opportunities.

        Args:
            market_data: Market data to analyze
            parameters: Optional parameters for the analyzer

        Returns:
            Analysis result with confluence zones and strength metrics

        Raises:
            AnalyzerNotFoundError: If the confluence analyzer is not found
            InsufficientDataError: If there is insufficient data for analysis
            AnalysisError: If an error occurs during analysis
        """
        """
        Run confluence analysis

        Args:
            market_data: Market data to analyze
            parameters: Optional parameters for the analyzer

        Returns:
            Analysis result

        Raises:
            AnalyzerNotFoundError: If the confluence analyzer is not found
            InsufficientDataError: If there is insufficient data for analysis
            AnalysisError: If an error occurs during analysis
        """
        # Generate a correlation ID for this operation
        correlation_id = generate_correlation_id()

        try:
            # Get the analyzer (this will raise AnalyzerNotFoundError if not found)
            analyzer = await self.get_analyzer("confluence", parameters)

            # Check if data is sufficient
            if isinstance(market_data, MarketData) and not market_data.is_valid():
                raise InsufficientDataError(
                    message="Insufficient data for confluence analysis",
                    symbol=getattr(market_data, "symbol", None),
                    timeframe=getattr(market_data, "timeframe", None),
                    available_points=len(market_data.close) if hasattr(market_data, "close") else 0,
                    correlation_id=correlation_id
                )

            # Run the analysis with performance monitoring
            start_time = time.time()
            result = await analyzer.execute(market_data)
            execution_time = time.time() - start_time

            # Log performance metrics
            logger.debug(
                "Confluence analysis completed",
                extra={
                    "correlation_id": correlation_id,
                    "execution_time": execution_time,
                    "is_valid": result.is_valid if hasattr(result, "is_valid") else True,
                    "symbol": getattr(market_data, "symbol", "unknown") if isinstance(market_data, MarketData) else "multiple"
                }
            )

            return result

        except (AnalyzerNotFoundError, InsufficientDataError):
            # Re-raise these specific exceptions
            raise

        except Exception as e:
            # Wrap other exceptions in AnalysisError
            logger.error(
                f"Error during confluence analysis: {str(e)}",
                extra={"correlation_id": correlation_id},
                exc_info=True
            )

            symbol = getattr(market_data, "symbol", None) if isinstance(market_data, MarketData) else None
            timeframe = getattr(market_data, "timeframe", None) if isinstance(market_data, MarketData) else None

            raise AnalysisError(
                message="Error during confluence analysis",
                error_code="CONFLUENCE_ANALYSIS_ERROR",
                details={
                    "error": str(e),
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "data_type": type(market_data).__name__
                },
                correlation_id=correlation_id
            )

    async def list_available_analyzers(self) -> List[Dict[str, Any]]:
        """
        Get list of available analyzers

        Returns:
            List of analyzer information
        """
        if not self._initialized:
            await self.initialize()

        available = []

        # Add currently initialized analyzers
        for name, analyzer in self.analyzers.items():
            available.append({
                "name": name,
                "type": analyzer.__class__.__name__,
                "active": True,
                "parameters": analyzer.parameters
            })

        # Add discovered but not initialized analyzers
        for name, analyzer_class in self.analyzer_classes.items():
            if name not in self.analyzers:
                available.append({
                    "name": name,
                    "type": analyzer_class.__name__,
                    "active": False,
                    "parameters": {}
                })

        return available

    async def get_analyzer_details(self, analyzer_name: str) -> Optional[Dict[str, Any]]:
        """
        Get details about a specific analyzer

        Args:
            analyzer_name: Name of the analyzer

        Returns:
            Dictionary with analyzer details or None if not found
        """
        if not self._initialized:
            await self.initialize()

        # Check initialized analyzers
        if analyzer_name in self.analyzers:
            analyzer = self.analyzers[analyzer_name]
            performance_metrics = await analyzer.get_performance_metrics()
            return {
                "name": analyzer_name,
                "type": analyzer.__class__.__name__,
                "active": True,
                "parameters": analyzer.parameters,
                "performance": performance_metrics
            }

        # Check discovered but not initialized analyzers
        if analyzer_name in self.analyzer_classes:
            analyzer_class = self.analyzer_classes[analyzer_name]
            return {
                "name": analyzer_name,
                "type": analyzer_class.__name__,
                "active": False,
                "parameters": {},
                "description": analyzer_class.__doc__ or "No description available"
            }

        return None

    async def get_analyzer_performance(self, analyzer_name: str) -> Optional[Dict[str, Any]]:
        """
        Get performance metrics for a specific analyzer

        Args:
            analyzer_name: Name of the analyzer

        Returns:
            Dictionary with performance metrics or None if not found
        """
        analyzer = await self.get_analyzer(analyzer_name)

        if not analyzer:
            return None

        return await analyzer.get_performance_metrics()

    async def get_analyzer_effectiveness(
        self,
        analyzer_name: str,
        timeframe: Optional[str] = None,
        instrument: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Get effectiveness metrics for a specific analyzer

        Args:
            analyzer_name: Name of the analyzer
            timeframe: Optional timeframe filter
            instrument: Optional instrument filter

        Returns:
            Dictionary with effectiveness metrics or None if not found
        """
        analyzer = await self.get_analyzer(analyzer_name)

        if not analyzer:
            return None

        # Get summary from effectiveness logger
        summary = analyzer.effectiveness_logger.get_summary()

        # Apply filters if provided
        if timeframe and "by_timeframe" in summary:
            if timeframe in summary["by_timeframe"]:
                # Return only this timeframe's metrics
                return {
                    "analyzer_name": analyzer_name,
                    "timeframe": timeframe,
                    **summary["by_timeframe"][timeframe]
                }
            else:
                return {
                    "analyzer_name": analyzer_name,
                    "timeframe": timeframe,
                    "error": f"No data for timeframe {timeframe}"
                }

        if instrument and "by_instrument" in summary:
            if instrument in summary["by_instrument"]:
                # Return only this instrument's metrics
                return {
                    "analyzer_name": analyzer_name,
                    "instrument": instrument,
                    **summary["by_instrument"][instrument]
                }
            else:
                return {
                    "analyzer_name": analyzer_name,
                    "instrument": instrument,
                    "error": f"No data for instrument {instrument}"
                }

        # Return full summary
        return summary

    async def record_effectiveness(
        self,
        analyzer_name: str,
        analysis_result: Dict[str, Any],
        actual_outcome: Dict[str, Any],
        timeframe: Optional[str] = None,
        instrument: Optional[str] = None
    ) -> Optional[str]:
        """
        Record effectiveness for a specific analyzer result

        Args:
            analyzer_name: Name of the analyzer
            analysis_result: The original analysis result
            actual_outcome: The actual market outcome
            timeframe: Optional timeframe
            instrument: Optional instrument

        Returns:
            Log ID or None if failed
        """
        analyzer = await self.get_analyzer(analyzer_name)

        if not analyzer:
            logger.warning(f"Cannot record effectiveness: analyzer {analyzer_name} not found")
            return None

        # Convert to AnalysisResult if needed
        if isinstance(analysis_result, dict):
            result = AnalysisResult(
                analyzer_name=analyzer_name,
                result_data=analysis_result.get("result_data", {}),
                is_valid=analysis_result.get("is_valid", True),
                metadata=analysis_result.get("metadata", {})
            )
        else:
            result = analysis_result

        # Log effectiveness
        log_id = analyzer.log_effectiveness(
            analysis_result=result,
            actual_outcome=actual_outcome,
            timeframe=timeframe,
            instrument=instrument
        )

        return log_id
