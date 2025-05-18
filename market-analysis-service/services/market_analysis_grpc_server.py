import grpc
import logging
from datetime import datetime
from typing import List, Dict, Any

# Import generated gRPC code
import common_lib.grpc.market_analysis.market_analysis_service_pb2 as market_analysis_service_pb2
import common_lib.grpc.market_analysis.market_analysis_service_pb2_grpc as market_analysis_service_pb2_grpc
import common_lib.grpc.common.common_types_pb2 as common_types_pb2
import common_lib.grpc.common.error_types_pb2 as error_types_pb2

# Import existing service dependencies
from market_analysis_service.services.market_analysis_service import MarketAnalysisService
from market_analysis_service.adapters.data_pipeline_adapter import DataPipelineAdapter
from market_analysis_service.adapters.analysis_coordinator_adapter import AnalysisCoordinatorAdapter
from market_analysis_service.adapters.feature_store_adapter import FeatureStoreAdapter
from market_analysis_service.repositories.analysis_repository import AnalysisRepository

logger = logging.getLogger(__name__)

class MarketAnalysisServicer(market_analysis_service_pb2_grpc.MarketAnalysisServiceServicer):
    """
    gRPC server implementation for the Market Analysis Service.
    """

    def __init__(self, market_analysis_service: MarketAnalysisService):
        """
        Initialize the gRPC servicer with the core MarketAnalysisService.

        Args:
            market_analysis_service: An instance of the core MarketAnalysisService.
        """
        self.market_analysis_service = market_analysis_service

    async def CalculateIndicators(
        self,
        request: market_analysis_service_pb2.CalculateIndicatorsRequest,
        context: grpc.ServicerContext
    ) -> market_analysis_service_pb2.CalculateIndicatorsResponse:
        """
        Implements the CalculateIndicators RPC method.
        """
        logger.info(f"Received CalculateIndicators request for symbol: {request.symbol.symbol}, timeframe: {request.timeframe.timeframe}")

        try:
            # Convert protobuf Timestamp to datetime
            start_date = datetime.fromtimestamp(request.start_date.seconds)
            end_date = datetime.fromtimestamp(request.end_date.seconds) if request.end_date.seconds else None

            # Get market data with indicators
            data = await self.market_analysis_service.data_pipeline_adapter.get_market_data(
                symbol=request.symbol.symbol,
                timeframe=request.timeframe.timeframe,
                start_date=start_date,
                end_date=end_date,
                include_indicators=True
            )

            if data.empty:
                return market_analysis_service_pb2.CalculateIndicatorsResponse(
                    error=error_types_pb2.ErrorResponse(
                        code=common_types_pb2.ErrorCode.NOT_FOUND,
                        message="No market data found for the specified criteria."
                    )
                )

            # Extract indicators
            indicators = []
            indicator_columns = [col for col in data.columns if col not in ["open", "high", "low", "close", "volume", "timestamp"]]

            for indicator_name in indicator_columns:
                if request.indicators and indicator_name not in request.indicators:
                    continue

                indicator_values = {}
                for index, row in data.iterrows():
                    timestamp_seconds = int(row['timestamp'].timestamp())
                    indicator_values[timestamp_seconds] = row[indicator_name]

                indicator_proto = market_analysis_service_pb2.Indicator(
                    name=indicator_name,
                    values=indicator_values,
                    parameters=request.parameters,
                    metadata=common_types_pb2.Metadata()  # Placeholder
                )
                indicators.append(indicator_proto)

            return market_analysis_service_pb2.CalculateIndicatorsResponse(
                indicators=indicators
            )

        except Exception as e:
            logger.error(f"Error calculating indicators for {request.symbol.symbol}: {e}", exc_info=True)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"An internal error occurred: {e}")
            return market_analysis_service_pb2.CalculateIndicatorsResponse(
                error=error_types_pb2.ErrorResponse(
                    code=common_types_pb2.ErrorCode.INTERNAL_ERROR,
                    message=f"An internal error occurred: {e}"
                )
            )

    async def DetectPatterns(
        self,
        request: market_analysis_service_pb2.DetectPatternsRequest,
        context: grpc.ServicerContext
    ) -> market_analysis_service_pb2.DetectPatternsResponse:
        """
        Implements the DetectPatterns RPC method.
        """
        logger.info(f"Received DetectPatterns request for symbol: {request.symbol.symbol}, timeframe: {request.timeframe.timeframe}")

        try:
            # Convert protobuf Timestamp to datetime
            start_date = datetime.fromtimestamp(request.start_date.seconds)
            end_date = datetime.fromtimestamp(request.end_date.seconds) if request.end_date.seconds else None

            # Detect patterns
            patterns = await self.market_analysis_service.detect_patterns(
                symbol=request.symbol.symbol,
                timeframe=request.timeframe.timeframe,
                start_date=start_date,
                end_date=end_date,
                pattern_types=request.pattern_types,
                min_strength=request.min_strength
            )

            # Convert patterns to protobuf format
            proto_patterns = []
            for pattern in patterns:
                proto_pattern = market_analysis_service_pb2.Pattern(
                    id=pattern.get("id", ""),
                    type=pattern.get("type", ""),
                    start_time=market_analysis_service_pb2.Timestamp(seconds=int(pattern.get("start_time", datetime.now()).timestamp())),
                    end_time=market_analysis_service_pb2.Timestamp(seconds=int(pattern.get("end_time", datetime.now()).timestamp())),
                    strength=pattern.get("strength", 0.0),
                    direction=pattern.get("direction", ""),
                    target_price=pattern.get("target_price", 0.0),
                    stop_loss_price=pattern.get("stop_loss_price", 0.0),
                    points=pattern.get("points", {}),
                    metadata=common_types_pb2.Metadata()  # Placeholder
                )
                proto_patterns.append(proto_pattern)

            return market_analysis_service_pb2.DetectPatternsResponse(
                patterns=proto_patterns
            )

        except Exception as e:
            logger.error(f"Error detecting patterns for {request.symbol.symbol}: {e}", exc_info=True)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"An internal error occurred: {e}")
            return market_analysis_service_pb2.DetectPatternsResponse(
                error=error_types_pb2.ErrorResponse(
                    code=common_types_pb2.ErrorCode.INTERNAL_ERROR,
                    message=f"An internal error occurred: {e}"
                )
            )

    async def DetectSupportResistance(
        self,
        request: market_analysis_service_pb2.DetectSupportResistanceRequest,
        context: grpc.ServicerContext
    ) -> market_analysis_service_pb2.DetectSupportResistanceResponse:
        """
        Implements the DetectSupportResistance RPC method.
        """
        logger.info(f"Received DetectSupportResistance request for symbol: {request.symbol.symbol}, timeframe: {request.timeframe.timeframe}")

        try:
            # Convert protobuf Timestamp to datetime
            start_date = datetime.fromtimestamp(request.start_date.seconds)
            end_date = datetime.fromtimestamp(request.end_date.seconds) if request.end_date.seconds else None

            # Detect support and resistance levels
            levels = await self.market_analysis_service.detect_support_resistance(
                symbol=request.symbol.symbol,
                timeframe=request.timeframe.timeframe,
                start_date=start_date,
                end_date=end_date,
                method=request.method,
                min_strength=request.min_strength
            )

            # Convert levels to protobuf format
            proto_levels = []
            for level in levels:
                proto_level = market_analysis_service_pb2.Level(
                    id=level.get("id", ""),
                    type=level.get("type", ""),
                    price=level.get("price", 0.0),
                    strength=level.get("strength", 0.0),
                    start_time=market_analysis_service_pb2.Timestamp(seconds=int(level.get("start_time", datetime.now()).timestamp())),
                    end_time=market_analysis_service_pb2.Timestamp(seconds=int(level.get("end_time", datetime.now()).timestamp())) if level.get("end_time") else None,
                    touches=level.get("touches", 0),
                    metadata=common_types_pb2.Metadata()  # Placeholder
                )
                proto_levels.append(proto_level)

            return market_analysis_service_pb2.DetectSupportResistanceResponse(
                levels=proto_levels
            )

        except Exception as e:
            logger.error(f"Error detecting support/resistance for {request.symbol.symbol}: {e}", exc_info=True)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"An internal error occurred: {e}")
            return market_analysis_service_pb2.DetectSupportResistanceResponse(
                error=error_types_pb2.ErrorResponse(
                    code=common_types_pb2.ErrorCode.INTERNAL_ERROR,
                    message=f"An internal error occurred: {e}"
                )
            )

    async def DetectMarketRegime(
        self,
        request: market_analysis_service_pb2.DetectMarketRegimeRequest,
        context: grpc.ServicerContext
    ) -> market_analysis_service_pb2.DetectMarketRegimeResponse:
        """
        Implements the DetectMarketRegime RPC method.
        """
        logger.info(f"Received DetectMarketRegime request for symbol: {request.symbol.symbol}, timeframe: {request.timeframe.timeframe}")

        try:
            # Convert protobuf Timestamp to datetime
            start_date = datetime.fromtimestamp(request.start_date.seconds)
            end_date = datetime.fromtimestamp(request.end_date.seconds) if request.end_date.seconds else None

            # Detect market regime
            regimes = await self.market_analysis_service.detect_market_regime(
                symbol=request.symbol.symbol,
                timeframe=request.timeframe.timeframe,
                start_date=start_date,
                end_date=end_date,
                method=request.method
            )

            # Convert regimes to protobuf format
            proto_regimes = []
            for regime in regimes:
                proto_regime = market_analysis_service_pb2.MarketRegime(
                    id=regime.get("id", ""),
                    type=regime.get("type", ""),
                    start_time=market_analysis_service_pb2.Timestamp(seconds=int(regime.get("start_time", datetime.now()).timestamp())),
                    end_time=market_analysis_service_pb2.Timestamp(seconds=int(regime.get("end_time", datetime.now()).timestamp())) if regime.get("end_time") else None,
                    strength=regime.get("strength", 0.0),
                    direction=regime.get("direction", ""),
                    volatility=regime.get("volatility", 0.0),
                    metadata=common_types_pb2.Metadata()  # Placeholder
                )
                proto_regimes.append(proto_regime)

            return market_analysis_service_pb2.DetectMarketRegimeResponse(
                regimes=proto_regimes
            )

        except Exception as e:
            logger.error(f"Error detecting market regime for {request.symbol.symbol}: {e}", exc_info=True)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"An internal error occurred: {e}")
            return market_analysis_service_pb2.DetectMarketRegimeResponse(
                error=error_types_pb2.ErrorResponse(
                    code=common_types_pb2.ErrorCode.INTERNAL_ERROR,
                    message=f"An internal error occurred: {e}"
                )
            )

    async def PerformCorrelationAnalysis(
        self,
        request: market_analysis_service_pb2.PerformCorrelationAnalysisRequest,
        context: grpc.ServicerContext
    ) -> market_analysis_service_pb2.PerformCorrelationAnalysisResponse:
        """
        Implements the PerformCorrelationAnalysis RPC method.
        """
        logger.info(f"Received PerformCorrelationAnalysis request for {len(request.symbols)} symbols, timeframe: {request.timeframe.timeframe}")

        try:
            # Convert protobuf Timestamp to datetime
            start_date = datetime.fromtimestamp(request.start_date.seconds)
            end_date = datetime.fromtimestamp(request.end_date.seconds) if request.end_date.seconds else None

            # Perform correlation analysis
            correlations = await self.market_analysis_service.perform_correlation_analysis(
                symbols=request.symbols,
                timeframe=request.timeframe.timeframe,
                start_date=start_date,
                end_date=end_date,
                method=request.method,
                window_size=request.window_size
            )

            # Convert correlations to protobuf format
            proto_correlations = []
            for correlation in correlations:
                # Create time series map
                time_series = {}
                for ts_data in correlation.get("time_series", []):
                    timestamp_seconds = int(ts_data.get("timestamp", datetime.now()).timestamp())
                    time_series[timestamp_seconds] = ts_data.get("value", 0.0)

                proto_correlation = market_analysis_service_pb2.Correlation(
                    id=correlation.get("id", ""),
                    symbol1=common_types_pb2.Symbol(symbol=correlation.get("symbol1", "")),
                    symbol2=common_types_pb2.Symbol(symbol=correlation.get("symbol2", "")),
                    coefficient=correlation.get("coefficient", 0.0),
                    start_time=market_analysis_service_pb2.Timestamp(seconds=int(correlation.get("start_time", datetime.now()).timestamp())),
                    end_time=market_analysis_service_pb2.Timestamp(seconds=int(correlation.get("end_time", datetime.now()).timestamp())),
                    time_series=time_series,
                    metadata=common_types_pb2.Metadata()  # Placeholder
                )
                proto_correlations.append(proto_correlation)

            return market_analysis_service_pb2.PerformCorrelationAnalysisResponse(
                correlations=proto_correlations
            )

        except Exception as e:
            logger.error(f"Error performing correlation analysis: {e}", exc_info=True)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"An internal error occurred: {e}")
            return market_analysis_service_pb2.PerformCorrelationAnalysisResponse(
                error=error_types_pb2.ErrorResponse(
                    code=common_types_pb2.ErrorCode.INTERNAL_ERROR,
                    message=f"An internal error occurred: {e}"
                )
            )

    async def PerformVolatilityAnalysis(
        self,
        request: market_analysis_service_pb2.PerformVolatilityAnalysisRequest,
        context: grpc.ServicerContext
    ) -> market_analysis_service_pb2.PerformVolatilityAnalysisResponse:
        """
        Implements the PerformVolatilityAnalysis RPC method.
        """
        logger.info(f"Received PerformVolatilityAnalysis request for symbol: {request.symbol.symbol}, timeframe: {request.timeframe.timeframe}")

        try:
            # Convert protobuf Timestamp to datetime
            start_date = datetime.fromtimestamp(request.start_date.seconds)
            end_date = datetime.fromtimestamp(request.end_date.seconds) if request.end_date.seconds else None

            # Perform volatility analysis
            volatility_data = await self.market_analysis_service.perform_volatility_analysis(
                symbol=request.symbol.symbol,
                timeframe=request.timeframe.timeframe,
                start_date=start_date,
                end_date=end_date,
                method=request.method,
                window_size=request.window_size
            )

            # Convert regimes to volatility analysis
            proto_regimes = []
            for regime in volatility_data.get("regimes", []):
                proto_regime = market_analysis_service_pb2.MarketRegime(
                    id=regime.get("id", ""),
                    type=regime.get("type", ""),
                    start_time=market_analysis_service_pb2.Timestamp(seconds=int(regime.get("start_time", datetime.now()).timestamp())),
                    end_time=market_analysis_service_pb2.Timestamp(seconds=int(regime.get("end_time", datetime.now()).timestamp())) if regime.get("end_time") else None,
                    strength=regime.get("strength", 0.0),
                    direction=regime.get("direction", ""),
                    volatility=regime.get("volatility", 0.0),
                    metadata=common_types_pb2.Metadata()  # Placeholder
                )
                proto_regimes.append(proto_regime)

            # Create time series map
            time_series = {}
            for ts_data in volatility_data.get("time_series", []):
                timestamp_seconds = int(ts_data.get("timestamp", datetime.now()).timestamp())
                time_series[timestamp_seconds] = ts_data.get("value", 0.0)

            # Create volatility analysis response
            proto_volatility_analysis = market_analysis_service_pb2.VolatilityAnalysis(
                id=volatility_data.get("id", ""),
                symbol=common_types_pb2.Symbol(symbol=volatility_data.get("symbol", "")),
                timeframe=common_types_pb2.Timeframe(timeframe=volatility_data.get("timeframe", "")),
                start_time=market_analysis_service_pb2.Timestamp(seconds=int(volatility_data.get("start_time", datetime.now()).timestamp())),
                end_time=market_analysis_service_pb2.Timestamp(seconds=int(volatility_data.get("end_time", datetime.now()).timestamp())),
                historical_volatility=volatility_data.get("historical_volatility", 0.0),
                implied_volatility=volatility_data.get("implied_volatility", 0.0),
                time_series=time_series,
                volatility_regimes=proto_regimes,
                metadata=common_types_pb2.Metadata()  # Placeholder
            )

            return market_analysis_service_pb2.PerformVolatilityAnalysisResponse(
                analysis=proto_volatility_analysis
            )

        except Exception as e:
            logger.error(f"Error performing volatility analysis for {request.symbol.symbol}: {e}", exc_info=True)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"An internal error occurred: {e}")
            return market_analysis_service_pb2.PerformVolatilityAnalysisResponse(
                error=error_types_pb2.ErrorResponse(
                    code=common_types_pb2.ErrorCode.INTERNAL_ERROR,
                    message=f"An internal error occurred: {e}"
                )
            )
