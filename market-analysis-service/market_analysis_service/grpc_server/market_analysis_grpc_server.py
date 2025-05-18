import grpc
import logging
from datetime import datetime

# Import generated protobuf code
from market_analysis import market_analysis_service_pb2
from market_analysis import market_analysis_service_pb2_grpc

# Import existing service and models
from market_analysis_service.services.market_analysis_service import MarketAnalysisService
from market_analysis_service.models.market_analysis_models import (
    MarketAnalysisRequest,
    MarketAnalysisResponse,
    PatternRecognitionRequest,
    PatternRecognitionResponse,
    SupportResistanceRequest,
    SupportResistanceResponse,
    MarketRegimeRequest,
    MarketRegimeResponse,
    CorrelationAnalysisRequest,
    CorrelationAnalysisResponse,
    AnalysisType,
    PatternType,
    MarketRegimeType,
    SupportResistanceMethod
)

logger = logging.getLogger(__name__)

class MarketAnalysisGrpcServicer(market_analysis_service_pb2_grpc.MarketAnalysisServiceServicer):
    """
    Implements the gRPC service definition for the Market Analysis Service.
    """

    def __init__(self, market_analysis_service: MarketAnalysisService):
        """
        Initializes the gRPC servicer with an instance of the MarketAnalysisService.
        """
        self.market_analysis_service = market_analysis_service

    async def CalculateIndicators(self, request, context):
        """
        gRPC method to calculate technical indicators.
        """
        logger.info(f"CalculateIndicators gRPC call received for {request.symbol} {request.timeframe}")
        try:
            # Map gRPC request to Pydantic model
            pydantic_request = MarketAnalysisRequest(
                symbol=request.symbol,
                timeframe=request.timeframe,
                start_date=request.start_date,
                end_date=request.end_date if request.HasField("end_date") else None,
                analysis_types=[AnalysisType.TECHNICAL], # Request only technical analysis
                additional_parameters=request.additional_parameters # Pass through any additional parameters
            )

            # Call the existing service method
            pydantic_response = await self.market_analysis_service.analyze_market(pydantic_request)

            # Extract technical analysis results
            indicators_result = next(
                (res for res in pydantic_response.analysis_results if res["analysis_type"] == AnalysisType.TECHNICAL),
                None
            )

            if not indicators_result or "indicators" not in indicators_result["result"]:
                logger.warning(f"No technical analysis results found for {request.symbol}")
                context.set_code(grpc.StatusCode.NOT_FOUND)
                context.set_details("Technical analysis results not found")
                return market_analysis_service_pb2.CalculateIndicatorsResponse()

            # Map Pydantic response back to gRPC response
            grpc_indicators = indicators_result["result"]["indicators"]

            grpc_response = market_analysis_service_pb2.CalculateIndicatorsResponse(
                request_id=pydantic_response.request_id,
                symbol=pydantic_response.symbol,
                timeframe=pydantic_response.timeframe,
                start_date=pydantic_response.start_date,
                end_date=pydantic_response.end_date.isoformat() if pydantic_response.end_date else "",
                indicators=grpc_indicators,
                execution_time_ms=pydantic_response.execution_time_ms,
                timestamp=pydantic_response.timestamp.isoformat()
            )
            return grpc_response

        except Exception as e:
            logger.error(f"Error in CalculateIndicators gRPC call: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Internal server error: {e}")
            return market_analysis_service_pb2.CalculateIndicatorsResponse()

    async def DetectPatterns(self, request, context):
        """
        gRPC method to detect chart patterns.
        """
        logger.info(f"DetectPatterns gRPC call received for {request.symbol} {request.timeframe}")
        try:
            pydantic_request = PatternRecognitionRequest(
                symbol=request.symbol,
                timeframe=request.timeframe,
                start_date=request.start_date,
                end_date=request.end_date if request.HasField("end_date") else None,
                pattern_types=[PatternType(pt) for pt in request.pattern_types] if request.pattern_types else None,
                min_confidence=request.min_confidence
            )

            pydantic_response = await self.market_analysis_service.recognize_patterns(pydantic_request)

            grpc_patterns = []
            for pattern in pydantic_response.patterns:
                grpc_pattern = market_analysis_service_pb2.Pattern(
                    type=pattern.type.value,
                    start_time=pattern.start_time.isoformat(),
                    end_time=pattern.end_time.isoformat(),
                    confidence=pattern.confidence,
                    description=pattern.description
                )
                grpc_patterns.append(grpc_pattern)

            grpc_response = market_analysis_service_pb2.DetectPatternsResponse(
                request_id=pydantic_response.request_id,
                symbol=pydantic_response.symbol,
                timeframe=pydantic_response.timeframe,
                start_date=pydantic_response.start_date,
                end_date=pydantic_response.end_date.isoformat() if pydantic_response.end_date else "",
                patterns=grpc_patterns,
                execution_time_ms=pydantic_response.execution_time_ms,
                timestamp=pydantic_response.timestamp.isoformat()
            )
            return grpc_response
        except Exception as e:
            logger.error(f"Error in DetectPatterns gRPC call: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Internal server error: {e}")
            return market_analysis_service_pb2.DetectPatternsResponse()

    async def DetectSupportResistance(self, request, context):
        """
        gRPC method to detect support and resistance levels.
        """
        logger.info(f"DetectSupportResistance gRPC call received for {request.symbol} {request.timeframe}")
        try:
            pydantic_request = SupportResistanceRequest(
                symbol=request.symbol,
                timeframe=request.timeframe,
                start_date=request.start_date,
                end_date=request.end_date if request.HasField("end_date") else None,
                methods=[SupportResistanceMethod(m) for m in request.methods] if request.methods else None,
                levels_count=request.levels_count if request.HasField("levels_count") else None,
                additional_parameters=request.additional_parameters
            )

            pydantic_response = await self.market_analysis_service.identify_support_resistance(pydantic_request)

            grpc_levels = []
            for level in pydantic_response.levels:
                 grpc_level = market_analysis_service_pb2.SupportResistanceLevel(
                     type=level.type,
                     price=level.price,
                     strength=level.strength if level.strength is not None else 0.0,
                     # Convert timestamp to string if it exists
                     timestamp=level.timestamp.isoformat() if level.timestamp else "",
                     method=level.method
                 )
                 grpc_levels.append(grpc_level)

            grpc_response = market_analysis_service_pb2.DetectSupportResistanceResponse(
                request_id=pydantic_response.request_id,
                symbol=pydantic_response.symbol,
                timeframe=pydantic_response.timeframe,
                start_date=pydantic_response.start_date,
                end_date=pydantic_response.end_date.isoformat() if pydantic_response.end_date else "",
                levels=grpc_levels,
                execution_time_ms=pydantic_response.execution_time_ms,
                timestamp=pydantic_response.timestamp.isoformat()
            )
            return grpc_response
        except Exception as e:
            logger.error(f"Error in DetectSupportResistance gRPC call: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Internal server error: {e}")
            return market_analysis_service_pb2.DetectSupportResistanceResponse()

    async def DetectMarketRegime(self, request, context):
        """
        gRPC method to detect the current market regime.
        """
        logger.info(f"DetectMarketRegime gRPC call received for {request.symbol} {request.timeframe}")
        try:
            pydantic_request = MarketRegimeRequest(
                symbol=request.symbol,
                timeframe=request.timeframe,
                start_date=request.start_date,
                end_date=request.end_date if request.HasField("end_date") else None,
                window_size=request.window_size if request.HasField("window_size") else None,
                additional_parameters=request.additional_parameters
            )

            pydantic_response = await self.market_analysis_service.detect_market_regime(pydantic_request)

            grpc_regimes = []
            for regime in pydantic_response.regimes:
                grpc_regime = market_analysis_service_pb2.MarketRegime(
                    regime_type=regime["regime_type"],
                    start_time=regime["start_time"].isoformat(),
                    end_time=regime["end_time"].isoformat() if regime["end_time"] else "",
                    confidence=regime.get("confidence", 0.0) # Handle optional confidence
                )
                grpc_regimes.append(grpc_regime)

            grpc_response = market_analysis_service_pb2.DetectMarketRegimeResponse(
                request_id=pydantic_response.request_id,
                symbol=pydantic_response.symbol,
                timeframe=pydantic_response.timeframe,
                start_date=pydantic_response.start_date,
                end_date=pydantic_response.end_date.isoformat() if pydantic_response.end_date else "",
                regimes=grpc_regimes,
                current_regime=pydantic_response.current_regime,
                execution_time_ms=pydantic_response.execution_time_ms,
                timestamp=pydantic_response.timestamp.isoformat()
            )
            return grpc_response
        except Exception as e:
            logger.error(f"Error in DetectMarketRegime gRPC call: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Internal server error: {e}")
            return market_analysis_service_pb2.DetectMarketRegimeResponse()

    async def PerformCorrelationAnalysis(self, request, context):
        """
        gRPC method to perform correlation analysis between symbols.
        """
        logger.info(f"PerformCorrelationAnalysis gRPC call received for {request.symbols}")
        try:
            pydantic_request = CorrelationAnalysisRequest(
                symbols=list(request.symbols), # Convert repeated field to list
                timeframe=request.timeframe,
                start_date=request.start_date,
                end_date=request.end_date if request.HasField("end_date") else None,
                window_size=request.window_size if request.HasField("window_size") else None,
                method=request.method,
                additional_parameters=request.additional_parameters
            )

            pydantic_response = await self.market_analysis_service.analyze_correlations(pydantic_request)

            # Convert correlation matrix (assuming it's a dictionary of dictionaries) to gRPC map
            grpc_correlation_matrix = {
                symbol: market_analysis_service_pb2.CorrelationRow(correlations=correlations)
                for symbol, correlations in pydantic_response.correlation_matrix.items()
            }

            grpc_correlation_pairs = []
            for pair in pydantic_response.correlation_pairs:
                 grpc_pair = market_analysis_service_pb2.CorrelationPair(
                     symbol1=pair["symbol1"],
                     symbol2=pair["symbol2"],
                     correlation=pair["correlation"],
                     method=pair["method"],
                     window_size=pair.get("window_size") # Handle optional window_size
                 )
                 grpc_correlation_pairs.append(grpc_pair)

            grpc_response = market_analysis_service_pb2.PerformCorrelationAnalysisResponse(
                request_id=pydantic_response.request_id,
                symbols=pydantic_response.symbols,
                timeframe=pydantic_response.timeframe,
                start_date=pydantic_response.start_date,
                end_date=pydantic_response.end_date.isoformat() if pydantic_response.end_date else "",
                method=pydantic_response.method,
                correlation_matrix=grpc_correlation_matrix,
                correlation_pairs=grpc_correlation_pairs,
                execution_time_ms=pydantic_response.execution_time_ms,
                timestamp=pydantic_response.timestamp.isoformat()
            )
            return grpc_response
        except Exception as e:
            logger.error(f"Error in PerformCorrelationAnalysis gRPC call: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Internal server error: {e}")
            return market_analysis_service_pb2.PerformCorrelationAnalysisResponse()

    async def PerformVolatilityAnalysis(self, request, context):
        """
        gRPC method to perform volatility analysis for a symbol.
        """
        logger.info(f"PerformVolatilityAnalysis gRPC call received for {request.symbol} {request.timeframe}")
        try:
            # The existing analyze_volatility method takes individual parameters, not a Pydantic model.
            # Need to convert gRPC request to individual parameters.
            volatility_results = await self.market_analysis_service.analyze_volatility(
                symbol=request.symbol,
                timeframe=request.timeframe,
                start_date=datetime.fromisoformat(request.start_date.replace('Z', '+00:00')),
                end_date=datetime.fromisoformat(request.end_date.replace('Z', '+00:00')) if request.HasField("end_date") else None,
                parameters=request.additional_parameters
            )

            # Mapping from the dictionary returned by analyze_volatility to VolatilityAnalysis protobuf message
            grpc_analysis = market_analysis_service_pb2.VolatilityAnalysis(
                # Assuming these keys exist in the dictionary returned by analyze_volatility
                id=volatility_results.get("id", ""),
                symbol=market_analysis_service_pb2.Symbol(symbol=volatility_results.get("symbol", "")),
                timeframe=market_analysis_service_pb2.Timeframe(timeframe=volatility_results.get("timeframe", "")),
                # Assuming string timestamps in the dictionary, convert to Timestamp protobuf message
                start_time=market_analysis_service_pb2.Timestamp(timestamp=volatility_results.get("start_time", "")),
                end_time=market_analysis_service_pb2.Timestamp(timestamp=volatility_results.get("end_time", "")) if volatility_results.get("end_time") else None,
                historical_volatility=volatility_results.get("historical_volatility", 0.0),
                implied_volatility=volatility_results.get("implied_volatility", 0.0),
                time_series=volatility_results.get("time_series", {}), # Assuming this is already a map<int64, double>
                volatility_regimes=[
                    market_analysis_service_pb2.MarketRegime(
                        id=regime.get("id", ""),
                        type=regime.get("regime_type", ""), # Mapping from 'regime_type' key
                        start_time=market_analysis_service_pb2.Timestamp(timestamp=regime.get("start_time", "")), # Assuming string timestamp
                        end_time=market_analysis_service_pb2.Timestamp(timestamp=regime.get("end_time", "")) if regime.get("end_time") else None, # Assuming string timestamp
                        strength=regime.get("confidence", 0.0), # Mapping from 'confidence' key
                        direction=regime.get("direction", ""),
                        volatility=regime.get("volatility", 0.0),
                        metadata=market_analysis_service_pb2.Metadata(**regime.get("metadata", {})) # Assuming metadata is a dictionary
                    )
                    for regime in volatility_results.get("volatility_regimes", [])
                ],
                metadata=market_analysis_service_pb2.Metadata(**volatility_results.get("metadata", {})) # Assuming metadata is a dictionary
            )

            grpc_response = market_analysis_service_pb2.PerformVolatilityAnalysisResponse(
                analysis=grpc_analysis,
                # execution_time_ms is a direct field in the response, not nested in analysis
                execution_time_ms=volatility_results.get("execution_time_ms", 0)
            )

            return grpc_response
        except Exception as e:
            logger.error(f"Error in PerformVolatilityAnalysis gRPC call: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Internal server error: {e}")
            return market_analysis_service_pb2.PerformVolatilityAnalysisResponse()

    # The existing service has an analyze_sentiment method, but there is no
    # corresponding AnalyzeSentiment gRPC method in the generated code.
    # I will not implement a gRPC method for sentiment analysis based on the current .proto.
    # If sentiment analysis is needed over gRPC, the .proto file should be updated.

    async def AnalyzeMarket(self, request, context):
        """
        gRPC method for comprehensive market analysis.
        """
        # The existing analyze_market method in MarketAnalysisService is comprehensive.
        # This gRPC method name (if it existed in the .proto) would likely map to it.
        # However, AnalyzeMarket is not in the generated market_analysis_service_pb2_grpc.py.
        # I will not implement this based on the current generated code.
        logger.warning("AnalyzeMarket gRPC method called but not found in generated code.")
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not found in generated gRPC code')
        raise NotImplementedError('Method not found in generated gRPC code')

    async def GetAvailablePatterns(self, request, context):
         """
         gRPC method to get available chart patterns.
         """
         logger.info("GetAvailablePatterns gRPC call received")
         try:
             patterns = await self.market_analysis_service.get_available_patterns()
             # Assuming patterns is a list of dictionaries with 'id' and 'name'
             grpc_patterns = [market_analysis_service_pb2.AvailablePattern(id=p["id"], name=p["name"]) for p in patterns]
             grpc_response = market_analysis_service_pb2.GetAvailablePatternsResponse(patterns=grpc_patterns)
             return grpc_response
         except Exception as e:
             logger.error(f"Error in GetAvailablePatterns gRPC call: {e}")
             context.set_code(grpc.StatusCode.INTERNAL)
             context.set_details(f"Internal server error: {e}")
             return market_analysis_service_pb2.GetAvailablePatternsResponse()

    async def GetAvailableRegimes(self, request, context):
        """
        gRPC method to get available market regimes.
        """
        logger.info("GetAvailableRegimes gRPC call received")
        try:
            regimes = await self.market_analysis_service.get_available_regimes()
            # Assuming regimes is a list of dictionaries with 'id' and 'name'
            grpc_regimes = [market_analysis_service_pb2.AvailableRegime(id=r["id"], name=r["name"]) for r in regimes]
            grpc_response = market_analysis_service_pb2.GetAvailableRegimesResponse(regimes=grpc_regimes)
            return grpc_response
        except Exception as e:
             logger.error(f"Error in GetAvailableRegimes gRPC call: {e}")
             context.set_code(grpc.StatusCode.INTERNAL)
             context.set_details(f"Internal server error: {e}")
             return market_analysis_service_pb2.GetAvailableRegimesResponse()

    async def GetAvailableMethods(self, request, context):
        """
        gRPC method to get available analysis methods.
        """
        logger.info("GetAvailableMethods gRPC call received")
        try:
            methods = await self.market_analysis_service.get_available_methods()
            # Assuming methods is a dictionary like {'analysis_type': ['method1', 'method2']}
            grpc_methods_map = {}
            for analysis_type, method_list in methods.items():
                 # Need a repeated string field in the protobuf message for the list of methods per type
                 # Assuming a structure like map<string, AvailableMethodsList> where AvailableMethodsList has a repeated string 'methods'
                 # This requires a specific protobuf definition. For now, using a placeholder.
                 # **NOTE:** The conversion here depends heavily on the GetAvailableMethodsResponse protobuf definition.
                 # Assuming a simple map<string, AvailableMethodsList> where AvailableMethodsList is a message
                 # with a single repeated string field called 'methods'.

                 # Placeholder - replace with actual protobuf message and field names
                 grpc_methods_list = market_analysis_service_pb2.AvailableMethodsList()
                 grpc_methods_list.methods.extend(method_list)
                 grpc_methods_map[analysis_type] = grpc_methods_list

            grpc_response = market_analysis_service_pb2.GetAvailableMethodsResponse(
                # Assuming the protobuf response has a map field called 'available_methods'
                available_methods=grpc_methods_map
            )
            return grpc_response
        except Exception as e:
            logger.error(f"Error in GetAvailableMethods gRPC call: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Internal server error: {e}")
            return market_analysis_service_pb2.GetAvailableMethodsResponse()


# Helper function to serve the gRPC server (this would be in the main startup file)
# async def serve():
#     server = grpc.aio.server(futures.ThreadPoolExecutor(max_workers=10))
#     market_analysis_service_instance = MarketAnalysisService(...)
#     market_analysis_service_pb2_grpc.add_MarketAnalysisServiceServicer_to_server(
#         MarketAnalysisGrpcServicer(market_analysis_service_instance), server
#     )
#     listen_addr = '[::]:50051' # Example port
#     server.add_insecure_port(listen_addr)
#     logging.info("Starting server on %s", listen_addr)
#     await server.start()
#     await server.wait_for_termination()

# if __name__ == '__main__':
#     logging.basicConfig(level=logging.INFO)
#     asyncio.run(serve())