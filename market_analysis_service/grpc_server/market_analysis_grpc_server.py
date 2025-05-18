"""
Market Analysis Service gRPC Server Implementation.
"""

"""
Market Analysis Service gRPC Server Implementation.
"""

import grpc
import asyncio
from concurrent import futures
from datetime import datetime

# Import standardized logging setup
from common_lib.monitoring import setup_logging, get_logger

# Import generated protobuf stubs
from common_lib.grpc.market_analysis import market_analysis_service_pb2
from common_lib.grpc.market_analysis import market_analysis_service_pb2_grpc
from common_lib.grpc.common import common_types_pb2
from common_lib.grpc.common import error_types_pb2

# Import monitoring and tracing components
from common_lib.monitoring import MetricsManager, TracingManager, track_time, trace_function

# Import existing Market Analysis Service logic
from market_analysis_service.services.market_analysis_service import MarketAnalysisService as CoreMarketAnalysisService
from market_analysis_service.models.market_analysis_models import MarketAnalysisRequest, AnalysisType
from market_analysis_service.utils.dependency_injection import get_market_analysis_service



class MarketAnalysisServiceServicer(market_analysis_service_pb2_grpc.MarketAnalysisServiceServicer):
    """
    Implements the gRPC methods for the Market Analysis Service.
    """

    def __init__(self, market_analysis_logic: CoreMarketAnalysisService):
        """
        Initializes the servicer.
        Args:
            market_analysis_logic: An instance of the core MarketAnalysisService logic.
        """
        self.market_analysis_logic = market_analysis_logic
        self.logger = logger

    @trace_function
    async def CalculateIndicators(self, request: market_analysis_service_pb2.CalculateIndicatorsRequest, context: grpc.aio.ServicerContext) -> market_analysis_service_pb2.CalculateIndicatorsResponse:
        """
        Handles the CalculateIndicators gRPC call.
        """
        method_name = "CalculateIndicators"
        self.logger.info(f"{method_name} request received for symbol: {request.symbol.name}")
        response = market_analysis_service_pb2.CalculateIndicatorsResponse()
        status = "UNKNOWN"
        start_time = time.time()
        try:
            # Increment total requests counter
            grpc_requests_total.labels(method=method_name, status="IN_PROGRESS").inc()

            # Convert gRPC request to internal service request model
            # Convert gRPC request to internal service request model
            service_request = MarketAnalysisRequest(
                symbol=request.symbol.name,
                timeframe=request.timeframe.name,
                start_date=datetime.fromtimestamp(request.start_date.seconds).isoformat(),
                end_date=datetime.fromtimestamp(request.end_date.seconds).isoformat() if request.end_date.seconds > 0 else None,
                analysis_types=[AnalysisType.TECHNICAL], # Request only technical analysis for indicators
                additional_parameters={'indicators': list(request.indicators), 'indicator_parameters': dict(request.parameters)}
            )

            # Call the core service logic
            service_response = await self.market_analysis_logic.analyze_market(service_request)

            # Map the result from the service response to the gRPC response
            # The analyze_market method returns indicators within the 'indicators' key of the result
            if service_response.analysis_results:
                for result in service_response.analysis_results:
                    if 'indicators' in result:
                        for indicator_name, values in result['indicators'].items():
                            indicator_msg = market_analysis_service_pb2.Indicator(
                                name=indicator_name,
                                values={int(datetime.fromisoformat(ts).timestamp()): val for ts, val in values.items()} # Assuming values are dict of isoformat timestamp to value
                            )
                            response.indicators.append(indicator_msg)

            # Handle potential errors returned by the service logic
            if service_response.error:
                 response.error.CopyFrom(error_types_pb2.ErrorResponse(
                     code=error_types_pb2.ErrorCode.INTERNAL_ERROR, # Map service error to gRPC error code
                     message=service_response.error.message,
                     details=service_response.error.details
                 ))
                 context.set_code(grpc.StatusCode.INTERNAL)
                 context.set_details(service_response.error.message)

            self.logger.info(f"CalculateIndicators response sent for symbol: {request.symbol.name}")
            return response

        except Exception as e:
            self.logger.error(f"Error in CalculateIndicators: {e}", exc_info=True)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f'Internal server error: {str(e)}')
            # Return an error response message
            response.error.CopyFrom(error_types_pb2.ErrorResponse(
                code=error_types_pb2.ErrorCode.INTERNAL_ERROR,
                message=f'Internal server error: {str(e)}'
            ))
            return response
        finally:
            # Record request duration and update status counter
            grpc_request_duration_seconds.labels(method=method_name).observe(time.time() - start_time)
            grpc_requests_total.labels(method=method_name, status=status).inc()

    @trace_function
    async def DetectPatterns(self, request: market_analysis_service_pb2.DetectPatternsRequest, context: grpc.aio.ServicerContext) -> market_analysis_service_pb2.DetectPatternsResponse:
        """
        Handles the DetectPatterns gRPC call.
        """\
        method_name = "DetectPatterns"
        self.logger.info(f"{method_name} request received for symbol: {request.symbol.name}")
        response = market_analysis_service_pb2.DetectPatternsResponse()
        status = "UNKNOWN"
        start_time = time.time()
        try:
            # Increment total requests counter
            grpc_requests_total.labels(method=method_name, status="IN_PROGRESS").inc()
            service_request = MarketAnalysisRequest(
                symbol=request.symbol.name,
                timeframe=request.timeframe.name,
                start_date=datetime.fromtimestamp(request.start_date.seconds).isoformat(),
                end_date=datetime.fromtimestamp(request.end_date.seconds).isoformat() if request.end_date.seconds > 0 else None,
                analysis_types=[AnalysisType.PATTERN_DETECTION],
                additional_parameters={'patterns': list(request.patterns)}
            )

            service_response = await self.market_analysis_logic.analyze_market(service_request)

            if service_response.analysis_results:
                for result in service_response.analysis_results:
                    if 'patterns' in result:
                        for pattern_name, occurrences in result['patterns'].items():
                            pattern_msg = market_analysis_service_pb2.Pattern(
                                name=pattern_name,
                                occurrences=[market_analysis_service_pb2.PatternOccurrence(start_timestamp=int(datetime.fromisoformat(o['start_timestamp']).timestamp()), end_timestamp=int(datetime.fromisoformat(o['end_timestamp']).timestamp()), confidence=o.get('confidence', 0.0)) for o in occurrences]
                            )
                            response.patterns.append(pattern_msg)

            if service_response.error:
                 response.error.CopyFrom(error_types_pb2.ErrorResponse(
                     code=error_types_pb2.ErrorCode.INTERNAL_ERROR,
                     message=service_response.error.message,
                     details=service_response.error.details
                 ))
                 context.set_code(grpc.StatusCode.INTERNAL)
                 context.set_details(service_response.error.message)

            status = "OK"
            self.logger.info(f"{method_name} response sent for symbol: {request.symbol.name}")
            return response

        except Exception as e:
            status = "ERROR"
            self.logger.error(f"Error in DetectPatterns: {e}", exc_info=True)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f'Internal server error: {str(e)}')
            response.error.CopyFrom(error_types_pb2.ErrorResponse(
                code=error_types_pb2.ErrorCode.INTERNAL_ERROR,
                message=f'Internal server error: {str(e)}'
            ))
            return response
        finally:
            # Record request duration and update status counter
            grpc_request_duration_seconds.labels(method=method_name).observe(time.time() - start_time)
            grpc_requests_total.labels(method=method_name, status=status).inc()

    @trace_function
     async def DetectSupportResistance(self, request: market_analysis_service_pb2.DetectSupportResistanceRequest, context: grpc.aio.ServicerContext) -> market_analysis_service_pb2.DetectSupportResistanceResponse:
         """
         Handles the DetectSupportResistance gRPC call.
         """\
         method_name = "DetectSupportResistance"
         self.logger.info(f"{method_name} request received for symbol: {request.symbol.name}")
         response = market_analysis_service_pb2.DetectSupportResistanceResponse()
         status = "UNKNOWN"
         start_time = time.time()
         try:
             # Increment total requests counter
             grpc_requests_total.labels(method=method_name, status="IN_PROGRESS").inc()
            service_request = MarketAnalysisRequest(
                symbol=request.symbol.name,
                timeframe=request.timeframe.name,
                start_date=datetime.fromtimestamp(request.start_date.seconds).isoformat(),
                end_date=datetime.fromtimestamp(request.end_date.seconds).isoformat() if request.end_date.seconds > 0 else None,
                analysis_types=[AnalysisType.SUPPORT_RESISTANCE],
                additional_parameters={}
            )

            service_response = await self.market_analysis_logic.analyze_market(service_request)

            if service_response.analysis_results:
                for result in service_response.analysis_results:
                    if 'support_resistance' in result:
                        sr_data = result['support_resistance']
                        for level_type, levels in sr_data.items():
                            if level_type == 'support':
                                response.support_levels.extend([market_analysis_service_pb2.Level(value=l['value'], type=l.get('type', common_types_pb2.LevelType.UNKNOWN_LEVEL_TYPE), strength=l.get('strength', 0.0)) for l in levels])
                            elif level_type == 'resistance':
                                response.resistance_levels.extend([market_analysis_service_pb2.Level(value=l['value'], type=l.get('type', common_types_pb2.LevelType.UNKNOWN_LEVEL_TYPE), strength=l.get('strength', 0.0)) for l in levels])

            if service_response.error:
                 response.error.CopyFrom(error_types_pb2.ErrorResponse(
                     code=error_types_pb2.ErrorCode.INTERNAL_ERROR,
                     message=service_response.error.message,
                     details=service_response.error.details
                 ))
                 context.set_code(grpc.StatusCode.INTERNAL)
                 context.set_details(service_response.error.message)

            status = "OK"
             self.logger.info(f"{method_name} response sent for symbol: {request.symbol.name}")
             return response
 
         except Exception as e:
             status = "ERROR"
            self.logger.error(f"Error in DetectSupportResistance: {e}", exc_info=True)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f'Internal server error: {str(e)}')
            response.error.CopyFrom(error_types_pb2.ErrorResponse(
                code=error_types_pb2.ErrorCode.INTERNAL_ERROR,
                message=f'Internal server error: {str(e)}'
            ))
            return response
        finally:
            # Record request duration and update status counter
            grpc_request_duration_seconds.labels(method=method_name).observe(time.time() - start_time)
            grpc_requests_total.labels(method=method_name, status=status).inc()

    @trace_function
    async def DetectMarketRegime(self, request: market_analysis_service_pb2.DetectMarketRegimeRequest, context: grpc.aio.ServicerContext) -> market_analysis_service_pb2.DetectMarketRegimeResponse:
        """
        Handles the DetectMarketRegime gRPC call.
        """\
        method_name = "DetectMarketRegime"
        self.logger.info(f"{method_name} request received for symbol: {request.symbol.name}")
        response = market_analysis_service_pb2.DetectMarketRegimeResponse()
        status = "UNKNOWN"
        start_time = time.time()
        try:
            # Increment total requests counter
            grpc_requests_total.labels(method=method_name, status="IN_PROGRESS").inc()
            service_request = MarketAnalysisRequest(
                symbol=request.symbol.name,
                timeframe=request.timeframe.name,
                start_date=datetime.fromtimestamp(request.start_date.seconds).isoformat(),
                end_date=datetime.fromtimestamp(request.end_date.seconds).isoformat() if request.end_date.seconds > 0 else None,
                analysis_types=[AnalysisType.MARKET_REGIME],
                additional_parameters={}
            )

            service_response = await self.market_analysis_logic.analyze_market(service_request)

            if service_response.analysis_results:
                for result in service_response.analysis_results:
                    if 'market_regime' in result:
                        regime_data = result['market_regime']
                        response.current_regime = common_types_pb2.MarketRegime.Value(regime_data.get('current_regime', 'UNKNOWN_REGIME'))
                        response.regime_history.extend([market_analysis_service_pb2.RegimePeriod(regime=common_types_pb2.MarketRegime.Value(p['regime']), start_timestamp=int(datetime.fromisoformat(p['start_timestamp']).timestamp()), end_timestamp=int(datetime.fromisoformat(p['end_timestamp']).timestamp()) if p.get('end_timestamp') else 0) for p in regime_data.get('regime_history', [])])

            if service_response.error:
                 response.error.CopyFrom(error_types_pb2.ErrorResponse(
                     code=error_types_pb2.ErrorCode.INTERNAL_ERROR,
                     message=service_response.error.message,
                     details=service_response.error.details
                 ))
                 context.set_code(grpc.StatusCode.INTERNAL)
                 context.set_details(service_response.error.message)

            status = "OK"
            self.logger.info(f"{method_name} response sent for symbol: {request.symbol.name}")
            return response

        except Exception as e:
            status = "ERROR"
            self.logger.error(f"Error in DetectMarketRegime: {e}", exc_info=True)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f'Internal server error: {str(e)}')
            response.error.CopyFrom(error_types_pb2.ErrorResponse(
                code=error_types_pb2.ErrorCode.INTERNAL_ERROR,
                message=f'Internal server error: {str(e)}'
            ))
            return response
        finally:
            # Record request duration and update status counter
            grpc_request_duration_seconds.labels(method=method_name).observe(time.time() - start_time)
            grpc_requests_total.labels(method=method_name, status=status).inc()

    @trace_function
    async def PerformCorrelationAnalysis(self, request: market_analysis_service_pb2.PerformCorrelationAnalysisRequest, context: grpc.aio.ServicerContext) -> market_analysis_service_pb2.PerformCorrelationAnalysisResponse:
        """
        Handles the PerformCorrelationAnalysis gRPC call.
        """\
        method_name = "PerformCorrelationAnalysis"
        self.logger.info(f"{method_name} request received for symbols: {', '.join([s.name for s in request.symbols])}")
        response = market_analysis_service_pb2.PerformCorrelationAnalysisResponse()
        status = "UNKNOWN"
        start_time = time.time()
        try:
            # Increment total requests counter
            grpc_requests_total.labels(method=method_name, status="IN_PROGRESS").inc()
            # Correlation analysis typically involves multiple symbols, but the core service analyze_market is designed for a single symbol.
            # A proper implementation might require a different method in the core service or iterating calls.
            # For now, let's assume analyze_market can handle multiple symbols if passed in additional_parameters or a dedicated method exists.
            # Assuming analyze_market can take a list of symbols and AnalysisType.CORRELATION
            service_request = MarketAnalysisRequest(
                symbol=request.symbols[0].name if request.symbols else '', # Assuming the first symbol is the primary, or need a different approach
                timeframe=request.timeframe.name,
                start_date=datetime.fromtimestamp(request.start_date.seconds).isoformat(),
                end_date=datetime.fromtimestamp(request.end_date.seconds).isoformat() if request.end_date.seconds > 0 else None,
                analysis_types=[AnalysisType.CORRELATION],
                additional_parameters={'symbols': [s.name for s in request.symbols]}
            )

            service_response = await self.market_analysis_logic.analyze_market(service_request)

            if service_response.analysis_results:
                 for result in service_response.analysis_results:
                    if 'correlations' in result:
                        for symbol_pair, correlation_value in result['correlations'].items():
                            response.correlations[symbol_pair] = correlation_value

            if service_response.error:
                 response.error.CopyFrom(error_types_pb2.ErrorResponse(
                     code=error_types_pb2.ErrorCode.INTERNAL_ERROR,
                     message=service_response.error.message,
                     details=service_response.error.details
                 ))
                 context.set_code(grpc.StatusCode.INTERNAL)
                 context.set_details(service_response.error.message)

            status = "OK"
            self.logger.info(f"{method_name} response sent for symbols: {', '.join([s.name for s in request.symbols])}")
            return response

        except Exception as e:
            status = "ERROR"
            self.logger.error(f"Error in PerformCorrelationAnalysis: {e}", exc_info=True)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f'Internal server error: {str(e)}')
            response.error.CopyFrom(error_types_pb2.ErrorResponse(
                code=error_types_pb2.ErrorCode.INTERNAL_ERROR,
                message=f'Internal server error: {str(e)}'
            ))
            return response
        finally:
            # Record request duration and update status counter
            grpc_request_duration_seconds.labels(method=method_name).observe(time.time() - start_time)
            grpc_requests_total.labels(method=method_name, status=status).inc()

    @trace_function
    async def PerformVolatilityAnalysis(self, request: market_analysis_service_pb2.PerformVolatilityAnalysisRequest, context: grpc.aio.ServicerContext) -> market_analysis_service_pb2.PerformVolatilityAnalysisResponse:
        """
        Handles the PerformVolatilityAnalysis gRPC call.
        """\
        method_name = "PerformVolatilityAnalysis"
        self.logger.info(f"{method_name} request received for symbol: {request.symbol.name}")
        response = market_analysis_service_pb2.PerformVolatilityAnalysisResponse()
        status = "UNKNOWN"
        start_time = time.time()
        try:
            # Increment total requests counter
            grpc_requests_total.labels(method=method_name, status="IN_PROGRESS").inc()
            service_request = MarketAnalysisRequest(
                symbol=request.symbol.name,
                timeframe=request.timeframe.name,
                start_date=datetime.fromtimestamp(request.start_date.seconds).isoformat(),
                end_date=datetime.fromtimestamp(request.end_date.seconds).isoformat() if request.end_date.seconds > 0 else None,
                analysis_types=[AnalysisType.VOLATILITY],
                additional_parameters={}
            )

            service_response = await self.market_analysis_logic.analyze_market(service_request)

            if service_response.analysis_results:
                for result in service_response.analysis_results:
                    if 'volatility' in result:
                        volatility_data = result['volatility']
                        response.current_volatility = volatility_data.get('current_volatility', 0.0)
                        response.volatility_history.extend([market_analysis_service_pb2.VolatilityPeriod(timestamp=int(datetime.fromisoformat(v['timestamp']).timestamp()), value=v['value']) for v in volatility_data.get('volatility_history', [])])

            if service_response.error:
                 response.error.CopyFrom(error_types_pb2.ErrorResponse(
                     code=error_types_pb2.ErrorCode.INTERNAL_ERROR,
                     message=service_response.error.message,
                     details=service_response.error.details
                 ))
                 context.set_code(grpc.StatusCode.INTERNAL)
                 context.set_details(service_response.error.message)

            status = "OK"
            self.logger.info(f"{method_name} response sent for symbol: {request.symbol.name}")
            return response

        except Exception as e:
            status = "ERROR"
            self.logger.error(f"Error in PerformVolatilityAnalysis: {e}", exc_info=True)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f'Internal server error: {str(e)}')
            response.error.CopyFrom(error_types_pb2.ErrorResponse(
                code=error_types_pb2.ErrorCode.INTERNAL_ERROR,
                message=f'Internal server error: {str(e)}'
            ))
            return response
        finally:
            # Record request duration and update status counter
            grpc_request_duration_seconds.labels(method=method_name).observe(time.time() - start_time)
            grpc_requests_total.labels(method=method_name, status=status).inc()


async def serve():
    """
    Starts the gRPC server for the Market Analysis Service.
    """
    server = grpc.aio.server(futures.ThreadPoolExecutor(max_workers=10)) # TODO: Configure max_workers appropriately

    # Initialize monitoring and tracing managers
    metrics_manager = MetricsManager(service_name="market-analysis-service") # TODO: Get service name from config
    tracing_manager = TracingManager(service_name="market-analysis-service") # TODO: Get service name from config

    # Define gRPC specific metrics
    grpc_requests_total = metrics_manager.create_counter(
        name="grpc_requests_total",
        description="Total number of gRPC requests",
        labels=["method", "status"]
    )
    grpc_request_duration_seconds = metrics_manager.create_histogram(
        name="grpc_request_duration_seconds",
        description="gRPC request duration in seconds",
        labels=["method"],
        buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5]
    )

    # Get the actual core MarketAnalysisService instance using dependency injection
    core_service_instance = get_market_analysis_service()

    # Initialize the servicer with the core service instance
    servicer = MarketAnalysisServiceServicer(market_analysis_logic=core_service_instance)

    market_analysis_service_pb2_grpc.add_MarketAnalysisServiceServicer_to_server(
        servicer, server)

    # TODO: Configure the server port and security
    listen_addr = '[::]:50051' # Example port
    server.add_insecure_port(listen_addr)

    logger.info(f"Starting Market Analysis gRPC server on {listen_addr}")
    await server.start()
    await server.wait_for_termination() # Keep the server running

    # TODO: Implement proper shutdown handling

if __name__ == '__main__':
    # Set up standardized logging
    setup_logging(service_name="market-analysis-service", log_level="INFO") # TODO: Get service name and log level from config
    logger = get_logger("market-analysis-service") # Get logger after setup
    asyncio.run(serve())