import grpc
import asyncio
from concurrent import futures
import logging

from generated_protos import analysis_engine_pb2, analysis_engine_pb2_grpc
from google.protobuf.timestamp_pb2 import Timestamp

# For gRPC Health Checking Protocol
from grpc_health.v1 import health
from grpc_health.v1 import health_pb2
from grpc_health.v1 import health_pb2_grpc

from analysis_engine.core.container import ServiceContainer # Import ServiceContainer
from analysis_engine.config import get_settings # Ensure get_settings is imported

logger = logging.getLogger(__name__)

class AnalysisEngineServicer(analysis_engine_pb2_grpc.AnalysisEngineServicer):
    """
    gRPC servicer for the Analysis Engine.
    """
    def __init__(self, service_container: ServiceContainer):
        self.service_container = service_container
        logger.info("AnalysisEngineServicer initialized with ServiceContainer.")

    async def GetMarketOverview(self, request: analysis_engine_pb2.GetMarketOverviewRequest, context):
        logger.info(f"GetMarketOverview called for symbol: {request.symbol}, timeframe: {request.timeframe}, lookback: {request.lookback_days} days")
        response = analysis_engine_pb2.MarketOverviewResponse()

        try:
            # Retrieve dependencies from the service container
            # Assuming ServiceContainer provides access to these adapter instances
            # This might need adjustment based on actual ServiceContainer implementation
            # For now, we'll assume it mirrors the structure of ServiceDependencies from service_dependencies.py
            
            # TODO: Confirm how to get specific adapters from self.service_container.
            # For now, assuming direct access or via a getter method on service_container
            # that uses the common_adapter_factory, similar to service_dependencies.py
            # If ServiceContainer IS the ServiceDependencies instance, then this is direct.
            # If ServiceContainer WRAPS ServiceDependencies, it might be self.service_container.dependencies.get_trading_gateway()
            
            # Using service_dependencies singleton for now, as ServiceContainer structure is not fully clear from previous steps.
            # Ideally, this should come from self.service_container passed in __init__
            from analysis_engine.core.service_dependencies import service_dependencies # Direct import for now
            
            trading_gateway = await service_dependencies.get_trading_gateway()
            feature_provider = await service_dependencies.get_feature_provider()
            ml_model_registry = await service_dependencies.get_ml_model_registry()
            # risk_manager = await service_dependencies.get_risk_manager() # get_risk_manager is defined

            # Logic adapted from api/v1/integrated_analysis.py:get_market_overview
            from datetime import datetime, timedelta, timezone

            end_time = datetime.now(timezone.utc)
            start_time = end_time - timedelta(days=request.lookback_days)

            # 1. Market Data from Trading Gateway
            # The interface ITradingGateway expects get_market_data(symbol, timeframe, start_time, end_time)
            # The response from adapter is likely a Pydantic model or dict, not raw data.
            # The REST endpoint's market_data has .data attribute. Assuming similar structure.
            raw_market_data = await trading_gateway.get_market_data(
                symbol=request.symbol,
                timeframe=request.timeframe,
                start_time=start_time,
                end_time=end_time
            )
            
            if raw_market_data and raw_market_data.data: # Assuming .data holds list of candles
                latest_candle = raw_market_data.data[-1] # Assuming list of dicts/objects
                response.market_data.open = latest_candle.get("open", 0.0)
                response.market_data.high = latest_candle.get("high", 0.0)
                response.market_data.low = latest_candle.get("low", 0.0)
                response.market_data.close = latest_candle.get("close", 0.0)
                response.market_data.volume = int(latest_candle.get("volume", 0)) # Ensure int
                
                # Timestamp conversion: Assuming latest_candle['timestamp'] is a datetime object or float/int seconds
                candle_timestamp_val = latest_candle.get("timestamp")
                if isinstance(candle_timestamp_val, datetime):
                    response.market_data.timestamp.FromDatetime(candle_timestamp_val)
                elif isinstance(candle_timestamp_val, (int, float)):
                     response.market_data.timestamp.FromSeconds(int(candle_timestamp_val))
                else: # Fallback to current time if no valid timestamp
                    response.market_data.timestamp.GetCurrentTime()

            # 2. Feature Data (Technical Indicators) from Feature Store
            # The REST endpoint gets a list of available features and then fetches data for a few.
            # For gRPC, let's assume the request might specify indicators or we fetch defaults.
            # For now, replicating the "fetch a few available ones" logic.
            try:
                available_features = await feature_provider.get_available_features(
                    symbol=request.symbol, 
                    timeframe=request.timeframe
                ) # Adjusted to new interface
                
                # Fetch data for a limited number of features
                # The feature_provider.get_feature_data returns a FeatureData object.
                # The REST API converts it to dict. Here we need the latest value.
                # This part is highly dependent on the actual structure of FeatureData
                # and how to get a singular, latest value for an indicator.
                # The dummy data in grpc_server.py was: response.technical_indicators["SMA_50"] = 1.12000
                
                # Simplified: If request.indicator_names (from GetIndicatorsRequest) was part of GetMarketOverviewRequest,
                # we'd use that. Otherwise, using a fixed list or first few available.
                # This placeholder doesn't fetch actual values yet.
                # TODO: Replace with actual feature fetching and value extraction
                if available_features:
                    for feature_name in available_features[:2]: # Example: first two
                         # Assume get_latest_feature_value is a hypothetical method or needs complex logic
                         # based on get_feature_data's full timeseries output.
                         # For now, using placeholder values.
                        response.technical_indicators[feature_name.name] = 0.0 # Placeholder
                response.technical_indicators["SMA_50_placeholder"] = 1.12000 
                response.technical_indicators["RSI_14_placeholder"] = 55.5

            except Exception as e:
                logger.warning(f"Error fetching feature data for {request.symbol}: {e}", exc_info=True)


            # 3. Model Predictions from ML Model Registry
            # The REST endpoint lists models and picks a few.
            # This also needs adaptation for how ML predictions are actually retrieved and structured.
            # Assuming list_models gives models applicable to the symbol/timeframe or general models.
            try:
                models = await ml_model_registry.list_models(symbol=request.symbol, timeframe=request.timeframe)
                # The REST API iterates and gets model_info, then makes up a prediction.
                # TODO: Replace with actual prediction logic if possible, or structured placeholders.
                if models:
                    for model_meta in models[:1]: # Example: first model
                        # prediction_data = await ml_model_registry.get_prediction(model_meta.id, request.symbol, request.timeframe)
                        # This is a guess; actual method depends on IMLModelRegistry interface for predictions.
                        response.model_predictions[model_meta.name] = "UP_placeholder" # Placeholder
                response.model_predictions["trend_placeholder"] = "UP"
                response.model_predictions["confidence_placeholder"] = "0.75"

            except Exception as e:
                logger.warning(f"Error fetching model predictions for {request.symbol}: {e}", exc_info=True)

            # 4. Risk Assessment (Simplified, as RiskManagerDep was not in REST example directly, but is in deps)
            # The REST example for market_overview didn't directly use risk_manager, but get_trading_opportunity did.
            # The gRPC MarketOverviewResponse has a RiskAssessment field.
            # Let's add a basic placeholder for it.
            # risk_manager = await service_dependencies.get_risk_manager() # If needed
            # For now, just placeholder values as in the original gRPC dummy data.
            response.risk_assessment.volatility = 0.005
            response.risk_assessment.var = 0.01 # Value at Risk
            response.risk_assessment.sentiment = "NEUTRAL_placeholder"

            return response

        except grpc.aio.AioRpcError as e: # Catch errors from dependent gRPC services if any
            logger.error(f"gRPC error during GetMarketOverview for {request.symbol}: {e.details()}", exc_info=True)
            await context.abort(e.code(), f"Upstream gRPC error: {e.details()}")
        except Exception as e:
            logger.error(f"Failed to get market overview for {request.symbol}: {e}", exc_info=True)
            await context.abort(grpc.StatusCode.INTERNAL, f"Internal server error: {str(e)}")

    async def GetTradingOpportunity(self, request: analysis_engine_pb2.GetTradingOpportunityRequest, context):
        logger.info(f"GetTradingOpportunity called for symbol: {request.symbol}, account: {request.account_id}, risk: {request.risk_percentage}%")
        response = analysis_engine_pb2.TradingOpportunityResponse()

        try:
            from analysis_engine.core.service_dependencies import service_dependencies # Direct import for now
            from datetime import datetime, timedelta, timezone
            from common_lib.models.trading import OrderSide as CommonOrderSide # For mapping

            trading_gateway = await service_dependencies.get_trading_gateway()
            ml_model_registry = await service_dependencies.get_ml_model_registry()
            risk_manager = await service_dependencies.get_risk_manager()
            # feature_provider = await service_dependencies.get_feature_provider() # Not used in REST version

            # 1. Account Info from Trading Gateway
            account_info = await trading_gateway.get_account_info(request.account_id)
            response.account_balance = account_info.balance

            # 2. Market Data from Trading Gateway
            end_time = datetime.now(timezone.utc)
            start_time = end_time - timedelta(days=7) # Fixed lookback as in REST
            
            market_data_result = await trading_gateway.get_market_data(
                symbol=request.symbol,
                timeframe=request.timeframe,
                start_time=start_time,
                end_time=end_time
            )
            current_price = 0.0
            if market_data_result and market_data_result.data:
                current_price = market_data_result.data[-1].get("close", 0.0)
            
            if current_price == 0.0:
                await context.abort(grpc.StatusCode.FAILED_PRECONDITION, "Could not retrieve current price for symbol.")
                return

            # 3. Calculate Entry and Exit Points (simplified example from REST)
            # TODO: This logic should ideally be more sophisticated, perhaps from another service or library
            response.entry_price = current_price
            response.risk_details.stop_loss_price = current_price * 0.98  # 2% below
            response.risk_details.take_profit_price = current_price * 1.04 # 4% above
            
            # 4. Calculate Position Size from Risk Management
            position_size_result_dict = await risk_manager.calculate_position_size(
                symbol=request.symbol,
                account_id=request.account_id,
                risk_percentage=request.risk_percentage,
                entry_price=response.entry_price,
                stop_loss_price=response.risk_details.stop_loss_price
            )
            # The result from adapter is a dict: {"position_size": float, "risk_amount": float, "reward_amount": float, "risk_reward_ratio": float}
            response.position_size = position_size_result_dict.get("position_size", 0.0)
            response.risk_details.risk_reward_ratio = position_size_result_dict.get("risk_reward_ratio", 0.0)
            # Note: TradingOpportunityResponse doesn't have risk_amount, reward_amount directly.

            # 5. Model Predictions from ML Model Registry
            models = await ml_model_registry.list_models(symbol=request.symbol, timeframe=request.timeframe)
            model_predictions_map = {}
            bullish_count = 0
            bearish_count = 0
            if models:
                for model_meta in models[:3]: # Limit to 3 models as in REST
                    try:
                        # Placeholder logic from REST - replace with actual prediction call if available
                        # prediction = await ml_model_registry.get_prediction(model_meta.id, request.symbol, request.timeframe) 
                        # model_predictions_map[model_meta.name] = prediction.value 
                        # For now, replicating REST's placeholder:
                        is_bullish = model_meta.id.endswith("1") # Dummy logic from REST
                        model_predictions_map[model_meta.name] = "bullish_placeholder" if is_bullish else "bearish_placeholder"
                        if is_bullish: bullish_count +=1
                        else: bearish_count += 1
                    except Exception as e_model:
                        logger.warning(f"Error getting prediction for model {model_meta.id}: {e_model}", exc_info=True)
            response.model_predictions.update(model_predictions_map)

            # 6. Determine Trade Direction
            common_trade_direction = CommonOrderSide.BUY if bullish_count > bearish_count else CommonOrderSide.SELL
            if common_trade_direction == CommonOrderSide.BUY:
                response.trade_direction = analysis_engine_pb2.OrderSide.BUY
            else:
                response.trade_direction = analysis_engine_pb2.OrderSide.SELL
            
            # Target price based on take_profit_price (can be refined)
            response.target_price = response.risk_details.take_profit_price

            return response

        except grpc.aio.AioRpcError as e: # Catch errors from dependent gRPC services if any
            logger.error(f"gRPC error during GetTradingOpportunity for {request.symbol}: {e.details()}", exc_info=True)
            await context.abort(e.code(), f"Upstream gRPC error: {e.details()}")
        except Exception as e:
            logger.error(f"Failed to get trading opportunity for {request.symbol}: {e}", exc_info=True)
            await context.abort(grpc.StatusCode.INTERNAL, f"Internal server error: {str(e)}")

    # Custom HealthCheck method is removed as per Step 1 & 2.
    # Standard gRPC Health Checking Protocol will be used.

    async def GetIndicators(self, request, context):
        logger.info(f"GetIndicators called for symbol: {request.symbol}")
        # Placeholder
        response = analysis_engine_pb2.GetIndicatorsResponse()
        response.indicators["MACD_12_26_9"] = 0.0005
        response.indicators["CCI_20"] = 110.0
        return response

    async def GetPatterns(self, request, context):
        logger.info(f"GetPatterns called for symbol: {request.symbol}")
        # Placeholder
        response = analysis_engine_pb2.GetPatternsResponse()
        response.patterns["HeadAndShoulders"] = False
        response.patterns["DoubleTop"] = True
        return response

    async def PerformAnalysis(self, request, context):
        logger.info(f"PerformAnalysis called for symbol: {request.symbol} with type: {request.analysis_type}")
        # Placeholder
        response = analysis_engine_pb2.PerformAnalysisResponse()
        response.summary = f"Analysis for {request.symbol} ({request.analysis_type}) completed."
        response.details["key_finding_1"] = "Market is currently range-bound."
        response.details["recommendation"] = "Wait for breakout."
        return response

async def serve():
    """
    Starts the gRPC server.
    Args:
        service_container (ServiceContainer): The application's service container.
    """
    server = grpc.aio.server(futures.ThreadPoolExecutor(max_workers=10))
    
    # Add AnalysisEngineServicer, passing the service_container
    analysis_engine_pb2_grpc.add_AnalysisEngineServicer_to_server(
        AnalysisEngineServicer(service_container=service_container), server
    )

    # Setup gRPC Health Checking Protocol (assuming this was correctly added from previous subtask)
    health_servicer = health.HealthServicer()
    health_pb2_grpc.add_HealthServicer_to_server(health_servicer, server)
    service_name = analysis_engine_pb2.DESCRIPTOR.services_by_name['AnalysisEngine'].full_name
    health_servicer.set(service_name, health_pb2.HealthCheckResponse.SERVING)
    health_servicer.set("", health_pb2.HealthCheckResponse.SERVING) # For overall server health

    settings = get_settings() # Use imported get_settings
    grpc_port = settings.GRPC_PORT
    server.add_insecure_port(f'[::]:{grpc_port}')
    
    logger.info(f"Starting gRPC server on port {grpc_port}...")
    await server.start()
    logger.info(f"gRPC server started successfully on port {grpc_port}.")
    
    try:
        await server.wait_for_termination()
    except KeyboardInterrupt:
        logger.info("gRPC server shutting down due to KeyboardInterrupt...")
        await server.stop(0) # Graceful stop
        logger.info("gRPC server stopped.")
    except asyncio.CancelledError:
        logger.info("gRPC server task cancelled, shutting down...")
        await server.stop(0)
        logger.info("gRPC server stopped.")


if __name__ == '__main__':
    # This is for standalone testing of the gRPC server,
    # requires a mock or real ServiceContainer.
    # For actual service running, core.main.py is the entry point.
    logging.basicConfig(level=logging.INFO)
    
    async def main_standalone():
        logger.warning("grpc_server.py is not intended to be run standalone without a proper ServiceContainer.")
        logger.warning("Run via core/main.py for integrated FastAPI and gRPC server startup.")
        # Example for testing (would require mocking or full setup):
        # settings_instance = get_settings() 
        # test_container = ServiceContainer(settings=settings_instance)
        # await test_container.initialize_dependencies() 
        # await serve(service_container=test_container)
        # await test_container.shutdown_dependencies()
        pass

    asyncio.run(main_standalone())
