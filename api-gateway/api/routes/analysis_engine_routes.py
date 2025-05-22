import logging
from fastapi import APIRouter, Depends, HTTPException, status, Path, Query, Body
from typing import Optional

# Assuming the gRPC client and generated protos are importable
# Adjust path if necessary based on project structure
try:
    from ...adapters.analysis_engine_grpc_client import AnalysisEngineGrpcClient
    from ...generated_protos import analysis_engine_pb2
    from ...config.config import get_service_config # To get grpc_url
    from ...config.config_schema import ServiceConfig
except ImportError:
    logging.error("API Gateway: Could not import necessary modules for analysis_engine_routes.py. Ensure paths are correct and stubs generated.")
    # Dummy classes for initial loading if imports fail
    class AnalysisEngineGrpcClient: pass
    class analysis_engine_pb2:
        class GetMarketOverviewRequest: pass
        class MarketOverviewResponse: pass
        class GetTradingOpportunityRequest: pass
        class TradingOpportunityResponse: pass
        class OrderSide: pass # Enum
    class ServiceConfig: analysis_engine_grpc_url: Optional[str] = ""
    def get_service_config(): return ServiceConfig()


logger = logging.getLogger(__name__)
router = APIRouter(prefix="/analysis-engine-service", tags=["Analysis Engine gRPC"])

# Dependency to get gRPC client
async def get_grpc_client(service_config: ServiceConfig = Depends(get_service_config)):
    if not service_config.analysis_engine_grpc_url:
        logger.error("API Gateway: Analysis Engine gRPC URL is not configured.")
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Analysis Engine Service is not configured.")
    
    client = AnalysisEngineGrpcClient(grpc_server_address=service_config.analysis_engine_grpc_url)
    try:
        yield client
    finally:
        await client.close()

@router.get("/integrated-analysis/market-overview/{symbol}", 
            response_model=Optional[dict]) # Using dict for flexibility, ideally map to Pydantic model
async def get_market_overview_grpc(
    symbol: str = Path(..., description="Trading symbol, e.g., EURUSD"),
    timeframe: str = Query("H1", description="Timeframe for analysis, e.g., M15, H1, D1"),
    lookback_days: int = Query(30, description="Number of days for lookback period for market data"),
    client: AnalysisEngineGrpcClient = Depends(get_grpc_client)
):
    logger.info(f"API Gateway: Received REST request for market overview (gRPC) for {symbol}")
    
    grpc_response = await client.get_market_overview(
        symbol=symbol,
        timeframe=timeframe,
        lookback_days=lookback_days
    )

    if grpc_response is None:
        logger.error(f"API Gateway: No response from Analysis Engine gRPC for market overview of {symbol}")
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Analysis Engine Service did not return a market overview.")

    # Convert gRPC response to dict for FastAPI JSON response
    # This can be more sophisticated with Pydantic models mirroring gRPC messages
    market_data_dict = {
        "open": grpc_response.market_data.open,
        "high": grpc_response.market_data.high,
        "low": grpc_response.market_data.low,
        "close": grpc_response.market_data.close,
        "volume": grpc_response.market_data.volume,
        "timestamp": grpc_response.market_data.timestamp.ToSeconds() if grpc_response.market_data.HasField("timestamp") else None,
    }
    risk_assessment_dict = {
        "volatility": grpc_response.risk_assessment.volatility,
        "var": grpc_response.risk_assessment.var,
        "sentiment": grpc_response.risk_assessment.sentiment,
    }
    
    response_dict = {
        "market_data": market_data_dict,
        "technical_indicators": dict(grpc_response.technical_indicators),
        "model_predictions": dict(grpc_response.model_predictions),
        "risk_assessment": risk_assessment_dict,
    }
    return response_dict

@router.post("/integrated-analysis/trading-opportunity", 
             response_model=Optional[dict]) # Using dict for flexibility
async def get_trading_opportunity_grpc(
    # Assuming these parameters would come in the body of a POST request for a RESTful approach
    symbol: str = Body(..., embed=True, description="Trading symbol, e.g., EURUSD"),
    timeframe: str = Body(..., embed=True, description="Timeframe for analysis, e.g., M15, H1, D1"),
    account_id: str = Body(..., embed=True, description="Account ID for context"),
    risk_percentage: float = Body(..., embed=True, gt=0, lt=100, description="Risk percentage for position sizing"),
    client: AnalysisEngineGrpcClient = Depends(get_grpc_client)
):
    logger.info(f"API Gateway: Received REST request for trading opportunity (gRPC) for {symbol}")

    grpc_response = await client.get_trading_opportunity(
        symbol=symbol,
        timeframe=timeframe,
        account_id=account_id,
        risk_percentage=risk_percentage
    )

    if grpc_response is None:
        logger.error(f"API Gateway: No response from Analysis Engine gRPC for trading opportunity of {symbol}")
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Analysis Engine Service did not return a trading opportunity.")

    # Convert gRPC response to dict
    risk_details_dict = {
        "stop_loss_price": grpc_response.risk_details.stop_loss_price,
        "take_profit_price": grpc_response.risk_details.take_profit_price,
        "risk_reward_ratio": grpc_response.risk_details.risk_reward_ratio,
    }
    
    # Mapping enum OrderSide to string
    order_side_str = analysis_engine_pb2.OrderSide.Name(grpc_response.trade_direction)

    response_dict = {
        "account_balance": grpc_response.account_balance,
        "trade_direction": order_side_str,
        "entry_price": grpc_response.entry_price,
        "target_price": grpc_response.target_price,
        "position_size": grpc_response.position_size,
        "risk_details": risk_details_dict,
        "model_predictions": dict(grpc_response.model_predictions),
    }
    return response_dict

@router.get("/health/analysis-engine", tags=["Health Checks"])
async def health_check_analysis_engine_grpc(
    client: AnalysisEngineGrpcClient = Depends(get_grpc_client)
):
    logger.info("API Gateway: Performing health check on Analysis Engine via gRPC.")
    grpc_response = await client.health_check_analysis_engine()
    if grpc_response and grpc_response.status == analysis_engine_pb2.HealthCheckResponse.SERVING:
        return {"status": "SERVING"}
    else:
        logger.error("API Gateway: Analysis Engine gRPC health check failed or service not serving.")
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Analysis Engine Service is not healthy.")
