"""
Integrated Analysis API.

This module provides API endpoints for integrated analysis that combines data and functionality
from multiple services. It demonstrates how to use the common interfaces and adapters to interact
with other services without direct dependencies.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

from fastapi import APIRouter, Query, Path, HTTPException, status

from common_lib.models.trading import OrderType, OrderSide

from analysis_engine.core.service_dependencies import (
    TradingGatewayDep,
    FeatureProviderDep,
    MLModelRegistryDep,
    RiskManagerDep
)


logger = logging.getLogger(__name__)
router = APIRouter(prefix="/integrated-analysis", tags=["Integrated Analysis"])


@router.get("/market-overview/{symbol}")
async def get_market_overview(
    symbol: str = Path(..., description="Trading symbol"),
    timeframe: str = Query("1h", description="Timeframe for analysis"),
    lookback_days: int = Query(7, description="Number of days to look back"),
    trading_gateway: TradingGatewayDep = None,
    feature_provider: FeatureProviderDep = None,
    ml_model_registry: MLModelRegistryDep = None,
    risk_manager: RiskManagerDep = None
):
    """
    Get a comprehensive market overview for a symbol.
    
    This endpoint combines data from multiple services:
    - Market data from Trading Gateway
    - Feature data from Feature Store
    - Model predictions from ML Integration
    - Risk assessment from Risk Management
    
    Args:
        symbol: Trading symbol
        timeframe: Timeframe for analysis
        lookback_days: Number of days to look back
        trading_gateway: Trading Gateway dependency
        feature_provider: Feature Provider dependency
        ml_model_registry: ML Model Registry dependency
        risk_manager: Risk Manager dependency
        
    Returns:
        Comprehensive market overview
    """
    try:
        # Calculate time range
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(days=lookback_days)
        
        # Get market data from Trading Gateway
        market_data = await trading_gateway.get_market_data(
            symbol=symbol,
            timeframe=timeframe,
            start_time=start_time,
            end_time=end_time
        )
        
        # Get feature data from Feature Store
        features = await feature_provider.get_available_features()
        feature_data = {}
        
        for feature_name in features[:5]:  # Limit to 5 features for performance
            try:
                data = await feature_provider.get_feature_data(
                    feature_name=feature_name,
                    symbol=symbol,
                    timeframe=timeframe,
                    start_time=start_time,
                    end_time=end_time
                )
                feature_data[feature_name] = data.to_dict(orient="records")
            except Exception as e:
                logger.warning(f"Error getting feature data for {feature_name}: {str(e)}")
        
        # Get model predictions from ML Integration
        models = await ml_model_registry.list_models()
        model_predictions = {}
        
        for model in models[:3]:  # Limit to 3 models for performance
            try:
                model_id = model["id"]
                model_info = await ml_model_registry.get_model_info(model_id)
                
                if model_info.get("target_symbol") == symbol:
                    # This is just a placeholder - in a real implementation,
                    # you would get actual predictions from the model
                    model_predictions[model_id] = {
                        "model_name": model_info.get("name", "Unknown"),
                        "prediction": "bullish" if model_id.endswith("1") else "bearish",
                        "confidence": 0.75,
                        "timeframe": timeframe
                    }
            except Exception as e:
                logger.warning(f"Error getting model predictions for {model['id']}: {str(e)}")
        
        # Get risk assessment from Risk Management
        try:
            position_risk = await risk_manager.get_position_risk(symbol)
            risk_assessment = {
                "position_risk": position_risk,
                "risk_level": "medium",
                "max_position_size": position_risk.get("max_position_size", 0.0),
                "recommended_stop_loss": market_data.data[-1]["close"] * 0.98  # 2% below current price
            }
        except Exception as e:
            logger.warning(f"Error getting risk assessment: {str(e)}")
            risk_assessment = {
                "risk_level": "unknown",
                "error": str(e)
            }
        
        # Combine all data into a comprehensive market overview
        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "analysis_time": datetime.utcnow().isoformat(),
            "market_data": {
                "current_price": market_data.data[-1]["close"] if market_data.data else None,
                "price_change_24h": calculate_price_change(market_data.data, 24) if market_data.data else None,
                "price_change_7d": calculate_price_change(market_data.data, 7 * 24) if market_data.data else None,
                "volume_24h": calculate_volume(market_data.data, 24) if market_data.data else None,
                "data_points": len(market_data.data) if market_data.data else 0
            },
            "technical_indicators": feature_data,
            "model_predictions": model_predictions,
            "risk_assessment": risk_assessment
        }
    except Exception as e:
        logger.error(f"Error in get_market_overview: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error generating market overview: {str(e)}"
        )


@router.get("/trading-opportunity/{symbol}")
async def get_trading_opportunity(
    symbol: str = Path(..., description="Trading symbol"),
    timeframe: str = Query("1h", description="Timeframe for analysis"),
    account_id: str = Query(..., description="Trading account ID"),
    risk_percentage: float = Query(1.0, description="Risk percentage (0-100)"),
    trading_gateway: TradingGatewayDep = None,
    feature_provider: FeatureProviderDep = None,
    ml_model_registry: MLModelRegistryDep = None,
    risk_manager: RiskManagerDep = None
):
    """
    Get trading opportunity analysis for a symbol.
    
    This endpoint combines data from multiple services to provide a comprehensive
    trading opportunity analysis, including entry/exit points and position sizing.
    
    Args:
        symbol: Trading symbol
        timeframe: Timeframe for analysis
        account_id: Trading account ID
        risk_percentage: Risk percentage (0-100)
        trading_gateway: Trading Gateway dependency
        feature_provider: Feature Provider dependency
        ml_model_registry: ML Model Registry dependency
        risk_manager: Risk Manager dependency
        
    Returns:
        Trading opportunity analysis
    """
    try:
        # Get account info from Trading Gateway
        account_info = await trading_gateway.get_account_info(account_id)
        
        # Get market data from Trading Gateway
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(days=7)
        
        market_data = await trading_gateway.get_market_data(
            symbol=symbol,
            timeframe=timeframe,
            start_time=start_time,
            end_time=end_time
        )
        
        # Get current price
        current_price = market_data.data[-1]["close"] if market_data.data else 0.0
        
        # Calculate entry and exit points (simplified example)
        entry_price = current_price
        stop_loss_price = current_price * 0.98  # 2% below current price
        take_profit_price = current_price * 1.04  # 4% above current price
        
        # Calculate position size based on risk
        position_size_result = await risk_manager.calculate_position_size(
            symbol=symbol,
            account_id=account_id,
            risk_percentage=risk_percentage,
            entry_price=entry_price,
            stop_loss_price=stop_loss_price
        )
        
        # Get model predictions
        models = await ml_model_registry.list_models()
        model_predictions = {}
        
        for model in models[:3]:  # Limit to 3 models for performance
            try:
                model_id = model["id"]
                model_info = await ml_model_registry.get_model_info(model_id)
                
                if model_info.get("target_symbol") == symbol:
                    # This is just a placeholder - in a real implementation,
                    # you would get actual predictions from the model
                    model_predictions[model_id] = {
                        "model_name": model_info.get("name", "Unknown"),
                        "prediction": "bullish" if model_id.endswith("1") else "bearish",
                        "confidence": 0.75,
                        "timeframe": timeframe
                    }
            except Exception as e:
                logger.warning(f"Error getting model predictions for {model['id']}: {str(e)}")
        
        # Determine trade direction based on model predictions
        bullish_count = sum(1 for p in model_predictions.values() if p["prediction"] == "bullish")
        bearish_count = sum(1 for p in model_predictions.values() if p["prediction"] == "bearish")
        
        trade_direction = OrderSide.BUY if bullish_count > bearish_count else OrderSide.SELL
        
        # Return trading opportunity analysis
        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "analysis_time": datetime.utcnow().isoformat(),
            "account_balance": account_info.balance,
            "trade_direction": trade_direction,
            "entry_price": entry_price,
            "stop_loss_price": stop_loss_price,
            "take_profit_price": take_profit_price,
            "position_size": position_size_result.get("position_size", 0.0),
            "risk_amount": position_size_result.get("risk_amount", 0.0),
            "reward_amount": position_size_result.get("reward_amount", 0.0),
            "risk_reward_ratio": position_size_result.get("risk_reward_ratio", 0.0),
            "model_predictions": model_predictions,
            "confidence_level": "high" if abs(bullish_count - bearish_count) > 1 else "medium"
        }
    except Exception as e:
        logger.error(f"Error in get_trading_opportunity: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error generating trading opportunity: {str(e)}"
        )


def calculate_price_change(data: List[Dict[str, Any]], hours: int) -> float:
    """
    Calculate price change over a period.
    
    Args:
        data: Price data
        hours: Number of hours to look back
        
    Returns:
        Price change percentage
    """
    if not data or len(data) < 2:
        return 0.0
    
    current_price = data[-1]["close"]
    
    # Find the price 'hours' ago
    for i in range(len(data) - 2, -1, -1):
        if data[-1]["timestamp"] - data[i]["timestamp"] >= hours * 3600:
            past_price = data[i]["close"]
            return (current_price - past_price) / past_price * 100
    
    # If we don't have enough data, use the oldest available price
    past_price = data[0]["close"]
    return (current_price - past_price) / past_price * 100


def calculate_volume(data: List[Dict[str, Any]], hours: int) -> float:
    """
    Calculate total volume over a period.
    
    Args:
        data: Price data
        hours: Number of hours to look back
        
    Returns:
        Total volume
    """
    if not data:
        return 0.0
    
    total_volume = 0.0
    current_time = data[-1]["timestamp"]
    
    for item in reversed(data):
        if current_time - item["timestamp"] <= hours * 3600:
            total_volume += item.get("volume", 0.0)
        else:
            break
    
    return total_volume
