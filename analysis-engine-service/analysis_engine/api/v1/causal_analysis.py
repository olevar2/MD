"""
Causal Analysis API Module

Provides API endpoints for causal analysis capabilities, allowing other
services to access causal insights, relationships, and counterfactual scenarios.
"""
import logging
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, HTTPException, Query, Depends
from pydantic import BaseModel

from analysis_engine.causal.integration.causal_integration import CausalInsightGenerator
from analysis_engine.api.v1.auth import get_current_user
from analysis_engine.models.user import User

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/causal", tags=["causal"])

# Models for request/response validation

class CounterfactualRequest(BaseModel):
    base_scenario: Dict[str, Any]
    interventions: List[Dict[str, Any]]
    description: Optional[str] = None

class SignalEnhancementRequest(BaseModel):
    signals: List[Dict[str, Any]]
    market_data: Dict[str, Any]
    config: Optional[Dict[str, Any]] = None

class RegimeDriverRequest(BaseModel):
    market_data: Dict[str, Any]
    regime_column: str
    feature_columns: List[str]
    config: Optional[Dict[str, Any]] = None

class CurrencyPairRequest(BaseModel):
    price_data: Dict[str, Dict[str, Any]]
    max_lag: Optional[int] = 5
    config: Optional[Dict[str, Any]] = None

class CorrelationRiskRequest(BaseModel):
    correlation_data: Dict[str, Any]
    config: Optional[Dict[str, Any]] = None

# API Endpoints

@router.post("/currency-pair-relationships", response_model=Dict[str, Any])
async def analyze_currency_pair_relationships(
    request: CurrencyPairRequest,
    current_user: User = Depends(get_current_user)
):
    """
    Discovers causal relationships between currency pairs.
    
    Uses Granger causality to identify which currency pairs lead or cause movements in others.
    """
    logger.info(f"Received request to analyze relationships between {len(request.price_data)} currency pairs")
    
    try:
        # Convert dict of dicts to dict of DataFrames
        price_data = {}
        for pair, data_dict in request.price_data.items():
            import pandas as pd
            if "ohlc" in data_dict:
                price_data[pair] = pd.DataFrame(data_dict["ohlc"])
            else:
                price_data[pair] = pd.DataFrame(data_dict)
        
        # Initialize CausalInsightGenerator with config if provided
        insight_generator = CausalInsightGenerator(request.config)
        
        # Run analysis
        results = insight_generator.discover_currency_pair_relationships(
            price_data=price_data,
            max_lag=request.max_lag
        )
        
        return results
    
    except Exception as e:
        logger.error(f"Error analyzing currency pair relationships: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@router.post("/regime-change-drivers", response_model=Dict[str, Any])
async def analyze_regime_change_drivers(
    request: RegimeDriverRequest,
    current_user: User = Depends(get_current_user)
):
    """
    Discovers causal factors that drive market regime changes.
    
    Identifies which features have the strongest causal influence on regime transitions.
    """
    logger.info(f"Received request to analyze regime change drivers with {len(request.feature_columns)} features")
    
    try:
        # Convert dict to DataFrame
        import pandas as pd
        market_data = pd.DataFrame(request.market_data)
        
        # Initialize CausalInsightGenerator with config if provided
        insight_generator = CausalInsightGenerator(request.config)
        
        # Run analysis
        results = insight_generator.detect_regime_change_drivers(
            market_data=market_data,
            regime_column=request.regime_column,
            feature_columns=request.feature_columns
        )
        
        return results
    
    except Exception as e:
        logger.error(f"Error analyzing regime change drivers: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@router.post("/enhance-trading-signals", response_model=Dict[str, Any])
async def enhance_trading_signals(
    request: SignalEnhancementRequest,
    current_user: User = Depends(get_current_user)
):
    """
    Enhances trading signals with causal insights.
    
    Adds confidence adjustments, explanatory factors, conflicting signals,
    and expected duration based on causal analysis.
    """
    logger.info(f"Received request to enhance {len(request.signals)} trading signals")
    
    try:
        # Convert dict to DataFrame
        import pandas as pd
        market_data = pd.DataFrame(request.market_data)
        
        # Initialize CausalInsightGenerator with config if provided
        insight_generator = CausalInsightGenerator(request.config)
        
        # Run enhancement
        enhanced_signals = insight_generator.enhance_trading_signals(
            signals=request.signals,
            market_data=market_data
        )
        
        return {
            "enhanced_signals": enhanced_signals,
            "count": len(enhanced_signals),
            "causal_factors_considered": ["volatility", "trend", "sentiment", "correlations"]
        }
    
    except Exception as e:
        logger.error(f"Error enhancing trading signals: {e}")
        raise HTTPException(status_code=500, detail=f"Enhancement failed: {str(e)}")

@router.post("/correlation-breakdown-risk", response_model=Dict[str, Any])
async def assess_correlation_breakdown_risk(
    request: CorrelationRiskRequest,
    current_user: User = Depends(get_current_user)
):
    """
    Uses causal models to assess correlation breakdown risk between assets.
    
    Identifies pairs at risk of correlation breakdown and potential triggers.
    """
    logger.info("Received request to assess correlation breakdown risk")
    
    try:
        # Initialize CausalInsightGenerator with config if provided
        insight_generator = CausalInsightGenerator(request.config)
        
        # Run assessment
        risk_assessment = insight_generator.assess_correlation_breakdown_risk(
            correlation_data=request.correlation_data
        )
        
        return risk_assessment
    
    except Exception as e:
        logger.error(f"Error assessing correlation breakdown risk: {e}")
        raise HTTPException(status_code=500, detail=f"Assessment failed: {str(e)}")

@router.post("/counterfactual-scenarios", response_model=Dict[str, Any])
async def generate_counterfactual_scenarios(
    request: CounterfactualRequest,
    current_user: User = Depends(get_current_user)
):
    """
    Generates multiple counterfactual scenarios for risk assessment and strategy testing.
    
    Creates "what-if" scenarios based on interventions to evaluate potential outcomes.
    """
    logger.info(f"Received request to generate {len(request.interventions)} counterfactual scenarios")
    
    try:
        # Initialize CausalInsightGenerator
        insight_generator = CausalInsightGenerator()
        
        # Generate scenarios
        scenarios = insight_generator.generate_counterfactual_scenarios(
            base_scenario=request.base_scenario,
            interventions=request.interventions
        )
        
        return scenarios
    
    except Exception as e:
        logger.error(f"Error generating counterfactual scenarios: {e}")
        raise HTTPException(status_code=500, detail=f"Scenario generation failed: {str(e)}")
