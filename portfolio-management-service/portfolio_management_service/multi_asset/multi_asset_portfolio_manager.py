"""
Multi-Asset Portfolio Manager

This module extends the portfolio management capability to handle positions
across different asset classes with unified risk calculation.
"""

import logging
from typing import Dict, List, Any, Optional, Union
from datetime import datetime

from portfolio_management_service.models.position import Position, PositionCreate, PositionUpdate
from portfolio_management_service.services.portfolio_service import PortfolioService
from analysis_engine.multi_asset.asset_registry import AssetClass, AssetRegistry
from analysis_engine.services.multi_asset_service import MultiAssetService

logger = logging.getLogger(__name__)


class MultiAssetPortfolioManager:
    """
    Manages portfolio positions across different asset classes
    
    This class provides unified portfolio tracking, risk calculation, and
    position management across diverse asset classes.
    """
    
    def __init__(
        self, 
        portfolio_service: Optional[PortfolioService] = None,
        multi_asset_service: Optional[MultiAssetService] = None
    ):
        """Initialize the multi-asset portfolio manager"""
        self.portfolio_service = portfolio_service or PortfolioService()
        self.multi_asset_service = multi_asset_service or MultiAssetService()
        self.logger = logging.getLogger(__name__)
        
    def create_position(self, position_data: Dict[str, Any]) -> Position:
        """
        Create a new position with asset-specific handling
        
        Args:
            position_data: Position data including symbol and asset class
            
        Returns:
            Created position
        """
        # Get asset information
        symbol = position_data.get("symbol")
        asset_info = self.multi_asset_service.get_asset_info(symbol)
        
        if not asset_info:
            self.logger.warning(f"Asset info not found for {symbol}")
            
        # Add asset class to position data if not present
        if "asset_class" not in position_data and asset_info:
            position_data["asset_class"] = asset_info.get("asset_class")
            
        # Apply asset-specific adjustments
        self._apply_asset_specific_parameters(position_data, asset_info)
        
        # Create position
        position_create = PositionCreate(**position_data)
        return self.portfolio_service.create_position(position_create)
    
    def get_portfolio_summary(self, account_id: str) -> Dict[str, Any]:
        """
        Get portfolio summary with asset class breakdown
        
        Args:
            account_id: Account ID
            
        Returns:
            Portfolio summary with asset class breakdown
        """
        # Get basic portfolio summary
        summary = self.portfolio_service.get_portfolio_summary(account_id)
        
        # Add asset class breakdown
        positions = summary.get("positions", [])
        
        # Group positions by asset class
        by_asset_class = {}
        for position in positions:
            symbol = position.get("symbol")
            asset_info = self.multi_asset_service.get_asset_info(symbol)
            asset_class = asset_info.get("asset_class") if asset_info else "unknown"
            
            if asset_class not in by_asset_class:
                by_asset_class[asset_class] = {
                    "count": 0,
                    "value": 0,
                    "profit_loss": 0,
                    "margin_used": 0
                }
                
            by_asset_class[asset_class]["count"] += 1
            by_asset_class[asset_class]["value"] += position.get("current_value", 0)
            by_asset_class[asset_class]["profit_loss"] += position.get("unrealized_pl", 0)
            by_asset_class[asset_class]["margin_used"] += position.get("margin_used", 0)
        
        # Calculate allocation percentages
        total_value = summary.get("total_value", 0)
        if total_value > 0:
            for asset_class in by_asset_class:
                by_asset_class[asset_class]["allocation_pct"] = (
                    (by_asset_class[asset_class]["value"] / total_value) * 100
                )
        
        # Add to summary
        summary["by_asset_class"] = by_asset_class
        
        # Add cross-asset risk metrics
        summary["cross_asset_risk"] = self._calculate_cross_asset_risk(positions)
        
        return summary
    
    def calculate_unified_risk(self, account_id: str) -> Dict[str, Any]:
        """
        Calculate unified risk metrics across all asset classes
        
        Args:
            account_id: Account ID
            
        Returns:
            Risk metrics
        """
        # Get positions
        positions = self.portfolio_service.get_positions(account_id)
        
        # Calculate value at risk across all positions
        var = self._calculate_value_at_risk(positions)
        
        # Calculate correlation-adjusted risk
        correlation_risk = self._calculate_correlation_adjusted_risk(positions)
        
        # Calculate asset class concentration risk
        concentration_risk = self._calculate_concentration_risk(positions)
        
        return {
            "value_at_risk": var,
            "correlation_risk": correlation_risk,
            "concentration_risk": concentration_risk,
            "timestamp": datetime.now().isoformat()
        }
        
    def get_asset_allocation_recommendations(self, account_id: str) -> Dict[str, Any]:
        """
        Get asset allocation recommendations
        
        Args:
            account_id: Account ID
            
        Returns:
            Asset allocation recommendations
        """        # Get current allocation
        summary = self.get_portfolio_summary(account_id)
        current_allocation = summary.get("by_asset_class", {})
        risk_profile = self.portfolio_service.get_risk_profile(account_id)
        market_conditions = self.market_data_service.get_market_conditions()
        
        # Use mean-variance optimization for allocation
        optimal_allocation = self._optimize_allocation(
            current_allocation=current_allocation, 
            risk_profile=risk_profile,
            market_conditions=market_conditions
        )
        
        return {
            "current_allocation": current_allocation,
            "recommended_allocation": optimal_allocation,
            "explanation": self._generate_allocation_explanation(optimal_allocation, risk_profile, market_conditions)
        }
    
    def _apply_asset_specific_parameters(self, position_data: Dict[str, Any], asset_info: Dict[str, Any]) -> None:
        """Apply asset-specific parameters to position data"""
        if not asset_info:
            return
            
        asset_class = asset_info.get("asset_class")
        trading_params = asset_info.get("trading_parameters", {})
        
        # Set proper margin rate if available
        if "margin_rate" in trading_params:
            position_data["margin_rate"] = trading_params["margin_rate"]
            
        # Set pip value for forex
        if asset_class == AssetClass.FOREX and "pip_value" in trading_params:
            position_data["pip_value"] = trading_params["pip_value"]
            
        # Set trading fee if available
        if "trading_fee" in trading_params:
            position_data["fee"] = trading_params["trading_fee"]
    
    def _calculate_value_at_risk(self, positions: List[Position]) -> Dict[str, float]:
        """Calculate value at risk for positions"""
        # Placeholder for now
        return {
            "var_95": 0.0,
            "var_99": 0.0
        }
    
    def _calculate_correlation_adjusted_risk(self, positions: List[Position]) -> Dict[str, float]:
        """Calculate correlation-adjusted risk for positions"""
        # Placeholder for now
        return {
            "adjusted_portfolio_risk": 0.0,
            "diversification_benefit": 0.0
        }
    
    def _calculate_concentration_risk(self, positions: List[Position]) -> Dict[str, float]:
        """Calculate concentration risk by asset class"""
        # Placeholder for now
        return {
            "concentration_score": 0.0,
            "max_concentrated_class": "none"
        }
    
    def _calculate_cross_asset_risk(self, positions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate cross-asset risk metrics"""
        # Placeholder for now
        return {
            "cross_correlation": 0.0,
            "diversification_score": 0.0
        }
    
    def _optimize_allocation(self, current_allocation: Dict[str, float], risk_profile: Dict[str, Any], 
                          market_conditions: Dict[str, Any]) -> Dict[str, float]:
        """
        Optimize asset allocation using mean-variance optimization.
        
        Args:
            current_allocation: Current asset allocation percentages
            risk_profile: User risk profile data
            market_conditions: Current market conditions
            
        Returns:
            Optimized asset allocation
        """
        # Get expected returns and covariance matrix for different asset classes
        expected_returns = self._get_expected_returns(market_conditions)
        covariance_matrix = self._get_covariance_matrix(market_conditions)
        
        # Determine risk tolerance from user's risk profile (0-1 scale)
        risk_tolerance = risk_profile.get("risk_tolerance", 0.5) 
        
        # Apply constraints based on risk profile
        # Higher risk tolerance allows for more volatile assets (crypto, stocks)
        constraints = {
            "forex": {"min": 0.2, "max": 0.7},
            "stocks": {"min": 0.1, "max": 0.5 + (risk_tolerance * 0.3)},
            "crypto": {"min": 0.0, "max": risk_tolerance * 0.4},
            "commodities": {"min": 0.05, "max": 0.3}
        }
        
        # Simple optimization: balance risk/return based on risk tolerance
        # For a full implementation, would use quadratic programming library
        allocation = {}
        
        # Higher risk tolerance emphasizes expected returns over risk
        # Lower risk tolerance emphasizes risk minimization
        for asset in expected_returns.keys():
            # Adjust allocation based on:
            # 1. Expected returns (higher = better)
            # 2. Risk (from covariance matrix diagonal - lower = better)
            # 3. Market condition factors
            # 4. Risk tolerance (how much to weight returns vs risk)
            
            # Calculate score for this asset
            return_score = expected_returns[asset] * risk_tolerance
            risk_score = covariance_matrix[asset][asset] * (1 - risk_tolerance)
            
            # Market conditions modifier (-0.2 to +0.2 based on conditions)
            market_modifier = market_conditions.get("asset_modifiers", {}).get(asset, 0)
            
            # Calculate initial proportional allocation
            allocation[asset] = max(
                constraints[asset]["min"],
                min(
                    constraints[asset]["max"],
                    return_score / risk_score + market_modifier
                )
            )
        
        # Normalize to ensure sum is 100%
        total = sum(allocation.values())
        for asset in allocation:
            allocation[asset] = round(allocation[asset] / total * 100)
        
        # Ensure we still hit 100% after rounding
        adjustment = 100 - sum(allocation.values())
        if adjustment != 0:
            # Add/subtract adjustment to largest allocation
            max_asset = max(allocation, key=allocation.get)
            allocation[max_asset] += adjustment
            
        return allocation
    
    def _get_expected_returns(self, market_conditions: Dict[str, Any]) -> Dict[str, float]:
        """Get expected returns for each asset class based on market conditions"""
        base_returns = {
            "forex": 0.05,
            "stocks": 0.08,
            "crypto": 0.15,
            "commodities": 0.06
        }
        
        # Apply market condition modifiers
        regime = market_conditions.get("regime", "normal")
        modifiers = {
            "bull": {"forex": -0.01, "stocks": 0.04, "crypto": 0.08, "commodities": -0.01},
            "bear": {"forex": 0.02, "stocks": -0.05, "crypto": -0.1, "commodities": 0.03},
            "volatile": {"forex": 0.01, "stocks": -0.02, "crypto": 0.05, "commodities": 0.02},
            "normal": {"forex": 0, "stocks": 0, "crypto": 0, "commodities": 0}
        }
        
        result = {}
        for asset, base_return in base_returns.items():
            result[asset] = base_return + modifiers.get(regime, {}).get(asset, 0)
            
        return result
    
    def _get_covariance_matrix(self, market_conditions: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
        """Get covariance matrix for asset classes based on market conditions"""
        # Base covariance matrix (simplified for clarity)
        # Diagonal elements represent variance (risk) of each asset
        base_matrix = {
            "forex": {"forex": 0.03, "stocks": 0.01, "crypto": 0.02, "commodities": 0.015},
            "stocks": {"forex": 0.01, "stocks": 0.05, "crypto": 0.03, "commodities": 0.02},
            "crypto": {"forex": 0.02, "stocks": 0.03, "crypto": 0.25, "commodities": 0.01},
            "commodities": {"forex": 0.015, "stocks": 0.02, "crypto": 0.01, "commodities": 0.04}
        }
        
        # Adjust covariance based on market regime
        regime = market_conditions.get("regime", "normal")
        
        # Apply multipliers to the covariance matrix based on regime
        multipliers = {
            "bull": 0.8,  # Lower covariance in bull markets
            "bear": 1.5,  # Higher covariance in bear markets
            "volatile": 2.0,  # Much higher covariance in volatile markets
            "normal": 1.0   # No change in normal markets
        }
        
        # Apply the multiplier
        multiplier = multipliers.get(regime, 1.0)
        result = {}
        
        for asset1, covs in base_matrix.items():
            result[asset1] = {}
            for asset2, cov in covs.items():
                result[asset1][asset2] = cov * multiplier
                
        return result
    
    def _generate_allocation_explanation(self, allocation: Dict[str, float], 
                                       risk_profile: Dict[str, Any],
                                       market_conditions: Dict[str, Any]) -> str:
        """Generate an explanation for the recommended allocation"""
        risk_level = risk_profile.get("risk_tolerance", 0.5)
        risk_text = "conservative" if risk_level < 0.3 else "moderate" if risk_level < 0.7 else "aggressive"
        regime = market_conditions.get("regime", "normal")
        
        explanation = f"Recommended allocation is optimized for your {risk_text} risk profile "
        explanation += f"and the current {regime} market conditions. "
        
        # Add asset-specific explanations
        if allocation.get("crypto", 0) > 30:
            explanation += "The allocation favors higher crypto exposure due to positive market momentum. "
        elif allocation.get("forex", 0) > 50:
            explanation += "The allocation emphasizes forex stability in current market conditions. "
        
        explanation += "The recommendation balances risk and potential returns while maintaining diversification."
        
        return explanation
