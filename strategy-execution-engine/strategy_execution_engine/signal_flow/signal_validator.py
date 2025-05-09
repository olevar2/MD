"""
Implementation of signal validation in the Strategy Execution Engine.
"""
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging

from common_lib.signal_flow.interface import ISignalValidator
from common_lib.signal_flow.model import SignalFlow, SignalValidationResult, SignalFlowState

class SignalValidator(ISignalValidator):
    """
    Validates trading signals before execution.
    
    Validation includes:
    1. Market data validation (price, spread, etc.)
    2. Risk checks (exposure, correlation, etc.)
    3. Strategy constraints (max positions, etc.)
    4. Technical validity (indicators, patterns)
    5. Position sizing and risk parameters
    """
    
    def __init__(
        self,
        risk_checker,  # Risk management service client
        market_data_service,  # Market data service client
        position_manager,  # Position management service client
        config: Dict[str, Any]
    ):
        self.risk_checker = risk_checker
        self.market_data = market_data_service
        self.position_manager = position_manager
        self.config = config
        self.logger = logging.getLogger(__name__)
        
    async def validate_signal(self, signal: SignalFlow) -> SignalValidationResult:
        """
        Validate a trading signal.
        
        Args:
            signal: The signal to validate
            
        Returns:
            SignalValidationResult containing validation results
        """
        validation_checks = {}
        risk_metrics = {}
        notes = []
        
        try:
            # 1. Market data validation
            market_valid, market_notes = await self._validate_market_data(signal)
            validation_checks["market_data_valid"] = market_valid
            notes.extend(market_notes)
            
            if not market_valid:
                return self._create_result(False, validation_checks, risk_metrics, notes)
                
            # 2. Risk checks
            risk_valid, risk_data = await self._validate_risk_parameters(signal)
            validation_checks["risk_checks_passed"] = risk_valid
            risk_metrics.update(risk_data)
            
            if not risk_valid:
                return self._create_result(False, validation_checks, risk_metrics, notes)
                
            # 3. Strategy constraints
            strategy_valid, strategy_notes = await self._validate_strategy_constraints(signal)
            validation_checks["strategy_constraints_met"] = strategy_valid
            notes.extend(strategy_notes)
            
            if not strategy_valid:
                return self._create_result(False, validation_checks, risk_metrics, notes)
                
            # 4. Technical validity
            tech_valid, tech_notes = self._validate_technical_parameters(signal)
            validation_checks["technical_parameters_valid"] = tech_valid
            notes.extend(tech_notes)
            
            if not tech_valid:
                return self._create_result(False, validation_checks, risk_metrics, notes)
                
            # 5. Position sizing
            position_valid, position_notes = await self._validate_position_sizing(signal)
            validation_checks["position_sizing_valid"] = position_valid
            notes.extend(position_notes)
            
            return self._create_result(
                all(validation_checks.values()),
                validation_checks,
                risk_metrics,
                notes
            )
            
        except Exception as e:
            self.logger.error(f"Error validating signal: {str(e)}", exc_info=True)
            notes.append(f"Validation error: {str(e)}")
            return self._create_result(False, validation_checks, risk_metrics, notes)
            
    async def _validate_market_data(self, signal: SignalFlow) -> tuple[bool, List[str]]:
        """Validate market data conditions"""
        notes = []
        try:
            # Get current market data
            market_data = await self.market_data.get_latest_data(signal.symbol)
            
            # Check if market is open
            if not market_data.get("market_open", False):
                notes.append("Market is closed")
                return False, notes
                
            # Check spread
            current_spread = market_data.get("spread", float("inf"))
            max_spread = self.config["max_spread"].get(signal.symbol, 0.0003)
            
            if current_spread > max_spread:
                notes.append(f"Spread too high: {current_spread} > {max_spread}")
                return False, notes
                
            # Check liquidity
            current_volume = market_data.get("volume", 0)
            min_volume = self.config["min_volume"].get(signal.symbol, 100000)
            
            if current_volume < min_volume:
                notes.append(f"Insufficient volume: {current_volume} < {min_volume}")
                return False, notes
                
            # Validate price
            current_price = market_data.get("price", 0)
            if abs(current_price - signal.suggested_entry) / current_price > 0.001:
                notes.append("Signal price significantly different from current price")
                return False, notes
                
            notes.append("Market conditions valid")
            return True, notes
            
        except Exception as e:
            notes.append(f"Market data validation error: {str(e)}")
            return False, notes
            
    async def _validate_risk_parameters(self, signal: SignalFlow) -> tuple[bool, Dict[str, float]]:
        """Validate risk management parameters"""
        try:
            risk_check = await self.risk_checker.check_signal_risk(
                symbol=signal.symbol,
                direction=signal.direction,
                size=signal.risk_parameters.get("position_size", 0),
                stop_loss=signal.suggested_stop,
                take_profit=signal.suggested_target
            )
            
            return risk_check["is_valid"], risk_check["metrics"]
            
        except Exception as e:
            self.logger.error(f"Risk validation error: {str(e)}")
            return False, {}
            
    async def _validate_strategy_constraints(self, signal: SignalFlow) -> tuple[bool, List[str]]:
        """Validate strategy-specific constraints"""
        notes = []
        try:
            # Check max positions per symbol
            current_positions = await self.position_manager.get_positions(signal.symbol)
            max_positions = self.config["max_positions_per_symbol"]
            
            if len(current_positions) >= max_positions:
                notes.append(f"Max positions ({max_positions}) reached for {signal.symbol}")
                return False, notes
                
            # Check correlation constraints
            if signal.metadata.get("correlated_pairs"):
                corr_exposure = await self.position_manager.get_correlated_exposure(
                    signal.symbol,
                    signal.metadata["correlated_pairs"]
                )
                if corr_exposure > self.config["max_correlation_exposure"]:
                    notes.append("Correlated exposure too high")
                    return False, notes
                    
            notes.append("Strategy constraints satisfied")
            return True, notes
            
        except Exception as e:
            notes.append(f"Strategy constraint validation error: {str(e)}")
            return False, notes
            
    def _validate_technical_parameters(self, signal: SignalFlow) -> tuple[bool, List[str]]:
        """Validate technical analysis parameters"""
        notes = []
        
        # Validate confidence
        if signal.confidence < self.config["min_signal_confidence"]:
            notes.append(f"Signal confidence too low: {signal.confidence}")
            return False, notes
            
        # Validate confluence
        if signal.confluence_score < self.config["min_confluence_score"]:
            notes.append(f"Insufficient confluence: {signal.confluence_score}")
            return False, notes
            
        # Validate technical context
        tech_context = signal.technical_context
        if not tech_context.get("trend_aligned", False):
            notes.append("Signal not aligned with trend")
            return False, notes
            
        notes.append("Technical parameters valid")
        return True, notes
        
    async def _validate_position_sizing(self, signal: SignalFlow) -> tuple[bool, List[str]]:
        """Validate position sizing parameters"""
        notes = []
        try:
            risk_params = signal.risk_parameters
            
            # Validate risk-reward ratio
            if risk_params.get("risk_reward_ratio", 0) < self.config["min_risk_reward_ratio"]:
                notes.append("Insufficient risk-reward ratio")
                return False, notes
                
            # Validate position size
            account_info = await self.position_manager.get_account_info()
            max_position_size = account_info["equity"] * self.config["max_position_size_percent"]
            
            if risk_params.get("position_size", 0) > max_position_size:
                notes.append("Position size too large")
                return False, notes
                
            notes.append("Position sizing valid")
            return True, notes
            
        except Exception as e:
            notes.append(f"Position sizing validation error: {str(e)}")
            return False, notes
            
    def _create_result(
        self,
        is_valid: bool,
        validation_checks: Dict[str, bool],
        risk_metrics: Dict[str, float],
        notes: List[str]
    ) -> SignalValidationResult:
        """Create a validation result object"""
        return SignalValidationResult(
            is_valid=is_valid,
            validation_checks=validation_checks,
            risk_metrics=risk_metrics,
            notes=notes
        )
