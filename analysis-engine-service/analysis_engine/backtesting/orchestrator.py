"""
Core components for the backtesting framework.
"""
import logging
import pandas as pd
from datetime import datetime
from typing import Dict, Any, List, Optional, Type, Union
import numpy as np # For calculations

from core_foundations.models.trading import Trade, Order, MarketData, Position, OrderType, OrderStatus, TradeSignal, ActionType
from core_foundations.models.portfolio import PortfolioSnapshot, PortfolioPerformanceMetrics
from core_foundations.models.risk import RiskParameters, DynamicRiskAdjustment
# Import actual Strategy/Agent base classes when available
# from strategy_execution_engine.strategies.base import BaseStrategy # Placeholder
# from ml_workbench_service.models.rl_agent import BaseRLAgent # Placeholder

# Import Data Providers
from .data_providers import BaseDataProvider, GeneratedDataProvider, HistoricalDatabaseProvider

# Simulators (adjust paths as needed)
from trading_gateway_service.simulation.forex_broker_simulator import ForexBrokerSimulator
from trading_gateway_service.simulation.market_regime_simulator import MarketRegimeSimulator, MarketRegimeGenerator
from trading_gateway_service.simulation.news_sentiment_simulator import NewsAndSentimentSimulator

# Risk Components (adjust paths as needed)
from risk_management_service.rl_risk_adapter import RLRiskAdapter
from risk_management_service.rl_risk_parameter_optimizer import RLRiskParameterOptimizer

from analysis_engine.caching.cache_service import cache_result # Added import

logger = logging.getLogger(__name__)

class BacktestingConfiguration:
    """Configuration settings for a backtesting run."""
    def __init__(
        self,
        start_date: datetime,
        end_date: datetime,
        initial_capital: float,
        symbols: List[str],
        strategy_class: Type, # Type[BaseStrategy] or Type[BaseRLAgent]
        strategy_config: Dict[str, Any],
        broker_config: Dict[str, Any],
        risk_parameters: RiskParameters, # Moved this parameter up
        market_regime_config: Optional[Dict[str, Any]] = None,
        news_sentiment_config: Optional[Dict[str, Any]] = None,
        # risk_parameters: RiskParameters, # Removed from original position
        use_rl_risk_optimization: bool = False,
        rl_risk_adapter_config: Optional[Dict[str, Any]] = None, # Config for adapter if needed
        rl_risk_optimizer_config: Optional[Dict[str, Any]] = None,
        data_source: str = "historical_db", # Or 'generated', 'live_replay' etc.
        data_parameters: Optional[Dict[str, Any]] = None, # Params for data source
    ):
        self.start_date = start_date
        self.end_date = end_date
        self.initial_capital = initial_capital
        self.symbols = symbols
        self.strategy_class = strategy_class
        self.strategy_config = strategy_config
        self.broker_config = broker_config
        self.market_regime_config = market_regime_config or {}
        self.news_sentiment_config = news_sentiment_config or {}
        self.risk_parameters = risk_parameters
        self.use_rl_risk_optimization = use_rl_risk_optimization
        self.rl_risk_adapter_config = rl_risk_adapter_config or {}
        self.rl_risk_optimizer_config = rl_risk_optimizer_config or {}
        self.data_source = data_source
        self.data_parameters = data_parameters or {}
        logger.info(f"BacktestingConfiguration created: {start_date} to {end_date} for {symbols}")


class BacktestingOrchestrator:
    """
    Orchestrates the backtesting process, integrating data, simulators,
    strategy execution, risk management, and performance analysis.
    """

    def __init__(self, config: BacktestingConfiguration):
        """
        Initializes the backtesting environment based on the configuration.
        """
        self.config = config
        self.results: Optional[BacktestingResults] = None
        self._initialize_components()
        logger.info("BacktestingOrchestrator initialized.")

    def _initialize_components(self):
        """Initialize all necessary components for the backtest."""
        logger.debug("Initializing backtesting components...")

        # 1. Data Provider
        self.data_provider: BaseDataProvider = self._setup_data_provider() # Type hint added
        logger.debug(f"Data provider setup for source: {self.config.data_source}")

        # 2. Simulators
        self.broker_simulator = ForexBrokerSimulator(**self.config.broker_config)
        logger.debug("ForexBrokerSimulator initialized.")

        self.market_regime_simulator = None
        if self.config.market_regime_config:
            # Assuming MarketRegimeGenerator is needed by MarketRegimeSimulator
            # This might need adjustment based on actual class structure
            regime_generator = MarketRegimeGenerator(config=self.config.market_regime_config.get('generator', {}))
            self.market_regime_simulator = MarketRegimeSimulator(
                generator=regime_generator,
                config=self.config.market_regime_config.get('simulator', {})
            )
            logger.debug("MarketRegimeSimulator initialized.")

        self.news_sentiment_simulator = None
        if self.config.news_sentiment_config:
            self.news_sentiment_simulator = NewsAndSentimentSimulator(**self.config.news_sentiment_config)
            logger.debug("NewsAndSentimentSimulator initialized.")

        # 3. Strategy / RL Agent
        # TODO: Replace placeholders with actual class instantiation
        # This assumes strategy_class is passed correctly in config
        if hasattr(self.config.strategy_class, 'get_action'): # Heuristic to check if it's an RL Agent
             # self.agent: BaseRLAgent = self.config.strategy_class(**self.config.strategy_config) # Placeholder
             self.strategy_or_agent = self.config.strategy_class(**self.config.strategy_config) # Use a generic name
             self.is_rl_agent = True
             logger.debug(f"RL Agent {self.config.strategy_class.__name__} initialized.")
        else: # Assume it's a traditional strategy
             # self.strategy: BaseStrategy = self.config.strategy_class(**self.config.strategy_config) # Placeholder
             self.strategy_or_agent = self.config.strategy_class(**self.config.strategy_config) # Use a generic name
             self.is_rl_agent = False
             logger.debug(f"Strategy {self.config.strategy_class.__name__} initialized.")


        # 4. Risk Management
        self.current_risk_parameters = self.config.risk_parameters.copy(deep=True)
        self.rl_risk_optimizer = None
        if self.config.use_rl_risk_optimization:
            # Requires an initialized RLRiskAdapter instance
            # This needs refinement - how is the adapter provided/initialized?
            # Maybe pass adapter instance or factory in config?
            # For now, assuming a mock/placeholder adapter can be created
            mock_rl_adapter = RLRiskAdapter() # Placeholder - Needs proper initialization
            self.rl_risk_optimizer = RLRiskParameterOptimizer(
                rl_risk_adapter=mock_rl_adapter,
                config=self.config.rl_risk_optimizer_config
            )
            logger.debug("RLRiskParameterOptimizer initialized.")
        else:
             logger.debug("RL Risk Optimization disabled.")


        # 5. Portfolio State
        self.portfolio_history: List[PortfolioSnapshot] = []
        self.trades: List[Trade] = []
        self.orders: List[Order] = []
        self.current_positions: Dict[str, Position] = {}
        self.current_balance = self.config.initial_capital
        self.current_equity = self.config.initial_capital
        self.current_timestamp: Optional[datetime] = None

        logger.info("Backtesting components initialization complete.")


    def _setup_data_provider(self) -> BaseDataProvider:
        """Sets up the data provider based on configuration."""
        source = self.config.data_source.lower()
        params = self.config.data_parameters or {}

        if source == 'generated':
            if not self.market_regime_simulator:
                raise ValueError("MarketRegimeSimulator must be configured for 'generated' data source.")
            # Pass the simulator instance and any specific generator params
            return GeneratedDataProvider(self.market_regime_simulator, params)
        elif source == 'historical_db':
            db_params = params.get('db_connection', {}) # Expecting DB connection details here
            if not db_params:
                 logger.warning("No 'db_connection' parameters found in data_parameters for historical_db source.")
            # Pass DB connection details and other relevant params
            return HistoricalDatabaseProvider(db_params, params)
        # TODO: Add elif for 'csv', 'live_replay' etc.
        else:
            raise NotImplementedError(f"Data source '{self.config.data_source}' is not supported.")

    @cache_result(ttl=86400) # Cache for 24 hours
    async def run(self):
        """Executes the backtest loop."""
        logger.info(f"Starting backtest run from {self.config.start_date} to {self.config.end_date}...")

        if not self.data_provider:
             raise RuntimeError("Data provider not initialized. Cannot run backtest.")

        # --- Backtesting Loop ---
        async for timestamp, market_data_batch in self.data_provider.stream_data(
            self.config.start_date, self.config.end_date, self.config.symbols
            # TODO: Pass timeframe from config if needed by provider
        ):
            self.current_timestamp = timestamp
            logger.debug(f"--- Processing Timestamp: {timestamp} ---")

            # 1. Update Simulators (Market Regime, News) if applicable
            current_regime = self.market_regime_simulator.get_current_regime(timestamp) if self.market_regime_simulator else None
            news_events = self.news_sentiment_simulator.get_events(timestamp) if self.news_sentiment_simulator else []
            # TODO: Potentially feed simulator outputs into strategy/agent state

            # 2. Update Broker Simulator State (e.g., spreads based on regime/news)
            self.broker_simulator.update_market_conditions(timestamp, current_regime, news_events)

            # 3. Update Portfolio Equity based on current prices & positions
            self._update_portfolio_equity(market_data_batch)

            # 4. Process Market Data for each symbol
            orders_to_submit: List[Order] = []
            for symbol, market_data in market_data_batch.items():
                if not market_data: # Skip if no data for this symbol at this timestamp
                    continue

                current_position = self.current_positions.get(symbol)
                logger.debug(f"Processing {symbol}: Price={market_data.close}, Position={current_position}")

                # 5. Check for Stop Loss / Take Profit triggers with Broker Simulator
                # This assumes broker simulator handles SL/TP based on market data feed
                triggered_orders = self.broker_simulator.check_conditional_orders(market_data, current_position)
                if triggered_orders:
                    logger.info(f"Conditional orders triggered for {symbol}: {triggered_orders}")
                    await self._process_executed_orders(triggered_orders)
                    # Update position state immediately after SL/TP execution
                    current_position = self.current_positions.get(symbol)


                # 6. Get RL Risk Adjustments (if enabled)
                risk_params_to_use = self.current_risk_parameters # Start with base
                if self.rl_risk_optimizer:
                    portfolio_snapshot = self._get_current_portfolio_snapshot() # Get current state
                    dynamic_adjustment: DynamicRiskAdjustment = await self.rl_risk_optimizer.suggest_risk_adjustments(
                        symbol=symbol,
                        current_parameters=self.config.risk_parameters, # Use original config as base
                        market_data=market_data,
                        current_position=current_position,
                        portfolio_snapshot=portfolio_snapshot
                    )
                    risk_params_to_use = self._apply_dynamic_risk_adjustments(
                        self.config.risk_parameters, dynamic_adjustment
                    )
                    logger.debug(f"Applied dynamic risk adjustments for {symbol}: {risk_params_to_use}")


                # 7. Get Strategy/Agent Signal/Action
                # Prepare state for the strategy/agent
                strategy_input_state = {
                     "timestamp": timestamp,
                     "market_data": market_data,
                     "current_position": current_position,
                     "portfolio_snapshot": self._get_current_portfolio_snapshot(),
                     "risk_parameters": risk_params_to_use,
                     "current_regime": current_regime,
                     "news_events": news_events,
                     # Add other relevant state: rl_insights? technical indicators?
                }

                signal_or_action: Union[TradeSignal, Any, None] = None # Use Any for RL action flexibility
                try:
                    if self.is_rl_agent:
                        # Assuming agent needs observation, returns action
                        # observation = self._prepare_rl_observation(strategy_input_state) # Needs implementation
                        # action = await self.strategy_or_agent.get_action(observation)
                        # signal_or_action = action # Store the raw action
                        logger.warning("RL Agent action generation not fully implemented.") # Placeholder
                    else:
                        # Assuming strategy returns a TradeSignal or None
                        signal: Optional[TradeSignal] = await self.strategy_or_agent.generate_signal(**strategy_input_state)
                        signal_or_action = signal
                        if signal:
                             logger.info(f"Strategy generated signal for {symbol}: {signal.action_type}")

                except Exception as e:
                     logger.error(f"Error getting signal/action for {symbol} from {self.config.strategy_class.__name__}: {e}", exc_info=True)
                     signal_or_action = None


                # 8. Generate Orders based on Signal/Action
                # Pass risk_params_to_use for order sizing, SL/TP calculation
                symbol_orders = self._translate_signal_to_orders(
                    symbol, signal_or_action, market_data, risk_params_to_use, current_position
                )
                orders_to_submit.extend(symbol_orders)


            # 9. Submit All New Orders for this Timestamp to Broker Simulator
            if orders_to_submit:
                logger.debug(f"Submitting {len(orders_to_submit)} orders for timestamp {timestamp}")
                executed_orders = await self.broker_simulator.process_orders(orders_to_submit)
                await self._process_executed_orders(executed_orders)


            # 10. Record Portfolio Snapshot for this timestamp
            self.portfolio_history.append(self._get_current_portfolio_snapshot())
            logger.debug(f"End of Timestamp {timestamp}: Equity={self.current_equity:.2f}, Balance={self.current_balance:.2f}")

            # --- End of Timestamp Loop ---

        logger.info("Backtest run finished.")

        # 11. Calculate Performance Metrics
        self.results = self._calculate_results()
        logger.info(f"Backtest results calculated. Final Equity: {self.results.final_equity:.2f}")
        logger.info(f"Metrics: {self.results.summary()}") # Log summary

        return self.results

    def _update_portfolio_equity(self, market_data_batch: Dict[str, MarketData]):
        """Updates current equity based on open positions and current market prices."""
        unrealized_pnl = 0.0
        for symbol, position in self.current_positions.items():
            if symbol in market_data_batch and market_data_batch[symbol]:
                market_data = market_data_batch[symbol]
                # Use broker simulator's method to get accurate PnL considering spread
                pnl = self.broker_simulator.calculate_unrealized_pnl(position, market_data)
                unrealized_pnl += pnl
            else:
                # If market data is missing, we can't update PnL accurately for this symbol
                # Option 1: Use last known PnL (might be stale)
                # Option 2: Log a warning and potentially skip PnL update for this symbol
                logger.warning(f"Missing market data for open position {symbol} at {self.current_timestamp}. Cannot update PnL accurately.")

        self.current_equity = self.current_balance + unrealized_pnl
        # TODO: Add margin calculations if needed

    def _apply_dynamic_risk_adjustments(
        self,
        base_parameters: RiskParameters,
        adjustment: DynamicRiskAdjustment
    ) -> RiskParameters:
        """Applies dynamic adjustments to a copy of the base risk parameters."""
        adjusted_params = base_parameters.copy(deep=True)

        if adjustment.position_size_scaling_factor is not None:
            adjusted_params.max_position_size_pct *= adjustment.position_size_scaling_factor
            logger.debug(f"Adjusted max_position_size_pct by factor {adjustment.position_size_scaling_factor:.2f}")

        if adjustment.stop_loss_adjustment_factor is not None:
             # Assuming stop_loss_pip is defined. If using relative SL, adjust that instead.
             if adjusted_params.stop_loss_pip:
                 adjusted_params.stop_loss_pip *= adjustment.stop_loss_adjustment_factor
                 logger.debug(f"Adjusted stop_loss_pip by factor {adjustment.stop_loss_adjustment_factor:.2f}")
             # TODO: Handle relative stop loss adjustments if applicable

        if adjustment.take_profit_adjustment_factor is not None:
             if adjusted_params.take_profit_pip:
                 adjusted_params.take_profit_pip *= adjustment.take_profit_adjustment_factor
                 logger.debug(f"Adjusted take_profit_pip by factor {adjustment.take_profit_adjustment_factor:.2f}")
             # TODO: Handle relative take profit adjustments

        if adjustment.max_leverage_adjustment_factor is not None:
             if adjusted_params.max_leverage:
                 adjusted_params.max_leverage *= adjustment.max_leverage_adjustment_factor
                 logger.debug(f"Adjusted max_leverage by factor {adjustment.max_leverage_adjustment_factor:.2f}")

        # TODO: Apply other adjustments if defined in DynamicRiskAdjustment model

        return adjusted_params


    def _translate_signal_to_orders(
        self,
        symbol: str,
        signal_or_action: Union[TradeSignal, Any, None],
        market_data: MarketData,
        risk_params: RiskParameters,
        current_position: Optional[Position]
    ) -> List[Order]:
        """Translates a strategy signal or RL action into concrete orders."""
        orders = []
        if signal_or_action is None:
            return orders

        action_type = None
        target_size = 0.0
        entry_price = None # For limit/stop orders
        sl_price = None
        tp_price = None

        # --- Determine Action and Size ---
        if isinstance(signal_or_action, TradeSignal): # Traditional Strategy Signal
            signal: TradeSignal = signal_or_action
            action_type = signal.action_type
            # Use risk_params to calculate size (e.g., % of equity)
            # This is a simplified example; real calculation needs account equity, leverage, contract size etc.
            if action_type == ActionType.BUY or action_type == ActionType.SELL:
                 # Simplified sizing based on % of current equity
                 equity_risk_per_trade = self.current_equity * risk_params.max_position_size_pct
                 # Need pip value and stop distance to calculate size accurately
                 # Placeholder size calculation - replace with proper logic from risk/portfolio service
                 target_size = 1000 # Dummy size
                 logger.warning("Order sizing uses placeholder logic. Implement proper calculation.")

            sl_price = signal.stop_loss
            tp_price = signal.take_profit
            entry_price = signal.entry_price # If it's a limit/stop signal

        elif self.is_rl_agent: # RL Agent Action
            # TODO: Interpret the RL agent's action. This depends heavily on the action space definition.
            # Example: action could be [action_type_index, size_percentage, sl_pips, tp_pips]
            # action = signal_or_action
            # action_type_index = action[0]
            # if action_type_index == 0: action_type = ActionType.HOLD # Or None
            # elif action_type_index == 1: action_type = ActionType.BUY
            # elif action_type_index == 2: action_type = ActionType.SELL
            # size_percentage = action[1]
            # target_size = self._calculate_size_from_percentage(size_percentage, risk_params) # Needs implementation
            # sl_pips = action[2]
            # tp_pips = action[3]
            # sl_price = self._calculate_sl_price(market_data.close, action_type, sl_pips) # Needs implementation
            # tp_price = self._calculate_tp_price(market_data.close, action_type, tp_pips) # Needs implementation
            logger.warning("RL action interpretation and order generation not implemented.")
            action_type = None # Prevent order generation until implemented

        else:
             logger.error(f"Unsupported signal/action type: {type(signal_or_action)}")
             return []


        # --- Generate Orders ---
        if action_type == ActionType.BUY:
            if current_position and current_position.quantity < 0: # Close short, potentially open long
                close_order = Order(
                    symbol=symbol, order_type=OrderType.MARKET, quantity=abs(current_position.quantity),
                    action_type=ActionType.BUY, timestamp=self.current_timestamp
                )
                orders.append(close_order)
                logger.debug(f"Generated order to close short position: {close_order}")
            if not current_position or current_position.quantity <= 0: # Open new long or add to existing
                 open_order = Order(
                     symbol=symbol, order_type=OrderType.MARKET, quantity=target_size,
                     action_type=ActionType.BUY, timestamp=self.current_timestamp,
                     stop_loss_price=sl_price, take_profit_price=tp_price
                 )
                 # TODO: Handle limit/stop entry orders based on entry_price
                 orders.append(open_order)
                 logger.debug(f"Generated order to open/add long position: {open_order}")

        elif action_type == ActionType.SELL:
             if current_position and current_position.quantity > 0: # Close long, potentially open short
                 close_order = Order(
                     symbol=symbol, order_type=OrderType.MARKET, quantity=abs(current_position.quantity),
                     action_type=ActionType.SELL, timestamp=self.current_timestamp
                 )
                 orders.append(close_order)
                 logger.debug(f"Generated order to close long position: {close_order}")
             if not current_position or current_position.quantity >= 0: # Open new short or add to existing
                 open_order = Order(
                     symbol=symbol, order_type=OrderType.MARKET, quantity=target_size, # Sell uses positive quantity here
                     action_type=ActionType.SELL, timestamp=self.current_timestamp,
                     stop_loss_price=sl_price, take_profit_price=tp_price
                 )
                 # TODO: Handle limit/stop entry orders based on entry_price
                 orders.append(open_order)
                 logger.debug(f"Generated order to open/add short position: {open_order}")

        elif action_type == ActionType.CLOSE:
             if current_position:
                 close_action = ActionType.SELL if current_position.quantity > 0 else ActionType.BUY
                 close_order = Order(
                     symbol=symbol, order_type=OrderType.MARKET, quantity=abs(current_position.quantity),
                     action_type=close_action, timestamp=self.current_timestamp
                 )
                 orders.append(close_order)
                 logger.debug(f"Generated order to close position based on signal: {close_order}")

        # TODO: Handle ActionType.HOLD (usually means do nothing)

        return orders


    async def _process_executed_orders(self, executed_orders: List[Order]):
         """Updates portfolio state based on executed orders from the simulator."""
         if not executed_orders:
             return

         logger.debug(f"Processing {len(executed_orders)} executed orders...")
         for order in executed_orders:
             if order.status != OrderStatus.FILLED:
                 logger.warning(f"Received non-filled order in execution processing: {order}")
                 continue # Should only process filled orders here

             self.orders.append(order) # Log all executed orders

             # --- Create Trade Object ---
             # Assumes order has fill_price and fill_timestamp from simulator
             trade = Trade(
                 trade_id=f"trade_{order.order_id}", # Generate unique trade ID
                 order_id=order.order_id,
                 symbol=order.symbol,
                 timestamp=order.fill_timestamp or self.current_timestamp, # Use fill time if available
                 quantity=order.quantity if order.action_type == ActionType.BUY else -order.quantity,
                 price=order.fill_price,
                 # TODO: Calculate commission, fees based on broker simulator config
                 commission=self.broker_simulator.calculate_commission(order),
                 pnl=0.0 # PnL is realized when position is closed/reduced
             )

             # --- Update Balance ---
             cost = trade.quantity * trade.price # Negative for BUY, Positive for SELL
             self.current_balance -= cost # Cost of shares/contracts
             self.current_balance -= trade.commission # Subtract commission
             logger.debug(f"Balance updated: {self.current_balance:.2f} after trade {trade.trade_id} (Cost: {cost:.2f}, Comm: {trade.commission:.2f})")


             # --- Update Position ---
             symbol = order.symbol
             position = self.current_positions.get(symbol)

             if not position: # New position
                 position = Position(
                     symbol=symbol,
                     quantity=trade.quantity,
                     average_entry_price=trade.price,
                     last_update_timestamp=trade.timestamp
                     # TODO: Add initial margin calculation
                 )
                 self.current_positions[symbol] = position
                 logger.debug(f"Opened new position: {position}")
             else: # Existing position
                 # Check if trade closes or reduces the position
                 if position.quantity * trade.quantity < 0: # Opposite signs mean closing/reducing
                     closed_quantity = min(abs(position.quantity), abs(trade.quantity))
                     realized_pnl = closed_quantity * (trade.price - position.average_entry_price) * np.sign(position.quantity)
                     trade.pnl = realized_pnl # Assign realized PnL to the closing trade part
                     self.current_balance += realized_pnl # Add realized PnL to balance
                     logger.info(f"Realized PnL: {realized_pnl:.2f} from closing {closed_quantity} units of {symbol}")

                     position.quantity += trade.quantity # Update remaining quantity
                     # If position fully closed
                     if abs(position.quantity) < 1e-9: # Use tolerance for float comparison
                         logger.debug(f"Closed position for {symbol}")
                         del self.current_positions[symbol]
                         position = None # Mark as closed
                     else: # Position reduced, average price remains the same
                          position.last_update_timestamp = trade.timestamp
                          logger.debug(f"Reduced position: {position}")

                 else: # Increasing position size
                     # Update average entry price
                     new_total_quantity = position.quantity + trade.quantity
                     new_total_cost = (position.average_entry_price * position.quantity) + (trade.price * trade.quantity)
                     position.average_entry_price = new_total_cost / new_total_quantity
                     position.quantity = new_total_quantity
                     position.last_update_timestamp = trade.timestamp
                     logger.debug(f"Increased position: {position}")


             self.trades.append(trade) # Add trade record *after* potential PnL calculation

         # Re-calculate equity after all trades in the batch are processed
         # This requires the latest market data, which might be tricky if called mid-loop
         # It's safer to rely on the _update_portfolio_equity call at the start of the next timestamp
         # self._update_portfolio_equity( LATEST_MARKET_DATA? ) # Or just update balance and let equity catch up


    def _get_current_portfolio_snapshot(self) -> PortfolioSnapshot:
        """Creates a snapshot of the current portfolio state."""
        # TODO: Implement margin calculations from broker simulator or portfolio service
        margin_used = self.broker_simulator.calculate_margin_used(list(self.current_positions.values()))
        free_margin = self.current_equity - margin_used

        return PortfolioSnapshot(
             timestamp=self.current_timestamp or datetime.now(), # Handle case before loop starts
             balance=self.current_balance,
             equity=self.current_equity,
             positions=list(self.current_positions.values()),
             open_orders=[], # TODO: Track open orders separately if needed by strategy
             margin_used=margin_used,
             free_margin=free_margin
         )

    def _calculate_results(self) -> 'BacktestingResults':
        """Calculates final performance metrics after the backtest."""
        logger.info("Calculating backtest performance metrics...")
        if not self.portfolio_history:
             logger.warning("No portfolio history recorded. Cannot calculate metrics.")
             return BacktestingResults(
                 config=self.config, portfolio_history=[], trades=[], orders=[],
                 metrics=PortfolioPerformanceMetrics(), final_equity=self.config.initial_capital
             )

        equity_curve = pd.Series([snap.equity for snap in self.portfolio_history], index=[snap.timestamp for snap in self.portfolio_history])
        initial_capital = self.config.initial_capital
        final_equity = equity_curve.iloc[-1]

        # --- Basic Metrics ---
        total_return_pct = (final_equity / initial_capital - 1) * 100 if initial_capital else 0
        total_trades = len(self.trades) # Count actual filled trades

        # --- Drawdown ---
        high_water_mark = equity_curve.cummax()
        drawdown = (equity_curve - high_water_mark) / high_water_mark
        max_drawdown_pct = drawdown.min() * 100 if not drawdown.empty else 0

        # --- Returns Analysis (Requires daily/periodic returns) ---
        # Resample equity curve to daily (or other period) to calculate returns
        # Example: Daily returns
        daily_equity = equity_curve.resample('D').last().ffill()
        daily_returns = daily_equity.pct_change().dropna()

        # --- Sharpe Ratio (Example using daily returns, assuming risk-free rate = 0) ---
        sharpe_ratio = 0.0
        if not daily_returns.empty and daily_returns.std() != 0:
            # Annualize Sharpe (assuming daily data -> sqrt(252))
            sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)


        # --- Trade Analysis ---
        win_rate_pct = 0.0
        profit_factor = 0.0
        total_profit = 0.0
        total_loss = 0.0
        winning_trades = 0
        losing_trades = 0

        # Analyze PnL from recorded trades (only those closing/reducing positions have PnL)
        trade_pnls = [t.pnl for t in self.trades if t.pnl is not None and abs(t.pnl) > 1e-9]

        if trade_pnls:
            winning_trades = sum(1 for pnl in trade_pnls if pnl > 0)
            losing_trades = len(trade_pnls) - winning_trades
            total_profit = sum(pnl for pnl in trade_pnls if pnl > 0)
            total_loss = abs(sum(pnl for pnl in trade_pnls if pnl < 0))

            if winning_trades + losing_trades > 0:
                win_rate_pct = (winning_trades / (winning_trades + losing_trades)) * 100

            if total_loss > 0:
                profit_factor = total_profit / total_loss
            elif total_profit > 0:
                profit_factor = np.inf # Only winning trades


        metrics = PortfolioPerformanceMetrics(
             total_return_pct=total_return_pct,
             sharpe_ratio=sharpe_ratio,
             max_drawdown_pct=max_drawdown_pct,
             win_rate_pct=win_rate_pct,
             profit_factor=profit_factor,
             total_trades=total_trades, # Use count of PnL trades or all trades? Decide consistency.
             total_profit=total_profit,
             total_loss=total_loss,
             # TODO: Add other metrics like Sortino, Calmar, avg trade duration, etc.
        )

        logger.info("Metrics calculation complete.")
        return BacktestingResults(
            config=self.config,
            portfolio_history=self.portfolio_history,
            trades=self.trades,
            orders=self.orders,
            metrics=metrics,
            final_equity=final_equity
        )


class BacktestingResults:
    """Holds the results of a backtesting run."""
    def __init__(
        self,
        config: BacktestingConfiguration,
        portfolio_history: List[PortfolioSnapshot],
        trades: List[Trade],
        orders: List[Order],
        metrics: PortfolioPerformanceMetrics,
        final_equity: float
    ):
        self.config = config
        self.portfolio_history = portfolio_history
        self.trades = trades
        self.orders = orders
        self.metrics = metrics
        self.final_equity = final_equity

    def plot_equity_curve(self):
        """Generates a plot of the equity curve (requires matplotlib)."""
        try:
            import matplotlib.pyplot as plt
            timestamps = [snap.timestamp for snap in self.portfolio_history]
            equity = [snap.equity for snap in self.portfolio_history]
            plt.figure(figsize=(12, 6))
            plt.plot(timestamps, equity, label='Equity Curve')
            plt.title(f"Backtest Equity Curve ({self.config.strategy_class.__name__})")
            plt.xlabel("Date")
            plt.ylabel("Portfolio Equity")
            plt.legend()
            plt.grid(True)
            plt.show()
        except ImportError:
            logger.warning("matplotlib not installed. Cannot plot equity curve.")
        except Exception as e:
             logger.error(f"Error plotting equity curve: {e}", exc_info=True)


    def summary(self) -> Dict[str, Any]:
         """Returns a dictionary summary of the backtest results."""
         return {
             "start_date": self.config.start_date.isoformat(),
             "end_date": self.config.end_date.isoformat(),
             "initial_capital": self.config.initial_capital,
             "final_equity": self.final_equity,
             "total_return_pct": self.metrics.total_return_pct,
             "sharpe_ratio": self.metrics.sharpe_ratio,
             "max_drawdown_pct": self.metrics.max_drawdown_pct,
             "win_rate_pct": self.metrics.win_rate_pct,
             "profit_factor": self.metrics.profit_factor,
             "total_trades": len(self.trades),
             # Add other relevant metrics
         }

# Example Usage (Conceptual)
async def run_backtest_example():
    from datetime import datetime
    # Assume RiskParameters, BaseStrategy/BaseRLAgent are defined elsewhere
    # Define placeholder strategy/agent and risk params
    class DummyStrategy: # Placeholder
        def __init__(self, **kwargs): pass
        async def generate_signal(self, **kwargs): return None # No action

    risk_params = RiskParameters(symbol='EURUSD', max_position_size_pct=0.01, stop_loss_pip=50, take_profit_pip=100)

    config = BacktestingConfiguration(
        start_date=datetime(2023, 1, 1),
        end_date=datetime(2023, 12, 31),
        initial_capital=10000.0,
        symbols=['EURUSD'],
        strategy_class=DummyStrategy,
        strategy_config={'param1': 'value1'},
        broker_config={'initial_balance': 10000.0, 'default_spread': 1.5}, # Example broker config
        risk_parameters=risk_params,
        use_rl_risk_optimization=True, # Example: Enable RL risk opt
        rl_risk_optimizer_config={'confidence_threshold_high': 0.75}, # Example optimizer config
        data_source='generated', # Example: Use generated data
        data_parameters={'regime_type': 'trending'} # Example data params
    )

    orchestrator = BacktestingOrchestrator(config)
    try:
        results = await orchestrator.run()
        print("Backtest Summary:")
        print(results.summary())
        # results.plot_equity_curve() # Uncomment to plot if matplotlib is installed
    except Exception as e:
        logger.error(f"Backtest failed: {e}", exc_info=True)

if __name__ == '__main__':
    import asyncio
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger.warning("Running this example directly will likely fail due to placeholders")
    logger.warning("and missing implementations (data provider connection, strategy/agent logic, RL adapter).")
    logger.warning("This script defines the backtesting framework structure.")
    # asyncio.run(run_backtest_example()) # Keep commented out

