-- TimeScaleDB Schema for Forex Trading Platform
-- Version: 0.1.0
-- Date: 2025-04-04

-- Enable TimescaleDB extension
CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;

-- Create schema for market data
CREATE SCHEMA IF NOT EXISTS market_data;

-- --------------------
-- Market Instruments
-- --------------------
CREATE TABLE market_data.instruments (
    symbol TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    type TEXT NOT NULL,  -- 'forex', 'stock', 'index', 'commodity', 'crypto'
    pip_size DOUBLE PRECISION NOT NULL,
    min_lot_size DOUBLE PRECISION NOT NULL,
    max_lot_size DOUBLE PRECISION NOT NULL,
    lot_step DOUBLE PRECISION NOT NULL,
    commission DOUBLE PRECISION,
    swap_long DOUBLE PRECISION,
    swap_short DOUBLE PRECISION,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Trading hours as a separate table with relationship to instruments
CREATE TABLE market_data.trading_hours (
    id SERIAL PRIMARY KEY,
    symbol TEXT NOT NULL REFERENCES market_data.instruments(symbol) ON DELETE CASCADE,
    day_of_week INTEGER NOT NULL CHECK (day_of_week BETWEEN 0 AND 6),  -- 0 = Sunday, 6 = Saturday
    open_time TIME NOT NULL,
    close_time TIME NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (symbol, day_of_week)
);

-- --------------------
-- OHLCV Data
-- --------------------
CREATE TABLE market_data.ohlcv (
    symbol TEXT NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    timeframe TEXT NOT NULL,  -- '1m', '5m', '15m', '30m', '1h', '4h', '1d', '1w', '1M'
    open DOUBLE PRECISION NOT NULL,
    high DOUBLE PRECISION NOT NULL,
    low DOUBLE PRECISION NOT NULL,
    close DOUBLE PRECISION NOT NULL,
    volume DOUBLE PRECISION NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (symbol, timestamp, timeframe)
);

-- Convert to TimescaleDB hypertable
SELECT create_hypertable('market_data.ohlcv', 'timestamp');

-- Index for more efficient queries
CREATE INDEX idx_ohlcv_symbol_timeframe ON market_data.ohlcv (symbol, timeframe, timestamp DESC);

-- --------------------
-- Tick Data
-- --------------------
CREATE TABLE market_data.tick_data (
    symbol TEXT NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    bid DOUBLE PRECISION NOT NULL,
    ask DOUBLE PRECISION NOT NULL,
    volume DOUBLE PRECISION,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (symbol, timestamp)
);

-- Convert to TimescaleDB hypertable
SELECT create_hypertable('market_data.tick_data', 'timestamp');

-- Index for more efficient queries
CREATE INDEX idx_tick_data_symbol ON market_data.tick_data (symbol, timestamp DESC);

-- --------------------
-- Technical Indicators
-- --------------------
CREATE SCHEMA IF NOT EXISTS feature_store;

CREATE TABLE feature_store.technical_indicators (
    symbol TEXT NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    timeframe TEXT NOT NULL,
    indicator_name TEXT NOT NULL,
    indicator_values JSONB NOT NULL,  -- Allows flexible indicator storage
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (symbol, timestamp, timeframe, indicator_name)
);

-- Convert to TimescaleDB hypertable
SELECT create_hypertable('feature_store.technical_indicators', 'timestamp');

-- Index for more efficient queries
CREATE INDEX idx_indicators_symbol_timeframe_name ON feature_store.technical_indicators (symbol, timeframe, indicator_name, timestamp DESC);

-- --------------------
-- Trading Records
-- --------------------
CREATE SCHEMA IF NOT EXISTS trading;

-- Orders table
CREATE TABLE trading.orders (
    order_id TEXT PRIMARY KEY,
    client_order_id TEXT,
    symbol TEXT NOT NULL,
    side TEXT NOT NULL,  -- 'buy', 'sell'
    order_type TEXT NOT NULL,  -- 'market', 'limit', 'stop', 'stop_limit'
    quantity DOUBLE PRECISION NOT NULL,
    price DOUBLE PRECISION,  -- NULL for market orders
    stop_price DOUBLE PRECISION,  -- NULL for non-stop orders
    take_profit DOUBLE PRECISION,
    stop_loss DOUBLE PRECISION,
    time_in_force TEXT DEFAULT 'GTC',
    status TEXT NOT NULL,  -- 'pending', 'open', 'filled', 'partially_filled', 'canceled', 'rejected', 'expired'
    filled_quantity DOUBLE PRECISION DEFAULT 0.0,
    average_fill_price DOUBLE PRECISION,
    commission DOUBLE PRECISION,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Index for order queries
CREATE INDEX idx_orders_symbol_status ON trading.orders (symbol, status);
CREATE INDEX idx_orders_created_at ON trading.orders (created_at DESC);

-- Trades table
CREATE TABLE trading.trades (
    trade_id TEXT PRIMARY KEY,
    order_id TEXT NOT NULL REFERENCES trading.orders(order_id),
    symbol TEXT NOT NULL,
    side TEXT NOT NULL,  -- 'buy', 'sell'
    quantity DOUBLE PRECISION NOT NULL,
    price DOUBLE PRECISION NOT NULL,
    commission DOUBLE PRECISION NOT NULL,
    realized_pnl DOUBLE PRECISION,
    trade_time TIMESTAMPTZ NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Index for trade queries
CREATE INDEX idx_trades_order_id ON trading.trades (order_id);
CREATE INDEX idx_trades_symbol_time ON trading.trades (symbol, trade_time DESC);

-- Positions table
CREATE TABLE trading.positions (
    position_id TEXT PRIMARY KEY,
    symbol TEXT NOT NULL,
    side TEXT NOT NULL,  -- 'long', 'short'
    quantity DOUBLE PRECISION NOT NULL,
    entry_price DOUBLE PRECISION NOT NULL,
    current_price DOUBLE PRECISION NOT NULL,
    unrealized_pnl DOUBLE PRECISION NOT NULL,
    realized_pnl DOUBLE PRECISION NOT NULL,
    take_profit DOUBLE PRECISION,
    stop_loss DOUBLE PRECISION,
    open_time TIMESTAMPTZ NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Index for position queries
CREATE INDEX idx_positions_symbol ON trading.positions (symbol);
CREATE INDEX idx_positions_open_time ON trading.positions (open_time DESC);

-- --------------------
-- Portfolio Records
-- --------------------
CREATE SCHEMA IF NOT EXISTS portfolio;

-- Account balances table
CREATE TABLE portfolio.account_balances (
    id SERIAL PRIMARY KEY,
    account_id TEXT NOT NULL,
    balance DOUBLE PRECISION NOT NULL,
    equity DOUBLE PRECISION NOT NULL,
    margin DOUBLE PRECISION NOT NULL,
    free_margin DOUBLE PRECISION NOT NULL,
    margin_level DOUBLE PRECISION,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Convert to TimescaleDB hypertable for time-series analysis
SELECT create_hypertable('portfolio.account_balances', 'timestamp');

-- Daily performance table
CREATE TABLE portfolio.daily_performance (
    id SERIAL PRIMARY KEY,
    account_id TEXT NOT NULL,
    date DATE NOT NULL,
    starting_balance DOUBLE PRECISION NOT NULL,
    ending_balance DOUBLE PRECISION NOT NULL,
    deposits DOUBLE PRECISION DEFAULT 0.0,
    withdrawals DOUBLE PRECISION DEFAULT 0.0,
    realized_pnl DOUBLE PRECISION NOT NULL,
    fees DOUBLE PRECISION NOT NULL,
    swaps DOUBLE PRECISION NOT NULL,
    trades_count INTEGER NOT NULL,
    win_count INTEGER NOT NULL,
    loss_count INTEGER NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (account_id, date)
);

-- --------------------
-- Risk Management
-- --------------------
CREATE SCHEMA IF NOT EXISTS risk;

-- Risk metrics table
CREATE TABLE risk.position_risk_metrics (
    id SERIAL PRIMARY KEY,
    position_id TEXT NOT NULL REFERENCES trading.positions(position_id) ON DELETE CASCADE,
    value_at_risk DOUBLE PRECISION NOT NULL,
    expected_shortfall DOUBLE PRECISION NOT NULL,
    max_drawdown DOUBLE PRECISION NOT NULL,
    sharpe_ratio DOUBLE PRECISION,
    sortino_ratio DOUBLE PRECISION,
    win_rate DOUBLE PRECISION,
    profit_factor DOUBLE PRECISION,
    risk_reward_ratio DOUBLE PRECISION,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Convert to TimescaleDB hypertable
SELECT create_hypertable('risk.position_risk_metrics', 'timestamp');

-- Risk limits table
CREATE TABLE risk.risk_limits (
    id SERIAL PRIMARY KEY,
    limit_type TEXT NOT NULL,  -- 'max_position', 'max_drawdown', 'daily_loss', etc.
    symbol TEXT,  -- NULL for account-wide limits
    value DOUBLE PRECISION NOT NULL,
    active BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (limit_type, symbol)
);

-- --------------------
-- Audit Logging
-- --------------------
CREATE SCHEMA IF NOT EXISTS audit;

-- System audit log
CREATE TABLE audit.system_logs (
    id SERIAL PRIMARY KEY,
    service TEXT NOT NULL,
    event_type TEXT NOT NULL,
    severity TEXT NOT NULL,  -- 'info', 'warning', 'error', 'critical'
    message TEXT NOT NULL,
    details JSONB,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Convert to TimescaleDB hypertable
SELECT create_hypertable('audit.system_logs', 'timestamp');

-- Index for log queries
CREATE INDEX idx_logs_service_severity_time ON audit.system_logs (service, severity, timestamp DESC);

-- Trading audit log - more specific to trading operations
CREATE TABLE audit.trading_logs (
    id SERIAL PRIMARY KEY,
    operation TEXT NOT NULL, -- 'order_create', 'order_modify', 'order_cancel', 'trade_execute', etc.
    entity_id TEXT NOT NULL, -- order_id, trade_id, etc.
    entity_type TEXT NOT NULL, -- 'order', 'trade', 'position', etc.
    user_id TEXT,
    details JSONB NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Convert to TimescaleDB hypertable
SELECT create_hypertable('audit.trading_logs', 'timestamp');

-- Index for trading audit queries
CREATE INDEX idx_trading_logs_operation_time ON audit.trading_logs (operation, timestamp DESC);
CREATE INDEX idx_trading_logs_entity ON audit.trading_logs (entity_type, entity_id);

-- --------------------
-- Continous Aggregates
-- --------------------

-- Example continous aggregate for OHLCV data
-- This will create a materialized view that is automatically updated
-- with daily OHLCV data based on 1-minute data
CREATE MATERIALIZED VIEW market_data.ohlcv_daily
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 day', timestamp) AS bucket,
    symbol,
    first(open, timestamp) AS open,
    max(high) AS high,
    min(low) AS low,
    last(close, timestamp) AS close,
    sum(volume) AS volume
FROM market_data.ohlcv
WHERE timeframe = '1m'
GROUP BY bucket, symbol;

-- Refresh policy to update the continuous aggregate
SELECT add_continuous_aggregate_policy('market_data.ohlcv_daily',
    start_offset => INTERVAL '3 days',
    end_offset => INTERVAL '1 hour',
    schedule_interval => INTERVAL '1 day');

-- -----------------------------
-- Default Retention Policies
-- -----------------------------

-- Set default retention policy for tick data (e.g., keep 90 days)
-- This can be adjusted based on storage capacity and requirements
-- SELECT add_retention_policy('market_data.tick_data', INTERVAL '90 days');

-- Optional: Set different retention policies for different tables
-- SELECT add_retention_policy('market_data.ohlcv', INTERVAL '5 years');
-- SELECT add_retention_policy('audit.system_logs', INTERVAL '1 year');

-- Comment out retention policies initially to avoid data deletion until confirmed