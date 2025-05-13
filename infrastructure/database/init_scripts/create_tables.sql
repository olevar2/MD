-- Create tables for Forex Trading Platform
-- This script creates the core tables for the platform

-- forex_platform database tables
\c forex_platform

-- Market Data Tables
CREATE TABLE IF NOT EXISTS market_data.ohlcv (
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

CREATE TABLE IF NOT EXISTS market_data.tick_data (
    symbol TEXT NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    bid DOUBLE PRECISION NOT NULL,
    ask DOUBLE PRECISION NOT NULL,
    volume DOUBLE PRECISION,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (symbol, timestamp)
);

CREATE TABLE IF NOT EXISTS market_data.symbols (
    symbol TEXT PRIMARY KEY,
    description TEXT,
    type TEXT NOT NULL,  -- 'forex', 'crypto', 'stock', etc.
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    pip_value DOUBLE PRECISION,
    lot_size INTEGER,
    min_lot DOUBLE PRECISION,
    max_lot DOUBLE PRECISION,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Historical Data Tables
CREATE TABLE IF NOT EXISTS historical_data.ohlcv_data (
    id SERIAL PRIMARY KEY,
    symbol TEXT NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    timeframe TEXT NOT NULL,
    open DOUBLE PRECISION NOT NULL,
    high DOUBLE PRECISION NOT NULL,
    low DOUBLE PRECISION NOT NULL,
    close DOUBLE PRECISION NOT NULL,
    volume DOUBLE PRECISION NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (symbol, timestamp, timeframe)
);

CREATE TABLE IF NOT EXISTS historical_data.tick_data (
    id SERIAL PRIMARY KEY,
    symbol TEXT NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    bid DOUBLE PRECISION NOT NULL,
    ask DOUBLE PRECISION NOT NULL,
    volume DOUBLE PRECISION,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (symbol, timestamp)
);

CREATE TABLE IF NOT EXISTS historical_data.alternative_data (
    id SERIAL PRIMARY KEY,
    source TEXT NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    data_type TEXT NOT NULL,
    data JSONB NOT NULL,
    metadata JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (source, timestamp, data_type)
);

-- Indicators Tables
CREATE TABLE IF NOT EXISTS indicators.indicator_registry (
    id SERIAL PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT,
    category TEXT NOT NULL,
    parameters JSONB NOT NULL DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (name)
);

CREATE TABLE IF NOT EXISTS indicators.indicator_values (
    id SERIAL PRIMARY KEY,
    indicator_id INTEGER NOT NULL REFERENCES indicators.indicator_registry(id),
    symbol TEXT NOT NULL,
    timeframe TEXT NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    value JSONB NOT NULL,
    parameters JSONB NOT NULL DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (indicator_id, symbol, timeframe, timestamp, parameters)
);

-- portfolio database tables
\c portfolio

CREATE TABLE IF NOT EXISTS portfolio.accounts (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT,
    broker TEXT NOT NULL,
    account_type TEXT NOT NULL,
    currency TEXT NOT NULL,
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS portfolio.account_balances (
    id SERIAL PRIMARY KEY,
    account_id TEXT NOT NULL REFERENCES portfolio.accounts(id),
    timestamp TIMESTAMPTZ NOT NULL,
    balance DOUBLE PRECISION NOT NULL,
    equity DOUBLE PRECISION NOT NULL,
    margin DOUBLE PRECISION NOT NULL,
    free_margin DOUBLE PRECISION NOT NULL,
    margin_level DOUBLE PRECISION,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (account_id, timestamp)
);

CREATE TABLE IF NOT EXISTS portfolio.positions (
    id TEXT PRIMARY KEY,
    account_id TEXT NOT NULL REFERENCES portfolio.accounts(id),
    symbol TEXT NOT NULL,
    direction TEXT NOT NULL CHECK (direction IN ('long', 'short')),
    open_time TIMESTAMPTZ NOT NULL,
    close_time TIMESTAMPTZ,
    open_price DOUBLE PRECISION NOT NULL,
    close_price DOUBLE PRECISION,
    volume DOUBLE PRECISION NOT NULL,
    stop_loss DOUBLE PRECISION,
    take_profit DOUBLE PRECISION,
    commission DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    swap DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    profit DOUBLE PRECISION,
    status TEXT NOT NULL CHECK (status IN ('open', 'closed', 'pending')),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- analysis_engine database tables
\c analysis_engine

CREATE TABLE IF NOT EXISTS analysis.market_regimes (
    id SERIAL PRIMARY KEY,
    symbol TEXT NOT NULL,
    timeframe TEXT NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    regime TEXT NOT NULL,
    confidence DOUBLE PRECISION NOT NULL,
    parameters JSONB NOT NULL DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (symbol, timeframe, timestamp, parameters)
);

CREATE TABLE IF NOT EXISTS analysis.support_resistance (
    id SERIAL PRIMARY KEY,
    symbol TEXT NOT NULL,
    timeframe TEXT NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    level_type TEXT NOT NULL CHECK (level_type IN ('support', 'resistance')),
    price DOUBLE PRECISION NOT NULL,
    strength DOUBLE PRECISION NOT NULL,
    parameters JSONB NOT NULL DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS analysis.patterns (
    id SERIAL PRIMARY KEY,
    symbol TEXT NOT NULL,
    timeframe TEXT NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    pattern_type TEXT NOT NULL,
    confidence DOUBLE PRECISION NOT NULL,
    parameters JSONB NOT NULL DEFAULT '{}',
    data JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ml_models database tables
\c ml_models

CREATE TABLE IF NOT EXISTS models.model_registry (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    version TEXT NOT NULL,
    description TEXT,
    model_type TEXT NOT NULL,
    framework TEXT NOT NULL,
    features JSONB NOT NULL,
    target TEXT NOT NULL,
    parameters JSONB NOT NULL DEFAULT '{}',
    metrics JSONB NOT NULL DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (name, version)
);

CREATE TABLE IF NOT EXISTS models.model_artifacts (
    id SERIAL PRIMARY KEY,
    model_id TEXT NOT NULL REFERENCES models.model_registry(id),
    artifact_type TEXT NOT NULL,
    artifact_path TEXT NOT NULL,
    size_bytes BIGINT NOT NULL,
    hash TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (model_id, artifact_type)
);

CREATE TABLE IF NOT EXISTS training.training_jobs (
    id TEXT PRIMARY KEY,
    model_id TEXT NOT NULL REFERENCES models.model_registry(id),
    status TEXT NOT NULL CHECK (status IN ('pending', 'running', 'completed', 'failed')),
    start_time TIMESTAMPTZ,
    end_time TIMESTAMPTZ,
    parameters JSONB NOT NULL DEFAULT '{}',
    metrics JSONB NOT NULL DEFAULT '{}',
    logs TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS evaluation.model_predictions (
    id SERIAL PRIMARY KEY,
    model_id TEXT NOT NULL REFERENCES models.model_registry(id),
    timestamp TIMESTAMPTZ NOT NULL,
    input_data JSONB NOT NULL,
    prediction JSONB NOT NULL,
    actual JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS evaluation.model_performance (
    id SERIAL PRIMARY KEY,
    model_id TEXT NOT NULL REFERENCES models.model_registry(id),
    evaluation_time TIMESTAMPTZ NOT NULL,
    dataset_id TEXT,
    metrics JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
