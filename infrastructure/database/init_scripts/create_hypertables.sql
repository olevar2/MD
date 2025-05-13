-- Create TimescaleDB hypertables for time-series data
-- This script converts appropriate tables to TimescaleDB hypertables

-- forex_platform database hypertables
\c forex_platform

-- Convert market_data.ohlcv to hypertable
SELECT create_hypertable('market_data.ohlcv', 'timestamp', if_not_exists => TRUE);

-- Convert market_data.tick_data to hypertable
SELECT create_hypertable('market_data.tick_data', 'timestamp', if_not_exists => TRUE);

-- Convert historical_data.ohlcv_data to hypertable
SELECT create_hypertable('historical_data.ohlcv_data', 'timestamp', if_not_exists => TRUE);

-- Convert historical_data.tick_data to hypertable
SELECT create_hypertable('historical_data.tick_data', 'timestamp', if_not_exists => TRUE);

-- Convert historical_data.alternative_data to hypertable
SELECT create_hypertable('historical_data.alternative_data', 'timestamp', if_not_exists => TRUE);

-- Convert indicators.indicator_values to hypertable
SELECT create_hypertable('indicators.indicator_values', 'timestamp', if_not_exists => TRUE);

-- portfolio database hypertables
\c portfolio

-- Convert portfolio.account_balances to hypertable
SELECT create_hypertable('portfolio.account_balances', 'timestamp', if_not_exists => TRUE);

-- analysis_engine database hypertables
\c analysis_engine

-- Convert analysis.market_regimes to hypertable
SELECT create_hypertable('analysis.market_regimes', 'timestamp', if_not_exists => TRUE);

-- Convert analysis.support_resistance to hypertable
SELECT create_hypertable('analysis.support_resistance', 'timestamp', if_not_exists => TRUE);

-- Convert analysis.patterns to hypertable
SELECT create_hypertable('analysis.patterns', 'timestamp', if_not_exists => TRUE);

-- ml_models database hypertables
\c ml_models

-- Convert evaluation.model_predictions to hypertable
SELECT create_hypertable('evaluation.model_predictions', 'timestamp', if_not_exists => TRUE);
