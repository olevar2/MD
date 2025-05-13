-- Create schemas for Forex Trading Platform
-- This script creates all the required schemas for the platform

-- Enable TimescaleDB extension
CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;

-- forex_platform database schemas
\c forex_platform

CREATE SCHEMA IF NOT EXISTS market_data;
CREATE SCHEMA IF NOT EXISTS historical_data;
CREATE SCHEMA IF NOT EXISTS indicators;

-- data_pipeline database schemas
\c data_pipeline

CREATE SCHEMA IF NOT EXISTS pipeline;
CREATE SCHEMA IF NOT EXISTS staging;
CREATE SCHEMA IF NOT EXISTS processed;

-- feature_store database schemas
\c feature_store

CREATE SCHEMA IF NOT EXISTS features;
CREATE SCHEMA IF NOT EXISTS metadata;
CREATE SCHEMA IF NOT EXISTS registry;

-- portfolio database schemas
\c portfolio

CREATE SCHEMA IF NOT EXISTS portfolio;
CREATE SCHEMA IF NOT EXISTS accounts;
CREATE SCHEMA IF NOT EXISTS performance;

-- analysis_engine database schemas
\c analysis_engine

CREATE SCHEMA IF NOT EXISTS analysis;
CREATE SCHEMA IF NOT EXISTS models;
CREATE SCHEMA IF NOT EXISTS results;

-- ml_models database schemas
\c ml_models

CREATE SCHEMA IF NOT EXISTS models;
CREATE SCHEMA IF NOT EXISTS training;
CREATE SCHEMA IF NOT EXISTS evaluation;
