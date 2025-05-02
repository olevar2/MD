/**
 * Authentication middleware for Trading Gateway Service.
 * Re-exports and configures security utilities from common-js-lib.
 */

const { apiKeyAuthMiddleware, jwtAuthMiddleware } = require('common-js-lib').security;

// Check if environment variables are set
if (!process.env.JWT_SECRET) {
  console.warn('WARNING: JWT_SECRET environment variable not set. Authentication will fail.');
}

// API Keys loaded from environment variables
const VALID_API_KEYS = {
  'analysis-engine-service': process.env.ANALYSIS_ENGINE_API_KEY,
  'portfolio-management-service': process.env.PORTFOLIO_API_KEY,
  'feature-store-service': process.env.FEATURE_STORE_API_KEY,
  'risk-management-service': process.env.RISK_MANAGEMENT_API_KEY
};

// Log warning for any missing API keys
Object.entries(VALID_API_KEYS).forEach(([service, key]) => {
  if (!key) {
    console.warn(`WARNING: API key for ${service} not set. Authentication for this service will fail.`);
  }
});

// JWT settings from environment variables
const JWT_SECRET = process.env.JWT_SECRET;

// Export pre-configured middleware functions
module.exports = {
  apiKeyAuth: apiKeyAuthMiddleware(VALID_API_KEYS),
  jwtAuth: (requiredScopes = []) => jwtAuthMiddleware(JWT_SECRET, requiredScopes),
  // Re-export the original functions for direct use
  rawApiKeyAuth: apiKeyAuthMiddleware,
  rawJwtAuth: jwtAuthMiddleware
};
