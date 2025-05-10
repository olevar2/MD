/**
 * Common JavaScript Library for the Forex Trading Platform
 *
 * This library provides standardized utilities and functionality shared across
 * all JavaScript components in the Forex trading platform.
 */

// Import modules
const security = require('./security');
const errors = require('./errors');
const errorHandler = require('./errorHandler');
const resilience = require('./resilience');
const apiClient = require('./apiClient');

// Export all modules
module.exports = {
  // Security module
  security,

  // Error classes
  errors,

  // Error handling utilities
  errorHandler,

  // Resilience patterns
  resilience,

  // API client
  apiClient,

  // Convenience exports
  ApiClient: apiClient.ApiClient,
  defaultApiClient: apiClient.apiClient
};
