/**
 * Error handling middleware for the Trading Gateway Service.
 * 
 * This middleware provides standardized error handling for the Express application,
 * mapping common-lib exception types to appropriate HTTP status codes and response formats.
 */

const logger = require('../utils/logger');

// Error types that correspond to common-lib exceptions
const ERROR_TYPES = {
  // Base error
  FOREX_PLATFORM_ERROR: 'ForexTradingPlatformError',
  
  // Data errors
  DATA_VALIDATION_ERROR: 'DataValidationError',
  DATA_FETCH_ERROR: 'DataFetchError',
  DATA_STORAGE_ERROR: 'DataStorageError',
  DATA_TRANSFORMATION_ERROR: 'DataTransformationError',
  
  // Service errors
  SERVICE_ERROR: 'ServiceError',
  SERVICE_UNAVAILABLE_ERROR: 'ServiceUnavailableError',
  SERVICE_TIMEOUT_ERROR: 'ServiceTimeoutError',
  
  // Trading errors
  TRADING_ERROR: 'TradingError',
  ORDER_EXECUTION_ERROR: 'OrderExecutionError',
  
  // Authentication/Authorization errors
  AUTHENTICATION_ERROR: 'AuthenticationError',
  AUTHORIZATION_ERROR: 'AuthorizationError',
  
  // Configuration errors
  CONFIGURATION_ERROR: 'ConfigurationError'
};

// Map error types to HTTP status codes
const ERROR_STATUS_CODES = {
  [ERROR_TYPES.FOREX_PLATFORM_ERROR]: 500,
  [ERROR_TYPES.DATA_VALIDATION_ERROR]: 400,
  [ERROR_TYPES.DATA_FETCH_ERROR]: 500,
  [ERROR_TYPES.DATA_STORAGE_ERROR]: 500,
  [ERROR_TYPES.DATA_TRANSFORMATION_ERROR]: 500,
  [ERROR_TYPES.SERVICE_ERROR]: 500,
  [ERROR_TYPES.SERVICE_UNAVAILABLE_ERROR]: 503,
  [ERROR_TYPES.SERVICE_TIMEOUT_ERROR]: 504,
  [ERROR_TYPES.TRADING_ERROR]: 400,
  [ERROR_TYPES.ORDER_EXECUTION_ERROR]: 400,
  [ERROR_TYPES.AUTHENTICATION_ERROR]: 401,
  [ERROR_TYPES.AUTHORIZATION_ERROR]: 403,
  [ERROR_TYPES.CONFIGURATION_ERROR]: 500
};

/**
 * Error handling middleware for Express
 */
function errorHandler(err, req, res, next) {
  // Default values
  let statusCode = 500;
  let errorType = 'InternalServerError';
  let message = 'An unexpected error occurred';
  let details = process.env.NODE_ENV === 'development' ? err.stack : undefined;
  
  // Check if this is a known error type
  if (err.errorType && ERROR_STATUS_CODES[err.errorType]) {
    statusCode = ERROR_STATUS_CODES[err.errorType];
    errorType = err.errorType;
    message = err.message || `${errorType} occurred`;
    details = err.details;
  } 
  // Handle validation errors from express-validator
  else if (err.array && typeof err.array === 'function') {
    statusCode = 400;
    errorType = ERROR_TYPES.DATA_VALIDATION_ERROR;
    message = 'Validation error';
    details = err.array();
  }
  // Handle other common error cases
  else if (err.statusCode) {
    statusCode = err.statusCode;
    message = err.message;
  }
  
  // Log the error with appropriate level based on severity
  if (statusCode >= 500) {
    logger.error(`${errorType}: ${message}`, {
      path: req.path,
      method: req.method,
      errorType,
      details,
      stack: err.stack
    });
  } else {
    logger.warn(`${errorType}: ${message}`, {
      path: req.path,
      method: req.method,
      errorType,
      details
    });
  }
  
  // Send standardized error response
  res.status(statusCode).json({
    error_type: errorType,
    message: message,
    details: details,
    timestamp: new Date().toISOString()
  });
}

module.exports = errorHandler;
