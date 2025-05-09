/**
 * Error handling middleware for the Trading Gateway Service.
 *
 * This middleware provides standardized error handling for the Express application,
 * mapping common-lib exception types to appropriate HTTP status codes and response formats.
 * It uses the errorBridge to handle errors consistently between JavaScript and Python components.
 */

const {
  convertPythonError,
  handleError
} = require('../utils/errorBridge');
const {
  ForexTradingPlatformError
} = require('../utils/errors');

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
  // Create context for error handling
  const context = {
    path: req.path,
    method: req.method,
    headers: req.headers,
    query: req.query,
    body: req.method !== 'GET' ? req.body : undefined
  };

  // Check if this is a Python error (from API response)
  if (err.response && err.response.data && err.response.data.error_type) {
    // Convert Python error to JavaScript error
    err = convertPythonError(err.response.data);
  }

  // Use the error bridge to handle the error (but don't rethrow)
  handleError(err, context, false);

  // Default values
  let statusCode = 500;
  let errorType = 'InternalServerError';
  let message = 'An unexpected error occurred';
  let details = process.env.NODE_ENV === 'development' ? err.stack : undefined;

  // Check if this is a ForexTradingPlatformError
  if (err instanceof ForexTradingPlatformError) {
    errorType = err.constructor.name;
    statusCode = ERROR_STATUS_CODES[errorType] || 500;
    message = err.message;
    details = err.details;
  }
  // Check if this is a known error type by errorType property
  else if (err.errorType && ERROR_STATUS_CODES[err.errorType]) {
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

  // Send standardized error response
  res.status(statusCode).json({
    error_type: errorType,
    message: message,
    details: details,
    timestamp: new Date().toISOString()
  });
}

module.exports = errorHandler;
