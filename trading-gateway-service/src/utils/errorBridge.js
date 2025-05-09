/**
 * Error Bridge for Trading Gateway Service
 * 
 * This module provides utilities to bridge error handling between JavaScript and Python
 * components of the Trading Gateway Service, ensuring consistent error handling across
 * the entire service.
 */

const { 
  ForexTradingPlatformError,
  DataValidationError,
  DataFetchError,
  DataStorageError,
  ServiceError,
  ServiceUnavailableError,
  ServiceTimeoutError,
  TradingError,
  OrderExecutionError,
  AuthenticationError,
  AuthorizationError
} = require('./errors');

const logger = require('./logger');

/**
 * Convert a Python error (received as JSON) to a JavaScript error
 * 
 * @param {Object|string} pythonError - Python error as JSON object or string
 * @returns {Error} JavaScript error object
 */
function convertPythonError(pythonError) {
  let errorData;
  
  if (typeof pythonError === 'string') {
    try {
      errorData = JSON.parse(pythonError);
    } catch (e) {
      // If it's not valid JSON, treat it as a simple error message
      return new ServiceError(pythonError);
    }
  } else {
    errorData = pythonError;
  }
  
  // Extract error information
  const errorType = errorData.error_type || 'ForexTradingPlatformError';
  const errorCode = errorData.error_code || 'UNKNOWN_ERROR';
  const message = errorData.message || 'Unknown error';
  const details = errorData.details || {};
  
  // Map Python error types to JavaScript error classes
  const errorMap = {
    'ForexTradingPlatformError': ForexTradingPlatformError,
    'ConfigurationError': ForexTradingPlatformError, // Use base error as fallback
    'DataError': DataValidationError, // Use closest match
    'DataValidationError': DataValidationError,
    'DataFetchError': DataFetchError,
    'DataStorageError': DataStorageError,
    'DataTransformationError': DataValidationError, // Use closest match
    'ServiceError': ServiceError,
    'ServiceUnavailableError': ServiceUnavailableError,
    'ServiceTimeoutError': ServiceTimeoutError,
    'TradingError': TradingError,
    'OrderExecutionError': OrderExecutionError,
    'AuthenticationError': AuthenticationError,
    'AuthorizationError': AuthorizationError,
    'BrokerConnectionError': ServiceUnavailableError, // Map to closest JS error
    'OrderValidationError': DataValidationError, // Map to closest JS error
    'MarketDataError': DataFetchError // Map to closest JS error
  };
  
  // Get the appropriate error class
  const ErrorClass = errorMap[errorType] || ForexTradingPlatformError;
  
  // Create and return the error
  return new ErrorClass(message, errorCode, details);
}

/**
 * Convert a JavaScript error to a format compatible with Python
 * 
 * @param {Error} error - JavaScript error
 * @returns {Object} Error in a format compatible with Python
 */
function convertToPythonError(error) {
  if (error instanceof ForexTradingPlatformError) {
    return {
      error_type: error.constructor.name,
      error_code: error.errorCode,
      message: error.message,
      details: error.details
    };
  } else {
    // Convert standard JavaScript error
    return {
      error_type: 'ServiceError',
      error_code: 'JS_EXCEPTION',
      message: error.message,
      details: {
        original_error: error.constructor.name,
        stack: error.stack
      }
    };
  }
}

/**
 * Handle an error with consistent logging and optional conversion
 * 
 * @param {Error} error - The error to handle
 * @param {Object} context - Additional context information
 * @param {boolean} convertToPython - Whether to convert to Python-compatible format
 * @returns {Object|Error} Handled error
 */
function handleError(error, context = {}, convertToPython = false) {
  // Default values
  let errorType = 'UnknownError';
  let errorCode = 'UNKNOWN_ERROR';
  let message = error.message || 'An unknown error occurred';
  let details = {};
  
  // Extract information based on error type
  if (error instanceof ForexTradingPlatformError) {
    errorType = error.constructor.name;
    errorCode = error.errorCode;
    message = error.message;
    details = error.details;
  } else if (typeof error === 'object' && error.error_type) {
    // It might be a Python error already in object form
    errorType = error.error_type;
    errorCode = error.error_code;
    message = error.message;
    details = error.details;
  }
  
  // Log the error with appropriate level based on severity
  if (errorType.includes('Error')) {
    logger.error(`${errorType}: ${message}`, {
      errorType,
      errorCode,
      details,
      ...context,
      stack: error.stack
    });
  } else {
    logger.warn(`${errorType}: ${message}`, {
      errorType,
      errorCode,
      details,
      ...context
    });
  }
  
  // Convert to Python format if requested
  if (convertToPython) {
    return convertToPythonError(error);
  }
  
  return error;
}

/**
 * Wrap a function with error handling
 * 
 * @param {Function} func - Function to wrap
 * @param {Object} options - Options for error handling
 * @returns {Function} Wrapped function
 */
function withErrorHandling(func, options = {}) {
  const { 
    context = {}, 
    convertToPython = false,
    rethrow = true,
    defaultErrorClass = ServiceError
  } = options;
  
  return function(...args) {
    try {
      return func(...args);
    } catch (error) {
      const contextWithArgs = {
        ...context,
        function: func.name,
        arguments: JSON.stringify(args.map(arg => 
          typeof arg === 'object' ? '[Object]' : arg
        ))
      };
      
      const handledError = handleError(error, contextWithArgs, convertToPython);
      
      if (rethrow) {
        if (error instanceof ForexTradingPlatformError) {
          throw error;
        } else {
          throw new defaultErrorClass(
            error.message || 'An error occurred',
            error.constructor.name,
            { originalError: error.constructor.name, stack: error.stack }
          );
        }
      }
      
      return handledError;
    }
  };
}

/**
 * Wrap an async function with error handling
 * 
 * @param {Function} func - Async function to wrap
 * @param {Object} options - Options for error handling
 * @returns {Function} Wrapped async function
 */
function withAsyncErrorHandling(func, options = {}) {
  const { 
    context = {}, 
    convertToPython = false,
    rethrow = true,
    defaultErrorClass = ServiceError
  } = options;
  
  return async function(...args) {
    try {
      return await func(...args);
    } catch (error) {
      const contextWithArgs = {
        ...context,
        function: func.name,
        arguments: JSON.stringify(args.map(arg => 
          typeof arg === 'object' ? '[Object]' : arg
        ))
      };
      
      const handledError = handleError(error, contextWithArgs, convertToPython);
      
      if (rethrow) {
        if (error instanceof ForexTradingPlatformError) {
          throw error;
        } else {
          throw new defaultErrorClass(
            error.message || 'An error occurred',
            error.constructor.name,
            { originalError: error.constructor.name, stack: error.stack }
          );
        }
      }
      
      return handledError;
    }
  };
}

module.exports = {
  convertPythonError,
  convertToPythonError,
  handleError,
  withErrorHandling,
  withAsyncErrorHandling
};
