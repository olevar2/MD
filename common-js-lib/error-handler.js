/**
 * Error Handler Utility for the Forex Trading Platform
 * 
 * This module provides utilities for handling errors in a standardized way across
 * all JavaScript components. It includes functions for logging errors, formatting
 * error messages, and handling errors in a consistent manner.
 */

const { 
  ForexTradingPlatformError,
  DataValidationError,
  NetworkError,
  ServiceUnavailableError
} = require('./errors');

/**
 * Generate a correlation ID for tracking errors
 * 
 * @returns {string} A unique correlation ID
 */
function generateCorrelationId() {
  return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {
    const r = Math.random() * 16 | 0;
    const v = c === 'x' ? r : (r & 0x3 | 0x8);
    return v.toString(16);
  });
}

/**
 * Get correlation ID from request headers or generate a new one
 * 
 * @param {Object} req - Express request object
 * @returns {string} Correlation ID
 */
function getCorrelationId(req) {
  if (req && req.headers && req.headers['x-correlation-id']) {
    return req.headers['x-correlation-id'];
  }
  return generateCorrelationId();
}

/**
 * Format error for display to the user
 * 
 * @param {Error} error - The error to format
 * @returns {string} A user-friendly error message
 */
function formatErrorMessage(error) {
  // Handle platform-specific errors
  if (error instanceof ForexTradingPlatformError) {
    return error.message;
  }
  
  // Handle Axios errors
  if (error.isAxiosError) {
    // Handle API error responses
    if (error.response && error.response.data) {
      if (error.response.data.message) {
        return error.response.data.message;
      }
      
      if (typeof error.response.data === 'string') {
        return error.response.data;
      }
    }
    
    // Handle network errors
    if (error.code === 'ECONNABORTED') {
      return 'The request timed out. Please try again.';
    }
    
    if (error.code === 'ERR_NETWORK') {
      return 'Network error. Please check your internet connection.';
    }
    
    return error.message || 'An error occurred while communicating with the server';
  }
  
  // Handle standard errors
  if (error instanceof Error) {
    return error.message;
  }
  
  // Handle string errors
  if (typeof error === 'string') {
    return error;
  }
  
  // Handle unknown errors
  return 'An unexpected error occurred';
}

/**
 * Map error to a platform-specific error
 * 
 * @param {Error} error - The error to map
 * @returns {Error} A platform-specific error
 */
function mapError(error) {
  // If it's already a platform error, return it
  if (error instanceof ForexTradingPlatformError) {
    return error;
  }
  
  // Handle Axios errors
  if (error.isAxiosError) {
    // Handle API error responses
    if (error.response) {
      const { status, data } = error.response;
      
      // If the response contains a platform error, convert it
      if (data && data.error_type) {
        return convertFromApiError(data);
      }
      
      // Map based on status code
      switch (status) {
        case 400:
          return new DataValidationError(
            data?.message || 'Invalid request data',
            data,
            { status, url: error.config?.url }
          );
        case 401:
          return new AuthenticationError(
            data?.message || 'Authentication failed',
            { status, url: error.config?.url }
          );
        case 403:
          return new AuthorizationError(
            data?.message || 'Not authorized',
            null,
            null,
            { status, url: error.config?.url }
          );
        case 404:
          return new DataFetchError(
            data?.message || 'Resource not found',
            error.config?.url,
            { status }
          );
        case 500:
          return new ServiceError(
            data?.message || 'Server error',
            null,
            { status, url: error.config?.url }
          );
        case 503:
          return new ServiceUnavailableError(
            null,
            { status, url: error.config?.url, message: data?.message }
          );
        case 504:
          return new ServiceTimeoutError(
            null,
            null,
            { status, url: error.config?.url, message: data?.message }
          );
        default:
          return new ForexTradingPlatformError(
            data?.message || `HTTP error ${status}`,
            'HTTP_ERROR',
            { status, url: error.config?.url }
          );
      }
    }
    
    // Handle network errors
    if (error.code === 'ECONNABORTED') {
      return new ServiceTimeoutError(
        null,
        error.config?.timeout,
        { url: error.config?.url }
      );
    }
    
    if (error.code === 'ERR_NETWORK') {
      return new NetworkError(
        'Network error. Please check your internet connection.',
        error.config?.url
      );
    }
    
    return new NetworkError(
      error.message || 'Network error',
      error.config?.url
    );
  }
  
  // Return the original error wrapped in a platform error
  return new ForexTradingPlatformError(
    error.message || 'An unexpected error occurred',
    'UNKNOWN_ERROR',
    { originalError: error.constructor.name }
  );
}

/**
 * Convert API error response to a platform-specific error
 * 
 * @param {Object} errorData - Error data from API response
 * @returns {Error} A platform-specific error
 */
function convertFromApiError(errorData) {
  // Import all error classes
  const errorClasses = require('./errors');
  
  // Get the error type from the response
  const errorType = errorData.error_type;
  
  // If the error type matches a platform error, create an instance of it
  if (errorType && errorClasses[errorType]) {
    const ErrorClass = errorClasses[errorType];
    return new ErrorClass(
      errorData.message || 'An error occurred',
      errorData.details || {}
    );
  }
  
  // If no matching error type, create a generic platform error
  return new ForexTradingPlatformError(
    errorData.message || 'An error occurred',
    errorData.error_code || 'UNKNOWN_ERROR',
    errorData.details || {}
  );
}

/**
 * Log error to console and optionally to a monitoring service
 * 
 * @param {Error} error - The error to log
 * @param {Object} context - Additional context information
 */
function logError(error, context = {}) {
  // Generate a correlation ID if not provided
  const correlationId = context.correlationId || generateCorrelationId();
  
  // Get error details
  const errorType = error instanceof ForexTradingPlatformError ? 
    error.constructor.name : 
    error.name || 'UnknownError';
  
  const errorCode = error instanceof ForexTradingPlatformError ? 
    error.errorCode : 
    'UNKNOWN_ERROR';
  
  const details = error instanceof ForexTradingPlatformError ? 
    error.details : 
    {};
  
  // Prepare context with correlation ID
  const enrichedContext = {
    ...context,
    correlationId,
    errorType,
    errorCode,
    timestamp: new Date().toISOString(),
    environment: process.env.NODE_ENV || 'development'
  };
  
  // Log to console
  console.error(
    `[${enrichedContext.timestamp}] [${errorType}] [${correlationId}]`,
    error.message,
    enrichedContext
  );
  
  // Log to error monitoring service in production
  if (process.env.NODE_ENV === 'production') {
    // This would be implemented with a service like Sentry
    // For now, just log to console
    console.info(`[MONITORING] Would send error to monitoring service: ${errorType} - ${error.message}`);
  }
  
  // Store in session storage for debugging if in browser environment
  if (typeof sessionStorage !== 'undefined') {
    try {
      // Get existing errors from session storage
      const storedErrorsJson = sessionStorage.getItem('forex_platform_errors');
      const storedErrors = storedErrorsJson ? JSON.parse(storedErrorsJson) : [];
      
      // Add new error
      storedErrors.unshift({
        message: error.message,
        errorType,
        errorCode,
        timestamp: enrichedContext.timestamp,
        stack: error.stack,
        context: enrichedContext
      });
      
      // Keep only the last 10 errors
      const limitedErrors = storedErrors.slice(0, 10);
      
      // Store back in session storage
      sessionStorage.setItem('forex_platform_errors', JSON.stringify(limitedErrors));
    } catch (e) {
      console.error('Failed to store error in session storage:', e);
    }
  }
  
  return correlationId;
}

/**
 * Handle error in a standardized way
 * 
 * @param {Error} error - The error to handle
 * @param {Object} context - Additional context information
 * @returns {string} Correlation ID
 */
function handleError(error, context = {}) {
  // Map error to a platform-specific error
  const mappedError = error instanceof ForexTradingPlatformError ? 
    error : 
    mapError(error);
  
  // Log the error
  const correlationId = logError(mappedError, context);
  
  // Return the correlation ID
  return correlationId;
}

/**
 * Create a custom error with a specific error type
 * 
 * @param {string} message - The error message
 * @param {string} errorType - The type of error
 * @param {Object} details - Additional error details
 * @returns {Error} A custom error
 */
function createError(message, errorType, details = {}) {
  // Import all error classes
  const errorClasses = require('./errors');
  
  // If the error type matches a platform error, create an instance of it
  if (errorType && errorClasses[errorType]) {
    const ErrorClass = errorClasses[errorType];
    return new ErrorClass(message, details);
  }
  
  // If no matching error type, create a generic platform error
  return new ForexTradingPlatformError(
    message,
    'UNKNOWN_ERROR',
    details
  );
}

/**
 * Higher-order function that wraps a function with error handling
 * 
 * @param {Function} fn - The function to wrap
 * @param {Object} context - Additional context information
 * @returns {Function} The wrapped function
 */
function withErrorHandling(fn, context = {}) {
  return function(...args) {
    try {
      return fn(...args);
    } catch (error) {
      // Add function information to context
      const enrichedContext = {
        ...context,
        function: fn.name || 'anonymous',
        arguments: args.map(arg => typeof arg === 'object' ? '[Object]' : String(arg))
      };
      
      // Handle the error
      handleError(error, enrichedContext);
      
      // Re-throw the error
      throw error;
    }
  };
}

/**
 * Higher-order function that wraps an async function with error handling
 * 
 * @param {Function} fn - The async function to wrap
 * @param {Object} context - Additional context information
 * @returns {Function} The wrapped async function
 */
function withAsyncErrorHandling(fn, context = {}) {
  return async function(...args) {
    try {
      return await fn(...args);
    } catch (error) {
      // Add function information to context
      const enrichedContext = {
        ...context,
        function: fn.name || 'anonymous',
        arguments: args.map(arg => typeof arg === 'object' ? '[Object]' : String(arg))
      };
      
      // Handle the error
      handleError(error, enrichedContext);
      
      // Re-throw the error
      throw error;
    }
  };
}

/**
 * Express middleware for handling errors
 * 
 * @param {Error} err - The error to handle
 * @param {Object} req - Express request object
 * @param {Object} res - Express response object
 * @param {Function} next - Express next function
 */
function errorMiddleware(err, req, res, next) {
  // Get correlation ID from request or generate a new one
  const correlationId = getCorrelationId(req);
  
  // Create context from request
  const context = {
    correlationId,
    path: req.path,
    method: req.method,
    ip: req.ip,
    userAgent: req.headers['user-agent']
  };
  
  // Map error to a platform-specific error
  const mappedError = err instanceof ForexTradingPlatformError ? 
    err : 
    mapError(err);
  
  // Log the error
  logError(mappedError, context);
  
  // Determine status code
  let statusCode = 500;
  
  if (mappedError instanceof DataValidationError) {
    statusCode = 400;
  } else if (mappedError instanceof AuthenticationError) {
    statusCode = 401;
  } else if (mappedError instanceof AuthorizationError) {
    statusCode = 403;
  } else if (mappedError instanceof ServiceUnavailableError) {
    statusCode = 503;
  } else if (mappedError instanceof ServiceTimeoutError) {
    statusCode = 504;
  }
  
  // Send response
  res.status(statusCode).json({
    error_type: mappedError.constructor.name,
    error_code: mappedError.errorCode,
    message: mappedError.message,
    correlation_id: correlationId,
    timestamp: new Date().toISOString(),
    ...(process.env.NODE_ENV === 'development' ? { details: mappedError.details } : {})
  });
}

/**
 * Get all errors stored in session storage
 * 
 * @returns {Array} Array of stored errors
 */
function getStoredErrors() {
  if (typeof sessionStorage === 'undefined') return [];
  
  try {
    const storedErrorsJson = sessionStorage.getItem('forex_platform_errors');
    return storedErrorsJson ? JSON.parse(storedErrorsJson) : [];
  } catch (e) {
    console.error('Failed to retrieve errors from session storage:', e);
    return [];
  }
}

/**
 * Clear all errors stored in session storage
 */
function clearStoredErrors() {
  if (typeof sessionStorage === 'undefined') return;
  
  try {
    sessionStorage.removeItem('forex_platform_errors');
  } catch (e) {
    console.error('Failed to clear errors from session storage:', e);
  }
}

module.exports = {
  // Error handling functions
  formatErrorMessage,
  mapError,
  handleError,
  logError,
  createError,
  withErrorHandling,
  withAsyncErrorHandling,
  errorMiddleware,
  
  // Correlation ID functions
  generateCorrelationId,
  getCorrelationId,
  
  // Session storage functions
  getStoredErrors,
  clearStoredErrors,
  
  // API error conversion
  convertFromApiError
};
