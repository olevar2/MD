/**
 * Custom error classes for the Trading Gateway Service.
 * 
 * These error classes correspond to the common-lib exceptions used in Python services,
 * providing a consistent error handling approach across the platform.
 */

/**
 * Base error class for all platform errors
 */
class ForexTradingPlatformError extends Error {
  constructor(message, errorCode, details = {}) {
    super(message || 'An error occurred in the Forex Trading Platform');
    this.name = this.constructor.name;
    this.errorType = this.constructor.name;
    this.errorCode = errorCode || 'FOREX_PLATFORM_ERROR';
    this.details = details;
    Error.captureStackTrace(this, this.constructor);
  }
  
  toJSON() {
    return {
      error_type: this.errorType,
      error_code: this.errorCode,
      message: this.message,
      details: this.details
    };
  }
}

/**
 * Error for data validation failures
 */
class DataValidationError extends ForexTradingPlatformError {
  constructor(message, data = null) {
    super(message || 'Data validation error', 'DATA_VALIDATION_ERROR', { data });
  }
}

/**
 * Error for data fetching failures
 */
class DataFetchError extends ForexTradingPlatformError {
  constructor(message, source = null, details = {}) {
    super(
      message || `Failed to fetch data${source ? ` from ${source}` : ''}`,
      'DATA_FETCH_ERROR',
      { source, ...details }
    );
  }
}

/**
 * Error for data storage failures
 */
class DataStorageError extends ForexTradingPlatformError {
  constructor(message, target = null, details = {}) {
    super(
      message || `Failed to store data${target ? ` to ${target}` : ''}`,
      'DATA_STORAGE_ERROR',
      { target, ...details }
    );
  }
}

/**
 * Error for service-related failures
 */
class ServiceError extends ForexTradingPlatformError {
  constructor(message, serviceName = null, details = {}) {
    super(
      message || `Error in service${serviceName ? `: ${serviceName}` : ''}`,
      'SERVICE_ERROR',
      { service_name: serviceName, ...details }
    );
  }
}

/**
 * Error for service unavailability
 */
class ServiceUnavailableError extends ServiceError {
  constructor(serviceName, message = null, details = {}) {
    super(
      message || `Service unavailable: ${serviceName || 'unknown'}`,
      'SERVICE_UNAVAILABLE_ERROR',
      { service_name: serviceName, ...details }
    );
  }
}

/**
 * Error for service timeouts
 */
class ServiceTimeoutError extends ServiceError {
  constructor(serviceName, timeout = null, message = null, details = {}) {
    super(
      message || `Service timeout${serviceName ? ` for ${serviceName}` : ''}${timeout ? ` after ${timeout}ms` : ''}`,
      'SERVICE_TIMEOUT_ERROR',
      { service_name: serviceName, timeout, ...details }
    );
  }
}

/**
 * Error for trading-related failures
 */
class TradingError extends ForexTradingPlatformError {
  constructor(message, details = {}) {
    super(message || 'Trading error', 'TRADING_ERROR', details);
  }
}

/**
 * Error for order execution failures
 */
class OrderExecutionError extends TradingError {
  constructor(message, orderId = null, details = {}) {
    super(
      message || `Failed to execute order${orderId ? ` ${orderId}` : ''}`,
      'ORDER_EXECUTION_ERROR',
      { order_id: orderId, ...details }
    );
  }
}

/**
 * Error for authentication failures
 */
class AuthenticationError extends ForexTradingPlatformError {
  constructor(message, details = {}) {
    super(message || 'Authentication failed', 'AUTHENTICATION_ERROR', details);
  }
}

/**
 * Error for authorization failures
 */
class AuthorizationError extends ForexTradingPlatformError {
  constructor(message, resource = null, action = null, details = {}) {
    let errorMessage = message;
    if (!errorMessage && resource && action) {
      errorMessage = `Not authorized to ${action} on ${resource}`;
    } else if (!errorMessage) {
      errorMessage = 'Authorization failed';
    }
    
    super(errorMessage, 'AUTHORIZATION_ERROR', { resource, action, ...details });
  }
}

module.exports = {
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
};
