/**
 * Error Classes for the Forex Trading Platform
 * 
 * This module defines a comprehensive standardized error hierarchy to be used across
 * all JavaScript components. These error classes mirror the Python exceptions in common-lib
 * to ensure consistent error handling across the platform.
 */

/**
 * Base error class for all platform errors
 */
class ForexTradingPlatformError extends Error {
  /**
   * Create a new ForexTradingPlatformError
   * 
   * @param {string} message - Error message
   * @param {string} errorCode - Error code for categorization
   * @param {Object} details - Additional error details
   */
  constructor(message = 'An error occurred in the Forex Trading Platform', errorCode = 'FOREX_PLATFORM_ERROR', details = {}) {
    super(message);
    this.name = this.constructor.name;
    this.errorCode = errorCode;
    this.details = details;
    this.timestamp = new Date().toISOString();
    
    // Capture stack trace
    if (Error.captureStackTrace) {
      Error.captureStackTrace(this, this.constructor);
    }
  }

  /**
   * Convert the error to a plain object for JSON serialization
   * 
   * @returns {Object} Error as a plain object
   */
  toJSON() {
    return {
      error_type: this.name,
      error_code: this.errorCode,
      message: this.message,
      details: this.details,
      timestamp: this.timestamp
    };
  }
}

/**
 * Configuration error
 */
class ConfigurationError extends ForexTradingPlatformError {
  /**
   * Create a new ConfigurationError
   * 
   * @param {string} message - Error message
   * @param {Object} details - Additional error details
   */
  constructor(message = 'Configuration error', details = {}) {
    super(message, 'CONFIG_ERROR', details);
  }
}

/**
 * Configuration not found error
 */
class ConfigNotFoundError extends ConfigurationError {
  /**
   * Create a new ConfigNotFoundError
   * 
   * @param {string} configName - Name of the configuration that was not found
   * @param {Object} details - Additional error details
   */
  constructor(configName, details = {}) {
    const message = configName ? `Configuration not found: ${configName}` : 'Configuration not found';
    super(message, { configName, ...details });
    this.errorCode = 'CONFIG_NOT_FOUND';
  }
}

/**
 * Configuration validation error
 */
class ConfigValidationError extends ConfigurationError {
  /**
   * Create a new ConfigValidationError
   * 
   * @param {Object} errors - Validation errors
   * @param {Object} details - Additional error details
   */
  constructor(errors = {}, details = {}) {
    super('Configuration validation failed', { validation_errors: errors, ...details });
    this.errorCode = 'CONFIG_VALIDATION_ERROR';
  }
}

/**
 * Base class for data errors
 */
class DataError extends ForexTradingPlatformError {
  /**
   * Create a new DataError
   * 
   * @param {string} message - Error message
   * @param {Object} details - Additional error details
   */
  constructor(message = 'Data error', details = {}) {
    super(message, 'DATA_ERROR', details);
  }
}

/**
 * Data validation error
 */
class DataValidationError extends DataError {
  /**
   * Create a new DataValidationError
   * 
   * @param {string} message - Error message
   * @param {*} data - The data that failed validation
   * @param {Object} details - Additional error details
   */
  constructor(message = 'Data validation failed', data = null, details = {}) {
    super(message, { data, ...details });
    this.errorCode = 'DATA_VALIDATION_ERROR';
    this.data = data;
  }
}

/**
 * Data fetch error
 */
class DataFetchError extends DataError {
  /**
   * Create a new DataFetchError
   * 
   * @param {string} message - Error message
   * @param {string} source - Source of the data
   * @param {Object} details - Additional error details
   */
  constructor(message = 'Failed to fetch data', source = null, details = {}) {
    super(message, { source, ...details });
    this.errorCode = 'DATA_FETCH_ERROR';
    this.source = source;
  }
}

/**
 * Data storage error
 */
class DataStorageError extends DataError {
  /**
   * Create a new DataStorageError
   * 
   * @param {string} message - Error message
   * @param {string} storageType - Type of storage
   * @param {Object} details - Additional error details
   */
  constructor(message = 'Failed to store data', storageType = null, details = {}) {
    const formattedMessage = storageType ? `Failed to store data in ${storageType}` : message;
    super(formattedMessage, { storage_type: storageType, ...details });
    this.errorCode = 'DATA_STORAGE_ERROR';
    this.storageType = storageType;
  }
}

/**
 * Data transformation error
 */
class DataTransformationError extends DataError {
  /**
   * Create a new DataTransformationError
   * 
   * @param {string} message - Error message
   * @param {string} transformationType - Type of transformation
   * @param {Object} details - Additional error details
   */
  constructor(message = 'Failed to transform data', transformationType = null, details = {}) {
    const formattedMessage = transformationType ? `Failed to transform data: ${transformationType}` : message;
    super(formattedMessage, { transformation_type: transformationType, ...details });
    this.errorCode = 'DATA_TRANSFORMATION_ERROR';
    this.transformationType = transformationType;
  }
}

/**
 * Base class for service errors
 */
class ServiceError extends ForexTradingPlatformError {
  /**
   * Create a new ServiceError
   * 
   * @param {string} message - Error message
   * @param {string} serviceName - Name of the service
   * @param {Object} details - Additional error details
   */
  constructor(message = 'Service error', serviceName = null, details = {}) {
    const formattedMessage = serviceName ? `Error in service: ${serviceName}` : message;
    super(formattedMessage, 'SERVICE_ERROR', { service_name: serviceName, ...details });
    this.serviceName = serviceName;
  }
}

/**
 * Service unavailable error
 */
class ServiceUnavailableError extends ServiceError {
  /**
   * Create a new ServiceUnavailableError
   * 
   * @param {string} serviceName - Name of the service
   * @param {Object} details - Additional error details
   */
  constructor(serviceName = null, details = {}) {
    const message = serviceName ? `Service unavailable: ${serviceName}` : 'Service unavailable';
    super(message, serviceName, details);
    this.errorCode = 'SERVICE_UNAVAILABLE_ERROR';
  }
}

/**
 * Service timeout error
 */
class ServiceTimeoutError extends ServiceError {
  /**
   * Create a new ServiceTimeoutError
   * 
   * @param {string} serviceName - Name of the service
   * @param {number} timeoutMs - Timeout in milliseconds
   * @param {Object} details - Additional error details
   */
  constructor(serviceName = null, timeoutMs = null, details = {}) {
    let message = 'Service timeout';
    if (serviceName) message += `: ${serviceName}`;
    if (timeoutMs) message += ` (${timeoutMs}ms)`;
    
    super(message, serviceName, { timeout_ms: timeoutMs, ...details });
    this.errorCode = 'SERVICE_TIMEOUT_ERROR';
    this.timeoutMs = timeoutMs;
  }
}

/**
 * Authentication error
 */
class AuthenticationError extends ForexTradingPlatformError {
  /**
   * Create a new AuthenticationError
   * 
   * @param {string} message - Error message
   * @param {Object} details - Additional error details
   */
  constructor(message = 'Authentication failed', details = {}) {
    super(message, 'AUTHENTICATION_ERROR', details);
  }
}

/**
 * Authorization error
 */
class AuthorizationError extends ForexTradingPlatformError {
  /**
   * Create a new AuthorizationError
   * 
   * @param {string} message - Error message
   * @param {string} resource - Resource being accessed
   * @param {string} action - Action being performed
   * @param {Object} details - Additional error details
   */
  constructor(message = null, resource = null, action = null, details = {}) {
    let formattedMessage = message;
    if (!formattedMessage && resource && action) {
      formattedMessage = `Not authorized to ${action} on ${resource}`;
    } else if (!formattedMessage) {
      formattedMessage = 'Authorization failed';
    }
    
    super(formattedMessage, 'AUTHORIZATION_ERROR', { resource, action, ...details });
    this.resource = resource;
    this.action = action;
  }
}

/**
 * Base class for trading errors
 */
class TradingError extends ForexTradingPlatformError {
  /**
   * Create a new TradingError
   * 
   * @param {string} message - Error message
   * @param {Object} details - Additional error details
   */
  constructor(message = 'Trading error', details = {}) {
    super(message, 'TRADING_ERROR', details);
  }
}

/**
 * Order execution error
 */
class OrderExecutionError extends TradingError {
  /**
   * Create a new OrderExecutionError
   * 
   * @param {string} message - Error message
   * @param {string} orderId - ID of the order
   * @param {Object} details - Additional error details
   */
  constructor(message = null, orderId = null, details = {}) {
    const formattedMessage = message || `Failed to execute order ${orderId || 'unknown'}`;
    super(formattedMessage, { order_id: orderId, ...details });
    this.errorCode = 'ORDER_EXECUTION_ERROR';
    this.orderId = orderId;
  }
}

/**
 * UI error
 */
class UIError extends ForexTradingPlatformError {
  /**
   * Create a new UIError
   * 
   * @param {string} message - Error message
   * @param {string} component - UI component where the error occurred
   * @param {Object} details - Additional error details
   */
  constructor(message = 'UI error', component = null, details = {}) {
    super(message, 'UI_ERROR', { component, ...details });
    this.component = component;
  }
}

/**
 * Rendering error
 */
class RenderingError extends UIError {
  /**
   * Create a new RenderingError
   * 
   * @param {string} message - Error message
   * @param {string} component - UI component where the error occurred
   * @param {Object} details - Additional error details
   */
  constructor(message = 'Rendering error', component = null, details = {}) {
    super(message, component, details);
    this.errorCode = 'RENDERING_ERROR';
  }
}

/**
 * Network error
 */
class NetworkError extends ForexTradingPlatformError {
  /**
   * Create a new NetworkError
   * 
   * @param {string} message - Error message
   * @param {string} url - URL that was being accessed
   * @param {Object} details - Additional error details
   */
  constructor(message = 'Network error', url = null, details = {}) {
    super(message, 'NETWORK_ERROR', { url, ...details });
    this.url = url;
  }
}

/**
 * Circuit breaker open error
 */
class CircuitBreakerOpenError extends ServiceUnavailableError {
  /**
   * Create a new CircuitBreakerOpenError
   * 
   * @param {string} serviceName - Name of the service
   * @param {Object} details - Additional error details
   */
  constructor(serviceName = null, details = {}) {
    const message = serviceName ? 
      `Circuit breaker open for service: ${serviceName}` : 
      'Circuit breaker open';
    
    super(serviceName, { ...details, circuit_breaker: true });
    this.message = message;
    this.errorCode = 'CIRCUIT_BREAKER_OPEN';
  }
}

/**
 * Retry exhausted error
 */
class RetryExhaustedError extends ServiceError {
  /**
   * Create a new RetryExhaustedError
   * 
   * @param {string} message - Error message
   * @param {number} attempts - Number of attempts made
   * @param {Object} details - Additional error details
   */
  constructor(message = 'Retry attempts exhausted', attempts = null, details = {}) {
    super(message, null, { attempts, ...details });
    this.errorCode = 'RETRY_EXHAUSTED';
    this.attempts = attempts;
  }
}

/**
 * Bulkhead full error
 */
class BulkheadFullError extends ServiceUnavailableError {
  /**
   * Create a new BulkheadFullError
   * 
   * @param {string} serviceName - Name of the service
   * @param {Object} details - Additional error details
   */
  constructor(serviceName = null, details = {}) {
    const message = serviceName ? 
      `Bulkhead full for service: ${serviceName}` : 
      'Bulkhead full';
    
    super(serviceName, { ...details, bulkhead: true });
    this.message = message;
    this.errorCode = 'BULKHEAD_FULL';
  }
}

module.exports = {
  // Base error
  ForexTradingPlatformError,
  
  // Configuration errors
  ConfigurationError,
  ConfigNotFoundError,
  ConfigValidationError,
  
  // Data errors
  DataError,
  DataValidationError,
  DataFetchError,
  DataStorageError,
  DataTransformationError,
  
  // Service errors
  ServiceError,
  ServiceUnavailableError,
  ServiceTimeoutError,
  
  // Authentication/Authorization errors
  AuthenticationError,
  AuthorizationError,
  
  // Trading errors
  TradingError,
  OrderExecutionError,
  
  // UI errors
  UIError,
  RenderingError,
  
  // Network errors
  NetworkError,
  
  // Resilience errors
  CircuitBreakerOpenError,
  RetryExhaustedError,
  BulkheadFullError
};
