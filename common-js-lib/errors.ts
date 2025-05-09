/**
 * Errors Module
 * 
 * This module defines standardized errors for the Forex Trading Platform.
 * These errors are used across all JavaScript/TypeScript services to ensure consistent error handling.
 * 
 * Key features:
 * 1. Hierarchical error structure
 * 2. Standardized error codes
 * 3. Detailed error information
 * 4. Support for correlation IDs
 */

/**
 * Base error for all Forex Trading Platform errors
 * 
 * All platform-specific errors should inherit from this class.
 */
export class ForexTradingPlatformError extends Error {
  /**
   * Error code for programmatic handling
   */
  code: string;
  
  /**
   * Additional error details
   */
  details: Record<string, any>;
  
  /**
   * Initialize the error
   * 
   * @param message Human-readable error message
   * @param code Error code for programmatic handling
   * @param details Additional error details
   */
  constructor(
    message: string,
    code: string = 'UNKNOWN_ERROR',
    details: Record<string, any> = {}
  ) {
    super(message);
    this.name = this.constructor.name;
    this.code = code;
    this.details = details;
    
    // Ensure stack trace is captured
    if (Error.captureStackTrace) {
      Error.captureStackTrace(this, this.constructor);
    }
  }
  
  /**
   * Get string representation of the error
   * 
   * @returns String representation
   */
  toString(): string {
    return `${this.name}[${this.code}]: ${this.message}`;
  }
}

// Configuration Errors

/**
 * Error raised for configuration errors
 */
export class ConfigurationError extends ForexTradingPlatformError {
  /**
   * Initialize the error
   * 
   * @param message Human-readable error message
   * @param code Error code for programmatic handling
   * @param details Additional error details
   */
  constructor(
    message: string,
    code: string = 'CONFIGURATION_ERROR',
    details: Record<string, any> = {}
  ) {
    super(message, code, details);
  }
}

// Data Errors

/**
 * Base error for data-related errors
 */
export class DataError extends ForexTradingPlatformError {
  /**
   * Initialize the error
   * 
   * @param message Human-readable error message
   * @param code Error code for programmatic handling
   * @param details Additional error details
   */
  constructor(
    message: string,
    code: string = 'DATA_ERROR',
    details: Record<string, any> = {}
  ) {
    super(message, code, details);
  }
}

/**
 * Error raised for data validation errors
 */
export class DataValidationError extends DataError {
  /**
   * Initialize the error
   * 
   * @param message Human-readable error message
   * @param code Error code for programmatic handling
   * @param details Additional error details
   */
  constructor(
    message: string,
    code: string = 'DATA_VALIDATION_ERROR',
    details: Record<string, any> = {}
  ) {
    super(message, code, details);
  }
}

/**
 * Error raised for errors when fetching data
 */
export class DataFetchError extends DataError {
  /**
   * Initialize the error
   * 
   * @param message Human-readable error message
   * @param code Error code for programmatic handling
   * @param details Additional error details
   */
  constructor(
    message: string,
    code: string = 'DATA_FETCH_ERROR',
    details: Record<string, any> = {}
  ) {
    super(message, code, details);
  }
}

/**
 * Error raised for errors when storing data
 */
export class DataStorageError extends DataError {
  /**
   * Initialize the error
   * 
   * @param message Human-readable error message
   * @param code Error code for programmatic handling
   * @param details Additional error details
   */
  constructor(
    message: string,
    code: string = 'DATA_STORAGE_ERROR',
    details: Record<string, any> = {}
  ) {
    super(message, code, details);
  }
}

/**
 * Error raised for errors when transforming data
 */
export class DataTransformationError extends DataError {
  /**
   * Initialize the error
   * 
   * @param message Human-readable error message
   * @param code Error code for programmatic handling
   * @param details Additional error details
   */
  constructor(
    message: string,
    code: string = 'DATA_TRANSFORMATION_ERROR',
    details: Record<string, any> = {}
  ) {
    super(message, code, details);
  }
}

// Service Errors

/**
 * Base error for service-related errors
 */
export class ServiceError extends ForexTradingPlatformError {
  /**
   * Initialize the error
   * 
   * @param message Human-readable error message
   * @param code Error code for programmatic handling
   * @param details Additional error details
   */
  constructor(
    message: string,
    code: string = 'SERVICE_ERROR',
    details: Record<string, any> = {}
  ) {
    super(message, code, details);
  }
}

/**
 * Error raised when a service is unavailable
 */
export class ServiceUnavailableError extends ServiceError {
  /**
   * Initialize the error
   * 
   * @param message Human-readable error message
   * @param code Error code for programmatic handling
   * @param details Additional error details
   */
  constructor(
    message: string,
    code: string = 'SERVICE_UNAVAILABLE',
    details: Record<string, any> = {}
  ) {
    super(message, code, details);
  }
}

/**
 * Error raised when a service request times out
 */
export class ServiceTimeoutError extends ServiceError {
  /**
   * Initialize the error
   * 
   * @param message Human-readable error message
   * @param code Error code for programmatic handling
   * @param details Additional error details
   */
  constructor(
    message: string,
    code: string = 'SERVICE_TIMEOUT',
    details: Record<string, any> = {}
  ) {
    super(message, code, details);
  }
}

// Authentication/Authorization Errors

/**
 * Error raised for authentication errors
 */
export class AuthenticationError extends ForexTradingPlatformError {
  /**
   * Initialize the error
   * 
   * @param message Human-readable error message
   * @param code Error code for programmatic handling
   * @param details Additional error details
   */
  constructor(
    message: string,
    code: string = 'AUTHENTICATION_ERROR',
    details: Record<string, any> = {}
  ) {
    super(message, code, details);
  }
}

/**
 * Error raised for authorization errors
 */
export class AuthorizationError extends ForexTradingPlatformError {
  /**
   * Initialize the error
   * 
   * @param message Human-readable error message
   * @param code Error code for programmatic handling
   * @param details Additional error details
   */
  constructor(
    message: string,
    code: string = 'AUTHORIZATION_ERROR',
    details: Record<string, any> = {}
  ) {
    super(message, code, details);
  }
}

// Network Errors

/**
 * Error raised for network-related errors
 */
export class NetworkError extends ForexTradingPlatformError {
  /**
   * Initialize the error
   * 
   * @param message Human-readable error message
   * @param code Error code for programmatic handling
   * @param details Additional error details
   */
  constructor(
    message: string,
    code: string = 'NETWORK_ERROR',
    details: Record<string, any> = {}
  ) {
    super(message, code, details);
  }
}

// Trading Errors

/**
 * Base error for trading-related errors
 */
export class TradingError extends ForexTradingPlatformError {
  /**
   * Initialize the error
   * 
   * @param message Human-readable error message
   * @param code Error code for programmatic handling
   * @param details Additional error details
   */
  constructor(
    message: string,
    code: string = 'TRADING_ERROR',
    details: Record<string, any> = {}
  ) {
    super(message, code, details);
  }
}

/**
 * Error raised for order execution errors
 */
export class OrderExecutionError extends TradingError {
  /**
   * Initialize the error
   * 
   * @param message Human-readable error message
   * @param code Error code for programmatic handling
   * @param details Additional error details
   */
  constructor(
    message: string,
    code: string = 'ORDER_EXECUTION_ERROR',
    details: Record<string, any> = {}
  ) {
    super(message, code, details);
  }
}

// Analysis Errors

/**
 * Base error for analysis-related errors
 */
export class AnalysisError extends ForexTradingPlatformError {
  /**
   * Initialize the error
   * 
   * @param message Human-readable error message
   * @param code Error code for programmatic handling
   * @param details Additional error details
   */
  constructor(
    message: string,
    code: string = 'ANALYSIS_ERROR',
    details: Record<string, any> = {}
  ) {
    super(message, code, details);
  }
}

// ML Errors

/**
 * Base error for ML-related errors
 */
export class MLError extends ForexTradingPlatformError {
  /**
   * Initialize the error
   * 
   * @param message Human-readable error message
   * @param code Error code for programmatic handling
   * @param details Additional error details
   */
  constructor(
    message: string,
    code: string = 'ML_ERROR',
    details: Record<string, any> = {}
  ) {
    super(message, code, details);
  }
}