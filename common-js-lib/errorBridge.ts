/**
 * Error Bridge Module
 * 
 * This module provides utilities for converting errors between JavaScript/TypeScript and Python.
 * It ensures consistent error handling across language boundaries in the Forex Trading Platform.
 * 
 * Key features:
 * 1. Bidirectional error conversion between JavaScript and Python
 * 2. Standardized error types across languages
 * 3. Consistent error structure and properties
 * 4. Correlation ID propagation
 */

import { 
  ForexTradingPlatformError,
  ConfigurationError,
  DataError,
  DataValidationError,
  DataFetchError,
  DataStorageError,
  DataTransformationError,
  ServiceError,
  ServiceUnavailableError,
  ServiceTimeoutError,
  AuthenticationError,
  AuthorizationError,
  NetworkError,
  TradingError,
  OrderExecutionError,
  AnalysisError,
  MLError
} from './errors';

// Mapping from JavaScript error types to Python exception types
const JS_TO_PYTHON_ERROR_MAPPING: Record<string, string> = {
  'ForexTradingPlatformError': 'ForexTradingPlatformError',
  'ConfigurationError': 'ConfigurationError',
  'DataError': 'DataError',
  'DataValidationError': 'DataValidationError',
  'DataFetchError': 'DataFetchError',
  'DataStorageError': 'DataStorageError',
  'DataTransformationError': 'DataTransformationError',
  'ServiceError': 'ServiceError',
  'ServiceUnavailableError': 'ServiceUnavailableError',
  'ServiceTimeoutError': 'ServiceTimeoutError',
  'AuthenticationError': 'AuthenticationError',
  'AuthorizationError': 'AuthorizationError',
  'NetworkError': 'NetworkError',
  'TradingError': 'TradingError',
  'OrderExecutionError': 'OrderExecutionError',
  'AnalysisError': 'AnalysisError',
  'MLError': 'MLError',
  // Add more mappings as needed
};

// Mapping from Python exception types to JavaScript error classes
const PYTHON_TO_JS_ERROR_MAPPING: Record<string, any> = {
  'ForexTradingPlatformError': ForexTradingPlatformError,
  'ConfigurationError': ConfigurationError,
  'DataError': DataError,
  'DataValidationError': DataValidationError,
  'DataFetchError': DataFetchError,
  'DataStorageError': DataStorageError,
  'DataTransformationError': DataTransformationError,
  'ServiceError': ServiceError,
  'ServiceUnavailableError': ServiceUnavailableError,
  'ServiceTimeoutError': ServiceTimeoutError,
  'AuthenticationError': AuthenticationError,
  'AuthorizationError': AuthorizationError,
  'NetworkError': NetworkError,
  'TradingError': TradingError,
  'OrderExecutionError': OrderExecutionError,
  'AnalysisError': AnalysisError,
  'MLError': MLError,
  // Add more mappings as needed
};

/**
 * Convert a JavaScript error to a Python error format
 * 
 * @param error JavaScript error to convert
 * @param correlationId Optional correlation ID for tracking
 * @returns Object representing the error in Python format
 */
export function convertToPythonError(
  error: Error,
  correlationId?: string
): Record<string, any> {
  // Get error type and details
  let errorType = 'ForexTradingPlatformError';
  let errorCode = 'UNKNOWN_ERROR';
  let message = error.message || 'Unknown error';
  let details: Record<string, any> = {};
  
  if (error instanceof ForexTradingPlatformError) {
    errorType = error.constructor.name;
    errorCode = error.code;
    message = error.message;
    details = error.details || {};
  }
  
  // Map to Python error type
  const pythonErrorType = JS_TO_PYTHON_ERROR_MAPPING[errorType] || 'ForexTradingPlatformError';
  
  // Create error object
  const errorObj = {
    error_type: pythonErrorType,
    error_code: errorCode,
    message: message,
    details: details,
    correlation_id: correlationId,
    timestamp: new Date().toISOString()
  };
  
  return errorObj;
}

/**
 * Convert a Python error to a JavaScript error
 * 
 * @param errorData Object representing the error in Python format
 * @returns JavaScript error
 */
export function convertFromPythonError(errorData: Record<string, any>): Error {
  // Extract error information
  const pythonErrorType = errorData.error_type || 'ForexTradingPlatformError';
  const errorCode = errorData.error_code || 'UNKNOWN_ERROR';
  const message = errorData.message || 'Unknown error';
  const details = errorData.details || {};
  const correlationId = errorData.correlation_id;
  
  // Map to JavaScript error class
  const ErrorClass = PYTHON_TO_JS_ERROR_MAPPING[pythonErrorType] || ForexTradingPlatformError;
  
  // Create error
  const error = new ErrorClass(message, errorCode, details);
  
  // Add correlation ID if available
  if (correlationId) {
    error.details.correlationId = correlationId;
  }
  
  return error;
}

/**
 * Handle an error response from a Python service
 * 
 * @param responseData Response data containing error information
 * @returns JavaScript error
 */
export function handlePythonErrorResponse(responseData: Record<string, any>): Error {
  // Check if response contains error information
  if (responseData.error) {
    return convertFromPythonError(responseData.error);
  }
  
  // Fallback for unexpected response format
  return new ServiceError(
    'Unexpected error response format',
    'UNEXPECTED_ERROR_FORMAT',
    { responseData }
  );
}

/**
 * Create a standardized error response for API endpoints
 * 
 * @param error The error to convert
 * @param correlationId Optional correlation ID for tracking
 * @param includeStack Whether to include stack trace in the response
 * @returns Standardized error response
 */
export function createErrorResponse(
  error: Error,
  correlationId?: string,
  includeStack: boolean = false
): Record<string, any> {
  // Get error type and details
  let errorType = 'ForexTradingPlatformError';
  let errorCode = 'UNKNOWN_ERROR';
  let message = error.message || 'Unknown error';
  let details: Record<string, any> = {};
  
  if (error instanceof ForexTradingPlatformError) {
    errorType = error.constructor.name;
    errorCode = error.code;
    message = error.message;
    details = error.details || {};
  }
  
  // Create error object
  const errorObj = {
    type: errorType,
    code: errorCode,
    message: message,
    details: details,
    correlationId: correlationId,
    timestamp: new Date().toISOString()
  };
  
  // Add stack trace if requested
  if (includeStack && error.stack) {
    errorObj.details.stack = error.stack;
  }
  
  // Create response
  const response = {
    error: errorObj,
    success: false
  };
  
  return response;
}

/**
 * Decorator for handling errors in async functions
 * 
 * @param func Function to decorate
 * @param context Additional context information
 * @param convertToPython Whether to convert errors to Python format
 * @param rethrow Whether to rethrow the error after handling
 * @returns Decorated function
 */
export function withAsyncErrorHandling(
  func: Function,
  context: Record<string, any> = {},
  convertToPython: boolean = false,
  rethrow: boolean = true
): Function {
  return async function(...args: any[]) {
    try {
      return await func(...args);
    } catch (error) {
      // Add function information to context
      const contextWithArgs = {
        ...context,
        function: func.name,
        arguments: JSON.stringify(args.map(arg => 
          typeof arg === 'object' ? '[Object]' : arg
        ))
      };
      
      // Log the error
      console.error(
        `Error in ${func.name}:`,
        error instanceof Error ? error.message : error,
        contextWithArgs
      );
      
      // Convert error if needed
      if (convertToPython) {
        const pythonError = convertToPythonError(
          error instanceof Error ? error : new Error(String(error)),
          contextWithArgs.correlationId
        );
        
        if (rethrow) {
          throw pythonError;
        }
        
        return pythonError;
      }
      
      // Rethrow original error if needed
      if (rethrow) {
        throw error;
      }
      
      return null;
    }
  };
}

/**
 * Decorator for handling errors in sync functions
 * 
 * @param func Function to decorate
 * @param context Additional context information
 * @param convertToPython Whether to convert errors to Python format
 * @param rethrow Whether to rethrow the error after handling
 * @returns Decorated function
 */
export function withErrorHandling(
  func: Function,
  context: Record<string, any> = {},
  convertToPython: boolean = false,
  rethrow: boolean = true
): Function {
  return function(...args: any[]) {
    try {
      return func(...args);
    } catch (error) {
      // Add function information to context
      const contextWithArgs = {
        ...context,
        function: func.name,
        arguments: JSON.stringify(args.map(arg => 
          typeof arg === 'object' ? '[Object]' : arg
        ))
      };
      
      // Log the error
      console.error(
        `Error in ${func.name}:`,
        error instanceof Error ? error.message : error,
        contextWithArgs
      );
      
      // Convert error if needed
      if (convertToPython) {
        const pythonError = convertToPythonError(
          error instanceof Error ? error : new Error(String(error)),
          contextWithArgs.correlationId
        );
        
        if (rethrow) {
          throw pythonError;
        }
        
        return pythonError;
      }
      
      // Rethrow original error if needed
      if (rethrow) {
        throw error;
      }
      
      return null;
    }
  };
}