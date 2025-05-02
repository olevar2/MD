/**
 * Error handling utilities for the UI Service.
 * 
 * This module provides centralized error handling functions for the UI Service,
 * including error logging, formatting, and notification.
 */

import { AxiosError } from 'axios';

// Error types that correspond to common-lib exceptions
export enum ErrorType {
  // Base error
  FOREX_PLATFORM_ERROR = 'ForexTradingPlatformError',
  
  // Data errors
  DATA_VALIDATION_ERROR = 'DataValidationError',
  DATA_FETCH_ERROR = 'DataFetchError',
  DATA_STORAGE_ERROR = 'DataStorageError',
  DATA_TRANSFORMATION_ERROR = 'DataTransformationError',
  
  // Service errors
  SERVICE_ERROR = 'ServiceError',
  SERVICE_UNAVAILABLE_ERROR = 'ServiceUnavailableError',
  SERVICE_TIMEOUT_ERROR = 'ServiceTimeoutError',
  
  // Trading errors
  TRADING_ERROR = 'TradingError',
  ORDER_EXECUTION_ERROR = 'OrderExecutionError',
  
  // Authentication/Authorization errors
  AUTHENTICATION_ERROR = 'AuthenticationError',
  AUTHORIZATION_ERROR = 'AuthorizationError',
  
  // Configuration errors
  CONFIGURATION_ERROR = 'ConfigurationError',
  
  // Network errors
  NETWORK_ERROR = 'NetworkError',
  
  // Unknown errors
  UNKNOWN_ERROR = 'UnknownError'
}

// Interface for standardized error response
export interface ErrorResponse {
  error_type: string;
  message: string;
  details?: any;
  timestamp?: string;
}

/**
 * Format error for display to the user
 * 
 * @param error The error to format
 * @returns A user-friendly error message
 */
export function formatErrorMessage(error: any): string {
  // Handle Axios errors
  if (isAxiosError(error)) {
    const axiosError = error as AxiosError<ErrorResponse>;
    
    // Handle API error responses
    if (axiosError.response?.data) {
      const errorData = axiosError.response.data as ErrorResponse;
      return errorData.message || 'An error occurred while communicating with the server';
    }
    
    // Handle network errors
    if (axiosError.code === 'ECONNABORTED') {
      return 'The request timed out. Please try again.';
    }
    
    if (axiosError.code === 'ERR_NETWORK') {
      return 'Network error. Please check your internet connection.';
    }
    
    return axiosError.message || 'An error occurred while communicating with the server';
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
 * Get error type from error object
 * 
 * @param error The error to analyze
 * @returns The error type
 */
export function getErrorType(error: any): ErrorType {
  // Handle Axios errors
  if (isAxiosError(error)) {
    const axiosError = error as AxiosError<ErrorResponse>;
    
    // Handle API error responses
    if (axiosError.response?.data && (axiosError.response.data as ErrorResponse).error_type) {
      return ((axiosError.response.data as ErrorResponse).error_type as ErrorType) || ErrorType.UNKNOWN_ERROR;
    }
    
    // Handle network errors
    if (axiosError.code === 'ECONNABORTED') {
      return ErrorType.SERVICE_TIMEOUT_ERROR;
    }
    
    if (axiosError.code === 'ERR_NETWORK') {
      return ErrorType.NETWORK_ERROR;
    }
    
    // Handle HTTP status codes
    if (axiosError.response) {
      switch (axiosError.response.status) {
        case 400:
          return ErrorType.DATA_VALIDATION_ERROR;
        case 401:
          return ErrorType.AUTHENTICATION_ERROR;
        case 403:
          return ErrorType.AUTHORIZATION_ERROR;
        case 404:
          return ErrorType.DATA_FETCH_ERROR;
        case 500:
          return ErrorType.FOREX_PLATFORM_ERROR;
        case 503:
          return ErrorType.SERVICE_UNAVAILABLE_ERROR;
        case 504:
          return ErrorType.SERVICE_TIMEOUT_ERROR;
        default:
          return ErrorType.UNKNOWN_ERROR;
      }
    }
    
    return ErrorType.NETWORK_ERROR;
  }
  
  // Handle standard errors
  if (error instanceof Error) {
    return ErrorType.UNKNOWN_ERROR;
  }
  
  // Handle unknown errors
  return ErrorType.UNKNOWN_ERROR;
}

/**
 * Log error to console and optionally to a logging service
 * 
 * @param error The error to log
 * @param context Additional context information
 */
export function logError(error: any, context?: Record<string, any>): void {
  const errorType = getErrorType(error);
  const timestamp = new Date().toISOString();
  
  // Log to console
  console.error(`[${timestamp}] [${errorType}]`, error, context || {});
  
  // TODO: Add logging to a service like Sentry or LogRocket
  // if (process.env.NODE_ENV === 'production') {
  //   // Log to external service
  // }
}

/**
 * Check if an error is an Axios error
 * 
 * @param error The error to check
 * @returns True if the error is an Axios error
 */
function isAxiosError(error: any): boolean {
  return error && error.isAxiosError === true;
}

/**
 * Handle API errors in a standardized way
 * 
 * @param error The error to handle
 * @param context Additional context information
 * @returns Formatted error message
 */
export function handleApiError(error: any, context?: Record<string, any>): string {
  // Log the error
  logError(error, context);
  
  // Return formatted message
  return formatErrorMessage(error);
}
