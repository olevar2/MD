/**
 * Enhanced Error Handling Utilities for the UI Service.
 *
 * This module provides comprehensive error handling functions for the UI Service,
 * including error logging, formatting, notification, and integration with error
 * monitoring services like Sentry.
 */

import { AxiosError } from 'axios';

// Mock Sentry import - in a real application, you would import Sentry
// import * as Sentry from '@sentry/react';

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

  // UI errors
  UI_ERROR = 'UIError',
  RENDERING_ERROR = 'RenderingError',

  // Unknown errors
  UNKNOWN_ERROR = 'UnknownError'
}

// Error severity levels
export enum ErrorSeverity {
  FATAL = 'fatal',
  ERROR = 'error',
  WARNING = 'warning',
  INFO = 'info',
  DEBUG = 'debug'
}

// Interface for standardized error response from API
export interface ErrorResponse {
  error_type: string;
  message: string;
  details?: any;
  timestamp?: string;
  correlation_id?: string;
  service?: string;
}

// Interface for error context
export interface ErrorContext {
  [key: string]: any;
  component?: string;
  action?: string;
  url?: string;
  method?: string;
  userId?: string;
  correlationId?: string;
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
      const errorTypeFromResponse = (axiosError.response.data as ErrorResponse).error_type;

      // Check if the error type from the response is a valid ErrorType
      if (Object.values(ErrorType).includes(errorTypeFromResponse as ErrorType)) {
        return errorTypeFromResponse as ErrorType;
      }

      return ErrorType.UNKNOWN_ERROR;
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
    // Check if the error name matches any ErrorType
    const errorName = error.name as string;
    if (Object.values(ErrorType).includes(errorName as ErrorType)) {
      return errorName as ErrorType;
    }

    return ErrorType.UNKNOWN_ERROR;
  }

  // Handle unknown errors
  return ErrorType.UNKNOWN_ERROR;
}

/**
 * Get error severity based on error type
 *
 * @param errorType The error type
 * @returns The error severity
 */
export function getErrorSeverity(errorType: ErrorType): ErrorSeverity {
  switch (errorType) {
    case ErrorType.AUTHENTICATION_ERROR:
    case ErrorType.AUTHORIZATION_ERROR:
    case ErrorType.DATA_VALIDATION_ERROR:
      return ErrorSeverity.WARNING;

    case ErrorType.SERVICE_ERROR:
    case ErrorType.SERVICE_UNAVAILABLE_ERROR:
    case ErrorType.SERVICE_TIMEOUT_ERROR:
    case ErrorType.NETWORK_ERROR:
    case ErrorType.TRADING_ERROR:
    case ErrorType.ORDER_EXECUTION_ERROR:
    case ErrorType.FOREX_PLATFORM_ERROR:
      return ErrorSeverity.ERROR;

    case ErrorType.RENDERING_ERROR:
    case ErrorType.UI_ERROR:
      return ErrorSeverity.ERROR;

    default:
      return ErrorSeverity.ERROR;
  }
}

/**
 * Log error to console and to error monitoring services
 *
 * @param error The error to log
 * @param context Additional context information
 */
export function logError(error: any, context: ErrorContext = {}): void {
  const errorType = getErrorType(error);
  const severity = getErrorSeverity(errorType);
  const timestamp = new Date().toISOString();

  // Generate a correlation ID if not provided
  const correlationId = context.correlationId || generateCorrelationId();

  // Prepare context with correlation ID
  const enrichedContext = {
    ...context,
    correlationId,
    errorType,
    severity,
    timestamp,
    environment: process.env.NODE_ENV || 'development',
    userAgent: typeof navigator !== 'undefined' ? navigator.userAgent : 'unknown'
  };

  // Log to console
  console.error(`[${timestamp}] [${errorType}] [${severity}] [${correlationId}]`, error, enrichedContext);

  // Log to Sentry or other error monitoring service in production
  if (process.env.NODE_ENV === 'production') {
    logToErrorMonitoringService(error, errorType, severity, enrichedContext);
  }

  // Store in session storage for debugging (limited to last 10 errors)
  storeErrorInSession(error, errorType, severity, enrichedContext);
}

/**
 * Log error to error monitoring service (e.g., Sentry)
 *
 * @param error The error to log
 * @param errorType The type of error
 * @param severity The severity of the error
 * @param context Additional context information
 */
function logToErrorMonitoringService(
  error: any,
  errorType: ErrorType,
  severity: ErrorSeverity,
  context: ErrorContext
): void {
  // In a real application, you would use Sentry or another error monitoring service
  // Example with Sentry:
  /*
  Sentry.withScope((scope) => {
    // Set the scope level based on severity
    scope.setLevel(severity as Sentry.Severity);

    // Add context information
    Object.entries(context).forEach(([key, value]) => {
      scope.setExtra(key, value);
    });

    // Set tags for filtering
    scope.setTag('errorType', errorType);
    scope.setTag('component', context.component || 'unknown');

    // Set user information if available
    if (context.userId) {
      scope.setUser({ id: context.userId });
    }

    // Capture the error
    Sentry.captureException(error);
  });
  */

  // For now, just log to console in a format that indicates it would be sent to Sentry
  console.info(`[SENTRY] Would send error to Sentry: ${errorType} - ${formatErrorMessage(error)}`);
}

/**
 * Store error in session storage for debugging
 *
 * @param error The error to store
 * @param errorType The type of error
 * @param severity The severity of the error
 * @param context Additional context information
 */
function storeErrorInSession(
  error: any,
  errorType: ErrorType,
  severity: ErrorSeverity,
  context: ErrorContext
): void {
  if (typeof sessionStorage === 'undefined') return;

  try {
    // Get existing errors from session storage
    const storedErrorsJson = sessionStorage.getItem('ui_errors');
    const storedErrors = storedErrorsJson ? JSON.parse(storedErrorsJson) : [];

    // Add new error
    storedErrors.unshift({
      message: formatErrorMessage(error),
      errorType,
      severity,
      timestamp: new Date().toISOString(),
      stack: error instanceof Error ? error.stack : undefined,
      context
    });

    // Keep only the last 10 errors
    const limitedErrors = storedErrors.slice(0, 10);

    // Store back in session storage
    sessionStorage.setItem('ui_errors', JSON.stringify(limitedErrors));
  } catch (e) {
    console.error('Failed to store error in session storage:', e);
  }
}

/**
 * Generate a correlation ID for tracking errors
 *
 * @returns A unique correlation ID
 */
export function generateCorrelationId(): string {
  return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {
    const r = Math.random() * 16 | 0;
    const v = c === 'x' ? r : (r & 0x3 | 0x8);
    return v.toString(16);
  });
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
export function handleApiError(error: any, context: ErrorContext = {}): string {
  // Get error type
  const errorType = getErrorType(error);

  // Extract correlation ID from error response if available
  let correlationId = context.correlationId;
  if (isAxiosError(error)) {
    const axiosError = error as AxiosError<ErrorResponse>;
    if (axiosError.response?.data?.correlation_id) {
      correlationId = axiosError.response.data.correlation_id;
    }
  }

  // Enrich context with error type and correlation ID
  const enrichedContext: ErrorContext = {
    ...context,
    errorType,
    correlationId: correlationId || generateCorrelationId()
  };

  // Log the error
  logError(error, enrichedContext);

  // Return formatted message
  return formatErrorMessage(error);
}

/**
 * Create a custom error with a specific error type
 *
 * @param message The error message
 * @param errorType The type of error
 * @param details Additional error details
 * @returns A custom error
 */
export function createError(
  message: string,
  errorType: ErrorType = ErrorType.UNKNOWN_ERROR,
  details: Record<string, any> = {}
): Error {
  const error = new Error(message);
  error.name = errorType;
  (error as any).details = details;
  return error;
}

/**
 * Get all errors stored in session storage
 *
 * @returns Array of stored errors
 */
export function getStoredErrors(): any[] {
  if (typeof sessionStorage === 'undefined') return [];

  try {
    const storedErrorsJson = sessionStorage.getItem('ui_errors');
    return storedErrorsJson ? JSON.parse(storedErrorsJson) : [];
  } catch (e) {
    console.error('Failed to retrieve errors from session storage:', e);
    return [];
  }
}

/**
 * Clear all errors stored in session storage
 */
export function clearStoredErrors(): void {
  if (typeof sessionStorage === 'undefined') return;

  try {
    sessionStorage.removeItem('ui_errors');
  } catch (e) {
    console.error('Failed to clear errors from session storage:', e);
  }
}
