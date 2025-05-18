/**
 * Centralized API Client for the UI Service
 * 
 * This module provides a standardized API client with error handling, retries,
 * circuit breaking, and other resilience patterns.
 */

import axios, { AxiosError, AxiosInstance, AxiosRequestConfig, AxiosResponse } from 'axios';
import { 
  ErrorType, 
  ErrorResponse, 
  handleApiError, 
  generateCorrelationId,
  createError
} from '../utils/errorHandler';

// Circuit breaker states
enum CircuitState {
  CLOSED = 'CLOSED',   // Normal operation, requests flow through
  OPEN = 'OPEN',       // Circuit is open, requests fail fast
  HALF_OPEN = 'HALF_OPEN' // Testing if service is back, allowing limited requests
}

// Default configuration
const DEFAULT_CONFIG = {
  baseURL: process.env.NEXT_PUBLIC_API_URL || 'http://localhost:3001/api',
  timeout: 10000,
  headers: {
    'Content-Type': 'application/json',
    'Accept': 'application/json'
  }
};

// Retry configuration
const DEFAULT_RETRY_CONFIG = {
  maxRetries: 3,
  initialDelayMs: 100,
  maxDelayMs: 3000,
  backoffFactor: 2,
  retryableStatusCodes: [408, 429, 500, 502, 503, 504]
};

// Circuit breaker configuration
const DEFAULT_CIRCUIT_BREAKER_CONFIG = {
  failureThreshold: 5,     // Number of failures before opening circuit
  resetTimeoutMs: 30000,   // Time before trying to close circuit again
  halfOpenMaxRequests: 3   // Max requests in half-open state
};

/**
 * API Client with built-in error handling, retries, and circuit breaking
 */
export class ApiClient {
  private axiosInstance: AxiosInstance;
  private retryConfig: typeof DEFAULT_RETRY_CONFIG;
  private circuitBreakerConfig: typeof DEFAULT_CIRCUIT_BREAKER_CONFIG;
  
  // Circuit breaker state
  private circuitState: CircuitState = CircuitState.CLOSED;
  private failureCount: number = 0;
  private circuitOpenTime: number = 0;
  private halfOpenSuccessCount: number = 0;
  
  /**
   * Create a new API client
   * 
   * @param config Axios configuration
   * @param retryConfig Retry configuration
   * @param circuitBreakerConfig Circuit breaker configuration
   */
  constructor(
    config: AxiosRequestConfig = {},
    retryConfig: Partial<typeof DEFAULT_RETRY_CONFIG> = {},
    circuitBreakerConfig: Partial<typeof DEFAULT_CIRCUIT_BREAKER_CONFIG> = {}
  ) {
    // Create Axios instance with merged config
    this.axiosInstance = axios.create({
      ...DEFAULT_CONFIG,
      ...config
    });
    
    // Set retry config
    this.retryConfig = {
      ...DEFAULT_RETRY_CONFIG,
      ...retryConfig
    };
    
    // Set circuit breaker config
    this.circuitBreakerConfig = {
      ...DEFAULT_CIRCUIT_BREAKER_CONFIG,
      ...circuitBreakerConfig
    };
    
    // Add request interceptor
    this.axiosInstance.interceptors.request.use(
      this.handleRequest.bind(this),
      this.handleRequestError.bind(this)
    );
    
    // Add response interceptor
    this.axiosInstance.interceptors.response.use(
      this.handleResponse.bind(this),
      this.handleResponseError.bind(this)
    );
  }
  
  /**
   * Make a GET request
   * 
   * @param url URL to request
   * @param config Request configuration
   * @returns Promise resolving to response data
   */
  async get<T = any>(url: string, config: AxiosRequestConfig = {}): Promise<T> {
    return this.request<T>({ ...config, method: 'GET', url });
  }
  
  /**
   * Make a POST request
   * 
   * @param url URL to request
   * @param data Data to send
   * @param config Request configuration
   * @returns Promise resolving to response data
   */
  async post<T = any>(url: string, data?: any, config: AxiosRequestConfig = {}): Promise<T> {
    return this.request<T>({ ...config, method: 'POST', url, data });
  }
  
  /**
   * Make a PUT request
   * 
   * @param url URL to request
   * @param data Data to send
   * @param config Request configuration
   * @returns Promise resolving to response data
   */
  async put<T = any>(url: string, data?: any, config: AxiosRequestConfig = {}): Promise<T> {
    return this.request<T>({ ...config, method: 'PUT', url, data });
  }
  
  /**
   * Make a DELETE request
   * 
   * @param url URL to request
   * @param config Request configuration
   * @returns Promise resolving to response data
   */
  async delete<T = any>(url: string, config: AxiosRequestConfig = {}): Promise<T> {
    return this.request<T>({ ...config, method: 'DELETE', url });
  }
  
  /**
   * Make a request with circuit breaker and retry logic
   * 
   * @param config Request configuration
   * @returns Promise resolving to response data
   */
  private async request<T = any>(config: AxiosRequestConfig): Promise<T> {
    // Check circuit breaker
    this.checkCircuitBreaker();
    
    try {
      const response = await this.axiosInstance.request<T>(config);
      return response.data;
    } catch (error) {
      // Let the response interceptor handle the error
      throw error;
    }
  }
  
  /**
   * Handle request interceptor
   * 
   * @param config Request configuration
   * @returns Modified request configuration
   */
  private handleRequest(config: AxiosRequestConfig): AxiosRequestConfig {
    // Add correlation ID to headers if not present
    if (!config.headers) {
      config.headers = {};
    }

    if (!config.headers['X-Correlation-ID']) {
      config.headers['X-Correlation-ID'] = generateCorrelationId();
    }

    // Add retry count to request if not present
    if (config.headers && typeof config.headers['X-Retry-Count'] === 'undefined') {
      config.headers['X-Retry-Count'] = 0;
    }

    // Log outgoing request
    console.info(`[API Client] Outgoing request: ${config.method?.toUpperCase()} ${config.url}`, { correlationId: config.headers['X-Correlation-ID'], retryCount: config.headers['X-Retry-Count'] });

    return config;
  }

  /**
   * Handle request error interceptor
   * 
   * @param error Request error
   * @returns Rejected promise with error
   */
  private handleRequestError(error: any): Promise<never> {
    // Log the error
    console.error('[API Client] Request error:', error);
    handleApiError(error, {
      action: 'request',
      url: error.config?.url,
      method: error.config?.method
    });

    return Promise.reject(error);
  }
  
  /**
   * Handle response interceptor
   * 
   * @param response Response object
   * @returns Modified response object
   */
  private handleResponse(response: AxiosResponse): AxiosResponse {
    // Log incoming response
    console.info(`[API Client] Incoming response: ${response.config.method?.toUpperCase()} ${response.config.url} - Status: ${response.status}`, { correlationId: response.config.headers?.['X-Correlation-ID'] });

    // Reset failure count on successful response if in half-open state
    if (this.circuitState === CircuitState.HALF_OPEN) {
      this.halfOpenSuccessCount++;

      // If we've had enough successful requests, close the circuit
      if (this.halfOpenSuccessCount >= this.circuitBreakerConfig.halfOpenMaxRequests) {
        this.closeCircuit();
      }
    }

    return response;
  }

  /**
   * Handle response error interceptor with retry logic
   * 
   * @param error Response error
   * @returns Promise with retried request or rejected with error
   */
  private async handleResponseError(error: any): Promise<any> {
    // Log response error
    console.error('[API Client] Response error:', error);

    // Get retry count from headers
    const config = error.config;
    if (!config) {
      return Promise.reject(error);
    }

    // Initialize retry count if not present
    if (!config.headers) {
      config.headers = {};
    }

    const retryCount = config.headers['X-Retry-Count'] || 0;

    // Check if we should retry
    if (
      retryCount < this.retryConfig.maxRetries &&
      this.isRetryableError(error)
    ) {
      // Increment retry count
      config.headers['X-Retry-Count'] = retryCount + 1;

      // Calculate delay with exponential backoff
      const delay = Math.min(
        this.retryConfig.initialDelayMs * Math.pow(this.retryConfig.backoffFactor, retryCount),
        this.retryConfig.maxDelayMs
      );

      // Wait for the delay
      console.warn(`[API Client] Retrying request: ${config.method?.toUpperCase()} ${config.url} in ${delay}ms (Attempt ${retryCount + 1})`, { correlationId: config.headers['X-Correlation-ID'] });
      await new Promise(resolve => setTimeout(resolve, delay));

      // Retry the request
      return this.axiosInstance.request(config);
    }

    // If we're in half-open state, increment failure count
    if (this.circuitState === CircuitState.HALF_OPEN) {
      this.openCircuit();
    } else if (this.circuitState === CircuitState.CLOSED) {
      // Increment failure count
      this.failureCount++;

      // Check if we should open the circuit
      if (this.failureCount >= this.circuitBreakerConfig.failureThreshold) {
        this.openCircuit();
      }
    }

    // Format and throw the error
    if (error.response) {
      // The request was made and the server responded with a status code
      // that falls out of the range of 2xx
      const errorResponse = error.response.data as ErrorResponse;
      const errorType = errorResponse?.error_type as ErrorType || ErrorType.UNKNOWN_ERROR;
      const message = errorResponse?.message || 'An error occurred';

      throw createError(message, errorType, {
        status: error.response.status,
        data: errorResponse,
        url: config.url,
        method: config.method
      });
    } else if (error.request) {
      // The request was made but no response was received
      throw createError(
        'No response received from server',
        ErrorType.NETWORK_ERROR,
        { url: config.url, method: config.method }
      );
    } else {
      // Something happened in setting up the request that triggered an Error
      throw createError(
        error.message || 'Request failed',
        ErrorType.UNKNOWN_ERROR,
        { url: config.url, method: config.method }
      );
    }
  }
  
  /**
   * Check if an error is retryable
   * 
   * @param error The error to check
   * @returns True if the error is retryable
   */
  private isRetryableError(error: any): boolean {
    // Network errors are retryable
    if (!error.response) {
      return true;
    }
    
    // Check if status code is in retryable list
    return this.retryConfig.retryableStatusCodes.includes(error.response.status);
  }
  
  /**
   * Check circuit breaker state before making a request
   * 
   * @throws Error if circuit is open
   */
  private checkCircuitBreaker(): void {
    const now = Date.now();
    
    if (this.circuitState === CircuitState.OPEN) {
      // Check if reset timeout has elapsed
      if (now - this.circuitOpenTime >= this.circuitBreakerConfig.resetTimeoutMs) {
        // Move to half-open state
        this.circuitState = CircuitState.HALF_OPEN;
        this.halfOpenSuccessCount = 0;
      } else {
        // Circuit is still open, fail fast
        throw createError(
          'Service is currently unavailable',
          ErrorType.SERVICE_UNAVAILABLE_ERROR,
          { circuitState: this.circuitState }
        );
      }
    }
  }
  
  /**
   * Open the circuit
   */
  private openCircuit(): void {
    this.circuitState = CircuitState.OPEN;
    this.circuitOpenTime = Date.now();
    this.failureCount = 0;
    console.warn('[API Client] Circuit breaker opened');
  }
  
  /**
   * Close the circuit
   */
  private closeCircuit(): void {
    this.circuitState = CircuitState.CLOSED;
    this.failureCount = 0;
    this.halfOpenSuccessCount = 0;
    console.info('[API Client] Circuit breaker closed');
  }
  
  /**
   * Get the current circuit breaker state
   * 
   * @returns The current circuit state
   */
  getCircuitState(): CircuitState {
    return this.circuitState;
  }
  
  /**
   * Reset the circuit breaker
   */
  resetCircuitBreaker(): void {
    this.closeCircuit();
  }
}

// Create and export a default API client instance
export const apiClient = new ApiClient();

// Export circuit state enum
export { CircuitState };
