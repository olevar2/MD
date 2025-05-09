/**
 * API Client for the Forex Trading Platform
 * 
 * This module provides a standardized API client with error handling, retries,
 * circuit breaking, and other resilience patterns.
 */

const axios = require('axios');

const { 
  ForexTradingPlatformError,
  NetworkError,
  ServiceUnavailableError,
  ServiceTimeoutError
} = require('./errors');

const {
  generateCorrelationId,
  handleError,
  mapError
} = require('./errorHandler');

const {
  CircuitBreaker,
  RetryPolicy,
  Bulkhead,
  withResilience
} = require('./resilience');

/**
 * Default API client configuration
 */
const DEFAULT_CONFIG = {
  // Base URL for API requests
  baseURL: process.env.API_BASE_URL || 'http://localhost:3001/api',
  
  // Request timeout in milliseconds
  timeout: 10000,
  
  // Default headers
  headers: {
    'Content-Type': 'application/json',
    'Accept': 'application/json'
  },
  
  // Retry configuration
  retry: {
    maxRetries: 3,
    initialDelayMs: 100,
    maxDelayMs: 3000,
    backoffFactor: 2,
    jitter: true
  },
  
  // Circuit breaker configuration
  circuitBreaker: {
    failureThreshold: 5,
    resetTimeoutMs: 30000,
    halfOpenMaxRequests: 3
  },
  
  // Bulkhead configuration
  bulkhead: {
    maxConcurrent: 10,
    maxQueue: 20
  }
};

/**
 * API Client with built-in error handling, retries, and circuit breaking
 */
class ApiClient {
  /**
   * Create a new API client
   * 
   * @param {Object} config - API client configuration
   */
  constructor(config = {}) {
    // Merge configuration with defaults
    this.config = {
      ...DEFAULT_CONFIG,
      ...config,
      headers: {
        ...DEFAULT_CONFIG.headers,
        ...config.headers
      },
      retry: {
        ...DEFAULT_CONFIG.retry,
        ...config.retry
      },
      circuitBreaker: {
        ...DEFAULT_CONFIG.circuitBreaker,
        ...config.circuitBreaker
      },
      bulkhead: {
        ...DEFAULT_CONFIG.bulkhead,
        ...config.bulkhead
      }
    };
    
    // Create axios instance
    this.axiosInstance = axios.create({
      baseURL: this.config.baseURL,
      timeout: this.config.timeout,
      headers: this.config.headers
    });
    
    // Create resilience components
    this.retryPolicy = new RetryPolicy(this.config.retry);
    this.circuitBreaker = new CircuitBreaker('api-client', this.config.circuitBreaker);
    this.bulkhead = new Bulkhead('api-client', this.config.bulkhead);
    
    // Add request interceptor
    this.axiosInstance.interceptors.request.use(
      this._handleRequest.bind(this),
      this._handleRequestError.bind(this)
    );
    
    // Add response interceptor
    this.axiosInstance.interceptors.response.use(
      this._handleResponse.bind(this),
      this._handleResponseError.bind(this)
    );
  }

  /**
   * Make a GET request
   * 
   * @param {string} url - URL to request
   * @param {Object} config - Request configuration
   * @returns {Promise<*>} Response data
   */
  async get(url, config = {}) {
    return this._request({
      ...config,
      method: 'get',
      url
    });
  }

  /**
   * Make a POST request
   * 
   * @param {string} url - URL to request
   * @param {*} data - Data to send
   * @param {Object} config - Request configuration
   * @returns {Promise<*>} Response data
   */
  async post(url, data, config = {}) {
    return this._request({
      ...config,
      method: 'post',
      url,
      data
    });
  }

  /**
   * Make a PUT request
   * 
   * @param {string} url - URL to request
   * @param {*} data - Data to send
   * @param {Object} config - Request configuration
   * @returns {Promise<*>} Response data
   */
  async put(url, data, config = {}) {
    return this._request({
      ...config,
      method: 'put',
      url,
      data
    });
  }

  /**
   * Make a DELETE request
   * 
   * @param {string} url - URL to request
   * @param {Object} config - Request configuration
   * @returns {Promise<*>} Response data
   */
  async delete(url, config = {}) {
    return this._request({
      ...config,
      method: 'delete',
      url
    });
  }

  /**
   * Make a request with resilience patterns
   * 
   * @param {Object} config - Request configuration
   * @returns {Promise<*>} Response data
   */
  async _request(config) {
    // Create context for error handling
    const context = {
      url: config.url,
      method: config.method,
      correlationId: config.headers?.['X-Correlation-ID'] || generateCorrelationId()
    };
    
    // Create a function to execute the request
    const executeRequest = async () => {
      try {
        const response = await this.axiosInstance.request(config);
        return response.data;
      } catch (error) {
        // Map error to a platform-specific error
        const mappedError = mapError(error);
        
        // Add request context to error
        if (mappedError instanceof ForexTradingPlatformError) {
          mappedError.details = {
            ...mappedError.details,
            url: config.url,
            method: config.method
          };
        }
        
        // Throw the mapped error
        throw mappedError;
      }
    };
    
    // Apply resilience patterns
    return withResilience(
      executeRequest,
      {
        retryPolicy: this.retryPolicy,
        circuitBreaker: this.circuitBreaker,
        bulkhead: this.bulkhead,
        timeoutMs: config.timeout || this.config.timeout
      },
      context
    )();
  }

  /**
   * Handle request interceptor
   * 
   * @param {Object} config - Request configuration
   * @returns {Object} Modified request configuration
   */
  _handleRequest(config) {
    // Add correlation ID to headers if not present
    if (!config.headers['X-Correlation-ID']) {
      config.headers['X-Correlation-ID'] = generateCorrelationId();
    }
    
    // Add retry count to headers if not present
    if (typeof config.headers['X-Retry-Count'] === 'undefined') {
      config.headers['X-Retry-Count'] = 0;
    }
    
    return config;
  }

  /**
   * Handle request error interceptor
   * 
   * @param {Error} error - Request error
   * @returns {Promise<never>} Rejected promise with error
   */
  _handleRequestError(error) {
    // Handle the error
    handleError(error, {
      phase: 'request',
      url: error.config?.url,
      method: error.config?.method
    });
    
    // Reject with the error
    return Promise.reject(error);
  }

  /**
   * Handle response interceptor
   * 
   * @param {Object} response - Response object
   * @returns {Object} Modified response object
   */
  _handleResponse(response) {
    // Reset circuit breaker on successful response
    if (this.circuitBreaker.getState() === 'HALF_OPEN') {
      this.circuitBreaker._onSuccess();
    }
    
    return response;
  }

  /**
   * Handle response error interceptor
   * 
   * @param {Error} error - Response error
   * @returns {Promise<never>} Rejected promise with error
   */
  _handleResponseError(error) {
    // Handle the error
    handleError(error, {
      phase: 'response',
      url: error.config?.url,
      method: error.config?.method,
      status: error.response?.status
    });
    
    // Reject with the error
    return Promise.reject(error);
  }

  /**
   * Get the current circuit breaker state
   * 
   * @returns {string} Circuit breaker state
   */
  getCircuitBreakerState() {
    return this.circuitBreaker.getState();
  }

  /**
   * Reset the circuit breaker
   */
  resetCircuitBreaker() {
    this.circuitBreaker.reset();
  }

  /**
   * Get the current bulkhead state
   * 
   * @returns {Object} Bulkhead state
   */
  getBulkheadState() {
    return this.bulkhead.getState();
  }
}

// Create and export a default API client instance
const apiClient = new ApiClient();

module.exports = {
  ApiClient,
  apiClient,
  DEFAULT_CONFIG
};
