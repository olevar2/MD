/**
 * Resilience Module for the Forex Trading Platform
 * 
 * This module provides resilience patterns for robust service communication in the
 * Forex trading platform. It includes circuit breaker, retry, timeout, and bulkhead
 * patterns to improve the reliability of service calls.
 */

const { 
  CircuitBreakerOpenError,
  RetryExhaustedError,
  BulkheadFullError,
  ServiceTimeoutError
} = require('./errors');

const { generateCorrelationId, logError } = require('./errorHandler');

/**
 * Circuit breaker states
 */
const CircuitState = {
  CLOSED: 'CLOSED',
  OPEN: 'OPEN',
  HALF_OPEN: 'HALF_OPEN'
};

/**
 * Default circuit breaker configuration
 */
const DEFAULT_CIRCUIT_BREAKER_CONFIG = {
  failureThreshold: 5,
  resetTimeoutMs: 30000,
  halfOpenMaxRequests: 3
};

/**
 * Circuit breaker implementation
 */
class CircuitBreaker {
  /**
   * Create a new circuit breaker
   * 
   * @param {string} name - Name of the circuit breaker
   * @param {Object} config - Circuit breaker configuration
   */
  constructor(name, config = {}) {
    this.name = name;
    this.config = { ...DEFAULT_CIRCUIT_BREAKER_CONFIG, ...config };
    this.state = CircuitState.CLOSED;
    this.failureCount = 0;
    this.successCount = 0;
    this.lastFailureTime = 0;
    this.listeners = [];
  }

  /**
   * Execute a function with circuit breaker protection
   * 
   * @param {Function} fn - Function to execute
   * @param {Object} context - Additional context information
   * @returns {Promise<*>} Result of the function
   * @throws {CircuitBreakerOpenError} If the circuit is open
   */
  async execute(fn, context = {}) {
    // Check if circuit is open
    this._checkState();
    
    try {
      // Execute the function
      const result = await fn();
      
      // Record success
      this._onSuccess();
      
      return result;
    } catch (error) {
      // Record failure
      this._onFailure(error, context);
      
      // Re-throw the error
      throw error;
    }
  }

  /**
   * Check the current state of the circuit breaker
   * 
   * @throws {CircuitBreakerOpenError} If the circuit is open
   */
  _checkState() {
    if (this.state === CircuitState.OPEN) {
      // Check if reset timeout has elapsed
      const now = Date.now();
      if (now - this.lastFailureTime >= this.config.resetTimeoutMs) {
        // Move to half-open state
        this._setState(CircuitState.HALF_OPEN);
      } else {
        // Circuit is still open, fail fast
        throw new CircuitBreakerOpenError(this.name, {
          resetTimeoutMs: this.config.resetTimeoutMs,
          timeElapsedMs: now - this.lastFailureTime
        });
      }
    }
  }

  /**
   * Handle successful execution
   */
  _onSuccess() {
    if (this.state === CircuitState.HALF_OPEN) {
      // Increment success count
      this.successCount++;
      
      // If we've had enough successful requests, close the circuit
      if (this.successCount >= this.config.halfOpenMaxRequests) {
        this._setState(CircuitState.CLOSED);
      }
    } else if (this.state === CircuitState.CLOSED) {
      // Reset failure count on success
      this.failureCount = 0;
    }
  }

  /**
   * Handle failed execution
   * 
   * @param {Error} error - The error that occurred
   * @param {Object} context - Additional context information
   */
  _onFailure(error, context = {}) {
    // Add circuit breaker information to context
    const enrichedContext = {
      ...context,
      circuitBreaker: this.name,
      circuitState: this.state
    };
    
    // Log the error
    logError(error, enrichedContext);
    
    if (this.state === CircuitState.HALF_OPEN) {
      // If we're in half-open state, any failure opens the circuit
      this._setState(CircuitState.OPEN);
    } else if (this.state === CircuitState.CLOSED) {
      // Increment failure count
      this.failureCount++;
      
      // Check if we should open the circuit
      if (this.failureCount >= this.config.failureThreshold) {
        this._setState(CircuitState.OPEN);
      }
    }
  }

  /**
   * Set the state of the circuit breaker
   * 
   * @param {string} newState - New state
   */
  _setState(newState) {
    const oldState = this.state;
    this.state = newState;
    
    // Reset counters
    if (newState === CircuitState.OPEN) {
      this.failureCount = 0;
      this.lastFailureTime = Date.now();
    } else if (newState === CircuitState.HALF_OPEN) {
      this.successCount = 0;
    } else if (newState === CircuitState.CLOSED) {
      this.failureCount = 0;
      this.successCount = 0;
    }
    
    // Notify listeners
    this._notifyListeners(oldState, newState);
    
    // Log state change
    console.info(`Circuit breaker ${this.name} state changed from ${oldState} to ${newState}`);
  }

  /**
   * Add a state change listener
   * 
   * @param {Function} listener - Listener function
   */
  addListener(listener) {
    this.listeners.push(listener);
  }

  /**
   * Remove a state change listener
   * 
   * @param {Function} listener - Listener function
   */
  removeListener(listener) {
    this.listeners = this.listeners.filter(l => l !== listener);
  }

  /**
   * Notify listeners of state change
   * 
   * @param {string} oldState - Old state
   * @param {string} newState - New state
   */
  _notifyListeners(oldState, newState) {
    this.listeners.forEach(listener => {
      try {
        listener(this.name, oldState, newState);
      } catch (error) {
        console.error(`Error in circuit breaker listener: ${error.message}`);
      }
    });
  }

  /**
   * Get the current state of the circuit breaker
   * 
   * @returns {string} Current state
   */
  getState() {
    return this.state;
  }

  /**
   * Reset the circuit breaker to closed state
   */
  reset() {
    this._setState(CircuitState.CLOSED);
  }
}

/**
 * Default retry configuration
 */
const DEFAULT_RETRY_CONFIG = {
  maxRetries: 3,
  initialDelayMs: 100,
  maxDelayMs: 3000,
  backoffFactor: 2,
  jitter: true,
  retryableErrors: [
    'NetworkError',
    'ServiceUnavailableError',
    'ServiceTimeoutError'
  ]
};

/**
 * Retry policy implementation
 */
class RetryPolicy {
  /**
   * Create a new retry policy
   * 
   * @param {Object} config - Retry configuration
   */
  constructor(config = {}) {
    this.config = { ...DEFAULT_RETRY_CONFIG, ...config };
  }

  /**
   * Execute a function with retry
   * 
   * @param {Function} fn - Function to execute
   * @param {Object} context - Additional context information
   * @returns {Promise<*>} Result of the function
   * @throws {RetryExhaustedError} If all retry attempts fail
   */
  async execute(fn, context = {}) {
    let lastError;
    
    for (let attempt = 1; attempt <= this.config.maxRetries + 1; attempt++) {
      try {
        // Execute the function
        return await fn();
      } catch (error) {
        lastError = error;
        
        // Check if we should retry
        if (attempt <= this.config.maxRetries && this._isRetryable(error)) {
          // Calculate delay with exponential backoff
          const delay = this._calculateDelay(attempt);
          
          // Log retry attempt
          console.info(`Retrying after error: ${error.message} (Attempt ${attempt}/${this.config.maxRetries}, Delay: ${delay}ms)`);
          
          // Wait for the delay
          await new Promise(resolve => setTimeout(resolve, delay));
        } else {
          // We've exhausted all retries or the error is not retryable
          break;
        }
      }
    }
    
    // If we get here, all retry attempts failed
    const enrichedContext = {
      ...context,
      retryAttempts: this.config.maxRetries,
      retryConfig: this.config
    };
    
    // Wrap the last error in a RetryExhaustedError
    const retryError = new RetryExhaustedError(
      `Retry attempts exhausted (${this.config.maxRetries})`,
      this.config.maxRetries,
      { originalError: lastError }
    );
    
    // Log the error
    logError(retryError, enrichedContext);
    
    // Re-throw the original error
    throw lastError;
  }

  /**
   * Check if an error is retryable
   * 
   * @param {Error} error - The error to check
   * @returns {boolean} True if the error is retryable
   */
  _isRetryable(error) {
    // Check if the error is in the list of retryable errors
    return this.config.retryableErrors.some(errorType => {
      // Check if the error is an instance of a platform error
      if (error.constructor && error.constructor.name === errorType) {
        return true;
      }
      
      // Check if the error has a name property that matches
      if (error.name === errorType) {
        return true;
      }
      
      return false;
    });
  }

  /**
   * Calculate delay for retry with exponential backoff
   * 
   * @param {number} attempt - Retry attempt number
   * @returns {number} Delay in milliseconds
   */
  _calculateDelay(attempt) {
    // Calculate base delay with exponential backoff
    let delay = this.config.initialDelayMs * Math.pow(this.config.backoffFactor, attempt - 1);
    
    // Apply maximum delay
    delay = Math.min(delay, this.config.maxDelayMs);
    
    // Apply jitter if enabled
    if (this.config.jitter) {
      // Add random jitter between 0% and 25%
      const jitterFactor = 1 + (Math.random() * 0.25);
      delay = Math.floor(delay * jitterFactor);
    }
    
    return delay;
  }
}

/**
 * Default bulkhead configuration
 */
const DEFAULT_BULKHEAD_CONFIG = {
  maxConcurrent: 10,
  maxQueue: 20
};

/**
 * Bulkhead implementation
 */
class Bulkhead {
  /**
   * Create a new bulkhead
   * 
   * @param {string} name - Name of the bulkhead
   * @param {Object} config - Bulkhead configuration
   */
  constructor(name, config = {}) {
    this.name = name;
    this.config = { ...DEFAULT_BULKHEAD_CONFIG, ...config };
    this.executingCount = 0;
    this.queue = [];
  }

  /**
   * Execute a function with bulkhead protection
   * 
   * @param {Function} fn - Function to execute
   * @param {Object} context - Additional context information
   * @returns {Promise<*>} Result of the function
   * @throws {BulkheadFullError} If the bulkhead is full
   */
  async execute(fn, context = {}) {
    // Check if we can execute immediately
    if (this.executingCount < this.config.maxConcurrent) {
      return this._doExecute(fn, context);
    }
    
    // Check if we can queue
    if (this.queue.length < this.config.maxQueue) {
      // Create a promise that will be resolved when the function is executed
      return new Promise((resolve, reject) => {
        this.queue.push({ fn, context, resolve, reject });
      });
    }
    
    // Bulkhead is full
    const error = new BulkheadFullError(this.name, {
      maxConcurrent: this.config.maxConcurrent,
      maxQueue: this.config.maxQueue,
      executingCount: this.executingCount,
      queueLength: this.queue.length
    });
    
    // Log the error
    logError(error, context);
    
    // Throw the error
    throw error;
  }

  /**
   * Execute a function and handle bulkhead accounting
   * 
   * @param {Function} fn - Function to execute
   * @param {Object} context - Additional context information
   * @returns {Promise<*>} Result of the function
   */
  async _doExecute(fn, context) {
    // Increment executing count
    this.executingCount++;
    
    try {
      // Execute the function
      return await fn();
    } finally {
      // Decrement executing count
      this.executingCount--;
      
      // Process next item in queue if any
      this._processQueue();
    }
  }

  /**
   * Process the next item in the queue
   */
  _processQueue() {
    // Check if we can process the next item
    if (this.executingCount < this.config.maxConcurrent && this.queue.length > 0) {
      // Get the next item
      const { fn, context, resolve, reject } = this.queue.shift();
      
      // Execute the function
      this._doExecute(fn, context)
        .then(resolve)
        .catch(reject);
    }
  }

  /**
   * Get the current state of the bulkhead
   * 
   * @returns {Object} Current state
   */
  getState() {
    return {
      name: this.name,
      executingCount: this.executingCount,
      queueLength: this.queue.length,
      maxConcurrent: this.config.maxConcurrent,
      maxQueue: this.config.maxQueue
    };
  }
}

/**
 * Execute a function with a timeout
 * 
 * @param {Function} fn - Function to execute
 * @param {number} timeoutMs - Timeout in milliseconds
 * @param {Object} context - Additional context information
 * @returns {Promise<*>} Result of the function
 * @throws {ServiceTimeoutError} If the function times out
 */
async function withTimeout(fn, timeoutMs, context = {}) {
  // Create a timeout promise
  const timeoutPromise = new Promise((_, reject) => {
    setTimeout(() => {
      reject(new ServiceTimeoutError(
        context.serviceName,
        timeoutMs,
        { function: fn.name || 'anonymous' }
      ));
    }, timeoutMs);
  });
  
  // Race the function against the timeout
  return Promise.race([fn(), timeoutPromise]);
}

/**
 * Higher-order function that wraps a function with resilience patterns
 * 
 * @param {Function} fn - Function to wrap
 * @param {Object} options - Resilience options
 * @param {Object} context - Additional context information
 * @returns {Function} The wrapped function
 */
function withResilience(fn, options = {}, context = {}) {
  // Extract options
  const {
    circuitBreaker,
    retryPolicy,
    bulkhead,
    timeoutMs
  } = options;
  
  // Return a wrapped function
  return async function(...args) {
    // Create a function that calls the original function with the provided arguments
    const execute = async () => {
      // Add timeout if specified
      if (timeoutMs) {
        return withTimeout(() => fn(...args), timeoutMs, context);
      }
      
      return fn(...args);
    };
    
    // Apply resilience patterns in order: bulkhead -> circuit breaker -> retry
    let resilientExecute = execute;
    
    // Apply retry policy if specified
    if (retryPolicy) {
      const retry = retryPolicy instanceof RetryPolicy ? 
        retryPolicy : 
        new RetryPolicy(retryPolicy);
      
      const retryExecute = resilientExecute;
      resilientExecute = () => retry.execute(retryExecute, context);
    }
    
    // Apply circuit breaker if specified
    if (circuitBreaker) {
      const cb = circuitBreaker instanceof CircuitBreaker ? 
        circuitBreaker : 
        new CircuitBreaker(context.serviceName || 'default', circuitBreaker);
      
      const cbExecute = resilientExecute;
      resilientExecute = () => cb.execute(cbExecute, context);
    }
    
    // Apply bulkhead if specified
    if (bulkhead) {
      const bh = bulkhead instanceof Bulkhead ? 
        bulkhead : 
        new Bulkhead(context.serviceName || 'default', bulkhead);
      
      const bhExecute = resilientExecute;
      resilientExecute = () => bh.execute(bhExecute, context);
    }
    
    // Execute the function with all resilience patterns
    return resilientExecute();
  };
}

module.exports = {
  // Circuit breaker
  CircuitBreaker,
  CircuitState,
  
  // Retry policy
  RetryPolicy,
  
  // Bulkhead
  Bulkhead,
  
  // Timeout
  withTimeout,
  
  // Combined resilience
  withResilience,
  
  // Default configurations
  DEFAULT_CIRCUIT_BREAKER_CONFIG,
  DEFAULT_RETRY_CONFIG,
  DEFAULT_BULKHEAD_CONFIG
};
