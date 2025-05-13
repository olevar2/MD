/**
 * API Client
 * 
 * This module provides a client for interacting with the Analysis Engine API.
 */

import axios from 'axios';

// Create axios instance
const apiClient = axios.create({
  baseURL: process.env.REACT_APP_API_URL || 'http://localhost:8000',
  timeout: 30000, // 30 seconds
  headers: {
    'Content-Type': 'application/json',
  }
});

// Request interceptor
apiClient.interceptors.request.use(
  (config) => {
    // Get API key from local storage
    const apiKey = localStorage.getItem('api_key');
    
    // Add API key to headers if available
    if (apiKey) {
      config.headers['X-API-Key'] = apiKey;
    }
    
    // Add request ID for tracing
    config.headers['X-Request-ID'] = generateRequestId();
    
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// Response interceptor
apiClient.interceptors.response.use(
  (response) => {
    return response;
  },
  (error) => {
    // Handle API errors
    if (error.response) {
      // The request was made and the server responded with a status code
      // that falls out of the range of 2xx
      const { status, data } = error.response;
      
      if (status === 401) {
        // Unauthorized - clear API key
        localStorage.removeItem('api_key');
        
        // Redirect to login page
        window.location.href = '/login';
      }
      
      // Create error message
      let errorMessage = 'An error occurred';
      
      if (data && data.error) {
        if (data.error.message) {
          errorMessage = data.error.message;
        } else if (typeof data.error === 'string') {
          errorMessage = data.error;
        }
      }
      
      // Create error object
      const apiError = new Error(errorMessage);
      apiError.status = status;
      apiError.data = data;
      
      return Promise.reject(apiError);
    } else if (error.request) {
      // The request was made but no response was received
      const networkError = new Error('Network error. Please check your connection.');
      networkError.status = 0;
      
      return Promise.reject(networkError);
    } else {
      // Something happened in setting up the request that triggered an Error
      return Promise.reject(error);
    }
  }
);

// Generate request ID
const generateRequestId = () => {
  return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, (c) => {
    const r = Math.random() * 16 | 0;
    const v = c === 'x' ? r : (r & 0x3 | 0x8);
    return v.toString(16);
  });
};

// API functions
export const api = {
  // Authentication
  auth: {
    login: async (credentials) => {
      const response = await apiClient.post('/auth/login', credentials);
      
      // Store API key
      if (response.data && response.data.api_key) {
        localStorage.setItem('api_key', response.data.api_key);
      }
      
      return response.data;
    },
    
    logout: async () => {
      // Clear API key
      localStorage.removeItem('api_key');
      
      return { success: true };
    },
    
    getProfile: async () => {
      const response = await apiClient.get('/auth/profile');
      return response.data;
    }
  },
  
  // Confluence detection
  confluence: {
    detect: async (params) => {
      const response = await apiClient.post('/confluence', params);
      return response.data;
    },
    
    detectML: async (params) => {
      const response = await apiClient.post('/ml/confluence', params);
      return response.data;
    }
  },
  
  // Divergence analysis
  divergence: {
    analyze: async (params) => {
      const response = await apiClient.post('/divergence', params);
      return response.data;
    },
    
    analyzeML: async (params) => {
      const response = await apiClient.post('/ml/divergence', params);
      return response.data;
    }
  },
  
  // Pattern recognition
  patterns: {
    recognize: async (params) => {
      const response = await apiClient.post('/patterns', params);
      return response.data;
    },
    
    recognizeML: async (params) => {
      const response = await apiClient.post('/ml/patterns', params);
      return response.data;
    }
  },
  
  // Currency strength
  currencyStrength: {
    get: async (params) => {
      const response = await apiClient.get('/currency-strength', { params });
      return response.data;
    }
  },
  
  // Related pairs
  relatedPairs: {
    get: async (symbol, params) => {
      const response = await apiClient.get(`/related-pairs/${symbol}`, { params });
      return response.data;
    }
  },
  
  // Price data
  priceData: {
    get: async (symbol, timeframe, params) => {
      const response = await apiClient.get(`/price-data/${symbol}/${timeframe}`, { params });
      return response.data;
    }
  },
  
  // System status
  system: {
    getStatus: async () => {
      const response = await apiClient.get('/system/status');
      return response.data;
    },
    
    getMetrics: async () => {
      const response = await apiClient.get('/system/metrics');
      return response.data;
    }
  },
  
  // ML models
  models: {
    list: async (params) => {
      const response = await apiClient.get('/ml/models', { params });
      return response.data;
    },
    
    getInfo: async (modelName) => {
      const response = await apiClient.get(`/ml/models/${modelName}`);
      return response.data;
    }
  }
};

export default apiClient;
