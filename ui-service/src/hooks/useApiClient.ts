/**
 * Hook for using the API client with React
 */

import { useState, useCallback, useEffect } from 'react';
import { apiClient, CircuitState } from '../api/apiClient';
import { ErrorType, getErrorType } from '../utils/errorHandler';
import { useSnackbar } from 'notistack';

interface UseApiClientOptions {
  showErrorNotification?: boolean;
  showSuccessNotification?: boolean;
  successMessage?: string;
  errorMessage?: string;
  onSuccess?: (data: any) => void;
  onError?: (error: any) => void;
}

/**
 * Hook for using the API client with React
 * 
 * @param options Configuration options
 * @returns Object with loading state, error state, data, and request methods
 */
export function useApiClient<T = any>(options: UseApiClientOptions = {}) {
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<Error | null>(null);
  const [errorType, setErrorType] = useState<ErrorType | null>(null);
  const [data, setData] = useState<T | null>(null);
  const [circuitState, setCircuitState] = useState<CircuitState>(apiClient.getCircuitState());
  
  const { enqueueSnackbar } = useSnackbar();
  
  const {
    showErrorNotification = true,
    showSuccessNotification = false,
    successMessage = 'Operation completed successfully',
    errorMessage,
    onSuccess,
    onError
  } = options;
  
  // Update circuit state when it changes
  useEffect(() => {
    const checkCircuitState = () => {
      const currentState = apiClient.getCircuitState();
      if (currentState !== circuitState) {
        setCircuitState(currentState);
        
        // Show notification when circuit opens
        if (currentState === CircuitState.OPEN) {
          enqueueSnackbar('Service is currently unavailable. Retrying in 30 seconds.', {
            variant: 'error',
            autoHideDuration: 5000
          });
        }
        
        // Show notification when circuit closes
        if (currentState === CircuitState.CLOSED && circuitState === CircuitState.HALF_OPEN) {
          enqueueSnackbar('Service is now available.', {
            variant: 'success',
            autoHideDuration: 3000
          });
        }
      }
    };
    
    // Check circuit state every 5 seconds
    const interval = setInterval(checkCircuitState, 5000);
    
    return () => clearInterval(interval);
  }, [circuitState, enqueueSnackbar]);
  
  /**
   * Handle successful response
   * 
   * @param responseData Response data
   * @param customOptions Custom options for this request
   */
  const handleSuccess = useCallback((
    responseData: T,
    customOptions: Partial<UseApiClientOptions> = {}
  ) => {
    setData(responseData);
    setError(null);
    setErrorType(null);
    
    const mergedOptions = { ...options, ...customOptions };
    
    if (mergedOptions.showSuccessNotification) {
      enqueueSnackbar(mergedOptions.successMessage, {
        variant: 'success',
        autoHideDuration: 3000
      });
    }
    
    if (mergedOptions.onSuccess) {
      mergedOptions.onSuccess(responseData);
    }
    
    return responseData;
  }, [options, enqueueSnackbar]);
  
  /**
   * Handle error response
   * 
   * @param error Error object
   * @param customOptions Custom options for this request
   */
  const handleError = useCallback((
    error: any,
    customOptions: Partial<UseApiClientOptions> = {}
  ) => {
    const errorObj = error instanceof Error ? error : new Error(String(error));
    const type = getErrorType(error);
    
    setError(errorObj);
    setErrorType(type);
    
    const mergedOptions = { ...options, ...customOptions };
    
    if (mergedOptions.showErrorNotification) {
      enqueueSnackbar(
        mergedOptions.errorMessage || errorObj.message,
        {
          variant: 'error',
          autoHideDuration: 5000
        }
      );
    }
    
    if (mergedOptions.onError) {
      mergedOptions.onError(error);
    }
    
    return null;
  }, [options, enqueueSnackbar]);
  
  /**
   * Make a GET request
   * 
   * @param url URL to request
   * @param params Query parameters
   * @param customOptions Custom options for this request
   * @returns Promise resolving to response data
   */
  const get = useCallback(async <R = T>(
    url: string,
    params?: Record<string, any>,
    customOptions?: Partial<UseApiClientOptions>
  ): Promise<R | null> => {
    setLoading(true);
    
    try {
      const response = await apiClient.get<R>(url, { params });
      return handleSuccess(response, customOptions) as R;
    } catch (error) {
      return handleError(error, customOptions);
    } finally {
      setLoading(false);
    }
  }, [handleSuccess, handleError]);
  
  /**
   * Make a POST request
   * 
   * @param url URL to request
   * @param data Data to send
   * @param customOptions Custom options for this request
   * @returns Promise resolving to response data
   */
  const post = useCallback(async <R = T>(
    url: string,
    data?: any,
    customOptions?: Partial<UseApiClientOptions>
  ): Promise<R | null> => {
    setLoading(true);
    
    try {
      const response = await apiClient.post<R>(url, data);
      return handleSuccess(response, customOptions) as R;
    } catch (error) {
      return handleError(error, customOptions);
    } finally {
      setLoading(false);
    }
  }, [handleSuccess, handleError]);
  
  /**
   * Make a PUT request
   * 
   * @param url URL to request
   * @param data Data to send
   * @param customOptions Custom options for this request
   * @returns Promise resolving to response data
   */
  const put = useCallback(async <R = T>(
    url: string,
    data?: any,
    customOptions?: Partial<UseApiClientOptions>
  ): Promise<R | null> => {
    setLoading(true);
    
    try {
      const response = await apiClient.put<R>(url, data);
      return handleSuccess(response, customOptions) as R;
    } catch (error) {
      return handleError(error, customOptions);
    } finally {
      setLoading(false);
    }
  }, [handleSuccess, handleError]);
  
  /**
   * Make a DELETE request
   * 
   * @param url URL to request
   * @param customOptions Custom options for this request
   * @returns Promise resolving to response data
   */
  const del = useCallback(async <R = T>(
    url: string,
    customOptions?: Partial<UseApiClientOptions>
  ): Promise<R | null> => {
    setLoading(true);
    
    try {
      const response = await apiClient.delete<R>(url);
      return handleSuccess(response, customOptions) as R;
    } catch (error) {
      return handleError(error, customOptions);
    } finally {
      setLoading(false);
    }
  }, [handleSuccess, handleError]);
  
  /**
   * Reset error state
   */
  const resetError = useCallback(() => {
    setError(null);
    setErrorType(null);
  }, []);
  
  /**
   * Reset data state
   */
  const resetData = useCallback(() => {
    setData(null);
  }, []);
  
  /**
   * Reset circuit breaker
   */
  const resetCircuitBreaker = useCallback(() => {
    apiClient.resetCircuitBreaker();
    setCircuitState(apiClient.getCircuitState());
  }, []);
  
  return {
    loading,
    error,
    errorType,
    data,
    circuitState,
    get,
    post,
    put,
    delete: del,
    resetError,
    resetData,
    resetCircuitBreaker
  };
}

/**
 * Hook for making GET requests
 * 
 * @param url URL to request
 * @param options Configuration options
 * @returns Object with loading state, error state, data, and fetch function
 */
export function useGet<T = any>(
  url: string,
  options: UseApiClientOptions = {}
) {
  const api = useApiClient<T>(options);
  
  const fetch = useCallback((
    params?: Record<string, any>,
    customOptions?: Partial<UseApiClientOptions>
  ) => {
    return api.get<T>(url, params, customOptions);
  }, [api, url]);
  
  return { ...api, fetch };
}

/**
 * Hook for making POST requests
 * 
 * @param url URL to request
 * @param options Configuration options
 * @returns Object with loading state, error state, data, and submit function
 */
export function usePost<T = any, D = any>(
  url: string,
  options: UseApiClientOptions = {}
) {
  const api = useApiClient<T>(options);
  
  const submit = useCallback((
    data?: D,
    customOptions?: Partial<UseApiClientOptions>
  ) => {
    return api.post<T>(url, data, customOptions);
  }, [api, url]);
  
  return { ...api, submit };
}

/**
 * Hook for making PUT requests
 * 
 * @param url URL to request
 * @param options Configuration options
 * @returns Object with loading state, error state, data, and update function
 */
export function usePut<T = any, D = any>(
  url: string,
  options: UseApiClientOptions = {}
) {
  const api = useApiClient<T>(options);
  
  const update = useCallback((
    data?: D,
    customOptions?: Partial<UseApiClientOptions>
  ) => {
    return api.put<T>(url, data, customOptions);
  }, [api, url]);
  
  return { ...api, update };
}

/**
 * Hook for making DELETE requests
 * 
 * @param url URL to request
 * @param options Configuration options
 * @returns Object with loading state, error state, data, and remove function
 */
export function useDelete<T = any>(
  url: string,
  options: UseApiClientOptions = {}
) {
  const api = useApiClient<T>(options);
  
  const remove = useCallback((
    customOptions?: Partial<UseApiClientOptions>
  ) => {
    return api.delete<T>(url, customOptions);
  }, [api, url]);
  
  return { ...api, remove };
}
