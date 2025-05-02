import { useState, useCallback } from 'react';
import axios, { AxiosRequestConfig, AxiosResponse, AxiosError } from 'axios';
import { handleApiError } from '../utils/errorHandler';
import { useSnackbar } from 'notistack';

interface UseApiOptions {
  showErrorNotification?: boolean;
  showSuccessNotification?: boolean;
  successMessage?: string;
}

/**
 * Custom hook for making API requests with built-in loading, error, and success states.
 * 
 * @param options Configuration options for the hook
 * @returns Object containing loading state, error state, data, and request function
 */
export function useApi<T = any>(options: UseApiOptions = {}) {
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [data, setData] = useState<T | null>(null);
  
  const { enqueueSnackbar } = useSnackbar();
  
  const {
    showErrorNotification = true,
    showSuccessNotification = false,
    successMessage = 'Operation completed successfully'
  } = options;
  
  /**
   * Execute an API request
   * 
   * @param config Axios request configuration
   * @param customOptions Options specific to this request
   * @returns Promise resolving to the response data
   */
  const request = useCallback(async <R = T>(
    config: AxiosRequestConfig,
    customOptions: Partial<UseApiOptions> = {}
  ): Promise<R | null> => {
    const requestOptions = { ...options, ...customOptions };
    
    setLoading(true);
    setError(null);
    
    try {
      const response: AxiosResponse<R> = await axios(config);
      
      setData(response.data as unknown as T);
      
      if (requestOptions.showSuccessNotification) {
        enqueueSnackbar(requestOptions.successMessage, { 
          variant: 'success',
          autoHideDuration: 3000
        });
      }
      
      return response.data;
    } catch (err) {
      const errorMessage = handleApiError(err, { 
        url: config.url,
        method: config.method
      });
      
      setError(errorMessage);
      
      if (requestOptions.showErrorNotification) {
        enqueueSnackbar(errorMessage, { 
          variant: 'error',
          autoHideDuration: 5000
        });
      }
      
      return null;
    } finally {
      setLoading(false);
    }
  }, [options, enqueueSnackbar]);
  
  return { loading, error, data, request };
}

/**
 * Custom hook for making GET requests with built-in loading, error, and success states.
 * 
 * @param url The URL to request
 * @param options Configuration options for the hook
 * @returns Object containing loading state, error state, data, and fetch function
 */
export function useGet<T = any>(url: string, options: UseApiOptions = {}) {
  const api = useApi<T>(options);
  
  const fetch = useCallback(async (
    params?: Record<string, any>,
    customOptions?: Partial<UseApiOptions>
  ): Promise<T | null> => {
    return api.request<T>({
      url,
      method: 'GET',
      params
    }, customOptions);
  }, [url, api]);
  
  return { ...api, fetch };
}

/**
 * Custom hook for making POST requests with built-in loading, error, and success states.
 * 
 * @param url The URL to request
 * @param options Configuration options for the hook
 * @returns Object containing loading state, error state, data, and submit function
 */
export function usePost<T = any, D = any>(url: string, options: UseApiOptions = {}) {
  const api = useApi<T>(options);
  
  const submit = useCallback(async (
    data?: D,
    customOptions?: Partial<UseApiOptions>
  ): Promise<T | null> => {
    return api.request<T>({
      url,
      method: 'POST',
      data
    }, customOptions);
  }, [url, api]);
  
  return { ...api, submit };
}
